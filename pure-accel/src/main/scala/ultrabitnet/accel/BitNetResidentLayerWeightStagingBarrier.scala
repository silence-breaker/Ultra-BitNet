package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentLayerWeightStagingBarrierFault {
  val None = 0x00
  val GrantIdentity = 0xc0
  val GrantLayout = 0xc1
  val EventProtocol = 0xc2
  val CompletionIdentity = 0xc3
  val UnexpectedCompletion = 0xc4
  val WorkloadIdentity = 0xc5
  val WorkloadFailedBeforeGrant = 0xc6
  val WorkloadFailedWithActiveLayer = 0xc7
  val ExecutionFailed = 0xc8
}

/** Static contract between the resident event and PS-to-PL staging domains. */
case class BitNetResidentLayerWeightStagingBarrierConfig(
    event: BitNetResidentLayerEventSchedulerConfig =
      BitNetResidentLayerEventSchedulerConfig(),
    staging: BitNetPsToPlWeightStagingConfig =
      BitNetPsToPlWeightStagingConfig()) {
  val model: BitNetConfig = event.model

  require(model == BitNetConfig.Production)
  require(staging.model == model)
  require(staging.epochWidth == event.epochWidth,
    "event and staging epochs must be one authenticated identity domain")
  require(staging.staging.layers.size == model.layerCount)
  require(staging.targetCount ==
    BitNetFiveBankStripedLayout.WeightGeometry.size)
}

/**
  * Binds each resident transformer layer to its real weight-staging grant.
  *
  * The first `AttentionNormReady` event of a layer is buffered until a grant
  * with the same request/epoch/layer identity has been accepted and every
  * advertised resident target range has been checked against the immutable
  * production layout.  The grant's weight slot is retained in a dedicated
  * register; `event.slot` remains only the activation-image ping-pong slot.
  * Later events of that layer pass through one at a time.
  *
  * A real successful `LayerCommit`, or any earlier authenticated execution
  * failure, terminates the weight lease.  Its scheduler data commit and the
  * staging compute completion form an atomic Stream fork: neither consumer
  * can accept its branch without the other accepting on the same edge.  All
  * outward identities are reconstructed from locked registers.
  *
  * A staging workload failure received before the grant converts an already
  * waiting first event into a bounded failed data commit, so DMA failure
  * cannot leave the scheduler waiting forever.  Identity/layout corruption
  * is reported using locked identities and then enters reset-only quarantine.
  * There is no production completion-injection or synthetic retirement path.
  */
class BitNetResidentLayerWeightStagingBarrier(
    cfg: BitNetResidentLayerWeightStagingBarrierConfig =
      BitNetResidentLayerWeightStagingBarrierConfig()) extends Component {
  private val model = cfg.model
  private val layout = cfg.staging.staging

  val io = new Bundle {
    val eventCommand = slave(Stream(
      BitNetResidentLayerEventCommand(cfg.event)))
    val executionEventCommand = master(Stream(
      BitNetResidentLayerEventCommand(cfg.event)))
    val executionDataCommit = slave(Stream(
      BitNetResidentLayerEvent(cfg.event)))
    val dataCommit = master(Stream(
      BitNetResidentLayerEvent(cfg.event)))

    val computeGrant = slave(Stream(
      BitNetWeightStagingComputeGrant(cfg.staging)))
    val computeCompletion = master(Stream(
      BitNetWeightStagingComputeCompletion(cfg.staging)))

    // The barrier observes the terminal staging result but preserves it as a
    // typed, backpressure-stable outward stream for the workload owner.
    val stagingCompletion = slave(Stream(
      BitNetWeightStagingWorkloadCompletion(cfg.staging)))
    val workloadCompletion = master(Stream(
      BitNetWeightStagingWorkloadCompletion(cfg.staging)))

    val busy = out Bool()
    val layerGrantActive = out Bool()
    val weightSlot = out UInt(cfg.staging.slotWidth bits)
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val faultCode = out Bits(8 bits)
    val acceptedEvents = out UInt(32 bits)
    val completedEvents = out UInt(32 bits)
    val completedLayers = out UInt(32 bits)
  }

  val eventValid = RegInit(False)
  val heldEvent = Reg(BitNetResidentLayerEventCommand(cfg.event)) init
    BitNetResidentLayerEventCommand(cfg.event).getZero
  val heldNeedsGrant = RegInit(False)
  val eventDispatched = RegInit(False)

  val grantValid = RegInit(False)
  val heldGrant = Reg(BitNetWeightStagingComputeGrant(cfg.staging)) init
    BitNetWeightStagingComputeGrant(cfg.staging).getZero
  val layerActive = RegInit(False)
  val activeGrant = Reg(BitNetWeightStagingComputeGrant(cfg.staging)) init
    BitNetWeightStagingComputeGrant(cfg.staging).getZero

  val workloadIdentityValid = RegInit(False)
  val workloadRequestId = Reg(UInt(model.requestIdWidth bits)) init 0
  val workloadEpoch = Reg(UInt(cfg.event.epochWidth bits)) init 0

  val preGrantFailureValid = RegInit(False)
  val preGrantFailure = Reg(
    BitNetWeightStagingWorkloadCompletion(cfg.staging)) init
      BitNetWeightStagingWorkloadCompletion(cfg.staging).getZero
  val computeAbortPending = RegInit(False)

  val workloadForwardValid = RegInit(False)
  val workloadForward = Reg(
    BitNetWeightStagingWorkloadCompletion(cfg.staging)) init
      BitNetWeightStagingWorkloadCompletion(cfg.staging).getZero

  val reportValid = RegInit(False)
  val reportEvent = Reg(BitNetResidentLayerEvent(cfg.event)) init
    BitNetResidentLayerEvent(cfg.event).getZero
  val reportCompute = Reg(
    BitNetWeightStagingComputeCompletion(cfg.staging)) init
      BitNetWeightStagingComputeCompletion(cfg.staging).getZero
  val reportNeedsCompute = RegInit(False)
  val reportThenQuarantine = RegInit(False)
  val reportEndsWorkload = RegInit(False)

  val quarantine = RegInit(False)
  val stickyFault = RegInit(False)
  val stickyFaultCode = Reg(Bits(8 bits)) init 0
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0
  val completedLayerCount = Reg(UInt(32 bits)) init 0

  private def rememberFault(code: Bits): Unit = {
    stickyFault := True
    when(!stickyFault) { stickyFaultCode := code }
  }

  private def fillEventReport(
      source: BitNetResidentLayerEventCommand,
      success: Bool,
      code: Bits): Unit = {
    reportEvent.requestId := source.requestId
    reportEvent.layer := source.layer
    reportEvent.epoch := source.epoch
    reportEvent.stage := source.stage
    reportEvent.slot := source.slot
    reportEvent.success := success
    reportEvent.faultCode := code
  }

  private def fillComputeReport(
      source: BitNetWeightStagingComputeGrant,
      success: Bool,
      code: Bits): Unit = {
    reportCompute.requestId := source.requestId
    reportCompute.epoch := source.epoch
    reportCompute.layer := source.layer
    reportCompute.slot := source.slot
    reportCompute.success := success
    reportCompute.faultCode := code
  }

  // ---------------------------------------------------------------------
  // Independent one-entry input buffers.  A grant may legitimately become
  // ready before its first scheduler event, or vice versa.
  // ---------------------------------------------------------------------
  io.eventCommand.ready := !quarantine && !eventValid && !reportValid
  when(io.eventCommand.fire) {
    heldEvent := io.eventCommand.payload
    heldNeedsGrant := !layerActive &&
      io.eventCommand.payload.stage ===
        BitNetResidentLayerEventStage.AttentionNormReady
    eventDispatched := False
    eventValid := True
    acceptedCount := acceptedCount + 1

    when(!workloadIdentityValid &&
        io.eventCommand.payload.stage ===
          BitNetResidentLayerEventStage.AttentionNormReady) {
      workloadIdentityValid := True
      workloadRequestId := io.eventCommand.payload.requestId
      workloadEpoch := io.eventCommand.payload.epoch
    }
  }

  io.computeGrant.ready := !quarantine && !grantValid && !layerActive &&
    !reportValid && !preGrantFailureValid
  when(io.computeGrant.fire) {
    heldGrant := io.computeGrant.payload
    grantValid := True
    when(!workloadIdentityValid) {
      workloadIdentityValid := True
      workloadRequestId := io.computeGrant.payload.requestId
      workloadEpoch := io.computeGrant.payload.epoch
    }
  }

  // Preserve the staging terminal result even while the internal failure
  // branch is being converted into a scheduler commit.
  val stagingCompletionIdentityMatches = workloadIdentityValid &&
    io.stagingCompletion.payload.requestId === workloadRequestId &&
    io.stagingCompletion.payload.epoch === workloadEpoch

  io.stagingCompletion.ready := !workloadForwardValid && !quarantine
  io.workloadCompletion.valid := workloadForwardValid
  io.workloadCompletion.payload := workloadForward
  when(io.stagingCompletion.fire) {
    workloadForwardValid := True
    workloadForward := io.stagingCompletion.payload

    when(!stagingCompletionIdentityMatches) {
      // The terminal result is still forwarded byte-for-byte to its owner,
      // but it may not mutate any state belonging to the locked workload.
      // In particular, a wrong-identity success can never impersonate the
      // matching terminal result and erase the identity needed by FinalNorm.
      rememberFault(B(
        BitNetResidentLayerWeightStagingBarrierFault.
          WorkloadIdentity, 8 bits))
      quarantine := True
    } otherwise {
      when(io.stagingCompletion.payload.success) {
        // Staging completion and inference completion are different
        // lifetimes.  Keep the authenticated request/epoch locked until the
        // matching FinalNorm execution report has really retired.
        computeAbortPending := False
      } otherwise {
        when(computeAbortPending) {
          // This is the expected workload-level echo of an earlier failed
          // compute completion already delivered by the atomic fork.
          computeAbortPending := False
          workloadIdentityValid := False
        } elsewhen(!layerActive) {
          preGrantFailureValid := True
          preGrantFailure := io.stagingCompletion.payload
        } otherwise {
          // The staging controller has abandoned a slot while its granted
          // layer still owns physical engines.  Cancellation is not safe.
          rememberFault(B(
            BitNetResidentLayerWeightStagingBarrierFault.
              WorkloadFailedWithActiveLayer, 8 bits))
          quarantine := True
        }
      }
    }
  }
  when(io.workloadCompletion.fire) { workloadForwardValid := False }

  // ---------------------------------------------------------------------
  // Immutable grant-layout authentication.  These tables are constants in
  // RTL; no descriptor RAM or software-programmable address is introduced.
  // ---------------------------------------------------------------------
  val grantLayerInRange = heldGrant.layer < model.layerCount
  val safeGrantLayer = UInt(model.layerWidth bits)
  safeGrantLayer := 0
  when(grantLayerInRange) { safeGrantLayer := heldGrant.layer }

  val expectedLayerBases = Vec(layout.layers.map(layer =>
    U(layer.destinationBase, model.addressWidth bits)))
  val layerGeometryMatches = grantLayerInRange &&
    heldGrant.plLayerBase === expectedLayerBases(safeGrantLayer) &&
    heldGrant.plLayerBytes === U(layout.layerBytes,
      heldGrant.plLayerBytes.getWidth bits)

  val targetGeometryMatches =
    layout.layers.head.targets.indices.map { targetIndex =>
      val targetBases = Vec(layout.layers.map(layer =>
        U(layer.destinationBase + layer.targets(targetIndex).layerOffset,
          model.addressWidth bits)))
      heldGrant.targetBase(targetIndex) === targetBases(safeGrantLayer) &&
        heldGrant.targetBytes(targetIndex) === U(
          layout.layers.head.targets(targetIndex).bytes,
          cfg.staging.targetByteWidth bits)
    }.reduce(_ && _)

  val grantEventIdentityMatches =
    heldGrant.requestId === heldEvent.requestId &&
      heldGrant.epoch === heldEvent.epoch &&
      heldGrant.layer === heldEvent.layer &&
      (!workloadIdentityValid ||
        (heldGrant.requestId === workloadRequestId &&
          heldGrant.epoch === workloadEpoch))

  val activeEventIdentityMatches =
    heldEvent.requestId === activeGrant.requestId &&
      heldEvent.epoch === activeGrant.epoch &&
      heldEvent.layer === activeGrant.layer &&
      (!workloadIdentityValid ||
        (heldEvent.requestId === workloadRequestId &&
          heldEvent.epoch === workloadEpoch))

  // A first event is dispatchable only after the grant has moved into the
  // dedicated activeGrant register.  Later events may never impersonate a
  // second AttentionNorm or a FinalNorm while the weight slot is active.
  val activeStageLegal = heldNeedsGrant ||
    (heldEvent.stage =/=
      BitNetResidentLayerEventStage.AttentionNormReady &&
      heldEvent.stage =/=
        BitNetResidentLayerEventStage.FinalNormReady)
  val finalNormIdentityMatches = !workloadIdentityValid ||
    (heldEvent.requestId === workloadRequestId &&
      heldEvent.epoch === workloadEpoch)
  val dispatchAuthorized = eventValid && !eventDispatched && !reportValid &&
    !quarantine &&
    ((layerActive && activeEventIdentityMatches && activeStageLegal) ||
      (!layerActive && !heldNeedsGrant &&
        heldEvent.stage ===
          BitNetResidentLayerEventStage.FinalNormReady &&
        finalNormIdentityMatches))

  io.executionEventCommand.valid := dispatchAuthorized
  io.executionEventCommand.payload := heldEvent
  when(io.executionEventCommand.fire) { eventDispatched := True }

  // A pre-grant workload failure has a complete scheduler identity but no
  // legal weight slot.  It therefore retires only the waiting scheduler event.
  val waitingForGrant = eventValid && heldNeedsGrant && !layerActive &&
    !eventDispatched && !reportValid && !quarantine
  when(waitingForGrant && preGrantFailureValid) {
    val workloadIdentityMatches =
      preGrantFailure.requestId === heldEvent.requestId &&
        preGrantFailure.epoch === heldEvent.epoch
    val workloadFault = Bits(8 bits)
    workloadFault := preGrantFailure.faultCode
    when(!preGrantFailure.faultCode.orR) {
      workloadFault := B(
        BitNetResidentLayerWeightStagingBarrierFault.
          WorkloadFailedBeforeGrant, 8 bits)
    }
    val reportFault = Mux(workloadIdentityMatches, workloadFault,
      B(BitNetResidentLayerWeightStagingBarrierFault.
        WorkloadIdentity, 8 bits))
    fillEventReport(heldEvent, False, reportFault)
    reportNeedsCompute := False
    reportThenQuarantine := !workloadIdentityMatches
    reportEndsWorkload := True
    reportValid := True
    preGrantFailureValid := False
    grantValid := False
    rememberFault(reportFault)
  } elsewhen(waitingForGrant && grantValid) {
    when(grantEventIdentityMatches && layerGeometryMatches &&
        targetGeometryMatches) {
      activeGrant := heldGrant
      layerActive := True
      grantValid := False
    } otherwise {
      val grantFault = Bits(8 bits)
      grantFault := B(
        BitNetResidentLayerWeightStagingBarrierFault.GrantIdentity,
        8 bits)
      when(grantEventIdentityMatches &&
          !(layerGeometryMatches && targetGeometryMatches)) {
        grantFault := B(
          BitNetResidentLayerWeightStagingBarrierFault.GrantLayout,
          8 bits)
      }
      fillEventReport(heldEvent, False, grantFault)
      fillComputeReport(heldGrant, False, grantFault)
      reportNeedsCompute := True
      reportThenQuarantine := True
      reportEndsWorkload := False
      reportValid := True
      grantValid := False
      rememberFault(grantFault)
    }
  }

  // A command outside the two legal dispatch cases is an identity/order
  // violation.  If a weight slot is active, fail both owners atomically.
  val activeCommandInvalid = eventValid && !eventDispatched &&
    !heldNeedsGrant && layerActive &&
    (!activeEventIdentityMatches || !activeStageLegal) &&
    !reportValid && !quarantine
  val idleCommandInvalid = eventValid && !eventDispatched &&
    !heldNeedsGrant && !layerActive &&
    heldEvent.stage =/=
      BitNetResidentLayerEventStage.FinalNormReady &&
    !reportValid && !quarantine
  val finalNormIdentityInvalid = eventValid && !eventDispatched &&
    !heldNeedsGrant && !layerActive &&
    heldEvent.stage === BitNetResidentLayerEventStage.FinalNormReady &&
    !finalNormIdentityMatches && !reportValid && !quarantine

  when(activeCommandInvalid) {
    val code = B(
      BitNetResidentLayerWeightStagingBarrierFault.EventProtocol, 8 bits)
    fillEventReport(heldEvent, False, code)
    fillComputeReport(activeGrant, False, code)
    reportNeedsCompute := True
    reportThenQuarantine := True
    reportEndsWorkload := False
    reportValid := True
    rememberFault(code)
  } elsewhen(idleCommandInvalid || finalNormIdentityInvalid) {
    val code = B(
      BitNetResidentLayerWeightStagingBarrierFault.EventProtocol, 8 bits)
    fillEventReport(heldEvent, False, code)
    reportNeedsCompute := False
    reportThenQuarantine := True
    reportEndsWorkload := False
    reportValid := True
    rememberFault(code)
  }

  // ---------------------------------------------------------------------
  // Execution completion authentication and layer-terminal fork creation.
  // ---------------------------------------------------------------------
  val completionExpected = eventValid && eventDispatched && !reportValid &&
    !quarantine
  val unexpectedCompletion = io.executionDataCommit.valid &&
    !completionExpected
  io.executionDataCommit.ready := completionExpected

  val executionIdentityMatches =
    io.executionDataCommit.requestId === heldEvent.requestId &&
      io.executionDataCommit.layer === heldEvent.layer &&
      io.executionDataCommit.epoch === heldEvent.epoch &&
      io.executionDataCommit.payload.stage === heldEvent.stage &&
      io.executionDataCommit.slot === heldEvent.slot

  when(unexpectedCompletion) {
    val code = B(
      BitNetResidentLayerWeightStagingBarrierFault.UnexpectedCompletion,
      8 bits)
    rememberFault(code)
    quarantine := True
  }

  when(io.executionDataCommit.fire) {
    val childFault = Bits(8 bits)
    childFault := io.executionDataCommit.faultCode
    when(!io.executionDataCommit.faultCode.orR) {
      childFault := B(
        BitNetResidentLayerWeightStagingBarrierFault.ExecutionFailed,
        8 bits)
    }
    val completionFault = Mux(executionIdentityMatches, childFault,
      B(BitNetResidentLayerWeightStagingBarrierFault.
        CompletionIdentity, 8 bits))
    val terminalLayerCommit =
      heldEvent.stage === BitNetResidentLayerEventStage.LayerCommit
    val boundedSuccess = executionIdentityMatches &&
      io.executionDataCommit.success
    val requiresComputeReport = layerActive &&
      (terminalLayerCommit || !boundedSuccess)

    when(terminalLayerCommit && !layerActive) {
      val code = B(
        BitNetResidentLayerWeightStagingBarrierFault.EventProtocol,
        8 bits)
      fillEventReport(heldEvent, False, code)
      reportNeedsCompute := False
      reportThenQuarantine := True
      reportEndsWorkload := False
      rememberFault(code)
    } otherwise {
      val reportSuccess = boundedSuccess
      val reportFault = Bits(8 bits)
      reportFault := completionFault
      when(reportSuccess) { reportFault := 0 }
      fillEventReport(heldEvent, reportSuccess, reportFault)
      reportNeedsCompute := requiresComputeReport
      reportThenQuarantine := !executionIdentityMatches
      reportEndsWorkload :=
        heldEvent.stage ===
          BitNetResidentLayerEventStage.FinalNormReady &&
          finalNormIdentityMatches && executionIdentityMatches
      when(requiresComputeReport) {
        // Only a matching successful LayerCommit can successfully retire the
        // granted weight slot.  Every earlier termination is a failure.
        val computeSuccess = terminalLayerCommit && reportSuccess
        fillComputeReport(activeGrant, computeSuccess, reportFault)
      }
      when(!reportSuccess) { rememberFault(reportFault) }
    }
    reportValid := True
  }

  // ---------------------------------------------------------------------
  // Scheduler-only reports are ordinary buffered Streams.  A layer terminal
  // report is an atomic two-way fork with stable, registered payloads.
  // ---------------------------------------------------------------------
  val outwardClean = !quarantine && !unexpectedCompletion
  io.dataCommit.valid := reportValid && outwardClean &&
    (!reportNeedsCompute || io.computeCompletion.ready)
  io.dataCommit.payload := reportEvent
  io.computeCompletion.valid := reportValid && outwardClean &&
    reportNeedsCompute && io.dataCommit.ready
  io.computeCompletion.payload := reportCompute

  val schedulerOnlyFire = io.dataCommit.fire && !reportNeedsCompute
  val atomicForkFire = io.dataCommit.fire && io.computeCompletion.fire &&
    reportNeedsCompute
  val reportDone = schedulerOnlyFire || atomicForkFire

  when(reportDone) {
    reportValid := False
    eventValid := False
    eventDispatched := False
    heldNeedsGrant := False
    completedCount := completedCount + 1

    when(reportNeedsCompute) {
      layerActive := False
      when(reportCompute.success) {
        completedLayerCount := completedLayerCount + 1
      } otherwise {
        computeAbortPending := True
      }
    }
    when(reportEndsWorkload) {
      workloadIdentityValid := False
      computeAbortPending := False
    }
    when(reportThenQuarantine) { quarantine := True }
  }

  io.busy := eventValid || grantValid || layerActive || reportValid ||
    preGrantFailureValid || workloadForwardValid
  io.layerGrantActive := layerActive
  io.weightSlot := activeGrant.slot
  io.quarantined := quarantine
  io.protocolFaultSticky := stickyFault
  io.faultCode := stickyFaultCode
  io.acceptedEvents := acceptedCount
  io.completedEvents := completedCount
  io.completedLayers := completedLayerCount
}

object GenerateBitNetResidentLayerWeightStagingBarrier extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-layer-weight-staging-barrier")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLayerWeightStagingBarrier())
}
