package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentMatrixStageOwner extends SpinalEnum(binarySequential) {
  val Idle, SegmentedQkv, Direct, FusedGateUp, Rejected = newElement()
}

object BitNetResidentMatrixStageControlFault {
  val UnsupportedProjection = 0xc1
  val CompletionIdentity = 0xc2
  val GateDoneIdentity = 0xc3
  val StrayOwnerTraffic = 0xc4
}

case class BitNetResidentMatrixStageControlSubsystemConfig(
    weight: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig(),
    qkv: BitNetSegmentedQkvProductionBridgeConfig =
      BitNetSegmentedQkvProductionBridgeConfig(),
    gate: BitNetResidentGateUpLeasedGatePipelineConfig =
      BitNetResidentGateUpLeasedGatePipelineConfig()) {
  val stage: BitNetResidentMatrixStageLaunchExpanderConfig = weight.stage
  val model: BitNetConfig = stage.model
  val endpoint: BitNetResidentLinearEndpointConfig = qkv.endpoint
  val layout: BitNetStreamingActivationLayout =
    BitNetStreamingActivationLayout(model)

  require(qkv.stage == stage)
  require(qkv.endpoint.feeder == stage.feeder)
  require(gate.projection == weight)
  require(gate.join.endpoint == endpoint)
  require(gate.gate.gate.model == model)
}

/**
  * The descriptor-free MatrixArray scheduler and its mutually exclusive
  * physical-owner router.
  *
  * This component deliberately contains no arithmetic implementation.  The
  * three child-facing port groups are useful both as an auditable one-hot
  * ownership boundary and as a focused verification seam.  Production wraps
  * it with the segmented-QKV, direct O/Down and fused GateUp slices below.
  */
class BitNetResidentMatrixStageOwnerRouter(
    cfg: BitNetResidentMatrixStageControlSubsystemConfig =
      BitNetResidentMatrixStageControlSubsystemConfig()) extends Component {
  private val s = cfg.stage
  private val model = cfg.model
  private val endpoint = cfg.endpoint
  private val qkvCfg = cfg.qkv
  private val gateCfg = cfg.gate
  private val gateCoreCfg = gateCfg.gate.gate
  private val poolCfg = gateCfg.gate.arithmetic

  val io = new Bundle {
    val command = slave(Stream(BitNetResidentMatrixStageCommand(s)))
    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))

    // Child projection and retirement boundaries.
    val qkvProjection = master(Stream(
      BitNetResidentMatrixProjectionLaunch(s)))
    val qkvProjectionCompletion = slave(Stream(
      BitNetResidentMatrixProjectionCompletion(s)))
    val qkvSupported = in Bool()

    val directProjection = master(Stream(
      BitNetResidentMatrixProjectionLaunch(s)))
    val directProjectionCompletion = slave(Stream(
      BitNetResidentMatrixProjectionCompletion(s)))
    val directSupported = in Bool()

    val gateProjection = master(Stream(
      BitNetResidentMatrixProjectionLaunch(s)))
    val gateProjectionCompletion = slave(Stream(
      BitNetResidentMatrixProjectionCompletion(s)))
    val gateSupported = in Bool()

    // Sole external TP4 linear endpoint.
    val linearLaunch = master(Stream(BitNetResidentLinearLaunch(endpoint)))
    val linearResult = slave(Stream(
      BitNetResidentFinalFeaturePair(endpoint)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(endpoint)))
    val qkvLinearLaunch = slave(Stream(
      BitNetResidentLinearLaunch(endpoint)))
    val qkvLinearResult = master(Stream(
      BitNetResidentFinalFeaturePair(endpoint)))
    val qkvLinearCompletion = master(Stream(
      BitNetResidentLinearCompletion(endpoint)))
    val directLinearLaunch = slave(Stream(
      BitNetResidentLinearLaunch(endpoint)))
    val directLinearCompletion = master(Stream(
      BitNetResidentLinearCompletion(endpoint)))
    val gateLinearLaunch = slave(Stream(
      BitNetResidentLinearLaunch(endpoint)))
    val gateLinearResult = master(Stream(
      BitNetResidentFinalFeaturePair(endpoint)))
    val gateLinearCompletion = master(Stream(
      BitNetResidentLinearCompletion(endpoint)))

    // Segmented-QKV resident side.
    val ropeCoefficient = slave(Stream(
      BitNetQkvFeatureRopeCoefficient(qkvCfg.coefficient)))
    val qkvRound = master(Stream(
      BitNetResidentQkvRoundCommand(qkvCfg.resident)))
    val qkvDot = master(Stream(BitNetQkvTp4DotBeat(qkvCfg.qkv)))
    val qkvCompletion = slave(Stream(
      BitNetResidentQkvRoundCompletion(qkvCfg.resident)))
    val qkvChildRopeCoefficient = master(Stream(
      BitNetQkvFeatureRopeCoefficient(qkvCfg.coefficient)))
    val qkvChildRound = slave(Stream(
      BitNetResidentQkvRoundCommand(qkvCfg.resident)))
    val qkvChildDot = slave(Stream(BitNetQkvTp4DotBeat(qkvCfg.qkv)))
    val qkvChildCompletion = master(Stream(
      BitNetResidentQkvRoundCompletion(qkvCfg.resident)))

    // Fused GateUp command, exact gate and shared arithmetic-pool side.
    val gateCommand = slave(Stream(
      BitNetSharedFfnGateCommand(gateCoreCfg)))
    val gateChildCommand = master(Stream(
      BitNetSharedFfnGateCommand(gateCoreCfg)))
    val gateChildDone = slave(Stream(BitNetSharedFfnGateDone(gateCoreCfg)))
    val wideWrite = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(model, cfg.layout))),
      gateCoreCfg.tokenLanes)
    val gateChildWideWrite = Vec(slave(Stream(
      BitNetPhysicalActivationWriteBeat(model, cfg.layout))),
      gateCoreCfg.tokenLanes)
    val rmsCommand = master(Stream(
      BitNetSharedRmsCommand(gateCoreCfg.rmsConfig)))
    val gateChildRmsCommand = slave(Stream(
      BitNetSharedRmsCommand(gateCoreCfg.rmsConfig)))
    val poolAcquire = master(Stream(
      BitNetSpuArithmeticLeaseRequest(poolCfg)))
    val gateChildPoolAcquire = slave(Stream(
      BitNetSpuArithmeticLeaseRequest(poolCfg)))
    val poolOperand = master(Stream(
      BitNetSpuArithmeticOperandBeat(poolCfg)))
    val gateChildPoolOperand = slave(Stream(
      BitNetSpuArithmeticOperandBeat(poolCfg)))
    val poolResult = slave(Stream(
      BitNetSpuArithmeticResultBeat(poolCfg)))
    val gateChildPoolResult = master(Stream(
      BitNetSpuArithmeticResultBeat(poolCfg)))
    val poolRelease = master(Stream(
      BitNetSpuArithmeticLeaseRelease(poolCfg)))
    val gateChildPoolRelease = slave(Stream(
      BitNetSpuArithmeticLeaseRelease(poolCfg)))

    val activeOwner = out(BitNetResidentMatrixStageOwner())
    val busy = out Bool()
    val rejectedProjections = out UInt(32 bits)
    val protocolFaultSticky = out Bool()
  }

  val expander = new BitNetResidentMatrixStageLaunchExpander(s)
  expander.io.command << io.command
  io.completion << expander.io.completion

  val owner = Reg(BitNetResidentMatrixStageOwner()) init
    BitNetResidentMatrixStageOwner.Idle
  val held = Reg(BitNetResidentMatrixProjectionLaunch(s)) init
    BitNetResidentMatrixProjectionLaunch(s).getZero
  val stickyFault = RegInit(False)
  val rejectedCount = Reg(UInt(32 bits)) init 0
  val rejectedPending = RegInit(False)
  val rejectedCompletion = Reg(
    BitNetResidentMatrixProjectionCompletion(s)) init
      BitNetResidentMatrixProjectionCompletion(s).getZero
  val gateDoneSeen = RegInit(False)
  val gateDoneIdentityOk = RegInit(False)

  // The expander computes one immutable physical descriptor from the semantic
  // stage command.  Register it before route authentication so the geometry
  // checks and atomic child-ready fanout cannot feed back into the expander's
  // part selector in the same cycle.  The owner already permits only one
  // projection in flight, so this adds one launch cycle without changing the
  // projection throughput contract.
  val projection = expander.io.projection.m2sPipe()
  // The child `supported` predicates include geometry products and atomic
  // ready fanout.  Feeding those predicates straight back into
  // `projection.fire` makes the router owner CE sit at the end of one long
  // descriptor -> validator -> child-ready combinational path.  Sample the
  // predicates while the m2s projection register is holding an immutable
  // descriptor, then route from the sampled decision on the following cycle.
  // This is a narrow control-only timing cut: it adds one validation cycle and
  // four FFs, while the wide projection payload remains registered exactly
  // once and stable until it is accepted.
  val routeCheckValid = RegInit(False)
  val qkvSupportedChecked = RegInit(False)
  val directSupportedChecked = RegInit(False)
  val gateSupportedChecked = RegInit(False)

  when(!projection.valid) {
    routeCheckValid := False
  } elsewhen(!routeCheckValid) {
    qkvSupportedChecked := io.qkvSupported
    directSupportedChecked := io.directSupported
    gateSupportedChecked := io.gateSupported
    routeCheckValid := True
  }

  val ownerIdle = owner === BitNetResidentMatrixStageOwner.Idle
  val qkvCategory = projection.payload.stage ===
    BitNetResidentLayerStage.QkvProjection &&
    (projection.payload.part === BitNetResidentMatrixProjectionPart.Query ||
      projection.payload.part === BitNetResidentMatrixProjectionPart.Key ||
      projection.payload.part === BitNetResidentMatrixProjectionPart.Value)
  val directCategory =
    (projection.payload.stage ===
      BitNetResidentLayerStage.AttentionOutputProjection &&
      projection.payload.part ===
        BitNetResidentMatrixProjectionPart.AttentionOutput) ||
    (projection.payload.stage === BitNetResidentLayerStage.DownProjection &&
      projection.payload.part === BitNetResidentMatrixProjectionPart.Down)
  val gateCategory = projection.payload.stage ===
    BitNetResidentLayerStage.GateUpProjection &&
    projection.payload.part ===
      BitNetResidentMatrixProjectionPart.FusedGateUp

  val qkvRoute = routeCheckValid && qkvCategory && qkvSupportedChecked
  val directRoute =
    routeCheckValid && directCategory && directSupportedChecked
  val gateRoute = routeCheckValid && gateCategory && gateSupportedChecked
  val knownCategory = qkvCategory || directCategory || gateCategory
  // GateUp waits for its paired scale command.  Only a present but rejected
  // command may turn into an immediate fail-closed projection completion.
  val rejectedRoute =
    routeCheckValid && ((qkvCategory && !qkvSupportedChecked) ||
      (directCategory && !directSupportedChecked) ||
      (!knownCategory) ||
      (gateCategory && io.gateCommand.valid && !gateSupportedChecked))

  io.qkvProjection.payload := projection.payload
  io.directProjection.payload := projection.payload
  io.gateProjection.payload := projection.payload
  io.qkvProjection.valid := projection.valid && ownerIdle && qkvRoute
  io.directProjection.valid := projection.valid && ownerIdle && directRoute
  io.gateProjection.valid := projection.valid && ownerIdle && gateRoute

  io.gateChildCommand.payload := io.gateCommand.payload
  io.gateChildCommand.valid := projection.valid && ownerIdle && gateRoute &&
    io.gateCommand.valid
  io.gateCommand.ready := False
  when(projection.valid && routeCheckValid && ownerIdle && gateCategory) {
    when(gateRoute) {
      io.gateCommand.ready := io.gateChildCommand.ready
    } elsewhen(io.gateCommand.valid && !rejectedPending) {
      io.gateCommand.ready := True
    }
  }

  projection.ready := False
  when(ownerIdle) {
    when(qkvRoute) {
      projection.ready := io.qkvProjection.ready
    } elsewhen(directRoute) {
      projection.ready := io.directProjection.ready
    } elsewhen(gateRoute) {
      projection.ready := io.gateProjection.ready
    } elsewhen(rejectedRoute && !rejectedPending) {
      projection.ready := True
    }
  }

  when(projection.fire) {
    routeCheckValid := False
    held := projection.payload
    gateDoneSeen := False
    gateDoneIdentityOk := False
    when(qkvRoute) {
      owner := BitNetResidentMatrixStageOwner.SegmentedQkv
    } elsewhen(directRoute) {
      owner := BitNetResidentMatrixStageOwner.Direct
    } elsewhen(gateRoute) {
      owner := BitNetResidentMatrixStageOwner.FusedGateUp
    } otherwise {
      owner := BitNetResidentMatrixStageOwner.Rejected
      rejectedCompletion.requestId := projection.payload.requestId
      rejectedCompletion.stepId := projection.payload.stepId
      rejectedCompletion.stage := projection.payload.stage
      rejectedCompletion.layer := projection.payload.layer
      rejectedCompletion.part := projection.payload.part
      rejectedCompletion.sequenceId := projection.payload.sequenceId
      rejectedCompletion.success := False
      rejectedCompletion.faultCode := B(
        BitNetResidentMatrixStageControlFault.UnsupportedProjection,
        8 bits)
      rejectedPending := True
      rejectedCount := rejectedCount + 1
      stickyFault := True
    }
  }

  /* ------------------------------------------------------------------
   * Authenticate exactly one selected child completion.
   * ------------------------------------------------------------------ */
  val selectedCompletion = BitNetResidentMatrixProjectionCompletion(s)
  selectedCompletion := rejectedCompletion
  val selectedValid = Bool()
  selectedValid := rejectedPending
  io.qkvProjectionCompletion.ready := False
  io.directProjectionCompletion.ready := False
  io.gateProjectionCompletion.ready := False

  switch(owner) {
    is(BitNetResidentMatrixStageOwner.SegmentedQkv) {
      selectedValid := io.qkvProjectionCompletion.valid
      selectedCompletion := io.qkvProjectionCompletion.payload
    }
    is(BitNetResidentMatrixStageOwner.Direct) {
      selectedValid := io.directProjectionCompletion.valid
      selectedCompletion := io.directProjectionCompletion.payload
    }
    is(BitNetResidentMatrixStageOwner.FusedGateUp) {
      selectedValid := io.gateProjectionCompletion.valid &&
        (gateDoneSeen || io.gateChildDone.fire)
      selectedCompletion := io.gateProjectionCompletion.payload
    }
    is(BitNetResidentMatrixStageOwner.Rejected) {
      selectedValid := rejectedPending
      selectedCompletion := rejectedCompletion
    }
  }

  val completionIdentityMatches =
    selectedCompletion.requestId === held.requestId &&
      selectedCompletion.stepId === held.stepId &&
      selectedCompletion.stage === held.stage &&
      selectedCompletion.layer === held.layer &&
      selectedCompletion.part === held.part &&
      selectedCompletion.sequenceId === held.sequenceId
  val gateDoneOkAfter = gateDoneIdentityOk ||
    (io.gateChildDone.fire &&
      io.gateChildDone.tag === held.stepId.resize(gateCoreCfg.tagWidth) &&
      io.gateChildDone.length === model.ffnSize &&
      !io.gateChildDone.protocolFault)

  expander.io.projectionCompletion.valid := selectedValid
  expander.io.projectionCompletion.payload := selectedCompletion
  when(selectedValid && !completionIdentityMatches) {
    expander.io.projectionCompletion.payload.requestId := held.requestId
    expander.io.projectionCompletion.payload.stepId := held.stepId
    expander.io.projectionCompletion.payload.stage := held.stage
    expander.io.projectionCompletion.payload.layer := held.layer
    expander.io.projectionCompletion.payload.part := held.part
    expander.io.projectionCompletion.payload.sequenceId := held.sequenceId
    expander.io.projectionCompletion.payload.success := False
    expander.io.projectionCompletion.payload.faultCode := B(
      BitNetResidentMatrixStageControlFault.CompletionIdentity, 8 bits)
  }
  when(owner === BitNetResidentMatrixStageOwner.FusedGateUp &&
      selectedValid && !gateDoneOkAfter) {
    expander.io.projectionCompletion.payload.success := False
    expander.io.projectionCompletion.payload.faultCode := B(
      BitNetResidentMatrixStageControlFault.GateDoneIdentity, 8 bits)
  }

  switch(owner) {
    is(BitNetResidentMatrixStageOwner.SegmentedQkv) {
      io.qkvProjectionCompletion.ready :=
        expander.io.projectionCompletion.ready
    }
    is(BitNetResidentMatrixStageOwner.Direct) {
      io.directProjectionCompletion.ready :=
        expander.io.projectionCompletion.ready
    }
    is(BitNetResidentMatrixStageOwner.FusedGateUp) {
      io.gateProjectionCompletion.ready :=
        expander.io.projectionCompletion.ready &&
          (gateDoneSeen || io.gateChildDone.fire)
    }
  }

  when(expander.io.projectionCompletion.fire) {
    when(!completionIdentityMatches ||
        (owner === BitNetResidentMatrixStageOwner.FusedGateUp &&
          !gateDoneOkAfter)) {
      stickyFault := True
    }
    when(owner === BitNetResidentMatrixStageOwner.Rejected) {
      rejectedPending := False
    }
    owner := BitNetResidentMatrixStageOwner.Idle
  }

  io.gateChildDone.ready :=
    owner === BitNetResidentMatrixStageOwner.FusedGateUp && !gateDoneSeen
  when(io.gateChildDone.fire) {
    gateDoneSeen := True
    gateDoneIdentityOk :=
      io.gateChildDone.tag === held.stepId.resize(gateCoreCfg.tagWidth) &&
        io.gateChildDone.length === model.ffnSize &&
        !io.gateChildDone.protocolFault
    when(io.gateChildDone.tag =/=
        held.stepId.resize(gateCoreCfg.tagWidth) ||
        io.gateChildDone.length =/= model.ffnSize ||
        io.gateChildDone.protocolFault) {
      stickyFault := True
    }
  }

  /* ------------------------------------------------------------------
   * One-hot TP4 linear endpoint.
   * ------------------------------------------------------------------ */
  io.linearLaunch.valid := False
  io.linearLaunch.payload := io.qkvLinearLaunch.payload
  io.qkvLinearLaunch.ready := False
  io.directLinearLaunch.ready := False
  io.gateLinearLaunch.ready := False
  switch(owner) {
    is(BitNetResidentMatrixStageOwner.SegmentedQkv) {
      io.linearLaunch.valid := io.qkvLinearLaunch.valid
      io.linearLaunch.payload := io.qkvLinearLaunch.payload
      io.qkvLinearLaunch.ready := io.linearLaunch.ready
    }
    is(BitNetResidentMatrixStageOwner.Direct) {
      io.linearLaunch.valid := io.directLinearLaunch.valid
      io.linearLaunch.payload := io.directLinearLaunch.payload
      io.directLinearLaunch.ready := io.linearLaunch.ready
    }
    is(BitNetResidentMatrixStageOwner.FusedGateUp) {
      io.linearLaunch.valid := io.gateLinearLaunch.valid
      io.linearLaunch.payload := io.gateLinearLaunch.payload
      io.gateLinearLaunch.ready := io.linearLaunch.ready
    }
  }

  io.qkvLinearResult.valid := False
  io.qkvLinearResult.payload := io.linearResult.payload
  io.gateLinearResult.valid := False
  io.gateLinearResult.payload := io.linearResult.payload
  io.linearResult.ready := False
  switch(owner) {
    is(BitNetResidentMatrixStageOwner.SegmentedQkv) {
      io.qkvLinearResult.valid := io.linearResult.valid
      io.linearResult.ready := io.qkvLinearResult.ready
    }
    is(BitNetResidentMatrixStageOwner.FusedGateUp) {
      io.gateLinearResult.valid := io.linearResult.valid
      io.linearResult.ready := io.gateLinearResult.ready
    }
    is(BitNetResidentMatrixStageOwner.Direct) {
      // O/Down terminate inside the compute island's fused residual endpoint.
      // Any outward pair is an invariant violation, but draining it prevents
      // a corrupt producer from holding the sole linear endpoint forever.
      io.linearResult.ready := True
      when(io.linearResult.fire) { stickyFault := True }
    }
  }

  io.qkvLinearCompletion.valid := False
  io.qkvLinearCompletion.payload := io.linearCompletion.payload
  io.directLinearCompletion.valid := False
  io.directLinearCompletion.payload := io.linearCompletion.payload
  io.gateLinearCompletion.valid := False
  io.gateLinearCompletion.payload := io.linearCompletion.payload
  io.linearCompletion.ready := False
  switch(owner) {
    is(BitNetResidentMatrixStageOwner.SegmentedQkv) {
      io.qkvLinearCompletion.valid := io.linearCompletion.valid
      io.linearCompletion.ready := io.qkvLinearCompletion.ready
    }
    is(BitNetResidentMatrixStageOwner.Direct) {
      io.directLinearCompletion.valid := io.linearCompletion.valid
      io.linearCompletion.ready := io.directLinearCompletion.ready
    }
    is(BitNetResidentMatrixStageOwner.FusedGateUp) {
      io.gateLinearCompletion.valid := io.linearCompletion.valid
      io.linearCompletion.ready := io.gateLinearCompletion.ready
    }
  }

  /* ------------------------------------------------------------------
   * Resident QKV side is open only for a selected Q/K/V projection.
   * ------------------------------------------------------------------ */
  val qkvOwns = owner === BitNetResidentMatrixStageOwner.SegmentedQkv
  io.qkvChildRopeCoefficient.valid := io.ropeCoefficient.valid && qkvOwns
  io.qkvChildRopeCoefficient.payload := io.ropeCoefficient.payload
  io.ropeCoefficient.ready := io.qkvChildRopeCoefficient.ready && qkvOwns
  io.qkvRound.valid := io.qkvChildRound.valid && qkvOwns
  io.qkvRound.payload := io.qkvChildRound.payload
  io.qkvChildRound.ready := io.qkvRound.ready && qkvOwns
  io.qkvDot.valid := io.qkvChildDot.valid && qkvOwns
  io.qkvDot.payload := io.qkvChildDot.payload
  io.qkvChildDot.ready := io.qkvDot.ready && qkvOwns
  io.qkvChildCompletion.valid := io.qkvCompletion.valid && qkvOwns
  io.qkvChildCompletion.payload := io.qkvCompletion.payload
  io.qkvCompletion.ready := io.qkvChildCompletion.ready && qkvOwns

  /* ------------------------------------------------------------------
   * Leased exact-gate side is open only for the fused GateUp owner.
   * ------------------------------------------------------------------ */
  val gateOwns = owner === BitNetResidentMatrixStageOwner.FusedGateUp
  for (lane <- 0 until gateCoreCfg.tokenLanes) {
    io.wideWrite(lane).valid := io.gateChildWideWrite(lane).valid && gateOwns
    io.wideWrite(lane).payload := io.gateChildWideWrite(lane).payload
    io.gateChildWideWrite(lane).ready := io.wideWrite(lane).ready && gateOwns
  }
  io.rmsCommand.valid := io.gateChildRmsCommand.valid && gateOwns
  io.rmsCommand.payload := io.gateChildRmsCommand.payload
  io.gateChildRmsCommand.ready := io.rmsCommand.ready && gateOwns
  io.poolAcquire.valid := io.gateChildPoolAcquire.valid && gateOwns
  io.poolAcquire.payload := io.gateChildPoolAcquire.payload
  io.gateChildPoolAcquire.ready := io.poolAcquire.ready && gateOwns
  io.poolOperand.valid := io.gateChildPoolOperand.valid && gateOwns
  io.poolOperand.payload := io.gateChildPoolOperand.payload
  io.gateChildPoolOperand.ready := io.poolOperand.ready && gateOwns
  io.gateChildPoolResult.valid := io.poolResult.valid && gateOwns
  io.gateChildPoolResult.payload := io.poolResult.payload
  io.poolResult.ready := io.gateChildPoolResult.ready && gateOwns
  io.poolRelease.valid := io.gateChildPoolRelease.valid && gateOwns
  io.poolRelease.payload := io.gateChildPoolRelease.payload
  io.gateChildPoolRelease.ready := io.poolRelease.ready && gateOwns

  val unselectedLinearTraffic =
    (!qkvOwns && io.qkvLinearLaunch.valid) ||
      (owner =/= BitNetResidentMatrixStageOwner.Direct &&
        io.directLinearLaunch.valid) ||
      (!gateOwns && io.gateLinearLaunch.valid)
  when(unselectedLinearTraffic || (io.gateChildDone.valid && !gateOwns)) {
    stickyFault := True
  }

  io.activeOwner := owner
  io.busy := expander.io.busy || owner =/=
    BitNetResidentMatrixStageOwner.Idle || rejectedPending
  io.rejectedProjections := rejectedCount
  io.protocolFaultSticky := stickyFault || expander.io.protocolFaultSticky
}

/**
  * Production MatrixArray stage subsystem.  It instantiates one scheduler and
  * exactly the three control slices; the TP4 cluster, memory crossbar,
  * arithmetic pool and activation memories remain unique external owners.
  */
class BitNetResidentMatrixStageControlSubsystem(
    cfg: BitNetResidentMatrixStageControlSubsystemConfig =
      BitNetResidentMatrixStageControlSubsystemConfig(),
    externalQkvProjectionPreprocess: Boolean = false) extends Component {
  private val model = cfg.model
  private val endpoint = cfg.endpoint
  private val prefetch = cfg.weight.prefetch
  private val gateCoreCfg = cfg.gate.gate.gate
  private val poolCfg = cfg.gate.gate.arithmetic

  val io = new Bundle {
    val command = slave(Stream(
      BitNetResidentMatrixStageCommand(cfg.stage)))
    /** The only public stage retirement. */
    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))

    /**
      * Optional typed in-series seam between the QKV stage owner and the
      * segmented bridge.  The resident full-layer shell inserts the RoPE-row
      * stager here so it can authenticate and echo the actual MatrixStage
      * sequenceId; stand-alone/OOC users retain the original direct wiring.
      */
    val qkvProjectionBeforePreprocess = if (externalQkvProjectionPreprocess)
      master(Stream(BitNetResidentMatrixProjectionLaunch(cfg.stage)))
    else null
    val qkvProjectionAfterPreprocess = if (externalQkvProjectionPreprocess)
      slave(Stream(BitNetResidentMatrixProjectionLaunch(cfg.stage)))
    else null

    val linearLaunch = master(Stream(BitNetResidentLinearLaunch(endpoint)))
    val linearResult = slave(Stream(
      BitNetResidentFinalFeaturePair(endpoint)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(endpoint)))
    val weightCommand = master(Stream(
      BitNetFiveBankStripedWeightCommand(prefetch)))
    val prefetchCompletion = slave(Stream(
      BitNetFiveBankStripedWeightCompletion(prefetch)))

    val ropeCoefficient = slave(Stream(
      BitNetQkvFeatureRopeCoefficient(cfg.qkv.coefficient)))
    val qkvRound = master(Stream(
      BitNetResidentQkvRoundCommand(cfg.qkv.resident)))
    val qkvDot = master(Stream(BitNetQkvTp4DotBeat(cfg.qkv.qkv)))
    val qkvCompletion = slave(Stream(
      BitNetResidentQkvRoundCompletion(cfg.qkv.resident)))

    val gateCommand = slave(Stream(
      BitNetSharedFfnGateCommand(gateCoreCfg)))
    val wideWrite = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(model, cfg.layout))),
      gateCoreCfg.tokenLanes)
    val rmsCommand = master(Stream(
      BitNetSharedRmsCommand(gateCoreCfg.rmsConfig)))
    val poolAcquire = master(Stream(
      BitNetSpuArithmeticLeaseRequest(poolCfg)))
    val poolOperand = master(Stream(
      BitNetSpuArithmeticOperandBeat(poolCfg)))
    val poolResult = slave(Stream(
      BitNetSpuArithmeticResultBeat(poolCfg)))
    val poolRelease = master(Stream(
      BitNetSpuArithmeticLeaseRelease(poolCfg)))

    val activeOwner = out(BitNetResidentMatrixStageOwner())
    val busy = out Bool()
    val dependentFailureDrainActive = out Bool()
    val fatalQkvRoundPoisoned = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val router = new BitNetResidentMatrixStageOwnerRouter(cfg)
  val qkv = new BitNetResidentSegmentedQkvProjectionControlSlice(
    cfg.weight, cfg.qkv)
  val direct = new BitNetResidentMatrixDirectProjectionControlSlice(
    cfg.weight)
  val gate = new BitNetResidentGateUpLeasedGatePipeline(cfg.gate)
  val sharedWeights = new BitNetResidentSharedMatrixWeightService(cfg.weight)

  router.io.command << io.command
  io.completion << router.io.completion

  if (externalQkvProjectionPreprocess) {
    io.qkvProjectionBeforePreprocess << router.io.qkvProjection
    qkv.io.projection << io.qkvProjectionAfterPreprocess
  } else {
    qkv.io.projection << router.io.qkvProjection
  }
  router.io.qkvProjectionCompletion << qkv.io.projectionCompletion
  router.io.qkvSupported := qkv.io.supported
  direct.io.projection << router.io.directProjection
  router.io.directProjectionCompletion << direct.io.completion
  router.io.directSupported := direct.io.supported
  gate.io.projection << router.io.gateProjection
  router.io.gateProjectionCompletion << gate.io.projectionCompletion
  router.io.gateSupported := gate.io.supported

  io.linearLaunch << router.io.linearLaunch
  router.io.linearResult << io.linearResult
  router.io.linearCompletion << io.linearCompletion
  router.io.qkvLinearLaunch << qkv.io.linearLaunch
  qkv.io.linearResult << router.io.qkvLinearResult
  qkv.io.linearCompletion << router.io.qkvLinearCompletion
  router.io.directLinearLaunch << direct.io.linearLaunch
  direct.io.linearCompletion << router.io.directLinearCompletion
  router.io.gateLinearLaunch << gate.io.linearLaunch
  gate.io.linearResult << router.io.gateLinearResult
  gate.io.linearCompletion << router.io.gateLinearCompletion

  sharedWeights.io.qkvProjection << qkv.io.weightProjection
  qkv.io.weightRetirement << sharedWeights.io.qkvRetirement
  sharedWeights.io.directProjection << direct.io.weightProjection
  direct.io.weightRetirement << sharedWeights.io.directRetirement
  sharedWeights.io.gateProjection << gate.io.weightProjection
  gate.io.weightRetirement << sharedWeights.io.gateRetirement
  io.weightCommand << sharedWeights.io.command
  sharedWeights.io.prefetchCompletion << io.prefetchCompletion

  router.io.ropeCoefficient << io.ropeCoefficient
  qkv.io.ropeCoefficient << router.io.qkvChildRopeCoefficient
  router.io.qkvChildRound << qkv.io.qkvRound
  io.qkvRound << router.io.qkvRound
  router.io.qkvChildDot << qkv.io.qkvDot
  io.qkvDot << router.io.qkvDot
  router.io.qkvCompletion << io.qkvCompletion
  qkv.io.qkvCompletion << router.io.qkvChildCompletion

  router.io.gateCommand << io.gateCommand
  gate.io.gateCommand << router.io.gateChildCommand
  router.io.gateChildDone << gate.io.done
  for (lane <- 0 until gateCoreCfg.tokenLanes) {
    router.io.gateChildWideWrite(lane) << gate.io.wideWrite(lane)
    io.wideWrite(lane) << router.io.wideWrite(lane)
  }
  router.io.gateChildRmsCommand << gate.io.rmsCommand
  io.rmsCommand << router.io.rmsCommand
  router.io.gateChildPoolAcquire << gate.io.poolAcquire
  io.poolAcquire << router.io.poolAcquire
  router.io.gateChildPoolOperand << gate.io.poolOperand
  io.poolOperand << router.io.poolOperand
  router.io.poolResult << io.poolResult
  gate.io.poolResult << router.io.gateChildPoolResult
  router.io.gateChildPoolRelease << gate.io.poolRelease
  io.poolRelease << router.io.poolRelease

  io.activeOwner := router.io.activeOwner
  io.busy := router.io.busy || qkv.io.busy || direct.io.busy || gate.io.busy ||
    sharedWeights.io.busy
  io.dependentFailureDrainActive := qkv.io.dependentFailureDrainActive ||
    gate.io.dependentFailureDrainActive
  io.fatalQkvRoundPoisoned := qkv.io.fatalRoundPoisoned
  io.protocolFaultSticky := router.io.protocolFaultSticky ||
    qkv.io.protocolFaultSticky || direct.io.protocolFaultSticky ||
    gate.io.protocolFaultSticky || sharedWeights.io.protocolFaultSticky
}

object GenerateBitNetResidentMatrixStageControlSubsystem extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-matrix-stage-control-subsystem")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentMatrixStageControlSubsystem())
}
