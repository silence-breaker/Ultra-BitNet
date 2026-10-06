package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Geometry shared by the matrix owner, resident image and sole TP4 feeder. */
case class BitNetResidentActivationReplayBinderConfig(
    linear: BitNetResidentLinearEndpointConfig =
      BitNetResidentLinearEndpointConfig(),
    activation: BitNetResidentProjectionActivationSubsystemConfig =
      BitNetResidentProjectionActivationSubsystemConfig()
) {
  val model: BitNetConfig = linear.model
  val tile: BitNetResidentActivationTileBufferConfig = activation.tile
  val stepIdWidth: Int = BitNetResidentLayerGeometry.stepIdWidth(model)
  val launchCountWidth: Int = 2

  require(activation.model == model)
  require(activation.tile.feeder == linear.feeder,
    "resident image tiles must target the sole production feeder")
  require(activation.quant.tagWidth == linear.matrix.sequenceIdWidth)
  require(tile.slotCount == 2 &&
    (tile.tileFeatures == 64 || tile.tileFeatures == 128))
  require(model.hiddenSize == tile.hiddenFeatures)
  require(model.ffnSize == tile.ffnFeatures)
}

/** Explicit lifetime binding between one RMS/quant image and its projections.
  *
  * `activationReuseTag == stepId` names the consuming matrix transaction.
  * `imageTag` names the physical producer: normally the same event, except
  * that Down consumes the image produced by the immediately preceding GateUp
  * event.  Epoch and slot protect reuse of the ping/pong storage.
  */
case class BitNetResidentActivationImageBinding(
    cfg: BitNetResidentActivationReplayBinderConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(cfg.stepIdWidth bits)
  val activationReuseTag = UInt(cfg.stepIdWidth bits)
  val imageTag = UInt(cfg.activation.quant.tagWidth bits)
  val epoch = UInt(cfg.tile.epochWidth bits)
  val slot = UInt(cfg.tile.slotWidth bits)
  val inputFeatures = UInt(cfg.tile.featureCountWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  /** Three means Q -> K -> V; one means one ordinary projection. */
  val expectedLaunchCount = UInt(cfg.launchCountWidth bits)
}

case class BitNetResidentActivationBindingCompletion(
    cfg: BitNetResidentActivationReplayBinderConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(cfg.stepIdWidth bits)
  val activationReuseTag = UInt(cfg.stepIdWidth bits)
  val imageTag = UInt(cfg.activation.quant.tagWidth bits)
  val epoch = UInt(cfg.tile.epochWidth bits)
  val slot = UInt(cfg.tile.slotWidth bits)
  val inputFeatures = UInt(cfg.tile.featureCountWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val expectedLaunchCount = UInt(cfg.launchCountWidth bits)
  val completedLaunchCount = UInt(cfg.launchCountWidth bits)
  val success = Bool()
  val faultCode = Bits(16 bits)
}

object BitNetResidentActivationReplayBinderFault {
  val InvalidBinding = 0
  val InvalidLaunch = 1
  val TileIdentity = 2
  val ComputeIdentity = 3
  val ComputeFailed = 4
  val ReplayIdentity = 5
  val ReplayFailed = 6
  val EarlyCompletion = 7
  val DuplicateCompletion = 8
  val StrayCompletion = 9
  val TileImageIdentity = 10
  val TileTokenMask = 11
  val TileSequence = 12
  val TileOrdinal = 13
  val TileFeature = 14
  val TileFraming = 15
}

/**
  * Binds each resident linear transaction to one authenticated i8 image replay.
  *
  * The block contains no tile buffer, quantizer, feeder or compute island.  A
  * held MatrixStageControl launch is atomically forked to the external compute
  * endpoint and to the resident image replay boundary.  Activation tiles are
  * authenticated and lowered to the existing normalized feeder type.
  * Arithmetic completion is withheld until both the real compute completion
  * and the real replay completion retire with the held identity.
  *
  * A failed side never releases the transaction early.  Replay failure causes
  * poisoned zero tiles to finish the compute-side fixed-length drain; compute
  * failure causes the remaining real image tiles to be consumed locally.  Q,
  * K and V can therefore reuse one binding exactly three times without a
  * software callback or another activation store.
  */
class BitNetResidentActivationReplayBinder(
    cfg: BitNetResidentActivationReplayBinderConfig =
      BitNetResidentActivationReplayBinderConfig()) extends Component {
  private val model = cfg.model
  private val linear = cfg.linear
  private val matrix = linear.matrix
  private val tileCfg = cfg.tile

  val io = new Bundle {
    val binding = slave(Stream(
      BitNetResidentActivationImageBinding(cfg)))

    // MatrixStageControl-facing transaction boundary.
    val linearLaunch = slave(Stream(BitNetResidentLinearLaunch(linear)))
    val linearCompletion = master(Stream(
      BitNetResidentLinearCompletion(linear)))

    // Sole TP4 compute/feeder boundary.  The matrix launch has already passed
    // the complete legacy-to-compact authentication boundary below, so only
    // the canonical program crosses the dependency-breaking composition seam.
    val computeLaunch = master(Stream(BitNetResidentLinearProgram(linear)))
    val activation = master(Stream(
      BitNetNormalizedActivationTile(linear.feeder)))
    val computeCompletion = slave(Stream(
      BitNetResidentLinearCompletion(linear)))

    // Public boundary of BitNetResidentProjectionActivationSubsystem.
    val replayCommand = master(Stream(
      BitNetResidentActivationReplayCommand(tileCfg)))
    val tile = slave(Stream(BitNetResidentActivationTile(tileCfg)))
    val replayCompletion = slave(Stream(
      BitNetResidentActivationReplayCompletion(tileCfg)))

    val bindingCompletion = master(Stream(
      BitNetResidentActivationBindingCompletion(cfg)))

    val busy = out Bool()
    val launchOrdinal = out UInt(cfg.launchCountWidth bits)
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(16 bits)
    val completedBindings = out UInt(32 bits)
    val completedTransactions = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val NoBinding, AwaitLaunch, ValidateLaunch, IssueBoth, Active,
        ReportLinear, ReportBinding = newElement()
  }
  val state = Reg(State()) init State.NoBinding

  val heldBinding = Reg(BitNetResidentActivationImageBinding(cfg)) init
    BitNetResidentActivationImageBinding(cfg).getZero
  val heldProgram = Reg(BitNetResidentLinearProgram(linear)) init
    BitNetResidentLinearProgram(linear).getZero
  // Internal replay/count/identity logic consumes the canonical expansion as
  // wires.  No second legacy launch image is retained in this component.
  val heldLaunch = BitNetResidentLinearProgramCodec.expand(
    heldProgram, linear)
  val launchOrdinalReg = Reg(UInt(cfg.launchCountWidth bits)) init 0
  val firstSequenceId = Reg(UInt(matrix.sequenceIdWidth bits)) init 0
  val firstLayer = Reg(UInt(model.layerWidth bits)) init 0
  val firstMode = Reg(BitNetMode()) init BitNetMode.Decode

  val bindingFault = Reg(Bits(16 bits)) init 0
  val transactionFault = Reg(Bits(16 bits)) init 0
  val stickyFault = RegInit(False)
  // Fault reporting is diagnostic and does not participate in transaction
  // control.  Keep each detection point local for one cycle before the
  // sticky reduction; otherwise synthesis builds one cross-block D cone from
  // every identity/protocol check into stickyFault.
  val stickyFaultEvents = Bits(17 bits)
  stickyFaultEvents := 0
  val stickyFaultEventsPipe = RegNext(stickyFaultEvents) init 0
  when(stickyFaultEventsPipe.orR) {
    stickyFault := True
  }
  val completedBindingCounter = Reg(UInt(32 bits)) init 0
  val completedTransactionCounter = Reg(UInt(32 bits)) init 0

  val linearReport = Reg(BitNetResidentLinearCompletion(linear)) init
    BitNetResidentLinearCompletion(linear).getZero
  val bindingReport = Reg(
    BitNetResidentActivationBindingCompletion(cfg)) init
      BitNetResidentActivationBindingCompletion(cfg).getZero

  // -----------------------------------------------------------------------
  // Binding admission: the consumer identity and physical image identity are
  // deliberately separate.  Hidden-size images are produced and consumed by
  // the same matrix event, so both tags match.  The FFN-size Down image is
  // produced by the immediately preceding GateUp event; it keeps GateUp's
  // physical imageTag while Down owns activationReuseTag/stepId.  Conflating
  // those two identities makes a real GateUp -> Down chain impossible.
  // -----------------------------------------------------------------------
  val bindingPrefixMask = (1 to model.tokenParallelism).map { count =>
    io.binding.payload.tokenMask === B((BigInt(1) << count) - 1,
      model.tokenParallelism bits)
  }.reduce(_ || _)
  val bindingFeatureCount =
    io.binding.payload.inputFeatures === model.hiddenSize ||
    io.binding.payload.inputFeatures === model.ffnSize
  val bindingLaunchCount =
    io.binding.payload.expectedLaunchCount === 1 ||
    io.binding.payload.expectedLaunchCount === 3
  val bindingConsumerTag =
    io.binding.payload.stepId === io.binding.payload.activationReuseTag
  val previousConsumerTag =
    (io.binding.payload.activationReuseTag - 1).resize(
      cfg.activation.quant.tagWidth)
  val bindingImageTag =
    (io.binding.payload.inputFeatures === model.hiddenSize &&
      io.binding.payload.imageTag ===
        io.binding.payload.activationReuseTag.resized) ||
      (io.binding.payload.inputFeatures === model.ffnSize &&
        io.binding.payload.activationReuseTag =/= 0 &&
        io.binding.payload.imageTag === previousConsumerTag)
  val bindingMultiShape =
    io.binding.payload.expectedLaunchCount =/= 3 ||
      io.binding.payload.inputFeatures === model.hiddenSize
  val bindingLegal = bindingPrefixMask && bindingFeatureCount &&
    bindingLaunchCount && bindingConsumerTag && bindingImageTag &&
    bindingMultiShape

  io.binding.ready := state === State.NoBinding

  def fillBindingReport(
      binding: BitNetResidentActivationImageBinding): Unit = {
    bindingReport.requestId := binding.requestId
    bindingReport.stepId := binding.stepId
    bindingReport.activationReuseTag := binding.activationReuseTag
    bindingReport.imageTag := binding.imageTag
    bindingReport.epoch := binding.epoch
    bindingReport.slot := binding.slot
    bindingReport.inputFeatures := binding.inputFeatures
    bindingReport.tokenMask := binding.tokenMask
    bindingReport.expectedLaunchCount := binding.expectedLaunchCount
  }

  when(io.binding.fire) {
    heldBinding := io.binding.payload
    launchOrdinalReg := 0
    bindingFault := 0
    transactionFault := 0
    fillBindingReport(io.binding.payload)
    bindingReport.completedLaunchCount := 0
    bindingReport.success := False
    bindingReport.faultCode := 0
    when(bindingLegal) {
      state := State.AwaitLaunch
    } otherwise {
      val invalid = B(
        BigInt(1) << BitNetResidentActivationReplayBinderFault.InvalidBinding,
        16 bits)
      bindingFault := invalid
      bindingReport.success := False
      bindingReport.faultCode := invalid
      stickyFaultEvents(0) := True
      state := State.ReportBinding
    }
  }

  // -----------------------------------------------------------------------
  // One launch is checked against the held binding and the Q/K/V reuse order.
  // -----------------------------------------------------------------------
  val incomingInputFeaturesWide = UInt(
    (linear.feeder.kTileCountWidth + 7) bits)
  incomingInputFeaturesWide :=
    (io.linearLaunch.payload.matrixTask.kTilesPerOutput.resize(
      linear.feeder.kTileCountWidth + 7) |<< 7).resized

  val expectedSegmentedKind = BitNetResidentLinearKind()
  expectedSegmentedKind := BitNetResidentLinearKind.QkvQuery
  when(launchOrdinalReg === 1) {
    expectedSegmentedKind := BitNetResidentLinearKind.QkvKey
  } elsewhen(launchOrdinalReg === 2) {
    expectedSegmentedKind := BitNetResidentLinearKind.QkvValue
  }
  val incomingSingleKind =
    io.linearLaunch.payload.kind ===
      BitNetResidentLinearKind.AttentionOutput ||
    io.linearLaunch.payload.kind === BitNetResidentLinearKind.GateUp ||
    io.linearLaunch.payload.kind === BitNetResidentLinearKind.Down
  val incomingKindLegal =
    (heldBinding.expectedLaunchCount === 3 &&
      io.linearLaunch.payload.kind === expectedSegmentedKind) ||
    (heldBinding.expectedLaunchCount === 1 && incomingSingleKind)
  val incomingSingleFeatureShape =
    heldBinding.expectedLaunchCount =/= 1 ||
      Mux(io.linearLaunch.payload.kind === BitNetResidentLinearKind.Down,
        heldBinding.inputFeatures === model.ffnSize,
        heldBinding.inputFeatures === model.hiddenSize)
  val incomingSequenceLegal =
    launchOrdinalReg === 0 ||
      io.linearLaunch.payload.sequenceId ===
        (firstSequenceId + launchOrdinalReg).resized
  val incomingRoundLegal =
    launchOrdinalReg === 0 ||
      (io.linearLaunch.payload.layer === firstLayer &&
        io.linearLaunch.payload.mode === firstMode)
  val incomingProgram = BitNetResidentLinearProgramCodec.encode(
    io.linearLaunch.payload, linear)
  // This is the sole legacy -> compact trust boundary.  In addition to the
  // Binder's image/reuse checks, require a complete bit-exact legacy protocol
  // round trip before any compute or replay side effect can be issued.
  val incomingLaunchEncodable =
    BitNetResidentLinearProgramCodec.launchEncodable(
      io.linearLaunch.payload, linear)
  // Keep the expensive compact-codec round trip and the independent protocol
  // checks on separate ingress registers.  Reducing them directly into the
  // binding fault enables previously built a cross-module owner -> DSP/CARRY
  // -> binder CE path.  The following ValidateLaunch cycle performs only a
  // shallow reduction over these registered results.
  val incomingLaunchChecks = Bits(5 bits)
  incomingLaunchChecks(0) :=
    io.linearLaunch.payload.requestId === heldBinding.requestId &&
    io.linearLaunch.payload.tokenMask === heldBinding.tokenMask &&
    io.linearLaunch.payload.matrixTask.activeTokenMask ===
      heldBinding.tokenMask &&
    io.linearLaunch.payload.sequenceId ===
      io.linearLaunch.payload.matrixTask.sequenceId
  incomingLaunchChecks(1) :=
    incomingInputFeaturesWide ===
      heldBinding.inputFeatures.resize(incomingInputFeaturesWide.getWidth) &&
    io.linearLaunch.payload.matrixTask.expectedBeats =/= 0 &&
    !io.linearLaunch.payload.matrixTask.expectedBeats(0) &&
    io.linearLaunch.payload.matrixTask.kTilesPerOutput =/= 0 &&
    io.linearLaunch.payload.matrixTask.operation ===
      BitNetContinuousDotKind.Projection &&
    !io.linearLaunch.payload.matrixTask.routeFinalToDynamicQuant &&
    !io.linearLaunch.payload.matrixTask.residualEnable
  incomingLaunchChecks(2) := incomingKindLegal && incomingSingleFeatureShape
  incomingLaunchChecks(3) := incomingLaunchEncodable
  incomingLaunchChecks(4) := incomingSequenceLegal && incomingRoundLegal
  val pendingLaunchChecks = Reg(Bits(5 bits)) init 0

  io.linearLaunch.ready := state === State.AwaitLaunch

  def initializeLinearReport(
      launch: BitNetResidentLinearLaunch): Unit = {
    linearReport.requestId := launch.requestId
    linearReport.layer := launch.layer
    linearReport.kind := launch.kind
    linearReport.sequenceId := launch.sequenceId
    linearReport.success := False
    linearReport.fatal := False
    linearReport.faultCode := 0
    linearReport.acceptedResultBeats := 0
    linearReport.emittedPairs := 0
  }

  val computeDone = RegInit(False)
  val replayDone = RegInit(False)
  val computeGood = RegInit(False)
  val replayGood = RegInit(False)
  val computeFatal = RegInit(False)
  val computeFaultCode = Reg(Bits(16 bits)) init 0
  val computeAcceptedBeats = Reg(UInt(32 bits)) init 0
  val computeEmittedPairs = Reg(UInt(32 bits)) init 0
  val syntheticDrain = RegInit(False)
  val tileCount = Reg(UInt(matrix.beatCountWidth bits)) init 0
  val expectedFeatureTile = Reg(UInt(tileCfg.tileIndexWidth bits)) init 0
  val feederOrdinal = Reg(UInt(matrix.beatCountWidth bits)) init 0

  when(io.linearLaunch.fire) {
    heldProgram := incomingProgram
    pendingLaunchChecks := incomingLaunchChecks
    initializeLinearReport(io.linearLaunch.payload)
    transactionFault := 0
    computeDone := False
    replayDone := False
    computeGood := False
    replayGood := False
    computeFatal := False
    computeFaultCode := 0
    computeAcceptedBeats := 0
    computeEmittedPairs := 0
    syntheticDrain := False
    tileCount := 0
    expectedFeatureTile := 0
    feederOrdinal := 0
    state := State.ValidateLaunch
  }

  when(state === State.ValidateLaunch) {
    when(pendingLaunchChecks.andR) {
      when(launchOrdinalReg === 0) {
        firstSequenceId := heldLaunch.sequenceId
        firstLayer := heldLaunch.layer
        firstMode := heldLaunch.mode
      }
      state := State.IssueBoth
    } otherwise {
      val invalid = B(
        BigInt(1) << BitNetResidentActivationReplayBinderFault.InvalidLaunch,
        16 bits)
      transactionFault := invalid
      bindingFault := bindingFault | invalid
      linearReport.faultCode := invalid
      stickyFaultEvents(1) := True
      state := State.ReportLinear
    }
  }

  // Both downstream commands fire on the same edge or neither fires.
  val atomicIssue = state === State.IssueBoth &&
    io.computeLaunch.ready && io.replayCommand.ready
  io.computeLaunch.valid := atomicIssue
  io.computeLaunch.payload := heldProgram

  io.replayCommand.valid := atomicIssue
  io.replayCommand.payload.imageTag := heldBinding.imageTag.resized
  io.replayCommand.payload.epoch := heldBinding.epoch
  io.replayCommand.payload.slot := heldBinding.slot
  io.replayCommand.payload.inputFeatures := heldBinding.inputFeatures
  io.replayCommand.payload.outputSequenceId := heldLaunch.sequenceId
  io.replayCommand.payload.beatOrdinalBase := 0
  io.replayCommand.payload.beatCount :=
    (heldLaunch.matrixTask.expectedBeats |>> 1).resized
  io.replayCommand.payload.startTile := 0

  when(atomicIssue) {
    state := State.Active
  }

  // -----------------------------------------------------------------------
  // Authenticated enhanced tile -> normalized sole-feeder tile.
  // -----------------------------------------------------------------------
  // One resident activation tile feeds one Map0 raw group.  The feeder emits
  // two matrix pair beats (lower then upper) from that held group, so replay
  // and drain accounting intentionally live in the expectedBeats/2 domain.
  val rawGroupCount = (heldLaunch.matrixTask.expectedBeats |>> 1).resize(
    matrix.beatCountWidth)
  val tilesRemaining = tileCount < rawGroupCount
  val expectedTileLast = tileCount === rawGroupCount - 1
  val imageTileCount = (heldBinding.inputFeatures |>> 7).resize(
    tileCfg.tileCountWidth)
  val realTileIdentity =
    io.tile.payload.imageTag === heldBinding.imageTag.resized &&
    io.tile.payload.epoch === heldBinding.epoch &&
    io.tile.payload.slot === heldBinding.slot &&
    io.tile.payload.tokenMask === heldBinding.tokenMask &&
    io.tile.payload.normalized.sequenceId === heldLaunch.sequenceId &&
    io.tile.payload.normalized.beatOrdinal === tileCount &&
    io.tile.payload.featureTile === expectedFeatureTile &&
    io.tile.payload.first === (tileCount === 0) &&
    io.tile.payload.last === expectedTileLast &&
    !io.tile.payload.normalized.fault

  io.activation.valid := state === State.Active && !computeDone &&
    tilesRemaining && (syntheticDrain || io.tile.valid)
  io.activation.payload.sequenceId := heldLaunch.sequenceId
  // Activation joins the raw group's lower phase.  At every K wrap the
  // feeder's upper K-wide ordinal interval has already been consumed from the
  // held activation, hence the lower ordinal jumps over that interval.
  io.activation.payload.beatOrdinal := feederOrdinal
  io.activation.payload.fault := syntheticDrain || !realTileIdentity ||
    transactionFault.orR
  for (lane <- 0 until model.tokenParallelism;
       site <- 0 until matrix.base.siteCount) {
    io.activation.payload.tokenActivation(lane)(site) :=
      io.tile.payload.normalized.tokenActivation(lane)(site)
  }
  when(syntheticDrain) {
    for (lane <- 0 until model.tokenParallelism;
         site <- 0 until matrix.base.siteCount) {
      io.activation.payload.tokenActivation(lane)(site) := 0
    }
  }

  io.tile.ready := state === State.Active && !syntheticDrain &&
    tilesRemaining && (computeDone || io.activation.ready)
  val realTileAdvance = io.tile.fire
  val syntheticTileAdvance = state === State.Active && syntheticDrain &&
    tilesRemaining && (computeDone || io.activation.ready)
  val tileAdvance = realTileAdvance || syntheticTileAdvance
  val tileCountAfter =
    (tileCount + tileAdvance.asUInt).resize(matrix.beatCountWidth)

  when(realTileAdvance && !realTileIdentity) {
    transactionFault(
      BitNetResidentActivationReplayBinderFault.TileIdentity) := True
    when(io.tile.payload.imageTag =/= heldBinding.imageTag.resized ||
        io.tile.payload.epoch =/= heldBinding.epoch ||
        io.tile.payload.slot =/= heldBinding.slot) {
      transactionFault(
        BitNetResidentActivationReplayBinderFault.TileImageIdentity) := True
    }
    when(io.tile.payload.tokenMask =/= heldBinding.tokenMask) {
      transactionFault(
        BitNetResidentActivationReplayBinderFault.TileTokenMask) := True
    }
    when(io.tile.payload.normalized.sequenceId =/= heldLaunch.sequenceId) {
      transactionFault(
        BitNetResidentActivationReplayBinderFault.TileSequence) := True
    }
    when(io.tile.payload.normalized.beatOrdinal =/= tileCount) {
      transactionFault(
        BitNetResidentActivationReplayBinderFault.TileOrdinal) := True
    }
    when(io.tile.payload.featureTile =/= expectedFeatureTile) {
      transactionFault(
        BitNetResidentActivationReplayBinderFault.TileFeature) := True
    }
    when(io.tile.payload.first =/= (tileCount === 0) ||
        io.tile.payload.last =/= expectedTileLast ||
        io.tile.payload.normalized.fault) {
      transactionFault(
        BitNetResidentActivationReplayBinderFault.TileFraming) := True
    }
    stickyFaultEvents(2) := True
  }
  when(tileAdvance) {
    tileCount := tileCountAfter
    when(expectedFeatureTile.resize(tileCfg.tileCountWidth) + 1 ===
        imageTileCount) {
      expectedFeatureTile := 0
      feederOrdinal := feederOrdinal +
        heldLaunch.matrixTask.kTilesPerOutput.resize(
          matrix.beatCountWidth) + 1
    } otherwise {
      expectedFeatureTile := expectedFeatureTile + 1
      feederOrdinal := feederOrdinal + 1
    }
    when(expectedTileLast) {
      syntheticDrain := False
    }
  }

  // -----------------------------------------------------------------------
  // Real compute/replay completions.  Register both completion streams before
  // identity and early-completion checks.  Those checks combine wide endpoint
  // counters with binder-local replay state; evaluating them directly on the
  // inter-module streams created another long control path into
  // transactionFault after launch authentication was pipelined.
  //
  // m2sPipe is a one-entry elastic slice: stale/duplicate traffic remains
  // drainable, with at most one cycle of ready backpressure while the local
  // copy is consumed.
  // -----------------------------------------------------------------------
  val computeCompletion = io.computeCompletion.m2sPipe()
  val replayCompletion = io.replayCompletion.m2sPipe()
  computeCompletion.ready := True
  replayCompletion.ready := True

  val computeCompletionIdentity =
    computeCompletion.payload.requestId === heldLaunch.requestId &&
    computeCompletion.payload.layer === heldLaunch.layer &&
    computeCompletion.payload.kind === heldLaunch.kind &&
    computeCompletion.payload.sequenceId === heldLaunch.sequenceId
  val computeCompletionSuccess = computeCompletionIdentity &&
    computeCompletion.payload.success &&
    computeCompletion.payload.faultCode === 0
  val computeCompletionEarly = tileCountAfter =/=
    rawGroupCount

  when(computeCompletion.fire) {
    when(state === State.Active) {
      when(!computeDone) {
        computeDone := True
        computeGood := computeCompletionSuccess && !computeCompletionEarly
        computeFatal := computeCompletion.payload.fatal
        computeFaultCode := computeCompletion.payload.faultCode
        computeAcceptedBeats :=
          computeCompletion.payload.acceptedResultBeats
        computeEmittedPairs := computeCompletion.payload.emittedPairs
        when(!computeCompletionIdentity) {
          transactionFault(
            BitNetResidentActivationReplayBinderFault.ComputeIdentity) :=
            True
          stickyFaultEvents(3) := True
        } elsewhen(!computeCompletion.payload.success ||
            computeCompletion.payload.faultCode =/= 0) {
          transactionFault(
            BitNetResidentActivationReplayBinderFault.ComputeFailed) := True
          stickyFaultEvents(4) := True
        }
        when(computeCompletionEarly) {
          transactionFault(
            BitNetResidentActivationReplayBinderFault.EarlyCompletion) :=
            True
          stickyFaultEvents(5) := True
        }
      } otherwise {
        transactionFault(
          BitNetResidentActivationReplayBinderFault.DuplicateCompletion) :=
          True
        stickyFaultEvents(6) := True
      }
    } elsewhen(state === State.ValidateLaunch || state === State.IssueBoth) {
      transactionFault(
        BitNetResidentActivationReplayBinderFault.EarlyCompletion) := True
      stickyFaultEvents(7) := True
    } elsewhen(state === State.ReportLinear) {
      val duplicate = B(
        BigInt(1) <<
          BitNetResidentActivationReplayBinderFault.DuplicateCompletion,
        16 bits)
      transactionFault := transactionFault | duplicate
      bindingFault := bindingFault | duplicate
      linearReport.success := False
      linearReport.faultCode := linearReport.faultCode | duplicate
      stickyFaultEvents(8) := True
    } otherwise {
      stickyFaultEvents(9) := True
    }
  }

  val replayCompletionIdentity =
    replayCompletion.payload.imageTag === heldBinding.imageTag.resized &&
    replayCompletion.payload.epoch === heldBinding.epoch &&
    replayCompletion.payload.slot === heldBinding.slot &&
    replayCompletion.payload.outputSequenceId === heldLaunch.sequenceId &&
    replayCompletion.payload.emittedBeats ===
      rawGroupCount
  val replayCompletionSuccess = replayCompletionIdentity &&
    replayCompletion.payload.success &&
    replayCompletion.payload.faultCode === 0 &&
    tileCountAfter === rawGroupCount

  when(replayCompletion.fire) {
    when(state === State.Active) {
      when(!replayDone) {
        replayDone := True
        replayGood := replayCompletionSuccess
        when(!replayCompletionIdentity) {
          transactionFault(
            BitNetResidentActivationReplayBinderFault.ReplayIdentity) := True
          stickyFaultEvents(10) := True
        } elsewhen(!replayCompletion.payload.success ||
            replayCompletion.payload.faultCode =/= 0) {
          transactionFault(
            BitNetResidentActivationReplayBinderFault.ReplayFailed) := True
          stickyFaultEvents(11) := True
        }
        when(tileCountAfter =/= rawGroupCount) {
          transactionFault(
            BitNetResidentActivationReplayBinderFault.EarlyCompletion) :=
            True
          syntheticDrain := True
          stickyFaultEvents(12) := True
        }
      } otherwise {
        transactionFault(
          BitNetResidentActivationReplayBinderFault.DuplicateCompletion) :=
          True
        stickyFaultEvents(13) := True
      }
    } elsewhen(state === State.ValidateLaunch || state === State.IssueBoth) {
      transactionFault(
        BitNetResidentActivationReplayBinderFault.EarlyCompletion) := True
      stickyFaultEvents(14) := True
    } elsewhen(state === State.ReportLinear) {
      val duplicate = B(
        BigInt(1) <<
          BitNetResidentActivationReplayBinderFault.DuplicateCompletion,
        16 bits)
      transactionFault := transactionFault | duplicate
      bindingFault := bindingFault | duplicate
      linearReport.success := False
      linearReport.faultCode := linearReport.faultCode | duplicate
      stickyFaultEvents(15) := True
    } otherwise {
      stickyFaultEvents(16) := True
    }
  }

  // Wait for both real completions and the complete activation-side drain.
  val transactionRetired = state === State.Active && computeDone &&
    replayDone && tileCount === rawGroupCount &&
    !computeCompletion.valid && !replayCompletion.valid &&
    !io.computeCompletion.valid && !io.replayCompletion.valid
  val transactionSuccessful = computeGood && replayGood &&
    !transactionFault.orR

  when(transactionRetired) {
    linearReport.requestId := heldLaunch.requestId
    linearReport.layer := heldLaunch.layer
    linearReport.kind := heldLaunch.kind
    linearReport.sequenceId := heldLaunch.sequenceId
    linearReport.success := transactionSuccessful
    linearReport.fatal := computeFatal
    linearReport.faultCode := transactionFault | computeFaultCode
    linearReport.acceptedResultBeats := computeAcceptedBeats
    linearReport.emittedPairs := computeEmittedPairs
    bindingFault := bindingFault | transactionFault | computeFaultCode
    when(!computeGood && !transactionFault.orR &&
        computeFaultCode === 0) {
      val computeFailed = B(
        BigInt(1) <<
          BitNetResidentActivationReplayBinderFault.ComputeFailed,
        16 bits)
      linearReport.faultCode := computeFailed
      bindingFault := bindingFault | computeFailed
    }
    when(!replayGood && !transactionFault.orR) {
      val replayFailed = B(
        BigInt(1) <<
          BitNetResidentActivationReplayBinderFault.ReplayFailed,
        16 bits)
      linearReport.faultCode := transactionFault | computeFaultCode |
        replayFailed
      bindingFault := bindingFault | transactionFault | computeFaultCode |
        replayFailed
    }
    completedTransactionCounter := completedTransactionCounter + 1
    state := State.ReportLinear
  }

  // A pending duplicate is drained before a nominal report can escape.
  io.linearCompletion.valid := state === State.ReportLinear &&
    !io.computeCompletion.valid && !io.replayCompletion.valid
  io.linearCompletion.payload := linearReport

  when(io.linearCompletion.fire) {
    val completedAfter =
      (launchOrdinalReg + 1).resize(cfg.launchCountWidth)
    when(completedAfter === heldBinding.expectedLaunchCount) {
      fillBindingReport(heldBinding)
      bindingReport.completedLaunchCount := completedAfter
      bindingReport.success := !bindingFault.orR
      bindingReport.faultCode := bindingFault
      state := State.ReportBinding
    } otherwise {
      launchOrdinalReg := launchOrdinalReg + 1
      state := State.AwaitLaunch
    }
  }

  // Late duplicates are also given priority over publishing binding success.
  io.bindingCompletion.valid := state === State.ReportBinding &&
    !io.computeCompletion.valid && !io.replayCompletion.valid
  io.bindingCompletion.payload := bindingReport
  when(io.bindingCompletion.fire) {
    completedBindingCounter := completedBindingCounter + 1
    state := State.NoBinding
  }

  io.busy := state =/= State.NoBinding
  io.launchOrdinal := launchOrdinalReg
  io.protocolFaultSticky := stickyFault
  io.faultBits := bindingFault | transactionFault
  io.completedBindings := completedBindingCounter
  io.completedTransactions := completedTransactionCounter
}

object GenerateBitNetResidentActivationReplayBinder extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-activation-replay-binder")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentActivationReplayBinder())
}
