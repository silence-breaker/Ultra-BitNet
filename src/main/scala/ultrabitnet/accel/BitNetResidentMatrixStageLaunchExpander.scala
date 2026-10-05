package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** The physical projection hidden behind one resident MatrixArray step. */
object BitNetResidentMatrixProjectionPart extends SpinalEnum(binarySequential) {
  val Query, Key, Value, AttentionOutput, Gate, Up, Down, FusedGateUp =
    newElement()
}

object BitNetResidentMatrixScaleIndex {
  val Query = 0
  val Key = 1
  val Value = 2
  val AttentionOutput = 3
  val Gate = 4
  val Up = 5
  val Down = 6
  val Count = 7
}

object BitNetResidentMatrixStageFault {
  val None = 0x00
  val InvalidStage = 0x81
  val ProjectionMismatch = 0x82
  val ProjectionFailed = 0x83
  val UnsupportedDirectLaunch = 0x84
  val LinearCompletionMismatch = 0x85
}

case class BitNetResidentMatrixStageLaunchExpanderConfig(
    model: BitNetConfig = BitNetConfig.Production,
    feeder: BitNetContinuousMatrixFeederConfig =
      BitNetContinuousMatrixFeederConfig(),
    prefetchEpochWidth: Int = 8
) {
  val matrix: BitNetContinuousMatrixConfig = feeder.matrix
  val kvFeatures: Int = model.kvHeadCount * model.headSize
  val qkvFeatures: Int = model.hiddenSize + 2 * kvFeatures
  val inputFeatureWidth: Int = log2Up(
    scala.math.max(model.hiddenSize, model.ffnSize) + 1)
  val partOrdinalWidth: Int = 2

  require(model.tokenParallelism == feeder.tokenLanes)
  require(model.engineCount == feeder.engineCount)
  require(model.tokenParallelism == 2 || model.tokenParallelism == 4)
  require(model.engineCount >= 1 && model.engineCount <= 2)
  require(matrix.base.siteCount == 64 || matrix.base.siteCount == 128)
  require(prefetchEpochWidth >= 4)
  require(model.hiddenSize % matrix.base.siteCount == 0 &&
    model.ffnSize % matrix.base.siteCount == 0)
  require(model.hiddenSize % 8 == 0 && model.ffnSize % 8 == 0)
  require(kvFeatures % 8 == 0)
  require(qkvFeatures <= (1 << matrix.base.outputIndexWidth))
}

/**
  * PL-resident context accompanying the semantic layer step.
  *
  * These are arithmetic results/parameters, not a software descriptor.  In
  * particular the seven scale entries are filled by the resident AUX/scale
  * path, while the activation factors are produced once by the preceding
  * normalization/quantization pass.  Gate and Up intentionally share the same
  * activation vector and factors.
  */
case class BitNetResidentMatrixStageCommand(
    cfg: BitNetResidentMatrixStageLaunchExpanderConfig) extends Bundle {
  val step = BitNetResidentLayerStep(cfg.model)
  val activationFactorQ16 = Vec(
    UInt(32 bits), cfg.model.tokenParallelism)
  val weightScaleQ16 = Vec(
    UInt(32 bits), BitNetResidentMatrixScaleIndex.Count)
}

/**
  * One complete physical projection.  Q/K/V remain separate matrices because
  * their semantic output shapes differ.  Gate and Up share one resident
  * `GateUpWeight` image whose complete four-row Map0 groups alternate, so they
  * launch as one physical projection and carry both scalar identities.
  */
case class BitNetResidentMatrixProjectionLaunch(
    cfg: BitNetResidentMatrixStageLaunchExpanderConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val stage = BitNetResidentLayerStage()
  val layer = UInt(cfg.model.layerWidth bits)
  val mode = BitNetMode()
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)

  val part = BitNetResidentMatrixProjectionPart()
  val partOrdinal = UInt(cfg.partOrdinalWidth bits)
  val partCount = UInt(cfg.partOrdinalWidth bits)
  val firstPart = Bool()
  val lastPart = Bool()
  val kind = BitNetResidentLinearKind()
  val weightTarget = BitNetAddressTarget()
  val scaleTarget = BitNetAddressTarget()
  val hasSecondaryScale = Bool()
  val secondaryScaleTarget = BitNetAddressTarget()
  val secondaryWeightScaleQ16 = UInt(32 bits)

  /** Matrix/weight-prefetch transaction identity generated entirely in PL. */
  val sequenceId = UInt(cfg.matrix.sequenceIdWidth bits)
  val prefetchEpoch = UInt(cfg.prefetchEpochWidth bits)
  /** All subprojections of one semantic step carry the same reuse tag. */
  val activationReuseTag = UInt(
    BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)

  val inputFeatures = UInt(cfg.inputFeatureWidth bits)
  val outputFeatures = UInt(cfg.matrix.base.outputIndexWidth bits)
  /** Semantic offset in a future Q|K|V result join; zero for other stages. */
  val globalOutputBase = UInt(cfg.matrix.base.outputIndexWidth bits)
  /** Number of five-bank commands issued per active engine. */
  val groupsPerEngine = UInt(cfg.matrix.base.outputIndexWidth bits)
  /** Current compute-island linearLaunch accepts this projection directly. */
  val directLinearCompatible = Bool()
  /** No additional result join is required before retiring the layer step. */
  val directStageCompletionCompatible = Bool()

  val matrixTask = BitNetContinuousMatrixTask(cfg.feeder)
}

case class BitNetResidentMatrixProjectionCompletion(
    cfg: BitNetResidentMatrixStageLaunchExpanderConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val stage = BitNetResidentLayerStage()
  val layer = UInt(cfg.model.layerWidth bits)
  val part = BitNetResidentMatrixProjectionPart()
  val sequenceId = UInt(cfg.matrix.sequenceIdWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/** Pure schedule oracle used by tests and the production image audit. */
object BitNetResidentMatrixStageOracle {
  case class Projection(
      part: BitNetResidentMatrixProjectionPart.E,
      kind: BitNetResidentLinearKind.E,
      weightTarget: BitNetAddressTarget.E,
      scaleTarget: BitNetAddressTarget.E,
      secondaryScaleTarget: Option[BitNetAddressTarget.E],
      inputFeatures: Int,
      outputFeatures: Int,
      globalOutputBase: Int,
      directLinearCompatible: Boolean,
      directStageCompletionCompatible: Boolean)

  def projections(model: BitNetConfig,
      stage: BitNetResidentLayerStage.E): Vector[Projection] = {
    val kv = model.kvHeadCount * model.headSize
    stage match {
      case BitNetResidentLayerStage.QkvProjection => Vector(
        Projection(BitNetResidentMatrixProjectionPart.Query,
          BitNetResidentLinearKind.Qkv,
          BitNetAddressTarget.QWeight,
          BitNetAddressTarget.QProjectionScale,
          None,
          model.hiddenSize, model.hiddenSize, 0, false, false),
        Projection(BitNetResidentMatrixProjectionPart.Key,
          BitNetResidentLinearKind.Qkv,
          BitNetAddressTarget.KWeight,
          BitNetAddressTarget.KProjectionScale,
          None,
          model.hiddenSize, kv, model.hiddenSize, false, false),
        Projection(BitNetResidentMatrixProjectionPart.Value,
          BitNetResidentLinearKind.Qkv,
          BitNetAddressTarget.VWeight,
          BitNetAddressTarget.VProjectionScale,
          None,
          model.hiddenSize, kv, model.hiddenSize + kv, false, false))
      case BitNetResidentLayerStage.AttentionOutputProjection => Vector(
        Projection(BitNetResidentMatrixProjectionPart.AttentionOutput,
          BitNetResidentLinearKind.AttentionOutput,
          BitNetAddressTarget.OWeight,
          BitNetAddressTarget.OProjectionScale,
          None,
          model.hiddenSize, model.hiddenSize, 0, true, true))
      case BitNetResidentLayerStage.GateUpProjection => Vector(
        Projection(BitNetResidentMatrixProjectionPart.FusedGateUp,
          BitNetResidentLinearKind.GateUp,
          BitNetAddressTarget.GateUpWeight,
          BitNetAddressTarget.GateProjectionScale,
          Some(BitNetAddressTarget.UpProjectionScale),
          model.hiddenSize, 2 * model.ffnSize, 0, true, false))
      case BitNetResidentLayerStage.DownProjection => Vector(
        Projection(BitNetResidentMatrixProjectionPart.Down,
          BitNetResidentLinearKind.Down,
          BitNetAddressTarget.DownWeight,
          BitNetAddressTarget.DownProjectionScale,
          None,
          model.ffnSize, model.hiddenSize, 0, true, true))
      case other => throw new IllegalArgumentException(
        s"$other is not a MatrixArray projection stage")
    }
  }
}

/**
  * Static, descriptor-free expansion of one MatrixArray semantic step.
  *
  * Exactly one projection may be outstanding.  QKV retires only after Query,
  * Key and Value completions; GateUp is one fused physical projection whose
  * downstream pairing/gate barrier supplies its single completion.  A
  * completion with the wrong identity is consumed fail-closed and terminates
  * the semantic step with an explicit fault.
  */
class BitNetResidentMatrixStageLaunchExpander(
    cfg: BitNetResidentMatrixStageLaunchExpanderConfig =
      BitNetResidentMatrixStageLaunchExpanderConfig()) extends Component {
  private val model = cfg.model
  private val matrix = cfg.matrix
  private val b = matrix.base

  val io = new Bundle {
    val command = slave(Stream(BitNetResidentMatrixStageCommand(cfg)))
    val projection = master(Stream(
      BitNetResidentMatrixProjectionLaunch(cfg)))
    val projectionCompletion = slave(Stream(
      BitNetResidentMatrixProjectionCompletion(cfg)))
    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))

    val busy = out Bool()
    val activePart = out(BitNetResidentMatrixProjectionPart())
    val issuedProjections = out UInt(32 bits)
    val completedProjections = out UInt(32 bits)
    val failedSteps = out UInt(32 bits)
    val protocolFaultSticky = out Bool()
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Issue, AwaitProjection, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentMatrixStageCommand(cfg)) init
    BitNetResidentMatrixStageCommand(cfg).getZero
  val partOrdinal = Reg(UInt(cfg.partOrdinalWidth bits)) init 0
  val report = Reg(BitNetResidentLayerStepCompletion(model)) init
    BitNetResidentLayerStepCompletion(model).getZero
  val stickyFault = RegInit(False)
  val issuedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0
  val failedCount = Reg(UInt(32 bits)) init 0

  val commandStageLegal = io.command.step.unit ===
    BitNetResidentLayerUnit.MatrixArray &&
    (io.command.step.stage === BitNetResidentLayerStage.QkvProjection ||
      io.command.step.stage ===
        BitNetResidentLayerStage.AttentionOutputProjection ||
      io.command.step.stage === BitNetResidentLayerStage.GateUpProjection ||
      io.command.step.stage === BitNetResidentLayerStage.DownProjection)
  io.command.ready := state === State.Idle

  def initializeReport(command: BitNetResidentMatrixStageCommand): Unit = {
    report.requestId := command.step.requestId
    report.stepId := command.step.stepId
    report.stage := command.step.stage
    report.layer := command.step.layer
    report.success := False
    report.faultCode := 0
  }

  when(io.command.fire) {
    initializeReport(io.command.payload)
    when(commandStageLegal) {
      held := io.command.payload
      partOrdinal := 0
      state := State.Issue
    } otherwise {
      report.faultCode := B(
        BitNetResidentMatrixStageFault.InvalidStage, 8 bits)
      stickyFault := True
      failedCount := failedCount + 1
      state := State.Report
    }
  }

  val launch = BitNetResidentMatrixProjectionLaunch(cfg)
  launch.requestId := held.step.requestId
  launch.stepId := held.step.stepId
  launch.stage := held.step.stage
  launch.layer := held.step.layer
  launch.mode := held.step.mode
  launch.tokenBase := held.step.tokenBase
  launch.tokenMask := held.step.activeTokenMask
  launch.partOrdinal := partOrdinal
  launch.firstPart := partOrdinal === 0
  launch.activationReuseTag := held.step.stepId

  val partCount = UInt(cfg.partOrdinalWidth bits)
  val inputFeatures = UInt(cfg.inputFeatureWidth bits)
  val outputFeatures = UInt(b.outputIndexWidth bits)
  val globalOutputBase = UInt(b.outputIndexWidth bits)
  val scaleIndex = UInt(log2Up(BitNetResidentMatrixScaleIndex.Count) bits)
  partCount := 1
  inputFeatures := U(model.hiddenSize, cfg.inputFeatureWidth bits)
  outputFeatures := U(model.hiddenSize, b.outputIndexWidth bits)
  globalOutputBase := 0
  scaleIndex := BitNetResidentMatrixScaleIndex.AttentionOutput
  launch.part := BitNetResidentMatrixProjectionPart.AttentionOutput
  launch.kind := BitNetResidentLinearKind.AttentionOutput
  launch.weightTarget := BitNetAddressTarget.OWeight
  launch.scaleTarget := BitNetAddressTarget.OProjectionScale
  launch.hasSecondaryScale := False
  launch.secondaryScaleTarget := BitNetAddressTarget.OProjectionScale
  launch.secondaryWeightScaleQ16 := 0
  launch.directLinearCompatible := True
  launch.directStageCompletionCompatible := True

  switch(held.step.stage) {
    is(BitNetResidentLayerStage.QkvProjection) {
      partCount := 3
      launch.kind := BitNetResidentLinearKind.Qkv
      launch.directLinearCompatible := False
      launch.directStageCompletionCompatible := False
      switch(partOrdinal) {
        is(0) {
          launch.part := BitNetResidentMatrixProjectionPart.Query
          launch.weightTarget := BitNetAddressTarget.QWeight
          launch.scaleTarget := BitNetAddressTarget.QProjectionScale
          scaleIndex := BitNetResidentMatrixScaleIndex.Query
        }
        is(1) {
          launch.part := BitNetResidentMatrixProjectionPart.Key
          launch.weightTarget := BitNetAddressTarget.KWeight
          launch.scaleTarget := BitNetAddressTarget.KProjectionScale
          outputFeatures := U(cfg.kvFeatures, b.outputIndexWidth bits)
          globalOutputBase := U(model.hiddenSize, b.outputIndexWidth bits)
          scaleIndex := BitNetResidentMatrixScaleIndex.Key
        }
        default {
          launch.part := BitNetResidentMatrixProjectionPart.Value
          launch.weightTarget := BitNetAddressTarget.VWeight
          launch.scaleTarget := BitNetAddressTarget.VProjectionScale
          outputFeatures := U(cfg.kvFeatures, b.outputIndexWidth bits)
          globalOutputBase := U(
            model.hiddenSize + cfg.kvFeatures, b.outputIndexWidth bits)
          scaleIndex := BitNetResidentMatrixScaleIndex.Value
        }
      }
    }
    is(BitNetResidentLayerStage.AttentionOutputProjection) {
      scaleIndex := BitNetResidentMatrixScaleIndex.AttentionOutput
    }
    is(BitNetResidentLayerStage.GateUpProjection) {
      partCount := 1
      outputFeatures := U(2 * model.ffnSize, b.outputIndexWidth bits)
      launch.part := BitNetResidentMatrixProjectionPart.FusedGateUp
      launch.kind := BitNetResidentLinearKind.GateUp
      launch.weightTarget := BitNetAddressTarget.GateUpWeight
      launch.scaleTarget := BitNetAddressTarget.GateProjectionScale
      launch.hasSecondaryScale := True
      launch.secondaryScaleTarget := BitNetAddressTarget.UpProjectionScale
      launch.secondaryWeightScaleQ16 :=
        held.weightScaleQ16(BitNetResidentMatrixScaleIndex.Up)
      // Matrix results retain their fused physical index.  The resident
      // two-window pairing boundary performs semantic Gate/Up retirement.
      launch.directLinearCompatible := True
      launch.directStageCompletionCompatible := False
      scaleIndex := BitNetResidentMatrixScaleIndex.Gate
    }
    is(BitNetResidentLayerStage.DownProjection) {
      launch.part := BitNetResidentMatrixProjectionPart.Down
      launch.kind := BitNetResidentLinearKind.Down
      launch.weightTarget := BitNetAddressTarget.DownWeight
      launch.scaleTarget := BitNetAddressTarget.DownProjectionScale
      inputFeatures := U(model.ffnSize, cfg.inputFeatureWidth bits)
      scaleIndex := BitNetResidentMatrixScaleIndex.Down
    }
  }

  launch.partCount := partCount
  launch.lastPart := partOrdinal === partCount - 1
  launch.inputFeatures := inputFeatures
  launch.outputFeatures := outputFeatures
  launch.globalOutputBase := globalOutputBase
  launch.groupsPerEngine := (outputFeatures >> 2).resized
  when(held.step.mode === BitNetMode.Decode) {
    launch.groupsPerEngine :=
      (outputFeatures >> log2Up(4 * model.engineCount)).resized
  }

  // requestId participates in the hardware-generated identity so two
  // consecutive resident runs do not normally reuse a sequence/tag.  Safety
  // still relies on exact completion matching, not on uniqueness alone.
  val sequenceBase = (
    held.step.requestId.resize(matrix.sequenceIdWidth) ^
      held.step.stepId.resize(matrix.sequenceIdWidth)).resize(
        matrix.sequenceIdWidth)
  launch.sequenceId := (sequenceBase + partOrdinal + 1).resized
  launch.prefetchEpoch := (
    held.step.requestId.resize(cfg.prefetchEpochWidth) ^
      held.step.stepId.resize(cfg.prefetchEpochWidth) ^
      partOrdinal.resize(cfg.prefetchEpochWidth)).resized

  val task = launch.matrixTask
  task.scheduleMode := held.step.mode
  task.operation := BitNetContinuousDotKind.Projection
  task.sequenceId := launch.sequenceId
  task.taskIdBase := (
    (held.step.stepId.resize(matrix.taskIdWidth) << 4) +
      partOrdinal.resize(matrix.taskIdWidth)).resized
  task.kTilesPerOutput :=
    (inputFeatures >> log2Up(matrix.base.siteCount)).resized
  task.activeTokenMask := held.step.activeTokenMask
  task.routeFinalToDynamicQuant := False
  task.residualEnable := False
  for (lane <- 0 until model.tokenParallelism) {
    task.tokenId(lane) := (
      held.step.tokenBase.resize(b.tokenIdWidth) + lane).resized
    task.activationFactorQ16(lane) := held.activationFactorQ16(lane)
  }
  for (engine <- 0 until model.engineCount) {
    task.weightScaleQ16(engine) := held.weightScaleQ16(scaleIndex)
    task.outputBase(engine) := 0
    task.outputEndExclusive(engine) := outputFeatures
  }
  when(held.step.mode === BitNetMode.Decode) {
    if (model.engineCount > 1) {
      task.outputEndExclusive(0) := (outputFeatures >> 1).resized
      task.outputBase(1) := (outputFeatures >> 1).resized
    }
  }
  // One compact raw group yields two matrix beats (rows 0/1 and rows 2/3).
  task.expectedBeats := (
    launch.groupsPerEngine.resize(matrix.beatCountWidth) *
      task.kTilesPerOutput.resize(matrix.beatCountWidth) * 2).resized

  io.projection.valid := state === State.Issue
  io.projection.payload := launch
  when(io.projection.fire) {
    issuedCount := issuedCount + 1
    state := State.AwaitProjection
  }

  val completionMatches =
    io.projectionCompletion.payload.requestId === held.step.requestId &&
    io.projectionCompletion.payload.stepId === held.step.stepId &&
    io.projectionCompletion.payload.stage === held.step.stage &&
    io.projectionCompletion.payload.layer === held.step.layer &&
    io.projectionCompletion.payload.part === launch.part &&
    io.projectionCompletion.payload.sequenceId === launch.sequenceId
  io.projectionCompletion.ready := state === State.AwaitProjection
  when(io.projectionCompletion.fire) {
    completedCount := completedCount + 1
    when(!completionMatches) {
      report.faultCode := B(
        BitNetResidentMatrixStageFault.ProjectionMismatch, 8 bits)
      stickyFault := True
      failedCount := failedCount + 1
      state := State.Report
    } elsewhen(!io.projectionCompletion.payload.success) {
      report.faultCode := io.projectionCompletion.payload.faultCode
      when(io.projectionCompletion.payload.faultCode === 0) {
        report.faultCode := B(
          BitNetResidentMatrixStageFault.ProjectionFailed, 8 bits)
      }
      stickyFault := True
      failedCount := failedCount + 1
      state := State.Report
    } elsewhen(launch.lastPart) {
      report.success := True
      report.faultCode := 0
      state := State.Report
    } otherwise {
      partOrdinal := partOrdinal + 1
      state := State.Issue
    }
  }

  io.completion.valid := state === State.Report
  io.completion.payload := report
  when(io.completion.fire) { state := State.Idle }

  io.busy := state =/= State.Idle
  io.activePart := launch.part
  io.issuedProjections := issuedCount
  io.completedProjections := completedCount
  io.failedSteps := failedCount
  io.protocolFaultSticky := stickyFault
}

/**
  * Real adapter for the existing compute-island `linearLaunch` boundary.
  *
  * Only O and Down are currently accepted.  Query/Key/Value use the island's
  * different atomic QKV protocol.  Gate/Up remain fail-closed because the
  * existing endpoint assumes fused physical Map0 bytes but the five-bank image
  * stores two independent matrices.
  */
class BitNetResidentMatrixLinearTransactionAdapter(
    cfg: BitNetResidentMatrixStageLaunchExpanderConfig =
      BitNetResidentMatrixStageLaunchExpanderConfig()) extends Component {
  val linearCfg = BitNetResidentLinearEndpointConfig(
    model = cfg.model, feeder = cfg.feeder)

  val io = new Bundle {
    val projection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(cfg)))
    val linearLaunch = master(Stream(BitNetResidentLinearLaunch(linearCfg)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(linearCfg)))
    val completion = master(Stream(
      BitNetResidentMatrixProjectionCompletion(cfg)))
    val active = out Bool()
    val activePart = out(BitNetResidentMatrixProjectionPart())
    val protocolFaultSticky = out Bool()
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssueLinear, AwaitLinear, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentMatrixProjectionLaunch(cfg)) init
    BitNetResidentMatrixProjectionLaunch(cfg).getZero
  val report = Reg(BitNetResidentMatrixProjectionCompletion(cfg)) init
    BitNetResidentMatrixProjectionCompletion(cfg).getZero
  val stickyFault = RegInit(False)

  val supported = io.projection.payload.directLinearCompatible &&
    io.projection.payload.part =/=
      BitNetResidentMatrixProjectionPart.Query &&
    io.projection.payload.part =/=
      BitNetResidentMatrixProjectionPart.Key &&
    io.projection.payload.part =/=
      BitNetResidentMatrixProjectionPart.Value
  io.projection.ready := state === State.Idle
  when(io.projection.fire) {
    held := io.projection.payload
    report.requestId := io.projection.payload.requestId
    report.stepId := io.projection.payload.stepId
    report.stage := io.projection.payload.stage
    report.layer := io.projection.payload.layer
    report.part := io.projection.payload.part
    report.sequenceId := io.projection.payload.sequenceId
    report.success := False
    report.faultCode := 0
    when(supported) {
      state := State.IssueLinear
    } otherwise {
      report.faultCode := B(
        BitNetResidentMatrixStageFault.UnsupportedDirectLaunch, 8 bits)
      stickyFault := True
      state := State.Report
    }
  }

  io.linearLaunch.valid := state === State.IssueLinear
  io.linearLaunch.requestId := held.requestId
  io.linearLaunch.layer := held.layer
  io.linearLaunch.kind := held.kind
  io.linearLaunch.mode := held.mode
  io.linearLaunch.tokenMask := held.tokenMask
  io.linearLaunch.sequenceId := held.sequenceId
  io.linearLaunch.matrixTask := held.matrixTask
  when(io.linearLaunch.fire) { state := State.AwaitLinear }

  val completionMatches =
    io.linearCompletion.payload.requestId === held.requestId &&
    io.linearCompletion.payload.layer === held.layer &&
    io.linearCompletion.payload.kind === held.kind &&
    io.linearCompletion.payload.sequenceId === held.sequenceId
  io.linearCompletion.ready := state === State.AwaitLinear
  when(io.linearCompletion.fire) {
    report.success := io.linearCompletion.payload.success && completionMatches
    report.faultCode := io.linearCompletion.payload.faultCode(7 downto 0)
    when(!completionMatches) {
      report.faultCode := B(
        BitNetResidentMatrixStageFault.LinearCompletionMismatch, 8 bits)
      stickyFault := True
    } elsewhen(!io.linearCompletion.payload.success) {
      stickyFault := True
    }
    state := State.Report
  }

  io.completion.valid := state === State.Report
  io.completion.payload := report
  when(io.completion.fire) { state := State.Idle }
  io.active := state =/= State.Idle
  io.activePart := held.part
  io.protocolFaultSticky := stickyFault
}

object GenerateBitNetResidentMatrixStageLaunchExpander extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-resident-matrix-stage-launch-expander")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentMatrixStageLaunchExpander())
}
