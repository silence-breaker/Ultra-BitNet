package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Static geometry of the production Q/K/V segmented bridge.
  *
  * The controller-facing QKV command type is intentionally the existing
  * resident QKV/attention contract.  No second Q/K/V store or software
  * packet format is introduced at this boundary.
  */
case class BitNetSegmentedQkvProductionBridgeConfig(
    stage: BitNetResidentMatrixStageLaunchExpanderConfig =
      BitNetResidentMatrixStageLaunchExpanderConfig(),
    endpoint: BitNetResidentLinearEndpointConfig =
      BitNetResidentLinearEndpointConfig(),
    resident: BitNetResidentQkvAttentionPipelineConfig =
      BitNetResidentQkvAttentionPipelineConfig()
) {
  val model: BitNetConfig = stage.model
  val matrix: BitNetContinuousMatrixConfig = endpoint.matrix
  val qkv: BitNetQkvPostprocessConfig = resident.qkv
  val coefficient: BitNetQkvClusterResultAssemblerConfig =
    BitNetQkvClusterResultAssemblerConfig(
      model = model, continuous = matrix, qkv = qkv)

  val queryFeatures: Int = model.hiddenSize
  val kvFeatures: Int = model.kvHeadCount * model.headSize
  val totalFeatures: Int = queryFeatures + 2 * kvFeatures
  // Only the far decode half is reordered.  The same physical memory is
  // reused by Q, K, and V; it is not a composite QKV activation RAM.
  val maxFarDecodePairs: Int = queryFeatures / 4
  val farAddressWidth: Int = log2Up(maxFarDecodePairs)

  require(stage.feeder == endpoint.feeder)
  require(resident.model == model)
  require(model == BitNetConfig.Production,
    "the current resident Q/K/V and KV-cache contract is production-shaped")
  require(matrix.base.outputLanes == 2 &&
    matrix.base.accumulatorWidth == 32)
  require(queryFeatures % 8 == 0 && kvFeatures % 8 == 0)
  require(totalFeatures ==
    (model.queryHeadCount + 2 * model.kvHeadCount) * model.headSize)
  require(maxFarDecodePairs * 2 * matrix.base.accumulatorWidth ==
    40960,
    "production decode reorder must remain the bounded 5 KiB far-half RAM")
}

object BitNetSegmentedQkvBridgeFault {
  val None = 0x00
  val InvalidProjection = 0x91
  val PartOrder = 0x92
  val ResultMetadata = 0x93
  val RopeMetadata = 0x94
  val LinearCompletion = 0x95
  val ResidentCompletion = 0x96
  val RoundPoisoned = 0x97
}

/** Pure schedule/mapping reference used by focused simulations. */
object BitNetSegmentedQkvBridgeOracle {
  final case class PartShape(
      part: BitNetResidentMatrixProjectionPart.E,
      typedKind: BitNetResidentLinearKind.E,
      qkvKind: BitNetQkvKind.E,
      outputFeatures: Int,
      globalOutputBase: Int)

  def parts(model: BitNetConfig = BitNetConfig.Production): Vector[PartShape] = {
    val kv = model.kvHeadCount * model.headSize
    Vector(
      PartShape(BitNetResidentMatrixProjectionPart.Query,
        BitNetResidentLinearKind.QkvQuery, BitNetQkvKind.Query,
        model.hiddenSize, 0),
      PartShape(BitNetResidentMatrixProjectionPart.Key,
        BitNetResidentLinearKind.QkvKey, BitNetQkvKind.Key,
        kv, model.hiddenSize),
      PartShape(BitNetResidentMatrixProjectionPart.Value,
        BitNetResidentLinearKind.QkvValue, BitNetQkvKind.Value,
        kv, model.hiddenSize + kv))
  }

  def decodeBufferedBytes(
      model: BitNetConfig = BitNetConfig.Production): Int =
    (model.hiddenSize / 2) * 4
}

/**
  * BitNet Q/K/V segmented production bridge.
  *
  * Query, Key, and Value retain separate five-bank images and are each
  * translated to a distinct typed transaction on the sole TP4 generic
  * final-pair path.  Prefill joins the two token-sharded engines one pair at
  * a time.  Decode streams engine 0's low row shard immediately and retains
  * only engine 1's far half in one 5 KiB RAM so the existing QKV/RoPE ingress
  * still observes complete heads in natural element order.  That ingress is
  * frame-serial and cannot accept an out-of-order feature scatter.
  *
  * A Query completion retires only Query, likewise Key.  Value is held behind
  * the resident Q/K/V controller completion, so the stage expander's third
  * completion is the real three-part/RoPE/KV-cache barrier.  Malformed active
  * transactions suppress further architectural output, drain the generic
  * endpoint and coefficient stream, and report failure.  If a resident round
  * was opened but cannot be aborted, `fatalRoundPoisoned` remains asserted
  * until reset; the bridge never pretends that a partial Query was full QKV.
  */
class BitNetSegmentedQkvProductionBridge(
    cfg: BitNetSegmentedQkvProductionBridgeConfig =
      BitNetSegmentedQkvProductionBridgeConfig()) extends Component {
  private val model = cfg.model
  private val matrix = cfg.matrix
  private val b = matrix.base
  private val engines = cfg.endpoint.engineCount
  private val locals = cfg.endpoint.localTokenLanes
  private val outputCountWidth = b.outputIndexWidth + 1
  private val pairCountWidth = b.outputIndexWidth

  val io = new Bundle {
    val projection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(cfg.stage)))

    val linearLaunch = master(Stream(
      BitNetResidentLinearLaunch(cfg.endpoint)))
    val linearResult = slave(Stream(
      BitNetResidentFinalFeaturePair(cfg.endpoint)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(cfg.endpoint)))

    val ropeCoefficient = slave(Stream(
      BitNetQkvFeatureRopeCoefficient(cfg.coefficient)))
    val qkvRound = master(Stream(
      BitNetResidentQkvRoundCommand(cfg.resident)))
    val qkvDot = master(Stream(BitNetQkvTp4DotBeat(cfg.qkv)))
    val qkvCompletion = slave(Stream(
      BitNetResidentQkvRoundCompletion(cfg.resident)))

    /**
      * A surrounding physical-retirement barrier asserts this if a part
      * failed outside the arithmetic bridge (for example in weight replay).
      * An already-open resident QKV round cannot then be reused safely.
      */
    val externalPoison = in Bool()

    val projectionCompletion = master(Stream(
      BitNetResidentMatrixProjectionCompletion(cfg.stage)))

    val supported = out Bool()
    val busy = out Bool()
    val failureDrainActive = out Bool()
    val protocolFaultSticky = out Bool()
    val fatalRoundPoisoned = out Bool()
    val acceptedParts = out UInt(32 bits)
    val completedParts = out UInt(32 bits)
    val emittedFeatures = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssueResidentRound, IssueLinear, Active, Drain,
        WaitResidentCompletion, Report, Poisoned = newElement()
  }
  object MapPhase extends SpinalEnum(binarySequential) {
    val Prefill, DecodeNear, DecodeFarIssue, DecodeFarWait,
        DecodeFarEmit, Done = newElement()
  }

  val state = Reg(State()) init State.Idle
  val mapPhase = Reg(MapPhase()) init MapPhase.Prefill
  val held = Reg(BitNetResidentMatrixProjectionLaunch(cfg.stage)) init
    BitNetResidentMatrixProjectionLaunch(cfg.stage).getZero
  val roundOpen = RegInit(False)
  val expectedPartOrdinal = Reg(UInt(cfg.stage.partOrdinalWidth bits)) init 0
  val roundRequestId = Reg(UInt(model.requestIdWidth bits)) init 0
  val roundStepId = Reg(UInt(
    BitNetResidentLayerGeometry.stepIdWidth(model) bits)) init 0
  val roundLayer = Reg(UInt(model.layerWidth bits)) init 0
  val roundMode = Reg(BitNetMode()) init BitNetMode.Decode
  val roundTokenBase = Reg(UInt(model.sequenceWidth bits)) init 0
  val roundTokenMask = Reg(Bits(model.tokenParallelism bits)) init 0
  val roundActivationReuseTag = Reg(UInt(
    BitNetResidentLayerGeometry.stepIdWidth(model) bits)) init 0
  val roundQuerySequence = Reg(UInt(matrix.sequenceIdWidth bits)) init 0

  val stickyFault = RegInit(False)
  val fatalPoison = RegInit(False)
  val jobFault = RegInit(False)
  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val poisonAfterReport = RegInit(False)
  val acceptedPartCount = Reg(UInt(32 bits)) init 0
  val completedPartCount = Reg(UInt(32 bits)) init 0
  val emittedFeatureCount = Reg(UInt(32 bits)) init 0

  val nextBase = Vec(Reg(UInt(b.outputIndexWidth bits)) init 0, engines)
  val expectedPairs = Vec(Reg(UInt(pairCountWidth bits)) init 0, engines)
  val acceptedPairs = Vec(Reg(UInt(pairCountWidth bits)) init 0, engines)
  val emittedInPart = Reg(UInt(outputCountWidth bits)) init 0
  val coefficientCount = Reg(UInt(outputCountWidth bits)) init 0
  val emitRow = Reg(UInt(log2Up(b.outputLanes) bits)) init 0
  val linearDone = RegInit(False)
  val linearSucceeded = RegInit(False)

  val pendingValid = Vec(RegInit(False), engines)
  val pendingPayload = Vec(
    Reg(BitNetResidentFinalFeaturePair(cfg.endpoint)) init
      BitNetResidentFinalFeaturePair(cfg.endpoint).getZero,
    engines)

  // At 32 bits/row and two rows/pair this is exactly 640 x 64 bits = 5 KiB
  // for the production Query projection.  K and V reuse its low 160 entries.
  val decodeFarMemory = Mem(
    Bits((b.outputLanes * b.accumulatorWidth) bits),
    cfg.maxFarDecodePairs)
  val farReplayPair = Reg(UInt(cfg.farAddressWidth bits)) init 0
  val farReplayRow = Reg(UInt(log2Up(b.outputLanes) bits)) init 0
  val farReadEnable = state === State.Active &&
    mapPhase === MapPhase.DecodeFarIssue
  val farReadWord = decodeFarMemory.readSync(farReplayPair, farReadEnable)

  val completionValid = RegInit(False)
  val completionReg = Reg(
    BitNetResidentMatrixProjectionCompletion(cfg.stage)) init
      BitNetResidentMatrixProjectionCompletion(cfg.stage).getZero
  io.projectionCompletion.valid := completionValid
  io.projectionCompletion.payload := completionReg
  when(io.projectionCompletion.fire) {
    completionValid := False
    completedPartCount := completedPartCount + 1
    when(poisonAfterReport) {
      fatalPoison := True
      state := State.Poisoned
    } otherwise {
      state := State.Idle
    }
  }

  def fillProjectionCompletion(
      projection: BitNetResidentMatrixProjectionLaunch,
      success: Bool,
      faultCode: Bits): Unit = {
    completionReg.requestId := projection.requestId
    completionReg.stepId := projection.stepId
    completionReg.stage := projection.stage
    completionReg.layer := projection.layer
    completionReg.part := projection.part
    completionReg.sequenceId := projection.sequenceId
    completionReg.success := success
    completionReg.faultCode := faultCode
    completionValid := True
  }

  val incomingTypedKind = BitNetResidentLinearKind()
  incomingTypedKind := BitNetResidentLinearKind.QkvQuery
  when(io.projection.part === BitNetResidentMatrixProjectionPart.Key) {
    incomingTypedKind := BitNetResidentLinearKind.QkvKey
  } elsewhen(io.projection.part ===
      BitNetResidentMatrixProjectionPart.Value) {
    incomingTypedKind := BitNetResidentLinearKind.QkvValue
  }

  val incomingOutputFeatures = UInt(outputCountWidth bits)
  val incomingGlobalBase = UInt(outputCountWidth bits)
  val incomingWeightTarget = BitNetAddressTarget()
  val incomingScaleTarget = BitNetAddressTarget()
  incomingOutputFeatures := U(cfg.queryFeatures, outputCountWidth bits)
  incomingGlobalBase := 0
  incomingWeightTarget := BitNetAddressTarget.QWeight
  incomingScaleTarget := BitNetAddressTarget.QProjectionScale
  when(io.projection.part === BitNetResidentMatrixProjectionPart.Key) {
    incomingOutputFeatures := U(cfg.kvFeatures, outputCountWidth bits)
    incomingGlobalBase := U(cfg.queryFeatures, outputCountWidth bits)
    incomingWeightTarget := BitNetAddressTarget.KWeight
    incomingScaleTarget := BitNetAddressTarget.KProjectionScale
  } elsewhen(io.projection.part ===
      BitNetResidentMatrixProjectionPart.Value) {
    incomingOutputFeatures := U(cfg.kvFeatures, outputCountWidth bits)
    incomingGlobalBase := U(
      cfg.queryFeatures + cfg.kvFeatures, outputCountWidth bits)
    incomingWeightTarget := BitNetAddressTarget.VWeight
    incomingScaleTarget := BitNetAddressTarget.VProjectionScale
  }

  val incomingPartRecognized =
    io.projection.part === BitNetResidentMatrixProjectionPart.Query ||
      io.projection.part === BitNetResidentMatrixProjectionPart.Key ||
      io.projection.part === BitNetResidentMatrixProjectionPart.Value
  val incomingIsQuery = io.projection.part ===
    BitNetResidentMatrixProjectionPart.Query
  val incomingIsKey = io.projection.part ===
    BitNetResidentMatrixProjectionPart.Key
  val incomingIsValue = io.projection.part ===
    BitNetResidentMatrixProjectionPart.Value

  val incomingOrdinalShape =
    (incomingIsQuery && io.projection.partOrdinal === 0 &&
      io.projection.firstPart && !io.projection.lastPart) ||
    (incomingIsKey && io.projection.partOrdinal === 1 &&
      !io.projection.firstPart && !io.projection.lastPart) ||
    (incomingIsValue && io.projection.partOrdinal === 2 &&
      !io.projection.firstPart && io.projection.lastPart)

  val incomingPrefixMask = (1 to model.tokenParallelism).map { count =>
    io.projection.tokenMask === B((BigInt(1) << count) - 1,
      model.tokenParallelism bits)
  }.reduce(_ || _)
  val incomingModeMaskLegal =
    (io.projection.mode === BitNetMode.Prefill && incomingPrefixMask) ||
      (io.projection.mode === BitNetMode.Decode &&
        io.projection.tokenMask === B(1, model.tokenParallelism bits))

  val incomingTask = io.projection.matrixTask
  val incomingKtiles = U(model.hiddenSize / b.siteCount,
    incomingTask.kTilesPerOutput.getWidth bits)
  val incomingExpectedGroups = UInt(b.outputIndexWidth bits)
  incomingExpectedGroups := (incomingOutputFeatures >> 2).resized
  when(io.projection.mode === BitNetMode.Decode) {
    incomingExpectedGroups := (incomingOutputFeatures >>
      log2Up(4 * model.engineCount)).resized
  }
  val incomingExpectedBeatsWide = incomingExpectedGroups.resize(
    matrix.beatCountWidth + incomingTask.kTilesPerOutput.getWidth) *
      incomingTask.kTilesPerOutput.resize(
        matrix.beatCountWidth + incomingTask.kTilesPerOutput.getWidth) * 2
  val incomingExpectedBeats = incomingExpectedBeatsWide.resize(
    matrix.beatCountWidth)
  val incomingHalf = (incomingOutputFeatures >> 1).resize(outputCountWidth)
  val incomingOutputShape = Bool()
  incomingOutputShape := (0 until model.engineCount).map { engine =>
    incomingTask.outputBase(engine) === 0 &&
      incomingTask.outputEndExclusive(engine).resize(outputCountWidth) ===
        incomingOutputFeatures
  }.reduce(_ && _)
  when(io.projection.mode === BitNetMode.Decode) {
    if (model.engineCount == 1) {
      incomingOutputShape := incomingTask.outputBase(0) === 0 &&
        incomingTask.outputEndExclusive(0).resize(outputCountWidth) ===
          incomingOutputFeatures
    } else {
      incomingOutputShape :=
        incomingTask.outputBase(0) === 0 &&
          incomingTask.outputEndExclusive(0).resize(outputCountWidth) ===
            incomingHalf &&
          incomingTask.outputBase(1).resize(outputCountWidth) === incomingHalf &&
          incomingTask.outputEndExclusive(1).resize(outputCountWidth) ===
            incomingOutputFeatures
    }
  }

  val incomingTokenIdsLegal = Vec(Bool(), model.tokenParallelism)
  for (lane <- 0 until model.tokenParallelism) {
    incomingTokenIdsLegal(lane) := incomingTask.tokenId(lane) ===
      (io.projection.tokenBase.resize(b.tokenIdWidth) + lane).resized
  }
  val incomingRoundIdentity =
    io.projection.requestId === roundRequestId &&
      io.projection.stepId === roundStepId &&
      io.projection.layer === roundLayer &&
      io.projection.mode === roundMode &&
      io.projection.tokenBase === roundTokenBase &&
      io.projection.tokenMask === roundTokenMask &&
      io.projection.activationReuseTag === roundActivationReuseTag &&
      io.projection.sequenceId ===
        (roundQuerySequence + io.projection.partOrdinal).resized
  val incomingOrderLegal =
    (incomingIsQuery && !roundOpen && io.projection.partOrdinal === 0) ||
      (!incomingIsQuery && roundOpen &&
        io.projection.partOrdinal === expectedPartOrdinal &&
        incomingRoundIdentity)
  val incomingStaticShape = incomingPartRecognized &&
    io.projection.payload.stage ===
      BitNetResidentLayerStage.QkvProjection &&
    io.projection.layer < model.layerCount &&
    io.projection.partCount === 3 && incomingOrdinalShape &&
    io.projection.kind === BitNetResidentLinearKind.Qkv &&
    io.projection.weightTarget === incomingWeightTarget &&
    io.projection.scaleTarget === incomingScaleTarget &&
    !io.projection.hasSecondaryScale &&
    !io.projection.directLinearCompatible &&
    !io.projection.directStageCompletionCompatible &&
    io.projection.inputFeatures === model.hiddenSize &&
    io.projection.outputFeatures.resize(outputCountWidth) ===
      incomingOutputFeatures &&
    io.projection.globalOutputBase.resize(outputCountWidth) ===
      incomingGlobalBase &&
    io.projection.groupsPerEngine === incomingExpectedGroups
  val incomingTaskShape =
    incomingTask.scheduleMode === io.projection.mode &&
      incomingTask.operation === BitNetContinuousDotKind.Projection &&
      incomingTask.sequenceId === io.projection.sequenceId &&
      incomingTask.kTilesPerOutput === incomingKtiles &&
      incomingTask.activeTokenMask === io.projection.tokenMask &&
      !incomingTask.routeFinalToDynamicQuant &&
      !incomingTask.residualEnable && incomingOutputShape &&
      incomingTask.expectedBeats === incomingExpectedBeats &&
      (1 until model.engineCount).map { engine =>
        incomingTask.weightScaleQ16(engine) === incomingTask.weightScaleQ16(0)
      }.foldLeft(True)(_ && _) &&
      incomingTokenIdsLegal.asBits.andR
  val incomingLegal = !fatalPoison && incomingOrderLegal &&
    incomingModeMaskLegal && incomingStaticShape && incomingTaskShape

  io.supported := incomingLegal

  io.projection.ready := state === State.Idle && !completionValid &&
    !fatalPoison
  when(io.projection.fire) {
    held := io.projection.payload
    jobFault := False
    jobFaultCode := 0
    poisonAfterReport := False
    linearDone := False
    linearSucceeded := False
    emittedInPart := 0
    coefficientCount := 0
    emitRow := 0
    farReplayPair := 0
    farReplayRow := 0
    pendingValid.foreach(_ := False)
    for (engine <- 0 until engines) {
      nextBase(engine) := io.projection.matrixTask.outputBase(engine)
      expectedPairs(engine) := ((
        io.projection.matrixTask.outputEndExclusive(engine).resize(
          outputCountWidth) -
          io.projection.matrixTask.outputBase(engine).resize(
            outputCountWidth)) >> 1).resized
      acceptedPairs(engine) := 0
    }

    when(incomingLegal) {
      acceptedPartCount := acceptedPartCount + 1
      when(io.projection.mode === BitNetMode.Prefill) {
        mapPhase := MapPhase.Prefill
      } otherwise {
        mapPhase := MapPhase.DecodeNear
      }
      when(incomingIsQuery) {
        roundRequestId := io.projection.requestId
        roundStepId := io.projection.stepId
        roundLayer := io.projection.layer
        roundMode := io.projection.mode
        roundTokenBase := io.projection.tokenBase
        roundTokenMask := io.projection.tokenMask
        roundActivationReuseTag := io.projection.activationReuseTag
        roundQuerySequence := io.projection.sequenceId
        state := State.IssueResidentRound
      } otherwise {
        state := State.IssueLinear
      }
    } otherwise {
      stickyFault := True
      poisonAfterReport := roundOpen
      val fault = Bits(8 bits)
      fault := B(BitNetSegmentedQkvBridgeFault.InvalidProjection, 8 bits)
      when(incomingPartRecognized && !incomingOrderLegal) {
        fault := B(BitNetSegmentedQkvBridgeFault.PartOrder, 8 bits)
      }
      fillProjectionCompletion(io.projection.payload, False, fault)
      state := State.Report
    }
  }

  io.qkvRound.valid := state === State.IssueResidentRound
  io.qkvRound.requestId := held.requestId
  io.qkvRound.layer := held.layer
  io.qkvRound.mode := held.mode
  io.qkvRound.tokenBase := held.tokenBase
  io.qkvRound.tokenMask := held.tokenMask
  for (lane <- 0 until model.tokenParallelism) {
    io.qkvRound.tokenId(lane) := held.matrixTask.tokenId(lane).resized
  }
  when(io.qkvRound.fire) {
    roundOpen := True
    expectedPartOrdinal := 0
    state := State.IssueLinear
  }

  val heldTypedKind = BitNetResidentLinearKind()
  heldTypedKind := BitNetResidentLinearKind.QkvQuery
  when(held.part === BitNetResidentMatrixProjectionPart.Key) {
    heldTypedKind := BitNetResidentLinearKind.QkvKey
  } elsewhen(held.part === BitNetResidentMatrixProjectionPart.Value) {
    heldTypedKind := BitNetResidentLinearKind.QkvValue
  }

  io.linearLaunch.valid := state === State.IssueLinear
  io.linearLaunch.requestId := held.requestId
  io.linearLaunch.layer := held.layer
  io.linearLaunch.kind := heldTypedKind
  io.linearLaunch.mode := held.mode
  io.linearLaunch.tokenMask := held.tokenMask
  io.linearLaunch.sequenceId := held.sequenceId
  io.linearLaunch.matrixTask := held.matrixTask
  when(io.linearLaunch.fire) { state := State.Active }

  val activePrefill = held.mode === BitNetMode.Prefill
  val activeEngineMask = Bits(engines bits)
  activeEngineMask := B((BigInt(1) << engines) - 1, engines bits)
  when(activePrefill) {
    for (engine <- 0 until engines) {
      activeEngineMask(engine) := held.tokenMask(
        engine * locals + locals - 1 downto engine * locals).orR
    }
  }

  val pairLegalByEngine = Vec(Bool(), engines)
  for (engine <- 0 until engines) {
    val expectedGlobalMask = Bits(model.tokenParallelism bits)
    expectedGlobalMask := B(1, model.tokenParallelism bits)
    when(activePrefill) {
      expectedGlobalMask := (held.tokenMask(
        engine * locals + locals - 1 downto engine * locals)
        .resize(model.tokenParallelism) << (engine * locals))
        .resize(model.tokenParallelism)
    }
    val expectedRowMask = Bits(b.outputLanes bits)
    for (row <- 0 until b.outputLanes) {
      expectedRowMask(row) :=
        nextBase(engine).resize(outputCountWidth) + row <
          held.matrixTask.outputEndExclusive(engine).resize(outputCountWidth)
    }
    val localGroup = ((nextBase(engine) -
      held.matrixTask.outputBase(engine)) >> 1).resize(b.outputIndexWidth)
    val localMetaLegal = Vec(Bool(), locals)
    for (local <- 0 until locals) {
      val globalLane = engine * locals + local
      val laneActive = Bool()
      laneActive := (if (local == 0) True else False)
      when(activePrefill) { laneActive := held.tokenMask(globalLane) }
      val expectedTaskId = UInt(matrix.taskIdWidth bits)
      expectedTaskId := (held.matrixTask.taskIdBase +
        (localGroup.resize(matrix.taskIdWidth) << 1) + engine).resized
      when(activePrefill) {
        expectedTaskId := (held.matrixTask.taskIdBase +
          (localGroup.resize(matrix.taskIdWidth) <<
            log2Up(model.tokenParallelism)) + globalLane).resized
      }
      val expectedToken = UInt(b.tokenIdWidth bits)
      val expectedFactor = UInt(32 bits)
      expectedToken := held.matrixTask.tokenId(0)
      expectedFactor := held.matrixTask.activationFactorQ16(0)
      when(activePrefill) {
        expectedToken := held.matrixTask.tokenId(globalLane)
        expectedFactor := held.matrixTask.activationFactorQ16(globalLane)
      }
      localMetaLegal(local) :=
        (laneActive &&
          io.linearResult.taskId(local) === expectedTaskId &&
          io.linearResult.tokenId(local) === expectedToken &&
          io.linearResult.activationFactorQ16(local) === expectedFactor) ||
        (!laneActive && io.linearResult.taskId(local) === 0 &&
          io.linearResult.tokenId(local) === 0 &&
          io.linearResult.activationFactorQ16(local) === 0)
    }
    pairLegalByEngine(engine) :=
      io.linearResult.engine === engine && activeEngineMask(engine) &&
      acceptedPairs(engine) < expectedPairs(engine) &&
      io.linearResult.requestId === held.requestId &&
      io.linearResult.layer === held.layer &&
      io.linearResult.kind === heldTypedKind &&
      io.linearResult.mode === held.mode &&
      io.linearResult.sequenceId === held.sequenceId &&
      io.linearResult.physicalOutputBase === nextBase(engine) &&
      io.linearResult.globalTokenLaneMask === expectedGlobalMask &&
      io.linearResult.outputLaneMask === expectedRowMask &&
      io.linearResult.weightScaleQ16 ===
        held.matrixTask.weightScaleQ16(engine) &&
      io.linearResult.physicalVectorFirst === (nextBase(engine) === 0) &&
      io.linearResult.physicalVectorLast ===
        (nextBase(engine).resize(outputCountWidth) + b.outputLanes >=
          held.matrixTask.outputEndExclusive(engine).resize(outputCountWidth)) &&
      localMetaLegal.asBits.andR
  }
  val inputPairLegal = pairLegalByEngine.asBits.orR

  val normalResultReady = Bool()
  normalResultReady := False
  when(io.linearResult.engine === 0) {
    normalResultReady := !pendingValid(0)
  } otherwise {
    if (engines > 1) {
      normalResultReady := Mux(activePrefill,
        !pendingValid(1), acceptedPairs(1) < expectedPairs(1))
    }
  }
  io.linearResult.ready := False
  when(state === State.Active) {
    io.linearResult.ready := Mux(jobFault, True, normalResultReady)
  } elsewhen(state === State.Drain || state === State.Idle ||
      state === State.Report || state === State.Poisoned) {
    io.linearResult.ready := True
  }

  when(io.linearResult.fire) {
    when(state === State.Active && !jobFault) {
      when(inputPairLegal) {
        for (engine <- 0 until engines) {
          when(io.linearResult.engine === engine) {
            acceptedPairs(engine) := acceptedPairs(engine) + 1
            nextBase(engine) := nextBase(engine) + b.outputLanes
          }
        }
        when(activePrefill || io.linearResult.engine === 0) {
          for (engine <- 0 until engines) {
            when(io.linearResult.engine === engine) {
              pendingPayload(engine) := io.linearResult.payload
              pendingValid(engine) := True
            }
          }
        } otherwise {
          if (engines > 1) {
            val packed = io.linearResult.rawRows(0)(1).asBits ##
              io.linearResult.rawRows(0)(0).asBits
            decodeFarMemory.write(
              acceptedPairs(1).resize(cfg.farAddressWidth), packed)
          }
        }
      } otherwise {
        jobFault := True
        stickyFault := True
        jobFaultCode := B(
          BitNetSegmentedQkvBridgeFault.ResultMetadata, 8 bits)
        pendingValid.foreach(_ := False)
      }
    } elsewhen(state =/= State.Drain) {
      stickyFault := True
    }
  }

  val expectedLinearPairs = UInt(32 bits)
  expectedLinearPairs := (0 until engines)
    .map(engine => expectedPairs(engine).resize(32)).reduce(_ + _)
  val expectedRawResults = UInt(32 bits)
  expectedRawResults := (held.matrixTask.expectedBeats.resize(32) *
    CountOne(activeEngineMask).resize(32)).resize(32)
  val linearCompletionMatches =
    io.linearCompletion.requestId === held.requestId &&
      io.linearCompletion.layer === held.layer &&
      io.linearCompletion.kind === heldTypedKind &&
      io.linearCompletion.sequenceId === held.sequenceId &&
      !io.linearCompletion.fatal &&
      io.linearCompletion.emittedPairs === expectedLinearPairs &&
      io.linearCompletion.acceptedResultBeats === expectedRawResults
  io.linearCompletion.ready :=
    (state === State.Active || state === State.Drain) && !linearDone
  when(io.linearCompletion.fire) {
    linearDone := True
    linearSucceeded := io.linearCompletion.success &&
      io.linearCompletion.faultCode === 0 && linearCompletionMatches
    when(!io.linearCompletion.success ||
        io.linearCompletion.faultCode =/= 0 || !linearCompletionMatches) {
      jobFault := True
      stickyFault := True
      jobFaultCode := B(
        BitNetSegmentedQkvBridgeFault.LinearCompletion, 8 bits)
      pendingValid.foreach(_ := False)
    }
  }
  when((state === State.Idle || state === State.Report ||
      state === State.Poisoned) && io.linearCompletion.valid) {
    io.linearCompletion.ready := True
    when(io.linearCompletion.fire) { stickyFault := True }
  }

  val allPrefillPending = (0 until engines).map { engine =>
    !activeEngineMask(engine) || pendingValid(engine)
  }.reduce(_ && _)
  val prefillPairsMatch = if (engines == 1) True else
    !activeEngineMask(1) ||
      (pendingPayload(0).physicalOutputBase ===
        pendingPayload(1).physicalOutputBase &&
        pendingPayload(0).outputLaneMask ===
          pendingPayload(1).outputLaneMask &&
        pendingPayload(0).weightScaleQ16 ===
          pendingPayload(1).weightScaleQ16)
  when(state === State.Active && !jobFault &&
      mapPhase === MapPhase.Prefill && allPrefillPending &&
      !prefillPairsMatch) {
    jobFault := True
    stickyFault := True
    jobFaultCode := B(
      BitNetSegmentedQkvBridgeFault.ResultMetadata, 8 bits)
    pendingValid.foreach(_ := False)
  }

  val prefillDataAvailable = mapPhase === MapPhase.Prefill &&
    allPrefillPending && prefillPairsMatch
  val nearDataAvailable = mapPhase === MapPhase.DecodeNear && pendingValid(0)
  val farDataAvailable = mapPhase === MapPhase.DecodeFarEmit
  val dataAvailable = state === State.Active && !jobFault &&
    (prefillDataAvailable || nearDataAvailable || farDataAvailable)

  val localFeature = UInt(b.outputIndexWidth bits)
  localFeature := pendingPayload(0).physicalOutputBase + emitRow
  when(farDataAvailable) {
    localFeature := ((held.outputFeatures.resize(b.outputIndexWidth) >> 1) +
      (farReplayPair.resize(b.outputIndexWidth) << 1) + farReplayRow)
      .resize(b.outputIndexWidth)
  }
  val globalFeature = (held.globalOutputBase.resize(outputCountWidth) +
    localFeature.resize(outputCountWidth)).resize(b.outputIndexWidth)
  val coefficientMatches =
    io.ropeCoefficient.sequenceId === held.sequenceId &&
      io.ropeCoefficient.globalFeature === globalFeature

  val selectedRaw = Vec(SInt(b.accumulatorWidth bits), model.tokenParallelism)
  val selectedFactor = Vec(UInt(32 bits), model.tokenParallelism)
  val selectedToken = Vec(UInt(cfg.qkv.tokenIdWidth bits),
    model.tokenParallelism)
  selectedRaw.foreach(_ := 0)
  selectedFactor.foreach(_ := 0)
  selectedToken.foreach(_ := 0)
  when(prefillDataAvailable) {
    for (lane <- 0 until model.tokenParallelism) {
      val engine = lane / locals
      val local = lane % locals
      when(held.tokenMask(lane)) {
        selectedRaw(lane) := pendingPayload(engine).rawRows(local)(emitRow)
        selectedFactor(lane) :=
          pendingPayload(engine).activationFactorQ16(local)
        selectedToken(lane) := pendingPayload(engine).tokenId(local).resized
      }
    }
  } elsewhen(nearDataAvailable) {
    selectedRaw(0) := pendingPayload(0).rawRows(0)(emitRow)
    selectedFactor(0) := pendingPayload(0).activationFactorQ16(0)
    selectedToken(0) := pendingPayload(0).tokenId(0).resized
  } elsewhen(farDataAvailable) {
    val farRaw = SInt(b.accumulatorWidth bits)
    farRaw := farReadWord(b.accumulatorWidth - 1 downto 0).asSInt
    when(farReplayRow === 1) {
      farRaw := farReadWord(2 * b.accumulatorWidth - 1 downto
        b.accumulatorWidth).asSInt
    }
    selectedRaw(0) := farRaw
    selectedFactor(0) := held.matrixTask.activationFactorQ16(0)
    selectedToken(0) := held.matrixTask.tokenId(0).resized
  }

  val selectedWeightScale = UInt(32 bits)
  selectedWeightScale := pendingPayload(0).weightScaleQ16
  if (engines > 1) {
    when(farDataAvailable) {
      selectedWeightScale := held.matrixTask.weightScaleQ16(1)
    }
  }
  val qkvKind = BitNetQkvKind()
  qkvKind := BitNetQkvKind.Query
  when(held.part === BitNetResidentMatrixProjectionPart.Key) {
    qkvKind := BitNetQkvKind.Key
  } elsewhen(held.part === BitNetResidentMatrixProjectionPart.Value) {
    qkvKind := BitNetQkvKind.Value
  }
  val element = localFeature(log2Up(model.headSize) - 1 downto 0)
  val head = (localFeature >> log2Up(model.headSize)).resize(
    cfg.qkv.headWidth)

  io.qkvDot.valid := dataAvailable && io.ropeCoefficient.valid &&
    coefficientMatches
  io.qkvDot.rawDot := selectedRaw
  io.qkvDot.activationFactorQ16 := selectedFactor
  io.qkvDot.weightScaleQ16 := selectedWeightScale
  io.qkvDot.tokenId := selectedToken
  io.qkvDot.tokenMask := held.tokenMask
  io.qkvDot.kind := qkvKind
  io.qkvDot.head := head
  io.qkvDot.element := element.resized
  io.qkvDot.cosine := io.ropeCoefficient.cosine
  io.qkvDot.sine := io.ropeCoefficient.sine
  io.qkvDot.first := element === 0
  io.qkvDot.last := element === model.headSize - 1

  io.ropeCoefficient.ready := False
  when(state === State.Active && !jobFault && dataAvailable) {
    io.ropeCoefficient.ready := Mux(coefficientMatches,
      io.qkvDot.ready, True)
  } elsewhen((state === State.Active && jobFault) || state === State.Drain) {
    io.ropeCoefficient.ready :=
      coefficientCount < held.outputFeatures.resize(outputCountWidth)
  } elsewhen(state === State.Idle || state === State.Report ||
      state === State.Poisoned) {
    io.ropeCoefficient.ready := True
  }

  when(io.ropeCoefficient.fire) {
    when(state === State.Active || state === State.Drain) {
      when(coefficientCount < held.outputFeatures.resize(outputCountWidth)) {
        coefficientCount := coefficientCount + 1
      }
      when(state === State.Active && !jobFault && !coefficientMatches) {
        jobFault := True
        stickyFault := True
        jobFaultCode := B(
          BitNetSegmentedQkvBridgeFault.RopeMetadata, 8 bits)
        pendingValid.foreach(_ := False)
      }
    } otherwise {
      stickyFault := True
    }
  }

  when(io.qkvDot.fire) {
    emittedInPart := emittedInPart + 1
    emittedFeatureCount := emittedFeatureCount + 1
    when(mapPhase === MapPhase.Prefill) {
      when(emitRow === b.outputLanes - 1) {
        emitRow := 0
        for (engine <- 0 until engines) {
          when(activeEngineMask(engine)) { pendingValid(engine) := False }
        }
        when(emittedInPart ===
            held.outputFeatures.resize(outputCountWidth) - 1) {
          mapPhase := MapPhase.Done
        }
      } otherwise {
        emitRow := emitRow + 1
      }
    } elsewhen(mapPhase === MapPhase.DecodeNear) {
      when(emitRow === b.outputLanes - 1) {
        emitRow := 0
        pendingValid(0) := False
        if (engines == 1) {
          when(emittedInPart ===
              held.outputFeatures.resize(outputCountWidth) - 1) {
            mapPhase := MapPhase.Done
          }
        }
      } otherwise {
        emitRow := emitRow + 1
      }
    } elsewhen(mapPhase === MapPhase.DecodeFarEmit) {
      when(farReplayRow === b.outputLanes - 1) {
        farReplayRow := 0
        when(emittedInPart ===
            held.outputFeatures.resize(outputCountWidth) - 1) {
          mapPhase := MapPhase.Done
        } otherwise {
          farReplayPair := farReplayPair + 1
          mapPhase := MapPhase.DecodeFarIssue
        }
      } otherwise {
        farReplayRow := farReplayRow + 1
      }
    }
  }

  val decodeNearFeatures =
    (held.outputFeatures.resize(outputCountWidth) >> 1).resize(outputCountWidth)
  if (engines > 1) {
    when(state === State.Active && !jobFault &&
        mapPhase === MapPhase.DecodeNear &&
        emittedInPart === decodeNearFeatures && !pendingValid(0) &&
        acceptedPairs(1) === expectedPairs(1)) {
      farReplayPair := 0
      farReplayRow := 0
      mapPhase := MapPhase.DecodeFarIssue
    }
  }
  when(state === State.Active && !jobFault &&
      mapPhase === MapPhase.DecodeFarIssue) {
    mapPhase := MapPhase.DecodeFarWait
  }
  when(state === State.Active && !jobFault &&
      mapPhase === MapPhase.DecodeFarWait) {
    mapPhase := MapPhase.DecodeFarEmit
  }

  when(state === State.Active && jobFault) {
    state := State.Drain
  }

  val normalDataDone = mapPhase === MapPhase.Done &&
    emittedInPart === held.outputFeatures.resize(outputCountWidth) &&
    coefficientCount === held.outputFeatures.resize(outputCountWidth)
  when(state === State.Active && !jobFault && linearDone &&
      linearSucceeded && normalDataDone) {
    when(held.part === BitNetResidentMatrixProjectionPart.Value) {
      state := State.WaitResidentCompletion
    } otherwise {
      fillProjectionCompletion(held, True,
        B(BitNetSegmentedQkvBridgeFault.None, 8 bits))
      expectedPartOrdinal := held.partOrdinal + 1
      state := State.Report
    }
  }

  val coefficientDrainDone = coefficientCount ===
    held.outputFeatures.resize(outputCountWidth)
  when(state === State.Drain && linearDone && coefficientDrainDone) {
    fillProjectionCompletion(held, False, jobFaultCode)
    poisonAfterReport := roundOpen
    state := State.Report
  }

  val residentCompletionMatches =
    io.qkvCompletion.requestId === roundRequestId &&
      io.qkvCompletion.layer === roundLayer &&
      io.qkvCompletion.tokenBase === roundTokenBase &&
      io.qkvCompletion.tokenMask === roundTokenMask
  io.qkvCompletion.ready := state === State.WaitResidentCompletion
  when(io.qkvCompletion.fire) {
    val residentSuccess = io.qkvCompletion.success &&
      io.qkvCompletion.faultCode === 0 && residentCompletionMatches
    fillProjectionCompletion(held, residentSuccess,
      Mux(residentSuccess,
        B(BitNetSegmentedQkvBridgeFault.None, 8 bits),
        B(BitNetSegmentedQkvBridgeFault.ResidentCompletion, 8 bits)))
    roundOpen := False
    poisonAfterReport := !residentCompletionMatches
    when(!residentSuccess) { stickyFault := True }
    state := State.Report
  }

  when(io.externalPoison) {
    stickyFault := True
    when(roundOpen) {
      fatalPoison := True
      state := State.Poisoned
    }
  }

  io.busy := state =/= State.Idle || roundOpen || completionValid
  io.failureDrainActive := state === State.Drain
  io.protocolFaultSticky := stickyFault
  io.fatalRoundPoisoned := fatalPoison
  io.acceptedParts := acceptedPartCount
  io.completedParts := completedPartCount
  io.emittedFeatures := emittedFeatureCount
}

object GenerateBitNetSegmentedQkvProductionBridge extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-segmented-qkv-production-bridge")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetSegmentedQkvProductionBridge())
}
