package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Semantic owner of one transaction on the sole model-wide linear array. */
object BitNetResidentLinearKind extends SpinalEnum(binarySequential) {
  // Keep the original values first: Qkv is the research-only atomic
  // composite transaction.  The three appended kinds are the production
  // segmented Q/K/V matrices and deliberately cannot alias that descriptor.
  val Qkv, AttentionOutput, GateUp, Down, LmHead,
      QkvQuery, QkvKey, QkvValue = newElement()
}

case class BitNetResidentLinearEndpointConfig(
    model: BitNetConfig = BitNetConfig.Production,
    feeder: BitNetContinuousMatrixFeederConfig =
      BitNetContinuousMatrixFeederConfig()
) {
  val matrix: BitNetContinuousMatrixConfig = feeder.matrix
  val engineCount: Int = feeder.engineCount
  val tokenLanes: Int = feeder.tokenLanes
  val localTokenLanes: Int = matrix.tokenLanesPerEngine
  val engineIndexWidth: Int = Math.max(1, log2Up(engineCount))

  require(model.engineCount == engineCount &&
    (engineCount == 1 || engineCount == 2))
  require(model.tokenParallelism == tokenLanes &&
    tokenLanes == engineCount * localTokenLanes)
  require(localTokenLanes == 2)
  require(matrix.base.outputLanes == 2)
  require(matrix.base.accumulatorWidth == 32)
  require(matrix.base.tokenIdWidth >= model.sequenceWidth)
}

/**
  * Generic transaction descriptor for every non-QKV linear operation.
  * QKV retains its richer atomic launch because it must also authenticate the
  * QKV assembler and the resident Q/K/V controller before the feeder starts.
  */
case class BitNetResidentLinearLaunch(
    cfg: BitNetResidentLinearEndpointConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val kind = BitNetResidentLinearKind()
  val mode = BitNetMode()
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val sequenceId = UInt(cfg.matrix.sequenceIdWidth bits)
  val matrixTask = BitNetContinuousMatrixTask(cfg.feeder)
}

/**
  * A final physical output pair.  This is deliberately not a cluster result:
  * every field has already been authenticated against the active launch and
  * partial-K returns never reach this boundary.
  *
  * In prefill, `engine` selects token lanes [1:0] or [3:2].  In decode both
  * engines carry logical token lane zero and `engine` instead identifies the
  * independent output shard.  Keeping the physical engine/base explicit lets
  * small downstream sinks perform semantic remapping without a staging RAM.
  */
case class BitNetResidentFinalFeaturePair(
    cfg: BitNetResidentLinearEndpointConfig) extends Bundle {
  private val b = cfg.matrix.base

  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val kind = BitNetResidentLinearKind()
  val mode = BitNetMode()
  val sequenceId = UInt(cfg.matrix.sequenceIdWidth bits)
  val engine = UInt(cfg.engineIndexWidth bits)
  val physicalOutputBase = UInt(b.outputIndexWidth bits)
  val globalTokenLaneMask = Bits(cfg.tokenLanes bits)
  val outputLaneMask = Bits(b.outputLanes bits)
  val taskId = Vec(UInt(cfg.matrix.taskIdWidth bits), cfg.localTokenLanes)
  val tokenId = Vec(UInt(b.tokenIdWidth bits), cfg.localTokenLanes)
  val rawRows = Vec(
    Vec(SInt(b.accumulatorWidth bits), b.outputLanes),
    cfg.localTokenLanes)
  val activationFactorQ16 = Vec(UInt(32 bits), cfg.localTokenLanes)
  val weightScaleQ16 = UInt(32 bits)
  val physicalVectorFirst = Bool()
  val physicalVectorLast = Bool()
}

/**
  * The only state that must survive output backpressure inside the linear
  * endpoint.  The engine is implicit in the Vec index and the local lanes are
  * implicit in `rawRows`; all semantic metadata is reconstructed from the
  * authenticated descriptor plus this logical output-group provenance.
  */
private[accel] case class BitNetResidentPendingRawPair(
    cfg: BitNetResidentLinearEndpointConfig) extends Bundle {
  private val b = cfg.matrix.base

  val logicalOutputGroup = UInt(b.outputIndexWidth bits)
  val rawRows = Vec(
    Vec(SInt(b.accumulatorWidth bits), b.outputLanes),
    cfg.localTokenLanes)
}

/** Completion is emitted only after all typed final pairs have retired. */
case class BitNetResidentLinearCompletion(
    cfg: BitNetResidentLinearEndpointConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val kind = BitNetResidentLinearKind()
  val sequenceId = UInt(cfg.matrix.sequenceIdWidth bits)
  val success = Bool()
  /** Set by the enclosing island when physical state cannot be aborted. */
  val fatal = Bool()
  val faultCode = Bits(16 bits)
  val acceptedResultBeats = UInt(32 bits)
  val emittedPairs = UInt(32 bits)
}

object BitNetResidentLinearFault {
  val InvalidLaunch = 0
  val ResultMetadata = 1
  val UnexpectedResult = 2
  val ResultFraming = 3
  val EndpointCompletion = 4
}

/** Shared launch predicates instantiated by the island and endpoint. */
object BitNetResidentLinearProtocol {
  def isSegmentedQkvKind(kind: BitNetResidentLinearKind.C): Bool =
    kind === BitNetResidentLinearKind.QkvQuery ||
      kind === BitNetResidentLinearKind.QkvKey ||
      kind === BitNetResidentLinearKind.QkvValue

  def isProjectionKind(kind: BitNetResidentLinearKind.C): Bool =
    kind === BitNetResidentLinearKind.AttentionOutput ||
      kind === BitNetResidentLinearKind.GateUp ||
      kind === BitNetResidentLinearKind.Down ||
      isSegmentedQkvKind(kind)

  def launchLegal(
      launch: BitNetResidentLinearLaunch,
      cfg: BitNetResidentLinearEndpointConfig): Bool = {
    val task = launch.matrixTask
    val m = cfg.matrix
    val b = m.base
    val prefill = launch.mode === BitNetMode.Prefill
    val decode = launch.mode === BitNetMode.Decode
    val prefixMask = (1 to cfg.tokenLanes).map { count =>
      launch.tokenMask === B((BigInt(1) << count) - 1,
        cfg.tokenLanes bits)
    }.reduce(_ || _)
    val modeMaskLegal = (prefill && prefixMask) ||
      (decode && launch.tokenMask === B(1, cfg.tokenLanes bits))
    val projection = isProjectionKind(launch.kind)
    val segmentedQkv = isSegmentedQkvKind(launch.kind)
    val operationLegal =
      (projection && task.operation ===
        BitNetContinuousDotKind.Projection) ||
      (launch.kind === BitNetResidentLinearKind.LmHead &&
        task.operation === BitNetContinuousDotKind.LmHead)
    // LM is physically always a single-token, two-vocabulary-shard scan.
    // A prompt-prefill controller may request the final prompt token's logits,
    // but it must lower that request to this decode-style physical schedule.
    val lmPhysicalModeLegal = launch.kind =/=
      BitNetResidentLinearKind.LmHead || decode
    val envelopeLegal = launch.kind =/=
      BitNetResidentLinearKind.Qkv &&
      launch.layer < cfg.model.layerCount &&
      launch.mode === task.scheduleMode &&
      launch.tokenMask === task.activeTokenMask &&
      launch.sequenceId === task.sequenceId &&
      !task.routeFinalToDynamicQuant && !task.residualEnable

    val spans = Vec(UInt((b.outputIndexWidth + 1) bits), cfg.engineCount)
    val groups = Vec(UInt((b.outputIndexWidth + 1) bits), cfg.engineCount)
    val boundsLegal = Vec(Bool(), cfg.engineCount)
    val expectedMatches = Vec(Bool(), cfg.engineCount)
    for (engine <- 0 until cfg.engineCount) {
      boundsLegal(engine) := task.outputEndExclusive(engine) >
        task.outputBase(engine)
      spans(engine) := task.outputEndExclusive(engine).resize(
        b.outputIndexWidth + 1) - task.outputBase(engine).resize(
          b.outputIndexWidth + 1)
      groups(engine) := spans(engine)
      when(projection) {
        groups(engine) := ((spans(engine) + 1) >> 1).resize(
          b.outputIndexWidth + 1)
      }
      val expectedProduct = groups(engine) *
        task.kTilesPerOutput.resize(cfg.feeder.kTileCountWidth)
      expectedMatches(engine) := expectedProduct ===
        task.expectedBeats.resize(expectedProduct.getWidth)
    }

    val prefillShared = if (cfg.engineCount == 1) True else
      task.outputBase(0) === task.outputBase(1) &&
        task.outputEndExclusive(0) === task.outputEndExclusive(1) &&
        task.weightScaleQ16(0) === task.weightScaleQ16(1)
    val decodeBalanced = if (cfg.engineCount == 1) True else
      groups(0) === groups(1)
    // Map0 storage is fetched as one group of four rows and always emits two
    // physical output pairs.  Production descriptors must expose that shape
    // exactly; accepting a two-row tail here would make the feeder fail only
    // after the stateful transaction had already started.
    val projectionAlignment = !projection ||
      ((0 until cfg.engineCount).map { engine =>
        task.outputBase(engine)(1 downto 0) === 0 &&
          task.outputEndExclusive(engine)(1 downto 0) === 0
      }.reduce(_ && _) && !task.expectedBeats(0) &&
        task.expectedBeats >= 2)
    // A prefill GateUp compact group is four physical rows, ordered as
    // [Gate lower, Gate upper, Up lower, Up upper].  Decode instead uses
    // engine 0 for Gate and engine 1 for Up.  The endpoint keeps only physical
    // engine/base; a later gate join assigns these semantic roles.
    val gateUpPrefillShape = launch.kind =/=
      BitNetResidentLinearKind.GateUp || !prefill ||
      (task.outputBase(0)(1 downto 0) === 0 &&
        spans(0)(1 downto 0) === 0)

    // Unlike the retained atomic Qkv research kind, every production Q/K/V
    // kind has one immutable matrix shape.  This prevents a mistyped
    // segmented launch from authenticating a composite or truncated image.
    val queryFeatures = cfg.model.hiddenSize
    val kvFeatures = cfg.model.kvHeadCount * cfg.model.headSize
    val typedOutputFeatures = UInt((b.outputIndexWidth + 1) bits)
    typedOutputFeatures := U(queryFeatures, b.outputIndexWidth + 1 bits)
    when(launch.kind === BitNetResidentLinearKind.QkvKey ||
        launch.kind === BitNetResidentLinearKind.QkvValue) {
      typedOutputFeatures := U(kvFeatures, b.outputIndexWidth + 1 bits)
    }
    val typedPrefillShape = (0 until cfg.engineCount).map { engine =>
      task.outputBase(engine) === 0 &&
        task.outputEndExclusive(engine).resize(b.outputIndexWidth + 1) ===
          typedOutputFeatures
    }.reduce(_ && _)
    val typedDecodeHalf = (typedOutputFeatures >> 1).resize(
      b.outputIndexWidth + 1)
    val typedDecodeShape = if (cfg.engineCount == 1) {
      task.outputBase(0) === 0 &&
        task.outputEndExclusive(0).resize(b.outputIndexWidth + 1) ===
          typedOutputFeatures
    } else {
      task.outputBase(0) === 0 &&
        task.outputEndExclusive(0).resize(b.outputIndexWidth + 1) ===
          typedDecodeHalf &&
        task.outputBase(1).resize(b.outputIndexWidth + 1) ===
          typedDecodeHalf &&
        task.outputEndExclusive(1).resize(b.outputIndexWidth + 1) ===
          typedOutputFeatures
    }
    val segmentedQkvShape = !segmentedQkv ||
      (task.kTilesPerOutput === cfg.model.hiddenSize / b.siteCount &&
        ((prefill && typedPrefillShape) || (decode && typedDecodeShape)))

    envelopeLegal && modeMaskLegal && operationLegal && lmPhysicalModeLegal &&
      task.expectedBeats =/= 0 && task.kTilesPerOutput =/= 0 &&
      boundsLegal.asBits.andR && expectedMatches.asBits.andR &&
      ((prefill && prefillShared) || (decode && decodeBalanced)) &&
      projectionAlignment && gateUpPrefillShape && segmentedQkvShape
  }
}

/** Pure reference for the fused Gate/Up physical schedule. */
object BitNetResidentLinearLayoutOracle {
  sealed trait GateUpRole
  case object GateLower extends GateUpRole
  case object GateUpper extends GateUpRole
  case object UpLower extends GateUpRole
  case object UpUpper extends GateUpRole

  def gateUpPrefillRole(pairOrdinal: Int, packedRow: Int): GateUpRole = {
    require(pairOrdinal >= 0 && (packedRow == 0 || packedRow == 1))
    (pairOrdinal & 1, packedRow) match {
      case (0, 0) => GateLower
      case (0, 1) => GateUpper
      case (1, 0) => UpLower
      case (1, 1) => UpUpper
    }
  }

  def gateUpDecodeRole(engine: Int, packedRow: Int): GateUpRole = {
    require((engine == 0 || engine == 1) &&
      (packedRow == 0 || packedRow == 1))
    (engine, packedRow) match {
      case (0, 0) => GateLower
      case (0, 1) => GateUpper
      case (1, 0) => UpLower
      case (1, 1) => UpUpper
    }
  }
}

/**
  * Authenticates raw TP2 returns and reduces them to typed final-K pairs.
  * It owns no matrix hardware and contains no large reorder memory: one
  * pending final pair per engine is enough to absorb simultaneous TP4 return
  * while propagating arbitrary downstream backpressure.
  */
class BitNetResidentLinearResultEndpoint(
    cfg: BitNetResidentLinearEndpointConfig =
      BitNetResidentLinearEndpointConfig()) extends Component {
  private val m = cfg.matrix
  private val b = m.base
  private val engines = cfg.engineCount
  private val locals = cfg.localTokenLanes

  val io = new Bundle {
    val launch = slave(Stream(BitNetResidentLinearLaunch(cfg)))
    val engineResult = Vec(
      slave(Stream(BitNetProductionCompactTp2DotResultBeat(m))), engines)
    val output = master(Stream(BitNetResidentFinalFeaturePair(cfg)))
    val completion = master(Stream(BitNetResidentLinearCompletion(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedLaunches = out UInt(32 bits)
    val completedLaunches = out UInt(32 bits)
    val failedLaunches = out UInt(32 bits)
    val emittedPairs = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Active, FaultDrain = newElement()
  }
  val state = Reg(State()) init State.Idle
  // Break the descriptor source -> launchLegal arithmetic -> state/completion
  // enable path.  launchLegal contains span/beat-count products and several
  // shape checks; evaluating it in the same cycle as the broadcast launch
  // made this slow-domain control path dominate board timing after routing.
  // The endpoint can safely authenticate one cycle later because engine
  // results arrive well after launch and launchPendingValid deasserts ready.
  val launchPendingValid = Reg(Bool()) init False
  val launchPending = Reg(BitNetResidentLinearLaunch(cfg)) init
    BitNetResidentLinearLaunch(cfg).getZero
  val descriptor = Reg(BitNetResidentLinearLaunch(cfg)) init
    BitNetResidentLinearLaunch(cfg).getZero
  val resultBeatCount = Vec(
    Reg(UInt(m.beatCountWidth bits)) init 0, engines)
  val kTileIndex = Vec(
    Reg(UInt(cfg.feeder.kTileCountWidth bits)) init 0, engines)
  // Map0 raw groups arrive K-major: lower rows 0/1, then upper rows 2/3,
  // before the feeder advances to the next K tile.  Keep the expected phase
  // explicitly instead of dividing a result count by kTilesPerOutput.  I8
  // QK/LM traffic never observes this state.
  val map0UpperPhase = Vec(RegInit(False), engines)
  val outputGroup = Vec(
    Reg(UInt(b.outputIndexWidth bits)) init 0, engines)
  val roundAcceptedResults = Reg(UInt(32 bits)) init 0
  val roundEmittedPairs = Reg(UInt(32 bits)) init 0
  val activeFaultCode = Reg(Bits(16 bits)) init 0

  val stickyFault = RegInit(False)
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0
  val failedCount = Reg(UInt(32 bits)) init 0
  val emittedCount = Reg(UInt(32 bits)) init 0

  val completionValid = RegInit(False)
  val completionReg = Reg(BitNetResidentLinearCompletion(cfg)) init
    BitNetResidentLinearCompletion(cfg).getZero
  io.completion.valid := completionValid
  io.completion.payload := completionReg
  when(io.completion.fire) { completionValid := False }

  val pendingValid = Vec(RegInit(False), engines)
  // No reset is required for the compact payload because pendingValid is the
  // sole visibility qualifier.  In particular, no request/layer/kind/mode,
  // token, task, scale, mask, or framing copy is retained here.
  val pendingRawPair = Vec(
    Reg(BitNetResidentPendingRawPair(cfg)), engines)
  val roundRobin = Reg(UInt(cfg.engineIndexWidth bits)) init 0
  val outputSelectionLocked = RegInit(False)
  val lockedOutputEngine = Reg(UInt(cfg.engineIndexWidth bits)) init 0
  val candidateEngine = UInt(cfg.engineIndexWidth bits)
  candidateEngine := 0
  if (engines > 1) {
    when(!pendingValid(0) || (pendingValid(1) && roundRobin === 1)) {
      candidateEngine := 1
    }
  }
  val selectedEngine = UInt(cfg.engineIndexWidth bits)
  selectedEngine := candidateEngine
  when(outputSelectionLocked) { selectedEngine := lockedOutputEngine }

  val selectedPending = BitNetResidentPendingRawPair(cfg)
  selectedPending := pendingRawPair(0)
  if (engines > 1) {
    when(selectedEngine === 1) {
      selectedPending := pendingRawPair(1)
    }
  }

  val descriptorPrefill = descriptor.mode === BitNetMode.Prefill
  val descriptorProjection =
    BitNetResidentLinearProtocol.isProjectionKind(descriptor.kind)
  val descriptorEngineMask = Bits(engines bits)
  descriptorEngineMask := B((BigInt(1) << engines) - 1, engines bits)
  when(descriptorPrefill) {
    for (engine <- 0 until engines) {
      descriptorEngineMask(engine) := descriptor.tokenMask(
        engine * locals + locals - 1 downto engine * locals).orR
    }
  }

  // Rebuild the rich external payload only at the endpoint boundary.  The
  // descriptor cannot change while a pending pair is visible because a new
  // launch is blocked through completion retirement, so every field below is
  // stable under arbitrary output backpressure.
  val selectedLocalMask = Bits(locals bits)
  selectedLocalMask := descriptor.tokenMask(locals - 1 downto 0)
  for (engine <- 1 until engines) {
    when(selectedEngine === engine) {
      selectedLocalMask := descriptor.tokenMask(
        (engine + 1) * locals - 1 downto engine * locals)
    }
  }
  when(!descriptorPrefill) { selectedLocalMask := B"2'b01" }

  val selectedGlobalMask = Bits(cfg.tokenLanes bits)
  selectedGlobalMask := B(1, cfg.tokenLanes bits)
  when(descriptorPrefill) {
    selectedGlobalMask := selectedLocalMask.resize(cfg.tokenLanes)
    for (engine <- 1 until engines) {
      when(selectedEngine === engine) {
        selectedGlobalMask := (selectedLocalMask.resize(cfg.tokenLanes) <<
          (engine * locals)).resize(cfg.tokenLanes)
      }
    }
  }

  val selectedOutputOffset = UInt(b.outputIndexWidth bits)
  selectedOutputOffset := selectedPending.logicalOutputGroup
  when(descriptorProjection) {
    selectedOutputOffset :=
      (selectedPending.logicalOutputGroup << 1).resize(b.outputIndexWidth)
  }
  val selectedDescriptorBase = UInt(b.outputIndexWidth bits)
  val selectedDescriptorEnd = UInt(b.outputIndexWidth bits)
  val selectedDescriptorScale = UInt(32 bits)
  selectedDescriptorBase := descriptor.matrixTask.outputBase(0)
  selectedDescriptorEnd := descriptor.matrixTask.outputEndExclusive(0)
  selectedDescriptorScale := descriptor.matrixTask.weightScaleQ16(0)
  for (engine <- 1 until engines) {
    when(selectedEngine === engine) {
      selectedDescriptorBase := descriptor.matrixTask.outputBase(engine)
      selectedDescriptorEnd := descriptor.matrixTask.outputEndExclusive(engine)
      selectedDescriptorScale := descriptor.matrixTask.weightScaleQ16(engine)
    }
  }
  val selectedPhysicalBase = (selectedDescriptorBase +
    selectedOutputOffset).resize(b.outputIndexWidth)
  val selectedBaseWide = selectedPhysicalBase.resize(
    b.outputIndexWidth + 1)
  val selectedEndWide = selectedDescriptorEnd.resize(b.outputIndexWidth + 1)
  val selectedLogicalRows = UInt(2 bits)
  selectedLogicalRows := b.outputLanes
  when(!descriptorProjection) { selectedLogicalRows := 1 }
  val selectedOutputLaneMask = Bits(b.outputLanes bits)
  for (row <- 0 until b.outputLanes) {
    selectedOutputLaneMask(row) := U(row, 2 bits) < selectedLogicalRows &&
      selectedBaseWide + row < selectedEndWide
  }

  io.output.valid := pendingValid.asBits.orR
  io.output.requestId := descriptor.requestId
  io.output.layer := descriptor.layer
  io.output.kind := descriptor.kind
  io.output.mode := descriptor.mode
  io.output.sequenceId := descriptor.sequenceId
  io.output.engine := selectedEngine
  io.output.physicalOutputBase := selectedPhysicalBase
  io.output.globalTokenLaneMask := selectedGlobalMask
  io.output.outputLaneMask := selectedOutputLaneMask
  for (local <- 0 until locals) {
    val selectedGlobalLane = UInt(Math.max(1, log2Up(cfg.tokenLanes)) bits)
    selectedGlobalLane := U(local, selectedGlobalLane.getWidth bits)
    for (engine <- 1 until engines) {
      when(selectedEngine === engine) {
        selectedGlobalLane := U(engine * locals + local,
          selectedGlobalLane.getWidth bits)
      }
    }

    val reconstructedTaskId = UInt(m.taskIdWidth bits)
    reconstructedTaskId := (descriptor.matrixTask.taskIdBase +
      (selectedPending.logicalOutputGroup.resize(m.taskIdWidth) << 1) +
      selectedEngine).resize(m.taskIdWidth)
    when(descriptorPrefill) {
      reconstructedTaskId := (descriptor.matrixTask.taskIdBase +
        (selectedPending.logicalOutputGroup.resize(m.taskIdWidth) <<
          log2Up(cfg.tokenLanes)) +
        selectedGlobalLane).resize(m.taskIdWidth)
    }

    val reconstructedTokenId = UInt(b.tokenIdWidth bits)
    reconstructedTokenId := descriptor.matrixTask.tokenId(0)
    val reconstructedFactor = UInt(32 bits)
    reconstructedFactor := descriptor.matrixTask.activationFactorQ16(0)
    when(descriptorPrefill) {
      reconstructedTokenId := descriptor.matrixTask.tokenId(
        selectedGlobalLane)
      reconstructedFactor := descriptor.matrixTask.activationFactorQ16(
        selectedGlobalLane)
    }

    io.output.taskId(local) := Mux(selectedLocalMask(local),
      reconstructedTaskId, U(0, m.taskIdWidth bits))
    io.output.tokenId(local) := Mux(selectedLocalMask(local),
      reconstructedTokenId, U(0, b.tokenIdWidth bits))
    io.output.activationFactorQ16(local) := Mux(
      selectedLocalMask(local), reconstructedFactor, U(0, 32 bits))
    for (row <- 0 until b.outputLanes) {
      // Inactive lanes are canonicalised once, when the compact pending
      // register is written.  Keep the 128-bit value path free of another
      // layer of output gating.
      io.output.rawRows(local)(row) := selectedPending.rawRows(local)(row)
    }
  }
  io.output.weightScaleQ16 := selectedDescriptorScale
  io.output.physicalVectorFirst := selectedPhysicalBase === 0
  io.output.physicalVectorLast :=
    selectedBaseWide + selectedLogicalRows >= selectedEndWide
  when(!io.output.valid) { outputSelectionLocked := False }
  when(io.output.valid && !io.output.ready && !outputSelectionLocked) {
    outputSelectionLocked := True
    lockedOutputEngine := selectedEngine
  }
  when(io.output.fire) {
    outputSelectionLocked := False
    for (engine <- 0 until engines) {
      when(selectedEngine === engine) { pendingValid(engine) := False }
    }
    roundRobin := selectedEngine + 1
    roundEmittedPairs := roundEmittedPairs + 1
    emittedCount := emittedCount + 1
  }

  val launchLegal = BitNetResidentLinearProtocol.launchLegal(
    launchPending, cfg)

  io.launch.ready := state === State.Idle && !completionValid &&
    !launchPendingValid
  when(io.launch.fire) {
    launchPending := io.launch.payload
    launchPendingValid := True
  }

  when(launchPendingValid && state === State.Idle && !completionValid) {
    launchPendingValid := False
    when(launchLegal) {
      descriptor := launchPending
      resultBeatCount.foreach(_ := 0)
      kTileIndex.foreach(_ := 0)
      map0UpperPhase.foreach(_ := False)
      outputGroup.foreach(_ := 0)
      pendingValid.foreach(_ := False)
      roundAcceptedResults := 0
      roundEmittedPairs := 0
      activeFaultCode := 0
      acceptedCount := acceptedCount + 1
      state := State.Active
    } otherwise {
      stickyFault := True
      failedCount := failedCount + 1
      completionReg.requestId := launchPending.requestId
      completionReg.layer := launchPending.layer
      completionReg.kind := launchPending.kind
      completionReg.sequenceId := launchPending.sequenceId
      completionReg.success := False
      completionReg.fatal := False
      completionReg.faultCode := 0
      completionReg.faultCode(
        BitNetResidentLinearFault.InvalidLaunch) := True
      completionReg.acceptedResultBeats := 0
      completionReg.emittedPairs := 0
      completionValid := True
    }
  }

  def fillCompletion(success: Bool, faultCode: Bits): Unit = {
    completionReg.requestId := descriptor.requestId
    completionReg.layer := descriptor.layer
    completionReg.kind := descriptor.kind
    completionReg.sequenceId := descriptor.sequenceId
    completionReg.success := success
    completionReg.fatal := False
    completionReg.faultCode := faultCode
    completionReg.acceptedResultBeats := roundAcceptedResults
    completionReg.emittedPairs := roundEmittedPairs
    completionValid := True
  }

  // The compact result authenticates only transport identity.  Rich task,
  // token, scale, output-mask and vector framing are reconstructed from the
  // immutable launch descriptor at the output boundary above.
  val expectedLocalMask = Vec(Bits(locals bits), engines)
  val expectedLogicalOutputGroup = Vec(
    UInt(b.outputIndexWidth bits), engines)
  val expectedFirstK = Vec(Bool(), engines)
  val engineExpectedLastK = Vec(Bool(), engines)
  val expectedAccumulatorTag = Vec(
    Vec(UInt(m.accumulatorTagWidth bits), locals), engines)

  for (engine <- 0 until engines) {
    expectedLocalMask(engine) := B"2'b01"
    when(descriptorPrefill) {
      expectedLocalMask(engine) := descriptor.tokenMask(
        engine * locals + locals - 1 downto engine * locals)
    }

    expectedLogicalOutputGroup(engine) := outputGroup(engine)
    when(descriptorProjection && map0UpperPhase(engine)) {
      expectedLogicalOutputGroup(engine) :=
        (outputGroup(engine) + 1).resize(b.outputIndexWidth)
    }

    expectedFirstK(engine) := kTileIndex(engine) === 0
    engineExpectedLastK(engine) := kTileIndex(engine) ===
      descriptor.matrixTask.kTilesPerOutput - 1

    for (local <- 0 until locals) {
      expectedAccumulatorTag(engine)(local) :=
        expectedLogicalOutputGroup(engine).resize(m.accumulatorTagWidth)
      when(descriptorPrefill) {
        expectedAccumulatorTag(engine)(local) :=
          ((expectedLogicalOutputGroup(engine).resize(
            m.accumulatorTagWidth) << 1) + local).resize(
              m.accumulatorTagWidth)
      }
    }
  }

  val engineFire = Vec(Bool(), engines)
  val engineMetadataLegal = Vec(Bool(), engines)
  for (engine <- 0 until engines) {
    val pair = io.engineResult(engine).payload
    val expectedFormatClass = UInt(1 bits)
    expectedFormatClass :=
      BitNetProductionCompactDotFormatClass.CanonicalI8Row
    when(descriptorProjection) {
      expectedFormatClass :=
        BitNetProductionCompactDotFormatClass.Map0KMajor
    }
    val rowsCanonical = Vec(Bool(), locals)
    for (local <- 0 until locals) {
      rowsCanonical(local) := True
      when(!expectedLocalMask(engine)(local)) {
        rowsCanonical(local) := pair.rawRows(local).asBits === 0
      } elsewhen(expectedFormatClass ===
          BitNetProductionCompactDotFormatClass.CanonicalI8Row) {
        rowsCanonical(local) := pair.rawRows(local)(1) === 0
      }
    }
    val implicitLane1Tag = (pair.baseAccumulatorTag + 1).resize(
      m.accumulatorTagWidth)
    engineMetadataLegal(engine) :=
      pair.validMask === expectedLocalMask(engine) &&
        pair.validMask =/= B"2'b00" &&
        pair.sequenceId === descriptor.sequenceId &&
        pair.baseAccumulatorTag === expectedAccumulatorTag(engine)(0) &&
        (!expectedLocalMask(engine)(1) ||
          implicitLane1Tag === expectedAccumulatorTag(engine)(1)) &&
        pair.firstK === expectedFirstK(engine) &&
        pair.lastK === engineExpectedLastK(engine) &&
        pair.formatClass === expectedFormatClass &&
        rowsCanonical.asBits.andR
    io.engineResult(engine).ready := state === State.Active &&
      descriptorEngineMask(engine) &&
      resultBeatCount(engine) < descriptor.matrixTask.expectedBeats &&
      (!engineExpectedLastK(engine) || !pendingValid(engine))
    engineFire(engine) := io.engineResult(engine).fire
  }

  val disabledResultFire = (0 until engines).map { engine =>
    state === State.Active && !descriptorEngineMask(engine) &&
      io.engineResult(engine).valid
  }.reduce(_ || _)
  when(state === State.Active) {
    for (engine <- 0 until engines) {
      when(!descriptorEngineMask(engine)) {
        io.engineResult(engine).ready := True
      }
    }
  }
  val metadataFault = (0 until engines).map { engine =>
    engineFire(engine) && !engineMetadataLegal(engine)
  }.reduce(_ || _) || disabledResultFire

  when(state === State.Active) {
    for (engine <- 0 until engines) {
      when(engineFire(engine)) {
        resultBeatCount(engine) := resultBeatCount(engine) + 1
        when(BitNetResidentLinearProtocol.isProjectionKind(
            descriptor.kind)) {
          // Lower and upper results describe the same K tile.  Only the upper
          // phase advances K; completing its final K advances by the two
          // physical pairs represented by one compact Map0 raw group.
          when(map0UpperPhase(engine)) {
            map0UpperPhase(engine) := False
            when(engineExpectedLastK(engine)) {
              kTileIndex(engine) := 0
              outputGroup(engine) := outputGroup(engine) + 2
            } otherwise {
              kTileIndex(engine) := kTileIndex(engine) + 1
            }
          } otherwise {
            map0UpperPhase(engine) := True
          }
        } otherwise {
          when(engineExpectedLastK(engine)) {
            kTileIndex(engine) := 0
            outputGroup(engine) := outputGroup(engine) + 1
          } otherwise {
            kTileIndex(engine) := kTileIndex(engine) + 1
          }
        }

        when(engineMetadataLegal(engine) &&
            engineExpectedLastK(engine) && !metadataFault) {
          pendingRawPair(engine).logicalOutputGroup :=
            expectedLogicalOutputGroup(engine)
          for (local <- 0 until locals) {
            for (row <- 0 until b.outputLanes) {
              pendingRawPair(engine).rawRows(local)(row) :=
                io.engineResult(engine).payload.rawRows(local)(row)
              when(!expectedLocalMask(engine)(local)) {
                pendingRawPair(engine).rawRows(local)(row) := 0
              }
            }
          }
          pendingValid(engine) := True
        }
      }
    }

    when(engineFire.asBits.orR) {
      roundAcceptedResults := roundAcceptedResults +
        CountOne(engineFire.asBits).resize(roundAcceptedResults.getWidth)
    }

    when(metadataFault) {
      stickyFault := True
      activeFaultCode := 0
      activeFaultCode(BitNetResidentLinearFault.ResultMetadata) := True
      pendingValid.foreach(_ := False)
      state := State.FaultDrain
    }
  }

  when(state === State.FaultDrain) {
    pendingValid.foreach(_ := False)
    val drainFire = Bits(engines bits)
    drainFire := 0
    for (engine <- 0 until engines) {
      io.engineResult(engine).ready := !descriptorEngineMask(engine) ||
        resultBeatCount(engine) < descriptor.matrixTask.expectedBeats
      when(io.engineResult(engine).fire && descriptorEngineMask(engine) &&
          resultBeatCount(engine) < descriptor.matrixTask.expectedBeats) {
        resultBeatCount(engine) := resultBeatCount(engine) + 1
        drainFire(engine) := True
      }
    }
    when(drainFire.orR) {
      roundAcceptedResults := roundAcceptedResults +
        CountOne(drainFire).resize(roundAcceptedResults.getWidth)
    }
  }

  val allResultsDone = (0 until engines).map { engine =>
    !descriptorEngineMask(engine) ||
      resultBeatCount(engine) === descriptor.matrixTask.expectedBeats
  }.reduce(_ && _)
  when(state === State.Active && allResultsDone && !pendingValid.asBits.orR) {
    completedCount := completedCount + 1
    fillCompletion(True, B(0, 16 bits))
    state := State.Idle
  }
  when(state === State.FaultDrain && allResultsDone) {
    failedCount := failedCount + 1
    fillCompletion(False, activeFaultCode)
    state := State.Idle
  }

  // Stray returns cannot block an upstream CDC FIFO and never become a typed
  // payload.  The island treats this sticky condition as an invariant fault.
  when(state === State.Idle) {
    for (engine <- 0 until engines) {
      io.engineResult(engine).ready := True
      when(io.engineResult(engine).fire) { stickyFault := True }
    }
  }

  io.busy := state =/= State.Idle || completionValid ||
    pendingValid.asBits.orR
  io.protocolFaultSticky := stickyFault
  io.acceptedLaunches := acceptedCount
  io.completedLaunches := completedCount
  io.failedLaunches := failedCount
  io.emittedPairs := emittedCount
}

object GenerateBitNetResidentLinearResultEndpoint extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-linear-result-endpoint")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLinearResultEndpoint())
}
