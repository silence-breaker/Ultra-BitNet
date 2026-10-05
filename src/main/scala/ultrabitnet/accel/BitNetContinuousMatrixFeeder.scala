package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** Dot operations admitted by the production continuous-array feeder.
  *
  * SV is deliberately not encoded here.  SV/context traffic is scheduled by
  * BitNetKvContextCoordinator because it owns the KV slot lifetime and the
  * Q15 factor metadata.  Keeping that ownership in one place prevents a
  * projection feeder from manufacturing context tags.
  */
object BitNetContinuousDotKind extends SpinalEnum(binarySequential) {
  val Projection, Qk, LmHead = newElement()
}

case class BitNetContinuousMatrixFeederConfig(
    matrix: BitNetContinuousMatrixConfig = BitNetContinuousMatrixConfig(),
    tokenLanes: Int = BitNetConfig.Production.tokenParallelism,
    engineCount: Int = BitNetConfig.Production.engineCount,
    kTileCountWidth: Int = 16
) {
  require(tokenLanes == 2 || tokenLanes == 4,
    "the production schedule must be TP2 or TP4")
  require(engineCount >= 1 && engineCount <= 2,
    "one or two TP2 continuous arrays are supported")
  require(matrix.tokenLanesPerEngine == 2)
  require(tokenLanes == engineCount * matrix.tokenLanesPerEngine)
  require(matrix.base.siteCount == 64 || matrix.base.siteCount == 128)
  require(kTileCountWidth >= 2)
}

/** One resident matrix task.  It describes a complete continuous sequence;
  * per-beat firstK/lastK, tags and row addresses are generated in hardware.
  *
  * `outputBase` and `outputEndExclusive` are absolute row indices for each
  * engine.  During prefill the two entries must be identical because the same
  * weight tile is broadcast.  During decode they name independent row/head/
  * vocabulary shards.  A Map0 projection is a group-of-four physical
  * contract: every base/span is four-row aligned and one raw group always
  * emits its lower and upper pair.  Therefore projection `expectedBeats` is
  * at least two and even; no implicit output-row tail is manufactured.
  */
case class BitNetContinuousMatrixTask(
    cfg: BitNetContinuousMatrixFeederConfig
) extends Bundle {
  private val m = cfg.matrix
  private val b = m.base

  val scheduleMode = BitNetMode()
  val operation = BitNetContinuousDotKind()
  val sequenceId = UInt(m.sequenceIdWidth bits)
  val taskIdBase = UInt(m.taskIdWidth bits)
  val expectedBeats = UInt(m.beatCountWidth bits)
  val kTilesPerOutput = UInt(cfg.kTileCountWidth bits)

  /** Prefill accepts only prefix masks 0001/0011/0111/1111.  Decode is 0001. */
  val activeTokenMask = Bits(cfg.tokenLanes bits)
  val tokenId = Vec(UInt(b.tokenIdWidth bits), cfg.tokenLanes)
  val activationFactorQ16 = Vec(UInt(32 bits), cfg.tokenLanes)

  val weightScaleQ16 = Vec(UInt(32 bits), cfg.engineCount)
  val outputBase = Vec(UInt(b.outputIndexWidth bits), cfg.engineCount)
  val outputEndExclusive = Vec(
    UInt(b.outputIndexWidth bits), cfg.engineCount)
  val routeFinalToDynamicQuant = Bool()
  val residualEnable = Bool()
}

/** Four on-chip activation vectors paired with one typed weight group. */
case class BitNetNormalizedActivationTile(
    cfg: BitNetContinuousMatrixFeederConfig
) extends Bundle {
  private val m = cfg.matrix
  val sequenceId = UInt(m.sequenceIdWidth bits)
  val beatOrdinal = UInt(m.beatCountWidth bits)
  val tokenActivation = Vec(
    Vec(SInt(m.base.activationWidth bits), m.base.siteCount), cfg.tokenLanes)
  val fault = Bool()
}

object BitNetContinuousFeederState extends SpinalEnum {
  val Idle, SendSequence, StreamTiles = newElement()
}

/**
  * Slow-domain production front-end for [[BitNetTp4MatrixCluster]].
  *
  * There is no GEMV packet or token-processor route at this boundary.  A task
  * is converted to two typed continuous sequences, while matched normalised
  * weight/activation tiles are converted directly to the two cluster beats.
  * The two beat outputs form an atomic barrier: either every enabled engine
  * accepts the tile or neither input advances.  Consequently arbitrary CDC
  * FIFO backpressure cannot duplicate one shard or lose the other.
  */
class BitNetContinuousMatrixFeeder(
    cfg: BitNetContinuousMatrixFeederConfig =
      BitNetContinuousMatrixFeederConfig()
) extends Component {
  private val m = cfg.matrix
  private val b = m.base

  val io = new Bundle {
    val task = slave(Stream(BitNetContinuousMatrixTask(cfg)))
    val map0RawGroup = slave(Stream(BitNetMap0RawGroupTile(cfg)))
    val i8WeightBeat = slave(Stream(BitNetI8WeightBeat(cfg)))
    val activationTile = slave(Stream(BitNetNormalizedActivationTile(cfg)))

    val sequence = Vec(
      master(Stream(BitNetContinuousSequence(m))), cfg.engineCount)
    val map0Beat = Vec(
      master(Stream(BitNetMap0PairMatrixBeat(cfg))), cfg.engineCount)
    val i8Beat = Vec(
      master(Stream(BitNetI8MatrixBeat(cfg))), cfg.engineCount)

    val active = out Bool()
    val activeEngineMask = out Bits(cfg.engineCount bits)
    val nextBeatOrdinal = out UInt(m.beatCountWidth bits)
    val nextKTile = out UInt(cfg.kTileCountWidth bits)
    val outputGroup = out UInt(b.outputIndexWidth bits)
    val holdActive = out Bool()
    val map0UpperPhase = out Bool()
    val pendingEngines = out Bits(cfg.engineCount bits)
    val protocolFaultSticky = out Bool()
    val acceptedTasks = out UInt(32 bits)
    val acceptedMap0RawGroups = out UInt(32 bits)
    val acceptedI8Beats = out UInt(32 bits)
    val droppedInputPairs = out UInt(32 bits)
  }

  val state = Reg(BitNetContinuousFeederState()) init (
    BitNetContinuousFeederState.Idle)
  val taskReg = Reg(BitNetContinuousMatrixTask(cfg)) init (
    BitNetContinuousMatrixTask(cfg).getZero)
  val engineMaskReg = Reg(Bits(cfg.engineCount bits)) init (0)
  val sequencePending = Reg(Bits(cfg.engineCount bits)) init (0)
  val nextI8BeatOrdinal = Reg(UInt(m.beatCountWidth bits)) init (0)
  val nextMap0LowerOrdinal = Reg(UInt(m.beatCountWidth bits)) init (0)
  val nextMap0UpperOrdinal = Reg(UInt(m.beatCountWidth bits)) init (0)
  val kTile = Reg(UInt(cfg.kTileCountWidth bits)) init (0)
  val outputGroupReg = Reg(UInt(b.outputIndexWidth bits)) init (0)

  // One input group is owned until every enabled engine has consumed it.
  // The upstream Streams remain unconsumed and therefore stable through the
  // complete ownership interval.  Only compact phase/pending state is kept
  // here; the cluster mailboxes are the unique wide payload registers.
  val holdValid = RegInit(False)
  val map0PairSelect = Reg(BitNetMap0PairSelect()) init
    BitNetMap0PairSelect.LowerRows01
  val beatPending = Reg(Bits(cfg.engineCount bits)) init 0

  val stickyFault = RegInit(False)
  val taskCount = Reg(UInt(32 bits)) init (0)
  val map0RawGroupCount = Reg(UInt(32 bits)) init (0)
  val i8BeatCount = Reg(UInt(32 bits)) init (0)
  val droppedCount = Reg(UInt(32 bits)) init (0)

  val taskIsPrefill = io.task.payload.scheduleMode === BitNetMode.Prefill
  val taskIsDecode = io.task.payload.scheduleMode === BitNetMode.Decode
  val prefillMaskLegal = (1 to cfg.tokenLanes).map { count =>
    io.task.payload.activeTokenMask === B((BigInt(1) << count) - 1,
      cfg.tokenLanes bits)
  }.reduce(_ || _)
  val decodeMaskLegal = io.task.payload.activeTokenMask ===
    B(1, cfg.tokenLanes bits)
  val prefillBoundsMatch = (1 until cfg.engineCount).map { engine =>
    io.task.payload.outputBase(0) === io.task.payload.outputBase(engine) &&
      io.task.payload.outputEndExclusive(0) ===
        io.task.payload.outputEndExclusive(engine) &&
      io.task.payload.weightScaleQ16(0) ===
        io.task.payload.weightScaleQ16(engine)
  }.foldLeft(True)(_ && _)
  val boundsLegal = (0 until cfg.engineCount).map { engine =>
    io.task.payload.outputEndExclusive(engine) >
      io.task.payload.outputBase(engine)
  }.reduce(_ && _)
  val quantRouteLegal = !io.task.payload.routeFinalToDynamicQuant ||
    io.task.payload.operation === BitNetContinuousDotKind.Projection
  val taskIsProjection = io.task.payload.operation ===
    BitNetContinuousDotKind.Projection
  val projectionBoundsAligned = (0 until cfg.engineCount).map { engine =>
    io.task.payload.outputBase(engine)(1 downto 0) === 0 &&
      io.task.payload.outputEndExclusive(engine)(1 downto 0) === 0
  }.reduce(_ && _)
  val projectionShapeLegal = !taskIsProjection ||
    (!io.task.payload.expectedBeats(0) &&
      io.task.payload.expectedBeats >=
        (io.task.payload.kTilesPerOutput.resize(m.beatCountWidth) << 1) &&
      projectionBoundsAligned)
  val taskLegal = io.task.payload.expectedBeats =/= 0 &&
    io.task.payload.kTilesPerOutput =/= 0 &&
    io.task.payload.kTilesPerOutput.resize(m.beatCountWidth) <=
      io.task.payload.expectedBeats && boundsLegal && quantRouteLegal &&
    projectionShapeLegal &&
    ((taskIsPrefill && prefillMaskLegal && prefillBoundsMatch) ||
      (taskIsDecode && decodeMaskLegal))

  val requestedEngineMask = Bits(cfg.engineCount bits)
  requestedEngineMask := B((BigInt(1) << cfg.engineCount) - 1,
    cfg.engineCount bits)
  when(taskIsPrefill) {
    for (engine <- 0 until cfg.engineCount) {
      requestedEngineMask(engine) := io.task.payload.activeTokenMask(
        engine * m.tokenLanesPerEngine + m.tokenLanesPerEngine - 1 downto
          engine * m.tokenLanesPerEngine).orR
    }
  }

  io.task.ready := state === BitNetContinuousFeederState.Idle
  when(io.task.fire) {
    when(taskLegal) {
      taskCount := taskCount + 1
      taskReg := io.task.payload
      engineMaskReg := requestedEngineMask
      sequencePending := requestedEngineMask
      nextI8BeatOrdinal := 0
      nextMap0LowerOrdinal := 0
      nextMap0UpperOrdinal :=
        io.task.payload.kTilesPerOutput.resize(m.beatCountWidth)
      kTile := 0
      outputGroupReg := 0
      holdValid := False
      beatPending := 0
      state := BitNetContinuousFeederState.SendSequence
    } otherwise {
      stickyFault := True
    }
  }

  val emittedMatrixMode = BitNetUnifiedMatrixMode()
  val emittedFormat = BitNetUnifiedOperandFormat()
  emittedMatrixMode := BitNetUnifiedMatrixMode.TernaryProjection
  emittedFormat := BitNetUnifiedOperandFormat.Map0Packed
  when(taskReg.operation === BitNetContinuousDotKind.Qk) {
    emittedMatrixMode := BitNetUnifiedMatrixMode.QkDotI8
    emittedFormat := BitNetUnifiedOperandFormat.I8Pair
  } elsewhen(taskReg.operation === BitNetContinuousDotKind.LmHead) {
    emittedMatrixMode := BitNetUnifiedMatrixMode.LmHeadDotI8
    emittedFormat := BitNetUnifiedOperandFormat.I8Pair
  }

  val sequenceFireMask = Bits(cfg.engineCount bits)
  for (engine <- 0 until cfg.engineCount) {
    val output = io.sequence(engine)
    output.valid := state === BitNetContinuousFeederState.SendSequence &&
      sequencePending(engine)
    output.payload.mode := emittedMatrixMode
    output.payload.format := emittedFormat
    output.payload.sequenceId := taskReg.sequenceId
    output.payload.expectedBeats := taskReg.expectedBeats
    output.payload.weightScaleQ16 := taskReg.weightScaleQ16(engine)
    output.payload.outputFeatureCount := taskReg.outputEndExclusive(engine)
    output.payload.routeFinalToDynamicQuant :=
      taskReg.routeFinalToDynamicQuant
    output.payload.residualEnable := taskReg.residualEnable
    for (lane <- 0 until m.tokenLanesPerEngine) {
      val globalPrefillLane = engine * m.tokenLanesPerEngine + lane
      output.payload.activationFactorQ16(lane) :=
        taskReg.activationFactorQ16(globalPrefillLane)
      when(taskReg.scheduleMode === BitNetMode.Decode) {
        output.payload.activationFactorQ16(lane) :=
          taskReg.activationFactorQ16(0)
      }
    }
    sequenceFireMask(engine) := output.fire
  }

  val sequenceRemaining = sequencePending & ~sequenceFireMask
  when(state === BitNetContinuousFeederState.SendSequence) {
    sequencePending := sequenceRemaining
    when(sequenceRemaining === 0) {
      state := BitNetContinuousFeederState.StreamTiles
    }
  }

  val isProjection = taskReg.operation ===
    BitNetContinuousDotKind.Projection
  val inputOpen = state === BitNetContinuousFeederState.StreamTiles
  val expectedFirstK = kTile === 0
  val expectedLastK = kTile === taskReg.kTilesPerOutput - 1

  val map0EngineTagsLegal = (0 until cfg.engineCount).map { engine =>
    val expectedBase = (taskReg.outputBase(engine) +
      (outputGroupReg << 1)).resize(b.outputIndexWidth)
    io.map0RawGroup.payload.engineTag(engine).outputBase === expectedBase &&
      io.map0RawGroup.payload.engineTag(engine).groupOrdinal ===
        (expectedBase >> 2).resize(b.outputIndexWidth)
  }.reduce(_ && _)
  val map0PrefillTagsMatch = (1 until cfg.engineCount).map { engine =>
    io.map0RawGroup.payload.engineTag(0).groupOrdinal ===
        io.map0RawGroup.payload.engineTag(engine).groupOrdinal &&
      io.map0RawGroup.payload.engineTag(0).outputBase ===
        io.map0RawGroup.payload.engineTag(engine).outputBase
  }.foldLeft(True)(_ && _)
  val map0TagsLegal =
    io.map0RawGroup.payload.sequenceId === taskReg.sequenceId &&
      io.activationTile.payload.sequenceId === taskReg.sequenceId &&
      io.map0RawGroup.payload.scheduleMode === taskReg.scheduleMode &&
      io.map0RawGroup.payload.kTile.resize(cfg.kTileCountWidth) === kTile &&
      io.map0RawGroup.payload.kTileCount.resize(cfg.kTileCountWidth) ===
        taskReg.kTilesPerOutput &&
      io.map0RawGroup.payload.lowerBeatOrdinal === nextMap0LowerOrdinal &&
      io.map0RawGroup.payload.upperBeatOrdinal === nextMap0UpperOrdinal &&
      io.activationTile.payload.beatOrdinal === nextMap0LowerOrdinal &&
      io.map0RawGroup.payload.firstRawGroup === expectedFirstK &&
      io.map0RawGroup.payload.lastRawGroup === expectedLastK &&
      map0EngineTagsLegal &&
      (taskReg.scheduleMode === BitNetMode.Decode || map0PrefillTagsMatch)
  val map0PayloadLegal = map0TagsLegal &&
    !io.map0RawGroup.payload.fault && !io.activationTile.payload.fault

  val expectedI8Use = BitNetI8WeightUse()
  expectedI8Use := BitNetI8WeightUse.Qk
  when(taskReg.operation === BitNetContinuousDotKind.LmHead) {
    expectedI8Use := BitNetI8WeightUse.LmHead
  }
  val i8EngineTagsLegal = (0 until cfg.engineCount).map { engine =>
    io.i8WeightBeat.payload.engineOutputBase(engine) ===
      (taskReg.outputBase(engine) + outputGroupReg).resize(b.outputIndexWidth)
  }.reduce(_ && _)
  val i8TagsLegal =
    io.i8WeightBeat.payload.sequenceId === taskReg.sequenceId &&
      io.activationTile.payload.sequenceId === taskReg.sequenceId &&
      io.i8WeightBeat.payload.beatOrdinal === nextI8BeatOrdinal &&
      io.activationTile.payload.beatOrdinal === nextI8BeatOrdinal &&
      io.i8WeightBeat.payload.use === expectedI8Use &&
      io.i8WeightBeat.payload.firstK === expectedFirstK &&
      io.i8WeightBeat.payload.lastK === expectedLastK &&
      i8EngineTagsLegal
  val i8PayloadLegal = i8TagsLegal && !io.i8WeightBeat.payload.fault &&
    !io.activationTile.payload.fault

  val selectedWeightValid = Mux(
    isProjection, io.map0RawGroup.valid, io.i8WeightBeat.valid)
  val selectedPayloadLegal = Mux(
    isProjection, map0PayloadLegal, i8PayloadLegal)
  val selectedPairValid = selectedWeightValid && io.activationTile.valid
  val freshOffer = !holdValid && selectedPairValid && selectedPayloadLegal
  val activeOffer = inputOpen && (holdValid || freshOffer)
  val activePairSelect = BitNetMap0PairSelect()
  activePairSelect := map0PairSelect
  when(!holdValid) {
    activePairSelect := BitNetMap0PairSelect.LowerRows01
  }
  val activePending = Bits(cfg.engineCount bits)
  activePending := beatPending
  when(!holdValid) {
    activePending := engineMaskReg
  }

  val pairOutputGroup = UInt(b.outputIndexWidth bits)
  pairOutputGroup := outputGroupReg
  when(activePairSelect === BitNetMap0PairSelect.UpperRows23) {
    pairOutputGroup := outputGroupReg + 1
  }
  val selectedMap0Ordinal = UInt(m.beatCountWidth bits)
  selectedMap0Ordinal := io.map0RawGroup.payload.lowerBeatOrdinal
  when(activePairSelect === BitNetMap0PairSelect.UpperRows23) {
    selectedMap0Ordinal := io.map0RawGroup.payload.upperBeatOrdinal
  }

  val outputFireMask = Bits(cfg.engineCount bits)
  for (engine <- 0 until cfg.engineCount) {
    val map0Output = io.map0Beat(engine)
    val i8Output = io.i8Beat(engine)
    map0Output.valid := activeOffer && isProjection && activePending(engine)
    i8Output.valid := activeOffer && !isProjection && activePending(engine)
    outputFireMask(engine) := map0Output.fire || i8Output.fire

    map0Output.payload.sequenceId := io.map0RawGroup.payload.sequenceId
    map0Output.payload.beatOrdinal := selectedMap0Ordinal
    map0Output.payload.pairSelect := activePairSelect
    i8Output.payload.sequenceId := io.i8WeightBeat.payload.sequenceId
    i8Output.payload.beatOrdinal := io.i8WeightBeat.payload.beatOrdinal
    i8Output.payload.use := io.i8WeightBeat.payload.use

    map0Output.payload.meta.laneMask := B"2'b01"
    i8Output.payload.meta.laneMask := B"2'b01"
    when(taskReg.scheduleMode === BitNetMode.Prefill) {
      val localMask = taskReg.activeTokenMask(
        engine * m.tokenLanesPerEngine + m.tokenLanesPerEngine - 1 downto
          engine * m.tokenLanesPerEngine)
      map0Output.payload.meta.laneMask := localMask
      i8Output.payload.meta.laneMask := localMask
    }

    val selectedRawEngine = UInt(log2Up(cfg.engineCount) bits)
    selectedRawEngine := engine
    when(taskReg.scheduleMode === BitNetMode.Prefill) {
      selectedRawEngine := 0
    }
    for (site <- 0 until b.siteCount) {
      val rawBytes = Vec(Bits(8 bits), cfg.engineCount)
      for (candidate <- 0 until cfg.engineCount) {
        rawBytes(candidate) := Mux(isProjection,
          io.map0RawGroup.payload.engineRawGroup(candidate)(site),
          io.i8WeightBeat.payload.engineSiteWeight(candidate)(site))
      }
      map0Output.payload.rawWeights(site) :=
        rawBytes(selectedRawEngine)(3 downto 0)
      when(activePairSelect === BitNetMap0PairSelect.UpperRows23) {
        map0Output.payload.rawWeights(site) :=
          rawBytes(selectedRawEngine)(7 downto 4)
      }
      i8Output.payload.rawWeights(site) := rawBytes(selectedRawEngine)

      for (lane <- 0 until m.tokenLanesPerEngine) {
        val globalPrefillLane = engine * m.tokenLanesPerEngine + lane
        map0Output.payload.meta.activations(lane)(site) :=
          io.activationTile.payload.tokenActivation(globalPrefillLane)(site)
        i8Output.payload.meta.activations(lane)(site) :=
          io.activationTile.payload.tokenActivation(globalPrefillLane)(site)
        when(taskReg.scheduleMode === BitNetMode.Decode) {
          map0Output.payload.meta.activations(lane)(site) :=
            io.activationTile.payload.tokenActivation(0)(site)
          i8Output.payload.meta.activations(lane)(site) :=
            io.activationTile.payload.tokenActivation(0)(site)
        }
      }
    }

    for (lane <- 0 until m.tokenLanesPerEngine) {
      val globalLane = engine * m.tokenLanesPerEngine + lane
      val prefillMap0TagWide =
        (pairOutputGroup.resize(m.accumulatorTagWidth + 1) << 1) + lane
      val prefillI8TagWide =
        (outputGroupReg.resize(m.accumulatorTagWidth + 1) << 1) + lane

      map0Output.payload.meta.accumulatorTag(lane) :=
        pairOutputGroup.resize(m.accumulatorTagWidth)
      i8Output.payload.meta.accumulatorTag(lane) :=
        outputGroupReg.resize(m.accumulatorTagWidth)
      when(taskReg.scheduleMode === BitNetMode.Prefill) {
        map0Output.payload.meta.accumulatorTag(lane) :=
          prefillMap0TagWide.resize(m.accumulatorTagWidth)
        i8Output.payload.meta.accumulatorTag(lane) :=
          prefillI8TagWide.resize(m.accumulatorTagWidth)
      }
      map0Output.payload.meta.firstK(lane) :=
        io.map0RawGroup.payload.firstRawGroup
      map0Output.payload.meta.lastK(lane) :=
        io.map0RawGroup.payload.lastRawGroup
      i8Output.payload.meta.firstK(lane) := io.i8WeightBeat.payload.firstK
      i8Output.payload.meta.lastK(lane) := io.i8WeightBeat.payload.lastK

      val map0DecodeTaskOffset =
        (pairOutputGroup.resize(m.taskIdWidth) << 1) + engine
      val map0PrefillTaskOffset =
        (pairOutputGroup.resize(m.taskIdWidth) << log2Up(cfg.tokenLanes)) +
          globalLane
      val i8DecodeTaskOffset =
        (outputGroupReg.resize(m.taskIdWidth) << 1) + engine
      val i8PrefillTaskOffset =
        (outputGroupReg.resize(m.taskIdWidth) << log2Up(cfg.tokenLanes)) +
          globalLane
      map0Output.payload.meta.taskId(lane) :=
        (taskReg.taskIdBase + map0DecodeTaskOffset).resize(m.taskIdWidth)
      i8Output.payload.meta.taskId(lane) :=
        (taskReg.taskIdBase + i8DecodeTaskOffset).resize(m.taskIdWidth)
      when(taskReg.scheduleMode === BitNetMode.Prefill) {
        map0Output.payload.meta.taskId(lane) :=
          (taskReg.taskIdBase + map0PrefillTaskOffset).resize(m.taskIdWidth)
        i8Output.payload.meta.taskId(lane) :=
          (taskReg.taskIdBase + i8PrefillTaskOffset).resize(m.taskIdWidth)
      }

      map0Output.payload.meta.tokenId(lane) := taskReg.tokenId(0)
      i8Output.payload.meta.tokenId(lane) := taskReg.tokenId(0)
      when(taskReg.scheduleMode === BitNetMode.Prefill) {
        map0Output.payload.meta.tokenId(lane) := taskReg.tokenId(globalLane)
        i8Output.payload.meta.tokenId(lane) := taskReg.tokenId(globalLane)
      }
      map0Output.payload.meta.outputBase(lane) :=
        (io.map0RawGroup.payload.engineTag(engine).outputBase +
          Mux(activePairSelect === BitNetMap0PairSelect.UpperRows23,
            U(2, b.outputIndexWidth bits), U(0, b.outputIndexWidth bits)))
          .resize(b.outputIndexWidth)
      i8Output.payload.meta.outputBase(lane) :=
        io.i8WeightBeat.payload.engineOutputBase(engine)
    }
  }

  val pendingAfterFire = activePending & ~outputFireMask
  val phaseCompletes = activeOffer && activePending.orR &&
    !pendingAfterFire.orR
  val transactionCompletes = phaseCompletes &&
    (!isProjection ||
      activePairSelect === BitNetMap0PairSelect.UpperRows23)
  val selectedPairReady = Mux(
    selectedPayloadLegal, transactionCompletes, !holdValid)
  io.map0RawGroup.ready := inputOpen && isProjection &&
    io.activationTile.valid && selectedPairReady
  io.i8WeightBeat.ready := inputOpen && !isProjection &&
    io.activationTile.valid && selectedPairReady
  io.activationTile.ready := inputOpen && selectedWeightValid &&
    selectedPairReady

  when(activeOffer && !transactionCompletes) {
    holdValid := True
    beatPending := pendingAfterFire
    map0PairSelect := activePairSelect
    when(phaseCompletes && isProjection &&
        activePairSelect === BitNetMap0PairSelect.LowerRows01) {
        map0PairSelect := BitNetMap0PairSelect.UpperRows23
        beatPending := engineMaskReg
    }
  }

  when(io.map0RawGroup.fire) {
    holdValid := False
    beatPending := 0
    map0PairSelect := BitNetMap0PairSelect.LowerRows01
    when(map0PayloadLegal) {
      map0RawGroupCount := map0RawGroupCount + 1
      when(expectedLastK) {
        kTile := 0
        outputGroupReg := outputGroupReg + 2
        nextMap0LowerOrdinal := nextMap0LowerOrdinal +
          taskReg.kTilesPerOutput.resize(m.beatCountWidth) + 1
        nextMap0UpperOrdinal := nextMap0UpperOrdinal +
          taskReg.kTilesPerOutput.resize(m.beatCountWidth) + 1
      } otherwise {
        kTile := kTile + 1
        nextMap0LowerOrdinal := nextMap0LowerOrdinal + 1
        nextMap0UpperOrdinal := nextMap0UpperOrdinal + 1
      }
      when(io.map0RawGroup.payload.upperBeatOrdinal ===
          taskReg.expectedBeats - 1) {
        when(!io.map0RawGroup.payload.lastRawGroup) { stickyFault := True }
        state := BitNetContinuousFeederState.Idle
        engineMaskReg := 0
      }
    } otherwise {
      stickyFault := True
      droppedCount := droppedCount + 1
    }
  }

  when(io.i8WeightBeat.fire) {
    holdValid := False
    beatPending := 0
    when(i8PayloadLegal) {
      i8BeatCount := i8BeatCount + 1
      nextI8BeatOrdinal := nextI8BeatOrdinal + 1
      when(expectedLastK) {
        kTile := 0
        outputGroupReg := outputGroupReg + 1
      } otherwise {
        kTile := kTile + 1
      }
      when(io.i8WeightBeat.payload.beatOrdinal ===
          taskReg.expectedBeats - 1) {
        when(!io.i8WeightBeat.payload.lastK) { stickyFault := True }
        state := BitNetContinuousFeederState.Idle
        engineMaskReg := 0
      }
    } otherwise {
      stickyFault := True
      droppedCount := droppedCount + 1
    }
  }

  io.active := state =/= BitNetContinuousFeederState.Idle
  io.activeEngineMask := engineMaskReg
  io.nextBeatOrdinal := Mux(isProjection,
    nextMap0LowerOrdinal, nextI8BeatOrdinal)
  io.nextKTile := kTile
  io.outputGroup := outputGroupReg
  io.holdActive := holdValid
  io.map0UpperPhase := map0PairSelect ===
    BitNetMap0PairSelect.UpperRows23
  io.pendingEngines := beatPending
  io.protocolFaultSticky := stickyFault
  io.acceptedTasks := taskCount
  io.acceptedMap0RawGroups := map0RawGroupCount
  io.acceptedI8Beats := i8BeatCount
  io.droppedInputPairs := droppedCount
}

/** Pure-Scala reference mapper used by schedule generation and unit tests. */
object BitNetContinuousMatrixFeederOracle {
  sealed trait Schedule
  case object Prefill extends Schedule
  case object Decode extends Schedule

  /**
    * One Map0 result in the order in which the feeder presents it to an
    * engine.  A compact raw group is consumed K-major: lower rows 0/1 and
    * upper rows 2/3 are emitted next to each other before the next K tile.
    * Consequently the semantic beat ordinals are
    * `0, kTilesPerOutput, 1, kTilesPerOutput + 1, ...`, rather than the
    * pair-major `0, 1, 2, ...` order used by I8 traffic.
    */
  final case class ProjectionArrival(
      logicalPair: Int,
      kTile: Int,
      upper: Boolean,
      semanticBeatOrdinal: Int)

  def projectionArrival(
      arrivalOrdinal: Int,
      kTilesPerOutput: Int): ProjectionArrival = {
    require(arrivalOrdinal >= 0)
    require(kTilesPerOutput > 0)
    val rawGroupOrdinal = arrivalOrdinal / 2
    val upper = (arrivalOrdinal & 1) != 0
    val pairBlock = rawGroupOrdinal / kTilesPerOutput
    val kTile = rawGroupOrdinal % kTilesPerOutput
    val logicalPair = pairBlock * 2 + (if (upper) 1 else 0)
    ProjectionArrival(
      logicalPair = logicalPair,
      kTile = kTile,
      upper = upper,
      semanticBeatOrdinal = logicalPair * kTilesPerOutput + kTile)
  }

  final case class LaneRoute(
      engine: Int,
      localLane: Int,
      globalTokenLane: Int,
      weightShard: Int,
      accumulatorTag: Int,
      taskId: Int,
      outputBase: Int,
      firstK: Boolean,
      lastK: Boolean)

  def routes(
      schedule: Schedule,
      activeTokenMask: Int,
      beatOrdinal: Int,
      kTilesPerOutput: Int,
      taskIdBase: Int,
      engineOutputBase: Seq[Int],
      projection: Boolean,
      accumulatorTags: Int = 64
  ): Seq[LaneRoute] = {
    require(kTilesPerOutput > 0)
    require(engineOutputBase.length == 2)
    val outputGroup = beatOrdinal / kTilesPerOutput
    val k = beatOrdinal % kTilesPerOutput
    val outputStride = if (projection) 2 else 1
    schedule match {
      case Prefill =>
        require(Set(1, 3, 7, 15).contains(activeTokenMask))
        for {
          globalLane <- 0 until 4
          if ((activeTokenMask >> globalLane) & 1) != 0
        } yield LaneRoute(
          engine = globalLane / 2,
          localLane = globalLane % 2,
          globalTokenLane = globalLane,
          weightShard = 0,
          accumulatorTag = (outputGroup * 2 + globalLane % 2) %
            accumulatorTags,
          taskId = taskIdBase + outputGroup * 4 + globalLane,
          outputBase = engineOutputBase.head + outputGroup * outputStride,
          firstK = k == 0,
          lastK = k == kTilesPerOutput - 1)
      case Decode =>
        require(activeTokenMask == 1)
        for (engine <- 0 until 2) yield LaneRoute(
          engine = engine,
          localLane = 0,
          globalTokenLane = 0,
          weightShard = engine,
          accumulatorTag = outputGroup % accumulatorTags,
          taskId = taskIdBase + outputGroup * 2 + engine,
          outputBase = engineOutputBase(engine) + outputGroup * outputStride,
          firstK = k == 0,
          lastK = k == kTilesPerOutput - 1)
    }
  }
}

object GenerateBitNetContinuousMatrixFeeder extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-continuous-matrix-feeder")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetContinuousMatrixFeeder())
}
