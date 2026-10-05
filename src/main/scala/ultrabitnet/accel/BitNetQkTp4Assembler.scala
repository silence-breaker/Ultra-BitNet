package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Geometry of the packet-free, pass-driven QK assembler.
  *
  * This block owns no multiplier array.  It presents native I8-pair beats to
  * the same two TP2 continuous engines used by projection/SV/LM head, while
  * preserving BitNet's mode-dependent four-lane interpretation.
  */
case class BitNetQkTp4AssemblerConfig(
    model: BitNetConfig = BitNetConfig.Production,
    continuous: BitNetContinuousMatrixConfig =
      BitNetContinuousMatrixConfig(),
    maxSeq: Int = 4096,
    factorWidth: Int = 32,
    metaFifoDepth: Int = 64,
    resultFifoDepth: Int = 8
) {
  require(model.engineCount >= 1 && model.engineCount <= 2)
  require(model.queryHeadCount == 20 && model.kvHeadCount == 5)
  require(model.gqaGroupSize == 4 && model.headSize == 128)
  require(continuous.tokenLanesPerEngine == 2)
  require(model.tokenParallelism ==
    model.engineCount * continuous.tokenLanesPerEngine)
  require(model.headSize % continuous.base.siteCount == 0)
  require(continuous.base.activationWidth == 8)
  require(continuous.base.accumulatorWidth == 32)
  require(continuous.base.tokenIdWidth >= model.sequenceWidth)
  require(maxSeq >= 64 && maxSeq <= model.maxSequenceLength && isPow2(maxSeq))
  require(factorWidth == 32)
  require(metaFifoDepth >= 64 && isPow2(metaFifoDepth))
  require(resultFifoDepth >= 4 && isPow2(resultFifoDepth))

  val laneCount: Int = model.tokenParallelism
  val engineCount: Int = model.engineCount
  val lanesPerEngine: Int = continuous.tokenLanesPerEngine
  val feeder: BitNetContinuousMatrixFeederConfig =
    BitNetContinuousMatrixFeederConfig(
      matrix = continuous,
      tokenLanes = laneCount,
      engineCount = engineCount)
  val queryHeadWidth: Int = log2Up(model.queryHeadCount)
  val kvHeadWidth: Int = log2Up(model.kvHeadCount)
  val passIndexWidth: Int = log2Up(model.queryHeadCount)
  val headMicrotiles: Int = model.headSize / continuous.base.siteCount
  val headMicrotileWidth: Int = scala.math.max(1, log2Up(headMicrotiles))
  require(headMicrotiles == 1 || headMicrotiles == 2)
}

/** Four resident int8 Q rows and their positive Q16.16 dequant factors. */
case class BitNetQkQueryBlock(cfg: BitNetQkTp4AssemblerConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val laneMask = Bits(cfg.laneCount bits)
  val tokenId = Vec(UInt(cfg.model.sequenceWidth bits), cfg.laneCount)
  val queryHeadId = Vec(UInt(cfg.queryHeadWidth bits), cfg.laneCount)
  val queryPosition = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.laneCount)
  val keyCount = Vec(UInt(cfg.model.sequenceWidth bits), cfg.laneCount)
  val q = Vec(
    Vec(SInt(cfg.continuous.base.activationWidth bits),
      cfg.model.headSize), cfg.laneCount)
  val qFactorQ16 = Vec(UInt(cfg.factorWidth bits), cfg.laneCount)
}

/** One ordered K-cache row, accepted exactly once and broadcast to engines. */
case class BitNetQkKeyBeat(cfg: BitNetQkTp4AssemblerConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val k = Vec(
    SInt(cfg.continuous.base.activationWidth bits), cfg.model.headSize)
  val kFactorQ16 = UInt(cfg.factorWidth bits)
  val keyPosition = UInt(cfg.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
  val success = Bool()
}

/**
  * Four raw S32 scores before dequantisation/head scaling.
  *
  * `laneValid` applies causality, per-token keyCount and engine tail masks.
  * Invalid rawDot lanes are forced to zero, while complete metadata remains
  * available for deterministic buffered-softmax routing.
  */
case class BitNetQkTp4RawScoreBeat(cfg: BitNetQkTp4AssemblerConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val laneValid = Bits(cfg.laneCount bits)
  val tokenId = Vec(UInt(cfg.model.sequenceWidth bits), cfg.laneCount)
  val queryHeadId = Vec(UInt(cfg.queryHeadWidth bits), cfg.laneCount)
  val queryPosition = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.laneCount)
  val keyCount = Vec(UInt(cfg.model.sequenceWidth bits), cfg.laneCount)
  val rawDot = Vec(
    SInt(cfg.continuous.base.accumulatorWidth bits), cfg.laneCount)
  val qFactorQ16 = Vec(UInt(cfg.factorWidth bits), cfg.laneCount)
  val kFactorQ16 = UInt(cfg.factorWidth bits)
  val taskTag = Vec(UInt(cfg.continuous.taskIdWidth bits), cfg.laneCount)
  val keyPosition = UInt(cfg.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
}

private[accel] case class BitNetQkKeyMeta(
    cfg: BitNetQkTp4AssemblerConfig) extends Bundle {
  val kFactorQ16 = UInt(cfg.factorWidth bits)
  val keyPosition = UInt(cfg.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
  val protocolValid = Bool()
}

object BitNetQkTp4AssemblerMath {
  def dot(query: Seq[Int], key: Seq[Int]): BigInt = {
    require(query.length == 128 && key.length == 128)
    require(query.forall(v => v >= -128 && v <= 127))
    require(key.forall(v => v >= -128 && v <= 127))
    query.zip(key).map { case (q, k) => BigInt(q) * k }.sum
  }

  case class RawScore(
      rawDot: Seq[BigInt],
      laneValid: Seq[Boolean],
      keyPosition: Int,
      keyIndex: Int)

  /** Pure oracle for one pass; the common K row is consumed only once/key. */
  def assemble(
      laneMask: Int,
      queryPosition: Seq[Int],
      keyCount: Seq[Int],
      queries: Seq[Seq[Int]],
      keys: Seq[(Int, Seq[Int])]): Seq[RawScore] = {
    require(queryPosition.length == 4 && keyCount.length == 4)
    require(queries.length == 4 && queries.forall(_.length == 128))
    keys.zipWithIndex.map { case ((position, key), index) =>
      val valid = (0 until 4).map { lane =>
        ((laneMask >> lane) & 1) != 0 &&
          index < keyCount(lane) && position <= queryPosition(lane)
      }
      RawScore(
        rawDot = (0 until 4).map { lane =>
          if (valid(lane)) dot(queries(lane), key) else BigInt(0)
        },
        laneValid = valid,
        keyPosition = position,
        keyIndex = index)
    }
  }
}

/**
  * BitNet QK TP4 assembler.
  *
  * A prefill pass maps four physical lanes to four tokens sharing one Q head;
  * a decode pass maps them to one token's four GQA heads.  One K row enters a
  * synchronous active-engine broadcast and one metadata word enters a FIFO.
  * The two independent TP2 returns are buffered and joined with that metadata
  * only when every active engine is present, so score backpressure cannot
  * reorder K factors/positions or release one engine early.
  */
class BitNetQkTp4Assembler(
    cfg: BitNetQkTp4AssemblerConfig = BitNetQkTp4AssemblerConfig())
    extends Component {
  private val lanes = cfg.laneCount
  private val engines = cfg.engineCount
  private val m = cfg.continuous
  private val b = m.base
  private val feederCfg = cfg.feeder

  val io = new Bundle {
    val pass = slave(Stream(BitNetAttentionPass(cfg.model)))
    /**
      * Borrowed query storage.
      *
      * Unlike an ordinary one-cycle Stream consumer, this assembler keeps
      * `ready` low while it uses the payload.  The source must therefore hold
      * the complete query stable, and the terminal `ready && valid` is the
      * release handshake after the last raw score has been accepted.
      */
    val query = slave(Stream(BitNetQkQueryBlock(cfg)))
    /** Cancels a borrowed query and discards bounded per-pass bookkeeping. */
    val abort = in Bool()
    val key = slave(Stream(BitNetQkKeyBeat(cfg)))
    val engineSequence = Vec(
      master(Stream(BitNetContinuousSequence(m))), engines)
    val engineBeat = Vec(
      master(Stream(BitNetI8MatrixBeat(feederCfg))), engines)
    val engineResult = Vec(
      slave(Stream(BitNetProductionCompactTp2DotResultBeat(m))), engines)
    val score = master(Stream(BitNetQkTp4RawScoreBeat(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedKeys = out UInt(32 bits)
    val emittedScores = out UInt(32 bits)
    val completedPasses = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Idle, WaitQuery, IssueSequences, StreamKeys, DrainResults = newElement()
  }
  val state = Reg(State()) init State.Idle
  val activePass = Reg(BitNetAttentionPass(cfg.model)) init (
    BitNetAttentionPass(cfg.model).getZero)
  val activeEngineMask = Reg(Bits(engines bits)) init 0
  val sequencePending = Reg(Bits(engines bits)) init 0
  // The 4096-bit Q block remains owned by the resident Query adapter.  There
  // is deliberately no second queryData register bank in this component.
  // Stream stability under backpressure is the storage contract.
  val queryBorrowed = RegInit(False)
  val ingressIndex = Reg(UInt(cfg.model.sequenceWidth bits)) init 0
  val fault = RegInit(False)
  val acceptedKeys = Reg(UInt(32 bits)) init 0
  val emittedScores = Reg(UInt(32 bits)) init 0
  val completedPasses = Reg(UInt(32 bits)) init 0
  io.busy := state =/= State.Idle
  io.protocolFaultSticky := fault
  io.acceptedKeys := acceptedKeys
  io.emittedScores := emittedScores
  io.completedPasses := completedPasses

  val inputActiveKeyCount = Vec(UInt(cfg.model.sequenceWidth bits), lanes)
  for (lane <- 0 until lanes) {
    inputActiveKeyCount(lane) := Mux(io.pass.laneMask(lane),
      io.pass.keyCount(lane), U(0, cfg.model.sequenceWidth bits))
  }
  val derivedMaximum = inputActiveKeyCount.drop(1).foldLeft(
    inputActiveKeyCount(0)) { (maximum, candidate) =>
      Mux(maximum > candidate, maximum, candidate)
    }
  val prefillRole = io.pass.mode === BitNetMode.Prefill &&
    io.pass.laneRole === BitNetAttentionLaneRole.TokenParallel
  val decodeRole = io.pass.mode === BitNetMode.Decode &&
    io.pass.laneRole === BitNetAttentionLaneRole.HeadParallel
  val legalPrefillMask = (1 to lanes).map { count =>
    io.pass.laneMask === B((BigInt(1) << count) - 1, lanes bits)
  }.reduce(_ || _)
  val decodePassesPerKv = cfg.model.gqaGroupSize / lanes
  val laneLegal = Vec(Bool(), lanes)
  for (lane <- 0 until lanes) {
    val decodeSubpass = io.pass.passIndex % decodePassesPerKv
    val decodeHead = ((io.pass.kvHead.resize(cfg.queryHeadWidth) << 2) +
      decodeSubpass * lanes + lane).resize(cfg.queryHeadWidth)
    laneLegal(lane) := !io.pass.laneMask(lane) ||
      (io.pass.keyCount(lane) =/= 0 &&
        io.pass.keyCount(lane) <= U(cfg.maxSeq,
          cfg.model.sequenceWidth bits) &&
        ((prefillRole && io.pass.queryHeadId(lane) ===
          io.pass.passIndex.resize(cfg.queryHeadWidth)) ||
         (decodeRole && io.pass.queryHeadId(lane) === decodeHead &&
          io.pass.tokenId(lane) === io.pass.tokenId(0))))
  }
  val roleLegal = (prefillRole && legalPrefillMask &&
    io.pass.kvHead ===
      (io.pass.passIndex >> log2Up(cfg.model.gqaGroupSize)).resized) ||
    (decodeRole && io.pass.laneMask.andR &&
      (io.pass.passIndex / decodePassesPerKv).resize(cfg.kvHeadWidth) ===
        io.pass.kvHead)
  val passLegal = io.pass.layer < cfg.model.layerCount &&
    io.pass.kvHead < cfg.model.kvHeadCount && roleLegal &&
    laneLegal.asBits.andR && io.pass.maxKeyCount === derivedMaximum &&
    io.pass.maxKeyCount =/= 0 && io.pass.maxKeyCount <=
      U(cfg.maxSeq, cfg.model.sequenceWidth bits)
  val inputEngineMask = Bits(engines bits)
  for (engine <- 0 until engines) {
    val firstLane = engine * cfg.lanesPerEngine
    inputEngineMask(engine) := io.pass.laneMask(
      firstLane + cfg.lanesPerEngine - 1 downto firstLane).orR
  }

  io.pass.ready := state === State.Idle
  when(io.pass.fire) {
    when(passLegal) {
      activePass := io.pass.payload
      activeEngineMask := inputEngineMask
      acceptedKeys := 0
      ingressIndex := 0
      state := State.WaitQuery
    } otherwise {
      fault := True
    }
  }

  val queryMatchesPass = io.query.requestId === activePass.requestId &&
    io.query.layer === activePass.layer &&
    io.query.laneRole === activePass.laneRole &&
    io.query.passIndex === activePass.passIndex &&
    io.query.kvHead === activePass.kvHead &&
    io.query.laneMask === activePass.laneMask &&
    io.query.tokenId.asBits === activePass.tokenId.asBits &&
    io.query.queryHeadId.asBits === activePass.queryHeadId.asBits &&
    io.query.queryPosition.asBits === activePass.queryPosition.asBits &&
    io.query.keyCount.asBits === activePass.keyCount.asBits
  val rejectQuery = state === State.WaitQuery && io.query.valid &&
    !queryMatchesPass && !io.abort
  // A matching query is observed (borrowed), not consumed.  The release
  // condition is assigned after the score join below.
  io.query.ready := rejectQuery
  when(state === State.WaitQuery && io.query.valid && !io.abort) {
    when(queryMatchesPass) {
      queryBorrowed := True
      sequencePending := activeEngineMask
      state := State.IssueSequences
    } otherwise {
      fault := True
      state := State.Idle
    }
  }

  // Valid and all identity fields are required throughout the borrow.  A
  // source which revokes a non-aborted Stream payload is a protocol failure;
  // gating every new side effect in that cycle keeps the failure fail-stop.
  val borrowedQueryPresent = io.query.valid && queryMatchesPass
  val borrowViolation = queryBorrowed && !io.abort && !borrowedQueryPresent

  val activeSequenceId = (activePass.requestId.resize(m.sequenceIdWidth) +
    activePass.passIndex.resize(m.sequenceIdWidth)).resize(m.sequenceIdWidth)
  val sequenceFire = Bits(engines bits)
  for (engine <- 0 until engines) {
    val firstLane = engine * cfg.lanesPerEngine
    val sequence = io.engineSequence(engine)
    sequence.valid := state === State.IssueSequences &&
      sequencePending(engine) && borrowedQueryPresent
    sequence.payload.mode := BitNetUnifiedMatrixMode.QkDotI8
    sequence.payload.format := BitNetUnifiedOperandFormat.I8Pair
    sequence.payload.sequenceId := activeSequenceId
    sequence.payload.expectedBeats :=
      (activePass.maxKeyCount * cfg.headMicrotiles)
        .resize(m.beatCountWidth)
    for (local <- 0 until cfg.lanesPerEngine) {
      sequence.payload.activationFactorQ16(local) :=
        io.query.qFactorQ16(firstLane + local)
    }
    sequence.payload.weightScaleQ16 := U(1 << 16, 32 bits)
    sequence.payload.outputFeatureCount :=
      activePass.maxKeyCount.resize(b.outputIndexWidth)
    sequence.payload.routeFinalToDynamicQuant := False
    sequence.payload.residualEnable := False
    sequenceFire(engine) := sequence.fire
  }
  val pendingAfter = sequencePending & ~sequenceFire
  when(state === State.IssueSequences) {
    sequencePending := pendingAfter
    when(pendingAfter === 0) { state := State.StreamKeys }
  }

  val keyMetaFifo = StreamFifo(
    BitNetQkKeyMeta(cfg), cfg.metaFifoDepth)
  val keyMicrotile = Reg(UInt(cfg.headMicrotileWidth bits)) init 0
  val keyPending = Reg(Bits(engines bits)) init 0
  val keyStreamOpen = state === State.StreamKeys && borrowedQueryPresent
  val finalKeyMicrotile = keyMicrotile === cfg.headMicrotiles - 1

  val expectedFirst = ingressIndex === 0
  val expectedLast = ingressIndex === activePass.maxKeyCount - 1
  val keyMatchesPass = io.key.requestId === activePass.requestId &&
    io.key.layer === activePass.layer &&
    io.key.passIndex === activePass.passIndex &&
    io.key.kvHead === activePass.kvHead &&
    io.key.keyIndex === ingressIndex &&
    io.key.firstInSequence === expectedFirst &&
    io.key.lastInSequence === expectedLast && io.key.success
  keyMetaFifo.io.push.valid := io.key.fire
  keyMetaFifo.io.push.kFactorQ16 := io.key.kFactorQ16
  keyMetaFifo.io.push.keyPosition := io.key.keyPosition
  keyMetaFifo.io.push.keyIndex := ingressIndex
  keyMetaFifo.io.push.firstInSequence := expectedFirst
  keyMetaFifo.io.push.lastInSequence := expectedLast
  keyMetaFifo.io.push.protocolValid := keyMatchesPass

  when(io.key.fire) {
    acceptedKeys := acceptedKeys + 1
    when(!keyMatchesPass) { fault := True }
    when(expectedLast) {
      state := State.DrainResults
    } otherwise {
      ingressIndex := ingressIndex + 1
    }
  }

  val keyBeatFire = Bits(engines bits)
  for (engine <- 0 until engines) {
    val firstLane = engine * cfg.lanesPerEngine
    val beat = io.engineBeat(engine)
    beat.valid := io.key.valid && keyStreamOpen && keyPending(engine) &&
      (!finalKeyMicrotile || keyMetaFifo.io.push.ready)
    keyBeatFire(engine) := beat.fire
    beat.payload.sequenceId := activeSequenceId
    beat.payload.beatOrdinal :=
      (ingressIndex.resize(m.beatCountWidth) * cfg.headMicrotiles +
        keyMicrotile.resize(m.beatCountWidth)).resized
    beat.payload.use := BitNetI8WeightUse.Qk
    beat.payload.meta.laneMask := activePass.laneMask(
      firstLane + cfg.lanesPerEngine - 1 downto firstLane)
    for (site <- 0 until b.siteCount) {
      val headFeature = UInt(log2Up(cfg.model.headSize) bits)
      headFeature := (keyMicrotile * b.siteCount + site)
        .resize(headFeature.getWidth)
      beat.payload.rawWeights(site) := io.key.k(headFeature).asBits
      for (local <- 0 until cfg.lanesPerEngine) {
        beat.payload.meta.activations(local)(site) :=
          io.query.q(firstLane + local)(headFeature)
      }
    }
    for (local <- 0 until cfg.lanesPerEngine) {
      val lane = firstLane + local
      val taskTag = ((ingressIndex.resize(m.taskIdWidth) << 2) + lane)
        .resize(m.taskIdWidth)
      beat.payload.meta.accumulatorTag(local) :=
        ((ingressIndex.resize(m.accumulatorTagWidth) << 1) + local)
          .resize(m.accumulatorTagWidth)
      beat.payload.meta.firstK(local) := keyMicrotile === 0
      beat.payload.meta.lastK(local) := finalKeyMicrotile
      beat.payload.meta.taskId(local) := taskTag
      beat.payload.meta.tokenId(local) :=
        activePass.tokenId(lane).resize(b.tokenIdWidth)
      beat.payload.meta.outputBase(local) :=
        ingressIndex.resize(b.outputIndexWidth)
    }
  }

  val keyPendingAfter = keyPending & ~keyBeatFire
  val keyMicrotileDone = io.key.valid && keyStreamOpen &&
    keyPendingAfter === 0
  io.key.ready := keyMicrotileDone && finalKeyMicrotile &&
    keyMetaFifo.io.push.ready
  when(state === State.IssueSequences && pendingAfter === 0) {
    keyMicrotile := 0
    keyPending := activeEngineMask
  }
  when(keyMicrotileDone) {
    keyPending := activeEngineMask
    when(finalKeyMicrotile) {
      keyMicrotile := 0
    } otherwise {
      keyMicrotile := keyMicrotile + 1
    }
  } otherwise {
    when(keyStreamOpen) {
      keyPending := keyPendingAfter
    }
  }

  val resultFifos = Array.fill(engines)(
    StreamFifo(BitNetProductionCompactTp2DotResultBeat(m),
      cfg.resultFifoDepth))
  for (engine <- 0 until engines) {
    val acceptingResults = state === State.StreamKeys ||
      state === State.DrainResults
    val terminalResult = io.engineResult(engine).lastK
    resultFifos(engine).io.push.valid := io.engineResult(engine).valid &&
      acceptingResults && activeEngineMask(engine) && terminalResult
    resultFifos(engine).io.push.payload := io.engineResult(engine).payload
    io.engineResult(engine).ready := Mux(
      acceptingResults && activeEngineMask(engine) && terminalResult,
      resultFifos(engine).io.push.ready, True)
    when(io.engineResult(engine).valid && acceptingResults &&
        !activeEngineMask(engine)) {
      fault := True
    }
  }

  val allActiveResultsValid = (0 until engines).map { engine =>
    !activeEngineMask(engine) || resultFifos(engine).io.pop.valid
  }.reduce(_ && _)
  val outputState = state === State.StreamKeys ||
    state === State.DrainResults
  io.score.valid := outputState && keyMetaFifo.io.pop.valid &&
    allActiveResultsValid && borrowedQueryPresent

  val resultMetadataLegal = Vec(Bool(), engines)
  for (engine <- 0 until engines) {
    val firstLane = engine * cfg.lanesPerEngine
    val pair = resultFifos(engine).io.pop.payload
    val expectedMask = activePass.laneMask(
      firstLane + cfg.lanesPerEngine - 1 downto firstLane)
    val expectedBaseTag =
      (keyMetaFifo.io.pop.keyIndex.resize(m.accumulatorTagWidth) << 1)
        .resize(m.accumulatorTagWidth)
    val rowsCanonical = Vec(Bool(), cfg.lanesPerEngine)
    for (local <- 0 until cfg.lanesPerEngine) {
      rowsCanonical(local) := pair.rawRows(local)(1) === 0
      when(!expectedMask(local)) {
        rowsCanonical(local) := pair.rawRows(local).asBits === 0
      }
    }
    resultMetadataLegal(engine) := !activeEngineMask(engine) ||
      (pair.validMask === expectedMask && pair.validMask =/= B"2'b00" &&
        pair.sequenceId === activeSequenceId &&
        pair.baseAccumulatorTag === expectedBaseTag &&
        (!expectedMask(1) ||
          (pair.baseAccumulatorTag + 1).resize(m.accumulatorTagWidth) ===
            (expectedBaseTag + 1).resize(m.accumulatorTagWidth)) &&
        pair.firstK === Bool(cfg.headMicrotiles == 1) && pair.lastK &&
        pair.formatClass ===
          BitNetProductionCompactDotFormatClass.CanonicalI8Row &&
        rowsCanonical.asBits.andR)
  }
  val allResultMetadataLegal = resultMetadataLegal.asBits.andR

  io.score.requestId := activePass.requestId
  io.score.layer := activePass.layer
  io.score.laneRole := activePass.laneRole
  io.score.passIndex := activePass.passIndex
  io.score.kvHead := activePass.kvHead
  io.score.tokenId := activePass.tokenId
  io.score.queryHeadId := activePass.queryHeadId
  io.score.queryPosition := activePass.queryPosition
  io.score.keyCount := activePass.keyCount
  io.score.qFactorQ16 := io.query.qFactorQ16
  io.score.kFactorQ16 := keyMetaFifo.io.pop.kFactorQ16
  io.score.keyPosition := keyMetaFifo.io.pop.keyPosition
  io.score.keyIndex := keyMetaFifo.io.pop.keyIndex
  io.score.firstInSequence := keyMetaFifo.io.pop.firstInSequence
  io.score.lastInSequence := keyMetaFifo.io.pop.lastInSequence
  for (lane <- 0 until lanes) {
    val engine = lane / cfg.lanesPerEngine
    val local = lane % cfg.lanesPerEngine
    val pair = resultFifos(engine).io.pop.payload
    val causalValid = keyMetaFifo.io.pop.keyIndex <
      activePass.keyCount(lane) && keyMetaFifo.io.pop.keyPosition <=
      activePass.queryPosition(lane)
    val valid = activePass.laneMask(lane) && causalValid &&
      keyMetaFifo.io.pop.protocolValid && allResultMetadataLegal &&
      pair.validMask(local)
    io.score.laneValid(lane) := valid
    io.score.rawDot(lane) := 0
    when(valid) {
      io.score.rawDot(lane) := pair.rawRows(local)(0)
    }
    io.score.taskTag(lane) :=
      ((keyMetaFifo.io.pop.keyIndex.resize(m.taskIdWidth) << 2) + lane)
        .resize(m.taskIdWidth)
  }

  keyMetaFifo.io.pop.ready := io.score.fire
  for (engine <- 0 until engines) {
    resultFifos(engine).io.pop.ready := io.score.fire &&
      activeEngineMask(engine)
  }
  when(io.score.fire) {
    emittedScores := emittedScores + 1
    when(!keyMetaFifo.io.pop.protocolValid || !allResultMetadataLegal) {
      fault := True
    }
    when(keyMetaFifo.io.pop.lastInSequence) {
      completedPasses := completedPasses + 1
      queryBorrowed := False
      state := State.Idle
    }
  }

  // Release is atomic with acceptance of the final raw score.  Score
  // backpressure consequently extends the borrow and keeps Q stable without
  // adding any buffer or bubble to the key/score cadence.
  val releaseQuery = io.score.fire &&
    keyMetaFifo.io.pop.lastInSequence && queryBorrowed
  when(releaseQuery) {
    io.query.ready := True
  }

  // Abort is not itself a protocol fault.  Flush only bounded assembler
  // bookkeeping; late external matrix returns are accepted and dropped by
  // the existing inactive-state result policy.  A non-abort source revoke is
  // handled identically but records the contract violation.
  keyMetaFifo.io.flush := io.abort || borrowViolation
  for (engine <- 0 until engines) {
    resultFifos(engine).io.flush := io.abort || borrowViolation
  }
  when(io.abort || borrowViolation) {
    state := State.Idle
    activeEngineMask := 0
    sequencePending := 0
    queryBorrowed := False
    ingressIndex := 0
    when(borrowViolation) { fault := True }
  }
}
