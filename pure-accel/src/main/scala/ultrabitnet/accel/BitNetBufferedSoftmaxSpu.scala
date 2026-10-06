package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Physical fixed-point contract for the resident BitNet softmax path.
  *
  * Unlike the earlier tile-online prototype, this block retains the complete
  * score vector in PL BRAM.  It first finds the sequence maximum, replays all
  * scores to form the final denominator, and only then emits normalized
  * probabilities.  The downstream matrix engines can therefore keep the SV
  * context in their DSP48 P/C ring and never perform an external context-RAM
  * read/modify/write.
  */
case class BitNetBufferedSoftmaxConfig(
    model: BitNetConfig = BitNetConfig.Production,
    maxSeq: Int = 4096,
    scoreWidth: Int = 16,
    scoreFractionalBits: Int = 12,
    expWidth: Int = 16,
    expFractionalBits: Int = 15,
    sumWidth: Int = 40,
    reciprocalWidth: Int = 17,
    probabilityFifoDepth: Int = 16,
    probabilityReplays: Int = 1
) {
  require(model.engineCount >= 1 && model.engineCount <= 2 &&
    model.tokenParallelism == model.engineCount * 2,
    "the native attention datapath must use one or two TP2 engines")
  require(model.queryHeadCount == 20 && model.kvHeadCount == 5 &&
    model.gqaGroupSize == 4,
    "Ultra-BitNet production is fixed at 20Q/5KV with GQA4")
  require(model.headSize == 128, "the production head dimension is 128")
  require(maxSeq >= 64 && maxSeq <= model.maxSequenceLength && isPow2(maxSeq),
    "production uses 4096; smaller powers of two are permitted in simulation")
  require(scoreWidth == 16 && scoreFractionalBits == 12)
  require(expWidth == 16 && expFractionalBits == 15)
  require(sumWidth >= log2Up(maxSeq) + expFractionalBits + 1)
  require(sumWidth >= 33,
    "the rounded 2^31 numerator must fit the reciprocal divider")
  require(reciprocalWidth >= 17)
  require(probabilityFifoDepth >= 8)
  require(probabilityReplays == 1 || probabilityReplays == 2)

  val physicalLanes: Int = model.tokenParallelism
  val positionWidth: Int = log2Up(maxSeq)
  val keyCountWidth: Int = log2Up(maxSeq + 1)
  val layerCount: Int = model.layerCount
  val layerWidth: Int = model.layerWidth
  val kvHeadCount: Int = model.kvHeadCount
  val kvHeadWidth: Int = log2Up(model.kvHeadCount)
  val queryHeadWidth: Int = log2Up(model.queryHeadCount)
  val passIndexWidth: Int = log2Up(model.queryHeadCount)
  val tokenIdWidth: Int = model.sequenceWidth
  val requestIdWidth: Int = model.requestIdWidth
  val expOne: Int = 1 << expFractionalBits
  val probabilityOne: Int = 1 << expFractionalBits
  val dividerCycles: Int = sumWidth
}

case class BitNetBufferedAttentionCommand(
    cfg: BitNetBufferedSoftmaxConfig
) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val layerIndex = UInt(cfg.layerWidth bits)
  val mode = BitNetMode()
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val laneMask = Bits(cfg.physicalLanes bits)
  val tokenId = Vec(UInt(cfg.tokenIdWidth bits), cfg.physicalLanes)
  val queryHeadId = Vec(UInt(cfg.queryHeadWidth bits), cfg.physicalLanes)
  val queryPosition = Vec(UInt(cfg.positionWidth bits), cfg.physicalLanes)
  val keyCount = Vec(UInt(cfg.keyCountWidth bits), cfg.physicalLanes)
  /** Number of physical key rows collected, one through maxSeq. */
  val maxKeyCount = UInt(cfg.keyCountWidth bits)
  val causal = Bool()
}

case class BitNetBufferedAttentionScoreBeat(
    cfg: BitNetBufferedSoftmaxConfig
) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val layerIndex = UInt(cfg.layerWidth bits)
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val score = Vec(SInt(cfg.scoreWidth bits), cfg.physicalLanes)
  val keyPosition = UInt(cfg.positionWidth bits)
  /** Per-lane upstream QK validity; retained through both BRAM replays. */
  val laneValid = Bits(cfg.physicalLanes bits)
  /** Shared physical/protocol validity for this cached key row. */
  val positionValid = Bool()
  val first = Bool()
  val last = Bool()
}

/**
  * One final four-physical-lane probability vector for a single cached key.
  * The weights are unsigned Q1.15.  There is deliberately no tile-alpha or
  * denominator side stream: normalization is complete at this boundary.
  */
case class BitNetBufferedAttentionProbability(
    cfg: BitNetBufferedSoftmaxConfig
) extends Bundle {
  val weight = Vec(UInt(cfg.expWidth bits), cfg.physicalLanes)
  val keyPosition = UInt(cfg.positionWidth bits)
  val keyIndex = UInt(cfg.positionWidth bits)
  val requestId = UInt(cfg.requestIdWidth bits)
  val mode = BitNetMode()
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.passIndexWidth bits)
  val laneMask = Bits(cfg.physicalLanes bits)
  val allMasked = Bits(cfg.physicalLanes bits)
  val layerIndex = UInt(cfg.layerWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val tokenId = Vec(UInt(cfg.tokenIdWidth bits), cfg.physicalLanes)
  val queryHeadId = Vec(UInt(cfg.queryHeadWidth bits), cfg.physicalLanes)
  val queryPosition = Vec(UInt(cfg.positionWidth bits), cfg.physicalLanes)
  val keyCount = Vec(UInt(cfg.keyCountWidth bits), cfg.physicalLanes)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
}

/**
  * Complete-score buffered softmax, matching BitNet's production data
  * motion: score FIFO/BRAM -> final max/norm -> normalized replay -> resident
  * DSP AXPY.  Four score lanes and four exp ports sustain one key per cycle in
  * both replay passes.  Four small restoring dividers run in parallel once per
  * command; the probability replay then uses four 16x17 DSP multiplies.
  */
class BitNetBufferedSoftmaxSpu(
    cfg: BitNetBufferedSoftmaxConfig = BitNetBufferedSoftmaxConfig()
) extends Component {
  val io = new Bundle {
    val command = slave(Stream(BitNetBufferedAttentionCommand(cfg)))
    val score = slave(Stream(BitNetBufferedAttentionScoreBeat(cfg)))
    val probability = master(Stream(
      BitNetBufferedAttentionProbability(cfg)))

    val busy = out Bool()
    val protocolFault = out Bool()
    val acceptedScores = out UInt(32 bits)
    val emittedProbabilities = out UInt(32 bits)
    val completedCommands = out UInt(32 bits)
    /** Stable after denominator replay; exposed only for verification/DFX. */
    val finalDenominator = out Vec(
      UInt(cfg.sumWidth bits), cfg.physicalLanes)
    val finalMaximum = out Vec(
      SInt(cfg.scoreWidth bits), cfg.physicalLanes)
    val dfxFinalReciprocal = out Vec(
      UInt(cfg.reciprocalWidth bits), cfg.physicalLanes)
    /* DFX trace only; it does not participate in flow control. */
    val dfxDenominatorExpValid = out Bool()
    val dfxDenominatorExpIndex = out UInt(cfg.positionWidth bits)
    val dfxDenominatorExpWeight = out Vec(
      UInt(cfg.expWidth bits), cfg.physicalLanes)
  }

  object State extends SpinalEnum {
    val WaitCommand, Collect, SumReplay, DividerLoad, DividerRun,
        ProbabilityStart, ProbabilityReplay = newElement()
  }
  val state = Reg(State()) init State.WaitCommand

  val activeRequestId = Reg(UInt(cfg.requestIdWidth bits)) init 0
  val activeLayerIndex = Reg(UInt(cfg.layerWidth bits)) init 0
  val activeMode = Reg(BitNetMode()) init BitNetMode.Decode
  val activeLaneRole = Reg(BitNetAttentionLaneRole()) init
    BitNetAttentionLaneRole.HeadParallel
  val activePassIndex = Reg(UInt(cfg.passIndexWidth bits)) init 0
  val activeKvHead = Reg(UInt(cfg.kvHeadWidth bits)) init 0
  val activeLaneMask = Reg(Bits(cfg.physicalLanes bits)) init 0
  val activeTokenId = Vec(
    Reg(UInt(cfg.tokenIdWidth bits)) init 0, cfg.physicalLanes)
  val activeQueryHeadId = Vec(
    Reg(UInt(cfg.queryHeadWidth bits)) init 0, cfg.physicalLanes)
  val activeQueryPosition = Vec(
    Reg(UInt(cfg.positionWidth bits)) init 0, cfg.physicalLanes)
  val activeKeyCount = Vec(
    Reg(UInt(cfg.keyCountWidth bits)) init 0, cfg.physicalLanes)
  val activeMaxKeyCount = Reg(UInt(cfg.keyCountWidth bits)) init 0
  val activeCausal = RegInit(True)

  val fault = RegInit(False)
  val acceptedCounter = Reg(UInt(32 bits)) init 0
  val emittedCounter = Reg(UInt(32 bits)) init 0
  val completedCounter = Reg(UInt(32 bits)) init 0
  io.protocolFault := fault
  io.acceptedScores := acceptedCounter
  io.emittedProbabilities := emittedCounter
  io.completedCommands := completedCounter
  io.busy := state =/= State.WaitCommand

  /* Four 4096x16 score banks.  Each is a single-write/single-read BRAM. */
  val scoreMemory = Array.fill(cfg.physicalLanes)(
    Mem(SInt(cfg.scoreWidth bits), cfg.maxSeq))
  scoreMemory.foreach(_.addAttribute("ram_style", "block"))

  /* Key position and fail-closed validity survive both score replays. */
  private val metaWidth = cfg.positionWidth + 1 + cfg.physicalLanes
  val metaMemory = Mem(Bits(metaWidth bits), cfg.maxSeq)
  metaMemory.addAttribute("ram_style", "block")

  val collectIndex = Reg(UInt(cfg.positionWidth bits)) init 0
  val replayIndex = Reg(UInt(cfg.positionWidth bits)) init 0
  val replayAllIssued = RegInit(False)
  val probabilityReplay = Reg(
    UInt(log2Up(cfg.probabilityReplays + 1) bits)) init 0
  val laneHasValue = Reg(Bits(cfg.physicalLanes bits)) init 0
  val maximum = Vec(
    Reg(SInt(cfg.scoreWidth bits)) init 0, cfg.physicalLanes)
  val denominator = Vec(
    Reg(UInt(cfg.sumWidth bits)) init 0, cfg.physicalLanes)
  val reciprocal = Vec(
    Reg(UInt(cfg.reciprocalWidth bits)) init 0,
    cfg.physicalLanes)

  io.finalDenominator := denominator
  io.finalMaximum := maximum
  io.dfxFinalReciprocal := reciprocal

  io.command.ready := state === State.WaitCommand
  io.score.ready := state === State.Collect

  val commandLaneLegal = Vec(Bool(), cfg.physicalLanes)
  val commandActiveCount = Vec(UInt(cfg.keyCountWidth bits),
    cfg.physicalLanes)
  for (lane <- 0 until cfg.physicalLanes) {
    val countLegal = io.command.keyCount(lane) =/= 0 &&
      io.command.keyCount(lane) <= io.command.maxKeyCount &&
      io.command.keyCount(lane) <= cfg.maxSeq
    val prefillHeadLegal = io.command.queryHeadId(lane) ===
      io.command.passIndex.resize(cfg.queryHeadWidth)
    val decodePassesPerKv = cfg.model.gqaGroupSize / cfg.physicalLanes
    val decodeSubpass = io.command.passIndex % decodePassesPerKv
    val decodeHeadLegal = io.command.queryHeadId(lane) ===
      ((io.command.kvHead.resize(cfg.queryHeadWidth) << 2) +
        decodeSubpass * cfg.physicalLanes + lane).resize(cfg.queryHeadWidth)
    val decodeTokenLegal = io.command.tokenId(lane) ===
      io.command.tokenId(0) &&
      io.command.queryPosition(lane) === io.command.queryPosition(0) &&
      io.command.keyCount(lane) === io.command.keyCount(0)
    commandLaneLegal(lane) := !io.command.laneMask(lane) ||
      (countLegal &&
        ((io.command.mode === BitNetMode.Prefill && prefillHeadLegal) ||
          (io.command.mode === BitNetMode.Decode && decodeHeadLegal &&
            decodeTokenLegal)))
    commandActiveCount(lane) := Mux(io.command.laneMask(lane),
      io.command.keyCount(lane), U(0, cfg.keyCountWidth bits))
  }
  val commandComputedMax = commandActiveCount.drop(1).foldLeft(
    commandActiveCount(0)) { (maximum, candidate) =>
      Mux(maximum > candidate, maximum, candidate)
    }
  val commandRoleLegal =
    (io.command.mode === BitNetMode.Prefill &&
      io.command.laneRole === BitNetAttentionLaneRole.TokenParallel) ||
    (io.command.mode === BitNetMode.Decode &&
      io.command.laneRole === BitNetAttentionLaneRole.HeadParallel &&
      io.command.laneMask.andR)
  val commandLegal = io.command.maxKeyCount =/= 0 &&
    io.command.maxKeyCount <= cfg.maxSeq &&
    io.command.maxKeyCount === commandComputedMax &&
    io.command.laneMask.orR && commandLaneLegal.asBits.andR &&
    commandRoleLegal &&
    io.command.layerIndex < cfg.layerCount &&
    io.command.kvHead < cfg.kvHeadCount &&
    io.command.passIndex < cfg.model.queryHeadCount

  when(io.command.fire) {
    fault := False
    acceptedCounter := 0
    emittedCounter := 0
    when(!commandLegal) {
      fault := True
      state := State.WaitCommand
    } otherwise {
      activeRequestId := io.command.requestId
      activeLayerIndex := io.command.layerIndex
      activeMode := io.command.mode
      activeLaneRole := io.command.laneRole
      activePassIndex := io.command.passIndex
      activeKvHead := io.command.kvHead
      activeLaneMask := io.command.laneMask
      for (lane <- 0 until cfg.physicalLanes) {
        activeTokenId(lane) := io.command.tokenId(lane)
        activeQueryHeadId(lane) := io.command.queryHeadId(lane)
        activeQueryPosition(lane) := io.command.queryPosition(lane)
        activeKeyCount(lane) := io.command.keyCount(lane)
      }
      activeMaxKeyCount := io.command.maxKeyCount
      activeCausal := io.command.causal
      collectIndex := 0
      replayIndex := 0
      replayAllIssued := False
      probabilityReplay := 0
      laneHasValue := 0
      for (lane <- 0 until cfg.physicalLanes) {
        maximum(lane) := 0
        denominator(lane) := 0
        reciprocal(lane) := 0
      }
      state := State.Collect
    }
  }

  val collectExpectedLast =
    collectIndex.resize(cfg.keyCountWidth) === activeMaxKeyCount - 1
  val collectAccepted = Vec(Bool(), cfg.physicalLanes)
  for (lane <- 0 until cfg.physicalLanes) {
    val laneInRange = collectIndex.resize(cfg.keyCountWidth) <
      activeKeyCount(lane)
    val lanePositionAccepted = io.score.positionValid &&
      (!activeCausal || io.score.keyPosition <= activeQueryPosition(lane))
    collectAccepted(lane) := lanePositionAccepted && laneInRange &&
      activeLaneMask(lane) && io.score.laneValid(lane)
    scoreMemory(lane).write(
      address = collectIndex,
      data = io.score.score(lane),
      enable = io.score.fire)
  }
  val metaWriteData = io.score.laneValid ## io.score.positionValid.asBits ##
    io.score.keyPosition.asBits
  metaMemory.write(
    address = collectIndex,
    data = metaWriteData,
    enable = io.score.fire)

  when(io.score.fire) {
    acceptedCounter := acceptedCounter + 1
    when(io.score.requestId =/= activeRequestId ||
        io.score.layerIndex =/= activeLayerIndex ||
        io.score.passIndex =/= activePassIndex ||
        io.score.kvHead =/= activeKvHead ||
        io.score.first =/= (collectIndex === 0) ||
        io.score.last =/= collectExpectedLast) {
      fault := True
    }
    for (lane <- 0 until cfg.physicalLanes) {
      when(collectAccepted(lane)) {
        when(!laneHasValue(lane) ||
            io.score.score(lane) > maximum(lane)) {
          maximum(lane) := io.score.score(lane)
        }
        laneHasValue(lane) := True
      }
    }
    when(collectExpectedLast) {
      collectIndex := 0
      for (lane <- 0 until cfg.physicalLanes) {
        denominator(lane) := 0
      }
      state := State.SumReplay
    } otherwise {
      collectIndex := collectIndex + 1
    }
  }

  /*
   * A single score read pipeline is reused by denominator and probability
   * replay, exactly as in BitNet.  No second score copy is constructed.
   */
  val probabilityFifo = new StreamFifo(
    BitNetBufferedAttentionProbability(cfg),
    cfg.probabilityFifoDepth)

  val sumIssue = state === State.SumReplay && !replayAllIssued
  val probabilityIssue = state === State.ProbabilityReplay &&
    !replayAllIssued &&
    probabilityFifo.io.availability > 4
  val scoreReadIssue = sumIssue || probabilityIssue

  val scoreReadData = Vec(SInt(cfg.scoreWidth bits),
    cfg.physicalLanes)
  for (lane <- 0 until cfg.physicalLanes) {
    scoreReadData(lane) := scoreMemory(lane).readSync(
      replayIndex, scoreReadIssue)
  }
  val scoreStageValid = RegNext(scoreReadIssue) init False
  val scoreStageProbability = RegNextWhen(probabilityIssue,
    scoreReadIssue) init False
  val scoreStageIndex = RegNextWhen(replayIndex,
    scoreReadIssue) init 0

  /* Drain the final synchronous score response for one clock. */
  val scoreStageLast = scoreStageIndex.resize(cfg.keyCountWidth) ===
    activeMaxKeyCount - 1
  val scoreFinalPending = RegInit(False)
  val scoreFinalProbability = RegInit(False)
  when(scoreStageValid && scoreStageLast) {
    scoreFinalPending := True
    scoreFinalProbability := scoreStageProbability
  }
  when(scoreFinalPending) {
    scoreFinalPending := False
  }
  val lookupValid = (scoreStageValid && !scoreStageLast) ||
    scoreFinalPending
  val lookupProbability = Mux(scoreFinalPending,
    scoreFinalProbability, scoreStageProbability)
  val lookupIndex = Mux(scoreFinalPending,
    (activeMaxKeyCount - 1).resize(cfg.positionWidth), scoreStageIndex)

  /* Two synchronous exp ports per initialized block ROM. */
  private val expConstants = (0 to 128).map { index =>
    B(Math.round(Math.exp(-index.toDouble / 16.0) * cfg.expOne),
      cfg.expWidth bits)
  }
  val expRoms = Array.fill((cfg.physicalLanes + 1) / 2)(
    Mem(Bits(cfg.expWidth bits), initialContent = expConstants))
  expRoms.foreach(_.addAttribute("ram_style", "block"))

  val expAddress = Vec(UInt(8 bits), cfg.physicalLanes)
  val expTail = Bits(cfg.physicalLanes bits)
  for (lane <- 0 until cfg.physicalLanes) {
    val difference = (
      maximum(lane).resize(cfg.scoreWidth + 1) -
        scoreReadData(lane).resize(cfg.scoreWidth + 1)
    ).asUInt
    val extended = difference.resize(difference.getWidth + 1)
    val roundedStep = ((extended + U(128, extended.getWidth bits)) |>> 8)
      .resize(9)
    expAddress(lane) := roundedStep.resize(8)
    expTail(lane) := roundedStep > 128
    when(roundedStep > 128) { expAddress(lane) := 0 }
  }

  val expReadData = Vec(Bits(cfg.expWidth bits),
    cfg.physicalLanes)
  for (lane <- 0 until cfg.physicalLanes) {
    expReadData(lane) := expRoms(lane / 2).readSync(
      expAddress(lane), lookupValid)
  }

  val romStageValid = RegNext(lookupValid) init False
  val romStageProbability = RegNextWhen(lookupProbability,
    lookupValid) init False
  val romStageTail = RegNextWhen(expTail,
    lookupValid) init 0
  val romStageIndex = RegNextWhen(lookupIndex,
    lookupValid) init 0
  // Metadata and exp ROM are issued together and return on the same edge.
  // Keeping both reads synchronous avoids a 4096x17 distributed-memory copy
  // without adding a replay pipeline stage.
  val romStageMeta = metaMemory.readSync(lookupIndex, lookupValid)
  val romStagePosition = romStageMeta(cfg.positionWidth - 1 downto 0)
    .asUInt
  val romStagePositionValid = romStageMeta(cfg.positionWidth)
  val romStageLaneValid = romStageMeta(
    metaWidth - 1 downto cfg.positionWidth + 1)
  val romStageAccepted = Bits(cfg.physicalLanes bits)
  for (lane <- 0 until cfg.physicalLanes) {
    val laneInRange = romStageIndex.resize(cfg.keyCountWidth) <
      activeKeyCount(lane)
    val lanePositionAccepted = romStagePositionValid &&
      (!activeCausal || romStagePosition <= activeQueryPosition(lane))
    romStageAccepted(lane) := laneInRange && lanePositionAccepted &&
      activeLaneMask(lane) && romStageLaneValid(lane) && laneHasValue(lane)
  }

  val expWeight = Vec(UInt(cfg.expWidth bits),
    cfg.physicalLanes)
  for (lane <- 0 until cfg.physicalLanes) {
    expWeight(lane) := expReadData(lane).asUInt
    when(!romStageAccepted(lane) || romStageTail(lane)) {
      expWeight(lane) := 0
    }
  }

  /* Register the complete ROM response atomically, data and metadata alike. */
  val normalizedStageValid = RegNext(romStageValid) init False
  val normalizedStageProbability = RegNextWhen(romStageProbability,
    romStageValid) init False
  val normalizedStageIndex = RegNextWhen(romStageIndex,
    romStageValid) init 0
  val normalizedStagePosition = RegNextWhen(romStagePosition,
    romStageValid) init 0
  val normalizedStageWeight = Vec(
    Reg(UInt(cfg.expWidth bits)) init 0, cfg.physicalLanes)
  when(romStageValid) {
    for (lane <- 0 until cfg.physicalLanes) {
      normalizedStageWeight(lane) := expWeight(lane)
    }
  }

  io.dfxDenominatorExpValid := normalizedStageValid &&
    !normalizedStageProbability
  io.dfxDenominatorExpIndex := normalizedStageIndex
  io.dfxDenominatorExpWeight := normalizedStageWeight

  val issueExpectedLast =
    replayIndex.resize(cfg.keyCountWidth) === activeMaxKeyCount - 1
  when(scoreReadIssue) {
    when(issueExpectedLast) {
      replayAllIssued := True
    } otherwise {
      replayIndex := replayIndex + 1
    }
  }

  when(normalizedStageValid && !normalizedStageProbability) {
    for (lane <- 0 until cfg.physicalLanes) {
      val added = denominator(lane).resize(cfg.sumWidth + 1) +
        normalizedStageWeight(lane).resize(cfg.sumWidth + 1)
      denominator(lane) := added.resize(cfg.sumWidth)
    }
    when(normalizedStageIndex.resize(cfg.keyCountWidth) ===
        activeMaxKeyCount - 1) {
      state := State.DividerLoad
    }
  }

  /*
   * Four restoring dividers compute round(2^31 / denominator) in parallel.
   * This is a fixed cfg.sumWidth-cycle command overhead, not a per-key divide.
   */
  val dividerDividend = Vec(
    Reg(UInt(cfg.sumWidth bits)) init 0, cfg.physicalLanes)
  val dividerDivisor = Vec(
    Reg(UInt(cfg.sumWidth bits)) init 0, cfg.physicalLanes)
  val dividerRemainder = Vec(
    Reg(UInt((cfg.sumWidth + 1) bits)) init 0,
    cfg.physicalLanes)
  val dividerQuotient = Vec(
    Reg(UInt(cfg.sumWidth bits)) init 0, cfg.physicalLanes)
  val dividerCount = Reg(UInt(log2Up(cfg.sumWidth) bits)) init 0

  when(state === State.DividerLoad) {
    dividerCount := 0
    for (lane <- 0 until cfg.physicalLanes) {
      val roundedNumerator =
        U(BigInt(1) << 31, cfg.sumWidth bits) +
          (denominator(lane) |>> 1)
      dividerDividend(lane) := roundedNumerator
      dividerDivisor(lane) := denominator(lane)
      dividerRemainder(lane) := 0
      dividerQuotient(lane) := 0
    }
    state := State.DividerRun
  }

  when(state === State.DividerRun) {
    for (lane <- 0 until cfg.physicalLanes) {
      val shiftedRemainder =
        (dividerRemainder(lane)(cfg.sumWidth - 1 downto 0) ##
          dividerDividend(lane).msb).asUInt
      val quotientShift =
        (dividerQuotient(lane)(cfg.sumWidth - 2 downto 0) ##
          B"0").asUInt
      val quotientNext = UInt(cfg.sumWidth bits)
      quotientNext := quotientShift
      dividerRemainder(lane) := shiftedRemainder
      when(dividerDivisor(lane) =/= 0 &&
          shiftedRemainder >= dividerDivisor(lane).resize(cfg.sumWidth + 1)) {
        dividerRemainder(lane) := shiftedRemainder -
          dividerDivisor(lane).resize(cfg.sumWidth + 1)
        quotientNext := quotientShift | 1
      }
      dividerQuotient(lane) := quotientNext
      dividerDividend(lane) :=
        (dividerDividend(lane)(cfg.sumWidth - 2 downto 0) ## B"0")
          .asUInt
      when(dividerCount === cfg.sumWidth - 1) {
        when(dividerDivisor(lane) === 0) {
          reciprocal(lane) := 0
        } otherwise {
          reciprocal(lane) := quotientNext.resize(cfg.reciprocalWidth)
        }
      }
    }
    when(dividerCount === cfg.sumWidth - 1) {
      state := State.ProbabilityStart
    } otherwise {
      dividerCount := dividerCount + 1
    }
  }

  when(state === State.ProbabilityStart) {
    replayIndex := 0
    replayAllIssued := False
    state := State.ProbabilityReplay
  }

  /* Registered four-DSP probability scale stage. */
  private val productWidth = cfg.expWidth + cfg.reciprocalWidth
  val probabilityProduct = Vec(UInt(productWidth bits),
    cfg.physicalLanes)
  val probabilityProductReg = Vec(
    Reg(UInt(productWidth bits)) init 0, cfg.physicalLanes)
  for (lane <- 0 until cfg.physicalLanes) {
    probabilityProduct(lane) := normalizedStageWeight(lane) *
      reciprocal(lane)
    probabilityProduct(lane).addAttribute("use_dsp", "yes")
    when(normalizedStageValid && normalizedStageProbability) {
      probabilityProductReg(lane) := probabilityProduct(lane)
    }
  }
  val probabilityStageValid = RegNext(
    normalizedStageValid && normalizedStageProbability) init False
  val probabilityStageIndex = RegNextWhen(normalizedStageIndex,
    normalizedStageValid && normalizedStageProbability) init 0
  val probabilityStagePosition = RegNextWhen(normalizedStagePosition,
    normalizedStageValid && normalizedStageProbability) init 0

  probabilityFifo.io.push.valid := probabilityStageValid
  probabilityFifo.io.push.keyPosition := probabilityStagePosition
  probabilityFifo.io.push.keyIndex := probabilityStageIndex
  probabilityFifo.io.push.requestId := activeRequestId
  probabilityFifo.io.push.mode := activeMode
  probabilityFifo.io.push.laneRole := activeLaneRole
  probabilityFifo.io.push.passIndex := activePassIndex
  probabilityFifo.io.push.laneMask := activeLaneMask
  probabilityFifo.io.push.allMasked := ~laneHasValue
  probabilityFifo.io.push.layerIndex := activeLayerIndex
  probabilityFifo.io.push.kvHead := activeKvHead
  probabilityFifo.io.push.firstInSequence := probabilityStageIndex === 0
  probabilityFifo.io.push.lastInSequence :=
    probabilityStageIndex.resize(cfg.keyCountWidth) === activeMaxKeyCount - 1
  for (lane <- 0 until cfg.physicalLanes) {
    probabilityFifo.io.push.tokenId(lane) := activeTokenId(lane)
    probabilityFifo.io.push.queryHeadId(lane) := activeQueryHeadId(lane)
    probabilityFifo.io.push.queryPosition(lane) := activeQueryPosition(lane)
    probabilityFifo.io.push.keyCount(lane) := activeKeyCount(lane)
    val rounded = (
      probabilityProductReg(lane).resize(productWidth + 1) +
        U(BigInt(1) << 15, (productWidth + 1) bits)
    ) |>> 16
    probabilityFifo.io.push.weight(lane) := rounded.resize(cfg.expWidth)
    when(rounded > cfg.probabilityOne) {
      probabilityFifo.io.push.weight(lane) := cfg.probabilityOne
    }
  }

  io.probability << probabilityFifo.io.pop
  when(io.probability.fire) {
    emittedCounter := emittedCounter + 1
    when(io.probability.lastInSequence) {
      when(probabilityReplay === cfg.probabilityReplays - 1) {
        completedCounter := completedCounter + 1
        state := State.WaitCommand
      } otherwise {
        probabilityReplay := probabilityReplay + 1
        replayIndex := 0
        replayAllIssued := False
        state := State.ProbabilityReplay
      }
    }
  }
}

/** Bit-accurate software oracle for the complete-score buffered contract. */
object BitNetBufferedSoftmaxMath {
  case class Lane(
      active: Boolean,
      tokenId: Int,
      queryHeadId: Int,
      queryPosition: Int,
      keyCount: Int
  )

  case class Position(
      scoresQ12: Seq[Int],
      keyPosition: Int,
      valid: Boolean = true,
      laneValid: Seq[Boolean] = Seq.fill(4)(true)
  )

  case class Probability(
      weightQ15: Seq[Int],
      keyPosition: Int,
      keyIndex: Int,
      first: Boolean,
      last: Boolean
  )

  case class Result(
      probabilities: Seq[Probability],
      denominatorQ15: Seq[BigInt],
      reciprocalQ16: Seq[BigInt],
      finalMaxQ12: Seq[Int],
      allMasked: Seq[Boolean]
  )

  private val ExpOne = 1 << 15
  private val ExpLut = (0 to 128).map { index =>
    Math.round(Math.exp(-index.toDouble / 16.0) * ExpOne).toInt
  }

  def expNegative(magnitudeQ12: Int): Int = {
    require(magnitudeQ12 >= 0)
    val index = (magnitudeQ12 + 128) >> 8
    if (index <= 128) ExpLut(index) else 0
  }

  def compute(
      lanes: Seq[Lane],
      positions: Seq[Position],
      causal: Boolean = true
  ): Result = {
    require(lanes.length == 4)
    require(positions.nonEmpty && positions.length <= 4096)
    require(positions.forall(position => position.scoresQ12.length == 4 &&
      position.laneValid.length == 4))
    require(lanes.forall(lane => !lane.active ||
      (lane.keyCount > 0 && lane.keyCount <= positions.length)))

    val accepted = positions.indices.map { index =>
      lanes.zipWithIndex.map { case (lane, laneIndex) =>
        lane.active && index < lane.keyCount && positions(index).valid &&
          positions(index).laneValid(laneIndex) &&
          (!causal || positions(index).keyPosition <= lane.queryPosition)
      }
    }
    val has = (0 until 4).map(lane =>
      accepted.exists(_(lane)))
    val maximum = (0 until 4).map { lane =>
      positions.indices.filter(accepted(_)(lane))
        .map(positions(_).scoresQ12(lane)).reduceOption(_ max _).getOrElse(0)
    }
    val exp = positions.indices.map { index =>
      (0 until 4).map { lane =>
        if (accepted(index)(lane))
          expNegative(maximum(lane) - positions(index).scoresQ12(lane))
        else 0
      }
    }
    val denominator = (0 until 4).map(lane =>
      exp.map(_(lane)).map(BigInt(_)).sum)
    val reciprocal = denominator.map { value =>
      if (value == 0) BigInt(0)
      else ((BigInt(1) << 31) + value / 2) / value
    }
    val probabilities = positions.indices.map { index =>
      val weights = (0 until 4).map { lane =>
        val scaled = (BigInt(exp(index)(lane)) * reciprocal(lane) +
          (BigInt(1) << 15)) >> 16
        scaled.min(BigInt(ExpOne)).toInt
      }
      Probability(weights, positions(index).keyPosition, index,
        first = index == 0, last = index == positions.length - 1)
    }
    Result(probabilities, denominator, reciprocal, maximum,
      has.map(!_))
  }
}
