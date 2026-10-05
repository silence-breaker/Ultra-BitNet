package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Geometry and protocol sizing for the BitNet streaming array.
  *
  * Unlike [[BitNetUnifiedMatrixArrayCore]], weights are not installed in a
  * stationary job bank.  Every accepted beat carries the next 128 weights.
  * A sequence locks only the arithmetic interpretation; it may contain an
  * entire projection, QK sweep, SV sweep or LM-head scan without an
  * intermediate drain.
  */
case class BitNetContinuousMatrixConfig(
    base: BitNetUnifiedMatrixConfig = BitNetUnifiedMatrixConfig(),
    tokenLanesPerEngine: Int = 2,
    accumulatorTags: Int = 64,
    sequenceIdWidth: Int = 12,
    taskIdWidth: Int = 20,
    beatCountWidth: Int = 24,
    resultFifoDepth: Int = 128,
    factorWidth: Int = 18,
    factorFractionWidth: Int = 5
) {
  require(tokenLanesPerEngine == 2)
  require(accumulatorTags >= 32 && isPow2(accumulatorTags))
  require(resultFifoDepth >= 64 && isPow2(resultFifoDepth))
  require(factorWidth == base.dspBWidth)
  require(factorFractionWidth >= 5)
  require(base.siteCount == 64 || base.siteCount == 128)
  require(base.physicalDsps == base.siteCount + base.numChains - 1)

  val laneMaskWidth: Int = tokenLanesPerEngine
  val accumulatorTagWidth: Int = log2Up(accumulatorTags)
  val outstandingWidth: Int = log2Up(resultFifoDepth + 1)
}

case class BitNetContinuousSequence(cfg: BitNetContinuousMatrixConfig)
    extends Bundle {
  val mode = BitNetUnifiedMatrixMode()
  val format = BitNetUnifiedOperandFormat()
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  // Counts external weight beats.  A TP2 beat still counts once even though
  // the fast array replays its weights for token lane 1 on the following clock.
  val expectedBeats = UInt(cfg.beatCountWidth bits)
  // Projection-only scale/join contract.  Both factors are unsigned Q16.16;
  // their product converts a final raw S32 ternary dot to resident Q16.16 in
  // NativeScaleResidual.  Activation factors are constant for each token lane
  // during a projection sequence, so they need not consume 37 metadata stages.
  val activationFactorQ16 = Vec(UInt(32 bits), cfg.tokenLanesPerEngine)
  val weightScaleQ16 = UInt(32 bits)
  val outputFeatureCount = UInt(cfg.base.outputIndexWidth bits)
  val routeFinalToDynamicQuant = Bool()
  val residualEnable = Bool()
}

case class BitNetContinuousMatrixBeat(
    cfg: BitNetContinuousMatrixConfig,
    includeGenericSv: Boolean = true
)
    extends Bundle {
  private val b = cfg.base

  // Map0Packed: bits 1:0 are output row 0 and 3:2 output row 1.
  // I8Pair: the complete byte is a signed int8 K/LM weight.
  val rawWeights = Vec(Bits(8 bits), b.siteCount)
  val activations = Vec(
    Vec(SInt(b.activationWidth bits), b.siteCount),
    cfg.tokenLanesPerEngine
  )
  // Legal dot masks are 01 (decode) and 11 (prefill TP2).  SV uses 01.
  val laneMask = Bits(cfg.laneMaskWidth bits)
  val accumulatorTag = Vec(
    UInt(cfg.accumulatorTagWidth bits),
    cfg.tokenLanesPerEngine
  )
  val firstK = Vec(Bool(), cfg.tokenLanesPerEngine)
  val lastK = Vec(Bool(), cfg.tokenLanesPerEngine)
  val taskId = Vec(UInt(cfg.taskIdWidth bits), cfg.tokenLanesPerEngine)
  val tokenId = Vec(UInt(b.tokenIdWidth bits), cfg.tokenLanesPerEngine)
  val outputBase = Vec(UInt(b.outputIndexWidth bits), cfg.tokenLanesPerEngine)

  // SV uses lane 0 only.  ContextScale places the context limb on DSP A and
  // an unsigned Q1.15 factor on B.  AXPY places an effective U18/S18 factor
  // on A and the signed V sample on B, so unsigned factors with bit 17 set are
  // represented exactly rather than becoming negative DSP-B values.
  val svValue = if (includeGenericSv)
    Vec(SInt(b.dspAWidth bits), b.siteCount) else null
  val svFactorRaw = if (includeGenericSv) Bits(cfg.factorWidth bits) else null
  val svFactorSigned = if (includeGenericSv) Bool() else null
  val svFactorFractionBits = if (includeGenericSv)
    UInt(cfg.factorFractionWidth bits) else null
  val svLimbShift = if (includeGenericSv) UInt(5 bits) else null
  val contextBank = if (includeGenericSv)
    UInt(b.contextBankWidth bits) else null
  val contextSlot = if (includeGenericSv)
    UInt(b.contextSlotWidth bits) else null
  val headLane = if (includeGenericSv) UInt(2 bits) else null
}

/** Metadata belonging to one internal token microphase. */
case class BitNetContinuousMicroMeta(
    cfg: BitNetContinuousMatrixConfig,
    includeGenericSv: Boolean = true
)
    extends Bundle {
  private val b = cfg.base
  val accumulatorTag = UInt(cfg.accumulatorTagWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val taskId = UInt(cfg.taskIdWidth bits)
  val tokenId = UInt(b.tokenIdWidth bits)
  val outputBase = UInt(b.outputIndexWidth bits)
  val tokenLane = UInt(log2Up(cfg.tokenLanesPerEngine) bits)
  /** External lane mask shared by both microphases of one accepted beat. */
  val inputLaneMask = Bits(cfg.laneMaskWidth bits)
  val contextBank = if (includeGenericSv)
    UInt(b.contextBankWidth bits) else null
  val contextSlot = if (includeGenericSv)
    UInt(b.contextSlotWidth bits) else null
  val headLane = if (includeGenericSv) UInt(2 bits) else null
  val svLimbShift = if (includeGenericSv) UInt(5 bits) else null
  val svFactorSigned = if (includeGenericSv) Bool() else null
  val svFactorFractionBits = if (includeGenericSv)
    UInt(cfg.factorFractionWidth bits) else null
}

case class BitNetContinuousDotResult(cfg: BitNetContinuousMatrixConfig)
    extends Bundle {
  private val b = cfg.base
  val mode = BitNetUnifiedMatrixMode()
  val format = BitNetUnifiedOperandFormat()
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val taskId = UInt(cfg.taskIdWidth bits)
  val tokenId = UInt(b.tokenIdWidth bits)
  val outputBase = UInt(b.outputIndexWidth bits)
  // Final projection values are converted as
  // rawDot * activationFactorQ16 * weightScaleQ16, then framed for the dynamic
  // i8 quantizer.  The fields are still returned on partial-K results so a
  // coordinator can remain packet-free; consumers act only when lastK is set.
  val activationFactorQ16 = UInt(32 bits)
  val weightScaleQ16 = UInt(32 bits)
  val routeFinalToDynamicQuant = Bool()
  val residualEnable = Bool()
  val vectorFirst = Bool()
  val vectorLast = Bool()
  val outputLaneMask = Bits(b.outputLanes bits)
  val tokenLane = UInt(log2Up(cfg.tokenLanesPerEngine) bits)
  /** Lets the related-clock bridge distinguish TP2 pairs from lane-0 beats. */
  val inputLaneMask = Bits(cfg.laneMaskWidth bits)
  val accumulatorTag = UInt(cfg.accumulatorTagWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val values = Vec(SInt(b.accumulatorWidth bits), b.outputLanes)
  val tileValues = Vec(SInt(b.accumulatorWidth bits), b.outputLanes)
}

case class BitNetContinuousSvRawResult(cfg: BitNetContinuousMatrixConfig)
    extends Bundle {
  private val b = cfg.base
  val mode = BitNetUnifiedMatrixMode()
  val format = BitNetUnifiedOperandFormat()
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val taskId = UInt(cfg.taskIdWidth bits)
  val tokenId = UInt(b.tokenIdWidth bits)
  val contextBank = UInt(b.contextBankWidth bits)
  val contextSlot = UInt(b.contextSlotWidth bits)
  val headLane = UInt(2 bits)
  val svLimbShift = UInt(5 bits)
  val svFactorSigned = Bool()
  val svFactorFractionBits = UInt(cfg.factorFractionWidth bits)
  val rawP = Vec(SInt(b.dspPWidth bits), b.siteCount)
}

/** One slow-domain TP2 return.  Prefill sets validMask=11; decode and a
  * partially populated prefill engine set validMask=01. */
case class BitNetTp2DotResultBeat(cfg: BitNetContinuousMatrixConfig)
    extends Bundle {
  val validMask = Bits(cfg.tokenLanesPerEngine bits)
  val results = Vec(
    BitNetContinuousDotResult(cfg), cfg.tokenLanesPerEngine)
}

/**
  * Narrow view of one held 128-site SV result.
  *
  * The related-clock cluster never captures a second 6144-bit copy.  It keeps
  * the fast core result stalled/stable and exposes `groupLanes` sites per slow
  * transfer.  Only the final group handshake releases the core result.  This
  * grouped transport is also the output shape required by the planned
  * resident-DSP context accumulator; it does not imply per-key BRAM RMW is the
  * final SV architecture.
  */
case class BitNetContinuousSvResultGroup(
    cfg: BitNetContinuousMatrixConfig,
    groupLanes: Int = 8
) extends Bundle {
  private val b = cfg.base
  require(groupLanes > 0 && isPow2(groupLanes))
  require(b.siteCount % groupLanes == 0)
  private val groupCount = b.siteCount / groupLanes

  val mode = BitNetUnifiedMatrixMode()
  val format = BitNetUnifiedOperandFormat()
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val taskId = UInt(cfg.taskIdWidth bits)
  val tokenId = UInt(b.tokenIdWidth bits)
  val contextBank = UInt(b.contextBankWidth bits)
  val contextSlot = UInt(b.contextSlotWidth bits)
  val headLane = UInt(2 bits)
  val svLimbShift = UInt(5 bits)
  val svFactorSigned = Bool()
  val svFactorFractionBits = UInt(cfg.factorFractionWidth bits)
  val groupIndex = UInt(log2Up(groupCount) bits)
  val rawP = Vec(SInt(b.dspPWidth bits), groupLanes)
  val first = Bool()
  val last = Bool()
}

/**
  * Production continuous 128-site / 143-DSP matrix array.
  *
  * The key scheduling unit is a long sequence, not a tile-sized job.  At 532
  * MHz, a prefill beat executes lane 0 and then replays the same 1024 weight
  * bits for lane 1; therefore one engine consumes one 266-MHz beat every two
  * fast clocks (TP2), and two engines sharing that beat implement TP4 with 286
  * DSPs.  Decode accepts lane-0 beats without waiting for the 37-cycle dot
  * latency.  Only a sequence/mode transition drains the pipelines.
  *
  * Dot weights and activations advance down each eight-site chain exactly like
  * BitNet's wkv_pipe: compact raw bytes are delayed, then decoded at the
  * consuming site.  Partial sums remain in DSP48E2 C/P cascade stages.  Sixty
  * four tagged cross-K accumulators and a one-entry write bypass permit
  * interleaved rows without a read-after-write bubble.
  *
  * SV uses the same 128 multipliers and an elastic four-cycle vector path.
  * Context tags remain busy until the corresponding output is consumed, so
  * backpressure cannot overwrite a bank/slot or lose a 6144-bit result.
  */
class BitNetContinuousMatrixArrayCore(
    cfg: BitNetContinuousMatrixConfig = BitNetContinuousMatrixConfig(),
    useVendorPrimitive: Boolean = true,
    productionPruneGenericSv: Boolean = true,
    productionCompactDotResult: Boolean = false
) extends Component {
  private val b = cfg.base
  private val map0CodecCfg = BitNetTernaryMap0DspOperandCodecConfig(
    laneSeparation = b.laneSeparation,
    dspAWidth = 30,
    dspDWidth = b.dspAWidth
  )
  // ADREG/B2 replace the former per-site fabric A/D/B operand stage.  The
  // first-site latency therefore remains four clocks while 75 fabric FFs per
  // site move into already-instantiated DSP48E2 pipeline registers.
  private val hardenedOperandPipelineStages = 1
  private val siteLatency = hardenedOperandPipelineStages + b.sitePipelineStages
  // DSP48E2 PCOUT/PCIN is a one-cycle systolic hop.  Only the first site pays
  // the complete operand/M/P latency; each later site is fed one cycle later
  // and consumes the preceding site's dedicated cascade result.  The former
  // fabric-C implementation paid a complete site latency at every hop.
  private val chainLatency = siteLatency + b.chainLength - 1
  private val dotLatency = chainLatency + b.reductionLevels + 1
  private val svLatency = siteLatency
  private val includeGenericSv = !productionPruneGenericSv

  val io = new Bundle {
    val sequence = slave(Stream(BitNetContinuousSequence(cfg)))
    val beat = slave(Stream(BitNetContinuousMatrixBeat(
      cfg, includeGenericSv)))
    /** Rich research/debug return.  Production clusters prune it at source so
      * the 128-deep result FIFO never stores reconstructible metadata. */
    val dotResult = if (!productionCompactDotResult)
      master(Stream(BitNetContinuousDotResult(cfg))) else null
    /** Production return retained by the core FIFO at one result/fast-cycle. */
    val compactDotResult = if (productionCompactDotResult)
      master(Stream(BitNetProductionCompactDotResult(cfg))) else null
    val svResult = if (includeGenericSv)
      master(Stream(BitNetContinuousSvRawResult(cfg))) else null

    // Production resident-SV path.  These ports arbitrate the same 128 MAC
    // DSPs as sequence/beat above; they are not a sidecar arithmetic array.
    val residentSequence = slave(Stream(BitNetResidentSvSequence(cfg)))
    val residentBeat = slave(Stream(BitNetResidentSvBeat(cfg)))
    val residentRawResult = master(Stream(BitNetResidentSvRawResult(cfg)))

    val sequenceActive = out Bool()
    val residentActive = out Bool()
    val residentDraining = out Bool()
    val residentReplayingLane1 = out Bool()
    val draining = out Bool()
    val replayingLane1 = out Bool()
    val contextBusy = if (includeGenericSv)
      out Bits((b.contextBanks * b.contextSlots) bits) else null
    val protocolFault = out Bool()
    val beatsRemaining = out UInt(cfg.beatCountWidth bits)
    val reservedResults = out UInt(cfg.outstandingWidth bits)
    val dotPipelineLatency = out UInt(8 bits)
    val svPipelineLatency = if (includeGenericSv) out UInt(8 bits) else null
    val acceptedSequences = out UInt(32 bits)
    val acceptedBeats = out UInt(32 bits)
    val issuedMicrophases = out UInt(32 bits)
    val emittedResults = out UInt(32 bits)
  }

  private def isDot(mode: BitNetUnifiedMatrixMode.C): Bool =
    mode === BitNetUnifiedMatrixMode.TernaryProjection ||
      mode === BitNetUnifiedMatrixMode.QkDotI8 ||
      mode === BitNetUnifiedMatrixMode.LmHeadDotI8

  private def isSv(mode: BitNetUnifiedMatrixMode.C): Bool =
    mode === BitNetUnifiedMatrixMode.SvContextScaleQ15 ||
      mode === BitNetUnifiedMatrixMode.SvAxpyQ15

  private def legalModeFormat(
      mode: BitNetUnifiedMatrixMode.C,
      format: BitNetUnifiedOperandFormat.C
  ): Bool = {
    val legal = Bool()
    if (includeGenericSv) {
      // Research enumerates every mode, so no default assignment is needed.
      // Keeping this as a separate Scala branch prevents any generic-SV mode
      // compare from being elaborated into the production legality cone.
      switch(mode) {
        is(BitNetUnifiedMatrixMode.TernaryProjection) {
          legal := format === BitNetUnifiedOperandFormat.Map0Packed
        }
        is(BitNetUnifiedMatrixMode.QkDotI8) {
          legal := format === BitNetUnifiedOperandFormat.I8Pair
        }
        is(BitNetUnifiedMatrixMode.LmHeadDotI8) {
          legal := format === BitNetUnifiedOperandFormat.I8Pair
        }
        is(BitNetUnifiedMatrixMode.SvContextScaleQ15) {
          legal := format === BitNetUnifiedOperandFormat.Q15TileAlpha ||
            format === BitNetUnifiedOperandFormat.Q15FinalReciprocal
        }
        is(BitNetUnifiedMatrixMode.SvAxpyQ15) {
          legal := format === BitNetUnifiedOperandFormat.Q15Probability
        }
      }
    } else {
      legal := False
      switch(mode) {
        is(BitNetUnifiedMatrixMode.TernaryProjection) {
          legal := format === BitNetUnifiedOperandFormat.Map0Packed
        }
        is(BitNetUnifiedMatrixMode.QkDotI8) {
          legal := format === BitNetUnifiedOperandFormat.I8Pair
        }
        is(BitNetUnifiedMatrixMode.LmHeadDotI8) {
          legal := format === BitNetUnifiedOperandFormat.I8Pair
        }
      }
    }
    legal
  }

  val active = RegInit(False)
  val matrixCommandPending = RegInit(False)
  val closing = RegInit(False)
  val mode = Reg(BitNetUnifiedMatrixMode()) init (
    BitNetUnifiedMatrixMode.TernaryProjection
  )
  val format = Reg(BitNetUnifiedOperandFormat()) init (
    BitNetUnifiedOperandFormat.Map0Packed
  )
  val sequenceId = Reg(UInt(cfg.sequenceIdWidth bits)) init (0)
  val activationFactorQ16 = Vec(
    Reg(UInt(32 bits)) init (0),
    cfg.tokenLanesPerEngine
  )
  val weightScaleQ16 = Reg(UInt(32 bits)) init (0)
  val outputFeatureCount = Reg(UInt(b.outputIndexWidth bits)) init (0)
  val routeFinalToDynamicQuant = RegInit(False)
  val residualEnable = RegInit(False)
  val activeDotMode = if (includeGenericSv) RegInit(False) else True
  val activeSvMode = if (includeGenericSv) RegInit(False) else null
  val remaining = Reg(UInt(cfg.beatCountWidth bits)) init (0)
  val outstanding = Reg(UInt(cfg.outstandingWidth bits)) init (0)
  val stickyFault = RegInit(False)
  // -----------------------------------------------------------------------
  // DSP-resident SV controller.  It is mutually exclusive with the ordinary
  // continuous matrix sequence and drives the very same site primitives below.
  // -----------------------------------------------------------------------
  val residentState = Reg(BitNetResidentSvArrayState()) init (
    BitNetResidentSvArrayState.Idle)
  val residentCommandPending = RegInit(False)

  val residentFault = RegInit(False)
  val residentSchedule = Reg(BitNetResidentSvSchedule()) init (
    BitNetResidentSvSchedule.PrefillTokenParallel)
  val residentSequenceId = Reg(UInt(cfg.sequenceIdWidth bits)) init (0)
  val residentKvHead = Reg(UInt(b.contextBankWidth bits)) init (0)
  val residentFeatureHalf = RegInit(False)
  val residentTokenTag = Vec(
    Reg(UInt(cfg.taskIdWidth bits)) init (0), cfg.tokenLanesPerEngine)
  val residentTokenId = Vec(
    Reg(UInt(b.tokenIdWidth bits)) init (0), cfg.tokenLanesPerEngine)
  private val residentQueryHeadWidth = log2Up(b.contextBanks * 4)
  val residentQueryHead = Vec(
    Reg(UInt(residentQueryHeadWidth bits)) init (0),
    cfg.tokenLanesPerEngine)
  val residentContextSlot = Vec(
    Reg(UInt(b.contextSlotWidth bits)) init (0),
    cfg.tokenLanesPerEngine)
  val residentLaneMask = Reg(Bits(cfg.tokenLanesPerEngine bits)) init (0)
  val residentRemaining = Reg(UInt(cfg.beatCountWidth bits)) init (0)
  val residentNextKey = Reg(UInt(cfg.beatCountWidth bits)) init (0)

  val residentMaskLegal = io.residentSequence.queryMask === B"2'b01" ||
    io.residentSequence.queryMask === B"2'b11"
  val residentLane0HeadLegal =
    io.residentSequence.queryHeadId(0) < b.contextBanks * 4 &&
      (io.residentSequence.queryHeadId(0) >> 2)
        .resize(b.contextBankWidth) === io.residentSequence.kvHead
  val residentLane1HeadLegal = !io.residentSequence.queryMask(1) ||
    (io.residentSequence.queryHeadId(1) < b.contextBanks * 4 &&
      (io.residentSequence.queryHeadId(1) >> 2)
        .resize(b.contextBankWidth) === io.residentSequence.kvHead)
  val residentPrefillRelationLegal = !io.residentSequence.queryMask(1) ||
    (io.residentSequence.queryHeadId(0) ===
      io.residentSequence.queryHeadId(1) &&
      io.residentSequence.tokenTag(0) =/=
        io.residentSequence.tokenTag(1))
  val residentDecodeRelationLegal = !io.residentSequence.queryMask(1) ||
    (io.residentSequence.tokenId(0) === io.residentSequence.tokenId(1) &&
      io.residentSequence.tokenTag(0) === io.residentSequence.tokenTag(1) &&
      io.residentSequence.queryHeadId(0) =/=
        io.residentSequence.queryHeadId(1))
  val residentRelationLegal = Bool()
  residentRelationLegal := residentDecodeRelationLegal
  when(io.residentSequence.schedule ===
      BitNetResidentSvSchedule.PrefillTokenParallel) {
    residentRelationLegal := residentPrefillRelationLegal
  }
  val residentSequenceLegal = io.residentSequence.expectedKeys =/= 0 &&
    io.residentSequence.kvHead < b.contextBanks && residentMaskLegal &&
    residentLane0HeadLegal && residentLane1HeadLegal && residentRelationLegal
  // Sequence launch is a staged control-plane operation.  Keep ready/fire a
  // short state-only path, hold the command locally, then start Run after the
  // pipelined checker below.  The two launch cycles are amortized over the
  // whole context and do not reduce resident beat throughput.
  io.residentSequence.ready := !active &&
    residentState === BitNetResidentSvArrayState.Idle &&
    !residentCommandPending && !matrixCommandPending

  // Protocol checking is deliberately pipelined and kept off the resident
  // payload/CE path.  A sticky diagnostic may arrive a few cycles after the
  // offending transfer; compute acceptance and DSP issue never wait for the
  // wide metadata reduction.
  val residentSequenceCheck0 = RegNext(io.residentSequence.fire) init (False)
  val residentSequenceBad0 = Vec(RegInit(False), 12)
  when(io.residentSequence.fire) {
    residentSequenceBad0(0) := io.residentSequence.expectedKeys === 0
    residentSequenceBad0(1) := io.residentSequence.kvHead >= b.contextBanks
    residentSequenceBad0(2) := !residentMaskLegal
    residentSequenceBad0(3) :=
      io.residentSequence.queryHeadId(0) >= b.contextBanks * 4
    residentSequenceBad0(4) :=
      (io.residentSequence.queryHeadId(0) >> 2)
        .resize(b.contextBankWidth) =/= io.residentSequence.kvHead
    residentSequenceBad0(5) := io.residentSequence.queryMask(1) &&
      io.residentSequence.queryHeadId(1) >= b.contextBanks * 4
    residentSequenceBad0(6) := io.residentSequence.queryMask(1) &&
      (io.residentSequence.queryHeadId(1) >> 2)
        .resize(b.contextBankWidth) =/= io.residentSequence.kvHead
    residentSequenceBad0(7) := io.residentSequence.queryMask(1) &&
      io.residentSequence.schedule ===
        BitNetResidentSvSchedule.PrefillTokenParallel &&
      io.residentSequence.queryHeadId(0) =/=
        io.residentSequence.queryHeadId(1)
    residentSequenceBad0(8) := io.residentSequence.queryMask(1) &&
      io.residentSequence.schedule ===
        BitNetResidentSvSchedule.PrefillTokenParallel &&
      io.residentSequence.tokenTag(0) === io.residentSequence.tokenTag(1)
    residentSequenceBad0(9) := io.residentSequence.queryMask(1) &&
      io.residentSequence.schedule ===
        BitNetResidentSvSchedule.DecodeHeadParallel &&
      io.residentSequence.tokenId(0) =/= io.residentSequence.tokenId(1)
    residentSequenceBad0(10) := io.residentSequence.queryMask(1) &&
      io.residentSequence.schedule ===
        BitNetResidentSvSchedule.DecodeHeadParallel &&
      io.residentSequence.tokenTag(0) =/= io.residentSequence.tokenTag(1)
    residentSequenceBad0(11) := io.residentSequence.queryMask(1) &&
      io.residentSequence.schedule ===
        BitNetResidentSvSchedule.DecodeHeadParallel &&
      io.residentSequence.queryHeadId(0) ===
        io.residentSequence.queryHeadId(1)
  }
  val residentSequenceCheck1 = RegNext(residentSequenceCheck0) init (False)
  val residentSequenceBad1 = Vec(RegInit(False), 3)
  when(residentSequenceCheck0) {
    residentSequenceBad1(0) := residentSequenceBad0.take(4).reduce(_ || _)
    residentSequenceBad1(1) := residentSequenceBad0.slice(4, 8).reduce(_ || _)
    residentSequenceBad1(2) := residentSequenceBad0.drop(8).reduce(_ || _)
  }
  when(io.residentSequence.fire) {
    residentCommandPending := True
    residentSchedule := io.residentSequence.schedule
    residentSequenceId := io.residentSequence.sequenceId
    residentKvHead := io.residentSequence.kvHead
    residentFeatureHalf := io.residentSequence.featureHalf
    residentTokenTag := io.residentSequence.tokenTag
    residentTokenId := io.residentSequence.tokenId
    residentQueryHead := io.residentSequence.queryHeadId
    residentContextSlot := io.residentSequence.contextSlot
    residentLaneMask := io.residentSequence.queryMask
    residentRemaining := io.residentSequence.expectedKeys
    residentNextKey := 0
  }

  val residentReplayValid = RegInit(False)
  val residentReplayRawV = Vec(
    Reg(SInt(b.activationWidth bits)) init (0), b.siteCount)
  val residentReplayFactor = Reg(UInt(cfg.factorWidth bits)) init (0)
  val residentReplayLast = RegInit(False)
  val residentTagsMatch = (0 until cfg.tokenLanesPerEngine).map { lane =>
    !residentLaneMask(lane) ||
      (io.residentBeat.tokenTag(lane) === residentTokenTag(lane) &&
        io.residentBeat.tokenId(lane) === residentTokenId(lane))
  }.reduce(_ && _)
  val residentBeatLegal =
    io.residentBeat.sequenceId === residentSequenceId && residentTagsMatch &&
      io.residentBeat.keyIndex === residentNextKey &&
      io.residentBeat.firstKey === (residentNextKey === 0) &&
      io.residentBeat.lastKey === (residentRemaining === 1)
  io.residentBeat.ready :=
    residentState === BitNetResidentSvArrayState.Run &&
      !residentReplayValid

  val residentDirectFire = io.residentBeat.fire
  val residentReplayFire = residentReplayValid
  val residentMicroFire = residentDirectFire || residentReplayFire

  // Register the independent comparisons first, then reduce only registered
  // booleans.  This removes the 20+-bit framing comparisons from the single
  // residentFault setup path seen by Vivado at the x2 clock.
  val residentBeatCheck0 = RegNext(residentDirectFire) init (False)
  val residentBeatBad0 = Vec(RegInit(False), 8)
  when(residentDirectFire) {
    residentBeatBad0(0) := io.residentBeat.sequenceId =/= residentSequenceId
    residentBeatBad0(1) := io.residentBeat.keyIndex =/= residentNextKey
    residentBeatBad0(2) := io.residentBeat.firstKey =/= (residentNextKey === 0)
    residentBeatBad0(3) := io.residentBeat.lastKey =/= (residentRemaining === 1)
    residentBeatBad0(4) := residentLaneMask(0) &&
      io.residentBeat.tokenTag(0) =/= residentTokenTag(0)
    residentBeatBad0(5) := residentLaneMask(0) &&
      io.residentBeat.tokenId(0) =/= residentTokenId(0)
    residentBeatBad0(6) := residentLaneMask(1) &&
      io.residentBeat.tokenTag(1) =/= residentTokenTag(1)
    residentBeatBad0(7) := residentLaneMask(1) &&
      io.residentBeat.tokenId(1) =/= residentTokenId(1)
  }
  val residentBeatCheck1 = RegNext(residentBeatCheck0) init (False)
  val residentBeatBad1 = Vec(RegInit(False), 2)
  when(residentBeatCheck0) {
    residentBeatBad1(0) := residentBeatBad0.take(4).reduce(_ || _)
    residentBeatBad1(1) := residentBeatBad0.drop(4).reduce(_ || _)
  }

  when(residentSequenceCheck1) {
    residentCommandPending := False
    when(residentSequenceBad1.reduce(_ || _)) {
      residentFault := True
    } otherwise {
      residentState := BitNetResidentSvArrayState.Run
    }
  }
  when(residentBeatCheck1 && residentBeatBad1.reduce(_ || _)) {
    residentFault := True
  }
  val residentSelectedFactor = UInt(cfg.factorWidth bits)
  residentSelectedFactor := io.residentBeat.factorRaw(0)
  when(residentReplayValid) {
    residentSelectedFactor := residentReplayFactor
  }
  val residentSelectedRawV = Vec(SInt(b.activationWidth bits), b.siteCount)
  for (site <- 0 until b.siteCount) {
    residentSelectedRawV(site) := io.residentBeat.rawV(site)
    when(residentReplayValid) {
      residentSelectedRawV(site) := residentReplayRawV(site)
    }
  }
  when(residentDirectFire) {
    for (site <- 0 until b.siteCount) {
      residentReplayRawV(site) := io.residentBeat.rawV(site)
    }
    residentReplayFactor := Mux(
      residentLaneMask(1), io.residentBeat.factorRaw(1), U(0))
    residentReplayLast := io.residentBeat.lastKey
    residentReplayValid := True
    residentRemaining := residentRemaining - 1
    residentNextKey := residentNextKey + 1
  }
  when(residentReplayFire) { residentReplayValid := False }

  val residentMicroIndex = Reg(UInt((cfg.beatCountWidth + 1) bits)) init (0)
  val residentCurrentAccumulate = residentMicroIndex >= 2
  when(io.residentSequence.fire) { residentMicroIndex := 0 }
  when(residentMicroFire) { residentMicroIndex := residentMicroIndex + 1 }

  val residentDrainRemaining = Reg(UInt(2 bits)) init (0)
  val residentAlignIssued = RegInit(False)
  val residentMaintenance =
    residentState === BitNetResidentSvArrayState.Drain ||
      ((residentState === BitNetResidentSvArrayState.AlignLane0 ||
        residentState === BitNetResidentSvArrayState.AlignLane1) &&
        !residentAlignIssued)
  val residentAdvance = residentMicroFire || residentMaintenance
  val residentAdvanceForState = RegNext(residentAdvance) init (False)
  val residentRequestedAccumulate = Mux(
    residentMicroFire, residentCurrentAccumulate, True)

  when(residentReplayFire && residentReplayLast) {
    residentState := BitNetResidentSvArrayState.Drain
    residentDrainRemaining := 2
    residentAlignIssued := False
  }
  when(residentState === BitNetResidentSvArrayState.Drain) {
    when(residentDrainRemaining === 1) {
      residentDrainRemaining := 0
      residentState := BitNetResidentSvArrayState.AlignLane0
      residentAlignIssued := False
    } otherwise {
      residentDrainRemaining := residentDrainRemaining - 1
    }
  }
  when(residentState === BitNetResidentSvArrayState.AlignLane0) {
    when(!residentAlignIssued) {
      residentAlignIssued := True
    } elsewhen(residentAdvanceForState) {
      residentAlignIssued := False
      residentState := BitNetResidentSvArrayState.EmitLane0
    }
  }
  when(residentState === BitNetResidentSvArrayState.AlignLane1) {
    when(!residentAlignIssued) {
      residentAlignIssued := True
    } elsewhen(residentAdvanceForState) {
      residentAlignIssued := False
      residentState := BitNetResidentSvArrayState.EmitLane1
    }
  }

  val dynamicRouteLegal = !io.sequence.routeFinalToDynamicQuant ||
    (io.sequence.mode === BitNetUnifiedMatrixMode.TernaryProjection &&
      io.sequence.outputFeatureCount =/= 0)
  val sequenceLegal = legalModeFormat(io.sequence.mode, io.sequence.format) &&
    io.sequence.expectedBeats =/= 0 && dynamicRouteLegal

  // Ordinary matrix launches use the same staged control-plane policy as the
  // resident command.  This prevents expectedBeats/mode legality from feeding
  // ready/fire and dozens of fast-domain CEs in one 532 MHz cycle.
  io.sequence.ready := !active &&
    residentState === BitNetResidentSvArrayState.Idle &&
    !matrixCommandPending && !residentCommandPending &&
    !io.residentSequence.valid
  val matrixSequenceCheck0 = RegNext(io.sequence.fire) init (False)
  val matrixSequenceBad0 = Vec(RegInit(False), 4)
  when(io.sequence.fire) {
    matrixSequenceBad0(0) :=
      !legalModeFormat(io.sequence.mode, io.sequence.format)
    matrixSequenceBad0(1) := io.sequence.expectedBeats === 0
    matrixSequenceBad0(2) := io.sequence.routeFinalToDynamicQuant &&
      io.sequence.mode =/= BitNetUnifiedMatrixMode.TernaryProjection
    matrixSequenceBad0(3) := io.sequence.routeFinalToDynamicQuant &&
      io.sequence.outputFeatureCount === 0
  }
  val matrixSequenceCheck1 = RegNext(matrixSequenceCheck0) init (False)
  val matrixSequenceBad1 = RegNext(matrixSequenceBad0.reduce(_ || _)) init (False)
  when(io.sequence.fire) {
    matrixCommandPending := True
    closing := False
    mode := io.sequence.mode
    format := io.sequence.format
    sequenceId := io.sequence.sequenceId
    remaining := io.sequence.expectedBeats
    outstanding := 0
    activationFactorQ16 := io.sequence.activationFactorQ16
    weightScaleQ16 := io.sequence.weightScaleQ16
    outputFeatureCount := io.sequence.outputFeatureCount
    routeFinalToDynamicQuant := io.sequence.routeFinalToDynamicQuant
    residualEnable := io.sequence.residualEnable
    if (includeGenericSv) activeDotMode := isDot(io.sequence.mode)
    if (includeGenericSv) activeSvMode := isSv(io.sequence.mode)
  }
  when(matrixSequenceCheck1) {
    matrixCommandPending := False
    when(matrixSequenceBad1) {
      stickyFault := True
    } otherwise {
      active := True
    }
  }

  // Only compact data needed by the second token lane are replayed.  SV is
  // lane-0-only and never consumes this storage.
  val replayValid = RegInit(False)
  val replayWeights = Vec(Reg(Bits(8 bits)) init (0), b.siteCount)
  val replayActivation = Vec(
    Reg(SInt(b.activationWidth bits)) init (0),
    b.siteCount
  )
  val replayMeta = Reg(BitNetContinuousMicroMeta(
    cfg, includeGenericSv)) init (
    BitNetContinuousMicroMeta(cfg, includeGenericSv).getZero
  )
  // Sample the potential lane-1 payload continuously.  Qualifying 2048 data
  // registers with beatFire/mode/laneMask creates a very high-fanout six-LUT
  // CE path.  replayValid alone determines whether the sampled payload is
  // consumed, so unconditional sampling is numerically equivalent.
  for (site <- 0 until b.siteCount) {
    replayWeights(site) := io.beat.rawWeights(site)
    replayActivation(site) := io.beat.activations(1)(site)
  }
  replayMeta.accumulatorTag := io.beat.accumulatorTag(1)
  replayMeta.firstK := io.beat.firstK(1)
  replayMeta.lastK := io.beat.lastK(1)
  replayMeta.taskId := io.beat.taskId(1)
  replayMeta.tokenId := io.beat.tokenId(1)
  replayMeta.outputBase := io.beat.outputBase(1)
  replayMeta.tokenLane := 1
  replayMeta.inputLaneMask := io.beat.laneMask
  if (includeGenericSv) {
    replayMeta.contextBank := 0
    replayMeta.contextSlot := 0
    replayMeta.headLane := 0
    replayMeta.svLimbShift := 0
    replayMeta.svFactorSigned := False
    replayMeta.svFactorFractionBits := 0
  }

  val svValidPipe = if (includeGenericSv)
    Vec(RegInit(False), svLatency) else null
  val svMetaPipe = if (includeGenericSv) Vec(
    Reg(BitNetContinuousMicroMeta(cfg, includeGenericSv)) init (
      BitNetContinuousMicroMeta(cfg, includeGenericSv).getZero
    ), svLatency) else null
  val svOutValid = if (includeGenericSv)
    svValidPipe(svLatency - 1) else False
  val svAdvance = if (includeGenericSv)
    !svOutValid || io.svResult.ready else True
  // The research-only elastic SV pipe is itself the complete ownership queue.
  // Production never elaborates the pipe or its wide context hazard compare.
  val requestedContextFree = if (includeGenericSv) {
    val contextIndexLegal = io.beat.contextBank < b.contextBanks
    val requestedContextConflicts = (0 until svLatency).map { stage =>
      val stageLeaves =
        if (stage == svLatency - 1) io.svResult.ready else False
      svValidPipe(stage) && !stageLeaves &&
        svMetaPipe(stage).contextBank === io.beat.contextBank &&
        svMetaPipe(stage).contextSlot === io.beat.contextSlot
    }.reduce(_ || _)
    contextIndexLegal && !requestedContextConflicts
  } else True

  val laneMaskLegal = io.beat.laneMask === B"2'b01" ||
    (activeDotMode && io.beat.laneMask === B"2'b11")
  // Context scale is intentionally unsigned Q1.15 on DSP B.  AXPY accepts the
  // complete U18/S18 space because its factor is routed to DSP A.
  val factorLegal = if (includeGenericSv)
    mode =/= BitNetUnifiedMatrixMode.SvContextScaleQ15 ||
      (!io.beat.svFactorSigned && !io.beat.svFactorRaw.msb)
    else True
  val beatPayloadLegal = if (includeGenericSv)
    laneMaskLegal && factorLegal &&
      (!activeSvMode ||
        (io.beat.laneMask === B"2'b01" && requestedContextFree))
    else laneMaskLegal
  val requestedReservations = UInt(2 bits)
  requestedReservations := 1
  when(io.beat.laneMask(1)) { requestedReservations := 2 }
  // One prior beat may be in the registered reservation stage.  Three slots
  // cover that pending decode result plus a new TP2 beat without overflow.
  val dotHasCredit = outstanding <= cfg.resultFifoDepth - 3

  // Production receives only the cluster-authenticated typed Map0/I8 stream.
  // Register its admission decision so the mailbox valid bit reaches the
  // `remaining` clock-enable through one small gate instead of the complete
  // active/closing/replay/credit cone.  TP2 replay explicitly closes the next
  // fast cycle; a conservative depth-5 threshold covers the registered
  // decision and the existing reservation stage without reducing steady-state
  // cadence.  Research generic-SV keeps its context-dependent combinational
  // ready contract.
  val productionBeatReady = if (!includeGenericSv) RegInit(False) else null
  if (!includeGenericSv) {
    val conservativeProductionCredit =
      outstanding <= cfg.resultFifoDepth - 5
    when(!active || closing || remaining === 0) {
      productionBeatReady := False
    } otherwise {
      productionBeatReady := conservativeProductionCredit
      when(io.beat.fire &&
          (io.beat.laneMask(1) || remaining === 1)) {
        productionBeatReady := False
      }
    }
  }

  io.beat.ready := (if (includeGenericSv) {
    active && !closing && remaining =/= 0 && !replayValid &&
      beatPayloadLegal && Mux(activeSvMode, svAdvance, dotHasCredit)
  } else {
    productionBeatReady && laneMaskLegal
  })
  when(io.beat.valid && active && !closing && !laneMaskLegal) {
    stickyFault := True
  }
  if (includeGenericSv) {
    when(io.beat.valid && active && !closing && !factorLegal) {
      stickyFault := True
    }
  }

  val beatFire = io.beat.fire
  val beatIsDot = activeDotMode
  when(beatFire) {
    remaining := remaining - 1
    when(remaining === 1) { closing := True }
    when(beatIsDot && io.beat.laneMask(1)) {
      replayValid := True
    }
  }

  // A replay consumes its already-reserved result slot and cannot stall.
  val replayFire = replayValid
  when(replayFire) { replayValid := False }
  // Derive the dot launch from the dot-mode acceptance predicate itself.
  // `beatFire && beatIsDot` is logically identical in a dot sequence, but it
  // lets the SV context-hazard comparator feed the dot-valid SRL through the
  // shared ready mux.  That false cone was the final two-endpoint, 4 ps OOC
  // violation at 532 MHz.  Keeping a typed launch predicate also makes the
  // BitNet split between matrix and resident-SV issue explicit.
  val dotBeatFire = if (includeGenericSv) {
    io.beat.valid && activeDotMode && active && !closing &&
      remaining =/= 0 && !replayValid && laneMaskLegal && dotHasCredit
  } else beatFire
  val dotMicroFire = dotBeatFire || replayFire
  val svMicroFire = if (includeGenericSv)
    beatFire && activeSvMode else False

  val microWeights = Vec(Bits(8 bits), b.siteCount)
  val microActivation = Vec(SInt(b.activationWidth bits), b.siteCount)
  val microMeta = BitNetContinuousMicroMeta(cfg, includeGenericSv)
  val replaySelectPerChain = Vec(Bool(), b.numChains)
  for (chain <- 0 until b.numChains) {
    replaySelectPerChain(chain) := replayValid
    replaySelectPerChain(chain).addAttribute("keep", "true")
    replaySelectPerChain(chain).addAttribute("max_fanout", "16")
  }
  for (site <- 0 until b.siteCount) {
    microWeights(site) := io.beat.rawWeights(site)
    microActivation(site) := io.beat.activations(0)(site)
    when(replaySelectPerChain(site / b.chainLength)) {
      microWeights(site) := replayWeights(site)
      microActivation(site) := replayActivation(site)
    }
  }
  microMeta.accumulatorTag := io.beat.accumulatorTag(0)
  microMeta.firstK := io.beat.firstK(0)
  microMeta.lastK := io.beat.lastK(0)
  microMeta.taskId := io.beat.taskId(0)
  microMeta.tokenId := io.beat.tokenId(0)
  microMeta.outputBase := io.beat.outputBase(0)
  microMeta.tokenLane := 0
  microMeta.inputLaneMask := io.beat.laneMask
  if (includeGenericSv) {
    microMeta.contextBank := io.beat.contextBank
    microMeta.contextSlot := io.beat.contextSlot
    microMeta.headLane := io.beat.headLane
    microMeta.svLimbShift := io.beat.svLimbShift
    microMeta.svFactorSigned := io.beat.svFactorSigned
    microMeta.svFactorFractionBits := io.beat.svFactorFractionBits
  }
  when(replayValid) {
    microMeta := replayMeta
  }

  val dotValidPipe = Vec(RegInit(False), dotLatency)
  val dotMetaPipe = Vec(
    Reg(BitNetContinuousMicroMeta(cfg, includeGenericSv)) init (
      BitNetContinuousMicroMeta(cfg, includeGenericSv).getZero
    ),
    dotLatency
  )
  dotValidPipe(0) := dotMicroFire
  // Metadata shifts every cycle; validity alone qualifies it.  Clock-enabling
  // the wide stage from dotMicroFire creates a mode/ready fanout path to every
  // CE pin and misses 532 MHz even though the data path is already registered.
  dotMetaPipe(0) := microMeta
  for (stage <- 1 until dotLatency) {
    dotValidPipe(stage) := dotValidPipe(stage - 1)
    dotMetaPipe(stage) := dotMetaPipe(stage - 1)
  }

  if (includeGenericSv) {
    when(svAdvance) {
      svValidPipe(0) := svMicroFire
      svMetaPipe(0) := microMeta
      for (stage <- 1 until svLatency) {
        svValidPipe(stage) := svValidPipe(stage - 1)
        svMetaPipe(stage) := svMetaPipe(stage - 1)
      }
    }
  }

  val chainOutputs = Vec(SInt(b.dspPWidth bits), b.numChains)
  val svOutputs = Vec(SInt(b.dspPWidth bits), b.siteCount)
  val residentIsActive =
    residentState =/= BitNetResidentSvArrayState.Idle
  for (chain <- 0 until b.numChains) {
    // Physically separate CE cones cap route fanout at 532 MHz.
    val chainAdvance = if (includeGenericSv) {
      val value = Bool()
      value := !activeSvMode || svAdvance
      value
    } else True
    chainAdvance.addAttribute("keep", "true")
    chainAdvance.addAttribute("max_fanout", "16")

    // Resident-SV arrives through a held related-clock mailbox, but its
    // acceptance pulse is replicated once per chain before reaching the input
    // register CEs.  A resident beat enters hardened AREG/B1 on its fire cycle;
    // the existing registered advance then moves ADREG/B2, MREG and PREG in
    // lockstep.  This is cycle-for-cycle equivalent to the removed fabric
    // operand register followed by AREG/B1, MREG and PREG, including gaps,
    // lane replay and the maintenance/drain events.
    val residentAdvanceStage = RegNext(residentAdvance) init (False)
    residentAdvanceStage.addAttribute("keep", "true")
    residentAdvanceStage.addAttribute("max_fanout", "16")
    val residentAccumulateStage = RegInit(False)
    residentAccumulateStage.addAttribute("keep", "true")
    residentAccumulateStage.addAttribute("max_fanout", "16")
    when(residentAdvance) {
      residentAccumulateStage := residentRequestedAccumulate
    }
    // Delay the accumulate/overwrite choice to the MREG issue cycle so
    // OPMODEREG reaches PREG with the product it describes.  Qualifying this
    // register with the staged advance preserves alignment across arbitrary
    // gaps between resident keys.
    val residentAccumulateDspStage = RegInit(False)
    residentAccumulateDspStage.addAttribute("keep", "true")
    residentAccumulateDspStage.addAttribute("max_fanout", "16")
    when(residentAdvanceStage) {
      residentAccumulateDspStage := residentAccumulateStage
    }
    val inputAdvance = Bool()
    val preaddAdvance = Bool()
    val multiplyAdvance = Bool()
    val outputAdvance = Bool()
    inputAdvance := chainAdvance
    preaddAdvance := chainAdvance
    multiplyAdvance := chainAdvance
    outputAdvance := chainAdvance
    when(residentIsActive) {
      inputAdvance := residentAdvance
      preaddAdvance := residentAdvanceStage
      multiplyAdvance := residentAdvanceStage
      outputAdvance := residentAdvanceStage
    }
    Seq(inputAdvance, preaddAdvance, multiplyAdvance, outputAdvance).foreach {
      advance =>
        advance.addAttribute("keep", "true")
        advance.addAttribute("max_fanout", "16")
    }
    val siteOutputs = new Array[SInt](b.chainLength)
    val siteCascadeOutputs = new Array[Bits](b.chainLength)
    for (siteInChain <- 0 until b.chainLength) {
      val site = chain * b.chainLength + siteInChain
      // BitNet systolic skew: PCOUT from site n and the product for
      // site n+1 meet one fast clock later.  A full-site skew was required only
      // by the removed fabric C capture path.
      val offset = siteInChain
      val delayedRaw = if (offset == 0) microWeights(site) else
        Delay(microWeights(site), offset, init = B(0, 8 bits))
      val delayedActivation = if (offset == 0) microActivation(site) else
        Delay(
          microActivation(site),
          offset,
          init = S(0, b.activationWidth bits)
        )

      // Map0 stays compact through the systolic raw-byte skew.  At the
      // consuming DSP boundary the codec drives the high packed row on A and
      // the low packed row on D; the hardened pre-adder recreates the former
      // fabric packMap0Pair value.  Other modes put their original operand on
      // A and drive D to zero.
      val map0Codec = new BitNetTernaryMap0DspOperandCodec(map0CodecCfg)
      map0Codec.io.raw := delayedRaw
      val dotA = SInt(map0CodecCfg.dspAWidth bits)
      val dotD = SInt(map0CodecCfg.dspDWidth bits)
      dotA := delayedRaw.asSInt.resize(map0CodecCfg.dspAWidth)
      dotD := 0
      when(mode === BitNetUnifiedMatrixMode.TernaryProjection) {
        dotA := map0Codec.io.operands.a.asSInt
        dotD := map0Codec.io.operands.d.asSInt
      }

      val selectedA = SInt(map0CodecCfg.dspAWidth bits)
      val selectedD = SInt(map0CodecCfg.dspDWidth bits)
      val selectedB = SInt(b.dspBWidth bits)
      if (includeGenericSv) {
        val unsignedFactorA = io.beat.svFactorRaw.asUInt
          .resize(map0CodecCfg.dspAWidth).asSInt
        val signedFactorA = io.beat.svFactorRaw.asSInt
          .resize(map0CodecCfg.dspAWidth)
        val svFactorA = SInt(map0CodecCfg.dspAWidth bits)
        svFactorA := unsignedFactorA
        when(io.beat.svFactorSigned) { svFactorA := signedFactorA }

        selectedA := io.beat.svValue(site).resize(map0CodecCfg.dspAWidth)
        selectedD := 0
        selectedB :=
          io.beat.svFactorRaw.asUInt.resize(b.dspBWidth).asSInt
        when(mode === BitNetUnifiedMatrixMode.SvAxpyQ15) {
          selectedA := svFactorA
          selectedB := io.beat.svValue(site).resize(b.dspBWidth)
        }
        when(activeDotMode) {
          selectedA := dotA
          selectedD := dotD
          selectedB := delayedActivation.resize(b.dspBWidth)
        }
      } else {
        selectedA := dotA
        selectedD := dotD
        selectedB := delayedActivation.resize(b.dspBWidth)
      }

      // Resident AXPY places its positive UQ3.15 factor on DSP A and signed
      // int8 V on DSP B.  A direct/replay fire is captured immediately by the
      // hardened AREG/B1 pair; maintenance cycles inject zero through the same
      // path.  No per-site fabric operand register remains.
      val residentInputA = SInt(map0CodecCfg.dspAWidth bits)
      residentInputA := residentSelectedFactor
        .resize(map0CodecCfg.dspAWidth).asSInt
      val residentInputB = SInt(b.dspBWidth bits)
      residentInputB := residentSelectedRawV(site).resize(b.dspBWidth)
      when(residentMaintenance) {
        residentInputA := 0
        residentInputB := 0
      }
      val siteInputA = Mux(residentIsActive, residentInputA, selectedA)
      val siteInputD = Mux(
        residentIsActive,
        S(0, map0CodecCfg.dspDWidth bits),
        selectedD
      )
      val siteInputB = Mux(residentIsActive, residentInputB, selectedB)

      if (useVendorPrimitive) {
        val dsp = new BitNetPackedMacDsp48E2(
          opmodeReg = 1,
          aMultSel = "AD",
          dReg = 1,
          adReg = 1,
          bReg = 2
        )
        dsp.A := siteInputA.asBits
        dsp.B := siteInputB.asBits
        // C is reserved exclusively for resident AXPY self-feedback.  Dot
        // accumulation never enters fabric: adjacent sites use the dedicated
        // PC cascade below.
        dsp.C := dsp.P
        dsp.D := siteInputD.asBits
        dsp.CARRYIN := False
        dsp.ACIN := 0
        dsp.BCIN := 0
        if (siteInChain == 0) dsp.PCIN := 0
        else dsp.PCIN := siteCascadeOutputs(siteInChain - 1)
        dsp.CARRYCASCIN := False
        dsp.MULTSIGNIN := False
        dsp.ALUMODE := B"4'b0000"
        // INMODE[4]=0 selects B2, aligned with the new ADREG stage.
        dsp.INMODE := B"5'b00101"
        dsp.OPMODE := B"9'b000000101" // M for SV
        if (siteInChain != 0) {
          // X/Y=M, Z=PCIN.  This is the same hardened cascade topology used
          // by BitNet's vec_dsp; no 48-bit fabric adder/mux is inferred.
          when(activeDotMode) { dsp.OPMODE := B"9'b000010101" }
        }
        when(residentIsActive) {
          dsp.OPMODE := B"9'b000000101" // first two overwrite stale P/C
          when(residentAccumulateDspStage) {
            dsp.OPMODE := B"9'b110000101" // M + W(CREG)
          }
        }
        dsp.CARRYINSEL := 0
        dsp.CEA1 := inputAdvance
        dsp.CEA2 := False
        dsp.CEB1 := inputAdvance
        dsp.CEB2 := preaddAdvance
        dsp.CEC := outputAdvance
        dsp.CED := inputAdvance
        dsp.CEAD := preaddAdvance
        dsp.CEM := multiplyAdvance
        dsp.CEP := outputAdvance
        dsp.CECARRYIN := False
        dsp.CECTRL := multiplyAdvance
        dsp.CEINMODE := False
        dsp.CEALUMODE := False
        dsp.RSTA := False
        dsp.RSTB := False
        dsp.RSTC := False
        dsp.RSTD := False
        dsp.RSTM := False
        dsp.RSTP := False
        dsp.RSTALLCARRYIN := False
        dsp.RSTCTRL := False
        dsp.RSTINMODE := False
        dsp.RSTALUMODE := False
        siteOutputs(siteInChain) = dsp.P.asSInt
        siteCascadeOutputs(siteInChain) = dsp.PCOUT
      } else {
        val aReg = Reg(SInt(map0CodecCfg.dspAWidth bits)) init (0)
        val dReg = Reg(SInt(map0CodecCfg.dspDWidth bits)) init (0)
        val b1Reg = Reg(SInt(b.dspBWidth bits)) init (0)
        val b2Reg = Reg(SInt(b.dspBWidth bits)) init (0)
        val adReg = Reg(SInt(map0CodecCfg.dspDWidth bits)) init (0)
        val cReg = Reg(SInt(b.dspPWidth bits)) init (0)
        val mReg = Reg(SInt(b.dspPWidth bits)) init (0)
        val residentOpReg = RegInit(False)
        val pReg = Reg(SInt(b.dspPWidth bits)) init (0)
        when(inputAdvance) {
          aReg := siteInputA
          dReg := siteInputD
          b1Reg := siteInputB
        }
        when(preaddAdvance) {
          b2Reg := b1Reg
          val aAtPreadder = aReg(
            map0CodecCfg.dspDWidth - 1 downto 0
          )
          adReg := (aAtPreadder + dReg)
            .resize(map0CodecCfg.dspDWidth)
        }
        when(multiplyAdvance) {
          mReg := (adReg * b2Reg).resize(b.dspPWidth)
          residentOpReg := residentAccumulateDspStage
        }
        when(outputAdvance) {
          cReg := pReg
          when(residentIsActive) {
            pReg := mReg
            when(residentOpReg) {
              pReg := (mReg + cReg).resize(b.dspPWidth)
            }
          } elsewhen(activeDotMode) {
            if (siteInChain == 0) pReg := mReg
            else pReg := (mReg + siteOutputs(siteInChain - 1))
              .resize(b.dspPWidth)
          } otherwise {
            pReg := mReg
          }
        }
        siteOutputs(siteInChain) = pReg
      }
      svOutputs(site) := siteOutputs(siteInChain)
    }
    chainOutputs(chain) := siteOutputs(b.chainLength - 1)
  }

  // The same 15 DSP adders reduce all dot modes.  Packed ternary lanes remain
  // packed until after this tree, then receive one exact borrow correction.
  var reduceLevel: Seq[SInt] = chainOutputs.toSeq
  for (level <- 0 until b.reductionLevels) {
    val levelValid = dotValidPipe(chainLatency + level - 1)
    reduceLevel = reduceLevel.grouped(2).map {
      case Seq(left, right) =>
        val sum = (left + right).resize(b.dspPWidth)
        val sumReg = Reg(SInt(b.dspPWidth bits)) init (0)
        sum.addAttribute("use_dsp", "yes")
        sumReg.addAttribute("use_dsp", "yes")
        when(levelValid) { sumReg := sum }
        sumReg
      case Seq(single) => single
    }.toSeq
  }
  val reduced = reduceLevel.head
  val lowField = reduced(b.laneSeparation - 1 downto 0)
  val lowLane = lowField.resize(b.accumulatorWidth)
  val highRaw = (reduced >> b.laneSeparation).resize(b.accumulatorWidth)
  val borrow = SInt(b.accumulatorWidth bits)
  borrow := 0
  when(lowField.msb) { borrow := 1 }
  val tile = Vec(SInt(b.accumulatorWidth bits), b.outputLanes)
  tile(0) := reduced.resize(b.accumulatorWidth)
  tile(1) := 0
  when(mode === BitNetUnifiedMatrixMode.TernaryProjection) {
    tile(0) := lowLane
    tile(1) := (highRaw + borrow).resize(b.accumulatorWidth)
  }
  val tileReg = Vec(
    Reg(SInt(b.accumulatorWidth bits)) init (0),
    b.outputLanes
  )
  when(dotValidPipe(dotLatency - 2)) { tileReg := tile }

  // One LUTRAM word stores both S32 lanes.  A one-entry forwarding path makes
  // consecutive phases of the same K tag deterministic and bubble-free.
  val accumulatorMem = Mem(Bits((b.accumulatorWidth * b.outputLanes) bits),
    cfg.accumulatorTags)
  val dotValid = dotValidPipe(dotLatency - 1)
  val dotMeta = dotMetaPipe(dotLatency - 1)
  val memoryPreviousBits = accumulatorMem.readAsync(dotMeta.accumulatorTag)
  val lastWriteValid = RegInit(False)
  val lastWriteTag = Reg(UInt(cfg.accumulatorTagWidth bits)) init (0)
  val lastWriteBits = Reg(Bits((b.accumulatorWidth * b.outputLanes) bits)) init (0)
  val selectedPreviousBits = Bits((b.accumulatorWidth * b.outputLanes) bits)
  selectedPreviousBits := memoryPreviousBits
  when(lastWriteValid && lastWriteTag === dotMeta.accumulatorTag) {
    selectedPreviousBits := lastWriteBits
  }

  val accumulated = Vec(SInt(b.accumulatorWidth bits), b.outputLanes)
  for (lane <- 0 until b.outputLanes) {
    val previous = selectedPreviousBits(
      (lane + 1) * b.accumulatorWidth - 1 downto lane * b.accumulatorWidth
    ).asSInt
    accumulated(lane) := (previous + tileReg(lane)).resize(b.accumulatorWidth)
    when(dotMeta.firstK) { accumulated(lane) := tileReg(lane) }
  }
  val accumulatedBits = accumulated.asBits
  when(dotValid) {
    accumulatorMem.write(dotMeta.accumulatorTag, accumulatedBits)
    lastWriteValid := True
    lastWriteTag := dotMeta.accumulatorTag
    lastWriteBits := accumulatedBits
  }

  // Register the completed result before the BRAM FIFO.  Without this stage,
  // tagged bypass + S32 add + vector-boundary carry chain all terminate on a
  // FIFO BRAM data pin and miss 532 MHz.  The stage adds one cycle of latency
  // but retains one result per fast clock initiation rate.
  val dotEmitValid = RegNext(dotValid) init (False)
  if (productionCompactDotResult) {
    // Collapse the scalar before the 128-deep ownership FIFO.  Sequence,
    // descriptor, token, scale and vector-boundary metadata are already held
    // by the slow endpoint and are reconstructed there after authentication.
    val compactResultReg =
      Reg(BitNetProductionCompactDotResult(cfg)) init (
        BitNetProductionCompactDotResult(cfg).getZero)
    val compactSourceMap0 =
      mode === BitNetUnifiedMatrixMode.TernaryProjection &&
        format === BitNetUnifiedOperandFormat.Map0Packed
    val compactSourceI8 =
      (mode === BitNetUnifiedMatrixMode.QkDotI8 ||
        mode === BitNetUnifiedMatrixMode.LmHeadDotI8) &&
        format === BitNetUnifiedOperandFormat.I8Pair
    val compactSingle = dotMeta.inputLaneMask === B"2'b01" &&
      dotMeta.tokenLane === 0
    val compactPairLane0 = dotMeta.inputLaneMask === B"2'b11" &&
      dotMeta.tokenLane === 0
    val compactPairLane1 = dotMeta.inputLaneMask === B"2'b11" &&
      dotMeta.tokenLane === 1

    when(dotValid) {
      compactResultReg.sequenceId := sequenceId
      compactResultReg.accumulatorTag := dotMeta.accumulatorTag
      compactResultReg.firstK := dotMeta.firstK
      compactResultReg.lastK := dotMeta.lastK
      compactResultReg.formatClass :=
        BitNetProductionCompactDotFormatClass.CanonicalI8Row
      when(compactSourceMap0) {
        compactResultReg.formatClass :=
          BitNetProductionCompactDotFormatClass.Map0KMajor
      }
      compactResultReg.values(0) := accumulated(0)
      compactResultReg.values(1) := 0
      when(compactSourceMap0) {
        compactResultReg.values(1) := accumulated(1)
      }
      compactResultReg.pairPhase :=
        BitNetProductionCompactDotPairPhase.invalid(cfg)
      when((compactSourceMap0 || compactSourceI8) && compactSingle) {
        compactResultReg.pairPhase :=
          BitNetProductionCompactDotPairPhase.SingleOrDecode
      } elsewhen((compactSourceMap0 || compactSourceI8) &&
          compactPairLane0) {
        compactResultReg.pairPhase :=
          BitNetProductionCompactDotPairPhase.pairedLane(0, cfg)
      } elsewhen((compactSourceMap0 || compactSourceI8) &&
          compactPairLane1) {
        compactResultReg.pairPhase :=
          BitNetProductionCompactDotPairPhase.pairedLane(1, cfg)
      }
    }
    val compactPush = Stream(BitNetProductionCompactDotResult(cfg))
    compactPush.valid := dotEmitValid
    compactPush.payload := compactResultReg
    val compactFifo = StreamFifo(
      BitNetProductionCompactDotResult(cfg), cfg.resultFifoDepth)
    compactFifo.io.push << compactPush
    io.compactDotResult << compactFifo.io.pop
    when(compactPush.valid && !compactPush.ready) { stickyFault := True }
  } else {
    val dotResultReg = Reg(BitNetContinuousDotResult(cfg)) init (
      BitNetContinuousDotResult(cfg).getZero
    )
    val outputBaseWide = dotMeta.outputBase.resize(b.outputIndexWidth + 1)
    val outputCountWide = outputFeatureCount.resize(b.outputIndexWidth + 1)
    // Packed ternary projection produces two physical rows per beat.  I8Pair
    // QK/LM uses one signed byte per site and therefore produces exactly one
    // logical row; the second DSP result lane is padding and must never be
    // advertised as a neighbouring vocabulary/key row.
    val logicalRowsPerBeat = UInt(2 bits)
    logicalRowsPerBeat := b.outputLanes
    when(format === BitNetUnifiedOperandFormat.I8Pair) {
      logicalRowsPerBeat := 1
    }
    when(dotValid) {
      dotResultReg.mode := mode
      dotResultReg.format := format
      dotResultReg.sequenceId := sequenceId
      dotResultReg.taskId := dotMeta.taskId
      dotResultReg.tokenId := dotMeta.tokenId
      dotResultReg.outputBase := dotMeta.outputBase
      dotResultReg.activationFactorQ16 := activationFactorQ16(dotMeta.tokenLane)
      dotResultReg.weightScaleQ16 := weightScaleQ16
      dotResultReg.routeFinalToDynamicQuant := routeFinalToDynamicQuant
      dotResultReg.residualEnable := residualEnable
      dotResultReg.vectorFirst := dotMeta.lastK && dotMeta.outputBase === 0
      dotResultReg.vectorLast := dotMeta.lastK &&
        outputBaseWide + logicalRowsPerBeat >= outputCountWide
      for (lane <- 0 until b.outputLanes) {
        dotResultReg.outputLaneMask(lane) :=
          U(lane, 2 bits) < logicalRowsPerBeat &&
            outputBaseWide + lane < outputCountWide
      }
      dotResultReg.tokenLane := dotMeta.tokenLane
      dotResultReg.inputLaneMask := dotMeta.inputLaneMask
      dotResultReg.accumulatorTag := dotMeta.accumulatorTag
      dotResultReg.firstK := dotMeta.firstK
      dotResultReg.lastK := dotMeta.lastK
      dotResultReg.values := accumulated
      dotResultReg.tileValues := tileReg
    }
    val dotPush = Stream(BitNetContinuousDotResult(cfg))
    dotPush.valid := dotEmitValid
    dotPush.payload := dotResultReg
    val dotFifo = StreamFifo(
      BitNetContinuousDotResult(cfg), cfg.resultFifoDepth)
    dotFifo.io.push << dotPush
    io.dotResult << dotFifo.io.pop
    when(dotPush.valid && !dotPush.ready) { stickyFault := True }
  }

  if (includeGenericSv) {
    val svMeta = svMetaPipe(svLatency - 1)
    io.svResult.valid := svOutValid
    io.svResult.mode := mode
    io.svResult.format := format
    io.svResult.sequenceId := sequenceId
    io.svResult.taskId := svMeta.taskId
    io.svResult.tokenId := svMeta.tokenId
    io.svResult.contextBank := svMeta.contextBank
    io.svResult.contextSlot := svMeta.contextSlot
    io.svResult.headLane := svMeta.headLane
    io.svResult.svLimbShift := svMeta.svLimbShift
    io.svResult.svFactorSigned := svMeta.svFactorSigned
    io.svResult.svFactorFractionBits := svMeta.svFactorFractionBits
    io.svResult.rawP := svOutputs
  }

  // The consumer holds ready low while a slow-domain Vec8 mux walks the
  // complete P vector.  No second 6144-bit result register is created here.
  val residentEmitLane1 =
    residentState === BitNetResidentSvArrayState.EmitLane1
  io.residentRawResult.valid :=
    residentState === BitNetResidentSvArrayState.EmitLane0 ||
      residentEmitLane1
  io.residentRawResult.schedule := residentSchedule
  io.residentRawResult.sequenceId := residentSequenceId
  io.residentRawResult.tokenTag := Mux(
    residentEmitLane1, residentTokenTag(1), residentTokenTag(0))
  io.residentRawResult.tokenId := Mux(
    residentEmitLane1, residentTokenId(1), residentTokenId(0))
  io.residentRawResult.kvHead := residentKvHead
  io.residentRawResult.queryHeadId := Mux(
    residentEmitLane1, residentQueryHead(1), residentQueryHead(0))
  io.residentRawResult.contextSlot := Mux(
    residentEmitLane1, residentContextSlot(1), residentContextSlot(0))
  io.residentRawResult.residentLane := residentEmitLane1.asUInt
  io.residentRawResult.featureHalf := residentFeatureHalf
  io.residentRawResult.rawContext := svOutputs
  when(io.residentRawResult.fire) {
    when(residentState === BitNetResidentSvArrayState.EmitLane0 &&
        residentLaneMask(1)) {
      residentState := BitNetResidentSvArrayState.AlignLane1
      residentAlignIssued := False
    } otherwise {
      residentState := BitNetResidentSvArrayState.Idle
    }
  }

  // Retirement retains the architectural contract that a sequence owns the
  // array until its result is actually consumed.  Registering that event
  // breaks the downstream FIFO-ready -> outstanding/closing combinational
  // path at 532 MHz without changing ownership semantics; the only cost is a
  // single drain cycle after the final handshake.
  val ordinaryDotResultFire =
    if (productionCompactDotResult) io.compactDotResult.fire
    else io.dotResult.fire
  val selectedResultFire = RegNext(
    if (includeGenericSv) io.svResult.fire || ordinaryDotResultFire
    else ordinaryDotResultFire) init (False)
  // Reserve results one registered cycle after beat acceptance.  Minimum SV
  // latency is four cycles, so this cannot race a result; it removes mode and
  // ready decode from the outstanding counter's clock-enable cone.
  val reserveValid = RegNext(beatFire) init (False)
  val reserveCount = Reg(UInt(2 bits)) init (0)
  reserveCount := requestedReservations
  when(reserveValid && !selectedResultFire) {
    outstanding := outstanding + reserveCount.resize(cfg.outstandingWidth)
  } elsewhen (!reserveValid && selectedResultFire) {
    outstanding := outstanding - 1
  } elsewhen (reserveValid && selectedResultFire) {
    outstanding := outstanding + reserveCount.resize(cfg.outstandingWidth) - 1
  }

  when(closing && selectedResultFire && outstanding === 1 && !reserveValid &&
      !replayValid) {
    active := False
    closing := False
  }

  val sequenceCount = Reg(UInt(32 bits)) init (0)
  val beatCount = Reg(UInt(32 bits)) init (0)
  val microCount = Reg(UInt(32 bits)) init (0)
  val resultCount = Reg(UInt(32 bits)) init (0)
  when(io.sequence.fire) { sequenceCount := sequenceCount + 1 }
  when(reserveValid) {
    beatCount := beatCount + 1
    microCount := microCount + reserveCount.resize(32)
  }
  when(selectedResultFire) { resultCount := resultCount + 1 }

  io.sequenceActive := active || residentIsActive
  io.residentActive := residentIsActive
  io.residentDraining := residentIsActive &&
    residentState =/= BitNetResidentSvArrayState.Run
  io.residentReplayingLane1 := residentReplayValid
  io.draining := (active && closing) || io.residentDraining
  io.replayingLane1 := replayValid
  if (includeGenericSv) {
    val contextBusyView = Bits((b.contextBanks * b.contextSlots) bits)
    for (bank <- 0 until b.contextBanks; slot <- 0 until b.contextSlots) {
      contextBusyView(bank * b.contextSlots + slot) :=
        (0 until svLatency).map { stage =>
          svValidPipe(stage) &&
            svMetaPipe(stage).contextBank === bank &&
            svMetaPipe(stage).contextSlot === slot
        }.reduce(_ || _)
    }
    io.contextBusy := contextBusyView
  }
  io.protocolFault := stickyFault || residentFault
  io.beatsRemaining := Mux(residentIsActive, residentRemaining, remaining)
  io.reservedResults := outstanding
  io.dotPipelineLatency := dotLatency + 1
  if (includeGenericSv) io.svPipelineLatency := svLatency
  io.acceptedSequences := sequenceCount
  io.acceptedBeats := beatCount
  io.issuedMicrophases := microCount
  io.emittedResults := resultCount
}

object BitNetContinuousMatrixOracle {
  def decodeMap0(code: Int): Int = code & 3 match {
    case 1 => -1
    case 3 => 1
    case _ => 0
  }

  def ternaryDot2(
      activations: Seq[Int],
      rawWeights: Seq[Int]
  ): (Int, Int) = {
    require(activations.length == rawWeights.length)
    val low = activations.indices.map { index =>
      activations(index) * decodeMap0(rawWeights(index))
    }.sum
    val high = activations.indices.map { index =>
      activations(index) * decodeMap0(rawWeights(index) >> 2)
    }.sum
    (low, high)
  }

  def i8(value: Int): Int = {
    val byte = value & 0xff
    if ((byte & 0x80) != 0) byte - 256 else byte
  }

  def i8Dot(activations: Seq[Int], rawWeights: Seq[Int]): Int = {
    require(activations.length == rawWeights.length)
    activations.indices.map(i => activations(i) * i8(rawWeights(i))).sum
  }
}

object GenerateBitNetContinuousMatrixArrayCore extends App {
  SpinalConfig(
    targetDirectory = "target/generated-bitnet-continuous-matrix-array-core",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetContinuousMatrixArrayCore())
}
