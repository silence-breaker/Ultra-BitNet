package ultrabitnet.accel

import spinal.core._

/** Physical-lane meaning is fixed for a whole resident key sweep. */
object BitNetResidentSvSchedule extends SpinalEnum(binarySequential) {
  // Two lanes are two prefill tokens processed by the same query/KV head.
  val PrefillTokenParallel, DecodeHeadParallel = newElement()
}

/** Shared controller phases used by the isolated proof and unified core. */
object BitNetResidentSvArrayState extends SpinalEnum(binarySequential) {
  val Idle, Run, Drain, AlignLane0, EmitLane0, AlignLane1, EmitLane1 =
    newElement()
}

/**
  * Sequence descriptor for the BitNet DSP-resident SV reduction.
  *
  * In prefill, one physical engine owns two tokens for one fixed query head;
  * the two engines therefore retain BitNet's TP4 token/SPU mapping.  In
  * decode, the four physical lanes are repurposed as the four query heads of
  * one GQA group.  `kvHead` is deliberately sized from the configured number
  * of KV banks (five for the 20Q/5KV model), rather than baking an eight-head
  * assumption into the protocol.
  *
  * The arithmetic contract is fixed: each factor is unsigned UQ3.15 and each
  * V element is signed int8.  Accumulators stay as signed S48 values in the
  * DSP48E2 P/C ring until the complete key sweep has finished.
  */
case class BitNetResidentSvSequence(cfg: BitNetContinuousMatrixConfig)
    extends Bundle {
  private val b = cfg.base

  val schedule = BitNetResidentSvSchedule()
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val expectedKeys = UInt(cfg.beatCountWidth bits)
  val kvHead = UInt(b.contextBankWidth bits)
  /** Selects feature 0..sites-1 or sites..2*sites-1 for an S64 core. */
  val featureHalf = Bool()

  // All lane metadata is explicit because its equality relation changes with
  // schedule: prefill has distinct tokens/equal Q head, decode has equal token
  // and distinct Q heads.
  val tokenTag = Vec(
    UInt(cfg.taskIdWidth bits), cfg.tokenLanesPerEngine)
  val tokenId = Vec(
    UInt(b.tokenIdWidth bits), cfg.tokenLanesPerEngine)
  val queryHeadId = Vec(
    UInt(log2Up(b.contextBanks * 4) bits), cfg.tokenLanesPerEngine)
  // 01 permits a single tail head; normal GQA4 engine traffic uses 11.
  val queryMask = Bits(cfg.tokenLanesPerEngine bits)
  // Logical ownership metadata only: the accumulator payload itself is not
  // stored in a BRAM context slot.
  val contextSlot = Vec(
    UInt(b.contextSlotWidth bits), cfg.tokenLanesPerEngine)
}

/**
  * One native 128-element V row plus both resident-lane probability factors.
  *
  * The beat is accepted once at the slow/native key rate, then replayed as
  * two adjacent fast-clock microphases.  Thus V is fetched and broadcast once
  * per key for both resident lanes.  Decode uses the two lanes as Q heads;
  * prefill uses them as tokens and retains head-major scheduling.
  */
case class BitNetResidentSvBeat(cfg: BitNetContinuousMatrixConfig)
    extends Bundle {
  private val b = cfg.base

  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  // Repeated at the beat boundary so a corrupted/misrouted held payload is
  // detected before it can alter either resident context.
  val tokenTag = Vec(
    UInt(cfg.taskIdWidth bits), cfg.tokenLanesPerEngine)
  val tokenId = Vec(
    UInt(b.tokenIdWidth bits), cfg.tokenLanesPerEngine)
  val keyIndex = UInt(cfg.beatCountWidth bits)
  val firstKey = Bool()
  val lastKey = Bool()
  val rawV = Vec(SInt(b.activationWidth bits), b.siteCount)
  // Unsigned UQ3.15; the two factors correspond to sequence.queryHeadId(0/1).
  val factorRaw = Vec(UInt(cfg.factorWidth bits), cfg.tokenLanesPerEngine)
}

/**
  * Fast-domain zero-copy view of one complete resident lane.  `rawContext`
  * aliases the 128 DSP P outputs while valid; keeping ready low freezes those
  * P/C states.  The related x1/x2 bridge converts this to groups of eight in
  * the slow domain without a second 6144-bit mailbox.
  */
case class BitNetResidentSvRawResult(cfg: BitNetContinuousMatrixConfig)
    extends Bundle {
  private val b = cfg.base

  val schedule = BitNetResidentSvSchedule()
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val tokenTag = UInt(cfg.taskIdWidth bits)
  val tokenId = UInt(b.tokenIdWidth bits)
  val kvHead = UInt(b.contextBankWidth bits)
  val queryHeadId = UInt(log2Up(b.contextBanks * 4) bits)
  val contextSlot = UInt(b.contextSlotWidth bits)
  val residentLane = UInt(log2Up(cfg.tokenLanesPerEngine) bits)
  val featureHalf = Bool()
  val rawContext = Vec(SInt(b.dspPWidth bits), b.siteCount)
}

/**
  * Slow-domain grouped view of one completed resident head context.
  *
  * A complete 128-element result is emitted as 16 groups of eight.  The
  * resident DSP state is not released/reused until the final group fires.
  */
case class BitNetResidentSvResultGroup(
    cfg: BitNetContinuousMatrixConfig,
    groupLanes: Int = 8
) extends Bundle {
  private val b = cfg.base
  require(groupLanes > 0 && isPow2(groupLanes))
  require(b.siteCount % groupLanes == 0)
  private val groupCount = b.siteCount / groupLanes

  val schedule = BitNetResidentSvSchedule()
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val tokenTag = UInt(cfg.taskIdWidth bits)
  val tokenId = UInt(b.tokenIdWidth bits)
  val kvHead = UInt(b.contextBankWidth bits)
  val queryHeadId = UInt(log2Up(b.contextBanks * 4) bits)
  val contextSlot = UInt(b.contextSlotWidth bits)
  val residentLane = UInt(log2Up(cfg.tokenLanesPerEngine) bits)
  val featureHalf = Bool()
  val groupIndex = UInt(log2Up(groupCount) bits)
  val rawContext = Vec(SInt(b.dspPWidth bits), groupLanes)
  val first = Bool()
  val last = Bool()
}
