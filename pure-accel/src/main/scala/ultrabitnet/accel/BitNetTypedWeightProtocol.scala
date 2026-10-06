package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** The physical payload kind.  This is a control tag, not permission to
  * widen every Map0 site to the I8 representation. */
object BitNetWeightPayloadKind extends SpinalEnum(binarySequential) {
  val Map0Pair4, I8 = newElement()
}

object BitNetMap0PairSelect extends SpinalEnum(binarySequential) {
  val LowerRows01, UpperRows23 = newElement()
}

object BitNetI8WeightUse extends SpinalEnum(binarySequential) {
  val Qk, LmHead = newElement()
}

/** Per-engine identity carried beside one packed Map0 group-of-four tile. */
case class BitNetMap0RawGroupEngineTag(
    cfg: BitNetContinuousMatrixFeederConfig) extends Bundle {
  private val base = cfg.matrix.base

  /** Group number in units of four output rows. */
  val groupOrdinal = UInt(base.outputIndexWidth bits)
  /** Absolute output row represented by bits [1:0] of every packed byte. */
  val outputBase = UInt(base.outputIndexWidth bits)
}

/**
  * Compact memory-to-feeder boundary for ternary Map0 weights.
  *
  * Each engine carries exactly 128 raw bytes (1024 bits).  Byte `site` keeps
  * the DDR group-of-four encoding: low nibble is rows 0/1 and high nibble is
  * rows 2/3.  One feeder-local hold produces two 128x4-bit beats carrying the
  * explicit lower/upper ordinals below.  `firstRawGroup`/`lastRawGroup` refer
  * to the K sweep of this four-row group, not the complete layer sequence.
  */
case class BitNetMap0RawGroupTile(
    cfg: BitNetContinuousMatrixFeederConfig,
    maxKFeatures: Int = 6912) extends Bundle {
  private val matrix = cfg.matrix
  private val base = matrix.base
  private val maxKTiles = maxKFeatures / base.siteCount
  private val kTileIndexWidth = log2Up(maxKTiles)
  private val kTileCountWidth = log2Up(maxKTiles + 1)

  require(maxKFeatures >= base.siteCount)
  require(maxKFeatures % base.siteCount == 0)

  val sequenceId = UInt(matrix.sequenceIdWidth bits)
  val scheduleMode = BitNetMode()
  val engineTag = Vec(BitNetMap0RawGroupEngineTag(cfg), cfg.engineCount)
  val kTile = UInt(kTileIndexWidth bits)
  val kTileCount = UInt(kTileCountWidth bits)
  val lowerBeatOrdinal = UInt(matrix.beatCountWidth bits)
  val upperBeatOrdinal = UInt(matrix.beatCountWidth bits)
  val firstRawGroup = Bool()
  val lastRawGroup = Bool()
  val engineRawGroup = Vec(
    Vec(Bits(8 bits), base.siteCount), cfg.engineCount)
  val fault = Bool()
}

/** Tags on a byte-wide QK/LM operand beat.  This stream is deliberately
  * disjoint from Map0 raw groups and never enters a nibble selector. */
case class BitNetI8WeightBeat(
    cfg: BitNetContinuousMatrixFeederConfig) extends Bundle {
  private val matrix = cfg.matrix
  private val base = matrix.base

  val sequenceId = UInt(matrix.sequenceIdWidth bits)
  val beatOrdinal = UInt(matrix.beatCountWidth bits)
  val use = BitNetI8WeightUse()
  val engineOutputBase = Vec(
    UInt(base.outputIndexWidth bits), cfg.engineCount)
  val firstK = Bool()
  val lastK = Bool()
  val engineSiteWeight = Vec(
    Vec(Bits(8 bits), base.siteCount), cfg.engineCount)
  val fault = Bool()
}

/** All non-weight fields of one per-engine continuous-array beat. */
case class BitNetTypedMatrixBeatMeta(
    cfg: BitNetContinuousMatrixFeederConfig) extends Bundle {
  private val matrix = cfg.matrix
  private val base = matrix.base

  val activations = Vec(
    Vec(SInt(base.activationWidth bits), base.siteCount),
    matrix.tokenLanesPerEngine)
  val laneMask = Bits(matrix.laneMaskWidth bits)
  val accumulatorTag = Vec(
    UInt(matrix.accumulatorTagWidth bits), matrix.tokenLanesPerEngine)
  val firstK = Vec(Bool(), matrix.tokenLanesPerEngine)
  val lastK = Vec(Bool(), matrix.tokenLanesPerEngine)
  val taskId = Vec(UInt(matrix.taskIdWidth bits),
    matrix.tokenLanesPerEngine)
  val tokenId = Vec(UInt(base.tokenIdWidth bits),
    matrix.tokenLanesPerEngine)
  val outputBase = Vec(UInt(base.outputIndexWidth bits),
    matrix.tokenLanesPerEngine)
}

/** Feeder output for one engine and one selected Map0 row pair. */
case class BitNetMap0PairMatrixBeat(
    cfg: BitNetContinuousMatrixFeederConfig) extends Bundle {
  private val matrix = cfg.matrix
  private val base = matrix.base

  val sequenceId = UInt(matrix.sequenceIdWidth bits)
  val beatOrdinal = UInt(matrix.beatCountWidth bits)
  val pairSelect = BitNetMap0PairSelect()
  /** Still packed: [1:0] and [3:2] are the two ternary row codes. */
  val rawWeights = Vec(Bits(4 bits), base.siteCount)
  val meta = BitNetTypedMatrixBeatMeta(cfg)
}

/** Feeder output for one engine and one byte-wide QK/LM dot beat. */
case class BitNetI8MatrixBeat(
    cfg: BitNetContinuousMatrixFeederConfig) extends Bundle {
  private val matrix = cfg.matrix
  private val base = matrix.base

  val sequenceId = UInt(matrix.sequenceIdWidth bits)
  val beatOrdinal = UInt(matrix.beatCountWidth bits)
  val use = BitNetI8WeightUse()
  val rawWeights = Vec(Bits(8 bits), base.siteCount)
  val meta = BitNetTypedMatrixBeatMeta(cfg)
}

object BitNetTypedWeightProtocolMath {
  /** Exact software model of the feeder-local nibble selection. */
  def splitMap0RawGroup(raw: Seq[Int]): (Vector[Int], Vector[Int]) = {
    require(raw.length == 128)
    require(raw.forall(value => value >= 0 && value <= 0xff))
    (
      raw.map(_ & 0x0f).toVector,
      raw.map(value => (value >> 4) & 0x0f).toVector
    )
  }
}
