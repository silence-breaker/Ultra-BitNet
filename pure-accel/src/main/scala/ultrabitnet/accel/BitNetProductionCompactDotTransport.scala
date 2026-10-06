package ultrabitnet.accel

import spinal.core._

/**
  * Compact format classes carried by the production dot-result transport.
  *
  * The class is deliberately narrower than [[BitNetUnifiedOperandFormat]]:
  * the active sequence has already authenticated the exact mode/format pair.
  * Map0 keeps both physical result rows in their feeder K-major identity;
  * QK/LM use one canonical I8 row and require every padding row to be zero.
  */
object BitNetProductionCompactDotFormatClass {
  val Map0KMajor = 0
  val CanonicalI8Row = 1
  val encodings: Seq[Int] = Seq(Map0KMajor, CanonicalI8Row)
}

/**
  * Two-bit scalar-to-TP2 assembly phase.
  *
  * `SingleOrDecode` is the only phase legal for a `validMask=01` result.
  * Paired lane phases are one plus the local lane number.  The remaining
  * all-ones encoding is poison.  This phase never identifies Map0 lower or
  * upper rows: that identity remains in the accumulator tag and the feeder's
  * K-major counter.
  */
object BitNetProductionCompactDotPairPhase {
  val SingleOrDecode = 0

  def pairedLane(localLane: Int,
      cfg: BitNetContinuousMatrixConfig): Int = {
    require(localLane >= 0 && localLane < cfg.tokenLanesPerEngine)
    localLane + 1
  }

  def invalid(cfg: BitNetContinuousMatrixConfig): Int = {
    val width = BitNetProductionCompactDotTransportOracle.widths(cfg)
      .pairPhaseWidth
    (1 << width) - 1
  }
}

/** Pure-Scala layout and fail-closed assembly oracle for the Stage-C schema. */
object BitNetProductionCompactDotTransportOracle {
  final case class Widths(
      valuesWidth: Int,
      rawRowsWidth: Int,
      sequenceIdWidth: Int,
      accumulatorTagWidth: Int,
      pairPhaseWidth: Int,
      formatClassWidth: Int,
      validMaskWidth: Int,
      scalarWidth: Int,
      tp2PairWidth: Int)

  /** Metadata retained by the compact scalar while it awaits TP2 assembly. */
  final case class ScalarIdentity(
      sequenceId: BigInt,
      accumulatorTag: BigInt,
      firstK: Boolean,
      lastK: Boolean,
      pairPhase: Int,
      formatClass: Int)

  private val productionReference = BitNetContinuousMatrixConfig()
  private val flagWidth = 1

  /**
    * Derive every payload width from the matrix configuration.  The requires
    * intentionally reject a research geometry at this production boundary;
    * silently changing one of these fields would break endpoint metadata
    * reconstruction even if the resulting Bundle still elaborated.
    */
  def widths(cfg: BitNetContinuousMatrixConfig): Widths = {
    require(cfg.base.outputLanes == productionReference.base.outputLanes,
      "compact production dot transport requires canonical output lanes")
    require(cfg.base.accumulatorWidth ==
      productionReference.base.accumulatorWidth,
      "compact production dot transport requires canonical accumulator width")
    require(cfg.tokenLanesPerEngine ==
      productionReference.tokenLanesPerEngine,
      "compact production dot transport requires canonical TP2 geometry")
    require(cfg.sequenceIdWidth == productionReference.sequenceIdWidth,
      "compact production dot transport requires canonical sequence width")
    require(cfg.accumulatorTagWidth ==
      productionReference.accumulatorTagWidth,
      "compact production dot transport requires canonical tag width")

    val valuesWidth = cfg.base.outputLanes * cfg.base.accumulatorWidth
    val rawRowsWidth = cfg.tokenLanesPerEngine * valuesWidth
    val pairPhaseWidth = log2Up(cfg.tokenLanesPerEngine + 1)
    val formatClassWidth = log2Up(
      BitNetProductionCompactDotFormatClass.encodings.size)
    val validMaskWidth = cfg.laneMaskWidth
    val firstLastWidth = 2 * flagWidth
    val scalarWidth = valuesWidth + cfg.sequenceIdWidth +
      cfg.accumulatorTagWidth + firstLastWidth + pairPhaseWidth +
      formatClassWidth
    val tp2PairWidth = rawRowsWidth + validMaskWidth + cfg.sequenceIdWidth +
      cfg.accumulatorTagWidth + firstLastWidth + formatClassWidth

    require(pairPhaseWidth > log2Up(cfg.tokenLanesPerEngine),
      "pair phase must retain a distinct single/decode encoding")
    require(formatClassWidth == flagWidth,
      "production compact format class must remain a single bit")

    Widths(
      valuesWidth = valuesWidth,
      rawRowsWidth = rawRowsWidth,
      sequenceIdWidth = cfg.sequenceIdWidth,
      accumulatorTagWidth = cfg.accumulatorTagWidth,
      pairPhaseWidth = pairPhaseWidth,
      formatClassWidth = formatClassWidth,
      validMaskWidth = validMaskWidth,
      scalarWidth = scalarWidth,
      tp2PairWidth = tp2PairWidth)
  }

  private def fitsUnsigned(value: BigInt, width: Int): Boolean =
    value >= 0 && value < (BigInt(1) << width)

  def laneAccumulatorTag(
      baseAccumulatorTag: BigInt,
      localLane: Int,
      cfg: BitNetContinuousMatrixConfig): BigInt = {
    val layout = widths(cfg)
    require(fitsUnsigned(baseAccumulatorTag, layout.accumulatorTagWidth))
    require(localLane >= 0 && localLane < cfg.tokenLanesPerEngine)
    val tagMask = (BigInt(1) << layout.accumulatorTagWidth) - 1
    (baseAccumulatorTag + localLane) & tagMask
  }

  def singleValidMask(cfg: BitNetContinuousMatrixConfig): Int = {
    widths(cfg)
    1
  }

  def fullValidMask(cfg: BitNetContinuousMatrixConfig): Int = {
    val layout = widths(cfg)
    (1 << layout.validMaskWidth) - 1
  }

  val failStopValidMask: Int = 0

  private def scalarEncodingLegal(
      scalar: ScalarIdentity,
      cfg: BitNetContinuousMatrixConfig): Boolean = {
    val layout = widths(cfg)
    fitsUnsigned(scalar.sequenceId, layout.sequenceIdWidth) &&
      fitsUnsigned(scalar.accumulatorTag, layout.accumulatorTagWidth) &&
      BitNetProductionCompactDotFormatClass.encodings.contains(
        scalar.formatClass) &&
      scalar.pairPhase >= 0 &&
      scalar.pairPhase < (1 << layout.pairPhaseWidth)
  }

  /**
    * Validate an assembled result and collapse every malformed combination to
    * `validMask=00`.  A consumer observing a valid transport beat with that
    * mask must raise its bridge fault and remain fail-stopped.
    *
    * A single/decode beat is exactly phase 00 with no second scalar.  A TP2
    * beat is phase 01 followed by phase 10, with identical sequence/framing/
    * format metadata and lane-1 tag equal to lane-0 tag plus one modulo the
    * configured tag width.
    */
  def canonicalValidMask(
      claimedValidMask: Int,
      lane0: ScalarIdentity,
      lane1: Option[ScalarIdentity],
      cfg: BitNetContinuousMatrixConfig): Int = {
    val singleMask = singleValidMask(cfg)
    val pairMask = fullValidMask(cfg)
    val singleLegal = claimedValidMask == singleMask && lane1.isEmpty &&
      scalarEncodingLegal(lane0, cfg) &&
      lane0.pairPhase == BitNetProductionCompactDotPairPhase.SingleOrDecode

    val pairLegal = lane1.exists { second =>
      claimedValidMask == pairMask &&
        scalarEncodingLegal(lane0, cfg) &&
        scalarEncodingLegal(second, cfg) &&
        lane0.pairPhase ==
          BitNetProductionCompactDotPairPhase.pairedLane(0, cfg) &&
        second.pairPhase ==
          BitNetProductionCompactDotPairPhase.pairedLane(1, cfg) &&
        second.sequenceId == lane0.sequenceId &&
        second.accumulatorTag ==
          laneAccumulatorTag(lane0.accumulatorTag, 1, cfg) &&
        second.firstK == lane0.firstK &&
        second.lastK == lane0.lastK &&
        second.formatClass == lane0.formatClass
    }

    if (singleLegal) singleMask
    else if (pairLegal) pairMask
    else failStopValidMask
  }

  /** Compact transport preserves the feeder's lower/upper K-major identity. */
  def map0KMajorSemanticOrdinal(
      arrivalOrdinal: Int,
      kTilesPerOutput: Int): Int =
    BitNetContinuousMatrixFeederOracle.projectionArrival(
      arrivalOrdinal, kTilesPerOutput).semanticBeatOrdinal

  /**
    * Preserve both Map0 rows, but canonicalise the padding lanes of QK/LM I8
    * results to zero before they enter a TP2 pair.
    */
  def canonicalRows(
      rows: Seq[BigInt],
      formatClass: Int,
      cfg: BitNetContinuousMatrixConfig): Seq[BigInt] = {
    widths(cfg)
    require(rows.length == cfg.base.outputLanes)
    require(BitNetProductionCompactDotFormatClass.encodings.contains(
      formatClass))
    if (formatClass == BitNetProductionCompactDotFormatClass.Map0KMajor)
      rows
    else rows.indices.map(index => if (index == 0) rows.head else BigInt(0))
  }
}

/**
  * One fast-domain scalar result before local-lane pairing.
  *
  * This is a Stage-C transport schema only; no existing production port uses
  * it yet.  Rich semantic metadata remains authenticated at the endpoint and
  * is reconstructed from its immutable launch descriptor.
  */
case class BitNetProductionCompactDotResult(
    cfg: BitNetContinuousMatrixConfig) extends Bundle {
  private val b = cfg.base
  private val layout = BitNetProductionCompactDotTransportOracle.widths(cfg)

  val values = Vec(SInt(b.accumulatorWidth bits), b.outputLanes)
  val sequenceId = UInt(layout.sequenceIdWidth bits)
  val accumulatorTag = UInt(layout.accumulatorTagWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val pairPhase = UInt(layout.pairPhaseWidth bits)
  val formatClass = UInt(layout.formatClassWidth bits)
}

/**
  * One slow-domain TP2 result after compact scalar assembly.
  *
  * `rawRows(localLane)(row)` is canonical: Map0 retains both rows, while
  * QK/LM require row 1 to be zero.  For `validMask=11`, local lane 1 has the
  * implicit accumulator tag `baseAccumulatorTag + 1`; `validMask=01` contains
  * only lane 0.  A published `validMask=00` is poison and must fail-stop the
  * downstream bridge.
  */
case class BitNetProductionCompactTp2DotResultBeat(
    cfg: BitNetContinuousMatrixConfig) extends Bundle {
  private val b = cfg.base
  private val layout = BitNetProductionCompactDotTransportOracle.widths(cfg)

  val rawRows = Vec(
    Vec(SInt(b.accumulatorWidth bits), b.outputLanes),
    cfg.tokenLanesPerEngine)
  val validMask = Bits(layout.validMaskWidth bits)
  val sequenceId = UInt(layout.sequenceIdWidth bits)
  val baseAccumulatorTag = UInt(layout.accumulatorTagWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val formatClass = UInt(layout.formatClassWidth bits)
}
