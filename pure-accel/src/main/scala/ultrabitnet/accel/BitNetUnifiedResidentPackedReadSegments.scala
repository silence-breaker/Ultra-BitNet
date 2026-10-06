package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * One token lane from a packed resident row.
  *
  * This is intentionally different from the packed-write segment ABI.  A
  * write segment is eight complete 128-bit banks.  A read segment instead
  * gathers one byte from the selected token lane at every one of the 128
  * feature sites:
  *
  *   segmentData(site) =
  *     words(site >> 2).feature(site & 3)(segmentIndex)
  */
case class BitNetUnifiedResidentPackedI8TokenLaneSegment(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  private val layout = cfg.layout
  val value = Vec(
    SInt(layout.packedWidth bits),
    layout.bankCount * layout.packedFeaturesPerWord)
}

/** Physical row identity plus one 1,024-bit token-lane segment. */
case class BitNetUnifiedResidentTensorFastPackedReadSegmentResponse(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val scratchpad = cfg.scratchpad
  private val layout = scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  val tag = UInt(cfg.tagWidth bits)
  val segmentIndex = UInt(log2Up(layout.packedTransportSegmentCount) bits)
  val last = Bool()
  val data = BitNetUnifiedResidentPackedI8TokenLaneSegment(scratchpad)
}

/** Semantic row identity, fail-stop status and one token-lane segment. */
case class BitNetUnifiedResidentPackedI8ReadSegmentData(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  private val layout = cfg.layout
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(layout.localRowWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  val segmentIndex = UInt(log2Up(layout.packedTransportSegmentCount) bits)
  val last = Bool()
  val data = BitNetUnifiedResidentPackedI8TokenLaneSegment(cfg)
  val fault = Bool()
  val faultCode = Bits(4 bits)
}

/** One physical packed-row response transported as four parallel token lanes.
  *
  * Unlike [[BitNetUnifiedResidentTensorFastPackedReadSegmentResponse]], this
  * is an atomic production payload rather than one beat of a four-beat serial
  * protocol.  Row identity is shared once and all four 1,024-bit segments are
  * present on the same backend response pulse.  The related-clock receiver may
  * subsequently retain a four-bit pending mask, but must never copy this wide
  * payload into another register bank.
  */
case class BitNetUnifiedResidentTensorFastPackedReadParallelPayload(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val scratchpad = cfg.scratchpad
  private val layout = scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  val laneMask = Bits(layout.packedTransportSegmentCount bits)
  val segments = Vec(
    BitNetUnifiedResidentPackedI8TokenLaneSegment(scratchpad),
    layout.packedTransportSegmentCount)
}

/** Related-clock physical response channel for one parallel packed row.
  *
  * The fast backend emits the payload above atomically as a Flow.  The bridge
  * converts that pulse into this independently acknowledged slow-domain
  * channel: shared physical identity and all segment data travel forward,
  * while only the four ready bits travel back toward the bridge owner.
  */
case class BitNetUnifiedResidentTensorFastPackedReadParallelChannel(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig)
    extends Bundle with IMasterSlave {
  private val scratchpad = cfg.scratchpad
  private val layout = scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  val laneMask = Bits(layout.packedTransportSegmentCount bits)
  val segmentValid = Bits(layout.packedTransportSegmentCount bits)
  val segmentReady = Bits(layout.packedTransportSegmentCount bits)
  val segments = Vec(
    BitNetUnifiedResidentPackedI8TokenLaneSegment(scratchpad),
    layout.packedTransportSegmentCount)

  override def asMaster(): Unit = {
    out(
      absoluteRow,
      tag,
      bankMask,
      laneMask,
      segmentValid,
      segments)
    in(segmentReady)
  }

  def fireMask: Bits = segmentValid & segmentReady
}

/** Slow-domain production response channel for one parallel packed row.
  *
  * Header and segment data are owned by the producer while any bit of
  * `segmentValid` remains asserted.  A consumer may acknowledge independent
  * lanes with `segmentReady`; the logical row retires only after every
  * requested lane has fired.  Metadata is shared, so mismatched row/fault
  * identity between lanes is structurally impossible.
  */
case class BitNetUnifiedResidentPackedI8ReadParallelChannel(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig)
    extends Bundle with IMasterSlave {
  private val layout = cfg.layout
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(layout.localRowWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  val laneMask = Bits(layout.packedTransportSegmentCount bits)
  val fault = Bool()
  val faultCode = Bits(4 bits)
  val segmentValid = Bits(layout.packedTransportSegmentCount bits)
  val segmentReady = Bits(layout.packedTransportSegmentCount bits)
  val segments = Vec(
    BitNetUnifiedResidentPackedI8TokenLaneSegment(cfg),
    layout.packedTransportSegmentCount)

  override def asMaster(): Unit = {
    out(
      region,
      localRow,
      bankMask,
      laneMask,
      fault,
      faultCode,
      segmentValid,
      segments)
    in(segmentReady)
  }

  def fireMask: Bits = segmentValid & segmentReady
}

/** Stable elaboration markers for the production compatibility adapter. */
object BitNetUnifiedResidentPackedReadParallelAdapterStorageContract {
  val RetainedPayloadSnapshots: Int = 0
  val RetainedPayloadBits: Int = 0
  val AtomicLegacyRowFire: Boolean = true
}

/** Pure-wiring compatibility endpoint from the parallel production ABI to the
  * legacy monolithic row Stream.
  *
  * This adapter is deliberately all-or-nothing: a legacy row is never exposed
  * until all four token-lane segments are valid, and all four are acknowledged
  * on the one legacy fire.  The fixed join equation is only wire permutation;
  * this component contains no payload register, pending state or wide mux.
  */
class BitNetUnifiedResidentPackedReadParallelToLegacyRowAdapter(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()) extends Component {
  private val layout = cfg.layout
  private val segmentCount = layout.packedTransportSegmentCount

  val retainedPayloadSnapshots: Int =
    BitNetUnifiedResidentPackedReadParallelAdapterStorageContract
      .RetainedPayloadSnapshots
  val retainedPayloadBits: Int =
    BitNetUnifiedResidentPackedReadParallelAdapterStorageContract
      .RetainedPayloadBits
  val atomicLegacyRowFire: Boolean =
    BitNetUnifiedResidentPackedReadParallelAdapterStorageContract
      .AtomicLegacyRowFire
  require(segmentCount == layout.tokenLanes)
  require(retainedPayloadSnapshots == 0)
  require(retainedPayloadBits == 0)
  require(atomicLegacyRowFire)

  val io = new Bundle {
    val parallel = slave(
      BitNetUnifiedResidentPackedI8ReadParallelChannel(cfg))
    val legacy = master(Stream(
      BitNetUnifiedResidentPackedI8ReadData(cfg)))
  }

  val fullLaneMask = B(
    (BigInt(1) << segmentCount) - 1, segmentCount bits)
  val completeRowValid = io.parallel.segmentValid === fullLaneMask &&
    io.parallel.laneMask === fullLaneMask

  io.legacy.valid := completeRowValid
  io.legacy.region := io.parallel.region
  io.legacy.localRow := io.parallel.localRow
  io.legacy.bankMask := io.parallel.bankMask
  io.legacy.fault := io.parallel.fault
  io.legacy.faultCode := io.parallel.faultCode

  for (bank <- 0 until layout.bankCount;
       wordFeature <- 0 until layout.packedFeaturesPerWord;
       lane <- 0 until layout.tokenLanes) {
    val site = bank * layout.packedFeaturesPerWord + wordFeature
    io.legacy.words(bank).feature(wordFeature)(lane) :=
      io.parallel.segments(lane).value(site)
  }

  io.parallel.segmentReady := 0
  when(completeRowValid && io.legacy.ready) {
    io.parallel.segmentReady := fullLaneMask
  }
}

/** Bit-exact production layout oracle for 32 x 64-bit packed rows. */
object BitNetUnifiedResidentPackedReadSegmentLayoutOracle {
  val BankCount: Int = 32
  val WordWidth: Int = 64
  val FeatureCountPerWord: Int = 4
  val TokenLaneCount: Int = 2
  val ElementWidth: Int = 8
  val SegmentFeatureCount: Int = BankCount * FeatureCountPerWord
  val SegmentWidth: Int = SegmentFeatureCount * ElementWidth
  val SegmentCount: Int = TokenLaneCount

  require(SegmentFeatureCount == 128)
  require(SegmentWidth == 1024)

  private val wordLimit = BigInt(1) << WordWidth
  private val segmentLimit = BigInt(1) << SegmentWidth

  private def validateWords(words: Seq[BigInt]): Unit = {
    require(words.size == BankCount,
      s"packed row must contain $BankCount words, got ${words.size}")
    words.foreach { word =>
      require(word >= 0 && word < wordLimit,
        s"packed word exceeds $WordWidth bits: $word")
    }
  }

  private def validateSegments(segments: Seq[BigInt]): Unit = {
    require(segments.size == SegmentCount,
      s"packed row must contain $SegmentCount segments, got ${segments.size}")
    segments.foreach { segment =>
      require(segment >= 0 && segment < segmentLimit,
        s"packed segment exceeds $SegmentWidth bits")
    }
  }

  /** Extracts one token lane using the production feature/bank equation. */
  def segmentData(words: Seq[BigInt], segmentIndex: Int): BigInt = {
    validateWords(words)
    require(segmentIndex >= 0 && segmentIndex < SegmentCount)

    (0 until SegmentFeatureCount).foldLeft(BigInt(0)) {
      case (result, site) =>
        val bank = site >> 2
        val wordFeature = site & 3
        val sourceByte = wordFeature * TokenLaneCount + segmentIndex
        val value = (words(bank) >> (sourceByte * ElementWidth)) & 0xff
        result | (value << (site * ElementWidth))
    }
  }

  def split(words: Seq[BigInt]): Vector[BigInt] = {
    validateWords(words)
    Vector.tabulate(SegmentCount)(index => segmentData(words, index))
  }

  /** Reconstructs the original 32 physical words without changing bit order. */
  def join(segments: Seq[BigInt]): Vector[BigInt] = {
    validateSegments(segments)
    Vector.tabulate(BankCount) { bank =>
      (0 until FeatureCountPerWord).foldLeft(BigInt(0)) {
        case (wordByFeature, wordFeature) =>
          (0 until TokenLaneCount).foldLeft(wordByFeature) {
            case (word, tokenLane) =>
              val site = bank * FeatureCountPerWord + wordFeature
              val value =
                (segments(tokenLane) >> (site * ElementWidth)) & 0xff
              val destinationByte =
                wordFeature * TokenLaneCount + tokenLane
              word | (value << (destinationByte * ElementWidth))
          }
      }
    }
  }
}

/** Two independently acknowledged segments still retire one logical row. */
object BitNetUnifiedResidentPackedReadPendingOracle {
  val SegmentCount: Int = 2
  val FullMask: Int = (1 << SegmentCount) - 1

  case class Step(
      pendingBefore: Int,
      acknowledgeMask: Int,
      acceptedMask: Int,
      duplicateMask: Int,
      pendingAfter: Int,
      completedNow: Boolean,
      protocolFault: Boolean)

  def start: Int = FullMask

  def acknowledge(pending: Int, acknowledgeMask: Int): Step = {
    require((pending & ~FullMask) == 0,
      f"pending mask 0x$pending%x exceeds two segments")
    require((acknowledgeMask & ~FullMask) == 0,
      f"ack mask 0x$acknowledgeMask%x exceeds two segments")

    val accepted = pending & acknowledgeMask
    val duplicate = acknowledgeMask & ~pending
    val next = pending & ~accepted
    Step(
      pendingBefore = pending,
      acknowledgeMask = acknowledgeMask,
      acceptedMask = accepted,
      duplicateMask = duplicate,
      pendingAfter = next,
      completedNow = pending != 0 && accepted != 0 && next == 0,
      protocolFault = duplicate != 0)
  }

  def acknowledgeIndex(pending: Int, segmentIndex: Int): Step = {
    require(segmentIndex >= 0 && segmentIndex < SegmentCount)
    acknowledge(pending, 1 << segmentIndex)
  }
}

/** Pure Scala models used to audit a complete two-segment response. */
object BitNetUnifiedResidentPackedReadSegmentProtocolOracle {
  import BitNetUnifiedResidentPackedReadSegmentLayoutOracle._

  object Reason {
    val SegmentCount = "segment-count"
    val SegmentIndex = "segment-index"
    val SegmentSet = "segment-set"
    val Last = "last"
    val AbsoluteRow = "absolute-row"
    val LocalRow = "local-row"
    val Region = "region"
    val Tag = "tag"
    val BankMask = "bank-mask"
    val Fault = "fault"
    val FaultCode = "fault-code"
    val FaultData = "fault-data"
    val DataWidth = "data-width"
  }

  case class Validation(reasons: Vector[String]) {
    val legal: Boolean = reasons.isEmpty
    val failClosed: Boolean = !legal
  }

  case class PhysicalSegment(
      absoluteRow: Int,
      bankMask: BigInt,
      tag: Int,
      segmentIndex: Int,
      last: Boolean,
      data: BigInt)

  case class SemanticSegment(
      region: Int,
      localRow: Int,
      bankMask: BigInt,
      segmentIndex: Int,
      last: Boolean,
      data: BigInt,
      fault: Boolean,
      faultCode: Int)

  private def framingReasons(
      count: Int,
      indexes: Seq[Int],
      lasts: Seq[(Int, Boolean)],
      data: Seq[BigInt]): Vector[String] = {
    val reasons = Vector.newBuilder[String]
    if (count != SegmentCount) reasons += Reason.SegmentCount
    if (indexes.exists(index => index < 0 || index >= SegmentCount))
      reasons += Reason.SegmentIndex
    if (indexes.sorted != (0 until SegmentCount))
      reasons += Reason.SegmentSet
    if (lasts.exists { case (index, last) =>
        last != (index == SegmentCount - 1) })
      reasons += Reason.Last
    if (data.exists(value => value < 0 || value >= (BigInt(1) << SegmentWidth)))
      reasons += Reason.DataWidth
    reasons.result().distinct
  }

  def validatePhysical(segments: Seq[PhysicalSegment]): Validation = {
    val framing = framingReasons(
      segments.size,
      segments.map(_.segmentIndex),
      segments.map(segment => segment.segmentIndex -> segment.last),
      segments.map(_.data))
    val reasons = Vector.newBuilder[String]
    reasons ++= framing
    segments.headOption.foreach { first =>
      if (segments.exists(_.absoluteRow != first.absoluteRow))
        reasons += Reason.AbsoluteRow
      if (segments.exists(_.bankMask != first.bankMask))
        reasons += Reason.BankMask
      if (segments.exists(_.tag != first.tag))
        reasons += Reason.Tag
    }
    Validation(reasons.result().distinct)
  }

  def validateSemantic(segments: Seq[SemanticSegment]): Validation = {
    val framing = framingReasons(
      segments.size,
      segments.map(_.segmentIndex),
      segments.map(segment => segment.segmentIndex -> segment.last),
      segments.map(_.data))
    val reasons = Vector.newBuilder[String]
    reasons ++= framing
    segments.headOption.foreach { first =>
      if (segments.exists(_.region != first.region))
        reasons += Reason.Region
      if (segments.exists(_.localRow != first.localRow))
        reasons += Reason.LocalRow
      if (segments.exists(_.bankMask != first.bankMask))
        reasons += Reason.BankMask
      if (segments.exists(_.fault != first.fault))
        reasons += Reason.Fault
      if (segments.exists(_.faultCode != first.faultCode))
        reasons += Reason.FaultCode
      if (first.fault && first.faultCode == 0)
        reasons += Reason.FaultCode
      if (!first.fault && first.faultCode != 0)
        reasons += Reason.FaultCode
      if (segments.exists(segment => segment.fault && segment.data != 0))
        reasons += Reason.FaultData
    }
    Validation(reasons.result().distinct)
  }
}
