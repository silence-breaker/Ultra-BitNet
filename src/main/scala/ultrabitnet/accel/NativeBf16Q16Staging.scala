package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** One raw AUX.BIN BF16 word in a packet-free resident-data burst. */
case class NativeBf16StageBeat() extends Bundle {
  val raw = Bits(16 bits)
  val first = Bool()
  val last = Bool()
}

/**
  * Signed Q16.16 staging result and fail-visible IEEE-754 classification.
  *
  * The original word is retained so firmware can identify a corrupt AUX.BIN
  * entry without reconstructing it from a saturated value.  `overflow` is set
  * for both finite saturation and infinity.  `special` is set only for Inf or
  * NaN.  NaN has the deterministic numeric result zero; infinity saturates by
  * sign.  `underflow` identifies a finite nonzero BF16 value which rounds to
  * Q16.16 zero.  Negative zero remains observable through `raw(15)`.
  */
case class NativeBf16Q16Result() extends Bundle {
  val value = SInt(32 bits)
  val raw = Bits(16 bits)
  val first = Bool()
  val last = Bool()
  val isZero = Bool()
  val isSubnormal = Bool()
  val underflow = Bool()
  val overflow = Bool()
  val isInf = Bool()
  val isNaN = Bool()
  val special = Bool()
}

/** Registered boundary between BF16 shift/round and signed saturation. */
private[accel] case class NativeBf16Q16MagnitudeStage() extends Bundle {
  val magnitude = UInt(40 bits)
  val raw = Bits(16 bits)
  val first = Bool()
  val last = Bool()
  val sign = Bool()
  val isZero = Bool()
  val isSubnormal = Bool()
  val finiteNonzero = Bool()
  val finiteExponentOverflow = Bool()
  val isInf = Bool()
  val isNaN = Bool()
  val special = Bool()
}

/**
  * Packet-free BF16 to signed-Q16.16 resident parameter converter.
  *
  * Finite values are converted with round-to-nearest, ties away from zero,
  * matching the fixed-point convention used by native RMSNorm and FFN math.
  * Exact conversion reduces to an 8-bit significand and a bounded binary
  * shift; no floating-point core, divider, packet parser or micro-op is used.
  * The registered Stream boundary sustains one AUX.BIN word per cycle and
  * holds every result/status bit stable while the consumer is stalled.
  *
  * This single format boundary is shared by RMS gamma, the four subnorm data
  * sets and all seven projection scales.
  */
class NativeBf16Q16Staging extends Component {
  val io = new Bundle {
    val input = slave(Stream(NativeBf16StageBeat()))
    val output = master(Stream(NativeBf16Q16Result()))

    /** Sticky until reset, so a special AUX word cannot be silently dropped. */
    val specialFaultSticky = out Bool()
    /** Sticky finite overflow, kept separate from Inf/NaN corruption. */
    val finiteOverflowSticky = out Bool()
    val acceptedWords = out UInt(32 bits)
    val emittedWords = out UInt(32 bits)
  }

  /*
   * Three elastic register boundaries are intentional.  The raw AUX BRAM
   * beat is first captured, the bounded BF16 shift/round is then captured,
   * and only the final stage performs sign/saturation.  This keeps a BRAM
   * output from traversing both barrel/rounding logic and the 32-bit signed
   * negate path in one 266 MHz cycle while retaining one-word/cycle
   * throughput under arbitrary downstream backpressure.
   */
  val captured = io.input.m2sPipe()
  val raw = captured.raw
  val sign = raw(15)
  val exponent = raw(14 downto 7).asUInt
  val fraction = raw(6 downto 0).asUInt
  val exponentZero = exponent === 0
  val exponentAllOnes = exponent === U(255, 8 bits)
  val fractionZero = fraction === 0
  val isZero = exponentZero && fractionZero
  val isSubnormal = exponentZero && !fractionZero
  val isInf = exponentAllOnes && fractionZero
  val isNaN = exponentAllOnes && !fractionZero
  val special = isInf || isNaN
  val finiteNonzero = !exponentAllOnes && !isZero

  // A normal BF16 significand is 1.fraction, represented here as the exact
  // unsigned integer [128,255].  Q16.16 raw magnitude is
  // significand * 2^(exponent-118).
  val significand = (B"1'b1" ## fraction.asBits).asUInt
  val magnitude = UInt(40 bits)
  magnitude := 0

  // Exponents 118..142 need a bounded left shift of at most 24.  Larger
  // finite exponents are known overflow before a wide barrel shift is built.
  val leftShift = UInt(5 bits)
  leftShift := 0
  when(exponent >= U(118, 8 bits) && exponent <= U(142, 8 bits)) {
    leftShift := (exponent - U(118, 8 bits)).resize(5)
    magnitude := (significand.resize(40) |<< leftShift).resize(40)
  }

  // Exponents 110..117 require fixed right shifts 8..1.  For a positive
  // significand, ties-away rounding is quotient + the highest discarded bit;
  // the fixed slices avoid a second barrel shifter and compare/subtract chain.
  val roundedRight = UInt(9 bits)
  roundedRight := 0
  switch(exponent) {
    for (shift <- 1 to 8) {
      is(U(118 - shift, 8 bits)) {
        if (shift == 8) {
          roundedRight := significand(7).asUInt.resize(9)
        } else {
          roundedRight := (
            significand(7 downto shift).resize(9) +
              significand(shift - 1).asUInt.resize(9)).resize(9)
        }
      }
    }
  }
  when(exponent >= U(110, 8 bits) && exponent < U(118, 8 bits)) {
    magnitude := roundedRight.resize(40)
  }

  val finiteExponentOverflow = !exponentAllOnes && exponent > U(142, 8 bits)

  val magnitudeResult = Stream(NativeBf16Q16MagnitudeStage())
  magnitudeResult.valid := captured.valid
  captured.ready := magnitudeResult.ready
  magnitudeResult.magnitude := magnitude
  magnitudeResult.raw := raw
  magnitudeResult.first := captured.first
  magnitudeResult.last := captured.last
  magnitudeResult.sign := sign
  magnitudeResult.isZero := isZero
  magnitudeResult.isSubnormal := isSubnormal
  magnitudeResult.finiteNonzero := finiteNonzero
  magnitudeResult.finiteExponentOverflow := finiteExponentOverflow
  magnitudeResult.isInf := isInf
  magnitudeResult.isNaN := isNaN
  magnitudeResult.special := special

  val staged = magnitudeResult.m2sPipe()
  val positiveLimit = U(BigInt(Int.MaxValue), 40 bits)
  val negativeLimit = U(BigInt(1) << 31, 40 bits)
  val finiteMagnitudeOverflow = staged.finiteNonzero &&
    ((staged.sign && staged.magnitude > negativeLimit) ||
      (!staged.sign && staged.magnitude > positiveLimit))
  val finiteOverflow = staged.finiteExponentOverflow ||
    finiteMagnitudeOverflow

  val convertedValue = SInt(32 bits)
  convertedValue := 0
  when(staged.isNaN) {
    convertedValue := 0
  } elsewhen (staged.isInf || finiteOverflow) {
    when(staged.sign) {
      convertedValue := S(Int.MinValue, 32 bits)
    } otherwise {
      convertedValue := S(Int.MaxValue, 32 bits)
    }
  } otherwise {
    when(staged.sign) {
      convertedValue := (-staged.magnitude.resize(33).asSInt).resize(32)
    } otherwise {
      convertedValue := staged.magnitude.resize(32).asSInt
    }
  }

  val underflow = staged.finiteNonzero && !finiteOverflow &&
    staged.magnitude === 0

  val converted = Stream(NativeBf16Q16Result())
  converted.valid := staged.valid
  staged.ready := converted.ready
  converted.value := convertedValue
  converted.raw := staged.raw
  converted.first := staged.first
  converted.last := staged.last
  converted.isZero := staged.isZero
  converted.isSubnormal := staged.isSubnormal
  converted.underflow := underflow
  converted.overflow := finiteOverflow || staged.isInf
  converted.isInf := staged.isInf
  converted.isNaN := staged.isNaN
  converted.special := staged.special

  io.output << converted.m2sPipe()

  val specialFault = RegInit(False)
  val finiteOverflowFault = RegInit(False)
  val acceptedCounter = Reg(UInt(32 bits)) init (0)
  val emittedCounter = Reg(UInt(32 bits)) init (0)
  when(io.input.fire) {
    acceptedCounter := acceptedCounter + 1
    val acceptedExponent = io.input.raw(14 downto 7)
    when(acceptedExponent === B(255, 8 bits)) {
      specialFault := True
    }
  }
  when(converted.fire && finiteOverflow) { finiteOverflowFault := True }
  when(io.output.fire) { emittedCounter := emittedCounter + 1 }

  io.specialFaultSticky := specialFault
  io.finiteOverflowSticky := finiteOverflowFault
  io.acceptedWords := acceptedCounter
  io.emittedWords := emittedCounter
}

case class NativeBf16Q16OracleResult(
    value: Int,
    raw: Int,
    isZero: Boolean,
    isSubnormal: Boolean,
    underflow: Boolean,
    overflow: Boolean,
    isInf: Boolean,
    isNaN: Boolean,
    special: Boolean
)

/** Bit-accurate pure-Scala reference for [[NativeBf16Q16Staging]]. */
object NativeBf16Q16Math {
  private val PositiveLimit = BigInt(Int.MaxValue)
  private val NegativeLimit = BigInt(1) << 31

  def convert(rawWord: Int): NativeBf16Q16OracleResult = {
    val raw = rawWord & 0xffff
    val negative = (raw & 0x8000) != 0
    val exponent = (raw >>> 7) & 0xff
    val fraction = raw & 0x7f
    val isZero = exponent == 0 && fraction == 0
    val isSubnormal = exponent == 0 && fraction != 0
    val isInf = exponent == 0xff && fraction == 0
    val isNaN = exponent == 0xff && fraction != 0
    val special = isInf || isNaN

    val magnitude = if (exponent == 0 || exponent == 0xff) {
      BigInt(0)
    } else {
      val significand = BigInt(128 + fraction)
      val shift = exponent - 118
      if (shift >= 0) {
        significand << shift
      } else {
        val right = -shift
        if (right > 8) {
          BigInt(0)
        } else {
          val quotient = significand >> right
          val remainder = significand - (quotient << right)
          if ((remainder << 1) >= (BigInt(1) << right)) quotient + 1
          else quotient
        }
      }
    }

    val limit = if (negative) NegativeLimit else PositiveLimit
    val finiteOverflow = !special && !isZero && magnitude > limit
    val overflow = finiteOverflow || isInf
    val value = if (isNaN) {
      0
    } else if (overflow) {
      if (negative) Int.MinValue else Int.MaxValue
    } else if (negative) {
      (-magnitude).toInt
    } else {
      magnitude.toInt
    }
    val underflow = !special && !isZero && !finiteOverflow && magnitude == 0

    NativeBf16Q16OracleResult(
      value = value,
      raw = raw,
      isZero = isZero,
      isSubnormal = isSubnormal,
      underflow = underflow,
      overflow = overflow,
      isInf = isInf,
      isNaN = isNaN,
      special = special
    )
  }
}

/** Production AUX/gamma staging RTL entry point. */
object GenerateNativeBf16Q16Staging extends App {
  SpinalConfig(
    targetDirectory = "target/generated-bitnet-bf16-q16-staging",
    oneFilePerComponent = false
  ).generateVerilog(new NativeBf16Q16Staging)
}
