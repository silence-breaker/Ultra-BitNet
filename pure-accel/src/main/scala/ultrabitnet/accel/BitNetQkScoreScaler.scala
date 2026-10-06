package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Fixed-point and routing contract for the production QK score scaler.
  *
  * A matrix result is a signed integer dot product.  Query and key dynamic
  * quantisers attach positive Q16.16 dequantisation factors.  The scaler emits
  * the signed Q4.12 value consumed by the complete-score buffered softmax
  * lane:
  *
  *   round_away(dot * queryFactorQ16 * keyFactorQ16 /
  *              (2^20 * sqrt(128)))
  *
  * The complete irrational denominator `2^20 * sqrt(128)` is represented with
  * 32 fractional bits.  Its error is below 0.5 ulp at that boundary and the
  * integer contract is exposed by [[BitNetQkScoreScalerMath]], so software
  * and RTL use one definition.
  */
case class BitNetQkScoreScalerConfig(
    tokenIdWidth: Int = 16,
    headWidth: Int = 5,
    keyPositionWidth: Int = 12,
    tagWidth: Int = 20,
    rawDotWidth: Int = 32,
    factorWidth: Int = 32,
    scoreWidth: Int = 16,
    scoreFractionBits: Int = 12
) {
  require(tokenIdWidth >= 12)
  require(headWidth >= 5, "twenty query heads require five bits")
  require(keyPositionWidth >= 12, "the production KV cache holds 4096 keys")
  require(tagWidth >= 1)
  require(rawDotWidth == 32)
  require(factorWidth == 32)
  require(scoreWidth == 16 && scoreFractionBits == 12,
    "the resident buffered-softmax path consumes signed Q4.12")

  val factorLimbWidth: Int = 16
  val factorProductWidth: Int = 64
  val nonSaturatedFactorWidth: Int = 39
  val dotMagnitudeWidth: Int = 32
  val magnitudeProductWidth: Int =
    nonSaturatedFactorWidth + dotMagnitudeWidth
  val sqrtFractionBits: Int = 32
  val divisionRemainderWidth: Int =
    nonSaturatedFactorWidth + sqrtFractionBits
  val quotientWidth: Int = scoreWidth - 1
}

case class BitNetQkScoreInput(cfg: BitNetQkScoreScalerConfig)
    extends Bundle {
  val rawDot = SInt(cfg.rawDotWidth bits)
  val queryFactorQ16 = UInt(cfg.factorWidth bits)
  val keyFactorQ16 = UInt(cfg.factorWidth bits)
  val tokenId = UInt(cfg.tokenIdWidth bits)
  /** Absolute query-head number, zero through nineteen. */
  val head = UInt(cfg.headWidth bits)
  val keyPosition = UInt(cfg.keyPositionWidth bits)
  /** Packet-free completion/routing tag, normally the matrix taskId. */
  val tag = UInt(cfg.tagWidth bits)
}

case class BitNetQkScoreOutput(cfg: BitNetQkScoreScalerConfig)
    extends Bundle {
  val scoreQ12 = SInt(cfg.scoreWidth bits)
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val head = UInt(cfg.headWidth bits)
  val keyPosition = UInt(cfg.keyPositionWidth bits)
  val tag = UInt(cfg.tagWidth bits)
}

case class BitNetQkScoreMeta(cfg: BitNetQkScoreScalerConfig)
    extends Bundle {
  val negative = Bool()
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val head = UInt(cfg.headWidth bits)
  val keyPosition = UInt(cfg.keyPositionWidth bits)
  val tag = UInt(cfg.tagWidth bits)
}

case class BitNetQkFactorPartials(
    cfg: BitNetQkScoreScalerConfig) extends Bundle {
  val lowLow = UInt(32 bits)
  val lowHigh = UInt(32 bits)
  val highLow = UInt(32 bits)
  val highHigh = UInt(32 bits)
  val dotMagnitude = UInt(cfg.dotMagnitudeWidth bits)
  val meta = BitNetQkScoreMeta(cfg)
}

case class BitNetQkFactorProduct(
    cfg: BitNetQkScoreScalerConfig) extends Bundle {
  val factorLow = UInt(cfg.nonSaturatedFactorWidth bits)
  val factorTooLarge = Bool()
  val dotMagnitude = UInt(cfg.dotMagnitudeWidth bits)
  val meta = BitNetQkScoreMeta(cfg)
}

case class BitNetQkRadix16(
    cfg: BitNetQkScoreScalerConfig) extends Bundle {
  val partial = Vec(UInt(cfg.magnitudeProductWidth bits), 16)
  val factorTooLarge = Bool()
  val dotNonZero = Bool()
  val meta = BitNetQkScoreMeta(cfg)
}

case class BitNetQkRadix8(
    cfg: BitNetQkScoreScalerConfig) extends Bundle {
  val partial = Vec(UInt(cfg.magnitudeProductWidth bits), 8)
  val factorTooLarge = Bool()
  val dotNonZero = Bool()
  val meta = BitNetQkScoreMeta(cfg)
}

case class BitNetQkRadix4(
    cfg: BitNetQkScoreScalerConfig) extends Bundle {
  val partial = Vec(UInt(cfg.magnitudeProductWidth bits), 4)
  val factorTooLarge = Bool()
  val dotNonZero = Bool()
  val meta = BitNetQkScoreMeta(cfg)
}

case class BitNetQkRadix2(
    cfg: BitNetQkScoreScalerConfig) extends Bundle {
  val partial = Vec(UInt(cfg.magnitudeProductWidth bits), 2)
  val factorTooLarge = Bool()
  val dotNonZero = Bool()
  val meta = BitNetQkScoreMeta(cfg)
}

case class BitNetQkMagnitudeProduct(
    cfg: BitNetQkScoreScalerConfig) extends Bundle {
  val magnitude = UInt(cfg.magnitudeProductWidth bits)
  val factorTooLarge = Bool()
  val dotNonZero = Bool()
  val meta = BitNetQkScoreMeta(cfg)
}

case class BitNetQkDivideState(
    cfg: BitNetQkScoreScalerConfig) extends Bundle {
  val remainder = UInt(cfg.divisionRemainderWidth bits)
  val quotient = UInt(cfg.quotientWidth bits)
  val clamp = Bool()
  val meta = BitNetQkScoreMeta(cfg)
}

/**
  * BitNet packet-free QK dequantisation and head scaling.
  *
  * Four DSP48E2s form one exact pipelined U32 x U32 multiplier.  Applying the
  * raw dot with another inferred multiplier would at least double that cost.
  * Instead, sixteen radix-4 partials pass through four registered adder-tree
  * levels.  A fifteen-stage restoring divider then converts to Q4.12.  Every
  * stage is elastic: the block accepts one score per clock in steady state and
  * holds all payload and routing metadata under arbitrary output backpressure.
  */
class BitNetQkScoreScaler(
    cfg: BitNetQkScoreScalerConfig = BitNetQkScoreScalerConfig())
    extends Component {
  val io = new Bundle {
    val input = slave(Stream(BitNetQkScoreInput(cfg)))
    val output = master(Stream(BitNetQkScoreOutput(cfg)))
  }

  private val denominatorQ32 = BigInt("b504f333f9de65", 16)
  private val saturationNumerator = BigInt("5a821f1784", 16)

  val rawWide = io.input.rawDot.resize(cfg.rawDotWidth + 1)
  val rawNegative = rawWide.msb
  val rawMagnitude = UInt(cfg.dotMagnitudeWidth bits)
  when(rawNegative) {
    rawMagnitude := (-rawWide).asUInt.resize(cfg.dotMagnitudeWidth)
  } otherwise {
    rawMagnitude := rawWide.asUInt.resize(cfg.dotMagnitudeWidth)
  }

  val queryLow = io.input.queryFactorQ16(15 downto 0)
  val queryHigh = io.input.queryFactorQ16(31 downto 16)
  val keyLow = io.input.keyFactorQ16(15 downto 0)
  val keyHigh = io.input.keyFactorQ16(31 downto 16)

  val factorPartials = Stream(BitNetQkFactorPartials(cfg))
  factorPartials.valid := io.input.valid
  io.input.ready := factorPartials.ready
  factorPartials.lowLow := queryLow * keyLow
  factorPartials.lowHigh := queryLow * keyHigh
  factorPartials.highLow := queryHigh * keyLow
  factorPartials.highHigh := queryHigh * keyHigh
  factorPartials.lowLow.addAttribute("use_dsp", "yes")
  factorPartials.lowHigh.addAttribute("use_dsp", "yes")
  factorPartials.highLow.addAttribute("use_dsp", "yes")
  factorPartials.highHigh.addAttribute("use_dsp", "yes")
  factorPartials.dotMagnitude := rawMagnitude
  factorPartials.meta.negative := rawNegative
  factorPartials.meta.tokenId := io.input.tokenId
  factorPartials.meta.head := io.input.head
  factorPartials.meta.keyPosition := io.input.keyPosition
  factorPartials.meta.tag := io.input.tag
  val registeredFactorPartials = factorPartials.m2sPipe()

  val cross = registeredFactorPartials.lowHigh.resize(33) +
    registeredFactorPartials.highLow.resize(33)
  val factorProductWide =
    registeredFactorPartials.lowLow.resize(cfg.factorProductWidth) +
      (cross.resize(cfg.factorProductWidth) |<< 16) +
      (registeredFactorPartials.highHigh.resize(cfg.factorProductWidth) |<< 32)

  val factorProduct = Stream(BitNetQkFactorProduct(cfg))
  factorProduct.valid := registeredFactorPartials.valid
  registeredFactorPartials.ready := factorProduct.ready
  factorProduct.factorLow := factorProductWide(
    cfg.nonSaturatedFactorWidth - 1 downto 0)
  factorProduct.factorTooLarge := factorProductWide(
    cfg.factorProductWidth - 1 downto cfg.nonSaturatedFactorWidth).orR
  factorProduct.dotMagnitude := registeredFactorPartials.dotMagnitude
  factorProduct.meta := registeredFactorPartials.meta
  val registeredFactorProduct = factorProduct.m2sPipe()

  val radix16 = Stream(BitNetQkRadix16(cfg))
  radix16.valid := registeredFactorProduct.valid
  registeredFactorProduct.ready := radix16.ready
  radix16.factorTooLarge := registeredFactorProduct.factorTooLarge
  radix16.dotNonZero := registeredFactorProduct.dotMagnitude.orR
  radix16.meta := registeredFactorProduct.meta

  // No '*' operator is used here: these are radix-4 selection and add paths,
  // intentionally mapped to LUT carry chains rather than additional DSPs.
  for (digit <- 0 until 16) {
    val base = (registeredFactorProduct.factorLow
      .resize(cfg.magnitudeProductWidth) |<< (digit * 2))
      .resize(cfg.magnitudeProductWidth)
    val twice = (base |<< 1).resize(cfg.magnitudeProductWidth)
    val three = (base + twice).resize(cfg.magnitudeProductWidth)
    switch(registeredFactorProduct.dotMagnitude(digit * 2 + 1 downto digit * 2)) {
      is(U"00") { radix16.partial(digit) := 0 }
      is(U"01") { radix16.partial(digit) := base }
      is(U"10") { radix16.partial(digit) := twice }
      default { radix16.partial(digit) := three }
    }
  }
  val registeredRadix16 = radix16.m2sPipe()

  val radix8 = Stream(BitNetQkRadix8(cfg))
  radix8.valid := registeredRadix16.valid
  registeredRadix16.ready := radix8.ready
  for (index <- 0 until 8) {
    radix8.partial(index) := (registeredRadix16.partial(index * 2) +
      registeredRadix16.partial(index * 2 + 1)).resized
  }
  radix8.factorTooLarge := registeredRadix16.factorTooLarge
  radix8.dotNonZero := registeredRadix16.dotNonZero
  radix8.meta := registeredRadix16.meta
  val registeredRadix8 = radix8.m2sPipe()

  val radix4 = Stream(BitNetQkRadix4(cfg))
  radix4.valid := registeredRadix8.valid
  registeredRadix8.ready := radix4.ready
  for (index <- 0 until 4) {
    radix4.partial(index) := (registeredRadix8.partial(index * 2) +
      registeredRadix8.partial(index * 2 + 1)).resized
  }
  radix4.factorTooLarge := registeredRadix8.factorTooLarge
  radix4.dotNonZero := registeredRadix8.dotNonZero
  radix4.meta := registeredRadix8.meta
  val registeredRadix4 = radix4.m2sPipe()

  val radix2 = Stream(BitNetQkRadix2(cfg))
  radix2.valid := registeredRadix4.valid
  registeredRadix4.ready := radix2.ready
  for (index <- 0 until 2) {
    radix2.partial(index) := (registeredRadix4.partial(index * 2) +
      registeredRadix4.partial(index * 2 + 1)).resized
  }
  radix2.factorTooLarge := registeredRadix4.factorTooLarge
  radix2.dotNonZero := registeredRadix4.dotNonZero
  radix2.meta := registeredRadix4.meta
  val registeredRadix2 = radix2.m2sPipe()

  val magnitudeProduct = Stream(BitNetQkMagnitudeProduct(cfg))
  magnitudeProduct.valid := registeredRadix2.valid
  registeredRadix2.ready := magnitudeProduct.ready
  magnitudeProduct.magnitude := (registeredRadix2.partial(0) +
    registeredRadix2.partial(1)).resized
  magnitudeProduct.factorTooLarge := registeredRadix2.factorTooLarge
  magnitudeProduct.dotNonZero := registeredRadix2.dotNonZero
  magnitudeProduct.meta := registeredRadix2.meta
  val registeredMagnitude = magnitudeProduct.m2sPipe()

  val magnitudeHigh = registeredMagnitude.magnitude(
    cfg.magnitudeProductWidth - 1 downto cfg.nonSaturatedFactorWidth).orR
  val magnitudeLow = registeredMagnitude.magnitude(
    cfg.nonSaturatedFactorWidth - 1 downto 0)
  val initialClamp =
    (registeredMagnitude.factorTooLarge && registeredMagnitude.dotNonZero) ||
      magnitudeHigh || magnitudeLow >= U(saturationNumerator,
        cfg.nonSaturatedFactorWidth bits)

  val divideStart = Stream(BitNetQkDivideState(cfg))
  divideStart.valid := registeredMagnitude.valid
  registeredMagnitude.ready := divideStart.ready
  divideStart.remainder := (magnitudeLow
    .resize(cfg.divisionRemainderWidth) |<< cfg.sqrtFractionBits).resized
  divideStart.quotient := 0
  divideStart.clamp := initialClamp
  divideStart.meta := registeredMagnitude.meta
  var divideStage = divideStart.m2sPipe()

  // For non-clamped values the integer quotient is at most 32767, so bits
  // 14:0 are sufficient.  Clamped values traverse the same elastic stages.
  for (bit <- (cfg.quotientWidth - 1) to 0 by -1) {
    val next = Stream(BitNetQkDivideState(cfg))
    val shiftedDenominator = U(
      denominatorQ32 << bit, cfg.divisionRemainderWidth bits)
    val take = !divideStage.clamp &&
      divideStage.remainder >= shiftedDenominator
    next.valid := divideStage.valid
    divideStage.ready := next.ready
    next.remainder := divideStage.remainder
    next.quotient := divideStage.quotient
    when(take) {
      next.remainder := divideStage.remainder - shiftedDenominator
      next.quotient(bit) := True
    }
    when(divideStage.clamp) {
      next.remainder := 0
      next.quotient := 0
    }
    next.clamp := divideStage.clamp
    next.meta := divideStage.meta
    divideStage = next.m2sPipe()
  }

  val twiceRemainder =
    (divideStage.remainder.resize(cfg.divisionRemainderWidth + 1) |<< 1)
  val roundUp = twiceRemainder >= U(
    denominatorQ32, cfg.divisionRemainderWidth + 1 bits)
  val roundedMagnitude = UInt((cfg.scoreWidth + 1) bits)
  roundedMagnitude := divideStage.quotient.resize(cfg.scoreWidth + 1)
  when(divideStage.clamp) {
    roundedMagnitude := U(BigInt(1) << (cfg.scoreWidth - 1),
      cfg.scoreWidth + 1 bits)
  } elsewhen (roundUp) {
    roundedMagnitude := (divideStage.quotient.resize(cfg.scoreWidth + 1) + 1)
  }

  val outputStage = Stream(BitNetQkScoreOutput(cfg))
  outputStage.valid := divideStage.valid
  divideStage.ready := outputStage.ready
  outputStage.tokenId := divideStage.meta.tokenId
  outputStage.head := divideStage.meta.head
  outputStage.keyPosition := divideStage.meta.keyPosition
  outputStage.tag := divideStage.meta.tag
  outputStage.scoreQ12 := 0
  when(divideStage.meta.negative) {
    when(roundedMagnitude >= U(BigInt(1) << (cfg.scoreWidth - 1),
        cfg.scoreWidth + 1 bits)) {
      outputStage.scoreQ12 := S(-(BigInt(1) << (cfg.scoreWidth - 1)),
        cfg.scoreWidth bits)
    } otherwise {
      outputStage.scoreQ12 := (-roundedMagnitude.asSInt)
        .resize(cfg.scoreWidth)
    }
  } otherwise {
    when(roundedMagnitude >= U(BigInt(1) << (cfg.scoreWidth - 1),
        cfg.scoreWidth + 1 bits)) {
      outputStage.scoreQ12 := S((BigInt(1) << (cfg.scoreWidth - 1)) - 1,
        cfg.scoreWidth bits)
    } otherwise {
      outputStage.scoreQ12 := roundedMagnitude.resize(cfg.scoreWidth).asSInt
    }
  }

  io.output << outputStage.m2sPipe()
}

/** Bit-exact software contract used by host checks and Verilator tests. */
object BitNetQkScoreScalerMath {
  /** round((2^20 * sqrt(128)) * 2^32). */
  val DenominatorQ32: BigInt = BigInt("b504f333f9de65", 16)
  private val FractionBits = 32
  private val PositiveMax = BigInt(32767)
  private val NegativeMagnitudeMax = BigInt(32768)

  private def unsigned32(value: Long): BigInt =
    BigInt(value & 0xffffffffL)

  /** Round magnitude to nearest, with an exact half rounded away from zero. */
  def apply(rawDot: Int, queryFactorQ16: Long, keyFactorQ16: Long): Int = {
    val magnitude = if (rawDot < 0) -BigInt(rawDot) else BigInt(rawDot)
    val numerator = magnitude * unsigned32(queryFactorQ16) *
      unsigned32(keyFactorQ16)
    val scaledNumerator = numerator << FractionBits
    val roundedMagnitude =
      (scaledNumerator * 2 + DenominatorQ32) / (DenominatorQ32 * 2)

    if (rawDot < 0) {
      -roundedMagnitude.min(NegativeMagnitudeMax).toInt
    } else {
      roundedMagnitude.min(PositiveMax).toInt
    }
  }
}

object GenerateBitNetQkScoreScaler extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-qk-score-scaler")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetQkScoreScaler())
}
