package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Scaling contract between raw matrix accumulators and Q16.16 SPU data. */
case class NativeScaleResidualConfig(
    lanes: Int = 4,
    accumulatorWidth: Int = 32,
    valueWidth: Int = 32,
    factorWidth: Int = 32,
    fractionBits: Int = 16
) {
  require(lanes > 0)
  require(accumulatorWidth == 32 && valueWidth == 32)
  require(factorWidth == 32 && fractionBits == 16)

  val factorProductWidth: Int = factorWidth * 2
  val magnitudeWidth: Int = accumulatorWidth + 1
  val scaledProductWidth: Int = magnitudeWidth + factorWidth
}

case class NativeScaleResidualBeat(cfg: NativeScaleResidualConfig)
    extends Bundle {
  /** Existing PL-resident Q16.16 value; ignored when residualEnable is false. */
  val residual = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  /** Raw integer dot-product emitted by the matrix engine. */
  val accumulator = Vec(SInt(cfg.accumulatorWidth bits), cfg.lanes)
  /** Per-token dynamic-dequant factor, unsigned Q16.16. */
  val activationFactor = Vec(UInt(cfg.factorWidth bits), cfg.lanes)
  /** Projection weight scale loaded once per projection, unsigned Q16.16. */
  val weightScale = UInt(cfg.factorWidth bits)
  val laneMask = Bits(cfg.lanes bits)
  val residualEnable = Bool()
  val first = Bool()
  val last = Bool()
}

case class NativeScaleFactorStage(cfg: NativeScaleResidualConfig)
    extends Bundle {
  val residual = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  val accumulator = Vec(SInt(cfg.accumulatorWidth bits), cfg.lanes)
  val deltaFactor = Vec(UInt(cfg.factorWidth bits), cfg.lanes)
  val laneMask = Bits(cfg.lanes bits)
  val residualEnable = Bool()
  val first = Bool()
  val last = Bool()
}

case class NativeScaleFactorProductStage(cfg: NativeScaleResidualConfig)
    extends Bundle {
  val residual = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  val accumulator = Vec(SInt(cfg.accumulatorWidth bits), cfg.lanes)
  val factorProduct = Vec(UInt(cfg.factorProductWidth bits), cfg.lanes)
  val laneMask = Bits(cfg.lanes bits)
  val residualEnable = Bool()
  val first = Bool()
  val last = Bool()
}

case class NativeScaleMagnitudeStage(cfg: NativeScaleResidualConfig)
    extends Bundle {
  val residual = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  val accumulatorMagnitude = Vec(UInt(cfg.magnitudeWidth bits), cfg.lanes)
  val accumulatorNegative = Bits(cfg.lanes bits)
  val deltaFactor = Vec(UInt(cfg.factorWidth bits), cfg.lanes)
  val laneMask = Bits(cfg.lanes bits)
  val residualEnable = Bool()
  val first = Bool()
  val last = Bool()
}

case class NativeScaleProductStage(cfg: NativeScaleResidualConfig)
    extends Bundle {
  val residual = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  val scaledMagnitude = Vec(UInt(cfg.scaledProductWidth bits), cfg.lanes)
  val accumulatorNegative = Bits(cfg.lanes bits)
  val laneMask = Bits(cfg.lanes bits)
  val residualEnable = Bool()
  val first = Bool()
  val last = Bool()
}

case class NativeScaleResidualResult(cfg: NativeScaleResidualConfig)
    extends Bundle {
  /** Saturated Q16.16 projection value before the optional residual add. */
  val scaledDelta = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  /** Saturated Q16.16 final value. */
  val value = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  val laneMask = Bits(cfg.lanes bits)
  val first = Bool()
  val last = Bool()
}

case class NativeScaleDeltaStage(cfg: NativeScaleResidualConfig)
    extends Bundle {
  val residual = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  val scaledDelta = Vec(SInt(cfg.valueWidth bits), cfg.lanes)
  val laneMask = Bits(cfg.lanes bits)
  val residualEnable = Bool()
  val first = Bool()
  val last = Bool()
}

/**
  * Packet-free, lane-serial matrix scale fusion and residual adder.
  *
  * It implements the native fixed-point equivalent of
  *
  *   delta = accumulator * activation_dequant * weight_scale
  *   result = residualEnable ? residual + delta : delta
  *
  * All externally visible values are signed Q16.16.  Factor multiplication is
  * rounded to nearest before the raw integer accumulator is applied.  Both the
  * scaled projection and residual sum saturate instead of wrapping.
  *
  * The production TP4 matrix emits one final feature only after all twenty
  * hidden-dimension K tiles have been consumed.  Replicating both wide
  * multipliers four times therefore cannot improve the DDR/matrix roofline.
  * This implementation holds one TP4 beat, visits one token lane per slow
  * clock and shares one U32 x U32 factor multiplier plus one U33 x U32 raw-dot
  * multiplier.  On DSP48E2 each wide multiplier maps to four sites, reducing
  * this block from 32 to eight DSPs while retaining the original Vec4 stream
  * boundary and bit-exact numeric contract.
  *
  * Every wide arithmetic operation ends at a register.  The complete result
  * is held stable under arbitrary output backpressure, and a new TP4 beat is
  * accepted as soon as the previous result is consumed.  No activation data
  * is spilled or exposed outside the unique matrix-result endpoint.
  */
class NativeScaleResidual(
    cfg: NativeScaleResidualConfig = NativeScaleResidualConfig())
    extends Component {
  require(cfg.lanes == 4,
    "the production scale endpoint is fixed to BitNet TP4")

  val io = new Bundle {
    val input = slave(Stream(NativeScaleResidualBeat(cfg)))
    val output = master(Stream(NativeScaleResidualResult(cfg)))
  }

  private val positiveMax = BigInt("7fffffff", 16)
  private val negativeMagnitudeMax = BigInt(1) << (cfg.valueWidth - 1)

  val held = Reg(NativeScaleResidualBeat(cfg)) init (
    NativeScaleResidualBeat(cfg).getZero)
  val issuing = RegInit(False)
  val issueLane = Reg(UInt(log2Up(cfg.lanes) bits)) init 0

  /* Stage 0: select one lane and register both multiplier operand sets. */
  val operandValid = RegNext(issuing) init False
  val operandLane = RegNextWhen(issueLane, issuing) init 0
  val operandActivationFactor = Reg(UInt(cfg.factorWidth bits)) init 0
  val operandWeightScale = Reg(UInt(cfg.factorWidth bits)) init 0
  val operandAccumulator = Reg(SInt(cfg.accumulatorWidth bits)) init 0
  val operandResidual = Reg(SInt(cfg.valueWidth bits)) init 0
  val operandLaneActive = RegInit(False)

  when(issuing) {
    operandActivationFactor := held.activationFactor(issueLane)
    operandWeightScale := held.weightScale
    operandAccumulator := held.accumulator(issueLane)
    operandResidual := held.residual(issueLane)
    operandLaneActive := held.laneMask(issueLane)
  }

  /* Stage 1: one shared U32 x U32 factor multiplier (four DSP48E2s). */
  val factorProduct = (
    operandActivationFactor * operandWeightScale
  ).resize(cfg.factorProductWidth)
  factorProduct.addAttribute("use_dsp", "yes")
  val factorProductReg = Reg(UInt(cfg.factorProductWidth bits)) init 0
  val factorProductValid = RegNext(operandValid) init False
  val factorProductLane = RegNextWhen(operandLane, operandValid) init 0
  val factorProductAccumulator = Reg(SInt(cfg.accumulatorWidth bits)) init 0
  val factorProductResidual = Reg(SInt(cfg.valueWidth bits)) init 0
  val factorProductLaneActive = RegInit(False)
  when(operandValid) {
    factorProductReg := factorProduct
    factorProductAccumulator := operandAccumulator
    factorProductResidual := operandResidual
    factorProductLaneActive := operandLaneActive
  }

  /* Stage 2: round/saturate the factor and form the exact dot magnitude. */
  val roundedFactor = (
    factorProductReg.resize(cfg.factorProductWidth + 1) +
      U(BigInt(1) << (cfg.fractionBits - 1),
        cfg.factorProductWidth + 1 bits)
  ).resize(cfg.factorProductWidth + 1)
  val shiftedFactor = roundedFactor |>> cfg.fractionBits
  val nextDeltaFactor = UInt(cfg.factorWidth bits)
  when(shiftedFactor >
      U(BigInt("ffffffff", 16), shiftedFactor.getWidth bits)) {
    nextDeltaFactor := U(BigInt("ffffffff", 16), cfg.factorWidth bits)
  } otherwise {
    nextDeltaFactor := shiftedFactor.resize(cfg.factorWidth)
  }

  val accumulatorWide = factorProductAccumulator.resize(cfg.magnitudeWidth)
  val nextAccumulatorNegative = accumulatorWide.msb
  val nextAccumulatorMagnitude = UInt(cfg.magnitudeWidth bits)
  nextAccumulatorMagnitude := accumulatorWide.asUInt
  when(nextAccumulatorNegative) {
    nextAccumulatorMagnitude := (-accumulatorWide).asUInt
  }

  val factorValid = RegNext(factorProductValid) init False
  val factorLane = RegNextWhen(factorProductLane, factorProductValid) init 0
  val deltaFactorReg = Reg(UInt(cfg.factorWidth bits)) init 0
  val accumulatorMagnitudeReg = Reg(UInt(cfg.magnitudeWidth bits)) init 0
  val accumulatorNegativeReg = RegInit(False)
  val factorResidualReg = Reg(SInt(cfg.valueWidth bits)) init 0
  val factorLaneActiveReg = RegInit(False)
  when(factorProductValid) {
    deltaFactorReg := nextDeltaFactor
    accumulatorMagnitudeReg := nextAccumulatorMagnitude
    accumulatorNegativeReg := nextAccumulatorNegative
    factorResidualReg := factorProductResidual
    factorLaneActiveReg := factorProductLaneActive
  }

  /* Stage 3: one shared U33 x U32 projection multiplier (four DSP48E2s). */
  val scaledMagnitude = (
    accumulatorMagnitudeReg * deltaFactorReg
  ).resize(cfg.scaledProductWidth)
  scaledMagnitude.addAttribute("use_dsp", "yes")
  val scaledMagnitudeReg = Reg(UInt(cfg.scaledProductWidth bits)) init 0
  val scaledProductValid = RegNext(factorValid) init False
  val scaledProductLane = RegNextWhen(factorLane, factorValid) init 0
  val scaledProductNegative = RegInit(False)
  val scaledProductResidual = Reg(SInt(cfg.valueWidth bits)) init 0
  val scaledProductLaneActive = RegInit(False)
  when(factorValid) {
    scaledMagnitudeReg := scaledMagnitude
    scaledProductNegative := accumulatorNegativeReg
    scaledProductResidual := factorResidualReg
    scaledProductLaneActive := factorLaneActiveReg
  }

  /* Stage 4: saturate the signed projection before the optional residual. */
  val saturatedDelta = SInt(cfg.valueWidth bits)
  saturatedDelta := 0
  when(scaledProductNegative) {
    when(scaledMagnitudeReg >=
        U(negativeMagnitudeMax, cfg.scaledProductWidth bits)) {
      saturatedDelta := S(-negativeMagnitudeMax, cfg.valueWidth bits)
    } otherwise {
      saturatedDelta := (-scaledMagnitudeReg
        .resize(cfg.valueWidth + 1).asSInt).resize(cfg.valueWidth)
    }
  } otherwise {
    when(scaledMagnitudeReg >
        U(positiveMax, cfg.scaledProductWidth bits)) {
      saturatedDelta := S(positiveMax, cfg.valueWidth bits)
    } otherwise {
      saturatedDelta := scaledMagnitudeReg
        .resize(cfg.valueWidth).asSInt
    }
  }

  val deltaValid = RegNext(scaledProductValid) init False
  val deltaLane = RegNextWhen(scaledProductLane, scaledProductValid) init 0
  val deltaReg = Reg(SInt(cfg.valueWidth bits)) init 0
  val deltaResidualReg = Reg(SInt(cfg.valueWidth bits)) init 0
  val deltaLaneActiveReg = RegInit(False)
  when(scaledProductValid) {
    deltaReg := Mux(scaledProductLaneActive, saturatedDelta,
      S(0, cfg.valueWidth bits))
    deltaResidualReg := scaledProductResidual
    deltaLaneActiveReg := scaledProductLaneActive
  }

  /* Stage 5: residual saturation and Vec4 reassembly. */
  val residualSum = deltaResidualReg.resize(cfg.valueWidth + 1) +
    deltaReg.resize(cfg.valueWidth + 1)
  val saturatedResidualSum = SInt(cfg.valueWidth bits)
  when(residualSum > S(positiveMax, cfg.valueWidth + 1 bits)) {
    saturatedResidualSum := S(positiveMax, cfg.valueWidth bits)
  } elsewhen (residualSum <
      S(-negativeMagnitudeMax, cfg.valueWidth + 1 bits)) {
    saturatedResidualSum := S(-negativeMagnitudeMax, cfg.valueWidth bits)
  } otherwise {
    saturatedResidualSum := residualSum.resize(cfg.valueWidth)
  }

  val outputValid = RegInit(False)
  val outputDelta = Vec(Reg(SInt(cfg.valueWidth bits)) init 0, cfg.lanes)
  val outputValue = Vec(Reg(SInt(cfg.valueWidth bits)) init 0, cfg.lanes)

  io.input.ready := !issuing && !operandValid && !factorProductValid &&
    !factorValid && !scaledProductValid && !deltaValid && !outputValid
  when(io.input.fire) {
    held := io.input.payload
    issuing := True
    issueLane := 0
    for (lane <- 0 until cfg.lanes) {
      outputDelta(lane) := 0
      outputValue(lane) := 0
    }
  }

  when(issuing) {
    when(issueLane === cfg.lanes - 1) {
      issuing := False
    } otherwise {
      issueLane := issueLane + 1
    }
  }

  when(deltaValid) {
    outputDelta(deltaLane) := deltaReg
    outputValue(deltaLane) := S(0, cfg.valueWidth bits)
    when(deltaLaneActiveReg) {
      outputValue(deltaLane) := Mux(
        held.residualEnable,
        saturatedResidualSum,
        deltaReg)
    }
    when(deltaLane === cfg.lanes - 1) {
      outputValid := True
    }
  }

  io.output.valid := outputValid
  io.output.scaledDelta := outputDelta
  io.output.value := outputValue
  io.output.laneMask := held.laneMask
  io.output.first := held.first
  io.output.last := held.last
  when(io.output.fire) { outputValid := False }
}

object NativeScaleResidualMath {
  private val PositiveMax = BigInt("7fffffff", 16)
  private val NegativeMin = -(BigInt(1) << 31)

  def saturateI32(value: BigInt): Int =
    value.max(NegativeMin).min(PositiveMax).toInt

  def factorQ16(activationFactor: Long, weightScale: Long): Long = {
    val product = BigInt(activationFactor & 0xffffffffL) *
      BigInt(weightScale & 0xffffffffL)
    ((product + (BigInt(1) << 15)) >> 16)
      .min(BigInt("ffffffff", 16)).toLong
  }

  def apply(accumulator: Int, activationFactor: Long, weightScale: Long,
            residual: Int, residualEnable: Boolean): (Int, Int) = {
    val factor = BigInt(factorQ16(activationFactor, weightScale))
    val delta = saturateI32(BigInt(accumulator) * factor)
    val value = if (residualEnable) {
      saturateI32(BigInt(residual) + delta)
    } else delta
    (delta, value)
  }
}
