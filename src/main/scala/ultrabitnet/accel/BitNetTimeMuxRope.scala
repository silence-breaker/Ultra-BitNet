package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Production RoPE implementation for XCZU3EG.
  *
  * The four-token reference implementation evaluates sixteen 32x16 products
  * in parallel.  Because one UltraScale+ DSP48E2 is only 27x18, Vivado maps
  * that expression to 32 DSPs.  RoPE is not on the matrix/DDR roofline, so the
  * production datapath holds one TP4 pair and visits one token lane per cycle.
  * Four 32x16 products are shared by all lanes (eight DSP48E2 after mapping).
  *
  * A complete input beat therefore takes four issue cycles plus the registered
  * result stage.  Q and K reuse this same block; V bypasses it.  All metadata
  * and inactive-lane zeros are held under arbitrary output backpressure.
  */
class BitNetTimeMuxRope(
    cfg: NativeRopeConfig = NativeRopeConfig()
) extends Component {
  require(cfg.tokenLanes == 2 || cfg.tokenLanes == 4,
    "the production scheduler must be TP2 or TP4")

  val io = new Bundle {
    val input = slave(Stream(NativeRopeBeat(cfg)))
    val output = master(Stream(NativeRopeResult(cfg)))
    val busy = out Bool()
    val acceptedBeats = out UInt(32 bits)
    val emittedBeats = out UInt(32 bits)
  }

  val held = Reg(NativeRopeBeat(cfg)) init (NativeRopeBeat(cfg).getZero)
  val issuing = RegInit(False)
  val issueLane = Reg(UInt(log2Up(cfg.tokenLanes) bits)) init 0

  // The lane mux/token mask, multiplier and add/sub each terminate in a
  // register.  Without these boundaries Vivado placed mask -> 4:1 mux ->
  // two-DSP 32x16 multiply -> 50-bit carry chain in one 266 MHz cycle.
  val operandValid = RegNext(issuing) init False
  val operandLane = RegNextWhen(issueLane, issuing) init 0
  val productValid = RegNext(operandValid) init False
  val productLane = RegNextWhen(operandLane, operandValid) init 0
  val rawValid = RegNext(productValid) init False
  val rawLane = RegNextWhen(productLane, productValid) init 0
  val rawLower = Reg(SInt(cfg.rawWidth bits)) init 0
  val rawUpper = Reg(SInt(cfg.rawWidth bits)) init 0

  val outputValid = RegInit(False)
  val outputLower = Vec(Reg(SInt(cfg.dataWidth bits)) init 0, cfg.tokenLanes)
  val outputUpper = Vec(Reg(SInt(cfg.dataWidth bits)) init 0, cfg.tokenLanes)

  io.input.ready := !issuing && !operandValid && !productValid &&
    !rawValid && !outputValid
  when(io.input.fire) {
    held := io.input.payload
    issuing := True
    issueLane := 0
    for (lane <- 0 until cfg.tokenLanes) {
      outputLower(lane) := 0
      outputUpper(lane) := 0
    }
  }

  val operandX0 = Reg(SInt(cfg.dataWidth bits)) init 0
  val operandX1 = Reg(SInt(cfg.dataWidth bits)) init 0
  val operandCosine = Reg(SInt(cfg.coefficientWidth bits)) init 0
  val operandSine = Reg(SInt(cfg.coefficientWidth bits)) init 0
  when(issuing) {
    operandX0 := Mux(held.tokenMask(issueLane), held.lower(issueLane),
      S(0, cfg.dataWidth bits))
    operandX1 := Mux(held.tokenMask(issueLane), held.upper(issueLane),
      S(0, cfg.dataWidth bits))
    operandCosine := held.cosine(issueLane)
    operandSine := held.sine(issueLane)
  }

  // Four physical 32x16 multipliers.  Vivado implements each with two DSP48E2
  // slices, so this block has a hard eight-DSP budget.
  val x0Cos = (operandX0 * operandCosine).resize(cfg.multiplyWidth)
  val x1Sin = (operandX1 * operandSine).resize(cfg.multiplyWidth)
  val x1Cos = (operandX1 * operandCosine).resize(cfg.multiplyWidth)
  val x0Sin = (operandX0 * operandSine).resize(cfg.multiplyWidth)
  Seq(x0Cos, x1Sin, x1Cos, x0Sin).foreach(
    _.addAttribute("use_dsp", "yes"))

  val productX0Cos = Reg(SInt(cfg.multiplyWidth bits)) init 0
  val productX1Sin = Reg(SInt(cfg.multiplyWidth bits)) init 0
  val productX1Cos = Reg(SInt(cfg.multiplyWidth bits)) init 0
  val productX0Sin = Reg(SInt(cfg.multiplyWidth bits)) init 0
  when(operandValid) {
    productX0Cos := x0Cos
    productX1Sin := x1Sin
    productX1Cos := x1Cos
    productX0Sin := x0Sin
  }

  when(productValid) {
    rawLower := (productX0Cos.resize(cfg.rawWidth) -
      productX1Sin.resize(cfg.rawWidth)).resize(cfg.rawWidth)
    rawUpper := (productX1Cos.resize(cfg.rawWidth) +
      productX0Sin.resize(cfg.rawWidth)).resize(cfg.rawWidth)
  }

  when(issuing) {
    when(issueLane === cfg.tokenLanes - 1) {
      issuing := False
    } otherwise {
      issueLane := issueLane + 1
    }
  }

  private def roundQ15(value: SInt): SInt = {
    val wide = value.resize(cfg.rawWidth + 1)
    val magnitude = UInt((cfg.rawWidth + 1) bits)
    magnitude := Mux(wide.msb, (-wide).asUInt, wide.asUInt)
    val roundedMagnitude = ((magnitude +
      U(BigInt(1) << (cfg.coefficientFractionBits - 1), magnitude.getWidth bits))
      >> cfg.coefficientFractionBits).resize(magnitude.getWidth)
    val rounded = SInt((cfg.rawWidth + 1) bits)
    rounded := Mux(wide.msb, -roundedMagnitude.asSInt,
      roundedMagnitude.asSInt)
    rounded
  }

  private def saturateI32(value: SInt): SInt = {
    val result = SInt(cfg.dataWidth bits)
    when(value > S(BigInt(Int.MaxValue), value.getWidth bits)) {
      result := S(Int.MaxValue, cfg.dataWidth bits)
    } elsewhen (value < S(BigInt(Int.MinValue), value.getWidth bits)) {
      result := S(Int.MinValue, cfg.dataWidth bits)
    } otherwise {
      result := value.resize(cfg.dataWidth)
    }
    result
  }

  when(rawValid) {
    outputLower(rawLane) := saturateI32(roundQ15(rawLower))
    outputUpper(rawLane) := saturateI32(roundQ15(rawUpper))
    when(rawLane === cfg.tokenLanes - 1) {
      outputValid := True
    }
  }

  io.output.valid := outputValid
  io.output.lower := outputLower
  io.output.upper := outputUpper
  io.output.tokenMask := held.tokenMask
  io.output.head := held.head
  io.output.pair := held.pair
  io.output.first := held.first
  io.output.last := held.last
  when(io.output.fire) { outputValid := False }

  val accepted = Reg(UInt(32 bits)) init 0
  val emitted = Reg(UInt(32 bits)) init 0
  when(io.input.fire) { accepted := accepted + 1 }
  when(io.output.fire) { emitted := emitted + 1 }
  io.acceptedBeats := accepted
  io.emittedBeats := emitted
  io.busy := issuing || operandValid || productValid || rawValid || outputValid
}

object GenerateBitNetTimeMuxRope extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-time-mux-rope")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetTimeMuxRope())
}
