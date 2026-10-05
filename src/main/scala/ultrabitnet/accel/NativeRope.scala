package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** Split-half RoPE stream used directly between QKV projection and KV append. */
case class NativeRopeConfig(
    tokenLanes: Int = 4,
    headCount: Int = 20,
    headSize: Int = 128,
    dataWidth: Int = 32,
    coefficientWidth: Int = 16,
    coefficientFractionBits: Int = 15
) {
  require(tokenLanes > 0 && isPow2(tokenLanes))
  require(headCount > 0)
  require(headSize > 0 && headSize % 2 == 0)
  require(dataWidth == 32)
  require(coefficientWidth == 16 && coefficientFractionBits == 15)

  val headIndexWidth: Int = scala.math.max(1, log2Up(headCount))
  val pairIndexWidth: Int = scala.math.max(1, log2Up(headSize / 2))
  val multiplyWidth: Int = dataWidth + coefficientWidth
  val rawWidth: Int = multiplyWidth + 2
}

case class NativeRopeBeat(cfg: NativeRopeConfig) extends Bundle {
  val lower = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val upper = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  /** Each prefill lane has a different position and therefore coefficients. */
  val cosine = Vec(SInt(cfg.coefficientWidth bits), cfg.tokenLanes)
  val sine = Vec(SInt(cfg.coefficientWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val head = UInt(cfg.headIndexWidth bits)
  val pair = UInt(cfg.pairIndexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class NativeRopeRaw(cfg: NativeRopeConfig) extends Bundle {
  val lower = Vec(SInt(cfg.rawWidth bits), cfg.tokenLanes)
  val upper = Vec(SInt(cfg.rawWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val head = UInt(cfg.headIndexWidth bits)
  val pair = UInt(cfg.pairIndexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class NativeRopeResult(cfg: NativeRopeConfig) extends Bundle {
  val lower = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val upper = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val head = UInt(cfg.headIndexWidth bits)
  val pair = UInt(cfg.pairIndexWidth bits)
  val first = Bool()
  val last = Bool()
}

/**
  * Four multipliers per active physical SPU lane implement:
  *
  *   y0 = round_sat((x0*cos - x1*sin) / 2^15)
  *   y1 = round_sat((x1*cos + x0*sin) / 2^15)
  *
  * Multiplication and round/saturate are separated by registered Stream
  * boundaries. There is no scratch/uop/packet round trip.
  */
class NativeRope(cfg: NativeRopeConfig = NativeRopeConfig()) extends Component {
  val io = new Bundle {
    val input = slave(Stream(NativeRopeBeat(cfg)))
    val output = master(Stream(NativeRopeResult(cfg)))
    val acceptedBeats = out UInt(32 bits)
    val emittedBeats = out UInt(32 bits)
  }

  val raw = Stream(NativeRopeRaw(cfg))
  raw.valid := io.input.valid
  io.input.ready := raw.ready
  raw.tokenMask := io.input.tokenMask
  raw.head := io.input.head
  raw.pair := io.input.pair
  raw.first := io.input.first
  raw.last := io.input.last

  for (lane <- 0 until cfg.tokenLanes) {
    val x0Cos =
      (io.input.lower(lane) * io.input.cosine(lane)).resize(cfg.multiplyWidth)
    val x1Sin =
      (io.input.upper(lane) * io.input.sine(lane)).resize(cfg.multiplyWidth)
    val x1Cos =
      (io.input.upper(lane) * io.input.cosine(lane)).resize(cfg.multiplyWidth)
    val x0Sin =
      (io.input.lower(lane) * io.input.sine(lane)).resize(cfg.multiplyWidth)
    x0Cos.addAttribute("use_dsp", "yes")
    x1Sin.addAttribute("use_dsp", "yes")
    x1Cos.addAttribute("use_dsp", "yes")
    x0Sin.addAttribute("use_dsp", "yes")
    raw.lower(lane) :=
      (x0Cos.resize(cfg.rawWidth) - x1Sin.resize(cfg.rawWidth)).resize(cfg.rawWidth)
    raw.upper(lane) :=
      (x1Cos.resize(cfg.rawWidth) + x0Sin.resize(cfg.rawWidth)).resize(cfg.rawWidth)
  }

  val multiplied = raw.m2sPipe()
  val rounded = Stream(NativeRopeResult(cfg))
  rounded.valid := multiplied.valid
  multiplied.ready := rounded.ready
  rounded.tokenMask := multiplied.tokenMask
  rounded.head := multiplied.head
  rounded.pair := multiplied.pair
  rounded.first := multiplied.first
  rounded.last := multiplied.last

  def roundQ15(value: SInt): SInt = {
    val biased = SInt((cfg.rawWidth + 1) bits)
    when(value.msb) {
      biased := value.resize(cfg.rawWidth + 1) +
        S((BigInt(1) << (cfg.coefficientFractionBits - 1)) - 1,
          cfg.rawWidth + 1 bits)
    } otherwise {
      biased := value.resize(cfg.rawWidth + 1) +
        S(BigInt(1) << (cfg.coefficientFractionBits - 1), cfg.rawWidth + 1 bits)
    }
    (biased >> cfg.coefficientFractionBits).resize(cfg.rawWidth + 1)
  }

  def saturateI32(value: SInt): SInt = {
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

  for (lane <- 0 until cfg.tokenLanes) {
    rounded.lower(lane) := 0
    rounded.upper(lane) := 0
    when(multiplied.tokenMask(lane)) {
      rounded.lower(lane) := saturateI32(roundQ15(multiplied.lower(lane)))
      rounded.upper(lane) := saturateI32(roundQ15(multiplied.upper(lane)))
    }
  }
  io.output << rounded.m2sPipe()

  val accepted = Reg(UInt(32 bits)) init 0
  val emitted = Reg(UInt(32 bits)) init 0
  when(io.input.fire) { accepted := accepted + 1 }
  when(io.output.fire) { emitted := emitted + 1 }
  io.acceptedBeats := accepted
  io.emittedBeats := emitted
}

object NativeRopeMath {
  private val MinI32 = BigInt(Int.MinValue)
  private val MaxI32 = BigInt(Int.MaxValue)

  def roundQ15(value: BigInt): BigInt = {
    val bias = if (value < 0) (BigInt(1) << 14) - 1 else BigInt(1) << 14
    (value + bias) >> 15
  }

  def saturateI32(value: BigInt): Int =
    value.max(MinI32).min(MaxI32).toInt

  def rotate(lower: Int, upper: Int, cosine: Int, sine: Int): (Int, Int) = {
    val lo = BigInt(lower) * cosine - BigInt(upper) * sine
    val hi = BigInt(upper) * cosine + BigInt(lower) * sine
    (saturateI32(roundQ15(lo)), saturateI32(roundQ15(hi)))
  }
}

object GenerateNativeRope extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-rope")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new NativeRope(NativeRopeConfig()))
}
