package ultrabitnet.accel

import spinal.core._

/**
  * Pin widths and packed-lane geometry of the DSP48E2 Map0 pre-adder codec.
  *
  * DSP48E2 exposes a 30-bit A port and a 27-bit D/pre-adder datapath.  The
  * packed ternary dot product keeps the low output row in D and places the
  * high output row at bit `laneSeparation` of A.  Both values are already
  * sign-extended at the primitive boundary, so selecting AMULTSEL="AD" makes
  * the DSP pre-adder reproduce the old packed Map0 operand exactly.
  */
case class BitNetTernaryMap0DspOperandCodecConfig(
    laneSeparation: Int = 18,
    dspAWidth: Int = 30,
    dspDWidth: Int = 27
) {
  require(dspAWidth == 30, "DSP48E2 A must be 30 bits")
  require(dspDWidth == 27, "DSP48E2 D and A+D pre-adder must be 27 bits")
  require(laneSeparation >= 2)
  require(laneSeparation < dspDWidth - 1,
    "the shifted signed high row must fit in the 27-bit pre-adder")

  val highSignWidth: Int = dspAWidth - laneSeparation
}

case class BitNetTernaryMap0DspOperands(
    cfg: BitNetTernaryMap0DspOperandCodecConfig
) extends Bundle {
  val a = Bits(cfg.dspAWidth bits)
  val d = Bits(cfg.dspDWidth bits)
}

/**
  * BitNet Map0-to-DSP48E2 A/D operand codec for one matrix site.
  *
  * `raw(1 downto 0)` is the low output row and `raw(3 downto 2)` is the high
  * output row.  Map0 is 00->0, 01->-1, 10->0, 11->+1.  Bits 7:4 are ignored.
  *
  * The implementation intentionally contains no arithmetic operator.  Each
  * ternary value is the sign extension of the two-bit word
  * `{isNegative,isNonZero}`; the high row shift is a static concatenation.
  * The only non-wire logic is the two negative-code Boolean terms.
  */
class BitNetTernaryMap0DspOperandCodec(
    cfg: BitNetTernaryMap0DspOperandCodecConfig =
      BitNetTernaryMap0DspOperandCodecConfig()
) extends Component {
  val io = new Bundle {
    val raw = in Bits(8 bits)
    val operands = out(BitNetTernaryMap0DspOperands(cfg))
  }

  private def decodeMap0Bits(code: Bits, width: Int): Bits = {
    require(width >= 2)
    val isNegative = code(0) && !code(1)
    // 00 -> 00, 01 -> 11 (-1), 10 -> 00, 11 -> 01 (+1).
    val compact = (isNegative ## code(0)).asSInt
    compact.resize(width).asBits
  }

  val low = decodeMap0Bits(io.raw(1 downto 0), cfg.dspDWidth)
  val high = decodeMap0Bits(
    io.raw(3 downto 2),
    cfg.highSignWidth
  )

  io.operands.a := high ## B(0, cfg.laneSeparation bits)
  io.operands.d := low
}

/** A packet-free 128-site elaboration target for isolated resource probes. */
class BitNetTernaryMap0DspOperandCodecArray(
    siteCount: Int = 128,
    cfg: BitNetTernaryMap0DspOperandCodecConfig =
      BitNetTernaryMap0DspOperandCodecConfig()
) extends Component {
  require(siteCount > 0)

  val io = new Bundle {
    val raw = in Vec(Bits(8 bits), siteCount)
    val a = out Vec(Bits(cfg.dspAWidth bits), siteCount)
    val d = out Vec(Bits(cfg.dspDWidth bits), siteCount)
  }

  for (site <- 0 until siteCount) {
    val codec = new BitNetTernaryMap0DspOperandCodec(cfg)
    codec.io.raw := io.raw(site)
    io.a(site) := codec.io.operands.a
    io.d(site) := codec.io.operands.d
  }
}

/** Pure-Scala bit-exact oracle shared by unit and integration tests. */
object BitNetTernaryMap0DspOperandCodecOracle {
  def decodeMap0(code: Int): Int = code & 3 match {
    case 1 => -1
    case 3 => 1
    case _ => 0
  }

  def lowCode(raw: Int): Int = raw & 3

  def highCode(raw: Int): Int = (raw >> 2) & 3

  /** Mathematical value produced by the former fabric packMap0Pair lookup. */
  def oldPackMap0Pair(raw: Int, laneSeparation: Int = 18): BigInt =
    BigInt(decodeMap0(lowCode(raw))) +
      (BigInt(decodeMap0(highCode(raw))) << laneSeparation)

  /** Mathematical value driven onto DSP A before its pre-adder. */
  def aSigned(raw: Int, laneSeparation: Int = 18): BigInt =
    BigInt(decodeMap0(highCode(raw))) << laneSeparation

  /** Mathematical value driven onto DSP D before its pre-adder. */
  def dSigned(raw: Int): BigInt = BigInt(decodeMap0(lowCode(raw)))

  def toTwos(value: BigInt, width: Int): BigInt = {
    require(width > 0)
    value & ((BigInt(1) << width) - 1)
  }

  def fromTwos(bits: BigInt, width: Int): BigInt = {
    require(width > 0)
    val masked = toTwos(bits, width)
    val sign = BigInt(1) << (width - 1)
    if ((masked & sign) == 0) masked else masked - (BigInt(1) << width)
  }

  def aBits(
      raw: Int,
      cfg: BitNetTernaryMap0DspOperandCodecConfig =
        BitNetTernaryMap0DspOperandCodecConfig()
  ): BigInt = toTwos(aSigned(raw, cfg.laneSeparation), cfg.dspAWidth)

  def dBits(
      raw: Int,
      cfg: BitNetTernaryMap0DspOperandCodecConfig =
        BitNetTernaryMap0DspOperandCodecConfig()
  ): BigInt = toTwos(dSigned(raw), cfg.dspDWidth)

  /** Signed 27-bit DSP pre-adder result; all 16 Map0 pairs are overflow-free. */
  def preadderSigned(
      raw: Int,
      cfg: BitNetTernaryMap0DspOperandCodecConfig =
        BitNetTernaryMap0DspOperandCodecConfig()
  ): BigInt = {
    val aAtPreadder = fromTwos(aBits(raw, cfg), cfg.dspDWidth)
    val dAtPreadder = fromTwos(dBits(raw, cfg), cfg.dspDWidth)
    fromTwos(
      toTwos(aAtPreadder + dAtPreadder, cfg.dspDWidth),
      cfg.dspDWidth
    )
  }
}

object GenerateBitNetTernaryMap0DspOperandCodecArray extends App {
  val target = sys.env.getOrElse(
    "BITNET_MAP0_CODEC_RTL_DIR",
    "target/generated-bitnet-map0-dsp-codec-array"
  )
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetTernaryMap0DspOperandCodecArray())
}
