package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Production geometry of the shared, externally replayed dynamic-i8 SPU.
  *
  * Unlike [[NativeDynamicI8Quantizer]], this block never stores the activation
  * vector.  The first pass only accumulates four maxima.  After the shared
  * divider bank has produced the reciprocal and dequantization factor, an
  * explicit request asks the PL activation fabric to replay the same image.
  * Thus hidden and FFN vectors share one circuit without allocating a private
  * 4 x 6912 x 32-bit RAM beside the resident activation stores.
  */
case class BitNetSharedDynamicI8QuantizerConfig(
    tokenLanes: Int = BitNetConfig.Production.tokenParallelism,
    maxVectorLength: Int = 6912,
    inputWidth: Int = 32,
    inputFractionBits: Int = 16,
    reciprocalFractionBits: Int = 33,
    factorWidth: Int = 32,
    tagWidth: Int = 12,
    epochWidth: Int = 8
) {
  require(tokenLanes == 2 || tokenLanes == 4,
    "the production quantizer must be TP2 or TP4")
  require(maxVectorLength >= 6912,
    "one shared unit must cover both hidden and FFN activations")
  require(inputWidth == 32 && inputFractionBits == 16,
    "the resident activation boundary is signed Q16.16")
  require(reciprocalFractionBits >= inputWidth + 1,
    "the reciprocal proposal must be within one exact i8 code")
  require(factorWidth >= inputWidth)
  require(tagWidth >= 1 && epochWidth >= 1)

  val lengthWidth: Int = log2Up(maxVectorLength + 1)
  val indexWidth: Int = log2Up(maxVectorLength)
  val magnitudeWidth: Int = inputWidth + 1
  // 127 * 2^33 occupies exactly 40 bits; the extra MSB is retained by the
  // restoring implementation and matches NativeDynamicI8Quantizer.
  val dividerWidth: Int = reciprocalFractionBits + 8
  val dividerCountWidth: Int = log2Up(dividerWidth)
  val reciprocalWidth: Int = dividerWidth
  val reciprocalProductWidth: Int = magnitudeWidth + reciprocalWidth
  val exactProductWidth: Int = magnitudeWidth + 9
}

case class BitNetSharedDynamicQuantCommand(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val vectorLength = UInt(cfg.lengthWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
}

/** First-pass activation beat.  Ownership and framing are checked on every
  * accepted beat; values belonging to inactive token lanes are ignored. */
case class BitNetSharedDynamicQuantMeasureBeat(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val value = Vec(SInt(cfg.inputWidth bits), cfg.tokenLanes)
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

/** Stable request issued after all four reciprocal/factor divisions finish. */
case class BitNetSharedDynamicQuantReplayRequest(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val vectorLength = UInt(cfg.lengthWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
}

/** Externally replayed Q16.16 activation beat.
  *
  * The explicit index, ownership and framing fields make stale-buffer or
  * wrong-image replays fail closed instead of silently quantizing them.
  */
case class BitNetSharedDynamicQuantReplayBeat(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val value = Vec(SInt(cfg.inputWidth bits), cfg.tokenLanes)
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantResultBeat(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val value = Vec(SInt(8 bits), cfg.tokenLanes)
  /** Unsigned Q16.16 max(abs(x))/127, repeated for a packet-free join. */
  val dequantFactorQ16 = Vec(UInt(cfg.factorWidth bits), cfg.tokenLanes)
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantDone(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val vectorLength = UInt(cfg.lengthWidth bits)
  val emittedFeatures = UInt(cfg.lengthWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/** Internal elastic replay stages.  They contain one Vec4 beat each, never a
  * vector-length-dependent store. */
case class BitNetSharedDynamicQuantCaptured(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val magnitude = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  val negative = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantReciprocalPartials(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val p00 = Vec(UInt(34 bits), cfg.tokenLanes) // 17 x 17
  val p10 = Vec(UInt(33 bits), cfg.tokenLanes) // 16 x 17
  val p01 = Vec(UInt(41 bits), cfg.tokenLanes) // 17 x 24
  val p11 = Vec(UInt(40 bits), cfg.tokenLanes) // 16 x 24
  val twiceNumerator = Vec(
    UInt(cfg.exactProductWidth bits), cfg.tokenLanes)
  val negative = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantReciprocalPairs(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val low = Vec(UInt(50 bits), cfg.tokenLanes)
  val high = Vec(UInt(cfg.reciprocalProductWidth bits), cfg.tokenLanes)
  val twiceNumerator = Vec(
    UInt(cfg.exactProductWidth bits), cfg.tokenLanes)
  val negative = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantReciprocalProduct(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val product = Vec(
    UInt(cfg.reciprocalProductWidth bits), cfg.tokenLanes)
  val twiceNumerator = Vec(
    UInt(cfg.exactProductWidth bits), cfg.tokenLanes)
  val negative = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantProposed(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val provisional = Vec(UInt(8 bits), cfg.tokenLanes)
  val twiceNumerator = Vec(
    UInt(cfg.exactProductWidth bits), cfg.tokenLanes)
  val negative = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantCenterPartials(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val low = Vec(UInt(25 bits), cfg.tokenLanes) // 17 x 8
  val high = Vec(UInt(24 bits), cfg.tokenLanes) // 16 x 8
  val provisional = Vec(UInt(8 bits), cfg.tokenLanes)
  val twiceNumerator = Vec(
    UInt(cfg.exactProductWidth bits), cfg.tokenLanes)
  val negative = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantCenterProduct(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val product = Vec(
    UInt(cfg.exactProductWidth bits), cfg.tokenLanes)
  val provisional = Vec(UInt(8 bits), cfg.tokenLanes)
  val twiceNumerator = Vec(
    UInt(cfg.exactProductWidth bits), cfg.tokenLanes)
  val negative = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedDynamicQuantCorrected(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val magnitude = Vec(UInt(8 bits), cfg.tokenLanes)
  val negative = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

/** Stable software/RTL diagnostic allocation. */
object BitNetSharedDynamicQuantFault {
  val None: Int = 0x00
  val InvalidCommand: Int = 0xd0
  val MeasureMetadata: Int = 0xd1
  val MeasureFraming: Int = 0xd2
  val ReplayMetadata: Int = 0xd3
  val ReplayFraming: Int = 0xd4
}

/**
  * RAM-free, two-pass dynamic Q16.16 -> i8 quantizer for resident inference.
  *
  * Arithmetic is bit-exact with [[NativeDynamicI8QuantizerMath]]:
  *
  *   q = sat_i8(sign(x) * floor(abs(x)*127/maxAbs + 0.5))
  *   factorQ16 = maxAbs == 0 ? 65536 : floor((maxAbs + 63)/127)
  *
  * The Q33 reciprocal only proposes an i8 code.  Exact integer threshold
  * comparisons correct the proposal by at most one, making finite reciprocal
  * precision invisible at the interface (including ties away from zero and
  * Int.MinValue).  Four fixed-latency restoring dividers are reused for the
  * reciprocal and factor phases.  There is no `/`, `%`, vector Mem, legacy
  * packet/uop route, or host callback in this component.
  *
  * A malformed command/measure/replay emits no data after detection.  The
  * typed failure completion is held until accepted, after which the unit is
  * immediately reusable; the diagnostic sticky bit intentionally remains set.
  */
class BitNetSharedDynamicI8Quantizer(
    cfg: BitNetSharedDynamicI8QuantizerConfig =
      BitNetSharedDynamicI8QuantizerConfig()) extends Component {
  val io = new Bundle {
    val command = slave(Stream(BitNetSharedDynamicQuantCommand(cfg)))
    val measureInput = slave(Stream(
      BitNetSharedDynamicQuantMeasureBeat(cfg)))
    val replayRequest = master(Stream(
      BitNetSharedDynamicQuantReplayRequest(cfg)))
    val replayInput = slave(Stream(
      BitNetSharedDynamicQuantReplayBeat(cfg)))
    val output = master(Stream(BitNetSharedDynamicQuantResultBeat(cfg)))
    val done = master(Stream(BitNetSharedDynamicQuantDone(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val completedCommands = out UInt(32 bits)
    /** Exposed for deterministic integration diagnostics. */
    val dequantFactorQ16 = out Vec(
      UInt(cfg.factorWidth bits), cfg.tokenLanes)
  }

  object State extends SpinalEnum {
    val Idle, Measure, ScalePrepare, ScaleDivide, FactorPrepare,
        FactorDivide, RequestReplay, Replay, ReplayDrain,
        Complete = newElement()
  }
  val state = Reg(State()) init State.Idle

  val tagReg = Reg(UInt(cfg.tagWidth bits)) init 0
  val epochReg = Reg(UInt(cfg.epochWidth bits)) init 0
  val lengthReg = Reg(UInt(cfg.lengthWidth bits)) init 0
  val tokenMaskReg = Reg(Bits(cfg.tokenLanes bits)) init 0
  val measureIndex = Reg(UInt(cfg.indexWidth bits)) init 0
  val replayIndex = Reg(UInt(cfg.indexWidth bits)) init 0
  val emittedFeatures = Reg(UInt(cfg.lengthWidth bits)) init 0

  val maxAbs = Vec(
    Reg(UInt(cfg.magnitudeWidth bits)) init 0, cfg.tokenLanes)
  val reciprocal = Vec(
    Reg(UInt(cfg.reciprocalWidth bits)) init 0, cfg.tokenLanes)
  val factor = Vec(
    Reg(UInt(cfg.factorWidth bits)) init 0, cfg.tokenLanes)
  io.dequantFactorQ16 := factor

  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)
  val completedCounter = Reg(UInt(32 bits)) init 0
  io.protocolFaultSticky := stickyFault
  io.completedCommands := completedCounter

  def magnitudeOf(value: SInt): UInt = {
    val widened = value.resize(cfg.magnitudeWidth)
    val magnitude = UInt(cfg.magnitudeWidth bits)
    when(widened.msb) {
      magnitude := (-widened).asUInt
    } otherwise {
      magnitude := widened.asUInt
    }
    magnitude
  }

  val measureMagnitude = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  val replayMagnitude = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    measureMagnitude(lane) := magnitudeOf(io.measureInput.payload.value(lane))
    replayMagnitude(lane) := magnitudeOf(io.replayInput.payload.value(lane))
  }

  // Exactly four physical restoring dividers.  The same register bank first
  // divides 127*2^33 by maxAbs and then (maxAbs+63) by 127.
  val dividerDividend = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init 0, cfg.tokenLanes)
  val dividerDivisor = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init 0, cfg.tokenLanes)
  val dividerRemainder = Vec(
    Reg(UInt((cfg.dividerWidth + 1) bits)) init 0, cfg.tokenLanes)
  val dividerQuotient = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init 0, cfg.tokenLanes)
  val dividerIteration =
    Reg(UInt(cfg.dividerCountWidth bits)) init 0

  val dividerDividendNext = Vec(UInt(cfg.dividerWidth bits), cfg.tokenLanes)
  val dividerRemainderNext =
    Vec(UInt((cfg.dividerWidth + 1) bits), cfg.tokenLanes)
  val dividerQuotientNext = Vec(UInt(cfg.dividerWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val shiftedRemainder = UInt((cfg.dividerWidth + 1) bits)
    shiftedRemainder := (
      dividerRemainder(lane)(cfg.dividerWidth - 1 downto 0) ##
        dividerDividend(lane).msb).asUInt
    val canSubtract = shiftedRemainder >=
      dividerDivisor(lane).resize(cfg.dividerWidth + 1)
    dividerRemainderNext(lane) := shiftedRemainder
    when(canSubtract) {
      dividerRemainderNext(lane) :=
        (shiftedRemainder -
          dividerDivisor(lane).resize(cfg.dividerWidth + 1))
          .resize(cfg.dividerWidth + 1)
    }
    dividerDividendNext(lane) :=
      (dividerDividend(lane) |<< 1).resize(cfg.dividerWidth)
    val shiftedQuotient =
      (dividerQuotient(lane) |<< 1).resize(cfg.dividerWidth)
    dividerQuotientNext(lane) := shiftedQuotient
    when(canSubtract) {
      dividerQuotientNext(lane) :=
        shiftedQuotient | U(1, cfg.dividerWidth bits)
    }
  }

  val expectedMeasureFirst = measureIndex === 0
  val expectedMeasureLast =
    measureIndex.resize(cfg.lengthWidth) === (lengthReg - 1)
  val measureMetadataLegal =
    io.measureInput.payload.tag === tagReg &&
      io.measureInput.payload.epoch === epochReg &&
      io.measureInput.payload.featureIndex === measureIndex &&
      io.measureInput.payload.tokenMask === tokenMaskReg
  val measureFramingLegal =
    io.measureInput.payload.first === expectedMeasureFirst &&
      io.measureInput.payload.last === expectedMeasureLast

  val expectedReplayFirst = replayIndex === 0
  val expectedReplayLast =
    replayIndex.resize(cfg.lengthWidth) === (lengthReg - 1)
  val replayMetadataLegal =
    io.replayInput.payload.tag === tagReg &&
      io.replayInput.payload.epoch === epochReg &&
      io.replayInput.payload.featureIndex === replayIndex &&
      io.replayInput.payload.tokenMask === tokenMaskReg
  val replayFramingLegal =
    io.replayInput.payload.first === expectedReplayFirst &&
      io.replayInput.payload.last === expectedReplayLast
  val replayBeatLegal = replayMetadataLegal && replayFramingLegal

  io.command.ready := state === State.Idle
  io.measureInput.ready := state === State.Measure

  io.replayRequest.valid := state === State.RequestReplay
  io.replayRequest.payload.tag := tagReg
  io.replayRequest.payload.epoch := epochReg
  io.replayRequest.payload.vectorLength := lengthReg
  io.replayRequest.payload.tokenMask := tokenMaskReg

  // The one-beat elastic stages preserve II=1 while putting a real timing
  // endpoint after every DSP/adder level; no stage scales with vectorLength.
  val capture = Stream(BitNetSharedDynamicQuantCaptured(cfg))
  capture.valid := state === State.Replay &&
    io.replayInput.valid && replayBeatLegal
  capture.payload.featureIndex := replayIndex
  capture.payload.first := expectedReplayFirst
  capture.payload.last := expectedReplayLast
  for (lane <- 0 until cfg.tokenLanes) {
    capture.payload.magnitude(lane) := replayMagnitude(lane)
    capture.payload.negative(lane) :=
      io.replayInput.payload.value(lane).msb
  }
  val captured = capture.m2sPipe()

  // Explicit unsigned 2x2 limb decomposition prevents Vivado from creating a
  // four-DSP combinational cascade for 33x41.  Each partial product is one
  // independently registered DSP48E2 operation.
  val partialWire = Stream(
    BitNetSharedDynamicQuantReciprocalPartials(cfg))
  partialWire.valid := captured.valid
  captured.ready := partialWire.ready
  partialWire.payload.featureIndex := captured.payload.featureIndex
  partialWire.payload.first := captured.payload.first
  partialWire.payload.last := captured.payload.last
  partialWire.payload.negative := captured.payload.negative
  for (lane <- 0 until cfg.tokenLanes) {
    val magnitudeLow = captured.payload.magnitude(lane)(16 downto 0)
    val magnitudeHigh = captured.payload.magnitude(lane)(32 downto 17)
    val reciprocalLow = reciprocal(lane)(16 downto 0)
    val reciprocalHigh = reciprocal(lane)(40 downto 17)
    val p00 = magnitudeLow * reciprocalLow
    val p10 = magnitudeHigh * reciprocalLow
    val p01 = magnitudeLow * reciprocalHigh
    val p11 = magnitudeHigh * reciprocalHigh
    p00.addAttribute("use_dsp", "yes")
    p10.addAttribute("use_dsp", "yes")
    p01.addAttribute("use_dsp", "yes")
    p11.addAttribute("use_dsp", "yes")
    partialWire.payload.p00(lane) := p00
    partialWire.payload.p10(lane) := p10
    partialWire.payload.p01(lane) := p01
    partialWire.payload.p11(lane) := p11

    // abs*127 is a constant shift/subtract, not another multiplier/DSP.
    val magnitudeWide = captured.payload.magnitude(lane)
      .resize(cfg.exactProductWidth)
    val exactNumerator =
      ((magnitudeWide |<< 7).resize(cfg.exactProductWidth) -
        magnitudeWide).resize(cfg.exactProductWidth)
    partialWire.payload.twiceNumerator(lane) :=
      (exactNumerator |<< 1).resize(cfg.exactProductWidth)
  }
  val partials = partialWire.m2sPipe()

  // First adder level keeps the two reciprocal limbs independent.
  val pairWire = Stream(BitNetSharedDynamicQuantReciprocalPairs(cfg))
  pairWire.valid := partials.valid
  partials.ready := pairWire.ready
  pairWire.payload.featureIndex := partials.payload.featureIndex
  pairWire.payload.first := partials.payload.first
  pairWire.payload.last := partials.payload.last
  pairWire.payload.negative := partials.payload.negative
  pairWire.payload.twiceNumerator := partials.payload.twiceNumerator
  for (lane <- 0 until cfg.tokenLanes) {
    pairWire.payload.low(lane) :=
      (partials.payload.p00(lane).resize(50) +
        (partials.payload.p10(lane).resize(50) |<< 17))
        .resize(50)
    pairWire.payload.high(lane) :=
      ((partials.payload.p01(lane)
          .resize(cfg.reciprocalProductWidth) |<< 17) +
        (partials.payload.p11(lane)
          .resize(cfg.reciprocalProductWidth) |<< 34))
        .resize(cfg.reciprocalProductWidth)
  }
  val pairs = pairWire.m2sPipe()

  val productWire = Stream(
    BitNetSharedDynamicQuantReciprocalProduct(cfg))
  productWire.valid := pairs.valid
  pairs.ready := productWire.ready
  productWire.payload.featureIndex := pairs.payload.featureIndex
  productWire.payload.first := pairs.payload.first
  productWire.payload.last := pairs.payload.last
  productWire.payload.negative := pairs.payload.negative
  productWire.payload.twiceNumerator := pairs.payload.twiceNumerator
  for (lane <- 0 until cfg.tokenLanes) {
    productWire.payload.product(lane) :=
      (pairs.payload.low(lane).resize(cfg.reciprocalProductWidth) +
        pairs.payload.high(lane)).resize(cfg.reciprocalProductWidth)
  }
  val products = productWire.m2sPipe()

  val proposalWire = Stream(BitNetSharedDynamicQuantProposed(cfg))
  proposalWire.valid := products.valid
  products.ready := proposalWire.ready
  proposalWire.payload.featureIndex := products.payload.featureIndex
  proposalWire.payload.first := products.payload.first
  proposalWire.payload.last := products.payload.last
  proposalWire.payload.negative := products.payload.negative
  proposalWire.payload.twiceNumerator := products.payload.twiceNumerator
  for (lane <- 0 until cfg.tokenLanes) {
    // (product + 2^32) >> 33 equals product[73:33] + product[32].
    // Writing it this way avoids a 75-bit rounding carry chain.
    val productHigh = products.payload.product(lane)(
      cfg.reciprocalProductWidth - 1 downto cfg.reciprocalFractionBits)
    val provisionalWide =
      (productHigh.resize(productHigh.getWidth + 1) +
        products.payload.product(lane)(
          cfg.reciprocalFractionBits - 1).asUInt
          .resize(productHigh.getWidth + 1))
        .resize(productHigh.getWidth + 1)
    proposalWire.payload.provisional(lane) :=
      provisionalWide.resize(8)
    when(provisionalWide > U(127, provisionalWide.getWidth bits)) {
      proposalWire.payload.provisional(lane) := 127
    }
  }
  val proposed = proposalWire.m2sPipe()

  // The 33x8 exact-center multiply is split into two independently registered
  // DSPs per lane for the same reason as the reciprocal product.
  val centerPartialWire = Stream(
    BitNetSharedDynamicQuantCenterPartials(cfg))
  centerPartialWire.valid := proposed.valid
  proposed.ready := centerPartialWire.ready
  centerPartialWire.payload.featureIndex := proposed.payload.featureIndex
  centerPartialWire.payload.first := proposed.payload.first
  centerPartialWire.payload.last := proposed.payload.last
  centerPartialWire.payload.negative := proposed.payload.negative
  centerPartialWire.payload.provisional := proposed.payload.provisional
  centerPartialWire.payload.twiceNumerator := proposed.payload.twiceNumerator
  for (lane <- 0 until cfg.tokenLanes) {
    val centerLow = maxAbs(lane)(16 downto 0) *
      proposed.payload.provisional(lane)
    val centerHigh = maxAbs(lane)(32 downto 17) *
      proposed.payload.provisional(lane)
    centerLow.addAttribute("use_dsp", "yes")
    centerHigh.addAttribute("use_dsp", "yes")
    centerPartialWire.payload.low(lane) := centerLow
    centerPartialWire.payload.high(lane) := centerHigh
  }
  val centerPartials = centerPartialWire.m2sPipe()

  val centerProductWire = Stream(
    BitNetSharedDynamicQuantCenterProduct(cfg))
  centerProductWire.valid := centerPartials.valid
  centerPartials.ready := centerProductWire.ready
  centerProductWire.payload.featureIndex :=
    centerPartials.payload.featureIndex
  centerProductWire.payload.first := centerPartials.payload.first
  centerProductWire.payload.last := centerPartials.payload.last
  centerProductWire.payload.negative := centerPartials.payload.negative
  centerProductWire.payload.provisional :=
    centerPartials.payload.provisional
  centerProductWire.payload.twiceNumerator :=
    centerPartials.payload.twiceNumerator
  for (lane <- 0 until cfg.tokenLanes) {
    centerProductWire.payload.product(lane) :=
      (centerPartials.payload.low(lane).resize(cfg.exactProductWidth) +
        (centerPartials.payload.high(lane)
          .resize(cfg.exactProductWidth) |<< 17))
        .resize(cfg.exactProductWidth)
  }
  val centerProducts = centerProductWire.m2sPipe()

  val correctionWire = Stream(BitNetSharedDynamicQuantCorrected(cfg))
  correctionWire.valid := centerProducts.valid
  centerProducts.ready := correctionWire.ready
  correctionWire.payload.featureIndex := centerProducts.payload.featureIndex
  correctionWire.payload.first := centerProducts.payload.first
  correctionWire.payload.last := centerProducts.payload.last
  correctionWire.payload.negative := centerProducts.payload.negative
  for (lane <- 0 until cfg.tokenLanes) {
    // One center product replaces the original pair of (2c+/-1)*max
    // multipliers: thresholds are 2*(c*max)-max and 2*(c*max)+max.
    val twiceCenter =
      (centerProducts.payload.product(lane) |<< 1)
        .resize(cfg.exactProductWidth)
    val maximumWide = maxAbs(lane).resize(cfg.exactProductWidth)
    val lowerThreshold = UInt(cfg.exactProductWidth bits)
    lowerThreshold := 0
    when(centerProducts.payload.provisional(lane) =/= 0) {
      lowerThreshold :=
        (twiceCenter - maximumWide).resize(cfg.exactProductWidth)
    }
    val upperThreshold =
      (twiceCenter + maximumWide).resize(cfg.exactProductWidth)

    correctionWire.payload.magnitude(lane) :=
      centerProducts.payload.provisional(lane)
    when(maxAbs(lane) =/= 0 &&
        centerProducts.payload.provisional(lane) =/= 0 &&
        centerProducts.payload.twiceNumerator(lane) < lowerThreshold) {
      correctionWire.payload.magnitude(lane) :=
        centerProducts.payload.provisional(lane) - 1
    } elsewhen (maxAbs(lane) =/= 0 &&
        centerProducts.payload.provisional(lane) < 127 &&
        centerProducts.payload.twiceNumerator(lane) >= upperThreshold) {
      correctionWire.payload.magnitude(lane) :=
        centerProducts.payload.provisional(lane) + 1
    }
  }
  // Keep the arithmetic-to-output timing cut on the compact 51-bit corrected
  // beat.  The following skid stage registers the downstream ready path
  // without adding forward latency when the output is not stalled.
  val correctedM2s = correctionWire.m2sPipe()
  val corrected = correctedM2s.s2mPipe()

  val replayPipelineEmpty =
    !captured.valid && !partials.valid && !pairs.valid && !products.valid &&
      !proposed.valid && !centerPartials.valid && !centerProducts.valid &&
      !correctedM2s.valid && !corrected.valid
  // A malformed beat waits for older legal results to drain, then is consumed
  // with no output.  Therefore no result can appear after fault detection.
  io.replayInput.ready := state === State.Replay &&
    Mux(replayBeatLegal, capture.ready, replayPipelineEmpty)

  io.output.valid := corrected.valid
  corrected.ready := io.output.ready
  io.output.payload.tag := tagReg
  io.output.payload.epoch := epochReg
  io.output.payload.featureIndex := corrected.payload.featureIndex
  io.output.payload.tokenMask := tokenMaskReg
  io.output.payload.first := corrected.payload.first
  io.output.payload.last := corrected.payload.last
  for (lane <- 0 until cfg.tokenLanes) {
    io.output.payload.dequantFactorQ16(lane) := factor(lane)
    io.output.payload.value(lane) := 0
    val signedCode = SInt(9 bits)
    when(corrected.payload.negative(lane)) {
      signedCode :=
        (-corrected.payload.magnitude(lane).resize(9).asSInt).resize(9)
    } otherwise {
      signedCode := corrected.payload.magnitude(lane).resize(9).asSInt
    }
    when(tokenMaskReg(lane)) {
      io.output.payload.value(lane) := signedCode.resize(8)
    }
  }

  io.done.valid := state === State.Complete
  io.done.payload.tag := tagReg
  io.done.payload.epoch := epochReg
  io.done.payload.vectorLength := lengthReg
  io.done.payload.emittedFeatures := emittedFeatures
  io.done.payload.success := jobFaultCode ===
    B(BitNetSharedDynamicQuantFault.None, 8 bits)
  io.done.payload.faultCode := jobFaultCode
  io.busy := state =/= State.Idle

  when(io.output.fire) {
    emittedFeatures := emittedFeatures + 1
    when(io.output.payload.last) {
      state := State.Complete
    }
  }

  switch(state) {
    is(State.Idle) {
      when(io.command.fire) {
        tagReg := io.command.payload.tag
        epochReg := io.command.payload.epoch
        lengthReg := io.command.payload.vectorLength
        tokenMaskReg := io.command.payload.tokenMask
        measureIndex := 0
        replayIndex := 0
        emittedFeatures := 0
        jobFaultCode := B(BitNetSharedDynamicQuantFault.None, 8 bits)
        for (lane <- 0 until cfg.tokenLanes) {
          maxAbs(lane) := 0
          reciprocal(lane) := 0
          factor(lane) := U(
            BigInt(1) << cfg.inputFractionBits, cfg.factorWidth bits)
        }
        when(io.command.payload.vectorLength === 0 ||
            io.command.payload.vectorLength >
              U(cfg.maxVectorLength, cfg.lengthWidth bits) ||
            !io.command.payload.tokenMask.orR) {
          jobFaultCode :=
            B(BitNetSharedDynamicQuantFault.InvalidCommand, 8 bits)
          stickyFault := True
          state := State.Complete
        } otherwise {
          state := State.Measure
        }
      }
    }

    is(State.Measure) {
      when(io.measureInput.fire) {
        when(!measureMetadataLegal) {
          jobFaultCode :=
            B(BitNetSharedDynamicQuantFault.MeasureMetadata, 8 bits)
          stickyFault := True
          state := State.Complete
        } elsewhen (!measureFramingLegal) {
          jobFaultCode :=
            B(BitNetSharedDynamicQuantFault.MeasureFraming, 8 bits)
          stickyFault := True
          state := State.Complete
        } otherwise {
          for (lane <- 0 until cfg.tokenLanes) {
            when(tokenMaskReg(lane) &&
                measureMagnitude(lane) > maxAbs(lane)) {
              maxAbs(lane) := measureMagnitude(lane)
            }
          }
          when(expectedMeasureLast) {
            state := State.ScalePrepare
          } otherwise {
            measureIndex := measureIndex + 1
          }
        }
      }
    }

    is(State.ScalePrepare) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := U(
          BigInt(127) << cfg.reciprocalFractionBits,
          cfg.dividerWidth bits)
        dividerDivisor(lane) := maxAbs(lane).resize(cfg.dividerWidth)
        when(maxAbs(lane) === 0) { dividerDivisor(lane) := 1 }
        dividerRemainder(lane) := 0
        dividerQuotient(lane) := 0
      }
      dividerIteration := 0
      state := State.ScaleDivide
    }

    is(State.ScaleDivide) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := dividerDividendNext(lane)
        dividerRemainder(lane) := dividerRemainderNext(lane)
        dividerQuotient(lane) := dividerQuotientNext(lane)
      }
      when(dividerIteration ===
          U(cfg.dividerWidth - 1, cfg.dividerCountWidth bits)) {
        for (lane <- 0 until cfg.tokenLanes) {
          reciprocal(lane) := dividerQuotientNext(lane)
          when(maxAbs(lane) === 0) { reciprocal(lane) := 0 }
        }
        state := State.FactorPrepare
      } otherwise {
        dividerIteration := dividerIteration + 1
      }
    }

    is(State.FactorPrepare) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) :=
          (maxAbs(lane).resize(cfg.dividerWidth) +
            U(63, cfg.dividerWidth bits)).resize(cfg.dividerWidth)
        dividerDivisor(lane) := U(127, cfg.dividerWidth bits)
        dividerRemainder(lane) := 0
        dividerQuotient(lane) := 0
      }
      dividerIteration := 0
      state := State.FactorDivide
    }

    is(State.FactorDivide) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := dividerDividendNext(lane)
        dividerRemainder(lane) := dividerRemainderNext(lane)
        dividerQuotient(lane) := dividerQuotientNext(lane)
      }
      when(dividerIteration ===
          U(cfg.dividerWidth - 1, cfg.dividerCountWidth bits)) {
        for (lane <- 0 until cfg.tokenLanes) {
          factor(lane) := dividerQuotientNext(lane).resize(cfg.factorWidth)
          when(maxAbs(lane) === 0) {
            factor(lane) := U(
              BigInt(1) << cfg.inputFractionBits, cfg.factorWidth bits)
          }
        }
        replayIndex := 0
        emittedFeatures := 0
        state := State.RequestReplay
      } otherwise {
        dividerIteration := dividerIteration + 1
      }
    }

    is(State.RequestReplay) {
      when(io.replayRequest.fire) {
        state := State.Replay
      }
    }

    is(State.Replay) {
      when(io.replayInput.fire) {
        when(!replayMetadataLegal) {
          jobFaultCode :=
            B(BitNetSharedDynamicQuantFault.ReplayMetadata, 8 bits)
          stickyFault := True
          state := State.Complete
        } elsewhen (!replayFramingLegal) {
          jobFaultCode :=
            B(BitNetSharedDynamicQuantFault.ReplayFraming, 8 bits)
          stickyFault := True
          state := State.Complete
        } otherwise {
          when(expectedReplayLast) {
            // Completion is ordered after the registered last result is
            // actually accepted, not merely after its replay input arrives.
            state := State.ReplayDrain
          } otherwise {
            replayIndex := replayIndex + 1
          }
        }
      }
    }

    is(State.ReplayDrain) {
      // The common output-fire logic retires the last elastic result.
    }

    is(State.Complete) {
      when(io.done.fire) {
        completedCounter := completedCounter + 1
        state := State.Idle
      }
    }
  }
}

object GenerateBitNetSharedDynamicI8Quantizer extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-shared-dynamic-i8-quantizer")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetSharedDynamicI8Quantizer())
}
