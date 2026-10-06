package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Geometry of the four-physical-lane QK dequantisation stage. */
case class BitNetQkVectorScalerConfig(
    qk: BitNetQkTp4AssemblerConfig = BitNetQkTp4AssemblerConfig(),
    scalar: BitNetQkScoreScalerConfig = BitNetQkScoreScalerConfig(),
    metaFifoDepth: Int = 64
) {
  require(qk.laneCount == 2 || qk.laneCount == 4)
  require(scalar.rawDotWidth == qk.continuous.base.accumulatorWidth)
  require(scalar.factorWidth == qk.factorWidth)
  require(scalar.tokenIdWidth >= qk.model.sequenceWidth)
  require(scalar.headWidth >= qk.queryHeadWidth)
  // sequenceWidth also represents the count value 4096, whereas an absolute
  // key position is only 0..4095 and therefore deliberately remains 12 bits.
  require(scalar.keyPositionWidth >= log2Up(qk.maxSeq))
  require(scalar.tagWidth == qk.continuous.taskIdWidth)
  require(metaFifoDepth >= 4 && isPow2(metaFifoDepth))

  val laneCount: Int = qk.laneCount
  /** One arithmetic lane serially drains the physical score vector. */
  val arithmeticLaneCount: Int = 1
  val serializedIssueCycles: Int = laneCount / arithmeticLaneCount
  require(laneCount % arithmeticLaneCount == 0)
  /** Exact low magnitude plus a sticky >= 2^W overflow bit. */
  val cappedMagnitudeWidth: Int = scalar.nonSaturatedFactorWidth + 1
}

/** Four scaled Q4.12 scores with the pass metadata needed by softmax. */
case class BitNetQkScaledScoreBeat(cfg: BitNetQkVectorScalerConfig)
    extends Bundle {
  val requestId = UInt(cfg.qk.model.requestIdWidth bits)
  val layer = UInt(cfg.qk.model.layerWidth bits)
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.qk.passIndexWidth bits)
  val kvHead = UInt(cfg.qk.kvHeadWidth bits)
  val laneValid = Bits(cfg.laneCount bits)
  val tokenId = Vec(
    UInt(cfg.qk.model.sequenceWidth bits), cfg.laneCount)
  val queryHeadId = Vec(
    UInt(cfg.qk.queryHeadWidth bits), cfg.laneCount)
  val queryPosition = Vec(
    UInt(cfg.qk.model.sequenceWidth bits), cfg.laneCount)
  val keyCount = Vec(
    UInt(cfg.qk.model.sequenceWidth bits), cfg.laneCount)
  val scoreQ12 = Vec(SInt(cfg.scalar.scoreWidth bits), cfg.laneCount)
  val taskTag = Vec(
    UInt(cfg.qk.continuous.taskIdWidth bits), cfg.laneCount)
  val keyPosition = UInt(cfg.qk.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.qk.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
  /** False invalidates the entire vector but still lets the pipeline drain. */
  val protocolValid = Bool()
}

private[accel] case class BitNetQkVectorMeta(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val requestId = UInt(cfg.qk.model.requestIdWidth bits)
  val layer = UInt(cfg.qk.model.layerWidth bits)
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.qk.passIndexWidth bits)
  val kvHead = UInt(cfg.qk.kvHeadWidth bits)
  val laneValid = Bits(cfg.laneCount bits)
  val tokenId = Vec(
    UInt(cfg.qk.model.sequenceWidth bits), cfg.laneCount)
  val queryHeadId = Vec(
    UInt(cfg.qk.queryHeadWidth bits), cfg.laneCount)
  val queryPosition = Vec(
    UInt(cfg.qk.model.sequenceWidth bits), cfg.laneCount)
  val keyCount = Vec(
    UInt(cfg.qk.model.sequenceWidth bits), cfg.laneCount)
  val taskTag = Vec(
    UInt(cfg.qk.continuous.taskIdWidth bits), cfg.laneCount)
  val keyPosition = UInt(cfg.qk.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.qk.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
}

/** Arithmetic-only payloads for the lockstep TP4 scaler.  Routing metadata is
  * intentionally absent: it lives once in BitNetQkVectorMeta's external
  * FIFO while these stages retain only values required by the arithmetic.
  */
private[accel] case class BitNetQkVectorArithmeticInput(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val rawDot = Vec(
    SInt(cfg.scalar.rawDotWidth bits), cfg.arithmeticLaneCount)
  val queryFactorQ16 = Vec(
    UInt(cfg.scalar.factorWidth bits), cfg.arithmeticLaneCount)
  val keyFactorQ16 = UInt(cfg.scalar.factorWidth bits)
}

private[accel] case class BitNetQkVectorArithmeticOutput(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val scoreQ12 = Vec(
    SInt(cfg.scalar.scoreWidth bits), cfg.arithmeticLaneCount)
}

private[accel] case class BitNetQkVectorFactorPartials(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val lowLow = Vec(UInt(32 bits), cfg.arithmeticLaneCount)
  val lowHigh = Vec(UInt(32 bits), cfg.arithmeticLaneCount)
  val highLow = Vec(UInt(32 bits), cfg.arithmeticLaneCount)
  val highHigh = Vec(UInt(32 bits), cfg.arithmeticLaneCount)
  val dotMagnitude = Vec(
    UInt(cfg.scalar.dotMagnitudeWidth bits), cfg.arithmeticLaneCount)
  val negative = Bits(cfg.arithmeticLaneCount bits)
}

private[accel] case class BitNetQkVectorFactorProduct(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val factorLow = Vec(
    UInt(cfg.scalar.nonSaturatedFactorWidth bits), cfg.arithmeticLaneCount)
  val factorTooLarge = Bits(cfg.arithmeticLaneCount bits)
  val dotMagnitude = Vec(
    UInt(cfg.scalar.dotMagnitudeWidth bits), cfg.arithmeticLaneCount)
  val negative = Bits(cfg.arithmeticLaneCount bits)
}

private[accel] case class BitNetQkVectorMagnitudePartials(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val lowLow = Vec(UInt(36 bits), cfg.arithmeticLaneCount)
  val lowHigh = Vec(UInt(32 bits), cfg.arithmeticLaneCount)
  val highLow = Vec(UInt(39 bits), cfg.arithmeticLaneCount)
  val highHigh = Vec(UInt(35 bits), cfg.arithmeticLaneCount)
  val factorTooLarge = Bits(cfg.arithmeticLaneCount bits)
  val dotNonZero = Bits(cfg.arithmeticLaneCount bits)
  val negative = Bits(cfg.arithmeticLaneCount bits)
}

private[accel] case class BitNetQkVectorMagnitudeCross(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val lowLow = Vec(UInt(36 bits), cfg.arithmeticLaneCount)
  val cross = Vec(UInt(40 bits), cfg.arithmeticLaneCount)
  val highHigh = Vec(UInt(35 bits), cfg.arithmeticLaneCount)
  val factorTooLarge = Bits(cfg.arithmeticLaneCount bits)
  val dotNonZero = Bits(cfg.arithmeticLaneCount bits)
  val negative = Bits(cfg.arithmeticLaneCount bits)
}

private[accel] case class BitNetQkVectorMagnitudeProduct(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val magnitude = Vec(
    UInt(cfg.cappedMagnitudeWidth bits), cfg.arithmeticLaneCount)
  val factorTooLarge = Bits(cfg.arithmeticLaneCount bits)
  val dotNonZero = Bits(cfg.arithmeticLaneCount bits)
  val negative = Bits(cfg.arithmeticLaneCount bits)
}

private[accel] case class BitNetQkVectorDivideState(
    cfg: BitNetQkVectorScalerConfig) extends Bundle {
  val remainder = Vec(
    UInt(cfg.scalar.divisionRemainderWidth bits), cfg.arithmeticLaneCount)
  val quotient = Vec(
    UInt(cfg.scalar.quotientWidth bits), cfg.arithmeticLaneCount)
  val clamp = Bits(cfg.arithmeticLaneCount bits)
  val negative = Bits(cfg.arithmeticLaneCount bits)
}

/** One elastic-control pipeline carrying the serialized physical score lane.
  *
  * There is exactly one valid/ready register per stage.  The factor product
  * uses four 16x16 DSP limbs total.  TP4 routing metadata remains outside this
  * long pipeline while the wrapper issues four logical lanes in order.
  */
private[accel] class BitNetQkSerializedLaneElasticPipeline(
    cfg: BitNetQkVectorScalerConfig) extends Component {
  val io = new Bundle {
    val input = slave(Stream(BitNetQkVectorArithmeticInput(cfg)))
    val output = master(Stream(BitNetQkVectorArithmeticOutput(cfg)))
  }

  private val scalar = cfg.scalar
  private val denominatorQ32 = BigInt("b504f333f9de65", 16)
  private val saturationNumerator = BigInt("5a821f1784", 16)
  private val magnitudeValueWidth = scalar.nonSaturatedFactorWidth
  require(cfg.cappedMagnitudeWidth == magnitudeValueWidth + 1)
  require(saturationNumerator < (BigInt(1) << magnitudeValueWidth))

  val factorPartials = Stream(BitNetQkVectorFactorPartials(cfg))
  factorPartials.valid := io.input.valid
  io.input.ready := factorPartials.ready
  val keyLow = io.input.keyFactorQ16(15 downto 0)
  val keyHigh = io.input.keyFactorQ16(31 downto 16)
  for (lane <- 0 until cfg.arithmeticLaneCount) {
    val rawWide = io.input.rawDot(lane).resize(scalar.rawDotWidth + 1)
    val rawNegative = rawWide.msb
    val rawMagnitude = UInt(scalar.dotMagnitudeWidth bits)
    when(rawNegative) {
      rawMagnitude := (-rawWide).asUInt.resize(scalar.dotMagnitudeWidth)
    } otherwise {
      rawMagnitude := rawWide.asUInt.resize(scalar.dotMagnitudeWidth)
    }

    val queryLow = io.input.queryFactorQ16(lane)(15 downto 0)
    val queryHigh = io.input.queryFactorQ16(lane)(31 downto 16)
    factorPartials.lowLow(lane) := queryLow * keyLow
    factorPartials.lowHigh(lane) := queryLow * keyHigh
    factorPartials.highLow(lane) := queryHigh * keyLow
    factorPartials.highHigh(lane) := queryHigh * keyHigh
    factorPartials.lowLow(lane).addAttribute("use_dsp", "yes")
    factorPartials.lowHigh(lane).addAttribute("use_dsp", "yes")
    factorPartials.highLow(lane).addAttribute("use_dsp", "yes")
    factorPartials.highHigh(lane).addAttribute("use_dsp", "yes")
    factorPartials.dotMagnitude(lane) := rawMagnitude
    factorPartials.negative(lane) := rawNegative
  }
  val registeredFactorPartials = factorPartials.m2sPipe()

  val factorProduct = Stream(BitNetQkVectorFactorProduct(cfg))
  factorProduct.valid := registeredFactorPartials.valid
  registeredFactorPartials.ready := factorProduct.ready
  factorProduct.negative := registeredFactorPartials.negative
  for (lane <- 0 until cfg.arithmeticLaneCount) {
    val cross = registeredFactorPartials.lowHigh(lane).resize(33) +
      registeredFactorPartials.highLow(lane).resize(33)
    val factorProductWide =
      registeredFactorPartials.lowLow(lane).resize(
        scalar.factorProductWidth) +
        (cross.resize(scalar.factorProductWidth) |<< 16) +
        (registeredFactorPartials.highHigh(lane).resize(
          scalar.factorProductWidth) |<< 32)
    factorProduct.factorLow(lane) := factorProductWide(
      scalar.nonSaturatedFactorWidth - 1 downto 0)
    factorProduct.factorTooLarge(lane) := factorProductWide(
      scalar.factorProductWidth - 1 downto
        scalar.nonSaturatedFactorWidth).orR
    factorProduct.dotMagnitude(lane) :=
      registeredFactorPartials.dotMagnitude(lane)
  }
  val registeredFactorProduct = factorProduct.m2sPipe()

  val magnitudePartials = Stream(BitNetQkVectorMagnitudePartials(cfg))
  magnitudePartials.valid := registeredFactorProduct.valid
  registeredFactorProduct.ready := magnitudePartials.ready
  magnitudePartials.factorTooLarge := registeredFactorProduct.factorTooLarge
  magnitudePartials.negative := registeredFactorProduct.negative
  for (lane <- 0 until cfg.arithmeticLaneCount) {
    magnitudePartials.dotNonZero(lane) :=
      registeredFactorProduct.dotMagnitude(lane).orR
    val factorLow = registeredFactorProduct.factorLow(lane)(17 downto 0)
    val factorHigh = registeredFactorProduct.factorLow(lane)(38 downto 18)
    val dotLow = registeredFactorProduct.dotMagnitude(lane)(17 downto 0)
    val dotHigh = registeredFactorProduct.dotMagnitude(lane)(31 downto 18)
    magnitudePartials.lowLow(lane) := factorLow * dotLow
    magnitudePartials.lowHigh(lane) := factorLow * dotHigh
    magnitudePartials.highLow(lane) := factorHigh * dotLow
    magnitudePartials.highHigh(lane) := factorHigh * dotHigh
    magnitudePartials.lowLow(lane).addAttribute("use_dsp", "yes")
    magnitudePartials.lowHigh(lane).addAttribute("use_dsp", "yes")
    magnitudePartials.highLow(lane).addAttribute("use_dsp", "yes")
    magnitudePartials.highHigh(lane).addAttribute("use_dsp", "yes")
  }
  val registeredMagnitudePartials = magnitudePartials.m2sPipe()

  val magnitudeCross = Stream(BitNetQkVectorMagnitudeCross(cfg))
  magnitudeCross.valid := registeredMagnitudePartials.valid
  registeredMagnitudePartials.ready := magnitudeCross.ready
  magnitudeCross.factorTooLarge := registeredMagnitudePartials.factorTooLarge
  magnitudeCross.dotNonZero := registeredMagnitudePartials.dotNonZero
  magnitudeCross.negative := registeredMagnitudePartials.negative
  for (lane <- 0 until cfg.arithmeticLaneCount) {
    magnitudeCross.lowLow(lane) := registeredMagnitudePartials.lowLow(lane)
    magnitudeCross.cross(lane) :=
      registeredMagnitudePartials.lowHigh(lane).resize(40) +
        registeredMagnitudePartials.highLow(lane).resize(40)
    magnitudeCross.highHigh(lane) :=
      registeredMagnitudePartials.highHigh(lane)
  }
  val registeredMagnitudeCross = magnitudeCross.m2sPipe()

  val magnitudeProduct = Stream(BitNetQkVectorMagnitudeProduct(cfg))
  magnitudeProduct.valid := registeredMagnitudeCross.valid
  registeredMagnitudeCross.ready := magnitudeProduct.ready
  magnitudeProduct.factorTooLarge := registeredMagnitudeCross.factorTooLarge
  magnitudeProduct.dotNonZero := registeredMagnitudeCross.dotNonZero
  magnitudeProduct.negative := registeredMagnitudeCross.negative
  for (lane <- 0 until cfg.arithmeticLaneCount) {
    val magnitudeWide = UInt(scalar.magnitudeProductWidth bits)
    magnitudeWide :=
      registeredMagnitudeCross.lowLow(lane).resize(
        scalar.magnitudeProductWidth) +
        (registeredMagnitudeCross.cross(lane).resize(
          scalar.magnitudeProductWidth) |<< 18) +
        (registeredMagnitudeCross.highHigh(lane).resize(
          scalar.magnitudeProductWidth) |<< 36)
    magnitudeProduct.magnitude(lane)(magnitudeValueWidth - 1 downto 0) :=
      magnitudeWide(magnitudeValueWidth - 1 downto 0)
    magnitudeProduct.magnitude(lane).msb :=
      magnitudeWide(scalar.magnitudeProductWidth - 1 downto
        magnitudeValueWidth).orR
  }
  val registeredMagnitude = magnitudeProduct.m2sPipe()

  val divideStart = Stream(BitNetQkVectorDivideState(cfg))
  divideStart.valid := registeredMagnitude.valid
  registeredMagnitude.ready := divideStart.ready
  divideStart.negative := registeredMagnitude.negative
  for (lane <- 0 until cfg.arithmeticLaneCount) {
    val magnitudeOverflow = registeredMagnitude.magnitude(lane).msb
    val magnitudeLow = registeredMagnitude.magnitude(lane)(
      scalar.nonSaturatedFactorWidth - 1 downto 0)
    val initialClamp =
      (registeredMagnitude.factorTooLarge(lane) &&
        registeredMagnitude.dotNonZero(lane)) ||
        magnitudeOverflow || magnitudeLow >= U(saturationNumerator,
          scalar.nonSaturatedFactorWidth bits)
    divideStart.remainder(lane) := (magnitudeLow
      .resize(scalar.divisionRemainderWidth) |<<
        scalar.sqrtFractionBits).resized
    divideStart.quotient(lane) := 0
    divideStart.clamp(lane) := initialClamp
  }
  var divideStage = divideStart.m2sPipe()

  for (bit <- (scalar.quotientWidth - 1) to 0 by -1) {
    val next = Stream(BitNetQkVectorDivideState(cfg))
    val shiftedDenominator = U(
      denominatorQ32 << bit, scalar.divisionRemainderWidth bits)
    next.valid := divideStage.valid
    divideStage.ready := next.ready
    next.clamp := divideStage.clamp
    next.negative := divideStage.negative
    for (lane <- 0 until cfg.arithmeticLaneCount) {
      val remainderDifference = UInt(
        (scalar.divisionRemainderWidth + 1) bits)
      remainderDifference :=
        divideStage.remainder(lane).resize(
          scalar.divisionRemainderWidth + 1) -
          shiftedDenominator.resize(scalar.divisionRemainderWidth + 1)
      remainderDifference.addAttribute("use_dsp", "yes")
      val take = !divideStage.clamp(lane) && !remainderDifference.msb
      next.remainder(lane) := divideStage.remainder(lane)
      next.quotient(lane) := divideStage.quotient(lane)
      when(take) {
        next.remainder(lane) := remainderDifference(
          scalar.divisionRemainderWidth - 1 downto 0)
        next.quotient(lane)(bit) := True
      }
      when(divideStage.clamp(lane)) {
        next.remainder(lane) := 0
        next.quotient(lane) := 0
      }
    }
    divideStage = next.m2sPipe()
  }

  val outputStage = Stream(BitNetQkVectorArithmeticOutput(cfg))
  outputStage.valid := divideStage.valid
  divideStage.ready := outputStage.ready
  for (lane <- 0 until cfg.arithmeticLaneCount) {
    val twiceRemainder =
      (divideStage.remainder(lane).resize(
        scalar.divisionRemainderWidth + 1) |<< 1)
    val roundUp = twiceRemainder >= U(
      denominatorQ32, scalar.divisionRemainderWidth + 1 bits)
    val roundedMagnitude = UInt((scalar.scoreWidth + 1) bits)
    roundedMagnitude :=
      divideStage.quotient(lane).resize(scalar.scoreWidth + 1)
    when(divideStage.clamp(lane)) {
      roundedMagnitude := U(BigInt(1) << (scalar.scoreWidth - 1),
        scalar.scoreWidth + 1 bits)
    } elsewhen (roundUp) {
      roundedMagnitude :=
        divideStage.quotient(lane).resize(scalar.scoreWidth + 1) + 1
    }

    outputStage.scoreQ12(lane) := 0
    when(divideStage.negative(lane)) {
      when(roundedMagnitude >= U(BigInt(1) << (scalar.scoreWidth - 1),
          scalar.scoreWidth + 1 bits)) {
        outputStage.scoreQ12(lane) :=
          S(-(BigInt(1) << (scalar.scoreWidth - 1)),
            scalar.scoreWidth bits)
      } otherwise {
        outputStage.scoreQ12(lane) :=
          (-roundedMagnitude.asSInt).resize(scalar.scoreWidth)
      }
    } otherwise {
      when(roundedMagnitude >= U(BigInt(1) << (scalar.scoreWidth - 1),
          scalar.scoreWidth + 1 bits)) {
        outputStage.scoreQ12(lane) :=
          S((BigInt(1) << (scalar.scoreWidth - 1)) - 1,
            scalar.scoreWidth bits)
      } otherwise {
        outputStage.scoreQ12(lane) :=
          roundedMagnitude.resize(scalar.scoreWidth).asSInt
      }
    }
  }

  io.output << outputStage.m2sPipe()
}

object BitNetQkVectorScalerMath {
  def apply(
      rawDot: Seq[Int],
      queryFactorQ16: Seq[Long],
      keyFactorQ16: Long,
      laneValid: Seq[Boolean]): Seq[Int] = {
    require(rawDot.length == 4 && queryFactorQ16.length == 4)
    require(laneValid.length == 4)
    (0 until 4).map { lane =>
      if (laneValid(lane)) {
        BitNetQkScoreScalerMath(
          rawDot(lane), queryFactorQ16(lane), keyFactorQ16)
      } else 0
    }
  }
}

/** Atomic TP4 wrapper around one time-multiplexed QK arithmetic lane.
  *
  * Packed KV replay needs eight 128-bit beats to construct one key row, so it
  * cannot produce score vectors faster than one every eight slow clocks.  This
  * wrapper serializes the four logical TP4 lanes over four clocks and therefore
  * stays ahead of that production boundary while replacing four copies of the
  * factor/reduction/divide pipeline with one.  Metadata remains vector-atomic
  * in one external FIFO and completed scalar scores are reassembled in order.
  */
class BitNetQkVectorScaler(
    cfg: BitNetQkVectorScalerConfig = BitNetQkVectorScalerConfig())
    extends Component {
  val io = new Bundle {
    val input = slave(Stream(BitNetQkTp4RawScoreBeat(cfg.qk)))
    val output = master(Stream(BitNetQkScaledScoreBeat(cfg)))
    val protocolFaultSticky = out Bool()
    val acceptedVectors = out UInt(32 bits)
    val emittedVectors = out UInt(32 bits)
  }

  val arithmetic = new BitNetQkSerializedLaneElasticPipeline(cfg)
  val metadata = StreamFifo(
    BitNetQkVectorMeta(cfg), cfg.metaFifoDepth)

  val heldRawDot = Vec(
    Reg(SInt(cfg.scalar.rawDotWidth bits)) init 0, cfg.laneCount)
  val heldQueryFactor = Vec(
    Reg(UInt(cfg.scalar.factorWidth bits)) init 0, cfg.laneCount)
  val heldKeyFactor = Reg(UInt(cfg.scalar.factorWidth bits)) init 0
  val serialBusy = RegInit(False)
  val serialLane = Reg(UInt(log2Up(cfg.laneCount) bits)) init 0
  val lastSerialLane = U(cfg.laneCount - 1, serialLane.getWidth bits)

  arithmetic.io.input.valid := serialBusy
  arithmetic.io.input.rawDot(0) := heldRawDot(serialLane)
  arithmetic.io.input.queryFactorQ16(0) := heldQueryFactor(serialLane)
  arithmetic.io.input.keyFactorQ16 := heldKeyFactor

  // On the fourth arithmetic issue a new vector may replace the held vector
  // in the same edge.  Thus sustained input cadence is exactly four clocks,
  // not five, while metadata admission remains atomic with vector capture.
  val serialRetiring = serialBusy && serialLane === lastSerialLane &&
    arithmetic.io.input.ready
  val vectorSlotAvailable = !serialBusy || serialRetiring
  io.input.ready := vectorSlotAvailable && metadata.io.push.ready
  metadata.io.push.valid := io.input.valid && vectorSlotAvailable

  when(io.input.fire) {
    heldRawDot := io.input.rawDot
    heldQueryFactor := io.input.qFactorQ16
    heldKeyFactor := io.input.kFactorQ16
  }
  when(arithmetic.io.input.fire) {
    when(serialLane === lastSerialLane) {
      serialLane := 0
      serialBusy := io.input.fire
    } otherwise {
      serialLane := serialLane + 1
    }
  }
  when(!serialBusy && io.input.fire) {
    serialBusy := True
    serialLane := 0
  }

  for (lane <- 0 until cfg.laneCount) {
    metadata.io.push.tokenId(lane) := io.input.tokenId(lane)
    metadata.io.push.queryHeadId(lane) := io.input.queryHeadId(lane)
    metadata.io.push.queryPosition(lane) := io.input.queryPosition(lane)
    metadata.io.push.keyCount(lane) := io.input.keyCount(lane)
    metadata.io.push.taskTag(lane) := io.input.taskTag(lane)
  }
  metadata.io.push.requestId := io.input.requestId
  metadata.io.push.layer := io.input.layer
  metadata.io.push.laneRole := io.input.laneRole
  metadata.io.push.passIndex := io.input.passIndex
  metadata.io.push.kvHead := io.input.kvHead
  metadata.io.push.laneValid := io.input.laneValid
  metadata.io.push.keyPosition := io.input.keyPosition
  metadata.io.push.keyIndex := io.input.keyIndex
  metadata.io.push.firstInSequence := io.input.firstInSequence
  metadata.io.push.lastInSequence := io.input.lastInSequence

  val scoreAssembly = Vec(
    Reg(SInt(cfg.scalar.scoreWidth bits)) init 0, cfg.laneCount)
  val completedScores = Vec(
    Reg(SInt(cfg.scalar.scoreWidth bits)) init 0, cfg.laneCount)
  val completedValid = RegInit(False)
  val collectLane = Reg(UInt(log2Up(cfg.laneCount) bits)) init 0
  val lastCollectLane = U(cfg.laneCount - 1, collectLane.getWidth bits)

  // A separate assembly vector lets three lanes of the next vector drain even
  // while one completed vector is backpressured.  Only its fourth lane waits
  // for the completed slot to retire, preserving scalar/vector ordering.
  val joinedValid = metadata.io.pop.valid && completedValid
  io.output.valid := joinedValid
  metadata.io.pop.ready := io.output.fire
  val completedSlotAvailable = !completedValid || io.output.fire
  arithmetic.io.output.ready := collectLane =/= lastCollectLane ||
    completedSlotAvailable

  when(io.output.fire) {
    completedValid := False
  }
  when(arithmetic.io.output.fire) {
    when(collectLane === lastCollectLane) {
      for (lane <- 0 until cfg.laneCount - 1) {
        completedScores(lane) := scoreAssembly(lane)
      }
      completedScores(cfg.laneCount - 1) :=
        arithmetic.io.output.scoreQ12(0)
      completedValid := True
      collectLane := 0
    } otherwise {
      scoreAssembly(collectLane) := arithmetic.io.output.scoreQ12(0)
      collectLane := collectLane + 1
    }
  }

  val orphanArithmetic = arithmetic.io.output.valid &&
    !metadata.io.pop.valid
  val orphanCompleted = completedValid && !metadata.io.pop.valid

  io.output.requestId := metadata.io.pop.requestId
  io.output.layer := metadata.io.pop.layer
  io.output.laneRole := metadata.io.pop.laneRole
  io.output.passIndex := metadata.io.pop.passIndex
  io.output.kvHead := metadata.io.pop.kvHead
  io.output.tokenId := metadata.io.pop.tokenId
  io.output.queryHeadId := metadata.io.pop.queryHeadId
  io.output.queryPosition := metadata.io.pop.queryPosition
  io.output.keyCount := metadata.io.pop.keyCount
  io.output.taskTag := metadata.io.pop.taskTag
  io.output.keyPosition := metadata.io.pop.keyPosition
  io.output.keyIndex := metadata.io.pop.keyIndex
  io.output.firstInSequence := metadata.io.pop.firstInSequence
  io.output.lastInSequence := metadata.io.pop.lastInSequence
  val fault = RegInit(False)
  val vectorProtocolValid = !fault && !orphanArithmetic && !orphanCompleted
  io.output.protocolValid := vectorProtocolValid
  for (lane <- 0 until cfg.laneCount) {
    io.output.laneValid(lane) :=
      metadata.io.pop.laneValid(lane) && vectorProtocolValid
    io.output.scoreQ12(lane) := 0
    when(metadata.io.pop.laneValid(lane) && vectorProtocolValid) {
      io.output.scoreQ12(lane) := completedScores(lane)
    }
  }

  val accepted = Reg(UInt(32 bits)) init 0
  val emitted = Reg(UInt(32 bits)) init 0
  when(io.input.fire) { accepted := accepted + 1 }
  when(orphanArithmetic || orphanCompleted) { fault := True }
  when(io.output.fire) {
    emitted := emitted + 1
    when(!vectorProtocolValid) { fault := True }
  }
  io.protocolFaultSticky := fault
  io.acceptedVectors := accepted
  io.emittedVectors := emitted
}

object GenerateBitNetQkVectorScaler extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-qk-vector-scaler")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetQkVectorScaler())
}
