package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Geometry and fixed-point contract of the resident dynamic i8 quantizer.
  *
  * Input samples are signed Q16.16.  The returned dequantization factor is an
  * unsigned Q16.16 value, i.e. `max(abs(x)) / 127`; Q16.16 is sufficient for
  * the entire signed-i32 input range.  The reciprocal used internally has 33
  * fractional bits.  It is produced once per active token lane by a 41-cycle
  * restoring divider and is never exposed as part of the architectural data
  * format.
  *
  * One beat is one feature position across all token lanes.  Hidden and FFN
  * vectors use the same circuit with different inferred RAM depths.
  */
case class NativeDynamicI8QuantizerConfig(
    tokenLanes: Int = 4,
    vectorLength: Int = 2560,
    inputWidth: Int = 32,
    inputFractionBits: Int = 16,
    reciprocalFractionBits: Int = 33,
    factorWidth: Int = 32,
    replayFifoDepth: Int = 8
) {
  require(tokenLanes > 0 && isPow2(tokenLanes),
    "tokenLanes must be a positive power of two")
  require(vectorLength >= 2, "a dynamic-quantized vector needs at least two entries")
  require(inputWidth == 32 && inputFractionBits == 16,
    "the native activation boundary is signed Q16.16")
  require(reciprocalFractionBits >= inputWidth + 1,
    "the reciprocal must keep the provisional quotient within one exact code")
  require(factorWidth >= inputWidth,
    "unsigned Q16.16 must cover max_abs/127 for every signed i32 input")
  require(replayFifoDepth >= 4 && isPow2(replayFifoDepth),
    "replayFifoDepth must be a power of two of at least four")

  val indexWidth: Int = log2Up(vectorLength)
  val magnitudeWidth: Int = inputWidth + 1
  val dividerWidth: Int = reciprocalFractionBits + 8
  val dividerCountWidth: Int = log2Up(dividerWidth)
  val reciprocalWidth: Int = dividerWidth
  val replayEntryWidth: Int = tokenLanes * inputWidth
  val reciprocalProductWidth: Int = magnitudeWidth + reciprocalWidth
  val exactProductWidth: Int = magnitudeWidth + 9
}

object NativeDynamicI8QuantizerConfig {
  def hidden(tokenLanes: Int = 4): NativeDynamicI8QuantizerConfig =
    NativeDynamicI8QuantizerConfig(tokenLanes = tokenLanes, vectorLength = 2560)

  def ffn(tokenLanes: Int = 4): NativeDynamicI8QuantizerConfig =
    NativeDynamicI8QuantizerConfig(tokenLanes = tokenLanes, vectorLength = 6912)
}

case class NativeDynamicQuantBeat(cfg: NativeDynamicI8QuantizerConfig)
    extends Bundle {
  val value = Vec(SInt(cfg.inputWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

case class NativeDynamicQuantResult(cfg: NativeDynamicI8QuantizerConfig)
    extends Bundle {
  val value = Vec(SInt(8 bits), cfg.tokenLanes)
  /** Unsigned Q16.16; repeated on every replay beat for a packet-free join. */
  val dequantFactor = Vec(UInt(cfg.factorWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

case class NativeDynamicQuantReplay(
    cfg: NativeDynamicI8QuantizerConfig
) extends Bundle {
  val data = Bits(cfg.replayEntryWidth bits)
  val index = UInt(cfg.indexWidth bits)
}

/**
  * Packet-free two-pass dynamic activation quantizer.
  *
  * Pass one writes the complete Q16.16 vector to an inferred synchronous RAM
  * and finds one max_abs per token lane.  A bank of small bit-serial restoring
  * dividers then computes floor(127*2^33/max_abs), once per lane.  The same
  * divider bank is reused to return round(max_abs/127) in unsigned Q16.16.
  * Pass two replays RAM at one feature beat per cycle and computes the i8 data.
  *
  * The Q33 reciprocal approximation is used only to propose an i8 code.  Its
  * pre-rounding error is at most 0.25 i8 LSB over the full signed-i32 range.
  * Two exact integer threshold comparisons correct that proposal by at most
  * one.  Consequently the externally visible result is exactly:
  *
  *   sat_i8(sign(x) * floor(abs(x)*127/max_abs + 0.5))
  *
  * including ties away from zero.  There is no combinational variable divider,
  * legacy packet, token-processor route, or micro-op dependency.  All Stream
  * payloads remain stable under downstream backpressure.
  */
class NativeDynamicI8Quantizer(
    cfg: NativeDynamicI8QuantizerConfig =
      NativeDynamicI8QuantizerConfig.hidden()
) extends Component {
  val io = new Bundle {
    val input = slave(Stream(NativeDynamicQuantBeat(cfg)))
    val output = master(Stream(NativeDynamicQuantResult(cfg)))
    val busy = out Bool()
    val protocolFault = out Bool()
    val acceptedVectors = out UInt(32 bits)
    val emittedVectors = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Collect, ScalePrepare, ScaleDivide, FactorPrepare, FactorDivide,
        Replay = newElement()
  }
  val state = Reg(State()) init State.Collect

  val storage = Mem(Bits(cfg.replayEntryWidth bits), cfg.vectorLength)
  val collecting = RegInit(False)
  val collectIndex = Reg(UInt(cfg.indexWidth bits)) init (0)
  val activeTokenMask = Reg(Bits(cfg.tokenLanes bits)) init (0)
  val fault = RegInit(False)
  val maxAbs = Vec(Reg(UInt(cfg.magnitudeWidth bits)) init (0), cfg.tokenLanes)

  val acceptedVectorCounter = Reg(UInt(32 bits)) init (0)
  val emittedVectorCounter = Reg(UInt(32 bits)) init (0)
  io.acceptedVectors := acceptedVectorCounter
  io.emittedVectors := emittedVectorCounter
  io.protocolFault := fault
  io.input.ready := state === State.Collect

  val packedInput = Bits(cfg.replayEntryWidth bits)
  for (lane <- 0 until cfg.tokenLanes) {
    packedInput((lane + 1) * cfg.inputWidth - 1 downto lane * cfg.inputWidth) :=
      io.input.value(lane).asBits
  }

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

  val inputMagnitude = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    inputMagnitude(lane) := magnitudeOf(io.input.value(lane))
  }

  val acceptedIndex = UInt(cfg.indexWidth bits)
  acceptedIndex := collectIndex
  when(io.input.first) { acceptedIndex := 0 }
  val acceptedAtEnd = acceptedIndex ===
    U(cfg.vectorLength - 1, cfg.indexWidth bits)
  val acceptedHasStart = collecting || io.input.first

  when(io.input.fire) {
    when(io.input.first) {
      fault := False
      when(collecting) { fault := True }
      collecting := True
      collectIndex := 0
      activeTokenMask := io.input.tokenMask
      when(!io.input.tokenMask.orR) { fault := True }
    }

    when(!acceptedHasStart) {
      fault := True
    } otherwise {
      // Gate the inferred RAM port explicitly.  A literal True here lowers to
      // an always-on write port even though this call is textually inside the
      // input-fire scope, corrupting the last element during divider cycles.
      storage.write(acceptedIndex, packedInput, enable = io.input.fire)

      for (lane <- 0 until cfg.tokenLanes) {
        val previous = UInt(cfg.magnitudeWidth bits)
        previous := maxAbs(lane)
        when(io.input.first) { previous := 0 }
        when(io.input.tokenMask(lane)) {
          when(inputMagnitude(lane) > previous) {
            maxAbs(lane) := inputMagnitude(lane)
          } otherwise {
            maxAbs(lane) := previous
          }
        } elsewhen (io.input.first) {
          maxAbs(lane) := 0
        }
      }

      when(!io.input.first && io.input.tokenMask =/= activeTokenMask) {
        fault := True
      }
      when(io.input.last =/= acceptedAtEnd) {
        fault := True
      }

      when(acceptedAtEnd) {
        collecting := False
        when(io.input.last) {
          state := State.ScalePrepare
          acceptedVectorCounter := acceptedVectorCounter + 1
        }
      } elsewhen (io.input.last) {
        // An early last discards the incomplete vector.
        collecting := False
      } otherwise {
        collectIndex := acceptedIndex + 1
      }
    }
  }

  // Exactly one fixed-latency restoring divider per token lane.  The bank is
  // reused for reciprocal scale and Q16.16 dequant-factor division.
  val dividerDividend = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0), cfg.tokenLanes)
  val dividerDivisor = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0), cfg.tokenLanes)
  val dividerRemainder = Vec(
    Reg(UInt((cfg.dividerWidth + 1) bits)) init (0), cfg.tokenLanes)
  val dividerQuotient = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0), cfg.tokenLanes)
  val dividerIteration = Reg(UInt(cfg.dividerCountWidth bits)) init (0)

  val dividerDividendNext = Vec(UInt(cfg.dividerWidth bits), cfg.tokenLanes)
  val dividerRemainderNext =
    Vec(UInt((cfg.dividerWidth + 1) bits), cfg.tokenLanes)
  val dividerQuotientNext = Vec(UInt(cfg.dividerWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val shiftedRemainder = UInt((cfg.dividerWidth + 1) bits)
    shiftedRemainder := (
      dividerRemainder(lane)(cfg.dividerWidth - 1 downto 0) ##
        dividerDividend(lane).msb
    ).asUInt
    val canSubtract = shiftedRemainder >=
      dividerDivisor(lane).resize(cfg.dividerWidth + 1)
    dividerRemainderNext(lane) := shiftedRemainder
    when(canSubtract) {
      dividerRemainderNext(lane) := (
        shiftedRemainder - dividerDivisor(lane).resize(cfg.dividerWidth + 1)
      ).resize(cfg.dividerWidth + 1)
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

  val reciprocal = Vec(
    Reg(UInt(cfg.reciprocalWidth bits)) init (0), cfg.tokenLanes)
  val dequantFactor = Vec(
    Reg(UInt(cfg.factorWidth bits)) init (0), cfg.tokenLanes)

  val replayReadIndex = Reg(UInt(cfg.indexWidth bits)) init (0)
  val allReplayReadsIssued = RegInit(False)

  switch(state) {
    is(State.Collect) {
      // Collection and framing recovery are handled above.
    }

    is(State.ScalePrepare) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := U(
          BigInt(127) << cfg.reciprocalFractionBits,
          cfg.dividerWidth bits
        )
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
        // maxAbs is already a Q16.16 raw integer.  Adding 127/2 implements
        // nearest rounding for the returned Q16.16 factor (127 is odd).
        dividerDividend(lane) := (
          maxAbs(lane).resize(cfg.dividerWidth) + U(63, cfg.dividerWidth bits)
        ).resize(cfg.dividerWidth)
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
          dequantFactor(lane) :=
            dividerQuotientNext(lane).resize(cfg.factorWidth)
          when(maxAbs(lane) === 0) {
            dequantFactor(lane) :=
              U(BigInt(1) << cfg.inputFractionBits, cfg.factorWidth bits)
          }
        }
        replayReadIndex := 0
        allReplayReadsIssued := False
        state := State.Replay
      } otherwise {
        dividerIteration := dividerIteration + 1
      }
    }

    is(State.Replay) {
      // RAM issue and pipeline drain are handled below.
    }
  }

  val replayFifo = new StreamFifo(
    NativeDynamicQuantReplay(cfg), cfg.replayFifoDepth)
  val replayReadIssue = state === State.Replay &&
    !allReplayReadsIssued && replayFifo.io.availability > 1
  val replayReadData = storage.readSync(replayReadIndex, replayReadIssue)
  val replayResponseValid = RegNext(replayReadIssue) init (False)
  val replayResponseIndex =
    RegNextWhen(replayReadIndex, replayReadIssue) init (0)

  replayFifo.io.push.valid := replayResponseValid
  replayFifo.io.push.data := replayReadData
  replayFifo.io.push.index := replayResponseIndex

  when(replayReadIssue) {
    when(replayReadIndex === U(cfg.vectorLength - 1, cfg.indexWidth bits)) {
      allReplayReadsIssued := True
    } otherwise {
      replayReadIndex := replayReadIndex + 1
    }
  }

  val replayValues = Vec(SInt(cfg.inputWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    replayValues(lane) := replayFifo.io.pop.data(
      (lane + 1) * cfg.inputWidth - 1 downto lane * cfg.inputWidth
    ).asSInt
  }

  val quantized = Stream(NativeDynamicQuantResult(cfg))
  quantized.valid := replayFifo.io.pop.valid
  replayFifo.io.pop.ready := quantized.ready
  quantized.tokenMask := activeTokenMask
  quantized.first := replayFifo.io.pop.index === 0
  quantized.last := replayFifo.io.pop.index ===
    U(cfg.vectorLength - 1, cfg.indexWidth bits)

  for (lane <- 0 until cfg.tokenLanes) {
    quantized.dequantFactor(lane) := dequantFactor(lane)
    quantized.value(lane) := 0

    val magnitude = magnitudeOf(replayValues(lane))
    val reciprocalProduct =
      (magnitude * reciprocal(lane)).resize(cfg.reciprocalProductWidth)
    reciprocalProduct.addAttribute("use_dsp", "yes")
    val biasedProduct = (
      reciprocalProduct.resize(cfg.reciprocalProductWidth + 1) +
        U(BigInt(1) << (cfg.reciprocalFractionBits - 1),
          (cfg.reciprocalProductWidth + 1) bits)
    ).resize(cfg.reciprocalProductWidth + 1)
    val provisionalWide = biasedProduct |>> cfg.reciprocalFractionBits
    val provisional = UInt(8 bits)
    when(provisionalWide > U(127, provisionalWide.getWidth bits)) {
      provisional := 127
    } otherwise {
      provisional := provisionalWide.resize(8)
    }

    // Exact correction makes the reciprocal's finite precision invisible.
    // For candidate c, c is correct iff
    //   (2c-1)*max <= 2*abs*127 < (2c+1)*max.
    val exactNumerator =
      (magnitude * U(127, 8 bits)).resize(cfg.exactProductWidth)
    val twiceNumerator =
      (exactNumerator |<< 1).resize(cfg.exactProductWidth)
    val lowerCoefficient = UInt(9 bits)
    lowerCoefficient := 0
    when(provisional =/= 0) {
      lowerCoefficient :=
        ((provisional.resize(9) |<< 1) - 1).resize(9)
    }
    val upperCoefficient =
      ((provisional.resize(9) |<< 1) + 1).resize(9)
    val lowerThreshold =
      (maxAbs(lane) * lowerCoefficient).resize(cfg.exactProductWidth)
    val upperThreshold =
      (maxAbs(lane) * upperCoefficient).resize(cfg.exactProductWidth)
    val correctedMagnitude = UInt(8 bits)
    correctedMagnitude := provisional
    when(maxAbs(lane) =/= 0 && provisional =/= 0 &&
        twiceNumerator < lowerThreshold) {
      correctedMagnitude := provisional - 1
    } elsewhen (maxAbs(lane) =/= 0 && provisional < 127 &&
        twiceNumerator >= upperThreshold) {
      correctedMagnitude := provisional + 1
    }

    val signedCode = SInt(9 bits)
    when(replayValues(lane).msb) {
      signedCode := (-correctedMagnitude.resize(9).asSInt).resize(9)
    } otherwise {
      signedCode := correctedMagnitude.resize(9).asSInt
    }

    when(activeTokenMask(lane)) {
      when(signedCode > S(127, 9 bits)) {
        quantized.value(lane) := 127
      } elsewhen (signedCode < S(-128, 9 bits)) {
        quantized.value(lane) := -128
      } otherwise {
        quantized.value(lane) := signedCode.resize(8)
      }
    }
  }

  io.output << quantized.m2sPipe()

  when(io.output.fire && io.output.last) {
    state := State.Collect
    emittedVectorCounter := emittedVectorCounter + 1
  }

  io.busy := state =/= State.Collect || collecting ||
    replayFifo.io.occupancy =/= 0 || quantized.valid || io.output.valid
}

case class NativeDynamicQuantizedVector(
    value: Seq[Seq[Int]],
    dequantFactorQ16: Seq[Long]
)

/** Bit-accurate architectural oracle for [[NativeDynamicI8Quantizer]]. */
object NativeDynamicI8QuantizerMath {
  val OneQ16: Long = 1L << 16

  private def magnitude(value: Int): BigInt = BigInt(value).abs

  private def exactCode(value: Int, maxAbs: BigInt): Int = {
    if (maxAbs == 0) {
      0
    } else {
      val numerator = magnitude(value) * 127
      val base = numerator / maxAbs
      val remainder = numerator % maxAbs
      val rounded = if (remainder * 2 >= maxAbs) base + 1 else base
      val signed = if (value < 0) -rounded else rounded
      signed.max(BigInt(-128)).min(BigInt(127)).toInt
    }
  }

  def quantize(tokens: Seq[Seq[Int]]): NativeDynamicQuantizedVector = {
    require(tokens.nonEmpty && tokens.head.nonEmpty)
    require(tokens.forall(_.length == tokens.head.length))
    val maxima = tokens.map(token => token.map(magnitude).max)
    val factors = maxima.map { maximum =>
      if (maximum == 0) OneQ16
      else ((maximum + 63) / 127).toLong
    }
    val values = tokens.zip(maxima).map { case (token, maximum) =>
      token.map(value => exactCode(value, maximum))
    }
    NativeDynamicQuantizedVector(values, factors)
  }

  /** Maximum absolute Q16.16 factor error versus the real max_abs/127. */
  def factorErrorRaw(maxAbsRaw: BigInt, factorRaw: Long): BigDecimal = {
    if (maxAbsRaw == 0) BigDecimal(0)
    else (BigDecimal(factorRaw) - BigDecimal(maxAbsRaw) / BigDecimal(127)).abs
  }
}

object GenerateNativeDynamicI8QuantizerHidden extends App {
  SpinalConfig(
    targetDirectory = "target/generated-bitnet-dynamic-i8-hidden",
    oneFilePerComponent = false
  ).generateVerilog(new NativeDynamicI8Quantizer(
    NativeDynamicI8QuantizerConfig.hidden()))
}

object GenerateNativeDynamicI8QuantizerFfn extends App {
  SpinalConfig(
    targetDirectory = "target/generated-bitnet-dynamic-i8-ffn",
    oneFilePerComponent = false
  ).generateVerilog(new NativeDynamicI8Quantizer(
    NativeDynamicI8QuantizerConfig.ffn()))
}

object GenerateNativeDynamicI8QuantizerHead extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-dynamic-i8-head")
  SpinalConfig(
    targetDirectory = target,
    oneFilePerComponent = false
  ).generateVerilog(new NativeDynamicI8Quantizer(
    NativeDynamicI8QuantizerConfig(tokenLanes = 4, vectorLength = 128)))
}
