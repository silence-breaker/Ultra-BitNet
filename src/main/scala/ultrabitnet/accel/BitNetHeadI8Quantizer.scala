package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** Scalar replay contract used by the native Q/K/V path.
  *
  * BitNet quantises a 128-element head through one shared conversion
  * pipeline.  Keeping the lane and element coordinates on the stream lets the
  * KV writer, RoPE path and score path consume the result without rebuilding a
  * four-lane packet or writing a QKV intermediate to DDR.
  */
case class BitNetHeadQuantSample(cfg: NativeDynamicI8QuantizerConfig)
    extends Bundle {
  val value = SInt(8 bits)
  val dequantFactor = UInt(cfg.factorWidth bits)
  val tokenLane = UInt(log2Up(cfg.tokenLanes) bits)
  val element = UInt(cfg.indexWidth bits)
  val tokenActive = Bool()
  val first = Bool()
  val last = Bool()
}

case class BitNetHeadQuantOperand(cfg: NativeDynamicI8QuantizerConfig)
    extends Bundle {
  val magnitude = UInt(cfg.magnitudeWidth bits)
  val reciprocal = UInt(cfg.reciprocalWidth bits)
  val maximum = UInt(cfg.magnitudeWidth bits)
  val dequantFactor = UInt(cfg.factorWidth bits)
  val negative = Bool()
  val tokenLane = UInt(log2Up(cfg.tokenLanes) bits)
  val element = UInt(cfg.indexWidth bits)
  val tokenActive = Bool()
}

case class BitNetHeadQuantPartial(cfg: NativeDynamicI8QuantizerConfig)
    extends Bundle {
  val lowLow = UInt(35 bits)
  val lowHigh = UInt(40 bits)
  val highLow = UInt(34 bits)
  val highHigh = UInt(39 bits)
  val magnitude = UInt(cfg.magnitudeWidth bits)
  val maximum = UInt(cfg.magnitudeWidth bits)
  val dequantFactor = UInt(cfg.factorWidth bits)
  val negative = Bool()
  val tokenLane = UInt(log2Up(cfg.tokenLanes) bits)
  val element = UInt(cfg.indexWidth bits)
  val tokenActive = Bool()
}

case class BitNetHeadQuantSum(cfg: NativeDynamicI8QuantizerConfig)
    extends Bundle {
  val even = UInt(cfg.reciprocalProductWidth bits)
  val odd = UInt(cfg.reciprocalProductWidth bits)
  val magnitude = UInt(cfg.magnitudeWidth bits)
  val maximum = UInt(cfg.magnitudeWidth bits)
  val dequantFactor = UInt(cfg.factorWidth bits)
  val negative = Bool()
  val tokenLane = UInt(log2Up(cfg.tokenLanes) bits)
  val element = UInt(cfg.indexWidth bits)
  val tokenActive = Bool()
}

case class BitNetHeadQuantMul(cfg: NativeDynamicI8QuantizerConfig)
    extends Bundle {
  val product = UInt(cfg.reciprocalProductWidth bits)
  val magnitude = UInt(cfg.magnitudeWidth bits)
  val maximum = UInt(cfg.magnitudeWidth bits)
  val dequantFactor = UInt(cfg.factorWidth bits)
  val negative = Bool()
  val tokenLane = UInt(log2Up(cfg.tokenLanes) bits)
  val element = UInt(cfg.indexWidth bits)
  val tokenActive = Bool()
}

case class BitNetHeadQuantCandidate(
    cfg: NativeDynamicI8QuantizerConfig
) extends Bundle {
  val candidate = UInt(8 bits)
  val twiceNumerator = UInt(cfg.exactProductWidth bits)
  val maximum = UInt(cfg.magnitudeWidth bits)
  val dequantFactor = UInt(cfg.factorWidth bits)
  val negative = Bool()
  val tokenLane = UInt(log2Up(cfg.tokenLanes) bits)
  val element = UInt(cfg.indexWidth bits)
  val tokenActive = Bool()
}

case class BitNetHeadQuantCoefficient(
    cfg: NativeDynamicI8QuantizerConfig
) extends Bundle {
  val candidate = UInt(8 bits)
  val lowerCoefficient = UInt(9 bits)
  val twiceNumerator = UInt(cfg.exactProductWidth bits)
  val maximum = UInt(cfg.magnitudeWidth bits)
  val dequantFactor = UInt(cfg.factorWidth bits)
  val negative = Bool()
  val tokenLane = UInt(log2Up(cfg.tokenLanes) bits)
  val element = UInt(cfg.indexWidth bits)
  val tokenActive = Bool()
}

case class BitNetHeadQuantThreshold(
    cfg: NativeDynamicI8QuantizerConfig
) extends Bundle {
  val candidate = UInt(8 bits)
  val twiceNumerator = UInt(cfg.exactProductWidth bits)
  val lowerThreshold = UInt(cfg.exactProductWidth bits)
  val upperThreshold = UInt(cfg.exactProductWidth bits)
  val maximum = UInt(cfg.magnitudeWidth bits)
  val dequantFactor = UInt(cfg.factorWidth bits)
  val negative = Bool()
  val tokenLane = UInt(log2Up(cfg.tokenLanes) bits)
  val element = UInt(cfg.indexWidth bits)
  val tokenActive = Bool()
}

/** BitNet, head-local dynamic i8 quantiser.
  *
  * The generic TP4 quantiser evaluated four wide reciprocal/correction paths
  * in parallel.  That cost 32 DSPs and put a 38-level path between RAM and the
  * output.  A 128-dimensional Q/K/V head is memory-bandwidth limited, so this
  * production implementation retains the four-token max/divider bank but
  * replays one token lane per cycle through a single pipelined arithmetic
  * path.  It therefore emits 512 scalar samples for a full TP4 head while the
  * next matrix head can be overlapped by the surrounding ping-pong scheduler.
  *
  * Quantisation remains bit exact:
  *
  *   sign(x) * round_away(abs(x) * 127 / max_abs)
  *
  * A Q33 reciprocal proposes a code.  One exact lower threshold multiply is
  * shared for all lanes; the upper threshold is derived by adding 2*max_abs.
  * Every arithmetic boundary is registered and honours Stream backpressure.
  */
class BitNetHeadI8Quantizer(
    cfg: NativeDynamicI8QuantizerConfig = NativeDynamicI8QuantizerConfig(
      tokenLanes = 4,
      vectorLength = 128
    )
) extends Component {
  require(cfg.tokenLanes == 2 || cfg.tokenLanes == 4,
    "the head quantiser supports the TP2/TP4 BitNet replay paths")
  require(cfg.magnitudeWidth == 33 && cfg.reciprocalWidth == 41,
    "the registered DSP decomposition implements the Q16.16/Q33 contract")

  val io = new Bundle {
    val input = slave(Stream(NativeDynamicQuantBeat(cfg)))
    val output = master(Stream(BitNetHeadQuantSample(cfg)))
    val busy = out Bool()
    val protocolFault = out Bool()
    val acceptedVectors = out UInt(32 bits)
    val emittedVectors = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Collect, ReciprocalPrepare, ReciprocalDivide, FactorPrepare,
        FactorDivide, Replay = newElement()
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

  io.protocolFault := fault
  io.acceptedVectors := acceptedVectorCounter
  io.emittedVectors := emittedVectorCounter
  io.input.ready := state === State.Collect

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

  val packedInput = Bits(cfg.replayEntryWidth bits)
  val inputMagnitude = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    packedInput((lane + 1) * cfg.inputWidth - 1 downto lane * cfg.inputWidth) :=
      io.input.value(lane).asBits
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
      // Mem.write's explicit enable is not inherited reliably from the
      // surrounding `when` scope during Verilog lowering.  Keep the actual
      // Stream handshake on the port or the final address is overwritten
      // while the divider/replay phases run and upstream payload changes.
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
      when(io.input.last =/= acceptedAtEnd) { fault := True }

      when(acceptedAtEnd) {
        collecting := False
        when(io.input.last) {
          state := State.ReciprocalPrepare
          acceptedVectorCounter := acceptedVectorCounter + 1
        }
      } elsewhen (io.input.last) {
        collecting := False
      } otherwise {
        collectIndex := acceptedIndex + 1
      }
    }
  }

  // Four small restoring dividers run only once per head.  They use LUT carry
  // chains, not DSPs, and are shared between reciprocal and factor creation.
  val dividerDividend = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0), cfg.tokenLanes)
  val dividerDivisor = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0), cfg.tokenLanes)
  val dividerRemainder = Vec(
    Reg(UInt((cfg.dividerWidth + 1) bits)) init (0), cfg.tokenLanes)
  val dividerQuotient = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0), cfg.tokenLanes)
  val dividerIteration = Reg(UInt(cfg.dividerCountWidth bits)) init (0)

  val dividendNext = Vec(UInt(cfg.dividerWidth bits), cfg.tokenLanes)
  val remainderNext = Vec(UInt((cfg.dividerWidth + 1) bits), cfg.tokenLanes)
  val quotientNext = Vec(UInt(cfg.dividerWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val shiftedRemainder = UInt((cfg.dividerWidth + 1) bits)
    shiftedRemainder := (
      dividerRemainder(lane)(cfg.dividerWidth - 1 downto 0) ##
        dividerDividend(lane).msb
    ).asUInt
    val subtract = shiftedRemainder >=
      dividerDivisor(lane).resize(cfg.dividerWidth + 1)
    remainderNext(lane) := shiftedRemainder
    when(subtract) {
      remainderNext(lane) := (
        shiftedRemainder - dividerDivisor(lane).resize(cfg.dividerWidth + 1)
      ).resize(cfg.dividerWidth + 1)
    }
    dividendNext(lane) :=
      (dividerDividend(lane) |<< 1).resize(cfg.dividerWidth)
    val shiftedQuotient =
      (dividerQuotient(lane) |<< 1).resize(cfg.dividerWidth)
    quotientNext(lane) := shiftedQuotient
    when(subtract) {
      quotientNext(lane) := shiftedQuotient |
        U(1, cfg.dividerWidth bits)
    }
  }

  val reciprocal = Vec(
    Reg(UInt(cfg.reciprocalWidth bits)) init (0), cfg.tokenLanes)
  val dequantFactor = Vec(
    Reg(UInt(cfg.factorWidth bits)) init (0), cfg.tokenLanes)
  val replayReadIndex = Reg(UInt(cfg.indexWidth bits)) init (0)
  val allReplayReadsIssued = RegInit(False)

  switch(state) {
    is(State.Collect) {}

    is(State.ReciprocalPrepare) {
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
      state := State.ReciprocalDivide
    }

    is(State.ReciprocalDivide) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := dividendNext(lane)
        dividerRemainder(lane) := remainderNext(lane)
        dividerQuotient(lane) := quotientNext(lane)
      }
      when(dividerIteration ===
          U(cfg.dividerWidth - 1, cfg.dividerCountWidth bits)) {
        for (lane <- 0 until cfg.tokenLanes) {
          reciprocal(lane) := quotientNext(lane)
          when(maxAbs(lane) === 0) { reciprocal(lane) := 0 }
        }
        state := State.FactorPrepare
      } otherwise {
        dividerIteration := dividerIteration + 1
      }
    }

    is(State.FactorPrepare) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := (
          maxAbs(lane).resize(cfg.dividerWidth) +
            U(63, cfg.dividerWidth bits)
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
        dividerDividend(lane) := dividendNext(lane)
        dividerRemainder(lane) := remainderNext(lane)
        dividerQuotient(lane) := quotientNext(lane)
      }
      when(dividerIteration ===
          U(cfg.dividerWidth - 1, cfg.dividerCountWidth bits)) {
        for (lane <- 0 until cfg.tokenLanes) {
          dequantFactor(lane) := quotientNext(lane).resize(cfg.factorWidth)
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

    is(State.Replay) {}
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

  val replayLane = Reg(UInt(log2Up(cfg.tokenLanes) bits)) init (0)
  val replayValues = Vec(SInt(cfg.inputWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    replayValues(lane) := replayFifo.io.pop.data(
      (lane + 1) * cfg.inputWidth - 1 downto lane * cfg.inputWidth
    ).asSInt
  }

  val issue = Stream(BitNetHeadQuantOperand(cfg))
  issue.valid := state === State.Replay && replayFifo.io.pop.valid
  replayFifo.io.pop.ready := issue.fire &&
    replayLane === U(cfg.tokenLanes - 1, replayLane.getWidth bits)

  val selectedValue = SInt(cfg.inputWidth bits)
  val selectedMaximum = UInt(cfg.magnitudeWidth bits)
  val selectedReciprocal = UInt(cfg.reciprocalWidth bits)
  val selectedFactor = UInt(cfg.factorWidth bits)
  val selectedActive = Bool()
  selectedValue := replayValues(replayLane)
  selectedMaximum := maxAbs(replayLane)
  selectedReciprocal := reciprocal(replayLane)
  selectedFactor := dequantFactor(replayLane)
  selectedActive := activeTokenMask(replayLane)
  val selectedMagnitude = magnitudeOf(selectedValue)
  issue.magnitude := selectedMagnitude
  issue.reciprocal := selectedReciprocal
  issue.maximum := selectedMaximum
  issue.dequantFactor := selectedFactor
  issue.negative := selectedValue.msb
  issue.tokenLane := replayLane
  issue.element := replayFifo.io.pop.index
  issue.tokenActive := selectedActive

  when(issue.fire) {
    when(replayLane === U(cfg.tokenLanes - 1, replayLane.getWidth bits)) {
      replayLane := 0
    } otherwise {
      replayLane := replayLane + 1
    }
  }

  // A monolithic unsigned 33x41 multiply is four cascaded DSP48s and does not
  // close at 266 MHz.  Register the selected operands, form four independent
  // <=23x17 partial products (one DSP48 each), register them, then use a
  // two-level registered adder tree.  The arithmetic is identical and still
  // accepts one scalar every clock.
  val operandReg = issue.m2sPipe()
  val partial = Stream(BitNetHeadQuantPartial(cfg))
  partial.valid := operandReg.valid
  operandReg.ready := partial.ready
  val magnitudeLow = operandReg.magnitude(16 downto 0)
  val magnitudeHigh = operandReg.magnitude(32 downto 17)
  val reciprocalLow = operandReg.reciprocal(17 downto 0)
  val reciprocalHigh = operandReg.reciprocal(40 downto 18)
  val lowLow = (magnitudeLow * reciprocalLow).resize(35)
  val lowHigh = (magnitudeLow * reciprocalHigh).resize(40)
  val highLow = (magnitudeHigh * reciprocalLow).resize(34)
  val highHigh = (magnitudeHigh * reciprocalHigh).resize(39)
  lowLow.addAttribute("use_dsp", "yes")
  lowHigh.addAttribute("use_dsp", "yes")
  highLow.addAttribute("use_dsp", "yes")
  highHigh.addAttribute("use_dsp", "yes")
  partial.lowLow := lowLow
  partial.lowHigh := lowHigh
  partial.highLow := highLow
  partial.highHigh := highHigh
  partial.magnitude := operandReg.magnitude
  partial.maximum := operandReg.maximum
  partial.dequantFactor := operandReg.dequantFactor
  partial.negative := operandReg.negative
  partial.tokenLane := operandReg.tokenLane
  partial.element := operandReg.element
  partial.tokenActive := operandReg.tokenActive

  val partialReg = partial.m2sPipe()
  val sum = Stream(BitNetHeadQuantSum(cfg))
  sum.valid := partialReg.valid
  partialReg.ready := sum.ready
  sum.even := (
    partialReg.lowLow.resize(cfg.reciprocalProductWidth + 1) +
      (partialReg.highHigh.resize(cfg.reciprocalProductWidth + 1) |<< 35)
  ).resize(cfg.reciprocalProductWidth)
  sum.odd := (
    (partialReg.lowHigh.resize(cfg.reciprocalProductWidth + 1) |<< 18) +
      (partialReg.highLow.resize(cfg.reciprocalProductWidth + 1) |<< 17)
  ).resize(cfg.reciprocalProductWidth)
  sum.magnitude := partialReg.magnitude
  sum.maximum := partialReg.maximum
  sum.dequantFactor := partialReg.dequantFactor
  sum.negative := partialReg.negative
  sum.tokenLane := partialReg.tokenLane
  sum.element := partialReg.element
  sum.tokenActive := partialReg.tokenActive

  val sumReg = sum.m2sPipe()
  val product = Stream(BitNetHeadQuantMul(cfg))
  product.valid := sumReg.valid
  sumReg.ready := product.ready
  product.product := (
    sumReg.even.resize(cfg.reciprocalProductWidth + 1) +
      sumReg.odd.resize(cfg.reciprocalProductWidth + 1)
  ).resize(cfg.reciprocalProductWidth)
  product.magnitude := sumReg.magnitude
  product.maximum := sumReg.maximum
  product.dequantFactor := sumReg.dequantFactor
  product.negative := sumReg.negative
  product.tokenLane := sumReg.tokenLane
  product.element := sumReg.element
  product.tokenActive := sumReg.tokenActive

  val multiplied = product.m2sPipe()
  val candidate = Stream(BitNetHeadQuantCandidate(cfg))
  candidate.valid := multiplied.valid
  multiplied.ready := candidate.ready
  val biasedProduct = (
    multiplied.product.resize(cfg.reciprocalProductWidth + 1) +
      U(BigInt(1) << (cfg.reciprocalFractionBits - 1),
        (cfg.reciprocalProductWidth + 1) bits)
  ).resize(cfg.reciprocalProductWidth + 1)
  val proposedWide = biasedProduct |>> cfg.reciprocalFractionBits
  candidate.candidate := proposedWide.resize(8)
  when(proposedWide > U(127, proposedWide.getWidth bits)) {
    candidate.candidate := 127
  }
  val numerator = (
    multiplied.magnitude.resize(cfg.exactProductWidth) *
      U(127, 7 bits)
  ).resize(cfg.exactProductWidth)
  numerator.addAttribute("use_dsp", "no")
  candidate.twiceNumerator :=
    (numerator |<< 1).resize(cfg.exactProductWidth)
  candidate.maximum := multiplied.maximum
  candidate.dequantFactor := multiplied.dequantFactor
  candidate.negative := multiplied.negative
  candidate.tokenLane := multiplied.tokenLane
  candidate.element := multiplied.element
  candidate.tokenActive := multiplied.tokenActive

  val candidateReg = candidate.m2sPipe()
  val coefficient = Stream(BitNetHeadQuantCoefficient(cfg))
  coefficient.valid := candidateReg.valid
  candidateReg.ready := coefficient.ready
  val lowerCoefficient = UInt(9 bits)
  lowerCoefficient := 0
  when(candidateReg.candidate =/= 0) {
    lowerCoefficient :=
      ((candidateReg.candidate.resize(9) |<< 1) - 1).resize(9)
  }
  coefficient.candidate := candidateReg.candidate
  coefficient.lowerCoefficient := lowerCoefficient
  coefficient.twiceNumerator := candidateReg.twiceNumerator
  coefficient.maximum := candidateReg.maximum
  coefficient.dequantFactor := candidateReg.dequantFactor
  coefficient.negative := candidateReg.negative
  coefficient.tokenLane := candidateReg.tokenLane
  coefficient.element := candidateReg.element
  coefficient.tokenActive := candidateReg.tokenActive

  // Keep coefficient generation, the cascaded 33x9 multiply and the 42-bit
  // upper-threshold add on separate clock boundaries.  This is deliberately
  // deeper than the generic quantiser: throughput remains one scalar/cycle,
  // while each DSP stage has a realistic 266 MHz OOC path.
  val coefficientReg = coefficient.m2sPipe()
  val lower = Stream(BitNetHeadQuantThreshold(cfg))
  lower.valid := coefficientReg.valid
  coefficientReg.ready := lower.ready
  val lowerProduct =
    (coefficientReg.maximum * coefficientReg.lowerCoefficient).resize(
      cfg.exactProductWidth)
  lowerProduct.addAttribute("use_dsp", "yes")
  lower.lowerThreshold := lowerProduct
  lower.upperThreshold := 0
  lower.maximum := coefficientReg.maximum
  lower.candidate := coefficientReg.candidate
  lower.twiceNumerator := coefficientReg.twiceNumerator
  lower.dequantFactor := coefficientReg.dequantFactor
  lower.negative := coefficientReg.negative
  lower.tokenLane := coefficientReg.tokenLane
  lower.element := coefficientReg.element
  lower.tokenActive := coefficientReg.tokenActive

  val lowerReg = lower.m2sPipe()
  val threshold = Stream(BitNetHeadQuantThreshold(cfg))
  threshold.valid := lowerReg.valid
  lowerReg.ready := threshold.ready
  threshold.lowerThreshold := lowerReg.lowerThreshold
  threshold.upperThreshold := (
    lowerReg.lowerThreshold.resize(cfg.exactProductWidth + 1) +
      (lowerReg.maximum.resize(cfg.exactProductWidth + 1) |<< 1)
  ).resize(cfg.exactProductWidth)
  when(lowerReg.candidate === 0) {
    threshold.upperThreshold :=
      lowerReg.maximum.resize(cfg.exactProductWidth)
  }
  threshold.maximum := lowerReg.maximum
  threshold.candidate := lowerReg.candidate
  threshold.twiceNumerator := lowerReg.twiceNumerator
  threshold.dequantFactor := lowerReg.dequantFactor
  threshold.negative := lowerReg.negative
  threshold.tokenLane := lowerReg.tokenLane
  threshold.element := lowerReg.element
  threshold.tokenActive := lowerReg.tokenActive

  val thresholdReg = threshold.m2sPipe()
  val result = Stream(BitNetHeadQuantSample(cfg))
  result.valid := thresholdReg.valid
  thresholdReg.ready := result.ready
  val corrected = UInt(8 bits)
  corrected := thresholdReg.candidate
  when(thresholdReg.upperThreshold =/= 0 &&
      thresholdReg.candidate =/= 0 &&
      thresholdReg.twiceNumerator < thresholdReg.lowerThreshold) {
    corrected := thresholdReg.candidate - 1
  } elsewhen (thresholdReg.upperThreshold =/= 0 &&
      thresholdReg.candidate < 127 &&
      thresholdReg.twiceNumerator >= thresholdReg.upperThreshold) {
    corrected := thresholdReg.candidate + 1
  }
  val signedCode = SInt(9 bits)
  when(thresholdReg.negative) {
    signedCode := (-corrected.resize(9).asSInt).resize(9)
  } otherwise {
    signedCode := corrected.resize(9).asSInt
  }
  result.value := signedCode.resize(8)
  when(!thresholdReg.tokenActive) { result.value := 0 }
  result.dequantFactor := thresholdReg.dequantFactor
  result.tokenLane := thresholdReg.tokenLane
  result.element := thresholdReg.element
  result.tokenActive := thresholdReg.tokenActive
  result.first := thresholdReg.element === 0 && thresholdReg.tokenLane === 0
  result.last := thresholdReg.element ===
    U(cfg.vectorLength - 1, cfg.indexWidth bits) &&
    thresholdReg.tokenLane === U(cfg.tokenLanes - 1,
      thresholdReg.tokenLane.getWidth bits)

  io.output << result.m2sPipe()
  when(io.output.fire && io.output.last) {
    state := State.Collect
    emittedVectorCounter := emittedVectorCounter + 1
  }

  io.busy := state =/= State.Collect || collecting ||
    replayFifo.io.occupancy =/= 0 || issue.valid || operandReg.valid ||
    partialReg.valid || sumReg.valid || multiplied.valid || candidateReg.valid ||
    coefficientReg.valid || lowerReg.valid ||
    thresholdReg.valid || result.valid || io.output.valid
}
