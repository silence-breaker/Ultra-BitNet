package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Fixed-point shape of the packet-free GQA online-attention prototype.
  *
  * One KV head is shared by `queryHeadsPerKv` query heads (four in BitNet).
  * A V vector is streamed one scalar at a time.  `firstValue`/`lastValue`
  * delimit the vector only; there is no packet header, route word, uop, or
  * score/probability scratch buffer.  Scores are signed Q4.12 and V/context
  * values are signed Q16.16.
  *
  * For every causally-valid key the core maintains, independently per query
  * head, the stable online-softmax recurrence
  *
  *   m' = max(m, score)
  *   l' = l * exp(m-m') + exp(score-m')
  *   o' = o * exp(m-m') + V * exp(score-m')
  *
  * `l` is Q*.15 and `o` is signed Q*.31.  Every rescale is rounded to nearest,
  * with exact ties away from zero.  Final Q16.16 context is produced by one
  * reciprocal per query head followed by a streamed multiply; neither scores
  * nor probabilities are materialized.
  *
  * The exp prototype quantizes a non-positive delta to the nearest 1/16 and
  * returns round(exp(delta) * 2^15).  Deltas below -8 become zero.  Thus a
  * single LUT result has at most roughly 0.03127 absolute error from input
  * quantization plus 0.5 Q1.15 LSB; the discarded tail is below 0.000336.
  * This 129-entry combinational LUT is intentionally an arithmetic prototype.
  * A replicated/BRAM-banked LUT or polynomial SPU implementation remains a
  * physical-design choice, without changing this stream contract or oracle.
  * Likewise, the two logical asynchronous context reads make the recurrence
  * easy to verify but are not a BRAM inference promise; the production SPU
  * should bank or time-multiplex this 4 x 128 x 64-bit state after timing and
  * resource exploration.  The generated RTL is synthesizable, while resource
  * closure of those two structures is deliberately outside this prototype.
  */
case class NativeOnlineAttentionConfig(
    queryHeadsPerKv: Int = 4,
    headDim: Int = 128,
    positionWidth: Int = 15,
    scoreWidth: Int = 16,
    scoreFractionalBits: Int = 12,
    valueWidth: Int = 32,
    valueFractionalBits: Int = 16,
    expWidth: Int = 16,
    expFractionalBits: Int = 15,
    sumWidth: Int = 48,
    contextWidth: Int = 64,
    reciprocalWidth: Int = 48,
    reciprocalFractionalBits: Int = 46
) {
  require(queryHeadsPerKv > 0 && isPow2(queryHeadsPerKv))
  require(headDim >= 2 && isPow2(headDim))
  require(positionWidth > 0)
  require(scoreWidth == 16 && scoreFractionalBits == 12,
    "the online-attention score contract is signed Q4.12")
  require(valueWidth == 32 && valueFractionalBits == 16,
    "the native activation contract is signed Q16.16")
  require(expWidth == 16 && expFractionalBits == 15,
    "the exp LUT contract is unsigned Q1.15")
  require(sumWidth >= positionWidth + expFractionalBits + 1)
  require(contextWidth >= valueWidth + positionWidth + expFractionalBits)
  require(reciprocalWidth > reciprocalFractionalBits)

  val valueIndexWidth: Int = log2Up(headDim)
  val dividerWidth: Int = 64
  val dividerCountWidth: Int = log2Up(dividerWidth)
  val expOne: Int = 1 << expFractionalBits
}

case class NativeOnlineAttentionBeat(cfg: NativeOnlineAttentionConfig)
    extends Bundle {
  val score = Vec(SInt(cfg.scoreWidth bits), cfg.queryHeadsPerKv)
  val value = SInt(cfg.valueWidth bits)
  val valueIndex = UInt(cfg.valueIndexWidth bits)

  val queryPosition = UInt(cfg.positionWidth bits)
  val keyPosition = UInt(cfg.positionWidth bits)
  val queryMask = Bits(cfg.queryHeadsPerKv bits)
  val positionValid = Bool()

  val firstKey = Bool()
  val lastKey = Bool()
  val firstValue = Bool()
  val lastValue = Bool()
}

case class NativeOnlineAttentionResult(cfg: NativeOnlineAttentionConfig)
    extends Bundle {
  val value = Vec(SInt(cfg.valueWidth bits), cfg.queryHeadsPerKv)
  val valueIndex = UInt(cfg.valueIndexWidth bits)
  val queryMask = Bits(cfg.queryHeadsPerKv bits)
  val allMasked = Bits(cfg.queryHeadsPerKv bits)
  val first = Bool()
  val last = Bool()
}

class NativeOnlineAttention(
    cfg: NativeOnlineAttentionConfig = NativeOnlineAttentionConfig()
) extends Component {
  val io = new Bundle {
    val input = slave(Stream(NativeOnlineAttentionBeat(cfg)))
    val output = master(Stream(NativeOnlineAttentionResult(cfg)))
    val busy = out Bool()
    val protocolFault = out Bool()
    val acceptedSequences = out UInt(32 bits)
    val emittedSequences = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Collect, ReciprocalPrepare, ReciprocalDivide, Emit = newElement()
  }
  val state = Reg(State()) init State.Collect

  val maxScore = Vec(
    Reg(SInt(cfg.scoreWidth bits)) init S(-(BigInt(1) << (cfg.scoreWidth - 1)),
      cfg.scoreWidth bits),
    cfg.queryHeadsPerKv
  )
  val sum = Vec(Reg(UInt(cfg.sumWidth bits)) init 0, cfg.queryHeadsPerKv)
  val hasValid = Reg(Bits(cfg.queryHeadsPerKv bits)) init 0
  val alpha = Vec(Reg(UInt(cfg.expWidth bits)) init 0, cfg.queryHeadsPerKv)
  val beta = Vec(Reg(UInt(cfg.expWidth bits)) init 0, cfg.queryHeadsPerKv)

  val context = Array.fill(cfg.queryHeadsPerKv)(
    Mem(SInt(cfg.contextWidth bits), cfg.headDim)
  )

  val sequenceActive = RegInit(False)
  val activeQueryMask = Reg(Bits(cfg.queryHeadsPerKv bits)) init 0
  val expectedValueIndex = Reg(UInt(cfg.valueIndexWidth bits)) init 0
  val activeQueryPosition = Reg(UInt(cfg.positionWidth bits)) init 0
  val activeKeyPosition = Reg(UInt(cfg.positionWidth bits)) init 0
  val activePositionValid = RegInit(False)
  val fault = RegInit(False)

  val acceptedCounter = Reg(UInt(32 bits)) init 0
  val emittedCounter = Reg(UInt(32 bits)) init 0
  io.acceptedSequences := acceptedCounter
  io.emittedSequences := emittedCounter
  io.protocolFault := fault

  // Constant Q1.15 exp(-n/16) table, n=0..128.  32768 represents exactly 1.
  private val expConstants = (0 to 128).map { index =>
    Math.round(Math.exp(-index.toDouble / 16.0) * cfg.expOne).toInt
  }
  val expLut = Vec(expConstants.map(value => U(value, cfg.expWidth bits)))

  def expNegative(magnitudeQ12: UInt): UInt = {
    val extended = magnitudeQ12.resize(magnitudeQ12.getWidth + 1)
    val roundedStep = ((extended + U(128, extended.getWidth bits)) |>> 8)
      .resize(9)
    val result = UInt(cfg.expWidth bits)
    result := 0
    when(roundedStep <= 128) {
      result := expLut(roundedStep.resize(8))
    }
    result
  }

  def roundSignedShiftAndSaturate(
      input: SInt,
      shift: Int,
      outputWidth: Int
  ): SInt = {
    val magnitude = UInt(input.getWidth bits)
    when(input.msb) {
      magnitude := (-input).asUInt
    } otherwise {
      magnitude := input.asUInt
    }
    val biased = (
      magnitude.resize(input.getWidth + 1) +
        U(BigInt(1) << (shift - 1), (input.getWidth + 1) bits)
    ).resize(input.getWidth + 1)
    val rounded = (biased |>> shift).resize(input.getWidth + 1 - shift)
    val result = SInt(outputWidth bits)
    when(input.msb) {
      when(rounded >= U(BigInt(1) << (outputWidth - 1), rounded.getWidth bits)) {
        result := S(-(BigInt(1) << (outputWidth - 1)), outputWidth bits)
      } otherwise {
        result := (-rounded.resize(outputWidth).asSInt).resize(outputWidth)
      }
    } otherwise {
      when(rounded > U((BigInt(1) << (outputWidth - 1)) - 1,
          rounded.getWidth bits)) {
        result := S((BigInt(1) << (outputWidth - 1)) - 1, outputWidth bits)
      } otherwise {
        result := rounded.resize(outputWidth).asSInt
      }
    }
    result
  }

  def saturatingAdd(left: SInt, right: SInt, outputWidth: Int): SInt = {
    val wide = left.resize(outputWidth + 1) + right.resize(outputWidth + 1)
    val result = SInt(outputWidth bits)
    when(wide > S((BigInt(1) << (outputWidth - 1)) - 1, wide.getWidth bits)) {
      result := S((BigInt(1) << (outputWidth - 1)) - 1, outputWidth bits)
    } elsewhen (wide < S(-(BigInt(1) << (outputWidth - 1)), wide.getWidth bits)) {
      result := S(-(BigInt(1) << (outputWidth - 1)), outputWidth bits)
    } otherwise {
      result := wide.resize(outputWidth)
    }
    result
  }

  io.input.ready := state === State.Collect

  val factorAlpha = Vec(UInt(cfg.expWidth bits), cfg.queryHeadsPerKv)
  val factorBeta = Vec(UInt(cfg.expWidth bits), cfg.queryHeadsPerKv)
  val nextMax = Vec(SInt(cfg.scoreWidth bits), cfg.queryHeadsPerKv)
  val nextSum = Vec(UInt(cfg.sumWidth bits), cfg.queryHeadsPerKv)
  val nextHasValid = Bits(cfg.queryHeadsPerKv bits)

  for (query <- 0 until cfg.queryHeadsPerKv) {
    val priorHasValid = Bool()
    priorHasValid := hasValid(query)
    when(io.input.firstKey && io.input.firstValue) {
      priorHasValid := False
    }

    val priorSum = UInt(cfg.sumWidth bits)
    priorSum := sum(query)
    when(io.input.firstKey && io.input.firstValue) { priorSum := 0 }

    factorAlpha(query) := Mux(priorHasValid,
      U(cfg.expOne, cfg.expWidth bits), U(0, cfg.expWidth bits))
    factorBeta(query) := 0
    nextMax(query) := maxScore(query)
    nextSum(query) := priorSum
    nextHasValid(query) := priorHasValid

    val causallyValid = io.input.positionValid &&
      io.input.queryMask(query) &&
      io.input.keyPosition <= io.input.queryPosition

    when(causallyValid) {
      nextHasValid(query) := True
      when(!priorHasValid) {
        factorAlpha(query) := 0
        factorBeta(query) := cfg.expOne
        nextMax(query) := io.input.score(query)
        nextSum(query) := cfg.expOne
      } elsewhen (io.input.score(query) > maxScore(query)) {
        val difference = (
          io.input.score(query).resize(cfg.scoreWidth + 1) -
            maxScore(query).resize(cfg.scoreWidth + 1)
        ).asUInt
        factorAlpha(query) := expNegative(difference)
        factorBeta(query) := cfg.expOne
        nextMax(query) := io.input.score(query)

        val product = priorSum * factorAlpha(query)
        val scaled = (
          product.resize(product.getWidth + 1) +
            U(BigInt(1) << (cfg.expFractionalBits - 1),
              (product.getWidth + 1) bits)
        ) |>> cfg.expFractionalBits
        nextSum(query) := (
          scaled.resize(cfg.sumWidth + 1) +
            factorBeta(query).resize(cfg.sumWidth + 1)
        ).resize(cfg.sumWidth)
      } otherwise {
        val difference = (
          maxScore(query).resize(cfg.scoreWidth + 1) -
            io.input.score(query).resize(cfg.scoreWidth + 1)
        ).asUInt
        factorAlpha(query) := cfg.expOne
        factorBeta(query) := expNegative(difference)
        nextMax(query) := maxScore(query)
        nextSum(query) := (
          priorSum.resize(cfg.sumWidth + 1) +
            factorBeta(query).resize(cfg.sumWidth + 1)
        ).resize(cfg.sumWidth)
      }
    }
  }

  for (query <- 0 until cfg.queryHeadsPerKv) {
    val selectedAlpha = UInt(cfg.expWidth bits)
    val selectedBeta = UInt(cfg.expWidth bits)
    selectedAlpha := alpha(query)
    selectedBeta := beta(query)
    when(io.input.firstValue) {
      selectedAlpha := factorAlpha(query)
      selectedBeta := factorBeta(query)
    }

    val oldContext = context(query).readAsync(io.input.valueIndex)
    val positiveAlpha = (B(0, 1 bits) ## selectedAlpha.asBits).asSInt
    val oldProduct = oldContext * positiveAlpha
    val scaledOld = roundSignedShiftAndSaturate(
      oldProduct,
      cfg.expFractionalBits,
      cfg.contextWidth
    )
    val positiveBeta = (B(0, 1 bits) ## selectedBeta.asBits).asSInt
    val weightedValue = (
      io.input.value.resize(cfg.valueWidth + 1) * positiveBeta
    ).resize(cfg.contextWidth)
    val updatedContext = saturatingAdd(scaledOld, weightedValue, cfg.contextWidth)

    when(io.input.fire) {
      context(query).write(
        address = io.input.valueIndex,
        data = updatedContext,
        enable = True
      )
    }
  }

  when(io.input.fire) {
    when(io.input.firstKey && io.input.firstValue) {
      fault := False
      sequenceActive := True
      activeQueryMask := io.input.queryMask
      activeQueryPosition := io.input.queryPosition
    }

    when(!sequenceActive && !(io.input.firstKey && io.input.firstValue)) {
      fault := True
    }
    when(io.input.firstValue =/= (io.input.valueIndex === 0)) {
      fault := True
    }
    when(io.input.lastValue =/=
        (io.input.valueIndex === U(cfg.headDim - 1, cfg.valueIndexWidth bits))) {
      fault := True
    }
    when(!io.input.firstValue && io.input.valueIndex =/= expectedValueIndex) {
      fault := True
    }
    when(sequenceActive && io.input.queryPosition =/= activeQueryPosition) {
      fault := True
    }
    when(sequenceActive && io.input.queryMask =/= activeQueryMask) {
      fault := True
    }

    when(io.input.firstValue) {
      expectedValueIndex := 1
      activeKeyPosition := io.input.keyPosition
      activePositionValid := io.input.positionValid
      for (query <- 0 until cfg.queryHeadsPerKv) {
        alpha(query) := factorAlpha(query)
        beta(query) := factorBeta(query)
        maxScore(query) := nextMax(query)
        sum(query) := nextSum(query)
        hasValid(query) := nextHasValid(query)
      }
    } otherwise {
      expectedValueIndex := expectedValueIndex + 1
      when(io.input.keyPosition =/= activeKeyPosition ||
          io.input.positionValid =/= activePositionValid) {
        fault := True
      }
    }

    when(io.input.lastKey && io.input.lastValue) {
      sequenceActive := False
      acceptedCounter := acceptedCounter + 1
      state := State.ReciprocalPrepare
    }
  }

  // Four parallel restoring dividers calculate round(2^46 / l).  This is a
  // once-per-query-head operation, so output normalization subsequently runs
  // at one vector element per cycle instead of instantiating four dividers per
  // output element.
  val dividerDividend = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init 0,
    cfg.queryHeadsPerKv
  )
  val dividerDivisor = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init 1,
    cfg.queryHeadsPerKv
  )
  val dividerRemainder = Vec(
    Reg(UInt((cfg.dividerWidth + 1) bits)) init 0,
    cfg.queryHeadsPerKv
  )
  val dividerQuotient = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init 0,
    cfg.queryHeadsPerKv
  )
  val reciprocal = Vec(
    Reg(UInt(cfg.reciprocalWidth bits)) init 0,
    cfg.queryHeadsPerKv
  )
  val dividerIteration = Reg(UInt(cfg.dividerCountWidth bits)) init 0

  val dividendNext = Vec(UInt(cfg.dividerWidth bits), cfg.queryHeadsPerKv)
  val remainderNext = Vec(
    UInt((cfg.dividerWidth + 1) bits),
    cfg.queryHeadsPerKv
  )
  val quotientNext = Vec(UInt(cfg.dividerWidth bits), cfg.queryHeadsPerKv)
  for (query <- 0 until cfg.queryHeadsPerKv) {
    val shiftedRemainder = (
      dividerRemainder(query)(cfg.dividerWidth - 1 downto 0) ##
        dividerDividend(query).msb
    ).asUInt
    val canSubtract = shiftedRemainder >=
      dividerDivisor(query).resize(cfg.dividerWidth + 1)
    remainderNext(query) := shiftedRemainder
    when(canSubtract) {
      remainderNext(query) := (
        shiftedRemainder - dividerDivisor(query).resize(cfg.dividerWidth + 1)
      ).resize(cfg.dividerWidth + 1)
    }
    dividendNext(query) :=
      (dividerDividend(query) |<< 1).resize(cfg.dividerWidth)
    val shiftedQuotient =
      (dividerQuotient(query) |<< 1).resize(cfg.dividerWidth)
    quotientNext(query) := shiftedQuotient
    when(canSubtract) {
      quotientNext(query) :=
        shiftedQuotient | U(1, cfg.dividerWidth bits)
    }
  }

  val outputIndex = Reg(UInt(cfg.valueIndexWidth bits)) init 0

  switch(state) {
    is(State.Collect) {}

    is(State.ReciprocalPrepare) {
      for (query <- 0 until cfg.queryHeadsPerKv) {
        val roundedDividend = (
          U(BigInt(1) << cfg.reciprocalFractionalBits,
            cfg.dividerWidth bits) +
            (sum(query).resize(cfg.dividerWidth) |>> 1)
        ).resize(cfg.dividerWidth)
        dividerDividend(query) := Mux(hasValid(query), roundedDividend, U(0))
        dividerDivisor(query) := Mux(hasValid(query),
          sum(query).resize(cfg.dividerWidth), U(1, cfg.dividerWidth bits))
        dividerRemainder(query) := 0
        dividerQuotient(query) := 0
      }
      dividerIteration := 0
      state := State.ReciprocalDivide
    }

    is(State.ReciprocalDivide) {
      for (query <- 0 until cfg.queryHeadsPerKv) {
        dividerDividend(query) := dividendNext(query)
        dividerRemainder(query) := remainderNext(query)
        dividerQuotient(query) := quotientNext(query)
      }
      when(dividerIteration ===
          U(cfg.dividerWidth - 1, cfg.dividerCountWidth bits)) {
        for (query <- 0 until cfg.queryHeadsPerKv) {
          reciprocal(query) := Mux(hasValid(query),
            quotientNext(query).resize(cfg.reciprocalWidth), U(0))
        }
        outputIndex := 0
        state := State.Emit
      } otherwise {
        dividerIteration := dividerIteration + 1
      }
    }

    is(State.Emit) {
      when(io.output.fire) {
        when(outputIndex === U(cfg.headDim - 1, cfg.valueIndexWidth bits)) {
          emittedCounter := emittedCounter + 1
          state := State.Collect
        } otherwise {
          outputIndex := outputIndex + 1
        }
      }
    }
  }

  io.output.valid := state === State.Emit
  io.output.valueIndex := outputIndex
  io.output.queryMask := activeQueryMask
  io.output.allMasked := ~hasValid
  io.output.first := outputIndex === 0
  io.output.last := outputIndex === U(cfg.headDim - 1, cfg.valueIndexWidth bits)
  for (query <- 0 until cfg.queryHeadsPerKv) {
    val numerator = context(query).readAsync(outputIndex)
    val positiveReciprocal =
      (B(0, 1 bits) ## reciprocal(query).asBits).asSInt
    val normalizedProduct = numerator * positiveReciprocal
    io.output.value(query) := roundSignedShiftAndSaturate(
      normalizedProduct,
      cfg.reciprocalFractionalBits,
      cfg.valueWidth
    )
  }

  io.busy := state =/= State.Collect || sequenceActive
}

/** Bit-accurate software oracle for [[NativeOnlineAttention]]. */
object NativeOnlineAttentionMath {
  case class Position(
      scoresQ12: Seq[Int],
      valueQ16: Seq[Int],
      keyPosition: Int,
      valid: Boolean = true
  )

  private val ExpOne = BigInt(1 << 15)
  private val Signed64Min = -(BigInt(1) << 63)
  private val Signed64Max = (BigInt(1) << 63) - 1
  private val Signed32Min = BigInt(Int.MinValue)
  private val Signed32Max = BigInt(Int.MaxValue)
  private val ExpLut = (0 to 128).map { index =>
    BigInt(Math.round(Math.exp(-index.toDouble / 16.0) * (1 << 15)))
  }

  private def expNegative(magnitudeQ12: Int): BigInt = {
    require(magnitudeQ12 >= 0)
    val index = (magnitudeQ12 + 128) >> 8
    if (index <= 128) ExpLut(index) else BigInt(0)
  }

  private def roundShift(value: BigInt, shift: Int): BigInt = {
    val rounded = (value.abs + (BigInt(1) << (shift - 1))) >> shift
    if (value.signum < 0) -rounded else rounded
  }

  private def sat64(value: BigInt): BigInt =
    value.max(Signed64Min).min(Signed64Max)

  private def sat32(value: BigInt): Int =
    value.max(Signed32Min).min(Signed32Max).toInt

  def compute(
      queryPosition: Int,
      queryMask: Seq[Boolean],
      positions: Seq[Position]
  ): Seq[Seq[Int]] = {
    require(queryMask.nonEmpty)
    require(positions.nonEmpty)
    val headDim = positions.head.valueQ16.length
    require(headDim > 0)
    require(positions.forall(_.valueQ16.length == headDim))
    require(positions.forall(_.scoresQ12.length == queryMask.length))

    queryMask.indices.map { query =>
      var maximum = 0
      var sum = BigInt(0)
      var hasValid = false
      val context = Array.fill(headDim)(BigInt(0))

      positions.foreach { position =>
        val accepted = queryMask(query) && position.valid &&
          position.keyPosition <= queryPosition
        var alpha = if (hasValid) ExpOne else BigInt(0)
        var beta = BigInt(0)

        if (accepted) {
          val score = position.scoresQ12(query)
          if (!hasValid) {
            maximum = score
            sum = ExpOne
            alpha = 0
            beta = ExpOne
            hasValid = true
          } else if (score > maximum) {
            alpha = expNegative(score - maximum)
            beta = ExpOne
            maximum = score
            sum = ((sum * alpha + (BigInt(1) << 14)) >> 15) + beta
          } else {
            alpha = ExpOne
            beta = expNegative(maximum - score)
            sum += beta
          }
        }

        for (index <- 0 until headDim) {
          val scaledOld = roundShift(context(index) * alpha, 15)
          val weightedValue = BigInt(position.valueQ16(index)) * beta
          context(index) = sat64(sat64(scaledOld) + weightedValue)
        }
      }

      if (!hasValid) {
        Seq.fill(headDim)(0)
      } else {
        val reciprocal = ((BigInt(1) << 46) + sum / 2) / sum
        context.map(value => sat32(roundShift(value * reciprocal, 46))).toSeq
      }
    }
  }
}
