package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Fixed production arithmetic for a packet-free resident RMSNorm stream.
  *
  * Input activations and gamma are signed Q16.16.  One beat carries the same
  * feature index for all token lanes; gamma is consequently broadcast across
  * the lanes.  A complete vector is buffered in an inferred synchronous RAM
  * while every active lane accumulates sum(x^2).  The core then computes:
  *
  *   meanQ32 = round(sum(xQ16^2) / vectorLength)
  *   rmsQ16  = floor_sqrt(meanQ32 + epsilonQ32)
  *   gainQ30 = round(2^46 / rmsQ16)
  *   yQ16    = sat_i32(round_away((xQ16 * gammaQ16 * gainQ30) / 2^46))
  *
  * Mean and reciprocal use parallel, one-bit-per-cycle restoring dividers;
  * sqrt uses a parallel radix-4 restoring implementation.  Their latency is
  * fixed (96 + 48 + 96 cycles) and independent of input data.  Once the
  * gains are available, the buffered vector is replayed through registered
  * multiply/multiply/round stages at one feature beat per cycle.  All public
  * interfaces obey Stream backpressure.
  *
  * `first` must be asserted on feature zero and `last` on feature
  * vectorLength-1.  A framing error drops the incomplete vector; a framing
  * or token-mask mismatch raises protocolFault until the next `first` beat.
  */
case class NativeRmsNormConfig(
    tokenLanes: Int = 4,
    vectorLength: Int = 2560,
    dataWidth: Int = 32,
    fractionalBits: Int = 16,
    epsilonQ32: BigInt = BigInt(42950), // round(1e-5 * 2^32)
    replayFifoDepth: Int = 8
) {
  require(tokenLanes > 0 && isPow2(tokenLanes),
    "tokenLanes must be a positive power of two")
  require(vectorLength >= 2, "vectorLength must be at least two")
  require(dataWidth == 32, "the production BitNet activation format is signed i32")
  require(fractionalBits == 16, "the production BitNet activation format is Q16.16")
  require(epsilonQ32 > 0 && epsilonQ32 < (BigInt(1) << 63),
    "epsilonQ32 must be a positive Q32.32 value")
  require(replayFifoDepth >= 4 && isPow2(replayFifoDepth),
    "replayFifoDepth must be a power of two of at least four")

  val indexWidth: Int = log2Up(vectorLength)
  val entryWidth: Int = dataWidth * (tokenLanes + 1)
  val dividerWidth: Int = 96
  val dividerCountWidth: Int = log2Up(dividerWidth)
  val sqrtWidth: Int = dividerWidth / 2
  val sqrtCountWidth: Int = log2Up(sqrtWidth)
  val inputProductWidth: Int = 66
  val gainWidth: Int = 48
  val normalizedProductWidth: Int = inputProductWidth + gainWidth + 1
  val normalizeShift: Int = 46
}

case class NativeRmsNormBeat(cfg: NativeRmsNormConfig) extends Bundle {
  val value = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val gamma = SInt(cfg.dataWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

case class NativeRmsNormResult(cfg: NativeRmsNormConfig) extends Bundle {
  val value = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

case class NativeRmsReplayEntry(cfg: NativeRmsNormConfig)
    extends Bundle {
  val data = Bits(cfg.entryWidth bits)
  val index = UInt(cfg.indexWidth bits)
}

case class NativeRmsInputProduct(cfg: NativeRmsNormConfig)
    extends Bundle {
  val valueGamma = Vec(SInt(cfg.inputProductWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

case class NativeRmsNormalizedProduct(cfg: NativeRmsNormConfig)
    extends Bundle {
  val product = Vec(SInt(cfg.normalizedProductWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

class NativeRmsNorm(
    cfg: NativeRmsNormConfig = NativeRmsNormConfig()
) extends Component {
  val io = new Bundle {
    val input = slave(Stream(NativeRmsNormBeat(cfg)))
    val output = master(Stream(NativeRmsNormResult(cfg)))
    val busy = out Bool()
    val protocolFault = out Bool()
    val acceptedVectors = out UInt(32 bits)
    val emittedVectors = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Collect, MeanPrepare, MeanDivide, SqrtPrepare, SqrtIterate,
        GainPrepare, GainDivide, Replay = newElement()
  }
  val state = Reg(State()) init State.Collect

  val storage = Mem(Bits(cfg.entryWidth bits), cfg.vectorLength)
  val collecting = RegInit(False)
  val collectIndex = Reg(UInt(cfg.indexWidth bits)) init (0)
  val activeTokenMask = Reg(Bits(cfg.tokenLanes bits)) init (0)
  val fault = RegInit(False)
  val sumSquares = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0),
    cfg.tokenLanes
  )

  val acceptedVectorCounter = Reg(UInt(32 bits)) init (0)
  val emittedVectorCounter = Reg(UInt(32 bits)) init (0)

  io.input.ready := state === State.Collect
  io.protocolFault := fault
  io.acceptedVectors := acceptedVectorCounter
  io.emittedVectors := emittedVectorCounter

  val inputEntry = Bits(cfg.entryWidth bits)
  for (lane <- 0 until cfg.tokenLanes) {
    inputEntry((lane + 1) * cfg.dataWidth - 1 downto lane * cfg.dataWidth) :=
      io.input.value(lane).asBits
  }
  inputEntry(cfg.entryWidth - 1 downto cfg.tokenLanes * cfg.dataWidth) :=
    io.input.gamma.asBits

  val acceptedIndex = UInt(cfg.indexWidth bits)
  acceptedIndex := collectIndex
  when(io.input.first) { acceptedIndex := 0 }
  val acceptedAtEnd = acceptedIndex === U(cfg.vectorLength - 1, cfg.indexWidth bits)
  val acceptedHasStart = collecting || io.input.first

  val inputSquares = Vec(UInt(cfg.dividerWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val widened = io.input.value(lane).resize(cfg.dataWidth + 1)
    inputSquares(lane) := (widened * widened).asUInt.resize(cfg.dividerWidth)
  }

  when(io.input.fire) {
    // A new `first` beat is also the explicit recovery point after a framing
    // fault.  If it arrives mid-vector, the old partial vector is discarded.
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
      storage.write(
        address = acceptedIndex,
        data = inputEntry,
        enable = True
      )

      for (lane <- 0 until cfg.tokenLanes) {
        val square = Mux(io.input.tokenMask(lane), inputSquares(lane), U(0))
        when(io.input.first) {
          sumSquares(lane) := square
        } otherwise {
          sumSquares(lane) :=
            (sumSquares(lane) + square).resize(cfg.dividerWidth)
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
          state := State.MeanPrepare
          acceptedVectorCounter := acceptedVectorCounter + 1
        }
      } elsewhen (io.input.last) {
        // An early `last` never exposes a partially-normalized vector.
        collecting := False
      } otherwise {
        collectIndex := acceptedIndex + 1
      }
    }
  }

  // One restoring divider per token lane.  This keeps prefill lanes in
  // lockstep and avoids a variable-latency operator or combinational `/`.
  val dividerDividend = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0),
    cfg.tokenLanes
  )
  val dividerDivisor = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0),
    cfg.tokenLanes
  )
  val dividerRemainder = Vec(
    Reg(UInt((cfg.dividerWidth + 1) bits)) init (0),
    cfg.tokenLanes
  )
  val dividerQuotient = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0),
    cfg.tokenLanes
  )
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

  val meanSquareQ32 = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0),
    cfg.tokenLanes
  )
  val rmsQ16 = Vec(
    Reg(UInt(cfg.sqrtWidth bits)) init (0),
    cfg.tokenLanes
  )
  val gainQ30 = Vec(
    Reg(UInt(cfg.gainWidth bits)) init (0),
    cfg.tokenLanes
  )

  // Parallel restoring base-4 square roots consume two Q32.32 radicand bits
  // per cycle and return floor(sqrt(.)) in Q16.16 after 48 cycles.
  val sqrtRadicand = Vec(
    Reg(UInt(cfg.dividerWidth bits)) init (0),
    cfg.tokenLanes
  )
  val sqrtRemainder = Vec(
    Reg(UInt((cfg.sqrtWidth + 2) bits)) init (0),
    cfg.tokenLanes
  )
  val sqrtRoot = Vec(
    Reg(UInt(cfg.sqrtWidth bits)) init (0),
    cfg.tokenLanes
  )
  val sqrtIteration = Reg(UInt(cfg.sqrtCountWidth bits)) init (0)

  val sqrtRadicandNext = Vec(UInt(cfg.dividerWidth bits), cfg.tokenLanes)
  val sqrtRemainderNext =
    Vec(UInt((cfg.sqrtWidth + 2) bits), cfg.tokenLanes)
  val sqrtRootNext = Vec(UInt(cfg.sqrtWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val shiftedRemainder = UInt((cfg.sqrtWidth + 2) bits)
    shiftedRemainder := (
      (sqrtRemainder(lane) |<< 2).resize(cfg.sqrtWidth + 2) |
        sqrtRadicand(lane)(cfg.dividerWidth - 1 downto cfg.dividerWidth - 2)
          .resize(cfg.sqrtWidth + 2)
    ).resize(cfg.sqrtWidth + 2)
    val trial = (
      (sqrtRoot(lane).resize(cfg.sqrtWidth + 2) |<< 2) |
        U(1, cfg.sqrtWidth + 2 bits)
    ).resize(cfg.sqrtWidth + 2)
    val canSubtract = shiftedRemainder >= trial
    sqrtRemainderNext(lane) := shiftedRemainder
    when(canSubtract) {
      sqrtRemainderNext(lane) :=
        (shiftedRemainder - trial).resize(cfg.sqrtWidth + 2)
    }
    sqrtRootNext(lane) := (sqrtRoot(lane) |<< 1).resize(cfg.sqrtWidth)
    when(canSubtract) {
      sqrtRootNext(lane) :=
        (sqrtRoot(lane) |<< 1).resize(cfg.sqrtWidth) |
          U(1, cfg.sqrtWidth bits)
    }
    sqrtRadicandNext(lane) :=
      (sqrtRadicand(lane) |<< 2).resize(cfg.dividerWidth)
  }

  val allReplayReadsIssued = RegInit(False)
  val replayReadIndex = Reg(UInt(cfg.indexWidth bits)) init (0)

  switch(state) {
    is(State.Collect) {
      // Collection is handled above so malformed framing can be recovered
      // without adding command/control coupling to this math block.
    }

    is(State.MeanPrepare) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := (
          sumSquares(lane) + U(cfg.vectorLength / 2, cfg.dividerWidth bits)
        ).resize(cfg.dividerWidth)
        dividerDivisor(lane) := U(cfg.vectorLength, cfg.dividerWidth bits)
        dividerRemainder(lane) := 0
        dividerQuotient(lane) := 0
      }
      dividerIteration := 0
      state := State.MeanDivide
    }

    is(State.MeanDivide) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := dividerDividendNext(lane)
        dividerRemainder(lane) := dividerRemainderNext(lane)
        dividerQuotient(lane) := dividerQuotientNext(lane)
      }
      when(dividerIteration ===
          U(cfg.dividerWidth - 1, cfg.dividerCountWidth bits)) {
        for (lane <- 0 until cfg.tokenLanes) {
          meanSquareQ32(lane) := (
            dividerQuotientNext(lane) +
              U(cfg.epsilonQ32, cfg.dividerWidth bits)
          ).resize(cfg.dividerWidth)
        }
        state := State.SqrtPrepare
      } otherwise {
        dividerIteration := dividerIteration + 1
      }
    }

    is(State.SqrtPrepare) {
      for (lane <- 0 until cfg.tokenLanes) {
        sqrtRadicand(lane) := meanSquareQ32(lane)
        sqrtRemainder(lane) := 0
        sqrtRoot(lane) := 0
      }
      sqrtIteration := 0
      state := State.SqrtIterate
    }

    is(State.SqrtIterate) {
      for (lane <- 0 until cfg.tokenLanes) {
        sqrtRadicand(lane) := sqrtRadicandNext(lane)
        sqrtRemainder(lane) := sqrtRemainderNext(lane)
        sqrtRoot(lane) := sqrtRootNext(lane)
      }
      when(sqrtIteration === U(cfg.sqrtWidth - 1, cfg.sqrtCountWidth bits)) {
        for (lane <- 0 until cfg.tokenLanes) {
          rmsQ16(lane) := sqrtRootNext(lane)
        }
        state := State.GainPrepare
      } otherwise {
        sqrtIteration := sqrtIteration + 1
      }
    }

    is(State.GainPrepare) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := (
          U(BigInt(1) << cfg.normalizeShift, cfg.dividerWidth bits) +
            (rmsQ16(lane).resize(cfg.dividerWidth) |>> 1)
        ).resize(cfg.dividerWidth)
        // Epsilon guarantees a nonzero rms.  The fallback keeps the divider
        // fail-safe if parameters are corrupted during future refactoring.
        dividerDivisor(lane) := rmsQ16(lane).resize(cfg.dividerWidth)
        when(rmsQ16(lane) === 0) { dividerDivisor(lane) := 1 }
        dividerRemainder(lane) := 0
        dividerQuotient(lane) := 0
      }
      dividerIteration := 0
      state := State.GainDivide
    }

    is(State.GainDivide) {
      for (lane <- 0 until cfg.tokenLanes) {
        dividerDividend(lane) := dividerDividendNext(lane)
        dividerRemainder(lane) := dividerRemainderNext(lane)
        dividerQuotient(lane) := dividerQuotientNext(lane)
      }
      when(dividerIteration ===
          U(cfg.dividerWidth - 1, cfg.dividerCountWidth bits)) {
        for (lane <- 0 until cfg.tokenLanes) {
          gainQ30(lane) := dividerQuotientNext(lane).resize(cfg.gainWidth)
        }
        replayReadIndex := 0
        allReplayReadsIssued := False
        state := State.Replay
      } otherwise {
        dividerIteration := dividerIteration + 1
      }
    }

    is(State.Replay) {
      // Address issue and pipeline drain are handled below.  The last public
      // output handshake returns the core to Collect.
    }
  }

  // Synchronous replay RAM plus a FIFO decouples the one-cycle RAM response
  // from arbitrary downstream backpressure.  Requiring two free FIFO slots
  // before issuing accounts for the one outstanding synchronous response.
  val replayFifo = new StreamFifo(
    NativeRmsReplayEntry(cfg),
    cfg.replayFifoDepth
  )
  val replayReadIssue = state === State.Replay &&
    !allReplayReadsIssued && replayFifo.io.availability > 1
  val replayReadData = storage.readSync(replayReadIndex, replayReadIssue)
  val replayResponseValid = RegNext(replayReadIssue) init (False)
  val replayResponseIndex = RegNextWhen(replayReadIndex, replayReadIssue) init (0)

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

  val replayValues = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    replayValues(lane) := replayFifo.io.pop.data(
      (lane + 1) * cfg.dataWidth - 1 downto lane * cfg.dataWidth
    ).asSInt
  }
  val replayGamma = replayFifo.io.pop.data(
    cfg.entryWidth - 1 downto cfg.tokenLanes * cfg.dataWidth
  ).asSInt

  val inputProductStage = Stream(NativeRmsInputProduct(cfg))
  inputProductStage.valid := replayFifo.io.pop.valid
  replayFifo.io.pop.ready := inputProductStage.ready
  inputProductStage.tokenMask := activeTokenMask
  inputProductStage.first := replayFifo.io.pop.index === 0
  inputProductStage.last := replayFifo.io.pop.index ===
    U(cfg.vectorLength - 1, cfg.indexWidth bits)
  for (lane <- 0 until cfg.tokenLanes) {
    inputProductStage.valueGamma(lane) := (
      replayValues(lane).resize(cfg.dataWidth + 1) *
        replayGamma.resize(cfg.dataWidth + 1)
    ).resize(cfg.inputProductWidth)
  }

  val inputProduct = inputProductStage.m2sPipe()
  val normalizedProductStage = Stream(NativeRmsNormalizedProduct(cfg))
  normalizedProductStage.valid := inputProduct.valid
  inputProduct.ready := normalizedProductStage.ready
  normalizedProductStage.tokenMask := inputProduct.tokenMask
  normalizedProductStage.first := inputProduct.first
  normalizedProductStage.last := inputProduct.last
  for (lane <- 0 until cfg.tokenLanes) {
    val positiveGain = (B(0, 1 bits) ## gainQ30(lane).asBits).asSInt
    normalizedProductStage.product(lane) := (
      inputProduct.valueGamma(lane) * positiveGain
    ).resize(cfg.normalizedProductWidth)
  }

  val normalizedProduct = normalizedProductStage.m2sPipe()
  val roundedStage = Stream(NativeRmsNormResult(cfg))
  roundedStage.valid := normalizedProduct.valid
  normalizedProduct.ready := roundedStage.ready
  roundedStage.tokenMask := normalizedProduct.tokenMask
  roundedStage.first := normalizedProduct.first
  roundedStage.last := normalizedProduct.last
  for (lane <- 0 until cfg.tokenLanes) {
    val product = normalizedProduct.product(lane)
    val magnitude = UInt(cfg.normalizedProductWidth bits)
    when(product.msb) {
      magnitude := (-product).asUInt
    } otherwise {
      magnitude := product.asUInt
    }
    val biased = (
      magnitude.resize(cfg.normalizedProductWidth + 1) +
        U(BigInt(1) << (cfg.normalizeShift - 1),
          (cfg.normalizedProductWidth + 1) bits)
    ).resize(cfg.normalizedProductWidth + 1)
    val rounded = (biased |>> cfg.normalizeShift)
      .resize(cfg.normalizedProductWidth + 1 - cfg.normalizeShift)

    val value = SInt(cfg.dataWidth bits)
    value := 0
    when(normalizedProduct.tokenMask(lane)) {
      when(product.msb) {
        when(rounded >= U(BigInt(1) << (cfg.dataWidth - 1), rounded.getWidth bits)) {
          value := S(Int.MinValue, cfg.dataWidth bits)
        } otherwise {
          value := (-rounded.resize(cfg.dataWidth).asSInt).resize(cfg.dataWidth)
        }
      } otherwise {
        when(rounded > U(Int.MaxValue, rounded.getWidth bits)) {
          value := S(Int.MaxValue, cfg.dataWidth bits)
        } otherwise {
          value := rounded.resize(cfg.dataWidth).asSInt
        }
      }
    }
    roundedStage.value(lane) := value
  }

  io.output << roundedStage.m2sPipe()

  when(io.output.fire && io.output.last) {
    state := State.Collect
    emittedVectorCounter := emittedVectorCounter + 1
  }

  io.busy := state =/= State.Collect || collecting ||
    replayFifo.io.occupancy =/= 0 || inputProduct.valid ||
    normalizedProduct.valid || roundedStage.valid || io.output.valid
}

/** Bit-accurate pure-Scala model of [[NativeRmsNorm]]. */
object NativeRmsNormMath {
  val DefaultEpsilonQ32: BigInt = BigInt(42950)
  private val SignedMin = BigInt(Int.MinValue)
  private val SignedMax = BigInt(Int.MaxValue)

  private def floorSqrt(value: BigInt): BigInt = {
    require(value >= 0)
    if (value == 0) {
      BigInt(0)
    } else {
      var x = BigInt(1) << ((value.bitLength + 1) / 2)
      var next = (x + value / x) >> 1
      while (next < x) {
        x = next
        next = (x + value / x) >> 1
      }
      x
    }
  }

  private def roundMagnitudeAndSaturate(product: BigInt, shift: Int): Int = {
    val roundedMagnitude = (product.abs + (BigInt(1) << (shift - 1))) >> shift
    val signed = if (product.signum < 0) -roundedMagnitude else roundedMagnitude
    signed.max(SignedMin).min(SignedMax).toInt
  }

  def normalize(
      tokens: Seq[Seq[Int]],
      gamma: Seq[Int],
      epsilonQ32: BigInt = DefaultEpsilonQ32
  ): Seq[Seq[Int]] = {
    require(tokens.nonEmpty)
    require(gamma.nonEmpty)
    require(tokens.forall(_.length == gamma.length))
    require(epsilonQ32 > 0)

    val length = gamma.length
    tokens.map { token =>
      val sumSquares = token.foldLeft(BigInt(0)) { (sum, value) =>
        sum + BigInt(value) * BigInt(value)
      }
      val meanQ32 = (sumSquares + length / 2) / length
      val rmsQ16 = floorSqrt(meanQ32 + epsilonQ32)
      val gainQ30 = ((BigInt(1) << 46) + rmsQ16 / 2) / rmsQ16
      token.zip(gamma).map { case (value, weight) =>
        val product = BigInt(value) * BigInt(weight) * gainQ30
        roundMagnitudeAndSaturate(product, 46)
      }
    }
  }
}
