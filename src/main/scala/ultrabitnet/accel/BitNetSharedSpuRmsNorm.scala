package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Arithmetic contract for the resident, packet-free RMSNorm SPU.
  *
  * External activations and gamma are signed Q16.16.  Before multiplication
  * they are explicitly saturated to symmetric signed-27 Q10.16
  * [-67108863, +67108863].  Saturation is never silent: it sets
  * `overflowSticky` and the per-job overflow result.  The symmetric bound
  * makes each magnitude exactly 26 bits.  A product is reconstructed exactly
  * from unsigned magnitude limbs (26 x 18 and 26 x 8).  Four lanes therefore
  * need eight shared DSP48
  * multipliers, reused by the measure, x*gamma and *gain phases.
  *
  * The replay arithmetic deliberately has a narrowing point:
  *
  *   pQ16 = sat_i27(round_away((xQ16 * gammaQ16) / 2^16))
  *   yQ16 = sat_i32(round_away((pQ16 * gainQ16) / 2^16))
  *
  * gainQ16 is round(2^32 / floor_sqrt(mean(x^2)+epsilon)).  Mean division,
  * integer square root and reciprocal division are fixed-width iterative
  * datapaths.  There are no `/`, `%`, combinational sqrt blocks, vector RAMs,
  * packets, uops or host callbacks in this component.
  */
case class BitNetSharedSpuRmsNormConfig(
    tokenLanes: Int = BitNetConfig.Production.tokenParallelism,
    maxLength: Int = 6912,
    dataWidth: Int = 32,
    fractionalBits: Int = 16,
    narrowWidth: Int = 27,
    limbWidth: Int = 18,
    accumulatorWidth: Int = 66,
    epsilonQ32: BigInt = BigInt(42950),
    tagWidth: Int = 8
) {
  require(tokenLanes == 2 || tokenLanes == 4,
    "the native shared SPU must be TP2 or TP4")
  require(maxLength >= 6912, "production hidden/FFN lengths must fit")
  require(dataWidth == 32 && fractionalBits == 16,
    "the resident activation contract is signed Q16.16")
  require(narrowWidth == 27 && limbWidth == 18,
    "DSP48E2 decomposition is fixed at 26x18 + 26x8 magnitudes")
  require(accumulatorWidth >= 66,
    "6912 maximum-magnitude squares require 66 accumulator bits")
  require(epsilonQ32 > 0 && epsilonQ32 < (BigInt(1) << 53))
  require(tagWidth >= 1)

  val lengthWidth: Int = log2Up(maxLength + 1)
  val indexWidth: Int = log2Up(maxLength)
  val magnitudeWidth: Int = narrowWidth - 1
  val highLimbWidth: Int = magnitudeWidth - limbWidth
  val lowProductWidth: Int = magnitudeWidth + limbWidth
  val highProductWidth: Int = magnitudeWidth + highLimbWidth
  val productWidth: Int = magnitudeWidth * 2
  val sqrtInputWidth: Int = 54
  val sqrtRootWidth: Int = sqrtInputWidth / 2
  val sqrtRemainderWidth: Int = sqrtRootWidth + 3
  val gainDivideWidth: Int = 40
}

case class BitNetSharedRmsCommand(cfg: BitNetSharedSpuRmsNormConfig)
    extends Bundle {
  val length = UInt(cfg.lengthWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val sourceBuffer = BitNetPhysicalActivationBuffer()
  val tag = UInt(cfg.tagWidth bits)
}

case class BitNetSharedRmsMeasureBeat(
    cfg: BitNetSharedSpuRmsNormConfig) extends Bundle {
  val value = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedRmsReplayBeat(
    cfg: BitNetSharedSpuRmsNormConfig) extends Bundle {
  val value = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val gamma = SInt(cfg.dataWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
}

/** Handshake from the SPU to the activation-fabric replay coordinator. */
case class BitNetSharedRmsReplayRequest(
    cfg: BitNetSharedSpuRmsNormConfig) extends Bundle {
  val length = UInt(cfg.lengthWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val sourceBuffer = BitNetPhysicalActivationBuffer()
  val tag = UInt(cfg.tagWidth bits)
  /** Reads precede writes by the SPU pipeline, so the same pool is legal. */
  val inPlace = Bool()
}

case class BitNetSharedRmsResultBeat(
    cfg: BitNetSharedSpuRmsNormConfig) extends Bundle {
  val value = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
  val tag = UInt(cfg.tagWidth bits)
}

case class BitNetSharedRmsDone(
    cfg: BitNetSharedSpuRmsNormConfig) extends Bundle {
  val tag = UInt(cfg.tagWidth bits)
  val protocolFault = Bool()
  val overflow = Bool()
}

class BitNetSharedSpuRmsNorm(
    cfg: BitNetSharedSpuRmsNormConfig =
      BitNetSharedSpuRmsNormConfig()) extends Component {
  val io = new Bundle {
    val command = slave(Stream(BitNetSharedRmsCommand(cfg)))
    val measureInput = slave(Stream(BitNetSharedRmsMeasureBeat(cfg)))
    val replayRequest = master(Stream(BitNetSharedRmsReplayRequest(cfg)))
    val replayInput = slave(Stream(BitNetSharedRmsReplayBeat(cfg)))
    val output = master(Stream(BitNetSharedRmsResultBeat(cfg)))
    val done = master(Stream(BitNetSharedRmsDone(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
    /** Exposed for deterministic diagnostics and bit-accurate integration. */
    val gainQ16 = out Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  }

  object State extends SpinalEnum {
    val Idle, MeasureAccept, MeasureMultiply, MeasureCombine, MeasureCommit,
        MeanInit, MeanIter, MeanRound,
        SqrtInit, SqrtIter, GainInit, GainIter, GainRound, LaneAdvance,
        RequestReplay, ReplayAccept, ReplayProductMultiply,
        ReplayProductCombine,
        ReplayProductNarrow, ReplayGainMultiply, ReplayGainCombine,
        ReplayOutput, Complete = newElement()
  }
  val state = Reg(State()) init State.Idle

  val lengthReg = Reg(UInt(cfg.lengthWidth bits)) init 0
  val tokenMaskReg = Reg(Bits(cfg.tokenLanes bits)) init 0
  val sourceBufferReg = Reg(BitNetPhysicalActivationBuffer())
    .init(BitNetPhysicalActivationBuffer.HiddenResident)
  val tagReg = Reg(UInt(cfg.tagWidth bits)) init 0
  val measureIndex = Reg(UInt(cfg.indexWidth bits)) init 0
  val replayIndex = Reg(UInt(cfg.indexWidth bits)) init 0
  val laneIndex = Reg(UInt(log2Up(cfg.tokenLanes) bits)) init 0

  val sumSquares = Vec(
    Reg(UInt(cfg.accumulatorWidth bits)) init 0,
    cfg.tokenLanes)
  val gains = Vec(Reg(UInt(cfg.magnitudeWidth bits)) init 0, cfg.tokenLanes)
  io.gainQ16 := gains

  val globalProtocolFault = RegInit(False)
  val globalOverflow = RegInit(False)
  val jobProtocolFault = RegInit(False)
  val jobOverflow = RegInit(False)
  io.protocolFaultSticky := globalProtocolFault
  io.overflowSticky := globalOverflow

  private val narrowPositiveMax = (BigInt(1) << (cfg.narrowWidth - 1)) - 1
  private val narrowNegativeMin = -narrowPositiveMax

  /** Explicit signed-Q16.16 narrowing; the returned magnitude is exact. */
  def narrowedMagnitude(value: SInt): (SInt, UInt, Bool, Bool) = {
    val narrowed = SInt(cfg.narrowWidth bits)
    val clipped = Bool()
    when(value > S(narrowPositiveMax, cfg.dataWidth bits)) {
      narrowed := S(narrowPositiveMax, cfg.narrowWidth bits)
      clipped := True
    } elsewhen(value < S(narrowNegativeMin, cfg.dataWidth bits)) {
      narrowed := S(narrowNegativeMin, cfg.narrowWidth bits)
      clipped := True
    } otherwise {
      narrowed := value.resize(cfg.narrowWidth)
      clipped := False
    }
    val negative = narrowed.msb
    val magnitude = UInt(cfg.magnitudeWidth bits)
    magnitude := Mux(
      negative,
      (-narrowed.resize(cfg.narrowWidth + 1)).asUInt
        .resize(cfg.magnitudeWidth),
      narrowed.asUInt.resize(cfg.magnitudeWidth))
    (narrowed, magnitude, negative, clipped)
  }

  val measureMagnitude = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  val measureClipped = Bits(cfg.tokenLanes bits)
  val replayXMagnitude = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  val replayXNegative = Bits(cfg.tokenLanes bits)
  val replayXClipped = Bits(cfg.tokenLanes bits)
  val replayGammaMagnitude = UInt(cfg.magnitudeWidth bits)
  val replayGammaNegative = Bool()
  val replayGammaClipped = Bool()

  for (lane <- 0 until cfg.tokenLanes) {
    val (_, measureMag, _, measureClip) =
      narrowedMagnitude(io.measureInput.payload.value(lane))
    measureMagnitude(lane) := measureMag
    measureClipped(lane) := measureClip

    val (_, replayMag, replayNeg, replayClip) =
      narrowedMagnitude(io.replayInput.payload.value(lane))
    replayXMagnitude(lane) := replayMag
    replayXNegative(lane) := replayNeg
    replayXClipped(lane) := replayClip
  }
  val (_, gammaMag, gammaNeg, gammaClip) =
    narrowedMagnitude(io.replayInput.payload.gamma)
  replayGammaMagnitude := gammaMag
  replayGammaNegative := gammaNeg
  replayGammaClipped := gammaClip

  /*
   * The activation/gamma memories terminate here.  In particular, do not let
   * a BRAM output traverse the signed clamp/absolute-value carry chain and a
   * DSP48 in the same cycle.  These registers are also the ownership point
   * for an accepted Stream beat: downstream multiplier cycles never observe
   * live input payloads after ready is withdrawn.
   */
  val stagedMeasureMagnitude = Vec(
    Reg(UInt(cfg.magnitudeWidth bits)) init 0, cfg.tokenLanes)
  val stagedReplayXMagnitude = Vec(
    Reg(UInt(cfg.magnitudeWidth bits)) init 0, cfg.tokenLanes)
  val stagedReplayXNegative = Reg(Bits(cfg.tokenLanes bits)) init 0
  val stagedReplayGammaMagnitude =
    Reg(UInt(cfg.magnitudeWidth bits)) init 0
  val stagedReplayGammaNegative = RegInit(False)

  val productReg = Vec(Reg(UInt(cfg.productWidth bits)) init 0,
    cfg.tokenLanes)
  val replayProductNegative = Reg(Bits(cfg.tokenLanes bits)) init 0
  val replayIntermediateMagnitude = Vec(
    Reg(UInt(cfg.magnitudeWidth bits)) init 0,
    cfg.tokenLanes)
  val replayIntermediateNegative = Reg(Bits(cfg.tokenLanes bits)) init 0

  /*
   * One exact magnitude multiplier per lane.  Keeping both products outside
   * state branches makes the physical DSP bank unambiguous to synthesis.
   */
  val multiplierA = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  val multiplierB = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    multiplierA(lane) := 0
    multiplierB(lane) := 0
    when(state === State.MeasureMultiply) {
      multiplierA(lane) := stagedMeasureMagnitude(lane)
      multiplierB(lane) := stagedMeasureMagnitude(lane)
    }
    when(state === State.ReplayProductMultiply) {
      multiplierA(lane) := stagedReplayXMagnitude(lane)
      multiplierB(lane) := stagedReplayGammaMagnitude
    }
    when(state === State.ReplayGainMultiply) {
      multiplierA(lane) := replayIntermediateMagnitude(lane)
      multiplierB(lane) := gains(lane)
    }
  }

  val multiplierLow = Vec(UInt(cfg.lowProductWidth bits), cfg.tokenLanes)
  val multiplierHigh = Vec(UInt(cfg.highProductWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val low = multiplierA(lane) *
      multiplierB(lane)(cfg.limbWidth - 1 downto 0)
    val high = multiplierA(lane) *
      multiplierB(lane)(cfg.magnitudeWidth - 1 downto cfg.limbWidth)
    low.addAttribute("use_dsp", "yes")
    high.addAttribute("use_dsp", "yes")
    multiplierLow(lane) := low
    multiplierHigh(lane) := high
  }

  val multiplierLowReg = Vec(
    Reg(UInt(cfg.lowProductWidth bits)) init 0, cfg.tokenLanes)
  val multiplierHighReg = Vec(
    Reg(UInt(cfg.highProductWidth bits)) init 0, cfg.tokenLanes)
  val reconstructedProduct = Vec(UInt(cfg.productWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val shiftedHigh =
      (multiplierHighReg(lane).resize(cfg.productWidth) |<< cfg.limbWidth)
        .resize(cfg.productWidth)
    reconstructedProduct(lane) :=
      (multiplierLowReg(lane).resize(cfg.productWidth) + shiftedHigh)
        .resize(cfg.productWidth)
  }

  /* Positive magnitude rounding and explicit signed-27 saturation. */
  val intermediateRounded = Vec(UInt((cfg.productWidth + 1) bits),
    cfg.tokenLanes)
  val intermediateMagnitudeNext = Vec(UInt(cfg.magnitudeWidth bits),
    cfg.tokenLanes)
  val intermediateOverflow = Bits(cfg.tokenLanes bits)
  for (lane <- 0 until cfg.tokenLanes) {
    intermediateRounded(lane) :=
      ((productReg(lane).resize(cfg.productWidth + 1) +
        U(BigInt(1) << (cfg.fractionalBits - 1),
          cfg.productWidth + 1 bits)) >> cfg.fractionalBits)
        .resize(cfg.productWidth + 1)
    val limit = UInt((cfg.productWidth + 1) bits)
    limit := U(narrowPositiveMax, cfg.productWidth + 1 bits)
    intermediateOverflow(lane) := intermediateRounded(lane) > limit
    intermediateMagnitudeNext(lane) := Mux(
      intermediateRounded(lane) > limit,
      limit.resize(cfg.magnitudeWidth),
      intermediateRounded(lane).resize(cfg.magnitudeWidth))
  }

  val outputRounded = Vec(UInt((cfg.productWidth + 1) bits), cfg.tokenLanes)
  val outputOverflow = Bits(cfg.tokenLanes bits)
  val outputValue = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    outputRounded(lane) :=
      ((productReg(lane).resize(cfg.productWidth + 1) +
        U(BigInt(1) << (cfg.fractionalBits - 1),
          cfg.productWidth + 1 bits)) >> cfg.fractionalBits)
        .resize(cfg.productWidth + 1)
    val outputLimit = UInt((cfg.productWidth + 1) bits)
    outputLimit := Mux(
      replayIntermediateNegative(lane),
      U(BigInt(1) << (cfg.dataWidth - 1), cfg.productWidth + 1 bits),
      U((BigInt(1) << (cfg.dataWidth - 1)) - 1,
        cfg.productWidth + 1 bits))
    outputOverflow(lane) := outputRounded(lane) > outputLimit
    val saturatedMagnitude = Mux(
      outputRounded(lane) > outputLimit,
      outputLimit,
      outputRounded(lane)).resize(cfg.dataWidth + 1)
    val signedPositive = saturatedMagnitude.asSInt.resize(cfg.dataWidth + 1)
    val signedValue = Mux(
      replayIntermediateNegative(lane),
      -signedPositive,
      signedPositive)
    outputValue(lane) := signedValue.resize(cfg.dataWidth)
  }

  /* Shared fixed-latency divider registers. */
  val divisionDividend = Reg(UInt(cfg.accumulatorWidth bits)) init 0
  val divisionDivisor = Reg(UInt(cfg.accumulatorWidth bits)) init 1
  val divisionRemainder = Reg(UInt((cfg.accumulatorWidth + 1) bits)) init 0
  val divisionQuotient = Reg(UInt(cfg.accumulatorWidth bits)) init 0
  val divisionCycles = Reg(UInt(log2Up(cfg.accumulatorWidth + 1) bits)) init 0
  // Both runtime divisors are narrow: length is 13 bits and rms is 27 bits.
  // Restoring division guarantees remainder<divisor, so carrying 67 bits
  // through the compare/subtract loop would only waste LUTs and timing.
  val divisionShiftedRemainder =
    (divisionRemainder(cfg.sqrtRootWidth - 1 downto 0) ##
      divisionDividend.msb).asUInt
  val divisionNarrowDivisor =
    divisionDivisor(cfg.sqrtRootWidth downto 0)
  val divisionSubtract = divisionShiftedRemainder >=
    divisionNarrowDivisor

  val meanRoundedReg = Reg(UInt(cfg.sqrtInputWidth bits)) init 0

  /* Shared radix-4 restoring integer square root. */
  val sqrtRadicand = Reg(UInt(cfg.sqrtInputWidth bits)) init 0
  val sqrtRemainder = Reg(UInt(cfg.sqrtRemainderWidth bits)) init 0
  val sqrtRoot = Reg(UInt(cfg.sqrtRootWidth bits)) init 0
  val sqrtCycles = Reg(UInt(log2Up(cfg.sqrtRootWidth + 1) bits)) init 0
  val sqrtBrought =
    ((sqrtRemainder.resize(cfg.sqrtRemainderWidth + 2) |<< 2) |
      sqrtRadicand(cfg.sqrtInputWidth - 1 downto
        cfg.sqrtInputWidth - 2).resize(cfg.sqrtRemainderWidth + 2))
      .resize(cfg.sqrtRemainderWidth + 2)
  val sqrtTrial =
    ((sqrtRoot.resize(cfg.sqrtRemainderWidth + 2) |<< 2) |
      U(1, cfg.sqrtRemainderWidth + 2 bits))
      .resize(cfg.sqrtRemainderWidth + 2)
  val sqrtTake = sqrtBrought >= sqrtTrial

  io.command.ready := state === State.Idle
  io.measureInput.ready := state === State.MeasureAccept
  io.replayInput.ready := state === State.ReplayAccept
  io.busy := state =/= State.Idle

  io.replayRequest.valid := state === State.RequestReplay
  io.replayRequest.payload.length := lengthReg
  io.replayRequest.payload.tokenMask := tokenMaskReg
  io.replayRequest.payload.sourceBuffer := sourceBufferReg
  io.replayRequest.payload.tag := tagReg
  io.replayRequest.payload.inPlace := True

  io.output.valid := state === State.ReplayOutput
  io.output.payload.tokenMask := tokenMaskReg
  io.output.payload.featureIndex := replayIndex
  io.output.payload.first := replayIndex === 0
  io.output.payload.last :=
    replayIndex.resize(cfg.lengthWidth) === (lengthReg - 1)
  io.output.payload.tag := tagReg
  for (lane <- 0 until cfg.tokenLanes) {
    io.output.payload.value(lane) := Mux(
      tokenMaskReg(lane), outputValue(lane), S(0, cfg.dataWidth bits))
  }

  io.done.valid := state === State.Complete
  io.done.payload.tag := tagReg
  io.done.payload.protocolFault := jobProtocolFault
  io.done.payload.overflow := jobOverflow

  val expectedMeasureFirst = measureIndex === 0
  val expectedMeasureLast =
    measureIndex.resize(cfg.lengthWidth) === (lengthReg - 1)
  val expectedReplayFirst = replayIndex === 0
  val expectedReplayLast =
    replayIndex.resize(cfg.lengthWidth) === (lengthReg - 1)

  switch(state) {
    is(State.Idle) {
      when(io.command.fire) {
        lengthReg := io.command.payload.length
        tokenMaskReg := io.command.payload.tokenMask
        sourceBufferReg := io.command.payload.sourceBuffer
        tagReg := io.command.payload.tag
        measureIndex := 0
        replayIndex := 0
        laneIndex := 0
        jobProtocolFault := False
        jobOverflow := False
        for (lane <- 0 until cfg.tokenLanes) {
          sumSquares(lane) := 0
          gains(lane) := 0
        }
        when(io.command.payload.length === 0 ||
          io.command.payload.length > cfg.maxLength ||
          !io.command.payload.tokenMask.orR) {
          jobProtocolFault := True
          globalProtocolFault := True
          state := State.Complete
        } otherwise {
          state := State.MeasureAccept
        }
      }
    }

    is(State.MeasureAccept) {
      when(io.measureInput.fire) {
        for (lane <- 0 until cfg.tokenLanes) {
          stagedMeasureMagnitude(lane) := measureMagnitude(lane)
        }
        when(io.measureInput.payload.tokenMask =/= tokenMaskReg ||
          io.measureInput.payload.first =/= expectedMeasureFirst ||
          io.measureInput.payload.last =/= expectedMeasureLast) {
          jobProtocolFault := True
          globalProtocolFault := True
        }
        when((measureClipped & tokenMaskReg).orR) {
          jobOverflow := True
          globalOverflow := True
        }
        state := State.MeasureMultiply
      }
    }

    is(State.MeasureMultiply) {
      for (lane <- 0 until cfg.tokenLanes) {
        multiplierLowReg(lane) := multiplierLow(lane)
        multiplierHighReg(lane) := multiplierHigh(lane)
      }
      state := State.MeasureCombine
    }

    is(State.MeasureCombine) {
      for (lane <- 0 until cfg.tokenLanes) {
        productReg(lane) := reconstructedProduct(lane)
      }
      state := State.MeasureCommit
    }

    is(State.MeasureCommit) {
      for (lane <- 0 until cfg.tokenLanes) {
        when(tokenMaskReg(lane)) {
          sumSquares(lane) :=
            (sumSquares(lane) + productReg(lane)
              .resize(cfg.accumulatorWidth)).resize(cfg.accumulatorWidth)
        }
      }
      when(expectedMeasureLast) {
        laneIndex := 0
        state := State.MeanInit
      } otherwise {
        measureIndex := measureIndex + 1
        state := State.MeasureAccept
      }
    }

    is(State.MeanInit) {
      when(!tokenMaskReg(laneIndex)) {
        gains(laneIndex) := U(0, cfg.magnitudeWidth bits)
        state := State.LaneAdvance
      } otherwise {
        divisionDividend := sumSquares(laneIndex)
        divisionDivisor := lengthReg.resize(cfg.accumulatorWidth)
        divisionRemainder := 0
        divisionQuotient := 0
        divisionCycles := cfg.accumulatorWidth
        state := State.MeanIter
      }
    }

    is(State.MeanIter) {
      divisionDividend :=
        (divisionDividend |<< 1).resize(cfg.accumulatorWidth)
      divisionQuotient :=
        ((divisionQuotient |<< 1).resize(cfg.accumulatorWidth) |
          divisionSubtract.asUInt.resize(cfg.accumulatorWidth))
      divisionRemainder := Mux(
        divisionSubtract,
        divisionShiftedRemainder -
          divisionNarrowDivisor,
        divisionShiftedRemainder).resize(cfg.accumulatorWidth + 1)
      divisionCycles := divisionCycles - 1
      when(divisionCycles === 1) {
        state := State.MeanRound
      }
    }

    is(State.MeanRound) {
      val remainderTwice =
        (divisionRemainder(cfg.lengthWidth - 1 downto 0)
          .resize(cfg.lengthWidth + 1) |<< 1)
      val roundUp = remainderTwice >=
        divisionDivisor(cfg.lengthWidth downto 0)
      val rounded =
        (divisionQuotient(cfg.sqrtInputWidth - 1 downto 0) +
          roundUp.asUInt.resize(cfg.sqrtInputWidth))
          .resize(cfg.sqrtInputWidth)
      meanRoundedReg := rounded
      // By construction sum <= length * (2^26-1)^2, so the rounded mean
      // is strictly below 2^52 and cannot overflow sqrtInputWidth=54.
      state := State.SqrtInit
    }

    is(State.SqrtInit) {
      val radicandWide = meanRoundedReg.resize(cfg.sqrtInputWidth + 1) +
        U(cfg.epsilonQ32, cfg.sqrtInputWidth + 1 bits)
      sqrtRadicand := radicandWide.resize(cfg.sqrtInputWidth)
      sqrtRemainder := 0
      sqrtRoot := 0
      sqrtCycles := cfg.sqrtRootWidth
      // The preceding bound plus epsilonQ32 < 2^53 proves this resize exact.
      state := State.SqrtIter
    }

    is(State.SqrtIter) {
      sqrtRadicand := (sqrtRadicand |<< 2).resize(cfg.sqrtInputWidth)
      sqrtRemainder := Mux(
        sqrtTake,
        (sqrtBrought - sqrtTrial).resize(cfg.sqrtRemainderWidth),
        sqrtBrought.resize(cfg.sqrtRemainderWidth))
      sqrtRoot :=
        ((sqrtRoot |<< 1).resize(cfg.sqrtRootWidth) |
          sqrtTake.asUInt.resize(cfg.sqrtRootWidth))
      sqrtCycles := sqrtCycles - 1
      when(sqrtCycles === 1) {
        state := State.GainInit
      }
    }

    is(State.GainInit) {
      /* Left-align the 40-bit reciprocal numerator in the shared divider. */
      divisionDividend :=
        (U(BigInt(1) << 32, cfg.accumulatorWidth bits) |
          U(0, cfg.accumulatorWidth bits)) |<<
          (cfg.accumulatorWidth - cfg.gainDivideWidth)
      divisionDivisor := sqrtRoot.resize(cfg.accumulatorWidth)
      divisionRemainder := 0
      divisionQuotient := 0
      divisionCycles := cfg.gainDivideWidth
      state := State.GainIter
    }

    is(State.GainIter) {
      divisionDividend :=
        (divisionDividend |<< 1).resize(cfg.accumulatorWidth)
      divisionQuotient :=
        ((divisionQuotient |<< 1).resize(cfg.accumulatorWidth) |
          divisionSubtract.asUInt.resize(cfg.accumulatorWidth))
      divisionRemainder := Mux(
        divisionSubtract,
        divisionShiftedRemainder -
          divisionNarrowDivisor,
        divisionShiftedRemainder).resize(cfg.accumulatorWidth + 1)
      divisionCycles := divisionCycles - 1
      when(divisionCycles === 1) {
        state := State.GainRound
      }
    }

    is(State.GainRound) {
      val remainderTwice =
        (divisionRemainder(cfg.sqrtRootWidth - 1 downto 0)
          .resize(cfg.sqrtRootWidth + 1) |<< 1)
      val roundUp = remainderTwice >=
        divisionDivisor(cfg.sqrtRootWidth downto 0)
      val rounded =
        (divisionQuotient(cfg.gainDivideWidth - 1 downto 0) +
          roundUp.asUInt.resize(cfg.gainDivideWidth))
          .resize(cfg.gainDivideWidth)
      when(rounded > narrowPositiveMax) {
        gains(laneIndex) := U(narrowPositiveMax, cfg.magnitudeWidth bits)
        jobOverflow := True
        globalOverflow := True
      } otherwise {
        gains(laneIndex) := rounded.resize(cfg.magnitudeWidth)
      }
      state := State.LaneAdvance
    }

    is(State.LaneAdvance) {
      when(laneIndex === cfg.tokenLanes - 1) {
        replayIndex := 0
        state := State.RequestReplay
      } otherwise {
        laneIndex := laneIndex + 1
        state := State.MeanInit
      }
    }

    is(State.RequestReplay) {
      when(io.replayRequest.fire) {
        state := State.ReplayAccept
      }
    }

    is(State.ReplayAccept) {
      when(io.replayInput.fire) {
        for (lane <- 0 until cfg.tokenLanes) {
          stagedReplayXMagnitude(lane) := replayXMagnitude(lane)
          stagedReplayXNegative(lane) := replayXNegative(lane)
        }
        stagedReplayGammaMagnitude := replayGammaMagnitude
        stagedReplayGammaNegative := replayGammaNegative
        when(io.replayInput.payload.tokenMask =/= tokenMaskReg ||
          io.replayInput.payload.first =/= expectedReplayFirst ||
          io.replayInput.payload.last =/= expectedReplayLast) {
          jobProtocolFault := True
          globalProtocolFault := True
        }
        when(((replayXClipped & tokenMaskReg).orR) ||
          replayGammaClipped) {
          jobOverflow := True
          globalOverflow := True
        }
        state := State.ReplayProductMultiply
      }
    }

    is(State.ReplayProductMultiply) {
      for (lane <- 0 until cfg.tokenLanes) {
        multiplierLowReg(lane) := multiplierLow(lane)
        multiplierHighReg(lane) := multiplierHigh(lane)
        replayProductNegative(lane) :=
          stagedReplayXNegative(lane) ^ stagedReplayGammaNegative
      }
      state := State.ReplayProductCombine
    }

    is(State.ReplayProductCombine) {
      for (lane <- 0 until cfg.tokenLanes) {
        productReg(lane) := reconstructedProduct(lane)
      }
      state := State.ReplayProductNarrow
    }

    is(State.ReplayProductNarrow) {
      for (lane <- 0 until cfg.tokenLanes) {
        replayIntermediateMagnitude(lane) :=
          Mux(tokenMaskReg(lane), intermediateMagnitudeNext(lane), U(0))
        replayIntermediateNegative(lane) :=
          tokenMaskReg(lane) && replayProductNegative(lane) &&
            intermediateMagnitudeNext(lane) =/= 0
      }
      when((intermediateOverflow & tokenMaskReg).orR) {
        jobOverflow := True
        globalOverflow := True
      }
      state := State.ReplayGainMultiply
    }

    is(State.ReplayGainMultiply) {
      for (lane <- 0 until cfg.tokenLanes) {
        multiplierLowReg(lane) := multiplierLow(lane)
        multiplierHighReg(lane) := multiplierHigh(lane)
      }
      state := State.ReplayGainCombine
    }

    is(State.ReplayGainCombine) {
      for (lane <- 0 until cfg.tokenLanes) {
        productReg(lane) := reconstructedProduct(lane)
      }
      state := State.ReplayOutput
    }

    is(State.ReplayOutput) {
      when(io.output.fire) {
        when((outputOverflow & tokenMaskReg).orR) {
          jobOverflow := True
          globalOverflow := True
        }
        when(expectedReplayLast) {
          state := State.Complete
        } otherwise {
          replayIndex := replayIndex + 1
          state := State.ReplayAccept
        }
      }
    }

    is(State.Complete) {
      when(io.done.fire) {
        state := State.Idle
      }
    }
  }
}

/** Bit-accurate Scala oracle for the explicit production narrowing contract. */
object BitNetSharedSpuRmsNormMath {
  case class Result(
      values: Seq[Seq[Int]],
      gainQ16: Seq[BigInt],
      overflow: Boolean)

  private val NarrowMax = (BigInt(1) << 26) - 1
  private val NarrowMin = -NarrowMax
  private val I32Min = -(BigInt(1) << 31)
  private val I32Max = (BigInt(1) << 31) - 1

  private def clamp(value: BigInt, minimum: BigInt, maximum: BigInt):
      (BigInt, Boolean) =
    if (value < minimum) (minimum, true)
    else if (value > maximum) (maximum, true)
    else (value, false)

  private def roundAwayMagnitude(productMagnitude: BigInt): BigInt =
    (productMagnitude + (BigInt(1) << 15)) >> 16

  private def roundedSignedProduct(a: BigInt, b: BigInt): BigInt = {
    val magnitude = roundAwayMagnitude(a.abs * b.abs)
    if ((a < 0) ^ (b < 0)) -magnitude else magnitude
  }

  private def integerSqrt(value: BigInt): BigInt = {
    require(value >= 0)
    if (value == 0) return 0
    var lower = BigInt(0)
    var upper = BigInt(1) << ((value.bitLength + 1) / 2)
    while (lower + 1 < upper) {
      val middle = (lower + upper) >> 1
      if (middle * middle <= value) lower = middle else upper = middle
    }
    lower
  }

  private def divideNearest(numerator: BigInt, denominator: BigInt): BigInt = {
    require(numerator >= 0 && denominator > 0)
    val quotient = numerator / denominator
    val remainder = numerator % denominator
    quotient + (if (remainder * 2 >= denominator) 1 else 0)
  }

  def normalize(
      tokens: Seq[Seq[Int]],
      gamma: Seq[Int],
      tokenMask: Int,
      epsilonQ32: BigInt = BigInt(42950)): Result = {
    require(tokens.length == 4)
    require(tokens.map(_.length).distinct.size == 1)
    require(tokens.head.nonEmpty && gamma.length == tokens.head.length)

    var overflow = false
    val narrowedTokens = tokens.zipWithIndex.map { case (token, lane) => token.map { raw =>
      val (value, clipped) = clamp(BigInt(raw), NarrowMin, NarrowMax)
      if (((tokenMask >> lane) & 1) != 0) overflow ||= clipped
      value
    }}
    val narrowedGamma = gamma.map { raw =>
      val (value, clipped) = clamp(BigInt(raw), NarrowMin, NarrowMax)
      overflow ||= clipped
      value
    }

    val gains = narrowedTokens.indices.map { lane =>
      if (((tokenMask >> lane) & 1) == 0) BigInt(0)
      else {
        val sum = narrowedTokens(lane).map(x => x * x).sum
        val mean = divideNearest(sum, tokens(lane).length)
        val rms = integerSqrt(mean + epsilonQ32)
        val rawGain = divideNearest(BigInt(1) << 32, rms)
        val (gain, clipped) = clamp(rawGain, 0, NarrowMax)
        overflow ||= clipped
        gain
      }
    }

    val output = tokens.indices.map { lane =>
      tokens(lane).indices.map { index =>
        if (((tokenMask >> lane) & 1) == 0) 0
        else {
          val first = roundedSignedProduct(
            narrowedTokens(lane)(index), narrowedGamma(index))
          val (intermediate, firstClipped) =
            clamp(first, NarrowMin, NarrowMax)
          overflow ||= firstClipped
          val second = roundedSignedProduct(intermediate, gains(lane))
          val (result, secondClipped) = clamp(second, I32Min, I32Max)
          overflow ||= secondClipped
          result.toInt
        }
      }
    }
    Result(output, gains, overflow)
  }
}

object GenerateBitNetSharedSpuRmsNorm extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-shared-spu-rmsnorm")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetSharedSpuRmsNorm())
}
