package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Physical/numeric contract of the production Ultra-BitNet FFN gate.
  *
  * The fused Gate+Up projection emits raw signed-i32 accumulators.  Each
  * token lane has a positive Q16.16 Gate scale and Up scale (the product of
  * the activation scale and the corresponding BF16 row scale, converted by
  * the scale-staging block).  The gate first evaluates
  *
  *   g = sat_symmetric_s27(round_away(gateRaw * gateScaleQ16 / 2^16))
  *   u = sat_symmetric_s27(round_away(upRaw   * upScaleQ16   / 2^16))
  *   y = sat_i32(round_away((g > 0 ? g*g*u : 0) / 2^32))
  *
  * so g, u and y are signed Q16.16.  The S27 narrowing is intentional and
  * matches the input contract of [[BitNetSharedSpuRmsNorm]]; every clamp is
  * surfaced through the per-frame and sticky overflow outputs.
  *
  * There are only eight physical 26x18 unsigned magnitude multipliers (two
  * DSP48E2 sites for each of four token lanes).  They are time-shared across
  * projection scaling, g^2, and the three limbs of g^2*|u|.  This preserves a
  * single rounding point for the cubic product without the reference core's
  * four parallel 32x32 and four parallel 64x32 multipliers.  The throughput
  * trade-off is harmless behind the much longer matrix/DDR phase and avoids
  * an extra Gate/Up activation RAM.
  */
case class BitNetSharedSpuFfnGateConfig(
    model: BitNetConfig = BitNetConfig.Production,
    maxLength: Int = 6912,
    dataWidth: Int = 32,
    fractionalBits: Int = 16,
    narrowWidth: Int = 27,
    limbWidth: Int = 18,
    tagWidth: Int = 8
) {
  require(model.tokenParallelism == 2 || model.tokenParallelism == 4,
    "the production shared FFN gate must be TP2 or TP4")
  require(model.spuCount == 4,
    "WideWork scatter assumes four activation stripes")
  require(maxLength == model.ffnSize && maxLength == 6912,
    "the resident FFN frame is exactly 6912 features")
  require(dataWidth == 32 && fractionalBits == 16,
    "the resident activation contract is signed Q16.16")
  require(narrowWidth == 27 && limbWidth == 18,
    "DSP48E2 limb decomposition is fixed at 26x18")
  require(tagWidth >= 1)

  val tokenLanes: Int = model.tokenParallelism
  val magnitudeWidth: Int = narrowWidth - 1
  val rawMagnitudeWidth: Int = dataWidth
  val scaleWidth: Int = magnitudeWidth
  val lengthWidth: Int = log2Up(maxLength + 1)
  val indexWidth: Int = log2Up(maxLength)
  val scaleProductWidth: Int = rawMagnitudeWidth + scaleWidth
  val squareWidth: Int = magnitudeWidth * 2
  val cubicWidth: Int = squareWidth + magnitudeWidth
  val roundedCubicWidth: Int = cubicWidth + 1 - 2 * fractionalBits
  val multiplierProductWidth: Int = magnitudeWidth + limbWidth

  def rmsConfig: BitNetSharedSpuRmsNormConfig =
    BitNetSharedSpuRmsNormConfig(
      tokenLanes = tokenLanes,
      maxLength = maxLength,
      dataWidth = dataWidth,
      fractionalBits = fractionalBits,
      narrowWidth = narrowWidth,
      limbWidth = limbWidth,
      tagWidth = tagWidth
    )
}

/** One resident FFN-gate job.  Combined scales are positive Q16.16. */
case class BitNetSharedFfnGateCommand(
    cfg: BitNetSharedSpuFfnGateConfig) extends Bundle {
  val length = UInt(cfg.lengthWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val mode = BitNetMode()
  /**
    * Permit semantic feature addresses to arrive out of order only when the
    * stream is produced by the resident GateUp pairing boundary.  That
    * boundary has already authenticated every physical Gate/Up row and
    * guarantees a one-to-one semantic mapping.  Direct/research callers leave
    * this clear and retain the stricter monotonically increasing contract.
    */
  val authenticatedScatter = Bool()
  val gateScaleQ16 = Vec(UInt(cfg.scaleWidth bits), cfg.tokenLanes)
  val upScaleQ16 = Vec(UInt(cfg.scaleWidth bits), cfg.tokenLanes)
  val tag = UInt(cfg.tagWidth bits)
}

/** One feature from a fused [Gate rows][Up rows] projection result stream. */
case class BitNetFusedGateUpBeat(
    cfg: BitNetSharedSpuFfnGateConfig) extends Bundle {
  val gateRaw = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val upRaw = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val featureIndex = UInt(cfg.indexWidth bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetSharedFfnGateDone(
    cfg: BitNetSharedSpuFfnGateConfig) extends Bundle {
  val tag = UInt(cfg.tagWidth bits)
  val length = UInt(cfg.lengthWidth bits)
  val protocolFault = Bool()
  val overflow = Bool()
  /** False only for a rejected zero/out-of-range command. */
  val rmsCommandIssued = Bool()
}

class BitNetSharedSpuFfnGate(
    cfg: BitNetSharedSpuFfnGateConfig =
      BitNetSharedSpuFfnGateConfig()) extends Component {
  private val model = cfg.model
  private val layout = BitNetStreamingActivationLayout(model)
  private val rmsCfg = cfg.rmsConfig

  val io = new Bundle {
    val command = slave(Stream(BitNetSharedFfnGateCommand(cfg)))
    val fusedInput = slave(Stream(BitNetFusedGateUpBeat(cfg)))

    /**
      * Direct writes to the four token ports of the liveness-overlaid
      * activation fabric.  Each result updates one of the four WideWork
      * stripes; per-port pending bits make arbitrary independent
      * backpressure safe without duplicating writes.
      */
    val wideWrite = Vec(
      master(Stream(BitNetPhysicalActivationWriteBeat(model, layout))),
      cfg.tokenLanes)

    /** Issued only after every WideWork write in the frame has completed. */
    val rmsCommand = master(Stream(BitNetSharedRmsCommand(rmsCfg)))
    val done = master(Stream(BitNetSharedFfnGateDone(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
    val acceptedFrames = out UInt(32 bits)
    val completedFrames = out UInt(32 bits)
    val acceptedFeatures = out UInt(32 bits)
    val emittedFeatures = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Idle, AcceptFeature,
        GateScaleMultiply, GateScaleCombine, GateScaleRound,
        UpScaleMultiply, UpScaleCombine, UpScaleRound,
        SquareMultiply, SquareCombine,
        CubicLowMidMultiply, CubicHighMultiply, CubicCombine, CubicRound,
        Saturate, WriteWide, IssueRms, Complete = newElement()
  }
  val state = Reg(State()) init State.Idle
  io.busy := state =/= State.Idle

  val lengthReg = Reg(UInt(cfg.lengthWidth bits)) init 0
  val tokenMaskReg = Reg(Bits(cfg.tokenLanes bits)) init 0
  val tagReg = Reg(UInt(cfg.tagWidth bits)) init 0
  val gateScaleReg = Vec(
    Reg(UInt(cfg.scaleWidth bits)) init 0, cfg.tokenLanes)
  val upScaleReg = Vec(
    Reg(UInt(cfg.scaleWidth bits)) init 0, cfg.tokenLanes)
  /** Stream ordinal and destination address are deliberately distinct.
    * Decode can interleave the two output shards while WideWork remains a
    * random-access, PL-local destination. */
  val featureOrdinal = Reg(UInt(cfg.indexWidth bits)) init 0
  val featureAddress = Reg(UInt(cfg.indexWidth bits)) init 0
  val authenticatedScatterReg = RegInit(False)

  val globalProtocolFault = RegInit(False)
  val globalOverflow = RegInit(False)
  val jobProtocolFault = RegInit(False)
  val jobOverflow = RegInit(False)
  val issueRmsForJob = RegInit(False)
  io.protocolFaultSticky := globalProtocolFault
  io.overflowSticky := globalOverflow

  val acceptedFrameCounter = Reg(UInt(32 bits)) init 0
  val completedFrameCounter = Reg(UInt(32 bits)) init 0
  val acceptedFeatureCounter = Reg(UInt(32 bits)) init 0
  val emittedFeatureCounter = Reg(UInt(32 bits)) init 0
  io.acceptedFrames := acceptedFrameCounter
  io.completedFrames := completedFrameCounter
  io.acceptedFeatures := acceptedFeatureCounter
  io.emittedFeatures := emittedFeatureCounter

  // Capture the exact unsigned magnitude of signed i32, including INT_MIN.
  def rawMagnitude(value: SInt): UInt = {
    val extended = value.resize(cfg.dataWidth + 1)
    Mux(value.msb, (-extended).asUInt, extended.asUInt)
      .resize(cfg.rawMagnitudeWidth)
  }

  val rawGateMagnitude = Vec(
    Reg(UInt(cfg.rawMagnitudeWidth bits)) init 0, cfg.tokenLanes)
  val rawUpMagnitude = Vec(
    Reg(UInt(cfg.rawMagnitudeWidth bits)) init 0, cfg.tokenLanes)
  val rawGatePositive = Reg(Bits(cfg.tokenLanes bits)) init 0
  val rawUpNegative = Reg(Bits(cfg.tokenLanes bits)) init 0

  val gateMagnitude = Vec(
    Reg(UInt(cfg.magnitudeWidth bits)) init 0, cfg.tokenLanes)
  val upMagnitude = Vec(
    Reg(UInt(cfg.magnitudeWidth bits)) init 0, cfg.tokenLanes)
  val gateEnabled = Reg(Bits(cfg.tokenLanes bits)) init 0
  val upNegative = Reg(Bits(cfg.tokenLanes bits)) init 0

  // Declared before the multiplier input mux because its three 18-bit limbs
  // feed the two cubic multiply phases below.
  val squareReg = Vec(Reg(UInt(cfg.squareWidth bits)) init 0, cfg.tokenLanes)

  /*
   * Exactly eight 26x18 multipliers.  All wide products below are rebuilt
   * from these registered magnitude limbs.
   */
  val multiplierA = Vec(UInt(cfg.magnitudeWidth bits), cfg.tokenLanes)
  val multiplierB0 = Vec(UInt(cfg.limbWidth bits), cfg.tokenLanes)
  val multiplierB1 = Vec(UInt(cfg.limbWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    multiplierA(lane) := 0
    multiplierB0(lane) := 0
    multiplierB1(lane) := 0

    when(state === State.GateScaleMultiply) {
      multiplierA(lane) := gateScaleReg(lane)
      multiplierB0(lane) :=
        rawGateMagnitude(lane)(cfg.limbWidth - 1 downto 0)
      multiplierB1(lane) :=
        rawGateMagnitude(lane)(cfg.rawMagnitudeWidth - 1 downto cfg.limbWidth)
          .resize(cfg.limbWidth)
    } elsewhen(state === State.UpScaleMultiply) {
      multiplierA(lane) := upScaleReg(lane)
      multiplierB0(lane) :=
        rawUpMagnitude(lane)(cfg.limbWidth - 1 downto 0)
      multiplierB1(lane) :=
        rawUpMagnitude(lane)(cfg.rawMagnitudeWidth - 1 downto cfg.limbWidth)
          .resize(cfg.limbWidth)
    } elsewhen(state === State.SquareMultiply) {
      multiplierA(lane) := gateMagnitude(lane)
      multiplierB0(lane) := gateMagnitude(lane)(cfg.limbWidth - 1 downto 0)
      multiplierB1(lane) :=
        gateMagnitude(lane)(cfg.magnitudeWidth - 1 downto cfg.limbWidth)
          .resize(cfg.limbWidth)
    } elsewhen(state === State.CubicLowMidMultiply) {
      multiplierA(lane) := upMagnitude(lane)
      multiplierB0(lane) := squareReg(lane)(cfg.limbWidth - 1 downto 0)
      multiplierB1(lane) := squareReg(lane)(2 * cfg.limbWidth - 1 downto
        cfg.limbWidth)
    } elsewhen(state === State.CubicHighMultiply) {
      multiplierA(lane) := upMagnitude(lane)
      multiplierB0(lane) :=
        squareReg(lane)(cfg.squareWidth - 1 downto 2 * cfg.limbWidth)
          .resize(cfg.limbWidth)
      multiplierB1(lane) := 0
    }
  }

  val product0 = Vec(UInt(cfg.multiplierProductWidth bits), cfg.tokenLanes)
  val product1 = Vec(UInt(cfg.multiplierProductWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val p0 = multiplierA(lane) * multiplierB0(lane)
    val p1 = multiplierA(lane) * multiplierB1(lane)
    p0.addAttribute("use_dsp", "yes")
    p1.addAttribute("use_dsp", "yes")
    product0(lane) := p0
    product1(lane) := p1
  }

  val product0Reg = Vec(
    Reg(UInt(cfg.multiplierProductWidth bits)) init 0, cfg.tokenLanes)
  val product1Reg = Vec(
    Reg(UInt(cfg.multiplierProductWidth bits)) init 0, cfg.tokenLanes)
  val scaleProductReg = Vec(
    Reg(UInt(cfg.scaleProductWidth bits)) init 0, cfg.tokenLanes)
  val cubicLowMidReg = Vec(
    Reg(UInt((2 * cfg.limbWidth + cfg.magnitudeWidth) bits)) init 0,
    cfg.tokenLanes)
  val cubicHighReg = Vec(
    Reg(UInt(cfg.multiplierProductWidth bits)) init 0, cfg.tokenLanes)
  val cubicProductReg = Vec(
    Reg(UInt(cfg.cubicWidth bits)) init 0, cfg.tokenLanes)
  val cubicRoundedReg = Vec(
    Reg(UInt(cfg.roundedCubicWidth bits)) init 0, cfg.tokenLanes)
  val resultReg = Vec(Reg(SInt(cfg.dataWidth bits)) init 0, cfg.tokenLanes)

  // Registered reconstruction nodes keep every carry chain separate from a
  // DSP output and from the following rounding/saturation comparison.
  val reconstructedScale = Vec(UInt(cfg.scaleProductWidth bits), cfg.tokenLanes)
  val reconstructedSquare = Vec(UInt(cfg.squareWidth bits), cfg.tokenLanes)
  val reconstructedLowMid = Vec(
    UInt((2 * cfg.limbWidth + cfg.magnitudeWidth) bits), cfg.tokenLanes)
  val reconstructedCubic = Vec(UInt(cfg.cubicWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    reconstructedScale(lane) :=
      (product0Reg(lane).resize(cfg.scaleProductWidth) +
        (product1Reg(lane).resize(cfg.scaleProductWidth) |<< cfg.limbWidth))
        .resize(cfg.scaleProductWidth)
    reconstructedSquare(lane) :=
      (product0Reg(lane).resize(cfg.squareWidth) +
        (product1Reg(lane).resize(cfg.squareWidth) |<< cfg.limbWidth))
        .resize(cfg.squareWidth)
    reconstructedLowMid(lane) :=
      (product0Reg(lane).resize(reconstructedLowMid(lane).getWidth) +
        (product1Reg(lane).resize(reconstructedLowMid(lane).getWidth) |<<
          cfg.limbWidth)).resize(reconstructedLowMid(lane).getWidth)
    reconstructedCubic(lane) :=
      (cubicLowMidReg(lane).resize(cfg.cubicWidth) +
        (cubicHighReg(lane).resize(cfg.cubicWidth) |<<
          (2 * cfg.limbWidth))).resize(cfg.cubicWidth)
  }

  val narrowLimit = U((BigInt(1) << cfg.magnitudeWidth) - 1,
    (cfg.scaleProductWidth + 1) bits)
  val scaledRounded = Vec(UInt((cfg.scaleProductWidth + 1) bits),
    cfg.tokenLanes)
  val scaledOverflow = Bits(cfg.tokenLanes bits)
  for (lane <- 0 until cfg.tokenLanes) {
    scaledRounded(lane) :=
      ((scaleProductReg(lane).resize(cfg.scaleProductWidth + 1) +
        U(BigInt(1) << (cfg.fractionalBits - 1),
          (cfg.scaleProductWidth + 1) bits)) >> cfg.fractionalBits)
        .resize(cfg.scaleProductWidth + 1)
    scaledOverflow(lane) := scaledRounded(lane) > narrowLimit
  }

  val roundedCubic = Vec(UInt(cfg.roundedCubicWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    roundedCubic(lane) :=
      ((cubicProductReg(lane).resize(cfg.cubicWidth + 1) +
        U(BigInt(1) << (2 * cfg.fractionalBits - 1),
          (cfg.cubicWidth + 1) bits)) >> (2 * cfg.fractionalBits))
        .resize(cfg.roundedCubicWidth)
  }

  val finalOverflow = Bits(cfg.tokenLanes bits)
  val finalValue = Vec(SInt(cfg.dataWidth bits), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    val limit = UInt(cfg.roundedCubicWidth bits)
    limit := Mux(
      upNegative(lane),
      U(BigInt(1) << (cfg.dataWidth - 1), cfg.roundedCubicWidth bits),
      U((BigInt(1) << (cfg.dataWidth - 1)) - 1,
        cfg.roundedCubicWidth bits))
    val enabled = gateEnabled(lane) && tokenMaskReg(lane)
    finalOverflow(lane) := enabled && cubicRoundedReg(lane) > limit
    val saturatedMagnitude = Mux(cubicRoundedReg(lane) > limit,
      limit, cubicRoundedReg(lane)).resize(cfg.dataWidth + 1)
    val signedMagnitude = saturatedMagnitude.asSInt
    finalValue(lane) := 0
    when(enabled) {
      when(upNegative(lane)) {
        finalValue(lane) := (-signedMagnitude).resize(cfg.dataWidth)
      } otherwise {
        finalValue(lane) := saturatedMagnitude.resize(cfg.dataWidth).asSInt
      }
    }
  }

  // ------------------------------------------------------------------
  // Public streams and frame protocol.
  // ------------------------------------------------------------------
  io.command.ready := state === State.Idle
  io.fusedInput.ready := state === State.AcceptFeature

  val commandLegal = io.command.payload.length =/= 0 &&
    io.command.payload.length <= U(cfg.maxLength, cfg.lengthWidth bits)
  val expectedLastFeature = featureOrdinal.resize(cfg.lengthWidth) ===
    (lengthReg - 1).resized
  val inputAddressLegal = io.fusedInput.payload.featureIndex.resize(
    cfg.lengthWidth) < lengthReg
  val inputOrderLegal = authenticatedScatterReg ||
    io.fusedInput.payload.featureIndex === featureOrdinal
  val beatProtocolFault =
    io.fusedInput.payload.tokenMask =/= tokenMaskReg ||
      !inputAddressLegal || !inputOrderLegal ||
      io.fusedInput.payload.first =/= (featureOrdinal === 0) ||
      io.fusedInput.payload.last =/= expectedLastFeature

  val writePending = Reg(Bits(cfg.tokenLanes bits)) init 0
  val completedWrites = Bits(cfg.tokenLanes bits)
  for (token <- 0 until cfg.tokenLanes) {
    completedWrites(token) := io.wideWrite(token).fire
    io.wideWrite(token).valid :=
      state === State.WriteWide && writePending(token)
    io.wideWrite(token).payload.buffer :=
      BitNetPhysicalActivationBuffer.WideWork
    io.wideWrite(token).payload.group :=
      (featureAddress >> log2Up(model.spuCount)).resized
    io.wideWrite(token).payload.featureValid :=
      UIntToOh(featureAddress(log2Up(model.spuCount) - 1 downto 0),
        model.spuCount).asBits
    for (shard <- 0 until model.spuCount) {
      io.wideWrite(token).payload.values(shard) := 0
    }
    io.wideWrite(token).payload.values(
      featureAddress(log2Up(model.spuCount) - 1 downto 0)) :=
      resultReg(token)
  }
  val pendingAfterHandshakes = writePending & ~completedWrites

  io.rmsCommand.valid := state === State.IssueRms
  io.rmsCommand.payload.length := lengthReg.resized
  io.rmsCommand.payload.tokenMask := tokenMaskReg
  io.rmsCommand.payload.sourceBuffer :=
    BitNetPhysicalActivationBuffer.WideWork
  io.rmsCommand.payload.tag := tagReg

  io.done.valid := state === State.Complete
  io.done.payload.tag := tagReg
  io.done.payload.length := lengthReg
  io.done.payload.protocolFault := jobProtocolFault
  io.done.payload.overflow := jobOverflow
  io.done.payload.rmsCommandIssued := issueRmsForJob

  switch(state) {
    is(State.Idle) {
      when(io.command.fire) {
        acceptedFrameCounter := acceptedFrameCounter + 1
        lengthReg := io.command.payload.length
        tokenMaskReg := io.command.payload.tokenMask
        tagReg := io.command.payload.tag
        featureOrdinal := 0
        featureAddress := 0
        authenticatedScatterReg := io.command.payload.authenticatedScatter
        jobProtocolFault := !commandLegal
        jobOverflow := False
        issueRmsForJob := commandLegal
        for (lane <- 0 until cfg.tokenLanes) {
          gateScaleReg(lane) := io.command.payload.gateScaleQ16(lane)
          upScaleReg(lane) := io.command.payload.upScaleQ16(lane)
        }
        when(commandLegal) {
          state := State.AcceptFeature
        } otherwise {
          globalProtocolFault := True
          state := State.Complete
        }
      }
    }

    is(State.AcceptFeature) {
      when(io.fusedInput.fire) {
        acceptedFeatureCounter := acceptedFeatureCounter + 1
        featureAddress := io.fusedInput.payload.featureIndex
        when(beatProtocolFault) {
          jobProtocolFault := True
          globalProtocolFault := True
        }
        for (lane <- 0 until cfg.tokenLanes) {
          rawGateMagnitude(lane) := rawMagnitude(
            io.fusedInput.payload.gateRaw(lane))
          rawUpMagnitude(lane) := rawMagnitude(
            io.fusedInput.payload.upRaw(lane))
          rawGatePositive(lane) :=
            io.fusedInput.payload.gateRaw(lane) > 0
          rawUpNegative(lane) := io.fusedInput.payload.upRaw(lane).msb
        }
        state := State.GateScaleMultiply
      }
    }

    is(State.GateScaleMultiply) {
      for (lane <- 0 until cfg.tokenLanes) {
        product0Reg(lane) := product0(lane)
        product1Reg(lane) := product1(lane)
      }
      state := State.GateScaleCombine
    }
    is(State.GateScaleCombine) {
      for (lane <- 0 until cfg.tokenLanes) {
        scaleProductReg(lane) := reconstructedScale(lane)
      }
      state := State.GateScaleRound
    }
    is(State.GateScaleRound) {
      for (lane <- 0 until cfg.tokenLanes) {
        gateMagnitude(lane) := Mux(scaledOverflow(lane),
          U((BigInt(1) << cfg.magnitudeWidth) - 1,
            cfg.magnitudeWidth bits),
          scaledRounded(lane).resize(cfg.magnitudeWidth))
        gateEnabled(lane) := rawGatePositive(lane) &&
          scaledRounded(lane) =/= 0 && tokenMaskReg(lane)
      }
      when((scaledOverflow & tokenMaskReg).orR) {
        jobOverflow := True
        globalOverflow := True
      }
      state := State.UpScaleMultiply
    }

    is(State.UpScaleMultiply) {
      for (lane <- 0 until cfg.tokenLanes) {
        product0Reg(lane) := product0(lane)
        product1Reg(lane) := product1(lane)
      }
      state := State.UpScaleCombine
    }
    is(State.UpScaleCombine) {
      for (lane <- 0 until cfg.tokenLanes) {
        scaleProductReg(lane) := reconstructedScale(lane)
      }
      state := State.UpScaleRound
    }
    is(State.UpScaleRound) {
      for (lane <- 0 until cfg.tokenLanes) {
        upMagnitude(lane) := Mux(scaledOverflow(lane),
          U((BigInt(1) << cfg.magnitudeWidth) - 1,
            cfg.magnitudeWidth bits),
          scaledRounded(lane).resize(cfg.magnitudeWidth))
        upNegative(lane) := rawUpNegative(lane)
      }
      when((scaledOverflow & tokenMaskReg).orR) {
        jobOverflow := True
        globalOverflow := True
      }
      state := State.SquareMultiply
    }

    is(State.SquareMultiply) {
      for (lane <- 0 until cfg.tokenLanes) {
        product0Reg(lane) := product0(lane)
        product1Reg(lane) := product1(lane)
      }
      state := State.SquareCombine
    }
    is(State.SquareCombine) {
      for (lane <- 0 until cfg.tokenLanes) {
        squareReg(lane) := reconstructedSquare(lane)
      }
      state := State.CubicLowMidMultiply
    }

    is(State.CubicLowMidMultiply) {
      for (lane <- 0 until cfg.tokenLanes) {
        product0Reg(lane) := product0(lane)
        product1Reg(lane) := product1(lane)
      }
      state := State.CubicHighMultiply
    }
    is(State.CubicHighMultiply) {
      for (lane <- 0 until cfg.tokenLanes) {
        cubicLowMidReg(lane) := reconstructedLowMid(lane)
        cubicHighReg(lane) := product0(lane)
      }
      state := State.CubicCombine
    }
    is(State.CubicCombine) {
      for (lane <- 0 until cfg.tokenLanes) {
        cubicProductReg(lane) := reconstructedCubic(lane)
      }
      state := State.CubicRound
    }
    is(State.CubicRound) {
      for (lane <- 0 until cfg.tokenLanes) {
        cubicRoundedReg(lane) := roundedCubic(lane)
      }
      state := State.Saturate
    }
    is(State.Saturate) {
      for (lane <- 0 until cfg.tokenLanes) {
        resultReg(lane) := finalValue(lane)
      }
      when(finalOverflow.orR) {
        jobOverflow := True
        globalOverflow := True
      }
      writePending := tokenMaskReg
      state := State.WriteWide
    }

    is(State.WriteWide) {
      writePending := pendingAfterHandshakes
      when(pendingAfterHandshakes === 0) {
        emittedFeatureCounter := emittedFeatureCounter + 1
        when(expectedLastFeature) {
          state := State.IssueRms
        } otherwise {
          featureOrdinal := featureOrdinal + 1
          state := State.AcceptFeature
        }
      }
    }

    is(State.IssueRms) {
      when(io.rmsCommand.fire) {
        state := State.Complete
      }
    }

    is(State.Complete) {
      when(io.done.fire) {
        completedFrameCounter := completedFrameCounter + 1
        state := State.Idle
      }
    }
  }
}

/** Bit-accurate Scala oracle for the production scaling/narrowing contract. */
object BitNetSharedSpuFfnGateMath {
  val NarrowMax: BigInt = (BigInt(1) << 26) - 1
  val NarrowMin: BigInt = -NarrowMax
  val I32Min: BigInt = -(BigInt(1) << 31)
  val I32Max: BigInt = (BigInt(1) << 31) - 1
  val ScaleMax: BigInt = NarrowMax

  case class Scaled(valueQ16: BigInt, overflow: Boolean)
  case class Result(valueQ16: Int, overflow: Boolean,
                    gateQ16: Int, upQ16: Int)

  private def clamp(value: BigInt, minimum: BigInt, maximum: BigInt):
      (BigInt, Boolean) = {
    if (value < minimum) (minimum, true)
    else if (value > maximum) (maximum, true)
    else (value, false)
  }

  /** Symmetric nearest rounding; a half LSB is rounded away from zero. */
  private def roundAwayShift(value: BigInt, shift: Int): BigInt = {
    val roundedMagnitude = (value.abs + (BigInt(1) << (shift - 1))) >> shift
    if (value < 0) -roundedMagnitude else roundedMagnitude
  }

  def scaleProjection(raw: Int, scaleQ16: BigInt): Scaled = {
    require(scaleQ16 >= 0 && scaleQ16 <= ScaleMax)
    val rounded = roundAwayShift(BigInt(raw) * scaleQ16, 16)
    val (clamped, overflow) = clamp(rounded, NarrowMin, NarrowMax)
    Scaled(clamped, overflow)
  }

  def apply(gateRaw: Int, gateScaleQ16: BigInt,
            upRaw: Int, upScaleQ16: BigInt): Result = {
    val gate = scaleProjection(gateRaw, gateScaleQ16)
    val up = scaleProjection(upRaw, upScaleQ16)
    if (gate.valueQ16 <= 0) {
      Result(0, gate.overflow || up.overflow,
        gate.valueQ16.toInt, up.valueQ16.toInt)
    } else {
      val cubic = gate.valueQ16 * gate.valueQ16 * up.valueQ16
      val rounded = roundAwayShift(cubic, 32)
      val (clamped, outputOverflow) = clamp(rounded, I32Min, I32Max)
      Result(clamped.toInt, gate.overflow || up.overflow || outputOverflow,
        gate.valueQ16.toInt, up.valueQ16.toInt)
    }
  }
}

/** Clearly named production generation entry; NativeRelu2Gate stays reference. */
object GenerateBitNetSharedSpuFfnGate extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-shared-spu-ffn-gate")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetSharedSpuFfnGate())
}
