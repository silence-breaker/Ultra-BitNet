package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** The five operations which share one physical 128-site DSP array. */
object BitNetUnifiedMatrixMode extends SpinalEnum(binarySequential) {
  val TernaryProjection, QkDotI8, SvContextScaleQ15, SvAxpyQ15,
      LmHeadDotI8 = newElement()
}

/**
  * Operand interpretation is locked together with the mode for the complete
  * job.  TileAlpha and FinalReciprocal intentionally use the same arithmetic;
  * the distinct encodings make an accidental attention scheduling mix-up
  * observable at the hardware boundary.
  */
object BitNetUnifiedOperandFormat extends SpinalEnum(binarySequential) {
  val Map0Packed, I8Pair, Q15TileAlpha, Q15Probability,
      Q15FinalReciprocal = newElement()
}

case class BitNetUnifiedMatrixConfig(
    chainLength: Int = 8,
    numChains: Int = 16,
    outputLanes: Int = 2,
    dotSlots: Int = 4,
    contextSlots: Int = 4,
    contextBanks: Int = 5,
    weightBanks: Int = 2,
    activationWidth: Int = 8,
    q15Width: Int = 16,
    contextWidth: Int = 27,
    laneSeparation: Int = 18,
    dspAWidth: Int = 27,
    dspBWidth: Int = 18,
    dspPWidth: Int = 48,
    accumulatorWidth: Int = 32,
    epochWidth: Int = 12,
    tokenIdWidth: Int = 16,
    outputIndexWidth: Int = 18,
    phaseCountWidth: Int = 16,
    featureIndexWidth: Int = 7,
    resultFifoDepth: Int = 64,
    sitePipelineStages: Int = 3
) {
  require(chainLength == 8)
  require(numChains == 8 || numChains == 16)
  require(outputLanes == 2)
  require(dotSlots >= 4 && isPow2(dotSlots))
  require(contextSlots >= 4 && isPow2(contextSlots))
  require(contextBanks >= 5)
  require(weightBanks == 2)
  require(activationWidth == 8)
  require(q15Width == 16)
  require(contextWidth >= 24 && contextWidth <= dspAWidth)
  require(laneSeparation >= 17 && laneSeparation <= 24)
  require(dspAWidth == 27 && dspBWidth == 18 && dspPWidth == 48)
  require(sitePipelineStages == 3)
  require(accumulatorWidth >= 24 && accumulatorWidth <= dspPWidth)
  require(resultFifoDepth >= 32 && isPow2(resultFifoDepth))

  val siteCount: Int = chainLength * numChains
  val packedWeightWidth: Int = siteCount * outputLanes * 2
  val weightBankWidth: Int = log2Up(weightBanks)
  val dotSlotWidth: Int = log2Up(dotSlots)
  val contextSlotWidth: Int = log2Up(contextSlots)
  val contextBankWidth: Int = log2Up(contextBanks)
  val reductionLevels: Int = log2Up(numChains)
  val chainPipelineLatency: Int = chainLength * sitePipelineStages
  val unpackPipelineStages: Int = 1
  val dotPipelineLatency: Int =
    chainPipelineLatency + reductionLevels + unpackPipelineStages
  val svPipelineLatency: Int = sitePipelineStages
  val physicalMacDsps: Int = siteCount
  val physicalReductionDsps: Int = numChains - 1
  val physicalDsps: Int = physicalMacDsps + physicalReductionDsps

  require(siteCount == 64 || siteCount == 128)
  require(physicalDsps == siteCount + numChains - 1)
  require(siteCount * (1 << (activationWidth - 1)) <
    (1 << (laneSeparation - 1)))
}

case class BitNetUnifiedWeightLoad(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  val bank = UInt(cfg.weightBankWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  // Each site is {map0 output lane 1, map0 output lane 0}.
  val packedMap0 = Bits(cfg.packedWeightWidth bits)
}

case class BitNetUnifiedJob(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  val mode = BitNetUnifiedMatrixMode()
  val format = BitNetUnifiedOperandFormat()
  val weightBank = UInt(cfg.weightBankWidth bits)
  val weightEpoch = UInt(cfg.epochWidth bits)
  // A job consumes exactly this many accepted phases.  Mode/format/bank/epoch
  // cannot change until every corresponding result has been consumed.
  val phaseCount = UInt(cfg.phaseCountWidth bits)
}

case class BitNetUnifiedPhase(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  val lhs = Vec(SInt(cfg.activationWidth bits), cfg.siteCount)
  val rhs = Vec(SInt(cfg.activationWidth bits), cfg.siteCount)
  val dotSlot = UInt(cfg.dotSlotWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val outputBase = UInt(cfg.outputIndexWidth bits)

  // A tiled-attention SPU presents all four GQA probabilities/alphas with the
  // V vector.  One physical pass selects headLane; four rotating context slots
  // therefore update the four query heads without a second multiplier array.
  val q15Factors = Vec(SInt(cfg.q15Width bits), 4)
  val headLane = UInt(2 bits)
  val contextSlot = UInt(cfg.contextSlotWidth bits)
}

case class BitNetUnifiedResult(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  val mode = BitNetUnifiedMatrixMode()
  val format = BitNetUnifiedOperandFormat()
  val weightBank = UInt(cfg.weightBankWidth bits)
  val weightEpoch = UInt(cfg.epochWidth bits)
  val values = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  val tileValues = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  val dotSlot = UInt(cfg.dotSlotWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val outputBase = UInt(cfg.outputIndexWidth bits)
  val contextCommit = Bool()
  val contextSlot = UInt(cfg.contextSlotWidth bits)
  val headLane = UInt(2 bits)
}

case class BitNetUnifiedContextReadCommand(
    cfg: BitNetUnifiedMatrixConfig
) extends Bundle {
  val bank = UInt(cfg.weightBankWidth bits)
  val slot = UInt(cfg.contextSlotWidth bits)
  val feature = UInt(cfg.featureIndexWidth bits)
}

case class BitNetUnifiedContextReadResponse(
    cfg: BitNetUnifiedMatrixConfig
) extends Bundle {
  val bank = UInt(cfg.weightBankWidth bits)
  val slot = UInt(cfg.contextSlotWidth bits)
  val feature = UInt(cfg.featureIndexWidth bits)
  // Context is a signed Q*.15 numerator.  Keeping 27 bits fits DSP48E2's A
  // input and exactly covers the bounded int8 weighted-attention numerator.
  val valueQ15 = SInt(cfg.contextWidth bits)
}

/** Metadata shifted beside, never through, the arithmetic array. */
case class BitNetUnifiedPhaseMeta(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  val dotSlot = UInt(cfg.dotSlotWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val outputBase = UInt(cfg.outputIndexWidth bits)
  val contextSlot = UInt(cfg.contextSlotWidth bits)
  val headLane = UInt(2 bits)
}

/**
  * Bit-accurate architectural reference model for the unified matrix engine.
  *
  * DO NOT instantiate this class in the production top.  Its intentionally
  * direct 8-way context-register mux and parallel round/saturate network cost
  * 64,944 LUTs in xczu3eg OOC synthesis.  Production uses
  * [[BitNetUnifiedMatrixArrayCore]] plus the external bank-local context
  * coordinator; that core is 7,568 LUTs / 143 DSPs and closes 532 MHz.
  *
  * There is exactly one 128-site 27x18 multiplier array.  Projection packs two
  * ternary output rows into each site; QK and LM use the same sites as int8
  * products and the same 8-site chains/16-to-1 reduction tree.  SV selects a
  * three-cycle non-reduction path through those sites, either context*alpha or
  * V*probability+context.  It does not instantiate an attention GEMV.
  *
  * Job locking is intentionally strict.  A new mode cannot be accepted until
  * all input phases have traversed the array and all result acknowledgements,
  * including ones held by downstream backpressure, have drained.
  */
class BitNetUnifiedMatrixEngine(
    cfg: BitNetUnifiedMatrixConfig = BitNetUnifiedMatrixConfig(),
    useVendorPrimitive: Boolean = true
) extends Component {
  val io = new Bundle {
    val weightLoad = slave(Stream(BitNetUnifiedWeightLoad(cfg)))
    val job = slave(Stream(BitNetUnifiedJob(cfg)))
    val phase = slave(Stream(BitNetUnifiedPhase(cfg)))
    val result = master(Stream(BitNetUnifiedResult(cfg)))
    val contextRead = slave(Stream(BitNetUnifiedContextReadCommand(cfg)))
    val contextReadResult = master(
      Stream(BitNetUnifiedContextReadResponse(cfg))
    )

    val weightBankValid = out Bits(cfg.weightBanks bits)
    val bankBusy = out Bits(cfg.weightBanks bits)
    val contextBusy = out Bits((cfg.weightBanks * cfg.contextSlots) bits)
    val jobActive = out Bool()
    val draining = out Bool()
    val protocolFault = out Bool()
    val acceptedJobs = out UInt(32 bits)
    val acceptedPhases = out UInt(32 bits)
    val emittedResults = out UInt(32 bits)
  }

  private def packWeightPair(lowCode: Bits, highCode: Bits): SInt = {
    val selector = highCode ## lowCode
    val packed = SInt(cfg.dspAWidth bits)
    switch(selector) {
      for (highCodeValue <- 0 until 4; lowCodeValue <- 0 until 4) {
        val low = lowCodeValue match {
          case 1 => -1
          case 3 => 1
          case _ => 0
        }
        val high = highCodeValue match {
          case 1 => -1
          case 3 => 1
          case _ => 0
        }
        val value = BigInt(low) + (BigInt(high) << cfg.laneSeparation)
        is(B((highCodeValue << 2) | lowCodeValue, 4 bits)) {
          packed := S(value, cfg.dspAWidth bits)
        }
      }
    }
    packed
  }

  private def isDot(mode: BitNetUnifiedMatrixMode.C): Bool =
    mode === BitNetUnifiedMatrixMode.TernaryProjection ||
      mode === BitNetUnifiedMatrixMode.QkDotI8 ||
      mode === BitNetUnifiedMatrixMode.LmHeadDotI8

  private def isSv(mode: BitNetUnifiedMatrixMode.C): Bool =
    mode === BitNetUnifiedMatrixMode.SvContextScaleQ15 ||
      mode === BitNetUnifiedMatrixMode.SvAxpyQ15

  /** Symmetric round-to-nearest, ties away from zero, then signed saturate. */
  private def roundQ15Saturate(value: SInt): SInt = {
    val wideWidth = value.getWidth + 1
    val wide = value.resize(wideWidth)
    val magnitude = UInt(wideWidth bits)
    when(wide.msb) {
      magnitude := (-wide).asUInt
    } otherwise {
      magnitude := wide.asUInt
    }
    val roundedMagnitude =
      ((magnitude + U(1 << 14, wideWidth bits)) >> 15).resize(wideWidth)
    val rounded = SInt(wideWidth bits)
    when(wide.msb) {
      rounded := -roundedMagnitude.asSInt
    } otherwise {
      rounded := roundedMagnitude.asSInt
    }

    val result = SInt(cfg.contextWidth bits)
    val maximum = (BigInt(1) << (cfg.contextWidth - 1)) - 1
    val minimum = -(BigInt(1) << (cfg.contextWidth - 1))
    when(rounded > S(maximum, wideWidth bits)) {
      result := S(maximum, cfg.contextWidth bits)
    } elsewhen (rounded < S(minimum, wideWidth bits)) {
      result := S(minimum, cfg.contextWidth bits)
    } otherwise {
      result := rounded.resize(cfg.contextWidth)
    }
    result
  }

  private def saturateContext(value: SInt): SInt = {
    val width = value.getWidth
    val result = SInt(cfg.contextWidth bits)
    val maximum = (BigInt(1) << (cfg.contextWidth - 1)) - 1
    val minimum = -(BigInt(1) << (cfg.contextWidth - 1))
    when(value > S(maximum, width bits)) {
      result := S(maximum, cfg.contextWidth bits)
    } elsewhen (value < S(minimum, width bits)) {
      result := S(minimum, cfg.contextWidth bits)
    } otherwise {
      result := value.resize(cfg.contextWidth)
    }
    result
  }

  // Projection weights are expanded only at the load boundary.  No ternary
  // decoder remains on the 532 MHz token path.
  val expandedWeights = Vec(
    Vec(Reg(SInt(cfg.dspAWidth bits)) init (0), cfg.siteCount),
    cfg.weightBanks
  )
  val weightEpoch = Vec(Reg(UInt(cfg.epochWidth bits)) init (0), cfg.weightBanks)
  val weightValid = Vec(RegInit(False), cfg.weightBanks)

  val active = RegInit(False)
  val closing = RegInit(False)
  val jobMode = Reg(BitNetUnifiedMatrixMode()) init (
    BitNetUnifiedMatrixMode.TernaryProjection
  )
  val jobFormat = Reg(BitNetUnifiedOperandFormat()) init (
    BitNetUnifiedOperandFormat.Map0Packed
  )
  val jobBank = Reg(UInt(cfg.weightBankWidth bits)) init (0)
  val jobEpoch = Reg(UInt(cfg.epochWidth bits)) init (0)
  val phasesRemaining = Reg(UInt(cfg.phaseCountWidth bits)) init (0)
  val outstanding = Reg(UInt(log2Up(cfg.resultFifoDepth + 1) bits)) init (0)
  val stickyFault = RegInit(False)

  def legalModeFormat(
      mode: BitNetUnifiedMatrixMode.C,
      format: BitNetUnifiedOperandFormat.C
  ): Bool = {
    val legal = Bool()
    switch(mode) {
      is(BitNetUnifiedMatrixMode.TernaryProjection) {
        legal := format === BitNetUnifiedOperandFormat.Map0Packed
      }
      is(BitNetUnifiedMatrixMode.QkDotI8) {
        legal := format === BitNetUnifiedOperandFormat.I8Pair
      }
      is(BitNetUnifiedMatrixMode.LmHeadDotI8) {
        legal := format === BitNetUnifiedOperandFormat.I8Pair
      }
      is(BitNetUnifiedMatrixMode.SvContextScaleQ15) {
        legal := format === BitNetUnifiedOperandFormat.Q15TileAlpha ||
          format === BitNetUnifiedOperandFormat.Q15FinalReciprocal
      }
      is(BitNetUnifiedMatrixMode.SvAxpyQ15) {
        legal := format === BitNetUnifiedOperandFormat.Q15Probability
      }
    }
    legal
  }

  val jobHasProjectionWeights =
    weightValid(io.job.weightBank) &&
      weightEpoch(io.job.weightBank) === io.job.weightEpoch
  val jobBankBeingLoaded = io.weightLoad.valid &&
    io.weightLoad.bank === io.job.weightBank
  val jobPayloadLegal = legalModeFormat(io.job.mode, io.job.format) &&
    io.job.phaseCount =/= 0 &&
    (io.job.mode =/= BitNetUnifiedMatrixMode.TernaryProjection ||
      (jobHasProjectionWeights && !jobBankBeingLoaded))

  io.job.ready := !active && jobPayloadLegal
  when(io.job.valid && !active && !jobPayloadLegal) {
    stickyFault := True
  }
  when(io.job.fire) {
    active := True
    closing := False
    jobMode := io.job.mode
    jobFormat := io.job.format
    jobBank := io.job.weightBank
    jobEpoch := io.job.weightEpoch
    phasesRemaining := io.job.phaseCount
    outstanding := 0
  }

  io.weightLoad.ready := !(active && io.weightLoad.bank === jobBank)
  when(io.weightLoad.fire) {
    weightEpoch(io.weightLoad.bank) := io.weightLoad.epoch
    weightValid(io.weightLoad.bank) := True
    for (site <- 0 until cfg.siteCount) {
      val base = site * cfg.outputLanes * 2
      expandedWeights(io.weightLoad.bank)(site) := packWeightPair(
        io.weightLoad.packedMap0(base + 1 downto base),
        io.weightLoad.packedMap0(base + 3 downto base + 2)
      )
    }
  }

  // Four slots per physical bank make a one-phase-per-cycle GQA schedule free
  // of the three-cycle SV read-after-write hazard.  The scoreboard also makes
  // arbitrary/non-rotating software or controller schedules safe.
  val context = Vec(
    Vec(
      Vec(Reg(SInt(cfg.contextWidth bits)) init (0), cfg.siteCount),
      cfg.contextSlots
    ),
    cfg.weightBanks
  )
  val contextSlotBusy = Vec(
    Vec(RegInit(False), cfg.contextSlots),
    cfg.weightBanks
  )
  val selectedContextBusy = contextSlotBusy(jobBank)(io.phase.contextSlot)

  val hasCredit = outstanding =/= cfg.resultFifoDepth
  val svHazardFree = !isSv(jobMode) || !selectedContextBusy
  io.phase.ready := active && !closing && phasesRemaining =/= 0 &&
    hasCredit && svHazardFree
  val phaseFire = io.phase.fire
  val dotFire = phaseFire && isDot(jobMode)
  val svFire = phaseFire && isSv(jobMode)

  when(phaseFire) {
    phasesRemaining := phasesRemaining - 1
    when(phasesRemaining === 1) { closing := True }
  }
  when(svFire) {
    contextSlotBusy(jobBank)(io.phase.contextSlot) := True
  }

  val metaPipe = Vec(
    Reg(BitNetUnifiedPhaseMeta(cfg)) init (BitNetUnifiedPhaseMeta(cfg).getZero),
    cfg.dotPipelineLatency
  )
  val acceptedMeta = BitNetUnifiedPhaseMeta(cfg)
  acceptedMeta.dotSlot := io.phase.dotSlot
  acceptedMeta.firstK := io.phase.firstK
  acceptedMeta.lastK := io.phase.lastK
  acceptedMeta.tokenId := io.phase.tokenId
  acceptedMeta.outputBase := io.phase.outputBase
  acceptedMeta.contextSlot := io.phase.contextSlot
  acceptedMeta.headLane := io.phase.headLane
  when(phaseFire) { metaPipe(0) := acceptedMeta }
  for (stage <- 1 until cfg.dotPipelineLatency) {
    metaPipe(stage) := metaPipe(stage - 1)
  }

  val dotValidPipe = Vec(RegInit(False), cfg.dotPipelineLatency)
  val svValidPipe = Vec(RegInit(False), cfg.svPipelineLatency)
  dotValidPipe(0) := dotFire
  svValidPipe(0) := svFire
  for (stage <- 1 until cfg.dotPipelineLatency) {
    dotValidPipe(stage) := dotValidPipe(stage - 1)
  }
  for (stage <- 1 until cfg.svPipelineLatency) {
    svValidPipe(stage) := svValidPipe(stage - 1)
  }

  val chainOutputs = Vec(SInt(cfg.dspPWidth bits), cfg.numChains)
  val svSiteOutputs = Vec(SInt(cfg.dspPWidth bits), cfg.siteCount)
  for (chain <- 0 until cfg.numChains) {
    val siteOutputs = new Array[SInt](cfg.chainLength)
    for (siteInChain <- 0 until cfg.chainLength) {
      val site = chain * cfg.chainLength + siteInChain
      val siteOffset = siteInChain * cfg.sitePipelineStages

      // Dot operands move systolically through an 8-site chain.  SV presents
      // all 128 independent features together and exits after the site's own
      // A/B, M and P registers.
      val delayedLhs = if (siteOffset == 0) {
        io.phase.lhs(site)
      } else {
        Delay(io.phase.lhs(site), siteOffset, init = S(0, cfg.activationWidth bits))
      }
      val delayedRhs = if (siteOffset == 0) {
        io.phase.rhs(site)
      } else {
        Delay(io.phase.rhs(site), siteOffset, init = S(0, cfg.activationWidth bits))
      }

      val selectedPackedWeight = SInt(cfg.dspAWidth bits)
      selectedPackedWeight := expandedWeights(0)(site)
      when(jobBank === 1) {
        selectedPackedWeight := expandedWeights(1)(site)
      }

      val selectedFactor = io.phase.q15Factors(io.phase.headLane)
      val selectedContext = context(jobBank)(io.phase.contextSlot)(site)
      val aOperand = SInt(cfg.dspAWidth bits)
      val bOperand = SInt(cfg.dspBWidth bits)
      val cOperand = SInt(cfg.dspPWidth bits)

      switch(jobMode) {
        is(BitNetUnifiedMatrixMode.TernaryProjection) {
          aOperand := selectedPackedWeight
          bOperand := delayedLhs.resize(cfg.dspBWidth)
          if (siteInChain > 0) {
            cOperand := siteOutputs(siteInChain - 1)
          } else {
            cOperand := 0
          }
        }
        is(BitNetUnifiedMatrixMode.QkDotI8) {
          aOperand := delayedLhs.resize(cfg.dspAWidth)
          bOperand := delayedRhs.resize(cfg.dspBWidth)
          if (siteInChain > 0) {
            cOperand := siteOutputs(siteInChain - 1)
          } else {
            cOperand := 0
          }
        }
        is(BitNetUnifiedMatrixMode.LmHeadDotI8) {
          aOperand := delayedLhs.resize(cfg.dspAWidth)
          bOperand := delayedRhs.resize(cfg.dspBWidth)
          if (siteInChain > 0) {
            cOperand := siteOutputs(siteInChain - 1)
          } else {
            cOperand := 0
          }
        }
        is(BitNetUnifiedMatrixMode.SvContextScaleQ15) {
          aOperand := selectedContext.resize(cfg.dspAWidth)
          bOperand := selectedFactor.resize(cfg.dspBWidth)
          cOperand := 0
        }
        is(BitNetUnifiedMatrixMode.SvAxpyQ15) {
          aOperand := io.phase.rhs(site).resize(cfg.dspAWidth)
          bOperand := selectedFactor.resize(cfg.dspBWidth)
          cOperand := selectedContext.resize(cfg.dspPWidth)
        }
      }

      if (useVendorPrimitive) {
        val dsp = new BitNetPackedMacDsp48E2
        dsp.A := aOperand.resize(30).asBits
        dsp.B := bOperand.asBits
        dsp.C := cOperand.asBits
        dsp.D := 0
        dsp.CARRYIN := False
        dsp.ACIN := 0
        dsp.BCIN := 0
        dsp.PCIN := 0
        dsp.CARRYCASCIN := False
        dsp.MULTSIGNIN := False
        dsp.ALUMODE := B"4'b0000"
        dsp.INMODE := B"5'b10101"
        dsp.OPMODE := B"9'b000110101" // P=M+C
        dsp.CARRYINSEL := B"3'b000"
        dsp.CEA1 := True
        dsp.CEA2 := False
        dsp.CEB1 := True
        dsp.CEB2 := False
        dsp.CEC := True
        dsp.CED := False
        dsp.CEAD := False
        dsp.CEM := True
        dsp.CEP := True
        dsp.CECARRYIN := False
        dsp.CECTRL := False
        dsp.CEINMODE := False
        dsp.CEALUMODE := False
        dsp.RSTA := False
        dsp.RSTB := False
        dsp.RSTC := False
        dsp.RSTD := False
        dsp.RSTM := False
        dsp.RSTP := False
        dsp.RSTALLCARRYIN := False
        dsp.RSTCTRL := False
        dsp.RSTINMODE := False
        dsp.RSTALUMODE := False
        siteOutputs(siteInChain) = dsp.P.asSInt
      } else {
        // One portable multiply per physical site.  This is the bit-accurate
        // Verilator model of the same A/B, M, P DSP pipeline.
        val aReg = Reg(SInt(cfg.dspAWidth bits)) init (0)
        val bReg = Reg(SInt(cfg.dspBWidth bits)) init (0)
        val cReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        aReg := aOperand
        bReg := bOperand
        cReg := cOperand
        val mReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        val cDelay = Reg(SInt(cfg.dspPWidth bits)) init (0)
        mReg := (aReg * bReg).resize(cfg.dspPWidth)
        cDelay := cReg
        val pReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        pReg := (mReg + cDelay).resize(cfg.dspPWidth)
        siteOutputs(siteInChain) = pReg
      }
      svSiteOutputs(site) := siteOutputs(siteInChain)
    }
    chainOutputs(chain) := siteOutputs(cfg.chainLength - 1)
  }

  // The same chain outputs feed one registered 16-to-1 tree.  Vivado maps its
  // 15 additions to DSP48E2 ALUs, giving 128 + 15 = 143 DSPs per engine.
  var reduceLevel: Seq[SInt] = chainOutputs.toSeq
  for (level <- 0 until cfg.reductionLevels) {
    val levelValid = dotValidPipe(cfg.chainPipelineLatency + level - 1)
    reduceLevel = reduceLevel.grouped(2).map {
      case Seq(left, right) =>
        val sum = (left + right).resize(cfg.dspPWidth)
        val sumReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        sum.addAttribute("use_dsp", "yes")
        sumReg.addAttribute("use_dsp", "yes")
        when(levelValid) { sumReg := sum }
        sumReg
      case Seq(single) => single
    }.toSeq
  }
  val reducedDot = reduceLevel.head

  val lane0Field = reducedDot(cfg.laneSeparation - 1 downto 0)
  val unpackedLane0 = lane0Field.resize(cfg.accumulatorWidth)
  val highRaw = (reducedDot >> cfg.laneSeparation).resize(cfg.accumulatorWidth)
  val borrowCorrection = SInt(cfg.accumulatorWidth bits)
  borrowCorrection := 0
  when(lane0Field.msb) { borrowCorrection := 1 }
  val unpackedLane1 = (highRaw + borrowCorrection).resize(cfg.accumulatorWidth)

  val dotTile = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  dotTile(0) := reducedDot.resize(cfg.accumulatorWidth)
  dotTile(1) := 0
  when(jobMode === BitNetUnifiedMatrixMode.TernaryProjection) {
    dotTile(0) := unpackedLane0
    dotTile(1) := unpackedLane1
  }
  val registeredDotTile = Vec(
    Reg(SInt(cfg.accumulatorWidth bits)) init (0),
    cfg.outputLanes
  )
  val unpackIndex = cfg.dotPipelineLatency - 2
  when(dotValidPipe(unpackIndex)) { registeredDotTile := dotTile }

  val dotAccumulators = Vec(
    Vec(Reg(SInt(cfg.accumulatorWidth bits)) init (0), cfg.outputLanes),
    cfg.dotSlots
  )
  val dotResultIndex = cfg.dotPipelineLatency - 1
  val dotResultValid = dotValidPipe(dotResultIndex)
  val dotResultMeta = metaPipe(dotResultIndex)
  val accumulatedDot = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  for (lane <- 0 until cfg.outputLanes) {
    val previous = SInt(cfg.accumulatorWidth bits)
    previous := dotAccumulators(dotResultMeta.dotSlot)(lane)
    when(dotResultMeta.firstK) { previous := 0 }
    accumulatedDot(lane) :=
      (previous + registeredDotTile(lane)).resize(cfg.accumulatorWidth)
    when(dotResultValid) {
      dotAccumulators(dotResultMeta.dotSlot)(lane) := accumulatedDot(lane)
    }
  }

  val svResultIndex = cfg.svPipelineLatency - 1
  val svResultValid = svValidPipe(svResultIndex)
  val svResultMeta = metaPipe(svResultIndex)
  when(svResultValid) {
    for (site <- 0 until cfg.siteCount) {
      val scaled = roundQ15Saturate(svSiteOutputs(site))
      val axpy = saturateContext(svSiteOutputs(site))
      when(jobMode === BitNetUnifiedMatrixMode.SvContextScaleQ15) {
        context(jobBank)(svResultMeta.contextSlot)(site) := scaled
      } otherwise {
        context(jobBank)(svResultMeta.contextSlot)(site) := axpy
      }
    }
    contextSlotBusy(jobBank)(svResultMeta.contextSlot) := False
  }

  val generated = Stream(BitNetUnifiedResult(cfg))
  generated.valid := dotResultValid || svResultValid
  generated.mode := jobMode
  generated.format := jobFormat
  generated.weightBank := jobBank
  generated.weightEpoch := jobEpoch
  generated.values := accumulatedDot
  generated.tileValues := registeredDotTile
  generated.dotSlot := dotResultMeta.dotSlot
  generated.firstK := dotResultMeta.firstK
  generated.lastK := dotResultMeta.lastK
  generated.tokenId := dotResultMeta.tokenId
  generated.outputBase := dotResultMeta.outputBase
  generated.contextCommit := False
  generated.contextSlot := dotResultMeta.contextSlot
  generated.headLane := dotResultMeta.headLane
  when(svResultValid) {
    generated.values.foreach(_ := 0)
    generated.tileValues.foreach(_ := 0)
    generated.dotSlot := svResultMeta.dotSlot
    generated.firstK := svResultMeta.firstK
    generated.lastK := svResultMeta.lastK
    generated.tokenId := svResultMeta.tokenId
    generated.outputBase := svResultMeta.outputBase
    generated.contextCommit := True
    generated.contextSlot := svResultMeta.contextSlot
    generated.headLane := svResultMeta.headLane
  }

  val resultFifo = StreamFifo(BitNetUnifiedResult(cfg), cfg.resultFifoDepth)
  resultFifo.io.push << generated
  io.result << resultFifo.io.pop

  // Credit covers both pipeline-resident and FIFO-resident phases.  Therefore
  // generated.ready is an invariant even under unbounded result backpressure.
  when(generated.valid && !generated.ready) { stickyFault := True }
  switch((phaseFire ## io.result.fire).asBits) {
    is(B"2'b10") { outstanding := outstanding + 1 }
    is(B"2'b01") { outstanding := outstanding - 1 }
  }
  when(closing && io.result.fire && outstanding === 1) {
    active := False
    closing := False
  }

  // One-entry elastic context readback.  Reads are rejected while that slot is
  // being modified; no ambiguous read-during-write RAM behaviour leaks out.
  val readValid = RegInit(False)
  val readPayload = Reg(BitNetUnifiedContextReadResponse(cfg)) init (
    BitNetUnifiedContextReadResponse(cfg).getZero
  )
  val readSlotBusy = contextSlotBusy(io.contextRead.bank)(io.contextRead.slot)
  io.contextRead.ready := (!readValid || io.contextReadResult.ready) &&
    !readSlotBusy
  when(io.contextRead.ready) {
    readValid := io.contextRead.valid
    when(io.contextRead.valid) {
      readPayload.bank := io.contextRead.bank
      readPayload.slot := io.contextRead.slot
      readPayload.feature := io.contextRead.feature
      readPayload.valueQ15 := context(io.contextRead.bank)(io.contextRead.slot)(
        io.contextRead.feature
      )
    }
  }
  io.contextReadResult.valid := readValid
  io.contextReadResult.payload := readPayload

  val acceptedJobCount = Reg(UInt(32 bits)) init (0)
  val acceptedPhaseCount = Reg(UInt(32 bits)) init (0)
  val emittedResultCount = Reg(UInt(32 bits)) init (0)
  when(io.job.fire) { acceptedJobCount := acceptedJobCount + 1 }
  when(phaseFire) { acceptedPhaseCount := acceptedPhaseCount + 1 }
  when(io.result.fire) { emittedResultCount := emittedResultCount + 1 }

  io.weightBankValid := weightValid.asBits
  for (bank <- 0 until cfg.weightBanks) {
    io.bankBusy(bank) := active && jobBank === bank
  }
  for (bank <- 0 until cfg.weightBanks; slot <- 0 until cfg.contextSlots) {
    io.contextBusy(bank * cfg.contextSlots + slot) := contextSlotBusy(bank)(slot)
  }
  io.jobActive := active
  io.draining := active && closing
  io.protocolFault := stickyFault
  io.acceptedJobs := acceptedJobCount
  io.acceptedPhases := acceptedPhaseCount
  io.emittedResults := emittedResultCount
}

object BitNetUnifiedMatrixOracle {
  val ContextWidth = 48
  val ContextLimbBits = 24

  def decodeMap0(code: Int): Int = code & 3 match {
    case 1 => -1
    case 3 => 1
    case _ => 0
  }

  def ternaryDot2(
      lhs: Seq[Int],
      lane0: Seq[Int],
      lane1: Seq[Int]
  ): (Int, Int) = (
    lhs.indices.map(i => lhs(i) * decodeMap0(lane0(i))).sum,
    lhs.indices.map(i => lhs(i) * decodeMap0(lane1(i))).sum
  )

  def i8Dot(lhs: Seq[Int], rhs: Seq[Int]): Int = {
    require(lhs.length == rhs.length)
    lhs.indices.map(i => lhs(i) * rhs(i)).sum
  }

  def roundQ15(value: Long): Long = {
    val magnitude = if (value < 0) -value else value
    val rounded = (magnitude + (1L << 14)) >> 15
    if (value < 0) -rounded else rounded
  }

  def saturateContext(value: Long, width: Int = 27): Int = {
    val maximum = (1L << (width - 1)) - 1
    val minimum = -(1L << (width - 1))
    scala.math.max(minimum, scala.math.min(maximum, value)).toInt
  }

  def scaleContext(valueQ15: Int, alphaQ15: Int, width: Int = 27): Int =
    saturateContext(roundQ15(valueQ15.toLong * alphaQ15.toLong), width)

  def axpyContext(
      contextQ15: Int,
      valueI8: Int,
      probabilityQ15: Int,
      width: Int = 27
  ): Int = saturateContext(
    contextQ15.toLong + valueI8.toLong * probabilityQ15.toLong,
    width
  )

  /** Exact two-pass representation used by the production 27x18 DSP array. */
  def splitContextS48(value: Long): (Int, Int) = {
    val minimum = -(1L << (ContextWidth - 1))
    val maximum = (1L << (ContextWidth - 1)) - 1
    require(value >= minimum && value <= maximum)
    val lowUnsigned = (value & ((1L << ContextLimbBits) - 1)).toInt
    val highSigned = (value >> ContextLimbBits).toInt
    (lowUnsigned, highSigned)
  }

  /** Reconstruct the unrounded S48-context times UInt Q1.15 product. */
  def reconstructContextScaleProduct(
      lowProduct: Long,
      highProduct: Long
  ): Long = lowProduct + (highProduct << ContextLimbBits)

  def packCodes(lane0: Seq[Int], lane1: Seq[Int]): BigInt = {
    require(lane0.length == lane1.length)
    lane0.indices.foldLeft(BigInt(0)) { case (packed, index) =>
      packed |
        (BigInt(lane0(index) & 3) << (index * 4)) |
        (BigInt(lane1(index) & 3) << (index * 4 + 2))
    }
  }
}

object GenerateBitNetUnifiedMatrixEngine extends App {
  val positional = BitNetResearchGenerationPolicy
    .requireOptIn("GenerateBitNetUnifiedMatrixEngine", args)
  require(positional.length <= 1,
    "usage: GenerateBitNetUnifiedMatrixEngine " +
      "--allow-legacy-research [target-directory]")
  SpinalConfig(
    targetDirectory = positional.headOption.getOrElse(
      "target/generated-bitnet-unified-matrix"),
    oneFilePerComponent = false
  ).generateVerilog(new BitNetUnifiedMatrixEngine())
}
