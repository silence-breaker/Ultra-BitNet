package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * A phase after the 266 MHz mode-specific mapper has selected operands.
  *
  * This is the deliberate 266->532 MHz boundary.  The fast array sees only a
  * DSP-native stationary bank plus activation/SV-limb values; it contains no
 * 8-way context mux, weight decoder,
  * reciprocal unit, command CAM or parallel saturation network.
  */
case class BitNetUnifiedMappedPhase(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  // Dot modes multiply this int8 vector by the selected stationary operand
  // bank.  The mapper loads packed ternary pairs or sign-extended int8 K/LM
  // weights into that bank before starting the job.
  val activation = Vec(SInt(cfg.activationWidth bits), cfg.siteCount)
  // SV modes use one independent A value per feature and a shared Q1.15 B.
  // SvContextScale supplies contextQ15; SvAxpy supplies V and returns a raw
  // V*probability delta.  Context addition/saturation belongs to the external
  // bank-local BRAM coordinator.
  val svValue = Vec(SInt(cfg.dspAWidth bits), cfg.siteCount)
  // Tiled-attention Q1.15 is unsigned: 0x8000 is exactly 1.0.  It is zero
  // extended to signed 18-bit DSP B, never reinterpreted as SInt16.
  val svFactorQ15 = UInt(cfg.q15Width bits)
  // S48 context is never narrowed.  The coordinator presents low24 unsigned
  // (zero-extended into svValue) and high24 signed in two scale phases, then
  // reconstructs rawLow + (rawHigh << 24) before Q15 rounding.
  val svLimbShift = UInt(5 bits)
  val dotSlot = UInt(cfg.dotSlotWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val outputBase = UInt(cfg.outputIndexWidth bits)
  // The bank/slot are returned with raw SV results so a bank-local coordinator
  // can perform BRAM writeback without reconstructing ordering state.
  val contextBank = UInt(cfg.contextBankWidth bits)
  val contextSlot = UInt(cfg.contextSlotWidth bits)
  val headLane = UInt(2 bits)
}

case class BitNetUnifiedMappedWeightLoad(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  val bank = UInt(cfg.weightBankWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  // Already decoded/formatted for DSP A: packed ternary pair or sign-extended
  // int8 K/LM value.  This makes the 532 MHz core independent of map encoding.
  val values = Vec(SInt(cfg.dspAWidth bits), cfg.siteCount)
}

case class BitNetUnifiedDotResult(cfg: BitNetUnifiedMatrixConfig)
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
}

case class BitNetUnifiedSvRawResult(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  val mode = BitNetUnifiedMatrixMode()
  val format = BitNetUnifiedOperandFormat()
  val contextBank = UInt(cfg.contextBankWidth bits)
  val contextSlot = UInt(cfg.contextSlotWidth bits)
  val headLane = UInt(2 bits)
  val svLimbShift = UInt(5 bits)
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val rawP = Vec(SInt(cfg.dspPWidth bits), cfg.siteCount)
}

case class BitNetUnifiedArrayMeta(cfg: BitNetUnifiedMatrixConfig)
    extends Bundle {
  val dotSlot = UInt(cfg.dotSlotWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val outputBase = UInt(cfg.outputIndexWidth bits)
  val contextBank = UInt(cfg.contextBankWidth bits)
  val contextSlot = UInt(cfg.contextSlotWidth bits)
  val headLane = UInt(2 bits)
  val svLimbShift = UInt(5 bits)
}

/**
  * Production 532 MHz unified DSP array core.
  *
  * - Projection: mapper supplies packed map-0 pair on A and int8 activation on
  *   B; the array returns two exact output rows per site.
  * - QK/LM: mapper supplies sign-extended int8 pairs; the same chains/tree
  *   return one exact S32 dot product.
  * - SV scale/delta: mapper supplies one context limb or V plus an unsigned
  *   Q1.15 factor.  The same 128 DSP sites return raw S48 values in four cycles;
  *   the external coordinator reconstructs S48 context scale or adds the delta.
  *
  * SV output is an elastic Stream.  Backpressure freezes the operand register
  * and all three DSP stages, preserving every vector without a 6144-bit FIFO.
  * Four external context slots can rotate at II=1 because SV latency is four
  * cycles.  Dot results use a small credit-covered FIFO and never stall the
  * systolic chain.
  */
class BitNetUnifiedMatrixArrayCore(
    cfg: BitNetUnifiedMatrixConfig = BitNetUnifiedMatrixConfig(),
    useVendorPrimitive: Boolean = true
) extends Component {
  private val operandPipelineStages = 1
  private val siteLatency = operandPipelineStages + cfg.sitePipelineStages
  private val chainLatency = cfg.chainLength * siteLatency
  private val dotLatency = chainLatency + cfg.reductionLevels + 1
  private val svLatency = siteLatency

  val io = new Bundle {
    val weightLoad = slave(Stream(BitNetUnifiedMappedWeightLoad(cfg)))
    val job = slave(Stream(BitNetUnifiedJob(cfg)))
    val phase = slave(Stream(BitNetUnifiedMappedPhase(cfg)))
    val dotResult = master(Stream(BitNetUnifiedDotResult(cfg)))
    val svResult = master(Stream(BitNetUnifiedSvRawResult(cfg)))

    val jobActive = out Bool()
    val draining = out Bool()
    val protocolFault = out Bool()
    val dotPipelineLatency = out UInt(8 bits)
    val svPipelineLatency = out UInt(8 bits)
    val acceptedJobs = out UInt(32 bits)
    val acceptedPhases = out UInt(32 bits)
    val emittedResults = out UInt(32 bits)
  }

  private def isDot(mode: BitNetUnifiedMatrixMode.C): Bool =
    mode === BitNetUnifiedMatrixMode.TernaryProjection ||
      mode === BitNetUnifiedMatrixMode.QkDotI8 ||
      mode === BitNetUnifiedMatrixMode.LmHeadDotI8

  private def isSv(mode: BitNetUnifiedMatrixMode.C): Bool =
    mode === BitNetUnifiedMatrixMode.SvContextScaleQ15 ||
      mode === BitNetUnifiedMatrixMode.SvAxpyQ15

  private def legalModeFormat(
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

  val active = RegInit(False)
  val closing = RegInit(False)
  val mode = Reg(BitNetUnifiedMatrixMode()) init (
    BitNetUnifiedMatrixMode.TernaryProjection
  )
  val format = Reg(BitNetUnifiedOperandFormat()) init (
    BitNetUnifiedOperandFormat.Map0Packed
  )
  val weightBank = Reg(UInt(cfg.weightBankWidth bits)) init (0)
  val weightEpoch = Reg(UInt(cfg.epochWidth bits)) init (0)
  val phasesRemaining = Reg(UInt(cfg.phaseCountWidth bits)) init (0)
  val outstanding = Reg(UInt(log2Up(cfg.resultFifoDepth + 1) bits)) init (0)
  val stickyFault = RegInit(False)

  val stationaryOperands = Vec(
    Vec(Reg(SInt(cfg.dspAWidth bits)) init (0), cfg.siteCount),
    cfg.weightBanks
  )
  val loadedEpoch = Vec(Reg(UInt(cfg.epochWidth bits)) init (0), cfg.weightBanks)
  val loadedValid = Vec(RegInit(False), cfg.weightBanks)

  val jobNeedsWeights = isDot(io.job.mode)
  val requestedWeightsReady = loadedValid(io.job.weightBank) &&
    loadedEpoch(io.job.weightBank) === io.job.weightEpoch
  val requestedBankBeingLoaded = io.weightLoad.valid &&
    io.weightLoad.bank === io.job.weightBank
  val jobLegal = legalModeFormat(io.job.mode, io.job.format) &&
    io.job.phaseCount =/= 0 &&
    (!jobNeedsWeights || (requestedWeightsReady && !requestedBankBeingLoaded))
  io.job.ready := !active && jobLegal
  when(io.job.valid && !active && !jobLegal) { stickyFault := True }
  when(io.job.fire) {
    active := True
    closing := False
    mode := io.job.mode
    format := io.job.format
    weightBank := io.job.weightBank
    weightEpoch := io.job.weightEpoch
    phasesRemaining := io.job.phaseCount
    outstanding := 0
  }

  io.weightLoad.ready := !(active && isDot(mode) &&
    io.weightLoad.bank === weightBank)
  when(io.weightLoad.fire) {
    loadedEpoch(io.weightLoad.bank) := io.weightLoad.epoch
    loadedValid(io.weightLoad.bank) := True
    for (site <- 0 until cfg.siteCount) {
      stationaryOperands(io.weightLoad.bank)(site) := io.weightLoad.values(site)
    }
  }

  // Dot output is small enough for a conventional FIFO.  SV instead freezes
  // the physical array, avoiding a wide FIFO and preserving arbitrary
  // backpressure exactly.
  val svValidPipe = Vec(RegInit(False), svLatency)
  val svOutValid = svValidPipe(svLatency - 1)
  val svAdvance = !svOutValid || io.svResult.ready
  val dotHasCredit = outstanding =/= cfg.resultFifoDepth
  io.phase.ready := active && !closing && phasesRemaining =/= 0 &&
    Mux(isSv(mode), svAdvance, dotHasCredit)
  val phaseFire = io.phase.fire
  val dotFire = phaseFire && isDot(mode)
  val svFire = phaseFire && isSv(mode)

  when(phaseFire) {
    phasesRemaining := phasesRemaining - 1
    when(phasesRemaining === 1) { closing := True }
  }

  val acceptedMeta = BitNetUnifiedArrayMeta(cfg)
  acceptedMeta.dotSlot := io.phase.dotSlot
  acceptedMeta.firstK := io.phase.firstK
  acceptedMeta.lastK := io.phase.lastK
  acceptedMeta.tokenId := io.phase.tokenId
  acceptedMeta.outputBase := io.phase.outputBase
  acceptedMeta.contextBank := io.phase.contextBank
  acceptedMeta.contextSlot := io.phase.contextSlot
  acceptedMeta.headLane := io.phase.headLane
  acceptedMeta.svLimbShift := io.phase.svLimbShift

  val dotValidPipe = Vec(RegInit(False), dotLatency)
  val dotMetaPipe = Vec(
    Reg(BitNetUnifiedArrayMeta(cfg)) init (BitNetUnifiedArrayMeta(cfg).getZero),
    dotLatency
  )
  dotValidPipe(0) := dotFire
  when(dotFire) { dotMetaPipe(0) := acceptedMeta }
  for (stage <- 1 until dotLatency) {
    dotValidPipe(stage) := dotValidPipe(stage - 1)
    when(dotValidPipe(stage - 1)) {
      dotMetaPipe(stage) := dotMetaPipe(stage - 1)
    }
  }

  val svMetaPipe = Vec(
    Reg(BitNetUnifiedArrayMeta(cfg)) init (BitNetUnifiedArrayMeta(cfg).getZero),
    svLatency
  )
  when(svAdvance) {
    svValidPipe(0) := svFire
    when(svFire) { svMetaPipe(0) := acceptedMeta }
    for (stage <- 1 until svLatency) {
      svValidPipe(stage) := svValidPipe(stage - 1)
      when(svValidPipe(stage - 1)) {
        svMetaPipe(stage) := svMetaPipe(stage - 1)
      }
    }
  }
  val chainOutputs = Vec(SInt(cfg.dspPWidth bits), cfg.numChains)
  val svOutputs = Vec(SInt(cfg.dspPWidth bits), cfg.siteCount)
  for (chain <- 0 until cfg.numChains) {
    // Keep sixteen physically separate CE cones.  A single mode/ready LUT
    // driving every CE pin missed 532 MHz on route delay despite one logic
    // level; one cone per 8-site chain bounds fanout without changing state.
    val chainAdvance = Bool()
    chainAdvance := !isSv(mode) || svAdvance
    chainAdvance.addAttribute("keep", "true")
    chainAdvance.addAttribute("max_fanout", "16")
    val siteOutputs = new Array[SInt](cfg.chainLength)
    for (siteInChain <- 0 until cfg.chainLength) {
      val site = chain * cfg.chainLength + siteInChain
      val offset = siteInChain * siteLatency
      val dotActivation = if (offset == 0) io.phase.activation(site) else
        Delay(
          io.phase.activation(site),
          offset,
          init = S(0, cfg.activationWidth bits)
        )

      val selectedStationary = SInt(cfg.dspAWidth bits)
      selectedStationary := stationaryOperands(0)(site)
      when(weightBank === 1) {
        selectedStationary := stationaryOperands(1)(site)
      }

      val selectedA = SInt(cfg.dspAWidth bits)
      val selectedB = SInt(cfg.dspBWidth bits)
      val selectedC = SInt(cfg.dspPWidth bits)
      selectedA := io.phase.svValue(site)
      selectedB := io.phase.svFactorQ15.resize(cfg.dspBWidth).asSInt
      if (siteInChain == 0) {
        selectedC := 0
      } else {
        selectedC := siteOutputs(siteInChain - 1)
      }
      when(isDot(mode)) {
        selectedA := selectedStationary
        selectedB := dotActivation.resize(cfg.dspBWidth)
      }

      // Register the mode mux before the DSP input registers.  This preserves
      // the 532 MHz A/B/C timing contract at the cost of one latency stage,
      // while II remains one phase per fast clock.
      val aOperand = Reg(SInt(cfg.dspAWidth bits)) init (0)
      val bOperand = Reg(SInt(cfg.dspBWidth bits)) init (0)
      val cOperand = Reg(SInt(cfg.dspPWidth bits)) init (0)
      when(chainAdvance) {
        aOperand := selectedA
        bOperand := selectedB
        cOperand := selectedC
      }

      if (useVendorPrimitive) {
        val dsp = new BitNetPackedMacDsp48E2(opmodeReg = 1)
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
        // Dot: P=M+C.  SV: P=M only; AXPY addition/saturation is performed by
        // the bank-local context coordinator, so no 48-bit C mux exists here.
        dsp.OPMODE := B"9'b000000101"
        when(isDot(mode)) { dsp.OPMODE := B"9'b000110101" }
        dsp.CARRYINSEL := 0
        dsp.CEA1 := chainAdvance
        dsp.CEA2 := False
        dsp.CEB1 := chainAdvance
        dsp.CEB2 := False
        dsp.CEC := chainAdvance
        dsp.CED := False
        dsp.CEAD := False
        dsp.CEM := chainAdvance
        dsp.CEP := chainAdvance
        dsp.CECARRYIN := False
        dsp.CECTRL := chainAdvance
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
        val aReg = Reg(SInt(cfg.dspAWidth bits)) init (0)
        val bReg = Reg(SInt(cfg.dspBWidth bits)) init (0)
        val cReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        val mReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        val cDelay = Reg(SInt(cfg.dspPWidth bits)) init (0)
        val pReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        when(chainAdvance) {
          aReg := aOperand
          bReg := bOperand
          cReg := cOperand
          mReg := (aReg * bReg).resize(cfg.dspPWidth)
          cDelay := cReg
          when(isDot(mode)) {
            pReg := (mReg + cDelay).resize(cfg.dspPWidth)
          } otherwise {
            pReg := mReg
          }
        }
        siteOutputs(siteInChain) = pReg
      }
      svOutputs(site) := siteOutputs(siteInChain)
    }
    chainOutputs(chain) := siteOutputs(cfg.chainLength - 1)
  }

  var reduceLevel: Seq[SInt] = chainOutputs.toSeq
  for (level <- 0 until cfg.reductionLevels) {
    val levelValid = dotValidPipe(chainLatency + level - 1)
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
  val reduced = reduceLevel.head

  val lowField = reduced(cfg.laneSeparation - 1 downto 0)
  val ternaryLow = lowField.resize(cfg.accumulatorWidth)
  val ternaryHighRaw =
    (reduced >> cfg.laneSeparation).resize(cfg.accumulatorWidth)
  val ternaryBorrow = SInt(cfg.accumulatorWidth bits)
  ternaryBorrow := 0
  when(lowField.msb) { ternaryBorrow := 1 }

  val tile = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  tile(0) := reduced.resize(cfg.accumulatorWidth)
  tile(1) := 0
  when(mode === BitNetUnifiedMatrixMode.TernaryProjection) {
    tile(0) := ternaryLow
    tile(1) := (ternaryHighRaw + ternaryBorrow).resize(cfg.accumulatorWidth)
  }
  val tileReg = Vec(
    Reg(SInt(cfg.accumulatorWidth bits)) init (0),
    cfg.outputLanes
  )
  when(dotValidPipe(dotLatency - 2)) { tileReg := tile }

  val accumulators = Vec(
    Vec(Reg(SInt(cfg.accumulatorWidth bits)) init (0), cfg.outputLanes),
    cfg.dotSlots
  )
  val dotValid = dotValidPipe(dotLatency - 1)
  val dotMeta = dotMetaPipe(dotLatency - 1)
  val accumulated = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  for (lane <- 0 until cfg.outputLanes) {
    val previous = SInt(cfg.accumulatorWidth bits)
    previous := accumulators(dotMeta.dotSlot)(lane)
    when(dotMeta.firstK) { previous := 0 }
    accumulated(lane) := (previous + tileReg(lane)).resize(cfg.accumulatorWidth)
    when(dotValid) { accumulators(dotMeta.dotSlot)(lane) := accumulated(lane) }
  }

  val dotPush = Stream(BitNetUnifiedDotResult(cfg))
  dotPush.valid := dotValid
  dotPush.mode := mode
  dotPush.format := format
  dotPush.weightBank := weightBank
  dotPush.weightEpoch := weightEpoch
  dotPush.values := accumulated
  dotPush.tileValues := tileReg
  dotPush.dotSlot := dotMeta.dotSlot
  dotPush.firstK := dotMeta.firstK
  dotPush.lastK := dotMeta.lastK
  dotPush.tokenId := dotMeta.tokenId
  dotPush.outputBase := dotMeta.outputBase
  val dotFifo = StreamFifo(BitNetUnifiedDotResult(cfg), cfg.resultFifoDepth)
  dotFifo.io.push << dotPush
  io.dotResult << dotFifo.io.pop
  when(dotPush.valid && !dotPush.ready) { stickyFault := True }

  val svMeta = svMetaPipe(svLatency - 1)
  io.svResult.valid := svOutValid
  io.svResult.mode := mode
  io.svResult.format := format
  io.svResult.contextBank := svMeta.contextBank
  io.svResult.contextSlot := svMeta.contextSlot
  io.svResult.headLane := svMeta.headLane
  io.svResult.svLimbShift := svMeta.svLimbShift
  io.svResult.tokenId := svMeta.tokenId
  io.svResult.rawP := svOutputs

  val selectedResultFire = Mux(isSv(mode), io.svResult.fire, io.dotResult.fire)
  switch((phaseFire ## selectedResultFire).asBits) {
    is(B"2'b10") { outstanding := outstanding + 1 }
    is(B"2'b01") { outstanding := outstanding - 1 }
  }
  when(closing && selectedResultFire && outstanding === 1) {
    active := False
    closing := False
  }

  val jobs = Reg(UInt(32 bits)) init (0)
  val phases = Reg(UInt(32 bits)) init (0)
  val results = Reg(UInt(32 bits)) init (0)
  when(io.job.fire) { jobs := jobs + 1 }
  when(phaseFire) { phases := phases + 1 }
  when(selectedResultFire) { results := results + 1 }

  io.jobActive := active
  io.draining := active && closing
  io.protocolFault := stickyFault
  io.dotPipelineLatency := dotLatency
  io.svPipelineLatency := svLatency
  io.acceptedJobs := jobs
  io.acceptedPhases := phases
  io.emittedResults := results
}

object GenerateBitNetUnifiedMatrixArrayCore extends App {
  val positional = BitNetResearchGenerationPolicy
    .requireOptIn("GenerateBitNetUnifiedMatrixArrayCore", args)
  require(positional.length <= 1,
    "usage: GenerateBitNetUnifiedMatrixArrayCore " +
      "--allow-legacy-research [target-directory]")
  SpinalConfig(
    targetDirectory = positional.headOption.getOrElse(
      "target/generated-bitnet-unified-array-core"),
    oneFilePerComponent = false
  ).generateVerilog(new BitNetUnifiedMatrixArrayCore())
}
