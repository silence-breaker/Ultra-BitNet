package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Two-output-lane DSP packing geometry for the native BitNet matrix engine.
  *
  * Both output lanes multiply the same int8 activation by independent map-0
  * ternary weights.  This is the useful matrix-vector reuse pattern: one DSP
  * multiplier produces two packed products without the cross terms that would
  * appear if both multiplier operands contained two independent values.
  */
case class PackedDspTernaryMatrixConfig(
    chainLength: Int = 8,
    numChains: Int = 16,
    outputLanes: Int = 2,
    tokenSlots: Int = 2,
    weightBanks: Int = 2,
    activationWidth: Int = 8,
    laneSeparation: Int = 18,
    dspAWidth: Int = 27,
    dspBWidth: Int = 18,
    dspPWidth: Int = 48,
    accumulatorWidth: Int = 32,
    epochWidth: Int = 12,
    tokenIdWidth: Int = 16,
    outputIndexWidth: Int = 18,
    sitePipelineStages: Int = 3
) {
  require(chainLength == 8)
  require(numChains == 16)
  require(outputLanes == 2)
  require(tokenSlots == 2)
  require(weightBanks == 2)
  require(activationWidth == 8)
  require(laneSeparation >= 17 && laneSeparation <= 24)
  require(dspAWidth == 27 && dspBWidth == 18 && dspPWidth == 48)
  require(sitePipelineStages == 3, "A/B input, M, and P must all be registered")
  require(accumulatorWidth >= 24 && accumulatorWidth <= dspPWidth)

  val siteCount: Int = chainLength * numChains
  val codesPerSite: Int = outputLanes
  val packedWeightWidth: Int = siteCount * outputLanes * 2
  val weightBankWidth: Int = log2Up(weightBanks)
  val tokenSlotWidth: Int = log2Up(tokenSlots)
  val reductionLevels: Int = log2Up(numChains)
  val chainPipelineLatency: Int = chainLength * sitePipelineStages
  // The packed lane extraction/borrow correction is registered before the
  // per-token cross-K accumulator.  This keeps each 532 MHz stage to one
  // carry-chain operation instead of combining unpack and accumulation.
  val unpackPipelineStages: Int = 1
  val pipelineLatency: Int =
    chainPipelineLatency + reductionLevels + unpackPipelineStages

  // One tile contains 128 signed int8 terms.  Eighteen bits leave more than
  // two guard bits beyond the exact signed dot-product requirement.
  val worstTileMagnitude: Int = siteCount * (1 << (activationWidth - 1))
  require(worstTileMagnitude < (1 << (laneSeparation - 1)))
}

case class PackedDspTernaryWeightLoad(cfg: PackedDspTernaryMatrixConfig)
    extends Bundle {
  val bank = UInt(cfg.weightBankWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  // Per site: bits 1:0 are output lane 0, bits 3:2 are output lane 1.
  val packedMap0 = Bits(cfg.packedWeightWidth bits)
}

case class PackedDspTernaryTokenPhase(cfg: PackedDspTernaryMatrixConfig)
    extends Bundle {
  val bank = UInt(cfg.weightBankWidth bits)
  val weightEpoch = UInt(cfg.epochWidth bits)
  val slot = UInt(cfg.tokenSlotWidth bits)
  val activations = Vec(SInt(cfg.activationWidth bits), cfg.siteCount)
  val firstK = Bool()
  val lastK = Bool()
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val outputBase = UInt(cfg.outputIndexWidth bits)
}

case class PackedDspTernaryMatrixResult(cfg: PackedDspTernaryMatrixConfig)
    extends Bundle {
  val values = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  val tileValues = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  val slot = UInt(cfg.tokenSlotWidth bits)
  val firstK = Bool()
  val lastK = Bool()
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val outputBase = UInt(cfg.outputIndexWidth bits)
}

/**
  * Minimal DSP48E2 primitive view for one fully-pipelined M+C operation.
  *
  * Keeping this wrapper local prevents the production engine from depending on
  * BitNet's source tree while still making the AREG/BREG/MREG/CREG/PREG
  * contract unambiguous to Vivado.  Unused primitive outputs are intentionally
  * left open by named-port instantiation.
  */
class BitNetPackedMacDsp48E2(
    opmodeReg: Int = 0,
    aMultSel: String = "A",
    dReg: Int = 0,
    adReg: Int = 0,
    bReg: Int = 1
) extends BlackBox {
  require(aMultSel == "A" || aMultSel == "AD")
  require(dReg == 0 || dReg == 1)
  require(adReg == 0 || adReg == 1)
  require(bReg >= 0 && bReg <= 2)
  setDefinitionName("DSP48E2")
  addGeneric("A_INPUT", "DIRECT")
  addGeneric("B_INPUT", "DIRECT")
  addGeneric("AMULTSEL", aMultSel)
  addGeneric("BMULTSEL", "B")
  addGeneric("PREADDINSEL", "A")
  addGeneric("USE_MULT", "MULTIPLY")
  addGeneric("USE_SIMD", "ONE48")
  addGeneric("AREG", 1)
  addGeneric("BREG", bReg)
  addGeneric("CREG", 1)
  addGeneric("DREG", dReg)
  addGeneric("ADREG", adReg)
  addGeneric("MREG", 1)
  addGeneric("PREG", 1)
  addGeneric("ACASCREG", 1)
  addGeneric("BCASCREG", 1)
  addGeneric("CARRYINREG", 0)
  addGeneric("CARRYINSELREG", 0)
  addGeneric("INMODEREG", 0)
  addGeneric("OPMODEREG", opmodeReg)
  addGeneric("ALUMODEREG", 0)

  val A = in Bits (30 bits)
  val B = in Bits (18 bits)
  val C = in Bits (48 bits)
  val D = in Bits (27 bits)
  val CARRYIN = in Bool()
  val ACIN = in Bits (30 bits)
  val BCIN = in Bits (18 bits)
  val PCIN = in Bits (48 bits)
  val CARRYCASCIN = in Bool()
  val MULTSIGNIN = in Bool()
  val ALUMODE = in Bits (4 bits)
  val INMODE = in Bits (5 bits)
  val OPMODE = in Bits (9 bits)
  val CARRYINSEL = in Bits (3 bits)

  val CEA1, CEA2, CEB1, CEB2, CEC, CED, CEAD, CEM, CEP = in Bool()
  val CECARRYIN, CECTRL, CEINMODE, CEALUMODE = in Bool()
  val RSTA, RSTB, RSTC, RSTD, RSTM, RSTP = in Bool()
  val RSTALLCARRYIN, RSTCTRL, RSTINMODE, RSTALUMODE = in Bool()
  val CLK = in Bool()
  val P = out Bits (48 bits)
  // Dedicated vertical cascade output.  Production native arrays must route
  // partial sums through PCOUT/PCIN instead of copying the 48-bit P bus back
  // into fabric and then into the next DSP's C port.
  val PCOUT = out Bits (48 bits)

  noIoPrefix()
  mapClockDomain(clock = CLK)
}

/**
  * BitNet packed ternary engine with an explicit 532 MHz pipeline.
  * This block implements the packed ternary projection mode only; QK, SV and
  * int8 LM-head modes require operand/result routing around the same physical
  * DSP array and are deliberately not claimed by this component.
  *
  * Each physical site computes, in one DSP48E2:
  *
  *   P += activation * (weight0 + (weight1 << laneSeparation))
  *
  * weight0/weight1 are independent {-1,0,+1} map-0 values and activation is
  * shared.  The complete 128-site accumulation remains packed; after the final
  * reduction, lane 0 is the signed low field and lane 1 is the arithmetic high
  * field plus one when the low field is negative (the exact two's-complement
  * borrow correction).  No approximation or saturation occurs inside a tile.
  *
  * Every MAC site has three explicit stages: selected A/B operands, multiplier
  * M, and accumulator P.  This is the inference form of DSP48E2 AREG/BREG,
  * MREG and PREG.  A token still enters every 2x clock; deeper latency does not
  * reduce initiation rate.  The registered 16-to-1 tree adds 15 DSP ALUs, so a
  * correctly inferred engine is 128 MAC DSPs + 15 reduction DSPs = 143 DSPs
  * while performing 256 ternary MACs per cycle.
  */
class BitNetPackedDspTernaryMatrixEngine(
    cfg: PackedDspTernaryMatrixConfig = PackedDspTernaryMatrixConfig(),
    useVendorPrimitive: Boolean = true
) extends Component {
  val io = new Bundle {
    val weightLoad = slave(Stream(PackedDspTernaryWeightLoad(cfg)))
    val tokenPhase = slave(Stream(PackedDspTernaryTokenPhase(cfg)))
    val result = master(Flow(PackedDspTernaryMatrixResult(cfg)))

    val weightBankValid = out Bits(cfg.weightBanks bits)
    val bankBusy = out Bits(cfg.weightBanks bits)
    val busy = out Bool()
    val acceptedWeightLoads = out UInt(32 bits)
    val acceptedTokenPhases = out UInt(32 bits)
    val emittedTileResults = out UInt(32 bits)
  }

  private def decodeMap0(code: Bits): SInt = {
    val decoded = SInt(cfg.dspAWidth bits)
    decoded := 0
    switch(code) {
      is(B"2'b01") { decoded := -1 }
      is(B"2'b11") { decoded := 1 }
    }
    decoded
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
        val packedValue = BigInt(low) + (BigInt(high) << cfg.laneSeparation)
        is(B((highCodeValue << 2) | lowCodeValue, 4 bits)) {
          packed := S(packedValue, cfg.dspAWidth bits)
        }
      }
    }
    packed
  }

  // Decode once at the load boundary.  The token-rate path sees only a 2:1
  // bank mux followed by the DSP A/B input register, not map decoding logic.
  val expandedWeights = Vec(
    Vec(Reg(SInt(cfg.dspAWidth bits)) init (0), cfg.siteCount),
    cfg.weightBanks
  )
  val weightEpoch = Vec(Reg(UInt(cfg.epochWidth bits)) init (0), cfg.weightBanks)
  val weightValid = Vec(RegInit(False), cfg.weightBanks)

  val validPipe = Vec(RegInit(False), cfg.pipelineLatency)
  val bankPipe = Vec(
    Reg(UInt(cfg.weightBankWidth bits)) init (0),
    cfg.pipelineLatency
  )
  val slotPipe = Vec(
    Reg(UInt(cfg.tokenSlotWidth bits)) init (0),
    cfg.pipelineLatency
  )
  val firstKPipe = Vec(RegInit(False), cfg.pipelineLatency)
  val lastKPipe = Vec(RegInit(False), cfg.pipelineLatency)
  val tokenIdPipe = Vec(
    Reg(UInt(cfg.tokenIdWidth bits)) init (0),
    cfg.pipelineLatency
  )
  val outputBasePipe = Vec(
    Reg(UInt(cfg.outputIndexWidth bits)) init (0),
    cfg.pipelineLatency
  )

  val bankIsBusy = Bits(cfg.weightBanks bits)
  for (bank <- 0 until cfg.weightBanks) {
    bankIsBusy(bank) := validPipe.zip(bankPipe).map { case (valid, selected) =>
      valid && selected === bank
    }.reduce(_ || _)
  }
  io.bankBusy := bankIsBusy
  io.weightBankValid := weightValid.asBits
  io.busy := validPipe.asBits.orR

  io.weightLoad.ready := !bankIsBusy(io.weightLoad.bank)
  when(io.weightLoad.fire) {
    weightEpoch(io.weightLoad.bank) := io.weightLoad.epoch
    weightValid(io.weightLoad.bank) := True
    for (site <- 0 until cfg.siteCount) {
      val base = site * cfg.outputLanes * 2
      val lowCode = io.weightLoad.packedMap0(base + 1 downto base)
      val highCode = io.weightLoad.packedMap0(base + 3 downto base + 2)
      expandedWeights(io.weightLoad.bank)(site) := packWeightPair(lowCode, highCode)
    }
  }

  val selectedBankReady = weightValid(io.tokenPhase.bank) &&
    weightEpoch(io.tokenPhase.bank) === io.tokenPhase.weightEpoch
  val selectedBankBeingWritten = io.weightLoad.valid &&
    io.weightLoad.bank === io.tokenPhase.bank
  io.tokenPhase.ready := selectedBankReady && !selectedBankBeingWritten
  val phaseFire = io.tokenPhase.fire

  validPipe(0) := phaseFire
  when(phaseFire) {
    bankPipe(0) := io.tokenPhase.bank
    slotPipe(0) := io.tokenPhase.slot
    firstKPipe(0) := io.tokenPhase.firstK
    lastKPipe(0) := io.tokenPhase.lastK
    tokenIdPipe(0) := io.tokenPhase.tokenId
    outputBasePipe(0) := io.tokenPhase.outputBase
  }
  for (stage <- 1 until cfg.pipelineLatency) {
    validPipe(stage) := validPipe(stage - 1)
    when(validPipe(stage - 1)) {
      bankPipe(stage) := bankPipe(stage - 1)
      slotPipe(stage) := slotPipe(stage - 1)
      firstKPipe(stage) := firstKPipe(stage - 1)
      lastKPipe(stage) := lastKPipe(stage - 1)
      tokenIdPipe(stage) := tokenIdPipe(stage - 1)
      outputBasePipe(stage) := outputBasePipe(stage - 1)
    }
  }

  val chainOutputs = Vec(SInt(cfg.dspPWidth bits), cfg.numChains)
  for (chain <- 0 until cfg.numChains) {
    val siteOutputs = new Array[SInt](cfg.chainLength)

    for (siteInChain <- 0 until cfg.chainLength) {
      val site = chain * cfg.chainLength + siteInChain
      val siteOffset = siteInChain * cfg.sitePipelineStages
      val inputValid = if (siteInChain == 0) {
        phaseFire
      } else {
        validPipe(siteOffset - 1)
      }
      val selectedBank = if (siteInChain == 0) {
        io.tokenPhase.bank
      } else {
        bankPipe(siteOffset - 1)
      }
      val delayedActivation = if (siteInChain == 0) {
        io.tokenPhase.activations(site)
      } else {
        Delay(
          io.tokenPhase.activations(site),
          siteOffset,
          init = S(0, cfg.activationWidth bits)
        )
      }

      val selectedWeight = SInt(cfg.dspAWidth bits)
      selectedWeight := expandedWeights(0)(site)
      when(selectedBank === 1) {
        selectedWeight := expandedWeights(1)(site)
      }
      val previousP = if (siteInChain == 0) {
        S(0, cfg.dspPWidth bits)
      } else {
        siteOutputs(siteInChain - 1)
      }

      if (useVendorPrimitive) {
        // DSP48E2 is explicitly configured as M+C.  A/B/C capture together,
        // M captures one cycle later, and P captures after another cycle.
        val dsp = new BitNetPackedMacDsp48E2
        dsp.A := selectedWeight.resize(30).asBits
        dsp.B := delayedActivation.resize(cfg.dspBWidth).asBits
        dsp.C := previousP.asBits
        dsp.D := 0
        dsp.CARRYIN := False
        dsp.ACIN := 0
        dsp.BCIN := 0
        dsp.PCIN := 0
        dsp.CARRYCASCIN := False
        dsp.MULTSIGNIN := False

        // W=0, X=M, Y=M, Z=C -> P=M+C; INMODE selects the single A1/B1
        // input registers and gates off the unused D pre-adder input.
        dsp.ALUMODE := B"4'b0000"
        dsp.INMODE := B"5'b10101"
        dsp.OPMODE := B"9'b000110101"
        dsp.CARRYINSEL := B"3'b000"

        // Keep the fine-grained DSP pipeline advancing every compute clock.
        // Stream validity is carried by the parallel metadata pipeline, so
        // values captured during bubbles are never committed.  Besides making
        // bubbles behave like a conventional fixed-latency pipeline, this
        // avoids distributing phaseFire/weightEpoch comparisons to hundreds
        // of DSP CE pins on the 532 MHz clock domain.
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

        // Valid metadata is reset and masks stale/bubble data, so the datapath
        // reset pins can remain inactive as in BitNet's MPU.
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
        // Portable bit-accurate model used by Verilator.  It preserves the
        // same three-cycle site latency and II=1 as the DSP48E2 primitive.
        val aReg = Reg(SInt(cfg.dspAWidth bits)) init (0)
        val bReg = Reg(SInt(cfg.dspBWidth bits)) init (0)
        val cReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        aReg := selectedWeight
        bReg := delayedActivation.resize(cfg.dspBWidth)
        cReg := previousP
        val mReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        val cDelay = Reg(SInt(cfg.dspPWidth bits)) init (0)
        mReg := (aReg * bReg).resize(cfg.dspPWidth)
        cDelay := cReg
        val pReg = Reg(SInt(cfg.dspPWidth bits)) init (0)
        pReg := (mReg + cDelay).resize(cfg.dspPWidth)
        siteOutputs(siteInChain) = pReg
      }
    }
    chainOutputs(chain) := siteOutputs(cfg.chainLength - 1)
  }

  // Packed sums stay packed through the tree.  Carries/borrows between lanes
  // are corrected once, after the complete 128-site dot product.
  var reduceLevel: Seq[SInt] = chainOutputs.toSeq
  for (level <- 0 until cfg.reductionLevels) {
    val levelValid = validPipe(cfg.chainPipelineLatency + level - 1)
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
  val packedTileSum = reduceLevel.head

  val lane0Field = packedTileSum(cfg.laneSeparation - 1 downto 0)
  val lane0Tile = lane0Field.resize(cfg.accumulatorWidth)
  val highRaw = (packedTileSum >> cfg.laneSeparation).resize(cfg.accumulatorWidth)
  val borrowCorrection = SInt(cfg.accumulatorWidth bits)
  borrowCorrection := 0
  when(lane0Field.msb) { borrowCorrection := 1 }
  val lane1Tile = (highRaw + borrowCorrection).resize(cfg.accumulatorWidth)
  val tileValues = Vec(lane0Tile, lane1Tile)

  val unpackIndex = cfg.pipelineLatency - 2
  val unpackedTileValues = Vec(
    Reg(SInt(cfg.accumulatorWidth bits)) init (0),
    cfg.outputLanes
  )
  when(validPipe(unpackIndex)) {
    unpackedTileValues := tileValues
  }

  val slotAccumulators = Vec(
    Vec(Reg(SInt(cfg.accumulatorWidth bits)) init (0), cfg.outputLanes),
    cfg.tokenSlots
  )
  val resultIndex = cfg.pipelineLatency - 1
  val resultValid = validPipe(resultIndex)
  val resultSlot = slotPipe(resultIndex)
  val accumulatedValues = Vec(SInt(cfg.accumulatorWidth bits), cfg.outputLanes)
  for (lane <- 0 until cfg.outputLanes) {
    val previous = SInt(cfg.accumulatorWidth bits)
    previous := slotAccumulators(resultSlot)(lane)
    when(firstKPipe(resultIndex)) { previous := 0 }
    accumulatedValues(lane) := (previous + unpackedTileValues(lane))
      .resize(cfg.accumulatorWidth)
    when(resultValid) {
      slotAccumulators(resultSlot)(lane) := accumulatedValues(lane)
    }
  }

  io.result.valid := resultValid
  io.result.values := accumulatedValues
  io.result.tileValues := unpackedTileValues
  io.result.slot := resultSlot
  io.result.firstK := firstKPipe(resultIndex)
  io.result.lastK := lastKPipe(resultIndex)
  io.result.tokenId := tokenIdPipe(resultIndex)
  io.result.outputBase := outputBasePipe(resultIndex)

  val weightLoadCount = Reg(UInt(32 bits)) init (0)
  val tokenPhaseCount = Reg(UInt(32 bits)) init (0)
  val tileResultCount = Reg(UInt(32 bits)) init (0)
  when(io.weightLoad.fire) { weightLoadCount := weightLoadCount + 1 }
  when(phaseFire) { tokenPhaseCount := tokenPhaseCount + 1 }
  when(resultValid) { tileResultCount := tileResultCount + 1 }
  io.acceptedWeightLoads := weightLoadCount
  io.acceptedTokenPhases := tokenPhaseCount
  io.emittedTileResults := tileResultCount
}

object PackedDspTernaryMatrixOracle {
  def decodeMap0(code: Int): Int = code & 3 match {
    case 1 => -1
    case 3 => 1
    case _ => 0
  }

  def dot2(
      activations: Seq[Int],
      lane0Codes: Seq[Int],
      lane1Codes: Seq[Int]
  ): (Int, Int) = {
    require(activations.length == lane0Codes.length)
    require(activations.length == lane1Codes.length)
    val lane0 = activations.indices.map { index =>
      activations(index) * decodeMap0(lane0Codes(index))
    }.sum
    val lane1 = activations.indices.map { index =>
      activations(index) * decodeMap0(lane1Codes(index))
    }.sum
    (lane0, lane1)
  }

  def packCodes(lane0Codes: Seq[Int], lane1Codes: Seq[Int]): BigInt = {
    require(lane0Codes.length == lane1Codes.length)
    lane0Codes.indices.foldLeft(BigInt(0)) { case (packed, index) =>
      packed |
        (BigInt(lane0Codes(index) & 3) << (index * 4)) |
        (BigInt(lane1Codes(index) & 3) << (index * 4 + 2))
    }
  }

  /** Bit-accurate packed arithmetic, including the high-lane borrow fix. */
  def packedDot2(
      activations: Seq[Int],
      lane0Codes: Seq[Int],
      lane1Codes: Seq[Int],
      laneSeparation: Int = 18
  ): (Int, Int) = {
    val modulus = BigInt(1) << 48
    val packed = activations.indices.foldLeft(BigInt(0)) { case (sum, index) =>
      val w0 = decodeMap0(lane0Codes(index))
      val w1 = decodeMap0(lane1Codes(index))
      sum + BigInt(activations(index)) * (BigInt(w0) + (BigInt(w1) << laneSeparation))
    }
    val p48 = ((packed % modulus) + modulus) % modulus
    val lowMask = (BigInt(1) << laneSeparation) - 1
    val lowBits = p48 & lowMask
    val lowSign = BigInt(1) << (laneSeparation - 1)
    val low = if ((lowBits & lowSign) != 0) lowBits - (BigInt(1) << laneSeparation) else lowBits
    val signedP = if ((p48 & (BigInt(1) << 47)) != 0) p48 - modulus else p48
    val highRaw = signedP >> laneSeparation
    val high = highRaw + (if (low < 0) 1 else 0)
    (low.toInt, high.toInt)
  }
}

object GenerateBitNetPackedDspTernaryMatrixEngine extends App {
  SpinalConfig(
    targetDirectory = "target/generated-bitnet-packed-dsp-matrix",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetPackedDspTernaryMatrixEngine())
}
