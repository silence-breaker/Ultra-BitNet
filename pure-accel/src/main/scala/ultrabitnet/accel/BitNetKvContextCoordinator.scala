package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Physical contract of one bank-local GQA4 value/context coordinator.
  *
  * The coordinator owns one shared probability/scale multiplier.  It schedules
  * the SV modes of the same continuous 128-site array used by projection/QK/LM
  * through the native streams below.  Keeping this boundary independent of the
  * 266->532 MHz mapper avoids duplicating an attention matrix engine or moving
  * score/probability/context data to DDR.
  */
case class BitNetKvContextConfig(
    model: BitNetConfig = BitNetConfig.Production,
    attention: BitNetTiledAttentionConfig = BitNetTiledAttentionConfig(),
    continuous: BitNetContinuousMatrixConfig =
      BitNetContinuousMatrixConfig(),
    contextWidth: Int = 48,
    valueWidth: Int = 8,
    valueScaleWidth: Int = 32,
    dequantLanes: Int = 8,
    tileBuffers: Int = 2,
    phaseTagWidth: Int = 7
) {
  val matrix: BitNetUnifiedMatrixConfig = continuous.base
  require(model.layerCount == 30)
  require(model.kvHeadCount == 5)
  require(model.headSize == attention.headDim)
  require(attention.queryHeadsPerKv == 4 && attention.tileKeys == 16)
  require(contextWidth == 48,
    "the online attention numerator is retained at full S48 precision")
  require(valueWidth == 8)
  require(valueScaleWidth == 32)
  require(dequantLanes == 8 && attention.headDim % dequantLanes == 0)
  require(tileBuffers == 2)
  require(matrix.siteCount == attention.headDim)
  require(matrix.dspAWidth == 27 && matrix.dspPWidth == contextWidth)
  require(continuous.factorWidth == 18 &&
    continuous.factorFractionWidth >= 5)
  require(phaseTagWidth >= log2Up(attention.tileKeys *
    attention.queryHeadsPerKv))

  val bankWidth: Int = log2Up(model.kvHeadCount)
  val layerWidth: Int = model.layerWidth
  val tokenIdWidth: Int = 16
  val requestTagWidth: Int = log2Up(tileBuffers * attention.tileKeys)
  val tileBufferWidth: Int = log2Up(tileBuffers)
  val featureWidth: Int = log2Up(attention.headDim)
}

case class BitNetKvContextCommand(cfg: BitNetKvContextConfig)
    extends Bundle {
  val layer = UInt(cfg.layerWidth bits)
  /** KV head n is resident in physical DDR bank n. */
  val kvHead = UInt(cfg.bankWidth bits)
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val queryMask = Bits(cfg.attention.queryHeadsPerKv bits)
}

/** One semantic 128-byte V read; the downstream AGU resolves its address. */
case class BitNetKvValueReadCommand(cfg: BitNetKvContextConfig)
    extends Bundle {
  val tag = UInt(cfg.requestTagWidth bits)
  val bank = UInt(cfg.bankWidth bits)
  val layer = UInt(cfg.layerWidth bits)
  val keyPosition = UInt(cfg.attention.positionWidth bits)
  val keyIndex = UInt(cfg.attention.positionWidth bits)
  val tileIndex = UInt(cfg.attention.tileIndexWidth bits)
  val tileBuffer = UInt(cfg.tileBufferWidth bits)
  val tileOffset = UInt(cfg.attention.tileOffsetWidth bits)
}

/** Tagged responses may return in arbitrary order. */
case class BitNetKvValueReadResponse(cfg: BitNetKvContextConfig)
    extends Bundle {
  val tag = UInt(cfg.requestTagWidth bits)
  val bank = UInt(cfg.bankWidth bits)
  val layer = UInt(cfg.layerWidth bits)
  val keyPosition = UInt(cfg.attention.positionWidth bits)
  val keyIndex = UInt(cfg.attention.positionWidth bits)
  val value = Vec(SInt(cfg.valueWidth bits), cfg.attention.headDim)
  /** Positive Q16.16 dequant scale from the paired KvValueScale object. */
  val valueScaleQ16 = UInt(cfg.valueScaleWidth bits)
  val success = Bool()
}

/** Scalar stream into the WideWork activation owner (128 features x 4 heads). */
case class BitNetKvContextOutput(cfg: BitNetKvContextConfig)
    extends Bundle {
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val kvHead = UInt(cfg.bankWidth bits)
  val queryHeadLane = UInt(2 bits)
  val feature = UInt(cfg.featureWidth bits)
  /** Normalized, dequantized context in Q16.16. */
  val valueQ16 = SInt(cfg.contextWidth bits)
  val masked = Bool()
  val first = Bool()
  val last = Bool()
}

/**
  * SV-only production phase for the continuous physical array.
  *
  * Context scaling uses unsigned Q1.15 in the low 16 bits.  AXPY uses the
  * complete unsigned UQ3.15 range, so the factor must not pass through the
  * legacy 16-bit `svFactorQ15` bundle.  Raw signed int8 V remains in svValue;
  * the continuous core routes factor to DSP A and V to DSP B.
  */
case class BitNetContinuousSvPhase(cfg: BitNetContinuousMatrixConfig)
    extends Bundle {
  private val b = cfg.base
  val svValue = Vec(SInt(b.dspAWidth bits), b.siteCount)
  val svFactorRaw = Bits(cfg.factorWidth bits)
  val svFactorSigned = Bool()
  val svFactorFractionBits = UInt(cfg.factorFractionWidth bits)
  val svLimbShift = UInt(5 bits)
  val taskId = UInt(cfg.taskIdWidth bits)
  val tokenId = UInt(b.tokenIdWidth bits)
  val contextBank = UInt(b.contextBankWidth bits)
  val contextSlot = UInt(b.contextSlotWidth bits)
  val headLane = UInt(2 bits)
}

/**
  * Bank-local V-tile and online-context coordinator.
  *
  * Two 16x128-byte V memories ping-pong.  Probability metadata never leaves
  * this block; tagged V responses can complete out of order.  Four independent
  * S48 context banks correspond to the four query heads of one KV head.  Tile
  * alpha, every probability/V AXPY, and the final reciprocal scale are issued
  * as mode-locked jobs to one external unified-array adapter.
  */
class BitNetKvContextCoordinator(
    cfg: BitNetKvContextConfig = BitNetKvContextConfig()
) extends Component {
  private val a = cfg.attention
  private val heads = a.queryHeadsPerKv
  private val keys = a.tileKeys

  val io = new Bundle {
    val command = slave(Stream(BitNetKvContextCommand(cfg)))
    val tileUpdate = slave(Stream(BitNetAttentionTileUpdate(a)))
    val probability = slave(Stream(BitNetAttentionProbability(a)))
    val normalization = slave(Stream(BitNetAttentionNormalization(a)))

    val valueReadCommand = master(Stream(BitNetKvValueReadCommand(cfg)))
    val valueReadResponse = slave(Stream(BitNetKvValueReadResponse(cfg)))

    /** Direct production interface of the single continuous physical array. */
    val svSequence = master(Stream(BitNetContinuousSequence(cfg.continuous)))
    val svPhase = master(Stream(BitNetContinuousSvPhase(cfg.continuous)))
    val svResult = slave(Stream(BitNetContinuousSvResultGroup(
      cfg.continuous, cfg.dequantLanes)))

    val contextOut = master(Stream(BitNetKvContextOutput(cfg)))

    val busy = out Bool()
    val protocolFault = out Bool()
    val acceptedValueReads = out UInt(32 bits)
    val completedValueReads = out UInt(32 bits)
    val submittedSvPhases = out UInt(32 bits)
    val completedCommands = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Idle, WaitTile,
        ScaleJob, ScaleLoadIssue, ScaleLoadCapture, ScaleRun,
        ScaleWrite, ScaleRound, ScaleCommit,
        ValueLoad, ValueCapture, ValueAppend, ValueWaitFactor,
        AxpyJob, AxpyRun,
        AxpyContextDelta, AxpyContextSum, AxpyContextCommit,
        WaitNormalization, ReciprocalStart, ReciprocalRun, NormalizeJob,
        EmitLoadIssue, EmitLoadCapture, EmitContext = newElement()
  }
  val state = Reg(State()) init State.Idle

  val fault = RegInit(False)
  val commandActive = RegInit(False)
  val activeLayer = Reg(UInt(cfg.layerWidth bits)) init 0
  val activeKvHead = Reg(UInt(cfg.bankWidth bits)) init 0
  val activeTokenId = Reg(UInt(cfg.tokenIdWidth bits)) init 0
  val activeQueryMask = Reg(Bits(heads bits)) init 0
  io.protocolFault := fault
  io.busy := commandActive || state =/= State.Idle

  val acceptedReadCount = Reg(UInt(32 bits)) init 0
  val completedReadCount = Reg(UInt(32 bits)) init 0
  val submittedPhaseCount = Reg(UInt(32 bits)) init 0
  val completedCommandCount = Reg(UInt(32 bits)) init 0
  val sequenceCounter = Reg(UInt(cfg.continuous.sequenceIdWidth bits)) init 0
  val activeSequenceId = Reg(UInt(cfg.continuous.sequenceIdWidth bits)) init 0
  io.acceptedValueReads := acceptedReadCount
  io.completedValueReads := completedReadCount
  io.submittedSvPhases := submittedPhaseCount
  io.completedCommands := completedCommandCount

  val commandLegal = io.command.kvHead < cfg.model.kvHeadCount &&
    io.command.layer < cfg.model.layerCount
  io.command.ready := state === State.Idle
  when(io.command.fire) {
    fault := !commandLegal
    acceptedReadCount := 0
    completedReadCount := 0
    submittedPhaseCount := 0
    when(commandLegal) {
      commandActive := True
      activeLayer := io.command.layer
      activeKvHead := io.command.kvHead
      activeTokenId := io.command.tokenId
      activeQueryMask := io.command.queryMask
      state := State.WaitTile
    }
  }

  // ---------------------------------------------------------------------
  // Input-side tile allocation and tagged V reads.
  // ---------------------------------------------------------------------
  val fillBank = Reg(Bool()) init False
  val fillActive = RegInit(False)
  val fillOffset = Reg(UInt(a.tileOffsetWidth bits)) init 0
  val nextFillTile = Reg(UInt(a.tileIndexWidth bits)) init 0

  val tileValid = Vec(Reg(Bool()) init False, cfg.tileBuffers)
  val tileCollected = Vec(Reg(Bool()) init False, cfg.tileBuffers)
  val tileAlpha = Vec((0 until cfg.tileBuffers).map { _ =>
    Vec(Reg(UInt(a.expWidth bits)) init 0, heads)
  })
  val tileAllMasked = Vec(Reg(Bits(heads bits)) init 0, cfg.tileBuffers)
  val tileIndex = Vec(Reg(UInt(a.tileIndexWidth bits)) init 0, cfg.tileBuffers)
  val tileKeyCount = Vec(
    Reg(UInt(a.tileKeyCountWidth bits)) init 0, cfg.tileBuffers)
  val tileLast = Vec(Reg(Bool()) init False, cfg.tileBuffers)

  val probabilityWeight = Vec((0 until cfg.tileBuffers).map { _ =>
    Vec((0 until keys).map { _ =>
      Vec(Reg(UInt(a.expWidth bits)) init 0, heads)
    })
  })
  val probabilityKeyPosition = Vec((0 until cfg.tileBuffers).map { _ =>
    Vec(Reg(UInt(a.positionWidth bits)) init 0, keys)
  })
  val probabilityKeyIndex = Vec((0 until cfg.tileBuffers).map { _ =>
    Vec(Reg(UInt(a.positionWidth bits)) init 0, keys)
  })
  val probabilityValueScaleQ16 = Vec((0 until cfg.tileBuffers).map { _ =>
    Vec(Reg(UInt(cfg.valueScaleWidth bits)) init 0, keys)
  })
  val slotAllocated = Vec(Reg(Bits(keys bits)) init 0, cfg.tileBuffers)
  val valueReady = Vec(Reg(Bits(keys bits)) init 0, cfg.tileBuffers)

  private val valueGroups = a.headDim / cfg.dequantLanes
  private val rawGroupWidth = cfg.dequantLanes * cfg.valueWidth
  private val rawVectorWidth = a.headDim * cfg.valueWidth
  private val valueMemoryDepth = keys * valueGroups
  private val valueMemoryAddressWidth = log2Up(valueMemoryDepth)

  // Each V tile is deliberately narrow: 64 bits x 256.  A full-width
  // 1024-bit memory inferred 36 RAMB36 equivalents for the two ping-pong
  // halves on UltraScale+, while this layout consumes one RAMB18 per half.
  val valueTileMemory = Array.fill(cfg.tileBuffers)(
    Mem(Bits(rawGroupWidth bits), valueMemoryDepth)
      .addAttribute("ram_style", "block")
  )

  // A tagged response is accepted once, then serialized into the selected
  // narrow V BRAM.  valueReady is asserted only after group 15 is committed,
  // so the processing walker can never observe a partially written vector.
  val responseWriteActive = RegInit(False)
  val responseWriteBank = Reg(Bool()) init False
  val responseWriteOffset = Reg(UInt(a.tileOffsetWidth bits)) init 0
  val responseWriteGroup = Reg(UInt(log2Up(valueGroups) bits)) init 0
  val responseWriteShift = Reg(Bits(a.headDim * cfg.valueWidth bits)) init 0
  val responseWriteScaleQ16 = Reg(UInt(cfg.valueScaleWidth bits)) init 0

  io.tileUpdate.ready := commandActive && !fillActive && !tileValid(fillBank.asUInt)
  when(io.tileUpdate.fire) {
    val bank = fillBank.asUInt
    tileValid(bank) := True
    tileCollected(bank) := False
    tileIndex(bank) := io.tileUpdate.tileIndex
    tileKeyCount(bank) := io.tileUpdate.keyCount
    tileAllMasked(bank) := io.tileUpdate.allMasked
    tileLast(bank) := io.tileUpdate.lastTile
    slotAllocated(bank) := B(0, keys bits)
    valueReady(bank) := B(0, keys bits)
    for (head <- 0 until heads) {
      tileAlpha(bank)(head) := io.tileUpdate.alpha(head)
    }
    fillActive := True
    fillOffset := 0
    when(io.tileUpdate.tileIndex =/= nextFillTile ||
        io.tileUpdate.firstTile =/= (nextFillTile === 0) ||
        io.tileUpdate.queryMask =/= activeQueryMask ||
        io.tileUpdate.keyCount === 0) {
      fault := True
    }
  }

  val fillSlotFree = !slotAllocated(fillBank.asUInt)(fillOffset)
  val probabilityHasWeight = io.probability.weight.map(_.orR).orR
  val probabilityCanFire = commandActive && fillActive && fillSlotFree &&
    (!probabilityHasWeight || io.valueReadCommand.ready)
  io.probability.ready := probabilityCanFire

  io.valueReadCommand.valid := io.probability.valid && commandActive &&
    fillActive && fillSlotFree && probabilityHasWeight
  io.valueReadCommand.tag := (fillBank ## fillOffset).asUInt
  io.valueReadCommand.bank := activeKvHead
  io.valueReadCommand.layer := activeLayer
  io.valueReadCommand.keyPosition := io.probability.keyPosition
  io.valueReadCommand.keyIndex := io.probability.keyIndex
  io.valueReadCommand.tileIndex := io.probability.tileIndex
  io.valueReadCommand.tileBuffer := fillBank.asUInt
  io.valueReadCommand.tileOffset := fillOffset

  when(io.probability.fire) {
    val bank = fillBank.asUInt
    val offset = fillOffset
    slotAllocated(bank)(offset) := True
    probabilityKeyPosition(bank)(offset) := io.probability.keyPosition
    probabilityKeyIndex(bank)(offset) := io.probability.keyIndex
    for (head <- 0 until heads) {
      probabilityWeight(bank)(offset)(head) := io.probability.weight(head)
    }

    when(!probabilityHasWeight) {
      valueReady(bank)(offset) := True
      probabilityValueScaleQ16(bank)(offset) := U(0, cfg.valueScaleWidth bits)
    } otherwise {
      acceptedReadCount := acceptedReadCount + 1
    }

    val expectedLast = offset.resize(a.tileKeyCountWidth) ===
      tileKeyCount(bank) - 1
    when(io.probability.tileIndex =/= tileIndex(bank) ||
        io.probability.firstInTile =/= (offset === 0) ||
        io.probability.lastInTile =/= expectedLast ||
        io.probability.queryMask =/= activeQueryMask) {
      fault := True
    }

    when(expectedLast) {
      tileCollected(bank) := True
      fillActive := False
      fillBank := !fillBank
      nextFillTile := nextFillTile + 1
      when(io.probability.lastInSequence =/= tileLast(bank)) { fault := True }
    } otherwise {
      fillOffset := fillOffset + 1
    }
  }

  val responseBank = io.valueReadResponse.tag(cfg.requestTagWidth - 1)
  val responseOffset = io.valueReadResponse.tag(a.tileOffsetWidth - 1 downto 0)
  val responseAllocated = slotAllocated(responseBank.asUInt)(responseOffset)
  val responseNotReady = !valueReady(responseBank.asUInt)(responseOffset)
  val responseMetadataMatches =
    io.valueReadResponse.bank === activeKvHead &&
      io.valueReadResponse.layer === activeLayer &&
      io.valueReadResponse.keyPosition ===
        probabilityKeyPosition(responseBank.asUInt)(responseOffset) &&
      io.valueReadResponse.keyIndex ===
        probabilityKeyIndex(responseBank.asUInt)(responseOffset)
  val responseMatches = commandActive && responseAllocated &&
    responseNotReady && responseMetadataMatches
  io.valueReadResponse.ready := !responseWriteActive
  when(io.valueReadResponse.fire) {
    when(responseMatches) {
      responseWriteActive := True
      responseWriteBank := responseBank
      responseWriteOffset := responseOffset
      responseWriteGroup := 0
      responseWriteShift := Mux(io.valueReadResponse.success,
        io.valueReadResponse.value.asBits,
        B(0, a.headDim * cfg.valueWidth bits))
      responseWriteScaleQ16 := Mux(
        io.valueReadResponse.success,
        io.valueReadResponse.valueScaleQ16,
        U(0, cfg.valueScaleWidth bits))
      when(!io.valueReadResponse.success ||
          !io.valueReadResponse.valueScaleQ16.orR) { fault := True }
    } otherwise {
      // Failed, duplicate, stale and unsolicited responses are consumed so a
      // DDR crossbar cannot deadlock, but the command remains visibly failed.
      fault := True
    }
  }

  val responseWriteAddress = UInt(valueMemoryAddressWidth bits)
  responseWriteAddress := ((responseWriteOffset.resize(valueMemoryAddressWidth) <<
    log2Up(valueGroups)) + responseWriteGroup.resize(valueMemoryAddressWidth))
    .resize(valueMemoryAddressWidth)
  when(responseWriteActive) {
    when(responseWriteBank) {
      valueTileMemory(1).write(responseWriteAddress,
        responseWriteShift(rawGroupWidth - 1 downto 0))
    } otherwise {
      valueTileMemory(0).write(responseWriteAddress,
        responseWriteShift(rawGroupWidth - 1 downto 0))
    }
    responseWriteShift := responseWriteShift |>> rawGroupWidth
    when(responseWriteGroup === valueGroups - 1) {
      valueReady(responseWriteBank.asUInt)(responseWriteOffset) := True
      probabilityValueScaleQ16(responseWriteBank.asUInt)(responseWriteOffset) :=
        responseWriteScaleQ16
      completedReadCount := completedReadCount + 1
      responseWriteActive := False
    } otherwise {
      responseWriteGroup := responseWriteGroup + 1
    }
  }

  def validKeyMask(count: UInt): Bits = {
    val mask = Bits(keys bits)
    for (key <- 0 until keys) mask(key) := U(key) < count
    mask
  }
  val processBank = Reg(Bool()) init False
  val processKey = Reg(UInt(a.tileOffsetWidth bits)) init 0
  val processMask = validKeyMask(tileKeyCount(processBank.asUInt))
  val processTileDataReady = tileValid(processBank.asUInt) &&
    tileCollected(processBank.asUInt) &&
    ((valueReady(processBank.asUInt) & processMask) === processMask)

  // ---------------------------------------------------------------------
  // Bank-local S48 context BRAM and one mode-locked SV scheduler.
  // ---------------------------------------------------------------------
  private val contextGroups = a.headDim / cfg.dequantLanes
  private val contextGroupWidth = cfg.dequantLanes * cfg.contextWidth
  private val contextDepth = heads * contextGroups
  private val contextAddressWidth = log2Up(contextDepth)
  val contextMemory = Mem(Bits(contextGroupWidth bits), contextDepth)
    .addAttribute("ram_style", "block")

  val contextClearActive = RegInit(False)
  val contextClearAddress = Reg(UInt(contextAddressWidth bits)) init 0
  when(io.command.fire && commandLegal) {
    contextClearActive := True
    contextClearAddress := 0
  }
  when(contextClearActive) {
    when(contextClearAddress === contextDepth - 1) {
      contextClearActive := False
    } otherwise {
      contextClearAddress := contextClearAddress + 1
    }
  }
  val processTileReady = processTileDataReady && !contextClearActive

  // The one BRAM stores {head, featureGroup}; a 384-bit word contains eight
  // adjacent S48 Q16.16 values.  This replaces four 128-entry register banks
  // and their 4:1/128-way mux cone.
  def contextAddress(head: UInt, group: UInt): UInt =
    (head ## group).asUInt.resize(contextAddressWidth)

  val reciprocalQ15 = Vec(Reg(UInt(a.expWidth bits)) init 0, heads)
  val scaleHead = Reg(UInt(2 bits)) init 0
  val scaleLimbIssued = Reg(UInt(2 bits)) init 0
  val scaleLowSeen = RegInit(False)
  val scaleNormalizing = RegInit(False)
  val contextGroup = Reg(UInt(log2Up(contextGroups) bits)) init 0
  val contextVectorGroups = Vec(
    Reg(Bits(contextGroupWidth bits)) init 0, contextGroups)
  // Low-limb scale products are parked in one small local BRAM until the
  // corresponding high-limb result group arrives.  No 6144-bit result or
  // parallel product bank crosses the 532->266 MHz boundary.
  val scaleLowProductMemory = Mem(Bits(contextGroupWidth bits), contextGroups)
    .addAttribute("ram_style", "block")
  val scaleHighProductGroupReg = Reg(Bits(contextGroupWidth bits)) init 0
  private val scaleProductWidth = cfg.contextWidth + a.expWidth + 1
  val scaleCombined = Vec(
    Reg(SInt(scaleProductWidth bits)) init 0, cfg.dequantLanes)
  val scaleWriteDataReg = Reg(Bits(contextGroupWidth bits)) init 0

  val axpyHead = Reg(UInt(2 bits)) init 0
  val axpyPhaseIssued = RegInit(False)
  val axpyProductGroupReg = Reg(Bits(contextGroupWidth bits)) init 0
  val axpyContextGroupReg = Reg(Bits(contextGroupWidth bits)) init 0
  val axpyDeltaReg = Vec(
    Reg(SInt((cfg.contextWidth + 1) bits)) init 0, cfg.dequantLanes)
  val axpyWriteDataReg = Reg(Bits(contextGroupWidth bits)) init 0
  val axpyJobActive = RegInit(False)

  val currentValueScaleQ16 = Reg(UInt(cfg.valueScaleWidth bits)) init 0
  val rawValueOutputShift = Reg(Bits(rawVectorWidth bits)) init 0
  val valueGroup = Reg(UInt(log2Up(valueGroups) bits)) init 0
  val valueRawGroup = Reg(Bits(rawGroupWidth bits)) init 0
  val rawValue = Vec(SInt(cfg.valueWidth bits), a.headDim)
  for (feature <- 0 until a.headDim) {
    rawValue(feature) := Mux(currentValueScaleQ16.orR,
      rawValueOutputShift(
        (feature + 1) * cfg.valueWidth - 1 downto
          feature * cfg.valueWidth).asSInt,
      S(0, cfg.valueWidth bits))
  }

  val valueReadEnable = state === State.ValueLoad ||
    (state === State.ValueAppend && valueGroup =/= valueGroups - 1)
  val valueReadGroup = UInt(log2Up(valueGroups) bits)
  valueReadGroup := 0
  when(state === State.ValueAppend) {
    valueReadGroup := (valueGroup + 1).resized
  }
  val valueReadAddress = UInt(valueMemoryAddressWidth bits)
  valueReadAddress := ((processKey.resize(valueMemoryAddressWidth) <<
    log2Up(valueGroups)) + valueReadGroup.resize(valueMemoryAddressWidth))
    .resize(valueMemoryAddressWidth)
  val valueRead0 = valueTileMemory(0).readSync(valueReadAddress,
    valueReadEnable && !processBank)
  val valueRead1 = valueTileMemory(1).readSync(valueReadAddress,
    valueReadEnable && processBank)
  val selectedValueGroup = Mux(processBank, valueRead1, valueRead0)

  // BitNet probability/V-scale path.  One shared 16x16 DSP walks
  // the low and high limbs of the Q16.16 scale while the narrow V BRAM is
  // being read.  Four UQ3.15 effective factors are therefore ready before the
  // first AXPY phase without adding a key-level scheduling bubble.
  private val effectiveFactorWidth = cfg.continuous.factorWidth
  private val factorProductWidth = a.expWidth + 16
  val effectiveFactorRaw = Vec(
    Reg(UInt(effectiveFactorWidth bits)) init 0, heads)
  val factorScaleActive = RegInit(False)
  val factorScaleHead = Reg(UInt(2 bits)) init 0
  val factorScalePhase = Reg(UInt(2 bits)) init 0
  val factorLowProduct = Reg(UInt(factorProductWidth bits)) init 0
  val factorHighProduct = Reg(UInt(factorProductWidth bits)) init 0
  val factorScaleLimb = UInt(16 bits)
  factorScaleLimb := currentValueScaleQ16(15 downto 0)
  when(factorScalePhase === 1) {
    factorScaleLimb := currentValueScaleQ16(31 downto 16)
  }
  val factorLimbProduct = UInt(factorProductWidth bits)
  factorLimbProduct :=
    (probabilityWeight(processBank.asUInt)(processKey)(factorScaleHead) *
      factorScaleLimb).resize(factorProductWidth)
  factorLimbProduct.addAttribute("use_dsp", "yes")
  val factorCombined = UInt(49 bits)
  factorCombined := factorLowProduct.resize(49) +
    (factorHighProduct.resize(49) << 16).resize(49)
  val factorRounded = UInt(49 bits)
  factorRounded := ((factorCombined + U(1 << 15, 49 bits)) >> 16).resize(49)
  val factorOverflow = factorRounded >
    U((BigInt(1) << effectiveFactorWidth) - 1, 49 bits)

  private def saturateSigned(value: SInt, targetWidth: Int): SInt = {
    val result = SInt(targetWidth bits)
    val maximum = (BigInt(1) << (targetWidth - 1)) - 1
    val minimum = -(BigInt(1) << (targetWidth - 1))
    when(value > S(maximum, value.getWidth bits)) {
      result := S(maximum, targetWidth bits)
    } elsewhen (value < S(minimum, value.getWidth bits)) {
      result := S(minimum, targetWidth bits)
    } otherwise {
      result := value.resize(targetWidth)
    }
    result
  }

  private def saturateS48(value: SInt): SInt =
    saturateSigned(value, cfg.contextWidth)

  /** Reconstruct low24 + (signedHigh24 << 24) before the single Q15 round. */
  private def combineScaleProducts(low: SInt, high: SInt): SInt =
    (low.resize(scaleProductWidth) +
      (high.resize(scaleProductWidth) << 24).resize(scaleProductWidth))
      .resize(scaleProductWidth)

  private def roundScaleCombined(combined: SInt): SInt = {
    val wide = combined.resize(scaleProductWidth + 1)
    val magnitude = UInt((scaleProductWidth + 1) bits)
    magnitude := Mux(wide.msb, (-wide).asUInt, wide.asUInt)
    val roundedMagnitude =
      ((magnitude + U(1 << 14, magnitude.getWidth bits)) >> 15)
        .resize(scaleProductWidth + 1)
    val rounded = SInt((scaleProductWidth + 1) bits)
    rounded := Mux(wide.msb, -roundedMagnitude.asSInt,
      roundedMagnitude.asSInt)
    saturateS48(rounded)
  }

  // One synchronous context read port is shared by scale, AXPY and output.
  // A following group is issued while the current group is consumed, so the
  // grouped BRAM contributes one startup cycle but no per-group bubble.
  val outputHead = Reg(UInt(2 bits)) init 0
  val outputGroup = Reg(UInt(log2Up(contextGroups) bits)) init 0
  val outputLane = Reg(UInt(log2Up(cfg.dequantLanes) bits)) init 0
  val outputGroupBits = Reg(Bits(contextGroupWidth bits)) init 0

  val contextReadEnable = Bool()
  val contextReadAddress = UInt(contextAddressWidth bits)
  contextReadEnable := False
  contextReadAddress := 0
  when(state === State.ScaleLoadIssue) {
    contextReadEnable := True
    contextReadAddress := contextAddress(scaleHead,
      U(0, log2Up(contextGroups) bits))
  } elsewhen (state === State.ScaleLoadCapture &&
      contextGroup =/= contextGroups - 1) {
    contextReadEnable := True
    contextReadAddress := contextAddress(scaleHead,
      (contextGroup + 1).resized)
  } elsewhen (state === State.AxpyRun && io.svResult.fire) {
    contextReadEnable := True
    contextReadAddress := contextAddress(axpyHead,
      io.svResult.groupIndex.resized)
  } elsewhen (state === State.EmitLoadIssue) {
    contextReadEnable := True
    contextReadAddress := contextAddress(outputHead, outputGroup)
  }
  val contextReadData = contextMemory.readSync(contextReadAddress,
    enable = contextReadEnable)

  val scaleLowWrite = state === State.ScaleRun && io.svResult.fire &&
    !scaleLowSeen && io.svResult.svLimbShift === 0
  scaleLowProductMemory.write(io.svResult.groupIndex.resized,
    io.svResult.rawP.asBits, enable = scaleLowWrite)
  val scaleLowRead = state === State.ScaleRun && io.svResult.fire &&
    scaleLowSeen && io.svResult.svLimbShift === 24
  val scaleLowReadData = scaleLowProductMemory.readSync(
    io.svResult.groupIndex.resized, enable = scaleLowRead)

  val scaleCombinedNext = Vec(SInt(scaleProductWidth bits), cfg.dequantLanes)
  val scaleRoundedValues = Vec(SInt(cfg.contextWidth bits), cfg.dequantLanes)
  val axpyDeltaNext = Vec(SInt((cfg.contextWidth + 1) bits), cfg.dequantLanes)
  val axpySummedValues = Vec(SInt(cfg.contextWidth bits), cfg.dequantLanes)
  val selectedScaleLowGroup = scaleLowReadData
  val selectedScaleHighGroup = scaleHighProductGroupReg
  val selectedAxpyProductGroup = axpyProductGroupReg
  for (lane <- 0 until cfg.dequantLanes) {
    val low = selectedScaleLowGroup((lane + 1) * cfg.contextWidth - 1 downto
      lane * cfg.contextWidth).asSInt
    val high = selectedScaleHighGroup((lane + 1) * cfg.contextWidth - 1 downto
      lane * cfg.contextWidth).asSInt
    scaleCombinedNext(lane) := combineScaleProducts(low, high)
    scaleRoundedValues(lane) := roundScaleCombined(scaleCombined(lane))

    val rawProduct = selectedAxpyProductGroup(
      (lane + 1) * cfg.contextWidth - 1 downto
      lane * cfg.contextWidth).asSInt
    // raw V * UQ3.15 factor is Q15.  The resident context is Q16.16, hence
    // the exact conversion is a one-bit left shift (no second rounding).
    axpyDeltaNext(lane) :=
      (rawProduct.resize(cfg.contextWidth + 1) << 1)
        .resize(cfg.contextWidth + 1)
    val oldContext = axpyContextGroupReg(
      (lane + 1) * cfg.contextWidth - 1 downto
        lane * cfg.contextWidth).asSInt
    val sum = (oldContext.resize(cfg.contextWidth + 1) +
      axpyDeltaReg(lane)).resize(cfg.contextWidth + 1)
    axpySummedValues(lane) := saturateS48(sum)
  }

  val contextWriteEnable = Bool()
  val contextWriteAddress = UInt(contextAddressWidth bits)
  val contextWriteData = Bits(contextGroupWidth bits)
  contextWriteEnable := contextClearActive
  contextWriteAddress := contextClearAddress
  contextWriteData := B(0, contextGroupWidth bits)
  when(state === State.ScaleCommit) {
    contextWriteEnable := True
    contextWriteAddress := contextAddress(scaleHead, contextGroup)
    contextWriteData := scaleWriteDataReg
  } elsewhen (state === State.AxpyContextCommit) {
    contextWriteEnable := True
    contextWriteAddress := contextAddress(axpyHead, contextGroup)
    contextWriteData := axpyWriteDataReg
  }
  contextMemory.write(contextWriteAddress, contextWriteData,
    enable = contextWriteEnable)

  val activeAxpy = axpyJobActive || state === State.AxpyJob
  val normalizeFormat = state === State.NormalizeJob ||
    (scaleNormalizing && state =/= State.ScaleJob)
  val scheduledMode = BitNetUnifiedMatrixMode()
  val scheduledFormat = BitNetUnifiedOperandFormat()
  scheduledMode := BitNetUnifiedMatrixMode.SvContextScaleQ15
  scheduledFormat := BitNetUnifiedOperandFormat.Q15TileAlpha
  when(activeAxpy) {
    scheduledMode := BitNetUnifiedMatrixMode.SvAxpyQ15
    scheduledFormat := BitNetUnifiedOperandFormat.Q15Probability
  } elsewhen (normalizeFormat) {
    scheduledFormat := BitNetUnifiedOperandFormat.Q15FinalReciprocal
  }

  io.svSequence.valid := state === State.ScaleJob || state === State.AxpyJob ||
    state === State.NormalizeJob
  io.svSequence.mode := scheduledMode
  io.svSequence.format := scheduledFormat
  io.svSequence.sequenceId := sequenceCounter
  io.svSequence.expectedBeats := heads * 2
  when(state === State.AxpyJob) {
    io.svSequence.expectedBeats :=
      (tileKeyCount(processBank.asUInt)
        .resize(cfg.continuous.beatCountWidth) * heads)
        .resize(cfg.continuous.beatCountWidth)
  }
  io.svSequence.activationFactorQ16.foreach(_ := 0)
  io.svSequence.weightScaleQ16 := 0
  io.svSequence.outputFeatureCount := 0
  io.svSequence.routeFinalToDynamicQuant := False
  io.svSequence.residualEnable := False
  when(io.svSequence.fire) {
    activeSequenceId := sequenceCounter
    sequenceCounter := sequenceCounter + 1
  }

  val inScaleRun = state === State.ScaleRun
  val inRunState = inScaleRun || state === State.AxpyRun
  io.svPhase.valid := Mux(inScaleRun, scaleLimbIssued < 2,
    state === State.AxpyRun && !axpyPhaseIssued)
  io.svPhase.tokenId := activeTokenId
  io.svPhase.taskId := activeSequenceId.resized
  io.svPhase.contextBank := activeKvHead.resized
  io.svPhase.contextSlot := Mux(inScaleRun, scaleHead, axpyHead).resized
  io.svPhase.headLane := Mux(inScaleRun, scaleHead, axpyHead)
  io.svPhase.svLimbShift := Mux(inScaleRun && scaleLimbIssued === 1,
    U(24, 5 bits), U(0, 5 bits))
  io.svPhase.svFactorRaw :=
    tileAlpha(processBank.asUInt)(scaleHead).resize(
      cfg.continuous.factorWidth).asBits
  io.svPhase.svFactorSigned := False
  io.svPhase.svFactorFractionBits := 15
  when(state === State.AxpyRun) {
    io.svPhase.svFactorRaw := effectiveFactorRaw(axpyHead).asBits
  } elsewhen (scaleNormalizing) {
    io.svPhase.svFactorRaw := reciprocalQ15(scaleHead)
      .resize(cfg.continuous.factorWidth).asBits
  }
  for (feature <- 0 until a.headDim) {
    val group = feature / cfg.dequantLanes
    val lane = feature % cfg.dequantLanes
    val selectedContext = contextVectorGroups(group)(
      (lane + 1) * cfg.contextWidth - 1 downto
        lane * cfg.contextWidth).asSInt
    val low24 = selectedContext.asBits(23 downto 0).asUInt
      .resize(cfg.matrix.dspAWidth).asSInt
    val high24 = (selectedContext >> 24).resize(cfg.matrix.dspAWidth)
    io.svPhase.svValue(feature) := low24
    when(inScaleRun && scaleLimbIssued === 1) {
      io.svPhase.svValue(feature) := high24
    } elsewhen (state === State.AxpyRun) {
      io.svPhase.svValue(feature) :=
        rawValue(feature).resize(cfg.matrix.dspAWidth)
    }
  }

  when(io.svPhase.fire) {
    submittedPhaseCount := submittedPhaseCount + 1
    when(inScaleRun) {
      scaleLimbIssued := scaleLimbIssued + 1
    } otherwise {
      axpyPhaseIssued := True
    }
  }

  val commonResultMatches = inRunState &&
    io.svResult.mode === scheduledMode &&
    io.svResult.format === scheduledFormat &&
    io.svResult.sequenceId === activeSequenceId &&
    io.svResult.taskId === activeSequenceId.resized &&
    io.svResult.tokenId === activeTokenId &&
    io.svResult.contextBank === activeKvHead.resized &&
    io.svResult.contextSlot === io.svResult.headLane &&
    !io.svResult.svFactorSigned &&
    io.svResult.svFactorFractionBits === 15
  val resultGroupMatches =
    io.svResult.groupIndex === contextGroup &&
    io.svResult.first === (contextGroup === 0) &&
    io.svResult.last === (contextGroup === contextGroups - 1)
  val scaleResultMatches = commonResultMatches && inScaleRun &&
    io.svResult.headLane === scaleHead &&
    (io.svResult.svLimbShift === 0 || io.svResult.svLimbShift === 24) &&
    resultGroupMatches
  val axpyResultMatches = commonResultMatches && state === State.AxpyRun &&
    axpyPhaseIssued && io.svResult.headLane === axpyHead &&
    io.svResult.svLimbShift === 0 && resultGroupMatches
  io.svResult.ready := inRunState

  when(io.svResult.fire) {
    when(scaleResultMatches && io.svResult.svLimbShift === 0 && !scaleLowSeen) {
      when(contextGroup === contextGroups - 1) {
        scaleLowSeen := True
        contextGroup := 0
      } otherwise {
        contextGroup := contextGroup + 1
      }
    } elsewhen (scaleResultMatches && io.svResult.svLimbShift === 24 &&
        scaleLowSeen) {
      scaleHighProductGroupReg := io.svResult.rawP.asBits
      state := State.ScaleWrite
    } elsewhen (axpyResultMatches) {
      axpyProductGroupReg := io.svResult.rawP.asBits
      state := State.AxpyContextDelta
    } otherwise {
      fault := True
    }
  }

  // ---------------------------------------------------------------------
  // Normalization latch and iterative reciprocal.  No divider/DSP is
  // replicated per head: one 31-step restoring divider is shared by GQA4.
  // ---------------------------------------------------------------------
  val normalizationHeld = RegInit(False)
  val denominator = Vec(Reg(UInt(a.sumWidth bits)) init 0, heads)
  val finalAllMasked = Reg(Bits(heads bits)) init 0
  io.normalization.ready := commandActive && !normalizationHeld
  when(io.normalization.fire) {
    normalizationHeld := True
    finalAllMasked := io.normalization.allMasked
    for (head <- 0 until heads) denominator(head) :=
      io.normalization.denominator(head)
    when(io.normalization.queryMask =/= activeQueryMask) { fault := True }
  }

  val reciprocalHead = Reg(UInt(2 bits)) init 0
  val divideBit = Reg(UInt(5 bits)) init 30
  val divideDividend = Reg(UInt(31 bits)) init 0
  val divideDivisor = Reg(UInt((a.sumWidth + 1) bits)) init 0
  val divideRemainder = Reg(UInt((a.sumWidth + 1) bits)) init 0
  val divideQuotient = Reg(UInt(31 bits)) init 0

  val shiftedRemainder = UInt((a.sumWidth + 1) bits)
  shiftedRemainder := (divideRemainder << 1).resize(a.sumWidth + 1)
  shiftedRemainder(0) := divideDividend(divideBit)
  val subtractDivider = shiftedRemainder >= divideDivisor
  val nextRemainder = UInt((a.sumWidth + 1) bits)
  nextRemainder := Mux(subtractDivider,
    shiftedRemainder - divideDivisor, shiftedRemainder)
  val nextQuotient = UInt(31 bits)
  nextQuotient := divideQuotient
  nextQuotient(divideBit) := subtractDivider

  // Scalar output walker over the grouped context BRAM.
  val outputFeature = (outputGroup ## outputLane).asUInt
    .resize(cfg.featureWidth)
  val selectedOutputContext = Vec((0 until cfg.dequantLanes).map { lane =>
    outputGroupBits((lane + 1) * cfg.contextWidth - 1 downto
      lane * cfg.contextWidth).asSInt
  })(outputLane)
  io.contextOut.valid := state === State.EmitContext
  io.contextOut.tokenId := activeTokenId
  io.contextOut.kvHead := activeKvHead
  io.contextOut.queryHeadLane := outputHead
  io.contextOut.feature := outputFeature
  io.contextOut.masked := finalAllMasked(outputHead)
  io.contextOut.valueQ16 := Mux(finalAllMasked(outputHead),
    S(0, cfg.contextWidth bits), selectedOutputContext)
  io.contextOut.first := outputHead === 0 && outputFeature === 0
  io.contextOut.last := outputHead === heads - 1 &&
    outputFeature === a.headDim - 1

  // Three-cycle/head scaler runs independently of the value-BRAM walker:
  // low limb multiply, high limb multiply, combine/round/saturate.
  when(factorScaleActive) {
    switch(factorScalePhase) {
      is(U(0, 2 bits)) {
        factorLowProduct := factorLimbProduct
        factorScalePhase := 1
      }
      is(U(1, 2 bits)) {
        factorHighProduct := factorLimbProduct
        factorScalePhase := 2
      }
      default {
        when(factorOverflow) {
          effectiveFactorRaw(factorScaleHead) :=
            U((BigInt(1) << effectiveFactorWidth) - 1,
              effectiveFactorWidth bits)
          fault := True
        } otherwise {
          effectiveFactorRaw(factorScaleHead) :=
            factorRounded.resize(effectiveFactorWidth)
        }
        factorScalePhase := 0
        when(factorScaleHead === heads - 1) {
          factorScaleActive := False
        } otherwise {
          factorScaleHead := factorScaleHead + 1
        }
      }
    }
  }

  // ---------------------------------------------------------------------
  // Processing state machine.  Jobs remain mode locked while BRAM groups are
  // loaded or committed; score/probability/context never spill to DDR.
  // ---------------------------------------------------------------------
  switch(state) {
    is(State.Idle) {}

    is(State.WaitTile) {
      when(processTileReady) {
        scaleNormalizing := False
        state := State.ScaleJob
      }
    }

    is(State.ScaleJob) {
      when(io.svSequence.fire) {
        scaleHead := 0
        scaleLimbIssued := 0
        scaleLowSeen := False
        scaleNormalizing := False
        contextGroup := 0
        state := State.ScaleLoadIssue
      }
    }

    is(State.ScaleLoadIssue) {
      contextGroup := 0
      state := State.ScaleLoadCapture
    }

    is(State.ScaleLoadCapture) {
      contextVectorGroups(contextGroup) := contextReadData
      when(contextGroup === contextGroups - 1) {
        scaleLimbIssued := 0
        scaleLowSeen := False
        contextGroup := 0
        state := State.ScaleRun
      } otherwise {
        contextGroup := contextGroup + 1
      }
    }

    // Result capture moves this state to ScaleWrite.
    is(State.ScaleRun) {}

    is(State.ScaleWrite) {
      for (lane <- 0 until cfg.dequantLanes) {
        scaleCombined(lane) := scaleCombinedNext(lane)
      }
      state := State.ScaleRound
    }

    is(State.ScaleRound) {
      scaleWriteDataReg := scaleRoundedValues.asBits
      state := State.ScaleCommit
    }

    is(State.ScaleCommit) {
      when(contextGroup === contextGroups - 1) {
        scaleLowSeen := False
        scaleLimbIssued := 0
        contextGroup := 0
        when(scaleHead === heads - 1) {
          when(scaleNormalizing) {
            outputHead := 0
            outputGroup := 0
            outputLane := 0
            state := State.EmitLoadIssue
          } otherwise {
            processKey := 0
            axpyHead := 0
            axpyJobActive := False
            state := State.ValueLoad
          }
        } otherwise {
          scaleHead := scaleHead + 1
          state := State.ScaleLoadIssue
        }
      } otherwise {
        contextGroup := contextGroup + 1
        state := State.ScaleRun
      }
    }

    is(State.ValueLoad) {
      currentValueScaleQ16 :=
        probabilityValueScaleQ16(processBank.asUInt)(processKey)
      rawValueOutputShift := B(0, rawVectorWidth bits)
      valueGroup := 0
      factorScaleActive := True
      factorScaleHead := 0
      factorScalePhase := 0
      effectiveFactorRaw.foreach(_ := 0)
      state := State.ValueCapture
    }

    is(State.ValueCapture) {
      valueRawGroup := selectedValueGroup
      state := State.ValueAppend
    }

    is(State.ValueAppend) {
      rawValueOutputShift := valueRawGroup ##
        rawValueOutputShift(rawVectorWidth - 1 downto rawGroupWidth)
      when(valueGroup === valueGroups - 1) {
        state := State.ValueWaitFactor
      } otherwise {
        valueGroup := valueGroup + 1
        state := State.ValueCapture
      }
    }

    is(State.ValueWaitFactor) {
      when(!factorScaleActive) {
        axpyHead := 0
        axpyPhaseIssued := False
        when(axpyJobActive) {
          state := State.AxpyRun
        } otherwise {
          state := State.AxpyJob
        }
      }
    }

    /* The first key opens one mode-locked job; later keys resume it. */
    is(State.AxpyJob) {
      when(io.svSequence.fire) {
        axpyJobActive := True
        axpyHead := 0
        axpyPhaseIssued := False
        state := State.AxpyRun
      }
    }

    /*
     * Each key issues four independent query-head phases only after its raw V
     * vector and four effective UQ3.15 factors are resident.  The physical
     * array remains sequence-locked.
     */
    // Each grouped result capture moves this state to AxpyContextDelta.
    is(State.AxpyRun) {}

    is(State.AxpyContextDelta) {
      axpyContextGroupReg := contextReadData
      for (lane <- 0 until cfg.dequantLanes) {
        axpyDeltaReg(lane) := axpyDeltaNext(lane)
      }
      state := State.AxpyContextSum
    }

    is(State.AxpyContextSum) {
      axpyWriteDataReg := axpySummedValues.asBits
      state := State.AxpyContextCommit
    }

    is(State.AxpyContextCommit) {
      when(contextGroup === contextGroups - 1) {
        contextGroup := 0
        axpyPhaseIssued := False
        when(axpyHead === heads - 1) {
          axpyHead := 0
          when(processKey.resize(a.tileKeyCountWidth) ===
              tileKeyCount(processBank.asUInt) - 1) {
          val bank = processBank.asUInt
          tileValid(bank) := False
          tileCollected(bank) := False
          slotAllocated(bank) := B(0, keys bits)
          valueReady(bank) := B(0, keys bits)
          axpyJobActive := False
            processBank := !processBank
            when(tileLast(bank)) {
              state := State.WaitNormalization
          } otherwise {
            state := State.WaitTile
          }
          } otherwise {
            processKey := processKey + 1
            state := State.ValueLoad
          }
        } otherwise {
          axpyHead := axpyHead + 1
          state := State.AxpyRun
        }
      } otherwise {
        contextGroup := contextGroup + 1
        state := State.AxpyRun
      }
    }

    is(State.WaitNormalization) {
      when(normalizationHeld) {
        reciprocalHead := 0
        state := State.ReciprocalStart
      }
    }

    is(State.ReciprocalStart) {
      val d = denominator(reciprocalHead)
      when(finalAllMasked(reciprocalHead) || d === 0) {
        reciprocalQ15(reciprocalHead) := U(0, a.expWidth bits)
        when(reciprocalHead === heads - 1) {
          state := State.NormalizeJob
        } otherwise {
          reciprocalHead := reciprocalHead + 1
        }
      } elsewhen (d <= U(1 << 15, a.sumWidth bits)) {
        // Legal softmax streams have D>=1.0 because at least one maximum has
        // weight 0x8000.  Saturation also makes malformed tiny-D streams safe.
        reciprocalQ15(reciprocalHead) := U(1 << 15, a.expWidth bits)
        when(reciprocalHead === heads - 1) {
          state := State.NormalizeJob
        } otherwise {
          reciprocalHead := reciprocalHead + 1
        }
      } otherwise {
        val roundedNumerator =
          (U(BigInt(1) << 30, (a.sumWidth + 1) bits) +
            (d.resize(a.sumWidth + 1) >> 1)).resize(a.sumWidth + 1)
        divideDividend := roundedNumerator.resize(31)
        divideDivisor := d.resize(a.sumWidth + 1)
        divideRemainder := 0
        divideQuotient := 0
        divideBit := 30
        state := State.ReciprocalRun
      }
    }

    is(State.ReciprocalRun) {
      divideRemainder := nextRemainder
      divideQuotient := nextQuotient
      when(divideBit === 0) {
        reciprocalQ15(reciprocalHead) := Mux(nextQuotient > (1 << 15),
          U(1 << 15, a.expWidth bits), nextQuotient.resize(a.expWidth))
        when(reciprocalHead === heads - 1) {
          state := State.NormalizeJob
        } otherwise {
          reciprocalHead := reciprocalHead + 1
          state := State.ReciprocalStart
        }
      } otherwise {
        divideBit := divideBit - 1
      }
    }

    is(State.NormalizeJob) {
      when(io.svSequence.fire) {
        scaleHead := 0
        scaleLimbIssued := 0
        scaleLowSeen := False
        scaleNormalizing := True
        contextGroup := 0
        state := State.ScaleLoadIssue
      }
    }

    is(State.EmitLoadIssue) {
      state := State.EmitLoadCapture
    }

    is(State.EmitLoadCapture) {
      outputGroupBits := contextReadData
      outputLane := 0
      state := State.EmitContext
    }

    is(State.EmitContext) {
      when(io.contextOut.fire) {
        when(outputLane === cfg.dequantLanes - 1) {
          outputLane := 0
          when(outputGroup === contextGroups - 1) {
            outputGroup := 0
            when(outputHead === heads - 1) {
              completedCommandCount := completedCommandCount + 1
              commandActive := False
              normalizationHeld := False
              scaleNormalizing := False
              fillBank := False
              fillActive := False
              nextFillTile := 0
              processBank := False
              state := State.Idle
            } otherwise {
              outputHead := outputHead + 1
              state := State.EmitLoadIssue
            }
          } otherwise {
            outputGroup := outputGroup + 1
            state := State.EmitLoadIssue
          }
        } otherwise {
          outputLane := outputLane + 1
        }
      }
    }
  }
}

/** Bit-accurate software model of the coordinator's S48/Q15 contract. */
object BitNetKvContextMath {
  private val ContextMin = -(BigInt(1) << 47)
  private val ContextMax = (BigInt(1) << 47) - 1
  val EffectiveFactorMax: BigInt = (BigInt(1) << 18) - 1

  private def saturate48(value: BigInt): BigInt =
    value.max(ContextMin).min(ContextMax)

  /** Symmetric round-to-nearest, ties away from zero. */
  def roundShift15(value: BigInt): BigInt = {
    val magnitude = value.abs
    val rounded = (magnitude + (BigInt(1) << 14)) >> 15
    if (value < 0) -rounded else rounded
  }

  def scale(context: BigInt, factorQ15: Int): BigInt = {
    require(factorQ15 >= 0 && factorQ15 <= 32768)
    saturate48(roundShift15(context * factorQ15))
  }

  case class EffectiveFactor(rawUq3_15: BigInt, saturated: Boolean)

  /**
    * BitNet probability/V-scale join.
    *
    * Q1.15 * Q16.16 has 31 fractional bits; rounding right by 16 produces
    * UQ3.15 for the continuous array's unsigned 18-bit DSP-A operand.
    */
  def effectiveFactor(probabilityQ15: Int,
                      valueScaleQ16: BigInt): EffectiveFactor = {
    require(probabilityQ15 >= 0 && probabilityQ15 <= 32768)
    require(valueScaleQ16 >= 0 && valueScaleQ16 < (BigInt(1) << 32))
    val rounded = (BigInt(probabilityQ15) * valueScaleQ16 +
      (BigInt(1) << 15)) >> 16
    EffectiveFactor(rounded.min(EffectiveFactorMax),
      rounded > EffectiveFactorMax)
  }

  def axpy(contextQ16: BigInt,
           valueI8: Int,
           valueScaleQ16: BigInt,
           probabilityQ15: Int): BigInt = {
    require(probabilityQ15 >= 0 && probabilityQ15 <= 32768)
    require(valueI8 >= -128 && valueI8 <= 127)
    val factor = effectiveFactor(probabilityQ15, valueScaleQ16)
    // rawV * UQ3.15 is Q15; shift once to resident Q16.16.
    saturate48(contextQ16 + BigInt(valueI8) * factor.rawUq3_15 * 2)
  }

  /** Q15 encoding of 1/(denominatorQ15 / 2^15). */
  def reciprocalQ15(denominatorQ15: BigInt, allMasked: Boolean): Int = {
    if (allMasked || denominatorQ15 == 0) 0
    else {
      val rounded = ((BigInt(1) << 30) + denominatorQ15 / 2) /
        denominatorQ15
      rounded.min(BigInt(1) << 15).toInt
    }
  }

  case class Result(contextQ16: Vector[Vector[BigInt]],
                    reciprocal: Vector[Int])

  def compute(
      softmax: BitNetTiledAttentionMath.Result,
      valuesByKeyIndex: IndexedSeq[IndexedSeq[Int]],
      valueScalesQ16: IndexedSeq[BigInt],
      tileKeys: Int = 16
  ): Result = {
    require(valuesByKeyIndex.length == softmax.probabilities.length)
    require(valueScalesQ16.length == softmax.probabilities.length)
    require(valuesByKeyIndex.forall(_.length == 128))
    val context = Array.fill(4, 128)(BigInt(0))

    softmax.updates.zipWithIndex.foreach { case (update, tile) =>
      for (head <- 0 until 4; feature <- 0 until 128) {
        context(head)(feature) = scale(context(head)(feature),
          update.alphaQ15(head))
      }
      softmax.probabilities.slice(tile * tileKeys,
        scala.math.min((tile + 1) * tileKeys,
          softmax.probabilities.length)).foreach { probability =>
        val value = valuesByKeyIndex(probability.keyIndex)
        for (head <- 0 until 4; feature <- 0 until 128) {
          context(head)(feature) = axpy(context(head)(feature),
            value(feature), valueScalesQ16(probability.keyIndex),
            probability.weightQ15(head))
        }
      }
    }

    val reciprocal = Vector.tabulate(4)(head => reciprocalQ15(
      softmax.denominatorQ15(head), softmax.allMasked(head)))
    for (head <- 0 until 4; feature <- 0 until 128) {
      context(head)(feature) = scale(context(head)(feature), reciprocal(head))
      if (softmax.allMasked(head)) context(head)(feature) = 0
    }
    Result(context.map(_.toVector).toVector, reciprocal)
  }
}

object GenerateBitNetKvContextCoordinator extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-kv-context-coordinator")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetKvContextCoordinator())
}
