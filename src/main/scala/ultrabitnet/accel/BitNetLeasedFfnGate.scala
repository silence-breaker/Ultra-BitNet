package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Configuration for the production, arithmetic-pool-backed FFN gate.
  *
  * `strictFeatureOrder` is the resident-microprogram contract.  Prefill uses
  * semantic order 0..length-1; decode uses the fixed two-engine schedule
  * lower-group, upper-group, lower-group, upper-group.  Both schedules are
  * one-to-one and are authenticated from the existing expected ordinal, so
  * the compile-time mode elaborates no 6912-entry seen RAM.
  *
  * Generic/research compositions retain authenticated scatter by leaving the
  * flag clear.  In that mode `seenGenerationWidth` is deliberately large
  * enough that duplicate-feature authentication does not require clearing
  * 6912 entries between decode tokens.  The generation RAM is cleared once
  * after reset and again only on generation rollover.
  */
case class BitNetLeasedFfnGateConfig(
    gate: BitNetSharedSpuFfnGateConfig =
      BitNetSharedSpuFfnGateConfig(),
    arithmetic: BitNetSharedSpuArithmeticPoolConfig =
      BitNetSharedSpuArithmeticPoolConfig(),
    strictFeatureOrder: Boolean = false,
    seenGenerationWidth: Int = 32
) {
  require(gate.tokenLanes == 2 || gate.tokenLanes == 4,
    "the resident FFN gate must be TP2 or TP4")
  require(arithmetic.clientCount == BitNetSpuArithmeticClient.Count)
  require(arithmetic.multiplierCount == 8 &&
    arithmetic.operandAWidth == gate.magnitudeWidth &&
    arithmetic.operandBWidth == gate.limbWidth,
    "the leased FFN gate requires the shared eight-site 26x18 bank")
  require(arithmetic.productWidth == gate.multiplierProductWidth)
  require(arithmetic.sequenceWidth >= log2Up(gate.maxLength * 5),
    "one pool sequence value is required for each feature/phase pair")
  require(seenGenerationWidth >= 2,
    "generic scatter authentication needs a nontrivial generation counter")
}

/**
  * Production FFN gate client for [[BitNetSharedSpuArithmeticPool]].
  *
  * The block implements the exact resident Q16.16 contract
  *
  *   g = sat_s27(round_away(gateRaw * gateScale / 2^16))
  *   u = sat_s27(round_away(upRaw   * upScale   / 2^16))
  *   y = sat_i32(round_away(g > 0 ? g*g*u : 0, 2^32))
  *
  * without owning a multiplier.  One command-level FfnGate lease covers all
  * five limb beats per semantic feature and is released before RMSNorm is
  * requested.  A strict resident build accepts only the next expected feature
  * index, irrespective of the command's generic scatter capability, and owns
  * no feature-sized authentication RAM.  A generic build may be out of order
  * only when the command carries that capability; its generation RAM proves
  * that every semantic address occurs exactly once.  Range, identity,
  * first/last and inactive-lane checks remain mandatory in both modes.
  *
  * A malformed job is fail closed: no RMS command is published, the remaining
  * input frame is drained, and any earlier WideWork writes remain unpublished.
  * The old [[BitNetSharedSpuFfnGate]] remains the explicitly separate
  * research implementation with local multipliers.
  */
class BitNetLeasedFfnGate(
    cfg: BitNetLeasedFfnGateConfig =
      BitNetLeasedFfnGateConfig()) extends Component {
  private val gateCfg = cfg.gate
  private val poolCfg = cfg.arithmetic
  private val model = gateCfg.model
  private val layout = BitNetStreamingActivationLayout(model)
  private val rmsCfg = gateCfg.rmsConfig
  private val owner = BitNetSpuArithmeticClient.FfnGate
  private val sitesPerLane = 2
  private val phaseCount = 5
  private val cubicLowMidWidth =
    2 * gateCfg.limbWidth + gateCfg.magnitudeWidth

  val io = new Bundle {
    val command = slave(Stream(BitNetSharedFfnGateCommand(gateCfg)))
    val fusedInput = slave(Stream(BitNetFusedGateUpBeat(gateCfg)))

    val wideWrite = Vec(
      master(Stream(BitNetPhysicalActivationWriteBeat(model, layout))),
      gateCfg.tokenLanes)
    val rmsCommand = master(Stream(BitNetSharedRmsCommand(rmsCfg)))
    val done = master(Stream(BitNetSharedFfnGateDone(gateCfg)))

    val poolAcquire = master(Stream(
      BitNetSpuArithmeticLeaseRequest(poolCfg)))
    val poolOperand = master(Stream(
      BitNetSpuArithmeticOperandBeat(poolCfg)))
    val poolResult = slave(Stream(
      BitNetSpuArithmeticResultBeat(poolCfg)))
    val poolRelease = master(Stream(
      BitNetSpuArithmeticLeaseRelease(poolCfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
    val acceptedFrames = out UInt(32 bits)
    val completedFrames = out UInt(32 bits)
    val acceptedFeatures = out UInt(32 bits)
    val emittedFeatures = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val BootClear, Idle, Acquire, AcceptFeature, CheckFeature,
        IssueGateScale, WaitGateScale, RoundGateScale,
        IssueUpScale, WaitUpScale, RoundUpScale,
        IssueSquare, WaitSquare,
        IssueCubicLowMid, WaitCubicLowMid,
        IssueCubicHigh, WaitCubicHigh, RoundCubic, Saturate,
        WriteWide, FaultDrain, Release, IssueRms, Complete = newElement()
  }
  val state = Reg(State()) init (
    if (cfg.strictFeatureOrder) State.Idle else State.BootClear)

  val lengthReg = Reg(UInt(gateCfg.lengthWidth bits)) init 0
  val tokenMaskReg = Reg(Bits(gateCfg.tokenLanes bits)) init 0
  val modeReg = Reg(BitNetMode()) init BitNetMode.Prefill
  val authenticatedScatterReg = RegInit(False)
  val commandTagReg = Reg(UInt(gateCfg.tagWidth bits)) init 0
  val gateScaleReg = Vec(
    Reg(UInt(gateCfg.scaleWidth bits)) init 0, gateCfg.tokenLanes)
  val upScaleReg = Vec(
    Reg(UInt(gateCfg.scaleWidth bits)) init 0, gateCfg.tokenLanes)

  val featureOrdinal = Reg(UInt(gateCfg.indexWidth bits)) init 0
  val featureAddress = Reg(UInt(gateCfg.indexWidth bits)) init 0
  val heldFeature = Reg(BitNetFusedGateUpBeat(gateCfg)) init (
    BitNetFusedGateUpBeat(gateCfg).getZero)

  val activeLeaseTag = Reg(UInt(poolCfg.tagWidth bits)) init 0
  val activeLeaseEpoch = Reg(UInt(poolCfg.epochWidth bits)) init 0
  val nextLeaseTag = Reg(UInt(poolCfg.tagWidth bits)) init 0
  val nextLeaseEpoch = Reg(UInt(poolCfg.epochWidth bits)) init 0

  val globalProtocolFault = RegInit(False)
  val globalOverflow = RegInit(False)
  val jobProtocolFault = RegInit(False)
  val jobOverflow = RegInit(False)
  val rmsCommandIssued = RegInit(False)
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

  /* ------------------------------------------------------------------
   * Feature-order authentication.
   *
   * These scalar generation registers are retained as a common control shape
   * so the generic mode can scrub on rollover.  The feature-sized memory
   * itself exists only in the generic elaboration branch below.
   * ------------------------------------------------------------------ */
  val clearAddress = Reg(UInt(gateCfg.indexWidth bits)) init 0
  val activeGeneration = Reg(UInt(cfg.seenGenerationWidth bits)) init 1
  val nextGeneration = Reg(UInt(cfg.seenGenerationWidth bits)) init 1
  val clearAfterJob = RegInit(False)

  val expectedLastFeature = featureOrdinal.resize(gateCfg.lengthWidth) ===
    (lengthReg - 1).resized
  val heldAddressLegal = heldFeature.featureIndex.resize(
    gateCfg.lengthWidth) < lengthReg
  val strictExpectedAddress = UInt(gateCfg.indexWidth bits)
  strictExpectedAddress := featureOrdinal
  val strictOrdinalGroup = (featureOrdinal >> 2).resize(
    gateCfg.indexWidth)
  val strictDecodeGroup = UInt(gateCfg.indexWidth bits)
  strictDecodeGroup := (strictOrdinalGroup >> 1).resized
  when(strictOrdinalGroup(0)) {
    strictDecodeGroup := ((strictOrdinalGroup >> 1) +
      (lengthReg >> 3).resize(gateCfg.indexWidth)).resized
  }
  when(modeReg === BitNetMode.Decode && authenticatedScatterReg) {
    strictExpectedAddress := ((strictDecodeGroup << 2) +
      featureOrdinal(1 downto 0)).resized
  }
  val heldOrderLegal = Bool()
  if (cfg.strictFeatureOrder) {
    heldOrderLegal := heldFeature.featureIndex === strictExpectedAddress
  } else {
    heldOrderLegal := heldFeature.featureIndex === featureOrdinal
    when(authenticatedScatterReg) { heldOrderLegal := True }
  }
  val heldInactiveLanesZero = Bool()
  heldInactiveLanesZero := True
  for (lane <- 0 until gateCfg.tokenLanes) {
    when(!tokenMaskReg(lane) &&
        (heldFeature.gateRaw(lane) =/= 0 ||
          heldFeature.upRaw(lane) =/= 0)) {
      heldInactiveLanesZero := False
    }
  }
  val heldBaseProtocolLegal =
    heldFeature.tokenMask === tokenMaskReg &&
      heldAddressLegal && heldOrderLegal && heldInactiveLanesZero &&
      heldFeature.first === (featureOrdinal === 0) &&
      heldFeature.last === expectedLastFeature
  val heldFeatureNotDuplicate = Bool()
  val heldProtocolLegal = heldBaseProtocolLegal && heldFeatureNotDuplicate

  if (cfg.strictFeatureOrder) {
    heldFeatureNotDuplicate := True
  } else {
    val seenMemory = Mem(UInt(cfg.seenGenerationWidth bits),
      gateCfg.maxLength)
    seenMemory.addAttribute("ram_style", "block")
    val incomingWithinMemory =
      io.fusedInput.payload.featureIndex < gateCfg.maxLength
    val safeReadAddress = Mux(incomingWithinMemory,
      io.fusedInput.payload.featureIndex,
      U(0, gateCfg.indexWidth bits))
    val seenReadEnable = state === State.AcceptFeature && io.fusedInput.fire
    val seenReadData = seenMemory.readSync(safeReadAddress, seenReadEnable)
    heldFeatureNotDuplicate := seenReadData =/= activeGeneration

    val seenWriteEnable =
      state === State.BootClear ||
        (state === State.CheckFeature && heldProtocolLegal)
    val seenWriteAddress = UInt(gateCfg.indexWidth bits)
    val seenWriteData = UInt(cfg.seenGenerationWidth bits)
    seenWriteAddress := clearAddress
    seenWriteData := 0
    when(state === State.CheckFeature) {
      seenWriteAddress := featureAddress
      seenWriteData := activeGeneration
    }
    seenMemory.write(seenWriteAddress, seenWriteData, seenWriteEnable)
  }

  /* ------------------------------------------------------------------
   * Signed magnitudes and registered product reconstructions.
   * ------------------------------------------------------------------ */
  def rawMagnitude(value: SInt): UInt = {
    val extended = value.resize(gateCfg.dataWidth + 1)
    Mux(value.msb, (-extended).asUInt, extended.asUInt)
      .resize(gateCfg.rawMagnitudeWidth)
  }

  val rawGateMagnitude = Vec(
    Reg(UInt(gateCfg.rawMagnitudeWidth bits)) init 0,
    gateCfg.tokenLanes)
  val rawUpMagnitude = Vec(
    Reg(UInt(gateCfg.rawMagnitudeWidth bits)) init 0,
    gateCfg.tokenLanes)
  val rawGatePositive = Reg(Bits(gateCfg.tokenLanes bits)) init 0
  val rawUpNegative = Reg(Bits(gateCfg.tokenLanes bits)) init 0

  val gateMagnitude = Vec(
    Reg(UInt(gateCfg.magnitudeWidth bits)) init 0,
    gateCfg.tokenLanes)
  val upMagnitude = Vec(
    Reg(UInt(gateCfg.magnitudeWidth bits)) init 0,
    gateCfg.tokenLanes)
  val gateEnabled = Reg(Bits(gateCfg.tokenLanes bits)) init 0
  val upNegative = Reg(Bits(gateCfg.tokenLanes bits)) init 0
  val scaleProductReg = Vec(
    Reg(UInt(gateCfg.scaleProductWidth bits)) init 0,
    gateCfg.tokenLanes)
  val squareReg = Vec(
    Reg(UInt(gateCfg.squareWidth bits)) init 0,
    gateCfg.tokenLanes)
  val cubicLowMidReg = Vec(
    Reg(UInt(cubicLowMidWidth bits)) init 0,
    gateCfg.tokenLanes)
  val cubicProductReg = Vec(
    Reg(UInt(gateCfg.cubicWidth bits)) init 0,
    gateCfg.tokenLanes)
  val cubicRoundedReg = Vec(
    Reg(UInt(gateCfg.roundedCubicWidth bits)) init 0,
    gateCfg.tokenLanes)
  val resultReg = Vec(
    Reg(SInt(gateCfg.dataWidth bits)) init 0,
    gateCfg.tokenLanes)

  def responseProduct0(lane: Int): UInt =
    io.poolResult.product(lane * sitesPerLane)
  def responseProduct1(lane: Int): UInt =
    io.poolResult.product(lane * sitesPerLane + 1)

  val reconstructedScale = Vec(
    UInt(gateCfg.scaleProductWidth bits), gateCfg.tokenLanes)
  val reconstructedSquare = Vec(
    UInt(gateCfg.squareWidth bits), gateCfg.tokenLanes)
  val reconstructedLowMid = Vec(
    UInt(cubicLowMidWidth bits), gateCfg.tokenLanes)
  val reconstructedCubic = Vec(
    UInt(gateCfg.cubicWidth bits), gateCfg.tokenLanes)
  for (lane <- 0 until gateCfg.tokenLanes) {
    reconstructedScale(lane) :=
      (responseProduct0(lane).resize(gateCfg.scaleProductWidth) +
        (responseProduct1(lane).resize(gateCfg.scaleProductWidth) |<<
          gateCfg.limbWidth)).resize(gateCfg.scaleProductWidth)
    reconstructedSquare(lane) :=
      (responseProduct0(lane).resize(gateCfg.squareWidth) +
        (responseProduct1(lane).resize(gateCfg.squareWidth) |<<
          gateCfg.limbWidth)).resize(gateCfg.squareWidth)
    reconstructedLowMid(lane) :=
      (responseProduct0(lane).resize(cubicLowMidWidth) +
        (responseProduct1(lane).resize(cubicLowMidWidth) |<<
          gateCfg.limbWidth)).resize(cubicLowMidWidth)
    reconstructedCubic(lane) :=
      (cubicLowMidReg(lane).resize(gateCfg.cubicWidth) +
        (responseProduct0(lane).resize(gateCfg.cubicWidth) |<<
          (2 * gateCfg.limbWidth))).resize(gateCfg.cubicWidth)
  }

  val narrowLimit = U((BigInt(1) << gateCfg.magnitudeWidth) - 1,
    (gateCfg.scaleProductWidth + 1) bits)
  val scaledRounded = Vec(
    UInt((gateCfg.scaleProductWidth + 1) bits), gateCfg.tokenLanes)
  val scaledOverflow = Bits(gateCfg.tokenLanes bits)
  for (lane <- 0 until gateCfg.tokenLanes) {
    scaledRounded(lane) :=
      ((scaleProductReg(lane).resize(gateCfg.scaleProductWidth + 1) +
        U(BigInt(1) << (gateCfg.fractionalBits - 1),
          (gateCfg.scaleProductWidth + 1) bits)) >> gateCfg.fractionalBits)
        .resize(gateCfg.scaleProductWidth + 1)
    scaledOverflow(lane) := scaledRounded(lane) > narrowLimit
  }

  val roundedCubic = Vec(
    UInt(gateCfg.roundedCubicWidth bits), gateCfg.tokenLanes)
  for (lane <- 0 until gateCfg.tokenLanes) {
    roundedCubic(lane) :=
      ((cubicProductReg(lane).resize(gateCfg.cubicWidth + 1) +
        U(BigInt(1) << (2 * gateCfg.fractionalBits - 1),
          (gateCfg.cubicWidth + 1) bits)) >>
          (2 * gateCfg.fractionalBits)).resize(
            gateCfg.roundedCubicWidth)
  }

  val finalOverflow = Bits(gateCfg.tokenLanes bits)
  val finalValue = Vec(SInt(gateCfg.dataWidth bits), gateCfg.tokenLanes)
  for (lane <- 0 until gateCfg.tokenLanes) {
    val limit = UInt(gateCfg.roundedCubicWidth bits)
    limit := Mux(
      upNegative(lane),
      U(BigInt(1) << (gateCfg.dataWidth - 1),
        gateCfg.roundedCubicWidth bits),
      U((BigInt(1) << (gateCfg.dataWidth - 1)) - 1,
        gateCfg.roundedCubicWidth bits))
    val enabled = gateEnabled(lane) && tokenMaskReg(lane)
    finalOverflow(lane) := enabled && cubicRoundedReg(lane) > limit
    val saturatedMagnitude = Mux(cubicRoundedReg(lane) > limit,
      limit, cubicRoundedReg(lane)).resize(gateCfg.dataWidth + 1)
    finalValue(lane) := 0
    when(enabled) {
      when(upNegative(lane)) {
        finalValue(lane) :=
          (-saturatedMagnitude.asSInt).resize(gateCfg.dataWidth)
      } otherwise {
        finalValue(lane) :=
          saturatedMagnitude.resize(gateCfg.dataWidth).asSInt
      }
    }
  }

  /* ------------------------------------------------------------------
   * Shared-arithmetic lease and five exact limb phases.
   * ------------------------------------------------------------------ */
  def sequenceForPhase(phase: Int): UInt = {
    val ordinal = featureOrdinal.resize(poolCfg.sequenceWidth)
    ((ordinal |<< 2) + ordinal +
      U(phase, poolCfg.sequenceWidth bits)).resize(poolCfg.sequenceWidth)
  }

  val fullActiveMask = Bits(poolCfg.multiplierCount bits)
  val highActiveMask = Bits(poolCfg.multiplierCount bits)
  fullActiveMask := 0
  highActiveMask := 0
  for (lane <- 0 until gateCfg.tokenLanes) {
    fullActiveMask(lane * sitesPerLane) := tokenMaskReg(lane)
    fullActiveMask(lane * sitesPerLane + 1) := tokenMaskReg(lane)
    highActiveMask(lane * sitesPerLane) := tokenMaskReg(lane)
  }

  io.poolAcquire.valid := state === State.Acquire
  io.poolAcquire.tag := activeLeaseTag
  io.poolAcquire.epoch := activeLeaseEpoch

  val issuingOperand =
    state === State.IssueGateScale ||
      state === State.IssueUpScale ||
      state === State.IssueSquare ||
      state === State.IssueCubicLowMid ||
      state === State.IssueCubicHigh
  io.poolOperand.valid := issuingOperand
  io.poolOperand.tag := activeLeaseTag
  io.poolOperand.epoch := activeLeaseEpoch
  io.poolOperand.sequence := sequenceForPhase(0)
  io.poolOperand.active := fullActiveMask
  io.poolOperand.first := False
  io.poolOperand.last := False
  io.poolOperand.operandA.foreach(_ := 0)
  io.poolOperand.operandB.foreach(_ := 0)

  when(state === State.IssueGateScale) {
    io.poolOperand.sequence := sequenceForPhase(0)
    io.poolOperand.first := featureOrdinal === 0
    for (lane <- 0 until gateCfg.tokenLanes) {
      val lowSite = lane * sitesPerLane
      io.poolOperand.operandA(lowSite) := gateScaleReg(lane)
      io.poolOperand.operandA(lowSite + 1) := gateScaleReg(lane)
      io.poolOperand.operandB(lowSite) :=
        rawGateMagnitude(lane)(gateCfg.limbWidth - 1 downto 0)
      io.poolOperand.operandB(lowSite + 1) :=
        rawGateMagnitude(lane)(gateCfg.rawMagnitudeWidth - 1 downto
          gateCfg.limbWidth).resized
    }
  }
  when(state === State.IssueUpScale) {
    io.poolOperand.sequence := sequenceForPhase(1)
    for (lane <- 0 until gateCfg.tokenLanes) {
      val lowSite = lane * sitesPerLane
      io.poolOperand.operandA(lowSite) := upScaleReg(lane)
      io.poolOperand.operandA(lowSite + 1) := upScaleReg(lane)
      io.poolOperand.operandB(lowSite) :=
        rawUpMagnitude(lane)(gateCfg.limbWidth - 1 downto 0)
      io.poolOperand.operandB(lowSite + 1) :=
        rawUpMagnitude(lane)(gateCfg.rawMagnitudeWidth - 1 downto
          gateCfg.limbWidth).resized
    }
  }
  when(state === State.IssueSquare) {
    io.poolOperand.sequence := sequenceForPhase(2)
    for (lane <- 0 until gateCfg.tokenLanes) {
      val lowSite = lane * sitesPerLane
      io.poolOperand.operandA(lowSite) := gateMagnitude(lane)
      io.poolOperand.operandA(lowSite + 1) := gateMagnitude(lane)
      io.poolOperand.operandB(lowSite) :=
        gateMagnitude(lane)(gateCfg.limbWidth - 1 downto 0)
      io.poolOperand.operandB(lowSite + 1) :=
        gateMagnitude(lane)(gateCfg.magnitudeWidth - 1 downto
          gateCfg.limbWidth).resized
    }
  }
  when(state === State.IssueCubicLowMid) {
    io.poolOperand.sequence := sequenceForPhase(3)
    for (lane <- 0 until gateCfg.tokenLanes) {
      val lowSite = lane * sitesPerLane
      io.poolOperand.operandA(lowSite) := upMagnitude(lane)
      io.poolOperand.operandA(lowSite + 1) := upMagnitude(lane)
      io.poolOperand.operandB(lowSite) :=
        squareReg(lane)(gateCfg.limbWidth - 1 downto 0)
      io.poolOperand.operandB(lowSite + 1) :=
        squareReg(lane)(2 * gateCfg.limbWidth - 1 downto
          gateCfg.limbWidth)
    }
  }
  when(state === State.IssueCubicHigh) {
    io.poolOperand.sequence := sequenceForPhase(4)
    io.poolOperand.active := highActiveMask
    io.poolOperand.last := expectedLastFeature
    for (lane <- 0 until gateCfg.tokenLanes) {
      val lowSite = lane * sitesPerLane
      io.poolOperand.operandA(lowSite) := upMagnitude(lane)
      io.poolOperand.operandB(lowSite) :=
        squareReg(lane)(gateCfg.squareWidth - 1 downto
          2 * gateCfg.limbWidth).resized
    }
  }

  val waitingForResult =
    state === State.WaitGateScale ||
      state === State.WaitUpScale ||
      state === State.WaitSquare ||
      state === State.WaitCubicLowMid ||
      state === State.WaitCubicHigh
  io.poolResult.ready := waitingForResult

  val expectedResultSequence = UInt(poolCfg.sequenceWidth bits)
  val expectedResultActive = Bits(poolCfg.multiplierCount bits)
  val expectedResultFirst = Bool()
  val expectedResultLast = Bool()
  expectedResultSequence := sequenceForPhase(0)
  expectedResultActive := fullActiveMask
  expectedResultFirst := False
  expectedResultLast := False
  when(state === State.WaitGateScale) {
    expectedResultSequence := sequenceForPhase(0)
    expectedResultFirst := featureOrdinal === 0
  } elsewhen(state === State.WaitUpScale) {
    expectedResultSequence := sequenceForPhase(1)
  } elsewhen(state === State.WaitSquare) {
    expectedResultSequence := sequenceForPhase(2)
  } elsewhen(state === State.WaitCubicLowMid) {
    expectedResultSequence := sequenceForPhase(3)
  } elsewhen(state === State.WaitCubicHigh) {
    expectedResultSequence := sequenceForPhase(4)
    expectedResultActive := highActiveMask
    expectedResultLast := expectedLastFeature
  }
  val resultMetadataValid =
    io.poolResult.owner === owner &&
      io.poolResult.tag === activeLeaseTag &&
      io.poolResult.epoch === activeLeaseEpoch &&
      io.poolResult.sequence === expectedResultSequence &&
      io.poolResult.active === expectedResultActive &&
      io.poolResult.first === expectedResultFirst &&
      io.poolResult.last === expectedResultLast

  io.poolRelease.valid := state === State.Release
  io.poolRelease.tag := activeLeaseTag
  io.poolRelease.epoch := activeLeaseEpoch

  /* ------------------------------------------------------------------
   * WideWork scatter, RMS handoff and public completion.
   * ------------------------------------------------------------------ */
  val writePending = Reg(Bits(gateCfg.tokenLanes bits)) init 0
  val completedWrites = Bits(gateCfg.tokenLanes bits)
  for (token <- 0 until gateCfg.tokenLanes) {
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
  io.rmsCommand.payload.tag := commandTagReg

  io.done.valid := state === State.Complete
  io.done.payload.tag := commandTagReg
  io.done.payload.length := lengthReg
  io.done.payload.protocolFault := jobProtocolFault
  io.done.payload.overflow := jobOverflow
  io.done.payload.rmsCommandIssued := rmsCommandIssued

  io.command.ready := state === State.Idle
  io.fusedInput.ready :=
    state === State.AcceptFeature || state === State.FaultDrain
  io.busy := state =/= State.Idle

  val commandLegal =
    io.command.payload.length =/= 0 &&
      io.command.payload.length <= gateCfg.maxLength &&
      io.command.payload.tokenMask.orR &&
      (!io.command.payload.authenticatedScatter ||
        io.command.payload.mode =/= BitNetMode.Decode ||
        io.command.payload.length(2 downto 0) === 0)

  switch(state) {
    is(State.BootClear) {
      if (cfg.strictFeatureOrder) {
        state := State.Idle
      } else {
        when(clearAddress === gateCfg.maxLength - 1) {
          clearAddress := 0
          state := State.Idle
        } otherwise {
          clearAddress := clearAddress + 1
        }
      }
    }

    is(State.Idle) {
      when(io.command.fire) {
        acceptedFrameCounter := acceptedFrameCounter + 1
        lengthReg := io.command.payload.length
        tokenMaskReg := io.command.payload.tokenMask
        modeReg := io.command.payload.mode
        authenticatedScatterReg :=
          io.command.payload.authenticatedScatter
        commandTagReg := io.command.payload.tag
        featureOrdinal := 0
        featureAddress := 0
        jobProtocolFault := !commandLegal
        jobOverflow := False
        rmsCommandIssued := False
        clearAfterJob := False
        for (lane <- 0 until gateCfg.tokenLanes) {
          gateScaleReg(lane) := io.command.payload.gateScaleQ16(lane)
          upScaleReg(lane) := io.command.payload.upScaleQ16(lane)
        }
        when(commandLegal) {
          activeLeaseTag := nextLeaseTag
          activeLeaseEpoch := nextLeaseEpoch
          when(nextLeaseTag.andR) {
            nextLeaseTag := 0
            nextLeaseEpoch := nextLeaseEpoch + 1
          } otherwise {
            nextLeaseTag := nextLeaseTag + 1
          }
          if (!cfg.strictFeatureOrder) {
            activeGeneration := nextGeneration
            when(nextGeneration.andR) {
              nextGeneration := 1
              clearAfterJob := True
            } otherwise {
              nextGeneration := nextGeneration + 1
            }
          }
          state := State.Acquire
        } otherwise {
          globalProtocolFault := True
          state := State.Complete
        }
      }
    }

    is(State.Acquire) {
      when(io.poolAcquire.fire) {
        state := State.AcceptFeature
      }
    }

    is(State.AcceptFeature) {
      when(io.fusedInput.fire) {
        acceptedFeatureCounter := acceptedFeatureCounter + 1
        heldFeature := io.fusedInput.payload
        featureAddress := io.fusedInput.payload.featureIndex
        state := State.CheckFeature
      }
    }

    is(State.CheckFeature) {
      when(heldProtocolLegal) {
        for (lane <- 0 until gateCfg.tokenLanes) {
          rawGateMagnitude(lane) := rawMagnitude(heldFeature.gateRaw(lane))
          rawUpMagnitude(lane) := rawMagnitude(heldFeature.upRaw(lane))
          rawGatePositive(lane) := heldFeature.gateRaw(lane) > 0
          rawUpNegative(lane) := heldFeature.upRaw(lane).msb
        }
        state := State.IssueGateScale
      } otherwise {
        jobProtocolFault := True
        globalProtocolFault := True
        when(expectedLastFeature) {
          state := State.Release
        } otherwise {
          featureOrdinal := featureOrdinal + 1
          state := State.FaultDrain
        }
      }
    }

    is(State.IssueGateScale) {
      when(io.poolOperand.fire) { state := State.WaitGateScale }
    }
    is(State.IssueUpScale) {
      when(io.poolOperand.fire) { state := State.WaitUpScale }
    }
    is(State.IssueSquare) {
      when(io.poolOperand.fire) { state := State.WaitSquare }
    }
    is(State.IssueCubicLowMid) {
      when(io.poolOperand.fire) { state := State.WaitCubicLowMid }
    }
    is(State.IssueCubicHigh) {
      when(io.poolOperand.fire) { state := State.WaitCubicHigh }
    }

    is(State.RoundGateScale) {
      for (lane <- 0 until gateCfg.tokenLanes) {
        gateMagnitude(lane) := Mux(
          scaledOverflow(lane),
          U((BigInt(1) << gateCfg.magnitudeWidth) - 1,
            gateCfg.magnitudeWidth bits),
          scaledRounded(lane).resize(gateCfg.magnitudeWidth))
        gateEnabled(lane) := rawGatePositive(lane) &&
          scaledRounded(lane) =/= 0 && tokenMaskReg(lane)
      }
      when((scaledOverflow & tokenMaskReg).orR) {
        jobOverflow := True
        globalOverflow := True
      }
      state := State.IssueUpScale
    }

    is(State.RoundUpScale) {
      for (lane <- 0 until gateCfg.tokenLanes) {
        upMagnitude(lane) := Mux(
          scaledOverflow(lane),
          U((BigInt(1) << gateCfg.magnitudeWidth) - 1,
            gateCfg.magnitudeWidth bits),
          scaledRounded(lane).resize(gateCfg.magnitudeWidth))
        upNegative(lane) := rawUpNegative(lane)
      }
      when((scaledOverflow & tokenMaskReg).orR) {
        jobOverflow := True
        globalOverflow := True
      }
      state := State.IssueSquare
    }

    is(State.RoundCubic) {
      for (lane <- 0 until gateCfg.tokenLanes) {
        cubicRoundedReg(lane) := roundedCubic(lane)
      }
      state := State.Saturate
    }

    is(State.Saturate) {
      for (lane <- 0 until gateCfg.tokenLanes) {
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
          state := State.Release
        } otherwise {
          featureOrdinal := featureOrdinal + 1
          state := State.AcceptFeature
        }
      }
    }

    is(State.FaultDrain) {
      when(io.fusedInput.fire) {
        acceptedFeatureCounter := acceptedFeatureCounter + 1
        when(expectedLastFeature) {
          state := State.Release
        } otherwise {
          featureOrdinal := featureOrdinal + 1
        }
      }
    }

    is(State.Release) {
      when(io.poolRelease.fire) {
        when(jobProtocolFault) {
          state := State.Complete
        } otherwise {
          state := State.IssueRms
        }
      }
    }

    is(State.IssueRms) {
      when(io.rmsCommand.fire) {
        rmsCommandIssued := True
        state := State.Complete
      }
    }

    is(State.Complete) {
      when(io.done.fire) {
        completedFrameCounter := completedFrameCounter + 1
        if (cfg.strictFeatureOrder) {
          state := State.Idle
        } else {
          when(clearAfterJob) {
            clearAddress := 0
            state := State.BootClear
          } otherwise {
            state := State.Idle
          }
        }
      }
    }
  }

  /* A correct response after a previously detected metadata violation is
   * consumed solely to make the lease releasable.  No partial value may reach
   * WideWork.  Invalid responses themselves are consumed and leave the client
   * waiting for a correctly authenticated response. */
  when(io.poolResult.fire) {
    when(!resultMetadataValid) {
      jobProtocolFault := True
      globalProtocolFault := True
    } otherwise {
      when(jobProtocolFault) {
        when(expectedLastFeature) {
          state := State.Release
        } otherwise {
          featureOrdinal := featureOrdinal + 1
          state := State.FaultDrain
        }
      } otherwise {
        switch(state) {
          is(State.WaitGateScale) {
            for (lane <- 0 until gateCfg.tokenLanes) {
              scaleProductReg(lane) := reconstructedScale(lane)
            }
            state := State.RoundGateScale
          }
          is(State.WaitUpScale) {
            for (lane <- 0 until gateCfg.tokenLanes) {
              scaleProductReg(lane) := reconstructedScale(lane)
            }
            state := State.RoundUpScale
          }
          is(State.WaitSquare) {
            for (lane <- 0 until gateCfg.tokenLanes) {
              squareReg(lane) := reconstructedSquare(lane)
            }
            state := State.IssueCubicLowMid
          }
          is(State.WaitCubicLowMid) {
            for (lane <- 0 until gateCfg.tokenLanes) {
              cubicLowMidReg(lane) := reconstructedLowMid(lane)
            }
            state := State.IssueCubicHigh
          }
          is(State.WaitCubicHigh) {
            for (lane <- 0 until gateCfg.tokenLanes) {
              cubicProductReg(lane) := reconstructedCubic(lane)
            }
            state := State.RoundCubic
          }
          default { }
        }
      }
    }
  }
}

object GenerateBitNetLeasedFfnGate extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-leased-ffn-gate")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetLeasedFfnGate())
}
