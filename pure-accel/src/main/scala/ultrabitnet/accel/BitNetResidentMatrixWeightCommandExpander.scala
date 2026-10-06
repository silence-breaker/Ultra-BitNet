package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentMatrixWeightFault {
  val None = 0x00
  val InvalidProjection = 0x91
  val AguResponse = 0x92
  val PrefetchCompletion = 0x93
  val PrefetchFailed = 0x94
}

case class BitNetResidentMatrixWeightCommandExpanderConfig(
    stage: BitNetResidentMatrixStageLaunchExpanderConfig =
      BitNetResidentMatrixStageLaunchExpanderConfig(),
    prefetch: BitNetFiveBankStripedWeightPrefetcherConfig =
      BitNetFiveBankStripedWeightPrefetcherConfig(),
    agu: BitNetFiveBankStripedAguConfig =
      BitNetFiveBankStripedAguConfig(),
    trustGeneratedProjection: Boolean = false
) {
  require(stage.model == BitNetConfig.Production)
  require(prefetch.model == stage.model)
  require(prefetch.feeder == stage.feeder)
  require(agu.model == stage.model)
  require(agu.layout == BitNetFiveBankStripedLayout())
  require(stage.prefetchEpochWidth == prefetch.epochWidth)
  require(prefetch.poolCount == stage.model.engineCount)
}

/** Completion of every five-bank group belonging to one projection. */
case class BitNetResidentMatrixWeightRetirement(
    cfg: BitNetResidentMatrixWeightCommandExpanderConfig) extends Bundle {
  val requestId = UInt(cfg.stage.model.requestIdWidth bits)
  val stepId = UInt(
    BitNetResidentLayerGeometry.stepIdWidth(cfg.stage.model) bits)
  val stage = BitNetResidentLayerStage()
  val layer = UInt(cfg.stage.model.layerWidth bits)
  val part = BitNetResidentMatrixProjectionPart()
  val sequenceId = UInt(cfg.stage.matrix.sequenceIdWidth bits)
  val target = BitNetAddressTarget()
  val success = Bool()
  val faultCode = Bits(8 bits)
  val issuedGroups = UInt(cfg.stage.matrix.base.outputIndexWidth bits)
  val completedGroups = UInt(cfg.stage.matrix.base.outputIndexWidth bits)
}

/**
  * Converts one static projection launch into ping-pong five-bank commands.
  *
  * Two AGU requests are generated for every command.  Prefill requests the
  * same group twice and the prefetcher broadcasts pool 0.  Decode requests the
  * lower and upper output-group halves concurrently for pools 0 and 1.  The
  * two command slots form a real scoreboard; neither slot can be recycled
  * until its fully authenticated prefetch completion retires.
  */
class BitNetResidentMatrixWeightCommandExpander(
    cfg: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig()) extends Component {
  private val s = cfg.stage
  private val p = cfg.prefetch
  private val a = cfg.agu
  private val model = s.model
  private val matrix = s.matrix
  private val b = matrix.base
  private val engines = model.engineCount
  private val banks = p.bankCount

  val io = new Bundle {
    val projection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(s)))
    val aguRequest = Vec(master(Stream(
      BitNetStripedTransformerGroupRequest(a))), engines)
    val aguResponse = Vec(slave(Stream(
      BitNetStripedTransformerGroupResponse(a))), engines)
    val command = master(Stream(
      BitNetFiveBankStripedWeightCommand(p)))
    val prefetchCompletion = slave(Stream(
      BitNetFiveBankStripedWeightCompletion(p)))
    val retirement = master(Stream(
      BitNetResidentMatrixWeightRetirement(cfg)))

    val busy = out Bool()
    val slotBusy = out Bits(p.slotCount bits)
    val issuedGroups = out UInt(32 bits)
    val completedGroups = out UInt(32 bits)
    val protocolFaultSticky = out Bool()
  }

  val active = RegInit(False)
  val held = Reg(BitNetResidentMatrixProjectionLaunch(s)) init
    BitNetResidentMatrixProjectionLaunch(s).getZero
  val nextCommand = Reg(UInt(b.outputIndexWidth bits)) init 0
  val allIssued = RegInit(False)
  val roundFailed = RegInit(False)
  val roundFault = Reg(Bits(8 bits)) init 0
  val requestPending = Reg(Bits(engines bits)) init 0
  val responseValid = Reg(Bits(engines bits)) init 0
  val responseHeld = Vec(
    Reg(BitNetStripedTransformerGroupResponse(a)) init
      BitNetStripedTransformerGroupResponse(a).getZero, engines)
  val occupied = Reg(Bits(p.slotCount bits)) init 0
  val expectedGroup = Vec.fill(p.slotCount)(Vec.fill(engines)(
    Reg(UInt(b.outputIndexWidth bits)) init 0))
  val expectedOutputBase = Vec.fill(p.slotCount)(Vec.fill(engines)(
    Reg(UInt(b.outputIndexWidth bits)) init 0))
  val expectedBeatBase = Vec.fill(p.slotCount)(
    Reg(UInt(matrix.beatCountWidth bits)) init 0)
  val expectedLast = Vec.fill(p.slotCount)(RegInit(False))
  val issuedRound = Reg(UInt(b.outputIndexWidth bits)) init 0
  val completedRound = Reg(UInt(b.outputIndexWidth bits)) init 0
  val stickyFault = RegInit(False)
  val issuedLifetime = Reg(UInt(32 bits)) init 0
  val completedLifetime = Reg(UInt(32 bits)) init 0

  val retirementValid = RegInit(False)
  val retirementReg = Reg(BitNetResidentMatrixWeightRetirement(cfg)) init
    BitNetResidentMatrixWeightRetirement(cfg).getZero
  io.retirement.valid := retirementValid
  io.retirement.payload := retirementReg
  when(io.retirement.fire) { retirementValid := False }

  val projectionLegal = if (cfg.trustGeneratedProjection) {
    None
  } else {
  val expectedInputFeatures = UInt(s.inputFeatureWidth bits)
  val expectedOutputFeatures = UInt(b.outputIndexWidth bits)
  expectedInputFeatures := U(model.hiddenSize, s.inputFeatureWidth bits)
  expectedOutputFeatures := U(model.hiddenSize, b.outputIndexWidth bits)
  when(io.projection.payload.weightTarget === BitNetAddressTarget.KWeight ||
      io.projection.payload.weightTarget === BitNetAddressTarget.VWeight) {
    expectedOutputFeatures := U(
      model.kvHeadCount * model.headSize, b.outputIndexWidth bits)
  }
  when(io.projection.payload.weightTarget === BitNetAddressTarget.GateWeight ||
      io.projection.payload.weightTarget === BitNetAddressTarget.UpWeight) {
    expectedOutputFeatures := U(model.ffnSize, b.outputIndexWidth bits)
  }
  when(io.projection.payload.weightTarget ===
      BitNetAddressTarget.GateUpWeight) {
    expectedOutputFeatures := U(2 * model.ffnSize, b.outputIndexWidth bits)
  }
  when(io.projection.payload.weightTarget === BitNetAddressTarget.DownWeight) {
    expectedInputFeatures := U(model.ffnSize, s.inputFeatureWidth bits)
  }

  val partTargetLegal =
    (io.projection.payload.part === BitNetResidentMatrixProjectionPart.Query &&
      io.projection.payload.weightTarget === BitNetAddressTarget.QWeight) ||
    (io.projection.payload.part === BitNetResidentMatrixProjectionPart.Key &&
      io.projection.payload.weightTarget === BitNetAddressTarget.KWeight) ||
    (io.projection.payload.part === BitNetResidentMatrixProjectionPart.Value &&
      io.projection.payload.weightTarget === BitNetAddressTarget.VWeight) ||
    (io.projection.payload.part ===
      BitNetResidentMatrixProjectionPart.AttentionOutput &&
      io.projection.payload.weightTarget === BitNetAddressTarget.OWeight) ||
    (io.projection.payload.part === BitNetResidentMatrixProjectionPart.Gate &&
      io.projection.payload.weightTarget === BitNetAddressTarget.GateWeight) ||
    (io.projection.payload.part === BitNetResidentMatrixProjectionPart.Up &&
      io.projection.payload.weightTarget === BitNetAddressTarget.UpWeight) ||
    (io.projection.payload.part ===
      BitNetResidentMatrixProjectionPart.FusedGateUp &&
      io.projection.payload.weightTarget ===
        BitNetAddressTarget.GateUpWeight) ||
    (io.projection.payload.part === BitNetResidentMatrixProjectionPart.Down &&
      io.projection.payload.weightTarget === BitNetAddressTarget.DownWeight)
  val partEnvelopeLegal = Bool()
  partEnvelopeLegal.allowOverride
  partEnvelopeLegal := False
  switch(io.projection.payload.part) {
    is(BitNetResidentMatrixProjectionPart.Query) {
      partEnvelopeLegal :=
        io.projection.payload.stage ===
          BitNetResidentLayerStage.QkvProjection &&
        io.projection.payload.kind === BitNetResidentLinearKind.Qkv &&
        io.projection.payload.scaleTarget ===
          BitNetAddressTarget.QProjectionScale &&
        io.projection.payload.globalOutputBase === 0
    }
    is(BitNetResidentMatrixProjectionPart.Key) {
      partEnvelopeLegal :=
        io.projection.payload.stage ===
          BitNetResidentLayerStage.QkvProjection &&
        io.projection.payload.kind === BitNetResidentLinearKind.Qkv &&
        io.projection.payload.scaleTarget ===
          BitNetAddressTarget.KProjectionScale &&
        io.projection.payload.globalOutputBase === model.hiddenSize
    }
    is(BitNetResidentMatrixProjectionPart.Value) {
      partEnvelopeLegal :=
        io.projection.payload.stage ===
          BitNetResidentLayerStage.QkvProjection &&
        io.projection.payload.kind === BitNetResidentLinearKind.Qkv &&
        io.projection.payload.scaleTarget ===
          BitNetAddressTarget.VProjectionScale &&
        io.projection.payload.globalOutputBase ===
          model.hiddenSize + model.kvHeadCount * model.headSize
    }
    is(BitNetResidentMatrixProjectionPart.AttentionOutput) {
      partEnvelopeLegal :=
        io.projection.payload.stage ===
          BitNetResidentLayerStage.AttentionOutputProjection &&
        io.projection.payload.kind ===
          BitNetResidentLinearKind.AttentionOutput &&
        io.projection.payload.scaleTarget ===
          BitNetAddressTarget.OProjectionScale &&
        io.projection.payload.globalOutputBase === 0
    }
    is(BitNetResidentMatrixProjectionPart.Gate) {
      partEnvelopeLegal :=
        io.projection.payload.stage ===
          BitNetResidentLayerStage.GateUpProjection &&
        io.projection.payload.kind === BitNetResidentLinearKind.GateUp &&
        io.projection.payload.scaleTarget ===
          BitNetAddressTarget.GateProjectionScale &&
        io.projection.payload.globalOutputBase === 0
    }
    is(BitNetResidentMatrixProjectionPart.Up) {
      partEnvelopeLegal :=
        io.projection.payload.stage ===
          BitNetResidentLayerStage.GateUpProjection &&
        io.projection.payload.kind === BitNetResidentLinearKind.GateUp &&
        io.projection.payload.scaleTarget ===
          BitNetAddressTarget.UpProjectionScale &&
        io.projection.payload.globalOutputBase === 0
    }
    is(BitNetResidentMatrixProjectionPart.FusedGateUp) {
      partEnvelopeLegal :=
        io.projection.payload.stage ===
          BitNetResidentLayerStage.GateUpProjection &&
        io.projection.payload.kind === BitNetResidentLinearKind.GateUp &&
        io.projection.payload.scaleTarget ===
          BitNetAddressTarget.GateProjectionScale &&
        io.projection.payload.hasSecondaryScale &&
        io.projection.payload.secondaryScaleTarget ===
          BitNetAddressTarget.UpProjectionScale &&
        io.projection.payload.globalOutputBase === 0
    }
    is(BitNetResidentMatrixProjectionPart.Down) {
      partEnvelopeLegal :=
        io.projection.payload.stage ===
          BitNetResidentLayerStage.DownProjection &&
        io.projection.payload.kind === BitNetResidentLinearKind.Down &&
        io.projection.payload.scaleTarget ===
          BitNetAddressTarget.DownProjectionScale &&
        io.projection.payload.globalOutputBase === 0
    }
  }
  val expectedGroups = UInt(b.outputIndexWidth bits)
  expectedGroups := (io.projection.payload.outputFeatures >> 2).resized
  when(io.projection.payload.mode === BitNetMode.Decode) {
    expectedGroups := (io.projection.payload.outputFeatures >>
      log2Up(4 * engines)).resized
  }
  val expectedK = (io.projection.payload.inputFeatures >> 7).resize(
    p.feeder.kTileCountWidth)
  val expectedBeats = (
    expectedGroups.resize(matrix.beatCountWidth) *
      expectedK.resize(matrix.beatCountWidth) * 2).resize(
        matrix.beatCountWidth)
  val prefillTaskRangesLegal = (0 until engines).map { engine =>
    io.projection.payload.matrixTask.outputBase(engine) === 0 &&
      io.projection.payload.matrixTask.outputEndExclusive(engine) ===
        io.projection.payload.outputFeatures
  }.reduce(_ && _)
  val decodeTaskRangesLegal = if (engines == 1) {
    io.projection.payload.matrixTask.outputBase(0) === 0 &&
      io.projection.payload.matrixTask.outputEndExclusive(0) ===
        io.projection.payload.outputFeatures
  } else {
    io.projection.payload.matrixTask.outputBase(0) === 0 &&
      io.projection.payload.matrixTask.outputEndExclusive(0) ===
        (io.projection.payload.outputFeatures >> 1).resized &&
      io.projection.payload.matrixTask.outputBase(1) ===
        (io.projection.payload.outputFeatures >> 1).resized &&
      io.projection.payload.matrixTask.outputEndExclusive(1) ===
        io.projection.payload.outputFeatures
  }
  val taskRangesLegal = Mux(
    io.projection.payload.mode === BitNetMode.Prefill,
    prefillTaskRangesLegal, decodeTaskRangesLegal)
    Some(partTargetLegal && partEnvelopeLegal &&
      io.projection.payload.layer < model.layerCount &&
      io.projection.payload.inputFeatures === expectedInputFeatures &&
      io.projection.payload.outputFeatures === expectedOutputFeatures &&
      io.projection.payload.groupsPerEngine === expectedGroups &&
      expectedGroups =/= 0 &&
      io.projection.payload.matrixTask.scheduleMode ===
        io.projection.payload.mode &&
      io.projection.payload.matrixTask.operation ===
        BitNetContinuousDotKind.Projection &&
      io.projection.payload.matrixTask.sequenceId ===
        io.projection.payload.sequenceId &&
      io.projection.payload.matrixTask.activeTokenMask ===
        io.projection.payload.tokenMask &&
      io.projection.payload.matrixTask.kTilesPerOutput === expectedK &&
      io.projection.payload.matrixTask.expectedBeats === expectedBeats &&
      !io.projection.payload.matrixTask.routeFinalToDynamicQuant &&
      !io.projection.payload.matrixTask.residualEnable && taskRangesLegal)
  }

  io.projection.ready := !active && !retirementValid
  when(io.projection.fire) {
    held := io.projection.payload
    nextCommand := 0
    allIssued := False
    roundFailed := False
    roundFault := 0
    requestPending := B((BigInt(1) << engines) - 1, engines bits)
    responseValid := 0
    occupied := 0
    issuedRound := 0
    completedRound := 0
    if (cfg.trustGeneratedProjection) {
      active := True
    } else {
      when(projectionLegal.get) {
        active := True
      } otherwise {
        retirementReg.requestId := io.projection.payload.requestId
        retirementReg.stepId := io.projection.payload.stepId
        retirementReg.stage := io.projection.payload.stage
        retirementReg.layer := io.projection.payload.layer
        retirementReg.part := io.projection.payload.part
        retirementReg.sequenceId := io.projection.payload.sequenceId
        retirementReg.target := io.projection.payload.weightTarget
        retirementReg.success := False
        retirementReg.faultCode := B(
          BitNetResidentMatrixWeightFault.InvalidProjection, 8 bits)
        retirementReg.issuedGroups := 0
        retirementReg.completedGroups := 0
        retirementValid := True
        stickyFault := True
      }
    }
  }

  val desiredGroup = Vec(UInt(a.outputGroupWidth bits), engines)
  for (engine <- 0 until engines) {
    desiredGroup(engine) := nextCommand.resized
  }
  if (engines > 1) {
    when(held.mode === BitNetMode.Decode) {
      desiredGroup(1) := (nextCommand + held.groupsPerEngine).resized
    }
  }
  for (engine <- 0 until engines) {
    io.aguRequest(engine).valid := active && !roundFailed && !allIssued &&
      requestPending(engine)
    io.aguRequest(engine).target := held.weightTarget
    io.aguRequest(engine).layer := held.layer
    io.aguRequest(engine).outputGroup := desiredGroup(engine)
    when(io.aguRequest(engine).fire) {
      requestPending(engine) := False
    }

    io.aguResponse(engine).ready := active && !roundFailed &&
      !requestPending(engine) && !responseValid(engine)
    when(io.aguResponse(engine).fire) {
      responseHeld(engine) := io.aguResponse(engine).payload
      responseValid(engine) := True
    }
  }

  val responseLegal = Vec(Bool(), engines)
  for (engine <- 0 until engines) {
    val addressesFit = Vec(Bool(), banks)
    for (bank <- 0 until banks) {
      addressesFit(bank) := responseHeld(engine).localAddress(bank)(
        model.addressWidth - 1 downto p.crossbar.localAddressWidth) === 0
    }
    responseLegal(engine) := responseHeld(engine).ok &&
      responseHeld(engine).fault === BitNetFiveBankStripedAguFault.None &&
      responseHeld(engine).target === held.weightTarget &&
      responseHeld(engine).layer === held.layer &&
      responseHeld(engine).outputGroup === desiredGroup(engine) &&
      responseHeld(engine).outputBase === (desiredGroup(engine) << 2).resized &&
      addressesFit.asBits.andR
  }
  val responsePairMatches = if (engines == 1) True else
    (0 until banks).map { bank =>
      responseHeld(0).inputStart(bank) ===
        responseHeld(1).inputStart(bank) &&
      responseHeld(0).inputFeatures(bank) ===
        responseHeld(1).inputFeatures(bank)
    }.reduce(_ && _)
  val responsesReadyAndLegal = responseValid.andR &&
    responseLegal.asBits.andR && responsePairMatches
  val responsesReadyButBad = responseValid.andR &&
    !responsesReadyAndLegal

  when(active && !roundFailed && responsesReadyButBad) {
    roundFailed := True
    roundFault := B(BitNetResidentMatrixWeightFault.AguResponse, 8 bits)
    stickyFault := True
    allIssued := True
    requestPending := 0
    responseValid := 0
  }

  val selectedSlot = nextCommand(p.slotWidth - 1 downto 0)
  val kTiles = held.matrixTask.kTilesPerOutput
  val beatBase = (
    nextCommand.resize(matrix.beatCountWidth) *
      (kTiles.resize(matrix.beatCountWidth) << 1)).resize(
        matrix.beatCountWidth)
  io.command.valid := active && !roundFailed && !allIssued &&
    responsesReadyAndLegal && !occupied(selectedSlot)
  io.command.sequenceId := held.sequenceId
  io.command.epoch := held.prefetchEpoch
  io.command.mode := held.mode
  io.command.target := held.weightTarget
  io.command.layer := held.layer
  io.command.slot := selectedSlot
  for (bank <- 0 until banks) {
    io.command.inputStart(bank) := responseHeld(0).inputStart(bank).resized
    io.command.inputFeatures(bank) := responseHeld(0).inputFeatures(bank).resized
  }
  for (engine <- 0 until engines) {
    io.command.engine(engine).groupOrdinal :=
      responseHeld(engine).outputGroup.resized
    io.command.engine(engine).outputBase :=
      responseHeld(engine).outputBase.resized
    io.command.engine(engine).beatOrdinalBase := beatBase
    for (bank <- 0 until banks) {
      io.command.engine(engine).localBase(bank) :=
        responseHeld(engine).localAddress(bank).resized
    }
  }
  io.command.lastGroup := nextCommand === held.groupsPerEngine - 1

  when(io.command.fire) {
    occupied(selectedSlot) := True
    expectedBeatBase(selectedSlot) := beatBase
    expectedLast(selectedSlot) := io.command.lastGroup
    for (engine <- 0 until engines) {
      expectedGroup(selectedSlot)(engine) :=
        io.command.engine(engine).groupOrdinal
      expectedOutputBase(selectedSlot)(engine) :=
        io.command.engine(engine).outputBase
    }
    issuedRound := issuedRound + 1
    issuedLifetime := issuedLifetime + 1
    responseValid := 0
    when(io.command.lastGroup) {
      allIssued := True
      requestPending := 0
    } otherwise {
      nextCommand := nextCommand + 1
      requestPending := B((BigInt(1) << engines) - 1, engines bits)
    }
  }

  val completionSlot = io.prefetchCompletion.payload.slot
  val completionActiveMask = Bits(engines bits)
  completionActiveMask := B(1, engines bits)
  when(held.mode === BitNetMode.Decode) {
    completionActiveMask := B((BigInt(1) << engines) - 1, engines bits)
  }
  val completionEngineMatches = Vec(Bool(), engines)
  for (engine <- 0 until engines) {
    val participates = if (engine == 0) True
      else held.mode === BitNetMode.Decode
    val expectedRawGroups = UInt(p.rawGroupCountWidth bits)
    expectedRawGroups := 0
    when(participates) { expectedRawGroups := kTiles.resized }
    val bankBeatMatches = Vec(Bool(), banks)
    for (bank <- 0 until banks) {
      bankBeatMatches(bank) := !participates ||
        io.prefetchCompletion.payload.bankReadBeats(engine)(bank) ===
          (BitNetFiveBankStripedLayout.partitionInput(
            model.hiddenSize, bank)._2 / 16)
      when(held.weightTarget === BitNetAddressTarget.DownWeight &&
          participates) {
        bankBeatMatches(bank) :=
          io.prefetchCompletion.payload.bankReadBeats(engine)(bank) ===
            (BitNetFiveBankStripedLayout.partitionInput(
              model.ffnSize, bank)._2 / 16)
      }
    }
    completionEngineMatches(engine) :=
      (!participates ||
        (io.prefetchCompletion.payload.groupOrdinal(engine) ===
          expectedGroup(completionSlot)(engine) &&
        io.prefetchCompletion.payload.outputBase(engine) ===
          expectedOutputBase(completionSlot)(engine) &&
        io.prefetchCompletion.payload.beatOrdinalBase(engine) ===
          expectedBeatBase(completionSlot) &&
        io.prefetchCompletion.payload.emittedRawGroups(engine) ===
          expectedRawGroups &&
        bankBeatMatches.asBits.andR))
  }
  val completionMatches = occupied(completionSlot) &&
    io.prefetchCompletion.payload.sequenceId === held.sequenceId &&
    io.prefetchCompletion.payload.epoch === held.prefetchEpoch &&
    io.prefetchCompletion.payload.mode === held.mode &&
    io.prefetchCompletion.payload.target === held.weightTarget &&
    io.prefetchCompletion.payload.layer === held.layer &&
    io.prefetchCompletion.payload.activePoolMask === completionActiveMask &&
    io.prefetchCompletion.payload.lastGroup === expectedLast(completionSlot) &&
    completionEngineMatches.asBits.andR
  val completionFailure = io.prefetchCompletion.fire &&
    (!completionMatches || !io.prefetchCompletion.payload.success)
  val completionFault = Bits(8 bits)
  completionFault := Mux(!completionMatches,
    B(BitNetResidentMatrixWeightFault.PrefetchCompletion, 8 bits),
    Mux(io.prefetchCompletion.payload.faultCode === 0,
      B(BitNetResidentMatrixWeightFault.PrefetchFailed, 8 bits),
      io.prefetchCompletion.payload.faultCode))
  io.prefetchCompletion.ready := active
  when(io.prefetchCompletion.fire) {
    when(occupied(completionSlot)) {
      occupied(completionSlot) := False
      completedRound := completedRound + 1
      completedLifetime := completedLifetime + 1
    }
    when(!completionMatches || !io.prefetchCompletion.payload.success) {
      roundFailed := True
      roundFault := completionFault
      stickyFault := True
      allIssued := True
      requestPending := 0
      responseValid := 0
    }
  }

  val occupiedAfterCompletion = Bits(p.slotCount bits)
  occupiedAfterCompletion := occupied
  when(io.prefetchCompletion.fire && occupied(completionSlot)) {
    occupiedAfterCompletion(completionSlot) := False
  }
  val completedAfter = UInt(b.outputIndexWidth bits)
  completedAfter := completedRound
  when(io.prefetchCompletion.fire && occupied(completionSlot)) {
    completedAfter := completedRound + 1
  }
  val failureAfter = roundFailed || completionFailure
  val mayRetire = active && (allIssued || failureAfter) &&
    !occupiedAfterCompletion.orR && !requestPending.orR &&
    !responseValid.orR && !retirementValid
  when(mayRetire) {
    retirementReg.requestId := held.requestId
    retirementReg.stepId := held.stepId
    retirementReg.stage := held.stage
    retirementReg.layer := held.layer
    retirementReg.part := held.part
    retirementReg.sequenceId := held.sequenceId
    retirementReg.target := held.weightTarget
    retirementReg.success := !failureAfter &&
      completedAfter === held.groupsPerEngine
    retirementReg.faultCode := Mux(completionFailure,
      completionFault, roundFault)
    when(!failureAfter && completedAfter =/= held.groupsPerEngine) {
      retirementReg.success := False
      retirementReg.faultCode := B(
        BitNetResidentMatrixWeightFault.PrefetchCompletion, 8 bits)
      stickyFault := True
    }
    retirementReg.issuedGroups := issuedRound
    retirementReg.completedGroups := completedAfter
    retirementValid := True
    active := False
  }

  io.busy := active || retirementValid
  io.slotBusy := occupied
  io.issuedGroups := issuedLifetime
  io.completedGroups := completedLifetime
  io.protocolFaultSticky := stickyFault
}

/** Production wiring of the command expander to the two existing striped AGUs. */
class BitNetResidentMatrixWeightCommandSubsystem(
    cfg: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig()) extends Component {
  val io = new Bundle {
    val projection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(cfg.stage)))
    val command = master(Stream(
      BitNetFiveBankStripedWeightCommand(cfg.prefetch)))
    val prefetchCompletion = slave(Stream(
      BitNetFiveBankStripedWeightCompletion(cfg.prefetch)))
    val retirement = master(Stream(
      BitNetResidentMatrixWeightRetirement(cfg)))
    val busy = out Bool()
    val slotBusy = out Bits(cfg.prefetch.slotCount bits)
    val protocolFaultSticky = out Bool()
  }

  val expander = new BitNetResidentMatrixWeightCommandExpander(cfg)
  val agus = Array.fill(cfg.stage.model.engineCount)(
    new BitNetFiveBankStripedTransformerAgu(cfg.agu))
  expander.io.projection << io.projection
  io.command << expander.io.command
  expander.io.prefetchCompletion << io.prefetchCompletion
  io.retirement << expander.io.retirement
  for (engine <- 0 until cfg.stage.model.engineCount) {
    agus(engine).io.request << expander.io.aguRequest(engine)
    expander.io.aguResponse(engine) << agus(engine).io.response
  }
  io.busy := expander.io.busy
  io.slotBusy := expander.io.slotBusy
  io.protocolFaultSticky := expander.io.protocolFaultSticky
}

object GenerateBitNetResidentMatrixWeightCommandSubsystem extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-resident-matrix-weight-command-subsystem")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentMatrixWeightCommandSubsystem())
}
