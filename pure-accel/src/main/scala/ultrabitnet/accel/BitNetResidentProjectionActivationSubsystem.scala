package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Geometry of the projection activation-image builder.
  *
  * The builder is deliberately a client of the model-wide dynamic-i8 service.
  * It owns only a four-feature physical-read staging register and transient
  * packed-row assembly; both ping/pong images reside in the unique unified
  * tensor scratchpad and it never owns a Q16.16 vector image.
  */
case class BitNetResidentProjectionActivationSubsystemConfig(
    model: BitNetConfig = BitNetConfig.Production,
    activationLayout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production),
    quant: BitNetSharedDynamicI8QuantizerConfig =
      BitNetSharedDynamicI8QuantizerConfig(),
    tile: BitNetResidentActivationTileBufferConfig =
      BitNetResidentActivationTileBufferConfig(),
    scratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()
) {
  require(model.hiddenSize == 2560 && model.ffnSize == 6912,
    "the production resident image has fixed hidden/FFN geometries")
  require((model.tokenParallelism == 2 || model.tokenParallelism == 4) &&
    model.spuCount == 4,
    "the physical activation reader must be TP2 or TP4")
  require(activationLayout.cfg == model)
  require(tile.hiddenFeatures == model.hiddenSize)
  require(tile.ffnFeatures == model.ffnSize)
  require(tile.tokenLanes == model.tokenParallelism)
  require(tile.feeder.matrix.base.siteCount == 64 ||
    tile.feeder.matrix.base.siteCount == 128)
  require(tile.tokenLanes <= scratchpad.layout.tokenLanes)
  require(tile.bankCount == scratchpad.layout.bankCount)
  require(tile.featuresPerBankWord ==
    scratchpad.layout.packedFeaturesPerWord)
  require(tile.bankWordWidth <= scratchpad.layout.wordWidth)
  require((tile.maxFeatures + 127) / 128 <=
    scratchpad.layout.projectionSlot0.rowCount)
  require(scratchpad.layout.projectionSlot0.rowCount ==
    scratchpad.layout.projectionSlot1.rowCount)
  require(quant.tokenLanes == model.tokenParallelism)
  require(quant.maxVectorLength >= model.ffnSize)
  require(quant.inputWidth == 32 && quant.factorWidth == 32)
  require(quant.tagWidth == tile.feeder.matrix.sequenceIdWidth,
    "quantizer tags and resident image tags are one ownership domain")
  require(quant.epochWidth == tile.epochWidth,
    "quantizer epochs and resident image epochs are one ownership domain")
}

/** One fixed-length activation-image request from the resident controller. */
case class BitNetResidentProjectionActivationCommand(
    cfg: BitNetResidentProjectionActivationSubsystemConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val tag = UInt(cfg.quant.tagWidth bits)
  val epoch = UInt(cfg.quant.epochWidth bits)
  val slot = UInt(cfg.tile.slotWidth bits)
  val sourceBuffer = BitNetPhysicalActivationBuffer()
  val featureCount = UInt(cfg.tile.featureCountWidth bits)
  /** Only 0001/0011/0111/1111 are legal production masks. */
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
}

/** Authenticated retirement of one resident activation image. */
case class BitNetResidentProjectionActivationCompletion(
    cfg: BitNetResidentProjectionActivationSubsystemConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val tag = UInt(cfg.quant.tagWidth bits)
  val epoch = UInt(cfg.quant.epochWidth bits)
  val slot = UInt(cfg.tile.slotWidth bits)
  val sourceBuffer = BitNetPhysicalActivationBuffer()
  val featureCount = UInt(cfg.tile.featureCountWidth bits)
  val acceptedFeatures = UInt(cfg.tile.featureCountWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val dequantFactorQ16 = Vec(UInt(32 bits), cfg.model.tokenParallelism)
  val success = Bool()
  /** Bit set, so simultaneous physical/quant/tile failures are not hidden. */
  val faultCode = Bits(16 bits)
}

/** Stable bit allocation for builder failures plus typed replay reject codes. */
object BitNetResidentProjectionActivationFault {
  val InvalidCommand = 0
  val PhysicalRead = 1
  val QuantReplayRequest = 2
  val QuantResult = 3
  val QuantDone = 4
  val TileFillCompletion = 5
  val ReplayCompletion = 6

  val InvalidReplayCommandCode = 0xb0
  val ReplayCompletionCode = 0xb1
}

/**
  * Builds the only projection activation image consumed by the TP4 feeder.
  *
  * Every legal request performs two complete group-by-four reads from the
  * selected physical activation buffer.  The first pass measures and the
  * second pass replays through an externally owned shared quantizer.  Its i8
  * results are committed as packed 128-feature rows into the projection
  * regions of the unique unified tensor scratchpad.  Fill of one slot and
  * feeder replay of the other slot remain independent.
  *
  * The wrapper maintains authoritative slot ownership.  In particular, the
  * packed controller's local ready bit is never exposed before an authenticated
  * quantizer `done`, fixed-length physical drain and authenticated fill
  * completion have all retired.  A malformed early `done` switches the tile
  * input to a fixed-length fault drain, preventing a half image from hanging
  * the buffer or becoming visible to the feeder.
  */
class BitNetResidentProjectionActivationSubsystem(
    cfg: BitNetResidentProjectionActivationSubsystemConfig =
      BitNetResidentProjectionActivationSubsystemConfig(),
    useVendorSegmentRam: Boolean = true)
    extends Component {
  private val model = cfg.model
  private val layout = cfg.activationLayout
  private val q = cfg.quant
  private val t = cfg.tile
  private val scratchpad = cfg.scratchpad
  private val matrix = t.feeder.matrix

  val io = new Bundle {
    val command = slave(Stream(
      BitNetResidentProjectionActivationCommand(cfg)))

    val activationReadCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))),
      model.tokenParallelism)
    val activationReadData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))),
      model.tokenParallelism)

    // Typed boundary of the sole model-wide dynamic-i8 quantizer.
    val quantCommand = master(Stream(BitNetSharedDynamicQuantCommand(q)))
    val quantMeasureInput = master(Stream(
      BitNetSharedDynamicQuantMeasureBeat(q)))
    val quantReplayRequest = slave(Stream(
      BitNetSharedDynamicQuantReplayRequest(q)))
    val quantReplayInput = master(Stream(
      BitNetSharedDynamicQuantReplayBeat(q)))
    val quantOutput = slave(Stream(BitNetSharedDynamicQuantResultBeat(q)))
    val quantDone = slave(Stream(BitNetSharedDynamicQuantDone(q)))

    // The sole typed activation feeder boundary.
    val replayCommand = slave(Stream(
      BitNetResidentActivationReplayCommand(t)))
    val tile = master(Stream(BitNetResidentActivationTile(t)))
    val replayCompletion = master(Stream(
      BitNetResidentActivationReplayCompletion(t)))

    // Direct clients of the one model-wide semantic tensor frontend.  There
    // is no projection activation data Mem below this boundary.
    val packedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(scratchpad)))
    val packedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(scratchpad)))
    val packedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(scratchpad)))
    val packedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(scratchpad)))

    val completion = master(Stream(
      BitNetResidentProjectionActivationCompletion(cfg)))

    // These are authoritative; the nested tile-buffer ready bits are hidden.
    val slotReady = out Bits(t.slotCount bits)
    val slotImageTag = out Vec(
      UInt(matrix.sequenceIdWidth bits), t.slotCount)
    val slotEpoch = out Vec(UInt(t.epochWidth bits), t.slotCount)
    val slotFeatureCount = out Vec(
      UInt(t.featureCountWidth bits), t.slotCount)
    val slotTokenMask = out Vec(Bits(t.tokenLanes bits), t.slotCount)
    val slotDequantFactorQ16 = out Vec(
      Vec(UInt(32 bits), t.tokenLanes), t.slotCount)

    val busy = out Bool()
    val fillActive = out Bool()
    val replayActive = out Bool()
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(16 bits)
    val completedImages = out UInt(32 bits)
    val completedReplays = out UInt(32 bits)
  }

  val residentImage = new BitNetUnifiedResidentProjectionTileController(
    BitNetUnifiedResidentProjectionTileControllerConfig(
      tile = t, scratchpad = scratchpad),
    useVendorSegmentRam = useVendorSegmentRam)
  residentImage.setName("unifiedResidentProjectionImageController")

  io.packedRead.valid := residentImage.io.packedRead.valid
  io.packedRead.payload := residentImage.io.packedRead.payload
  residentImage.io.packedRead.ready := io.packedRead.ready

  residentImage.io.packedReadData.valid := io.packedReadData.valid
  residentImage.io.packedReadData.payload := io.packedReadData.payload
  io.packedReadData.ready := residentImage.io.packedReadData.ready

  io.packedWrite.valid := residentImage.io.packedWrite.valid
  io.packedWrite.payload := residentImage.io.packedWrite.payload
  residentImage.io.packedWrite.ready := io.packedWrite.ready

  residentImage.io.packedWriteFault.valid := io.packedWriteFault.valid
  residentImage.io.packedWriteFault.payload := io.packedWriteFault.payload
  io.packedWriteFault.ready := residentImage.io.packedWriteFault.ready

  object JobState extends SpinalEnum(binarySequential) {
    val Idle, StartServices, Run, Respond = newElement()
  }
  object ReadState extends SpinalEnum(binarySequential) {
    val Dormant, MeasureIssue, MeasureWait, MeasureEmit,
        WaitReplayRequest, ReplayIssue, ReplayWait, ReplayEmit,
        Finished = newElement()
  }
  val jobState = Reg(JobState()) init JobState.Idle
  val readState = Reg(ReadState()) init ReadState.Dormant

  val active = Reg(BitNetResidentProjectionActivationCommand(cfg)) init
    BitNetResidentProjectionActivationCommand(cfg).getZero
  val jobFault = Reg(Bits(16 bits)) init 0
  val stickyFault = RegInit(False)

  val completionValid = RegInit(False)
  val completionPayload = Reg(
    BitNetResidentProjectionActivationCompletion(cfg)) init
      BitNetResidentProjectionActivationCompletion(cfg).getZero
  io.completion.valid := completionValid
  io.completion.payload := completionPayload

  val completedImageCounter = Reg(UInt(32 bits)) init 0
  val completedReplayCounter = Reg(UInt(32 bits)) init 0
  io.completedImages := completedImageCounter
  io.completedReplays := completedReplayCounter

  // -----------------------------------------------------------------------
  // Authoritative two-slot ownership.  Never expose the nested early ready.
  // -----------------------------------------------------------------------
  val authoritativeReady = Reg(Bits(t.slotCount bits)) init 0
  val authoritativeTag = Vec(
    Reg(UInt(matrix.sequenceIdWidth bits)) init 0, t.slotCount)
  val authoritativeEpoch = Vec(
    Reg(UInt(t.epochWidth bits)) init 0, t.slotCount)
  val authoritativeFeatureCount = Vec(
    Reg(UInt(t.featureCountWidth bits)) init 0, t.slotCount)
  val authoritativeMask = Vec(
    Reg(Bits(t.tokenLanes bits)) init 0, t.slotCount)
  val authoritativeFactor = Vec(
    Vec(Reg(UInt(32 bits)) init 0, t.tokenLanes), t.slotCount)

  io.slotReady := authoritativeReady
  io.slotImageTag := authoritativeTag
  io.slotEpoch := authoritativeEpoch
  io.slotFeatureCount := authoritativeFeatureCount
  io.slotTokenMask := authoritativeMask
  io.slotDequantFactorQ16 := authoritativeFactor

  // -----------------------------------------------------------------------
  // Authenticated replay wrapper around the unified ping/pong image regions.
  // -----------------------------------------------------------------------
  val replayActiveReg = RegInit(False)
  val replayHeld = Reg(BitNetResidentActivationReplayCommand(t)) init
    BitNetResidentActivationReplayCommand(t).getZero
  val replayCompletionValid = RegInit(False)
  val replayCompletionPayload = Reg(
    BitNetResidentActivationReplayCompletion(t)) init
      BitNetResidentActivationReplayCompletion(t).getZero

  io.replayCompletion.valid := replayCompletionValid
  io.replayCompletion.payload := replayCompletionPayload
  when(io.replayCompletion.fire) {
    replayCompletionValid := False
  }

  val replaySlotReady = authoritativeReady(io.replayCommand.payload.slot)
  val replayOwnershipLegal = replaySlotReady &&
    authoritativeTag(io.replayCommand.payload.slot) ===
      io.replayCommand.payload.imageTag &&
    authoritativeEpoch(io.replayCommand.payload.slot) ===
      io.replayCommand.payload.epoch &&
    authoritativeFeatureCount(io.replayCommand.payload.slot) ===
      io.replayCommand.payload.inputFeatures
  val replayFillCollision =
    (jobState === JobState.StartServices || jobState === JobState.Run) &&
      active.slot === io.replayCommand.payload.slot
  val publicReplayLegal = replayOwnershipLegal && !replayFillCollision
  val replayWrapperIdle = !replayActiveReg && !replayCompletionValid

  io.replayCommand.ready := replayWrapperIdle &&
    (!publicReplayLegal || residentImage.io.replayCommand.ready)
  residentImage.io.replayCommand.valid := replayWrapperIdle &&
    io.replayCommand.valid && publicReplayLegal
  residentImage.io.replayCommand.payload := io.replayCommand.payload

  when(io.replayCommand.fire) {
    when(publicReplayLegal) {
      replayActiveReg := True
      replayHeld := io.replayCommand.payload
    } otherwise {
      replayCompletionValid := True
      replayCompletionPayload.imageTag := io.replayCommand.payload.imageTag
      replayCompletionPayload.epoch := io.replayCommand.payload.epoch
      replayCompletionPayload.slot := io.replayCommand.payload.slot
      replayCompletionPayload.outputSequenceId :=
        io.replayCommand.payload.outputSequenceId
      replayCompletionPayload.emittedBeats := 0
      replayCompletionPayload.success := False
      replayCompletionPayload.faultCode := B(
        BitNetResidentProjectionActivationFault.InvalidReplayCommandCode,
        8 bits)
      stickyFault := True
    }
  }

  io.tile.valid := residentImage.io.tile.valid
  io.tile.payload := residentImage.io.tile.payload
  residentImage.io.tile.ready := io.tile.ready

  residentImage.io.replayCompletion.ready := replayActiveReg &&
    !replayCompletionValid
  val nestedReplayCompletionLegal =
    residentImage.io.replayCompletion.payload.imageTag ===
      replayHeld.imageTag &&
    residentImage.io.replayCompletion.payload.epoch === replayHeld.epoch &&
    residentImage.io.replayCompletion.payload.slot === replayHeld.slot &&
    residentImage.io.replayCompletion.payload.outputSequenceId ===
      replayHeld.outputSequenceId &&
    residentImage.io.replayCompletion.payload.emittedBeats ===
      replayHeld.beatCount &&
    residentImage.io.replayCompletion.payload.success &&
    residentImage.io.replayCompletion.payload.faultCode === 0

  when(residentImage.io.replayCompletion.fire) {
    replayActiveReg := False
    replayCompletionValid := True
    // Return held ownership, never untrusted completion ownership.
    replayCompletionPayload.imageTag := replayHeld.imageTag
    replayCompletionPayload.epoch := replayHeld.epoch
    replayCompletionPayload.slot := replayHeld.slot
    replayCompletionPayload.outputSequenceId := replayHeld.outputSequenceId
    replayCompletionPayload.emittedBeats :=
      residentImage.io.replayCompletion.payload.emittedBeats
    replayCompletionPayload.success := nestedReplayCompletionLegal
    replayCompletionPayload.faultCode := B(
      BitNetResidentProjectionActivationFault.ReplayCompletionCode,
      8 bits)
    when(nestedReplayCompletionLegal) {
      replayCompletionPayload.faultCode := 0
      completedReplayCounter := completedReplayCounter + 1
    } otherwise {
      stickyFault := True
    }
  }

  // -----------------------------------------------------------------------
  // Command admission and the quantizer/packed-controller start barrier.
  // -----------------------------------------------------------------------
  val commandPrefixMaskLegal = (1 to model.tokenParallelism).map { count =>
    io.command.payload.tokenMask === B((BigInt(1) << count) - 1,
      model.tokenParallelism bits)
  }.reduce(_ || _)
  val commandFeatureCountLegal =
    io.command.payload.featureCount === model.hiddenSize ||
    io.command.payload.featureCount === model.ffnSize
  val commandSourceCapacityLegal =
    io.command.payload.featureCount === model.hiddenSize ||
    io.command.payload.sourceBuffer ===
      BitNetPhysicalActivationBuffer.WideWork
  val commandReplayCollision = replayActiveReg &&
    replayHeld.slot === io.command.payload.slot
  val commandLegal = commandPrefixMaskLegal && commandFeatureCountLegal &&
    commandSourceCapacityLegal && !commandReplayCollision

  io.command.ready := jobState === JobState.Idle && !completionValid

  val quantCommandSent = RegInit(False)
  val fillCommandSent = RegInit(False)

  io.quantCommand.valid := jobState === JobState.StartServices &&
    !quantCommandSent
  io.quantCommand.payload.tag := active.tag
  io.quantCommand.payload.epoch := active.epoch
  io.quantCommand.payload.vectorLength := active.featureCount.resized
  io.quantCommand.payload.tokenMask := active.tokenMask

  residentImage.io.fillCommand.valid :=
    jobState === JobState.StartServices && !fillCommandSent
  residentImage.io.fillCommand.payload.imageTag := active.tag.resized
  residentImage.io.fillCommand.payload.epoch := active.epoch
  residentImage.io.fillCommand.payload.slot := active.slot
  residentImage.io.fillCommand.payload.featureCount := active.featureCount
  residentImage.io.fillCommand.payload.tokenMask := active.tokenMask

  // Per-job state reset below is deliberately explicit: an epoch/tag cannot
  // inherit a count or authenticated completion from the previous image.
  val quantResultCount = Reg(UInt(t.featureCountWidth bits)) init 0
  val quantDoneSeen = RegInit(False)
  val quantDoneGood = RegInit(False)
  val syntheticDrain = RegInit(False)
  val quantFactorValid = RegInit(False)
  val quantFactor = Vec(Reg(UInt(32 bits)) init 0, t.tokenLanes)
  val tileFillSeen = RegInit(False)
  val tileFillGood = RegInit(False)
  val tileFillFactor = Vec(Reg(UInt(32 bits)) init 0, t.tokenLanes)

  when(io.command.fire) {
    active := io.command.payload
    authoritativeReady(io.command.payload.slot) := False
    jobFault := 0
    quantCommandSent := False
    fillCommandSent := False
    quantResultCount := 0
    quantDoneSeen := False
    quantDoneGood := False
    syntheticDrain := False
    quantFactorValid := False
    tileFillSeen := False
    tileFillGood := False
    readState := ReadState.Dormant
    for (lane <- 0 until t.tokenLanes) {
      quantFactor(lane) := 0
      tileFillFactor(lane) := 0
    }

    when(commandLegal) {
      jobState := JobState.StartServices
    } otherwise {
      val invalidBits = B(
        BigInt(1) << BitNetResidentProjectionActivationFault.InvalidCommand,
        16 bits)
      jobFault := invalidBits
      completionValid := True
      completionPayload.requestId := io.command.payload.requestId
      completionPayload.tag := io.command.payload.tag
      completionPayload.epoch := io.command.payload.epoch
      completionPayload.slot := io.command.payload.slot
      completionPayload.sourceBuffer := io.command.payload.sourceBuffer
      completionPayload.featureCount := io.command.payload.featureCount
      completionPayload.acceptedFeatures := 0
      completionPayload.tokenMask := io.command.payload.tokenMask
      for (lane <- 0 until t.tokenLanes) {
        completionPayload.dequantFactorQ16(lane) := 0
      }
      completionPayload.success := False
      completionPayload.faultCode := invalidBits
      stickyFault := True
      jobState := JobState.Respond
    }
  }

  when(io.quantCommand.fire) { quantCommandSent := True }
  when(residentImage.io.fillCommand.fire) { fillCommandSent := True }
  val quantCommandAccepted = quantCommandSent || io.quantCommand.fire
  val fillCommandAccepted = fillCommandSent ||
    residentImage.io.fillCommand.fire
  when(jobState === JobState.StartServices && quantCommandAccepted &&
      fillCommandAccepted) {
    jobState := JobState.Run
    readState := ReadState.MeasureIssue
  }

  // -----------------------------------------------------------------------
  // Four independent physical lanes, group read followed by four stripes.
  // Commands and responses can be independently backpressured per lane.
  // -----------------------------------------------------------------------
  val readGroup = Reg(UInt(layout.groupIndexWidth bits)) init 0
  val readShard = Reg(UInt(log2Up(model.spuCount) bits)) init 0
  val readIssued = Reg(Bits(model.tokenParallelism bits)) init 0
  val readReceived = Reg(Bits(model.tokenParallelism bits)) init 0
  val heldReadValues = Vec(
    Vec(Reg(SInt(32 bits)) init 0, model.spuCount),
    model.tokenParallelism)

  val issuePhase = jobState === JobState.Run &&
    (readState === ReadState.MeasureIssue ||
      readState === ReadState.ReplayIssue)
  val waitPhase = jobState === JobState.Run &&
    (readState === ReadState.MeasureWait ||
      readState === ReadState.ReplayWait)

  val readIssueFire = Bits(model.tokenParallelism bits)
  val readResponseFire = Bits(model.tokenParallelism bits)
  readIssueFire := 0
  readResponseFire := 0

  for (lane <- 0 until model.tokenParallelism) {
    io.activationReadCommand(lane).valid := issuePhase &&
      active.tokenMask(lane) && !readIssued(lane)
    io.activationReadCommand(lane).payload.buffer := active.sourceBuffer
    io.activationReadCommand(lane).payload.group := readGroup
    io.activationReadData(lane).ready := waitPhase &&
      active.tokenMask(lane) && !readReceived(lane)
    readIssueFire(lane) := io.activationReadCommand(lane).fire
    readResponseFire(lane) := io.activationReadData(lane).fire
  }

  val issuedAfter = readIssued | readIssueFire
  val receivedAfter = readReceived | readResponseFire
  when(issuePhase) {
    readIssued := issuedAfter
    when((issuedAfter & active.tokenMask) === active.tokenMask) {
      readIssued := 0
      readReceived := 0
      when(readState === ReadState.MeasureIssue) {
        readState := ReadState.MeasureWait
      } otherwise {
        readState := ReadState.ReplayWait
      }
    }
  }

  when(waitPhase) {
    readReceived := receivedAfter
    for (lane <- 0 until model.tokenParallelism) {
      val responseLegal =
        io.activationReadData(lane).payload.buffer === active.sourceBuffer &&
        io.activationReadData(lane).payload.group === readGroup &&
        io.activationReadData(lane).payload.featureValid.andR &&
        !io.activationReadData(lane).payload.fault
      when(io.activationReadData(lane).fire) {
        for (shard <- 0 until model.spuCount) {
          heldReadValues(lane)(shard) := 0
          when(responseLegal) {
            heldReadValues(lane)(shard) :=
              io.activationReadData(lane).payload.values(shard)
          }
        }
        when(!responseLegal) {
          jobFault(
            BitNetResidentProjectionActivationFault.PhysicalRead) := True
          stickyFault := True
        }
      }
    }
    when((receivedAfter & active.tokenMask) === active.tokenMask) {
      readReceived := 0
      readShard := 0
      when(readState === ReadState.MeasureWait) {
        readState := ReadState.MeasureEmit
      } otherwise {
        readState := ReadState.ReplayEmit
      }
    }
  }

  val readFeatureIndex = UInt(q.indexWidth bits)
  readFeatureIndex := ((readGroup.resize(q.indexWidth) |<< 2) +
    readShard.resize(q.indexWidth)).resized

  io.quantMeasureInput.valid := jobState === JobState.Run &&
    readState === ReadState.MeasureEmit && !quantDoneSeen
  io.quantMeasureInput.payload.tag := active.tag
  io.quantMeasureInput.payload.epoch := active.epoch
  io.quantMeasureInput.payload.featureIndex := readFeatureIndex
  io.quantMeasureInput.payload.tokenMask := active.tokenMask
  io.quantMeasureInput.payload.first := readFeatureIndex === 0
  io.quantMeasureInput.payload.last :=
    readFeatureIndex.resize(t.featureCountWidth) === active.featureCount - 1

  io.quantReplayInput.valid := jobState === JobState.Run &&
    readState === ReadState.ReplayEmit && !quantDoneSeen
  io.quantReplayInput.payload.tag := active.tag
  io.quantReplayInput.payload.epoch := active.epoch
  io.quantReplayInput.payload.featureIndex := readFeatureIndex
  io.quantReplayInput.payload.tokenMask := active.tokenMask
  io.quantReplayInput.payload.first := readFeatureIndex === 0
  io.quantReplayInput.payload.last :=
    readFeatureIndex.resize(t.featureCountWidth) === active.featureCount - 1

  for (lane <- 0 until model.tokenParallelism) {
    io.quantMeasureInput.payload.value(lane) := 0
    io.quantReplayInput.payload.value(lane) := 0
    when(active.tokenMask(lane)) {
      io.quantMeasureInput.payload.value(lane) :=
        heldReadValues(lane)(readShard)
      io.quantReplayInput.payload.value(lane) :=
        heldReadValues(lane)(readShard)
    }
  }

  val measureEmitAdvance = jobState === JobState.Run &&
    readState === ReadState.MeasureEmit &&
    (quantDoneSeen || io.quantMeasureInput.fire)
  val replayEmitAdvance = jobState === JobState.Run &&
    readState === ReadState.ReplayEmit &&
    (quantDoneSeen || io.quantReplayInput.fire)
  val lastReadShard = readShard === model.spuCount - 1
  val readGroupCount = (active.featureCount |>> 2).resize(
    t.featureCountWidth)
  val lastReadGroup = readGroup.resize(t.featureCountWidth) ===
    readGroupCount - 1

  when(measureEmitAdvance) {
    when(lastReadShard) {
      readShard := 0
      readIssued := 0
      when(lastReadGroup) {
        readGroup := 0
        when(quantDoneSeen) {
          readState := ReadState.ReplayIssue
        } otherwise {
          readState := ReadState.WaitReplayRequest
        }
      } otherwise {
        readGroup := readGroup + 1
        readState := ReadState.MeasureIssue
      }
    } otherwise {
      readShard := readShard + 1
    }
  }

  io.quantReplayRequest.ready := jobState === JobState.Run &&
    readState === ReadState.WaitReplayRequest && !quantDoneSeen
  val quantReplayRequestLegal =
    io.quantReplayRequest.payload.tag === active.tag &&
    io.quantReplayRequest.payload.epoch === active.epoch &&
    io.quantReplayRequest.payload.vectorLength ===
      active.featureCount.resized &&
    io.quantReplayRequest.payload.tokenMask === active.tokenMask

  when(io.quantReplayRequest.fire) {
    when(!quantReplayRequestLegal) {
      jobFault(
        BitNetResidentProjectionActivationFault.QuantReplayRequest) := True
      stickyFault := True
    }
    readGroup := 0
    readShard := 0
    readIssued := 0
    readState := ReadState.ReplayIssue
  }

  when(replayEmitAdvance) {
    when(lastReadShard) {
      readShard := 0
      readIssued := 0
      when(lastReadGroup) {
        readGroup := 0
        readState := ReadState.Finished
      } otherwise {
        readGroup := readGroup + 1
        readState := ReadState.ReplayIssue
      }
    } otherwise {
      readShard := readShard + 1
    }
  }

  // -----------------------------------------------------------------------
  // Quant result -> resident i8 fill.  Metadata is rebuilt from held command
  // identity; mismatches poison but never shorten the configured image.
  // -----------------------------------------------------------------------
  val expectedQuantIndex = quantResultCount.resize(q.indexWidth)
  val expectedQuantFirst = quantResultCount === 0
  val expectedQuantLast = quantResultCount === active.featureCount - 1
  val quantFactorStable = (0 until t.tokenLanes).map { lane =>
    !quantFactorValid ||
      io.quantOutput.payload.dequantFactorQ16(lane) === quantFactor(lane)
  }.reduce(_ && _)
  val quantInactiveZero = (0 until t.tokenLanes).map { lane =>
    active.tokenMask(lane) || io.quantOutput.payload.value(lane) === 0
  }.reduce(_ && _)
  val quantResultMetadataLegal =
    io.quantOutput.payload.tag === active.tag &&
    io.quantOutput.payload.epoch === active.epoch &&
    io.quantOutput.payload.featureIndex === expectedQuantIndex &&
    io.quantOutput.payload.tokenMask === active.tokenMask &&
    io.quantOutput.payload.first === expectedQuantFirst &&
    io.quantOutput.payload.last === expectedQuantLast &&
    quantFactorStable && quantInactiveZero

  val quantResultSpace = quantResultCount < active.featureCount
  val realQuantResultEnabled = jobState === JobState.Run &&
    !syntheticDrain && quantResultSpace
  io.quantOutput.ready := realQuantResultEnabled &&
    residentImage.io.quantized.ready

  residentImage.io.quantized.valid := jobState === JobState.Run &&
    quantResultSpace &&
    (syntheticDrain || (!syntheticDrain && io.quantOutput.valid))
  residentImage.io.quantized.payload.imageTag := active.tag.resized
  residentImage.io.quantized.payload.epoch := active.epoch
  residentImage.io.quantized.payload.slot := active.slot
  residentImage.io.quantized.payload.featureIndex :=
    quantResultCount.resized
  residentImage.io.quantized.payload.tokenMask := active.tokenMask
  residentImage.io.quantized.payload.first := expectedQuantFirst
  residentImage.io.quantized.payload.last := expectedQuantLast
  residentImage.io.quantized.payload.fault := syntheticDrain ||
    !quantResultMetadataLegal || jobFault.orR
  for (lane <- 0 until t.tokenLanes) {
    residentImage.io.quantized.payload.value(lane) :=
      io.quantOutput.payload.value(lane)
    residentImage.io.quantized.payload.dequantFactorQ16(lane) :=
      io.quantOutput.payload.dequantFactorQ16(lane)
    when(syntheticDrain) {
      residentImage.io.quantized.payload.value(lane) := 0
      residentImage.io.quantized.payload.dequantFactorQ16(lane) := 0
      when(quantFactorValid) {
        residentImage.io.quantized.payload.dequantFactorQ16(lane) :=
          quantFactor(lane)
      }
    }
  }

  val residentQuantFire = residentImage.io.quantized.fire
  val realQuantFire = residentQuantFire && !syntheticDrain
  when(realQuantFire) {
    when(!quantResultMetadataLegal) {
      jobFault(BitNetResidentProjectionActivationFault.QuantResult) := True
      stickyFault := True
    }
    when(!quantFactorValid) {
      quantFactorValid := True
      for (lane <- 0 until t.tokenLanes) {
        quantFactor(lane) :=
          io.quantOutput.payload.dequantFactorQ16(lane)
      }
    }
  }
  when(residentQuantFire) {
    quantResultCount := quantResultCount + 1
    when(expectedQuantLast) {
      syntheticDrain := False
    }
  }

  // `done` is accepted throughout Run so an externally failed service cannot
  // deadlock measure/replay.  A short result stream is completed with poisoned
  // zero beats while both physical passes continue to their fixed boundary.
  io.quantDone.ready := jobState === JobState.Run && !quantDoneSeen
  val realResultCountAfter =
    (quantResultCount + realQuantFire.asUInt).resize(t.featureCountWidth)
  val quantDoneLegal =
    io.quantDone.payload.tag === active.tag &&
    io.quantDone.payload.epoch === active.epoch &&
    io.quantDone.payload.vectorLength === active.featureCount.resized &&
    io.quantDone.payload.emittedFeatures === active.featureCount.resized &&
    io.quantDone.payload.success && io.quantDone.payload.faultCode === 0 &&
    realResultCountAfter === active.featureCount

  when(io.quantDone.fire) {
    quantDoneSeen := True
    quantDoneGood := quantDoneLegal
    when(!quantDoneLegal) {
      jobFault(BitNetResidentProjectionActivationFault.QuantDone) := True
      stickyFault := True
    }
    when(realResultCountAfter =/= active.featureCount) {
      syntheticDrain := True
    }
    // A malformed early completion cannot strand the second physical pass.
    when(readState === ReadState.WaitReplayRequest) {
      readGroup := 0
      readShard := 0
      readIssued := 0
      readState := ReadState.ReplayIssue
    }
  }

  // -----------------------------------------------------------------------
  // Authenticate the packed-controller completion before publishing ownership.
  // -----------------------------------------------------------------------
  residentImage.io.fillCompletion.ready :=
    jobState === JobState.Run && !tileFillSeen
  val nestedFillFactorsLegal = (0 until t.tokenLanes).map { lane =>
    !quantFactorValid ||
      residentImage.io.fillCompletion.payload.dequantFactorQ16(lane) ===
        quantFactor(lane)
  }.reduce(_ && _)
  val nestedFillCompletionLegal =
    residentImage.io.fillCompletion.payload.imageTag === active.tag.resized &&
    residentImage.io.fillCompletion.payload.epoch === active.epoch &&
    residentImage.io.fillCompletion.payload.slot === active.slot &&
    residentImage.io.fillCompletion.payload.featureCount ===
      active.featureCount &&
    residentImage.io.fillCompletion.payload.acceptedFeatures ===
      active.featureCount &&
    residentImage.io.fillCompletion.payload.tokenMask === active.tokenMask &&
    residentImage.io.fillCompletion.payload.success &&
    residentImage.io.fillCompletion.payload.faultCode === 0 &&
    nestedFillFactorsLegal

  when(residentImage.io.fillCompletion.fire) {
    tileFillSeen := True
    tileFillGood := nestedFillCompletionLegal
    for (lane <- 0 until t.tokenLanes) {
      tileFillFactor(lane) :=
        residentImage.io.fillCompletion.payload.dequantFactorQ16(lane)
    }
    when(!nestedFillCompletionLegal) {
      jobFault(
        BitNetResidentProjectionActivationFault.TileFillCompletion) := True
      stickyFault := True
    }
  }

  val jobRetired = jobState === JobState.Run &&
    readState === ReadState.Finished && quantDoneSeen && tileFillSeen &&
    quantResultCount === active.featureCount
  val jobSuccessful = !jobFault.orR && quantDoneGood && tileFillGood

  when(jobRetired) {
    completionValid := True
    completionPayload.requestId := active.requestId
    completionPayload.tag := active.tag
    completionPayload.epoch := active.epoch
    completionPayload.slot := active.slot
    completionPayload.sourceBuffer := active.sourceBuffer
    completionPayload.featureCount := active.featureCount
    completionPayload.acceptedFeatures := quantResultCount
    completionPayload.tokenMask := active.tokenMask
    for (lane <- 0 until t.tokenLanes) {
      completionPayload.dequantFactorQ16(lane) := tileFillFactor(lane)
    }
    completionPayload.success := jobSuccessful
    completionPayload.faultCode := jobFault
    authoritativeReady(active.slot) := False
    when(jobSuccessful) {
      authoritativeReady(active.slot) := True
      authoritativeTag(active.slot) := active.tag.resized
      authoritativeEpoch(active.slot) := active.epoch
      authoritativeFeatureCount(active.slot) := active.featureCount
      authoritativeMask(active.slot) := active.tokenMask
      for (lane <- 0 until t.tokenLanes) {
        authoritativeFactor(active.slot)(lane) := tileFillFactor(lane)
      }
    }
    completedImageCounter := completedImageCounter + 1
    jobState := JobState.Respond
  }

  when(io.completion.fire) {
    completionValid := False
    jobState := JobState.Idle
    readState := ReadState.Dormant
  }

  io.busy := jobState =/= JobState.Idle
  io.fillActive := jobState === JobState.StartServices ||
    jobState === JobState.Run
  io.replayActive := replayActiveReg
  io.protocolFaultSticky := stickyFault ||
    residentImage.io.protocolFaultSticky
  io.faultBits := jobFault
}

object GenerateBitNetResidentProjectionActivationSubsystem extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-projection-activation-subsystem")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentProjectionActivationSubsystem())
}
