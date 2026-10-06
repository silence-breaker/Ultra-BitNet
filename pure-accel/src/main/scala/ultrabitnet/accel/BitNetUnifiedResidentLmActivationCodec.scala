package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Production geometry for the dense LM activation image in QueryLmAlias. */
case class BitNetUnifiedResidentLmActivationCodecConfig(
    resident: BitNetResidentStreamingLmHeadConfig =
      BitNetResidentStreamingLmHeadConfig(),
    scratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig(),
    coupledMatrixOutputs: Boolean = false) {
  val model: BitNetConfig = resident.model
  val lm: BitNetStreamingLmHeadConfig = resident.streaming
  val quant: BitNetSharedDynamicI8QuantizerConfig = resident.quant
  val feeder: BitNetContinuousMatrixFeederConfig = resident.linear.feeder
  val layout: BitNetUnifiedResidentTensorScratchpadLayout =
    scratchpad.layout

  val denseRowFeatures: Int = layout.bankCount *
    layout.packedFeaturesPerWord * layout.nativeSymbolCount
  val tilesPerDenseRow: Int = denseRowFeatures / lm.tileFeatures
  val denseRowCount: Int = lm.tileCount / tilesPerDenseRow
  val denseRowWidth: Int = scala.math.max(1, log2Up(denseRowCount))
  val acceptedCountWidth: Int = log2Up(model.hiddenSize + 1)

  require(model.hiddenSize == 2560,
    "the production LM activation image is exactly 2560 i8 values")
  require(model.tokenParallelism == 2 || model.tokenParallelism == 4)
  require((lm.tileFeatures == 64 && lm.tileCount == 40) ||
    (lm.tileFeatures == 128 && lm.tileCount == 20))
  require((tilesPerDenseRow == 2 || tilesPerDenseRow == 4) &&
    denseRowCount == 10)
  require(denseRowFeatures == 256)
  require(layout.bankCount == 32)
  require(layout.packedFeaturesPerWord == 4)
  require(layout.packedWidth == 8 && layout.wordWidth == 64)
  require(layout.nativeSymbolCount == 2)
  require(layout.queryLmAlias.rowCount >= denseRowCount)
  require(quant.tokenLanes == model.tokenParallelism)
  require(quant.maxVectorLength >= model.hiddenSize)
  require(quant.tagWidth <= feeder.matrix.sequenceIdWidth)
}

/** Authorises one exact 2560-feature capture from the shared quantizer. */
case class BitNetUnifiedResidentLmCaptureCommand(
    cfg: BitNetUnifiedResidentLmActivationCodecConfig) extends Bundle {
  val tag = UInt(cfg.quant.tagWidth bits)
  val epoch = UInt(cfg.quant.epochWidth bits)
}

case class BitNetUnifiedResidentLmCaptureCompletion(
    cfg: BitNetUnifiedResidentLmActivationCodecConfig) extends Bundle {
  val tag = UInt(cfg.quant.tagWidth bits)
  val epoch = UInt(cfg.quant.epochWidth bits)
  val acceptedFeatures = UInt(cfg.acceptedCountWidth bits)
  val committedRows = UInt(log2Up(cfg.denseRowCount + 1) bits)
  val dequantFactorQ16 = UInt(32 bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/** One vocabulary row-pair reuses all twenty K tiles of the resident image. */
case class BitNetUnifiedResidentLmReplayCommand(
    cfg: BitNetUnifiedResidentLmActivationCodecConfig) extends Bundle {
  private val matrix = cfg.feeder.matrix
  val imageTag = UInt(cfg.quant.tagWidth bits)
  val imageEpoch = UInt(cfg.quant.epochWidth bits)
  val sequenceId = UInt(matrix.sequenceIdWidth bits)
  val pairOrdinal = UInt(cfg.lm.pairOrdinalWidth bits)
  val beatOrdinalBase = UInt(matrix.beatCountWidth bits)
}

case class BitNetUnifiedResidentLmReplayCompletion(
    cfg: BitNetUnifiedResidentLmActivationCodecConfig) extends Bundle {
  val sequenceId = UInt(cfg.feeder.matrix.sequenceIdWidth bits)
  val pairOrdinal = UInt(cfg.lm.pairOrdinalWidth bits)
  val acceptedWeightTiles = UInt(log2Up(cfg.lm.tileCount + 1) bits)
  val emittedTiles = UInt(log2Up(cfg.lm.tileCount + 1) bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

object BitNetUnifiedResidentLmActivationFault {
  val None: Int = 0x00
  val CaptureMetadata: Int = 0xb0
  val CaptureFraming: Int = 0xb1
  val CaptureFactor: Int = 0xb2
  val CaptureInactiveLane: Int = 0xb3
  val QuantCompletion: Int = 0xb4
  val BackendWrite: Int = 0xb5
  val BackendWriteMetadata: Int = 0xb6
  val InvalidReplay: Int = 0xb7
  val BackendRead: Int = 0xb8
  val BackendReadMetadata: Int = 0xb9
  val WeightMetadata: Int = 0xba
  val StaleLease: Int = 0xbb
  val UnexpectedBackendResponse: Int = 0xbc
  val Aborted: Int = 0xbd
}

/** Scala oracle for the owner-specific dense packing used by this codec. */
object BitNetUnifiedResidentLmDenseLayout {
  case class Location(tile: Int, element: Int, localRow: Int, bank: Int,
      wordFeature: Int, wordToken: Int)

  def location(feature: Int): Location = {
    require(feature >= 0 && feature < 2560)
    val tile = feature >> 7
    val element = feature & 127
    Location(
      tile = tile,
      element = element,
      localRow = tile >> 1,
      bank = element >> 2,
      wordFeature = element & 3,
      wordToken = tile & 1)
  }
}

/**
  * Endpoint codec between the shared dynamic-i8 service, the unified
  * resident tensor and the continuous LM matrix feeder.  Capture staging is
  * two 1,024-bit native byte-writeable BRAM rows.  Replay retains the shared
  * packed response at its endpoint and does not copy it back through BRAM.
  *
  * Capture uses the token dimension of one packed word as a dense tile lane:
  * two consecutive 128-byte LM tiles occupy one physical 32-bank row.  Thus
  * the complete 2560-byte image commits as ten authenticated rows.  The
  * external alias lease remains owned by the future LM-head subsystem; this
  * block only samples `leaseValid`/`leaseGeneration`, fixes every semantic
  * command owner to LmHead and fails closed if that generation becomes stale.
  *
  * Replay holds each 2,048-bit packed response at its Stream source for two
  * LM tiles.  The selected token lane drives the activation sink directly;
  * the source response is released only when lane one's atomic weight/tile
  * join completes.  This keeps replay out of the capture BRAMs and avoids both
  * a 2,048-bit row register and a second 1,024-bit staging datapath.  A weight
  * tile and its activation tile share a two-sink pending barrier so independent
  * backpressure cannot advance or duplicate either side.
  */
class BitNetUnifiedResidentLmActivationCodec(
    cfg: BitNetUnifiedResidentLmActivationCodecConfig =
      BitNetUnifiedResidentLmActivationCodecConfig(),
    useVendorSegmentRam: Boolean = true) extends Component {
  private val model = cfg.model
  private val lm = cfg.lm
  private val q = cfg.quant
  private val feeder = cfg.feeder
  private val matrix = feeder.matrix
  private val layout = cfg.layout

  val io = new Bundle {
    val leaseValid = in Bool()
    val leaseGeneration = in UInt(layout.aliasGenerationWidth bits)
    // Job-scoped cancellation.  Idle cancellation is deliberately inert.
    // Once a Stream request has been made visible, the codec keeps it stable
    // through its terminal handshake and only suppresses later requests.
    val abort = in Bool()

    val captureCommand = slave(Stream(
      BitNetUnifiedResidentLmCaptureCommand(cfg)))
    val quantOutput = slave(Stream(
      BitNetSharedDynamicQuantResultBeat(q)))
    val quantDone = slave(Stream(BitNetSharedDynamicQuantDone(q)))
    val captureCompletion = master(Stream(
      BitNetUnifiedResidentLmCaptureCompletion(cfg)))

    val replayCommand = slave(Stream(
      BitNetUnifiedResidentLmReplayCommand(cfg)))
    val i8WeightInput = slave(Stream(BitNetI8WeightBeat(feeder)))
    val matrixI8Weight = master(Stream(BitNetI8WeightBeat(feeder)))
    val matrixActivation = master(Stream(
      BitNetNormalizedActivationTile(feeder)))
    val replayCompletion = master(Stream(
      BitNetUnifiedResidentLmReplayCompletion(cfg)))

    val packedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(cfg.scratchpad)))
    val packedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(cfg.scratchpad)))
    val packedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(cfg.scratchpad)))
    val packedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(cfg.scratchpad)))

    val imageReady = out Bool()
    val imageTag = out UInt(q.tagWidth bits)
    val imageEpoch = out UInt(q.epochWidth bits)
    val imageGeneration = out UInt(layout.aliasGenerationWidth bits)
    val imageDequantFactorQ16 = out UInt(32 bits)
    val captureActive = out Bool()
    val replayActive = out Bool()
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val authenticatedWriteCount = out UInt(32 bits)
    val authenticatedReadCount = out UInt(32 bits)
    val completedCaptures = out UInt(32 bits)
    val completedReplays = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Capture, CaptureWaitDone, Replay = newElement()
  }
  val state = Reg(State()) init State.Idle

  val stickyFault = RegInit(False)
  val authenticatedWriteCounter = Reg(UInt(32 bits)) init 0
  val authenticatedReadCounter = Reg(UInt(32 bits)) init 0
  val completedCaptureCounter = Reg(UInt(32 bits)) init 0
  val completedReplayCounter = Reg(UInt(32 bits)) init 0

  val imageReadyReg = RegInit(False)
  val imageTagReg = Reg(UInt(q.tagWidth bits)) init 0
  val imageEpochReg = Reg(UInt(q.epochWidth bits)) init 0
  val imageGenerationReg =
    Reg(UInt(layout.aliasGenerationWidth bits)) init 0
  val imageFactorReg = Reg(UInt(32 bits)) init 0

  io.imageReady := imageReadyReg
  io.imageTag := imageTagReg
  io.imageEpoch := imageEpochReg
  io.imageGeneration := imageGenerationReg
  io.imageDequantFactorQ16 := imageFactorReg
  io.captureActive := state === State.Capture ||
    state === State.CaptureWaitDone
  io.replayActive := state === State.Replay
  io.quarantined := stickyFault
  io.protocolFaultSticky := stickyFault
  io.authenticatedWriteCount := authenticatedWriteCounter
  io.authenticatedReadCount := authenticatedReadCounter
  io.completedCaptures := completedCaptureCounter
  io.completedReplays := completedReplayCounter

  // Once an LM lease is relinquished, the aliased physical rows no longer
  // authenticate the old image even if no replay command is active.
  when(imageReadyReg && (!io.leaseValid ||
      io.leaseGeneration =/= imageGenerationReg)) {
    imageReadyReg := False
  }

  val captureCompletionValid = RegInit(False)
  val captureCompletionPayload = Reg(
    BitNetUnifiedResidentLmCaptureCompletion(cfg)) init
      BitNetUnifiedResidentLmCaptureCompletion(cfg).getZero
  io.captureCompletion.valid := captureCompletionValid
  io.captureCompletion.payload := captureCompletionPayload
  when(io.captureCompletion.fire) {
    captureCompletionValid := False
  }

  val replayCompletionValid = RegInit(False)
  val replayCompletionPayload = Reg(
    BitNetUnifiedResidentLmReplayCompletion(cfg)) init
      BitNetUnifiedResidentLmReplayCompletion(cfg).getZero
  io.replayCompletion.valid := replayCompletionValid
  io.replayCompletion.payload := replayCompletionPayload
  when(io.replayCompletion.fire) {
    replayCompletionValid := False
  }

  val allBanks = B((BigInt(1) << layout.bankCount) - 1,
    layout.bankCount bits)

  // ------------------------------------------------------------------
  // Capture: 2560 scalar results -> twenty 1024-bit bank-group commits.
  // ------------------------------------------------------------------
  val captureTag = Reg(UInt(q.tagWidth bits)) init 0
  val captureEpoch = Reg(UInt(q.epochWidth bits)) init 0
  val captureGeneration =
    Reg(UInt(layout.aliasGenerationWidth bits)) init 0
  val captureExpected = Reg(UInt(q.lengthWidth bits)) init 0
  val captureAccepted = Reg(UInt(cfg.acceptedCountWidth bits)) init 0
  val captureCommittedRows =
    Reg(UInt(log2Up(cfg.denseRowCount + 1) bits)) init 0
  val captureFactor = Reg(UInt(32 bits)) init 0
  val captureFactorValid = RegInit(False)
  val capturePoison = RegInit(False)
  val captureFaultCode = Reg(Bits(8 bits)) init 0
  // A visible write may already have been selected by the packed-client
  // router even though ready is still low.  Remember that offer so abort can
  // never withdraw it; quantOutput is itself a retained Stream and therefore
  // keeps the final byte and metadata stable until the same terminal fire.
  val captureWriteOffered = RegInit(False)
  private val packedSegmentCount = layout.packedTransportSegmentCount
  private val packedSegmentWordCount = layout.bankCount / packedSegmentCount
  require(packedSegmentCount == 2)
  require(packedSegmentWordCount == 16)
  private val captureMetadataGenerationLow = 0
  private val captureMetadataSegmentLow = layout.aliasGenerationWidth
  private val captureMetadataRowLow =
    captureMetadataSegmentLow + log2Up(packedSegmentCount)
  private val captureMetadataWidth =
    captureMetadataRowLow + layout.localRowWidth
  private val segmentBufferCfg = BitNetLmPackedSegmentBufferConfig(
    segmentCount = packedSegmentCount,
    // Multiple logical segment groups share the same sixteen physical BRAMs:
    // 64x8 still occupies one RAMB36 per bank, while the second group hides
    // the prior row's bounded packed-write/post-fire retirement latency.
    ringDepth = 8,
    metadataWidth = captureMetadataWidth,
    scatterCapture = true)

  // The two logical 1,024-bit segments are scattered directly into native
  // byte-writeable BRAM banks.  Capture is already one byte per codec cycle,
  // so the first integration uses the current clock on both sides of the
  // reusable CDC-safe buffer without changing top-level clock interfaces.
  val segmentBuffer = new BitNetLmPackedSegmentBuffer(
    segmentBufferCfg,
    useVendorByteRam = useVendorSegmentRam,
    useCurrentClockDomain = true)
  segmentBuffer.setName("lmCaptureSegmentBuffer")
  private val codecClock = ClockDomain.current.readClockWire
  private val codecResetActive = ClockDomain.current.isResetActive
  private val codecResetn = !codecResetActive
  segmentBuffer.io.slowClk := codecClock
  segmentBuffer.io.fastClk := codecClock
  segmentBuffer.io.slowResetn := codecResetn
  segmentBuffer.io.fastResetn := codecResetn
  // Replay is deliberately zero-copy: capture owns all BRAM write ports, and
  // a retained packedReadData payload is the live source for two LM tiles.
  segmentBuffer.io.replaySegment.valid := False
  segmentBuffer.io.replaySegment.payload.metadata := 0
  segmentBuffer.io.replaySegment.payload.data := 0

  val captureWritePostFireWindow = RegInit(False)
  val captureLastWriteRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val captureLastWriteSegment =
    Reg(UInt(log2Up(packedSegmentCount) bits)) init 0

  io.captureCommand.ready := state === State.Idle &&
    !captureCompletionValid && !replayCompletionValid && !stickyFault &&
    segmentBuffer.io.captureIdle && segmentBuffer.io.readEmpty &&
    !segmentBuffer.io.writeFull
  when(io.captureCommand.fire) {
    state := State.Capture
    imageReadyReg := False
    captureTag := io.captureCommand.tag
    captureEpoch := io.captureCommand.epoch
    captureGeneration := io.leaseGeneration
    captureExpected := 0
    captureAccepted := 0
    captureCommittedRows := 0
    captureFactor := 0
    captureFactorValid := False
    capturePoison := !io.leaseValid
    captureFaultCode := B(BitNetUnifiedResidentLmActivationFault.None,
      8 bits)
    captureWriteOffered := False
    captureWritePostFireWindow := False
    when(!io.leaseValid) {
      captureFaultCode := B(
        BitNetUnifiedResidentLmActivationFault.StaleLease, 8 bits)
      stickyFault := True
    }
  }

  val captureLeaseHealthy = io.leaseValid &&
    io.leaseGeneration === captureGeneration
  val captureAtFirst = captureExpected === 0
  val captureAtLast = captureExpected === model.hiddenSize - 1
  val captureLocalRow = (captureExpected |>> 8)
    .resize(layout.localRowWidth)
  val captureBank = captureExpected(6 downto 2)
    .resize(layout.bankWidth)
  val captureWordFeature = captureExpected(1 downto 0)
  val captureWordToken = captureExpected(7)
  val captureSegmentIndex = captureBank(4).asUInt
  val captureSegmentWord = captureBank(3 downto 0)
  val captureAtSegmentEnd = captureWordToken &&
    captureSegmentWord.andR && captureWordFeature.andR

  val captureMetadataLegal =
      io.quantOutput.tag === captureTag &&
      io.quantOutput.epoch === captureEpoch &&
      io.quantOutput.featureIndex === captureExpected.resize(q.indexWidth) &&
      io.quantOutput.tokenMask === B(1, q.tokenLanes bits)
  val captureFramingLegal =
    io.quantOutput.first === captureAtFirst &&
      io.quantOutput.last === captureAtLast
  val captureFactorLegal = !captureFactorValid ||
    io.quantOutput.dequantFactorQ16(0) === captureFactor
  val captureInactiveLanesZero = (1 until q.tokenLanes).map { lane =>
    io.quantOutput.value(lane) === 0
  }.reduce(_ && _)

  val incomingCaptureFault = Bits(8 bits)
  incomingCaptureFault := B(
    BitNetUnifiedResidentLmActivationFault.None, 8 bits)
  when(!captureMetadataLegal) {
    incomingCaptureFault := B(
      BitNetUnifiedResidentLmActivationFault.CaptureMetadata, 8 bits)
  } elsewhen(!captureFramingLegal) {
    incomingCaptureFault := B(
      BitNetUnifiedResidentLmActivationFault.CaptureFraming, 8 bits)
  } elsewhen(!captureFactorLegal) {
    incomingCaptureFault := B(
      BitNetUnifiedResidentLmActivationFault.CaptureFactor, 8 bits)
  } elsewhen(!captureInactiveLanesZero) {
    incomingCaptureFault := B(
      BitNetUnifiedResidentLmActivationFault.CaptureInactiveLane, 8 bits)
  }
  val incomingCaptureLegal = incomingCaptureFault === 0

  val captureOwnerActive = state === State.Capture ||
    state === State.CaptureWaitDone
  val bufferedMetadata = segmentBuffer.io.segment.payload.metadata
  val bufferedGeneration = bufferedMetadata(
    captureMetadataSegmentLow - 1 downto
      captureMetadataGenerationLow).asUInt
  val bufferedSegmentIndex = bufferedMetadata(
    captureMetadataRowLow - 1 downto
      captureMetadataSegmentLow).asUInt
  val bufferedLocalRow = bufferedMetadata(
    captureMetadataWidth - 1 downto captureMetadataRowLow).asUInt
  val expectedWriteFaultMetadata =
    io.packedWriteFault.region ===
      BitNetUnifiedResidentTensorRegion.QueryLmAlias &&
    io.packedWriteFault.localRow === Mux(captureWritePostFireWindow,
        captureLastWriteRow, bufferedLocalRow) &&
    io.packedWriteFault.segmentIndex === Mux(captureWritePostFireWindow,
      captureLastWriteSegment, bufferedSegmentIndex)
  val writeFaultNow = io.packedWriteFault.valid
  val captureWriteFaultNow = writeFaultNow && captureOwnerActive
  val captureWriteFaultCodeNow = Bits(8 bits)
  captureWriteFaultCodeNow := Mux(expectedWriteFaultMetadata,
    B(BitNetUnifiedResidentLmActivationFault.BackendWrite, 8 bits),
    B(BitNetUnifiedResidentLmActivationFault.BackendWriteMetadata,
      8 bits))
  io.packedWriteFault.ready := True

  val captureAbortNow = io.abort &&
    captureOwnerActive
  val captureBufferFaultNow = segmentBuffer.io.protocolFaultSticky
  // Once poisoned, the exact accepted 2,560-beat quantizer transaction still
  // enters the segment buffer with internally generated legal framing.  Zero
  // data prevents corrupted user payload from reaching any unoffered write.
  val capturePoisonNow = capturePoison || !captureLeaseHealthy ||
    (io.quantOutput.valid && !incomingCaptureLegal) ||
    captureWriteFaultNow || captureAbortNow || captureBufferFaultNow

  val captureSegmentMetadata = Bits(captureMetadataWidth bits)
  captureSegmentMetadata := 0
  captureSegmentMetadata(
    captureMetadataSegmentLow - 1 downto
      captureMetadataGenerationLow) := captureGeneration.asBits
  captureSegmentMetadata(
    captureMetadataRowLow - 1 downto
      captureMetadataSegmentLow) := captureSegmentIndex.asBits
  captureSegmentMetadata(
    captureMetadataWidth - 1 downto captureMetadataRowLow) :=
      captureLocalRow.asBits
  val capturePhysicalByte =
    (captureSegmentWord ## captureWordFeature ## captureWordToken).asUInt
  val captureAtSegmentStart = !captureSegmentWord.orR &&
    !captureWordFeature.orR && !captureWordToken

  segmentBuffer.io.captureByte.valid :=
    state === State.Capture && io.quantOutput.valid
  segmentBuffer.io.captureByte.payload.metadata := captureSegmentMetadata
  segmentBuffer.io.captureByte.payload.slotIndex := captureSegmentIndex
  segmentBuffer.io.captureByte.payload.byteIndex := capturePhysicalByte
  segmentBuffer.io.captureByte.payload.data := Mux(
    capturePoisonNow,
    B(0, layout.packedWidth bits),
    io.quantOutput.value(0).asBits)
  segmentBuffer.io.captureByte.payload.first := captureAtSegmentStart
  segmentBuffer.io.captureByte.payload.last := captureAtSegmentEnd
  // Semantic capture faults are drained as legal zero data.  This bit is
  // reserved for a genuine segment-buffer framing failure, which is fail-stop.
  segmentBuffer.io.captureByte.payload.fault := False
  io.quantOutput.ready := state === State.Capture &&
    segmentBuffer.io.captureByte.ready

  io.packedWrite.valid :=
    captureOwnerActive && segmentBuffer.io.segment.valid &&
      !captureWritePostFireWindow && !writeFaultNow &&
      (captureWriteOffered || !capturePoisonNow)
  io.packedWrite.region :=
    BitNetUnifiedResidentTensorRegion.QueryLmAlias
  io.packedWrite.localRow := bufferedLocalRow
  io.packedWrite.segmentIndex := bufferedSegmentIndex
  io.packedWrite.symbolMask := B(
    (BigInt(1) << layout.nativeSymbolCount) - 1,
    layout.nativeSymbolCount bits)
  io.packedWrite.aliasOwner :=
    BitNetUnifiedResidentTensorAliasOwner.LmHead
  io.packedWrite.aliasGeneration := bufferedGeneration
  for (word <- 0 until packedSegmentWordCount) {
    io.packedWrite.segmentWords(word) := segmentBuffer.io.segment.payload.data(
      (word + 1) * layout.wordWidth - 1 downto word * layout.wordWidth)
  }

  // A poisoned, never-offered output is reclaimed locally.  Once packedWrite
  // has been visible while stalled, only its real fire/fault may release the
  // buffer row, preserving the Stream stability contract across abort.
  val discardBufferedSegment = captureOwnerActive &&
    segmentBuffer.io.segment.valid && capturePoisonNow &&
    !captureWriteOffered
  val captureBufferedSegmentReady = discardBufferedSegment ||
    io.packedWrite.fire ||
    (io.packedWriteFault.fire && captureWriteOffered)
  segmentBuffer.io.segment.ready := Mux(
    state === State.Replay,
    True,
    captureBufferedSegmentReady)

  when(io.packedWrite.valid && !io.packedWrite.ready) {
    captureWriteOffered := True
  }
  when(io.packedWrite.fire || io.packedWriteFault.fire) {
    captureWriteOffered := False
  }
  when(io.packedWrite.fire) {
    captureWritePostFireWindow := True
    captureLastWriteRow := io.packedWrite.localRow
    captureLastWriteSegment := io.packedWrite.segmentIndex
  }
  when(captureWritePostFireWindow && !io.packedWriteFault.valid) {
    captureWritePostFireWindow := False
    authenticatedWriteCounter := authenticatedWriteCounter + 1
    when(captureLastWriteSegment === packedSegmentCount - 1) {
      captureCommittedRows := captureCommittedRows + 1
    }
  }
  when(io.packedWriteFault.fire) {
    captureWritePostFireWindow := False
  }

  when(captureAbortNow && !capturePoison) {
    capturePoison := True
    captureFaultCode := B(
      BitNetUnifiedResidentLmActivationFault.Aborted, 8 bits)
    imageReadyReg := False
  }

  when(state === State.Capture && !captureLeaseHealthy &&
      !capturePoison) {
    capturePoison := True
    captureFaultCode := B(
      BitNetUnifiedResidentLmActivationFault.StaleLease, 8 bits)
    imageReadyReg := False
    stickyFault := True
  }

  when(io.packedWriteFault.fire) {
    stickyFault := True
    // A semantic frontend is expected to make write.ready mean authenticated
    // commit.  Nevertheless, fail closed if a protocol-violating delayed
    // fault arrives after the final feature retired but before quantDone is
    // accepted: CaptureWaitDone still owns that exact row transaction.
    when(state === State.Capture || state === State.CaptureWaitDone) {
      capturePoison := True
      imageReadyReg := False
      when(!capturePoison) {
        captureFaultCode := captureWriteFaultCodeNow
      }
    }
  }

  when(captureBufferFaultNow) {
    stickyFault := True
    imageReadyReg := False
    when(captureOwnerActive && !capturePoison) {
      capturePoison := True
      captureFaultCode := B(
        BitNetUnifiedResidentLmActivationFault.CaptureFraming, 8 bits)
    }
  }

  when(io.quantOutput.fire) {
    captureAccepted := captureAccepted + 1
    when(!captureFactorValid) {
      captureFactor := io.quantOutput.dequantFactorQ16(0)
      captureFactorValid := True
    }

    when(!incomingCaptureLegal && !capturePoison) {
      capturePoison := True
      captureFaultCode := incomingCaptureFault
      imageReadyReg := False
      stickyFault := True
    }

    when(captureAtLast) {
      state := State.CaptureWaitDone
    } otherwise {
      captureExpected := captureExpected + 1
    }
  }

  val captureWriteDrainComplete = segmentBuffer.io.captureIdle &&
    segmentBuffer.io.readEmpty && !segmentBuffer.io.writeFull &&
    !captureWriteOffered && !captureWritePostFireWindow &&
    !segmentBuffer.io.segment.valid
  io.quantDone.ready := state === State.CaptureWaitDone &&
    captureWriteDrainComplete &&
    (capturePoison || captureAbortNow ||
      captureCommittedRows === cfg.denseRowCount)
  val quantDoneLegal =
    io.quantDone.tag === captureTag &&
      io.quantDone.epoch === captureEpoch &&
      io.quantDone.vectorLength === model.hiddenSize &&
      io.quantDone.emittedFeatures === model.hiddenSize &&
      io.quantDone.success && io.quantDone.faultCode === 0 &&
      captureAccepted === model.hiddenSize &&
      captureCommittedRows === cfg.denseRowCount &&
      captureFactorValid && captureFactor =/= 0 &&
      captureLeaseHealthy && !capturePoison && !captureWriteFaultNow &&
      !captureAbortNow

  when(io.quantDone.fire) {
    val success = quantDoneLegal
    val abortOnlyTerminal =
      (captureAbortNow || (capturePoison && captureFaultCode ===
        B(BitNetUnifiedResidentLmActivationFault.Aborted, 8 bits))) &&
        io.quantDone.tag === captureTag &&
        io.quantDone.epoch === captureEpoch &&
        io.quantDone.vectorLength === model.hiddenSize &&
        io.quantDone.emittedFeatures === model.hiddenSize &&
        io.quantDone.success && io.quantDone.faultCode === 0 &&
        captureAccepted === model.hiddenSize && captureFactorValid &&
        captureFactor =/= 0 && captureLeaseHealthy &&
        !captureWriteFaultNow
    captureCompletionValid := True
    captureCompletionPayload.tag := captureTag
    captureCompletionPayload.epoch := captureEpoch
    captureCompletionPayload.acceptedFeatures := captureAccepted
    captureCompletionPayload.committedRows := captureCommittedRows
    captureCompletionPayload.dequantFactorQ16 := captureFactor
    captureCompletionPayload.success := success
    captureCompletionPayload.faultCode := B(
      BitNetUnifiedResidentLmActivationFault.None, 8 bits)
    when(!success) {
      captureCompletionPayload.faultCode :=
        B(BitNetUnifiedResidentLmActivationFault.QuantCompletion, 8 bits)
      when(capturePoison) {
        captureCompletionPayload.faultCode := captureFaultCode
      }
      when(captureAbortNow) {
        captureCompletionPayload.faultCode := B(
          BitNetUnifiedResidentLmActivationFault.Aborted, 8 bits)
      }
      // Register assignments above become visible only after this edge.  The
      // direct same-cycle term is therefore required when delayed write-fault
      // and quantDone handshake together.
      when(captureWriteFaultNow) {
        captureCompletionPayload.faultCode := captureWriteFaultCodeNow
      }
      when(!abortOnlyTerminal) { stickyFault := True }
      imageReadyReg := False
    } otherwise {
      imageReadyReg := True
      imageTagReg := captureTag
      imageEpochReg := captureEpoch
      imageGenerationReg := captureGeneration
      imageFactorReg := captureFactor
      completedCaptureCounter := completedCaptureCounter + 1
    }
    state := State.Idle
  }

  // ------------------------------------------------------------------
  // Replay: ten retained row reads feed twenty atomic weight/tile joins.
  // ------------------------------------------------------------------
  val replaySequence = Reg(UInt(matrix.sequenceIdWidth bits)) init 0
  val replayPair = Reg(UInt(lm.pairOrdinalWidth bits)) init 0
  val replayBeatBase = Reg(UInt(matrix.beatCountWidth bits)) init 0
  val replayGeneration =
    Reg(UInt(layout.aliasGenerationWidth bits)) init 0
  val nextReadRow = Reg(UInt(log2Up(cfg.denseRowCount + 1) bits)) init 0
  // `valid` itself is a claimable offer at the shared router boundary.  Keep
  // an offered command until fire; abort only prevents the next offer.
  val readCommandOffered = RegInit(False)
  val readOutstanding = RegInit(False)
  val expectedReadRow = Reg(UInt(layout.localRowWidth bits)) init 0
  // A retained 2,048-bit read response is consumed as two live tile lanes.
  // No activation payload is copied into a register or the capture BRAM ring.
  val replayLiveLane =
    Reg(UInt(log2Up(cfg.tilesPerDenseRow) bits)) init 0
  val expectedK = Reg(UInt(lm.tileIndexWidth bits)) init 0
  val acceptedWeights =
    Reg(UInt(log2Up(lm.tileCount + 1) bits)) init 0
  val emittedTiles = Reg(UInt(log2Up(lm.tileCount + 1) bits)) init 0
  val replayDrain = RegInit(False)
  val replayFaultTileNeeded = RegInit(False)
  val replayFaultCode = Reg(Bits(8 bits)) init 0
  val replayWeightsDrained = RegInit(False)
  val replaySuccessPending = RegInit(False)

  val joinValid = RegInit(False)
  // The standalone codec permits independently backpressured sinks and needs
  // to retain the complete weight beat after either side fires.  Canonical
  // production connects both outputs to one feeder whose ready equations are
  // identical for I8 traffic.  In that topology the upstream Stream retains
  // its payload until both sinks fire, so elaborating another 2,048-bit copy
  // only creates a wide register/mux bank without adding capacity.
  val joinWeightPending = if (!cfg.coupledMatrixOutputs) RegInit(False) else null
  val joinActivationPending =
    if (!cfg.coupledMatrixOutputs) RegInit(False) else null
  val joinWeight = if (!cfg.coupledMatrixOutputs) {
    Reg(BitNetI8WeightBeat(feeder)) init
      BitNetI8WeightBeat(feeder).getZero
  } else null
  val joinK = Reg(UInt(lm.tileIndexWidth bits)) init 0
  val joinFault = RegInit(False)
  val joinUsesLiveRow = RegInit(False)

  val replayLastOrdinal =
    io.replayCommand.beatOrdinalBase.resize(matrix.beatCountWidth + 1) +
      U(lm.tileCount - 1, matrix.beatCountWidth + 1 bits)
  val replayCommandLegal = imageReadyReg && io.leaseValid &&
    io.leaseGeneration === imageGenerationReg &&
    io.replayCommand.imageTag === imageTagReg &&
    io.replayCommand.imageEpoch === imageEpochReg &&
    io.replayCommand.pairOrdinal < lm.pairCount &&
    !replayLastOrdinal.msb

  io.replayCommand.ready := state === State.Idle &&
    !captureCompletionValid && !replayCompletionValid && !stickyFault &&
    segmentBuffer.io.captureIdle && segmentBuffer.io.readEmpty &&
    !segmentBuffer.io.writeFull && !segmentBuffer.io.segment.valid
  when(io.replayCommand.fire) {
    when(replayCommandLegal) {
      state := State.Replay
      replaySequence := io.replayCommand.sequenceId
      replayPair := io.replayCommand.pairOrdinal
      replayBeatBase := io.replayCommand.beatOrdinalBase
      replayGeneration := imageGenerationReg
      nextReadRow := 0
      readCommandOffered := False
      readOutstanding := False
      replayLiveLane := 0
      expectedK := 0
      acceptedWeights := 0
      emittedTiles := 0
      replayDrain := io.abort
      replayFaultTileNeeded := io.abort
      replayFaultCode := Mux(io.abort,
        B(BitNetUnifiedResidentLmActivationFault.Aborted, 8 bits),
        B(BitNetUnifiedResidentLmActivationFault.None, 8 bits))
      replayWeightsDrained := False
      replaySuccessPending := False
      joinValid := False
      if (!cfg.coupledMatrixOutputs) {
        joinWeightPending := False
        joinActivationPending := False
      }
      joinUsesLiveRow := False
    } otherwise {
      replayCompletionValid := True
      replayCompletionPayload.sequenceId :=
        io.replayCommand.sequenceId
      replayCompletionPayload.pairOrdinal :=
        io.replayCommand.pairOrdinal
      replayCompletionPayload.acceptedWeightTiles := 0
      replayCompletionPayload.emittedTiles := 0
      replayCompletionPayload.success := False
      replayCompletionPayload.faultCode := B(
        BitNetUnifiedResidentLmActivationFault.InvalidReplay, 8 bits)
      stickyFault := True
    }
  }

  val replayLeaseHealthy = io.leaseValid &&
    io.leaseGeneration === replayGeneration
  val replayAbortNow = io.abort && state === State.Replay
  val replayHealthyNow = state === State.Replay && !replayDrain &&
    !replayAbortNow && replayLeaseHealthy

  io.packedRead.valid := state === State.Replay && !readOutstanding &&
    nextReadRow < cfg.denseRowCount &&
    (readCommandOffered || replayHealthyNow)
  io.packedRead.region :=
    BitNetUnifiedResidentTensorRegion.QueryLmAlias
  io.packedRead.localRow := nextReadRow.resize(layout.localRowWidth)
  io.packedRead.bankMask := allBanks
  io.packedRead.aliasOwner :=
    BitNetUnifiedResidentTensorAliasOwner.LmHead
  io.packedRead.aliasGeneration := replayGeneration

  when(io.packedRead.valid && !io.packedRead.ready) {
    readCommandOffered := True
  }

  when(io.packedRead.fire) {
    readCommandOffered := False
    readOutstanding := True
    expectedReadRow := nextReadRow.resize(layout.localRowWidth)
    nextReadRow := nextReadRow + 1
  }

  val readResponseMetadataLegal = readOutstanding &&
    io.packedReadData.region ===
      BitNetUnifiedResidentTensorRegion.QueryLmAlias &&
    io.packedReadData.localRow === expectedReadRow &&
    io.packedReadData.bankMask === allBanks
  val unexpectedReadResponse = io.packedReadData.valid &&
    !readOutstanding

  when(unexpectedReadResponse) {
    stickyFault := True
    imageReadyReg := False
    when(state === State.Replay && !replayDrain) {
      replayDrain := True
      replayFaultTileNeeded := acceptedWeights < lm.tileCount
      when(acceptedWeights === lm.tileCount && !joinValid) {
        replayWeightsDrained := True
      }
      replayFaultCode := B(
        BitNetUnifiedResidentLmActivationFault.UnexpectedBackendResponse,
        8 bits)
    }
  }

  when(state === State.Replay && !replayDrain && !replayLeaseHealthy) {
    replayDrain := True
    // A live bound join already owns the retained response and is allowed to
    // finish before the row is discarded.  Only an unbound failure emits the
    // single synthetic fault tile.
    replayFaultTileNeeded := !joinValid &&
      acceptedWeights < lm.tileCount
    when(acceptedWeights === lm.tileCount && !joinValid) {
      replayWeightsDrained := True
    }
    replayFaultCode := B(
      BitNetUnifiedResidentLmActivationFault.StaleLease, 8 bits)
    stickyFault := True
    imageReadyReg := False
  }

  when(replayAbortNow && !replayDrain) {
    replayDrain := True
    replayFaultTileNeeded := !joinValid && acceptedWeights < lm.tileCount
    when(acceptedWeights === lm.tileCount && !joinValid) {
      replayWeightsDrained := True
    }
    replayFaultCode := B(
      BitNetUnifiedResidentLmActivationFault.Aborted, 8 bits)
    imageReadyReg := False
  }

  val expectedLiveRow = (expectedK |>> log2Up(cfg.tilesPerDenseRow))
    .resize(layout.localRowWidth)
  val expectedLiveLane = expectedK(log2Up(cfg.tilesPerDenseRow) - 1 downto 0)
  val replayLivePositionLegal = expectedReadRow === expectedLiveRow &&
    replayLiveLane === expectedLiveLane
  val readResponseLiveMetadataLegal = readResponseMetadataLegal &&
    replayLivePositionLegal
  val readResponseLiveLegal = readResponseLiveMetadataLegal &&
    !io.packedReadData.fault && replayLeaseHealthy
  val replayLiveTileLegal = readOutstanding &&
    io.packedReadData.valid && readResponseLiveLegal

  val expectedWeightOrdinal =
    (replayBeatBase + acceptedWeights.resize(matrix.beatCountWidth)).resized
  val incomingWeightLegal =
      io.i8WeightInput.sequenceId === replaySequence &&
      io.i8WeightInput.beatOrdinal === expectedWeightOrdinal &&
      io.i8WeightInput.use === BitNetI8WeightUse.LmHead &&
      (0 until feeder.engineCount).map { engine =>
        io.i8WeightInput.engineOutputBase(engine) === replayPair
      }.reduce(_ && _) &&
      io.i8WeightInput.firstK === (expectedK === 0) &&
      io.i8WeightInput.lastK === (expectedK === lm.tileCount - 1) &&
      !io.i8WeightInput.fault

  val canCaptureHealthyWeight = !replaySuccessPending && !replayDrain &&
    !replayAbortNow && replayLeaseHealthy &&
    replayLiveTileLegal
  val canCaptureFaultWeight = replayDrain && replayFaultTileNeeded
  val canDiscardFaultWeight = replayDrain && !replayFaultTileNeeded
  val canOfferJoin = canCaptureHealthyWeight || canCaptureFaultWeight
  val freshCoupledOffer = io.i8WeightInput.valid && canOfferJoin &&
    !joinValid
  val coupledOfferValid = joinValid || freshCoupledOffer
  if (cfg.coupledMatrixOutputs) {
    // Both feeder inputs see the same `inputOpen` condition and each ready is
    // gated only by the other valid.  Requiring both readies here therefore
    // makes the two transfers atomic while leaving valid independent of ready.
    io.i8WeightInput.ready := state === State.Replay &&
      !replaySuccessPending &&
      ((coupledOfferValid && io.matrixI8Weight.ready &&
        io.matrixActivation.ready) ||
        (!joinValid && canDiscardFaultWeight))
  } else {
    io.i8WeightInput.ready := state === State.Replay &&
      !replaySuccessPending && !joinValid &&
      (canOfferJoin || canDiscardFaultWeight)
  }

  val weightInputFire = io.i8WeightInput.fire
  val captureJoin = weightInputFire &&
    (if (cfg.coupledMatrixOutputs) coupledOfferValid else canOfferJoin)
  val incomingJoinFault = replayDrain || !incomingWeightLegal ||
    !replayLeaseHealthy
  val liveJoinUsesLiveRow = canCaptureHealthyWeight &&
    !incomingJoinFault

  // Fall through a newly bound weight and the selected lane of the retained
  // 2,048-bit response directly to the two matrix sinks.  If both are ready,
  // the next lane can fire on the next cycle.  Independent backpressure
  // converts only that transaction into the retained join state below while
  // packedReadData remains held at its source.
  val activeJoinValid =
    if (cfg.coupledMatrixOutputs) coupledOfferValid else joinValid || captureJoin
  val activeJoinK = Mux(joinValid, joinK, expectedK)
  val activeJoinFault = Mux(joinValid, joinFault, incomingJoinFault)
  val activeJoinUsesLiveRow = Mux(
    joinValid, joinUsesLiveRow, liveJoinUsesLiveRow)
  val activeWeightPending = if (cfg.coupledMatrixOutputs) activeJoinValid else
    Mux(joinValid, joinWeightPending, captureJoin)
  val activeActivationPending = if (cfg.coupledMatrixOutputs) activeJoinValid else
    Mux(joinValid, joinActivationPending, captureJoin)

  io.matrixI8Weight.valid := activeJoinValid && activeWeightPending
  io.matrixI8Weight.sequenceId := (if (cfg.coupledMatrixOutputs)
    io.i8WeightInput.sequenceId else
    Mux(joinValid, joinWeight.sequenceId, io.i8WeightInput.sequenceId))
  io.matrixI8Weight.beatOrdinal := (if (cfg.coupledMatrixOutputs)
    io.i8WeightInput.beatOrdinal else
    Mux(joinValid, joinWeight.beatOrdinal, io.i8WeightInput.beatOrdinal))
  io.matrixI8Weight.use := (if (cfg.coupledMatrixOutputs)
    io.i8WeightInput.use else
    Mux(joinValid, joinWeight.use, io.i8WeightInput.use))
  io.matrixI8Weight.firstK := (if (cfg.coupledMatrixOutputs)
    io.i8WeightInput.firstK else
    Mux(joinValid, joinWeight.firstK, io.i8WeightInput.firstK))
  io.matrixI8Weight.lastK := (if (cfg.coupledMatrixOutputs)
    io.i8WeightInput.lastK else
    Mux(joinValid, joinWeight.lastK, io.i8WeightInput.lastK))
  for (engine <- 0 until feeder.engineCount;
       site <- 0 until feeder.matrix.base.siteCount) {
    io.matrixI8Weight.engineSiteWeight(engine)(site) :=
      (if (cfg.coupledMatrixOutputs)
        io.i8WeightInput.engineSiteWeight(engine)(site)
      else Mux(joinValid,
        joinWeight.engineSiteWeight(engine)(site),
        io.i8WeightInput.engineSiteWeight(engine)(site)))
  }
  for (engine <- 0 until feeder.engineCount) {
    io.matrixI8Weight.engineOutputBase(engine) :=
      (if (cfg.coupledMatrixOutputs)
        io.i8WeightInput.engineOutputBase(engine)
      else Mux(joinValid,
        joinWeight.engineOutputBase(engine),
        io.i8WeightInput.engineOutputBase(engine)))
  }
  io.matrixI8Weight.fault := activeJoinFault

  io.matrixActivation.valid := activeJoinValid && activeActivationPending
  io.matrixActivation.sequenceId := io.matrixI8Weight.sequenceId
  io.matrixActivation.beatOrdinal := io.matrixI8Weight.beatOrdinal
  io.matrixActivation.fault := activeJoinFault
  val replayPhysicalToken =
    (expectedLiveLane / (128 / lm.tileFeatures)).resize(
      log2Up(layout.nativeSymbolCount))
  val replayFeatureBase =
    (expectedLiveLane % (128 / lm.tileFeatures)).resize(
      log2Up(128 / lm.tileFeatures)) * lm.tileFeatures
  for (lane <- 0 until model.tokenParallelism;
       site <- 0 until lm.tileFeatures) {
    if (lane == 0) {
      val physicalFeature = replayFeatureBase + site
      io.matrixActivation.tokenActivation(lane)(site) := Mux(
        activeJoinFault,
        S(0, layout.packedWidth bits),
        io.packedReadData.words((physicalFeature >> 2).resized)
          .feature(physicalFeature(1 downto 0))(
            replayPhysicalToken))
    } else {
      io.matrixActivation.tokenActivation(lane)(site) := 0
    }
  }

  val weightRemainingAfter = activeWeightPending &&
    !io.matrixI8Weight.ready
  val activationRemainingAfter = activeActivationPending &&
    !io.matrixActivation.ready
  val joinCompletes = activeJoinValid && !weightRemainingAfter &&
    !activationRemainingAfter

  val liveJoinCompletes = joinCompletes && activeJoinUsesLiveRow
  val responseHasBoundLiveJoin = joinValid && joinUsesLiveRow
  val healthyResponseRetires = readOutstanding && readResponseLiveLegal &&
    replayHealthyNow && replayLiveLane.andR && liveJoinCompletes
  val failedResponseCanRetire = readOutstanding &&
    (!readResponseLiveLegal || !replayHealthyNow) &&
    (!responseHasBoundLiveJoin || liveJoinCompletes)

  // In a healthy replay the producer owns one response through both joined
  // lanes, including independent sink stalls on lane one.  Abort or
  // lease loss may discard the unconsumed lanes, but never before an already
  // bound live join has reached both matrix sinks.
  io.packedReadData.ready := !readOutstanding ||
    healthyResponseRetires || failedResponseCanRetire

  when(io.packedReadData.fire && readOutstanding) {
    when(readResponseLiveLegal && replayHealthyNow) {
      // A healthy response can fire only with lane three's completed join.
      authenticatedReadCounter := authenticatedReadCounter + 1
    } elsewhen((replayDrain || replayAbortNow) &&
        readResponseLiveMetadataLegal &&
        !io.packedReadData.fault && replayLeaseHealthy) {
      // A read accepted before abort still owns one response.  Consume that
      // response without reviving the image or converting cancellation into
      // a protocol quarantine.
    } otherwise {
      stickyFault := True
      imageReadyReg := False
      when(state === State.Replay && !replayDrain) {
        replayDrain := True
        replayFaultTileNeeded := !joinValid &&
          acceptedWeights < lm.tileCount
        when(acceptedWeights === lm.tileCount && !joinValid) {
          replayWeightsDrained := True
        }
        replayFaultCode := Mux(readResponseLiveMetadataLegal,
          Mux(replayLeaseHealthy,
            B(BitNetUnifiedResidentLmActivationFault.BackendRead, 8 bits),
            B(BitNetUnifiedResidentLmActivationFault.StaleLease, 8 bits)),
          B(BitNetUnifiedResidentLmActivationFault.BackendReadMetadata,
            8 bits))
      }
    }
    readOutstanding := False
    replayLiveLane := 0
  }

  when(liveJoinCompletes && !io.packedReadData.fire) {
    replayLiveLane := replayLiveLane + 1
  }

  if (cfg.coupledMatrixOutputs) {
    // Snapshot only the compact control needed to keep a visible, stalled
    // offer semantically stable across abort/lease changes.  The upstream
    // weight payload remains unconsumed and is therefore stable by Stream
    // contract until both feeder inputs accept it together.
    when(freshCoupledOffer && !joinCompletes) {
      joinValid := True
      joinK := expectedK
      joinFault := incomingJoinFault
      joinUsesLiveRow := liveJoinUsesLiveRow
    }
    when(captureJoin) {
      joinValid := False
      joinUsesLiveRow := False
    }
  } else {
    when(captureJoin) {
      joinValid := !joinCompletes
      joinWeightPending := weightRemainingAfter
      joinActivationPending := activationRemainingAfter
      joinWeight.sequenceId := io.i8WeightInput.sequenceId
      joinWeight.beatOrdinal := io.i8WeightInput.beatOrdinal
      joinWeight.use := io.i8WeightInput.use
      joinWeight.firstK := io.i8WeightInput.firstK
      joinWeight.lastK := io.i8WeightInput.lastK
      for (engine <- 0 until feeder.engineCount;
           site <- 0 until feeder.matrix.base.siteCount) {
        joinWeight.engineSiteWeight(engine)(site) :=
          io.i8WeightInput.engineSiteWeight(engine)(site)
      }
      for (engine <- 0 until feeder.engineCount) {
        joinWeight.engineOutputBase(engine) :=
          io.i8WeightInput.engineOutputBase(engine)
      }
      joinWeight.fault := incomingJoinFault
    }
  }
  when(captureJoin) {
    joinK := activeJoinK
    joinFault := activeJoinFault
    joinUsesLiveRow := activeJoinUsesLiveRow
    acceptedWeights := acceptedWeights + 1
    when(activeJoinFault) {
      replayDrain := True
      replayFaultTileNeeded := False
      // A synthetic fault tile is the bounded terminal for an ordinary
      // abort, not evidence of a semantic protocol violation.  Metadata or
      // lease corruption remains quarantine-worthy; backend/read faults have
      // already set stickyFault at their point of detection.
      when(!incomingWeightLegal || !replayLeaseHealthy) {
        stickyFault := True
      }
      when(!replayDrain) {
        replayFaultCode := Mux(replayLeaseHealthy,
          B(BitNetUnifiedResidentLmActivationFault.WeightMetadata, 8 bits),
          B(BitNetUnifiedResidentLmActivationFault.StaleLease, 8 bits))
      }
    }
  }
  if (!cfg.coupledMatrixOutputs) {
    when(joinValid && !captureJoin) {
      when(io.matrixI8Weight.fire) {
        joinWeightPending := False
      }
      when(io.matrixActivation.fire) {
        joinActivationPending := False
      }
      when(joinCompletes) {
        joinValid := False
        joinWeightPending := False
        joinActivationPending := False
        joinUsesLiveRow := False
      }
    }
  }

  // Once the single visible fault tile has been bound, consume exactly the
  // rest of the declared twenty-weight boundary without issuing more reads.
  val discardWeightFire = weightInputFire && canDiscardFaultWeight
  when(discardWeightFire) {
    acceptedWeights := acceptedWeights + 1
    when(acceptedWeights === lm.tileCount - 1) {
      replayWeightsDrained := True
    } otherwise {
      expectedK := expectedK + 1
    }
  }

  when(joinCompletes) {
    emittedTiles := emittedTiles + 1

    when(activeJoinK === lm.tileCount - 1) {
      when(activeJoinFault || replayDrain || replayAbortNow) {
        replayWeightsDrained := acceptedWeights === lm.tileCount ||
          (captureJoin && acceptedWeights === lm.tileCount - 1)
      } otherwise {
        replaySuccessPending := True
      }
    } otherwise {
      expectedK := expectedK + 1
    }
  }

  when(state === State.Replay &&
      (replayDrain || replayAbortNow || !replayLeaseHealthy)) {
    replaySuccessPending := False
  }

  val captureBufferIdle = segmentBuffer.io.captureIdle &&
    segmentBuffer.io.readEmpty && !segmentBuffer.io.writeFull &&
    !segmentBuffer.io.segment.valid
  val healthyReplayTerminal = state === State.Replay &&
    replaySuccessPending && !replayDrain && !replayAbortNow &&
    replayLeaseHealthy && !joinValid && !readCommandOffered &&
    !readOutstanding && replayLiveLane === 0 && captureBufferIdle
  when(healthyReplayTerminal) {
    replayCompletionValid := True
    replayCompletionPayload.sequenceId := replaySequence
    replayCompletionPayload.pairOrdinal := replayPair
    replayCompletionPayload.acceptedWeightTiles := acceptedWeights
    replayCompletionPayload.emittedTiles := emittedTiles
    replayCompletionPayload.success := True
    replayCompletionPayload.faultCode := B(
      BitNetUnifiedResidentLmActivationFault.None, 8 bits)
    completedReplayCounter := completedReplayCounter + 1
    replaySuccessPending := False
    state := State.Idle
    readCommandOffered := False
    readOutstanding := False
  }

  val drainedReplayTerminal = state === State.Replay && replayDrain &&
    replayWeightsDrained && !joinValid && !readCommandOffered &&
    !readOutstanding && replayLiveLane === 0 && captureBufferIdle
  when(drainedReplayTerminal) {
    replayCompletionValid := True
    replayCompletionPayload.sequenceId := replaySequence
    replayCompletionPayload.pairOrdinal := replayPair
    replayCompletionPayload.acceptedWeightTiles := acceptedWeights
    replayCompletionPayload.emittedTiles := emittedTiles
    replayCompletionPayload.success := False
    replayCompletionPayload.faultCode := replayFaultCode
    state := State.Idle
    readCommandOffered := False
    readOutstanding := False
  }
}

object GenerateBitNetUnifiedResidentLmActivationCodec extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-lm-activation-codec")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedResidentLmActivationCodec())
}
