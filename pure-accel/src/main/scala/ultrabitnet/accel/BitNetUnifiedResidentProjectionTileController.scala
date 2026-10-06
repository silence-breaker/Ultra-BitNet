package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetUnifiedResidentProjectionTileControllerConfig(
    tile: BitNetResidentActivationTileBufferConfig =
      BitNetResidentActivationTileBufferConfig(),
    scratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()) {
  require(tile.tokenLanes <= scratchpad.layout.tokenLanes)
  require(tile.bankCount == scratchpad.layout.bankCount)
  require(tile.featuresPerBankWord ==
    scratchpad.layout.packedFeaturesPerWord)
  require(tile.bankWordWidth <= scratchpad.layout.wordWidth)
  require(tile.bankCount * tile.featuresPerBankWord == 128)
  require(tile.tileFeatures == 64 || tile.tileFeatures == 128)
  require((tile.maxFeatures + 127) / 128 <=
    scratchpad.layout.projectionSlot0.rowCount)
  require(scratchpad.layout.projectionSlot0.rowCount ==
    scratchpad.layout.projectionSlot1.rowCount)
}

/** Controller-specific backend failures use codes above the legacy
  * quantizer/framing range.  The four-bit semantic write/read code remains
  * available on the south interface for detailed diagnostics.
  */
object BitNetUnifiedResidentProjectionFault {
  val BackendWrite: Int = 0xa7
  val BackendWriteMetadata: Int = 0xa8
  val BackendRead: Int = 0xa9
  val BackendReadMetadata: Int = 0xaa
}

/**
  * BRAM-backed ping/pong projection image controller.
  *
  * Its north interface is field-for-field compatible with
  * [[BitNetResidentActivationTileBuffer]].  The old 32-bank local Mem image
  * is replaced by the packed-i8 semantic interface of the unique unified
  * resident tensor frontend.  A fill accumulates two alternating 32-feature,
  * 1024-bit bank-group segments in sixteen native byte-writeable 64-bit RAM
  * banks.  This lets the next segment keep accepting one feature per cycle
  * while the previous segment is committed and audited, without retaining a
  * 4096-bit row or a 1024-bit fabric-register staging slot.
  *
  * A failed fill invalidates its target slot.  Once poisoned, the controller
  * emits no more packed writes and consumes exactly the remaining declared
  * feature beats, preserving the fixed image boundary for the next command.
  * Replay has one semantic read outstanding and returns its 32 packed words
  * directly as one 128-feature x TP4 normalized tile without another data
  * image or response buffer.
  */
class BitNetUnifiedResidentProjectionTileController(
    cfg: BitNetUnifiedResidentProjectionTileControllerConfig =
      BitNetUnifiedResidentProjectionTileControllerConfig(),
    useVendorSegmentRam: Boolean = true)
    extends Component {
  private val tileCfg = cfg.tile
  private val scratchpadCfg = cfg.scratchpad
  private val layout = scratchpadCfg.layout
  private val m = tileCfg.feeder.matrix
  private val b = m.base

  val io = new Bundle {
    val fillCommand = slave(Stream(
      BitNetResidentActivationFillCommand(tileCfg)))
    val quantized = slave(Stream(
      BitNetResidentActivationQuantBeat(tileCfg)))
    val fillCompletion = master(Stream(
      BitNetResidentActivationFillCompletion(tileCfg)))

    val replayCommand = slave(Stream(
      BitNetResidentActivationReplayCommand(tileCfg)))
    val tile = master(Stream(BitNetResidentActivationTile(tileCfg)))
    val replayCompletion = master(Stream(
      BitNetResidentActivationReplayCompletion(tileCfg)))

    // These four Streams connect directly to the one semantic frontend.
    val packedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(scratchpadCfg)))
    val packedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(scratchpadCfg)))
    val packedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(scratchpadCfg)))
    val packedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(scratchpadCfg)))

    val slotReady = out Bits(tileCfg.slotCount bits)
    val slotImageTag = out Vec(
      UInt(m.sequenceIdWidth bits), tileCfg.slotCount)
    val slotEpoch = out Vec(
      UInt(tileCfg.epochWidth bits), tileCfg.slotCount)
    val slotFeatureCount = out Vec(
      UInt(tileCfg.featureCountWidth bits), tileCfg.slotCount)
    val slotTokenMask = out Vec(
      Bits(tileCfg.tokenLanes bits), tileCfg.slotCount)
    val slotDequantFactorQ16 = out Vec(
      Vec(UInt(32 bits), tileCfg.tokenLanes), tileCfg.slotCount)
    val fillActive = out Bool()
    val replayActive = out Bool()
    val protocolFaultSticky = out Bool()
    val completedFills = out UInt(32 bits)
    val completedReplays = out UInt(32 bits)
  }

  private def projectionRegion(slot: UInt):
      BitNetUnifiedResidentTensorRegion.C = {
    val region = BitNetUnifiedResidentTensorRegion()
    region := BitNetUnifiedResidentTensorRegion.ProjectionSlot0
    when(slot === 1) {
      region := BitNetUnifiedResidentTensorRegion.ProjectionSlot1
    }
    region
  }

  val allBanks = B(
    (BigInt(1) << layout.bankCount) - 1, layout.bankCount bits)

  // -------------------------------------------------------------------
  // Resident slot ownership and common diagnostics.
  // -------------------------------------------------------------------
  val slotReadyReg = Reg(Bits(tileCfg.slotCount bits)) init 0
  val slotImageTagReg = Vec(
    Reg(UInt(m.sequenceIdWidth bits)) init 0, tileCfg.slotCount)
  val slotEpochReg = Vec(
    Reg(UInt(tileCfg.epochWidth bits)) init 0, tileCfg.slotCount)
  val slotFeatureCountReg = Vec(
    Reg(UInt(tileCfg.featureCountWidth bits)) init 0, tileCfg.slotCount)
  val slotTileCountReg = Vec(
    Reg(UInt(tileCfg.tileCountWidth bits)) init 0, tileCfg.slotCount)
  val slotTokenMaskReg = Vec(
    Reg(Bits(tileCfg.tokenLanes bits)) init 0, tileCfg.slotCount)
  val slotFactorReg = Vec(
    Vec(Reg(UInt(32 bits)) init 0, tileCfg.tokenLanes),
    tileCfg.slotCount)

  val stickyFault = RegInit(False)
  val completedFillCounter = Reg(UInt(32 bits)) init 0
  val completedReplayCounter = Reg(UInt(32 bits)) init 0

  io.slotReady := slotReadyReg
  io.slotImageTag := slotImageTagReg
  io.slotEpoch := slotEpochReg
  io.slotFeatureCount := slotFeatureCountReg
  io.slotTokenMask := slotTokenMaskReg
  io.slotDequantFactorQ16 := slotFactorReg
  io.completedFills := completedFillCounter
  io.completedReplays := completedReplayCounter

  // -------------------------------------------------------------------
  // Retained completion stages.
  // -------------------------------------------------------------------
  val fillCompletionValid = RegInit(False)
  val fillCompletionPayload = Reg(
    BitNetResidentActivationFillCompletion(tileCfg)) init
      BitNetResidentActivationFillCompletion(tileCfg).getZero
  io.fillCompletion.valid := fillCompletionValid
  io.fillCompletion.payload := fillCompletionPayload
  when(io.fillCompletion.fire) {
    fillCompletionValid := False
  }

  val replayCompletionValid = RegInit(False)
  val replayCompletionPayload = Reg(
    BitNetResidentActivationReplayCompletion(tileCfg)) init
      BitNetResidentActivationReplayCompletion(tileCfg).getZero
  io.replayCompletion.valid := replayCompletionValid
  io.replayCompletion.payload := replayCompletionPayload
  when(io.replayCompletion.fire) {
    replayCompletionValid := False
  }

  // -------------------------------------------------------------------
  // Fill path: two BRAM-backed 1024-bit bank-group staging segments.
  // -------------------------------------------------------------------
  val fillActiveReg = RegInit(False)
  val fillImageTag = Reg(UInt(m.sequenceIdWidth bits)) init 0
  val fillEpoch = Reg(UInt(tileCfg.epochWidth bits)) init 0
  val fillSlot = Reg(UInt(tileCfg.slotWidth bits)) init 0
  val fillFeatureCount =
    Reg(UInt(tileCfg.featureCountWidth bits)) init 0
  val fillTokenMask = Reg(Bits(tileCfg.tokenLanes bits)) init 0
  val fillExpectedIndex =
    Reg(UInt(tileCfg.featureIndexWidth bits)) init 0
  val fillFactor = Vec(
    Reg(UInt(32 bits)) init 0, tileCfg.tokenLanes)
  val fillPoison = RegInit(False)
  val fillFaultCode = Reg(Bits(8 bits)) init 0
  private val packedSegmentCount = layout.packedTransportSegmentCount
  private val packedSegmentWordCount = layout.bankCount / packedSegmentCount
  require(packedSegmentCount == 2)
  require(packedSegmentWordCount == 16)
  private val segmentMetadataSlotLow = 0
  private val segmentMetadataSegmentLow =
    segmentMetadataSlotLow + tileCfg.slotWidth
  private val segmentMetadataRowLow =
    segmentMetadataSegmentLow + log2Up(packedSegmentCount)
  private val segmentMetadataWidth =
    segmentMetadataRowLow + layout.localRowWidth
  private val segmentBufferCfg =
    BitNetPackedSegmentWordWriteBufferConfig(
      segmentWidth = layout.packedTransportSegmentWidth,
      bankWidth = 64,
      ringDepth = 2,
      metadataWidth = segmentMetadataWidth)
  val segmentBuffer = new BitNetPackedSegmentWordWriteBuffer(
    segmentBufferCfg, useVendorByteRam = useVendorSegmentRam)
  segmentBuffer.setName("projectionSegmentBuffer")
  // This pointer follows physical commits rather than logical segment parity:
  // an aborted fill may reclaim an odd number of rows before the next image.
  val segmentWriteSlot = Reg(UInt(segmentBufferCfg.slotAddressWidth bits)) init 0
  val writeOffered = RegInit(False)
  val fillInputComplete = RegInit(False)
  val fillAuthenticatedRows =
    Reg(UInt(tileCfg.tileCountWidth bits)) init 0

  // A semantic reject can be reported one cycle after a physical write fires.
  // Audit that exact segment while the other 1024-bit buffer continues to fill.
  val writeFaultWindowActive = RegInit(False)
  val writeFaultContextValid = RegInit(False)
  val writeFaultContextRegion =
    Reg(BitNetUnifiedResidentTensorRegion()) init
      BitNetUnifiedResidentTensorRegion.ProjectionSlot0
  val writeFaultContextLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val writeFaultContextSegment =
    Reg(UInt(log2Up(packedSegmentCount) bits)) init 0

  def retireFill(finalFault: Bool, finalFaultCode: Bits): Unit = {
    fillActiveReg := False
    fillInputComplete := False
    writeOffered := False
    writeFaultWindowActive := False
    writeFaultContextValid := False
    fillCompletionValid := True
    fillCompletionPayload.imageTag := fillImageTag
    fillCompletionPayload.epoch := fillEpoch
    fillCompletionPayload.slot := fillSlot
    fillCompletionPayload.featureCount := fillFeatureCount
    fillCompletionPayload.acceptedFeatures :=
      (fillExpectedIndex.resize(tileCfg.featureCountWidth) + 1).resized
    fillCompletionPayload.tokenMask := fillTokenMask
    for (lane <- 0 until tileCfg.tokenLanes) {
      fillCompletionPayload.dequantFactorQ16(lane) := fillFactor(lane)
    }
    fillCompletionPayload.success := !finalFault
    fillCompletionPayload.faultCode :=
      Mux(finalFault, finalFaultCode,
        B(BitNetResidentActivationFault.None, 8 bits))

    when(finalFault) {
      slotReadyReg(fillSlot) := False
      stickyFault := True
    } otherwise {
      slotReadyReg(fillSlot) := True
      slotImageTagReg(fillSlot) := fillImageTag
      slotEpochReg(fillSlot) := fillEpoch
      slotFeatureCountReg(fillSlot) := fillFeatureCount
      slotTileCountReg(fillSlot) :=
        (fillFeatureCount / tileCfg.tileFeatures)
          .resize(tileCfg.tileCountWidth)
      slotTokenMaskReg(fillSlot) := fillTokenMask
      for (lane <- 0 until tileCfg.tokenLanes) {
        slotFactorReg(fillSlot)(lane) := fillFactor(lane)
      }
      completedFillCounter := completedFillCounter + 1
    }
  }

  io.fillActive := fillActiveReg

  val replayActiveReg = RegInit(False)
  val readOutstanding = RegInit(False)
  val replaySlot = Reg(UInt(tileCfg.slotWidth bits)) init 0
  io.replayActive := replayActiveReg

  val prefixMaskLegal = (1 to tileCfg.tokenLanes).map { count =>
    io.fillCommand.payload.tokenMask === B((BigInt(1) << count) - 1,
      tileCfg.tokenLanes bits)
  }.reduce(_ || _)
  val featureCountLegal = tileCfg.legalFeatureCounts.map { count =>
    io.fillCommand.payload.featureCount ===
      U(count, tileCfg.featureCountWidth bits)
  }.reduce(_ || _)
  val fillTargetProtected = replayActiveReg &&
    replaySlot === io.fillCommand.payload.slot

  val replayCommandBaseReady = !replayActiveReg &&
    !replayCompletionValid && !readOutstanding
  io.replayCommand.ready := replayCommandBaseReady
  val replayWinsSameSlot = io.replayCommand.valid &&
    replayCommandBaseReady &&
    io.replayCommand.payload.slot === io.fillCommand.payload.slot
  io.fillCommand.ready := !fillActiveReg && !fillCompletionValid &&
    !replayWinsSameSlot

  val fillCommandLegal = featureCountLegal && prefixMaskLegal &&
    !fillTargetProtected

  when(io.fillCommand.fire) {
    when(fillCommandLegal) {
      fillActiveReg := True
      fillImageTag := io.fillCommand.payload.imageTag
      fillEpoch := io.fillCommand.payload.epoch
      fillSlot := io.fillCommand.payload.slot
      fillFeatureCount := io.fillCommand.payload.featureCount
      fillTokenMask := io.fillCommand.payload.tokenMask
      fillExpectedIndex := 0
      fillPoison := False
      fillFaultCode := 0
      fillInputComplete := False
      fillAuthenticatedRows := 0
      writeOffered := False
      writeFaultWindowActive := False
      writeFaultContextValid := False
      slotReadyReg(io.fillCommand.payload.slot) := False
    } otherwise {
      stickyFault := True
      fillCompletionValid := True
      fillCompletionPayload.imageTag := io.fillCommand.payload.imageTag
      fillCompletionPayload.epoch := io.fillCommand.payload.epoch
      fillCompletionPayload.slot := io.fillCommand.payload.slot
      fillCompletionPayload.featureCount :=
        io.fillCommand.payload.featureCount
      fillCompletionPayload.acceptedFeatures := 0
      fillCompletionPayload.tokenMask :=
        io.fillCommand.payload.tokenMask
      for (lane <- 0 until tileCfg.tokenLanes) {
        fillCompletionPayload.dequantFactorQ16(lane) := 0
      }
      fillCompletionPayload.success := False
      fillCompletionPayload.faultCode := B(
        BitNetResidentActivationFault.InvalidFillCommand, 8 bits)
    }
  }

  val fillAtFirst = fillExpectedIndex === 0
  val fillAtEnd = fillExpectedIndex.resize(tileCfg.featureCountWidth) ===
    (fillFeatureCount - 1).resize(tileCfg.featureCountWidth)
  val fillMetadataLegal =
    io.quantized.payload.imageTag === fillImageTag &&
      io.quantized.payload.epoch === fillEpoch &&
      io.quantized.payload.slot === fillSlot &&
      io.quantized.payload.featureIndex === fillExpectedIndex &&
      io.quantized.payload.tokenMask === fillTokenMask
  val fillFramingLegal = io.quantized.payload.first === fillAtFirst &&
    io.quantized.payload.last === fillAtEnd
  val fillFactorsStable = (0 until tileCfg.tokenLanes).map { lane =>
    fillAtFirst ||
      io.quantized.payload.dequantFactorQ16(lane) === fillFactor(lane)
  }.reduce(_ && _)
  val fillInactiveLanesZero = (0 until tileCfg.tokenLanes).map { lane =>
    fillTokenMask(lane) || io.quantized.payload.value(lane) === 0
  }.reduce(_ && _)

  val incomingFaultCode = Bits(8 bits)
  incomingFaultCode := B(BitNetResidentActivationFault.None, 8 bits)
  when(!fillMetadataLegal) {
    incomingFaultCode := B(
      BitNetResidentActivationFault.BeatMetadata, 8 bits)
  } elsewhen (!fillFramingLegal) {
    incomingFaultCode := B(
      BitNetResidentActivationFault.BeatFraming, 8 bits)
  } elsewhen (!fillFactorsStable) {
    incomingFaultCode := B(
      BitNetResidentActivationFault.FactorChanged, 8 bits)
  } elsewhen (!fillInactiveLanesZero) {
    incomingFaultCode := B(
      BitNetResidentActivationFault.InactiveLaneNonZero, 8 bits)
  } elsewhen (io.quantized.payload.fault) {
    incomingFaultCode := B(
      BitNetResidentActivationFault.UpstreamFault, 8 bits)
  }
  val incomingBeatLegal = incomingFaultCode ===
    B(BitNetResidentActivationFault.None, 8 bits)

  val fillLocalRow = (fillExpectedIndex |>> 7)
    .resize(layout.localRowWidth)
  val fillSegmentIndex = fillExpectedIndex(6)
  val fillLocalWord = fillExpectedIndex(5 downto 2)
  val fillWordFeature = fillExpectedIndex(1 downto 0)
  val fillAtSegmentEnd = fillExpectedIndex(5 downto 0).andR

  val bufferedMetadata = segmentBuffer.io.segment.payload.metadata
  val bufferedSlot = bufferedMetadata(
    segmentMetadataSegmentLow - 1 downto
      segmentMetadataSlotLow).asUInt
  val bufferedSegmentIndex = bufferedMetadata(
    segmentMetadataRowLow - 1 downto
      segmentMetadataSegmentLow).asUInt
  val bufferedLocalRow = bufferedMetadata(
    segmentMetadataWidth - 1 downto segmentMetadataRowLow).asUInt
  val issueSegmentFull = segmentBuffer.io.segment.valid
  val issueRegion = projectionRegion(bufferedSlot)
  val issueLocalRow = bufferedLocalRow
  val issueSegmentIndex = bufferedSegmentIndex

  val writeFaultMatchesCurrent = issueSegmentFull &&
    io.packedWriteFault.segmentIndex === issueSegmentIndex &&
    io.packedWriteFault.region === issueRegion &&
    io.packedWriteFault.localRow === issueLocalRow
  val writeFaultMatchesContext = writeFaultContextValid &&
    io.packedWriteFault.segmentIndex === writeFaultContextSegment &&
    io.packedWriteFault.region === writeFaultContextRegion &&
    io.packedWriteFault.localRow === writeFaultContextLocalRow
  val writeFaultMetadataMatches = writeFaultMatchesContext ||
    writeFaultMatchesCurrent
  val writeFaultNow = io.packedWriteFault.valid
  val backendWriteFaultCodeNow = Bits(8 bits)
  backendWriteFaultCodeNow := Mux(writeFaultMetadataMatches,
    B(BitNetUnifiedResidentProjectionFault.BackendWrite, 8 bits),
    B(BitNetUnifiedResidentProjectionFault.BackendWriteMetadata, 8 bits))
  io.packedWriteFault.ready := True

  val incomingBeatFaultNow = io.quantized.valid && !incomingBeatLegal
  val segmentBufferFaultNow = segmentBuffer.io.protocolFaultSticky
  val fillPoisonNow = fillPoison || incomingBeatFaultNow || writeFaultNow ||
    segmentBufferFaultNow

  // Each feature contributes two token bytes.  Its feature index selects one
  // of four 16-bit positions in the native 64-bit packed word.
  val fillFeatureBytes = Bits(tileCfg.tokenLanes * layout.packedWidth bits)
  fillFeatureBytes := 0
  for (lane <- 0 until tileCfg.tokenLanes) {
    val low = lane * layout.packedWidth
    fillFeatureBytes(low + layout.packedWidth - 1 downto low) :=
      io.quantized.payload.value(lane).asBits
  }
  val fillWordData = fillFeatureBytes ## fillFeatureBytes ##
    fillFeatureBytes ## fillFeatureBytes
  val fillByteMask = Bits(segmentBufferCfg.bytesPerBank bits)
  fillByteMask := (B(3, segmentBufferCfg.bytesPerBank bits) |<<
    (fillWordFeature.resize(3) << 1)).resized
  val fillSegmentMetadata = Bits(segmentMetadataWidth bits)
  fillSegmentMetadata := 0
  fillSegmentMetadata(
    segmentMetadataSegmentLow - 1 downto segmentMetadataSlotLow) :=
      fillSlot.asBits
  fillSegmentMetadata(
    segmentMetadataRowLow - 1 downto segmentMetadataSegmentLow) :=
      fillSegmentIndex.asBits
  fillSegmentMetadata(
    segmentMetadataWidth - 1 downto segmentMetadataRowLow) :=
      fillLocalRow.asBits

  // Once poison is registered, finish only an already-partial BRAM row with
  // legal zero data and bypass every later feature.  The currently faulting
  // beat may enter the private row before fillPoison is clocked, but
  // fillPoisonNow blocks that row from ever reaching packedWrite.  Keeping the
  // current semantic checks out of the RAM write enable severs the long
  // framing/featureCount -> XPM EN path without changing healthy throughput.
  // The buffer fault bit is reserved for a genuine local protocol violation.
  val fillWordWriteRequired = !segmentBufferFaultNow &&
    (!fillPoison || segmentBuffer.io.partial)
  segmentBuffer.io.wordWrite.valid := fillActiveReg && io.quantized.valid &&
    fillWordWriteRequired
  segmentBuffer.io.wordWrite.payload.data := Mux(
    fillPoison, B(0, segmentBufferCfg.bankWidth bits), fillWordData)
  segmentBuffer.io.wordWrite.payload.byteMask := fillByteMask
  segmentBuffer.io.wordWrite.payload.bankIndex := fillLocalWord
  segmentBuffer.io.wordWrite.payload.slotAddress := segmentWriteSlot
  segmentBuffer.io.wordWrite.payload.commit := fillAtSegmentEnd
  segmentBuffer.io.wordWrite.payload.metadata := fillSegmentMetadata
  segmentBuffer.io.wordWrite.payload.fault := False
  when(segmentBuffer.io.wordWrite.fire && fillAtSegmentEnd) {
    segmentWriteSlot := segmentWriteSlot + 1
  }

  io.packedWrite.valid := fillActiveReg && issueSegmentFull &&
    !writeFaultWindowActive && !writeFaultNow &&
    (writeOffered || !fillPoisonNow)
  io.packedWrite.region := issueRegion
  io.packedWrite.localRow := issueLocalRow
  io.packedWrite.segmentIndex := issueSegmentIndex
  io.packedWrite.symbolMask := B(
    (BigInt(1) << layout.nativeSymbolCount) - 1,
    layout.nativeSymbolCount bits)
  io.packedWrite.aliasOwner :=
    BitNetUnifiedResidentTensorAliasOwner.Query
  io.packedWrite.aliasGeneration := 0
  for (word <- 0 until packedSegmentWordCount) {
    io.packedWrite.segmentWords(word) :=
      segmentBuffer.io.segment.payload.data(
        (word + 1) * layout.wordWidth - 1 downto word * layout.wordWidth)
  }

  io.quantized.ready := fillActiveReg &&
    (segmentBufferFaultNow || (fillPoisonNow && !segmentBuffer.io.partial) ||
      segmentBuffer.io.wordWrite.ready)

  // A poisoned, never-offered segment is reclaimed locally.  Once an output
  // has been visible while stalled, only its real fire/fault terminal may
  // release the BRAM row, preserving the packedWrite Stream contract.
  val discardBufferedSegment = fillActiveReg &&
    segmentBuffer.io.segment.valid && fillPoisonNow && !writeOffered
  segmentBuffer.io.segment.ready := discardBufferedSegment ||
    io.packedWrite.fire || (io.packedWriteFault.fire && writeOffered)

  when(io.packedWrite.valid && !io.packedWrite.ready) {
    writeOffered := True
  }

  when(io.packedWrite.fire) {
    writeOffered := False
    writeFaultWindowActive := True
    writeFaultContextValid := True
    writeFaultContextRegion := io.packedWrite.region
    writeFaultContextLocalRow := io.packedWrite.localRow
    writeFaultContextSegment := io.packedWrite.segmentIndex
  }

  when(io.packedWriteFault.fire) {
    stickyFault := True
    when(fillActiveReg) {
      slotReadyReg(fillSlot) := False
      when(!fillPoison) {
        fillPoison := True
        fillFaultCode := backendWriteFaultCodeNow
      }
      writeOffered := False
      writeFaultWindowActive := False
      writeFaultContextValid := False
    }
  }

  when(segmentBufferFaultNow) {
    stickyFault := True
    when(fillActiveReg && !fillPoison) {
      fillPoison := True
      fillFaultCode := B(
        BitNetResidentActivationFault.BeatFraming, 8 bits)
      slotReadyReg(fillSlot) := False
    }
  }

  when(io.quantized.fire) {
    when(fillAtFirst) {
      for (lane <- 0 until tileCfg.tokenLanes) {
        fillFactor(lane) :=
          io.quantized.payload.dequantFactorQ16(lane)
      }
    }

    when(!incomingBeatLegal && !fillPoison && !writeFaultNow) {
      fillPoison := True
      fillFaultCode := incomingFaultCode
      slotReadyReg(fillSlot) := False
      stickyFault := True
    }

    // After any fault, consume exactly featureCount beats rather than trusting
    // an early/late producer `last` marker as a new image boundary.
    when(fillAtEnd) {
      fillInputComplete := True
    } otherwise {
      fillExpectedIndex := fillExpectedIndex + 1
    }
  }

  // Exactly one complete cycle after a fired segment is reserved for its
  // delayed fault response.  Input filling continues in the opposite buffer.
  when(writeFaultWindowActive) {
    writeFaultWindowActive := False
    writeFaultContextValid := False
    when(!writeFaultNow && !fillPoison) {
      when(writeFaultContextSegment === packedSegmentCount - 1) {
        fillAuthenticatedRows := fillAuthenticatedRows + 1
      }
    }
  }

  val expectedAuthenticatedRows =
    (fillFeatureCount |>> 7).resize(tileCfg.tileCountWidth)
  val segmentBufferDrained = segmentBufferFaultNow ||
    (segmentBuffer.io.empty && !segmentBuffer.io.partial &&
      !segmentBuffer.io.segment.valid)
  val fillWriteDrainComplete = segmentBufferDrained && !writeOffered &&
    !writeFaultWindowActive
  when(fillActiveReg && fillInputComplete && fillWriteDrainComplete &&
      (fillPoison || writeFaultNow ||
        fillAuthenticatedRows === expectedAuthenticatedRows)) {
    val terminalFaultCode = Bits(8 bits)
    terminalFaultCode := fillFaultCode
    when(writeFaultNow && !fillPoison) {
      terminalFaultCode := backendWriteFaultCodeNow
    }
    retireFill(fillPoison || writeFaultNow, terminalFaultCode)
  }

  // -------------------------------------------------------------------
  // Replay path: one row request produces exactly one TP4 tile.
  // -------------------------------------------------------------------
  val replayImageTag = Reg(UInt(m.sequenceIdWidth bits)) init 0
  val replayEpoch = Reg(UInt(tileCfg.epochWidth bits)) init 0
  val replayOutputSequence = Reg(UInt(m.sequenceIdWidth bits)) init 0
  val replayBeatOrdinal = Reg(UInt(m.beatCountWidth bits)) init 0
  val replayBeatCount = Reg(UInt(m.beatCountWidth bits)) init 0
  val replayEmittedCount = Reg(UInt(m.beatCountWidth bits)) init 0
  val replayTileIndex = Reg(UInt(tileCfg.tileIndexWidth bits)) init 0

  val replaySlotReady = slotReadyReg(io.replayCommand.payload.slot)
  val replayTagMatches =
    slotImageTagReg(io.replayCommand.payload.slot) ===
      io.replayCommand.payload.imageTag &&
      slotEpochReg(io.replayCommand.payload.slot) ===
        io.replayCommand.payload.epoch
  val replayFeatureCountMatches =
    slotFeatureCountReg(io.replayCommand.payload.slot) ===
      io.replayCommand.payload.inputFeatures
  val replayStartInRange =
    io.replayCommand.payload.startTile.resize(tileCfg.tileCountWidth) <
      slotTileCountReg(io.replayCommand.payload.slot)
  val replayLastOrdinal =
    io.replayCommand.payload.beatOrdinalBase.resize(m.beatCountWidth + 1) +
      io.replayCommand.payload.beatCount.resize(m.beatCountWidth + 1) - 1
  val replayNoOrdinalWrap = !replayLastOrdinal.msb
  val replayTargetBeingFilled = fillActiveReg &&
    fillSlot === io.replayCommand.payload.slot
  val replayCommandLegal = replaySlotReady && replayTagMatches &&
    replayFeatureCountMatches && replayStartInRange &&
    io.replayCommand.payload.beatCount =/= 0 && replayNoOrdinalWrap &&
    !replayTargetBeingFilled

  when(io.replayCommand.fire) {
    when(replayCommandLegal) {
      replayActiveReg := True
      replayImageTag := io.replayCommand.payload.imageTag
      replayEpoch := io.replayCommand.payload.epoch
      replaySlot := io.replayCommand.payload.slot
      replayOutputSequence :=
        io.replayCommand.payload.outputSequenceId
      replayBeatOrdinal := io.replayCommand.payload.beatOrdinalBase
      replayBeatCount := io.replayCommand.payload.beatCount
      replayEmittedCount := 0
      replayTileIndex := io.replayCommand.payload.startTile
    } otherwise {
      stickyFault := True
      replayCompletionValid := True
      replayCompletionPayload.imageTag :=
        io.replayCommand.payload.imageTag
      replayCompletionPayload.epoch := io.replayCommand.payload.epoch
      replayCompletionPayload.slot := io.replayCommand.payload.slot
      replayCompletionPayload.outputSequenceId :=
        io.replayCommand.payload.outputSequenceId
      replayCompletionPayload.emittedBeats := 0
      replayCompletionPayload.success := False
      replayCompletionPayload.faultCode := B(
        BitNetResidentActivationFault.InvalidReplayCommand, 8 bits)
    }
  }

  val replayRegion = projectionRegion(replaySlot)
  private val logicalTilesPerRow = 128 / tileCfg.tileFeatures
  val replayLocalRow =
    (replayTileIndex / logicalTilesPerRow).resize(layout.localRowWidth)
  io.packedRead.valid := replayActiveReg && !readOutstanding
  io.packedRead.region := replayRegion
  io.packedRead.localRow := replayLocalRow
  io.packedRead.bankMask := allBanks
  io.packedRead.aliasOwner :=
    BitNetUnifiedResidentTensorAliasOwner.Query
  io.packedRead.aliasGeneration := 0

  val responseRegion = Reg(BitNetUnifiedResidentTensorRegion()) init
    BitNetUnifiedResidentTensorRegion.ProjectionSlot0
  val responseLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val responseImageTag = Reg(UInt(m.sequenceIdWidth bits)) init 0
  val responseEpoch = Reg(UInt(tileCfg.epochWidth bits)) init 0
  val responseSlot = Reg(UInt(tileCfg.slotWidth bits)) init 0
  val responseOutputSequence =
    Reg(UInt(m.sequenceIdWidth bits)) init 0
  val responseBeatOrdinal = Reg(UInt(m.beatCountWidth bits)) init 0
  val responseFeatureTile =
    Reg(UInt(tileCfg.tileIndexWidth bits)) init 0
  val responseFirst = RegInit(False)
  val responseLast = RegInit(False)

  when(io.packedRead.fire) {
    readOutstanding := True
    responseRegion := replayRegion
    responseLocalRow := replayLocalRow
    responseImageTag := replayImageTag
    responseEpoch := replayEpoch
    responseSlot := replaySlot
    responseOutputSequence := replayOutputSequence
    responseBeatOrdinal := replayBeatOrdinal
    responseFeatureTile := replayTileIndex
    responseFirst := replayEmittedCount === 0
    responseLast := replayEmittedCount === replayBeatCount - 1
  }

  val readResponseMetadataMatches = readOutstanding &&
    io.packedReadData.region === responseRegion &&
    io.packedReadData.localRow === responseLocalRow &&
    io.packedReadData.bankMask === allBanks
  val unexpectedReadResponseNow = io.packedReadData.valid &&
    !readOutstanding
  val readResponseFault = io.packedReadData.fault ||
    !readResponseMetadataMatches
  val readFaultNow = io.packedReadData.valid &&
    (unexpectedReadResponseNow || readResponseFault)

  io.tile.valid := io.packedReadData.valid && readOutstanding
  io.tile.payload.normalized.sequenceId := responseOutputSequence
  io.tile.payload.normalized.beatOrdinal := responseBeatOrdinal
  io.tile.payload.normalized.fault := readResponseFault
  val replayFeatureBase =
    (responseFeatureTile % logicalTilesPerRow) * tileCfg.tileFeatures
  for (feature <- 0 until tileCfg.tileFeatures;
       lane <- 0 until tileCfg.tokenLanes) {
    val physicalFeature = replayFeatureBase + feature
    io.tile.payload.normalized.tokenActivation(lane)(feature) :=
      io.packedReadData.words((physicalFeature >> 2).resized)
        .feature(physicalFeature(1 downto 0))(lane)
  }
  io.tile.payload.imageTag := responseImageTag
  io.tile.payload.epoch := responseEpoch
  io.tile.payload.slot := responseSlot
  io.tile.payload.featureTile := responseFeatureTile
  io.tile.payload.tokenMask := slotTokenMaskReg(responseSlot)
  for (lane <- 0 until tileCfg.tokenLanes) {
    io.tile.payload.dequantFactorQ16(lane) :=
      slotFactorReg(responseSlot)(lane)
  }
  io.tile.payload.first := responseFirst
  io.tile.payload.last := responseLast || readResponseFault

  // Unexpected responses are consumed and quarantined locally.  Expected
  // responses remain stable all the way through tile backpressure.
  io.packedReadData.ready := Mux(
    readOutstanding, io.tile.ready, True)

  when(readFaultNow) {
    stickyFault := True
    when(readOutstanding) {
      slotReadyReg(responseSlot) := False
    }
  }

  when(io.tile.fire) {
    val emittedNext =
      (replayEmittedCount + 1).resize(m.beatCountWidth)
    readOutstanding := False
    when(readResponseFault || responseLast) {
      replayActiveReg := False
      replayCompletionValid := True
      replayCompletionPayload.imageTag := responseImageTag
      replayCompletionPayload.epoch := responseEpoch
      replayCompletionPayload.slot := responseSlot
      replayCompletionPayload.outputSequenceId :=
        responseOutputSequence
      replayCompletionPayload.emittedBeats := emittedNext
      replayCompletionPayload.success := !readResponseFault
      replayCompletionPayload.faultCode := B(
        BitNetResidentActivationFault.None, 8 bits)
      when(readResponseFault) {
        replayCompletionPayload.success := False
        replayCompletionPayload.faultCode := Mux(
          readResponseMetadataMatches,
          B(BitNetUnifiedResidentProjectionFault.BackendRead, 8 bits),
          B(BitNetUnifiedResidentProjectionFault.BackendReadMetadata,
            8 bits))
      } otherwise {
        completedReplayCounter := completedReplayCounter + 1
      }
    } otherwise {
      replayEmittedCount := emittedNext
      replayBeatOrdinal := replayBeatOrdinal + 1
      when(replayTileIndex.resize(tileCfg.tileCountWidth) + 1 ===
          slotTileCountReg(replaySlot)) {
        replayTileIndex := 0
      } otherwise {
        replayTileIndex := replayTileIndex + 1
      }
    }
  }

  io.protocolFaultSticky := stickyFault || writeFaultNow || readFaultNow ||
    segmentBufferFaultNow
}

object GenerateBitNetUnifiedResidentProjectionTileController extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-projection-controller")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedResidentProjectionTileController())
}
