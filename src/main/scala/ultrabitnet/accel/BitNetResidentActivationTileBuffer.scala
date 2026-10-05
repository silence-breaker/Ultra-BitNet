package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Geometry of the PL-resident dynamic-i8 activation image.
  *
  * Four adjacent feature beats are packed into one 128-bit word.  The word is
  * striped by feature-quad over 32 independent memories, so one registered
  * read from every bank reconstructs 128 features x four tokens in one slow
  * clock.  Two complete slots permit the next image to be quantized while the
  * current image is repeatedly replayed by a projection.
  *
  * The slot bit is folded into each bank address, yielding 32 memories of
  * 128 bits x 108 rather than 64 independent shallow memories.  Each inferred
  * true-dual-port bank can write the inactive slot while reading the active
  * slot.  At the ZU3EG RAMB36 maximum port width this is about 64 RAMB36, half
  * the cost of mapping two separately inferred 128 x 54 banks per lane.
  */
case class BitNetResidentActivationTileBufferConfig(
    feeder: BitNetContinuousMatrixFeederConfig =
      BitNetContinuousMatrixFeederConfig(),
    hiddenFeatures: Int = 2560,
    ffnFeatures: Int = 6912,
    bankCount: Int = 32,
    featuresPerBankWord: Int = 4,
    slotCount: Int = 2,
    epochWidth: Int = 8
) {
  require(feeder.tokenLanes == 2 || feeder.tokenLanes == 4,
    "the resident activation image must be TP2 or TP4")
  require(feeder.matrix.base.activationWidth == 8)
  require(feeder.matrix.base.siteCount == 64 ||
    feeder.matrix.base.siteCount == 128)
  require(bankCount == 32,
    "32 parallel banks are required for one 128-feature tile per cycle")
  require(featuresPerBankWord == 4)
  require(slotCount == 2, "the production image store is ping/pong")
  require(hiddenFeatures > 0 && hiddenFeatures % 128 == 0)
  require(ffnFeatures > 0 && ffnFeatures % 128 == 0)
  require(epochWidth >= 4)

  val tokenLanes: Int = feeder.tokenLanes
  val maxFeatures: Int = scala.math.max(hiddenFeatures, ffnFeatures)
  val tileFeatures: Int = feeder.matrix.base.siteCount
  val maxTiles: Int = (maxFeatures + tileFeatures - 1) / tileFeatures
  val slotDepth: Int = maxTiles
  val bankDepth: Int = maxTiles * slotCount
  val bankWordWidth: Int =
    featuresPerBankWord * tokenLanes * feeder.matrix.base.activationWidth
  val slotWidth: Int = log2Up(slotCount)
  val bankWidth: Int = log2Up(bankCount)
  val bankAddressWidth: Int = scala.math.max(1, log2Up(bankDepth))
  val featureIndexWidth: Int = log2Up(maxFeatures)
  val featureCountWidth: Int = log2Up(maxFeatures + 1)
  val tileIndexWidth: Int = scala.math.max(1, log2Up(maxTiles))
  val tileCountWidth: Int = log2Up(maxTiles + 1)

  require(bankCount * featuresPerBankWord == 128)
  require(bankWordWidth == featuresPerBankWord * tokenLanes *
    feeder.matrix.base.activationWidth)
  require(maxFeatures == 6912 &&
    maxTiles == maxFeatures / tileFeatures,
    "the production FFN image must cover every physical matrix tile")

  def legalFeatureCounts: Seq[Int] = Seq(hiddenFeatures, ffnFeatures).distinct
}

case class BitNetResidentActivationFillCommand(
    cfg: BitNetResidentActivationTileBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  val imageTag = UInt(m.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val featureCount = UInt(cfg.featureCountWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
}

/** One feature position from [[NativeDynamicI8Quantizer]], with ownership tags.
  *
  * `dequantFactorQ16` and `tokenMask` are intentionally repeated exactly like
  * NativeDynamicQuantResult.  The buffer verifies that they are invariant over
  * the complete image instead of trusting an unframed sideband register.
  */
case class BitNetResidentActivationQuantBeat(
    cfg: BitNetResidentActivationTileBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  val imageTag = UInt(m.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val featureIndex = UInt(cfg.featureIndexWidth bits)
  val value = Vec(SInt(m.base.activationWidth bits), cfg.tokenLanes)
  val dequantFactorQ16 = Vec(UInt(32 bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val first = Bool()
  val last = Bool()
  /** Propagates an upstream quantizer/protocol failure without storing data. */
  val fault = Bool()
}

case class BitNetResidentActivationFillCompletion(
    cfg: BitNetResidentActivationTileBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  val imageTag = UInt(m.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val featureCount = UInt(cfg.featureCountWidth bits)
  val acceptedFeatures = UInt(cfg.featureCountWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val dequantFactorQ16 = Vec(UInt(32 bits), cfg.tokenLanes)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/** Replays an activation image over an arbitrary feeder ordinal interval.
  *
  * `startTile` is explicit, avoiding a variable hardware modulo for resumed
  * schedules.  Normal full-matrix and group-of-four schedules use zero.  The
  * buffer wraps at the resident image's tile count, so the same activation is
  * reused for every output-row group without another DDR or quantizer pass.
  */
case class BitNetResidentActivationReplayCommand(
    cfg: BitNetResidentActivationTileBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  val imageTag = UInt(m.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  /** Must match the resident slot; catches hidden/FFN image mix-ups. */
  val inputFeatures = UInt(cfg.featureCountWidth bits)
  val outputSequenceId = UInt(m.sequenceIdWidth bits)
  val beatOrdinalBase = UInt(m.beatCountWidth bits)
  val beatCount = UInt(m.beatCountWidth bits)
  val startTile = UInt(cfg.tileIndexWidth bits)
}

/** Enhanced resident tile.  `normalized` wires field-for-field to the existing
  * BitNetContinuousMatrixFeeder activation boundary; the remaining fields
  * retain image ownership and dynamic-dequant metadata for the task issuer.
  */
case class BitNetResidentActivationTile(
    cfg: BitNetResidentActivationTileBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  val normalized = BitNetNormalizedActivationTile(cfg.feeder)
  val imageTag = UInt(m.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val featureTile = UInt(cfg.tileIndexWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val dequantFactorQ16 = Vec(UInt(32 bits), cfg.tokenLanes)
  val first = Bool()
  val last = Bool()
}

case class BitNetResidentActivationReplayCompletion(
    cfg: BitNetResidentActivationTileBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  val imageTag = UInt(m.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val outputSequenceId = UInt(m.sequenceIdWidth bits)
  val emittedBeats = UInt(m.beatCountWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

object BitNetResidentActivationFault {
  val None: Int = 0x00
  val InvalidFillCommand: Int = 0xa0
  val BeatMetadata: Int = 0xa1
  val BeatFraming: Int = 0xa2
  val FactorChanged: Int = 0xa3
  val InactiveLaneNonZero: Int = 0xa4
  val UpstreamFault: Int = 0xa5
  val InvalidReplayCommand: Int = 0xa6
}

/** Pure layout helpers shared by linker diagnostics and tests. */
object BitNetResidentActivationTileBufferMath {
  case class Location(bank: Int, address: Int, wordFeature: Int)

  def location(feature: Int, bankCount: Int = 32): Location = {
    require(feature >= 0)
    val quad = feature / 4
    Location(
      bank = quad % bankCount,
      address = quad / bankCount,
      wordFeature = feature % 4)
  }

  def tileCount(featureCount: Int): Int = {
    require(featureCount > 0 && featureCount % 128 == 0)
    featureCount / 128
  }

  def packedWord(values: Seq[Seq[Int]]): BigInt = {
    require(values.length == 4 && values.forall(_.length == 4))
    values.zipWithIndex.foldLeft(BigInt(0)) { case (word, (feature, f)) =>
      feature.zipWithIndex.foldLeft(word) { case (partial, (value, token)) =>
        require(value >= -128 && value <= 127)
        partial | (BigInt(value & 0xff) << (f * 32 + token * 8))
      }
    }
  }
}

/**
  * Two-slot, one-tile-per-cycle PL-resident quantized activation store.
  *
  * There is one outstanding synchronous read at most.  Its response is the
  * output elastic stage itself: while stalled, no RAM address changes; while
  * accepted, the next 4096-bit tile is issued in the same cycle.  Thus a ready
  * consumer observes one tile per slow clock after the initial one-cycle RAM
  * latency, without an 8192-bit register FIFO.
  *
  * Any malformed fill poisons and invalidates the target slot.  Invalid replay
  * commands emit no tile.  Both paths return typed failure completions and set
  * a sticky protocol fault, which is the fail-closed contract used by the
  * resident layer controller.
  */
class BitNetResidentActivationTileBuffer(
    cfg: BitNetResidentActivationTileBufferConfig =
      BitNetResidentActivationTileBufferConfig()) extends Component {
  private val m = cfg.feeder.matrix
  private val b = m.base

  val io = new Bundle {
    val fillCommand = slave(Stream(
      BitNetResidentActivationFillCommand(cfg)))
    val quantized = slave(Stream(
      BitNetResidentActivationQuantBeat(cfg)))
    val fillCompletion = master(Stream(
      BitNetResidentActivationFillCompletion(cfg)))

    val replayCommand = slave(Stream(
      BitNetResidentActivationReplayCommand(cfg)))
    val tile = master(Stream(BitNetResidentActivationTile(cfg)))
    val replayCompletion = master(Stream(
      BitNetResidentActivationReplayCompletion(cfg)))

    val slotReady = out Bits(cfg.slotCount bits)
    val slotImageTag = out Vec(UInt(m.sequenceIdWidth bits), cfg.slotCount)
    val slotEpoch = out Vec(UInt(cfg.epochWidth bits), cfg.slotCount)
    val slotFeatureCount = out Vec(
      UInt(cfg.featureCountWidth bits), cfg.slotCount)
    val slotTokenMask = out Vec(Bits(cfg.tokenLanes bits), cfg.slotCount)
    val slotDequantFactorQ16 = out Vec(
      Vec(UInt(32 bits), cfg.tokenLanes), cfg.slotCount)
    val fillActive = out Bool()
    val replayActive = out Bool()
    val protocolFaultSticky = out Bool()
    val completedFills = out UInt(32 bits)
    val completedReplays = out UInt(32 bits)
  }

  val slotReadyReg = Reg(Bits(cfg.slotCount bits)) init (0)
  val slotImageTagReg = Vec(
    Reg(UInt(m.sequenceIdWidth bits)) init (0), cfg.slotCount)
  val slotEpochReg = Vec(
    Reg(UInt(cfg.epochWidth bits)) init (0), cfg.slotCount)
  val slotFeatureCountReg = Vec(
    Reg(UInt(cfg.featureCountWidth bits)) init (0), cfg.slotCount)
  val slotTileCountReg = Vec(
    Reg(UInt(cfg.tileCountWidth bits)) init (0), cfg.slotCount)
  val slotTokenMaskReg = Vec(
    Reg(Bits(cfg.tokenLanes bits)) init (0), cfg.slotCount)
  val slotFactorReg = Vec(
    Vec(Reg(UInt(32 bits)) init (0), cfg.tokenLanes), cfg.slotCount)

  io.slotReady := slotReadyReg
  io.slotImageTag := slotImageTagReg
  io.slotEpoch := slotEpochReg
  io.slotFeatureCount := slotFeatureCountReg
  io.slotTokenMask := slotTokenMaskReg
  io.slotDequantFactorQ16 := slotFactorReg

  val stickyFault = RegInit(False)
  val completedFillCounter = Reg(UInt(32 bits)) init (0)
  val completedReplayCounter = Reg(UInt(32 bits)) init (0)
  io.protocolFaultSticky := stickyFault
  io.completedFills := completedFillCounter
  io.completedReplays := completedReplayCounter

  val fillActiveReg = RegInit(False)
  val fillImageTag = Reg(UInt(m.sequenceIdWidth bits)) init (0)
  val fillEpoch = Reg(UInt(cfg.epochWidth bits)) init (0)
  val fillSlot = Reg(UInt(cfg.slotWidth bits)) init (0)
  val fillFeatureCount = Reg(UInt(cfg.featureCountWidth bits)) init (0)
  val fillTokenMask = Reg(Bits(cfg.tokenLanes bits)) init (0)
  val fillExpectedIndex = Reg(UInt(cfg.featureIndexWidth bits)) init (0)
  val fillFactor = Vec(
    Reg(UInt(32 bits)) init (0), cfg.tokenLanes)
  val fillPoison = RegInit(False)
  val fillFaultCode = Reg(Bits(8 bits)) init (0)
  val quadAssembly = Reg(Bits(cfg.bankWordWidth bits)) init (0)

  val replayActiveReg = RegInit(False)
  val replayImageTag = Reg(UInt(m.sequenceIdWidth bits)) init (0)
  val replayEpoch = Reg(UInt(cfg.epochWidth bits)) init (0)
  val replaySlot = Reg(UInt(cfg.slotWidth bits)) init (0)
  val replayOutputSequence = Reg(UInt(m.sequenceIdWidth bits)) init (0)
  val replayBeatOrdinal = Reg(UInt(m.beatCountWidth bits)) init (0)
  val replayBeatCount = Reg(UInt(m.beatCountWidth bits)) init (0)
  val replayIssuedCount = Reg(UInt(m.beatCountWidth bits)) init (0)
  val replayTileIndex = Reg(UInt(cfg.tileIndexWidth bits)) init (0)
  val replayLastIssued = RegInit(False)
  val readResponseValid = RegInit(False)

  io.fillActive := fillActiveReg
  io.replayActive := replayActiveReg

  val fillCompletionValid = RegInit(False)
  val fillCompletionPayload = Reg(
    BitNetResidentActivationFillCompletion(cfg)) init (
      BitNetResidentActivationFillCompletion(cfg).getZero)
  io.fillCompletion.valid := fillCompletionValid
  io.fillCompletion.payload := fillCompletionPayload
  when(io.fillCompletion.fire) {
    fillCompletionValid := False
  }

  val replayCompletionValid = RegInit(False)
  val replayCompletionPayload = Reg(
    BitNetResidentActivationReplayCompletion(cfg)) init (
      BitNetResidentActivationReplayCompletion(cfg).getZero)
  io.replayCompletion.valid := replayCompletionValid
  io.replayCompletion.payload := replayCompletionPayload
  when(io.replayCompletion.fire) {
    replayCompletionValid := False
  }

  // The ping/pong slot is part of the address: 32 x (128-bit x 108) true-dual
  // port banks, not 64 separately inferred 128-bit x 54 memories.  Port A is
  // the sequential quantizer fill and port B is the parallel replay read.
  val activationMemories = Array.tabulate(cfg.bankCount) { bank =>
    val memory = Mem(Bits(cfg.bankWordWidth bits), cfg.bankDepth)
    memory.setName(s"activationBank${bank}")
    memory.addAttribute("ram_style", "block")
    memory
  }

  val prefixMaskLegal = io.fillCommand.payload.tokenMask === B"4'b0001" ||
    io.fillCommand.payload.tokenMask === B"4'b0011" ||
    io.fillCommand.payload.tokenMask === B"4'b0111" ||
    io.fillCommand.payload.tokenMask === B"4'b1111"
  val featureCountLegal = cfg.legalFeatureCounts.map { count =>
    io.fillCommand.payload.featureCount ===
      U(count, cfg.featureCountWidth bits)
  }.reduce(_ || _)
  val fillTargetProtected = replayActiveReg &&
    replaySlot === io.fillCommand.payload.slot

  // A simultaneous replay of the same ready slot wins over overwrite.  Fills
  // of the other slot can still start on that cycle.
  val replayCommandBaseReady = !replayActiveReg &&
    !replayCompletionValid && !readResponseValid
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
      quadAssembly := 0
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
      fillCompletionPayload.tokenMask := io.fillCommand.payload.tokenMask
      for (lane <- 0 until cfg.tokenLanes) {
        fillCompletionPayload.dequantFactorQ16(lane) := 0
      }
      fillCompletionPayload.success := False
      fillCompletionPayload.faultCode := B(
        BitNetResidentActivationFault.InvalidFillCommand, 8 bits)
    }
  }

  io.quantized.ready := fillActiveReg
  val fillAtFirst = fillExpectedIndex === 0
  val fillAtEnd = fillExpectedIndex.resize(cfg.featureCountWidth) ===
    (fillFeatureCount - 1).resize(cfg.featureCountWidth)
  val fillMetadataLegal =
    io.quantized.payload.imageTag === fillImageTag &&
      io.quantized.payload.epoch === fillEpoch &&
      io.quantized.payload.slot === fillSlot &&
      io.quantized.payload.featureIndex === fillExpectedIndex &&
      io.quantized.payload.tokenMask === fillTokenMask
  val fillFramingLegal = io.quantized.payload.first === fillAtFirst &&
    io.quantized.payload.last === fillAtEnd
  val fillFactorsStable = (0 until cfg.tokenLanes).map { lane =>
    fillAtFirst ||
      io.quantized.payload.dequantFactorQ16(lane) === fillFactor(lane)
  }.reduce(_ && _)
  val fillInactiveLanesZero = (0 until cfg.tokenLanes).map { lane =>
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

  val packedFeature = Bits(cfg.tokenLanes * b.activationWidth bits)
  for (lane <- 0 until cfg.tokenLanes) {
    packedFeature((lane + 1) * b.activationWidth - 1 downto
      lane * b.activationWidth) := io.quantized.payload.value(lane).asBits
  }
  val nextQuadAssembly = Bits(cfg.bankWordWidth bits)
  nextQuadAssembly := quadAssembly
  when(fillExpectedIndex(1 downto 0) === 0) {
    nextQuadAssembly := 0
  }
  switch(fillExpectedIndex(1 downto 0)) {
    is(0) { nextQuadAssembly(31 downto 0) := packedFeature }
    is(1) { nextQuadAssembly(63 downto 32) := packedFeature }
    is(2) { nextQuadAssembly(95 downto 64) := packedFeature }
    default { nextQuadAssembly(127 downto 96) := packedFeature }
  }

  val fillCompletesWord = fillExpectedIndex(1 downto 0) === 3
  val fillWrite = io.quantized.fire && incomingBeatLegal &&
    !fillPoison && fillCompletesWord
  val fillQuadIndex = fillExpectedIndex |>> 2
  val fillBank = fillQuadIndex.resize(cfg.bankWidth)
  val fillTileAddress = (fillExpectedIndex |>> 7).resize(
    cfg.tileIndexWidth)
  val fillAddress = UInt(cfg.bankAddressWidth bits)
  fillAddress := fillTileAddress.resized
  when(fillSlot === 1) {
    fillAddress := (fillTileAddress.resize(cfg.bankAddressWidth) +
      U(cfg.slotDepth, cfg.bankAddressWidth bits)).resized
  }
  for (bank <- 0 until cfg.bankCount) {
    activationMemories(bank).write(
      address = fillAddress,
      data = nextQuadAssembly,
      enable = fillWrite && fillBank === bank)
  }

  when(io.quantized.fire) {
    quadAssembly := nextQuadAssembly
    when(fillAtFirst) {
      for (lane <- 0 until cfg.tokenLanes) {
        fillFactor(lane) := io.quantized.payload.dequantFactorQ16(lane)
      }
    }
    when(!incomingBeatLegal && !fillPoison) {
      fillPoison := True
      fillFaultCode := incomingFaultCode
      stickyFault := True
    }

    // Early last and missing last at the configured boundary both terminate
    // the poisoned image.  This prevents a malformed producer from writing
    // into the next slot/address indefinitely.
    when(io.quantized.payload.last || fillAtEnd) {
      val finalFault = fillPoison || !incomingBeatLegal
      val finalFaultCode = Bits(8 bits)
      finalFaultCode := incomingFaultCode
      when(fillPoison) { finalFaultCode := fillFaultCode }

      fillActiveReg := False
      fillCompletionValid := True
      fillCompletionPayload.imageTag := fillImageTag
      fillCompletionPayload.epoch := fillEpoch
      fillCompletionPayload.slot := fillSlot
      fillCompletionPayload.featureCount := fillFeatureCount
      fillCompletionPayload.acceptedFeatures :=
        (fillExpectedIndex.resize(cfg.featureCountWidth) + 1).resized
      fillCompletionPayload.tokenMask := fillTokenMask
      for (lane <- 0 until cfg.tokenLanes) {
        fillCompletionPayload.dequantFactorQ16(lane) := fillFactor(lane)
        when(fillAtFirst) {
          fillCompletionPayload.dequantFactorQ16(lane) :=
            io.quantized.payload.dequantFactorQ16(lane)
        }
      }
      fillCompletionPayload.success := !finalFault
      fillCompletionPayload.faultCode := finalFaultCode

      when(finalFault) {
        slotReadyReg(fillSlot) := False
        stickyFault := True
      } otherwise {
        slotReadyReg(fillSlot) := True
        slotImageTagReg(fillSlot) := fillImageTag
        slotEpochReg(fillSlot) := fillEpoch
        slotFeatureCountReg(fillSlot) := fillFeatureCount
        slotTileCountReg(fillSlot) :=
          (fillFeatureCount |>> 7).resize(cfg.tileCountWidth)
        slotTokenMaskReg(fillSlot) := fillTokenMask
        for (lane <- 0 until cfg.tokenLanes) {
          slotFactorReg(fillSlot)(lane) := fillFactor(lane)
        }
        completedFillCounter := completedFillCounter + 1
      }
    } otherwise {
      fillExpectedIndex := fillExpectedIndex + 1
    }
  }

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
    io.replayCommand.payload.startTile.resize(cfg.tileCountWidth) <
      slotTileCountReg(io.replayCommand.payload.slot)
  val replayLastOrdinal =
    io.replayCommand.payload.beatOrdinalBase.resize(m.beatCountWidth + 1) +
      io.replayCommand.payload.beatCount.resize(m.beatCountWidth + 1) - 1
  val replayNoOrdinalWrap = !replayLastOrdinal.msb
  val replayTargetBeingFilled = fillActiveReg &&
    fillSlot === io.replayCommand.payload.slot
  val replayCommandLegal = replaySlotReady && replayTagMatches &&
    replayFeatureCountMatches &&
    replayStartInRange && io.replayCommand.payload.beatCount =/= 0 &&
    replayNoOrdinalWrap && !replayTargetBeingFilled

  val responseSlot = Reg(UInt(cfg.slotWidth bits)) init (0)
  val responseImageTag = Reg(UInt(m.sequenceIdWidth bits)) init (0)
  val responseEpoch = Reg(UInt(cfg.epochWidth bits)) init (0)
  val responseOutputSequence = Reg(UInt(m.sequenceIdWidth bits)) init (0)
  val responseBeatOrdinal = Reg(UInt(m.beatCountWidth bits)) init (0)
  val responseFeatureTile = Reg(UInt(cfg.tileIndexWidth bits)) init (0)
  val responseFirst = RegInit(False)
  val responseLast = RegInit(False)

  when(io.replayCommand.fire) {
    when(replayCommandLegal) {
      replayActiveReg := True
      replayImageTag := io.replayCommand.payload.imageTag
      replayEpoch := io.replayCommand.payload.epoch
      replaySlot := io.replayCommand.payload.slot
      replayOutputSequence := io.replayCommand.payload.outputSequenceId
      replayBeatOrdinal := io.replayCommand.payload.beatOrdinalBase
      replayBeatCount := io.replayCommand.payload.beatCount
      replayIssuedCount := 0
      replayTileIndex := io.replayCommand.payload.startTile
      replayLastIssued := False
    } otherwise {
      stickyFault := True
      replayCompletionValid := True
      replayCompletionPayload.imageTag := io.replayCommand.payload.imageTag
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

  val readCanAdvance = !readResponseValid || io.tile.ready
  val readIssue = replayActiveReg && !replayLastIssued && readCanAdvance
  val issuingLast = replayIssuedCount === replayBeatCount - 1
  val replayAddress = UInt(cfg.bankAddressWidth bits)
  replayAddress := replayTileIndex.resized
  when(replaySlot === 1) {
    replayAddress := (replayTileIndex.resize(cfg.bankAddressWidth) +
      U(cfg.slotDepth, cfg.bankAddressWidth bits)).resized
  }
  val readData = Array.tabulate(cfg.bankCount) { bank =>
    activationMemories(bank).readSync(replayAddress, readIssue)
  }

  when(readIssue) {
    readResponseValid := True
    responseSlot := replaySlot
    responseImageTag := replayImageTag
    responseEpoch := replayEpoch
    responseOutputSequence := replayOutputSequence
    responseBeatOrdinal := replayBeatOrdinal
    responseFeatureTile := replayTileIndex
    responseFirst := replayIssuedCount === 0
    responseLast := issuingLast

    replayIssuedCount := replayIssuedCount + 1
    replayBeatOrdinal := replayBeatOrdinal + 1
    when(issuingLast) {
      replayLastIssued := True
    } otherwise {
      when(replayTileIndex.resize(cfg.tileCountWidth) + 1 ===
          slotTileCountReg(replaySlot)) {
        replayTileIndex := 0
      } otherwise {
        replayTileIndex := replayTileIndex + 1
      }
    }
  } elsewhen (io.tile.fire) {
    readResponseValid := False
  }

  io.tile.valid := readResponseValid
  io.tile.payload.normalized.sequenceId := responseOutputSequence
  io.tile.payload.normalized.beatOrdinal := responseBeatOrdinal
  io.tile.payload.normalized.fault := False
  for (bank <- 0 until cfg.bankCount;
       wordFeature <- 0 until cfg.featuresPerBankWord;
       lane <- 0 until cfg.tokenLanes) {
    val feature = bank * cfg.featuresPerBankWord + wordFeature
    val low = wordFeature * cfg.tokenLanes * b.activationWidth +
      lane * b.activationWidth
    io.tile.payload.normalized.tokenActivation(lane)(feature) :=
      readData(bank)(low + b.activationWidth - 1 downto low).asSInt
  }
  io.tile.payload.imageTag := responseImageTag
  io.tile.payload.epoch := responseEpoch
  io.tile.payload.slot := responseSlot
  io.tile.payload.featureTile := responseFeatureTile
  io.tile.payload.tokenMask := slotTokenMaskReg(responseSlot)
  for (lane <- 0 until cfg.tokenLanes) {
    io.tile.payload.dequantFactorQ16(lane) :=
      slotFactorReg(responseSlot)(lane)
  }
  io.tile.payload.first := responseFirst
  io.tile.payload.last := responseLast

  when(io.tile.fire && responseLast) {
    replayActiveReg := False
    replayCompletionValid := True
    replayCompletionPayload.imageTag := responseImageTag
    replayCompletionPayload.epoch := responseEpoch
    replayCompletionPayload.slot := responseSlot
    replayCompletionPayload.outputSequenceId := responseOutputSequence
    replayCompletionPayload.emittedBeats := replayBeatCount
    replayCompletionPayload.success := True
    replayCompletionPayload.faultCode := B(
      BitNetResidentActivationFault.None, 8 bits)
    completedReplayCounter := completedReplayCounter + 1
  }
}

object GenerateBitNetResidentActivationTileBuffer extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-activation-tile-buffer")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentActivationTileBuffer())
}
