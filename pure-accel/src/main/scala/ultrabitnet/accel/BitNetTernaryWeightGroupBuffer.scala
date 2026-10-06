package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Per-engine ping/pong storage for one compact group-of-four ternary row.
  *
  * The five-bank resident linker stores one byte per input feature.  Bits 3:0
  * encode the first output-row pair and bits 7:4 encode the second pair.  A
  * group is therefore exactly `inputFeatures` bytes, not four times that
  * value.  The buffer reads those compact bytes once and replays two complete
  * K sweeps to the continuous-array feeder.
  */
case class BitNetTernaryWeightGroupBufferConfig(
    feeder: BitNetContinuousMatrixFeederConfig =
      BitNetContinuousMatrixFeederConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    engineId: Int = 0,
    hiddenInputFeatures: Int = 2560,
    ffnInputFeatures: Int = 6912,
    epochWidth: Int = 8,
    replayFifoDepth: Int = 4
) {
  require(engineId >= 0 && engineId < feeder.engineCount)
  require(engineId < crossbar.matrixEngineClients)
  require(crossbar.dataWidth == 128)
  require(crossbar.maxBurstBeats == 256)
  require(feeder.matrix.base.siteCount == 64 ||
    feeder.matrix.base.siteCount == 128)
  require(hiddenInputFeatures > 0 && hiddenInputFeatures % 128 == 0)
  require(ffnInputFeatures > 0 && ffnInputFeatures % 128 == 0)
  require(epochWidth >= 4)
  require(replayFifoDepth >= 4 && isPow2(replayFifoDepth))

  val slotCount: Int = 2
  val beatBytes: Int = crossbar.dataWidth / 8
  val segmentsPerTile: Int = feeder.matrix.base.siteCount / beatBytes
  val maxInputFeatures: Int = scala.math.max(
    hiddenInputFeatures, ffnInputFeatures)
  val maxTiles: Int = maxInputFeatures / feeder.matrix.base.siteCount
  val maxReadBeats: Int = maxInputFeatures / beatBytes
  val inputFeatureWidth: Int = log2Up(maxInputFeatures + 1)
  val tileIndexWidth: Int = scala.math.max(1, log2Up(maxTiles))
  val memoryAddressWidth: Int = scala.math.max(1,
    log2Up(maxTiles * slotCount))
  val groupBeatWidth: Int = scala.math.max(1, log2Up(maxReadBeats + 1))
  val burstIndexWidth: Int = scala.math.max(1,
    log2Up((maxReadBeats + crossbar.maxBurstBeats - 1) /
      crossbar.maxBurstBeats))

  require(segmentsPerTile == 8,
    "one 128-feature tile must be assembled from eight 128-bit responses")
  require(maxReadBeats <= crossbar.maxBurstBeats * 2,
    "production 6912-byte groups require exactly two logical AXI requests")
}

case class BitNetTernaryWeightGroupCommand(
    cfg: BitNetTernaryWeightGroupBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  private val b = m.base

  val sequenceId = UInt(m.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(log2Up(cfg.slotCount) bits)
  val bank = UInt(cfg.crossbar.bankWidth bits)
  val localBase = UInt(cfg.crossbar.localAddressWidth bits)
  val inputFeatures = UInt(cfg.inputFeatureWidth bits)

  /** Group ordinal is local to this engine's current matrix sequence. */
  val groupOrdinal = UInt(b.outputIndexWidth bits)
  /** Absolute first output row of this compact group; it is four-row aligned. */
  val outputBase = UInt(b.outputIndexWidth bits)
  /** First feeder beat occupied by the lower pair of this group. */
  val beatOrdinalBase = UInt(m.beatCountWidth bits)
  val lastGroup = Bool()
}

/**
  * One-engine fragment of [[BitNetNormalizedWeightTile]].
  *
  * `sequenceId`, `beatOrdinal`, `weights` and `fault` map directly to the
  * corresponding normalized feeder boundary.  The remaining fields preserve
  * slot ownership for the two-engine join and verification logic.
  */
case class BitNetNormalizedWeightTileFragment(
    cfg: BitNetTernaryWeightGroupBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  private val b = m.base

  val sequenceId = UInt(m.sequenceIdWidth bits)
  val beatOrdinal = UInt(m.beatCountWidth bits)
  val engine = UInt(log2Up(cfg.feeder.engineCount) bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(log2Up(cfg.slotCount) bits)
  val groupOrdinal = UInt(b.outputIndexWidth bits)
  val outputBase = UInt(b.outputIndexWidth bits)
  val upperPair = Bool()
  val kTile = UInt(cfg.tileIndexWidth bits)
  val weights = Vec(Bits(8 bits), b.siteCount)
  val firstInGroup = Bool()
  val lastInGroup = Bool()
  val fault = Bool()
}

case class BitNetTernaryWeightGroupCompletion(
    cfg: BitNetTernaryWeightGroupBufferConfig) extends Bundle {
  private val m = cfg.feeder.matrix
  private val b = m.base

  val sequenceId = UInt(m.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val slot = UInt(log2Up(cfg.slotCount) bits)
  val bank = UInt(cfg.crossbar.bankWidth bits)
  val groupOrdinal = UInt(b.outputIndexWidth bits)
  val outputBase = UInt(b.outputIndexWidth bits)
  val inputFeatures = UInt(cfg.inputFeatureWidth bits)
  val readBeats = UInt(cfg.groupBeatWidth bits)
  val emittedTiles = UInt(log2Up(cfg.maxTiles * 2 + 1) bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
  val lastGroup = Bool()
}

object BitNetTernaryWeightGroupFault {
  val None: Int = 0x00
  val InvalidCommand: Int = 0xe0
  val ResponseMetadata: Int = 0xe1
  val AxiReadResponse: Int = 0xe2
  val ReplayOverflow: Int = 0xe3
}

object BitNetTernaryWeightGroupBufferMath {
  case class BeatLocation(segmentBank: Int, tile: Int)
  case class Burst(firstBeat: Int, beats: Int)
  case class NormalizedTile(
      beatOrdinal: Int,
      upperPair: Boolean,
      kTile: Int,
      weights: Vector[Int])

  def groupBytes(inputFeatures: Int): Int = {
    require(inputFeatures > 0 && inputFeatures % 128 == 0)
    inputFeatures
  }

  def kTiles(inputFeatures: Int): Int = groupBytes(inputFeatures) / 128

  def readBeats(inputFeatures: Int): Int = groupBytes(inputFeatures) / 16

  def beatLocation(beatIndex: Int): BeatLocation = {
    require(beatIndex >= 0)
    BeatLocation(segmentBank = beatIndex % 8, tile = beatIndex / 8)
  }

  def bursts(inputFeatures: Int, maxBurstBeats: Int = 256): Vector[Burst] = {
    require(maxBurstBeats > 0)
    val total = readBeats(inputFeatures)
    Vector.iterate(0, (total + maxBurstBeats - 1) / maxBurstBeats)(
      _ + maxBurstBeats).map { first =>
      Burst(first, scala.math.min(maxBurstBeats, total - first))
    }
  }

  /** Lower-pair full K sweep, then upper-pair full K sweep. */
  def normalize(
      compact: Seq[Int],
      beatOrdinalBase: Int = 0): Vector[NormalizedTile] = {
    require(compact.nonEmpty && compact.length % 128 == 0)
    require(compact.forall(value => value >= 0 && value <= 255))
    val tiles = compact.grouped(128).map(_.toVector).toVector
    Vector(false, true).flatMap { upper =>
      tiles.zipWithIndex.map { case (bytes, kTile) =>
        NormalizedTile(
          beatOrdinal = beatOrdinalBase +
            (if (upper) tiles.length else 0) + kTile,
          upperPair = upper,
          kTile = kTile,
          weights = bytes.map { byte =>
            if (upper) (byte >> 4) & 0x0f else byte & 0x0f
          })
      }
    }
  }
}

object BitNetTernaryWeightSlotState extends SpinalEnum(binarySequential) {
  val Empty, AwaitFill, Filling, Ready, Replaying, Complete = newElement()
}

/**
  * Per-engine double buffer for compact group-of-four ternary weights.
  *
  * The inactive slot can receive one 128-bit AXI beat every slow clock while
  * the active slot reads all eight segment memories and emits one 128-feature
  * tile every slow clock.  Slot state plus an explicit epoch prevents a stale
  * response or command from overwriting a group still owned by replay or a
  * backpressured completion.
  */
class BitNetTernaryWeightGroupBuffer(
    cfg: BitNetTernaryWeightGroupBufferConfig =
      BitNetTernaryWeightGroupBufferConfig()) extends Component {
  private val m = cfg.feeder.matrix
  private val b = m.base
  private val slotWidth = log2Up(cfg.slotCount)
  private val emittedCountWidth = log2Up(cfg.maxTiles * 2 + 1)

  val io = new Bundle {
    val command = slave(Stream(BitNetTernaryWeightGroupCommand(cfg)))
    val readRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar)))
    val readResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar)))
    val tile = master(Stream(BitNetNormalizedWeightTileFragment(cfg)))
    val completion = master(Stream(
      BitNetTernaryWeightGroupCompletion(cfg)))

    val slotOccupied = out Bits(cfg.slotCount bits)
    val slotReady = out Bits(cfg.slotCount bits)
    val fillActive = out Bool()
    val replayActive = out Bool()
    val sequenceActive = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedCommands = out UInt(32 bits)
    val issuedReadRequests = out UInt(32 bits)
    val acceptedReadBeats = out UInt(32 bits)
    val emittedTiles = out UInt(32 bits)
    val completedGroups = out UInt(32 bits)
  }

  val slotState = Vec.fill(cfg.slotCount)(
    Reg(BitNetTernaryWeightSlotState()) init
      BitNetTernaryWeightSlotState.Empty)
  val slotSequence = Vec.fill(cfg.slotCount)(
    Reg(UInt(m.sequenceIdWidth bits)) init 0)
  val slotEpoch = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.epochWidth bits)) init 0)
  val slotBank = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.crossbar.bankWidth bits)) init 0)
  val slotBase = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.crossbar.localAddressWidth bits)) init 0)
  val slotInputFeatures = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.inputFeatureWidth bits)) init 0)
  val slotGroupOrdinal = Vec.fill(cfg.slotCount)(
    Reg(UInt(b.outputIndexWidth bits)) init 0)
  val slotOutputBase = Vec.fill(cfg.slotCount)(
    Reg(UInt(b.outputIndexWidth bits)) init 0)
  val slotBeatOrdinalBase = Vec.fill(cfg.slotCount)(
    Reg(UInt(m.beatCountWidth bits)) init 0)
  val slotLastGroup = Vec.fill(cfg.slotCount)(Reg(Bool()) init False)
  val slotFault = Vec.fill(cfg.slotCount)(Reg(Bool()) init False)
  val slotFaultCode = Vec.fill(cfg.slotCount)(Reg(Bits(8 bits)) init 0)
  val slotReadBeatCount = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.groupBeatWidth bits)) init 0)
  val slotEmittedTileCount = Vec.fill(cfg.slotCount)(
    Reg(UInt(emittedCountWidth bits)) init 0)

  val occupied = Bits(cfg.slotCount bits)
  val ready = Bits(cfg.slotCount bits)
  for (slot <- 0 until cfg.slotCount) {
    occupied(slot) := slotState(slot) =/=
      BitNetTernaryWeightSlotState.Empty
    ready(slot) := slotState(slot) === BitNetTernaryWeightSlotState.Ready
  }
  io.slotOccupied := occupied
  io.slotReady := ready

  val sequenceOwned = RegInit(False)
  val activeSequence = Reg(UInt(m.sequenceIdWidth bits)) init 0
  val activeEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val activeInputFeatures = Reg(UInt(cfg.inputFeatureWidth bits)) init 0
  val nextReplayBeatOrdinal = Reg(UInt(m.beatCountWidth bits)) init 0
  val reservedBeatTail = Reg(UInt(m.beatCountWidth bits)) init 0
  val reservedGroupTail = Reg(UInt(b.outputIndexWidth bits)) init 0
  val reservedOutputTail = Reg(UInt(b.outputIndexWidth bits)) init 0
  val lastGroupReserved = RegInit(False)
  io.sequenceActive := sequenceOwned

  val stickyFault = RegInit(False)
  val acceptedCommandCount = Reg(UInt(32 bits)) init 0
  val issuedReadRequestCount = Reg(UInt(32 bits)) init 0
  val acceptedReadBeatCount = Reg(UInt(32 bits)) init 0
  val emittedTileCount = Reg(UInt(32 bits)) init 0
  val completedGroupCount = Reg(UInt(32 bits)) init 0
  io.protocolFaultSticky := stickyFault
  io.acceptedCommands := acceptedCommandCount
  io.issuedReadRequests := issuedReadRequestCount
  io.acceptedReadBeats := acceptedReadBeatCount
  io.emittedTiles := emittedTileCount
  io.completedGroups := completedGroupCount

  // Invalid, non-owning commands retire through a separate one-entry response.
  val rejectValid = RegInit(False)
  val rejectPayload = Reg(BitNetTernaryWeightGroupCompletion(cfg)) init (
    BitNetTernaryWeightGroupCompletion(cfg).getZero)

  val selectedCommandSlot = io.command.payload.slot
  val commandSlotEmpty = slotState(selectedCommandSlot) ===
    BitNetTernaryWeightSlotState.Empty
  val allSlotsEmpty = !occupied.orR
  io.command.ready := commandSlotEmpty && !rejectValid &&
    (sequenceOwned || allSlotsEmpty)

  val inputFeaturesSupported =
    io.command.payload.inputFeatures === cfg.hiddenInputFeatures ||
      io.command.payload.inputFeatures === cfg.ffnInputFeatures
  val commandBankLegal = io.command.payload.bank < cfg.crossbar.bankCount
  val commandBaseAligned = io.command.payload.localBase(6 downto 0) === 0
  val commandOutputAligned = io.command.payload.outputBase(1 downto 0) === 0
  val identityLegal = !sequenceOwned ||
    (io.command.payload.sequenceId === activeSequence &&
      io.command.payload.epoch === activeEpoch &&
      io.command.payload.inputFeatures === activeInputFeatures)
  val firstGroupLegal = sequenceOwned ||
    (io.command.payload.groupOrdinal === 0 &&
      io.command.payload.beatOrdinalBase === 0)
  val reservationContiguous = !sequenceOwned ||
    (!lastGroupReserved &&
      io.command.payload.groupOrdinal === reservedGroupTail &&
      io.command.payload.outputBase === reservedOutputTail &&
      io.command.payload.beatOrdinalBase === reservedBeatTail)
  val commandLegal = inputFeaturesSupported && commandBankLegal &&
    commandBaseAligned && commandOutputAligned && identityLegal &&
    firstGroupLegal && reservationContiguous

  when(io.command.fire) {
    when(commandLegal) {
      val slot = selectedCommandSlot
      slotState(slot) := BitNetTernaryWeightSlotState.AwaitFill
      slotSequence(slot) := io.command.payload.sequenceId
      slotEpoch(slot) := io.command.payload.epoch
      slotBank(slot) := io.command.payload.bank
      slotBase(slot) := io.command.payload.localBase
      slotInputFeatures(slot) := io.command.payload.inputFeatures
      slotGroupOrdinal(slot) := io.command.payload.groupOrdinal
      slotOutputBase(slot) := io.command.payload.outputBase
      slotBeatOrdinalBase(slot) := io.command.payload.beatOrdinalBase
      slotLastGroup(slot) := io.command.payload.lastGroup
      slotFault(slot) := False
      slotFaultCode(slot) := B(0, 8 bits)
      slotReadBeatCount(slot) := U(0, cfg.groupBeatWidth bits)
      slotEmittedTileCount(slot) := U(0, emittedCountWidth bits)
      acceptedCommandCount := acceptedCommandCount + 1
      val commandKTileCount =
        (io.command.payload.inputFeatures >> 7).resize(m.beatCountWidth)
      reservedBeatTail := (
        io.command.payload.beatOrdinalBase + (commandKTileCount << 1)
      ).resize(m.beatCountWidth)
      reservedGroupTail := io.command.payload.groupOrdinal + 1
      reservedOutputTail := io.command.payload.outputBase + 4
      when(io.command.payload.lastGroup) { lastGroupReserved := True }
      when(!sequenceOwned) {
        sequenceOwned := True
        activeSequence := io.command.payload.sequenceId
        activeEpoch := io.command.payload.epoch
        activeInputFeatures := io.command.payload.inputFeatures
        nextReplayBeatOrdinal := 0
        lastGroupReserved := io.command.payload.lastGroup
      }
    } otherwise {
      rejectValid := True
      rejectPayload.sequenceId := io.command.payload.sequenceId
      rejectPayload.epoch := io.command.payload.epoch
      rejectPayload.slot := io.command.payload.slot
      rejectPayload.bank := io.command.payload.bank
      rejectPayload.groupOrdinal := io.command.payload.groupOrdinal
      rejectPayload.outputBase := io.command.payload.outputBase
      rejectPayload.inputFeatures := io.command.payload.inputFeatures
      rejectPayload.readBeats := 0
      rejectPayload.emittedTiles := 0
      rejectPayload.success := False
      rejectPayload.faultCode := B(
        BitNetTernaryWeightGroupFault.InvalidCommand, 8 bits)
      rejectPayload.lastGroup := io.command.payload.lastGroup
      stickyFault := True
    }
  }

  // Eight 128-bit simple-dual-port banks. Slot ownership is part of the
  // address (slot 0: 0..53, slot 1: 54..107), halving the BRAM36 count versus
  // sixteen independent depth-54 memories while preserving one concurrent
  // AXI write and one replay read on different slots.
  val weightMemories = Array.tabulate(cfg.segmentsPerTile) { segment =>
      val memory = Mem(Bits(cfg.crossbar.dataWidth bits),
        cfg.maxTiles * cfg.slotCount)
      memory.setName(s"weightSegment${segment}")
      memory.addAttribute("ram_style", "block")
      memory
  }

  object FillState extends SpinalEnum(binarySequential) {
    val Idle, IssueBurst, ReceiveBurst = newElement()
  }
  val fillState = Reg(FillState()) init FillState.Idle
  val fillSlot = Reg(UInt(slotWidth bits)) init 0
  val fillBurstBase = Reg(UInt(cfg.groupBeatWidth bits)) init 0
  val fillBurstIndex = Reg(UInt(cfg.burstIndexWidth bits)) init 0
  val fillResponseBeat = Reg(UInt(cfg.crossbar.beatIndexWidth bits)) init 0

  val awaitingFillExists = Bool()
  val awaitingFillSlot = UInt(slotWidth bits)
  awaitingFillExists := False
  awaitingFillSlot := 0
  for (slot <- (0 until cfg.slotCount).reverse) {
    when(slotState(slot) === BitNetTernaryWeightSlotState.AwaitFill) {
      awaitingFillExists := True
      awaitingFillSlot := slot
    }
  }
  when(fillState === FillState.Idle && awaitingFillExists) {
    fillSlot := awaitingFillSlot
    slotState(awaitingFillSlot) := BitNetTernaryWeightSlotState.Filling
    fillBurstBase := 0
    fillBurstIndex := 0
    fillResponseBeat := 0
    fillState := FillState.IssueBurst
  }

  val fillTotalBeats = (slotInputFeatures(fillSlot) >> 4)
    .resize(cfg.groupBeatWidth)
  val fillRemaining = (fillTotalBeats - fillBurstBase)
    .resize(cfg.groupBeatWidth)
  val currentBurstBeats = UInt(cfg.crossbar.beatCountWidth bits)
  currentBurstBeats := fillRemaining.resized
  when(fillRemaining > cfg.crossbar.maxBurstBeats) {
    currentBurstBeats := cfg.crossbar.maxBurstBeats
  }

  val expectedTransaction = (
    slotEpoch(fillSlot).resize(cfg.crossbar.transactionIdWidth) ^
      (fillSlot.resize(cfg.crossbar.transactionIdWidth) << 1).resize(
        cfg.crossbar.transactionIdWidth) ^
      fillBurstIndex.resize(cfg.crossbar.transactionIdWidth)
  ).resize(cfg.crossbar.transactionIdWidth)
  val expectedTag = (
    slotSequence(fillSlot).resize(cfg.crossbar.tagWidth) ^
      (slotEpoch(fillSlot).resize(cfg.crossbar.tagWidth) << 12).resize(
        cfg.crossbar.tagWidth) ^
      (slotGroupOrdinal(fillSlot).resize(cfg.crossbar.tagWidth) << 2).resize(
        cfg.crossbar.tagWidth) ^
      (fillSlot.resize(cfg.crossbar.tagWidth) << 1).resize(
        cfg.crossbar.tagWidth) ^
      fillBurstIndex.resize(cfg.crossbar.tagWidth)
  ).resize(cfg.crossbar.tagWidth)

  io.readRequest.valid := fillState === FillState.IssueBurst
  io.readRequest.bank := slotBank(fillSlot)
  io.readRequest.transactionId := expectedTransaction
  io.readRequest.tag := expectedTag
  io.readRequest.localAddress := (
    slotBase(fillSlot) +
      (fillBurstBase.resize(cfg.crossbar.localAddressWidth) << 4)
  ).resize(cfg.crossbar.localAddressWidth)
  io.readRequest.beats := currentBurstBeats

  when(io.readRequest.fire) {
    fillResponseBeat := 0
    issuedReadRequestCount := issuedReadRequestCount + 1
    fillState := FillState.ReceiveBurst
  }

  io.readResponse.ready := fillState === FillState.ReceiveBurst
  val expectedResponseLast = fillResponseBeat ===
    (currentBurstBeats - 1).resize(cfg.crossbar.beatIndexWidth)
  val responseMetadataLegal =
    io.readResponse.bank === slotBank(fillSlot) &&
      io.readResponse.transactionId === expectedTransaction &&
      io.readResponse.tag === expectedTag &&
      io.readResponse.beatIndex === fillResponseBeat &&
      io.readResponse.last === expectedResponseLast
  val responseAxiLegal = !io.readResponse.fault &&
    io.readResponse.rresp === B"2'b00"
  val responseLegal = responseMetadataLegal && responseAxiLegal

  val globalFillBeat = (fillBurstBase +
    fillResponseBeat.resize(cfg.groupBeatWidth)).resize(cfg.groupBeatWidth)
  private val segmentIndexWidth = log2Up(cfg.segmentsPerTile)
  val writeSegment = globalFillBeat(segmentIndexWidth - 1 downto 0)
  val writeTile = (globalFillBeat >> segmentIndexWidth)
    .resize(cfg.tileIndexWidth)
  val writeMemoryAddress = UInt(cfg.memoryAddressWidth bits)
  writeMemoryAddress := writeTile.resized
  when(fillSlot === 1) {
    writeMemoryAddress := (U(cfg.maxTiles, cfg.memoryAddressWidth bits) +
      writeTile.resize(cfg.memoryAddressWidth)).resized
  }

  when(io.readResponse.fire) {
    acceptedReadBeatCount := acceptedReadBeatCount + 1
    slotReadBeatCount(fillSlot) := slotReadBeatCount(fillSlot) + 1
    when(!responseLegal) {
      stickyFault := True
      when(!slotFault(fillSlot)) {
        slotFault(fillSlot) := True
        slotFaultCode(fillSlot) := B(
          BitNetTernaryWeightGroupFault.ResponseMetadata, 8 bits)
        when(responseMetadataLegal && !responseAxiLegal) {
          slotFaultCode(fillSlot) := B(
            BitNetTernaryWeightGroupFault.AxiReadResponse, 8 bits)
        }
      }
    }

    for (segment <- 0 until cfg.segmentsPerTile) {
      when(responseLegal && writeSegment === segment) {
        weightMemories(segment).write(
          address = writeMemoryAddress,
          data = io.readResponse.data)
      }
    }

    when(expectedResponseLast) {
      val nextBurstBase = (fillBurstBase + currentBurstBeats.resize(
        cfg.groupBeatWidth)).resize(cfg.groupBeatWidth)
      when(nextBurstBase === fillTotalBeats) {
        when(slotFault(fillSlot) || !responseLegal) {
          slotState(fillSlot) := BitNetTernaryWeightSlotState.Complete
        } otherwise {
          slotState(fillSlot) := BitNetTernaryWeightSlotState.Ready
        }
        fillState := FillState.Idle
      } otherwise {
        fillBurstBase := nextBurstBase
        fillBurstIndex := fillBurstIndex + 1
        fillResponseBeat := 0
        fillState := FillState.IssueBurst
      }
    } otherwise {
      fillResponseBeat := fillResponseBeat + 1
    }
  }

  // -----------------------------------------------------------------------
  // Synchronous eight-bank replay with a reservation-backed elastic FIFO.
  // -----------------------------------------------------------------------
  val replaying = RegInit(False)
  val replaySlot = Reg(UInt(slotWidth bits)) init 0
  val replayUpperPair = RegInit(False)
  val replayTileIndex = Reg(UInt(cfg.tileIndexWidth bits)) init 0
  val allReplayReadsIssued = RegInit(False)
  io.replayActive := replaying
  io.fillActive := fillState =/= FillState.Idle

  val replayCandidateExists = Bool()
  val replayCandidateSlot = UInt(slotWidth bits)
  replayCandidateExists := False
  replayCandidateSlot := 0
  for (slot <- (0 until cfg.slotCount).reverse) {
    when(slotState(slot) === BitNetTernaryWeightSlotState.Ready &&
      slotBeatOrdinalBase(slot) === nextReplayBeatOrdinal) {
      replayCandidateExists := True
      replayCandidateSlot := slot
    }
  }

  when(!replaying && replayCandidateExists) {
    replaying := True
    replaySlot := replayCandidateSlot
    replayUpperPair := False
    replayTileIndex := 0
    allReplayReadsIssued := False
    slotState(replayCandidateSlot) :=
      BitNetTernaryWeightSlotState.Replaying
  }

  val replayFifo = new StreamFifo(
    BitNetNormalizedWeightTileFragment(cfg), cfg.replayFifoDepth)
  val replayResponseValid = RegInit(False)
  // Reserve FIFO capacity for both the synchronous-memory response already
  // in flight and the read issued this cycle. A simultaneous pop contributes
  // capacity immediately, so replay restarts without a bubble after arbitrary
  // downstream backpressure.
  val replayReservationWidth = log2Up(cfg.replayFifoDepth + 2)
  val replayPopFire = replayFifo.io.pop.valid && io.tile.ready
  val replayCapacityAfterCurrent = UInt(replayReservationWidth bits)
  replayCapacityAfterCurrent :=
    replayFifo.io.availability.resize(replayReservationWidth)
  when(replayPopFire) {
    replayCapacityAfterCurrent :=
      replayFifo.io.availability.resize(replayReservationWidth) + 1
  }
  val replayCommittedResponse =
    replayResponseValid.asUInt.resize(replayReservationWidth)
  val replayReadIssue = replaying && !allReplayReadsIssued &&
    replayCapacityAfterCurrent > replayCommittedResponse
  val replayKTileCount = (slotInputFeatures(replaySlot) >> 7)
    .resize(cfg.tileIndexWidth + 1)

  val replayMemoryAddress = UInt(cfg.memoryAddressWidth bits)
  replayMemoryAddress := replayTileIndex.resized
  when(replaySlot === 1) {
    replayMemoryAddress := (U(cfg.maxTiles, cfg.memoryAddressWidth bits) +
      replayTileIndex.resize(cfg.memoryAddressWidth)).resized
  }
  val replayReadData = Array.tabulate(cfg.segmentsPerTile) { segment =>
      weightMemories(segment).readSync(
        address = replayMemoryAddress,
        enable = replayReadIssue)
  }
  replayResponseValid := replayReadIssue
  val replayResponseSlot = RegNextWhen(replaySlot, replayReadIssue) init 0
  val replayResponseUpper = RegNextWhen(
    replayUpperPair, replayReadIssue) init False
  val replayResponseTile = RegNextWhen(
    replayTileIndex, replayReadIssue) init 0
  val replayResponseFirst = RegNextWhen(
    !replayUpperPair && replayTileIndex === 0, replayReadIssue) init False
  val replayResponseLast = RegNextWhen(
    replayUpperPair &&
      replayTileIndex.resize(cfg.tileIndexWidth + 1) ===
        replayKTileCount - 1,
    replayReadIssue) init False

  val selectedReadData = Vec(Bits(cfg.crossbar.dataWidth bits),
    cfg.segmentsPerTile)
  for (segment <- 0 until cfg.segmentsPerTile) {
    selectedReadData(segment) := replayReadData(segment)
  }

  replayFifo.io.push.valid := replayResponseValid
  replayFifo.io.push.sequenceId := slotSequence(replayResponseSlot)
  replayFifo.io.push.engine := cfg.engineId
  replayFifo.io.push.epoch := slotEpoch(replayResponseSlot)
  replayFifo.io.push.slot := replayResponseSlot
  replayFifo.io.push.groupOrdinal := slotGroupOrdinal(replayResponseSlot)
  replayFifo.io.push.outputBase := slotOutputBase(replayResponseSlot)
  replayFifo.io.push.upperPair := replayResponseUpper
  replayFifo.io.push.kTile := replayResponseTile
  replayFifo.io.push.beatOrdinal := (
    slotBeatOrdinalBase(replayResponseSlot) +
      Mux(replayResponseUpper,
        (slotInputFeatures(replayResponseSlot) >> 7)
          .resize(m.beatCountWidth),
        U(0, m.beatCountWidth bits)) +
      replayResponseTile.resize(m.beatCountWidth)
  ).resize(m.beatCountWidth)
  replayFifo.io.push.firstInGroup := replayResponseFirst
  replayFifo.io.push.lastInGroup := replayResponseLast
  replayFifo.io.push.fault := False
  for (feature <- 0 until b.siteCount) {
    val segment = feature / cfg.beatBytes
    val byte = feature % cfg.beatBytes
    val compact = selectedReadData(segment)(byte * 8 + 7 downto byte * 8)
    replayFifo.io.push.weights(feature) := B(0, 4 bits) ## compact(3 downto 0)
    when(replayResponseUpper) {
      replayFifo.io.push.weights(feature) := B(0, 4 bits) ## compact(7 downto 4)
    }
  }

  when(replayResponseValid && !replayFifo.io.push.ready) {
    stickyFault := True
    slotFault(replayResponseSlot) := True
    slotFaultCode(replayResponseSlot) := B(
      BitNetTernaryWeightGroupFault.ReplayOverflow, 8 bits)
  }

  when(replayReadIssue) {
    when(replayTileIndex.resize(cfg.tileIndexWidth + 1) ===
        replayKTileCount - 1) {
      replayTileIndex := 0
      when(replayUpperPair) {
        allReplayReadsIssued := True
      } otherwise {
        replayUpperPair := True
      }
    } otherwise {
      replayTileIndex := replayTileIndex + 1
    }
  }

  io.tile << replayFifo.io.pop
  when(io.tile.fire) {
    emittedTileCount := emittedTileCount + 1
    slotEmittedTileCount(io.tile.slot) :=
      slotEmittedTileCount(io.tile.slot) + 1
    when(io.tile.lastInGroup) {
      val completedSlot = io.tile.slot
      val completedKTileCount =
        (slotInputFeatures(completedSlot) >> 7).resize(m.beatCountWidth)
      slotState(completedSlot) := BitNetTernaryWeightSlotState.Complete
      nextReplayBeatOrdinal := (
        slotBeatOrdinalBase(completedSlot) +
          (completedKTileCount << 1)
      ).resize(m.beatCountWidth)
      replaying := False
      allReplayReadsIssued := False
    }
  }

  // -----------------------------------------------------------------------
  // Backpressure-safe completion; an owned slot is released only on fire.
  // -----------------------------------------------------------------------
  val completeExists = Bool()
  val completeSlot = UInt(slotWidth bits)
  completeExists := False
  completeSlot := 0
  when(slotState(0) === BitNetTernaryWeightSlotState.Complete) {
    completeExists := True
    completeSlot := 0
  }
  when(slotState(1) === BitNetTernaryWeightSlotState.Complete &&
    (slotState(0) =/= BitNetTernaryWeightSlotState.Complete ||
      slotBeatOrdinalBase(1) < slotBeatOrdinalBase(0))) {
    completeExists := True
    completeSlot := 1
  }

  io.completion.valid := rejectValid || completeExists
  io.completion.payload := rejectPayload
  when(!rejectValid) {
    io.completion.sequenceId := slotSequence(completeSlot)
    io.completion.epoch := slotEpoch(completeSlot)
    io.completion.slot := completeSlot
    io.completion.bank := slotBank(completeSlot)
    io.completion.groupOrdinal := slotGroupOrdinal(completeSlot)
    io.completion.outputBase := slotOutputBase(completeSlot)
    io.completion.inputFeatures := slotInputFeatures(completeSlot)
    io.completion.readBeats := slotReadBeatCount(completeSlot)
    io.completion.emittedTiles := slotEmittedTileCount(completeSlot)
    io.completion.success := !slotFault(completeSlot)
    io.completion.faultCode := slotFaultCode(completeSlot)
    io.completion.lastGroup := slotLastGroup(completeSlot)
  }

  when(io.completion.fire) {
    completedGroupCount := completedGroupCount + 1
    when(rejectValid) {
      rejectValid := False
    } otherwise {
      slotState(completeSlot) := BitNetTernaryWeightSlotState.Empty
      when(slotLastGroup(completeSlot) || slotFault(completeSlot)) {
        sequenceOwned := False
        lastGroupReserved := False
      }
    }
  }
}

object GenerateBitNetTernaryWeightGroupBuffer extends App {
  SpinalConfig(
    targetDirectory = "build/bitnet-ternary-weight-group-buffer",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetTernaryWeightGroupBuffer())
}
