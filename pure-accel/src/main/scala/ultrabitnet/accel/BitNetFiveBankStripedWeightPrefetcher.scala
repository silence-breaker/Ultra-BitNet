package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Production geometry for the five-bank, K-striped weight synchroniser. */
case class BitNetFiveBankStripedWeightPrefetcherConfig(
    model: BitNetConfig = BitNetConfig.Production,
    feeder: BitNetContinuousMatrixFeederConfig =
      BitNetContinuousMatrixFeederConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    epochWidth: Int = 8
) {
  require(model == BitNetConfig.Production)
  require(model.engineCount == feeder.engineCount &&
    (feeder.engineCount == 1 || feeder.engineCount == 2))
  require(crossbar.bankCount == 5 && crossbar.dataWidth == 128)
  require(feeder.matrix.base.siteCount == 64 ||
    feeder.matrix.base.siteCount == 128)
  require(model.hiddenSize == 2560 && model.ffnSize == 6912)
  require(epochWidth >= 4)

  val poolCount: Int = feeder.engineCount
  val slotCount: Int = 2
  val bankCount: Int = 5
  val beatBytes: Int = crossbar.dataWidth / 8
  val segmentsPerTile: Int = feeder.matrix.base.siteCount / beatBytes
  val maxInputFeatures: Int = model.ffnSize
  val maxTiles: Int = maxInputFeatures / feeder.matrix.base.siteCount
  val memoryDepth: Int = maxTiles * slotCount
  val inputFeatureWidth: Int = log2Up(maxInputFeatures + 1)
  val tileIndexWidth: Int = log2Up(maxTiles)
  val memoryAddressWidth: Int = log2Up(memoryDepth)
  val stripeBeatWidth: Int = log2Up(1408 / beatBytes + 1)
  val totalReadBeatWidth: Int = log2Up(maxInputFeatures / beatBytes + 1)
  val rawGroupCountWidth: Int = log2Up(maxTiles + 1)
  val slotWidth: Int = log2Up(slotCount)
  val poolWidth: Int = Math.max(1, log2Up(poolCount))

  require(segmentsPerTile == 4 || segmentsPerTile == 8)
  require(maxTiles == maxInputFeatures / feeder.matrix.base.siteCount &&
    memoryDepth == maxTiles * slotCount)
  require(1408 / beatBytes <= crossbar.maxBurstBeats)
}

/** One engine's group address vector, normally supplied by the striped AGU. */
case class BitNetStripedWeightEngineGroup(
    cfg: BitNetFiveBankStripedWeightPrefetcherConfig) extends Bundle {
  private val matrix = cfg.feeder.matrix
  private val base = matrix.base

  val groupOrdinal = UInt(base.outputIndexWidth bits)
  val outputBase = UInt(base.outputIndexWidth bits)
  val beatOrdinalBase = UInt(matrix.beatCountWidth bits)
  val localBase = Vec(
    UInt(cfg.crossbar.localAddressWidth bits), cfg.bankCount)
}

/**
  * One canonical prefill group or one pair of independent decode groups.
  *
  * `inputStart`/`inputFeatures` and `engine(e).localBase` are the typed output
  * of two striped transformer AGUs.  Prefill requires the two engine
  * descriptors to be identical and physically fills pool 0 once.  Decode
  * fills both pools concurrently and later joins their canonical K tiles.
  */
case class BitNetFiveBankStripedWeightCommand(
    cfg: BitNetFiveBankStripedWeightPrefetcherConfig) extends Bundle {
  private val matrix = cfg.feeder.matrix

  val sequenceId = UInt(matrix.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val mode = BitNetMode()
  val target = BitNetAddressTarget()
  val layer = UInt(cfg.model.layerWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val inputStart = Vec(UInt(cfg.inputFeatureWidth bits), cfg.bankCount)
  val inputFeatures = Vec(UInt(cfg.inputFeatureWidth bits), cfg.bankCount)
  val engine = Vec(BitNetStripedWeightEngineGroup(cfg), cfg.poolCount)
  val lastGroup = Bool()
}

case class BitNetFiveBankStripedWeightCompletion(
    cfg: BitNetFiveBankStripedWeightPrefetcherConfig) extends Bundle {
  private val matrix = cfg.feeder.matrix
  private val base = matrix.base

  val sequenceId = UInt(matrix.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val mode = BitNetMode()
  val target = BitNetAddressTarget()
  val layer = UInt(cfg.model.layerWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val activePoolMask = Bits(cfg.poolCount bits)
  val groupOrdinal = Vec(UInt(base.outputIndexWidth bits), cfg.poolCount)
  val outputBase = Vec(UInt(base.outputIndexWidth bits), cfg.poolCount)
  val beatOrdinalBase = Vec(
    UInt(matrix.beatCountWidth bits), cfg.poolCount)
  val bankReadBeats = Vec(
    Vec(UInt(cfg.stripeBeatWidth bits), cfg.bankCount), cfg.poolCount)
  /** Physical 128-byte group-of-four tiles emitted by the prefetcher. */
  val emittedRawGroups = Vec(
    UInt(cfg.rawGroupCountWidth bits), cfg.poolCount)
  val success = Bool()
  val faultCode = Bits(8 bits)
  val lastGroup = Bool()
}

object BitNetFiveBankStripedWeightFault {
  val None: Int = 0x00
  val InvalidCommand: Int = 0xd0
  val ResponseMetadata: Int = 0xd1
  val AxiReadResponse: Int = 0xd2
  val ReplayOverflow: Int = 0xd3
  val PartnerPoolFault: Int = 0xd4
}

object BitNetFiveBankStripedWeightMath {
  case class Stripe(inputStart: Int, inputFeatures: Int)
  case class Placement(tile: Int, segment: Int, memory: Int)
  case class RawGroupTile(
      lowerBeatOrdinal: Int,
      upperBeatOrdinal: Int,
      kTile: Int,
      firstRawGroup: Boolean,
      lastRawGroup: Boolean,
      engineRawGroup: Vector[Vector[Int]])

  def stripes(inputFeatures: Int): Vector[Stripe] = {
    require(inputFeatures == 2560 || inputFeatures == 6912)
    Vector.tabulate(5) { bank =>
      val (start, count) =
        BitNetFiveBankStripedLayout.partitionInput(inputFeatures, bank)
      Stripe(start, count)
    }
  }

  /**
    * Rotate each bank's eight 16-byte segments across the eight shared BRAMs.
    * Equal local beat indices from all five banks therefore target five
    * different memories, while replay still reads one word from every memory.
    */
  def placement(bank: Int, inputStart: Int, localBeat: Int): Placement = {
    require(bank >= 0 && bank < 5)
    require(inputStart >= 0 && inputStart % 128 == 0)
    require(localBeat >= 0)
    val globalBeat = inputStart / 16 + localBeat
    val segment = globalBeat % 8
    Placement(
      tile = globalBeat / 8,
      segment = segment,
      memory = (bank + segment) % 8)
  }

  def assemble(stripes: Seq[(Int, Seq[Int])], inputFeatures: Int): Vector[Int] = {
    val output = Array.fill(inputFeatures)(-1)
    stripes.foreach { case (start, bytes) =>
      require(start >= 0 && start + bytes.length <= inputFeatures)
      bytes.zipWithIndex.foreach { case (value, index) =>
        require(value >= 0 && value <= 255)
        require(output(start + index) == -1)
        output(start + index) = value
      }
    }
    require(output.forall(_ >= 0))
    output.toVector
  }

  /** Software oracle for the packed group-of-four prefetcher boundary. */
  def rawGroups(
      compactPerEngine: Seq[Seq[Int]],
      mode: BitNetMode.E,
      beatOrdinalBase: Int): Vector[RawGroupTile] = {
    require(compactPerEngine.size == 2)
    require(compactPerEngine.forall(bytes =>
      bytes.nonEmpty && bytes.length % 128 == 0))
    require(compactPerEngine.map(_.length).distinct.size == 1)
    val tiles = compactPerEngine.head.length / 128
    Vector.tabulate(tiles) { tile =>
      val packed = compactPerEngine.map(_.slice(
        tile * 128, (tile + 1) * 128).toVector).toVector
      val routed = if (mode == BitNetMode.Prefill)
        Vector(packed.head, packed.head) else packed
      RawGroupTile(
        lowerBeatOrdinal = beatOrdinalBase + tile,
        upperBeatOrdinal = beatOrdinalBase + tiles + tile,
        kTile = tile,
        firstRawGroup = tile == 0,
        lastRawGroup = tile == tiles - 1,
        engineRawGroup = routed)
    }
  }
}

object BitNetStripedWeightSlotState extends SpinalEnum(binarySequential) {
  val Empty, AwaitFill, Filling, Ready, Replaying, Complete = newElement()
}

object BitNetStripedWeightRoundState extends SpinalEnum(binarySequential) {
  val Empty, Active, Complete = newElement()
}

/**
  * Five-bank K-striped production weight prefetcher.
  *
  * The design owns two independent replay pools.  Each pool contains exactly
  * eight 128-bit, depth-108 memories; the slot bit is folded into the address.
  * Bank-dependent segment rotation permits five simultaneous writes without
  * replicating a complete group per bank.  Replay reads all eight memories in
  * parallel and restores natural segment order at one 128-feature tile per
  * slow clock.
  */
class BitNetFiveBankStripedWeightPrefetcher(
    cfg: BitNetFiveBankStripedWeightPrefetcherConfig =
      BitNetFiveBankStripedWeightPrefetcherConfig(),
    externalWeightTileCache: Boolean = false) extends Component {
  private val matrix = cfg.feeder.matrix
  private val base = matrix.base
  private val segmentIndexWidth = log2Up(cfg.segmentsPerTile)
  private val targets = BitNetFiveBankStripedLayout.WeightGeometry.map(_._1)
  private val tileCacheCfg = BitNetResidentSharedWeightTileCacheConfig(
    dataWidth = cfg.crossbar.dataWidth,
    poolCount = cfg.poolCount,
    memoryCount = cfg.segmentsPerTile,
    depth = cfg.memoryDepth,
    sequenceIdWidth = matrix.sequenceIdWidth,
    sourceEpochWidth = cfg.epochWidth)

  val io = new Bundle {
    val command = slave(Stream(BitNetFiveBankStripedWeightCommand(cfg)))
    val readRequest = Vec(master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar))), cfg.bankCount)
    val readResponse = Vec(slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar))), cfg.bankCount)
    val tile = master(Stream(BitNetMap0RawGroupTile(
      cfg.feeder, cfg.maxInputFeatures)))
    val completion = master(Stream(
      BitNetFiveBankStripedWeightCompletion(cfg)))

    // The production full shell connects these ports to the one physical
    // Matrix/LM tile cache.  Standalone/default instances keep their private
    // memories so the focused prefetch regression remains self-contained.
    val cacheWrite = Vec.fill(cfg.poolCount)(Vec.fill(cfg.segmentsPerTile)(
      master(Stream(BitNetResidentWeightTileCacheWrite(tileCacheCfg)))))
    val cacheRead = master(Stream(
      BitNetResidentWeightTileCacheRead(tileCacheCfg)))
    val cacheReadData = Vec.fill(cfg.poolCount)(Vec.fill(
      cfg.segmentsPerTile)(in Bits(cfg.crossbar.dataWidth bits)))
    val cacheAcquire = master(Stream(
      BitNetResidentWeightTileCacheAcquire(tileCacheCfg)))
    val cacheGrant = slave(Stream(
      BitNetResidentWeightTileCacheGrant(tileCacheCfg)))
    // The shared cache exposes its currently authenticated generation as an
    // independent sideband.  It is deliberately absent from private-cache
    // elaborations, where no external lease can authenticate memory access.
    val cacheActiveGeneration = if (externalWeightTileCache)
      in UInt(tileCacheCfg.generationWidth bits) else null
    val cacheRelease = master(Stream(
      BitNetResidentWeightTileCacheRelease(tileCacheCfg)))

    val slotOccupied = Vec(out Bits(cfg.slotCount bits), cfg.poolCount)
    val fillActive = out Bits(cfg.poolCount bits)
    val replayActive = out Bool()
    val sequenceActive = out Bool()
    /**
      * A malformed shared-cache grant is a permanent frontend fail-stop.
      * Reset is required before this client may acquire or use the cache again.
      */
    val cacheLeaseQuarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedCommands = out UInt(32 bits)
    val issuedReadRequests = out UInt(32 bits)
    val acceptedReadBeats = out UInt(32 bits)
    val emittedRawGroups = out UInt(32 bits)
    val completedGroups = out UInt(32 bits)
  }

  if (!externalWeightTileCache) {
    for (pool <- 0 until cfg.poolCount;
         memory <- 0 until cfg.segmentsPerTile) {
      io.cacheWrite(pool)(memory).valid := False
      io.cacheWrite(pool)(memory).payload :=
        BitNetResidentWeightTileCacheWrite(tileCacheCfg).getZero
    }
    io.cacheRead.valid := False
    io.cacheRead.payload := BitNetResidentWeightTileCacheRead(
      tileCacheCfg).getZero
    io.cacheAcquire.valid := False
    io.cacheAcquire.payload := BitNetResidentWeightTileCacheAcquire(
      tileCacheCfg).getZero
    io.cacheGrant.ready := False
    io.cacheRelease.valid := False
    io.cacheRelease.payload := BitNetResidentWeightTileCacheRelease(
      tileCacheCfg).getZero
  }

  // -----------------------------------------------------------------------
  // Round and slot ownership.
  // -----------------------------------------------------------------------
  val roundState = Vec.fill(cfg.slotCount)(
    Reg(BitNetStripedWeightRoundState()) init
      BitNetStripedWeightRoundState.Empty)
  val poolState = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Reg(BitNetStripedWeightSlotState()) init
      BitNetStripedWeightSlotState.Empty))

  val slotSequence = Vec.fill(cfg.slotCount)(
    Reg(UInt(matrix.sequenceIdWidth bits)) init 0)
  val slotEpoch = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.epochWidth bits)) init 0)
  val slotMode = Vec.fill(cfg.slotCount)(Reg(BitNetMode()) init
    BitNetMode.Prefill)
  val slotTarget = Vec.fill(cfg.slotCount)(Reg(BitNetAddressTarget()) init
    BitNetAddressTarget.QWeight)
  val slotLayer = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.model.layerWidth bits)) init 0)
  val slotLastGroup = Vec.fill(cfg.slotCount)(Reg(Bool()) init False)
  val slotInputStart = Vec.fill(cfg.slotCount)(Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.inputFeatureWidth bits)) init 0))
  val slotInputFeatures = Vec.fill(cfg.slotCount)(Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.inputFeatureWidth bits)) init 0))
  val slotTotalInputFeatures = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.inputFeatureWidth bits)) init 0)

  val slotGroupOrdinal = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Reg(UInt(base.outputIndexWidth bits)) init 0))
  val slotOutputBase = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Reg(UInt(base.outputIndexWidth bits)) init 0))
  val slotBeatOrdinalBase = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Reg(UInt(matrix.beatCountWidth bits)) init 0))
  val slotLocalBase = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Vec.fill(cfg.bankCount)(
      Reg(UInt(cfg.crossbar.localAddressWidth bits)) init 0)))
  val slotFault = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Reg(Bool()) init False))
  val slotFaultCode = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Reg(Bits(8 bits)) init 0))
  val slotBankReadBeats = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Vec.fill(cfg.bankCount)(Reg(UInt(cfg.stripeBeatWidth bits)) init 0)))
  val slotEmittedRawGroups = Vec.fill(cfg.poolCount)(Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.rawGroupCountWidth bits)) init 0))

  for (pool <- 0 until cfg.poolCount) {
    val occupied = Bits(cfg.slotCount bits)
    for (slot <- 0 until cfg.slotCount) {
      occupied(slot) := poolState(pool)(slot) =/=
        BitNetStripedWeightSlotState.Empty
    }
    io.slotOccupied(pool) := occupied
  }

  val sequenceOwned = RegInit(False)
  val activeSequence = Reg(UInt(matrix.sequenceIdWidth bits)) init 0
  val activeEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val activeMode = Reg(BitNetMode()) init BitNetMode.Prefill
  val activeTarget = Reg(BitNetAddressTarget()) init
    BitNetAddressTarget.QWeight
  val activeLayer = Reg(UInt(cfg.model.layerWidth bits)) init 0
  val reservedGroupTail = Vec.fill(cfg.poolCount)(
    Reg(UInt(base.outputIndexWidth bits)) init 0)
  val reservedOutputTail = Vec.fill(cfg.poolCount)(
    Reg(UInt(base.outputIndexWidth bits)) init 0)
  val reservedBeatTail = Vec.fill(cfg.poolCount)(
    Reg(UInt(matrix.beatCountWidth bits)) init 0)
  val nextReplayBeatOrdinal = Vec.fill(cfg.poolCount)(
    Reg(UInt(matrix.beatCountWidth bits)) init 0)
  val lastGroupReserved = RegInit(False)
  val cacheLeaseHeld = RegInit(False)
  val cacheAcquireIssued = RegInit(False)
  val cacheReleasePending = RegInit(False)
  val cacheLeaseSequence = Reg(UInt(matrix.sequenceIdWidth bits)) init 0
  val cacheLeaseEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val cacheLeaseGeneration = Reg(UInt(tileCacheCfg.generationWidth bits)) init 0
  val cacheGrantQuarantined = RegInit(False)
  val cacheInvalidGrantReleasePending = RegInit(False)
  val cacheInvalidGrantGeneration = Reg(UInt(
    tileCacheCfg.generationWidth bits)) init 0
  io.cacheLeaseQuarantined := cacheGrantQuarantined
  if (externalWeightTileCache) {
    io.sequenceActive := sequenceOwned || cacheAcquireIssued ||
      cacheLeaseHeld || cacheReleasePending ||
      cacheInvalidGrantReleasePending || cacheGrantQuarantined
  } else {
    io.sequenceActive := sequenceOwned
    cacheGrantQuarantined := False
    cacheInvalidGrantReleasePending := False
    cacheInvalidGrantGeneration := 0
  }

  val stickyFault = RegInit(False)
  val acceptedCommandCount = Reg(UInt(32 bits)) init 0
  val issuedReadRequestCount = Reg(UInt(32 bits)) init 0
  val acceptedReadBeatCount = Reg(UInt(32 bits)) init 0
  val emittedRawGroupCount = Reg(UInt(32 bits)) init 0
  val completedGroupCount = Reg(UInt(32 bits)) init 0
  io.protocolFaultSticky := stickyFault
  io.acceptedCommands := acceptedCommandCount
  io.issuedReadRequests := issuedReadRequestCount
  io.acceptedReadBeats := acceptedReadBeatCount
  io.emittedRawGroups := emittedRawGroupCount
  io.completedGroups := completedGroupCount

  val rejectValid = RegInit(False)
  val rejectPayload = Reg(BitNetFiveBankStripedWeightCompletion(cfg)) init
    BitNetFiveBankStripedWeightCompletion(cfg).getZero

  val selectedCommandSlot = io.command.payload.slot
  val commandRoundEmpty = roundState(selectedCommandSlot) ===
    BitNetStripedWeightRoundState.Empty
  val allRoundsEmpty = !roundState.map(_ =/=
    BitNetStripedWeightRoundState.Empty).asBits.orR
  val commandReadyBase = commandRoundEmpty && !rejectValid &&
    (sequenceOwned || allRoundsEmpty)
  val targetKnown = targets.map(io.command.payload.target === _).reduce(_ || _)
  val layerLegal = io.command.payload.layer < cfg.model.layerCount
  val expectedFfnStripe = io.command.payload.target ===
    BitNetAddressTarget.DownWeight
  val stripeLegal = Vec(Bool(), cfg.bankCount)
  val hiddenStarts = Vector(0, 512, 1024, 1536, 2048)
  val hiddenFeatures = Vector.fill(5)(512)
  val ffnStarts = Vector(0, 1408, 2816, 4224, 5632)
  val ffnFeatures = Vector(1408, 1408, 1408, 1408, 1280)
  for (bank <- 0 until cfg.bankCount) {
    val expectedStart = UInt(cfg.inputFeatureWidth bits)
    val expectedFeatures = UInt(cfg.inputFeatureWidth bits)
    expectedStart := U(hiddenStarts(bank), cfg.inputFeatureWidth bits)
    expectedFeatures := U(hiddenFeatures(bank), cfg.inputFeatureWidth bits)
    when(expectedFfnStripe) {
      expectedStart := U(ffnStarts(bank), cfg.inputFeatureWidth bits)
      expectedFeatures := U(ffnFeatures(bank), cfg.inputFeatureWidth bits)
    }
    stripeLegal(bank) :=
      io.command.payload.inputStart(bank) === expectedStart &&
      io.command.payload.inputFeatures(bank) === expectedFeatures
  }
  val allStripesLegal = stripeLegal.asBits.andR

  val selectedOutputGroups = UInt((base.outputIndexWidth + 1) bits)
  selectedOutputGroups := 0
  for (target <- targets) {
    when(io.command.payload.target === target) {
      selectedOutputGroups := U(
        BitNetFiveBankStripedLayout.outputFeatures(cfg.model, target) / 4,
        selectedOutputGroups.getWidth bits)
    }
  }
  val engineDescriptorLegal = Vec(Bool(), cfg.poolCount)
  for (pool <- 0 until cfg.poolCount) {
    val addressesAligned = Vec(Bool(), cfg.bankCount)
    for (bank <- 0 until cfg.bankCount) {
      addressesAligned(bank) :=
        io.command.payload.engine(pool).localBase(bank)(6 downto 0) === 0
    }
    engineDescriptorLegal(pool) := addressesAligned.asBits.andR &&
      io.command.payload.engine(pool).groupOrdinal.resize(
        selectedOutputGroups.getWidth) < selectedOutputGroups &&
      io.command.payload.engine(pool).outputBase ===
        (io.command.payload.engine(pool).groupOrdinal << 2).resized
  }
  val beatBasesJoinable = if (cfg.poolCount == 1) True else
    io.command.payload.engine(0).beatOrdinalBase ===
      io.command.payload.engine(1).beatOrdinalBase
  val prefillDescriptorsMatch = if (cfg.poolCount == 1) True else
    io.command.payload.engine(0).groupOrdinal ===
        io.command.payload.engine(1).groupOrdinal &&
      io.command.payload.engine(0).outputBase ===
        io.command.payload.engine(1).outputBase &&
      (0 until cfg.bankCount).map { bank =>
        io.command.payload.engine(0).localBase(bank) ===
          io.command.payload.engine(1).localBase(bank)
      }.reduce(_ && _)
  val modeContractLegal = beatBasesJoinable &&
    (io.command.payload.mode === BitNetMode.Decode ||
      prefillDescriptorsMatch)

  val identityLegal = !sequenceOwned ||
    (io.command.payload.sequenceId === activeSequence &&
      io.command.payload.epoch === activeEpoch &&
      io.command.payload.mode === activeMode &&
      io.command.payload.target === activeTarget &&
      io.command.payload.layer === activeLayer)
  val firstReservationLegal = sequenceOwned ||
    (0 until cfg.poolCount).map { pool =>
      io.command.payload.engine(pool).beatOrdinalBase === 0
    }.reduce(_ && _)
  val reservationLegal = !sequenceOwned ||
    (!lastGroupReserved && (0 until cfg.poolCount).map { pool =>
      io.command.payload.engine(pool).groupOrdinal ===
        reservedGroupTail(pool) &&
      io.command.payload.engine(pool).outputBase ===
        reservedOutputTail(pool) &&
      io.command.payload.engine(pool).beatOrdinalBase ===
        reservedBeatTail(pool)
    }.reduce(_ && _))
  val commandLegal = targetKnown && layerLegal && allStripesLegal &&
    engineDescriptorLegal.asBits.andR && modeContractLegal && identityLegal &&
    firstReservationLegal && reservationLegal

  if (!externalWeightTileCache) {
    // An invalid command inside an owned sequence is rejected only at a clean
    // round boundary.  That guarantees the rejection can retire the sequence
    // without abandoning an in-flight fill or replay.
    io.command.ready := commandReadyBase &&
      (!sequenceOwned || commandLegal || allRoundsEmpty)
  }

  if (externalWeightTileCache) {
    val firstLegalCommand = io.command.valid && !sequenceOwned &&
      commandReadyBase && commandLegal
    io.cacheAcquire.valid := firstLegalCommand && !cacheLeaseHeld &&
      !cacheAcquireIssued && !cacheReleasePending &&
      !cacheInvalidGrantReleasePending && !cacheGrantQuarantined
    io.cacheAcquire.sequenceId := io.command.sequenceId
    io.cacheAcquire.sourceEpoch := io.command.epoch
    when(io.cacheAcquire.fire) {
      cacheAcquireIssued := True
      cacheLeaseSequence := io.command.sequenceId
      cacheLeaseEpoch := io.command.epoch
    }

    io.cacheGrant.ready := cacheAcquireIssued && !cacheLeaseHeld &&
      !cacheReleasePending && !cacheInvalidGrantReleasePending &&
      !cacheGrantQuarantined
    val cacheGrantLegal =
      io.cacheGrant.sequenceId === cacheLeaseSequence &&
        io.cacheGrant.sourceEpoch === cacheLeaseEpoch &&
        io.cacheGrant.generation =/= 0 &&
        io.cacheGrant.generation === io.cacheActiveGeneration
    when(io.cacheGrant.fire) {
      cacheAcquireIssued := False
      when(cacheGrantLegal) {
        cacheLeaseHeld := True
        cacheLeaseGeneration := io.cacheGrant.generation
      } otherwise {
        // Never authenticate a bad grant.  A non-zero returned generation is
        // retained only for one best-effort release; it is never copied into
        // cacheLeaseGeneration and therefore cannot authorize a read/write.
        cacheLeaseHeld := False
        cacheLeaseGeneration := 0
        cacheGrantQuarantined := True
        stickyFault := True
        when(io.cacheGrant.generation =/= 0) {
          cacheInvalidGrantReleasePending := True
          cacheInvalidGrantGeneration := io.cacheGrant.generation
        }
      }
    }

    val commandMatchesLease =
      io.command.sequenceId === cacheLeaseSequence &&
        io.command.epoch === cacheLeaseEpoch
    io.command.ready := commandReadyBase && !cacheReleasePending &&
      !cacheInvalidGrantReleasePending && !cacheGrantQuarantined &&
      ((sequenceOwned && cacheLeaseHeld &&
          (commandLegal || allRoundsEmpty)) ||
        (!sequenceOwned && !commandLegal) ||
        (!sequenceOwned && commandLegal && cacheLeaseHeld &&
          !cacheReleasePending && commandMatchesLease))

    val normalCacheRelease = cacheReleasePending && cacheLeaseHeld &&
      !sequenceOwned && allRoundsEmpty && !cacheGrantQuarantined
    io.cacheRelease.valid := cacheInvalidGrantReleasePending ||
      normalCacheRelease
    io.cacheRelease.sequenceId := cacheLeaseSequence
    io.cacheRelease.sourceEpoch := cacheLeaseEpoch
    io.cacheRelease.generation := Mux(cacheInvalidGrantReleasePending,
      cacheInvalidGrantGeneration, cacheLeaseGeneration)
    when(io.cacheRelease.fire) {
      when(cacheInvalidGrantReleasePending) {
        cacheInvalidGrantReleasePending := False
        cacheInvalidGrantGeneration := 0
      } otherwise {
        cacheReleasePending := False
        cacheLeaseHeld := False
        cacheLeaseGeneration := 0
      }
    }
  }

  when(io.command.fire) {
    when(commandLegal) {
      val slot = selectedCommandSlot
      roundState(slot) := BitNetStripedWeightRoundState.Active
      slotSequence(slot) := io.command.payload.sequenceId
      slotEpoch(slot) := io.command.payload.epoch
      slotMode(slot) := io.command.payload.mode
      slotTarget(slot) := io.command.payload.target
      slotLayer(slot) := io.command.payload.layer
      slotLastGroup(slot) := io.command.payload.lastGroup
      slotTotalInputFeatures(slot) := Mux(expectedFfnStripe,
        U(cfg.model.ffnSize, cfg.inputFeatureWidth bits),
        U(cfg.model.hiddenSize, cfg.inputFeatureWidth bits))
      for (bank <- 0 until cfg.bankCount) {
        slotInputStart(slot)(bank) := io.command.payload.inputStart(bank)
        slotInputFeatures(slot)(bank) := io.command.payload.inputFeatures(bank)
      }
      for (pool <- 0 until cfg.poolCount) {
        slotGroupOrdinal(pool)(slot) :=
          io.command.payload.engine(pool).groupOrdinal
        slotOutputBase(pool)(slot) :=
          io.command.payload.engine(pool).outputBase
        slotBeatOrdinalBase(pool)(slot) :=
          io.command.payload.engine(pool).beatOrdinalBase
        for (bank <- 0 until cfg.bankCount) {
          slotLocalBase(pool)(slot)(bank) :=
            io.command.payload.engine(pool).localBase(bank)
          slotBankReadBeats(pool)(slot)(bank) :=
            U(0, cfg.stripeBeatWidth bits)
        }
        slotFault(pool)(slot) := False
        slotFaultCode(pool)(slot) := B(0, 8 bits)
        slotEmittedRawGroups(pool)(slot) :=
          U(0, cfg.rawGroupCountWidth bits)
        poolState(pool)(slot) := BitNetStripedWeightSlotState.AwaitFill
        if (pool == 1) {
          when(io.command.payload.mode === BitNetMode.Prefill) {
            poolState(pool)(slot) := BitNetStripedWeightSlotState.Empty
          }
        }

        val kTiles = Mux(expectedFfnStripe,
          U(cfg.model.ffnSize / 128, matrix.beatCountWidth bits),
          U(cfg.model.hiddenSize / 128, matrix.beatCountWidth bits))
        reservedGroupTail(pool) :=
          io.command.payload.engine(pool).groupOrdinal + 1
        reservedOutputTail(pool) :=
          io.command.payload.engine(pool).outputBase + 4
        reservedBeatTail(pool) := (
          io.command.payload.engine(pool).beatOrdinalBase + (kTiles << 1)
        ).resized
      }
      lastGroupReserved := io.command.payload.lastGroup
      acceptedCommandCount := acceptedCommandCount + 1
      when(!sequenceOwned) {
        sequenceOwned := True
        activeSequence := io.command.payload.sequenceId
        activeEpoch := io.command.payload.epoch
        activeMode := io.command.payload.mode
        activeTarget := io.command.payload.target
        activeLayer := io.command.payload.layer
        for (pool <- 0 until cfg.poolCount) {
          nextReplayBeatOrdinal(pool) := 0
        }
      }
    } otherwise {
      rejectValid := True
      rejectPayload.sequenceId := io.command.payload.sequenceId
      rejectPayload.epoch := io.command.payload.epoch
      rejectPayload.mode := io.command.payload.mode
      rejectPayload.target := io.command.payload.target
      rejectPayload.layer := io.command.payload.layer
      rejectPayload.slot := io.command.payload.slot
      rejectPayload.activePoolMask := B(0, cfg.poolCount bits)
      for (pool <- 0 until cfg.poolCount) {
        rejectPayload.groupOrdinal(pool) :=
          io.command.payload.engine(pool).groupOrdinal
        rejectPayload.outputBase(pool) :=
          io.command.payload.engine(pool).outputBase
        rejectPayload.beatOrdinalBase(pool) :=
          io.command.payload.engine(pool).beatOrdinalBase
        rejectPayload.emittedRawGroups(pool) := 0
        for (bank <- 0 until cfg.bankCount) {
          rejectPayload.bankReadBeats(pool)(bank) := 0
        }
      }
      rejectPayload.success := False
      rejectPayload.faultCode := B(
        BitNetFiveBankStripedWeightFault.InvalidCommand, 8 bits)
      rejectPayload.lastGroup := io.command.payload.lastGroup
      stickyFault := True
      // `io.command.ready` permits an invalid command in an existing sequence
      // only after every accepted round has drained.  Retire ownership here
      // so a fail-closed rejection cannot leak the shared cache lease and
      // permanently block LM head progress.
      when(sequenceOwned && allRoundsEmpty) {
        sequenceOwned := False
        lastGroupReserved := False
        if (externalWeightTileCache) {
          cacheReleasePending := True
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Two replay pools, eight rotated segment memories per pool.
  // -----------------------------------------------------------------------
  val weightMemories = if (externalWeightTileCache) None else Some(
    Array.tabulate(cfg.poolCount, cfg.segmentsPerTile) {
      (pool, memory) =>
        val mem = Mem(Bits(cfg.crossbar.dataWidth bits), cfg.memoryDepth)
        mem.setName(s"weightPool${pool}Memory${memory}")
        mem.addAttribute("ram_style", "block")
        mem
    })

  val poolFilling = Vec.fill(cfg.poolCount)(Reg(Bool()) init False)
  val poolFillSlot = Vec.fill(cfg.poolCount)(
    Reg(UInt(cfg.slotWidth bits)) init 0)
  val fillRequestIssued = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(
    Reg(Bool()) init False))
  val fillBankDone = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(
    Reg(Bool()) init False))
  val fillResponseBeat = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.crossbar.beatIndexWidth bits)) init 0))
  val requestPreferPool = Vec.fill(cfg.bankCount)(Reg(Bool()) init False)

  for (pool <- 0 until cfg.poolCount) {
    val awaitingExists = Bool()
    val awaitingSlot = UInt(cfg.slotWidth bits)
    awaitingExists := False
    awaitingSlot := 0
    for (slot <- (0 until cfg.slotCount).reverse) {
      when(poolState(pool)(slot) ===
        BitNetStripedWeightSlotState.AwaitFill) {
        awaitingExists := True
        awaitingSlot := slot
      }
    }
    when(!poolFilling(pool) && awaitingExists) {
      poolFilling(pool) := True
      poolFillSlot(pool) := awaitingSlot
      poolState(pool)(awaitingSlot) :=
        BitNetStripedWeightSlotState.Filling
      for (bank <- 0 until cfg.bankCount) {
        fillRequestIssued(pool)(bank) := False
        fillBankDone(pool)(bank) := False
        fillResponseBeat(pool)(bank) := 0
      }
    }
    io.fillActive(pool) := poolFilling(pool)
  }

  def expectedTransaction(pool: Int, bank: Int): UInt = {
    val slot = poolFillSlot(pool)
    val hashWidth = cfg.crossbar.transactionIdWidth - 1
    val hash = (
      slotEpoch(slot).resize(hashWidth) ^
      slotGroupOrdinal(pool)(slot).resize(hashWidth) ^
      U(bank * 13 + 3, hashWidth bits)
    ).resize(hashWidth)
    (hash.asBits ## B(pool, 1 bits)).asUInt
  }

  def expectedTag(pool: Int, bank: Int): UInt = {
    val slot = poolFillSlot(pool)
    (
      slotSequence(slot).resize(cfg.crossbar.tagWidth) ^
      (slotEpoch(slot).resize(cfg.crossbar.tagWidth) << 12).resize(
        cfg.crossbar.tagWidth) ^
      (slotGroupOrdinal(pool)(slot).resize(cfg.crossbar.tagWidth) << 2)
        .resize(cfg.crossbar.tagWidth) ^
      (U(pool, cfg.crossbar.tagWidth bits) << 1).resize(
        cfg.crossbar.tagWidth) ^
      U(bank, cfg.crossbar.tagWidth bits)
    ).resize(cfg.crossbar.tagWidth)
  }

  val requestChosenPool = Vec(UInt(cfg.poolWidth bits), cfg.bankCount)
  val requestFireBits = Bits(cfg.bankCount bits)
  val requestCandidate = Array.tabulate(cfg.poolCount, cfg.bankCount) {
    (pool, bank) => poolFilling(pool) && !fillRequestIssued(pool)(bank)
  }
  for (bank <- 0 until cfg.bankCount) {
    val choosePool1 = if (cfg.poolCount == 1) False else
      requestCandidate(1)(bank) &&
        (!requestCandidate(0)(bank) || requestPreferPool(bank))
    requestChosenPool(bank) := Mux(choosePool1, U(1), U(0))
    val chosen = requestChosenPool(bank)
    val chosenSlot = UInt(cfg.slotWidth bits)
    chosenSlot := poolFillSlot(0)
    val chosenLocalBase = UInt(cfg.crossbar.localAddressWidth bits)
    chosenLocalBase := slotLocalBase(0)(poolFillSlot(0))(bank)
    if (cfg.poolCount > 1) {
      when(chosen === 1) {
        chosenSlot := poolFillSlot(1)
        chosenLocalBase := slotLocalBase(1)(poolFillSlot(1))(bank)
      }
    }
    io.readRequest(bank).valid := (0 until cfg.poolCount)
      .map(pool => requestCandidate(pool)(bank)).reduce(_ || _)
    io.readRequest(bank).bank := U(bank, cfg.crossbar.bankWidth bits)
    io.readRequest(bank).transactionId := expectedTransaction(0, bank)
    io.readRequest(bank).tag := expectedTag(0, bank)
    if (cfg.poolCount > 1) {
      when(chosen === 1) {
        io.readRequest(bank).transactionId := expectedTransaction(1, bank)
        io.readRequest(bank).tag := expectedTag(1, bank)
      }
    }
    io.readRequest(bank).localAddress := chosenLocalBase
    io.readRequest(bank).beats :=
      (slotInputFeatures(chosenSlot)(bank) >> 4).resize(
        cfg.crossbar.beatCountWidth)
    requestFireBits(bank) := io.readRequest(bank).fire

    when(io.readRequest(bank).fire) {
      requestPreferPool(bank) := !requestChosenPool(bank).asBool
      when(requestChosenPool(bank) === 0) {
        fillRequestIssued(0)(bank) := True
      } otherwise {
        if (cfg.poolCount > 1) fillRequestIssued(1)(bank) := True
      }
    }
  }
  when(requestFireBits.orR) {
    issuedReadRequestCount := issuedReadRequestCount +
      CountOne(requestFireBits).resize(32)
  }

  // Route each bank response by the pool bit in the transaction id. A stale
  // pool bit falls back to the only outstanding pool; the full transaction,
  // tag and ordinal are still checked below and poison the complete round.
  val responseRoutePool = Vec(UInt(cfg.poolWidth bits), cfg.bankCount)
  val responseRouteValid = Vec(Bool(), cfg.bankCount)
  val responseCandidate = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(
    Bool()))
  val responseDesiredMemory = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(
    UInt(log2Up(cfg.segmentsPerTile) bits)))
  val responseGlobalTile = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(
    UInt(cfg.tileIndexWidth bits)))
  val responseLegal = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(Bool()))
  val responseMetadataLegal = Vec.fill(cfg.poolCount)(
    Vec.fill(cfg.bankCount)(Bool()))
  val responseAxiLegal = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(
    Bool()))
  val responseExpectedLast = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(
    Bool()))

  for (bank <- 0 until cfg.bankCount) {
    val outstanding0 = poolFilling(0) && fillRequestIssued(0)(bank) &&
      !fillBankDone(0)(bank)
    val outstanding1 = if (cfg.poolCount == 1) False else
      poolFilling(1) && fillRequestIssued(1)(bank) &&
        !fillBankDone(1)(bank)
    val encodedPool = io.readResponse(bank).transactionId(0).asUInt
    val encodedOutstanding = if (cfg.poolCount == 1) outstanding0 else
      Mux(encodedPool === 0, outstanding0, outstanding1)
    responseRoutePool(bank) := (if (cfg.poolCount == 1) U(0) else encodedPool)
    when(!encodedOutstanding) {
      responseRoutePool(bank) :=
        (if (cfg.poolCount == 1) U(0) else Mux(outstanding0, U(0), U(1)))
    }
    responseRouteValid(bank) := outstanding0 || outstanding1

    for (pool <- 0 until cfg.poolCount) {
      val slot = poolFillSlot(pool)
      responseCandidate(pool)(bank) := io.readResponse(bank).valid &&
        responseRouteValid(bank) && responseRoutePool(bank) === pool
      // `fillResponseBeat` is bank-local (at most 88 beats), while the stripe
      // base is global (up to 4224 / 16).  Keeping the sum at the local-counter
      // width aliases bank 4 of hidden K, and several FFN banks, back onto low
      // tile addresses.  Widen before adding so every K stripe remains unique.
      val globalBeat = (
        (slotInputStart(slot)(bank) >> 4).resize(cfg.totalReadBeatWidth) +
        fillResponseBeat(pool)(bank).resize(cfg.totalReadBeatWidth)
      ).resize(cfg.totalReadBeatWidth)
      val naturalSegment = globalBeat(segmentIndexWidth - 1 downto 0)
      responseDesiredMemory(pool)(bank) :=
        (naturalSegment.resize(segmentIndexWidth + 1) +
          U(bank, segmentIndexWidth + 1 bits))(
            segmentIndexWidth - 1 downto 0)
      responseGlobalTile(pool)(bank) :=
        (globalBeat >> segmentIndexWidth).resize(cfg.tileIndexWidth)
      val totalBeats = (slotInputFeatures(slot)(bank) >> 4).resize(
        cfg.crossbar.beatCountWidth)
      responseExpectedLast(pool)(bank) :=
        fillResponseBeat(pool)(bank) ===
          (totalBeats - 1).resize(cfg.crossbar.beatIndexWidth)
      responseMetadataLegal(pool)(bank) :=
        io.readResponse(bank).bank === bank &&
        io.readResponse(bank).transactionId ===
          expectedTransaction(pool, bank) &&
        io.readResponse(bank).tag === expectedTag(pool, bank) &&
        io.readResponse(bank).beatIndex === fillResponseBeat(pool)(bank) &&
        io.readResponse(bank).last === responseExpectedLast(pool)(bank)
      responseAxiLegal(pool)(bank) := !io.readResponse(bank).fault &&
        io.readResponse(bank).rresp === B"2'b00"
      responseLegal(pool)(bank) := responseMetadataLegal(pool)(bank) &&
        responseAxiLegal(pool)(bank)
    }
  }

  val responseWinnerValid = Vec.fill(cfg.poolCount)(
    Vec.fill(cfg.segmentsPerTile)(Bool()))
  val responseWinnerBank = Vec.fill(cfg.poolCount)(
    Vec.fill(cfg.segmentsPerTile)(UInt(cfg.crossbar.bankWidth bits)))
  for (pool <- 0 until cfg.poolCount; memory <- 0 until cfg.segmentsPerTile) {
    responseWinnerValid(pool)(memory) := False
    responseWinnerBank(pool)(memory) := 0
    for (bank <- (0 until cfg.bankCount).reverse) {
      when(responseCandidate(pool)(bank) &&
        responseDesiredMemory(pool)(bank) === memory) {
        responseWinnerValid(pool)(memory) := True
        responseWinnerBank(pool)(memory) := bank
      }
    }
  }

  for (bank <- 0 until cfg.bankCount) {
    io.readResponse(bank).ready := False
    for (pool <- 0 until cfg.poolCount; memory <- 0 until cfg.segmentsPerTile) {
      val slot = poolFillSlot(pool)
      val cacheWriteRequired = responseLegal(pool)(bank) &&
        !slotFault(pool)(slot)
      val cacheWriteReady = if (externalWeightTileCache)
        io.cacheWrite(pool)(memory).ready else True
      when(responseRoutePool(bank) === pool &&
        responseDesiredMemory(pool)(bank) === memory &&
        responseWinnerValid(pool)(memory) &&
        responseWinnerBank(pool)(memory) === bank) {
        io.readResponse(bank).ready := responseRouteValid(bank) &&
          (!cacheWriteRequired || cacheWriteReady)
      }
    }
  }

  // Exactly one write port per physical memory. Five equal-latency banks hit
  // five different rotated memories; skewed collisions are backpressured.
  for (pool <- 0 until cfg.poolCount; memory <- 0 until cfg.segmentsPerTile) {
    val winner = responseWinnerBank(pool)(memory)
    val winnerFire = responseWinnerValid(pool)(memory) &&
      io.readResponse(winner).fire && responseRoutePool(winner) === pool
    val slot = poolFillSlot(pool)
    val writeAddress = UInt(cfg.memoryAddressWidth bits)
    writeAddress := responseGlobalTile(pool)(winner).resized
    when(slot === 1) {
      writeAddress := (U(cfg.maxTiles, cfg.memoryAddressWidth bits) +
        responseGlobalTile(pool)(winner).resize(cfg.memoryAddressWidth)).resized
    }
    val writeEnable = winnerFire && responseLegal(pool)(winner) &&
      !slotFault(pool)(slot)
    if (externalWeightTileCache) {
      io.cacheWrite(pool)(memory).valid :=
        cacheLeaseHeld && !cacheGrantQuarantined &&
          responseWinnerValid(pool)(memory) &&
          responseRoutePool(winner) === pool &&
          responseLegal(pool)(winner) && !slotFault(pool)(slot)
      io.cacheWrite(pool)(memory).generation := cacheLeaseGeneration
      io.cacheWrite(pool)(memory).address := writeAddress
      io.cacheWrite(pool)(memory).data := io.readResponse(winner).data
    } else {
      weightMemories.get(pool)(memory).write(
        address = writeAddress,
        data = io.readResponse(winner).data,
        enable = writeEnable)
    }
  }

  val bankFinishing = Vec.fill(cfg.poolCount)(Vec.fill(cfg.bankCount)(Bool()))
  val illegalResponseFire = Vec.fill(cfg.poolCount)(
    Vec.fill(cfg.bankCount)(Bool()))
  val acceptedResponseFireBits = Bits(cfg.bankCount bits)
  for (bank <- 0 until cfg.bankCount) {
    acceptedResponseFireBits(bank) := io.readResponse(bank).fire
  }
  when(acceptedResponseFireBits.orR) {
    acceptedReadBeatCount := acceptedReadBeatCount +
      CountOne(acceptedResponseFireBits).resize(32)
  }
  for (pool <- 0 until cfg.poolCount; bank <- 0 until cfg.bankCount) {
    val routedFire = io.readResponse(bank).fire &&
      responseRoutePool(bank) === pool
    bankFinishing(pool)(bank) := routedFire &&
      (responseExpectedLast(pool)(bank) || io.readResponse(bank).last)
    illegalResponseFire(pool)(bank) := routedFire &&
      !responseLegal(pool)(bank)
    when(routedFire) {
      val slot = poolFillSlot(pool)
      slotBankReadBeats(pool)(slot)(bank) :=
        slotBankReadBeats(pool)(slot)(bank) + 1
      when(!responseLegal(pool)(bank)) {
        stickyFault := True
        when(!slotFault(pool)(slot)) {
          slotFault(pool)(slot) := True
          slotFaultCode(pool)(slot) := B(
            BitNetFiveBankStripedWeightFault.ResponseMetadata, 8 bits)
          when(responseMetadataLegal(pool)(bank) &&
            !responseAxiLegal(pool)(bank)) {
            slotFaultCode(pool)(slot) := B(
              BitNetFiveBankStripedWeightFault.AxiReadResponse, 8 bits)
          }
        }
      }
      when(bankFinishing(pool)(bank)) {
        fillBankDone(pool)(bank) := True
      } otherwise {
        fillResponseBeat(pool)(bank) := fillResponseBeat(pool)(bank) + 1
      }
    }
  }

  for (pool <- 0 until cfg.poolCount) {
    val doneAfter = Bits(cfg.bankCount bits)
    val anyIllegal = Bool()
    anyIllegal := False
    for (bank <- 0 until cfg.bankCount) {
      doneAfter(bank) := fillBankDone(pool)(bank) || bankFinishing(pool)(bank)
      when(illegalResponseFire(pool)(bank)) { anyIllegal := True }
    }
    when(poolFilling(pool) && doneAfter.andR) {
      val slot = poolFillSlot(pool)
      poolFilling(pool) := False
      when(slotFault(pool)(slot) || anyIllegal) {
        poolState(pool)(slot) := BitNetStripedWeightSlotState.Complete
      } otherwise {
        poolState(pool)(slot) := BitNetStripedWeightSlotState.Ready
      }
    }
  }

  // A failed prefill pool, or either failed decode pool, retires only after all
  // active banks have drained. No tile from that round is ever exposed.
  for (slot <- 0 until cfg.slotCount) {
    val prefillFailed = slotMode(slot) === BitNetMode.Prefill &&
      poolState(0)(slot) === BitNetStripedWeightSlotState.Complete &&
      slotFault(0)(slot)
    val decodeTerminal0 = poolState(0)(slot) ===
      BitNetStripedWeightSlotState.Ready ||
      poolState(0)(slot) === BitNetStripedWeightSlotState.Complete
    val decodeTerminal1 = if (cfg.poolCount == 1) True else
      poolState(1)(slot) === BitNetStripedWeightSlotState.Ready ||
        poolState(1)(slot) === BitNetStripedWeightSlotState.Complete
    val decodeFailed = slotMode(slot) === BitNetMode.Decode &&
      decodeTerminal0 && decodeTerminal1 &&
      (if (cfg.poolCount == 1) slotFault(0)(slot)
       else slotFault(0)(slot) || slotFault(1)(slot))
    when(roundState(slot) === BitNetStripedWeightRoundState.Active &&
      (prefillFailed || decodeFailed)) {
      roundState(slot) := BitNetStripedWeightRoundState.Complete
      for (pool <- 0 until cfg.poolCount) {
        val poolParticipates = if (pool == 0) True
          else slotMode(slot) === BitNetMode.Decode
        when(poolParticipates) {
          poolState(pool)(slot) := BitNetStripedWeightSlotState.Complete
          when(!slotFault(pool)(slot)) {
            slotFault(pool)(slot) := True
            slotFaultCode(pool)(slot) := B(
              BitNetFiveBankStripedWeightFault.PartnerPoolFault, 8 bits)
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Joined replay: one raw group-of-four K sweep.  Lower/upper row pairs are
  // selected downstream from the same held raw group, so BRAM read/unrotate
  // work is never repeated.
  // -----------------------------------------------------------------------
  val replaying = RegInit(False)
  val replaySlot = Reg(UInt(cfg.slotWidth bits)) init 0
  val replayTileIndex = Reg(UInt(cfg.tileIndexWidth bits)) init 0
  val allReplayReadsIssued = RegInit(False)
  io.replayActive := replaying

  val replayCandidateExists = Bool()
  val replayCandidateSlot = UInt(cfg.slotWidth bits)
  replayCandidateExists := False
  replayCandidateSlot := 0
  for (slot <- (0 until cfg.slotCount).reverse) {
    val prefillReady = slotMode(slot) === BitNetMode.Prefill &&
      poolState(0)(slot) === BitNetStripedWeightSlotState.Ready
    val decodeReady = slotMode(slot) === BitNetMode.Decode &&
      poolState(0)(slot) === BitNetStripedWeightSlotState.Ready &&
      (if (cfg.poolCount == 1) True else
        poolState(1)(slot) === BitNetStripedWeightSlotState.Ready)
    val ordinalReady = (0 until cfg.poolCount).map { pool =>
      slotBeatOrdinalBase(pool)(slot) === nextReplayBeatOrdinal(pool)
    }.reduce(_ && _)
    when(roundState(slot) === BitNetStripedWeightRoundState.Active &&
      (prefillReady || decodeReady) && ordinalReady) {
      replayCandidateExists := True
      replayCandidateSlot := slot
    }
  }

  when(!replaying && replayCandidateExists) {
    replaying := True
    replaySlot := replayCandidateSlot
    replayTileIndex := 0
    allReplayReadsIssued := False
    poolState(0)(replayCandidateSlot) :=
      BitNetStripedWeightSlotState.Replaying
    if (cfg.poolCount > 1) {
      when(slotMode(replayCandidateSlot) === BitNetMode.Decode) {
        poolState(1)(replayCandidateSlot) :=
          BitNetStripedWeightSlotState.Replaying
      }
    }
  }

  val replayResponseValid = RegInit(False)
  // The synchronous BRAM outputs themselves are the one-entry elastic stage.
  // A stalled tile freezes every read port and its metadata.  If the current
  // tile fires, the next read is issued in that same cycle, preserving II=1
  // without a 2048-bit-wide distributed-RAM FIFO.
  val replayResponseReady = !replayResponseValid || io.tile.ready
  val replayReadCandidate = replaying && !allReplayReadsIssued &&
    replayResponseReady
  val replayReadIssue = Bool()

  val replayKTileCount = (slotTotalInputFeatures(replaySlot) >> 7)
    .resize(cfg.tileIndexWidth + 1)
  val replayBank = UInt(cfg.crossbar.bankWidth bits)
  replayBank := 0
  for (bank <- 0 until cfg.bankCount) {
    val startTile = (slotInputStart(replaySlot)(bank) >> 7)
      .resize(cfg.tileIndexWidth + 1)
    val endTile = ((slotInputStart(replaySlot)(bank) +
      slotInputFeatures(replaySlot)(bank)) >> 7)
      .resize(cfg.tileIndexWidth + 1)
    when(replayTileIndex.resize(cfg.tileIndexWidth + 1) >= startTile &&
      replayTileIndex.resize(cfg.tileIndexWidth + 1) < endTile) {
      replayBank := bank
    }
  }

  val replayMemoryAddress = UInt(cfg.memoryAddressWidth bits)
  replayMemoryAddress := replayTileIndex.resized
  when(replaySlot === 1) {
    replayMemoryAddress := (U(cfg.maxTiles, cfg.memoryAddressWidth bits) +
      replayTileIndex.resize(cfg.memoryAddressWidth)).resized
  }
  if (externalWeightTileCache) {
    io.cacheRead.valid := cacheLeaseHeld && !cacheGrantQuarantined &&
      replayReadCandidate
    io.cacheRead.generation := cacheLeaseGeneration
    io.cacheRead.address := replayMemoryAddress.resized
    io.cacheRead.poolMask := B(1, cfg.poolCount bits)
    when(slotMode(replaySlot) === BitNetMode.Decode) {
      io.cacheRead.poolMask := B((BigInt(1) << cfg.poolCount) - 1,
        cfg.poolCount bits)
    }
    replayReadIssue := io.cacheRead.fire
  } else {
    replayReadIssue := replayReadCandidate
  }
  val replayReadData = Vec.fill(cfg.poolCount)(Vec.fill(cfg.segmentsPerTile)(
    Bits(cfg.crossbar.dataWidth bits)))
  for (pool <- 0 until cfg.poolCount; memory <- 0 until cfg.segmentsPerTile) {
    val poolParticipates = if (pool == 0) True
      else slotMode(replaySlot) === BitNetMode.Decode
    if (externalWeightTileCache) {
      replayReadData(pool)(memory) := io.cacheReadData(pool)(memory)
    } else {
      replayReadData(pool)(memory) :=
        weightMemories.get(pool)(memory).readSync(
          address = replayMemoryAddress,
          enable = replayReadIssue && poolParticipates)
    }
  }
  replayResponseValid :=
    (replayResponseValid && !io.tile.ready) || replayReadIssue
  val replayResponseSlot = RegNextWhen(replaySlot, replayReadIssue) init 0
  val replayResponseMode = RegNextWhen(
    slotMode(replaySlot), replayReadIssue) init BitNetMode.Prefill
  val replayResponseBank = RegNextWhen(replayBank, replayReadIssue) init 0
  val replayResponseTile = RegNextWhen(
    replayTileIndex, replayReadIssue) init 0
  val replayResponseLast = RegNextWhen(
    replayTileIndex.resize(cfg.tileIndexWidth + 1) === replayKTileCount - 1,
    replayReadIssue) init False

  io.tile.valid := replayResponseValid
  io.tile.sequenceId := slotSequence(replayResponseSlot)
  io.tile.scheduleMode := replayResponseMode
  io.tile.kTile := replayResponseTile
  io.tile.kTileCount := (slotTotalInputFeatures(replayResponseSlot) >> 7)
    .resize(cfg.rawGroupCountWidth)
  io.tile.lowerBeatOrdinal := (
    slotBeatOrdinalBase(0)(replayResponseSlot) +
      replayResponseTile.resize(matrix.beatCountWidth)
  ).resize(matrix.beatCountWidth)
  io.tile.upperBeatOrdinal := (
    slotBeatOrdinalBase(0)(replayResponseSlot) +
      (slotTotalInputFeatures(replayResponseSlot) >> 7).resize(
        matrix.beatCountWidth) +
      replayResponseTile.resize(matrix.beatCountWidth)
  ).resize(matrix.beatCountWidth)
  io.tile.firstRawGroup := replayResponseTile === 0
  io.tile.lastRawGroup := replayResponseLast
  io.tile.fault := False

  for (engine <- 0 until cfg.poolCount) {
    io.tile.engineTag(engine).groupOrdinal :=
      slotGroupOrdinal(engine)(replayResponseSlot)
    io.tile.engineTag(engine).outputBase :=
      slotOutputBase(engine)(replayResponseSlot)
  }

  val naturallyOrdered = Vec.fill(cfg.poolCount)(
    Vec.fill(cfg.segmentsPerTile)(Bits(cfg.crossbar.dataWidth bits)))
  for (pool <- 0 until cfg.poolCount; segment <- 0 until cfg.segmentsPerTile) {
    val rotatedMemory = (
      replayResponseBank.resize(segmentIndexWidth + 1) +
        U(segment, segmentIndexWidth + 1 bits))(
          segmentIndexWidth - 1 downto 0)
    val readVector = Vec(Bits(cfg.crossbar.dataWidth bits),
      cfg.segmentsPerTile)
    for (memory <- 0 until cfg.segmentsPerTile) {
      readVector(memory) := replayReadData(pool)(memory)
    }
    naturallyOrdered(pool)(segment) := readVector(rotatedMemory)
  }
  for (engine <- 0 until cfg.poolCount; feature <- 0 until base.siteCount) {
    val pool = engine
    val selectedPool = UInt(cfg.poolWidth bits)
    selectedPool := U(pool)
    when(replayResponseMode === BitNetMode.Prefill) {
      selectedPool := 0
    }
    val segment = feature / cfg.beatBytes
    val byte = feature % cfg.beatBytes
    val poolBytes = Vec(Bits(cfg.crossbar.dataWidth bits), cfg.poolCount)
    for (candidate <- 0 until cfg.poolCount) {
      poolBytes(candidate) := naturallyOrdered(candidate)(segment)
    }
    if (cfg.poolCount == 1) {
      io.tile.engineRawGroup(engine)(feature) :=
        poolBytes(0)(byte * 8 + 7 downto byte * 8)
    } else {
      io.tile.engineRawGroup(engine)(feature) :=
        poolBytes(selectedPool)(byte * 8 + 7 downto byte * 8)
    }
  }

  when(replayReadIssue) {
    when(replayTileIndex.resize(cfg.tileIndexWidth + 1) ===
      replayKTileCount - 1) {
      replayTileIndex := 0
      allReplayReadsIssued := True
    } otherwise {
      replayTileIndex := replayTileIndex + 1
    }
  }

  when(io.tile.fire) {
    emittedRawGroupCount := emittedRawGroupCount + 1
    val slot = replayResponseSlot
    slotEmittedRawGroups(0)(slot) :=
      slotEmittedRawGroups(0)(slot) + 1
    if (cfg.poolCount > 1) {
      when(io.tile.sequenceId === slotSequence(slot) &&
        slotMode(slot) === BitNetMode.Decode) {
        slotEmittedRawGroups(1)(slot) :=
          slotEmittedRawGroups(1)(slot) + 1
      }
    }
    when(replayResponseLast) {
      roundState(slot) := BitNetStripedWeightRoundState.Complete
      poolState(0)(slot) := BitNetStripedWeightSlotState.Complete
      if (cfg.poolCount > 1) {
        when(slotMode(slot) === BitNetMode.Decode) {
          poolState(1)(slot) := BitNetStripedWeightSlotState.Complete
        }
      }
      val completedKTileCount =
        (slotTotalInputFeatures(slot) >> 7).resize(matrix.beatCountWidth)
      for (pool <- 0 until cfg.poolCount) {
        nextReplayBeatOrdinal(pool) := (
          slotBeatOrdinalBase(pool)(slot) + (completedKTileCount << 1)
        ).resized
      }
      replaying := False
      allReplayReadsIssued := False
    }
  }

  // -----------------------------------------------------------------------
  // Ordered, backpressure-safe round completion.
  // -----------------------------------------------------------------------
  val completeExists = Bool()
  val completeSlot = UInt(cfg.slotWidth bits)
  completeExists := False
  completeSlot := 0
  when(roundState(0) === BitNetStripedWeightRoundState.Complete) {
    completeExists := True
    completeSlot := 0
  }
  when(roundState(1) === BitNetStripedWeightRoundState.Complete &&
    (roundState(0) =/= BitNetStripedWeightRoundState.Complete ||
      slotBeatOrdinalBase(0)(1) < slotBeatOrdinalBase(0)(0))) {
    completeExists := True
    completeSlot := 1
  }

  io.completion.valid := rejectValid || completeExists
  io.completion.payload := rejectPayload
  when(!rejectValid) {
    io.completion.sequenceId := slotSequence(completeSlot)
    io.completion.epoch := slotEpoch(completeSlot)
    io.completion.mode := slotMode(completeSlot)
    io.completion.target := slotTarget(completeSlot)
    io.completion.layer := slotLayer(completeSlot)
    io.completion.slot := completeSlot
    io.completion.activePoolMask := B(1, cfg.poolCount bits)
    when(slotMode(completeSlot) === BitNetMode.Decode) {
      io.completion.activePoolMask := B(
        (BigInt(1) << cfg.poolCount) - 1, cfg.poolCount bits)
    }
    for (pool <- 0 until cfg.poolCount) {
      io.completion.groupOrdinal(pool) :=
        slotGroupOrdinal(pool)(completeSlot)
      io.completion.outputBase(pool) := slotOutputBase(pool)(completeSlot)
      io.completion.beatOrdinalBase(pool) :=
        slotBeatOrdinalBase(pool)(completeSlot)
      io.completion.emittedRawGroups(pool) :=
        slotEmittedRawGroups(pool)(completeSlot)
      for (bank <- 0 until cfg.bankCount) {
        io.completion.bankReadBeats(pool)(bank) :=
          slotBankReadBeats(pool)(completeSlot)(bank)
      }
    }
    val pool0Success = !slotFault(0)(completeSlot)
    val pool1Success = if (cfg.poolCount == 1) True else
      slotMode(completeSlot) === BitNetMode.Prefill ||
        !slotFault(1)(completeSlot)
    io.completion.success := pool0Success && pool1Success
    io.completion.faultCode := slotFaultCode(0)(completeSlot)
    if (cfg.poolCount > 1) {
      when(!slotFault(0)(completeSlot) && slotFault(1)(completeSlot)) {
        io.completion.faultCode := slotFaultCode(1)(completeSlot)
      }
    }
    io.completion.lastGroup := slotLastGroup(completeSlot)
  }

  when(io.completion.fire) {
    completedGroupCount := completedGroupCount + 1
    when(rejectValid) {
      rejectValid := False
    } otherwise {
      val slot = completeSlot
      roundState(slot) := BitNetStripedWeightRoundState.Empty
      for (pool <- 0 until cfg.poolCount) {
        poolState(pool)(slot) := BitNetStripedWeightSlotState.Empty
      }
      val anySlotFault = (0 until cfg.poolCount)
        .map(pool => slotFault(pool)(slot)).reduce(_ || _)
      when(slotLastGroup(slot) || anySlotFault) {
        sequenceOwned := False
        lastGroupReserved := False
        if (externalWeightTileCache) {
          cacheReleasePending := True
        }
      }
    }
  }
}

object GenerateBitNetFiveBankStripedWeightPrefetcher extends App {
  SpinalConfig(
    targetDirectory = "build/bitnet-five-bank-striped-weight-prefetcher",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetFiveBankStripedWeightPrefetcher())
}
