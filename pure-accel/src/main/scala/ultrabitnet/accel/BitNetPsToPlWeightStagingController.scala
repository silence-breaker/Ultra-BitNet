package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** One bank-0 K stripe inside a layer-resident PL-DDR destination. */
case class BitNetPlWeightTargetRange(
    target: BitNetAddressTarget.E,
    layerOffset: BigInt,
    bytes: BigInt,
    inputStart: Int,
    inputFeatures: Int,
    outputFeatures: Int) {
  require(layerOffset >= 0 && bytes > 0)
  require(inputStart == 0,
    "bank 0 must own the first contiguous K stripe")
  require(inputFeatures > 0 && outputFeatures > 0)
}

/**
  * The immutable transformer bytes copied from one PS-DDR staging image into
  * their final, persistent PL-DDR addresses.
  *
  * A slot is an ownership/completion slot, not a layer-sized physical alias.
  * Consequently a completed layer remains resident after the slot is reused;
  * a later workload with the same model-image epoch performs no PS-to-PL
  * copy, even though its run/workload epoch must be new.
  */
case class BitNetPlWeightLayerRange(
    layer: Int,
    sourceOffset: BigInt,
    destinationBase: BigInt,
    bytes: BigInt,
    pages: Int,
    targets: Vector[BitNetPlWeightTargetRange]) {
  require(layer >= 0 && sourceOffset >= 0 && destinationBase >= 0)
  require(bytes > 0 && pages > 0)
}

case class BitNetPsToPlWeightStagingLayout(
    model: BitNetConfig,
    resident: BitNetFiveBankStripedLayout,
    pageBytes: Int,
    layers: Vector[BitNetPlWeightLayerRange],
    transformerDestinationBase: BigInt,
    transformerBytes: BigInt) {
  require(model == BitNetConfig.Production)
  require(resident.cfg == model)
  require(pageBytes == 4096,
    "production staging uses one legal 256-beat x 128-bit AXI page")
  require(layers.map(_.layer) == (0 until model.layerCount))
  require(layers.map(_.bytes).distinct.size == 1,
    "fixed 30-layer controller requires an equal bank-0 shard per layer")
  require(layers.forall(layer =>
    layer.bytes == BigInt(layer.pages) * pageBytes &&
      layer.sourceOffset % pageBytes == 0 &&
      layer.destinationBase % pageBytes == 0))
  require(layers.head.sourceOffset == 0)
  require(layers.sliding(2).forall {
    case Seq(left, right) =>
      left.sourceOffset + left.bytes == right.sourceOffset &&
        left.destinationBase + left.bytes == right.destinationBase
    case _ => true
  })
  require(layers.last.sourceOffset + layers.last.bytes == transformerBytes)
  require(layers.head.destinationBase == transformerDestinationBase)
  require(layers.last.destinationBase + layers.last.bytes ==
    transformerDestinationBase + transformerBytes)
  require(transformerDestinationBase + transformerBytes <=
    resident.bankCapacities.head)

  val layerBytes: BigInt = layers.head.bytes
  val layerPages: Int = layers.head.pages
  val targetCount: Int = layers.head.targets.size
}

object BitNetPsToPlWeightStagingLayout {
  val PageBytes: Int = 4096

  def apply(
      model: BitNetConfig = BitNetConfig.Production,
      resident: BitNetFiveBankStripedLayout =
        BitNetFiveBankStripedLayout()): BitNetPsToPlWeightStagingLayout = {
    require(model == BitNetConfig.Production && resident.cfg == model)
    val targets = BitNetFiveBankStripedLayout.WeightGeometry.map(_._1)
    val first = resident.weight(targets.head, 0, 0).localBase
    val ranges = Vector.tabulate(model.layerCount) { layer =>
      val shards = targets.map(target => resident.weight(target, layer, 0))
      require(shards.sliding(2).forall {
        case Seq(left, right) => left.localBase + left.bytes == right.localBase
        case _ => true
      }, s"layer $layer bank-0 target shards are not contiguous")
      val layerBase = shards.head.localBase
      val layerBytes = shards.map(_.bytes).sum
      require(layerBase % PageBytes == 0 && layerBytes % PageBytes == 0,
        s"layer $layer bank-0 shard is not page aligned")
      val targetRanges = shards.map { shard =>
        BitNetPlWeightTargetRange(
          target = shard.target,
          layerOffset = shard.localBase - layerBase,
          bytes = shard.bytes,
          inputStart = shard.inputStart,
          inputFeatures = shard.inputFeatures,
          outputFeatures = shard.outputFeatures)
      }
      BitNetPlWeightLayerRange(
        layer = layer,
        sourceOffset = layerBase - first,
        destinationBase = layerBase,
        bytes = layerBytes,
        pages = (layerBytes / PageBytes).toInt,
        targets = targetRanges)
    }
    val total = ranges.map(_.bytes).sum
    BitNetPsToPlWeightStagingLayout(
      model = model,
      resident = resident,
      pageBytes = PageBytes,
      layers = ranges,
      transformerDestinationBase = first,
      transformerBytes = total)
  }
}

case class BitNetPsToPlWeightStagingConfig(
    model: BitNetConfig = BitNetConfig.Production,
    resident: BitNetFiveBankStripedLayout =
      BitNetFiveBankStripedLayout(),
    pageBytes: Int = BitNetPsToPlWeightStagingLayout.PageBytes,
    maxOutstanding: Int = 8,
    epochWidth: Int = 16,
    psDdrCapacityBytes: BigInt = BigInt(4) << 30,
    plDdrCapacityBytes: BigInt = BigInt(1) << 30) {
  require(model == BitNetConfig.Production && resident.cfg == model)
  require(pageBytes == 4096 && pageBytes % 16 == 0)
  require(maxOutstanding >= 2 && (maxOutstanding & (maxOutstanding - 1)) == 0)
  require(epochWidth >= 8)
  require(psDdrCapacityBytes == (BigInt(4) << 30),
    "AXU3EG production PS DDR is 4 GiB")
  require(plDdrCapacityBytes == (BigInt(1) << 30),
    "AXU3EG production PL DDR is 1 GiB")

  val staging: BitNetPsToPlWeightStagingLayout =
    BitNetPsToPlWeightStagingLayout(model, resident)
  require(staging.pageBytes == pageBytes)
  require(staging.transformerDestinationBase + staging.transformerBytes <=
    plDdrCapacityBytes,
    "the persistent transformer shard must fit the physical PL DDR")

  val slotCount: Int = 2
  val slotWidth: Int = 1
  val beatBytes: Int = 16
  val beatsPerPage: Int = pageBytes / beatBytes
  val layerPages: Int = staging.layerPages
  val pageIndexWidth: Int = log2Up(layerPages)
  val pageCountWidth: Int = log2Up(layerPages + 1)
  val layerCountWidth: Int = log2Up(model.layerCount + 1)
  val outstandingIndexWidth: Int = log2Up(maxOutstanding)
  val byteCountWidth: Int = log2Up(pageBytes + 1)
  val targetCount: Int = staging.targetCount
  val targetByteWidth: Int = log2Up(
    staging.layers.head.targets.map(_.bytes).max.toInt + 1)
  // DMA completion identity must not depend on a software-supplied run epoch
  // being globally unique.  A private generation changes on every accepted
  // copy workload, so an old page completion cannot authenticate a later
  // image even if software repeats requestId/epoch.
  val dmaGenerationWidth: Int = 32
  val dmaTagWidth: Int = dmaGenerationWidth + model.layerWidth + slotWidth +
    pageIndexWidth
  val addressCalculationWidth: Int = model.addressWidth + 2

  require(beatsPerPage == 256)
  require(layerPages == 852,
    "production bank-0 layer shard must be exactly 852 pages")
  require(staging.layerBytes == 3489792)
  require(staging.transformerBytes == 104693760)
  require(dmaTagWidth <= 64,
    "the internal page-copy completion tag must remain a compact scalar")
}

object BitNetWeightStagingSlotState extends SpinalEnum(binarySequential) {
  val Empty, Filling, Ready, Computing = newElement()
}

object BitNetWeightStagingFault extends SpinalEnum(binarySequential) {
  // EpochSourceMismatch refers specifically to the resident imageEpoch/source
  // binding; the per-run epoch is allowed (and required) to change.
  val None,
      InvalidSourceRange,
      EpochSourceMismatch,
      DmaCompletionMismatch,
      DmaFailure,
      ComputeCompletionMismatch,
      ComputeFailure = newElement()
}

case class BitNetWeightStagingStart(
    cfg: BitNetPsToPlWeightStagingConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  /** Per-run identity.  Software must allocate a new value for every round. */
  val epoch = UInt(cfg.epochWidth bits)
  /**
    * Immutable model-image version.  Keep this stable across prefill/decode
    * rounds that use exactly the same weight bytes and source allocation.
    */
  val imageEpoch = UInt(cfg.epochWidth bits)
  /** Local byte address in the shared physical PS DDR. */
  val psSourceBase = UInt(cfg.model.addressWidth bits)
  /** Declared allocation size, including the complete bank0 transformer prefix. */
  val psSourceBytes = UInt((cfg.model.addressWidth + 1) bits)
}

/**
  * One PS-DDR-read to PL-DDR-write request.  The external copy engine may
  * retire requests in any order, but must echo `tag` exactly once.
  */
case class BitNetWeightStagingDmaCommand(
    cfg: BitNetPsToPlWeightStagingConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  /** Private controller generation; software cannot choose or reuse it. */
  val generation = UInt(cfg.dmaGenerationWidth bits)
  val tag = UInt(cfg.dmaTagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val page = UInt(cfg.pageIndexWidth bits)
  val target = BitNetAddressTarget()
  val sourceLocalAddress = UInt(cfg.model.addressWidth bits)
  val destinationLocalAddress = UInt(cfg.model.addressWidth bits)
  val bytes = UInt(cfg.byteCountWidth bits)
  /** Actual beat count; 256, not AXI LEN=255. */
  val beats = UInt(9 bits)
  val firstLayerPage = Bool()
  val lastLayerPage = Bool()
  val firstTargetPage = Bool()
  val lastTargetPage = Bool()
}

case class BitNetWeightStagingDmaCompletion(
    cfg: BitNetPsToPlWeightStagingConfig) extends Bundle {
  val tag = UInt(cfg.dmaTagWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/** Layer-compute permission.  All bank-0 addresses are final resident bases. */
case class BitNetWeightStagingComputeGrant(
    cfg: BitNetPsToPlWeightStagingConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val plLayerBase = UInt(cfg.model.addressWidth bits)
  val plLayerBytes = UInt(log2Up(cfg.staging.layerBytes.toInt + 1) bits)
  val targetBase = Vec(UInt(cfg.model.addressWidth bits), cfg.targetCount)
  val targetBytes = Vec(UInt(cfg.targetByteWidth bits), cfg.targetCount)
  /** True when no DMA was needed for this resident model-image epoch. */
  val reusedResident = Bool()
}

case class BitNetWeightStagingComputeCompletion(
    cfg: BitNetPsToPlWeightStagingConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val slot = UInt(cfg.slotWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

case class BitNetWeightStagingWorkloadCompletion(
    cfg: BitNetPsToPlWeightStagingConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val success = Bool()
  val fault = BitNetWeightStagingFault()
  val faultCode = Bits(8 bits)
  val residentLayerMask = Bits(cfg.model.layerCount bits)
}

/**
  * PL-owned, two-slot look-ahead controller for the only shard that must cross
  * from PS DDR to PL DDR.
  *
  * Banks 1..4 remain K-striped logical ranges in the single physical PS DDR
  * and are read in place through HP ports.  This controller copies only bank
  * 0's seven transformer target ranges.  It never copies activations, KV, LM
  * rows, AUX or RoPE, and it never asks the ARM to schedule individual pages.
  */
class BitNetPsToPlWeightStagingController(
    cfg: BitNetPsToPlWeightStagingConfig =
      BitNetPsToPlWeightStagingConfig()) extends Component {
  private val model = cfg.model
  private val layout = cfg.staging
  private val targets = BitNetFiveBankStripedLayout.WeightGeometry.map(_._1)

  val io = new Bundle {
    val start = slave(Stream(BitNetWeightStagingStart(cfg)))
    val dmaCommand = master(Stream(BitNetWeightStagingDmaCommand(cfg)))
    val dmaCompletion = slave(Stream(BitNetWeightStagingDmaCompletion(cfg)))
    val computeGrant = master(Stream(BitNetWeightStagingComputeGrant(cfg)))
    val computeCompletion =
      slave(Stream(BitNetWeightStagingComputeCompletion(cfg)))
    val completion =
      master(Stream(BitNetWeightStagingWorkloadCompletion(cfg)))

    val busy = out Bool()
    val faultSticky = out Bool()
    val residentLayerMask = out Bits(model.layerCount bits)
    val slotState = out Vec(BitNetWeightStagingSlotState(), cfg.slotCount)
    val outstanding = out UInt(log2Up(cfg.maxOutstanding + 1) bits)
  }

  val workloadActive = RegInit(False)
  val faulted = RegInit(False)
  val activeFault = Reg(BitNetWeightStagingFault()) init
    BitNetWeightStagingFault.None
  val activeFaultCode = Reg(Bits(8 bits)) init 0
  val heldRequest = Reg(UInt(model.requestIdWidth bits)) init 0
  // heldEpoch is deliberately the per-run identity echoed by every command,
  // grant and completion.  It is never used as a resident-cache key.
  val heldEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val heldImageEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val heldDmaGeneration = Reg(UInt(cfg.dmaGenerationWidth bits)) init 0
  val nextDmaGeneration = Reg(UInt(cfg.dmaGenerationWidth bits)) init 1
  val heldSourceBase = Reg(UInt(model.addressWidth bits)) init 0
  val heldSourceBytes = Reg(UInt((model.addressWidth + 1) bits)) init 0

  // Residency belongs to an immutable weight image, not to one inference run.
  val residentImageEpochValid = RegInit(False)
  val residentImageEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val residentSourceBase = Reg(UInt(model.addressWidth bits)) init 0
  val residentSourceBytes = Reg(UInt((model.addressWidth + 1) bits)) init 0
  val residentValid = Vec.fill(model.layerCount)(RegInit(False))

  val slots = Vec.fill(cfg.slotCount)(
    Reg(BitNetWeightStagingSlotState()) init BitNetWeightStagingSlotState.Empty)
  val slotLayer = Vec.fill(cfg.slotCount)(
    Reg(UInt(model.layerWidth bits)) init 0)
  val slotEpoch = Vec.fill(cfg.slotCount)(
    Reg(UInt(cfg.epochWidth bits)) init 0)
  val slotReused = Vec.fill(cfg.slotCount)(RegInit(False))

  val nextPrefetchLayer = Reg(UInt(cfg.layerCountWidth bits)) init 0
  val nextComputeLayer = Reg(UInt(cfg.layerCountWidth bits)) init 0
  val computeActive = RegInit(False)
  val activeComputeSlot = Reg(UInt(cfg.slotWidth bits)) init 0
  val activeComputeLayer = Reg(UInt(model.layerWidth bits)) init 0

  val fillActive = RegInit(False)
  val fillSlot = Reg(UInt(cfg.slotWidth bits)) init 0
  val fillLayer = Reg(UInt(model.layerWidth bits)) init 0
  val issuedPages = Reg(UInt(cfg.pageCountWidth bits)) init 0
  val completedPages = Reg(UInt(cfg.pageCountWidth bits)) init 0

  val entryValid = Vec.fill(cfg.maxOutstanding)(RegInit(False))
  val entryTag = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(cfg.dmaTagWidth bits)) init 0)
  val inflightCount = Reg(UInt(log2Up(cfg.maxOutstanding + 1) bits)) init 0

  val pendingValid = RegInit(False)
  val pendingEntry = Reg(UInt(cfg.outstandingIndexWidth bits)) init 0
  val pendingPayload = Reg(BitNetWeightStagingDmaCommand(cfg)) init
    BitNetWeightStagingDmaCommand(cfg).getZero

  val completionValid = RegInit(False)
  val completionPayload = Reg(BitNetWeightStagingWorkloadCompletion(cfg)) init
    BitNetWeightStagingWorkloadCompletion(cfg).getZero
  io.completion.valid := completionValid
  io.completion.payload := completionPayload
  when(io.completion.fire) { completionValid := False }

  val sticky = RegInit(False)
  io.faultSticky := sticky
  io.busy := workloadActive || completionValid || inflightCount =/= 0 ||
    pendingValid
  io.outstanding := inflightCount
  for (slot <- 0 until cfg.slotCount) io.slotState(slot) := slots(slot)
  for (layer <- 0 until model.layerCount)
    io.residentLayerMask(layer) := residentValid(layer)

  // -----------------------------------------------------------------------
  // One software submission declares the PS-DDR source allocation.  Every
  // subsequent layer/page decision is made here in PL.
  // -----------------------------------------------------------------------
  val sourceBaseWide = io.start.psSourceBase.resize(cfg.addressCalculationWidth)
  val sourceBytesWide = io.start.psSourceBytes.resize(cfg.addressCalculationWidth)
  val declaredSourceEnd = sourceBaseWide + sourceBytesWide
  val requiredSourceEnd = sourceBaseWide +
    U(layout.transformerBytes, cfg.addressCalculationWidth bits)
  val sourceAligned = io.start.psSourceBase(log2Up(cfg.pageBytes) - 1 downto 0) === 0
  val sourceRangeValid = sourceAligned &&
    io.start.psSourceBytes >= U(layout.transformerBytes,
      (model.addressWidth + 1) bits) &&
    !declaredSourceEnd.msb && !requiredSourceEnd.msb &&
    requiredSourceEnd <= declaredSourceEnd &&
    declaredSourceEnd <= U(cfg.psDdrCapacityBytes,
      cfg.addressCalculationWidth bits)
  val sameImageEpoch = residentImageEpochValid &&
    io.start.imageEpoch === residentImageEpoch
  val imageSourceMatches = !sameImageEpoch ||
    (io.start.psSourceBase === residentSourceBase &&
      io.start.psSourceBytes === residentSourceBytes)
  val startLegal = sourceRangeValid && imageSourceMatches

  io.start.ready := !workloadActive && !completionValid &&
    inflightCount === 0 && !pendingValid

  when(io.start.fire) {
    when(startLegal) {
      workloadActive := True
      faulted := False
      activeFault := BitNetWeightStagingFault.None
      activeFaultCode := 0
      heldRequest := io.start.requestId
      heldEpoch := io.start.epoch
      heldImageEpoch := io.start.imageEpoch
      heldDmaGeneration := nextDmaGeneration
      nextDmaGeneration := nextDmaGeneration + 1
      heldSourceBase := io.start.psSourceBase
      heldSourceBytes := io.start.psSourceBytes
      nextPrefetchLayer := 0
      nextComputeLayer := 0
      computeActive := False
      fillActive := False
      issuedPages := 0
      completedPages := 0
      for (slot <- 0 until cfg.slotCount) {
        slots(slot) := BitNetWeightStagingSlotState.Empty
        slotReused(slot) := False
      }
      when(!sameImageEpoch) {
        residentImageEpochValid := True
        residentImageEpoch := io.start.imageEpoch
        residentSourceBase := io.start.psSourceBase
        residentSourceBytes := io.start.psSourceBytes
        for (layer <- 0 until model.layerCount) residentValid(layer) := False
      }
    } otherwise {
      completionPayload.requestId := io.start.requestId
      completionPayload.epoch := io.start.epoch
      completionPayload.success := False
      completionPayload.fault := BitNetWeightStagingFault.InvalidSourceRange
      when(sourceRangeValid && !imageSourceMatches) {
        completionPayload.fault := BitNetWeightStagingFault.EpochSourceMismatch
      }
      completionPayload.faultCode := 0
      completionPayload.residentLayerMask := io.residentLayerMask
      completionValid := True
      sticky := True
    }
  }

  // -----------------------------------------------------------------------
  // Fill the current slot, then immediately use the other slot to copy the
  // next layer while the first layer computes.  Resident hits need no DMA.
  // -----------------------------------------------------------------------
  val freeSlotExists =
    slots(0) === BitNetWeightStagingSlotState.Empty ||
      slots(1) === BitNetWeightStagingSlotState.Empty
  val selectedFreeSlot = UInt(cfg.slotWidth bits)
  selectedFreeSlot := 0
  when(slots(0) =/= BitNetWeightStagingSlotState.Empty) {
    selectedFreeSlot := 1
  }
  val prefetchLayerLegal = nextPrefetchLayer < model.layerCount
  val safePrefetchLayer = UInt(model.layerWidth bits)
  safePrefetchLayer := 0
  when(prefetchLayerLegal) {
    safePrefetchLayer := nextPrefetchLayer.resize(model.layerWidth)
  }
  val activeImageMatchesResident = residentImageEpochValid &&
    heldImageEpoch === residentImageEpoch
  val selectedLayerResident = activeImageMatchesResident &&
    residentValid(safePrefetchLayer)
  val scheduleLayer = workloadActive && !faulted && !fillActive &&
    prefetchLayerLegal && freeSlotExists

  when(scheduleLayer) {
    slotLayer(selectedFreeSlot) := safePrefetchLayer
    slotEpoch(selectedFreeSlot) := heldEpoch
    slotReused(selectedFreeSlot) := selectedLayerResident
    nextPrefetchLayer := nextPrefetchLayer + 1
    when(selectedLayerResident) {
      slots(selectedFreeSlot) := BitNetWeightStagingSlotState.Ready
    } otherwise {
      slots(selectedFreeSlot) := BitNetWeightStagingSlotState.Filling
      fillActive := True
      fillSlot := selectedFreeSlot
      fillLayer := safePrefetchLayer
      issuedPages := 0
      completedPages := 0
    }
  }

  // -----------------------------------------------------------------------
  // Backpressure-stable DMA descriptor generation with a bounded tag table.
  // -----------------------------------------------------------------------
  val freeEntryExists = Bool()
  val selectedFreeEntry = UInt(cfg.outstandingIndexWidth bits)
  freeEntryExists := False
  selectedFreeEntry := 0
  for (entry <- (0 until cfg.maxOutstanding).reverse) {
    when(!entryValid(entry)) {
      freeEntryExists := True
      selectedFreeEntry := entry
    }
  }

  val layerSourceOffsets = Vec(layout.layers.map(layer =>
    U(layer.sourceOffset, model.addressWidth bits)))
  val layerDestinationBases = Vec(layout.layers.map(layer =>
    U(layer.destinationBase, model.addressWidth bits)))
  val currentLayerSourceOffset = layerSourceOffsets(fillLayer)
  val currentLayerDestinationBase = layerDestinationBases(fillLayer)
  val pageByteOffset =
    (issuedPages.resize(model.addressWidth) << log2Up(cfg.pageBytes)).resized
  val nextSourceAddress = heldSourceBase + currentLayerSourceOffset + pageByteOffset
  val nextDestinationAddress = currentLayerDestinationBase + pageByteOffset

  val selectedTarget = BitNetAddressTarget()
  val selectedTargetFirst = Bool()
  val selectedTargetLast = Bool()
  selectedTarget := targets.head
  selectedTargetFirst := issuedPages === 0
  selectedTargetLast := False
  var pageStart = 0
  for ((target, index) <- targets.zipWithIndex) {
    val pages = (layout.layers.head.targets(index).bytes / cfg.pageBytes).toInt
    val pageEnd = pageStart + pages
    when(issuedPages >= pageStart && issuedPages < pageEnd) {
      selectedTarget := target
      selectedTargetFirst := issuedPages === pageStart
      selectedTargetLast := issuedPages === pageEnd - 1
    }
    pageStart = pageEnd
  }

  val nextTag = UInt(cfg.dmaTagWidth bits)
  nextTag := (heldDmaGeneration.asBits ## fillLayer.asBits ## fillSlot.asBits ##
    issuedPages.resize(cfg.pageIndexWidth).asBits).asUInt

  val canPrepare = workloadActive && !faulted && fillActive &&
    !pendingValid && issuedPages < cfg.layerPages && freeEntryExists
  when(canPrepare) {
    pendingValid := True
    pendingEntry := selectedFreeEntry
    pendingPayload.requestId := heldRequest
    pendingPayload.generation := heldDmaGeneration
    pendingPayload.tag := nextTag
    pendingPayload.epoch := heldEpoch
    pendingPayload.layer := fillLayer
    pendingPayload.slot := fillSlot
    pendingPayload.page := issuedPages.resized
    pendingPayload.target := selectedTarget
    pendingPayload.sourceLocalAddress := nextSourceAddress
    pendingPayload.destinationLocalAddress := nextDestinationAddress
    pendingPayload.bytes := cfg.pageBytes
    pendingPayload.beats := cfg.beatsPerPage
    pendingPayload.firstLayerPage := issuedPages === 0
    pendingPayload.lastLayerPage := issuedPages === cfg.layerPages - 1
    pendingPayload.firstTargetPage := selectedTargetFirst
    pendingPayload.lastTargetPage := selectedTargetLast
  }

  io.dmaCommand.valid := pendingValid && workloadActive && !faulted
  io.dmaCommand.payload := pendingPayload
  val dmaIssueFire = io.dmaCommand.fire
  when(dmaIssueFire) {
    pendingValid := False
    entryValid(pendingEntry) := True
    entryTag(pendingEntry) := pendingPayload.tag
    issuedPages := issuedPages + 1
  }

  // -----------------------------------------------------------------------
  // DMA completions may be arbitrarily delayed or reordered.  A missing,
  // duplicate, stale-epoch or corrupted tag cannot make a layer Ready.
  // -----------------------------------------------------------------------
  val completionMatches = Vec(Bool(), cfg.maxOutstanding)
  for (entry <- 0 until cfg.maxOutstanding) {
    completionMatches(entry) := entryValid(entry) &&
      entryTag(entry) === io.dmaCompletion.tag
  }
  val completionMatchBits = completionMatches.asBits
  val completionMatchFound = completionMatchBits.orR
  val completionMatchEntry = OHToUInt(completionMatchBits)
    .resize(cfg.outstandingIndexWidth)
  io.dmaCompletion.ready := workloadActive
  val dmaCompletionFire = io.dmaCompletion.fire
  val dmaRetireFire = dmaCompletionFire && completionMatchFound

  when(dmaRetireFire) {
    entryValid(completionMatchEntry) := False
    completedPages := completedPages + 1
  }

  switch(dmaIssueFire ## dmaRetireFire) {
    is(B"2'b10") { inflightCount := inflightCount + 1 }
    is(B"2'b01") { inflightCount := inflightCount - 1 }
    default { }
  }

  val dmaTagFault = dmaCompletionFire && !completionMatchFound
  val dmaResponseFault = dmaRetireFire && !io.dmaCompletion.success

  val lastSuccessfulPage = !faulted && !dmaTagFault && !dmaResponseFault &&
    dmaRetireFire &&
    io.dmaCompletion.success &&
    issuedPages === cfg.layerPages && inflightCount === 1 && !pendingValid
  when(lastSuccessfulPage) {
    residentValid(fillLayer) := True
    slots(fillSlot) := BitNetWeightStagingSlotState.Ready
    slotReused(fillSlot) := False
    fillActive := False
  }

  // -----------------------------------------------------------------------
  // Compute is strictly layer ordered and only one layer may own the engines.
  // A slot cannot be recycled until its tagged completion returns.
  // -----------------------------------------------------------------------
  val readySlotFound = Bool()
  val selectedReadySlot = UInt(cfg.slotWidth bits)
  readySlotFound := False
  selectedReadySlot := 0
  for (slot <- (0 until cfg.slotCount).reverse) {
    when(slots(slot) === BitNetWeightStagingSlotState.Ready &&
      slotLayer(slot).resize(cfg.layerCountWidth) === nextComputeLayer) {
      readySlotFound := True
      selectedReadySlot := slot
    }
  }

  val selectedGrantLayer = slotLayer(selectedReadySlot)
  val grantLayerBases = Vec(layout.layers.map(layer =>
    U(layer.destinationBase, model.addressWidth bits)))
  io.computeGrant.valid := workloadActive && !faulted && !computeActive &&
    readySlotFound
  io.computeGrant.requestId := heldRequest
  io.computeGrant.epoch := heldEpoch
  io.computeGrant.layer := selectedGrantLayer
  io.computeGrant.slot := selectedReadySlot
  io.computeGrant.plLayerBase := grantLayerBases(selectedGrantLayer)
  io.computeGrant.plLayerBytes := U(layout.layerBytes,
    io.computeGrant.plLayerBytes.getWidth bits)
  io.computeGrant.reusedResident := slotReused(selectedReadySlot)

  for (targetIndex <- targets.indices) {
    val bases = Vec(layout.layers.map(layer =>
      U(layer.destinationBase + layer.targets(targetIndex).layerOffset,
        model.addressWidth bits)))
    io.computeGrant.targetBase(targetIndex) := bases(selectedGrantLayer)
    io.computeGrant.targetBytes(targetIndex) := U(
      layout.layers.head.targets(targetIndex).bytes,
      cfg.targetByteWidth bits)
  }

  when(io.computeGrant.fire) {
    slots(selectedReadySlot) := BitNetWeightStagingSlotState.Computing
    computeActive := True
    activeComputeSlot := selectedReadySlot
    activeComputeLayer := selectedGrantLayer
    nextComputeLayer := nextComputeLayer + 1
  }

  io.computeCompletion.ready := workloadActive
  val computeCompletionFire = io.computeCompletion.fire
  val computeCompletionMatches = computeActive &&
    io.computeCompletion.requestId === heldRequest &&
    io.computeCompletion.epoch === heldEpoch &&
    io.computeCompletion.layer === activeComputeLayer &&
    io.computeCompletion.slot === activeComputeSlot &&
    slots(activeComputeSlot) === BitNetWeightStagingSlotState.Computing
  val computeTagFault = computeCompletionFire && !computeCompletionMatches
  val computeResponseFault = computeCompletionFire &&
    computeCompletionMatches && !io.computeCompletion.success
  val activeFaultEvent = dmaTagFault || dmaResponseFault ||
    computeTagFault || computeResponseFault

  // A matching completion always retires the granted compute lease, including
  // a failed completion.  Keeping Computing asserted on the failure path would
  // self-deadlock the fault drain once terminal reporting is required to wait
  // for the authenticated lease to retire.
  when(computeCompletionFire && computeCompletionMatches) {
    slots(activeComputeSlot) := BitNetWeightStagingSlotState.Empty
    computeActive := False
    when(io.computeCompletion.success && !faulted && !activeFaultEvent &&
      activeComputeLayer === model.layerCount - 1) {
      workloadActive := False
      completionPayload.requestId := heldRequest
      completionPayload.epoch := heldEpoch
      completionPayload.success := True
      completionPayload.fault := BitNetWeightStagingFault.None
      completionPayload.faultCode := 0
      completionPayload.residentLayerMask := io.residentLayerMask
      completionValid := True
    }
  }

  // -----------------------------------------------------------------------
  // Fail closed.  Stop new copies/grants, invalidate the resident image,
  // issued DMA writes, and report only after every accepted tag and any
  // authenticated compute lease retire.  A compute-tag mismatch deliberately
  // remains fail-stop if no later matching completion arrives: the controller
  // cannot safely invent retirement for an unauthenticated owner.
  // -----------------------------------------------------------------------
  val selectedFault = BitNetWeightStagingFault()
  val selectedFaultCode = Bits(8 bits)
  selectedFault := BitNetWeightStagingFault.None
  selectedFaultCode := 0
  when(dmaTagFault) {
    selectedFault := BitNetWeightStagingFault.DmaCompletionMismatch
  } elsewhen (dmaResponseFault) {
    selectedFault := BitNetWeightStagingFault.DmaFailure
    selectedFaultCode := io.dmaCompletion.faultCode
  } elsewhen (computeTagFault) {
    selectedFault := BitNetWeightStagingFault.ComputeCompletionMismatch
  } elsewhen (computeResponseFault) {
    selectedFault := BitNetWeightStagingFault.ComputeFailure
    selectedFaultCode := io.computeCompletion.faultCode
  }

  when(workloadActive && !faulted && activeFaultEvent) {
    faulted := True
    activeFault := selectedFault
    activeFaultCode := selectedFaultCode
    pendingValid := False
    residentImageEpochValid := False
    for (layer <- 0 until model.layerCount) residentValid(layer) := False
    sticky := True
  }

  when(workloadActive && faulted && inflightCount === 0 && !pendingValid &&
    !computeActive && !completionValid) {
    workloadActive := False
    fillActive := False
    computeActive := False
    for (slot <- 0 until cfg.slotCount)
      slots(slot) := BitNetWeightStagingSlotState.Empty
    completionPayload.requestId := heldRequest
    completionPayload.epoch := heldEpoch
    completionPayload.success := False
    completionPayload.fault := activeFault
    completionPayload.faultCode := activeFaultCode
    completionPayload.residentLayerMask := 0
    completionValid := True
  }
}

object GenerateBitNetPsToPlWeightStagingController extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-ps-to-pl-weight-staging")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetPsToPlWeightStagingController())
}
