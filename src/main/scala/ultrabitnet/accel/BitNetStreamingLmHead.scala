package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Static geometry of the production K-striped streaming LM head. */
case class BitNetStreamingLmHeadConfig(
    hiddenSize: Int,
    vocabularySize: Int,
    tileFeatures: Int,
    bankCount: Int,
    engineCount: Int,
    outstandingRows: Int,
    addressWidth: Int,
    activationSlots: Int,
    tokenLanes: Int,
    stripeInputStarts: Vector[Int],
    stripeInputFeatures: Vector[Int],
    stripeWeightBases: Vector[BigInt],
    scaleRowStarts: Vector[Int],
    scaleRows: Vector[Int],
    scaleBases: Vector[BigInt]
) {
  require(hiddenSize > 0 && vocabularySize > 0)
  require(tileFeatures == 64 || tileFeatures == 128,
    "the native array consumes one 64- or 128-feature LM tile")
  require(hiddenSize % tileFeatures == 0)
  require(bankCount == 5, "production scheduler is a five-bank design")
  require(engineCount == 1 || engineCount == 2,
    "production scheduler targets one or two continuous arrays")
  require(outstandingRows >= 8 && isPow2(outstandingRows))
  require(addressWidth >= 32)
  require(activationSlots >= 2 && isPow2(activationSlots))
  require(tokenLanes > 0 && isPow2(tokenLanes))
  require(stripeInputStarts.size == bankCount)
  require(stripeInputFeatures.size == bankCount)
  require(stripeWeightBases.size == bankCount)
  require(scaleRowStarts.size == bankCount)
  require(scaleRows.size == bankCount)
  require(scaleBases.size == bankCount)
  require(stripeInputStarts.head == 0)
  require(stripeInputFeatures.forall(features =>
    features > 0 && features % tileFeatures == 0))
  require(stripeInputStarts.zip(stripeInputFeatures).sliding(2).forall {
    case Seq((leftStart, leftRows), (rightStart, _)) =>
      leftStart + leftRows == rightStart
    case _ => true
  })
  require(stripeInputStarts.last + stripeInputFeatures.last == hiddenSize)
  require(scaleRowStarts.head == 0)
  require(scaleRows.forall(_ > 0))
  require(scaleRowStarts.zip(scaleRows).sliding(2).forall {
    case Seq((leftStart, leftRows), (rightStart, _)) =>
      leftStart + leftRows == rightStart
    case _ => true
  })
  require(scaleRows.sum == vocabularySize)
  require(scaleRowStarts.last + scaleRows.last == vocabularySize)
  require(stripeWeightBases.forall(_ >= 0))
  require(scaleBases.forall(_ >= 0))
  require(stripeWeightBases.zip(stripeInputFeatures).forall {
    case (base, features) =>
      base + BigInt(vocabularySize) * features <=
        (BigInt(1) << addressWidth)
  })
  require(scaleBases.zip(scaleRows).forall { case (base, rows) =>
    base + BigInt(rows) * 4 <= (BigInt(1) << addressWidth)
  })

  val tileCount: Int = hiddenSize / tileFeatures
  val tileIndexWidth: Int = scala.math.max(1, log2Up(tileCount))
  val featureIndexWidth: Int = scala.math.max(1, log2Up(hiddenSize))
  val tokenIdWidth: Int = scala.math.max(1, log2Up(vocabularySize))
  val bankWidth: Int = log2Up(bankCount)
  val engineWidth: Int = scala.math.max(1, log2Up(engineCount))
  val requestIdWidth: Int = log2Up(outstandingRows)
  val activationSlotWidth: Int = log2Up(activationSlots)
  val tokenLaneWidth: Int = log2Up(tokenLanes)
  val rowCountWidth: Int = log2Up(vocabularySize + 1)
  val pairCount: Int = (vocabularySize + engineCount - 1) / engineCount
  val pairOrdinalWidth: Int = scala.math.max(1, log2Up(pairCount))
  val stripeFeatureWidth: Int = log2Up(hiddenSize + 1)
}

object BitNetStreamingLmHeadConfig {
  def production(
      layout: BitNetFiveBankStripedLayout =
        BitNetFiveBankStripedLayout(),
      tileFeatures: Int = 128
  ): BitNetStreamingLmHeadConfig = {
    val cfg = layout.cfg
    BitNetStreamingLmHeadConfig(
      hiddenSize = cfg.hiddenSize,
      vocabularySize = cfg.vocabularySize,
      tileFeatures = tileFeatures,
      bankCount = 5,
      engineCount = cfg.engineCount,
      outstandingRows = 64,
      addressWidth = cfg.addressWidth,
      activationSlots = cfg.tokenParallelism,
      tokenLanes = cfg.tokenParallelism,
      stripeInputStarts = layout.lmShards.map(_.inputStart),
      stripeInputFeatures = layout.lmShards.map(_.inputFeatures),
      stripeWeightBases = layout.lmShards.map(_.weightBase),
      scaleRowStarts = layout.lmShards.map(_.scaleRowStart),
      scaleRows = layout.lmShards.map(_.scaleRows),
      scaleBases = layout.lmShards.map(_.scaleBase)
    )
  }
}

case class BitNetLmHeadStart(cfg: BitNetStreamingLmHeadConfig)
    extends Bundle {
  /** Logical producer mode; it never changes the physical LM row mapping. */
  val logicalMode = BitNetMode()
  /** Active prefill lanes form a low-bit prefix; decode uses lane zero only. */
  val activeTokenMask = Bits(cfg.tokenLanes bits)
  /** Final prompt token lane, or zero for decode. */
  val finalTokenLane = UInt(cfg.tokenLaneWidth bits)
  /** Slot containing the final-norm output after its one dynamic-i8 pass. */
  val activationSlot = UInt(cfg.activationSlotWidth bits)
  /** Protects the two arrays from consuming a stale resident activation. */
  val activationEpoch = UInt(16 bits)
  /** Common positive Q16.16 factor; it cancels from argmax comparison. */
  val activationDequantFactorQ16 = UInt(32 bits)
}

/** One of the exactly 20 production activation tiles broadcast to both arrays. */
case class BitNetLmActivationTile(cfg: BitNetStreamingLmHeadConfig)
    extends Bundle {
  val logicalMode = BitNetMode()
  /** Always Decode: both arrays consume the same token and different rows. */
  val physicalMode = BitNetMode()
  val sourceTokenLane = UInt(cfg.tokenLaneWidth bits)
  val activationSlot = UInt(cfg.activationSlotWidth bits)
  val activationEpoch = UInt(16 bits)
  val tile = UInt(cfg.tileIndexWidth bits)
  val featureStart = UInt(cfg.featureIndexWidth bits)
  val features = UInt(log2Up(cfg.tileFeatures + 1) bits)
  val engineMask = Bits(cfg.engineCount bits)
  val first = Bool()
  val last = Bool()
}

/** One physical decode-row shard assigned to one of the two matrix arrays. */
case class BitNetLmEngineRowDescriptor(
    cfg: BitNetStreamingLmHeadConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val tokenId = UInt(cfg.tokenIdWidth bits)
  /** Every row consumes one contiguous K stripe from every memory bank. */
  val weightAddress = Vec(UInt(cfg.addressWidth bits), cfg.bankCount)
  /** F32 scales remain output-row-sharded and are fetched once per row. */
  val scaleBank = UInt(cfg.bankWidth bits)
  val scaleAddress = UInt(cfg.addressWidth bits)
}

/**
  * One row pair for the two physical arrays.  Engine 0 owns the even row and
  * engine 1 the odd row in both logical prefill and logical decode.
  */
case class BitNetLmRowPairRequest(cfg: BitNetStreamingLmHeadConfig)
    extends Bundle {
  val pairOrdinal = UInt(cfg.pairOrdinalWidth bits)
  val engineMask = Bits(cfg.engineCount bits)
  val engine = Vec(BitNetLmEngineRowDescriptor(cfg), cfg.engineCount)
  val inputStart = Vec(UInt(cfg.stripeFeatureWidth bits), cfg.bankCount)
  val inputFeatures = Vec(UInt(cfg.stripeFeatureWidth bits), cfg.bankCount)
  val firstPair = Bool()
  val lastPair = Bool()
}

/**
  * Row-level completion contract.  This is intentionally independent of the
  * internal continuous-array API.  A small adapter may join lastK dot output
  * with the raw F32 row scale and preserve this tag.
  */
case class BitNetLmRowCompletion(cfg: BitNetStreamingLmHeadConfig)
    extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val engine = UInt(cfg.engineWidth bits)
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val scaleBank = UInt(cfg.bankWidth bits)
  val dot = SInt(32 bits)
  val rowScaleF32 = Bits(32 bits)
  val tilesSeen = UInt(log2Up(cfg.tileCount + 1) bits)
  val bankMask = Bits(cfg.bankCount bits)
  val firstSeen = Bool()
  val lastSeen = Bool()
}

/** Only the winner and audit metadata leave the PL datapath. */
case class BitNetLmWinner(cfg: BitNetStreamingLmHeadConfig)
    extends Bundle {
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val dot = SInt(32 bits)
  val rowScaleF32 = Bits(32 bits)
  val bank = UInt(cfg.bankWidth bits)
  val activationDequantFactorQ16 = UInt(32 bits)
  val rowsCompared = UInt(cfg.rowCountWidth bits)
  val cycles = UInt(32 bits)
  val error = Bool()
}

case class BitNetLmQueuedCandidate(
    cfg: BitNetStreamingLmHeadConfig) extends Bundle {
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val bank = UInt(cfg.bankWidth bits)
  val dot = SInt(32 bits)
  val rowScaleF32 = Bits(32 bits)
  val framingFault = Bool()
}

/** Synchronous-read candidate queue backed by block RAM.
  *
  * `memoryCount` excludes the read pipeline and output register, while the
  * public occupancy covers all three locations.  This preserves exactly the
  * configured capacity and allows a push on the same edge that the full queue
  * pops.  The multiplier consumes at most one candidate every 25 cycles, so
  * the explicit read prefetch adds no steady-state LM row bottleneck.
  */
class BitNetLmCandidateBramFifo(cfg: BitNetStreamingLmHeadConfig)
    extends Component {
  private val depth = cfg.outstandingRows
  private val pointerWidth = log2Up(depth)
  private val occupancyWidth = log2Up(depth + 1)

  val io = new Bundle {
    val push = slave(Stream(BitNetLmQueuedCandidate(cfg)))
    val pop = master(Stream(BitNetLmQueuedCandidate(cfg)))
    val flush = in Bool()
    val occupancy = out UInt(occupancyWidth bits)
  }

  val memory = Mem(BitNetLmQueuedCandidate(cfg), depth)
  memory.addAttribute("ram_style", "block")
  memory.setName("candidateMemory")
  val writePointer = Reg(UInt(pointerWidth bits)) init 0
  val readPointer = Reg(UInt(pointerWidth bits)) init 0
  val memoryCount = Reg(UInt(occupancyWidth bits)) init 0
  val readPending = RegInit(False)
  val outputValid = RegInit(False)
  val outputPayload = Reg(BitNetLmQueuedCandidate(cfg)) init
    BitNetLmQueuedCandidate(cfg).getZero

  val totalOccupancy = UInt(occupancyWidth bits)
  totalOccupancy := (memoryCount + readPending.asUInt.resize(occupancyWidth) +
    outputValid.asUInt.resize(occupancyWidth)).resized
  io.occupancy := totalOccupancy
  io.pop.valid := outputValid
  io.pop.payload := outputPayload
  io.push.ready := totalOccupancy < depth || io.pop.fire

  val issueRead = memoryCount =/= 0 && !readPending &&
    (!outputValid || io.pop.ready)
  val readData = memory.readSync(readPointer, issueRead)

  when(io.push.fire) {
    memory.write(writePointer, io.push.payload)
    writePointer := writePointer + 1
  }
  when(issueRead) {
    readPointer := readPointer + 1
  }
  when(io.push.fire =/= issueRead) {
    when(io.push.fire) {
      memoryCount := memoryCount + 1
    } otherwise {
      memoryCount := memoryCount - 1
    }
  }

  readPending := issueRead
  when(io.pop.fire) {
    outputValid := False
  }
  when(readPending) {
    outputValid := True
    outputPayload := readData
  }

  when(io.flush) {
    writePointer := 0
    readPointer := 0
    memoryCount := 0
    readPending := False
    outputValid := False
  }
}

/**
  * Production streamed LM-head frontend and exact argmax backend.
  *
  * No logit RAM exists.  At most `outstandingRows` tags plus one queued row-dot
  * FIFO are resident.  The shared scaler uses a 24-cycle shift/add 32x24
  * multiply (zero DSPs) and converts the exact integer-times-F32 product to a
  * canonical sign/exponent/56-bit-significand tuple.  Comparing those tuples
  * is exactly equivalent to comparing real `dot * positiveFloatScale`, for
  * normal and subnormal IEEE-754 scales.  The activation dequantization factor
  * is common to every row and therefore cancels without approximation.
  */
class BitNetStreamingLmHead(
    cfg: BitNetStreamingLmHeadConfig =
      BitNetStreamingLmHeadConfig.production()) extends Component {
  val io = new Bundle {
    val start = slave(Stream(BitNetLmHeadStart(cfg)))
    // Job-scoped local flush.  The resident wrapper delays this signal while
    // a rowPair offer is already visible, so no claimed Stream is withdrawn.
    val abort = in Bool()
    val activationTile = master(Stream(BitNetLmActivationTile(cfg)))
    val rowPair = master(Stream(BitNetLmRowPairRequest(cfg)))
    val rowCompletion = slave(Stream(BitNetLmRowCompletion(cfg)))
    val winner = master(Stream(BitNetLmWinner(cfg)))

    val busy = out Bool()
    val protocolFault = out Bool()
    val rowsIssued = out UInt(cfg.rowCountWidth bits)
    val rowsAccepted = out UInt(cfg.rowCountWidth bits)
    val rowsCompared = out UInt(cfg.rowCountWidth bits)
    val outstanding = out UInt(log2Up(cfg.outstandingRows + 1) bits)
  }

  val running = RegInit(False)
  val broadcastDone = RegInit(False)
  val allRowsIssued = RegInit(False)
  val stickyFault = RegInit(False)
  val activationSlot = Reg(UInt(cfg.activationSlotWidth bits)) init (0)
  val activationEpoch = Reg(UInt(16 bits)) init (0)
  val activationFactor = Reg(UInt(32 bits)) init (0)
  val logicalMode = Reg(BitNetMode()) init BitNetMode.Decode
  val finalTokenLane = Reg(UInt(cfg.tokenLaneWidth bits)) init (0)
  val activationTileIndex = Reg(UInt(cfg.tileIndexWidth bits)) init (0)
  val cycleCounter = Reg(UInt(32 bits)) init (0)

  val issuedCount = Reg(UInt(cfg.rowCountWidth bits)) init (0)
  val acceptedCount = Reg(UInt(cfg.rowCountWidth bits)) init (0)
  val comparedCount = Reg(UInt(cfg.rowCountWidth bits)) init (0)

  val nextRowBase = Reg(UInt(cfg.tokenIdWidth bits)) init (0)
  val nextPairOrdinal = Reg(UInt(cfg.pairOrdinalWidth bits)) init (0)
  val nextRequestId = Reg(UInt(cfg.requestIdWidth bits)) init (0)

  val tagActive = Vec(RegInit(False), cfg.outstandingRows)
  // Request IDs and vocabulary rows start at zero and advance together.  The
  // request ID therefore supplies the token low bits and its LSB identifies
  // the physical engine.  Only token high bits and the row-sharded scale bank
  // need per-tag storage.  Even/odd banks retain two writes per issued pair
  // without building three wide 64:1 register muxes on completion.
  private val expectedTokenHighWidth =
    scala.math.max(0, cfg.tokenIdWidth - cfg.requestIdWidth)
  private val expectedMetadataWidth =
    expectedTokenHighWidth + cfg.bankWidth
  private val expectedMetadataDepth = cfg.outstandingRows / cfg.engineCount
  val expectedMetadataEven = Vec.fill(expectedMetadataDepth)(
    Reg(Bits(expectedMetadataWidth bits)) init 0)
  val expectedMetadataOdd = Vec.fill(expectedMetadataDepth)(
    Reg(Bits(expectedMetadataWidth bits)) init 0)
  expectedMetadataEven.setName("lmExpectedMetadataEven")
  expectedMetadataOdd.setName("lmExpectedMetadataOdd")

  val resultValid = RegInit(False)
  val result = Reg(BitNetLmWinner(cfg)) init (BitNetLmWinner(cfg).getZero)
  io.winner.valid := resultValid
  io.winner.payload := result
  when(io.winner.fire) { resultValid := False }

  // The candidate FIFO and multiplier must be empty before a new generation.
  val candidateFifo = new BitNetLmCandidateBramFifo(cfg)
  val multiplierBusy = RegInit(False)
  io.start.ready := !running && !resultValid && !multiplierBusy &&
    candidateFifo.io.occupancy === 0

  val validPrefillLane = Vec(Bool(), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    validPrefillLane(lane) :=
      io.start.activeTokenMask === B((BigInt(1) << (lane + 1)) - 1,
        cfg.tokenLanes bits) &&
      io.start.finalTokenLane === lane
  }
  val startModeLegal =
    (io.start.logicalMode === BitNetMode.Decode &&
      io.start.activeTokenMask === 1 && io.start.finalTokenLane === 0) ||
    (io.start.logicalMode === BitNetMode.Prefill &&
      validPrefillLane.asBits.orR)

  when(io.start.fire) {
    running := True
    broadcastDone := False
    allRowsIssued := False
    stickyFault := False
    activationSlot := io.start.activationSlot
    activationEpoch := io.start.activationEpoch
    activationFactor := io.start.activationDequantFactorQ16
    logicalMode := io.start.logicalMode
    finalTokenLane := io.start.finalTokenLane
    activationTileIndex := 0
    cycleCounter := 0
    issuedCount := 0
    acceptedCount := 0
    comparedCount := 0
    nextRowBase := 0
    nextPairOrdinal := 0
    nextRequestId := 0
    for (slot <- 0 until cfg.outstandingRows) {
      tagActive(slot) := False
    }
    when(io.start.activationDequantFactorQ16 === 0 || !startModeLegal) {
      stickyFault := True
    }
  }

  when(running) { cycleCounter := cycleCounter + 1 }

  // Broadcast the resident quantized final activation once, not once per row.
  io.activationTile.valid := running && !broadcastDone
  io.activationTile.logicalMode := logicalMode
  io.activationTile.physicalMode := BitNetMode.Decode
  io.activationTile.sourceTokenLane := finalTokenLane
  io.activationTile.activationSlot := activationSlot
  io.activationTile.activationEpoch := activationEpoch
  io.activationTile.tile := activationTileIndex
  io.activationTile.featureStart :=
    (activationTileIndex.resize(cfg.featureIndexWidth) << 7).resized
  io.activationTile.features := cfg.tileFeatures
  io.activationTile.engineMask := B((BigInt(1) << cfg.engineCount) - 1,
    cfg.engineCount bits)
  io.activationTile.first := activationTileIndex === 0
  io.activationTile.last := activationTileIndex === cfg.tileCount - 1
  when(io.activationTile.fire) {
    when(activationTileIndex === cfg.tileCount - 1) {
      broadcastDone := True
      activationTileIndex := 0
    } otherwise {
      activationTileIndex := activationTileIndex + 1
    }
  }

  private def constantVec(values: Vector[BigInt], width: Int): Vec[UInt] =
    Vec(values.map(value => U(value, width bits)))

  val stripeStarts = constantVec(cfg.stripeInputStarts.map(BigInt(_)),
    cfg.stripeFeatureWidth)
  val stripeFeatures = constantVec(cfg.stripeInputFeatures.map(BigInt(_)),
    cfg.stripeFeatureWidth)
  val weightBases = constantVec(cfg.stripeWeightBases, cfg.addressWidth)
  val scaleStarts = constantVec(cfg.scaleRowStarts.map(BigInt(_)),
    cfg.tokenIdWidth)
  val scaleEnds = constantVec(cfg.scaleRowStarts.zip(cfg.scaleRows).map {
    case (start, rows) => BigInt(start + rows)
  }, cfg.tokenIdWidth + 1)
  val scaleBases = constantVec(cfg.scaleBases, cfg.addressWidth)

  // Constant shift/add address paths keep the row-stride arithmetic out of DSPs.
  def multiplyByConstant(value: UInt, constant: Int, width: Int): UInt = {
    var sum = U(0, width bits)
    for (bit <- 0 until 31 if ((constant >>> bit) & 1) != 0) {
      sum = (sum + (value.resize(width) << bit)).resize(width)
    }
    sum
  }

  val pairEngineActive = Bits(cfg.engineCount bits)
  val pairRows = Vec(UInt(cfg.tokenIdWidth bits), cfg.engineCount)
  val pairRequestIds = Vec(UInt(cfg.requestIdWidth bits), cfg.engineCount)
  val pairScaleBanks = Vec(UInt(cfg.bankWidth bits), cfg.engineCount)
  val pairScaleAddresses = Vec(UInt(cfg.addressWidth bits), cfg.engineCount)
  val pairSlotsFree = Vec(Bool(), cfg.engineCount)
  for (engine <- 0 until cfg.engineCount) {
    val rowWide = nextRowBase.resize(cfg.tokenIdWidth + 1) + engine
    pairRows(engine) := rowWide.resize(cfg.tokenIdWidth)
    pairEngineActive(engine) := rowWide < cfg.vocabularySize
    pairRequestIds(engine) := (nextRequestId + engine).resized

    pairScaleBanks(engine) := 0
    pairScaleAddresses(engine) := 0
    for (bank <- 0 until cfg.bankCount) {
      when(rowWide >= scaleStarts(bank).resize(cfg.tokenIdWidth + 1) &&
          rowWide < scaleEnds(bank)) {
        pairScaleBanks(engine) := bank
        pairScaleAddresses(engine) := (
          scaleBases(bank) +
            ((pairRows(engine) - scaleStarts(bank)).resize(cfg.addressWidth) << 2)
        ).resized
      }
    }
    pairSlotsFree(engine) := !pairEngineActive(engine) ||
      !tagActive(pairRequestIds(engine))
  }
  val pairActiveRows = CountOne(pairEngineActive)

  io.rowPair.valid := running && broadcastDone && !allRowsIssued &&
    pairSlotsFree.asBits.andR
  io.rowPair.pairOrdinal := nextPairOrdinal
  io.rowPair.engineMask := pairEngineActive
  io.rowPair.firstPair := nextRowBase === 0
  io.rowPair.lastPair :=
    nextRowBase.resize(cfg.tokenIdWidth + 1) + cfg.engineCount >=
      cfg.vocabularySize
  for (bank <- 0 until cfg.bankCount) {
    io.rowPair.inputStart(bank) := stripeStarts(bank)
    io.rowPair.inputFeatures(bank) := stripeFeatures(bank)
  }
  for (engine <- 0 until cfg.engineCount) {
    io.rowPair.engine(engine).requestId := pairRequestIds(engine)
    io.rowPair.engine(engine).tokenId := pairRows(engine)
    io.rowPair.engine(engine).scaleBank := pairScaleBanks(engine)
    io.rowPair.engine(engine).scaleAddress := pairScaleAddresses(engine)
    for (bank <- 0 until cfg.bankCount) {
      io.rowPair.engine(engine).weightAddress(bank) := (
        weightBases(bank) + multiplyByConstant(
          pairRows(engine), cfg.stripeInputFeatures(bank), cfg.addressWidth)
      ).resized
    }
  }

  when(io.rowPair.fire) {
    issuedCount := (issuedCount + pairActiveRows.resize(cfg.rowCountWidth)).resized
    nextRequestId :=
      (nextRequestId + pairActiveRows.resize(cfg.requestIdWidth)).resized
    nextPairOrdinal := nextPairOrdinal + 1
    when(io.rowPair.lastPair) {
      allRowsIssued := True
    } otherwise {
      nextRowBase := nextRowBase + cfg.engineCount
    }
    for (engine <- 0 until cfg.engineCount; slot <- 0 until cfg.outstandingRows) {
      when(pairEngineActive(engine) && pairRequestIds(engine) === slot) {
        tagActive(slot) := True
      }
    }
    for (engine <- 0 until cfg.engineCount) {
      val metadata = if (expectedTokenHighWidth > 0) {
        pairRows(engine)(
          cfg.tokenIdWidth - 1 downto cfg.requestIdWidth).asBits ##
          pairScaleBanks(engine).asBits
      } else {
        pairScaleBanks(engine).asBits
      }
      val address = if (cfg.engineCount == 1) pairRequestIds(engine) else
        pairRequestIds(engine)(cfg.requestIdWidth - 1 downto 1)
      if (engine == 0) {
        when(pairEngineActive(engine)) {
          expectedMetadataEven(address) := metadata
        }
      } else {
        when(pairEngineActive(engine)) {
          expectedMetadataOdd(address) := metadata
        }
      }
    }
  }

  // Accept row completions in any order.  requestId is a bounded scoreboard
  // tag; corrupt metadata is replaced by the expected tag so recovery cannot
  // silently choose a wrong vocabulary row.
  val completionMetadataAddress = if (cfg.engineCount == 1)
    io.rowCompletion.requestId else
    io.rowCompletion.requestId(cfg.requestIdWidth - 1 downto 1)
  val completionMetadataEven = expectedMetadataEven(completionMetadataAddress)
  val completionMetadataOdd = expectedMetadataOdd(completionMetadataAddress)
  val completionExpectedMetadata = if (cfg.engineCount == 1)
    completionMetadataEven else Mux(
      io.rowCompletion.requestId(0),
      completionMetadataOdd,
      completionMetadataEven)
  val completionExpectedScaleBank = completionExpectedMetadata(
    cfg.bankWidth - 1 downto 0).asUInt
  val completionExpectedToken = UInt(cfg.tokenIdWidth bits)
  if (expectedTokenHighWidth > 0) {
    val completionExpectedTokenHigh = completionExpectedMetadata(
      expectedMetadataWidth - 1 downto cfg.bankWidth).asUInt
    completionExpectedToken := (
      completionExpectedTokenHigh.asBits ##
        io.rowCompletion.requestId.asBits).asUInt
  } else {
    completionExpectedToken := io.rowCompletion.requestId.resized
  }
  val completionExpectedEngine =
    io.rowCompletion.requestId(0).asUInt.resize(cfg.engineWidth)
  val completionSlotKnown = running && tagActive(io.rowCompletion.requestId)
  val completionMetadataGood =
    io.rowCompletion.tokenId === completionExpectedToken &&
      io.rowCompletion.scaleBank === completionExpectedScaleBank &&
      io.rowCompletion.engine === completionExpectedEngine
  val completionFramingGood =
    io.rowCompletion.tilesSeen === cfg.tileCount &&
      io.rowCompletion.bankMask ===
        B((BigInt(1) << cfg.bankCount) - 1, cfg.bankCount bits) &&
      io.rowCompletion.firstSeen && io.rowCompletion.lastSeen

  candidateFifo.io.push.valid := io.rowCompletion.valid && completionSlotKnown
  candidateFifo.io.push.tokenId := completionExpectedToken
  candidateFifo.io.push.bank := completionExpectedScaleBank
  candidateFifo.io.push.dot := io.rowCompletion.dot
  candidateFifo.io.push.rowScaleF32 := io.rowCompletion.rowScaleF32
  candidateFifo.io.push.framingFault :=
    !completionMetadataGood || !completionFramingGood
  io.rowCompletion.ready := Mux(completionSlotKnown,
    candidateFifo.io.push.ready, True)

  val completionFireKnown = io.rowCompletion.fire && completionSlotKnown
  when(io.rowCompletion.fire && !completionSlotKnown) {
    stickyFault := True
  }
  when(completionFireKnown) {
    acceptedCount := acceptedCount + 1
    when(!completionMetadataGood || !completionFramingGood) {
      stickyFault := True
    }
    for (slot <- 0 until cfg.outstandingRows) {
      when(io.rowCompletion.requestId === slot) { tagActive(slot) := False }
    }
  }

  // Exact shared row scaler.  A 24-cycle shift/add multiply is faster than the
  // aggregate 5-bank DDR row rate (one 2560-byte row in about 32 clocks).
  val candidateToken = Reg(UInt(cfg.tokenIdWidth bits)) init (0)
  val candidateBank = Reg(UInt(cfg.bankWidth bits)) init (0)
  val candidateDot = Reg(SInt(32 bits)) init (0)
  val candidateScale = Reg(Bits(32 bits)) init (0)
  val candidateScaleValid = RegInit(False)
  val candidateNegative = RegInit(False)
  val candidateZero = RegInit(False)
  val candidateBaseExponent = Reg(SInt(12 bits)) init (0)

  val multiplyMultiplier = Reg(UInt(24 bits)) init (0)
  val multiplyMultiplicand = Reg(UInt(56 bits)) init (0)
  val multiplyProduct = Reg(UInt(56 bits)) init (0)
  val multiplyCycle = Reg(UInt(5 bits)) init (0)

  val incomingExp = candidateFifo.io.pop.rowScaleF32(30 downto 23).asUInt
  val incomingFraction = candidateFifo.io.pop.rowScaleF32(22 downto 0).asUInt
  val incomingSignificand = UInt(24 bits)
  incomingSignificand := (U(1, 1 bits) ## incomingFraction).asUInt
  when(incomingExp === 0) {
    incomingSignificand := (U(0, 1 bits) ## incomingFraction).asUInt
  }
  val incomingScaleValid =
    !candidateFifo.io.pop.rowScaleF32.msb && incomingExp =/= 255 &&
      incomingSignificand =/= 0
  val incomingDotWide = candidateFifo.io.pop.dot.resize(33)
  val incomingMagnitude = UInt(32 bits)
  incomingMagnitude := candidateFifo.io.pop.dot.asUInt
  when(candidateFifo.io.pop.dot.msb) {
    incomingMagnitude := (-incomingDotWide).asUInt.resize(32)
  }
  val incomingBaseExponent = SInt(12 bits)
  incomingBaseExponent := incomingExp.resize(12).asSInt - S(150, 12 bits)
  when(incomingExp === 0) { incomingBaseExponent := S(-149, 12 bits) }

  candidateFifo.io.pop.ready := running && !multiplierBusy
  when(candidateFifo.io.pop.fire) {
    multiplierBusy := True
    multiplyCycle := 0
    // Iterate over the 24-bit F32 significand, not the 32-bit dot magnitude.
    // The exact product is unchanged and a new row can enter every 25 clocks.
    multiplyMultiplier := incomingSignificand
    multiplyMultiplicand := incomingMagnitude.resize(56)
    multiplyProduct := 0
    candidateToken := candidateFifo.io.pop.tokenId
    candidateBank := candidateFifo.io.pop.bank
    candidateDot := candidateFifo.io.pop.dot
    candidateScale := candidateFifo.io.pop.rowScaleF32
    candidateScaleValid := incomingScaleValid
    candidateNegative := candidateFifo.io.pop.dot.msb &&
      candidateFifo.io.pop.dot =/= 0
    candidateZero := candidateFifo.io.pop.dot === 0
    candidateBaseExponent := incomingBaseExponent
    when(candidateFifo.io.pop.framingFault || !incomingScaleValid) {
      stickyFault := True
    }
  }

  val addProduct = UInt(56 bits)
  addProduct := multiplyProduct
  when(multiplyMultiplier(0)) {
    addProduct := (multiplyProduct + multiplyMultiplicand).resized
  }

  // Break the exact comparison into product, normalize and compare stages.
  // This keeps the priority encoder/barrel shifter and the 56-bit comparator
  // off the shift/add carry path.
  val productStageValid = RegInit(False)
  val productStageValue = Reg(UInt(56 bits)) init (0)
  val productStageToken = Reg(UInt(cfg.tokenIdWidth bits)) init (0)
  val productStageBank = Reg(UInt(cfg.bankWidth bits)) init (0)
  val productStageDot = Reg(SInt(32 bits)) init (0)
  val productStageScale = Reg(Bits(32 bits)) init (0)
  val productStageScaleValid = RegInit(False)
  val productStageNegative = RegInit(False)
  val productStageZero = RegInit(False)
  val productStageBaseExponent = Reg(SInt(12 bits)) init (0)

  val productLead = UInt(6 bits)
  productLead := 0
  for (bit <- 0 until 56) {
    when(productStageValue(bit)) { productLead := bit }
  }
  val normalizeShift = (U(55, 6 bits) - productLead).resized
  val normalizedValue =
    (productStageValue |<< normalizeShift).resize(56)
  val normalizedTopExponent =
    (productStageBaseExponent + productLead.resize(12).asSInt).resize(12)

  val canonicalValid = RegInit(False)
  val canonicalScaleValid = RegInit(False)
  val canonicalNegative = RegInit(False)
  val canonicalZero = RegInit(False)
  val canonicalTopExponent = Reg(SInt(12 bits)) init (0)
  val canonicalNormalized = Reg(UInt(56 bits)) init (0)
  val canonicalToken = Reg(UInt(cfg.tokenIdWidth bits)) init (0)
  val canonicalDot = Reg(SInt(32 bits)) init (0)
  val canonicalScale = Reg(Bits(32 bits)) init (0)
  val canonicalBank = Reg(UInt(cfg.bankWidth bits)) init (0)

  productStageValid := False
  canonicalValid := False
  when(productStageValid) {
    canonicalValid := True
    canonicalScaleValid := productStageScaleValid
    canonicalNegative := productStageNegative
    canonicalZero := productStageZero
    canonicalTopExponent := normalizedTopExponent
    canonicalNormalized := normalizedValue
    canonicalToken := productStageToken
    canonicalDot := productStageDot
    canonicalScale := productStageScale
    canonicalBank := productStageBank
  }

  val bestValid = RegInit(False)
  val bestNegative = RegInit(False)
  val bestZero = RegInit(False)
  val bestTopExponent = Reg(SInt(12 bits)) init (0)
  val bestNormalized = Reg(UInt(56 bits)) init (0)
  val bestToken = Reg(UInt(cfg.tokenIdWidth bits)) init (0)
  val bestDot = Reg(SInt(32 bits)) init (0)
  val bestScale = Reg(Bits(32 bits)) init (0)
  val bestBank = Reg(UInt(cfg.bankWidth bits)) init (0)

  when(io.start.fire) {
    bestValid := False
    bestNegative := False
    bestZero := False
    bestTopExponent := 0
    bestNormalized := 0
    bestToken := 0
    bestDot := 0
    bestScale := 0
    bestBank := 0
  }

  val candidateNumericGreater = Bool()
  val candidateNumericEqual = Bool()
  candidateNumericGreater := False
  candidateNumericEqual := False
  when(canonicalZero && bestZero) {
    candidateNumericEqual := True
  } elsewhen (!canonicalNegative && !canonicalZero && bestNegative) {
    candidateNumericGreater := True
  } elsewhen (!canonicalNegative && !canonicalZero && bestZero) {
    candidateNumericGreater := True
  } elsewhen (canonicalZero && bestNegative) {
    candidateNumericGreater := True
  } elsewhen (canonicalNegative === bestNegative &&
      canonicalZero === bestZero) {
    when(canonicalTopExponent === bestTopExponent &&
        canonicalNormalized === bestNormalized) {
      candidateNumericEqual := True
    } elsewhen (!canonicalNegative) {
      candidateNumericGreater := canonicalTopExponent > bestTopExponent ||
        (canonicalTopExponent === bestTopExponent &&
          canonicalNormalized > bestNormalized)
    } otherwise {
      // Both are negative: the smaller exact magnitude is the larger score.
      candidateNumericGreater := canonicalTopExponent < bestTopExponent ||
        (canonicalTopExponent === bestTopExponent &&
          canonicalNormalized < bestNormalized)
    }
  }

  val takeCandidate = canonicalScaleValid &&
    (!bestValid || candidateNumericGreater ||
      (candidateNumericEqual && canonicalToken < bestToken))
  val finalBestValid = bestValid || canonicalScaleValid
  val selectedBestToken = Mux(takeCandidate, canonicalToken, bestToken)
  val selectedBestDot = Mux(takeCandidate, canonicalDot, bestDot)
  val selectedBestScale = Mux(takeCandidate, canonicalScale, bestScale)
  val selectedBestBank = Mux(takeCandidate, canonicalBank, bestBank)

  when(multiplierBusy) {
    multiplyProduct := addProduct
    multiplyMultiplier := (multiplyMultiplier |>> 1).resized
    multiplyMultiplicand := (multiplyMultiplicand |<< 1).resized
    when(multiplyCycle === 23) {
      multiplierBusy := False
      productStageValid := True
      productStageValue := addProduct
      productStageToken := candidateToken
      productStageBank := candidateBank
      productStageDot := candidateDot
      productStageScale := candidateScale
      productStageScaleValid := candidateScaleValid
      productStageNegative := candidateNegative
      productStageZero := candidateZero
      productStageBaseExponent := candidateBaseExponent
    } otherwise {
      multiplyCycle := multiplyCycle + 1
    }
  }

  when(canonicalValid) {
    comparedCount := comparedCount + 1
      when(takeCandidate) {
        bestValid := True
        bestNegative := canonicalNegative
        bestZero := canonicalZero
        bestTopExponent := canonicalTopExponent
        bestNormalized := canonicalNormalized
        bestToken := canonicalToken
        bestDot := canonicalDot
        bestScale := canonicalScale
        bestBank := canonicalBank
      }

      when(comparedCount === cfg.vocabularySize - 1) {
        running := False
        resultValid := True
        result.tokenId := selectedBestToken
        result.dot := selectedBestDot
        result.rowScaleF32 := selectedBestScale
        result.bank := selectedBestBank
        result.activationDequantFactorQ16 := activationFactor
        result.rowsCompared := comparedCount + 1
        result.cycles := cycleCounter
        result.error := stickyFault || !canonicalScaleValid ||
          !finalBestValid || !allRowsIssued
      }
  }

  val outstandingCount = CountOne(tagActive.asBits)
  candidateFifo.io.flush := io.abort &&
    (running || resultValid || multiplierBusy || productStageValid ||
      canonicalValid || candidateFifo.io.occupancy =/= 0)

  // Abort is not a protocol fault.  It discards only the resident reducer's
  // bounded bookkeeping after the wrapper has drained any already-visible
  // rowPair.  Late external result/completion streams are handled by the
  // resident coordinator, not by keeping this scoreboard alive forever.
  when(io.abort && (running || resultValid || multiplierBusy ||
      productStageValid || canonicalValid ||
      candidateFifo.io.occupancy =/= 0)) {
    running := False
    broadcastDone := False
    allRowsIssued := False
    resultValid := False
    multiplierBusy := False
    productStageValid := False
    canonicalValid := False
    for (slot <- 0 until cfg.outstandingRows) {
      tagActive(slot) := False
    }
  }
  io.busy := running || resultValid || multiplierBusy || productStageValid ||
    canonicalValid ||
    candidateFifo.io.occupancy =/= 0
  io.protocolFault := stickyFault
  io.rowsIssued := issuedCount
  io.rowsAccepted := acceptedCount
  io.rowsCompared := comparedCount
  io.outstanding := outstandingCount.resized
}

/** Exact Scala oracle used by random and IEEE-754 edge-case tests. */
object BitNetStreamingLmHeadMath {
  case class ExactScaled(dot: Int, scaleRaw: Int) {
    private val exponentField = (scaleRaw >>> 23) & 0xff
    private val fraction = scaleRaw & 0x7fffff
    val valid: Boolean = (scaleRaw & 0x80000000) == 0 &&
      exponentField != 0xff && (exponentField != 0 || fraction != 0)
    val significand: BigInt =
      if (exponentField == 0) BigInt(fraction)
      else BigInt((1 << 23) | fraction)
    val exponent: Int = if (exponentField == 0) -149 else exponentField - 150
    val signedMagnitude: BigInt = BigInt(dot) * significand
  }

  /** Returns -1/0/+1 exactly, without converting either score to Float. */
  def compare(left: ExactScaled, right: ExactScaled): Int = {
    require(left.valid && right.valid)
    val commonExponent = scala.math.min(left.exponent, right.exponent)
    val l = left.signedMagnitude << (left.exponent - commonExponent)
    val r = right.signedMagnitude << (right.exponent - commonExponent)
    l.compare(r)
  }

  def winner(rows: Seq[(Int, Int, Int)]): (Int, Int, Int) = {
    require(rows.nonEmpty)
    rows.filter { case (_, dot, scale) => ExactScaled(dot, scale).valid }
      .reduceLeft { (best, candidate) =>
        val cmp = compare(
          ExactScaled(candidate._2, candidate._3),
          ExactScaled(best._2, best._3))
        if (cmp > 0 || (cmp == 0 && candidate._1 < best._1)) candidate
        else best
      }
  }
}

object GenerateBitNetStreamingLmHead extends App {
  SpinalConfig(
    targetDirectory = "target/generated-bitnet-streaming-lm-head",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetStreamingLmHead())
}
