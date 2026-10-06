package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentRopeCoefficientFault {
  val None = 0x00
  val InvalidRound = 0xb0
  val ReadProtocol = 0xb1
  val ProjectionMetadata = 0xb2
  val ProjectionOrder = 0xb3
  val UnexpectedReadResponse = 0xb4
}

case class BitNetResidentRopeCoefficientStagerConfig(
    event: BitNetResidentLayerEventSchedulerConfig =
      BitNetResidentLayerEventSchedulerConfig(epochWidth = 8),
    bridge: BitNetSegmentedQkvProductionBridgeConfig =
      BitNetSegmentedQkvProductionBridgeConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    layout: BitNetFiveBankStripedLayout =
      BitNetFiveBankStripedLayout(),
    bank: Int = 0
) {
  val model: BitNetConfig = event.model
  val stage: BitNetResidentMatrixStageLaunchExpanderConfig = bridge.stage
  val coefficient: BitNetQkvClusterResultAssemblerConfig =
    bridge.coefficient

  val ropeAssetBase: BigInt = layout.ropeBases(bank)
  val ropeHeaderBytes: Int =
    BitNetAddressGeometry.RopeHeaderBytes.toInt
  val ropeTableBase: BigInt = ropeAssetBase + ropeHeaderBytes
  val rowBytes: Int = BitNetAddressGeometry.ropeRowBytes(model).toInt
  val rowBeats: Int = rowBytes / (crossbar.dataWidth / 8)
  val pairsPerRow: Int = model.headSize / 2
  val wordsPerBeat: Int = crossbar.dataWidth / 32
  val totalCoefficients: Int = bridge.totalFeatures

  require(model == BitNetConfig.Production)
  require(stage.model == model && bridge.model == model)
  require(layout.cfg == model)
  layout.validate()
  require(bank >= 0 && bank < crossbar.bankCount)
  require(crossbar.dataWidth == 128)
  require(ropeHeaderBytes == 48,
    "the deployed BPR1/Q15 asset has a fixed 48-byte header")
  require(rowBytes == pairsPerRow * 4 && rowBytes == 256,
    "one deployed RoPE row is 64 little-endian <cos:i16,sin:i16> words")
  require(rowBytes % (crossbar.dataWidth / 8) == 0)
  require(rowBeats > 0 && rowBeats <= crossbar.maxBurstBeats)
  require(isPow2(rowBeats) && wordsPerBeat == 4)
  require((ropeTableBase & (crossbar.dataWidth / 8 - 1)) == 0,
    "the first coefficient row must be aligned to one memory beat")
  require(BitNetAddressGeometry.RopeHeaderBytes +
    BigInt(model.maxSequenceLength) * rowBytes ==
      BitNetFiveBankStripedLayout.RopeBytes,
    "replicated RoPE allocation must contain exactly header plus all rows")
  require(ropeAssetBase >= 0 &&
    ropeAssetBase + BitNetFiveBankStripedLayout.RopeBytes <=
      (BigInt(1) << crossbar.localAddressWidth))
  require(totalCoefficients == model.hiddenSize +
    2 * model.kvHeadCount * model.headSize)
}

/**
  * Terminal status of one atomic resident QKV coefficient transaction.
  * This is deliberately not a LayerStep completion: the matrix bridge and the
  * resident Q/K/V/KV-cache path still own the architectural layer event.
  */
case class BitNetResidentRopeCoefficientCompletion(
    cfg: BitNetResidentRopeCoefficientStagerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val epoch = UInt(cfg.event.epochWidth bits)
  val mode = BitNetMode()
  val positionBase = UInt(cfg.model.sequenceWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
  val emittedCoefficients = UInt(32 bits)
}

object BitNetResidentRopeCoefficientStagerOracle {
  final case class Row(address: BigInt, bytes: Int, beats: Int)

  def row(cfg: BitNetResidentRopeCoefficientStagerConfig,
      position: Int): Row = {
    require(position >= 0 && position < cfg.model.maxSequenceLength)
    Row(cfg.ropeTableBase + BigInt(position) * cfg.rowBytes,
      cfg.rowBytes, cfg.rowBeats)
  }

  def pairForLocalFeature(localFeature: Int, headSize: Int = 128): Int = {
    require(localFeature >= 0 && headSize > 0 && isPow2(headSize))
    (localFeature % headSize) % (headSize / 2)
  }

  /** Deployed word format, matching export_emmc_package.py struct <hh>. */
  def unpackQ15(word: Long): (Short, Short) =
    ((word & 0xffffL).toShort, ((word >>> 16) & 0xffffL).toShort)
}

/**
  * PL-resident producer for the coefficient stream consumed by the segmented
  * Q/K/V bridge.
  *
  * One lowered `QkvResidentReady` command atomically authenticates the whole
  * Q/K/V round.  Before any projection is forwarded, this component fetches
  * each active absolute-position row exactly once from one of the five
  * replicated RoPE images and caches at most four 64-word Q1.15 rows.  Query
  * and Key then replay those rows for every head in natural feature order.
  * Value never performs a DDR read and emits the Q1.15 identity coefficient
  * solely because the existing bridge uses one lossless coefficient token as
  * its per-feature rendezvous.
  *
  * Projection sequence identities are never regenerated here.  The actual
  * typed projection launch is authenticated, forwarded unchanged, and its
  * exact `sequenceId` is echoed on every corresponding coefficient.  This
  * prevents a second copy of MatrixStage's sequence-id formula from silently
  * diverging.
  *
  * The memory boundary is a normal bank-local crossbar client.  Addresses are
  * derived from the validated five-bank layout: `ropeBase + 48 + position *
  * 256`.  Every transaction/tag/beat/last/RRESP field is checked.  Any fault
  * suppresses further coefficients and reports failure; a surrounding owner
  * must poison an already-forwarded bridge transaction rather than treating a
  * partial stream as valid QKV.
  */
class BitNetResidentRopeCoefficientStager(
    cfg: BitNetResidentRopeCoefficientStagerConfig =
      BitNetResidentRopeCoefficientStagerConfig()) extends Component {
  private val model = cfg.model
  private val stage = cfg.stage
  private val matrix = stage.matrix
  private val b = matrix.base
  private val lanes = model.tokenParallelism

  val io = new Bundle {
    val round = slave(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg.event)))

    /** In-series identity seam from MatrixStage to the segmented bridge. */
    val projection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(stage)))
    val projectionForward = master(Stream(
      BitNetResidentMatrixProjectionLaunch(stage)))

    val readRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar)))
    val readResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar)))

    val coefficient = master(Stream(
      BitNetQkvFeatureRopeCoefficient(cfg.coefficient)))
    val completion = master(Stream(
      BitNetResidentRopeCoefficientCompletion(cfg)))

    val busy = out Bool()
    val cacheReady = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedRounds = out UInt(32 bits)
    val completedRounds = out UInt(32 bits)
    val fetchedRows = out UInt(32 bits)
    val emittedCoefficients = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, SelectRow, IssueRead, ReceiveRead, WaitProjection,
        EmitCoefficient, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val heldRound = Reg(
    BitNetResidentLayerEventExecutorCommand(cfg.event)) init
      BitNetResidentLayerEventExecutorCommand(cfg.event).getZero
  val heldProjection = Reg(
    BitNetResidentMatrixProjectionLaunch(stage)) init
      BitNetResidentMatrixProjectionLaunch(stage).getZero

  val fetchLane = Reg(UInt(log2Up(lanes) bits)) init 0
  val cachedLaneMask = Reg(Bits(lanes bits)) init 0
  val expectedBeat = Reg(UInt(cfg.crossbar.beatIndexWidth bits)) init 0
  val activeTransactionId = Reg(
    UInt(cfg.crossbar.transactionIdWidth bits)) init 0
  val activeTag = Reg(UInt(cfg.crossbar.tagWidth bits)) init 0
  val expectedPartOrdinal = Reg(UInt(stage.partOrdinalWidth bits)) init 0
  val localFeature = Reg(UInt(b.outputIndexWidth bits)) init 0
  val partSequence = Vec(
    Reg(UInt(matrix.sequenceIdWidth bits)) init 0, 3)
  val seenPartSequence = Reg(Bits(3 bits)) init 0

  val jobFault = RegInit(False)
  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)
  val roundEmitted = Reg(UInt(32 bits)) init 0
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0
  val fetchedRowCount = Reg(UInt(32 bits)) init 0
  val emittedCount = Reg(UInt(32 bits)) init 0

  // One independent 2-Kibit row RAM per token lane gives four coefficient
  // reads per cycle while retaining one simple write port per fetched row.
  val rowCache = Array.fill(lanes)(
    Mem(Bits(cfg.crossbar.dataWidth bits), cfg.rowBeats))

  io.busy := state =/= State.Idle
  io.cacheReady := state === State.WaitProjection ||
    state === State.EmitCoefficient || state === State.Report
  io.protocolFaultSticky := stickyFault
  io.acceptedRounds := acceptedCount
  io.completedRounds := completedCount
  io.fetchedRows := fetchedRowCount
  io.emittedCoefficients := emittedCount

  private def setFault(code: Int): Unit = {
    when(!jobFault) { jobFaultCode := B(code, 8 bits) }
    jobFault := True
    stickyFault := True
  }

  val sourceIdentityLegal =
    io.round.event.requestId === io.round.step.requestId &&
      io.round.event.layer === io.round.step.layer &&
      io.round.event.mode === io.round.step.mode &&
      io.round.event.tokenBase === io.round.step.tokenBase &&
      io.round.event.tokenCount === io.round.step.tokenCount &&
      io.round.event.positionBase === io.round.step.positionBase &&
      io.round.event.activeTokenMask === io.round.step.activeTokenMask &&
      io.round.event.resetKv === io.round.step.resetKv
  val sourceCausalLegal = Bool()
  sourceCausalLegal := True
  for (lane <- 0 until lanes) {
    when(io.round.event.laneCausalLength(lane) =/=
        io.round.step.laneCausalLength(lane)) {
      sourceCausalLegal := False
    }
    val expectedCausal = UInt(model.sequenceWidth bits)
    expectedCausal := 0
    when(io.round.event.activeTokenMask(lane)) {
      expectedCausal := (io.round.event.positionBase + lane + 1).resized
    }
    when(io.round.event.laneCausalLength(lane) =/= expectedCausal) {
      sourceCausalLegal := False
    }
  }

  val expectedStepId = ((io.round.event.layer.resize(
    BitNetResidentLayerGeometry.stepIdWidth(model)) << 3) +
    U(BitNetResidentLayerEventStage.LayerOrder.indexOf(
      BitNetResidentLayerEventStage.QkvResidentReady),
      BitNetResidentLayerGeometry.stepIdWidth(model) bits)).resized
  val expectedMask = Bits(lanes bits)
  for (lane <- 0 until lanes) {
    expectedMask(lane) := io.round.event.tokenCount > lane
  }
  val positionEnd = io.round.event.positionBase.resize(
    model.sequenceWidth + 1) +
    io.round.event.tokenCount.resize(model.sequenceWidth + 1)
  val modeShapeLegal =
    (io.round.event.mode === BitNetMode.Prefill) ||
      (io.round.event.mode === BitNetMode.Decode &&
        io.round.event.tokenCount === 1 &&
        io.round.event.activeTokenMask === B(1, lanes bits))
  val roundLegal =
    io.round.event.stage ===
      BitNetResidentLayerEventStage.QkvResidentReady &&
      io.round.executor === BitNetResidentLayerEventExecutor.MatrixArray &&
      io.round.step.stage === BitNetResidentLayerStage.QkvProjection &&
      io.round.step.unit === BitNetResidentLayerUnit.MatrixArray &&
      io.round.step.layer < model.layerCount &&
      io.round.step.stepId === expectedStepId &&
      io.round.event.tokenCount =/= 0 &&
      io.round.event.tokenCount <= lanes &&
      io.round.event.activeTokenMask === expectedMask &&
      io.round.event.tokenBase === io.round.event.positionBase &&
      positionEnd <= model.maxSequenceLength &&
      modeShapeLegal && sourceIdentityLegal && sourceCausalLegal

  io.round.ready := state === State.Idle
  when(io.round.fire) {
    heldRound := io.round.payload
    fetchLane := 0
    cachedLaneMask := 0
    expectedPartOrdinal := 0
    localFeature := 0
    seenPartSequence := 0
    roundEmitted := 0
    jobFault := False
    jobFaultCode := 0
    acceptedCount := acceptedCount + 1
    when(roundLegal) {
      state := State.SelectRow
    } otherwise {
      jobFault := True
      jobFaultCode := B(
        BitNetResidentRopeCoefficientFault.InvalidRound, 8 bits)
      stickyFault := True
      state := State.Report
    }
  }

  when(state === State.SelectRow) {
    when(heldRound.step.activeTokenMask(fetchLane)) {
      state := State.IssueRead
    } elsewhen(fetchLane === lanes - 1) {
      state := State.WaitProjection
    } otherwise {
      fetchLane := fetchLane + 1
    }
  }

  val fetchPosition = (heldRound.step.positionBase.resize(
    model.sequenceWidth) + fetchLane.resize(model.sequenceWidth)).resized
  val fetchAddress = (U(cfg.ropeTableBase,
    cfg.crossbar.localAddressWidth bits) +
    fetchPosition.resize(cfg.crossbar.localAddressWidth) * cfg.rowBytes).resized
  val generatedTransactionId =
    (heldRound.step.requestId.resize(cfg.crossbar.transactionIdWidth) ^
      heldRound.step.stepId.resize(cfg.crossbar.transactionIdWidth) ^
      fetchLane.resize(cfg.crossbar.transactionIdWidth)).resized
  val generatedTag =
    (heldRound.step.requestId.resize(cfg.crossbar.tagWidth) ^
      heldRound.event.epoch.resize(cfg.crossbar.tagWidth) ^
      heldRound.step.stepId.resize(cfg.crossbar.tagWidth) ^
      fetchPosition.resize(cfg.crossbar.tagWidth)).resized

  io.readRequest.valid := state === State.IssueRead
  io.readRequest.bank := cfg.bank
  io.readRequest.transactionId := generatedTransactionId
  io.readRequest.tag := generatedTag
  io.readRequest.localAddress := fetchAddress
  io.readRequest.beats := cfg.rowBeats
  when(io.readRequest.fire) {
    activeTransactionId := io.readRequest.transactionId
    activeTag := io.readRequest.tag
    expectedBeat := 0
    state := State.ReceiveRead
  }

  // Outside ReceiveRead, accepting a response avoids wedging the shared
  // crossbar.  During an active round it poisons the transaction immediately.
  io.readResponse.ready := True
  val responseExpectedLast = expectedBeat === cfg.rowBeats - 1
  val responseLegal =
    io.readResponse.bank === cfg.bank &&
      io.readResponse.transactionId === activeTransactionId &&
      io.readResponse.tag === activeTag &&
      io.readResponse.beatIndex === expectedBeat &&
      io.readResponse.last === responseExpectedLast &&
      !io.readResponse.fault && io.readResponse.rresp === B"2'b00"

  when(io.readResponse.fire && state === State.ReceiveRead) {
    when(responseLegal) {
      for (lane <- 0 until lanes) {
        when(fetchLane === lane) {
          rowCache(lane).write(expectedBeat.resized,
            io.readResponse.data)
        }
      }
    } otherwise {
      setFault(BitNetResidentRopeCoefficientFault.ReadProtocol)
    }

    // Early last and missing last on the declared final beat both terminate
    // locally as a failed burst; neither condition can wait forever for a beat
    // that the malformed producer may never send.
    when(!responseLegal || io.readResponse.last || responseExpectedLast) {
      when(responseLegal) {
        cachedLaneMask(fetchLane) := True
        fetchedRowCount := fetchedRowCount + 1
        when(fetchLane === lanes - 1) {
          state := State.WaitProjection
        } otherwise {
          fetchLane := fetchLane + 1
          state := State.SelectRow
        }
      } otherwise {
        state := State.Report
      }
    } otherwise {
      expectedBeat := expectedBeat + 1
    }
  }

  when(io.readResponse.fire && state =/= State.ReceiveRead) {
    stickyFault := True
    when(state =/= State.Idle && state =/= State.Report) {
      setFault(
        BitNetResidentRopeCoefficientFault.UnexpectedReadResponse)
      state := State.Report
    }
  }

  val expectedPart = BitNetResidentMatrixProjectionPart()
  val expectedWeight = BitNetAddressTarget()
  val expectedScale = BitNetAddressTarget()
  val expectedOutputFeatures = UInt(b.outputIndexWidth bits)
  val expectedGlobalBase = UInt(b.outputIndexWidth bits)
  expectedPart := BitNetResidentMatrixProjectionPart.Query
  expectedWeight := BitNetAddressTarget.QWeight
  expectedScale := BitNetAddressTarget.QProjectionScale
  expectedOutputFeatures := model.hiddenSize
  expectedGlobalBase := 0
  when(expectedPartOrdinal === 1) {
    expectedPart := BitNetResidentMatrixProjectionPart.Key
    expectedWeight := BitNetAddressTarget.KWeight
    expectedScale := BitNetAddressTarget.KProjectionScale
    expectedOutputFeatures := cfg.bridge.kvFeatures
    expectedGlobalBase := model.hiddenSize
  } elsewhen(expectedPartOrdinal === 2) {
    expectedPart := BitNetResidentMatrixProjectionPart.Value
    expectedWeight := BitNetAddressTarget.VWeight
    expectedScale := BitNetAddressTarget.VProjectionScale
    expectedOutputFeatures := cfg.bridge.kvFeatures
    expectedGlobalBase := model.hiddenSize + cfg.bridge.kvFeatures
  }

  val projectionIdentityLegal =
    io.projection.requestId === heldRound.step.requestId &&
      io.projection.stepId === heldRound.step.stepId &&
      io.projection.payload.stage ===
        BitNetResidentLayerStage.QkvProjection &&
      io.projection.layer === heldRound.step.layer &&
      io.projection.mode === heldRound.step.mode &&
      io.projection.tokenBase === heldRound.step.tokenBase &&
      io.projection.tokenMask === heldRound.step.activeTokenMask &&
      io.projection.activationReuseTag === heldRound.step.stepId &&
      io.projection.matrixTask.scheduleMode === heldRound.step.mode &&
      io.projection.matrixTask.sequenceId === io.projection.sequenceId &&
      io.projection.matrixTask.activeTokenMask ===
        heldRound.step.activeTokenMask
  val projectionOrderLegal =
    io.projection.part === expectedPart &&
      io.projection.partOrdinal === expectedPartOrdinal &&
      io.projection.partCount === 3 &&
      io.projection.firstPart === (expectedPartOrdinal === 0) &&
      io.projection.lastPart === (expectedPartOrdinal === 2)
  val projectionShapeLegal =
    io.projection.kind === BitNetResidentLinearKind.Qkv &&
      io.projection.weightTarget === expectedWeight &&
      io.projection.scaleTarget === expectedScale &&
      !io.projection.hasSecondaryScale &&
      io.projection.inputFeatures === model.hiddenSize &&
      io.projection.outputFeatures === expectedOutputFeatures &&
      io.projection.globalOutputBase === expectedGlobalBase &&
      !io.projection.directLinearCompatible &&
      !io.projection.directStageCompletionCompatible
  val sequenceFresh = Bool()
  sequenceFresh := True
  when(expectedPartOrdinal === 1) {
    sequenceFresh := io.projection.sequenceId =/= partSequence(0)
  } elsewhen(expectedPartOrdinal === 2) {
    sequenceFresh := io.projection.sequenceId =/= partSequence(0) &&
      io.projection.sequenceId =/= partSequence(1)
  }
  val projectionLegal = projectionIdentityLegal &&
    projectionOrderLegal && projectionShapeLegal && sequenceFresh &&
    cachedLaneMask === heldRound.step.activeTokenMask

  io.projectionForward.valid := state === State.WaitProjection &&
    io.projection.valid && projectionLegal
  io.projectionForward.payload := io.projection.payload
  io.projection.ready := state === State.WaitProjection &&
    Mux(projectionLegal, io.projectionForward.ready, True)

  when(io.projection.fire && state === State.WaitProjection) {
    when(projectionLegal) {
      heldProjection := io.projection.payload
      partSequence(expectedPartOrdinal.resized) := io.projection.sequenceId
      seenPartSequence(expectedPartOrdinal.resized) := True
      localFeature := 0
      state := State.EmitCoefficient
    } otherwise {
      when(!projectionOrderLegal) {
        setFault(BitNetResidentRopeCoefficientFault.ProjectionOrder)
      } otherwise {
        setFault(BitNetResidentRopeCoefficientFault.ProjectionMetadata)
      }
      state := State.Report
    }
  }

  val pairIndex = localFeature(log2Up(cfg.pairsPerRow) - 1 downto 0)
  val rowBeatIndex = pairIndex(log2Up(cfg.pairsPerRow) - 1 downto 2)
  val wordIndex = pairIndex(1 downto 0)
  val cachedWords = Vec(Bits(32 bits), lanes)
  for (lane <- 0 until lanes) {
    val cachedBeat = rowCache(lane).readAsync(rowBeatIndex.resized)
    cachedWords(lane) := cachedBeat.subdivideIn(32 bits)(wordIndex)
  }

  val valuePart = heldProjection.part ===
    BitNetResidentMatrixProjectionPart.Value
  io.coefficient.valid := state === State.EmitCoefficient && !jobFault
  io.coefficient.sequenceId := heldProjection.sequenceId
  io.coefficient.globalFeature :=
    (heldProjection.globalOutputBase + localFeature).resized
  for (lane <- 0 until lanes) {
    io.coefficient.cosine(lane) := 0
    io.coefficient.sine(lane) := 0
    when(heldRound.step.activeTokenMask(lane)) {
      when(valuePart) {
        // 32767 is the exported/clamped Q1.15 representation of +1.0.
        io.coefficient.cosine(lane) := S(32767, 16 bits)
      } otherwise {
        io.coefficient.cosine(lane) :=
          cachedWords(lane)(15 downto 0).asSInt
        io.coefficient.sine(lane) :=
          cachedWords(lane)(31 downto 16).asSInt
      }
    }
  }

  when(io.coefficient.fire) {
    roundEmitted := roundEmitted + 1
    emittedCount := emittedCount + 1
    when(localFeature === heldProjection.outputFeatures - 1) {
      localFeature := 0
      when(expectedPartOrdinal === 2) {
        jobFault := False
        jobFaultCode := 0
        state := State.Report
      } otherwise {
        expectedPartOrdinal := expectedPartOrdinal + 1
        state := State.WaitProjection
      }
    } otherwise {
      localFeature := localFeature + 1
    }
  }

  io.completion.valid := state === State.Report
  io.completion.requestId := heldRound.step.requestId
  io.completion.stepId := heldRound.step.stepId
  io.completion.layer := heldRound.step.layer
  io.completion.epoch := heldRound.event.epoch
  io.completion.mode := heldRound.step.mode
  io.completion.positionBase := heldRound.step.positionBase
  io.completion.tokenMask := heldRound.step.activeTokenMask
  io.completion.success := !jobFault &&
    seenPartSequence === B"3'b111" &&
    roundEmitted === cfg.totalCoefficients
  io.completion.faultCode := jobFaultCode
  io.completion.emittedCoefficients := roundEmitted
  when(io.completion.fire) {
    completedCount := completedCount + 1
    state := State.Idle
  }
}

object GenerateBitNetResidentRopeCoefficientStager extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-rope-coefficient-stager")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentRopeCoefficientStager())
}
