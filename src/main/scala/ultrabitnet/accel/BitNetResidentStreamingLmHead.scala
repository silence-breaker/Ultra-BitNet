package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Integration geometry for the resident, one-token LM-head transaction. */
case class BitNetResidentStreamingLmHeadConfig(
    model: BitNetConfig = BitNetConfig.Production,
    linear: BitNetResidentLinearEndpointConfig =
      BitNetResidentLinearEndpointConfig(),
    streaming: BitNetStreamingLmHeadConfig =
      BitNetStreamingLmHeadConfig.production(),
    quant: BitNetSharedDynamicI8QuantizerConfig =
      BitNetSharedDynamicI8QuantizerConfig(),
    activationLayout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production),
    unifiedScratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig(),
    coupledMatrixOutputs: Boolean = false
) {
  require(linear.model == model)
  require(activationLayout.cfg == model)
  require(streaming.hiddenSize == model.hiddenSize)
  require(streaming.vocabularySize == model.vocabularySize)
  require(streaming.tokenLanes == model.tokenParallelism)
  require(streaming.engineCount == model.engineCount)
  require(streaming.tileFeatures == linear.matrix.base.siteCount)
  require(streaming.tileFeatures == 64 || streaming.tileFeatures == 128)
  require((model.tokenParallelism == 2 || model.tokenParallelism == 4) &&
    (model.engineCount == 1 || model.engineCount == 2))
  require(model.hiddenSize % model.spuCount == 0)
  require(model.hiddenSize % streaming.tileFeatures == 0)
  require(quant.tokenLanes == model.tokenParallelism)
  require(quant.maxVectorLength >= model.hiddenSize)
  require(quant.inputWidth == 32 && quant.factorWidth == 32)
  require(quant.tagWidth <= model.requestIdWidth)
  require(quant.epochWidth <= 16)
  require(unifiedScratchpad.layout.tokenLanes >= model.tokenParallelism)

  val activationGroupCount: Int = model.hiddenSize / model.spuCount
  val activationGroupWidth: Int =
    scala.math.max(1, log2Up(activationGroupCount))
  val activationShardWidth: Int =
    scala.math.max(1, log2Up(model.spuCount))
  val tileWordWidth: Int = streaming.tileFeatures * 8
  val physicalRowPairs: Int = (model.vocabularySize + model.engineCount - 1) /
    model.engineCount
  val expectedMatrixBeats: Int = physicalRowPairs * streaming.tileCount

  require(physicalRowPairs < (1 << linear.matrix.base.outputIndexWidth))
  require(expectedMatrixBeats < (1 << linear.matrix.beatCountWidth))
}

/** F32 row-scale return from the five-bank LM memory walker. */
case class BitNetResidentLmRowScaleResponse(
    cfg: BitNetResidentStreamingLmHeadConfig) extends Bundle {
  val requestId = UInt(cfg.streaming.requestIdWidth bits)
  val engine = UInt(cfg.streaming.engineWidth bits)
  val tokenId = UInt(cfg.streaming.tokenIdWidth bits)
  val scaleBank = UInt(cfg.streaming.bankWidth bits)
  val rowScaleF32 = Bits(32 bits)
  val fault = Bool()
}

/** The only architectural result of a greedy resident LM-head scan. */
case class BitNetResidentLmHeadCompletion(
    cfg: BitNetResidentStreamingLmHeadConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val mode = BitNetMode()
  val tokenPosition = UInt(cfg.model.sequenceWidth bits)
  val tokenId = UInt(cfg.streaming.tokenIdWidth bits)
  val dot = SInt(32 bits)
  val rowScaleF32 = Bits(32 bits)
  val activationDequantFactorQ16 = UInt(32 bits)
  val rowsCompared = UInt(cfg.streaming.rowCountWidth bits)
  val matrixWeightBeats = UInt(32 bits)
  val cycles = UInt(32 bits)
  val success = Bool()
  val fatal = Bool()
  val faultCode = Bits(16 bits)
}

/** Stable bit allocation used by the resident LM-head completion. */
object BitNetResidentLmHeadFault {
  val InvalidHandoff = 0
  val ActivationFabric = 1
  val DynamicQuantProtocol = 2
  val DynamicQuantCompletion = 3
  val RowDescriptor = 4
  val WeightProtocol = 5
  val ScaleProtocol = 6
  val LinearResult = 7
  val LinearCompletion = 8
  val Reducer = 9
  val UnifiedActivationCodec = 10
  val PackedSemantic = 11
  val Aborted = 12
}

/**
  * BitNet resident LM-head coordinator.
  *
  * The block deliberately owns no matrix array and no dynamic-quantisation
  * multiplier.  It is a client of the model-wide shared dynamic-i8 service
  * and of the sole TP4 compute island's typed `LmHead` linear lease.  The
  * selected final token is read twice from NarrowWork (measure/replay).  In
  * the production geometry its 2560 i8 values are packed into five rows of
  * the sole unified QueryLmAlias tensor and replayed beside streamed LM
  * weights.  The core therefore owns no production activation RAM; no
  * Q16.16 activation crosses PS/PL and no row logit is written to memory.
  *
  * [[BitNetStreamingLmHead]] remains the canonical five-bank row-descriptor
  * generator, bounded request scoreboard and exact F32-scale-aware online
  * argmax.  Its even/odd row mapping is lowered to the two physical decode
  * engines by treating each engine's output index as a row-pair ordinal.  An
  * odd vocabulary therefore executes one zero-padded physical tail row, which
  * is consumed locally and never enters the reducer.
  */
class BitNetResidentStreamingLmHead(
    cfg: BitNetResidentStreamingLmHeadConfig =
      BitNetResidentStreamingLmHeadConfig()) extends Component {
  private val model = cfg.model
  private val lm = cfg.streaming
  private val q = cfg.quant
  private val linear = cfg.linear
  private val matrix = linear.matrix
  private val base = matrix.base
  private val scratchpadCfg = cfg.unifiedScratchpad
  private val scratchpadLayout = scratchpadCfg.layout
  // Small unit tests retain their legacy arbitrary hidden-size geometry.  The
  // production image is deliberately exact so the reachable production RTL
  // can contain one codec and no private LM activation Mem.
  private val useUnifiedLmImage = model.hiddenSize == 2560 &&
    (lm.tileFeatures == 64 || lm.tileFeatures == 128) &&
    lm.tileCount * lm.tileFeatures == model.hiddenSize &&
    (model.tokenParallelism == 2 || model.tokenParallelism == 4) &&
    scratchpadLayout.tokenLanes == model.tokenParallelism

  val io = new Bundle {
    val handoff = slave(Stream(BitNetResidentLmHeadHandoff(model)))

    // Direct resident activation-fabric client.  Only handoff.tokenLane is
    // ever commanded; the quant service sees it lowered to physical lane 0.
    val activationReadCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(model, cfg.activationLayout))),
      model.tokenParallelism)
    val activationReadData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(model, cfg.activationLayout))),
      model.tokenParallelism)

    // Existing shared dynamic-i8 service boundary.  A production top connects
    // this client to the sole quantiser/arithmetic lease, rather than
    // instantiating another quantiser in the LM head.
    val quantCommand = master(Stream(BitNetSharedDynamicQuantCommand(q)))
    val quantMeasureInput = master(Stream(
      BitNetSharedDynamicQuantMeasureBeat(q)))
    val quantReplayRequest = slave(Stream(
      BitNetSharedDynamicQuantReplayRequest(q)))
    val quantReplayInput = master(Stream(
      BitNetSharedDynamicQuantReplayBeat(q)))
    val quantOutput = slave(Stream(BitNetSharedDynamicQuantResultBeat(q)))
    val quantDone = slave(Stream(BitNetSharedDynamicQuantDone(q)))

    // Production QueryLmAlias semantic client.  The subsystem owns the lease
    // lifetime; this core only authenticates every codec command against the
    // retained generation.
    val aliasLeaseValid = in Bool()
    val aliasLeaseGeneration = in UInt(
      scratchpadLayout.aliasGenerationWidth bits)
    val semanticPackedReadOutstanding = in Bool()
    val semanticPackedWriteOutstanding = in Bool()
    val abort = in Bool()
    val packedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(scratchpadCfg)))
    val packedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(scratchpadCfg)))
    val packedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(scratchpadCfg)))
    val packedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(scratchpadCfg)))

    // Five-bank row walk.  Weight payloads return on the disjoint typed-I8
    // stream; F32 scales use a small tagged response, never a logit buffer.
    val rowPair = master(Stream(BitNetLmRowPairRequest(lm)))
    val rowScale = slave(Stream(BitNetResidentLmRowScaleResponse(cfg)))
    val i8WeightInput = slave(Stream(BitNetI8WeightBeat(linear.feeder)))

    // Sole compute-island contract.  These ports wire one-for-one to
    // BitNetResidentQkvAttentionComputeIsland; no second cluster exists.
    val linearLaunch = master(Stream(BitNetResidentLinearLaunch(linear)))
    val matrixI8Weight = master(Stream(BitNetI8WeightBeat(linear.feeder)))
    val matrixActivation = master(Stream(
      BitNetNormalizedActivationTile(linear.feeder)))
    val linearResult = slave(Stream(
      BitNetResidentFinalFeaturePair(linear)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(linear)))

    val completion = master(Stream(BitNetResidentLmHeadCompletion(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(16 bits)
    val quantizedTiles = out UInt(log2Up(lm.tileCount + 1) bits)
    val rowsCompared = out UInt(lm.rowCountWidth bits)
    val matrixWeightBeats = out UInt(32 bits)
    val codecBusy = out Bool()
    val codecQuarantined = out Bool()
  }

  val activationCodec = if (useUnifiedLmImage)
    new BitNetUnifiedResidentLmActivationCodec(
      BitNetUnifiedResidentLmActivationCodecConfig(
        resident = cfg,
        scratchpad = scratchpadCfg,
        coupledMatrixOutputs = cfg.coupledMatrixOutputs))
  else null

  if (useUnifiedLmImage) {
    activationCodec.io.leaseValid := io.aliasLeaseValid
    activationCodec.io.leaseGeneration := io.aliasLeaseGeneration
    io.packedRead << activationCodec.io.packedRead
    activationCodec.io.packedReadData << io.packedReadData
    io.packedWrite << activationCodec.io.packedWrite
    activationCodec.io.packedWriteFault << io.packedWriteFault
    io.codecBusy := activationCodec.io.captureActive ||
      activationCodec.io.replayActive ||
      activationCodec.io.captureCompletion.valid ||
      activationCodec.io.replayCompletion.valid
    io.codecQuarantined := activationCodec.io.quarantined
  } else {
    io.packedRead.valid := False
    io.packedRead.payload :=
      BitNetUnifiedResidentPackedI8Read(scratchpadCfg).getZero
    io.packedReadData.ready := True
    io.packedWrite.valid := False
    io.packedWrite.payload :=
      BitNetUnifiedResidentPackedI8Write(scratchpadCfg).getZero
    io.packedWriteFault.ready := True
    io.codecBusy := False
    io.codecQuarantined := False
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, QuantCommand, MeasureIssue, MeasureWait, MeasureEmit,
        WaitReplayRequest, ReplayIssue, ReplayWait, ReplayEmit,
        WaitQuantDone, Launch, Run, Respond = newElement()
  }
  val state = Reg(State()) init State.Idle

  val activeHandoff = Reg(BitNetResidentLmHeadHandoff(model)) init
    BitNetResidentLmHeadHandoff(model).getZero
  val sourceLane = Reg(UInt(log2Up(model.tokenParallelism) bits)) init 0
  val activeQuantTag = Reg(UInt(q.tagWidth bits)) init 0
  val activeQuantEpoch = Reg(UInt(q.epochWidth bits)) init 0
  val nextQuantEpoch = Reg(UInt(q.epochWidth bits)) init 0
  val activeSequenceId = Reg(UInt(matrix.sequenceIdWidth bits)) init 0

  val jobFault = Reg(Bits(16 bits)) init 0
  val jobAbortRequested = RegInit(False)
  val stickyFault = RegInit(False)
  val cycleCounter = Reg(UInt(32 bits)) init 0
  when(state =/= State.Idle && state =/= State.Respond) {
    cycleCounter := cycleCounter + 1
  }

  val completionReg = Reg(BitNetResidentLmHeadCompletion(cfg)) init
    BitNetResidentLmHeadCompletion(cfg).getZero
  io.completion.valid := state === State.Respond
  io.completion.payload := completionReg
  when(io.completion.fire) {
    state := State.Idle
    jobAbortRequested := False
  }

  val activeJobState = state =/= State.Idle && state =/= State.Respond
  val jobAbortNow = jobAbortRequested || (io.abort && activeJobState)

  val handoffLaneLegal = io.handoff.tokenLane < model.tokenParallelism
  val handoffModeLegal =
    (io.handoff.mode === BitNetMode.Decode && io.handoff.tokenLane === 0) ||
      io.handoff.mode === BitNetMode.Prefill
  val handoffLeaseLegal = if (useUnifiedLmImage)
    io.aliasLeaseValid && io.aliasLeaseGeneration =/= 0
  else True
  val handoffLegal = io.handoff.finalLayer === model.layerCount - 1 &&
    io.handoff.activationBuffer === BitNetPhysicalActivationBuffer.NarrowWork &&
    handoffLaneLegal && handoffModeLegal && handoffLeaseLegal

  io.handoff.ready := state === State.Idle && !io.abort &&
    !io.codecQuarantined
  when(io.handoff.fire) {
    activeHandoff := io.handoff.payload
    sourceLane := io.handoff.tokenLane.resized
    activeQuantTag := io.handoff.requestId.resize(q.tagWidth)
    activeQuantEpoch := nextQuantEpoch
    nextQuantEpoch := nextQuantEpoch + 1
    activeSequenceId := (io.handoff.requestId.asBits ^
      io.handoff.tokenPosition.resize(model.requestIdWidth).asBits)
      .asUInt.resize(matrix.sequenceIdWidth)
    jobFault := 0
    jobAbortRequested := False
    cycleCounter := 0
    completionReg.requestId := io.handoff.requestId
    completionReg.mode := io.handoff.mode
    completionReg.tokenPosition := io.handoff.tokenPosition
    completionReg.tokenId := 0
    completionReg.dot := 0
    completionReg.rowScaleF32 := 0
    completionReg.activationDequantFactorQ16 := 0
    completionReg.rowsCompared := 0
    completionReg.matrixWeightBeats := 0
    completionReg.cycles := 0
    completionReg.success := False
    completionReg.fatal := False
    completionReg.faultCode := 0
    when(handoffLegal) {
      state := State.QuantCommand
    } otherwise {
      jobFault(BitNetResidentLmHeadFault.InvalidHandoff) := True
      stickyFault := True
      completionReg.success := False
      completionReg.fatal := False
      completionReg.faultCode :=
        B(BigInt(1) << BitNetResidentLmHeadFault.InvalidHandoff, 16 bits)
      state := State.Respond
    }
  }

  // -----------------------------------------------------------------------
  // Two resident NarrowWork passes through the external shared quantiser.
  // -----------------------------------------------------------------------
  val readGroup = Reg(UInt(cfg.activationGroupWidth bits)) init 0
  val readShard = Reg(UInt(cfg.activationShardWidth bits)) init 0
  val heldReadValues = Vec(Reg(SInt(32 bits)) init 0, model.spuCount)
  val heldReadFeatureValid = Reg(Bits(model.spuCount bits)) init 0
  val heldReadFault = RegInit(False)

  for (lane <- 0 until model.tokenParallelism) {
    io.activationReadCommand(lane).valid := False
    io.activationReadCommand(lane).payload.buffer :=
      BitNetPhysicalActivationBuffer.NarrowWork
    io.activationReadCommand(lane).payload.group := readGroup.resized
    io.activationReadData(lane).ready := False
    when(sourceLane === lane) {
      io.activationReadCommand(lane).valid :=
        state === State.MeasureIssue || state === State.ReplayIssue
      io.activationReadData(lane).ready :=
        state === State.MeasureWait || state === State.ReplayWait
    }
  }

  val selectedReadValid = Bool()
  val selectedRead = BitNetPhysicalActivationReadBeat(
    model, cfg.activationLayout)
  selectedReadValid := False
  selectedRead := BitNetPhysicalActivationReadBeat(
    model, cfg.activationLayout).getZero
  for (lane <- 0 until model.tokenParallelism) {
    when(sourceLane === lane) {
      selectedReadValid := io.activationReadData(lane).valid
      selectedRead := io.activationReadData(lane).payload
    }
  }

  val selectedReadCommandFire = (0 until model.tokenParallelism).map { lane =>
    io.activationReadCommand(lane).fire
  }.reduce(_ || _)
  val selectedReadFire = selectedReadValid &&
    (state === State.MeasureWait || state === State.ReplayWait)

  if (useUnifiedLmImage) {
    // Atomic two-sink launch: neither the shared quantizer nor the codec can
    // observe this transaction without the other accepting it in the same
    // cycle.
    io.quantCommand.valid := state === State.QuantCommand &&
      activationCodec.io.captureCommand.ready
    activationCodec.io.captureCommand.valid :=
      state === State.QuantCommand && io.quantCommand.ready
    activationCodec.io.captureCommand.tag := activeQuantTag
    activationCodec.io.captureCommand.epoch := activeQuantEpoch
  } else {
    io.quantCommand.valid := state === State.QuantCommand
  }
  io.quantCommand.payload.tag := activeQuantTag
  io.quantCommand.payload.epoch := activeQuantEpoch
  io.quantCommand.payload.vectorLength := model.hiddenSize
  io.quantCommand.payload.tokenMask := B(1, model.tokenParallelism bits)

  val readFeatureIndex = UInt(q.indexWidth bits)
  readFeatureIndex := (readGroup.resize(q.indexWidth) * model.spuCount +
    readShard.resize(q.indexWidth)).resized

  io.quantMeasureInput.valid := state === State.MeasureEmit
  io.quantMeasureInput.payload.tag := activeQuantTag
  io.quantMeasureInput.payload.epoch := activeQuantEpoch
  io.quantMeasureInput.payload.featureIndex := readFeatureIndex
  io.quantMeasureInput.payload.tokenMask := B(1, model.tokenParallelism bits)
  io.quantMeasureInput.payload.first := readFeatureIndex === 0
  io.quantMeasureInput.payload.last := readFeatureIndex === model.hiddenSize - 1
  for (lane <- 0 until model.tokenParallelism) {
    if (lane == 0)
      io.quantMeasureInput.payload.value(lane) := heldReadValues(readShard)
    else
      io.quantMeasureInput.payload.value(lane) := 0
  }

  io.quantReplayRequest.ready := state === State.WaitReplayRequest
  val replayRequestLegal =
    io.quantReplayRequest.tag === activeQuantTag &&
      io.quantReplayRequest.epoch === activeQuantEpoch &&
      io.quantReplayRequest.vectorLength === model.hiddenSize &&
      io.quantReplayRequest.tokenMask === B(1, model.tokenParallelism bits)

  io.quantReplayInput.valid := state === State.ReplayEmit
  io.quantReplayInput.payload.tag := activeQuantTag
  io.quantReplayInput.payload.epoch := activeQuantEpoch
  io.quantReplayInput.payload.featureIndex := readFeatureIndex
  io.quantReplayInput.payload.tokenMask := B(1, model.tokenParallelism bits)
  io.quantReplayInput.payload.first := readFeatureIndex === 0
  io.quantReplayInput.payload.last := readFeatureIndex === model.hiddenSize - 1
  for (lane <- 0 until model.tokenParallelism) {
    if (lane == 0)
      io.quantReplayInput.payload.value(lane) := heldReadValues(readShard)
    else
      io.quantReplayInput.payload.value(lane) := 0
  }

  when(io.quantCommand.fire) {
    readGroup := 0
    readShard := 0
    state := State.MeasureIssue
  }

  when(io.abort && activeJobState) {
    jobAbortRequested := True
    jobFault(BitNetResidentLmHeadFault.Aborted) := True
  }
  when(selectedReadCommandFire) {
    when(state === State.MeasureIssue) { state := State.MeasureWait }
    when(state === State.ReplayIssue) { state := State.ReplayWait }
  }
  when(selectedReadFire) {
    heldReadValues := selectedRead.values
    heldReadFeatureValid := selectedRead.featureValid
    heldReadFault := selectedRead.fault
    readShard := 0
    val responseLegal =
      selectedRead.buffer === BitNetPhysicalActivationBuffer.NarrowWork &&
        selectedRead.group === readGroup.resized && !selectedRead.fault &&
        selectedRead.featureValid.andR
    when(!responseLegal) {
      jobFault(BitNetResidentLmHeadFault.ActivationFabric) := True
      stickyFault := True
    }
    when(state === State.MeasureWait) { state := State.MeasureEmit }
    when(state === State.ReplayWait) { state := State.ReplayEmit }
  }

  when(io.quantMeasureInput.fire) {
    when(heldReadFault || !heldReadFeatureValid(readShard)) {
      jobFault(BitNetResidentLmHeadFault.ActivationFabric) := True
      stickyFault := True
    }
    when(readShard === model.spuCount - 1) {
      readShard := 0
      when(readGroup === cfg.activationGroupCount - 1) {
        readGroup := 0
        state := State.WaitReplayRequest
      } otherwise {
        readGroup := readGroup + 1
        state := State.MeasureIssue
      }
    } otherwise {
      readShard := readShard + 1
    }
  }

  when(io.quantReplayRequest.fire) {
    when(!replayRequestLegal) {
      jobFault(BitNetResidentLmHeadFault.DynamicQuantProtocol) := True
      stickyFault := True
    }
    readGroup := 0
    readShard := 0
    state := State.ReplayIssue
  }
  when(io.quantReplayInput.fire) {
    when(heldReadFault || !heldReadFeatureValid(readShard)) {
      jobFault(BitNetResidentLmHeadFault.ActivationFabric) := True
      stickyFault := True
    }
    when(readShard === model.spuCount - 1) {
      readShard := 0
      when(readGroup === cfg.activationGroupCount - 1) {
        readGroup := 0
        state := State.WaitQuantDone
      } otherwise {
        readGroup := readGroup + 1
        state := State.ReplayIssue
      }
    } otherwise {
      readShard := readShard + 1
    }
  }

  // Non-production test geometries retain the historical local store.  The
  // exact production branch elaborates only the unified codec above, so no
  // private tileMemory is reachable from a production LM hierarchy.
  val tileMemory = if (!useUnifiedLmImage)
    Mem(Bits(cfg.tileWordWidth bits), lm.tileCount)
      .addAttribute("ram_style", "block")
  else null
  val tilePack = if (!useUnifiedLmImage)
    Vec(Reg(Bits(8 bits)) init 0, lm.tileFeatures)
  else null
  val quantOutputCount = Reg(UInt(q.lengthWidth bits)) init 0
  val quantizedTileCount = Reg(UInt(log2Up(lm.tileCount + 1) bits)) init 0
  val quantFactor = Reg(UInt(32 bits)) init 0
  val quantFactorValid = RegInit(False)
  when(io.quantCommand.fire) {
    quantOutputCount := 0
    quantizedTileCount := 0
    quantFactor := 0
    quantFactorValid := False
  }

  if (useUnifiedLmImage) {
    activationCodec.io.quantOutput << io.quantOutput
    activationCodec.io.quantDone << io.quantDone
    activationCodec.io.captureCompletion.ready :=
      state === State.WaitQuantDone
    when(activationCodec.io.captureCompletion.fire) {
      quantOutputCount :=
        activationCodec.io.captureCompletion.acceptedFeatures.resized
      quantizedTileCount := Mux(
        activationCodec.io.captureCompletion.success,
        U(lm.tileCount, quantizedTileCount.getWidth bits),
        (activationCodec.io.captureCompletion.committedRows.resize(
          quantizedTileCount.getWidth) << 2).resized)
      quantFactor :=
        activationCodec.io.captureCompletion.dequantFactorQ16
      quantFactorValid :=
        activationCodec.io.captureCompletion.dequantFactorQ16 =/= 0
    }
  } else {
    val quantRunState = state === State.WaitReplayRequest ||
      state === State.ReplayIssue || state === State.ReplayWait ||
      state === State.ReplayEmit || state === State.WaitQuantDone
    io.quantOutput.ready := quantRunState
    val expectedQuantIndex = quantOutputCount.resize(q.indexWidth)
    val quantOutputLegal =
      io.quantOutput.tag === activeQuantTag &&
        io.quantOutput.epoch === activeQuantEpoch &&
        io.quantOutput.featureIndex === expectedQuantIndex &&
        io.quantOutput.tokenMask === B(1, model.tokenParallelism bits) &&
        io.quantOutput.first === (expectedQuantIndex === 0) &&
        io.quantOutput.last === (expectedQuantIndex === model.hiddenSize - 1) &&
        (!quantFactorValid ||
          io.quantOutput.dequantFactorQ16(0) === quantFactor)

    val quantSite = expectedQuantIndex(6 downto 0)
    val quantTile = (expectedQuantIndex |>> 7).resize(lm.tileIndexWidth)
    val quantWriteWord = Bits(cfg.tileWordWidth bits)
    for (site <- 0 until lm.tileFeatures) {
      quantWriteWord((site + 1) * 8 - 1 downto site * 8) := tilePack(site)
      when(quantSite === site) {
        quantWriteWord((site + 1) * 8 - 1 downto site * 8) :=
          io.quantOutput.value(0).asBits
      }
    }
    tileMemory.write(
      address = quantTile,
      data = quantWriteWord,
      enable = io.quantOutput.fire && quantSite === lm.tileFeatures - 1)

    when(io.quantOutput.fire) {
      when(!quantOutputLegal) {
        jobFault(BitNetResidentLmHeadFault.DynamicQuantProtocol) := True
        stickyFault := True
      }
      tilePack(quantSite) := io.quantOutput.value(0).asBits
      quantOutputCount := quantOutputCount + 1
      when(!quantFactorValid) {
        quantFactor := io.quantOutput.dequantFactorQ16(0)
        quantFactorValid := True
      }
      when(quantSite === lm.tileFeatures - 1) {
        quantizedTileCount := quantizedTileCount + 1
      }
    }

    io.quantDone.ready := state === State.WaitQuantDone
  }

  // Compile-time geometry selects exactly one terminal expression.  Avoid a
  // default hardware drive followed by a branch-local full override, which
  // Spinal correctly diagnoses as an assignment overlap.
  val captureTransactionFire = if (useUnifiedLmImage)
    activationCodec.io.captureCompletion.fire
  else io.quantDone.fire
  val captureTransactionLegal = if (useUnifiedLmImage)
    activationCodec.io.captureCompletion.tag === activeQuantTag &&
      activationCodec.io.captureCompletion.epoch === activeQuantEpoch &&
      activationCodec.io.captureCompletion.acceptedFeatures ===
        model.hiddenSize &&
      activationCodec.io.captureCompletion.committedRows ===
        (model.hiddenSize / (scratchpadLayout.bankCount *
          scratchpadLayout.packedFeaturesPerWord *
          scratchpadLayout.nativeSymbolCount)) &&
      activationCodec.io.captureCompletion.dequantFactorQ16 =/= 0 &&
      activationCodec.io.captureCompletion.success &&
      activationCodec.io.captureCompletion.faultCode === 0 &&
      activationCodec.io.imageReady &&
      activationCodec.io.imageGeneration === io.aliasLeaseGeneration
  else
    io.quantDone.tag === activeQuantTag &&
      io.quantDone.epoch === activeQuantEpoch &&
      io.quantDone.vectorLength === model.hiddenSize &&
      io.quantDone.emittedFeatures === model.hiddenSize &&
      io.quantDone.success && io.quantDone.faultCode === 0 &&
      quantOutputCount === model.hiddenSize &&
      quantizedTileCount === lm.tileCount && quantFactorValid &&
      quantFactor =/= 0

  // -----------------------------------------------------------------------
  // Existing row-descriptor/scoreboard/exact-argmax frontend.
  // -----------------------------------------------------------------------
  val reducer = new BitNetStreamingLmHead(lm)
  val reducerLaunchPending = RegInit(False)
  val linearLaunchPending = RegInit(False)

  val logicalMask = Bits(model.tokenParallelism bits)
  logicalMask := B(1, model.tokenParallelism bits)
  when(activeHandoff.mode === BitNetMode.Prefill) {
    for (lane <- 0 until model.tokenParallelism) {
      when(sourceLane === lane) {
        logicalMask := B((BigInt(1) << (lane + 1)) - 1,
          model.tokenParallelism bits)
      }
    }
  }

  reducer.io.start.valid := state === State.Launch && reducerLaunchPending
  reducer.io.start.payload.logicalMode := activeHandoff.mode
  reducer.io.start.payload.activeTokenMask := logicalMask
  reducer.io.start.payload.finalTokenLane := sourceLane.resized
  reducer.io.start.payload.activationSlot := 0
  reducer.io.start.payload.activationEpoch := activeQuantEpoch.resize(16)
  reducer.io.start.payload.activationDequantFactorQ16 := quantFactor

  // The reducer's activationTile is a descriptor-only audit stream.  The
  // actual i8 tile payload comes from QueryLmAlias in production (or the tiny
  // legacy test store) and is replayed once per weight beat.
  reducer.io.activationTile.ready := True
  when(reducer.io.activationTile.fire) {
    val activationDescriptorLegal =
      reducer.io.activationTile.physicalMode === BitNetMode.Decode &&
        reducer.io.activationTile.sourceTokenLane === sourceLane &&
        reducer.io.activationTile.features === lm.tileFeatures &&
        reducer.io.activationTile.engineMask === B(
          (BigInt(1) << model.engineCount) - 1, model.engineCount bits)
    when(!activationDescriptorLegal) {
      jobFault(BitNetResidentLmHeadFault.RowDescriptor) := True
      stickyFault := True
    }
  }

  io.linearLaunch.valid := state === State.Launch && linearLaunchPending
  io.linearLaunch.payload.requestId := activeHandoff.requestId
  io.linearLaunch.payload.layer := activeHandoff.finalLayer
  io.linearLaunch.payload.kind := BitNetResidentLinearKind.LmHead
  // LM is physically one token with two independent vocabulary shards even
  // when the logical producer was the final lane of a prefill round.
  io.linearLaunch.payload.mode := BitNetMode.Decode
  io.linearLaunch.payload.tokenMask := B(1, model.tokenParallelism bits)
  io.linearLaunch.payload.sequenceId := activeSequenceId
  io.linearLaunch.payload.matrixTask.scheduleMode := BitNetMode.Decode
  io.linearLaunch.payload.matrixTask.operation :=
    BitNetContinuousDotKind.LmHead
  io.linearLaunch.payload.matrixTask.sequenceId := activeSequenceId
  io.linearLaunch.payload.matrixTask.taskIdBase := 0
  io.linearLaunch.payload.matrixTask.expectedBeats := cfg.expectedMatrixBeats
  io.linearLaunch.payload.matrixTask.kTilesPerOutput := lm.tileCount
  io.linearLaunch.payload.matrixTask.activeTokenMask :=
    B(1, model.tokenParallelism bits)
  io.linearLaunch.payload.matrixTask.routeFinalToDynamicQuant := False
  io.linearLaunch.payload.matrixTask.residualEnable := False
  for (lane <- 0 until model.tokenParallelism) {
    io.linearLaunch.payload.matrixTask.tokenId(lane) :=
      activeHandoff.tokenPosition.resize(base.tokenIdWidth)
    io.linearLaunch.payload.matrixTask.activationFactorQ16(lane) := quantFactor
  }
  for (engine <- 0 until model.engineCount) {
    // Both engines count pair ordinals.  Row identity is
    // pairOrdinal*2+engine at the resident LM boundary.
    io.linearLaunch.payload.matrixTask.weightScaleQ16(engine) := 0x10000L
    io.linearLaunch.payload.matrixTask.outputBase(engine) := 0
    io.linearLaunch.payload.matrixTask.outputEndExclusive(engine) :=
      cfg.physicalRowPairs
  }

  val captureCodecFaultNow = if (useUnifiedLmImage)
    activationCodec.io.captureCompletion.valid &&
      (!activationCodec.io.captureCompletion.success ||
        activationCodec.io.captureCompletion.faultCode =/= 0)
  else False
  val captureTerminalFault = Bits(16 bits)
  captureTerminalFault := jobFault
  when(jobAbortNow) {
    captureTerminalFault(BitNetResidentLmHeadFault.Aborted) := True
  }
  when(captureCodecFaultNow) {
    captureTerminalFault(
      BitNetResidentLmHeadFault.UnifiedActivationCodec) := True
    jobFault(BitNetResidentLmHeadFault.UnifiedActivationCodec) := True
  }
  when(captureTransactionFire && !captureTransactionLegal) {
    captureTerminalFault(
      BitNetResidentLmHeadFault.DynamicQuantCompletion) := True
  }

  when(captureTransactionFire) {
    when(captureTransactionLegal && captureTerminalFault === 0) {
      reducerLaunchPending := True
      linearLaunchPending := True
      state := State.Launch
    } otherwise {
      jobFault(BitNetResidentLmHeadFault.DynamicQuantCompletion) := True
      when(!jobAbortNow) { stickyFault := True }
      completionReg.success := False
      completionReg.fatal := False
      completionReg.faultCode := captureTerminalFault
      completionReg.cycles := cycleCounter
      state := State.Respond
    }
  }

  val reducerLaunchAfter = reducerLaunchPending && !reducer.io.start.fire
  val linearLaunchAfter = linearLaunchPending && !io.linearLaunch.fire
  when(state === State.Launch) {
    when(reducer.io.start.fire) { reducerLaunchPending := False }
    when(io.linearLaunch.fire) { linearLaunchPending := False }
    when(!reducerLaunchAfter && !linearLaunchAfter) { state := State.Run }
  }

  // Mirror every accepted row descriptor into a small join scoreboard.  The
  // reducer owns allocation/reuse; this mirror only joins typed linear results
  // with independently returned F32 row scales.
  val rowExpected = Vec(RegInit(False), lm.outstandingRows)
  val expectedRowToken = Vec(
    Reg(UInt(lm.tokenIdWidth bits)) init 0, lm.outstandingRows)
  val expectedRowEngine = Vec(
    Reg(UInt(lm.engineWidth bits)) init 0, lm.outstandingRows)
  val expectedRowBank = Vec(
    Reg(UInt(lm.bankWidth bits)) init 0, lm.outstandingRows)
  val rowScaleValid = Vec(RegInit(False), lm.outstandingRows)
  val rowScaleValue = Vec(Reg(Bits(32 bits)) init 0, lm.outstandingRows)
  val rowScaleFault = Vec(RegInit(False), lm.outstandingRows)

  io.rowPair.payload := reducer.io.rowPair.payload
  // The atomic fork can expose exactly one side while the other is stalled.
  // Retain that already-visible offer across abort, then flush the reducer and
  // prevent every later row-pair/replay transaction.
  val rowPairOfferPending = RegInit(False)
  val allowRowPairAfterAbort = !jobAbortNow || rowPairOfferPending
  if (useUnifiedLmImage) {
    // Atomic row-pair fork.  The fetcher cannot begin returning its twenty
    // weight tiles until the codec has accepted the matching replay identity.
    io.rowPair.valid := reducer.io.rowPair.valid && state === State.Run &&
      activationCodec.io.replayCommand.ready && allowRowPairAfterAbort
    activationCodec.io.replayCommand.valid :=
      reducer.io.rowPair.valid && state === State.Run && io.rowPair.ready &&
        allowRowPairAfterAbort
    activationCodec.io.replayCommand.imageTag := activeQuantTag
    activationCodec.io.replayCommand.imageEpoch := activeQuantEpoch
    activationCodec.io.replayCommand.sequenceId := activeSequenceId
    activationCodec.io.replayCommand.pairOrdinal :=
      reducer.io.rowPair.pairOrdinal
    activationCodec.io.replayCommand.beatOrdinalBase :=
      (reducer.io.rowPair.pairOrdinal.resize(matrix.beatCountWidth) *
        lm.tileCount).resized
    reducer.io.rowPair.ready := io.rowPair.ready &&
      activationCodec.io.replayCommand.ready && state === State.Run &&
      allowRowPairAfterAbort
  } else {
    io.rowPair.valid := reducer.io.rowPair.valid && state === State.Run &&
      allowRowPairAfterAbort
    reducer.io.rowPair.ready := io.rowPair.ready && state === State.Run &&
      allowRowPairAfterAbort
  }
  val rowPairSideOffered = io.rowPair.valid && !io.rowPair.ready ||
    (if (useUnifiedLmImage)
      activationCodec.io.replayCommand.valid &&
        !activationCodec.io.replayCommand.ready
     else False)
  when(rowPairSideOffered) { rowPairOfferPending := True }
  when(io.rowPair.fire) { rowPairOfferPending := False }
  when(state === State.Launch) { rowPairOfferPending := False }
  reducer.io.abort := jobAbortNow && !rowPairOfferPending
  if (useUnifiedLmImage) {
    activationCodec.io.abort := jobAbortNow
  }
  when(io.rowPair.fire) {
    for (engine <- 0 until model.engineCount) {
      when(io.rowPair.engineMask(engine)) {
        val slot = io.rowPair.engine(engine).requestId
        when(rowExpected(slot) || rowScaleValid(slot)) {
          jobFault(BitNetResidentLmHeadFault.RowDescriptor) := True
          stickyFault := True
        }
        rowExpected(slot) := True
        expectedRowToken(slot) := io.rowPair.engine(engine).tokenId
        expectedRowEngine(slot) := U(engine, lm.engineWidth bits)
        expectedRowBank(slot) := io.rowPair.engine(engine).scaleBank
        rowScaleValid(slot) := False
        rowScaleFault(slot) := False
      }
    }
  }

  io.rowScale.ready := state === State.Run
  val scaleSlotKnown = rowExpected(io.rowScale.requestId)
  val scaleMetadataLegal = scaleSlotKnown &&
    io.rowScale.tokenId === expectedRowToken(io.rowScale.requestId) &&
    io.rowScale.engine === expectedRowEngine(io.rowScale.requestId) &&
    io.rowScale.scaleBank === expectedRowBank(io.rowScale.requestId)
  when(io.rowScale.fire) {
    when(jobAbortNow) {
      // Response to an already-issued row descriptor: drain only.
    } elsewhen(scaleMetadataLegal && !rowScaleValid(io.rowScale.requestId)) {
      rowScaleValid(io.rowScale.requestId) := True
      rowScaleValue(io.rowScale.requestId) := io.rowScale.rowScaleF32
      rowScaleFault(io.rowScale.requestId) := io.rowScale.fault
      when(io.rowScale.fault) {
        jobFault(BitNetResidentLmHeadFault.ScaleProtocol) := True
        stickyFault := True
      }
    } otherwise {
      jobFault(BitNetResidentLmHeadFault.ScaleProtocol) := True
      stickyFault := True
    }
  }

  // -----------------------------------------------------------------------
  // Replay the unified activation image beside every typed I8 weight beat.
  // -----------------------------------------------------------------------
  val matrixWeightBeatCount = Reg(UInt(32 bits)) init 0

  when(state === State.Launch && !reducerLaunchAfter && !linearLaunchAfter) {
    matrixWeightBeatCount := 0
    for (slot <- 0 until lm.outstandingRows) {
      rowExpected(slot) := False
      rowScaleValid(slot) := False
      rowScaleFault(slot) := False
    }
  }

  val replayStatus = if (useUnifiedLmImage) {
    activationCodec.io.i8WeightInput << io.i8WeightInput
    io.matrixI8Weight << activationCodec.io.matrixI8Weight
    io.matrixActivation << activationCodec.io.matrixActivation

    val expectedReplayPair = Reg(UInt(lm.pairOrdinalWidth bits)) init 0
    when(state === State.Launch && !reducerLaunchAfter && !linearLaunchAfter) {
      expectedReplayPair := 0
    }
    activationCodec.io.replayCompletion.ready := True
    val replayCompletionLegal =
      activationCodec.io.replayCompletion.sequenceId === activeSequenceId &&
        activationCodec.io.replayCompletion.pairOrdinal ===
          expectedReplayPair &&
        activationCodec.io.replayCompletion.acceptedWeightTiles ===
          lm.tileCount &&
        activationCodec.io.replayCompletion.emittedTiles === lm.tileCount &&
        activationCodec.io.replayCompletion.success &&
        activationCodec.io.replayCompletion.faultCode === 0
    when(activationCodec.io.replayCompletion.fire) {
      when(replayCompletionLegal) {
        expectedReplayPair := expectedReplayPair + 1
      } otherwise {
        jobFault(BitNetResidentLmHeadFault.UnifiedActivationCodec) := True
        when(!jobAbortNow) { stickyFault := True }
      }
    }
    when(io.matrixI8Weight.fire) {
      matrixWeightBeatCount := matrixWeightBeatCount + 1
    }
    when(activationCodec.io.protocolFaultSticky) {
      jobFault(BitNetResidentLmHeadFault.UnifiedActivationCodec) := True
      stickyFault := True
    }
    (
      activationCodec.io.replayActive ||
        activationCodec.io.replayCompletion.valid,
      activationCodec.io.replayCompletion.valid && !replayCompletionLegal
    )
  } else {
    val expectedWeightBeat = Reg(UInt(matrix.beatCountWidth bits)) init 0
    val expectedWeightK = Reg(UInt(lm.tileIndexWidth bits)) init 0
    val expectedWeightPair = Reg(UInt(lm.pairOrdinalWidth bits)) init 0
    val replayValid = RegInit(False)
    val replayWeightPending = RegInit(False)
    val replayActivationPending = RegInit(False)
    val replayWeight = Reg(BitNetI8WeightBeat(linear.feeder)) init
      BitNetI8WeightBeat(linear.feeder).getZero

    val replayWeightRemaining = replayWeightPending &&
      !io.matrixI8Weight.ready
    val replayActivationRemaining = replayActivationPending &&
      !io.matrixActivation.ready
    val replayCanAdvance = !replayValid ||
      (!replayWeightRemaining && !replayActivationRemaining)
    val incomingWeightLegal =
      io.i8WeightInput.sequenceId === activeSequenceId &&
        io.i8WeightInput.beatOrdinal === expectedWeightBeat &&
        io.i8WeightInput.use === BitNetI8WeightUse.LmHead &&
        (0 until model.engineCount).map { engine =>
          io.i8WeightInput.engineOutputBase(engine) === expectedWeightPair
        }.reduce(_ && _) &&
        io.i8WeightInput.firstK === (expectedWeightK === 0) &&
        io.i8WeightInput.lastK === (expectedWeightK === lm.tileCount - 1) &&
        !io.i8WeightInput.fault
    io.i8WeightInput.ready := state === State.Run && replayCanAdvance
    val weightIssue = io.i8WeightInput.fire
    val replayTileWord = tileMemory.readSync(
      address = expectedWeightK,
      enable = weightIssue)

    when(state === State.Launch && !reducerLaunchAfter && !linearLaunchAfter) {
      expectedWeightBeat := 0
      expectedWeightK := 0
      expectedWeightPair := 0
      replayValid := False
      replayWeightPending := False
      replayActivationPending := False
    }
    when(replayCanAdvance) {
      replayValid := weightIssue
      replayWeightPending := weightIssue
      replayActivationPending := weightIssue
      when(weightIssue) { replayWeight := io.i8WeightInput.payload }
    } otherwise {
      when(io.matrixI8Weight.fire) { replayWeightPending := False }
      when(io.matrixActivation.fire) { replayActivationPending := False }
    }
    when(weightIssue) {
      matrixWeightBeatCount := matrixWeightBeatCount + 1
      when(!incomingWeightLegal) {
        jobFault(BitNetResidentLmHeadFault.WeightProtocol) := True
        stickyFault := True
      }
      expectedWeightBeat := expectedWeightBeat + 1
      when(expectedWeightK === lm.tileCount - 1) {
        expectedWeightK := 0
        expectedWeightPair := expectedWeightPair + 1
      } otherwise {
        expectedWeightK := expectedWeightK + 1
      }
    }

    io.matrixI8Weight.valid := replayValid && replayWeightPending
    io.matrixI8Weight.payload := replayWeight
    io.matrixActivation.valid := replayValid && replayActivationPending
    io.matrixActivation.payload.sequenceId := replayWeight.sequenceId
    io.matrixActivation.payload.beatOrdinal := replayWeight.beatOrdinal
    io.matrixActivation.payload.fault := replayWeight.fault
    for (lane <- 0 until model.tokenParallelism;
         site <- 0 until lm.tileFeatures) {
      if (lane == 0)
        io.matrixActivation.payload.tokenActivation(lane)(site) :=
          replayTileWord((site + 1) * 8 - 1 downto site * 8).asSInt
      else
        io.matrixActivation.payload.tokenActivation(lane)(site) := 0
    }
    (replayValid, False)
  }
  val replayBusy = replayStatus._1
  val codecReplayFaultNow = replayStatus._2

  // -----------------------------------------------------------------------
  // Join final-K typed results with row scales and feed exact online argmax.
  // -----------------------------------------------------------------------
  val resultTokenWide = UInt((lm.tokenIdWidth + 1) bits)
  resultTokenWide := (
    (io.linearResult.physicalOutputBase.resize(lm.tokenIdWidth + 1) << 1)
      .resize(lm.tokenIdWidth + 1) +
      io.linearResult.engine.resize(lm.tokenIdWidth + 1)).resized
  val resultIsRealRow = resultTokenWide < model.vocabularySize
  val resultToken = resultTokenWide.resize(lm.tokenIdWidth)
  val resultRequestId = resultToken.resize(lm.requestIdWidth)
  val resultSlotKnown = rowExpected(resultRequestId) &&
    expectedRowToken(resultRequestId) === resultToken &&
    expectedRowEngine(resultRequestId) === io.linearResult.engine

  val resultMetadataLegal =
    io.linearResult.requestId === activeHandoff.requestId &&
      io.linearResult.layer === activeHandoff.finalLayer &&
      io.linearResult.kind === BitNetResidentLinearKind.LmHead &&
      io.linearResult.mode === BitNetMode.Decode &&
      io.linearResult.sequenceId === activeSequenceId &&
      io.linearResult.globalTokenLaneMask === B(1, model.tokenParallelism bits) &&
      io.linearResult.outputLaneMask === B"2'b01" &&
      io.linearResult.tokenId(0) ===
        activeHandoff.tokenPosition.resize(base.tokenIdWidth) &&
      io.linearResult.activationFactorQ16(0) === quantFactor
  val linearResultFaultNow = io.linearResult.fire && !jobAbortNow &&
    ((!resultIsRealRow && !(
      io.linearResult.requestId === activeHandoff.requestId &&
        io.linearResult.kind === BitNetResidentLinearKind.LmHead &&
        io.linearResult.sequenceId === activeSequenceId)) ||
      (resultIsRealRow && (!resultSlotKnown ||
        !rowScaleValid(resultRequestId) || !resultMetadataLegal)))

  reducer.io.rowCompletion.valid := io.linearResult.valid &&
    state === State.Run && !jobAbortNow && resultIsRealRow && resultSlotKnown &&
    rowScaleValid(resultRequestId)
  reducer.io.rowCompletion.payload.requestId := resultRequestId
  reducer.io.rowCompletion.payload.engine := io.linearResult.engine
  reducer.io.rowCompletion.payload.tokenId := resultToken
  reducer.io.rowCompletion.payload.scaleBank :=
    expectedRowBank(resultRequestId)
  reducer.io.rowCompletion.payload.dot := io.linearResult.rawRows(0)(0)
  reducer.io.rowCompletion.payload.rowScaleF32 :=
    rowScaleValue(resultRequestId)
  reducer.io.rowCompletion.payload.tilesSeen := lm.tileCount
  reducer.io.rowCompletion.payload.bankMask :=
    B((BigInt(1) << lm.bankCount) - 1, lm.bankCount bits)
  reducer.io.rowCompletion.payload.firstSeen :=
    resultMetadataLegal && !rowScaleFault(resultRequestId)
  reducer.io.rowCompletion.payload.lastSeen :=
    resultMetadataLegal && !rowScaleFault(resultRequestId)

  io.linearResult.ready := False
  when(state === State.Run) {
    when(jobAbortNow) {
      // Results belonging to the accepted linear launch are terminal drains
      // after cancellation; they must not re-enter the flushed reducer.
      io.linearResult.ready := True
    } elsewhen(!resultIsRealRow) {
      // Odd-vocabulary physical padding row: required for balanced decode
      // engines, but deliberately absent from the reducer scoreboard.
      io.linearResult.ready := True
    } elsewhen(!resultSlotKnown) {
      // Unknown real-row identities are terminally malformed.  Consume and
      // attribute them instead of waiting forever for a scale/tag which can
      // never legally arrive.
      io.linearResult.ready := True
    } elsewhen(resultSlotKnown && rowScaleValid(resultRequestId)) {
      io.linearResult.ready := reducer.io.rowCompletion.ready
    }
  }
  when(io.linearResult.fire) {
    when(jobAbortNow) {
      when(resultIsRealRow && resultSlotKnown) {
        rowExpected(resultRequestId) := False
        rowScaleValid(resultRequestId) := False
        rowScaleFault(resultRequestId) := False
      }
    } elsewhen(resultIsRealRow) {
      when(!resultSlotKnown || !rowScaleValid(resultRequestId) ||
          !resultMetadataLegal) {
        jobFault(BitNetResidentLmHeadFault.LinearResult) := True
        stickyFault := True
      }
      when(resultSlotKnown && rowScaleValid(resultRequestId)) {
        rowExpected(resultRequestId) := False
        rowScaleValid(resultRequestId) := False
        rowScaleFault(resultRequestId) := False
      }
    } otherwise {
      val paddingLegal =
        io.linearResult.requestId === activeHandoff.requestId &&
          io.linearResult.kind === BitNetResidentLinearKind.LmHead &&
          io.linearResult.sequenceId === activeSequenceId
      when(!paddingLegal) {
        jobFault(BitNetResidentLmHeadFault.LinearResult) := True
        stickyFault := True
      }
    }
  }

  val winnerSeen = RegInit(False)
  val winnerReg = Reg(BitNetLmWinner(lm)) init BitNetLmWinner(lm).getZero
  reducer.io.winner.ready := state === State.Run && !winnerSeen
  when(reducer.io.winner.fire) {
    winnerSeen := True
    winnerReg := reducer.io.winner.payload
    when(reducer.io.winner.error && !jobAbortNow) {
      jobFault(BitNetResidentLmHeadFault.Reducer) := True
      stickyFault := True
    }
  }

  val linearDoneSeen = RegInit(False)
  val linearDoneReg = Reg(BitNetResidentLinearCompletion(linear)) init
    BitNetResidentLinearCompletion(linear).getZero
  io.linearCompletion.ready := state === State.Run && !linearDoneSeen
  val linearDoneLegal =
    io.linearCompletion.requestId === activeHandoff.requestId &&
      io.linearCompletion.layer === activeHandoff.finalLayer &&
      io.linearCompletion.kind === BitNetResidentLinearKind.LmHead &&
      io.linearCompletion.sequenceId === activeSequenceId &&
      io.linearCompletion.success && !io.linearCompletion.fatal &&
      io.linearCompletion.faultCode === 0 &&
      io.linearCompletion.acceptedResultBeats ===
        U(BigInt(cfg.expectedMatrixBeats) * model.engineCount, 32 bits) &&
      io.linearCompletion.emittedPairs ===
        U(BigInt(cfg.physicalRowPairs) * model.engineCount, 32 bits)
  when(io.linearCompletion.fire) {
    linearDoneSeen := True
    linearDoneReg := io.linearCompletion.payload
    when(!linearDoneLegal) {
      jobFault(BitNetResidentLmHeadFault.LinearCompletion) := True
      when(!jobAbortNow) { stickyFault := True }
    }
  }

  when(state === State.Launch && !reducerLaunchAfter && !linearLaunchAfter) {
    winnerSeen := False
    linearDoneSeen := False
  }

  val winnerAvailable = winnerSeen || reducer.io.winner.fire
  val linearDoneAvailable = linearDoneSeen || io.linearCompletion.fire
  val selectedWinner = BitNetLmWinner(lm)
  selectedWinner := winnerReg
  when(reducer.io.winner.fire) { selectedWinner := reducer.io.winner.payload }
  val selectedLinearDone = BitNetResidentLinearCompletion(linear)
  selectedLinearDone := linearDoneReg
  when(io.linearCompletion.fire) {
    selectedLinearDone := io.linearCompletion.payload
  }
  val selectedLinearDoneLegal =
    selectedLinearDone.requestId === activeHandoff.requestId &&
      selectedLinearDone.layer === activeHandoff.finalLayer &&
      selectedLinearDone.kind === BitNetResidentLinearKind.LmHead &&
      selectedLinearDone.sequenceId === activeSequenceId &&
      selectedLinearDone.success && !selectedLinearDone.fatal &&
      selectedLinearDone.faultCode === 0 &&
      selectedLinearDone.acceptedResultBeats ===
        U(BigInt(cfg.expectedMatrixBeats) * model.engineCount, 32 bits) &&
      selectedLinearDone.emittedPairs ===
        U(BigInt(cfg.physicalRowPairs) * model.engineCount, 32 bits)
  val finalFault = Bits(16 bits)
  finalFault := jobFault
  when(jobAbortNow) {
    finalFault(BitNetResidentLmHeadFault.Aborted) := True
  }
  when(linearResultFaultNow) {
    finalFault(BitNetResidentLmHeadFault.LinearResult) := True
  }
  when(io.rowScale.fire && !jobAbortNow &&
      (!scaleMetadataLegal || io.rowScale.fault)) {
    finalFault(BitNetResidentLmHeadFault.ScaleProtocol) := True
  }
  when(codecReplayFaultNow) {
    finalFault(BitNetResidentLmHeadFault.UnifiedActivationCodec) := True
  }
  when(selectedWinner.error) {
    finalFault(BitNetResidentLmHeadFault.Reducer) := True
  }
  when(!selectedLinearDoneLegal) {
    finalFault(BitNetResidentLmHeadFault.LinearCompletion) := True
  }
  val matrixWeightBeatCountAfter = matrixWeightBeatCount +
    io.matrixI8Weight.fire.asUInt.resize(32)
  when(matrixWeightBeatCountAfter =/= cfg.expectedMatrixBeats) {
    finalFault(BitNetResidentLmHeadFault.WeightProtocol) := True
  }

  when(state === State.Run && !jobAbortNow && winnerAvailable &&
      linearDoneAvailable) {
    completionReg.requestId := activeHandoff.requestId
    completionReg.mode := activeHandoff.mode
    completionReg.tokenPosition := activeHandoff.tokenPosition
    completionReg.tokenId := selectedWinner.tokenId
    completionReg.dot := selectedWinner.dot
    completionReg.rowScaleF32 := selectedWinner.rowScaleF32
    completionReg.activationDequantFactorQ16 :=
      selectedWinner.activationDequantFactorQ16
    completionReg.rowsCompared := selectedWinner.rowsCompared
    completionReg.matrixWeightBeats := matrixWeightBeatCountAfter
    completionReg.cycles := cycleCounter
    completionReg.success := finalFault === 0 &&
      selectedWinner.rowsCompared === model.vocabularySize
    completionReg.fatal := selectedLinearDone.fatal
    completionReg.faultCode := finalFault
    when(finalFault.orR) { stickyFault := True }
    state := State.Respond
  }

  val packedSemanticBusy = io.semanticPackedReadOutstanding ||
    io.semanticPackedWriteOutstanding
  val abortRunDrained = state === State.Run && jobAbortNow &&
    !replayBusy && !io.codecBusy && !packedSemanticBusy &&
    !rowPairOfferPending && linearDoneAvailable
  when(abortRunDrained) {
    completionReg.requestId := activeHandoff.requestId
    completionReg.mode := activeHandoff.mode
    completionReg.tokenPosition := activeHandoff.tokenPosition
    completionReg.tokenId := 0
    completionReg.dot := 0
    completionReg.rowScaleF32 := 0
    completionReg.activationDequantFactorQ16 := quantFactor
    completionReg.rowsCompared := reducer.io.rowsCompared
    completionReg.matrixWeightBeats := matrixWeightBeatCountAfter
    completionReg.cycles := cycleCounter
    completionReg.success := False
    completionReg.fatal := selectedLinearDone.fatal
    completionReg.faultCode := finalFault |
      B(BigInt(1) << BitNetResidentLmHeadFault.Aborted, 16 bits)
    state := State.Respond
  }
  if (useUnifiedLmImage) {
    when(packedSemanticBusy && state === State.Idle && !io.codecBusy) {
      jobFault(BitNetResidentLmHeadFault.PackedSemantic) := True
      stickyFault := True
    }
  }
  io.busy := state =/= State.Idle || reducer.io.busy || replayBusy ||
    io.codecBusy || packedSemanticBusy
  io.protocolFaultSticky := stickyFault || reducer.io.protocolFault ||
    io.codecQuarantined
  io.faultBits := jobFault
  io.quantizedTiles := quantizedTileCount
  io.rowsCompared := reducer.io.rowsCompared
  io.matrixWeightBeats := matrixWeightBeatCount
}

object GenerateBitNetResidentStreamingLmHead extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-streaming-lm-head")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentStreamingLmHead())
}
