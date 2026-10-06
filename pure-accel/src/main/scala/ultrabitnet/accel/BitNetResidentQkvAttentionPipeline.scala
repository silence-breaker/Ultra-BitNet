package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetResidentQkvAttentionPipelineConfig(
    attention: BitNetResidentAttentionSubsystemConfig =
      BitNetResidentAttentionSubsystemConfig(),
    kvWriter: BitNetPackedKvAppendWriterConfig =
      BitNetPackedKvAppendWriterConfig()
) {
  val model: BitNetConfig = attention.model
  val qkv: BitNetQkvPostprocessConfig = attention.qResident.qkv
  val crossbar: BitNetBankLocalCrossbarConfig = attention.crossbar
  require(kvWriter.model == model)
  require(kvWriter.qkv == qkv)
  require(kvWriter.crossbar == crossbar)
  val expectedQkvFrames: Int = model.queryHeadCount + 2 * model.kvHeadCount
  val frameCountWidth: Int = log2Up(expectedQkvFrames + 1)
}

/** One projected TP4 round whose Q/K/V outputs remain entirely in PL. */
case class BitNetResidentQkvRoundCommand(
    cfg: BitNetResidentQkvAttentionPipelineConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val mode = BitNetMode()
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenId = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.model.tokenParallelism)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
}

case class BitNetResidentQkvRoundCompletion(
    cfg: BitNetResidentQkvAttentionPipelineConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/**
  * Production QKV-to-attention shell.
  *
  * Raw projection results flow through scale/RoPE/head-local i8 quantisation.
  * Q is retained in four on-chip resident memories, while K/V are appended to
  * their bank-local packed cache with WSTRB.  Attention cannot start until
  * both owners report completion.  The public boundary therefore contains no
  * software-assembled query block and no naked K/V vector injection port.
  */
class BitNetResidentQkvAttentionController(
    cfg: BitNetResidentQkvAttentionPipelineConfig =
      BitNetResidentQkvAttentionPipelineConfig(),
    useVendorPrimitive: Boolean = true) extends Component {
  private val lanes = cfg.model.tokenParallelism
  private val sharedQuantCfg = BitNetSharedDynamicI8QuantizerConfig()

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()

    val qkvRound = slave(Stream(BitNetResidentQkvRoundCommand(cfg)))
    val qkvDot = slave(Stream(BitNetQkvTp4DotBeat(cfg.qkv)))
    val qkvCompletion = master(Stream(
      BitNetResidentQkvRoundCompletion(cfg)))

    val projectionScaleAcquire = master(Stream(
      BitNetSpuArithmeticLeaseRequest(cfg.qkv.arithmetic)))
    val projectionScaleOperand = master(Stream(
      BitNetSpuArithmeticOperandBeat(cfg.qkv.arithmetic)))
    val projectionScaleResult = slave(Stream(
      BitNetSpuArithmeticResultBeat(cfg.qkv.arithmetic)))
    val projectionScaleRelease = master(Stream(
      BitNetSpuArithmeticLeaseRelease(cfg.qkv.arithmetic)))
    val spuPoolFault = in Bool()
    val quantCommand = master(Stream(
      BitNetSharedDynamicQuantCommand(sharedQuantCfg)))
    val quantMeasure = master(Stream(
      BitNetSharedDynamicQuantMeasureBeat(sharedQuantCfg)))
    val quantReplayRequest = slave(Stream(
      BitNetSharedDynamicQuantReplayRequest(sharedQuantCfg)))
    val quantReplay = master(Stream(
      BitNetSharedDynamicQuantReplayBeat(sharedQuantCfg)))
    val quantResult = slave(Stream(
      BitNetSharedDynamicQuantResultBeat(sharedQuantCfg)))
    val quantDone = slave(Stream(
      BitNetSharedDynamicQuantDone(sharedQuantCfg)))
    val linearScaleInput = slave(Stream(NativeScaleResidualBeat(
      NativeScaleResidualConfig(lanes = cfg.model.tokenParallelism))))
    val linearScaleOutput = master(Stream(NativeScaleResidualResult(
      NativeScaleResidualConfig(lanes = cfg.model.tokenParallelism))))

    val attentionCommand = slave(Stream(
      BitNetAttentionScheduleCommand(cfg.model)))
    val attentionCompletion = master(Stream(
      BitNetAttentionCompletion(cfg.model)))
    val queryAbort = in Bool()
    val queryPackedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(
        cfg.attention.unifiedScratchpad)))
    val queryPackedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(
        cfg.attention.unifiedScratchpad)))
    val queryPackedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(
        cfg.attention.unifiedScratchpad)))
    val queryPackedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(
        cfg.attention.unifiedScratchpad)))
    val queryAliasAcquire = master(Stream(
      BitNetUnifiedResidentAliasAcquire(
        cfg.attention.unifiedScratchpad)))
    val queryAliasGrant = slave(Stream(
      BitNetUnifiedResidentAliasGrant(
        cfg.attention.unifiedScratchpad)))
    val queryAliasRelease = master(Stream(
      BitNetUnifiedResidentAliasRelease(
        cfg.attention.unifiedScratchpad)))
    val queryAliasAcquireOwned = in Bool()
    val querySemanticPackedReadOutstanding = in Bool()
    val querySemanticPackedWriteOutstanding = in Bool()

    val kvWriteRequest = master(Stream(
      BitNetClientBankWriteRequest(cfg.crossbar)))
    val kvWriteData = master(Stream(
      BitNetClientBankWriteData(cfg.crossbar)))
    val kvWriteResponse = slave(Stream(
      BitNetClientBankWriteResponse(cfg.crossbar)))
    val kvReadRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar)))
    val kvReadResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar)))
    val writeData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(
        cfg.model, cfg.attention.activationLayout))), lanes)

    val matrixSequence = Vec(master(Stream(
      BitNetContinuousSequence(cfg.attention.continuous))),
      cfg.model.engineCount)
    val matrixI8Beat = Vec(master(Stream(
      BitNetI8MatrixBeat(BitNetContinuousMatrixFeederConfig(
        matrix = cfg.attention.continuous)))),
      cfg.model.engineCount)
    val matrixDotResult = Vec(slave(Stream(
      BitNetProductionCompactTp2DotResultBeat(
        cfg.attention.continuous))),
      cfg.model.engineCount)
    val matrixSvResult = if (!cfg.attention.cluster.productionPruneGenericSv)
      Vec(slave(Stream(
      BitNetContinuousSvResultGroup(
        cfg.attention.continuous, cfg.attention.residentGroupLanes))),
      cfg.model.engineCount) else null
    val matrixResidentSequence = Vec(master(Stream(
      BitNetResidentSvSequence(cfg.attention.continuous))),
      cfg.model.engineCount)
    val matrixResidentBeat = Vec(master(Stream(
      BitNetResidentSvBeat(cfg.attention.continuous))),
      cfg.model.engineCount)
    val matrixResidentResult = Vec(slave(Stream(
      BitNetResidentSvResultGroup(
        cfg.attention.continuous, cfg.attention.residentGroupLanes))),
      cfg.model.engineCount)
    val matrixSequenceActive = in Bits(cfg.model.engineCount bits)
    val matrixDraining = in Bits(cfg.model.engineCount bits)
    val matrixProtocolFaultSticky = in Bits(cfg.model.engineCount bits)

    val qkvImageReady = out Bool()
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedQkvRounds = out UInt(32 bits)
    val completedQkvRounds = out UInt(32 bits)
  }

  val attention = new BitNetResidentAttentionController(
    cfg.attention, useVendorPrimitive)
  attention.io.slowClk := io.slowClk
  attention.io.slowResetn := io.slowResetn

  for (engine <- 0 until cfg.model.engineCount) {
    io.matrixSequence(engine) << attention.io.matrixSequence(engine)
    io.matrixI8Beat(engine) << attention.io.matrixI8Beat(engine)
    attention.io.matrixDotResult(engine) << io.matrixDotResult(engine)
    if (!cfg.attention.cluster.productionPruneGenericSv)
      attention.io.matrixSvResult(engine) << io.matrixSvResult(engine)
    io.matrixResidentSequence(engine) <<
      attention.io.matrixResidentSequence(engine)
    io.matrixResidentBeat(engine) << attention.io.matrixResidentBeat(engine)
    attention.io.matrixResidentResult(engine) <<
      io.matrixResidentResult(engine)
  }
  attention.io.matrixSequenceActive := io.matrixSequenceActive
  attention.io.matrixDraining := io.matrixDraining
  attention.io.matrixProtocolFaultSticky :=
    io.matrixProtocolFaultSticky

  io.kvReadRequest << attention.io.kvReadRequest
  attention.io.kvReadResponse << io.kvReadResponse
  for (lane <- 0 until lanes) {
    io.writeData(lane) << attention.io.writeData(lane)
  }
  io.attentionCompletion << attention.io.completion
  io.queryPackedRead << attention.io.queryPackedRead
  attention.io.queryPackedReadData << io.queryPackedReadData
  io.queryPackedWrite << attention.io.queryPackedWrite
  attention.io.queryPackedWriteFault << io.queryPackedWriteFault
  io.queryAliasAcquire << attention.io.queryAliasAcquire
  attention.io.queryAliasGrant << io.queryAliasGrant
  io.queryAliasRelease << attention.io.queryAliasRelease
  attention.io.queryAliasAcquireOwned := io.queryAliasAcquireOwned
  attention.io.querySemanticPackedReadOutstanding :=
    io.querySemanticPackedReadOutstanding
  attention.io.querySemanticPackedWriteOutstanding :=
    io.querySemanticPackedWriteOutstanding

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  val slowArea = new ClockingArea(slowCd) {
    val postprocess = new BitNetQkvPostprocess(cfg.qkv)
    val kvWriter = new BitNetPackedKvAppendWriter(
      cfg.kvWriter, cfg.attention.residentLayout)

    io.projectionScaleAcquire << postprocess.io.poolAcquire
    io.projectionScaleOperand << postprocess.io.poolOperand
    postprocess.io.poolResult << io.projectionScaleResult
    io.projectionScaleRelease << postprocess.io.poolRelease
    postprocess.io.poolFault := io.spuPoolFault
    io.quantCommand << postprocess.io.quantCommand
    io.quantMeasure << postprocess.io.quantMeasure
    postprocess.io.quantReplayRequest << io.quantReplayRequest
    io.quantReplay << postprocess.io.quantReplay
    postprocess.io.quantResult << io.quantResult
    postprocess.io.quantDone << io.quantDone
    postprocess.io.linearScaleInput << io.linearScaleInput
    io.linearScaleOutput << postprocess.io.linearScaleOutput

    val activeRound = RegInit(False)
    val held = Reg(BitNetResidentQkvRoundCommand(cfg)) init (
      BitNetResidentQkvRoundCommand(cfg).getZero)
    // bit0=query owner, bit1=packed KV writer
    val commandPending = Reg(Bits(2 bits)) init 0
    val inputFrameCount = Reg(UInt(cfg.frameCountWidth bits)) init 0
    val qDone = RegInit(False)
    val kvDone = RegInit(False)
    val qSuccess = RegInit(False)
    val kvSuccess = RegInit(False)
    val qMetadataFault = RegInit(False)
    val kvMetadataFault = RegInit(False)
    val imageReady = RegInit(False)
    val faultSticky = RegInit(False)
    val failureCleanupPending = RegInit(False)

    val completionValid = RegInit(False)
    val completionPayload = Reg(BitNetResidentQkvRoundCompletion(cfg)) init (
      BitNetResidentQkvRoundCompletion(cfg).getZero)
    io.qkvCompletion.valid := completionValid
    io.qkvCompletion.payload := completionPayload
    when(io.qkvCompletion.fire) { completionValid := False }

    val acceptedCount = Reg(UInt(32 bits)) init 0
    val completedCount = Reg(UInt(32 bits)) init 0
    io.acceptedQkvRounds := acceptedCount
    io.completedQkvRounds := completedCount

    io.qkvRound.ready := !activeRound && !completionValid &&
      !failureCleanupPending
    when(io.qkvRound.fire) {
      held := io.qkvRound.payload
      activeRound := True
      imageReady := False
      commandPending := B"2'b11"
      inputFrameCount := 0
      qDone := False
      kvDone := False
      qSuccess := False
      kvSuccess := False
      qMetadataFault := False
      kvMetadataFault := False
      acceptedCount := acceptedCount + 1
    }

    attention.io.queryLoad.valid := activeRound && commandPending(0)
    attention.io.queryLoad.requestId := held.requestId
    attention.io.queryLoad.layer := held.layer
    attention.io.queryLoad.mode := held.mode
    attention.io.queryLoad.tokenBase := held.tokenBase
    attention.io.queryLoad.tokenId := held.tokenId
    attention.io.queryLoad.tokenMask := held.tokenMask
    when(attention.io.queryLoad.fire) { commandPending(0) := False }

    kvWriter.io.command.valid := activeRound && commandPending(1)
    kvWriter.io.command.requestId := held.requestId
    kvWriter.io.command.layer := held.layer
    kvWriter.io.command.mode := held.mode
    kvWriter.io.command.tokenBase := held.tokenBase
    kvWriter.io.command.tokenId := held.tokenId
    kvWriter.io.command.tokenMask := held.tokenMask
    when(kvWriter.io.command.fire) { commandPending(1) := False }

    val ingressOpen = activeRound && !commandPending.orR &&
      inputFrameCount < cfg.expectedQkvFrames
    postprocess.io.input.valid := io.qkvDot.valid && ingressOpen
    postprocess.io.input.payload := io.qkvDot.payload
    io.qkvDot.ready := postprocess.io.input.ready && ingressOpen
    when(io.qkvDot.fire && io.qkvDot.last) {
      inputFrameCount := inputFrameCount + 1
    }

    attention.io.querySample << postprocess.io.query
    kvWriter.io.key << postprocess.io.key
    kvWriter.io.value << postprocess.io.value
    io.kvWriteRequest << kvWriter.io.writeRequest
    io.kvWriteData << kvWriter.io.writeData
    kvWriter.io.writeResponse << io.kvWriteResponse

    val qCompletion = attention.io.queryLoadCompletion
    val qMetadataMatches = qCompletion.requestId === held.requestId &&
      qCompletion.layer === held.layer &&
      qCompletion.tokenBase === held.tokenBase &&
      qCompletion.tokenMask === held.tokenMask
    qCompletion.ready := activeRound && !qDone
    when(qCompletion.fire) {
      qDone := True
      qSuccess := qCompletion.success && qMetadataMatches
      when(!qMetadataMatches) {
        qMetadataFault := True
        faultSticky := True
      }
    }

    val kvCompletion = kvWriter.io.completion
    val kvMetadataMatches = kvCompletion.requestId === held.requestId &&
      kvCompletion.layer === held.layer &&
      kvCompletion.tokenBase === held.tokenBase &&
      kvCompletion.tokenMask === held.tokenMask
    kvCompletion.ready := activeRound && !kvDone
    when(kvCompletion.fire) {
      kvDone := True
      kvSuccess := kvCompletion.success && kvMetadataMatches
      when(!kvMetadataMatches) {
        kvMetadataFault := True
        faultSticky := True
      }
    }

    val qDoneAfter = qDone || qCompletion.fire
    val kvDoneAfter = kvDone || kvCompletion.fire
    val qSuccessAfter = Mux(qCompletion.fire,
      qCompletion.success && qMetadataMatches, qSuccess)
    val kvSuccessAfter = Mux(kvCompletion.fire,
      kvCompletion.success && kvMetadataMatches, kvSuccess)
    val allFramesAccepted = inputFrameCount === cfg.expectedQkvFrames
    val ingressFault = postprocess.io.protocolFault ||
      kvWriter.io.protocolFaultSticky ||
      attention.io.protocolFaultSticky ||
      qMetadataFault || kvMetadataFault

    val roundTerminal = activeRound && qDoneAfter && kvDoneAfter &&
      !completionValid && !failureCleanupPending
    val roundSuccess = qSuccessAfter && kvSuccessAfter &&
      allFramesAccepted && !ingressFault

    when(roundTerminal) {
      completionPayload.requestId := held.requestId
      completionPayload.layer := held.layer
      completionPayload.tokenBase := held.tokenBase
      completionPayload.tokenMask := held.tokenMask
      completionPayload.success := roundSuccess
      completionPayload.faultCode := 0
      when(!allFramesAccepted) { completionPayload.faultCode(0) := True }
      when(postprocess.io.protocolFault) { completionPayload.faultCode(1) := True }
      when(kvWriter.io.protocolFaultSticky) {
        completionPayload.faultCode(2) := True
      }
      when(attention.io.protocolFaultSticky) {
        completionPayload.faultCode(3) := True
      }
      when(qMetadataFault || kvMetadataFault) {
        completionPayload.faultCode(4) := True
      }
      when(!qSuccessAfter) { completionPayload.faultCode(5) := True }
      when(!kvSuccessAfter) { completionPayload.faultCode(6) := True }
      activeRound := False
      imageReady := roundSuccess
      when(roundSuccess) {
        completionValid := True
        completedCount := completedCount + 1
      } otherwise {
        failureCleanupPending := True
        faultSticky := True
      }
    }

    // A failed QKV round may leave a successfully loaded Q image resident
    // even though attention will never start.  Abort explicitly, then delay
    // the failed round completion until QueryLmAlias has actually released.
    attention.io.queryAbort := io.queryAbort || failureCleanupPending ||
      (roundTerminal && !roundSuccess)
    when(io.queryAbort) {
      imageReady := False
    }
    when(failureCleanupPending && !attention.io.queryLeaseHeld &&
        !attention.io.queryCleanupBusy && !completionValid) {
      failureCleanupPending := False
      completionValid := True
      completedCount := completedCount + 1
    }

    // A fixed controller may present the next stage command early, but it is
    // not accepted until Q and every packed K/V write are resident.
    attention.io.command.valid := io.attentionCommand.valid && imageReady
    attention.io.command.payload := io.attentionCommand.payload
    io.attentionCommand.ready := attention.io.command.ready && imageReady
    when(attention.io.completion.fire) { imageReady := False }

    io.qkvImageReady := imageReady
    io.protocolFaultSticky := faultSticky || ingressFault
    io.busy := activeRound || failureCleanupPending || completionValid ||
      postprocess.io.busy ||
      kvWriter.io.busy || attention.io.busy
  }
}

/**
  * Focused QKV/attention simulation shell.  A deployable model top uses
  * [[BitNetResidentQkvAttentionController]] behind the global matrix lease;
  * this wrapper retains one local cluster solely for stand-alone verification.
  */
class BitNetResidentQkvAttentionPipeline(
    cfg: BitNetResidentQkvAttentionPipelineConfig =
      BitNetResidentQkvAttentionPipelineConfig(),
    useVendorPrimitive: Boolean = true) extends Component {
  private val lanes = cfg.model.tokenParallelism
  private val engines = cfg.model.engineCount

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val qkvRound = slave(Stream(BitNetResidentQkvRoundCommand(cfg)))
    val qkvDot = slave(Stream(BitNetQkvTp4DotBeat(cfg.qkv)))
    val qkvCompletion = master(Stream(
      BitNetResidentQkvRoundCompletion(cfg)))
    val attentionCommand = slave(Stream(
      BitNetAttentionScheduleCommand(cfg.model)))
    val attentionCompletion = master(Stream(
      BitNetAttentionCompletion(cfg.model)))
    val queryAbort = in Bool()
    val queryPackedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(
        cfg.attention.unifiedScratchpad)))
    val queryPackedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(
        cfg.attention.unifiedScratchpad)))
    val queryPackedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(
        cfg.attention.unifiedScratchpad)))
    val queryPackedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(
        cfg.attention.unifiedScratchpad)))
    val queryAliasAcquire = master(Stream(
      BitNetUnifiedResidentAliasAcquire(
        cfg.attention.unifiedScratchpad)))
    val queryAliasGrant = slave(Stream(
      BitNetUnifiedResidentAliasGrant(
        cfg.attention.unifiedScratchpad)))
    val queryAliasRelease = master(Stream(
      BitNetUnifiedResidentAliasRelease(
        cfg.attention.unifiedScratchpad)))
    val queryAliasAcquireOwned = in Bool()
    val querySemanticPackedReadOutstanding = in Bool()
    val querySemanticPackedWriteOutstanding = in Bool()
    val kvWriteRequest = master(Stream(
      BitNetClientBankWriteRequest(cfg.crossbar)))
    val kvWriteData = master(Stream(
      BitNetClientBankWriteData(cfg.crossbar)))
    val kvWriteResponse = slave(Stream(
      BitNetClientBankWriteResponse(cfg.crossbar)))
    val kvReadRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar)))
    val kvReadResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar)))
    val writeData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(
        cfg.model, cfg.attention.activationLayout))), lanes)

    val qkvImageReady = out Bool()
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedQkvRounds = out UInt(32 bits)
    val completedQkvRounds = out UInt(32 bits)
  }

  val controller = new BitNetResidentQkvAttentionController(
    cfg, useVendorPrimitive)
  val cluster = new BitNetTp4MatrixCluster(
    cfg.attention.cluster, useVendorPrimitive)
  val slowCd = controller.slowCd
  val fastCd = cluster.fastCd

  controller.io.slowClk := io.slowClk
  controller.io.slowResetn := io.slowResetn
  cluster.io.slowClk := io.slowClk
  cluster.io.slowResetn := io.slowResetn
  cluster.io.fastClk := io.fastClk
  cluster.io.fastResetn := io.fastResetn

  controller.io.qkvRound << io.qkvRound
  controller.io.qkvDot << io.qkvDot
  io.qkvCompletion << controller.io.qkvCompletion
  controller.io.attentionCommand << io.attentionCommand
  io.attentionCompletion << controller.io.attentionCompletion
  controller.io.queryAbort := io.queryAbort
  io.queryPackedRead << controller.io.queryPackedRead
  controller.io.queryPackedReadData << io.queryPackedReadData
  io.queryPackedWrite << controller.io.queryPackedWrite
  controller.io.queryPackedWriteFault << io.queryPackedWriteFault
  io.queryAliasAcquire << controller.io.queryAliasAcquire
  controller.io.queryAliasGrant << io.queryAliasGrant
  io.queryAliasRelease << controller.io.queryAliasRelease
  controller.io.queryAliasAcquireOwned := io.queryAliasAcquireOwned
  controller.io.querySemanticPackedReadOutstanding :=
    io.querySemanticPackedReadOutstanding
  controller.io.querySemanticPackedWriteOutstanding :=
    io.querySemanticPackedWriteOutstanding
  io.kvWriteRequest << controller.io.kvWriteRequest
  io.kvWriteData << controller.io.kvWriteData
  controller.io.kvWriteResponse << io.kvWriteResponse
  io.kvReadRequest << controller.io.kvReadRequest
  controller.io.kvReadResponse << io.kvReadResponse
  for (lane <- 0 until lanes) {
    io.writeData(lane) << controller.io.writeData(lane)
  }

  for (engine <- 0 until engines) {
    cluster.io.sequence(engine) << controller.io.matrixSequence(engine)
    // This focused shell sends QK over the typed I8 mailbox and SV over the
    // resident stream.  Linear Map0 traffic has no producer here.
    cluster.io.map0Beat(engine).valid := False
    cluster.io.map0Beat(engine).payload :=
      BitNetMap0PairMatrixBeat(BitNetContinuousMatrixFeederConfig(
        matrix = cfg.attention.cluster.matrix)).getZero
    cluster.io.i8Beat(engine) << controller.io.matrixI8Beat(engine)
    if (!cfg.attention.cluster.productionPruneLegacyBeat) {
      cluster.io.beat(engine).valid := False
      cluster.io.beat(engine).payload := BitNetContinuousMatrixBeat(
        cfg.attention.cluster.matrix,
        includeGenericSv =
          !cfg.attention.cluster.productionPruneGenericSv).getZero
    }
    controller.io.matrixDotResult(engine) << cluster.io.dotResult(engine)
    if (!cfg.attention.cluster.productionPruneGenericSv)
      controller.io.matrixSvResult(engine) << cluster.io.svResult(engine)
    cluster.io.residentSequence(engine) <<
      controller.io.matrixResidentSequence(engine)
    cluster.io.residentBeat(engine) <<
      controller.io.matrixResidentBeat(engine)
    controller.io.matrixResidentResult(engine) <<
      cluster.io.residentResult(engine)
  }
  controller.io.matrixSequenceActive := cluster.io.sequenceActive
  controller.io.matrixDraining := cluster.io.draining
  controller.io.matrixProtocolFaultSticky :=
    cluster.io.protocolFaultSticky
  controller.io.linearScaleInput.valid := False
  controller.io.linearScaleInput.payload := NativeScaleResidualBeat(
    NativeScaleResidualConfig(lanes = cfg.model.tokenParallelism)).getZero
  controller.io.linearScaleOutput.ready := True

  // This class is only the focused simulation shell.  Keep its old closed
  // boundary by supplying a local pool; the deployable compute island owns
  // the one global pool and connects the controller to client zero instead.
  val localArithmeticArea = new ClockingArea(slowCd) {
    val pool = new BitNetSharedSpuArithmeticPool(cfg.qkv.arithmetic)
    pool.io.acquire(BitNetSpuArithmeticClient.ProjectionScale) <<
      controller.io.projectionScaleAcquire
    pool.io.operand(BitNetSpuArithmeticClient.ProjectionScale) <<
      controller.io.projectionScaleOperand
    controller.io.projectionScaleResult <<
      pool.io.result(BitNetSpuArithmeticClient.ProjectionScale)
    pool.io.release(BitNetSpuArithmeticClient.ProjectionScale) <<
      controller.io.projectionScaleRelease

    for (client <- 1 until cfg.qkv.arithmetic.clientCount) {
      pool.io.acquire(client).valid := False
      pool.io.acquire(client).payload :=
        BitNetSpuArithmeticLeaseRequest(cfg.qkv.arithmetic).getZero
      pool.io.operand(client).valid := False
      pool.io.operand(client).payload :=
        BitNetSpuArithmeticOperandBeat(cfg.qkv.arithmetic).getZero
      pool.io.result(client).ready := True
      pool.io.release(client).valid := False
      pool.io.release(client).payload :=
        BitNetSpuArithmeticLeaseRelease(cfg.qkv.arithmetic).getZero
    }
    pool.io.clearFault := False
    controller.io.spuPoolFault := pool.io.protocolFaultSticky
  }

  io.qkvImageReady := controller.io.qkvImageReady
  io.busy := controller.io.busy
  io.protocolFaultSticky := controller.io.protocolFaultSticky
  io.acceptedQkvRounds := controller.io.acceptedQkvRounds
  io.completedQkvRounds := controller.io.completedQkvRounds
}

object GenerateBitNetResidentQkvAttentionPipeline extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-qkv-attention-pipeline")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetResidentQkvAttentionPipeline())
}
