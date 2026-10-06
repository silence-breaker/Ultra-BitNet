package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Geometry of the production, pass-resident attention subsystem.
  *
  * The two TP2 matrix engines are a single physical resource.  QK and SV are
  * therefore deliberately phases of one transaction, rather than independent
  * clients behind an arbitration fabric.  This is the important scheduling
  * invariant inherited from BitNet: scores are completed and buffered
  * before the same DSP sites are turned into the resident SV accumulator.
  */
case class BitNetResidentAttentionSubsystemConfig(
    model: BitNetConfig = BitNetConfig.Production,
    continuous: BitNetContinuousMatrixConfig =
      BitNetContinuousMatrixConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    maxSeq: Int = 4096,
    qkMetaFifoDepth: Int = 64,
    qkResultFifoDepth: Int = 8,
    scalerMetaFifoDepth: Int = 64,
    probabilityFifoDepth: Int = 16,
    residentPreparedFifoDepth: Int = 16,
    residentGroupLanes: Int = 8,
    unifiedScratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig(),
    qkvPostprocess: BitNetQkvPostprocessConfig =
      BitNetQkvPostprocessConfig()
) {
  require(model.tokenParallelism == model.engineCount *
    continuous.tokenLanesPerEngine)
  require(model.queryHeadCount == 20 && model.kvHeadCount == 5)
  require(model.gqaGroupSize == 4 && model.headSize == 128)
  require(maxSeq >= 64 && maxSeq <= model.maxSequenceLength && isPow2(maxSeq))
  require(continuous.tokenLanesPerEngine == 2)
  require(model.headSize % continuous.base.siteCount == 0)
  require(residentGroupLanes == 8)

  val qk: BitNetQkTp4AssemblerConfig = BitNetQkTp4AssemblerConfig(
    model = model,
    continuous = continuous,
    maxSeq = maxSeq,
    metaFifoDepth = qkMetaFifoDepth,
    resultFifoDepth = qkResultFifoDepth)
  val scaler: BitNetQkVectorScalerConfig =
    BitNetQkVectorScalerConfig(
      qk = qk,
      scalar = BitNetQkScoreScalerConfig(
        tokenIdWidth = continuous.base.tokenIdWidth,
        headWidth = log2Up(model.queryHeadCount),
        keyPositionWidth = log2Up(model.maxSequenceLength),
        tagWidth = continuous.taskIdWidth),
      metaFifoDepth = scalerMetaFifoDepth)
  val softmax: BitNetBufferedSoftmaxConfig =
    BitNetBufferedSoftmaxConfig(
      model = model,
      maxSeq = maxSeq,
      probabilityFifoDepth = probabilityFifoDepth,
      probabilityReplays = model.headSize / continuous.base.siteCount)
  val qResident: BitNetQResidentBufferConfig =
    BitNetQResidentBufferConfig(
      model = model,
      qkv = qkvPostprocess,
      qk = qk)
  val unifiedQuery: BitNetUnifiedResidentQueryAdapterConfig =
    BitNetUnifiedResidentQueryAdapterConfig(
      resident = qResident, scratchpad = unifiedScratchpad)
  val resident: BitNetResidentAttentionConfig =
    BitNetResidentAttentionConfig(
      model = model,
      continuous = continuous,
      maxSeq = maxSeq,
      preparedFifoDepth = residentPreparedFifoDepth)
  val cluster: BitNetTp4MatrixClusterConfig =
    BitNetTp4MatrixClusterConfig(
      matrix = continuous,
      svGroupLanes = residentGroupLanes,
      logicalEngineCount = model.engineCount,
      physicalEngineCount = model.physicalMatrixEngineCount,
      productionPruneGenericSv = true)
  val writer: BitNetResidentContextWriterConfig =
    BitNetResidentContextWriterConfig(
      model = model,
      continuous = continuous,
      groupLanes = residentGroupLanes)
  val packedKv: BitNetPackedKvBlockReaderConfig =
    BitNetPackedKvBlockReaderConfig(
      model = model,
      crossbar = crossbar,
      maxSeq = maxSeq)
  val residentLayout: BitNetFiveBankStripedLayout =
    BitNetFiveBankStripedLayout(model)
  val activationLayout: BitNetStreamingActivationLayout =
    BitNetStreamingActivationLayout(model)

  // A key row becomes visible only after all eight 128-bit segments arrive.
  // The serialized scaler drains its four logical lanes in half that time, so
  // it cannot reduce production prefill/decode key throughput.
  require(packedKv.segmentsPerRow >= scaler.serializedIssueCycles,
    "QK lane serialization must stay ahead of packed KV row production")
}

/** Stable diagnostic bit allocation used by completion.faultCode. */
object BitNetResidentAttentionFault {
  val PassScheduler = 0
  val QkAssembly = 1
  val QkScaling = 2
  val BufferedSoftmax = 3
  val ResidentScheduler = 4
  val MatrixCluster = 5
  val ContextWriter = 6
  val NumericOrPhase = 7
}

/**
  * Complete BitNet attention path for one workload command.
  *
  * All ports except `fastClk/fastResetn` are in the related 1x `slowClk`
  * domain.  The bank-local memory boundary carries packed 33-beat KV blocks;
  * K is swept while scores are collected and V is independently swept after
  * softmax, so no second 4096x128 V copy is retained in the datapath.
  */
class BitNetResidentAttentionController(
    cfg: BitNetResidentAttentionSubsystemConfig =
      BitNetResidentAttentionSubsystemConfig(),
    useVendorPrimitive: Boolean = true) extends Component {
  private val lanes = cfg.model.tokenParallelism
  private val engines = cfg.model.engineCount

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()

    val command = slave(Stream(
      BitNetAttentionScheduleCommand(cfg.model)))
    /** Direct QKV-postprocess stream; no externally assembled 4x128 Q block. */
    val queryLoad = slave(Stream(BitNetQResidentLoadCommand(cfg.qResident)))
    val querySample = slave(Stream(BitNetQkvQuantizedSample(cfg.qResident.qkv)))
    val queryLoadCompletion = master(Stream(
      BitNetQResidentLoadCompletion(cfg.qResident)))
    val queryAbort = in Bool()
    val queryPackedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(cfg.unifiedScratchpad)))
    val queryPackedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(cfg.unifiedScratchpad)))
    val queryPackedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(cfg.unifiedScratchpad)))
    val queryPackedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(
        cfg.unifiedScratchpad)))
    val queryAliasAcquire = master(Stream(
      BitNetUnifiedResidentAliasAcquire(cfg.unifiedScratchpad)))
    val queryAliasGrant = slave(Stream(
      BitNetUnifiedResidentAliasGrant(cfg.unifiedScratchpad)))
    val queryAliasRelease = master(Stream(
      BitNetUnifiedResidentAliasRelease(cfg.unifiedScratchpad)))
    val queryAliasAcquireOwned = in Bool()
    val querySemanticPackedReadOutstanding = in Bool()
    val querySemanticPackedWriteOutstanding = in Bool()
    val queryLeaseHeld = out Bool()
    val queryCleanupBusy = out Bool()
    val queryQuarantined = out Bool()
    val kvReadRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar)))
    val kvReadResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar)))
    val writeData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(
        cfg.model, cfg.activationLayout))), lanes)
    val completion = master(Stream(BitNetAttentionCompletion(cfg.model)))

    // The controller owns no DSP array.  These typed streams form one atomic
    // client of the model-wide TP4 matrix lease fabric.
    val matrixSequence = Vec(master(Stream(
      BitNetContinuousSequence(cfg.continuous))), engines)
    /** QK uses the same compact typed-I8 mailbox as LM-head. */
    val matrixI8Beat = Vec(master(Stream(
      BitNetI8MatrixBeat(BitNetContinuousMatrixFeederConfig(
        matrix = cfg.continuous)))), engines)
    val matrixDotResult = Vec(slave(Stream(
      BitNetProductionCompactTp2DotResultBeat(cfg.continuous))), engines)
    val matrixSvResult = if (!cfg.cluster.productionPruneGenericSv)
      Vec(slave(Stream(
      BitNetContinuousSvResultGroup(
        cfg.continuous, cfg.residentGroupLanes))), engines) else null
    val matrixResidentSequence = Vec(master(Stream(
      BitNetResidentSvSequence(cfg.continuous))), engines)
    val matrixResidentBeat = Vec(master(Stream(
      BitNetResidentSvBeat(cfg.continuous))), engines)
    val matrixResidentResult = Vec(slave(Stream(
      BitNetResidentSvResultGroup(
        cfg.continuous, cfg.residentGroupLanes))), engines)
    val matrixSequenceActive = in Bits(engines bits)
    val matrixDraining = in Bits(engines bits)
    val matrixProtocolFaultSticky = in Bits(engines bits)

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val numericFaultSticky = out Bool()
    val faultBits = out Bits(8 bits)
    val acceptedCommands = out UInt(32 bits)
    val completedCommands = out UInt(32 bits)
    val completedPasses = out UInt(32 bits)
  }

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))
  val slowArea = new ClockingArea(slowCd) {
    val passScheduler = new BitNetAttentionPassScheduler(cfg.model)
    val qkAssembler = new BitNetQkTp4Assembler(cfg.qk)
    val qResident = new BitNetUnifiedResidentQueryAdapter(
      cfg.unifiedQuery, useVendorSegmentRam = useVendorPrimitive)
    val qkScaler = new BitNetQkVectorScaler(cfg.scaler)
    val softmax = new BitNetBufferedSoftmaxSpu(cfg.softmax)
    val resident = new BitNetResidentAttentionScheduler(cfg.resident)
    val packedKv = new BitNetPackedKvBlockReader(
      cfg.packedKv, cfg.residentLayout)
    val writer = new BitNetResidentContextWriter(
      cfg.writer, cfg.activationLayout)

    object Phase extends SpinalEnum {
      val Idle, StartFrontEnd, RunQk, WaitQkArrayIdle,
          StartResident, RunResident, WaitResidentHalf0Idle,
          StartResidentHalf1, RunResidentHalf1, RetireQuery,
          EmitCompletion = newElement()
    }
    val phase = Reg(Phase()) init Phase.Idle
    val activePass = Reg(BitNetAttentionPass(cfg.model)) init (
      BitNetAttentionPass(cfg.model).getZero)

    val faultSticky = RegInit(False)
    val numericSticky = RegInit(False)
    val passFaultBits = Reg(Bits(8 bits)) init 0
    val completionPayload = Reg(BitNetAttentionCompletion(cfg.model)) init (
      BitNetAttentionCompletion(cfg.model).getZero)
    val completedCommandCount = Reg(UInt(32 bits)) init 0
    val completedPassCount = Reg(UInt(32 bits)) init 0
    val queryIdentity = Reg(BitNetQResidentLoadCommand(cfg.qResident)) init
      BitNetQResidentLoadCommand(cfg.qResident).getZero
    val queryIdentityValid = RegInit(False)
    val queryRetireIssued = RegInit(False)

    // Do not permit a look-ahead workload command while the final pass is
    // still being drained.  The pass scheduler itself can otherwise become
    // idle as soon as its final pass is accepted by this subsystem.
    passScheduler.io.command.valid := io.command.valid &&
      phase === Phase.Idle && !passScheduler.io.pass.valid
    passScheduler.io.command.payload := io.command.payload
    io.command.ready := passScheduler.io.command.ready &&
      phase === Phase.Idle && !passScheduler.io.pass.valid

    passScheduler.io.pass.ready := phase === Phase.Idle
    when(passScheduler.io.pass.fire) {
      activePass := passScheduler.io.pass.payload
      // Faults are accumulated across all 20 prefill or five decode passes;
      // only the first pass of a new invocation starts a fresh barrier.
      when(passScheduler.io.pass.first) {
        passFaultBits := 0
      }
      phase := Phase.StartFrontEnd
    }

    // Q scalar samples stay on chip and are transposed according to the pass
    // lane role. Ordered K/V rows come from the packed resident image.
    qResident.io.load << io.queryLoad
    qResident.io.sample << io.querySample
    io.queryLoadCompletion << qResident.io.loadCompletion
    qResident.io.abort := io.queryAbort
    io.queryPackedRead << qResident.io.packedRead
    qResident.io.packedReadData << io.queryPackedReadData
    io.queryPackedWrite << qResident.io.packedWrite
    qResident.io.packedWriteFault << io.queryPackedWriteFault
    io.queryAliasAcquire << qResident.io.aliasAcquire
    qResident.io.aliasGrant << io.queryAliasGrant
    io.queryAliasRelease << qResident.io.aliasRelease
    qResident.io.aliasAcquireOwned := io.queryAliasAcquireOwned
    qResident.io.semanticPackedReadOutstanding :=
      io.querySemanticPackedReadOutstanding
    qResident.io.semanticPackedWriteOutstanding :=
      io.querySemanticPackedWriteOutstanding
    io.queryLeaseHeld := qResident.io.leaseHeld
    io.queryCleanupBusy := qResident.io.busy
    io.queryQuarantined := qResident.io.quarantined
    qkAssembler.io.query << qResident.io.block
    // QkAssembler borrows qResident's 4096-bit output register bank until the
    // last raw score is accepted.  Abort revokes both ends in the same slow
    // clock domain and flushes only the assembler's bounded bookkeeping.
    qkAssembler.io.abort := io.queryAbort
    qkScaler.io.input << qkAssembler.io.score
    io.kvReadRequest << packedKv.io.readRequest
    packedKv.io.readResponse << io.kvReadResponse

    when(io.queryLoad.fire) {
      queryIdentity := io.queryLoad.payload
      queryIdentityValid := True
      queryRetireIssued := False
    }

    // One atomic front-end start prevents a partially accepted pass from
    // leaving the score BRAM, QK array and context sink on different passes.
    val frontReady = qkAssembler.io.pass.ready && qResident.io.lookup.ready &&
      softmax.io.command.ready && writer.io.pass.ready &&
      packedKv.io.command.ready
    val frontValid = phase === Phase.StartFrontEnd && frontReady
    qkAssembler.io.pass.valid := frontValid
    qkAssembler.io.pass.payload := activePass
    qResident.io.lookup.valid := frontValid
    qResident.io.lookup.payload := activePass
    writer.io.pass.valid := frontValid
    writer.io.pass.payload := activePass

    packedKv.io.command.requestId := activePass.requestId
    packedKv.io.command.layer := activePass.layer
    packedKv.io.command.passIndex := activePass.passIndex
    packedKv.io.command.kvHead := activePass.kvHead
    packedKv.io.command.kind := BitNetPackedKvKind.Key
    packedKv.io.command.keyCount := activePass.maxKeyCount

    softmax.io.command.valid := frontValid
    softmax.io.command.requestId := activePass.requestId
    softmax.io.command.layerIndex := activePass.layer
    softmax.io.command.mode := activePass.mode
    softmax.io.command.laneRole := activePass.laneRole
    softmax.io.command.passIndex := activePass.passIndex
    softmax.io.command.kvHead := activePass.kvHead
    softmax.io.command.laneMask := activePass.laneMask
    softmax.io.command.tokenId := activePass.tokenId
    softmax.io.command.queryHeadId := activePass.queryHeadId
    for (lane <- 0 until lanes) {
      softmax.io.command.queryPosition(lane) :=
        activePass.queryPosition(lane).resize(cfg.softmax.positionWidth)
      softmax.io.command.keyCount(lane) :=
        activePass.keyCount(lane).resize(cfg.softmax.keyCountWidth)
    }
    softmax.io.command.maxKeyCount :=
      activePass.maxKeyCount.resize(cfg.softmax.keyCountWidth)
    softmax.io.command.causal := True

    when(frontValid) {
      phase := Phase.RunQk
    }

    // Explicit adapters keep the fixed-point and metadata boundaries visible.
    softmax.io.score.valid := qkScaler.io.output.valid
    qkScaler.io.output.ready := softmax.io.score.ready
    softmax.io.score.requestId := qkScaler.io.output.requestId
    softmax.io.score.layerIndex := qkScaler.io.output.layer
    softmax.io.score.passIndex := qkScaler.io.output.passIndex
    softmax.io.score.kvHead := qkScaler.io.output.kvHead
    softmax.io.score.score := qkScaler.io.output.scoreQ12
    softmax.io.score.laneValid := qkScaler.io.output.laneValid
    softmax.io.score.keyPosition :=
      qkScaler.io.output.keyPosition.resize(cfg.softmax.positionWidth)
    softmax.io.score.positionValid :=
      qkScaler.io.output.protocolValid &&
        qkScaler.io.output.keyPosition < cfg.maxSeq
    softmax.io.score.first := qkScaler.io.output.firstInSequence
    softmax.io.score.last := qkScaler.io.output.lastInSequence

    // K rows are consumed only by QK.  A wrong kind is still transferred with
    // success deasserted, making both the reader and assembler fail closed.
    qkAssembler.io.key.valid := packedKv.io.row.valid &&
      phase === Phase.RunQk
    qkAssembler.io.key.requestId := packedKv.io.row.requestId
    qkAssembler.io.key.layer := packedKv.io.row.layer
    qkAssembler.io.key.passIndex := packedKv.io.row.passIndex
    qkAssembler.io.key.kvHead := packedKv.io.row.kvHead
    qkAssembler.io.key.k := packedKv.io.row.row
    qkAssembler.io.key.kFactorQ16 := packedKv.io.row.factorQ16
    qkAssembler.io.key.keyPosition := packedKv.io.row.keyPosition
    qkAssembler.io.key.keyIndex := packedKv.io.row.keyIndex
    qkAssembler.io.key.firstInSequence :=
      packedKv.io.row.firstInSequence
    qkAssembler.io.key.lastInSequence :=
      packedKv.io.row.lastInSequence
    qkAssembler.io.key.success := packedKv.io.row.success &&
      packedKv.io.row.kind === BitNetPackedKvKind.Key

    for (engine <- 0 until engines) {
      io.matrixSequence(engine) << qkAssembler.io.engineSequence(engine)
      io.matrixI8Beat(engine) << qkAssembler.io.engineBeat(engine)
      qkAssembler.io.engineResult(engine) << io.matrixDotResult(engine)
    }

    // The raw QK score has entered the scaler when qkAssembler.busy falls.
    // The related-clock cluster is observed separately: a delayed active bit
    // or drain indication keeps residentSequence away from the same DSPs.
    when(phase === Phase.RunQk && !qkAssembler.io.busy) {
      phase := Phase.WaitQkArrayIdle
    }
    when(phase === Phase.WaitQkArrayIdle &&
        !io.matrixSequenceActive.orR && !io.matrixDraining.orR) {
      phase := Phase.StartResident
    }

    private val residentFeatureHalves =
      cfg.model.headSize / cfg.continuous.base.siteCount
    val residentStartPhase = phase === Phase.StartResident ||
      phase === Phase.StartResidentHalf1
    val residentStartReady = resident.io.pass.ready &&
      packedKv.io.command.ready
    val residentStartValid = residentStartPhase &&
      residentStartReady
    resident.io.pass.valid := residentStartValid
    resident.io.pass.payload := activePass
    resident.io.featureHalf := phase === Phase.StartResidentHalf1
    packedKv.io.command.valid := frontValid || residentStartValid
    when(residentStartValid) {
      packedKv.io.command.requestId := activePass.requestId
      packedKv.io.command.layer := activePass.layer
      packedKv.io.command.passIndex := activePass.passIndex
      packedKv.io.command.kvHead := activePass.kvHead
      packedKv.io.command.kind := BitNetPackedKvKind.Value
      packedKv.io.command.keyCount := activePass.maxKeyCount
    }
    when(residentStartValid) {
      when(phase === Phase.StartResidentHalf1) {
        phase := Phase.RunResidentHalf1
      } otherwise {
        phase := Phase.RunResident
      }
    }

    // Buffered probabilities are replayed only after the resident scheduler
    // owns the array.  V is independently re-read in this same ordered phase.
    val residentRunPhase = phase === Phase.RunResident ||
      phase === Phase.RunResidentHalf1
    resident.io.probability.valid := softmax.io.probability.valid &&
      residentRunPhase
    softmax.io.probability.ready := resident.io.probability.ready &&
      residentRunPhase
    resident.io.probability.requestId := softmax.io.probability.requestId
    resident.io.probability.layer := softmax.io.probability.layerIndex
    resident.io.probability.laneRole := softmax.io.probability.laneRole
    resident.io.probability.passIndex := softmax.io.probability.passIndex
    resident.io.probability.kvHead := softmax.io.probability.kvHead
    resident.io.probability.laneMask := softmax.io.probability.laneMask
    resident.io.probability.tokenId := softmax.io.probability.tokenId
    resident.io.probability.queryHeadId :=
      softmax.io.probability.queryHeadId
    for (lane <- 0 until lanes) {
      resident.io.probability.queryPosition(lane) :=
        softmax.io.probability.queryPosition(lane).resize(
          cfg.model.sequenceWidth)
      resident.io.probability.keyCount(lane) :=
        softmax.io.probability.keyCount(lane).resize(
          cfg.model.sequenceWidth)
    }
    resident.io.probability.weight := softmax.io.probability.weight
    resident.io.probability.allMasked := softmax.io.probability.allMasked
    resident.io.probability.keyPosition :=
      softmax.io.probability.keyPosition.resize(cfg.model.sequenceWidth)
    resident.io.probability.keyIndex :=
      softmax.io.probability.keyIndex.resize(cfg.model.sequenceWidth)
    resident.io.probability.firstInSequence :=
      softmax.io.probability.firstInSequence
    resident.io.probability.lastInSequence :=
      softmax.io.probability.lastInSequence

    resident.io.value.valid := packedKv.io.row.valid && residentRunPhase
    resident.io.value.requestId := packedKv.io.row.requestId
    resident.io.value.layer := packedKv.io.row.layer
    resident.io.value.passIndex := packedKv.io.row.passIndex
    resident.io.value.kvHead := packedKv.io.row.kvHead
    resident.io.value.value := packedKv.io.row.row
    resident.io.value.valueScaleQ16 := packedKv.io.row.factorQ16
    resident.io.value.keyPosition := packedKv.io.row.keyPosition
    resident.io.value.keyIndex := packedKv.io.row.keyIndex
    resident.io.value.firstInSequence :=
      packedKv.io.row.firstInSequence
    resident.io.value.lastInSequence :=
      packedKv.io.row.lastInSequence
    resident.io.value.success := packedKv.io.row.success &&
      packedKv.io.row.kind === BitNetPackedKvKind.Value
    packedKv.io.row.ready := False
    when(phase === Phase.RunQk) {
      packedKv.io.row.ready := qkAssembler.io.key.ready
    } elsewhen(residentRunPhase) {
      packedKv.io.row.ready := resident.io.value.ready
    }

    if (residentFeatureHalves == 2) {
      when(phase === Phase.RunResident && !resident.io.busy) {
        phase := Phase.WaitResidentHalf0Idle
      }
      when(phase === Phase.WaitResidentHalf0Idle &&
          !io.matrixSequenceActive.orR && !io.matrixDraining.orR) {
        phase := Phase.StartResidentHalf1
      }
    }
    for (engine <- 0 until engines) {
      io.matrixResidentSequence(engine) <<
        resident.io.residentSequence(engine)
      io.matrixResidentBeat(engine) << resident.io.residentBeat(engine)
      writer.io.result(engine) << io.matrixResidentResult(engine)
      // The non-resident SV return belongs to deprecated research modes.  It
      // is not elaborated in the production controller.
      if (!cfg.cluster.productionPruneGenericSv)
        io.matrixSvResult(engine).ready := True
    }
    for (lane <- 0 until lanes) {
      io.writeData(lane) << writer.io.writeData(lane)
    }

    // These events should be impossible by construction.  Keeping an explicit
    // checker makes a future refactor fail closed instead of reintroducing a
    // same-cycle dual owner for the shared matrix sites.
    val qkArrayFire = (0 until engines).map { engine =>
      qkAssembler.io.engineSequence(engine).fire ||
        qkAssembler.io.engineBeat(engine).fire
    }.reduce(_ || _)
    val residentArrayFire = (0 until engines).map { engine =>
      resident.io.residentSequence(engine).fire ||
        resident.io.residentBeat(engine).fire
    }.reduce(_ || _)
    val phaseCollision = qkArrayFire && residentArrayFire
    val kvKindMismatch = packedKv.io.row.valid &&
      ((phase === Phase.RunQk &&
        packedKv.io.row.kind =/= BitNetPackedKvKind.Key) ||
       (residentRunPhase &&
        packedKv.io.row.kind =/= BitNetPackedKvKind.Value))

    val liveFaultBits = Bits(8 bits)
    liveFaultBits := 0
    liveFaultBits(BitNetResidentAttentionFault.PassScheduler) :=
      passScheduler.io.protocolFaultSticky
    liveFaultBits(BitNetResidentAttentionFault.QkAssembly) :=
      qkAssembler.io.protocolFaultSticky ||
        packedKv.io.protocolFaultSticky ||
        qResident.io.protocolFaultSticky
    liveFaultBits(BitNetResidentAttentionFault.QkScaling) :=
      qkScaler.io.protocolFaultSticky
    liveFaultBits(BitNetResidentAttentionFault.BufferedSoftmax) :=
      softmax.io.protocolFault
    liveFaultBits(BitNetResidentAttentionFault.ResidentScheduler) :=
      resident.io.protocolFaultSticky
    liveFaultBits(BitNetResidentAttentionFault.MatrixCluster) :=
      io.matrixProtocolFaultSticky.orR
    liveFaultBits(BitNetResidentAttentionFault.ContextWriter) :=
      writer.io.protocolFaultSticky
    liveFaultBits(BitNetResidentAttentionFault.NumericOrPhase) :=
      resident.io.factorSaturationFault || writer.io.overflowSticky ||
        phaseCollision || kvKindMismatch

    when(liveFaultBits.orR) {
      faultSticky := True
      passFaultBits := passFaultBits | liveFaultBits
    }
    when(liveFaultBits(BitNetResidentAttentionFault.NumericOrPhase)) {
      numericSticky := True
    }

    val doneMatchesPass = writer.io.passDone.requestId ===
      activePass.requestId &&
      writer.io.passDone.layer === activePass.layer &&
      writer.io.passDone.tokenBase === activePass.tokenBase &&
      writer.io.passDone.passIndex === activePass.passIndex &&
      writer.io.passDone.laneMask === activePass.laneMask &&
      writer.io.passDone.first === activePass.first &&
      writer.io.passDone.last === activePass.last
    val doneFaultBits = passFaultBits | liveFaultBits |
      Mux(doneMatchesPass, B(0, 8 bits), B(1 <<
        BitNetResidentAttentionFault.ContextWriter, 8 bits)) |
      Mux(writer.io.passDone.fault, B(1 <<
        BitNetResidentAttentionFault.ContextWriter, 8 bits), B(0, 8 bits))

    writer.io.passDone.ready :=
      (if (residentFeatureHalves == 2)
        phase === Phase.RunResidentHalf1
      else phase === Phase.RunResident)
    when(writer.io.passDone.fire) {
      completedPassCount := completedPassCount + 1
      passFaultBits := doneFaultBits
      when(activePass.last) {
        completionPayload.requestId := activePass.requestId
        completionPayload.layer := activePass.layer
        completionPayload.tokenBase := activePass.tokenBase
        completionPayload.completedLaneMask := activePass.laneMask
        when(activePass.mode === BitNetMode.Decode) {
          completionPayload.completedLaneMask :=
            B(1, cfg.model.tokenParallelism bits)
        }
        completionPayload.success := writer.io.passDone.success &&
          doneMatchesPass && !doneFaultBits.orR
        completionPayload.faultCode := doneFaultBits
        queryRetireIssued := False
        phase := Phase.RetireQuery
      } otherwise {
        phase := Phase.Idle
      }
    }

    qResident.io.retire.valid := phase === Phase.RetireQuery &&
      queryIdentityValid && qResident.io.leaseHeld &&
      !queryRetireIssued && !qResident.io.quarantined
    qResident.io.retire.requestId := queryIdentity.requestId
    qResident.io.retire.layer := queryIdentity.layer
    qResident.io.retire.mode := queryIdentity.mode
    qResident.io.retire.tokenBase := queryIdentity.tokenBase
    qResident.io.retire.tokenId := queryIdentity.tokenId
    qResident.io.retire.tokenMask := queryIdentity.tokenMask
    qResident.io.retire.aliasGeneration := qResident.io.aliasGeneration
    when(qResident.io.retire.fire) {
      queryRetireIssued := True
    }

    // The attention completion cannot escape while QueryLmAlias is still
    // leased.  A prior fail-close/abort may already have started cleanup, in
    // which case there is no retire command to issue; both paths converge on
    // the same observed release barrier.
    when(phase === Phase.RetireQuery &&
        !qResident.io.leaseHeld && !qResident.io.busy) {
      queryIdentityValid := False
      queryRetireIssued := False
      phase := Phase.EmitCompletion
    }

    io.completion.valid := phase === Phase.EmitCompletion
    io.completion.payload := completionPayload
    when(io.completion.fire) {
      completedCommandCount := completedCommandCount + 1
      phase := Phase.Idle
    }

    io.busy := phase =/= Phase.Idle || passScheduler.io.busy ||
      passScheduler.io.pass.valid || qResident.io.busy
    io.protocolFaultSticky := faultSticky
    io.numericFaultSticky := numericSticky
    io.faultBits := passFaultBits | liveFaultBits
    io.acceptedCommands := passScheduler.io.acceptedCommands
    io.completedCommands := completedCommandCount
    io.completedPasses := completedPassCount
  }
}

/**
  * Stand-alone compatibility shell used by focused attention tests.
  *
  * The production model top instantiates [[BitNetResidentAttentionController]]
  * behind the model-wide matrix lease fabric.  This shell deliberately owns
  * exactly one cluster so the complete attention transaction can still be
  * elaborated and simulated in isolation; it must not be nested beside a
  * second projection cluster in a deployable top.
  */
class BitNetResidentAttentionSubsystem(
    cfg: BitNetResidentAttentionSubsystemConfig =
      BitNetResidentAttentionSubsystemConfig(),
    useVendorPrimitive: Boolean = true) extends Component {
  private val lanes = cfg.model.tokenParallelism
  private val engines = cfg.model.engineCount

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val command = slave(Stream(
      BitNetAttentionScheduleCommand(cfg.model)))
    val queryLoad = slave(Stream(BitNetQResidentLoadCommand(cfg.qResident)))
    val querySample = slave(Stream(
      BitNetQkvQuantizedSample(cfg.qResident.qkv)))
    val queryLoadCompletion = master(Stream(
      BitNetQResidentLoadCompletion(cfg.qResident)))
    val queryAbort = in Bool()
    val queryPackedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(cfg.unifiedScratchpad)))
    val queryPackedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(cfg.unifiedScratchpad)))
    val queryPackedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(cfg.unifiedScratchpad)))
    val queryPackedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(
        cfg.unifiedScratchpad)))
    val queryAliasAcquire = master(Stream(
      BitNetUnifiedResidentAliasAcquire(cfg.unifiedScratchpad)))
    val queryAliasGrant = slave(Stream(
      BitNetUnifiedResidentAliasGrant(cfg.unifiedScratchpad)))
    val queryAliasRelease = master(Stream(
      BitNetUnifiedResidentAliasRelease(cfg.unifiedScratchpad)))
    val queryAliasAcquireOwned = in Bool()
    val querySemanticPackedReadOutstanding = in Bool()
    val querySemanticPackedWriteOutstanding = in Bool()
    val kvReadRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar)))
    val kvReadResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar)))
    val writeData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(
        cfg.model, cfg.activationLayout))), lanes)
    val completion = master(Stream(BitNetAttentionCompletion(cfg.model)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val numericFaultSticky = out Bool()
    val faultBits = out Bits(8 bits)
    val acceptedCommands = out UInt(32 bits)
    val completedCommands = out UInt(32 bits)
    val completedPasses = out UInt(32 bits)
  }

  val controller = new BitNetResidentAttentionController(
    cfg, useVendorPrimitive)
  val cluster = new BitNetTp4MatrixCluster(
    cfg.cluster, useVendorPrimitive)
  // Preserve the stand-alone shell's public simulation clock handles.  Bind
  // them to top-level pins rather than child input aliases, whose simulation
  // nodes are not guaranteed to survive hierarchy pruning.
  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))
  val fastCd = ClockDomain(
    clock = io.fastClk,
    reset = io.fastResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  controller.io.slowClk := io.slowClk
  controller.io.slowResetn := io.slowResetn
  cluster.io.slowClk := io.slowClk
  cluster.io.slowResetn := io.slowResetn
  cluster.io.fastClk := io.fastClk
  cluster.io.fastResetn := io.fastResetn

  controller.io.command << io.command
  controller.io.queryLoad << io.queryLoad
  controller.io.querySample << io.querySample
  io.queryLoadCompletion << controller.io.queryLoadCompletion
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
  io.kvReadRequest << controller.io.kvReadRequest
  controller.io.kvReadResponse << io.kvReadResponse
  for (lane <- 0 until lanes) {
    io.writeData(lane) << controller.io.writeData(lane)
  }
  io.completion << controller.io.completion

  for (engine <- 0 until engines) {
    cluster.io.sequence(engine) << controller.io.matrixSequence(engine)
    // Attention-only stand-alone wrapper: QK owns the typed I8 mailbox and
    // resident SV owns its dedicated stream.  MAP0 and legacy generic traffic
    // remain quiescent.
    cluster.io.map0Beat(engine).valid := False
    cluster.io.map0Beat(engine).payload :=
      BitNetMap0PairMatrixBeat(BitNetContinuousMatrixFeederConfig(
        matrix = cfg.cluster.matrix)).getZero
    cluster.io.i8Beat(engine) << controller.io.matrixI8Beat(engine)
    if (!cfg.cluster.productionPruneLegacyBeat) {
      cluster.io.beat(engine).valid := False
      cluster.io.beat(engine).payload := BitNetContinuousMatrixBeat(
        cfg.cluster.matrix,
        includeGenericSv = !cfg.cluster.productionPruneGenericSv).getZero
    }
    controller.io.matrixDotResult(engine) << cluster.io.dotResult(engine)
    if (!cfg.cluster.productionPruneGenericSv)
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

  io.busy := controller.io.busy
  io.protocolFaultSticky := controller.io.protocolFaultSticky
  io.numericFaultSticky := controller.io.numericFaultSticky
  io.faultBits := controller.io.faultBits
  io.acceptedCommands := controller.io.acceptedCommands
  io.completedCommands := controller.io.completedCommands
  io.completedPasses := controller.io.completedPasses
}

object GenerateBitNetResidentAttentionSubsystem extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-attention-subsystem")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetResidentAttentionSubsystem())
}
