package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Production geometry of the single shared linear/attention physical plane. */
case class BitNetResidentSharedTp4ComputeDatapathConfig(
    pipeline: BitNetResidentQkvAttentionPipelineConfig =
      BitNetResidentQkvAttentionPipelineConfig(),
    leaseTagWidth: Int = 8
) {
  val model: BitNetConfig = pipeline.model
  val attention: BitNetResidentAttentionSubsystemConfig = pipeline.attention
  val continuous: BitNetContinuousMatrixConfig = attention.continuous
  val feeder: BitNetContinuousMatrixFeederConfig =
    BitNetContinuousMatrixFeederConfig(
      matrix = continuous,
      tokenLanes = model.tokenParallelism,
      engineCount = model.engineCount)
  val linearEndpoint: BitNetResidentLinearEndpointConfig =
    BitNetResidentLinearEndpointConfig(model = model, feeder = feeder)
  val projectionAssembler: BitNetResidentProjectionFeatureAssemblerConfig =
    BitNetResidentProjectionFeatureAssemblerConfig(linearEndpoint)
  val fusedResidual: BitNetResidentFusedResidualEndpointConfig =
    BitNetResidentFusedResidualEndpointConfig(
      projectionAssembler, attention.activationLayout)
  val lease: BitNetTp4MatrixClusterConfig = attention.cluster
  val arithmetic: BitNetSharedSpuArithmeticPoolConfig = pipeline.qkv.arithmetic
  val exposedArithmeticClients: Int = arithmetic.clientCount - 1

  require(model.tokenParallelism == model.engineCount *
    continuous.tokenLanesPerEngine)
  require(continuous eq feeder.matrix)
  require(lease.matrix eq continuous)
  require(lease.logicalEngineCount == model.engineCount)
  require(lease.physicalEngineCount == model.physicalMatrixEngineCount)
  require(lease.svGroupLanes == attention.residentGroupLanes)
  require(lease.productionPruneGenericSv && lease.productionPruneLegacyBeat,
    "the production plane must prune both legacy wide matrix mailboxes")
  require(exposedArithmeticClients == 5,
    "projection scale is private and exactly five SPU clients are external")
  require(leaseTagWidth >= 2 && leaseTagWidth <= continuous.sequenceIdWidth)
}

object BitNetResidentSharedTp4Fault {
  val InvalidLinearLaunch = 0
  val Feeder = 1
  val LinearEndpoint = 2
  val ProjectionAssembler = 3
  val FusedResidual = 4
  val QkvController = 5
  val AttentionController = 6
  val Lease = 7
  val MatrixCluster = 8
  val SpuArithmetic = 9
  val CompletionIdentity = 10
  val EarlyQkvDot = 11
  val UnexpectedMatrixReturn = 12
}

/** Compile-time-only completion injection for the focused control Spec. */
case class BitNetResidentSharedTp4ComputeDatapathHarnessIo(
    cfg: BitNetResidentSharedTp4ComputeDatapathConfig) extends Bundle {
  val qkvRoundStarted = out Bool()
  val linearEndpointStarted = out Bool()
  val feederStarted = out Bool()
  val attentionStarted = out Bool()
  val queryAbortPublished = out Bool()
  val localFailCloseNow = out Bool()
  val controllerQkvCompletion = slave(Stream(
    BitNetResidentQkvRoundCompletion(cfg.pipeline)))
  val controllerAttentionCompletion = slave(Stream(
    BitNetAttentionCompletion(cfg.model)))
  val linearEndpointCompletion = slave(Stream(
    BitNetResidentLinearCompletion(cfg.linearEndpoint)))
  val projectionAssemblyCompletion = slave(Stream(
    BitNetResidentProjectionAssemblyCompletion(cfg.projectionAssembler)))
  val fusedResidualCompletion = slave(Stream(
    BitNetResidentFusedResidualCompletion(cfg.fusedResidual)))
}

/**
  * Production-only shared TP4 compute datapath.
  *
  * Exactly one [[BitNetTp4MatrixLeaseFabric]] owns the sole TP4 cluster and
  * its two continuous TP2 arrays.  All segmented Q/K/V, O, GateUp, Down and
  * LM transactions enter through `linearLaunch`; the legacy composite QKV
  * transaction does not exist at this boundary.  Typed final Q/K/V pairs are
  * consumed by the external segmented bridge, which returns `qkvDot` directly
  * to the one resident QKV/attention controller.
  *
  * QKV postprocess does not own the matrix lease, so scale/RoPE/KV append may
  * overlap a later linear projection.  Resident attention, in contrast,
  * starts only after acquiring the attention facet of the same whole-cluster
  * lease.  Linear and attention therefore cannot double-own an array.
  */
class BitNetResidentSharedTp4ComputeDatapath(
    cfg: BitNetResidentSharedTp4ComputeDatapathConfig =
      BitNetResidentSharedTp4ComputeDatapathConfig(),
    useVendorPrimitive: Boolean = true,
    controlHarness: Boolean = false,
    trustedLinearLaunchIngress: Boolean = false) extends Component {
  private val engines = cfg.model.engineCount
  private val lanes = cfg.model.tokenParallelism
  private val sharedQuantCfg = BitNetSharedDynamicI8QuantizerConfig()

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val linearLaunch = slave(Stream(
      BitNetResidentLinearProgram(cfg.linearEndpoint)))
    val linearResult = master(Stream(
      BitNetResidentFinalFeaturePair(cfg.linearEndpoint)))
    val linearCompletion = master(Stream(
      BitNetResidentLinearCompletion(cfg.linearEndpoint)))

    val qkvRound = slave(Stream(BitNetResidentQkvRoundCommand(cfg.pipeline)))
    val qkvDot = slave(Stream(BitNetQkvTp4DotBeat(cfg.pipeline.qkv)))
    val qkvCompletion = master(Stream(
      BitNetResidentQkvRoundCompletion(cfg.pipeline)))
    val qkvQuantCommand = master(Stream(
      BitNetSharedDynamicQuantCommand(sharedQuantCfg)))
    val qkvQuantMeasure = master(Stream(
      BitNetSharedDynamicQuantMeasureBeat(sharedQuantCfg)))
    val qkvQuantReplayRequest = slave(Stream(
      BitNetSharedDynamicQuantReplayRequest(sharedQuantCfg)))
    val qkvQuantReplay = master(Stream(
      BitNetSharedDynamicQuantReplayBeat(sharedQuantCfg)))
    val qkvQuantResult = slave(Stream(
      BitNetSharedDynamicQuantResultBeat(sharedQuantCfg)))
    val qkvQuantDone = slave(Stream(
      BitNetSharedDynamicQuantDone(sharedQuantCfg)))

    val attentionCommand = slave(Stream(
      BitNetAttentionScheduleCommand(cfg.model)))
    val attentionCompletion = master(Stream(
      BitNetAttentionCompletion(cfg.model)))

    // QueryLmAlias semantic client.  The full-layer packed router owns the
    // shared frontend; this datapath exposes only Query's typed endpoint.
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
    /** Query acquire has been selected/locked by the shared alias router. */
    val queryAliasAcquireOwned = in Bool()
    val querySemanticPackedReadOutstanding = in Bool()
    val querySemanticPackedWriteOutstanding = in Bool()
    /** Parent packed/alias fabric failure: revoke Query valids and drain the
      * retained lease before the model-wide reset/recovery boundary. */
    val queryExternalAbort = in Bool()

    // Typed PL-resident feeder and memory boundaries.
    val map0RawGroup = slave(Stream(BitNetMap0RawGroupTile(
      cfg.feeder, cfg.model.ffnSize)))
    val i8WeightBeat = slave(Stream(BitNetI8WeightBeat(cfg.feeder)))
    val activationTile = slave(Stream(
      BitNetNormalizedActivationTile(cfg.feeder)))

    val kvWriteRequest = master(Stream(
      BitNetClientBankWriteRequest(cfg.pipeline.crossbar)))
    val kvWriteData = master(Stream(
      BitNetClientBankWriteData(cfg.pipeline.crossbar)))
    val kvWriteResponse = slave(Stream(
      BitNetClientBankWriteResponse(cfg.pipeline.crossbar)))
    val kvReadRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.pipeline.crossbar)))
    val kvReadResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.pipeline.crossbar)))
    val contextWrite = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(
        cfg.model, cfg.attention.activationLayout))), lanes)
    val residualReadCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(
        cfg.model, cfg.attention.activationLayout))), lanes)
    val residualReadData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(
        cfg.model, cfg.attention.activationLayout))), lanes)
    val residualWriteData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(
        cfg.model, cfg.attention.activationLayout))), lanes)

    // Physical client zero is private projection scaling.  Ports zero through
    // four below map to RMSNorm, FFN gate, dynamic quant, RoPE and head quant.
    val spuAcquire = Vec(slave(Stream(
      BitNetSpuArithmeticLeaseRequest(cfg.arithmetic))),
      cfg.exposedArithmeticClients)
    val spuOperand = Vec(slave(Stream(
      BitNetSpuArithmeticOperandBeat(cfg.arithmetic))),
      cfg.exposedArithmeticClients)
    val spuResult = Vec(master(Stream(
      BitNetSpuArithmeticResultBeat(cfg.arithmetic))),
      cfg.exposedArithmeticClients)
    val spuRelease = Vec(slave(Stream(
      BitNetSpuArithmeticLeaseRelease(cfg.arithmetic))),
      cfg.exposedArithmeticClients)
    val spuClearFault = in Bool()
    val spuPoolBusy = out Bool()
    val spuPoolOwned = out Bits(cfg.arithmetic.clientCount bits)
    val spuPoolActiveOwner = out UInt(cfg.arithmetic.clientWidth bits)
    val spuPoolFault = out Bool()

    val qkvImageReady = out Bool()
    val busy = out Bool()
    val fatalFailStop = out Bool()
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(16 bits)
    val linearLeaseGranted = out Bool()
    val attentionLeaseGranted = out Bool()
    val leaseReleaseSafe = out Bool()
    val leaseTag = out UInt(cfg.leaseTagWidth bits)
    val matrixSequenceActive = out Bits(engines bits)
    val matrixDraining = out Bits(engines bits)
    val activeLinearKind = out(BitNetResidentLinearKind())

    val harness = if (controlHarness)
      BitNetResidentSharedTp4ComputeDatapathHarnessIo(cfg) else null
  }

  // The lease fabric is the unique owner of the unique physical cluster.
  val leaseFabric = new BitNetTp4MatrixLeaseFabric(
    cfg.lease, useVendorPrimitive, cfg.leaseTagWidth)
  val controller = new BitNetResidentQkvAttentionController(
    cfg.pipeline, useVendorPrimitive)

  leaseFabric.io.slowClk := io.slowClk
  leaseFabric.io.slowResetn := io.slowResetn
  leaseFabric.io.fastClk := io.fastClk
  leaseFabric.io.fastResetn := io.fastResetn
  controller.io.slowClk := io.slowClk
  controller.io.slowResetn := io.slowResetn
  io.qkvQuantCommand << controller.io.quantCommand
  io.qkvQuantMeasure << controller.io.quantMeasure
  controller.io.quantReplayRequest << io.qkvQuantReplayRequest
  io.qkvQuantReplay << controller.io.quantReplay
  controller.io.quantResult << io.qkvQuantResult
  controller.io.quantDone << io.qkvQuantDone

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  val slowArea = new ClockingArea(slowCd) {
    val feeder = new BitNetContinuousMatrixFeeder(cfg.feeder)
    val finalEndpoint = new BitNetResidentLinearResultEndpoint(
      cfg.linearEndpoint)
    val projectionFeatureAssembler =
      new BitNetResidentProjectionFeatureAssembler(cfg.projectionAssembler)
    val fusedResidual =
      new BitNetResidentFusedResidualEndpoint(cfg.fusedResidual)
    val arithmeticPool = new BitNetSharedSpuArithmeticPool(cfg.arithmetic)

    // O/Down share the controller's sole projection scale datapath.
    controller.io.linearScaleInput << fusedResidual.io.scaleInput
    fusedResidual.io.scaleOutput << controller.io.linearScaleOutput
    for (lane <- 0 until lanes) {
      io.residualReadCommand(lane) << fusedResidual.io.readCommand(lane)
      fusedResidual.io.readData(lane) << io.residualReadData(lane)
      io.residualWriteData(lane) << fusedResidual.io.writeData(lane)
    }

    arithmeticPool.io.acquire(
      BitNetSpuArithmeticClient.ProjectionScale) <<
      controller.io.projectionScaleAcquire
    arithmeticPool.io.operand(
      BitNetSpuArithmeticClient.ProjectionScale) <<
      controller.io.projectionScaleOperand
    controller.io.projectionScaleResult << arithmeticPool.io.result(
      BitNetSpuArithmeticClient.ProjectionScale)
    arithmeticPool.io.release(
      BitNetSpuArithmeticClient.ProjectionScale) <<
      controller.io.projectionScaleRelease
    controller.io.spuPoolFault := arithmeticPool.io.protocolFaultSticky

    for (external <- 0 until cfg.exposedArithmeticClients) {
      val physicalClient = external + 1
      arithmeticPool.io.acquire(physicalClient) << io.spuAcquire(external)
      arithmeticPool.io.operand(physicalClient) << io.spuOperand(external)
      io.spuResult(external) << arithmeticPool.io.result(physicalClient)
      arithmeticPool.io.release(physicalClient) << io.spuRelease(external)
    }
    arithmeticPool.io.clearFault := io.spuClearFault
    io.spuPoolBusy := arithmeticPool.io.busy
    io.spuPoolOwned := arithmeticPool.io.owned
    io.spuPoolActiveOwner := arithmeticPool.io.activeOwner
    io.spuPoolFault := arithmeticPool.io.protocolFaultSticky

    object Phase extends SpinalEnum(binarySequential) {
      val Idle, AcquireLinear, StartLinearConsumers, StartFeeder, RunLinear,
          ReleaseLinear, AcquireAttention, StartAttention, RunAttention,
          ReleaseAttention, FatalDrain = newElement()
    }
    val phase = Reg(Phase()) init Phase.Idle
    // These ownership predicates qualify both result routing and fault
    // detection.  Keeping an out-of-phase matrix return stalled for one slow
    // clock lets the registered fail-stop below sever the related-clock
    // result-valid -> global abort combinational path.
    val linearOwnedPhase = phase === Phase.AcquireLinear ||
      phase === Phase.StartLinearConsumers || phase === Phase.StartFeeder ||
      phase === Phase.RunLinear || phase === Phase.ReleaseLinear
    val attentionOwnedPhase = phase === Phase.AcquireAttention ||
      phase === Phase.StartAttention || phase === Phase.RunAttention ||
      phase === Phase.ReleaseAttention
    val activeLinear = Reg(
      BitNetResidentLinearProgram(cfg.linearEndpoint)) init
      BitNetResidentLinearProgram(cfg.linearEndpoint).getZero
    // Expand only as combinational wires at the existing legacy consumer
    // boundary.  The shared owner context itself remains the 282-bit program.
    val expandedActiveLinear = BitNetResidentLinearProgramCodec.expand(
      activeLinear, cfg.linearEndpoint)
    val heldAttention = Reg(BitNetAttentionScheduleCommand(cfg.model)) init
      BitNetAttentionScheduleCommand(cfg.model).getZero
    val heldQkvRound = Reg(BitNetResidentQkvRoundCommand(cfg.pipeline)) init
      BitNetResidentQkvRoundCommand(cfg.pipeline).getZero

    val qkvRoundActive = RegInit(False)
    val imageMetadataValid = RegInit(False)
    val fatalLatched = RegInit(False)
    // The controller-facing abort is a registered publication boundary.
    // Internal fatal remains asserted through fatalLatched, while a pulsed
    // external abort is delayed by exactly one slow clock and then rearms.
    val queryAbortPublished = RegInit(False)
    val localFailCloseNow = Bool()
    val faultSticky = RegInit(False)
    val faultBitsReg = Reg(Bits(16 bits)) init 0
    val unexpectedLinearReturnPending = RegInit(False)

    val endpointDone = RegInit(False)
    val projectionDone = RegInit(False)
    val residualDone = RegInit(False)
    val endpointCompletionReg = Reg(
      BitNetResidentLinearCompletion(cfg.linearEndpoint)) init
      BitNetResidentLinearCompletion(cfg.linearEndpoint).getZero
    val projectionCompletionReg = Reg(
      BitNetResidentProjectionAssemblyCompletion(
        cfg.projectionAssembler)) init
      BitNetResidentProjectionAssemblyCompletion(
        cfg.projectionAssembler).getZero
    val residualCompletionReg = Reg(
      BitNetResidentFusedResidualCompletion(cfg.fusedResidual)) init
      BitNetResidentFusedResidualCompletion(cfg.fusedResidual).getZero
    val attentionCompletionReg = Reg(
      BitNetAttentionCompletion(cfg.model)) init
      BitNetAttentionCompletion(cfg.model).getZero

    val linearCompletionValid = RegInit(False)
    val linearCompletionReg = Reg(
      BitNetResidentLinearCompletion(cfg.linearEndpoint)) init
      BitNetResidentLinearCompletion(cfg.linearEndpoint).getZero
    val fatalLinearReportPending = RegInit(False)
    io.linearCompletion.valid := linearCompletionValid
    io.linearCompletion.payload := linearCompletionReg
    when(io.linearCompletion.fire) { linearCompletionValid := False }

    val qkvCompletionValid = RegInit(False)
    val qkvCompletionReg = Reg(
      BitNetResidentQkvRoundCompletion(cfg.pipeline)) init
      BitNetResidentQkvRoundCompletion(cfg.pipeline).getZero
    val fatalQkvReportPending = RegInit(False)
    io.qkvCompletion.valid := qkvCompletionValid
    io.qkvCompletion.payload := qkvCompletionReg
    when(io.qkvCompletion.fire) { qkvCompletionValid := False }

    val attentionCompletionValid = RegInit(False)
    val outwardAttentionCompletion = Reg(
      BitNetAttentionCompletion(cfg.model)) init
      BitNetAttentionCompletion(cfg.model).getZero
    val fatalAttentionReportPending = RegInit(False)
    io.attentionCompletion.valid := attentionCompletionValid
    io.attentionCompletion.payload := outwardAttentionCompletion
    when(io.attentionCompletion.fire) { attentionCompletionValid := False }

    // Resident KV/context and activation clients never cross a PS callback.
    io.kvWriteRequest << controller.io.kvWriteRequest
    io.kvWriteData << controller.io.kvWriteData
    controller.io.kvWriteResponse << io.kvWriteResponse
    io.kvReadRequest << controller.io.kvReadRequest
    controller.io.kvReadResponse << io.kvReadResponse
    for (lane <- 0 until lanes) {
      io.contextWrite(lane) << controller.io.writeData(lane)
    }
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

    // ------------------------------------------------------------------
    // One physical cluster, with mutually exclusive linear/attention routes.
    // ------------------------------------------------------------------
    for (engine <- 0 until engines) {
      leaseFabric.io.linearSequence(engine).valid :=
        feeder.io.sequence(engine).valid && !fatalLatched
      leaseFabric.io.linearSequence(engine).payload :=
        feeder.io.sequence(engine).payload
      feeder.io.sequence(engine).ready :=
        leaseFabric.io.linearSequence(engine).ready && !fatalLatched
      leaseFabric.io.linearMap0Beat(engine).valid :=
        feeder.io.map0Beat(engine).valid && !fatalLatched
      leaseFabric.io.linearMap0Beat(engine).payload :=
        feeder.io.map0Beat(engine).payload
      feeder.io.map0Beat(engine).ready :=
        leaseFabric.io.linearMap0Beat(engine).ready && !fatalLatched
      leaseFabric.io.linearI8Beat(engine).valid :=
        feeder.io.i8Beat(engine).valid && !fatalLatched
      leaseFabric.io.linearI8Beat(engine).payload :=
        feeder.io.i8Beat(engine).payload
      feeder.io.i8Beat(engine).ready :=
        leaseFabric.io.linearI8Beat(engine).ready && !fatalLatched

      finalEndpoint.io.engineResult(engine).valid :=
        leaseFabric.io.linearDotResult(engine).valid && linearOwnedPhase &&
          !fatalLatched
      finalEndpoint.io.engineResult(engine).payload :=
        leaseFabric.io.linearDotResult(engine).payload
      leaseFabric.io.linearDotResult(engine).ready := Mux(
        fatalLatched || unexpectedLinearReturnPending,
        True,
        finalEndpoint.io.engineResult(engine).ready && linearOwnedPhase)

      leaseFabric.io.attentionSequence(engine).valid :=
        controller.io.matrixSequence(engine).valid && !fatalLatched
      leaseFabric.io.attentionSequence(engine).payload :=
        controller.io.matrixSequence(engine).payload
      controller.io.matrixSequence(engine).ready :=
        leaseFabric.io.attentionSequence(engine).ready && !fatalLatched
      leaseFabric.io.attentionI8Beat(engine).valid :=
        controller.io.matrixI8Beat(engine).valid && !fatalLatched
      leaseFabric.io.attentionI8Beat(engine).payload :=
        controller.io.matrixI8Beat(engine).payload
      controller.io.matrixI8Beat(engine).ready :=
        leaseFabric.io.attentionI8Beat(engine).ready && !fatalLatched
      controller.io.matrixDotResult(engine).valid :=
        leaseFabric.io.attentionDotResult(engine).valid && !fatalLatched
      controller.io.matrixDotResult(engine).payload :=
        leaseFabric.io.attentionDotResult(engine).payload
      leaseFabric.io.attentionDotResult(engine).ready := Mux(
        fatalLatched, True, controller.io.matrixDotResult(engine).ready)

      leaseFabric.io.attentionResidentSequence(engine).valid :=
        controller.io.matrixResidentSequence(engine).valid && !fatalLatched
      leaseFabric.io.attentionResidentSequence(engine).payload :=
        controller.io.matrixResidentSequence(engine).payload
      controller.io.matrixResidentSequence(engine).ready :=
        leaseFabric.io.attentionResidentSequence(engine).ready &&
          !fatalLatched
      leaseFabric.io.attentionResidentBeat(engine).valid :=
        controller.io.matrixResidentBeat(engine).valid && !fatalLatched
      leaseFabric.io.attentionResidentBeat(engine).payload :=
        controller.io.matrixResidentBeat(engine).payload
      controller.io.matrixResidentBeat(engine).ready :=
        leaseFabric.io.attentionResidentBeat(engine).ready && !fatalLatched
      controller.io.matrixResidentResult(engine).valid :=
        leaseFabric.io.attentionResidentResult(engine).valid && !fatalLatched
      controller.io.matrixResidentResult(engine).payload :=
        leaseFabric.io.attentionResidentResult(engine).payload
      leaseFabric.io.attentionResidentResult(engine).ready := Mux(
        fatalLatched, True,
        controller.io.matrixResidentResult(engine).ready)
    }
    controller.io.matrixSequenceActive := Mux(
      leaseFabric.io.attentionGranted,
      leaseFabric.io.sequenceActive, B(0, engines bits))
    controller.io.matrixDraining := Mux(
      leaseFabric.io.attentionGranted,
      leaseFabric.io.draining, B(0, engines bits))
    controller.io.matrixProtocolFaultSticky :=
      leaseFabric.io.clusterProtocolFaultSticky

    // O/Down terminate in the fused residual writer.  Segmented Q/K/V,
    // GateUp and LM remain typed linear-result streams.
    val fusedProjectionKind =
      activeLinear.kind === BitNetResidentLinearKind.AttentionOutput ||
        activeLinear.kind === BitNetResidentLinearKind.Down
    projectionFeatureAssembler.io.input.valid :=
      finalEndpoint.io.output.valid && fusedProjectionKind && !fatalLatched
    projectionFeatureAssembler.io.input.payload :=
      finalEndpoint.io.output.payload
    io.linearResult.valid := finalEndpoint.io.output.valid &&
      !fusedProjectionKind && !fatalLatched
    io.linearResult.payload := finalEndpoint.io.output.payload
    finalEndpoint.io.output.ready := True
    when(!fatalLatched) {
      finalEndpoint.io.output.ready := io.linearResult.ready
      when(fusedProjectionKind) {
        finalEndpoint.io.output.ready :=
          projectionFeatureAssembler.io.input.ready
      }
    }
    fusedResidual.io.input.valid :=
      projectionFeatureAssembler.io.output.valid && !fatalLatched
    fusedResidual.io.input.payload :=
      projectionFeatureAssembler.io.output.payload
    projectionFeatureAssembler.io.output.ready := Mux(
      fatalLatched, True, fusedResidual.io.input.ready)

    val linearDataOpen = phase === Phase.RunLinear && !fatalLatched &&
      (if (controlHarness) False else True)
    feeder.io.map0RawGroup.valid := io.map0RawGroup.valid && linearDataOpen
    feeder.io.map0RawGroup.payload := io.map0RawGroup.payload
    io.map0RawGroup.ready := feeder.io.map0RawGroup.ready && linearDataOpen
    feeder.io.i8WeightBeat.valid := io.i8WeightBeat.valid && linearDataOpen
    feeder.io.i8WeightBeat.payload := io.i8WeightBeat.payload
    io.i8WeightBeat.ready := feeder.io.i8WeightBeat.ready && linearDataOpen
    feeder.io.activationTile.valid := io.activationTile.valid && linearDataOpen
    feeder.io.activationTile.payload := io.activationTile.payload
    io.activationTile.ready := feeder.io.activationTile.ready && linearDataOpen

    // ------------------------------------------------------------------
    // Direct resident QKV ingress.  It is independent of matrix lease state.
    // ------------------------------------------------------------------
    val qkvRoundCanStart = !qkvRoundActive && !imageMetadataValid &&
      !qkvCompletionValid && !localFailCloseNow
    if (controlHarness) {
      controller.io.qkvRound.valid := False
      controller.io.qkvRound.payload := io.qkvRound.payload
      io.qkvRound.ready := qkvRoundCanStart
      controller.io.qkvDot.valid := False
      controller.io.qkvDot.payload := io.qkvDot.payload
      io.qkvDot.ready := True
      controller.io.qkvCompletion.ready := True
    } else {
      controller.io.qkvRound.valid := io.qkvRound.valid && qkvRoundCanStart
      controller.io.qkvRound.payload := io.qkvRound.payload
      io.qkvRound.ready := controller.io.qkvRound.ready && qkvRoundCanStart
      controller.io.qkvDot.valid := io.qkvDot.valid && qkvRoundActive &&
        !fatalLatched
      controller.io.qkvDot.payload := io.qkvDot.payload
      io.qkvDot.ready := Mux(fatalLatched || !qkvRoundActive,
        True, controller.io.qkvDot.ready)
    }
    when(io.qkvRound.fire) {
      heldQkvRound := io.qkvRound.payload
      qkvRoundActive := True
      imageMetadataValid := False
    }
    val earlyQkvDot = io.qkvDot.fire && !qkvRoundActive && !fatalLatched

    val selectedQkvCompletion = if (controlHarness)
      io.harness.controllerQkvCompletion else controller.io.qkvCompletion
    selectedQkvCompletion.ready := qkvRoundActive &&
      !qkvCompletionValid && !fatalLatched
    if (controlHarness) {
      io.harness.qkvRoundStarted := io.qkvRound.fire
    }

    val qkvCompletionMatches =
      selectedQkvCompletion.requestId === heldQkvRound.requestId &&
        selectedQkvCompletion.layer === heldQkvRound.layer &&
        selectedQkvCompletion.tokenBase === heldQkvRound.tokenBase &&
        selectedQkvCompletion.tokenMask === heldQkvRound.tokenMask

    // ------------------------------------------------------------------
    // Linear/attention transaction starts and compile-time harness defaults.
    // ------------------------------------------------------------------
    finalEndpoint.io.launch.valid := False
    finalEndpoint.io.launch.payload := expandedActiveLinear
    projectionFeatureAssembler.io.launch.valid := False
    projectionFeatureAssembler.io.launch.payload := expandedActiveLinear
    fusedResidual.io.launch.valid := False
    fusedResidual.io.launch.payload := expandedActiveLinear
    feeder.io.task.valid := False
    feeder.io.task.payload := expandedActiveLinear.matrixTask
    controller.io.attentionCommand.valid := False
    controller.io.attentionCommand.payload := heldAttention

    val linearEndpointStarted = Bool()
    val feederStarted = Bool()
    val attentionStarted = Bool()
    linearEndpointStarted := False
    feederStarted := False
    attentionStarted := False

    if (controlHarness) {
      io.harness.linearEndpointStarted := linearEndpointStarted
      io.harness.feederStarted := feederStarted
      io.harness.attentionStarted := attentionStarted
      finalEndpoint.io.completion.ready := True
      projectionFeatureAssembler.io.completion.ready := True
      fusedResidual.io.completion.ready := True
      controller.io.attentionCompletion.ready := True
    }

    val selectedEndpointCompletion = if (controlHarness)
      io.harness.linearEndpointCompletion else finalEndpoint.io.completion
    val selectedProjectionCompletion = if (controlHarness)
      io.harness.projectionAssemblyCompletion
    else projectionFeatureAssembler.io.completion
    val selectedResidualCompletion = if (controlHarness)
      io.harness.fusedResidualCompletion else fusedResidual.io.completion
    val selectedAttentionCompletion = if (controlHarness)
      io.harness.controllerAttentionCompletion
    else controller.io.attentionCompletion

    selectedEndpointCompletion.ready := phase === Phase.RunLinear &&
      !endpointDone && !fatalLatched
    selectedProjectionCompletion.ready := phase === Phase.RunLinear &&
      fusedProjectionKind && !projectionDone && !fatalLatched
    selectedResidualCompletion.ready := phase === Phase.RunLinear &&
      fusedProjectionKind && !residualDone && !fatalLatched
    selectedAttentionCompletion.ready := phase === Phase.RunAttention &&
      !fatalLatched

    val endpointCompletionMatches =
      selectedEndpointCompletion.requestId === activeLinear.requestId &&
        selectedEndpointCompletion.layer === activeLinear.layer &&
        selectedEndpointCompletion.kind === activeLinear.kind &&
        selectedEndpointCompletion.sequenceId === activeLinear.sequenceId &&
        !selectedEndpointCompletion.fatal
    val projectionCompletionMatches =
      selectedProjectionCompletion.requestId === activeLinear.requestId &&
        selectedProjectionCompletion.layer === activeLinear.layer &&
        selectedProjectionCompletion.kind === activeLinear.kind &&
        selectedProjectionCompletion.sequenceId === activeLinear.sequenceId
    val residualCompletionMatches =
      selectedResidualCompletion.requestId === activeLinear.requestId &&
        selectedResidualCompletion.layer === activeLinear.layer &&
        selectedResidualCompletion.kind === activeLinear.kind &&
        selectedResidualCompletion.sequenceId === activeLinear.sequenceId
    val attentionCompletionMatches =
      selectedAttentionCompletion.requestId === heldAttention.requestId &&
        selectedAttentionCompletion.layer === heldAttention.layer &&
        selectedAttentionCompletion.tokenBase === heldAttention.tokenBase &&
        selectedAttentionCompletion.completedLaneMask ===
        Mux(heldAttention.mode === BitNetMode.Decode,
          B(1, cfg.model.tokenParallelism bits), heldAttention.activeTokenMask)

    // Component/completion faults are evaluated before any new issue so a
    // malformed return cannot cross with another accepted command.
    val endpointEventFault = selectedEndpointCompletion.fire &&
      (!selectedEndpointCompletion.success || !endpointCompletionMatches)
    val projectionEventFault = selectedProjectionCompletion.fire &&
      (!selectedProjectionCompletion.success ||
        !projectionCompletionMatches)
    val residualEventFault = selectedResidualCompletion.fire &&
      (!selectedResidualCompletion.success || !residualCompletionMatches)
    val attentionEventFault = selectedAttentionCompletion.fire &&
      (!selectedAttentionCompletion.success ||
        !attentionCompletionMatches)
    val qkvEventFault = selectedQkvCompletion.fire &&
      (!selectedQkvCompletion.success || !qkvCompletionMatches)
    val unexpectedLinearReturn = (0 until engines).map(engine =>
      leaseFabric.io.linearDotResult(engine).valid).reduce(_ || _) &&
      !linearOwnedPhase
    when(unexpectedLinearReturn) {
      unexpectedLinearReturnPending := True
    }

    val componentFaultBits = Bits(16 bits)
    componentFaultBits := 0
    componentFaultBits(BitNetResidentSharedTp4Fault.Feeder) :=
      linearOwnedPhase && feeder.io.protocolFaultSticky
    componentFaultBits(BitNetResidentSharedTp4Fault.LinearEndpoint) :=
      linearOwnedPhase &&
        (finalEndpoint.io.protocolFaultSticky || endpointEventFault)
    componentFaultBits(
      BitNetResidentSharedTp4Fault.ProjectionAssembler) :=
      linearOwnedPhase && fusedProjectionKind &&
        (projectionFeatureAssembler.io.protocolFaultSticky ||
          projectionEventFault)
    componentFaultBits(BitNetResidentSharedTp4Fault.FusedResidual) :=
      linearOwnedPhase && fusedProjectionKind &&
        (fusedResidual.io.protocolFaultSticky || residualEventFault)
    componentFaultBits(BitNetResidentSharedTp4Fault.QkvController) :=
      (qkvRoundActive || imageMetadataValid || attentionOwnedPhase) &&
        (controller.io.protocolFaultSticky || qkvEventFault)
    componentFaultBits(
      BitNetResidentSharedTp4Fault.AttentionController) :=
      attentionOwnedPhase && attentionEventFault
    componentFaultBits(BitNetResidentSharedTp4Fault.Lease) :=
      leaseFabric.io.leaseFaultSticky
    componentFaultBits(BitNetResidentSharedTp4Fault.MatrixCluster) :=
      leaseFabric.io.clusterProtocolFaultSticky.orR
    componentFaultBits(BitNetResidentSharedTp4Fault.SpuArithmetic) :=
      arithmeticPool.io.protocolFaultSticky
    componentFaultBits(BitNetResidentSharedTp4Fault.CompletionIdentity) :=
      (selectedQkvCompletion.fire && !qkvCompletionMatches) ||
        (selectedEndpointCompletion.fire && !endpointCompletionMatches) ||
        (selectedProjectionCompletion.fire &&
          !projectionCompletionMatches) ||
        (selectedResidualCompletion.fire && !residualCompletionMatches) ||
        (selectedAttentionCompletion.fire && !attentionCompletionMatches)
    componentFaultBits(BitNetResidentSharedTp4Fault.EarlyQkvDot) :=
      earlyQkvDot
    componentFaultBits(
      BitNetResidentSharedTp4Fault.UnexpectedMatrixReturn) :=
      unexpectedLinearReturnPending
    val fatalEvent = !fatalLatched && componentFaultBits.orR
    queryAbortPublished := fatalLatched || fatalEvent ||
      io.queryExternalAbort
    localFailCloseNow := fatalLatched || fatalEvent ||
      io.queryExternalAbort || queryAbortPublished
    controller.io.queryAbort := queryAbortPublished
    if (controlHarness) {
      io.harness.queryAbortPublished := queryAbortPublished
      io.harness.localFailCloseNow := localFailCloseNow
    }

    val controllerImageReady = if (controlHarness)
      imageMetadataValid else controller.io.qkvImageReady
    val attentionPrerequisites = imageMetadataValid &&
      controllerImageReady && !qkvCompletionValid
    val matrixIdle = phase === Phase.Idle && !linearCompletionValid &&
      !attentionCompletionValid && !localFailCloseNow &&
      !leaseFabric.io.leaseActive
    val attentionEligible = io.attentionCommand.valid &&
      attentionPrerequisites
    io.attentionCommand.ready := matrixIdle && attentionPrerequisites
    io.linearLaunch.ready := matrixIdle && !attentionEligible

    // Standalone/research compositions keep complete validation at this
    // public boundary.  The production full-layer shell may compile it out
    // only because its sole ingress is the immediately-upstream owner router,
    // which has already authenticated the same complete launch descriptor.
    def acceptLinearLaunch(): Unit = {
      activeLinear := io.linearLaunch.payload
      endpointDone := False
      projectionDone := False
      residualDone := False
      phase := Phase.AcquireLinear
    }
    if (trustedLinearLaunchIngress) {
      when(io.linearLaunch.fire) {
        acceptLinearLaunch()
      }
    } else {
      val genericLaunchLegal = BitNetResidentLinearProgramCodec.programLegal(
        io.linearLaunch.payload, cfg.linearEndpoint) &&
        io.linearLaunch.kind =/= BitNetResidentLinearKind.Qkv
      genericLaunchLegal.setName("genericLaunchLegal")
      when(io.linearLaunch.fire) {
        when(genericLaunchLegal) {
          acceptLinearLaunch()
        } otherwise {
          faultSticky := True
          faultBitsReg(
            BitNetResidentSharedTp4Fault.InvalidLinearLaunch) := True
          linearCompletionReg.requestId := io.linearLaunch.requestId
          linearCompletionReg.layer := io.linearLaunch.layer
          linearCompletionReg.kind := io.linearLaunch.kind
          linearCompletionReg.sequenceId := io.linearLaunch.sequenceId
          linearCompletionReg.success := False
          linearCompletionReg.fatal := False
          linearCompletionReg.faultCode := 0
          linearCompletionReg.faultCode(
            BitNetResidentLinearFault.InvalidLaunch) := True
          linearCompletionReg.acceptedResultBeats := 0
          linearCompletionReg.emittedPairs := 0
          linearCompletionValid := True
        }
      }
    }

    val attentionIdentityMatches =
      io.attentionCommand.requestId === heldQkvRound.requestId &&
        io.attentionCommand.layer === heldQkvRound.layer &&
        io.attentionCommand.tokenBase === heldQkvRound.tokenBase &&
        io.attentionCommand.mode === heldQkvRound.mode &&
        io.attentionCommand.activeTokenMask === heldQkvRound.tokenMask &&
        io.attentionCommand.tokenId.asBits === heldQkvRound.tokenId.asBits
    when(io.attentionCommand.fire) {
      when(attentionIdentityMatches) {
        heldAttention := io.attentionCommand.payload
        phase := Phase.AcquireAttention
      } otherwise {
        faultSticky := True
        faultBitsReg(
          BitNetResidentSharedTp4Fault.CompletionIdentity) := True
        outwardAttentionCompletion.requestId := io.attentionCommand.requestId
        outwardAttentionCompletion.layer := io.attentionCommand.layer
        outwardAttentionCompletion.tokenBase :=
          io.attentionCommand.tokenBase
        outwardAttentionCompletion.completedLaneMask := 0
        outwardAttentionCompletion.success := False
        outwardAttentionCompletion.faultCode := 0
        outwardAttentionCompletion.faultCode(
          BitNetResidentAttentionFault.NumericOrPhase) := True
        attentionCompletionValid := True
      }
    }

    val linearTag = activeLinear.sequenceId(
      cfg.leaseTagWidth - 1 downto 0)
    val attentionTagBase = heldAttention.requestId.resize(
      cfg.leaseTagWidth).asBits ^ heldAttention.tokenBase.resize(
      cfg.leaseTagWidth).asBits
    val attentionTag = (attentionTagBase ^
      B(BigInt(1) << (cfg.leaseTagWidth - 1),
        cfg.leaseTagWidth bits)).asUInt
    leaseFabric.io.linearAcquire.valid := phase === Phase.AcquireLinear &&
      !localFailCloseNow
    leaseFabric.io.linearAcquire.payload.tag := linearTag
    leaseFabric.io.linearRelease.valid := phase === Phase.ReleaseLinear &&
      leaseFabric.io.releaseSafe && !localFailCloseNow
    leaseFabric.io.linearRelease.payload.tag := linearTag
    leaseFabric.io.attentionAcquire.valid :=
      phase === Phase.AcquireAttention && !localFailCloseNow
    leaseFabric.io.attentionAcquire.payload.tag := attentionTag
    leaseFabric.io.attentionRelease.valid :=
      phase === Phase.ReleaseAttention && leaseFabric.io.releaseSafe &&
        !localFailCloseNow
    leaseFabric.io.attentionRelease.payload.tag := attentionTag

    when(phase === Phase.FatalDrain && leaseFabric.io.linearGranted &&
        leaseFabric.io.releaseSafe) {
      leaseFabric.io.linearRelease.valid := True
      leaseFabric.io.linearRelease.payload.tag := leaseFabric.io.leaseTag
    }
    when(phase === Phase.FatalDrain && leaseFabric.io.attentionGranted &&
        leaseFabric.io.releaseSafe) {
      leaseFabric.io.attentionRelease.valid := True
      leaseFabric.io.attentionRelease.payload.tag := leaseFabric.io.leaseTag
    }

    when(phase === Phase.AcquireLinear &&
        leaseFabric.io.linearAcquire.fire) {
      phase := Phase.StartLinearConsumers
    }
    when(phase === Phase.StartLinearConsumers) {
      if (controlHarness) {
        linearEndpointStarted := True
        phase := Phase.StartFeeder
      } else {
        when(fusedProjectionKind) {
          val allReady = finalEndpoint.io.launch.ready &&
            projectionFeatureAssembler.io.launch.ready &&
            fusedResidual.io.launch.ready
          finalEndpoint.io.launch.valid := allReady
          projectionFeatureAssembler.io.launch.valid := allReady
          fusedResidual.io.launch.valid := allReady
          when(finalEndpoint.io.launch.fire &&
              projectionFeatureAssembler.io.launch.fire &&
              fusedResidual.io.launch.fire) {
            linearEndpointStarted := True
            phase := Phase.StartFeeder
          }
        } otherwise {
          finalEndpoint.io.launch.valid := True
          when(finalEndpoint.io.launch.fire) {
            linearEndpointStarted := True
            phase := Phase.StartFeeder
          }
        }
      }
    }
    when(phase === Phase.StartFeeder) {
      if (controlHarness) {
        feederStarted := True
        phase := Phase.RunLinear
      } else {
        feeder.io.task.valid := True
        when(feeder.io.task.fire) {
          feederStarted := True
          phase := Phase.RunLinear
        }
      }
    }

    when(selectedEndpointCompletion.fire) {
      endpointDone := True
      endpointCompletionReg := selectedEndpointCompletion.payload
    }
    when(selectedProjectionCompletion.fire) {
      projectionDone := True
      projectionCompletionReg := selectedProjectionCompletion.payload
    }
    when(selectedResidualCompletion.fire) {
      residualDone := True
      residualCompletionReg := selectedResidualCompletion.payload
    }
    val endpointDoneAfter = endpointDone || selectedEndpointCompletion.fire
    val projectionDoneAfter = projectionDone ||
      selectedProjectionCompletion.fire
    val residualDoneAfter = residualDone || selectedResidualCompletion.fire
    val endpointSuccessAfter = Mux(selectedEndpointCompletion.fire,
      selectedEndpointCompletion.success && endpointCompletionMatches,
      endpointCompletionReg.success &&
        endpointCompletionReg.requestId === activeLinear.requestId &&
        endpointCompletionReg.layer === activeLinear.layer &&
        endpointCompletionReg.kind === activeLinear.kind &&
        endpointCompletionReg.sequenceId === activeLinear.sequenceId)
    val projectionSuccessAfter = Mux(selectedProjectionCompletion.fire,
      selectedProjectionCompletion.success && projectionCompletionMatches,
      projectionCompletionReg.success &&
        projectionCompletionReg.requestId === activeLinear.requestId &&
        projectionCompletionReg.layer === activeLinear.layer &&
        projectionCompletionReg.kind === activeLinear.kind &&
        projectionCompletionReg.sequenceId === activeLinear.sequenceId)
    val residualSuccessAfter = Mux(selectedResidualCompletion.fire,
      selectedResidualCompletion.success && residualCompletionMatches,
      residualCompletionReg.success &&
        residualCompletionReg.requestId === activeLinear.requestId &&
        residualCompletionReg.layer === activeLinear.layer &&
        residualCompletionReg.kind === activeLinear.kind &&
        residualCompletionReg.sequenceId === activeLinear.sequenceId)
    when(phase === Phase.RunLinear && !fusedProjectionKind &&
        endpointDoneAfter && endpointSuccessAfter && !localFailCloseNow) {
      phase := Phase.ReleaseLinear
    }
    when(phase === Phase.RunLinear && fusedProjectionKind &&
        endpointDoneAfter && projectionDoneAfter && residualDoneAfter &&
        endpointSuccessAfter && projectionSuccessAfter &&
        residualSuccessAfter && !localFailCloseNow) {
      phase := Phase.ReleaseLinear
    }
    when(phase === Phase.ReleaseLinear &&
        leaseFabric.io.linearRelease.fire) {
      linearCompletionReg := endpointCompletionReg
      linearCompletionValid := True
      phase := Phase.Idle
    }

    when(phase === Phase.AcquireAttention &&
        leaseFabric.io.attentionAcquire.fire) {
      phase := Phase.StartAttention
    }
    when(phase === Phase.StartAttention) {
      if (controlHarness) {
        attentionStarted := True
        phase := Phase.RunAttention
      } else {
        controller.io.attentionCommand.valid := True
        when(controller.io.attentionCommand.fire) {
          attentionStarted := True
          phase := Phase.RunAttention
        }
      }
    }
    when(selectedAttentionCompletion.fire &&
        selectedAttentionCompletion.success &&
        attentionCompletionMatches && !localFailCloseNow) {
      attentionCompletionReg := selectedAttentionCompletion.payload
      phase := Phase.ReleaseAttention
    }
    when(phase === Phase.ReleaseAttention &&
        leaseFabric.io.attentionRelease.fire) {
      outwardAttentionCompletion := attentionCompletionReg
      attentionCompletionValid := True
      imageMetadataValid := False
      phase := Phase.Idle
    }

    when(selectedQkvCompletion.fire && selectedQkvCompletion.success &&
        qkvCompletionMatches && !localFailCloseNow) {
      qkvCompletionReg := selectedQkvCompletion.payload
      qkvCompletionValid := True
      qkvRoundActive := False
      imageMetadataValid := True
    }

    // A post-start fault is globally fail-closed.  All matrix returns drain,
    // the exact held lease tag is released only at releaseSafe, and terminal
    // events become visible only after the physical cluster is unowned.
    when(fatalEvent) {
      fatalLatched := True
      faultSticky := True
      faultBitsReg := faultBitsReg | componentFaultBits
      imageMetadataValid := False

      when(qkvRoundActive && !qkvCompletionValid) {
        qkvCompletionReg.requestId := heldQkvRound.requestId
        qkvCompletionReg.layer := heldQkvRound.layer
        qkvCompletionReg.tokenBase := heldQkvRound.tokenBase
        qkvCompletionReg.tokenMask := heldQkvRound.tokenMask
        qkvCompletionReg.success := False
        qkvCompletionReg.faultCode := B"8'h80"
        when(selectedQkvCompletion.fire) {
          qkvCompletionReg.faultCode :=
            selectedQkvCompletion.faultCode | B"8'h80"
        }
        fatalQkvReportPending := True
      }
      qkvRoundActive := False

      when(linearOwnedPhase && !linearCompletionValid) {
        linearCompletionReg.requestId := activeLinear.requestId
        linearCompletionReg.layer := activeLinear.layer
        linearCompletionReg.kind := activeLinear.kind
        linearCompletionReg.sequenceId := activeLinear.sequenceId
        linearCompletionReg.success := False
        linearCompletionReg.fatal := True
        linearCompletionReg.faultCode := componentFaultBits
        linearCompletionReg.acceptedResultBeats := 0
        linearCompletionReg.emittedPairs := 0
        fatalLinearReportPending := True
      }
      when(attentionOwnedPhase && !attentionCompletionValid) {
        outwardAttentionCompletion.requestId := heldAttention.requestId
        outwardAttentionCompletion.layer := heldAttention.layer
        outwardAttentionCompletion.tokenBase := heldAttention.tokenBase
        outwardAttentionCompletion.completedLaneMask := 0
        outwardAttentionCompletion.success := False
        outwardAttentionCompletion.faultCode := B"8'h80"
        fatalAttentionReportPending := True
      }
      phase := Phase.FatalDrain
    }

    when(phase === Phase.FatalDrain && !leaseFabric.io.leaseActive) {
      when(fatalQkvReportPending && !qkvCompletionValid) {
        qkvCompletionValid := True
        fatalQkvReportPending := False
      }
      when(fatalLinearReportPending && !linearCompletionValid) {
        linearCompletionValid := True
        fatalLinearReportPending := False
      }
      when(fatalAttentionReportPending && !attentionCompletionValid) {
        attentionCompletionValid := True
        fatalAttentionReportPending := False
      }
    }

    when(componentFaultBits.orR) {
      faultSticky := True
      faultBitsReg := faultBitsReg | componentFaultBits
    }

    io.qkvImageReady := imageMetadataValid && controllerImageReady &&
      !fatalLatched
    io.busy := phase =/= Phase.Idle || qkvRoundActive || qkvCompletionValid ||
      linearCompletionValid || attentionCompletionValid || feeder.io.active ||
      finalEndpoint.io.busy || projectionFeatureAssembler.io.busy ||
      fusedResidual.io.busy ||
      (if (controlHarness) False else controller.io.busy)
    io.fatalFailStop := fatalLatched
    io.protocolFaultSticky := faultSticky ||
      leaseFabric.io.protocolFaultSticky || feeder.io.protocolFaultSticky ||
      finalEndpoint.io.protocolFaultSticky ||
      projectionFeatureAssembler.io.protocolFaultSticky ||
      fusedResidual.io.protocolFaultSticky ||
      controller.io.protocolFaultSticky ||
      arithmeticPool.io.protocolFaultSticky
    io.faultBits := faultBitsReg | componentFaultBits
    io.linearLeaseGranted := leaseFabric.io.linearGranted
    io.attentionLeaseGranted := leaseFabric.io.attentionGranted
    io.leaseReleaseSafe := leaseFabric.io.releaseSafe
    io.leaseTag := leaseFabric.io.leaseTag
    io.matrixSequenceActive := leaseFabric.io.sequenceActive
    io.matrixDraining := leaseFabric.io.draining
    io.activeLinearKind := activeLinear.kind
  }
}

object GenerateBitNetResidentSharedTp4ComputeDatapath extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-shared-tp4-compute-datapath")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentSharedTp4ComputeDatapath())
}
