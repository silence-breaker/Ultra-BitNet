package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Fixed production geometry of the sole QKV/attention matrix island. */
case class BitNetResidentQkvAttentionComputeIslandConfig(
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
  val assembler: BitNetQkvClusterResultAssemblerConfig =
    BitNetQkvClusterResultAssemblerConfig(
      model = model,
      continuous = continuous,
      qkv = pipeline.qkv,
      kTileCountWidth = feeder.kTileCountWidth)
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

  require(model.engineCount >= 1 && model.engineCount <= 2 &&
    model.tokenParallelism == model.engineCount * continuous.tokenLanesPerEngine)
  require(continuous eq feeder.matrix)
  require(continuous eq assembler.continuous)
  require(lease.matrix eq continuous)
  require(lease.svGroupLanes == attention.residentGroupLanes)
  require(leaseTagWidth >= 2 && leaseTagWidth <= continuous.sequenceIdWidth)
}

/**
  * One indivisible projection launch.  The redundant envelope fields are
  * intentional: they let the island compare the layer controller's identity
  * against all three resident consumers before any stateful block is started.
  */
case class BitNetResidentQkvAttentionAtomicLaunch(
    cfg: BitNetResidentQkvAttentionComputeIslandConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val mode = BitNetMode()
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val sequenceId = UInt(cfg.continuous.sequenceIdWidth bits)
  val qkvRound = BitNetResidentQkvRoundCommand(cfg.pipeline)
  val matrixTask = BitNetContinuousMatrixTask(cfg.feeder)
  val assemblerDescriptor = BitNetQkvClusterRoundDescriptor(cfg.assembler)
}

/** Stable diagnostic bit allocation for island-level failures. */
object BitNetQkvAttentionIslandFault {
  val LaunchInvariant = 0
  val Feeder = 1
  val Assembler = 2
  val QkvController = 3
  val Lease = 4
  val MatrixCluster = 5
  val CompletionMetadata = 6
  val UnexpectedMatrixReturn = 7
  val AttentionCommand = 8
  val AttentionCompletion = 9
  val LinearEndpoint = 10
  val SpuArithmetic = 11
  val FusedResidual = 12
  val ResidentTensor = 13
}

/** Terminal QKV event emitted only after a normal lease release or fail-stop. */
case class BitNetResidentQkvIslandCompletion(
    cfg: BitNetResidentQkvAttentionComputeIslandConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val sequenceId = UInt(cfg.continuous.sequenceIdWidth bits)
  val success = Bool()
  /** Fatal means the already-started resident controller cannot be aborted. */
  val fatal = Bool()
  val faultCode = Bits(16 bits)
  val emittedFeatures = UInt(32 bits)
}

/**
  * Compile-time-only control harness used by the focused Spec.  Production
  * elaboration omits these ports completely; no raw dot payload is exposed.
  */
case class BitNetQkvAttentionIslandHarnessIo(
    cfg: BitNetResidentQkvAttentionComputeIslandConfig) extends Bundle {
  val qkvConsumersStarted = out Bool()
  val genericEndpointStarted = out Bool()
  val feederStarted = out Bool()
  val attentionStarted = out Bool()
  val assemblerCompletion = slave(Stream(
    BitNetQkvAssemblerCompletion(cfg.assembler)))
  val controllerQkvCompletion = slave(Stream(
    BitNetResidentQkvRoundCompletion(cfg.pipeline)))
  val controllerAttentionCompletion = slave(Stream(
    BitNetAttentionCompletion(cfg.model)))
  val genericEndpointCompletion = slave(Stream(
    BitNetResidentLinearCompletion(cfg.linearEndpoint)))
  val projectionAssemblyCompletion = slave(Stream(
    BitNetResidentProjectionAssemblyCompletion(cfg.projectionAssembler)))
  val fusedResidualCompletion = slave(Stream(
    BitNetResidentFusedResidualCompletion(cfg.fusedResidual)))
}

/**
  * The only model-wide QKV/attention compute island.
  *
  * Exactly one [[BitNetTp4MatrixLeaseFabric]] owns the two physical TP2
  * arrays.  Projection uses its linear client through one continuous feeder;
  * QK and resident SV use its attention client through the resident
  * controller.  Raw cluster results terminate at the QKV assembler and never
  * cross this component boundary.
  */
class BitNetResidentQkvAttentionComputeIsland(
    cfg: BitNetResidentQkvAttentionComputeIslandConfig =
      BitNetResidentQkvAttentionComputeIslandConfig(),
    useVendorPrimitive: Boolean = true,
    controlHarness: Boolean = false) extends Component {
  private val engines = cfg.model.engineCount
  private val lanes = cfg.model.tokenParallelism
  private val m = cfg.continuous
  private val unifiedTensorCfg =
    BitNetUnifiedResidentTensorProductionShellConfig()

  require(cfg.attention.unifiedScratchpad ==
    unifiedTensorCfg.backend.scratchpad,
    "the standalone QKV island and its resident Query store must share " +
      "one packed layout and alias-generation domain")

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val qkvLaunch = slave(Stream(
      BitNetResidentQkvAttentionAtomicLaunch(cfg)))
    val qkvCompletion = master(Stream(
      BitNetResidentQkvIslandCompletion(cfg)))
    val linearLaunch = slave(Stream(
      BitNetResidentLinearLaunch(cfg.linearEndpoint)))
    val linearResult = master(Stream(
      BitNetResidentFinalFeaturePair(cfg.linearEndpoint)))
    val linearCompletion = master(Stream(
      BitNetResidentLinearCompletion(cfg.linearEndpoint)))
    val attentionCommand = slave(Stream(
      BitNetAttentionScheduleCommand(cfg.model)))
    val attentionCompletion = master(Stream(
      BitNetAttentionCompletion(cfg.model)))

    // Temporary typed PL-buffer boundaries.  No ARM/result-packet boundary is
    // introduced while the five-bank memory complex is integrated.
    val map0RawGroup = slave(Stream(BitNetMap0RawGroupTile(
      cfg.feeder, cfg.model.ffnSize)))
    val i8WeightBeat = slave(Stream(BitNetI8WeightBeat(cfg.feeder)))
    val activationTile = slave(Stream(
      BitNetNormalizedActivationTile(cfg.feeder)))
    val ropeCoefficient = slave(Stream(
      BitNetQkvFeatureRopeCoefficient(cfg.assembler)))

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

    // Exposed index 0..4 maps to physical client 1..5: RMSNorm, FFN gate,
    // dynamic quant, RoPE and head quant.  Physical client 0 is wired
    // privately to projection scaling below.
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
      BitNetQkvAttentionIslandHarnessIo(cfg) else null
  }

  // The tensor shell below is the sole physical Query image owner in this
  // standalone top; no private Q buffer or software-visible payload boundary
  // is reintroduced around the resident attention controller.
  val leaseFabric = new BitNetTp4MatrixLeaseFabric(
    cfg.lease, useVendorPrimitive, cfg.leaseTagWidth)
  val controller = new BitNetResidentQkvAttentionController(
    cfg.pipeline, useVendorPrimitive)
  val residentTensor =
    new BitNetUnifiedResidentTensorProductionShell(unifiedTensorCfg)
  val queryPackedReadJoin =
    new BitNetUnifiedResidentPackedReadParallelToLegacyRowAdapter(
      unifiedTensorCfg.backend.scratchpad)

  leaseFabric.io.slowClk := io.slowClk
  leaseFabric.io.slowResetn := io.slowResetn
  leaseFabric.io.fastClk := io.fastClk
  leaseFabric.io.fastResetn := io.fastResetn
  controller.io.slowClk := io.slowClk
  controller.io.slowResetn := io.slowResetn
  residentTensor.io.slowClk := io.slowClk
  residentTensor.io.slowResetn := io.slowResetn
  residentTensor.io.fastClk := io.fastClk
  residentTensor.io.fastResetn := io.fastResetn

  // Query owns the packed and QueryLmAlias semantic facets in this isolated
  // compute top.  Activation traffic belongs to the enclosing full-layer
  // composition, so both activation command inputs are sealed here and their
  // impossible response/fault streams are drained defensively.
  residentTensor.io.activationRead.valid := False
  residentTensor.io.activationRead.payload :=
    BitNetUnifiedResidentActivationGroupRead(
      unifiedTensorCfg.backend.scratchpad).getZero
  residentTensor.io.activationReadData.ready := True
  residentTensor.io.activationWrite.valid := False
  residentTensor.io.activationWrite.payload :=
    BitNetUnifiedResidentActivationGroupWrite(
      unifiedTensorCfg.backend.scratchpad).getZero
  residentTensor.io.activationWriteFault.ready := True

  residentTensor.io.packedRead << controller.io.queryPackedRead
  controller.io.queryPackedReadData << queryPackedReadJoin.io.legacy
  queryPackedReadJoin.io.parallel.region :=
    residentTensor.io.packedReadData.region
  queryPackedReadJoin.io.parallel.localRow :=
    residentTensor.io.packedReadData.localRow
  queryPackedReadJoin.io.parallel.bankMask :=
    residentTensor.io.packedReadData.bankMask
  queryPackedReadJoin.io.parallel.laneMask :=
    residentTensor.io.packedReadData.laneMask
  queryPackedReadJoin.io.parallel.fault :=
    residentTensor.io.packedReadData.fault
  queryPackedReadJoin.io.parallel.faultCode :=
    residentTensor.io.packedReadData.faultCode
  queryPackedReadJoin.io.parallel.segmentValid :=
    residentTensor.io.packedReadData.segmentValid
  for (lane <- 0 until unifiedTensorCfg.backend.scratchpad.layout
      .packedTransportSegmentCount) {
    queryPackedReadJoin.io.parallel.segments(lane) :=
      residentTensor.io.packedReadData.segments(lane)
  }
  residentTensor.io.packedReadData.segmentReady :=
    queryPackedReadJoin.io.parallel.segmentReady
  residentTensor.io.packedWrite << controller.io.queryPackedWrite
  controller.io.queryPackedWriteFault << residentTensor.io.packedWriteFault
  residentTensor.io.aliasAcquire << controller.io.queryAliasAcquire
  controller.io.queryAliasGrant << residentTensor.io.aliasGrant
  residentTensor.io.aliasRelease << controller.io.queryAliasRelease
  // This isolated top has no north-side alias router.  Before the direct
  // frontend acquire fires it retains no source identity, so abort may safely
  // cancel an unaccepted command after the adapter's one-cycle probe.
  controller.io.queryAliasAcquireOwned := False
  controller.io.querySemanticPackedReadOutstanding :=
    residentTensor.io.packedReadOutstanding
  controller.io.querySemanticPackedWriteOutstanding :=
    residentTensor.io.packedWriteOutstanding

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  val slowArea = new ClockingArea(slowCd) {
    val feeder = new BitNetContinuousMatrixFeeder(cfg.feeder)
    val assembler = new BitNetQkvClusterResultAssembler(cfg.assembler)
    val finalEndpoint = new BitNetResidentLinearResultEndpoint(
      cfg.linearEndpoint)
    val projectionFeatureAssembler =
      new BitNetResidentProjectionFeatureAssembler(cfg.projectionAssembler)
    val fusedResidual =
      new BitNetResidentFusedResidualEndpoint(cfg.fusedResidual)
    val arithmeticPool = new BitNetSharedSpuArithmeticPool(cfg.arithmetic)

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
      val Idle, AcquireLinear, StartQkvConsumers, StartFeeder, RunQkv,
          ReleaseLinear, AcquireAttention, StartAttention, RunAttention,
          ReleaseAttention, FatalStop = newElement()
    }
    val phase = Reg(Phase()) init Phase.Idle
    val activeLaunch = Reg(BitNetResidentQkvAttentionAtomicLaunch(cfg)) init
      BitNetResidentQkvAttentionAtomicLaunch(cfg).getZero
    val activeGenericLaunch = Reg(
      BitNetResidentLinearLaunch(cfg.linearEndpoint)) init
      BitNetResidentLinearLaunch(cfg.linearEndpoint).getZero
    val activeKind = Reg(BitNetResidentLinearKind()) init
      BitNetResidentLinearKind.Qkv
    val heldAttention = Reg(BitNetAttentionScheduleCommand(cfg.model)) init
      BitNetAttentionScheduleCommand(cfg.model).getZero
    val imageMetadataValid = RegInit(False)
    val fatalLatched = RegInit(False)
    val faultSticky = RegInit(False)
    val faultBitsReg = Reg(Bits(16 bits)) init 0

    val assemblerDone = RegInit(False)
    val controllerQkvDone = RegInit(False)
    val genericEndpointDone = RegInit(False)
    val projectionAssemblyDone = RegInit(False)
    val fusedResidualDone = RegInit(False)
    // The assembler's public counter is lifetime-cumulative.  Capture its
    // value at the atomic launch so a fatal completion still reports only the
    // features admitted during this round.
    val assemblerEmittedBase = Reg(UInt(32 bits)) init 0
    val endpointEmittedBase = Reg(UInt(32 bits)) init 0
    val assemblerCompletionReg = Reg(
      BitNetQkvAssemblerCompletion(cfg.assembler)) init
      BitNetQkvAssemblerCompletion(cfg.assembler).getZero
    val controllerQkvCompletionReg = Reg(
      BitNetResidentQkvRoundCompletion(cfg.pipeline)) init
      BitNetResidentQkvRoundCompletion(cfg.pipeline).getZero
    val attentionCompletionReg = Reg(
      BitNetAttentionCompletion(cfg.model)) init
      BitNetAttentionCompletion(cfg.model).getZero
    val endpointCompletionReg = Reg(
      BitNetResidentLinearCompletion(cfg.linearEndpoint)) init
      BitNetResidentLinearCompletion(cfg.linearEndpoint).getZero
    val projectionAssemblyCompletionReg = Reg(
      BitNetResidentProjectionAssemblyCompletion(
        cfg.projectionAssembler)) init
      BitNetResidentProjectionAssemblyCompletion(
        cfg.projectionAssembler).getZero
    val fusedResidualCompletionReg = Reg(
      BitNetResidentFusedResidualCompletion(cfg.fusedResidual)) init
      BitNetResidentFusedResidualCompletion(cfg.fusedResidual).getZero

    val qkvCompletionValid = RegInit(False)
    val fatalQkvReportPending = RegInit(False)
    val qkvCompletionReg = Reg(BitNetResidentQkvIslandCompletion(cfg)) init
      BitNetResidentQkvIslandCompletion(cfg).getZero
    io.qkvCompletion.valid := qkvCompletionValid
    io.qkvCompletion.payload := qkvCompletionReg
    when(io.qkvCompletion.fire) { qkvCompletionValid := False }

    val linearCompletionValid = RegInit(False)
    val fatalLinearReportPending = RegInit(False)
    val linearCompletionReg = Reg(
      BitNetResidentLinearCompletion(cfg.linearEndpoint)) init
      BitNetResidentLinearCompletion(cfg.linearEndpoint).getZero
    io.linearCompletion.valid := linearCompletionValid
    io.linearCompletion.payload := linearCompletionReg
    when(io.linearCompletion.fire) { linearCompletionValid := False }

    val attentionCompletionValid = RegInit(False)
    val fatalAttentionReportPending = RegInit(False)
    val outwardAttentionCompletion = Reg(
      BitNetAttentionCompletion(cfg.model)) init
      BitNetAttentionCompletion(cfg.model).getZero
    io.attentionCompletion.valid := attentionCompletionValid
    io.attentionCompletion.payload := outwardAttentionCompletion
    when(io.attentionCompletion.fire) { attentionCompletionValid := False }

    // ------------------------------------------------------------------
    // Atomic launch invariant checker.
    // ------------------------------------------------------------------
    val launch = io.qkvLaunch.payload
    val task = launch.matrixTask
    val desc = launch.assemblerDescriptor
    val round = launch.qkvRound
    val launchPrefill = launch.mode === BitNetMode.Prefill
    val launchDecode = launch.mode === BitNetMode.Decode
    val prefixMaskLegal = (1 to lanes).map { count =>
      launch.tokenMask === B((BigInt(1) << count) - 1, lanes bits)
    }.reduce(_ || _)
    val modeMaskLegal = (launchPrefill && prefixMaskLegal) ||
      (launchDecode && launch.tokenMask === B(1, lanes bits))
    val identityMatches = launch.requestId === round.requestId &&
      launch.layer === round.layer && launch.layer < cfg.model.layerCount &&
      launch.mode === round.mode && launch.tokenMask === round.tokenMask &&
      launch.mode === task.scheduleMode && launch.mode === desc.scheduleMode &&
      launch.tokenMask === task.activeTokenMask &&
      launch.tokenMask === desc.activeTokenMask &&
      launch.sequenceId === task.sequenceId &&
      launch.sequenceId === desc.sequenceId
    val framingMatches = task.operation ===
      BitNetContinuousDotKind.Projection &&
      !task.routeFinalToDynamicQuant && !task.residualEnable &&
      task.taskIdBase === desc.taskIdBase &&
      task.expectedBeats === desc.expectedBeats &&
      task.kTilesPerOutput === desc.kTilesPerOutput &&
      task.kTilesPerOutput =/= 0
    val vectorMatches = Vec(Bool(), lanes)
    for (lane <- 0 until lanes) {
      vectorMatches(lane) := task.tokenId(lane) === desc.tokenId(lane) &&
        task.tokenId(lane).resize(cfg.model.sequenceWidth) ===
          round.tokenId(lane) &&
        desc.tokenId(lane).resize(cfg.model.sequenceWidth) ===
          round.tokenId(lane) &&
        task.activationFactorQ16(lane) === desc.activationFactorQ16(lane)
    }
    val engineMatches = Vec(Bool(), engines)
    for (engine <- 0 until engines) {
      engineMatches(engine) :=
        task.weightScaleQ16(engine) === desc.weightScaleQ16(engine) &&
        task.outputBase(engine) === desc.outputBase(engine) &&
        task.outputEndExclusive(engine) === desc.outputEndExclusive(engine)
    }
    val totalFeatures = U(cfg.assembler.totalFeatures,
      m.base.outputIndexWidth bits)
    val halfFeatures = U(cfg.assembler.decodeFeaturesPerEngine,
      m.base.outputIndexWidth bits)
    val prefillLayout = (0 until engines).map { engine =>
      task.outputBase(engine) === 0 &&
        task.outputEndExclusive(engine) === totalFeatures &&
        task.weightScaleQ16(engine) === task.weightScaleQ16(0)
    }.reduce(_ && _)
    val decodeLayout = if (engines == 1) {
      task.outputBase(0) === 0 &&
        task.outputEndExclusive(0) === totalFeatures
    } else {
      val forward = task.outputBase(0) === 0 &&
        task.outputEndExclusive(0) === halfFeatures &&
        task.outputBase(1) === halfFeatures &&
        task.outputEndExclusive(1) === totalFeatures
      val reverse = task.outputBase(1) === 0 &&
        task.outputEndExclusive(1) === halfFeatures &&
        task.outputBase(0) === halfFeatures &&
        task.outputEndExclusive(0) === totalFeatures
      forward || reverse
    }
    val expectedPrefillBeats =
      (task.kTilesPerOutput.resize(m.beatCountWidth) *
        U(cfg.assembler.prefillGroups, m.beatCountWidth bits)).resize(
          m.beatCountWidth)
    val expectedDecodeBeats =
      (task.kTilesPerOutput.resize(m.beatCountWidth) *
        U(cfg.assembler.decodeGroupsPerEngine, m.beatCountWidth bits)).resize(
          m.beatCountWidth)
    val scheduleLegal =
      (launchPrefill && prefillLayout &&
        task.expectedBeats === expectedPrefillBeats) ||
      (launchDecode && decodeLayout &&
        task.expectedBeats === expectedDecodeBeats)
    val launchInvariantLegal = identityMatches && framingMatches &&
      vectorMatches.asBits.andR && engineMatches.asBits.andR &&
      modeMaskLegal && scheduleLegal

    // ------------------------------------------------------------------
    // Public memory/activation boundaries remain attached to the resident
    // controller.  They are PL-client streams, never ARM payload callbacks.
    // ------------------------------------------------------------------
    io.kvWriteRequest << controller.io.kvWriteRequest
    io.kvWriteData << controller.io.kvWriteData
    controller.io.kvWriteResponse << io.kvWriteResponse
    io.kvReadRequest << controller.io.kvReadRequest
    controller.io.kvReadResponse << io.kvReadResponse
    for (lane <- 0 until lanes) {
      io.contextWrite(lane) << controller.io.writeData(lane)
    }

    // ------------------------------------------------------------------
    // One physical cluster: linear projection client and attention client.
    // Fatal gating removes all new requests but continues draining returns.
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

      assembler.io.engineResult(engine).valid :=
        leaseFabric.io.linearDotResult(engine).valid && !fatalLatched &&
          activeKind === BitNetResidentLinearKind.Qkv
      assembler.io.engineResult(engine).payload :=
        leaseFabric.io.linearDotResult(engine).payload
      finalEndpoint.io.engineResult(engine).valid :=
        leaseFabric.io.linearDotResult(engine).valid && !fatalLatched &&
          activeKind =/= BitNetResidentLinearKind.Qkv
      finalEndpoint.io.engineResult(engine).payload :=
        leaseFabric.io.linearDotResult(engine).payload
      leaseFabric.io.linearDotResult(engine).ready := True
      when(!fatalLatched) {
        leaseFabric.io.linearDotResult(engine).ready :=
          assembler.io.engineResult(engine).ready
        when(activeKind =/= BitNetResidentLinearKind.Qkv) {
          leaseFabric.io.linearDotResult(engine).ready :=
            finalEndpoint.io.engineResult(engine).ready
        }
      }
      if (!cfg.lease.productionPruneGenericSv)
        leaseFabric.io.linearSvResult(engine).ready := True

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
      if (!cfg.lease.productionPruneLegacyBeat) {
        leaseFabric.io.attentionBeat(engine).valid := False
        leaseFabric.io.attentionBeat(engine).payload :=
          BitNetContinuousMatrixBeat(
            cfg.continuous,
            includeGenericSv = !cfg.lease.productionPruneGenericSv).getZero
      }

      controller.io.matrixDotResult(engine).valid :=
        leaseFabric.io.attentionDotResult(engine).valid && !fatalLatched
      controller.io.matrixDotResult(engine).payload :=
        leaseFabric.io.attentionDotResult(engine).payload
      leaseFabric.io.attentionDotResult(engine).ready :=
        Mux(fatalLatched, True,
          controller.io.matrixDotResult(engine).ready)
      if (!cfg.lease.productionPruneGenericSv) {
        controller.io.matrixSvResult(engine).valid :=
          leaseFabric.io.attentionSvResult(engine).valid && !fatalLatched
        controller.io.matrixSvResult(engine).payload :=
          leaseFabric.io.attentionSvResult(engine).payload
        leaseFabric.io.attentionSvResult(engine).ready :=
          Mux(fatalLatched, True,
            controller.io.matrixSvResult(engine).ready)
      }

      leaseFabric.io.attentionResidentSequence(engine).valid :=
        controller.io.matrixResidentSequence(engine).valid && !fatalLatched
      leaseFabric.io.attentionResidentSequence(engine).payload :=
        controller.io.matrixResidentSequence(engine).payload
      controller.io.matrixResidentSequence(engine).ready :=
        leaseFabric.io.attentionResidentSequence(engine).ready && !fatalLatched
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
      leaseFabric.io.attentionResidentResult(engine).ready :=
        Mux(fatalLatched, True,
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

    // QKV raw dots terminate internally.  Once a fatal condition is observed,
    // assembler output is drained/dropped and cannot enter the resident image.
    if (controlHarness) {
      controller.io.qkvDot.valid := False
      controller.io.qkvDot.payload := BitNetQkvTp4DotBeat(cfg.pipeline.qkv)
        .getZero
      assembler.io.output.ready := True
    } else {
      controller.io.qkvDot.valid := assembler.io.output.valid && !fatalLatched
      controller.io.qkvDot.payload := assembler.io.output.payload
      assembler.io.output.ready := Mux(
        fatalLatched, True, controller.io.qkvDot.ready)
    }

    // O/Down never expose raw pairs: they are reassembled to one TP4 feature,
    // fused with HiddenResident through the sole projection scaler and written
    // back before commit.  GateUp/LM keep the typed downstream stream.
    val fusedProjectionKind =
      activeKind === BitNetResidentLinearKind.AttentionOutput ||
        activeKind === BitNetResidentLinearKind.Down
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

    // Typed PL buffers are open only after both consumers and the feeder have
    // accepted the same launch.
    val linearDataOpen = phase === Phase.RunQkv && !fatalLatched &&
      (if (controlHarness) False else True)
    feeder.io.map0RawGroup.valid := io.map0RawGroup.valid && linearDataOpen
    feeder.io.map0RawGroup.payload := io.map0RawGroup.payload
    io.map0RawGroup.ready := feeder.io.map0RawGroup.ready && linearDataOpen
    feeder.io.i8WeightBeat.valid := io.i8WeightBeat.valid && linearDataOpen
    feeder.io.i8WeightBeat.payload := io.i8WeightBeat.payload
    io.i8WeightBeat.ready := feeder.io.i8WeightBeat.ready && linearDataOpen
    feeder.io.activationTile.valid :=
      io.activationTile.valid && linearDataOpen
    feeder.io.activationTile.payload := io.activationTile.payload
    io.activationTile.ready :=
      feeder.io.activationTile.ready && linearDataOpen
    val qkvRopeOpen = linearDataOpen &&
      activeKind === BitNetResidentLinearKind.Qkv
    assembler.io.ropeCoefficient.valid :=
      io.ropeCoefficient.valid && qkvRopeOpen
    assembler.io.ropeCoefficient.payload := io.ropeCoefficient.payload
    io.ropeCoefficient.ready :=
      assembler.io.ropeCoefficient.ready && qkvRopeOpen

    // Default start ports.  Production overrides them only at the atomic
    // barriers below; the harness keeps the real components quiescent.
    controller.io.qkvRound.valid := False
    controller.io.qkvRound.payload := activeLaunch.qkvRound
    assembler.io.round.valid := False
    assembler.io.round.payload := activeLaunch.assemblerDescriptor
    finalEndpoint.io.launch.valid := False
    finalEndpoint.io.launch.payload := activeGenericLaunch
    projectionFeatureAssembler.io.launch.valid := False
    projectionFeatureAssembler.io.launch.payload := activeGenericLaunch
    fusedResidual.io.launch.valid := False
    fusedResidual.io.launch.payload := activeGenericLaunch
    feeder.io.task.valid := False
    feeder.io.task.payload := activeLaunch.matrixTask
    when(activeKind =/= BitNetResidentLinearKind.Qkv) {
      feeder.io.task.payload := activeGenericLaunch.matrixTask
    }
    controller.io.attentionCommand.valid := False
    controller.io.attentionCommand.payload := heldAttention

    val consumersStarted = Bool()
    val genericEndpointStarted = Bool()
    val feederStarted = Bool()
    val attentionStarted = Bool()
    consumersStarted := False
    genericEndpointStarted := False
    feederStarted := False
    attentionStarted := False

    if (controlHarness) {
      io.harness.qkvConsumersStarted := consumersStarted
      io.harness.genericEndpointStarted := genericEndpointStarted
      io.harness.feederStarted := feederStarted
      io.harness.attentionStarted := attentionStarted
      // Real idle blocks must never retain a completion in harness mode.
      assembler.io.completion.ready := True
      controller.io.qkvCompletion.ready := True
      controller.io.attentionCompletion.ready := True
      finalEndpoint.io.completion.ready := True
      projectionFeatureAssembler.io.completion.ready := True
      fusedResidual.io.completion.ready := True
    }

    // Selected completion sources are compile-time constant.  The harness
    // therefore exercises this exact sequencer without adding production muxes.
    val selectedAssemblerCompletion = if (controlHarness)
      io.harness.assemblerCompletion else assembler.io.completion
    val selectedControllerQkvCompletion = if (controlHarness)
      io.harness.controllerQkvCompletion else controller.io.qkvCompletion
    val selectedAttentionCompletion = if (controlHarness)
      io.harness.controllerAttentionCompletion
    else controller.io.attentionCompletion
    val selectedEndpointCompletion = if (controlHarness)
      io.harness.genericEndpointCompletion
    else finalEndpoint.io.completion
    val selectedProjectionAssemblyCompletion = if (controlHarness)
      io.harness.projectionAssemblyCompletion
    else projectionFeatureAssembler.io.completion
    val selectedFusedResidualCompletion = if (controlHarness)
      io.harness.fusedResidualCompletion
    else fusedResidual.io.completion
    selectedAssemblerCompletion.ready := phase === Phase.RunQkv &&
      activeKind === BitNetResidentLinearKind.Qkv &&
      !assemblerDone && !fatalLatched
    selectedControllerQkvCompletion.ready := phase === Phase.RunQkv &&
      activeKind === BitNetResidentLinearKind.Qkv &&
      !controllerQkvDone && !fatalLatched
    selectedEndpointCompletion.ready := phase === Phase.RunQkv &&
      activeKind =/= BitNetResidentLinearKind.Qkv &&
      !genericEndpointDone && !fatalLatched
    selectedProjectionAssemblyCompletion.ready := phase === Phase.RunQkv &&
      fusedProjectionKind && !projectionAssemblyDone && !fatalLatched
    selectedFusedResidualCompletion.ready := phase === Phase.RunQkv &&
      fusedProjectionKind && !fusedResidualDone && !fatalLatched
    selectedAttentionCompletion.ready := phase === Phase.RunAttention &&
      !fatalLatched

    // Lease command defaults.  Release valid is asserted only with releaseSafe
    // because the lease fabric deliberately diagnoses an early held release.
    val activeLinearSequence = UInt(cfg.continuous.sequenceIdWidth bits)
    activeLinearSequence := activeLaunch.sequenceId
    when(activeKind =/= BitNetResidentLinearKind.Qkv) {
      activeLinearSequence := activeGenericLaunch.sequenceId
    }
    val linearTag = activeLinearSequence(
      cfg.leaseTagWidth - 1 downto 0)
    val attentionTag = (linearTag.asBits ^
      B(BigInt(1) << (cfg.leaseTagWidth - 1), cfg.leaseTagWidth bits)).asUInt
    leaseFabric.io.linearAcquire.valid := phase === Phase.AcquireLinear &&
      !fatalLatched
    leaseFabric.io.linearAcquire.payload.tag := linearTag
    leaseFabric.io.linearRelease.valid := phase === Phase.ReleaseLinear &&
      leaseFabric.io.releaseSafe && !fatalLatched
    leaseFabric.io.linearRelease.payload.tag := linearTag
    leaseFabric.io.attentionAcquire.valid :=
      phase === Phase.AcquireAttention && !fatalLatched
    leaseFabric.io.attentionAcquire.payload.tag := attentionTag
    leaseFabric.io.attentionRelease.valid :=
      phase === Phase.ReleaseAttention && leaseFabric.io.releaseSafe &&
        !fatalLatched
    leaseFabric.io.attentionRelease.payload.tag := attentionTag

    // In fatal-stop, revoke neither owner early.  Once the fabric itself says
    // the held cluster is quiescent, return the matching tag and stay stopped.
    when(phase === Phase.FatalStop && leaseFabric.io.linearGranted &&
        leaseFabric.io.releaseSafe) {
      leaseFabric.io.linearRelease.valid := True
      leaseFabric.io.linearRelease.payload.tag := leaseFabric.io.leaseTag
    }
    when(phase === Phase.FatalStop && leaseFabric.io.attentionGranted &&
        leaseFabric.io.releaseSafe) {
      leaseFabric.io.attentionRelease.valid := True
      leaseFabric.io.attentionRelease.payload.tag := leaseFabric.io.leaseTag
    }

    // ------------------------------------------------------------------
    // Transaction sequencer.
    // ------------------------------------------------------------------
    val controllerImageReady = if (controlHarness)
      imageMetadataValid else controller.io.qkvImageReady
    val idleLinearLaunchReady = phase === Phase.Idle &&
      !imageMetadataValid && !qkvCompletionValid &&
      !linearCompletionValid && !attentionCompletionValid && !fatalLatched
    io.qkvLaunch.ready := idleLinearLaunchReady
    // Existing QKV launch has fixed priority if both producers assert valid.
    io.linearLaunch.ready := idleLinearLaunchReady && !io.qkvLaunch.valid
    when(io.qkvLaunch.fire) {
      when(launchInvariantLegal) {
        activeLaunch := io.qkvLaunch.payload
        activeKind := BitNetResidentLinearKind.Qkv
        assemblerDone := False
        controllerQkvDone := False
        assemblerEmittedBase := assembler.io.emittedFeatures
        imageMetadataValid := False
        phase := Phase.AcquireLinear
      } otherwise {
        faultSticky := True
        faultBitsReg(BitNetQkvAttentionIslandFault.LaunchInvariant) := True
        qkvCompletionReg.requestId := launch.requestId
        qkvCompletionReg.layer := launch.layer
        qkvCompletionReg.sequenceId := launch.sequenceId
        qkvCompletionReg.success := False
        qkvCompletionReg.fatal := False
        qkvCompletionReg.faultCode := 0
        qkvCompletionReg.faultCode(
          BitNetQkvAttentionIslandFault.LaunchInvariant) := True
        qkvCompletionReg.emittedFeatures := 0
        qkvCompletionValid := True
      }
    }

    val genericLaunchLegal = BitNetResidentLinearProtocol.launchLegal(
      io.linearLaunch.payload, cfg.linearEndpoint)
    when(io.linearLaunch.fire) {
      when(genericLaunchLegal) {
        activeGenericLaunch := io.linearLaunch.payload
        activeKind := io.linearLaunch.kind
        genericEndpointDone := False
        projectionAssemblyDone := False
        fusedResidualDone := False
        endpointEmittedBase := finalEndpoint.io.emittedPairs
        imageMetadataValid := False
        phase := Phase.AcquireLinear
      } otherwise {
        faultSticky := True
        faultBitsReg(BitNetQkvAttentionIslandFault.LaunchInvariant) := True
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

    when(phase === Phase.AcquireLinear &&
        leaseFabric.io.linearAcquire.fire) {
      phase := Phase.StartQkvConsumers
    }

    when(phase === Phase.StartQkvConsumers) {
      if (controlHarness) {
        when(activeKind === BitNetResidentLinearKind.Qkv) {
          consumersStarted := True
        } otherwise {
          genericEndpointStarted := True
        }
        phase := Phase.StartFeeder
      } else {
        when(activeKind === BitNetResidentLinearKind.Qkv) {
          val bothReady =
            controller.io.qkvRound.ready && assembler.io.round.ready
          controller.io.qkvRound.valid := bothReady
          assembler.io.round.valid := bothReady
          when(controller.io.qkvRound.fire && assembler.io.round.fire) {
            consumersStarted := True
            phase := Phase.StartFeeder
          }
        } otherwise {
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
              genericEndpointStarted := True
              phase := Phase.StartFeeder
            }
          } otherwise {
            finalEndpoint.io.launch.valid := True
            when(finalEndpoint.io.launch.fire) {
              genericEndpointStarted := True
              phase := Phase.StartFeeder
            }
          }
        }
      }
    }

    when(phase === Phase.StartFeeder) {
      if (controlHarness) {
        feederStarted := True
        phase := Phase.RunQkv
      } else {
        feeder.io.task.valid := True
        when(feeder.io.task.fire) {
          feederStarted := True
          phase := Phase.RunQkv
        }
      }
    }

    val assemblerCompletionMatches =
      selectedAssemblerCompletion.sequenceId === activeLaunch.sequenceId &&
      selectedAssemblerCompletion.emittedFeatures ===
        U(cfg.assembler.totalFeatures, 32 bits) &&
      selectedAssemblerCompletion.faultCode ===
        BitNetQkvAssemblerFaultCode.None
    val controllerCompletionMatches =
      selectedControllerQkvCompletion.requestId ===
        activeLaunch.qkvRound.requestId &&
      selectedControllerQkvCompletion.layer === activeLaunch.qkvRound.layer &&
      selectedControllerQkvCompletion.tokenBase ===
        activeLaunch.qkvRound.tokenBase &&
      selectedControllerQkvCompletion.tokenMask ===
        activeLaunch.qkvRound.tokenMask
    val endpointCompletionMatches =
      selectedEndpointCompletion.requestId ===
        activeGenericLaunch.requestId &&
      selectedEndpointCompletion.layer === activeGenericLaunch.layer &&
      selectedEndpointCompletion.kind === activeKind &&
      selectedEndpointCompletion.sequenceId ===
        activeGenericLaunch.sequenceId &&
      !selectedEndpointCompletion.fatal
    val projectionAssemblyCompletionMatches =
      selectedProjectionAssemblyCompletion.requestId ===
        activeGenericLaunch.requestId &&
      selectedProjectionAssemblyCompletion.layer ===
        activeGenericLaunch.layer &&
      selectedProjectionAssemblyCompletion.kind === activeKind &&
      selectedProjectionAssemblyCompletion.sequenceId ===
        activeGenericLaunch.sequenceId
    val fusedResidualCompletionMatches =
      selectedFusedResidualCompletion.requestId ===
        activeGenericLaunch.requestId &&
      selectedFusedResidualCompletion.layer === activeGenericLaunch.layer &&
      selectedFusedResidualCompletion.kind === activeKind &&
      selectedFusedResidualCompletion.sequenceId ===
        activeGenericLaunch.sequenceId

    when(selectedAssemblerCompletion.fire) {
      assemblerDone := True
      assemblerCompletionReg := selectedAssemblerCompletion.payload
    }
    when(selectedControllerQkvCompletion.fire) {
      controllerQkvDone := True
      controllerQkvCompletionReg := selectedControllerQkvCompletion.payload
    }
    when(selectedEndpointCompletion.fire) {
      genericEndpointDone := True
      endpointCompletionReg := selectedEndpointCompletion.payload
      when(!selectedEndpointCompletion.success ||
          !endpointCompletionMatches) {
        faultSticky := True
        faultBitsReg(BitNetQkvAttentionIslandFault.LinearEndpoint) := True
      }
      when(!endpointCompletionMatches) {
        endpointCompletionReg.success := False
        endpointCompletionReg.fatal := False
        endpointCompletionReg.faultCode :=
          selectedEndpointCompletion.faultCode |
            B(BigInt(1) << BitNetResidentLinearFault.EndpointCompletion,
              16 bits)
      }
    }
    when(selectedProjectionAssemblyCompletion.fire) {
      projectionAssemblyDone := True
      projectionAssemblyCompletionReg :=
        selectedProjectionAssemblyCompletion.payload
      when(!selectedProjectionAssemblyCompletion.success ||
          !projectionAssemblyCompletionMatches) {
        faultSticky := True
        faultBitsReg(BitNetQkvAttentionIslandFault.FusedResidual) := True
      }
    }
    when(selectedFusedResidualCompletion.fire) {
      fusedResidualDone := True
      fusedResidualCompletionReg := selectedFusedResidualCompletion.payload
      when(!selectedFusedResidualCompletion.success ||
          !fusedResidualCompletionMatches) {
        faultSticky := True
        faultBitsReg(BitNetQkvAttentionIslandFault.FusedResidual) := True
      }
    }
    val assemblerDoneAfter = assemblerDone ||
      selectedAssemblerCompletion.fire
    val controllerDoneAfter = controllerQkvDone ||
      selectedControllerQkvCompletion.fire
    val genericEndpointDoneAfter = genericEndpointDone ||
      selectedEndpointCompletion.fire
    val projectionAssemblyDoneAfter = projectionAssemblyDone ||
      selectedProjectionAssemblyCompletion.fire
    val fusedResidualDoneAfter = fusedResidualDone ||
      selectedFusedResidualCompletion.fire
    val assemblerSuccessAfter = Mux(selectedAssemblerCompletion.fire,
      selectedAssemblerCompletion.success && assemblerCompletionMatches,
      assemblerCompletionReg.success &&
        assemblerCompletionReg.sequenceId === activeLaunch.sequenceId &&
        assemblerCompletionReg.emittedFeatures ===
          U(cfg.assembler.totalFeatures, 32 bits) &&
        assemblerCompletionReg.faultCode ===
          BitNetQkvAssemblerFaultCode.None)
    val controllerSuccessAfter = Mux(selectedControllerQkvCompletion.fire,
      selectedControllerQkvCompletion.success && controllerCompletionMatches,
      controllerQkvCompletionReg.success &&
        controllerQkvCompletionReg.requestId === activeLaunch.requestId &&
        controllerQkvCompletionReg.layer === activeLaunch.layer &&
        controllerQkvCompletionReg.tokenBase ===
          activeLaunch.qkvRound.tokenBase &&
        controllerQkvCompletionReg.tokenMask === activeLaunch.tokenMask)
    val genericEndpointSuccessAfter = Mux(selectedEndpointCompletion.fire,
      selectedEndpointCompletion.success && endpointCompletionMatches,
      endpointCompletionReg.success &&
        endpointCompletionReg.requestId === activeGenericLaunch.requestId &&
        endpointCompletionReg.layer === activeGenericLaunch.layer &&
        endpointCompletionReg.kind === activeKind &&
        endpointCompletionReg.sequenceId === activeGenericLaunch.sequenceId)
    val projectionAssemblySuccessAfter = Mux(
      selectedProjectionAssemblyCompletion.fire,
      selectedProjectionAssemblyCompletion.success &&
        projectionAssemblyCompletionMatches,
      projectionAssemblyCompletionReg.success &&
        projectionAssemblyCompletionReg.requestId ===
          activeGenericLaunch.requestId &&
        projectionAssemblyCompletionReg.layer === activeGenericLaunch.layer &&
        projectionAssemblyCompletionReg.kind === activeKind &&
        projectionAssemblyCompletionReg.sequenceId ===
          activeGenericLaunch.sequenceId)
    val fusedResidualSuccessAfter = Mux(
      selectedFusedResidualCompletion.fire,
      selectedFusedResidualCompletion.success &&
        fusedResidualCompletionMatches,
      fusedResidualCompletionReg.success &&
        fusedResidualCompletionReg.requestId ===
          activeGenericLaunch.requestId &&
        fusedResidualCompletionReg.layer === activeGenericLaunch.layer &&
        fusedResidualCompletionReg.kind === activeKind &&
        fusedResidualCompletionReg.sequenceId ===
          activeGenericLaunch.sequenceId)

    when(phase === Phase.RunQkv &&
        activeKind === BitNetResidentLinearKind.Qkv &&
        assemblerDoneAfter &&
        controllerDoneAfter && assemblerSuccessAfter &&
        controllerSuccessAfter) {
      phase := Phase.ReleaseLinear
    }
    when(phase === Phase.RunQkv &&
        activeKind =/= BitNetResidentLinearKind.Qkv &&
        !fusedProjectionKind && selectedEndpointCompletion.fire) {
      phase := Phase.ReleaseLinear
    }
    val fusedBarrierDone = genericEndpointDoneAfter &&
      projectionAssemblyDoneAfter && fusedResidualDoneAfter
    val fusedBarrierSuccess = genericEndpointSuccessAfter &&
      projectionAssemblySuccessAfter && fusedResidualSuccessAfter
    if (controlHarness) {
      // The focused control harness can inject a failed endpoint without
      // having performed physical writes, preserving its non-fatal recovery
      // test.  Production failures below remain fail-stop.
      when(phase === Phase.RunQkv && fusedProjectionKind &&
          fusedBarrierDone) {
        phase := Phase.ReleaseLinear
      }
    } else {
      when(phase === Phase.RunQkv && fusedProjectionKind &&
          fusedBarrierDone && fusedBarrierSuccess) {
        // HiddenResident destination-epoch commit barrier.  The matrix lease
        // cannot release while a final scale/write is held under backpressure.
        phase := Phase.ReleaseLinear
      }
    }
    when(phase === Phase.ReleaseLinear &&
        leaseFabric.io.linearRelease.fire) {
      when(activeKind === BitNetResidentLinearKind.Qkv) {
        qkvCompletionReg.requestId := activeLaunch.requestId
        qkvCompletionReg.layer := activeLaunch.layer
        qkvCompletionReg.sequenceId := activeLaunch.sequenceId
        qkvCompletionReg.success := True
        qkvCompletionReg.fatal := False
        qkvCompletionReg.faultCode := 0
        qkvCompletionReg.emittedFeatures :=
          U(cfg.assembler.totalFeatures, 32 bits)
        qkvCompletionValid := True
        imageMetadataValid := True
      } otherwise {
        linearCompletionReg := endpointCompletionReg
        linearCompletionValid := True
        imageMetadataValid := False
      }
      phase := Phase.Idle
    }

    val attentionIdentityMatches =
      io.attentionCommand.requestId === activeLaunch.requestId &&
      io.attentionCommand.layer === activeLaunch.layer &&
      io.attentionCommand.tokenBase === activeLaunch.qkvRound.tokenBase &&
      io.attentionCommand.mode === activeLaunch.mode &&
      io.attentionCommand.activeTokenMask === activeLaunch.tokenMask &&
      io.attentionCommand.tokenId.asBits ===
        activeLaunch.qkvRound.tokenId.asBits
    io.attentionCommand.ready := phase === Phase.Idle &&
      imageMetadataValid && controllerImageReady && !qkvCompletionValid &&
      !attentionCompletionValid && !fatalLatched
    when(io.attentionCommand.fire) {
      when(attentionIdentityMatches) {
        heldAttention := io.attentionCommand.payload
        phase := Phase.AcquireAttention
      } otherwise {
        faultSticky := True
        faultBitsReg(BitNetQkvAttentionIslandFault.AttentionCommand) := True
        outwardAttentionCompletion.requestId := io.attentionCommand.requestId
        outwardAttentionCompletion.layer := io.attentionCommand.layer
        outwardAttentionCompletion.tokenBase := io.attentionCommand.tokenBase
        outwardAttentionCompletion.completedLaneMask := 0
        outwardAttentionCompletion.success := False
        outwardAttentionCompletion.faultCode := 0
        outwardAttentionCompletion.faultCode(
          BitNetResidentAttentionFault.NumericOrPhase) := True
        attentionCompletionValid := True
      }
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

    val attentionCompletionMatches =
      selectedAttentionCompletion.requestId === heldAttention.requestId &&
      selectedAttentionCompletion.layer === heldAttention.layer &&
      selectedAttentionCompletion.tokenBase === heldAttention.tokenBase &&
      selectedAttentionCompletion.completedLaneMask ===
        Mux(heldAttention.mode === BitNetMode.Decode,
          B(1, cfg.model.tokenParallelism bits),
          heldAttention.activeTokenMask)
    when(selectedAttentionCompletion.fire) {
      attentionCompletionReg := selectedAttentionCompletion.payload
      when(!attentionCompletionMatches) {
        faultSticky := True
        faultBitsReg(
          BitNetQkvAttentionIslandFault.AttentionCompletion) := True
      }
      phase := Phase.ReleaseAttention
    }
    when(phase === Phase.ReleaseAttention &&
        leaseFabric.io.attentionRelease.fire) {
      outwardAttentionCompletion := attentionCompletionReg
      when(!(attentionCompletionReg.success &&
          attentionCompletionReg.requestId === heldAttention.requestId &&
          attentionCompletionReg.layer === heldAttention.layer &&
          attentionCompletionReg.tokenBase === heldAttention.tokenBase)) {
        outwardAttentionCompletion.success := False
        outwardAttentionCompletion.faultCode(
          BitNetResidentAttentionFault.NumericOrPhase) := True
      }
      attentionCompletionValid := True
      imageMetadataValid := False
      phase := Phase.Idle
    }

    // ------------------------------------------------------------------
    // Post-start failures are fatal because the existing resident QKV
    // controller has no abort input.  Report that distinction explicitly and
    // gate every error-bearing payload away from it.
    // ------------------------------------------------------------------
    val linearOwnedPhase = phase === Phase.AcquireLinear ||
      phase === Phase.StartQkvConsumers || phase === Phase.StartFeeder ||
      phase === Phase.RunQkv || phase === Phase.ReleaseLinear
    val attentionOwnedPhase = phase === Phase.AcquireAttention ||
      phase === Phase.StartAttention || phase === Phase.RunAttention ||
      phase === Phase.ReleaseAttention
    val unexpectedLinearSv = if (!cfg.lease.productionPruneGenericSv)
      (0 until engines).map(
        leaseFabric.io.linearSvResult(_).valid).reduce(_ || _)
    else False
    val endpointCompletionFault = if (controlHarness) False else
      selectedEndpointCompletion.fire &&
        (!selectedEndpointCompletion.success || !endpointCompletionMatches)
    val projectionCompletionFault = if (controlHarness) False else
      selectedProjectionAssemblyCompletion.fire &&
        (!selectedProjectionAssemblyCompletion.success ||
          !projectionAssemblyCompletionMatches)
    val residualCompletionFault = if (controlHarness) False else
      selectedFusedResidualCompletion.fire &&
        (!selectedFusedResidualCompletion.success ||
          !fusedResidualCompletionMatches)
    val componentFaultBits = Bits(16 bits)
    componentFaultBits := 0
    componentFaultBits(BitNetQkvAttentionIslandFault.Feeder) :=
      linearOwnedPhase && feeder.io.protocolFaultSticky
    componentFaultBits(BitNetQkvAttentionIslandFault.Assembler) :=
      linearOwnedPhase &&
        activeKind === BitNetResidentLinearKind.Qkv &&
        assembler.io.protocolFaultSticky
    componentFaultBits(BitNetQkvAttentionIslandFault.LinearEndpoint) :=
      linearOwnedPhase && activeKind =/=
        BitNetResidentLinearKind.Qkv &&
        (finalEndpoint.io.protocolFaultSticky ||
          endpointCompletionFault)
    componentFaultBits(BitNetQkvAttentionIslandFault.QkvController) :=
      ((linearOwnedPhase &&
        activeKind === BitNetResidentLinearKind.Qkv) ||
        attentionOwnedPhase) &&
        controller.io.protocolFaultSticky
    componentFaultBits(BitNetQkvAttentionIslandFault.Lease) :=
      leaseFabric.io.leaseFaultSticky
    componentFaultBits(BitNetQkvAttentionIslandFault.SpuArithmetic) :=
      arithmeticPool.io.protocolFaultSticky
    componentFaultBits(BitNetQkvAttentionIslandFault.FusedResidual) :=
      linearOwnedPhase && fusedProjectionKind &&
        (projectionFeatureAssembler.io.protocolFaultSticky ||
          fusedResidual.io.protocolFaultSticky ||
          projectionCompletionFault || residualCompletionFault)
    componentFaultBits(BitNetQkvAttentionIslandFault.ResidentTensor) :=
      residentTensor.io.protocolFaultSticky || residentTensor.io.quarantined
    componentFaultBits(BitNetQkvAttentionIslandFault.MatrixCluster) :=
      leaseFabric.io.clusterProtocolFaultSticky.orR
    componentFaultBits(
      BitNetQkvAttentionIslandFault.UnexpectedMatrixReturn) :=
      linearOwnedPhase && unexpectedLinearSv
    componentFaultBits(BitNetQkvAttentionIslandFault.CompletionMetadata) :=
      phase === Phase.RunQkv &&
        activeKind === BitNetResidentLinearKind.Qkv &&
        ((selectedAssemblerCompletion.fire &&
          (!selectedAssemblerCompletion.success ||
            !assemblerCompletionMatches)) ||
         (selectedControllerQkvCompletion.fire &&
          (!selectedControllerQkvCompletion.success ||
            !controllerCompletionMatches)))
    val fatalEvent = (linearOwnedPhase || attentionOwnedPhase) &&
      componentFaultBits.orR

    // A faulting resident image must abort Query capture/replay before the
    // island reports or releases the physical matrix lease.  Local Query
    // fail-close remains same-cycle; the sticky island latch preserves the
    // terminal diagnostic after cleanup.
    controller.io.queryAbort := fatalLatched || fatalEvent ||
      residentTensor.io.quarantined

    when(fatalEvent && !fatalLatched) {
      fatalLatched := True
      faultSticky := True
      faultBitsReg := faultBitsReg | componentFaultBits
      imageMetadataValid := False
      when(linearOwnedPhase &&
          activeKind === BitNetResidentLinearKind.Qkv &&
          !qkvCompletionValid) {
        qkvCompletionReg.requestId := activeLaunch.requestId
        qkvCompletionReg.layer := activeLaunch.layer
        qkvCompletionReg.sequenceId := activeLaunch.sequenceId
        qkvCompletionReg.success := False
        qkvCompletionReg.fatal := True
        qkvCompletionReg.faultCode := componentFaultBits
        qkvCompletionReg.emittedFeatures := Mux(
          selectedAssemblerCompletion.fire,
          selectedAssemblerCompletion.emittedFeatures,
          assembler.io.emittedFeatures - assemblerEmittedBase)
        fatalQkvReportPending := True
      }
      when(linearOwnedPhase &&
          activeKind =/= BitNetResidentLinearKind.Qkv &&
          !linearCompletionValid) {
        linearCompletionReg.requestId := activeGenericLaunch.requestId
        linearCompletionReg.layer := activeGenericLaunch.layer
        linearCompletionReg.kind := activeKind
        linearCompletionReg.sequenceId := activeGenericLaunch.sequenceId
        linearCompletionReg.success := False
        linearCompletionReg.fatal := True
        linearCompletionReg.faultCode := componentFaultBits
        linearCompletionReg.acceptedResultBeats := 0
        linearCompletionReg.emittedPairs :=
          finalEndpoint.io.emittedPairs - endpointEmittedBase
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
      phase := Phase.FatalStop
    }

    // Fatal status is observable immediately through fatalFailStop/faultBits,
    // but a terminal Stream event is still a lease-retirement barrier.  This
    // prevents software or an outer layer sequencer from recycling storage
    // while the physical arrays are draining an un-abortable transaction.
    when(phase === Phase.FatalStop && !leaseFabric.io.leaseActive) {
      when(fatalQkvReportPending && !qkvCompletionValid) {
        qkvCompletionValid := True
        fatalQkvReportPending := False
      }
      when(fatalAttentionReportPending && !attentionCompletionValid) {
        attentionCompletionValid := True
        fatalAttentionReportPending := False
      }
      when(fatalLinearReportPending && !linearCompletionValid) {
        linearCompletionValid := True
        fatalLinearReportPending := False
      }
    }

    when(componentFaultBits.orR) {
      faultSticky := True
      faultBitsReg := faultBitsReg | componentFaultBits
    }

    io.qkvImageReady := imageMetadataValid && controllerImageReady &&
      !fatalLatched && !residentTensor.io.quarantined
    io.busy := phase =/= Phase.Idle || qkvCompletionValid ||
      linearCompletionValid || attentionCompletionValid || feeder.io.active ||
      assembler.io.busy || finalEndpoint.io.busy ||
      projectionFeatureAssembler.io.busy || fusedResidual.io.busy ||
      (if (controlHarness) False else controller.io.busy)
    io.fatalFailStop := fatalLatched
    io.protocolFaultSticky := faultSticky ||
      leaseFabric.io.protocolFaultSticky || feeder.io.protocolFaultSticky ||
      assembler.io.protocolFaultSticky ||
      finalEndpoint.io.protocolFaultSticky ||
      projectionFeatureAssembler.io.protocolFaultSticky ||
      fusedResidual.io.protocolFaultSticky ||
      controller.io.protocolFaultSticky ||
      arithmeticPool.io.protocolFaultSticky ||
      residentTensor.io.protocolFaultSticky || residentTensor.io.quarantined
    io.faultBits := faultBitsReg | componentFaultBits
    io.linearLeaseGranted := leaseFabric.io.linearGranted
    io.attentionLeaseGranted := leaseFabric.io.attentionGranted
    io.leaseReleaseSafe := leaseFabric.io.releaseSafe
    io.leaseTag := leaseFabric.io.leaseTag
    io.matrixSequenceActive := leaseFabric.io.sequenceActive
    io.matrixDraining := leaseFabric.io.draining
    io.activeLinearKind := activeKind
  }
}

object GenerateBitNetResidentQkvAttentionComputeIsland extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-qkv-attention-compute-island")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentQkvAttentionComputeIsland())
}

/** Preferred production entry: the implementation above now owns every
  * QKV/O/GateUp/Down/LM linear transaction as well as resident attention. */
object GenerateBitNetResidentComputeIsland extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-compute-island")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentQkvAttentionComputeIsland())
}
