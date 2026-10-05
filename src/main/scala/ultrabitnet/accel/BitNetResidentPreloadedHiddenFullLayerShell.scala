package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Locked producer of the sole physical activation-fabric lease. */
object BitNetResidentFullLayerActivationLeaseSource
    extends SpinalEnum(binarySequential) {
  val Idle, HiddenPreload, LayerEvent, LmHead = newElement()
}

/**
  * Static contract of the first production-shaped full-layer composition.
  *
  * Every child is kept in the configuration so equality checks fail during
  * elaboration instead of silently truncating epochs, tags or memory ports.
  */
case class BitNetResidentPreloadedHiddenFullLayerShellConfig(
    event: BitNetResidentLayerEventSchedulerConfig,
    datapath: BitNetResidentSharedTp4ComputeDatapathConfig,
    matrixRope: BitNetResidentMatrixRopeEventBarrierConfig,
    layerRms: BitNetResidentRmsNormSubsystemConfig,
    lm: BitNetResidentLmFetchConfig,
    memory: BitNetProductionPlMemoryOwnershipHubConfig,
    staging: BitNetPsToPlWeightStagingConfig,
    copy: BitNetWeightCopyAxiConfig,
    bridge: BitNetPsToPlWeightHubBridgeConfig,
    barrier: BitNetResidentLayerWeightStagingBarrierConfig) {
  val model: BitNetConfig = event.model
  val layout: BitNetStreamingActivationLayout =
    BitNetStreamingActivationLayout(model)
  val matrix: BitNetResidentMatrixEventCoordinatorConfig =
    matrixRope.matrix
  val stage: BitNetResidentMatrixStageControlSubsystemConfig =
    matrix.prepared.matrix
  val crossbar: BitNetBankLocalCrossbarConfig = memory.memory.crossbar

  require(model == BitNetConfig.Production)
  require(event.epochWidth == 8 && event.slotCount == 2,
    "the production event/image identity is one 8-bit, two-slot domain")
  require(matrix.event == event && matrixRope.rope.event == event,
    "scheduler, matrix and RoPE must retain the complete event identity")
  require(matrixRope.rope.bridge == stage.qkv,
    "RoPE must preprocess the exact segmented-QKV bridge")
  require(stage.qkv.resident == datapath.pipeline,
    "MatrixStage QKV and the unique TP4 controller must be identical")
  require(stage.endpoint == datapath.linearEndpoint)
  require(stage.stage.feeder == datapath.feeder)
  require(stage.weight.prefetch.feeder == datapath.feeder)
  require(stage.weight.trustGeneratedProjection,
    "canonical full-layer may remove duplicate weight validation only behind the unique MatrixStage generator")
  require(matrix.prepared.binder.linear == datapath.linearEndpoint)
  require(matrix.activation.tile.feeder == datapath.feeder)

  require(layerRms.externalArithmetic &&
    layerRms.endpoint.externalArithmetic,
    "layer RMS must not instantiate a private arithmetic core")
  require(layerRms.endpoint.arithmetic == matrix.rms)
  require(layerRms.gamma.endpoint == layerRms.endpoint)
  require(matrix.downInput.postGate.arithmetic == matrix.rms)
  require(matrix.prepared.gate.rmsConfig == matrix.rms)

  require(matrix.prepared.gateScale.arithmetic == datapath.arithmetic)
  require(stage.gate.gate.arithmetic == datapath.arithmetic,
    "both logical FFN clients must borrow physical FfnGate client 2")
  require(stage.gate.gate.strictFeatureOrder,
    "the production resident microprogram must use expected-index FFN " +
      "authentication instead of a feature-sized scatter seen RAM")

  require(lm.resident.model == model)
  require(lm.resident.linear == datapath.linearEndpoint)
  require(lm.resident.quant == matrix.quant,
    "projection and LM must share exactly one dynamic-i8 service")
  require(lm.resident.coupledMatrixOutputs,
    "canonical LM weight/activation outputs must enter the same atomic feeder")
  require(lm.epochWidth == event.epochWidth)

  require(layerRms.endpoint.activationLayout == layout)
  require(matrix.activation.activationLayout == layout)
  require(matrix.downInput.postGate.activationLayout == layout)
  require(datapath.attention.activationLayout == layout)
  require(lm.resident.activationLayout == layout)

  require(staging.model == model && staging.epochWidth == event.epochWidth,
    "weight staging and resident events must share model/epoch identity")
  require(staging.resident == stage.weight.agu.layout &&
    staging.resident == matrixRope.rope.layout,
    "copy destinations, matrix AGU and RoPE must use one resident image")
  require(copy.staging == staging,
    "the page-copy engine must retire this exact staging controller")
  require(bridge.copy == copy && bridge.crossbar == crossbar,
    "the copy bridge must preserve the copy and memory-hub ABIs")
  require(bridge.sourceBank == 1 && bridge.destinationBank == 0,
    "production staging is PS bank 1 to PL bank 0 only")
  require(barrier.event == event && barrier.staging == staging,
    "the layer barrier must authenticate the same event/staging identity")

  require(stage.weight.prefetch.crossbar == crossbar)
  require(datapath.pipeline.crossbar == crossbar,
    "TP4 KV traffic and the full-layer memory hub must share one crossbar ABI")
  require(matrix.prepared.scales.crossbar == crossbar)
  require(matrix.downInput.postGate.crossbar == crossbar)
  require(matrixRope.rope.crossbar == crossbar)
  require(layerRms.gamma.crossbar == crossbar)
  require(lm.crossbar == crossbar)
  require(staging.resident.bankCapacities ==
    stage.weight.agu.layout.bankCapacities)
  require(memory.memory.axiAddressWidth == 40,
    "the AXU3EG physical DDR windows require a 40-bit AXI address")
  require(memory.memory.maxOutstanding == 4,
    "production keeps four logical bursts per bank; sixteen copies the wide " +
      "tag/state table into an unplaceable slow-domain control plane")
  require(memory.memory.physicalBase == Vector(
    BigInt(0),
    BigInt("800000000", 16),
    BigInt("810000000", 16),
    BigInt("820000000", 16),
    BigInt("830000000", 16)),
    "bank 0 is PL DDR; banks 1..4 are fixed PS HP physical windows")
  require(layerRms.gamma.bank == 0 &&
    matrix.prepared.scales.bank == 0 &&
    matrix.downInput.postGate.bank == 0 &&
    matrixRope.rope.bank == 0,
    "the four resident AUX users share the guarded bank-zero client")
}

object BitNetResidentPreloadedHiddenFullLayerShellConfig {
  /** One internally consistent production image. */
  def production(): BitNetResidentPreloadedHiddenFullLayerShellConfig = {
    val model = BitNetConfig.Production
    val layout = BitNetStreamingActivationLayout(model)
    val resident = BitNetFiveBankStripedLayout(model)
    val crossbar = BitNetBankLocalCrossbarConfig()
    val event = BitNetResidentLayerEventSchedulerConfig(
      model = model, epochWidth = 8, slotCount = 2)
    val qkvPostprocess = BitNetQkvPostprocessConfig(
      tokenLanes = model.tokenParallelism,
      externalDynamicQuant = true)
    val continuous = BitNetContinuousMatrixConfig(
      base = BitNetUnifiedMatrixConfig(numChains = 8))
    val attention = BitNetResidentAttentionSubsystemConfig(
      continuous = continuous,
      crossbar = crossbar,
      qkvPostprocess = qkvPostprocess)
    val pipeline = BitNetResidentQkvAttentionPipelineConfig(
      attention = attention,
      kvWriter = BitNetPackedKvAppendWriterConfig(
        qkv = qkvPostprocess, crossbar = crossbar))
    val datapath = BitNetResidentSharedTp4ComputeDatapathConfig(
      pipeline = pipeline)
    val memory = BitNetProductionPlMemoryOwnershipHubConfig(
      memory = BitNetFiveBankAxiMemoryComplexConfig(
        crossbar = crossbar,
        physicalBase = Vector(
          BigInt(0),
          BigInt("800000000", 16),
          BigInt("810000000", 16),
          BigInt("820000000", 16),
          BigInt("830000000", 16)),
        axiAddressWidth = 40,
        // Four entries cover Matrix/LM/KV/AUX ownership overlap while the
        // existing crossbar response FIFOs absorb arbitration latency.  The
        // production hub regression already exercises this exact geometry.
        maxOutstanding = 4))
    val baseMatrix = BitNetResidentMatrixEventCoordinatorConfig(event = event)
    val baseStage = baseMatrix.prepared.matrix
    val stageLaunch = baseStage.stage.copy(feeder = datapath.feeder)
    val s64Prefetch = baseStage.weight.prefetch.copy(feeder = datapath.feeder)
    val strictGate = baseStage.gate.gate.copy(
      strictFeatureOrder = true)
    val trustedWeight = baseStage.weight.copy(
      stage = stageLaunch,
      prefetch = s64Prefetch,
      trustGeneratedProjection = true)
    val gateProjection = baseStage.gate.join.projection.copy(
      endpoint = datapath.linearEndpoint)
    val gateJoin = baseStage.gate.join.copy(projection = gateProjection)
    val strictGatePipeline = baseStage.gate.copy(
      projection = trustedWeight,
      join = gateJoin,
      gate = strictGate)
    val productionQkvBridge = baseStage.qkv.copy(
      stage = stageLaunch,
      endpoint = datapath.linearEndpoint,
      resident = pipeline)
    val strictMatrixStage = baseStage.copy(
      weight = trustedWeight,
      qkv = productionQkvBridge,
      gate = strictGatePipeline)
    val s64ActivationTile = baseMatrix.prepared.activation.tile.copy(
      feeder = datapath.feeder)
    val s64Activation = baseMatrix.prepared.activation.copy(
      tile = s64ActivationTile)
    val s64Binder = baseMatrix.prepared.binder.copy(
      linear = datapath.linearEndpoint,
      activation = s64Activation)
    val s64Prepared: BitNetResidentPreparedMatrixStageCoordinatorConfig =
      baseMatrix.prepared.copy(
      matrix = strictMatrixStage,
      activation = s64Activation,
      binder = s64Binder)
    val s64DownInput: BitNetResidentFfnDownInputCoordinatorConfig =
      baseMatrix.downInput.copy(
      activation = s64Activation)
    val matrix = baseMatrix.copy(
      prepared = s64Prepared,
      downInput = s64DownInput)
    val rope = BitNetResidentRopeCoefficientStagerConfig(
      event = event,
      bridge = matrix.prepared.matrix.qkv,
      crossbar = memory.memory.crossbar,
      layout = resident,
      bank = 0)
    val matrixRope = BitNetResidentMatrixRopeEventBarrierConfig(
      matrix = matrix, rope = rope)

    val rmsEndpoint = BitNetResidentRmsNormEndpointConfig(
      model = model,
      activationLayout = layout,
      arithmetic = matrix.rms,
      externalArithmetic = true)
    val layerRms = BitNetResidentRmsNormSubsystemConfig(
      endpoint = rmsEndpoint,
      gamma = BitNetResidentRmsGammaStagerConfig(
        endpoint = rmsEndpoint,
        crossbar = memory.memory.crossbar,
        bank = 0,
        addresses = BitNetResidentRmsGammaAddressLayout.production()))

    val lmResident = BitNetResidentStreamingLmHeadConfig(
      model = model,
      linear = datapath.linearEndpoint,
      streaming = BitNetStreamingLmHeadConfig.production(
        tileFeatures = continuous.base.siteCount),
      quant = matrix.quant,
      activationLayout = layout,
      coupledMatrixOutputs = true)
    val lm = BitNetResidentLmFetchConfig(
      resident = lmResident,
      crossbar = memory.memory.crossbar,
      epochWidth = event.epochWidth)

    val staging = BitNetPsToPlWeightStagingConfig(
      model = model,
      resident = resident,
      epochWidth = event.epochWidth)
    val copy = BitNetWeightCopyAxiConfig(staging = staging)
    val bridge = BitNetPsToPlWeightHubBridgeConfig(
      copy = copy,
      crossbar = memory.memory.crossbar,
      sourceBank = 1,
      destinationBank = 0)
    val barrier = BitNetResidentLayerWeightStagingBarrierConfig(
      event = event,
      staging = staging)

    BitNetResidentPreloadedHiddenFullLayerShellConfig(
      event = event,
      datapath = datapath,
      matrixRope = matrixRope,
      layerRms = layerRms,
      lm = lm,
      memory = memory,
      staging = staging,
      copy = copy,
      bridge = bridge,
      barrier = barrier)
  }
}

/**
  * One atomic software submission for a preloaded-hidden transformer round.
  *
  * `psSourceBase` is bank-1-local, not a physical AXI address.  The suggested
  * Linux staging allocation starts at local offset 0x40000000, which the
  * fixed bank-1 window translates to physical 0x840000000.  Hardware does not
  * hard-code that offset: software supplies both the aligned base and the
  * complete allocation size on every accepted run.  `run.epoch` is a fresh
  * workload identity for every round, while `imageEpoch` identifies the
  * immutable model image and stays constant across prefill/decode runs until
  * those weight bytes change.
  */
case class BitNetResidentPreloadedHiddenFullLayerStart(
    cfg: BitNetResidentPreloadedHiddenFullLayerShellConfig) extends Bundle {
  val run = BitNetResidentLayerEventRun(cfg.event)
  val imageEpoch = UInt(cfg.staging.epochWidth bits)
  val psSourceBase = UInt(cfg.model.addressWidth bits)
  val psSourceBytes = UInt((cfg.model.addressWidth + 1) bits)
}

/**
  * One preloaded-hidden transformer round entirely below the PS/PL boundary.
  *
  * A successful HiddenResident preload is consumed by one real scheduler
  * start.  The shell then executes all thirty transformer layers, optional
  * final RMS and the complete streaming LM/argmax transaction.  Every event
  * completion below the scheduler is a physical retirement barrier; there is
  * no synthetic prefetch, compute, activation or LM acknowledgement.
  *
  * This is deliberately not the canonical production top yet.  It has no
  * embedding lookup and no autonomous prompt/decode token loop, so the next
  * round needs another typed HiddenResident preload.  Moreover only the FFN
  * logical arithmetic clients currently have an adapter to the shared TP4
  * SPU pool.  External SPU ports 0/2/3/4 are sealed and monitored; RMS and
  * dynamic quant each remain a unique dedicated service until that later
  * arithmetic-pool refactor is implemented.
  */
class BitNetResidentPreloadedHiddenFullLayerShell(
    cfg: BitNetResidentPreloadedHiddenFullLayerShellConfig =
      BitNetResidentPreloadedHiddenFullLayerShellConfig.production(),
    useVendorPrimitive: Boolean = true) extends Component {
  private val model = cfg.model
  private val layout = cfg.layout
  private val matrixCfg = cfg.matrix
  private val stageCfg = cfg.stage
  private val activationCfg = matrixCfg.activation
  private val binderCfg = matrixCfg.prepared.binder
  private val prefetchCfg = stageCfg.weight.prefetch
  private val crossbar = cfg.crossbar
  private val readCfg = cfg.memory.memory.read(0)
  private val writeCfg = cfg.memory.memory.write(0)
  private val preloadCfg = BitNetResidentHiddenPreloadEndpointConfig(
    model, layout)
  private val terminalCfg = BitNetResidentLmTerminalJoinConfig(
    schedule = cfg.event, lm = cfg.lm.resident)
  private val unifiedTensorCfg =
    BitNetUnifiedResidentTensorProductionShellConfig()
  private val unifiedScratchpadCfg = unifiedTensorCfg.backend.scratchpad
  private val clientCount = BitNetActivationClient.elements.size
  private val matrixClient = BitNetActivationClient.Matrix.position
  private val ffnExternalSpu =
    BitNetSpuArithmeticClient.FfnGate - 1
  private val preloadLeaseProducer =
    BitNetResidentActivationLeaseProducer.HiddenPreload.position
  private val layerLeaseProducer =
    BitNetResidentActivationLeaseProducer.LayerEvent.position
  private val lmLeaseProducer =
    BitNetResidentActivationLeaseProducer.LmHead.position

  require(ffnExternalSpu == 1,
    "physical FfnGate client 2 must map to external SPU index 1")
  require(cfg.datapath.attention.unifiedScratchpad == unifiedScratchpadCfg,
    "Query, Projection and the production resident tensor must share one " +
      "packed layout and alias-generation domain")
  require(cfg.lm.resident.unifiedScratchpad == unifiedScratchpadCfg,
    "LM capture/replay and Query must share the production resident tensor " +
      "layout and alias-generation domain")

  val io = new Bundle {
    // All components except the internal TP4 fast domain run in this explicit
    // slow domain.  There is no independent implicit shell clock and hence no
    // undeclared slow-clock CDC.
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val preloadCommand = slave(Stream(
      BitNetResidentHiddenPreloadCommand(preloadCfg)))
    val preloadInput = slave(Stream(
      BitNetResidentHiddenPreloadBeat(preloadCfg)))
    val preloadCompletion = master(Stream(
      BitNetResidentHiddenPreloadCompletion(preloadCfg)))

    val start = slave(Stream(
      BitNetResidentPreloadedHiddenFullLayerStart(cfg)))
    val completion = master(Stream(
      BitNetResidentFullRoundCompletion(terminalCfg)))

    val axiAr = Vec(master(Stream(BitNetAxiReadAddress(readCfg))),
      crossbar.bankCount)
    val axiR = Vec(slave(Stream(BitNetAxiReadData(readCfg))),
      crossbar.bankCount)
    val axiAw = Vec(master(Stream(BitNetAxiWriteAddress(writeCfg))),
      crossbar.bankCount)
    val axiW = Vec(master(Stream(BitNetAxiWriteData(writeCfg))),
      crossbar.bankCount)
    val axiB = Vec(slave(Stream(BitNetAxiWriteResponse(writeCfg))),
      crossbar.bankCount)

    val busy = out Bool()
    val hiddenReady = out Bool()
    val hiddenRequestId = out UInt(model.requestIdWidth bits)
    val hiddenTokenMask = out Bits(model.tokenParallelism bits)
    val activationLeaseSource = out(
      BitNetResidentFullLayerActivationLeaseSource())
    val activationOwnerValid = out Bool()
    val activationOwner = out(BitNetActivationClient())
    val linearOwner = out(BitNetResidentLinearPlaneOwner())
    val fatalFailStop = out Bool()
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(64 bits)
  }

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  // The full layer owns exactly one physical resident tensor image.  Its
  // semantic north ports are consumed inside slowArea; its physical backend
  // remains in the explicit 2x domain carried by this composition.
  val residentTensor =
    new BitNetUnifiedResidentTensorProductionShell(unifiedTensorCfg)
  residentTensor.io.slowClk := io.slowClk
  residentTensor.io.slowResetn := io.slowResetn
  residentTensor.io.fastClk := io.fastClk
  residentTensor.io.fastResetn := io.fastResetn

  val slowArea = new ClockingArea(slowCd) {
    // Fixed control chain.
    val scheduler = new BitNetResidentLayerEventScheduler(cfg.event)
    val execution = new BitNetResidentLayerEventExecutionAdapter(
      BitNetResidentLayerEventCommandLoweringConfig(cfg.event))
    val layerLease = new BitNetResidentLayerActivationLeaseCoordinator(
      cfg.event)

    // The only RMS service and its two address/control clients.
    val layerRms = new BitNetResidentRmsNormSubsystem(cfg.layerRms)
    val sharedRms = new BitNetResidentSharedRmsService(matrixCfg.rms)

    // Complete matrix event, QKV/RoPE terminal barrier and physical children.
    val matrixBarrier = new BitNetResidentMatrixRopeEventBarrier(
      cfg.matrixRope)
    val matrixEvent = new BitNetResidentMatrixEventCoordinator(matrixCfg)
    val projectionScales = new BitNetResidentProjectionScaleStager(
      matrixCfg.prepared.scales)
    val gateScales = new BitNetResidentGateScaleComposer(
      matrixCfg.prepared.gateScale)
    val projectionActivation =
      new BitNetResidentProjectionActivationSubsystem(
        activationCfg.copy(scratchpad = unifiedScratchpadCfg),
        useVendorSegmentRam = useVendorPrimitive)
    val packedRouter = new BitNetUnifiedPackedSemanticClientRouter(
      BitNetUnifiedPackedSemanticClientRouterConfig(unifiedScratchpadCfg))
    // Compatibility joins live only at the three legacy compute endpoints.
    // They are pure fixed wiring and retire all four parallel lanes in the
    // same slow cycle; the shared physical/semantic/router transport never
    // recreates a monolithic retained 4096-bit payload.
    val projectionPackedReadJoin =
      new BitNetUnifiedResidentPackedReadParallelToLegacyRowAdapter(
        unifiedScratchpadCfg)
    val queryPackedReadJoin =
      new BitNetUnifiedResidentPackedReadParallelToLegacyRowAdapter(
        unifiedScratchpadCfg)
    val lmPackedReadJoin =
      new BitNetUnifiedResidentPackedReadParallelToLegacyRowAdapter(
        unifiedScratchpadCfg)
    val aliasRouter = new BitNetUnifiedResidentAliasLeaseRouter(
      BitNetUnifiedResidentAliasLeaseRouterConfig(unifiedScratchpadCfg))
    val binder = new BitNetResidentActivationReplayBinder(binderCfg)
    val matrixStage = new BitNetResidentMatrixStageControlSubsystem(
      stageCfg, externalQkvProjectionPreprocess = true)
    val rope = new BitNetResidentRopeCoefficientStager(
      cfg.matrixRope.rope)
    val postGate = new BitNetResidentFfnPostGateRmsSubsystem(
      matrixCfg.downInput.postGate)

    // The sole linear/attention plane and all of its real feeder clients.
    val attention = new BitNetResidentAttentionEventAdapter(cfg.event)
    val linearRouter = new BitNetResidentLinearPlaneOwnerRouter(
      BitNetResidentLinearPlaneOwnerRouterConfig(cfg.datapath))
    val sharedTp4 = new BitNetResidentSharedTp4ComputeDatapath(
      cfg.datapath,
      useVendorPrimitive = useVendorPrimitive,
      trustedLinearLaunchIngress = true)
    val prefetch = new BitNetFiveBankStripedWeightPrefetcher(
      prefetchCfg, externalWeightTileCache = true)
    val weightTileCache = new BitNetResidentSharedWeightTileCache(
      BitNetResidentSharedWeightTileCacheConfig(
        dataWidth = prefetchCfg.crossbar.dataWidth,
        poolCount = prefetchCfg.poolCount,
        memoryCount = prefetchCfg.segmentsPerTile,
        depth = prefetchCfg.memoryDepth,
        sequenceIdWidth = cfg.datapath.continuous.sequenceIdWidth,
        sourceEpochWidth = prefetchCfg.epochWidth))
    val sharedQuant = new BitNetResidentSharedDynamicQuantService(
      matrixCfg.quant)
    val sharedFfn = new BitNetResidentSharedFfnArithmeticClient(
      cfg.datapath.arithmetic)

    // One physical activation store and its complete client plumbing.
    val matrixActivation = new BitNetResidentMatrixActivationPortMux(
      model, layout)
    val activationInterconnect =
      new BitNetResidentActivationClientInterconnect(model, layout)
    val activationFabric =
      new BitNetResidentUnifiedActivationFabricSubsystem(model, layout)
    val activationGlobalKill = Bool()
    // Cross-subsystem failures are accumulated through this register before
    // they may gate activation Streams.  Every child still performs its own
    // same-cycle fail-close; this register prevents a current fault/busy cone
    // from travelling through global kill and the fabric ready network back
    // into an unrelated producer FSM.
    val activationSystemKillSticky = RegInit(False)
    // Packed/alias/backend faults can be caused by Query or LM traffic itself.
    // Feeding same-cycle quarantine back into either client's abort input
    // would make its valid path part of the router's protocol detector.
    // Retain the transport failure at this explicit recovery boundary; every
    // producer still closes its own offending Stream in the detection cycle.
    val semanticTransportAbortSticky = RegInit(False)
    activationFabric.io.kill := activationGlobalKill
    val hiddenPreload = new BitNetResidentHiddenPreloadEndpoint(preloadCfg)
    val bootstrapWindowGuard =
      new BitNetResidentBootstrapWindowGuard
    val bootstrapAdmissionNow = Bool()
    val sourceLock = new BitNetResidentActivationLeaseSourceLock(model)

    residentTensor.io.activationRead <<
      activationFabric.io.unifiedActivationRead
    activationFabric.io.unifiedActivationReadData <<
      residentTensor.io.activationReadData
    residentTensor.io.activationWrite <<
      activationFabric.io.unifiedActivationWrite
    activationFabric.io.unifiedActivationWriteFault <<
      residentTensor.io.activationWriteFault

    // Projection ping/pong, Query and LM share the sole packed semantic
    // frontend.  The router retains owner/identity across deferred commits and
    // routes responses only to the authenticated client.
    val projectionPackedClient =
      BitNetUnifiedPackedSemanticClient.Projection.position
    val queryPackedClient =
      BitNetUnifiedPackedSemanticClient.Query.position
    val lmPackedClient =
      BitNetUnifiedPackedSemanticClient.LmHead.position

    packedRouter.io.clientPackedRead(projectionPackedClient) <<
      projectionActivation.io.packedRead
    projectionActivation.io.packedReadData <<
      projectionPackedReadJoin.io.legacy
    projectionPackedReadJoin.io.parallel.region :=
      packedRouter.io.clientPackedReadData(projectionPackedClient).region
    projectionPackedReadJoin.io.parallel.localRow :=
      packedRouter.io.clientPackedReadData(projectionPackedClient).localRow
    projectionPackedReadJoin.io.parallel.bankMask :=
      packedRouter.io.clientPackedReadData(projectionPackedClient).bankMask
    projectionPackedReadJoin.io.parallel.laneMask :=
      packedRouter.io.clientPackedReadData(projectionPackedClient).laneMask
    projectionPackedReadJoin.io.parallel.fault :=
      packedRouter.io.clientPackedReadData(projectionPackedClient).fault
    projectionPackedReadJoin.io.parallel.faultCode :=
      packedRouter.io.clientPackedReadData(projectionPackedClient).faultCode
    projectionPackedReadJoin.io.parallel.segmentValid :=
      packedRouter.io.clientPackedReadData(projectionPackedClient).segmentValid
    for (lane <- 0 until unifiedScratchpadCfg.layout
        .packedTransportSegmentCount) {
      projectionPackedReadJoin.io.parallel.segments(lane) :=
        packedRouter.io.clientPackedReadData(projectionPackedClient)
          .segments(lane)
    }
    packedRouter.io.clientPackedReadData(projectionPackedClient)
      .segmentReady := projectionPackedReadJoin.io.parallel.segmentReady
    packedRouter.io.clientPackedWrite(projectionPackedClient) <<
      projectionActivation.io.packedWrite
    projectionActivation.io.packedWriteFault <<
      packedRouter.io.clientPackedWriteFault(projectionPackedClient)

    packedRouter.io.clientPackedRead(queryPackedClient) <<
      sharedTp4.io.queryPackedRead
    // Query rows need the same authenticated elastic boundary as LM rows.
    // Without it, a packed header/fault comparison can propagate through the
    // Query replay queues, attention feeder and TP4 lease phase CE in one slow
    // cycle.  The full-row m2s stage keeps fault/data identity atomic, adds one
    // response cycle and still accepts one row on every slow clock.
    val queryPackedReadResponsePipe =
      queryPackedReadJoin.io.legacy.m2sPipe()
    sharedTp4.io.queryPackedReadData << queryPackedReadResponsePipe
    queryPackedReadJoin.io.parallel.region :=
      packedRouter.io.clientPackedReadData(queryPackedClient).region
    queryPackedReadJoin.io.parallel.localRow :=
      packedRouter.io.clientPackedReadData(queryPackedClient).localRow
    queryPackedReadJoin.io.parallel.bankMask :=
      packedRouter.io.clientPackedReadData(queryPackedClient).bankMask
    queryPackedReadJoin.io.parallel.laneMask :=
      packedRouter.io.clientPackedReadData(queryPackedClient).laneMask
    queryPackedReadJoin.io.parallel.fault :=
      packedRouter.io.clientPackedReadData(queryPackedClient).fault
    queryPackedReadJoin.io.parallel.faultCode :=
      packedRouter.io.clientPackedReadData(queryPackedClient).faultCode
    queryPackedReadJoin.io.parallel.segmentValid :=
      packedRouter.io.clientPackedReadData(queryPackedClient).segmentValid
    for (lane <- 0 until unifiedScratchpadCfg.layout
        .packedTransportSegmentCount) {
      queryPackedReadJoin.io.parallel.segments(lane) :=
        packedRouter.io.clientPackedReadData(queryPackedClient)
          .segments(lane)
    }
    packedRouter.io.clientPackedReadData(queryPackedClient).segmentReady :=
      queryPackedReadJoin.io.parallel.segmentReady
    packedRouter.io.clientPackedWrite(queryPackedClient) <<
      sharedTp4.io.queryPackedWrite
    sharedTp4.io.queryPackedWriteFault <<
      packedRouter.io.clientPackedWriteFault(queryPackedClient)

    residentTensor.io.packedRead << packedRouter.io.packedRead
    packedRouter.io.packedReadData.region :=
      residentTensor.io.packedReadData.region
    packedRouter.io.packedReadData.localRow :=
      residentTensor.io.packedReadData.localRow
    packedRouter.io.packedReadData.bankMask :=
      residentTensor.io.packedReadData.bankMask
    packedRouter.io.packedReadData.laneMask :=
      residentTensor.io.packedReadData.laneMask
    packedRouter.io.packedReadData.fault :=
      residentTensor.io.packedReadData.fault
    packedRouter.io.packedReadData.faultCode :=
      residentTensor.io.packedReadData.faultCode
    packedRouter.io.packedReadData.segmentValid :=
      residentTensor.io.packedReadData.segmentValid
    for (lane <- 0 until unifiedScratchpadCfg.layout
        .packedTransportSegmentCount) {
      packedRouter.io.packedReadData.segments(lane) :=
        residentTensor.io.packedReadData.segments(lane)
    }
    residentTensor.io.packedReadData.segmentReady :=
      packedRouter.io.packedReadData.segmentReady
    residentTensor.io.packedWrite << packedRouter.io.packedWrite
    packedRouter.io.packedWriteFault << residentTensor.io.packedWriteFault

    sharedTp4.io.querySemanticPackedReadOutstanding :=
      packedRouter.io.readOwnerValid &&
        packedRouter.io.readOwner ===
          BitNetUnifiedPackedSemanticClient.Query
    sharedTp4.io.querySemanticPackedWriteOutstanding :=
      packedRouter.io.writeOwnerValid &&
        packedRouter.io.writeOwner ===
          BitNetUnifiedPackedSemanticClient.Query
    when(packedRouter.io.quarantined || aliasRouter.io.quarantined ||
        residentTensor.io.quarantined) {
      semanticTransportAbortSticky := True
    }
    sharedTp4.io.queryExternalAbort := semanticTransportAbortSticky ||
      activationSystemKillSticky

    val queryAliasClient =
      BitNetUnifiedResidentTensorAliasOwner.Query.position
    val lmAliasClient =
      BitNetUnifiedResidentTensorAliasOwner.LmHead.position
    aliasRouter.io.clientAliasAcquire(queryAliasClient) <<
      sharedTp4.io.queryAliasAcquire
    sharedTp4.io.queryAliasGrant <<
      aliasRouter.io.clientAliasGrant(queryAliasClient)
    aliasRouter.io.clientAliasRelease(queryAliasClient) <<
      sharedTp4.io.queryAliasRelease
    sharedTp4.io.queryAliasAcquireOwned :=
      aliasRouter.io.clientLocked &&
        aliasRouter.io.lockedClient ===
          BitNetUnifiedResidentTensorAliasOwner.Query
    residentTensor.io.aliasAcquire << aliasRouter.io.backendAliasAcquire
    aliasRouter.io.backendAliasGrant << residentTensor.io.aliasGrant
    residentTensor.io.aliasRelease << aliasRouter.io.backendAliasRelease

    // Final RMS -> LM handoff -> real vocabulary scan/argmax join.
    val lm = new BitNetResidentLmHeadSubsystem(
      cfg.lm, externalWeightTileCache = true)
    packedRouter.io.clientPackedRead(lmPackedClient) << lm.io.packedRead
    // Authentication and parallel-to-legacy row assembly are complete at
    // this boundary.  Retain the complete row before the LM codec so packed
    // header/fault detection cannot feed combinationally through LM replay,
    // the matrix feeder and the shared TP2 phase controller.  This elastic
    // stage adds one response cycle, preserves one row per cycle throughput,
    // and keeps fault/data identity atomic under backpressure.
    val lmPackedReadResponsePipe = lmPackedReadJoin.io.legacy.m2sPipe()
    lm.io.packedReadData << lmPackedReadResponsePipe
    lmPackedReadJoin.io.parallel.region :=
      packedRouter.io.clientPackedReadData(lmPackedClient).region
    lmPackedReadJoin.io.parallel.localRow :=
      packedRouter.io.clientPackedReadData(lmPackedClient).localRow
    lmPackedReadJoin.io.parallel.bankMask :=
      packedRouter.io.clientPackedReadData(lmPackedClient).bankMask
    lmPackedReadJoin.io.parallel.laneMask :=
      packedRouter.io.clientPackedReadData(lmPackedClient).laneMask
    lmPackedReadJoin.io.parallel.fault :=
      packedRouter.io.clientPackedReadData(lmPackedClient).fault
    lmPackedReadJoin.io.parallel.faultCode :=
      packedRouter.io.clientPackedReadData(lmPackedClient).faultCode
    lmPackedReadJoin.io.parallel.segmentValid :=
      packedRouter.io.clientPackedReadData(lmPackedClient).segmentValid
    for (lane <- 0 until unifiedScratchpadCfg.layout
        .packedTransportSegmentCount) {
      lmPackedReadJoin.io.parallel.segments(lane) :=
        packedRouter.io.clientPackedReadData(lmPackedClient).segments(lane)
    }
    packedRouter.io.clientPackedReadData(lmPackedClient).segmentReady :=
      lmPackedReadJoin.io.parallel.segmentReady
    packedRouter.io.clientPackedWrite(lmPackedClient) << lm.io.packedWrite
    lm.io.packedWriteFault <<
      packedRouter.io.clientPackedWriteFault(lmPackedClient)
    lm.io.semanticPackedReadOutstanding :=
      packedRouter.io.readOwnerValid &&
        packedRouter.io.readOwner ===
          BitNetUnifiedPackedSemanticClient.LmHead
    lm.io.semanticPackedWriteOutstanding :=
      packedRouter.io.writeOwnerValid &&
        packedRouter.io.writeOwner ===
          BitNetUnifiedPackedSemanticClient.LmHead
    lm.io.abort := semanticTransportAbortSticky || activationSystemKillSticky
    aliasRouter.io.clientAliasAcquire(lmAliasClient) << lm.io.aliasAcquire
    lm.io.aliasGrant << aliasRouter.io.clientAliasGrant(lmAliasClient)
    aliasRouter.io.clientAliasRelease(lmAliasClient) << lm.io.aliasRelease
    val lmCompletionGate =
      new BitNetResidentLmCompletionAfterReleaseGate(cfg.lm.resident)
    val terminal = new BitNetResidentLmTerminalJoin(terminalCfg)

    // Exactly one AUX owner and one production five-bank memory hub.
    val aux = new BitNetResidentAuxReadOwnerMux(crossbar, auxBank = 0)
    val memory = new BitNetProductionPlMemoryOwnershipHub(cfg.memory)

    sharedTp4.io.slowClk := io.slowClk
    sharedTp4.io.slowResetn := io.slowResetn
    sharedTp4.io.fastClk := io.fastClk
    sharedTp4.io.fastResetn := io.fastResetn

    /* ------------------------------------------------------------------
     * Scheduler -> lowering -> activation lease -> physical executor.
     * ------------------------------------------------------------------ */
    // The board software preloads all five complete images into PS high DDR.
    // Bank 0 is therefore already resident before a round starts and no
    // per-layer PS-to-PL staging barrier is required.
    execution.io.eventCommand << scheduler.io.dataCommand
    scheduler.io.dataCommit << execution.io.dataCommit
    layerLease.io.command << execution.io.execution
    execution.io.executionCompletion << layerLease.io.completion

    layerRms.io.command << layerLease.io.rmsCommand
    layerLease.io.rmsCompletion << layerRms.io.completion
    matrixBarrier.io.command << layerLease.io.matrixCommand
    layerLease.io.matrixCompletion << matrixBarrier.io.completion
    attention.io.eventCommand << layerLease.io.attentionCommand
    layerLease.io.attentionCompletion << attention.io.completion

    /* ------------------------------------------------------------------
     * The unique RMS arithmetic service.
     * ------------------------------------------------------------------ */
    sharedRms.io.layerRms.command << layerRms.io.rmsCommand
    sharedRms.io.layerRms.measure << layerRms.io.rmsMeasure
    layerRms.io.rmsReplayRequest << sharedRms.io.layerRms.replayRequest
    sharedRms.io.layerRms.replay << layerRms.io.rmsReplay
    layerRms.io.rmsResult << sharedRms.io.layerRms.result
    layerRms.io.rmsDone << sharedRms.io.layerRms.done

    sharedRms.io.ffnPostGate.command << postGate.io.rmsCommand
    sharedRms.io.ffnPostGate.measure << postGate.io.rmsMeasure
    postGate.io.rmsReplayRequest << sharedRms.io.ffnPostGate.replayRequest
    sharedRms.io.ffnPostGate.replay << postGate.io.rmsReplay
    postGate.io.rmsResult << sharedRms.io.ffnPostGate.result
    postGate.io.rmsDone << sharedRms.io.ffnPostGate.done

    /* ------------------------------------------------------------------
     * Matrix/RoPE fork, preparation and physical MatrixStage.
     * ------------------------------------------------------------------ */
    matrixEvent.io.command << matrixBarrier.io.matrixCommand
    matrixBarrier.io.matrixCompletion << matrixEvent.io.completion
    rope.io.round << matrixBarrier.io.ropeRound
    matrixBarrier.io.ropeCompletion << rope.io.completion

    projectionScales.io.request << matrixEvent.io.scaleRequest
    matrixEvent.io.scaleSet << projectionScales.io.scales
    gateScales.io.command << matrixEvent.io.gateScaleCommand
    matrixEvent.io.gateScaleResult << gateScales.io.result
    binder.io.binding << matrixEvent.io.imageBinding
    matrixEvent.io.bindingCompletion << binder.io.bindingCompletion
    matrixStage.io.command << matrixEvent.io.matrixCommand
    matrixEvent.io.matrixCompletion << matrixStage.io.completion
    matrixStage.io.gateCommand << matrixEvent.io.gateCommand

    projectionActivation.io.command << matrixEvent.io.activationCommand
    matrixEvent.io.activationCompletion << projectionActivation.io.completion

    matrixEvent.io.gateRmsCommand << matrixStage.io.rmsCommand
    postGate.io.context << matrixEvent.io.postGateContext
    postGate.io.gateRmsCommand << matrixEvent.io.postGateRmsCommand
    matrixEvent.io.postGateCompletion << postGate.io.completion

    // The stager authenticates the actual MatrixStage sequenceId in series.
    rope.io.projection << matrixStage.io.qkvProjectionBeforePreprocess
    matrixStage.io.qkvProjectionAfterPreprocess << rope.io.projectionForward
    matrixStage.io.ropeCoefficient << rope.io.coefficient

    // Image lifetime binder: compute completion is not sent directly to the
    // MatrixStage.  Both real image replay and compute must retire first.
    binder.io.linearLaunch << matrixStage.io.linearLaunch
    matrixStage.io.linearCompletion << binder.io.linearCompletion
    // The binder performs an atomic fork by waiting for both downstream
    // ready signals.  The owner router computes ready from launch.valid, so
    // an M2S skid register is required at this composition boundary to keep
    // the Stream dependency acyclic while retaining the binder-side atomic
    // acceptance of compute and activation-replay commands.  This register
    // carries the 282-bit canonical program; the owner router expands it only
    // at the sole legacy physical-compute launch boundary.
    val layerMatrixLaunchPipe = binder.io.computeLaunch.m2sPipe()
    linearRouter.io.layerMatrixLaunch << layerMatrixLaunchPipe
    // Keep the authenticated tile, including its poison/fault identity,
    // atomic while cutting the global feeder-ready -> binder tile-count cone.
    // A generic S2M skid slice achieves that but adds a 2k-bit bypass mux.
    // This boundary instead uses one non-bypass slot: ready depends only on
    // local fullness and same-cycle pop/refill is deliberately disallowed.
    // The feeder consumes each raw group in lower/upper phases, so one tile
    // every two slow cycles matches its physical service rate.
    val layerMatrixActivationPipe = Stream(
      BitNetNormalizedActivationTile(binderCfg.linear.feeder))
    val layerMatrixActivationPipeValid = RegInit(False)
    val layerMatrixActivationPipePayload = Reg(
      BitNetNormalizedActivationTile(binderCfg.linear.feeder))
    binder.io.activation.ready := !layerMatrixActivationPipeValid
    when(binder.io.activation.fire) {
      layerMatrixActivationPipeValid := True
      layerMatrixActivationPipePayload := binder.io.activation.payload
    }
    layerMatrixActivationPipe.valid := layerMatrixActivationPipeValid
    layerMatrixActivationPipe.payload := layerMatrixActivationPipePayload
    when(layerMatrixActivationPipe.fire) {
      layerMatrixActivationPipeValid := False
    }
    linearRouter.io.layerMatrixActivationTile << layerMatrixActivationPipe
    binder.io.computeCompletion << linearRouter.io.layerMatrixCompletion
    matrixStage.io.linearResult << linearRouter.io.layerMatrixResult

    projectionActivation.io.replayCommand << binder.io.replayCommand
    binder.io.tile << projectionActivation.io.tile
    binder.io.replayCompletion << projectionActivation.io.replayCompletion

    prefetch.io.command << matrixStage.io.weightCommand
    matrixStage.io.prefetchCompletion << prefetch.io.completion
    linearRouter.io.layerMatrixMap0RawGroup << prefetch.io.tile

    // Matrix and LM acquire an authenticated, generation-tagged lease on the
    // same 2 x 8 x 108 tile image.  The owner is latched until an identity-
    // matched release; stale or overlapping access quarantines the cache.
    weightTileCache.io.matrixAcquire << prefetch.io.cacheAcquire
    prefetch.io.cacheGrant << weightTileCache.io.matrixGrant
    prefetch.io.cacheActiveGeneration := weightTileCache.io.activeGeneration
    weightTileCache.io.matrixRelease << prefetch.io.cacheRelease
    weightTileCache.io.lmAcquire << lm.io.cacheAcquire
    lm.io.cacheGrant << weightTileCache.io.lmGrant
    lm.io.cacheActiveGeneration := weightTileCache.io.activeGeneration
    weightTileCache.io.lmRelease << lm.io.cacheRelease
    for (pool <- 0 until prefetchCfg.poolCount;
         memoryIndex <- 0 until prefetchCfg.segmentsPerTile) {
      weightTileCache.io.matrixWrite(pool)(memoryIndex) <<
        prefetch.io.cacheWrite(pool)(memoryIndex)
      prefetch.io.cacheReadData(pool)(memoryIndex) :=
        weightTileCache.io.matrixReadData(pool)(memoryIndex)

      weightTileCache.io.lmWrite(pool)(memoryIndex) <<
        lm.io.cacheWrite(pool)(memoryIndex)
      lm.io.cacheReadData(pool)(memoryIndex) :=
        weightTileCache.io.lmReadData(pool)(memoryIndex)
    }
    weightTileCache.io.matrixRead << prefetch.io.cacheRead
    weightTileCache.io.lmRead << lm.io.cacheRead

    sharedTp4.io.qkvRound << matrixStage.io.qkvRound
    sharedTp4.io.qkvDot << matrixStage.io.qkvDot
    matrixStage.io.qkvCompletion << sharedTp4.io.qkvCompletion

    sharedTp4.io.attentionCommand << attention.io.attentionCommand
    attention.io.attentionCompletion << sharedTp4.io.attentionCompletion

    /* ------------------------------------------------------------------
     * LM and layer clients of the one linear plane.
     * ------------------------------------------------------------------ */
    sharedTp4.io.linearLaunch << linearRouter.io.linearLaunch
    sharedTp4.io.map0RawGroup << linearRouter.io.map0RawGroup
    sharedTp4.io.i8WeightBeat << linearRouter.io.i8WeightBeat
    sharedTp4.io.activationTile << linearRouter.io.activationTile
    linearRouter.io.linearResult << sharedTp4.io.linearResult
    linearRouter.io.linearCompletion << sharedTp4.io.linearCompletion

    linearRouter.io.lmHeadLaunch << lm.io.linearLaunch
    linearRouter.io.lmHeadI8WeightBeat << lm.io.matrixI8Weight
    linearRouter.io.lmHeadActivationTile << lm.io.matrixActivation
    lm.io.linearResult << linearRouter.io.lmHeadResult
    lm.io.linearCompletion << linearRouter.io.lmHeadCompletion

    /* ------------------------------------------------------------------
     * The unique dynamic-i8 service.
     * ------------------------------------------------------------------ */
    sharedQuant.io.projectionImage.command <<
      projectionActivation.io.quantCommand
    sharedQuant.io.projectionImage.measure <<
      projectionActivation.io.quantMeasureInput
    projectionActivation.io.quantReplayRequest <<
      sharedQuant.io.projectionImage.replayRequest
    sharedQuant.io.projectionImage.replay <<
      projectionActivation.io.quantReplayInput
    projectionActivation.io.quantOutput <<
      sharedQuant.io.projectionImage.output
    projectionActivation.io.quantDone <<
      sharedQuant.io.projectionImage.done

    sharedQuant.io.lmHead.command << lm.io.quantCommand
    sharedQuant.io.lmHead.measure << lm.io.quantMeasureInput
    lm.io.quantReplayRequest << sharedQuant.io.lmHead.replayRequest
    sharedQuant.io.lmHead.replay << lm.io.quantReplayInput
    lm.io.quantOutput << sharedQuant.io.lmHead.output
    lm.io.quantDone << sharedQuant.io.lmHead.done

    sharedQuant.io.qkvHead.command << sharedTp4.io.qkvQuantCommand
    sharedQuant.io.qkvHead.measure << sharedTp4.io.qkvQuantMeasure
    sharedTp4.io.qkvQuantReplayRequest <<
      sharedQuant.io.qkvHead.replayRequest
    sharedQuant.io.qkvHead.replay << sharedTp4.io.qkvQuantReplay
    sharedTp4.io.qkvQuantResult << sharedQuant.io.qkvHead.output
    sharedTp4.io.qkvQuantDone << sharedQuant.io.qkvHead.done

    /* ------------------------------------------------------------------
     * Both FFN logical users share physical SPU FfnGate client 2.
     * ------------------------------------------------------------------ */
    sharedFfn.io.gateScaleComposer.acquire << gateScales.io.poolAcquire
    sharedFfn.io.gateScaleComposer.operand << gateScales.io.poolOperand
    gateScales.io.poolResult << sharedFfn.io.gateScaleComposer.result
    sharedFfn.io.gateScaleComposer.release << gateScales.io.poolRelease

    sharedFfn.io.gateUpLeasedGate.acquire << matrixStage.io.poolAcquire
    sharedFfn.io.gateUpLeasedGate.operand << matrixStage.io.poolOperand
    matrixStage.io.poolResult << sharedFfn.io.gateUpLeasedGate.result
    sharedFfn.io.gateUpLeasedGate.release << matrixStage.io.poolRelease

    sharedTp4.io.spuAcquire(ffnExternalSpu) <<
      sharedFfn.io.physicalFfnGate.acquire
    sharedTp4.io.spuOperand(ffnExternalSpu) <<
      sharedFfn.io.physicalFfnGate.operand
    sharedFfn.io.physicalFfnGate.result <<
      sharedTp4.io.spuResult(ffnExternalSpu)
    sharedTp4.io.spuRelease(ffnExternalSpu) <<
      sharedFfn.io.physicalFfnGate.release

    val unusedSpuResultValid = Bits(cfg.datapath.exposedArithmeticClients bits)
    unusedSpuResultValid := 0
    for (external <- 0 until cfg.datapath.exposedArithmeticClients
         if external != ffnExternalSpu) {
      sharedTp4.io.spuAcquire(external).valid := False
      sharedTp4.io.spuAcquire(external).payload :=
        BitNetSpuArithmeticLeaseRequest(cfg.datapath.arithmetic).getZero
      sharedTp4.io.spuOperand(external).valid := False
      sharedTp4.io.spuOperand(external).payload :=
        BitNetSpuArithmeticOperandBeat(cfg.datapath.arithmetic).getZero
      sharedTp4.io.spuResult(external).ready := True
      sharedTp4.io.spuRelease(external).valid := False
      sharedTp4.io.spuRelease(external).payload :=
        BitNetSpuArithmeticLeaseRelease(cfg.datapath.arithmetic).getZero
      unusedSpuResultValid(external) :=
        sharedTp4.io.spuResult(external).valid
    }
    sharedTp4.io.spuClearFault := False

    /* ------------------------------------------------------------------
     * Matrix-event activation sources -> one semantic physical port.
     * ------------------------------------------------------------------ */
    for (lane <- 0 until model.tokenParallelism) {
      matrixActivation.io.projectionReadCommand(lane) <<
        projectionActivation.io.activationReadCommand(lane)
      projectionActivation.io.activationReadData(lane) <<
        matrixActivation.io.projectionReadData(lane)

      // The activation fabric's authenticated write-ready cone spans source
      // locking, arbitration, semantic ownership and the physical adapter.
      // Terminate that backward path at a skid register before it reaches the
      // Gate FSM state CE.  s2mPipe preserves payload/valid atomically under
      // stalls and retains one beat-per-cycle steady-state throughput.
      val gateWriteReadyPipe = matrixStage.io.wideWrite(lane).s2mPipe()
      matrixActivation.io.gateWriteData(lane) << gateWriteReadyPipe

      matrixActivation.io.postGateReadCommand(lane) <<
        postGate.io.activationReadCommand(lane)
      postGate.io.activationReadData(lane) <<
        matrixActivation.io.postGateReadData(lane)
      matrixActivation.io.postGateWriteData(lane) <<
        postGate.io.activationWriteData(lane)

      matrixActivation.io.residualReadCommand(lane) <<
        sharedTp4.io.residualReadCommand(lane)
      sharedTp4.io.residualReadData(lane) <<
        matrixActivation.io.residualReadData(lane)
      matrixActivation.io.residualWriteData(lane) <<
        sharedTp4.io.residualWriteData(lane)

      // Central read-command skid stages terminate the authenticated fabric's
      // ready cone before it fans back into Projection/PostGate/Residual, RMS
      // and LM producer FSMs.  They retain the complete command under stalls
      // and preserve one-command-per-cycle steady-state admission.
      val matrixReadReadyPipe =
        matrixActivation.io.fabricReadCommand(lane).s2mPipe()
      activationInterconnect.io.matrixReadCommand(lane) <<
        matrixReadReadyPipe
      matrixActivation.io.fabricReadData(lane) <<
        activationInterconnect.io.matrixReadData(lane)
      val matrixWriteReadyPipe =
        matrixActivation.io.fabricWriteData(lane).s2mPipe()
      activationInterconnect.io.matrixWriteData(lane) <<
        matrixWriteReadyPipe

      val rmsReadReadyPipe =
        layerRms.io.activationReadCommand(lane).s2mPipe()
      activationInterconnect.io.rmsReadCommand(lane) << rmsReadReadyPipe
      layerRms.io.activationReadData(lane) <<
        activationInterconnect.io.rmsReadData(lane)
      val rmsWriteReadyPipe =
        layerRms.io.activationWriteData(lane).s2mPipe()
      activationInterconnect.io.rmsWriteData(lane) << rmsWriteReadyPipe

      val attentionWriteReadyPipe =
        sharedTp4.io.contextWrite(lane).s2mPipe()
      activationInterconnect.io.attentionWriteData(lane) <<
        attentionWriteReadyPipe

      val lmReadReadyPipe = lm.io.activationReadCommand(lane).s2mPipe()
      activationInterconnect.io.lmReadCommand(lane) << lmReadReadyPipe
      lm.io.activationReadData(lane) <<
        activationInterconnect.io.lmReadData(lane)
    }

    activationInterconnect.io.ownerValid := activationFabric.io.ownerValid
    activationInterconnect.io.owner := activationFabric.io.owner

    /* ------------------------------------------------------------------
     * Three-source activation lease lock.  Producer identity remains
     * orthogonal to semantic client identity, so preload and layer Matrix
     * leases cannot be confused even when requestId is identical.
     * ------------------------------------------------------------------ */
    sourceLock.io.producerAcquire(preloadLeaseProducer) <<
      hiddenPreload.io.activationAcquire
    sourceLock.io.producerAcquire(layerLeaseProducer) <<
      layerLease.io.activationAcquire
    sourceLock.io.producerAcquire(lmLeaseProducer) <<
      lm.io.activationAcquire
    sourceLock.io.producerRelease(preloadLeaseProducer) <<
      hiddenPreload.io.activationRelease
    sourceLock.io.producerRelease(layerLeaseProducer) <<
      layerLease.io.activationRelease
    sourceLock.io.producerRelease(lmLeaseProducer) <<
      lm.io.activationRelease

    activationFabric.io.acquire << sourceLock.io.fabricAcquire
    activationFabric.io.release << sourceLock.io.fabricRelease
    sourceLock.io.fabricOwnerValid := activationFabric.io.ownerValid
    sourceLock.io.fabricOwner := activationFabric.io.owner
    sourceLock.io.fabricOwnerRequestId :=
      activationFabric.io.ownerRequestId
    sourceLock.io.fabricProtocolFaultSticky :=
      activationFabric.io.protocolFaultSticky

    hiddenPreload.io.activationOwnerValid :=
      sourceLock.io.producerOwnerValid(preloadLeaseProducer)
    hiddenPreload.io.activationOwner :=
      sourceLock.io.producerOwnerLease(preloadLeaseProducer).client
    hiddenPreload.io.activationOwnerRequestId :=
      sourceLock.io.producerOwnerLease(preloadLeaseProducer).requestId
    layerLease.io.activationOwnerValid :=
      sourceLock.io.producerOwnerValid(layerLeaseProducer)
    layerLease.io.activationOwner :=
      sourceLock.io.producerOwnerLease(layerLeaseProducer).client
    layerLease.io.activationOwnerRequestId :=
      sourceLock.io.producerOwnerLease(layerLeaseProducer).requestId

    val sourceIdle = !sourceLock.io.lockedSourceValid
    val preloadPortSelected =
      sourceLock.io.producerOwnerValid(preloadLeaseProducer)
    val layerPortSelected =
      sourceLock.io.producerOwnerValid(layerLeaseProducer)
    val leaseSource = BitNetResidentFullLayerActivationLeaseSource()
    leaseSource := BitNetResidentFullLayerActivationLeaseSource.Idle
    when(sourceLock.io.lockedSourceValid) {
      switch(sourceLock.io.lockedSource) {
        is(BitNetResidentActivationLeaseProducer.HiddenPreload) {
          leaseSource :=
            BitNetResidentFullLayerActivationLeaseSource.HiddenPreload
        }
        is(BitNetResidentActivationLeaseProducer.LayerEvent) {
          leaseSource :=
            BitNetResidentFullLayerActivationLeaseSource.LayerEvent
        }
        is(BitNetResidentActivationLeaseProducer.LmHead) {
          leaseSource := BitNetResidentFullLayerActivationLeaseSource.LmHead
        }
      }
    }

    val integrationQuarantine = RegInit(False)
    val integrationFaultSticky = RegInit(False)

    /* ------------------------------------------------------------------
     * Activation client wiring, with a source-locked preload/Matrix mux.
     * ------------------------------------------------------------------ */
    val preloadWriteValid = Bits(model.tokenParallelism bits)
    val matrixWriteValid = Bits(model.tokenParallelism bits)
    val matrixReadValid = Bits(model.tokenParallelism bits)
    for (lane <- 0 until model.tokenParallelism) {
      preloadWriteValid(lane) :=
        hiddenPreload.io.activationWriteData(lane).valid
      matrixWriteValid(lane) :=
        activationInterconnect.io.clientWriteData(matrixClient)(lane).valid
      matrixReadValid(lane) :=
        activationInterconnect.io.clientReadCommand(matrixClient)(lane).valid
    }
    val preloadMatrixCollision =
      (preloadWriteValid & matrixWriteValid).orR
    val activationPortViolation = preloadMatrixCollision ||
      (preloadPortSelected &&
        (matrixWriteValid.orR || matrixReadValid.orR)) ||
      (!preloadPortSelected && preloadWriteValid.orR) ||
      (!preloadPortSelected && !layerPortSelected &&
        (matrixWriteValid.orR || matrixReadValid.orR))
    val matrixPathEnabled = !integrationQuarantine &&
      !sourceLock.io.quarantined &&
      !activationPortViolation

    for (client <- 0 until clientCount; lane <- 0 until
        model.tokenParallelism) {
      if (client == matrixClient) {
        activationFabric.io.clientReadCommand(client)(lane).valid :=
          activationInterconnect.io.clientReadCommand(client)(lane).valid &&
            layerPortSelected && matrixPathEnabled
        activationFabric.io.clientReadCommand(client)(lane).payload :=
          activationInterconnect.io.clientReadCommand(client)(lane).payload
        activationInterconnect.io.clientReadCommand(client)(lane).ready :=
          activationFabric.io.clientReadCommand(client)(lane).ready &&
            layerPortSelected && matrixPathEnabled

        activationInterconnect.io.clientReadData(client)(lane) <<
          activationFabric.io.clientReadData(client)(lane)

        activationFabric.io.clientWriteData(client)(lane).valid := False
        activationFabric.io.clientWriteData(client)(lane).payload :=
          activationInterconnect.io.clientWriteData(client)(lane).payload
        activationInterconnect.io.clientWriteData(client)(lane).ready :=
          False
        hiddenPreload.io.activationWriteData(lane).ready := False
        when(preloadPortSelected && matrixPathEnabled) {
          activationFabric.io.clientWriteData(client)(lane).valid :=
            hiddenPreload.io.activationWriteData(lane).valid
          activationFabric.io.clientWriteData(client)(lane).payload :=
            hiddenPreload.io.activationWriteData(lane).payload
          hiddenPreload.io.activationWriteData(lane).ready :=
            activationFabric.io.clientWriteData(client)(lane).ready
        } elsewhen(layerPortSelected && matrixPathEnabled) {
          activationFabric.io.clientWriteData(client)(lane).valid :=
            activationInterconnect.io.clientWriteData(client)(lane).valid
          activationFabric.io.clientWriteData(client)(lane).payload :=
            activationInterconnect.io.clientWriteData(client)(lane).payload
          activationInterconnect.io.clientWriteData(client)(lane).ready :=
            activationFabric.io.clientWriteData(client)(lane).ready
        }
      } else {
        activationFabric.io.clientReadCommand(client)(lane) <<
          activationInterconnect.io.clientReadCommand(client)(lane)
        activationInterconnect.io.clientReadData(client)(lane) <<
          activationFabric.io.clientReadData(client)(lane)
        activationFabric.io.clientWriteData(client)(lane) <<
          activationInterconnect.io.clientWriteData(client)(lane)
      }
    }

    when(activationPortViolation || unusedSpuResultValid.orR) {
      integrationFaultSticky := True
      integrationQuarantine := True
    }

    /* ------------------------------------------------------------------
     * LM terminal join.  The release gate first buffers the real LM result,
     * then waits for source-locked release retirement and physical owner
     * clearing before it exposes that result to the terminal join.
     * ------------------------------------------------------------------ */
    terminal.io.schedulerHandoff << scheduler.io.lmHeadHandoff
    terminal.io.schedulerCompletion << scheduler.io.completion
    lm.io.handoff << terminal.io.lmHandoff

    lmCompletionGate.io.lmCompletion << lm.io.completion
    lmCompletionGate.io.releaseRetired << sourceLock.io.releaseRetired
    lmCompletionGate.io.fabricOwnerValid := activationFabric.io.ownerValid
    lmCompletionGate.io.fabricOwner := activationFabric.io.owner
    lmCompletionGate.io.fabricOwnerRequestId :=
      activationFabric.io.ownerRequestId
    lmCompletionGate.io.fabricProtocolFaultSticky :=
      activationFabric.io.protocolFaultSticky
    terminal.io.lmCompletion << lmCompletionGate.io.completion

    io.completion << terminal.io.completion

    /* ------------------------------------------------------------------
     * AUX whole-burst owner and the one five-bank memory ownership hub.
     * ------------------------------------------------------------------ */
    aux.io.layerRmsGammaRequest << layerRms.io.auxReadRequest
    layerRms.io.auxReadResponse << aux.io.layerRmsGammaResponse
    aux.io.projectionScaleRequest << projectionScales.io.readRequest
    projectionScales.io.readResponse << aux.io.projectionScaleResponse
    aux.io.postGateGammaRequest << postGate.io.auxReadRequest
    postGate.io.auxReadResponse << aux.io.postGateGammaResponse
    aux.io.ropeRequest << rope.io.readRequest
    rope.io.readResponse << aux.io.ropeResponse
    memory.io.rmsAuxReadRequest << aux.io.downstreamRequest
    aux.io.downstreamResponse << memory.io.rmsAuxReadResponse

    for (bank <- 0 until crossbar.bankCount) {
      memory.io.matrixReadRequest(bank) << prefetch.io.readRequest(bank)
      prefetch.io.readResponse(bank) << memory.io.matrixReadResponse(bank)

      memory.io.lmReadRequest(bank) << lm.io.lmReadRequest(bank)
      lm.io.lmReadResponse(bank) << memory.io.lmReadResponse(bank)

      io.axiAr(bank) << memory.io.axiAr(bank)
      memory.io.axiR(bank) << io.axiR(bank)
      io.axiAw(bank) << memory.io.axiAw(bank)
      io.axiW(bank) << memory.io.axiW(bank)
      memory.io.axiB(bank) << io.axiB(bank)
    }

    memory.io.kvReadRequest << sharedTp4.io.kvReadRequest
    sharedTp4.io.kvReadResponse << memory.io.kvReadResponse
    memory.io.kvWriteRequest << sharedTp4.io.kvWriteRequest
    memory.io.kvWriteData << sharedTp4.io.kvWriteData
    sharedTp4.io.kvWriteResponse << memory.io.kvWriteResponse

    // Keep the stable client-5 ABI on the memory hub, but make it unreachable
    // in the fully preloaded board image.  Constant propagation removes the
    // guard/arbitration branch and its former copy-engine payload path.
    memory.io.reservedReadRequest.valid := False
    memory.io.reservedReadRequest.payload.assignDontCare()
    memory.io.reservedReadResponse.ready := True
    memory.io.reservedWriteRequest.valid := False
    memory.io.reservedWriteRequest.payload.assignDontCare()
    memory.io.reservedWriteData.valid := False
    memory.io.reservedWriteData.payload.assignDontCare()
    memory.io.reservedWriteResponse.ready := True

    /* ------------------------------------------------------------------
     * Idle-only hidden bootstrap and exactly-one-round start gate.
     * ------------------------------------------------------------------ */
    hiddenPreload.io.command.valid := io.preloadCommand.valid &&
      bootstrapWindowGuard.io.commandPermit
    hiddenPreload.io.command.payload := io.preloadCommand.payload
    io.preloadCommand.ready := hiddenPreload.io.command.ready &&
      bootstrapWindowGuard.io.commandPermit
    hiddenPreload.io.input << io.preloadInput
    io.preloadCompletion << hiddenPreload.io.completion

    val memoryIdleChecks = Vec(Bool(), crossbar.bankCount)
    for (bank <- 0 until crossbar.bankCount) {
      memoryIdleChecks(bank) := memory.io.readOutstanding(bank) === 0 &&
        memory.io.writeOutstanding(bank) === 0 &&
        !memory.io.axiAr(bank).valid && !memory.io.axiAw(bank).valid &&
        !memory.io.axiW(bank).valid
    }
    val memoryIdle = memoryIdleChecks.asBits.andR
    val memoryResponseFaultSticky =
      memory.io.memoryReadFaultSticky.orR ||
        memory.io.memoryWriteFaultSticky.orR
    // The memory complex historically folds RRESP/BRESP into its broad
    // protocol signal.  Keep a response-only failure in the workload/data
    // category; routing or control metadata corruption remains protocol-fatal.
    val memoryControlProtocolFaultSticky =
      memory.io.routingFaultSticky ||
        (memory.io.protocolFaultSticky && !memoryResponseFaultSticky)
    val residentTensorIdle = activationFabric.io.idle &&
      !packedRouter.io.busy &&
      !aliasRouter.io.busy &&
      !residentTensor.io.activationReadOutstanding &&
      !residentTensor.io.activationWriteOutstanding &&
      !residentTensor.io.packedReadOutstanding &&
      !residentTensor.io.packedWriteOutstanding &&
      !residentTensor.io.semanticReadResponsePending &&
      !residentTensor.io.semanticWriteFaultPending &&
      !residentTensor.io.physicalCommandPending.orR &&
      !residentTensor.io.physicalResponsePending.orR &&
      !residentTensor.io.aliasOwnerValid
    val inferenceResourcesIdle = !scheduler.io.busy &&
      !execution.io.busy && !layerLease.io.busy &&
      !layerRms.io.busy && !sharedRms.io.busy &&
      !matrixBarrier.io.busy && !matrixEvent.io.busy &&
      !projectionScales.io.busy && !gateScales.io.busy &&
      !projectionActivation.io.busy && !binder.io.busy &&
      !matrixStage.io.busy && !rope.io.busy && !postGate.io.busy &&
      !attention.io.busy && !linearRouter.io.busy &&
      !sharedTp4.io.busy && !prefetch.io.sequenceActive &&
      !weightTileCache.io.ownerValid &&
      !weightTileCache.io.readPipelineValid &&
      !sharedQuant.io.busy && !sharedFfn.io.ownerValid &&
      !lm.io.busy && !lmCompletionGate.io.busy &&
      terminal.io.allowNextRound && !aux.io.busy && memoryIdle &&
      residentTensorIdle

    val subsystemFaultNow = scheduler.io.protocolFaultSticky ||
      execution.io.protocolFaultSticky || layerLease.io.protocolFaultSticky ||
      layerRms.io.protocolFaultSticky || sharedRms.io.protocolFaultSticky ||
      matrixBarrier.io.protocolFaultSticky ||
      matrixEvent.io.protocolFaultSticky ||
      projectionScales.io.protocolFaultSticky ||
      gateScales.io.protocolFaultSticky ||
      projectionActivation.io.protocolFaultSticky ||
      binder.io.protocolFaultSticky || matrixStage.io.protocolFaultSticky ||
      rope.io.protocolFaultSticky || postGate.io.protocolFaultSticky ||
      attention.io.protocolFaultSticky ||
      linearRouter.io.protocolFaultSticky ||
      sharedTp4.io.protocolFaultSticky || prefetch.io.protocolFaultSticky ||
      weightTileCache.io.protocolFaultSticky ||
      sharedQuant.io.protocolFaultSticky ||
      sharedFfn.io.protocolFaultSticky ||
      matrixActivation.io.protocolFaultSticky ||
      activationInterconnect.io.protocolFaultSticky ||
      activationFabric.io.protocolFaultSticky ||
      activationFabric.io.quarantined ||
      packedRouter.io.protocolFaultSticky ||
      aliasRouter.io.protocolFaultSticky ||
      residentTensor.io.protocolFaultSticky ||
      sourceLock.io.protocolFaultSticky ||
      lm.io.protocolFaultSticky ||
      lmCompletionGate.io.protocolFaultSticky ||
      terminal.io.protocolFaultSticky || aux.io.protocolFaultSticky ||
      memoryControlProtocolFaultSticky ||
      hiddenPreload.io.protocolFaultSticky || integrationFaultSticky

    val activationSystemFaultNow =
      matrixBarrier.io.quarantined || sharedTp4.io.fatalFailStop ||
      hiddenPreload.io.quarantined || sharedRms.io.quarantined ||
      sharedQuant.io.quarantined || sharedFfn.io.quarantined ||
      matrixActivation.io.quarantined ||
      activationInterconnect.io.quarantined ||
      linearRouter.io.quarantined || aux.io.quarantined ||
      sourceLock.io.quarantined || lmCompletionGate.io.quarantined ||
      prefetch.io.cacheLeaseQuarantined ||
      lm.io.weightLeaseQuarantined || lm.io.aliasLeaseQuarantined ||
      aliasRouter.io.quarantined || weightTileCache.io.quarantined ||
      packedRouter.io.quarantined ||
      residentTensor.io.quarantined ||
      residentTensor.io.protocolFaultSticky || unusedSpuResultValid.orR

    when(activationSystemFaultNow) {
      activationSystemKillSticky := True
    }

    // A source collision remains a local, same-cycle kill.  All wider system
    // faults are already fail-closed at their source and reach this fabric via
    // the reset-only registered accumulator above.
    activationGlobalKill := activationPortViolation ||
      integrationQuarantine || activationSystemKillSticky

    bootstrapAdmissionNow := inferenceResourcesIdle &&
      !subsystemFaultNow &&
      ((sourceIdle && !activationFabric.io.ownerValid) ||
        preloadPortSelected)
    bootstrapWindowGuard.io.admissionSafe := bootstrapAdmissionNow
    bootstrapWindowGuard.io.commandPresented := io.preloadCommand.valid
    bootstrapWindowGuard.io.commandAccepted := hiddenPreload.io.command.fire
    bootstrapWindowGuard.io.operationBusy := hiddenPreload.io.busy
    bootstrapWindowGuard.io.operationRetired :=
      hiddenPreload.io.completion.fire
    bootstrapWindowGuard.io.registeredGlobalKill :=
      activationSystemKillSticky || integrationQuarantine ||
        hiddenPreload.io.quarantined
    hiddenPreload.io.bootstrapWindowOpen :=
      bootstrapWindowGuard.io.windowOpen

    val hiddenReady = RegInit(False)
    val hiddenRequestId = Reg(UInt(model.requestIdWidth bits)) init 0
    val hiddenTokenMask = Reg(Bits(model.tokenParallelism bits)) init 0
    val preloadPrefixMask = (1 to model.tokenParallelism).map { count =>
      hiddenPreload.io.completion.tokenMask === B(
        (BigInt(1) << count) - 1, model.tokenParallelism bits)
    }.reduce(_ || _)
    val preloadCompletionAuthentic =
      hiddenPreload.io.completion.success && preloadPrefixMask &&
        hiddenPreload.io.completion.featureCount === model.hiddenSize &&
        hiddenPreload.io.completion.groupsCommitted === preloadCfg.groupCount

    when(hiddenPreload.io.command.fire) { hiddenReady := False }
    when(hiddenPreload.io.completion.fire) {
      hiddenReady := preloadCompletionAuthentic
      hiddenRequestId := hiddenPreload.io.completion.requestId
      hiddenTokenMask := hiddenPreload.io.completion.tokenMask
      when(!preloadCompletionAuthentic) {
        integrationFaultSticky := True
        integrationQuarantine := True
      }
    }

    val startPrefixMask = (1 to model.tokenParallelism).map { count =>
      io.start.payload.run.activeTokenMask === B(
        (BigInt(1) << count) - 1, model.tokenParallelism bits)
    }.reduce(_ || _)
    val startHiddenIdentity = hiddenReady && startPrefixMask &&
      io.start.payload.run.requestId === hiddenRequestId &&
      io.start.payload.run.activeTokenMask === hiddenTokenMask
    val startSafe = inferenceResourcesIdle && sourceIdle &&
      !activationFabric.io.ownerValid && !hiddenPreload.io.busy &&
      !io.preloadCommand.valid && !subsystemFaultNow && startHiddenIdentity
    scheduler.io.start.valid := io.start.valid && startSafe
    scheduler.io.start.payload := io.start.payload.run
    io.start.ready := startSafe && scheduler.io.start.ready
    when(io.start.fire) {
      hiddenReady := False
    }

    /* ------------------------------------------------------------------
     * Stable diagnostics.  Unused SPU returns are fatal because no later
     * transaction may consume a stale arithmetic result.
     * ------------------------------------------------------------------ */
    val faults = Bits(64 bits)
    faults := 0
    faults(0) := scheduler.io.protocolFaultSticky
    faults(1) := execution.io.protocolFaultSticky
    faults(2) := layerLease.io.protocolFaultSticky
    faults(3) := layerRms.io.protocolFaultSticky
    faults(4) := sharedRms.io.protocolFaultSticky
    faults(5) := matrixBarrier.io.protocolFaultSticky
    faults(6) := rope.io.protocolFaultSticky
    faults(7) := matrixEvent.io.protocolFaultSticky
    faults(8) := projectionScales.io.protocolFaultSticky
    faults(9) := gateScales.io.protocolFaultSticky
    faults(10) := projectionActivation.io.protocolFaultSticky
    faults(11) := binder.io.protocolFaultSticky
    faults(12) := matrixStage.io.protocolFaultSticky
    faults(13) := postGate.io.protocolFaultSticky
    faults(14) := attention.io.protocolFaultSticky
    faults(15) := linearRouter.io.protocolFaultSticky
    faults(16) := sharedTp4.io.protocolFaultSticky
    faults(17) := prefetch.io.protocolFaultSticky ||
      weightTileCache.io.protocolFaultSticky
    faults(18) := sharedQuant.io.protocolFaultSticky
    faults(19) := sharedFfn.io.protocolFaultSticky
    faults(20) := matrixActivation.io.protocolFaultSticky
    faults(21) := activationInterconnect.io.protocolFaultSticky
    faults(22) := activationFabric.io.protocolFaultSticky ||
      activationFabric.io.quarantined ||
      residentTensor.io.protocolFaultSticky
    faults(23) := activationFabric.io.readFaultSticky.orR ||
      residentTensor.io.rangeFaultSticky
    faults(24) := activationFabric.io.writeFaultSticky.orR ||
      residentTensor.io.formatFaultSticky
    faults(25) := activationFabric.io.collisionFaultSticky.orR ||
      residentTensor.io.collisionFaultSticky ||
      packedRouter.io.protocolFaultSticky ||
      aliasRouter.io.protocolFaultSticky
    faults(26) := lm.io.protocolFaultSticky
    faults(27) := terminal.io.protocolFaultSticky
    faults(28) := aux.io.protocolFaultSticky
    faults(29) := memory.io.protocolFaultSticky
    faults(30) := hiddenPreload.io.protocolFaultSticky
    faults(31) := hiddenPreload.io.quarantined
    faults(32) := integrationFaultSticky
    faults(33) := integrationQuarantine
    faults(34) := matrixBarrier.io.quarantined
    faults(35) := sharedTp4.io.fatalFailStop
    faults(36) := sharedRms.io.quarantined
    faults(37) := sharedQuant.io.quarantined
    faults(38) := sharedFfn.io.quarantined
    faults(43 downto 39) := memory.io.memoryReadFaultSticky
    faults(48 downto 44) := memory.io.memoryWriteFaultSticky
    faults(49) := memory.io.routingFaultSticky
    faults(50) := unusedSpuResultValid.orR
    faults(51) := matrixActivation.io.quarantined
    faults(52) := activationInterconnect.io.quarantined
    faults(53) := sharedTp4.io.spuPoolFault
    faults(54) := linearRouter.io.quarantined
    faults(55) := aux.io.quarantined
    faults(56) := sourceLock.io.protocolFaultSticky
    faults(57) := sourceLock.io.quarantined
    faults(58) := lmCompletionGate.io.protocolFaultSticky
    faults(59) := lmCompletionGate.io.quarantined
    // Former WeightStage diagnostics remain reserved in the stable ABI.
    faults(63 downto 60) := 0

    io.busy := !inferenceResourcesIdle || hiddenPreload.io.busy ||
      !sourceIdle || activationFabric.io.ownerValid || terminal.io.busy
    io.hiddenReady := hiddenReady
    io.hiddenRequestId := hiddenRequestId
    io.hiddenTokenMask := hiddenTokenMask
    io.activationLeaseSource := leaseSource
    io.activationOwnerValid := activationFabric.io.ownerValid
    io.activationOwner := activationFabric.io.owner
    io.linearOwner := linearRouter.io.activeOwner
    io.fatalFailStop := integrationQuarantine ||
      matrixBarrier.io.quarantined || sharedTp4.io.fatalFailStop ||
      hiddenPreload.io.quarantined || sharedRms.io.quarantined ||
      sharedQuant.io.quarantined || sharedFfn.io.quarantined ||
      matrixActivation.io.quarantined ||
      activationInterconnect.io.quarantined ||
      linearRouter.io.quarantined || aux.io.quarantined ||
      sourceLock.io.quarantined || lmCompletionGate.io.quarantined ||
      prefetch.io.cacheLeaseQuarantined ||
      lm.io.weightLeaseQuarantined || lm.io.aliasLeaseQuarantined ||
      aliasRouter.io.quarantined ||
      weightTileCache.io.quarantined ||
      activationFabric.io.protocolFaultSticky ||
      activationFabric.io.quarantined ||
      packedRouter.io.quarantined ||
      residentTensor.io.quarantined ||
      residentTensor.io.protocolFaultSticky || unusedSpuResultValid.orR
    io.protocolFaultSticky := subsystemFaultNow
    io.faultBits := faults
  }
}

object GenerateBitNetResidentPreloadedHiddenFullLayerShell extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-resident-preloaded-hidden-full-layer-shell")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentPreloadedHiddenFullLayerShell())
}
