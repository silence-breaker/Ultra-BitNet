package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentMatrixEventFault {
  val None = 0x00
  val InvalidCommand = 0xc8
  val MissingDownImage = 0xc9
  val PreparedIdentity = 0xca
  val PreparedFailed = 0xcb
  val DownInputIdentity = 0xcc
  val DownInputFailed = 0xcd
  val ActivationContention = 0xce
  val UnexpectedActivationCompletion = 0xcf
}

/**
  * Static contract of the matrix-event transaction coordinator.
  *
  * The event epoch and slot geometry deliberately equal the activation-image
  * geometry.  No lossy epoch truncation is allowed at this production seam.
  */
case class BitNetResidentMatrixEventCoordinatorConfig(
    event: BitNetResidentLayerEventSchedulerConfig =
      BitNetResidentLayerEventSchedulerConfig(epochWidth = 8),
    prepared: BitNetResidentPreparedMatrixStageCoordinatorConfig =
      BitNetResidentPreparedMatrixStageCoordinatorConfig(),
    downInput: BitNetResidentFfnDownInputCoordinatorConfig =
      BitNetResidentFfnDownInputCoordinatorConfig()
) {
  val model: BitNetConfig = event.model
  val activation: BitNetResidentProjectionActivationSubsystemConfig =
    prepared.activation
  val quant: BitNetSharedDynamicI8QuantizerConfig = activation.quant
  val tile: BitNetResidentActivationTileBufferConfig = activation.tile
  val rms: BitNetSharedSpuRmsNormConfig = downInput.rms

  require(model == BitNetConfig.Production)
  require(prepared.model == model && downInput.model == model)
  require(downInput.activation == activation,
    "GateUp and projection paths must share one activation-image builder")
  require(event.epochWidth == quant.epochWidth)
  require(event.slotCount == tile.slotCount)
  require(event.slotCount == 2,
    "GateUp produces the Down image in the following ping/pong slot")
  require(prepared.gate.rmsConfig == downInput.rms)
}

/**
  * Event-level owner for all resident matrix stages.
  *
  * QKV, O and Down retire when the prepared matrix transaction retires.
  * GateUp is different: its public event also owns FfnSubNorm and dynamic-i8
  * image creation.  The exact gated WideWork publication therefore flows
  * directly into the shared post-gate RMS client and the sole activation
  * builder.  Only after both the matrix transaction and the Down-ready image
  * authenticate does the GateUp event commit.
  *
  * The resulting Down image remains in the next PL ping/pong slot.  Down must
  * consume that exact image on the immediately following event; neither an
  * activation tensor nor a software descriptor crosses to PS.
  *
  * This component owns control state only.  It instantiates no arithmetic,
  * activation RAM, weight RAM, matrix cluster, RMS core or quantizer.
  */
class BitNetResidentMatrixEventCoordinator(
    cfg: BitNetResidentMatrixEventCoordinatorConfig =
      BitNetResidentMatrixEventCoordinatorConfig()) extends Component {
  private val model = cfg.model
  private val q = cfg.quant
  private val tile = cfg.tile
  private val gate = cfg.prepared.gate

  val io = new Bundle {
    val command = slave(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg.event)))

    // Prepared projection control boundaries.
    val scaleRequest = master(Stream(
      BitNetResidentProjectionScaleRequest(cfg.prepared.scales)))
    val scaleSet = slave(Stream(
      BitNetResidentProjectionScaleSet(cfg.prepared.scales)))
    val gateScaleCommand = master(Stream(
      BitNetResidentGateScaleComposeCommand(cfg.prepared.gateScale)))
    val gateScaleResult = slave(Stream(
      BitNetResidentGateScaleComposeResult(cfg.prepared.gateScale)))
    val imageBinding = master(Stream(
      BitNetResidentActivationImageBinding(cfg.prepared.binder)))
    val bindingCompletion = slave(Stream(
      BitNetResidentActivationBindingCompletion(cfg.prepared.binder)))
    val matrixCommand = master(Stream(
      BitNetResidentMatrixStageCommand(cfg.prepared.stage)))
    val gateCommand = master(Stream(BitNetSharedFfnGateCommand(gate)))
    val matrixCompletion = slave(Stream(
      BitNetResidentLayerStepCompletion(model)))

    // The unique activation builder is transaction-arbitrated between the
    // incoming projection image and the post-gate Down image.
    val activationCommand = master(Stream(
      BitNetResidentProjectionActivationCommand(cfg.activation)))
    val activationCompletion = slave(Stream(
      BitNetResidentProjectionActivationCompletion(cfg.activation)))

    // GateUp's matrix child emits this command after exact gating.  It is
    // atomically paired with resident event context inside this component.
    val gateRmsCommand = slave(Stream(BitNetSharedRmsCommand(cfg.rms)))
    val postGateContext = master(Stream(
      BitNetResidentFfnPostGateRmsContext(model)))
    val postGateRmsCommand = master(Stream(
      BitNetSharedRmsCommand(cfg.rms)))
    val postGateCompletion = slave(Stream(
      BitNetResidentFfnPostGateRmsCompletion(model)))

    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))
    val busy = out Bool()
    val downImageResident = out Bool()
    val protocolFaultSticky = out Bool()
    val faultCode = out Bits(8 bits)
    val completedEvents = out UInt(32 bits)
  }

  val prepared = new BitNetResidentPreparedMatrixStageCoordinator(
    cfg.prepared)
  val down = new BitNetResidentFfnDownInputCoordinator(cfg.downInput)

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssuePrepared, Active, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentLayerEventExecutorCommand(cfg.event)) init
    BitNetResidentLayerEventExecutorCommand(cfg.event).getZero

  val response = Reg(BitNetResidentLayerStepCompletion(model)) init
    BitNetResidentLayerStepCompletion(model).getZero
  val responseValid = RegInit(False)
  val preparedDone = RegInit(False)
  val preparedOk = RegInit(False)
  val preparedFault = Reg(Bits(8 bits)) init 0
  val downDone = RegInit(False)
  val downOk = RegInit(False)
  val downFault = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)
  val activeFault = Reg(Bits(8 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0

  val downImageValid = RegInit(False)
  val downImage = Reg(BitNetResidentFfnDownInputReady(cfg.downInput)) init
    BitNetResidentFfnDownInputReady(cfg.downInput).getZero

  val incomingQkv = io.command.step.stage ===
    BitNetResidentLayerStage.QkvProjection
  val incomingO = io.command.step.stage ===
    BitNetResidentLayerStage.AttentionOutputProjection
  val incomingGate = io.command.step.stage ===
    BitNetResidentLayerStage.GateUpProjection
  val incomingDown = io.command.step.stage ===
    BitNetResidentLayerStage.DownProjection
  val incomingMatrixStage = incomingQkv || incomingO || incomingGate ||
    incomingDown
  val incomingExecutorLegal =
    (incomingGate && io.command.executor ===
      BitNetResidentLayerEventExecutor.FfnGatePipeline) ||
      (!incomingGate && incomingMatrixStage && io.command.executor ===
        BitNetResidentLayerEventExecutor.MatrixArray)

  val sourceEventLegal =
    (incomingQkv && io.command.event.stage ===
      BitNetResidentLayerEventStage.QkvResidentReady) ||
      (incomingO && io.command.event.stage ===
        BitNetResidentLayerEventStage.AttentionBranchCommit) ||
      (incomingGate && io.command.event.stage ===
        BitNetResidentLayerEventStage.FfnDownInputReady) ||
      (incomingDown && io.command.event.stage ===
        BitNetResidentLayerEventStage.LayerCommit)
  val sourceIdentityLegal =
    io.command.event.requestId === io.command.step.requestId &&
      io.command.event.layer === io.command.step.layer &&
      io.command.event.mode === io.command.step.mode &&
      io.command.event.tokenBase === io.command.step.tokenBase &&
      io.command.event.tokenCount === io.command.step.tokenCount &&
      io.command.event.positionBase === io.command.step.positionBase &&
      io.command.event.activeTokenMask ===
        io.command.step.activeTokenMask &&
      io.command.event.resetKv === io.command.step.resetKv &&
      io.command.step.firstLayer === (io.command.event.layer === 0) &&
      io.command.step.lastLayer ===
        (io.command.event.layer === model.layerCount - 1) &&
      io.command.step.unit === BitNetResidentLayerUnit.MatrixArray
  val sourceCausalLegal = Bool()
  sourceCausalLegal := True
  for (lane <- 0 until model.tokenParallelism) {
    when(io.command.event.laneCausalLength(lane) =/=
        io.command.step.laneCausalLength(lane)) {
      sourceCausalLegal := False
    }
  }

  val incomingEventOrdinal =
    io.command.event.stage.asBits.asUInt.resize(
      BitNetResidentLayerGeometry.stepIdWidth(model))
  val incomingExpectedStepId =
    ((io.command.event.layer.resize(
      BitNetResidentLayerGeometry.stepIdWidth(model)) << 3) +
      incomingEventOrdinal).resize(
        BitNetResidentLayerGeometry.stepIdWidth(model))
  val incomingStepIdLegal =
    io.command.step.stepId === incomingExpectedStepId

  val expectedPreviousStep =
    (io.command.step.stepId - 1).resize(
      BitNetResidentLayerGeometry.stepIdWidth(model))
  val incomingDownImageLegal = downImageValid && downImage.success &&
    downImage.requestId === io.command.step.requestId &&
    downImage.stepId === expectedPreviousStep &&
    downImage.layer === io.command.step.layer &&
    downImage.mode === io.command.step.mode &&
    downImage.tokenMask === io.command.step.activeTokenMask &&
    downImage.imageTag === expectedPreviousStep.resize(q.tagWidth) &&
    downImage.imageEpoch === io.command.event.epoch &&
    downImage.imageSlot === io.command.event.slot.resize(tile.slotWidth) &&
    downImage.featureCount === model.ffnSize
  val incomingLegal = incomingMatrixStage && incomingExecutorLegal &&
    sourceEventLegal && sourceIdentityLegal && sourceCausalLegal &&
    incomingStepIdLegal &&
    (!incomingDown || incomingDownImageLegal)

  io.command.ready := state === State.Idle && !responseValid
  when(io.command.fire) {
    held := io.command.payload
    preparedDone := False
    preparedOk := False
    preparedFault := 0
    downDone := !incomingGate
    downOk := !incomingGate
    downFault := 0
    activeFault := 0
    when(incomingGate) { downImageValid := False }
    when(incomingLegal) {
      state := State.IssuePrepared
    } otherwise {
      response.requestId := io.command.step.requestId
      response.stepId := io.command.step.stepId
      response.stage := io.command.step.stage
      response.layer := io.command.step.layer
      response.success := False
      response.faultCode := B(
        BitNetResidentMatrixEventFault.InvalidCommand, 8 bits)
      when(incomingDown && !incomingDownImageLegal) {
        response.faultCode := B(
          BitNetResidentMatrixEventFault.MissingDownImage, 8 bits)
      }
      activeFault := B(
        BitNetResidentMatrixEventFault.InvalidCommand, 8 bits)
      when(incomingDown && !incomingDownImageLegal) {
        activeFault := B(
          BitNetResidentMatrixEventFault.MissingDownImage, 8 bits)
      }
      responseValid := True
      stickyFault := True
      state := State.Report
    }
  }

  val heldGate = held.step.stage ===
    BitNetResidentLayerStage.GateUpProjection
  val heldDown = held.step.stage ===
    BitNetResidentLayerStage.DownProjection
  val followingSlot = (held.event.slot + 1).resize(tile.slotWidth)

  prepared.io.request.valid := state === State.IssuePrepared
  prepared.io.request.step := held.step
  prepared.io.request.imageEpoch := held.event.epoch
  prepared.io.request.imageSlot := Mux(heldDown,
    downImage.imageSlot, held.event.slot.resize(tile.slotWidth))
  prepared.io.request.prebuiltDownImage := heldDown
  prepared.io.request.prebuiltImageTag := downImage.imageTag
  prepared.io.request.prebuiltActivationFactorQ16 :=
    downImage.activationFactorQ16
  when(prepared.io.request.fire) {
    when(heldDown) { downImageValid := False }
    state := State.Active
  }

  // Direct prepared-stage boundaries.
  io.scaleRequest << prepared.io.scaleRequest
  prepared.io.scaleSet << io.scaleSet
  io.gateScaleCommand << prepared.io.gateScaleCommand
  prepared.io.gateScaleResult << io.gateScaleResult
  io.imageBinding << prepared.io.imageBinding
  prepared.io.bindingCompletion << io.bindingCompletion
  io.matrixCommand << prepared.io.matrixCommand
  io.gateCommand << prepared.io.gateCommand
  prepared.io.matrixCompletion << io.matrixCompletion

  // Bind the GateUp RMS command atomically to the event that produced it.
  down.io.context.valid := state === State.Active && heldGate &&
    io.gateRmsCommand.valid
  down.io.context.requestId := held.step.requestId
  down.io.context.stepId := held.step.stepId
  down.io.context.layer := held.step.layer
  down.io.context.mode := held.step.mode
  down.io.context.tokenMask := held.step.activeTokenMask
  down.io.context.imageEpoch := held.event.epoch
  down.io.context.imageSlot := followingSlot
  down.io.gateRmsCommand.valid := state === State.Active && heldGate &&
    io.gateRmsCommand.valid
  down.io.gateRmsCommand.payload := io.gateRmsCommand.payload
  io.gateRmsCommand.ready := state === State.Active && heldGate &&
    down.io.context.ready && down.io.gateRmsCommand.ready

  io.postGateContext << down.io.postGateContext
  io.postGateRmsCommand << down.io.postGateRmsCommand
  down.io.postGateCompletion << io.postGateCompletion

  // One physical activation builder; ownership spans command -> completion.
  object ActivationOwner extends SpinalEnum(binarySequential) {
    val None, Prepared, Down = newElement()
  }
  val activationOwner = Reg(ActivationOwner()) init ActivationOwner.None
  val activationFree = activationOwner === ActivationOwner.None
  val preparedActivationSelected = activationFree &&
    prepared.io.activationCommand.valid
  val downActivationSelected = activationFree &&
    !prepared.io.activationCommand.valid && down.io.activationCommand.valid
  val activationContention = activationFree &&
    prepared.io.activationCommand.valid && down.io.activationCommand.valid

  io.activationCommand.valid := preparedActivationSelected ||
    downActivationSelected
  io.activationCommand.payload := prepared.io.activationCommand.payload
  when(downActivationSelected) {
    io.activationCommand.payload := down.io.activationCommand.payload
  }
  prepared.io.activationCommand.ready := preparedActivationSelected &&
    io.activationCommand.ready
  down.io.activationCommand.ready := downActivationSelected &&
    io.activationCommand.ready
  when(io.activationCommand.fire) {
    when(preparedActivationSelected) {
      activationOwner := ActivationOwner.Prepared
    } otherwise {
      activationOwner := ActivationOwner.Down
    }
  }
  when(activationContention) {
    stickyFault := True
    activeFault := B(
      BitNetResidentMatrixEventFault.ActivationContention, 8 bits)
  }

  prepared.io.activationCompletion.valid :=
    io.activationCompletion.valid &&
      activationOwner === ActivationOwner.Prepared
  prepared.io.activationCompletion.payload := io.activationCompletion.payload
  down.io.activationCompletion.valid := io.activationCompletion.valid &&
    activationOwner === ActivationOwner.Down
  down.io.activationCompletion.payload := io.activationCompletion.payload
  io.activationCompletion.ready := False
  when(activationOwner === ActivationOwner.Prepared) {
    io.activationCompletion.ready := prepared.io.activationCompletion.ready
  }
  when(activationOwner === ActivationOwner.Down) {
    io.activationCompletion.ready := down.io.activationCompletion.ready
  }
  when(io.activationCompletion.fire) {
    activationOwner := ActivationOwner.None
  }
  when(io.activationCompletion.valid &&
      activationOwner === ActivationOwner.None) {
    stickyFault := True
    activeFault := B(
      BitNetResidentMatrixEventFault.UnexpectedActivationCompletion,
      8 bits)
  }

  val preparedIdentity =
    prepared.io.completion.requestId === held.step.requestId &&
      prepared.io.completion.stepId === held.step.stepId &&
      prepared.io.completion.payload.stage === held.step.stage &&
      prepared.io.completion.layer === held.step.layer
  prepared.io.completion.ready := state === State.Active && !preparedDone

  val downReadyIdentity =
    down.io.ready.requestId === held.step.requestId &&
      down.io.ready.stepId === held.step.stepId &&
      down.io.ready.layer === held.step.layer &&
      down.io.ready.mode === held.step.mode &&
      down.io.ready.tokenMask === held.step.activeTokenMask &&
      down.io.ready.imageTag === held.step.stepId.resize(q.tagWidth) &&
      down.io.ready.imageEpoch === held.event.epoch &&
      down.io.ready.imageSlot === followingSlot &&
      down.io.ready.featureCount === model.ffnSize
  down.io.ready.ready := state === State.Active && heldGate && !downDone

  val preparedFire = prepared.io.completion.fire
  val preparedSuccessNow = preparedIdentity &&
    prepared.io.completion.success
  val preparedFaultNow = Mux(!preparedIdentity,
    B(BitNetResidentMatrixEventFault.PreparedIdentity, 8 bits),
    Mux(prepared.io.completion.faultCode.orR,
      prepared.io.completion.faultCode,
      B(BitNetResidentMatrixEventFault.PreparedFailed, 8 bits)))
  when(preparedFire) {
    preparedDone := True
    preparedOk := preparedSuccessNow
    preparedFault := Mux(preparedSuccessNow, B(0, 8 bits),
      preparedFaultNow)
    when(!preparedSuccessNow) {
      stickyFault := True
      activeFault := preparedFaultNow
    }
  }

  val downFire = down.io.ready.fire
  val downSuccessNow = downReadyIdentity && down.io.ready.success
  val downFaultNow = Mux(!downReadyIdentity,
    B(BitNetResidentMatrixEventFault.DownInputIdentity, 8 bits),
    Mux(down.io.ready.faultCode.orR, down.io.ready.faultCode,
      B(BitNetResidentMatrixEventFault.DownInputFailed, 8 bits)))
  when(downFire) {
    downDone := True
    downOk := downSuccessNow
    downFault := Mux(downSuccessNow, B(0, 8 bits), downFaultNow)
    when(downSuccessNow) {
      downImage := down.io.ready.payload
      downImageValid := True
    } otherwise {
      downImageValid := False
      stickyFault := True
      activeFault := downFaultNow
    }
  }

  val preparedDoneAfter = preparedDone || preparedFire
  val preparedOkAfter = Mux(preparedFire,
    preparedSuccessNow, preparedOk)
  val preparedFaultAfter = Mux(preparedFire,
    Mux(preparedSuccessNow, B(0, 8 bits), preparedFaultNow),
    preparedFault)
  val downDoneAfter = downDone || downFire
  val downOkAfter = Mux(downFire, downSuccessNow, downOk)
  val downFaultAfter = Mux(downFire,
    Mux(downSuccessNow, B(0, 8 bits), downFaultNow), downFault)

  when(state === State.Active && preparedDoneAfter && downDoneAfter &&
      !responseValid) {
    response.requestId := held.step.requestId
    response.stepId := held.step.stepId
    response.stage := held.step.stage
    response.layer := held.step.layer
    response.success := preparedOkAfter && downOkAfter && !activeFault.orR
    response.faultCode := 0
    when(!preparedOkAfter) {
      response.faultCode := preparedFaultAfter
    } elsewhen(!downOkAfter) {
      response.faultCode := downFaultAfter
    } elsewhen(activeFault.orR) {
      response.faultCode := activeFault
    }
    responseValid := True
    state := State.Report
  }

  io.completion.valid := responseValid
  io.completion.payload := response
  when(io.completion.fire) {
    responseValid := False
    completedCount := completedCount + 1
    state := State.Idle
  }

  io.busy := state =/= State.Idle || responseValid ||
    activationOwner =/= ActivationOwner.None
  io.downImageResident := downImageValid
  io.protocolFaultSticky := stickyFault ||
    prepared.io.protocolFaultSticky || down.io.protocolFaultSticky
  io.faultCode := activeFault
  io.completedEvents := completedCount
}

object GenerateBitNetResidentMatrixEventCoordinator extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-matrix-event-coordinator")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentMatrixEventCoordinator())
}
