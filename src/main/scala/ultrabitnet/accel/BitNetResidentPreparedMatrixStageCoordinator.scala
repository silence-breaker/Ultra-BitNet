package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentPreparedMatrixStageFault {
  val None = 0x00
  val InvalidRequest = 0xb8
  val ScaleIdentity = 0xb9
  val ScaleFailed = 0xba
  val ActivationIdentity = 0xbb
  val ActivationFailed = 0xbc
  val GateScaleIdentity = 0xbd
  val GateScaleFailed = 0xbe
  val MatrixIdentity = 0xbf
  val MatrixFailed = 0xc0
  val BindingIdentity = 0xc1
  val BindingFailed = 0xc2
}

/** Static contract of the complete pre-matrix preparation plane. */
case class BitNetResidentPreparedMatrixStageCoordinatorConfig(
    matrix: BitNetResidentMatrixStageControlSubsystemConfig =
      BitNetResidentMatrixStageControlSubsystemConfig(),
    activation: BitNetResidentProjectionActivationSubsystemConfig =
      BitNetResidentProjectionActivationSubsystemConfig(),
    scales: BitNetResidentProjectionScaleStagerConfig =
      BitNetResidentProjectionScaleStagerConfig(),
    binder: BitNetResidentActivationReplayBinderConfig =
      BitNetResidentActivationReplayBinderConfig(),
    gateScale: BitNetResidentGateScaleComposerConfig =
      BitNetResidentGateScaleComposerConfig()
) {
  val model: BitNetConfig = matrix.model
  val stage: BitNetResidentMatrixStageLaunchExpanderConfig = matrix.stage
  val gate: BitNetSharedSpuFfnGateConfig = matrix.gate.gate.gate
  val stepIdWidth: Int = BitNetResidentLayerGeometry.stepIdWidth(model)

  require(model == BitNetConfig.Production)
  require(activation.model == model && scales.model == model)
  require(binder.model == model && binder.linear == matrix.endpoint)
  require(binder.activation == activation)
  require(gateScale.model == model && gateScale.gate == gate)
  require(scales.epochWidth == activation.quant.epochWidth)
  require(scales.epochWidth == gateScale.epochWidth)
  require(activation.quant.tagWidth >= stepIdWidth)
}

/**
  * One matrix stage after event lowering.  Down consumes the image already
  * built by the post-gate coordinator; every other projection builds a fresh
  * hidden-size image from NarrowWork.
  */
case class BitNetResidentPreparedMatrixStageRequest(
    cfg: BitNetResidentPreparedMatrixStageCoordinatorConfig)
    extends Bundle {
  val step = BitNetResidentLayerStep(cfg.model)
  val imageEpoch = UInt(cfg.activation.quant.epochWidth bits)
  val imageSlot = UInt(cfg.activation.tile.slotWidth bits)
  val prebuiltDownImage = Bool()
  val prebuiltImageTag = UInt(cfg.activation.quant.tagWidth bits)
  val prebuiltActivationFactorQ16 = Vec(
    UInt(32 bits), cfg.model.tokenParallelism)
}

/**
  * Descriptor-free preparation and retirement coordinator for Q/K/V, O,
  * GateUp and Down.
  *
  * The block owns no arithmetic, activation memory, weight memory or matrix
  * array.  It fetches the seven resident layer scales and, except for Down,
  * asks the sole dynamic-i8 builder for one NarrowWork image.  It then binds
  * that image to the sole feeder before releasing the MatrixStage command.
  * GateUp additionally waits for the shared-pool scale composer and holds its
  * exact combined-scale command until the fused GateUp child accepts it.
  * Public completion is withheld until both MatrixStage and activation-image
  * lifetime retirement have authenticated the same transaction.
  */
class BitNetResidentPreparedMatrixStageCoordinator(
    cfg: BitNetResidentPreparedMatrixStageCoordinatorConfig =
      BitNetResidentPreparedMatrixStageCoordinatorConfig())
    extends Component {
  private val model = cfg.model
  private val q = cfg.activation.quant
  private val tile = cfg.activation.tile
  private val gate = cfg.gate

  val io = new Bundle {
    val request = slave(Stream(
      BitNetResidentPreparedMatrixStageRequest(cfg)))

    val scaleRequest = master(Stream(
      BitNetResidentProjectionScaleRequest(cfg.scales)))
    val scaleSet = slave(Stream(
      BitNetResidentProjectionScaleSet(cfg.scales)))

    val activationCommand = master(Stream(
      BitNetResidentProjectionActivationCommand(cfg.activation)))
    val activationCompletion = slave(Stream(
      BitNetResidentProjectionActivationCompletion(cfg.activation)))

    val gateScaleCommand = master(Stream(
      BitNetResidentGateScaleComposeCommand(cfg.gateScale)))
    val gateScaleResult = slave(Stream(
      BitNetResidentGateScaleComposeResult(cfg.gateScale)))

    val imageBinding = master(Stream(
      BitNetResidentActivationImageBinding(cfg.binder)))
    val bindingCompletion = slave(Stream(
      BitNetResidentActivationBindingCompletion(cfg.binder)))

    val matrixCommand = master(Stream(
      BitNetResidentMatrixStageCommand(cfg.stage)))
    val gateCommand = master(Stream(BitNetSharedFfnGateCommand(gate)))
    val matrixCompletion = slave(Stream(
      BitNetResidentLayerStepCompletion(model)))

    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val faultCode = out Bits(8 bits)
    val completedStages = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssuePreparation, WaitPreparation, EvaluatePreparation,
        IssueGateScale, WaitGateScale, EvaluateGateScale,
        IssueBinding, IssueMatrix, WaitExecution, EvaluateExecution,
        Report = newElement()
  }
  val state = Reg(State()) init State.Idle

  val held = Reg(BitNetResidentPreparedMatrixStageRequest(cfg)) init
    BitNetResidentPreparedMatrixStageRequest(cfg).getZero
  val scaleValues = Vec(
    Reg(UInt(32 bits)) init 0, BitNetResidentMatrixScaleIndex.Count)
  val activationFactors = Vec(
    Reg(UInt(32 bits)) init 0, model.tokenParallelism)
  val gateScaleValues = Reg(
    BitNetResidentGateScaleComposeResult(cfg.gateScale)) init
      BitNetResidentGateScaleComposeResult(cfg.gateScale).getZero

  val scaleIssued = RegInit(False)
  val activationIssued = RegInit(False)
  val scaleReceived = RegInit(False)
  val activationReceived = RegInit(False)
  val gateCommandPending = RegInit(False)
  val matrixReceived = RegInit(False)
  val bindingReceived = RegInit(False)
  val jobFault = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)
  val completedCounter = Reg(UInt(32 bits)) init 0

  val incomingMatrixStage =
    io.request.step.stage === BitNetResidentLayerStage.QkvProjection ||
      io.request.step.stage ===
        BitNetResidentLayerStage.AttentionOutputProjection ||
      io.request.step.stage === BitNetResidentLayerStage.GateUpProjection ||
      io.request.step.stage === BitNetResidentLayerStage.DownProjection
  val incomingDown = io.request.step.stage ===
    BitNetResidentLayerStage.DownProjection
  val incomingPrefixMask = (1 to model.tokenParallelism).map { count =>
    io.request.step.activeTokenMask === B((BigInt(1) << count) - 1,
      model.tokenParallelism bits)
  }.reduce(_ || _)
  val incomingLayerLegal = io.request.step.layer < model.layerCount
  // The post-gate image is produced while retiring GateUp (stage ordinal 6)
  // and consumed by the immediately following Down step (ordinal 7).  Its
  // physical image tag must therefore remain the preceding GateUp stepId;
  // activationReuseTag below intentionally remains the current Down stepId.
  val incomingPreviousStepTag =
    (io.request.step.stepId - 1).resize(q.tagWidth)
  val incomingPrebuiltLegal =
    (!incomingDown && !io.request.prebuiltDownImage) ||
      (incomingDown && io.request.prebuiltDownImage &&
        io.request.step.stepId =/= 0 &&
        io.request.prebuiltImageTag === incomingPreviousStepTag)
  val incomingLegal = incomingMatrixStage && incomingLayerLegal &&
    incomingPrefixMask &&
    io.request.step.unit === BitNetResidentLayerUnit.MatrixArray &&
    incomingPrebuiltLegal

  io.request.ready := state === State.Idle
  when(io.request.fire) {
    held := io.request.payload
    scaleValues.foreach(_ := 0)
    activationFactors.foreach(_ := 0)
    gateScaleValues := BitNetResidentGateScaleComposeResult(
      cfg.gateScale).getZero
    scaleIssued := False
    activationIssued := io.request.prebuiltDownImage
    scaleReceived := False
    activationReceived := io.request.prebuiltDownImage
    gateCommandPending := False
    matrixReceived := False
    bindingReceived := False
    jobFault := 0
    when(io.request.prebuiltDownImage) {
      for (lane <- 0 until model.tokenParallelism) {
        activationFactors(lane) :=
          io.request.prebuiltActivationFactorQ16(lane)
      }
    }
    when(incomingLegal) {
      state := State.IssuePreparation
    } otherwise {
      jobFault := B(
        BitNetResidentPreparedMatrixStageFault.InvalidRequest, 8 bits)
      stickyFault := True
      state := State.Report
    }
  }

  io.scaleRequest.valid := state === State.IssuePreparation && !scaleIssued
  io.scaleRequest.requestId := held.step.requestId
  io.scaleRequest.layer := held.step.layer
  io.scaleRequest.epoch := held.imageEpoch.resize(cfg.scales.epochWidth)
  when(io.scaleRequest.fire) { scaleIssued := True }

  io.activationCommand.valid := state === State.IssuePreparation &&
    !activationIssued && !held.prebuiltDownImage
  io.activationCommand.requestId := held.step.requestId
  io.activationCommand.tag := held.step.stepId.resize(q.tagWidth)
  io.activationCommand.epoch := held.imageEpoch
  io.activationCommand.slot := held.imageSlot
  io.activationCommand.sourceBuffer :=
    BitNetPhysicalActivationBuffer.NarrowWork
  io.activationCommand.featureCount := model.hiddenSize
  io.activationCommand.tokenMask := held.step.activeTokenMask
  when(io.activationCommand.fire) { activationIssued := True }

  val scaleIssuedAfter = scaleIssued || io.scaleRequest.fire
  val activationIssuedAfter = activationIssued || io.activationCommand.fire
  when(state === State.IssuePreparation && scaleIssuedAfter &&
      activationIssuedAfter) {
    state := State.WaitPreparation
  }

  val preparationOpen = state === State.IssuePreparation ||
    state === State.WaitPreparation
  io.scaleSet.ready := preparationOpen && scaleIssuedAfter && !scaleReceived
  io.activationCompletion.ready := preparationOpen &&
    activationIssuedAfter && !activationReceived && !held.prebuiltDownImage

  val scaleIdentity =
    io.scaleSet.requestId === held.step.requestId &&
      io.scaleSet.layer === held.step.layer &&
      io.scaleSet.epoch === held.imageEpoch.resize(cfg.scales.epochWidth)
  when(io.scaleSet.fire) {
    scaleReceived := True
    scaleValues := io.scaleSet.valueQ16
    when(!scaleIdentity) {
      jobFault := B(
        BitNetResidentPreparedMatrixStageFault.ScaleIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.scaleSet.success) {
      jobFault := Mux(io.scaleSet.faultCode.orR,
        io.scaleSet.faultCode,
        B(BitNetResidentPreparedMatrixStageFault.ScaleFailed, 8 bits))
      stickyFault := True
    }
  }

  val activationIdentity =
    io.activationCompletion.requestId === held.step.requestId &&
      io.activationCompletion.tag === held.step.stepId.resize(q.tagWidth) &&
      io.activationCompletion.epoch === held.imageEpoch &&
      io.activationCompletion.slot === held.imageSlot &&
      io.activationCompletion.sourceBuffer ===
        BitNetPhysicalActivationBuffer.NarrowWork &&
      io.activationCompletion.featureCount === model.hiddenSize &&
      io.activationCompletion.acceptedFeatures === model.hiddenSize &&
      io.activationCompletion.tokenMask === held.step.activeTokenMask
  when(io.activationCompletion.fire) {
    activationReceived := True
    for (lane <- 0 until model.tokenParallelism) {
      activationFactors(lane) :=
        io.activationCompletion.dequantFactorQ16(lane)
    }
    when(!activationIdentity) {
      jobFault := B(
        BitNetResidentPreparedMatrixStageFault.ActivationIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.activationCompletion.success) {
      jobFault := B(
        BitNetResidentPreparedMatrixStageFault.ActivationFailed, 8 bits)
      stickyFault := True
    }
  }

  when(state === State.WaitPreparation && scaleReceived &&
      activationReceived) {
    state := State.EvaluatePreparation
  }
  val heldGateUp = held.step.stage ===
    BitNetResidentLayerStage.GateUpProjection
  when(state === State.EvaluatePreparation) {
    when(jobFault.orR) {
      state := State.Report
    } elsewhen(heldGateUp) {
      state := State.IssueGateScale
    } otherwise {
      state := State.IssueBinding
    }
  }

  io.gateScaleCommand.valid := state === State.IssueGateScale
  io.gateScaleCommand.requestId := held.step.requestId
  io.gateScaleCommand.stepId := held.step.stepId
  io.gateScaleCommand.layer := held.step.layer
  io.gateScaleCommand.epoch := held.imageEpoch.resize(cfg.gateScale.epochWidth)
  io.gateScaleCommand.tokenMask := held.step.activeTokenMask
  io.gateScaleCommand.activationFactorQ16 := activationFactors
  io.gateScaleCommand.gateWeightScaleQ16 :=
    scaleValues(BitNetResidentMatrixScaleIndex.Gate)
  io.gateScaleCommand.upWeightScaleQ16 :=
    scaleValues(BitNetResidentMatrixScaleIndex.Up)
  when(io.gateScaleCommand.fire) { state := State.WaitGateScale }

  io.gateScaleResult.ready := state === State.WaitGateScale
  val gateScaleIdentity =
    io.gateScaleResult.requestId === held.step.requestId &&
      io.gateScaleResult.stepId === held.step.stepId &&
      io.gateScaleResult.layer === held.step.layer &&
      io.gateScaleResult.epoch ===
        held.imageEpoch.resize(cfg.gateScale.epochWidth) &&
      io.gateScaleResult.tokenMask === held.step.activeTokenMask
  when(io.gateScaleResult.fire) {
    gateScaleValues := io.gateScaleResult.payload
    when(!gateScaleIdentity) {
      jobFault := B(
        BitNetResidentPreparedMatrixStageFault.GateScaleIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.gateScaleResult.success) {
      jobFault := Mux(io.gateScaleResult.faultCode.orR,
        io.gateScaleResult.faultCode,
        B(BitNetResidentPreparedMatrixStageFault.GateScaleFailed, 8 bits))
      stickyFault := True
    }
    state := State.EvaluateGateScale
  }
  when(state === State.EvaluateGateScale) {
    when(jobFault.orR) { state := State.Report }
      .otherwise { state := State.IssueBinding }
  }

  val heldDown = held.step.stage === BitNetResidentLayerStage.DownProjection
  val bindingInputFeatures = UInt(tile.featureCountWidth bits)
  bindingInputFeatures := model.hiddenSize
  when(heldDown) { bindingInputFeatures := model.ffnSize }
  val bindingImageTag = UInt(q.tagWidth bits)
  bindingImageTag := held.step.stepId.resize(q.tagWidth)
  when(held.prebuiltDownImage) {
    bindingImageTag := held.prebuiltImageTag
  }
  val bindingLaunchCount = UInt(cfg.binder.launchCountWidth bits)
  bindingLaunchCount := 1
  when(held.step.stage === BitNetResidentLayerStage.QkvProjection) {
    bindingLaunchCount := 3
  }

  io.imageBinding.valid := state === State.IssueBinding
  io.imageBinding.requestId := held.step.requestId
  io.imageBinding.stepId := held.step.stepId
  io.imageBinding.activationReuseTag := held.step.stepId
  io.imageBinding.imageTag := bindingImageTag
  io.imageBinding.epoch := held.imageEpoch
  io.imageBinding.slot := held.imageSlot
  io.imageBinding.inputFeatures := bindingInputFeatures
  io.imageBinding.tokenMask := held.step.activeTokenMask
  io.imageBinding.expectedLaunchCount := bindingLaunchCount
  when(io.imageBinding.fire) {
    gateCommandPending := heldGateUp
    matrixReceived := False
    bindingReceived := False
    state := State.IssueMatrix
  }

  io.matrixCommand.valid := state === State.IssueMatrix
  io.matrixCommand.step := held.step
  io.matrixCommand.activationFactorQ16 := activationFactors
  io.matrixCommand.weightScaleQ16 := scaleValues
  when(io.matrixCommand.fire) { state := State.WaitExecution }

  io.gateCommand.valid := (state === State.IssueMatrix ||
    state === State.WaitExecution) && gateCommandPending
  io.gateCommand.length := model.ffnSize
  io.gateCommand.tokenMask := held.step.activeTokenMask
  io.gateCommand.mode := held.step.mode
  io.gateCommand.authenticatedScatter := True
  io.gateCommand.gateScaleQ16 := gateScaleValues.gateScaleQ16
  io.gateCommand.upScaleQ16 := gateScaleValues.upScaleQ16
  io.gateCommand.tag := held.step.stepId.resize(gate.tagWidth)
  when(io.gateCommand.fire) { gateCommandPending := False }

  // A child completion cannot be owned before MatrixStage has accepted the
  // command.  Keeping ready low in IssueMatrix prevents a stale, numerically
  // matching completion from an earlier request-id wrap from retiring a stage
  // that has not actually launched.
  val executionOpen = state === State.WaitExecution
  io.matrixCompletion.ready := executionOpen && !matrixReceived
  io.bindingCompletion.ready := executionOpen && !bindingReceived

  val matrixIdentity =
    io.matrixCompletion.requestId === held.step.requestId &&
      io.matrixCompletion.stepId === held.step.stepId &&
      io.matrixCompletion.payload.stage === held.step.stage &&
      io.matrixCompletion.layer === held.step.layer
  when(io.matrixCompletion.fire) {
    matrixReceived := True
    when(!matrixIdentity) {
      jobFault := B(
        BitNetResidentPreparedMatrixStageFault.MatrixIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.matrixCompletion.success) {
      jobFault := Mux(io.matrixCompletion.faultCode.orR,
        io.matrixCompletion.faultCode,
        B(BitNetResidentPreparedMatrixStageFault.MatrixFailed, 8 bits))
      stickyFault := True
    }
  }

  val bindingIdentity =
    io.bindingCompletion.requestId === held.step.requestId &&
      io.bindingCompletion.stepId === held.step.stepId &&
      io.bindingCompletion.activationReuseTag === held.step.stepId &&
      io.bindingCompletion.imageTag === bindingImageTag &&
      io.bindingCompletion.epoch === held.imageEpoch &&
      io.bindingCompletion.slot === held.imageSlot &&
      io.bindingCompletion.inputFeatures === bindingInputFeatures &&
      io.bindingCompletion.tokenMask === held.step.activeTokenMask &&
      io.bindingCompletion.expectedLaunchCount === bindingLaunchCount &&
      io.bindingCompletion.completedLaunchCount === bindingLaunchCount
  when(io.bindingCompletion.fire) {
    bindingReceived := True
    when(!bindingIdentity) {
      jobFault := B(
        BitNetResidentPreparedMatrixStageFault.BindingIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.bindingCompletion.success) {
      jobFault := B(
        BitNetResidentPreparedMatrixStageFault.BindingFailed, 8 bits)
      stickyFault := True
    }
  }

  when(state === State.WaitExecution && matrixReceived && bindingReceived &&
      !gateCommandPending) {
    state := State.EvaluateExecution
  }
  when(state === State.EvaluateExecution) { state := State.Report }

  io.completion.valid := state === State.Report
  io.completion.requestId := held.step.requestId
  io.completion.stepId := held.step.stepId
  io.completion.payload.stage := held.step.stage
  io.completion.layer := held.step.layer
  io.completion.success := !jobFault.orR
  io.completion.faultCode := jobFault
  when(io.completion.fire) {
    completedCounter := completedCounter + 1
    state := State.Idle
  }

  io.busy := state =/= State.Idle
  io.protocolFaultSticky := stickyFault
  io.faultCode := jobFault
  io.completedStages := completedCounter
}

object GenerateBitNetResidentPreparedMatrixStageCoordinator extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-prepared-matrix-stage-coordinator")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentPreparedMatrixStageCoordinator())
}
