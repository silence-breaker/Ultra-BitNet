package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentLayerActivationLeaseFault {
  val InvalidExecutor = 0xeb
  val CompletionIdentity = 0xec
  val CompletionFailed = 0xed
  val LostActivationOwner = 0xee
  val UnexpectedCompletion = 0xef
}

/**
  * Whole-event activation owner and physical-executor dispatcher.
  *
  * Every lowered resident event acquires exactly one logical client of the
  * sole activation RAM before its physical executor is allowed to observe the
  * command.  Completion is captured first, the matching activation lease is
  * then released only after all physical read/write traffic drains, and only
  * that release permits the completion to leave this component.
  *
  * Matrix subflows use one client for their complete lifetime:
  * QKV -> Matrix, O/Down -> Residual, and GateUp plus post-gate RMS/Down-image
  * creation -> Gate.  This lets their internal phases time-multiplex the same
  * four activation ports without PS-visible tensors or mid-event ownership
  * handoffs.
  */
class BitNetResidentLayerActivationLeaseCoordinator(
    cfg: BitNetResidentLayerEventSchedulerConfig =
      BitNetResidentLayerEventSchedulerConfig()) extends Component {
  private val model = cfg.model

  val io = new Bundle {
    val command = slave(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg)))

    val rmsCommand = master(Stream(BitNetResidentLayerStep(model)))
    val matrixCommand = master(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg)))
    val attentionCommand = master(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg)))

    val rmsCompletion = slave(Stream(
      BitNetResidentLayerStepCompletion(model)))
    val matrixCompletion = slave(Stream(
      BitNetResidentLayerStepCompletion(model)))
    val attentionCompletion = slave(Stream(
      BitNetResidentLayerStepCompletion(model)))

    val activationAcquire = master(Stream(
      BitNetActivationFabricLease(model)))
    val activationRelease = master(Stream(
      BitNetActivationFabricLease(model)))
    val activationOwnerValid = in Bool()
    val activationOwner = in(BitNetActivationClient())
    val activationOwnerRequestId = in UInt(model.requestIdWidth bits)

    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val completedEvents = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Acquire, Dispatch, WaitCompletion, Release, Report,
        Quarantine = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentLayerEventExecutorCommand(cfg)) init
    BitNetResidentLayerEventExecutorCommand(cfg).getZero
  val leaseClient = Reg(BitNetActivationClient()) init
    BitNetActivationClient.Rms
  val response = Reg(BitNetResidentLayerStepCompletion(model)) init
    BitNetResidentLayerStepCompletion(model).getZero
  val responseValid = RegInit(False)
  val stickyFault = RegInit(False)
  val completedCount = Reg(UInt(32 bits)) init 0

  val incomingRms = io.command.executor ===
    BitNetResidentLayerEventExecutor.RmsNorm
  val incomingMatrix = io.command.executor ===
    BitNetResidentLayerEventExecutor.MatrixArray
  val incomingAttention = io.command.executor ===
    BitNetResidentLayerEventExecutor.Attention
  val incomingGate = io.command.executor ===
    BitNetResidentLayerEventExecutor.FfnGatePipeline

  val incomingRmsStage = io.command.step.stage ===
    BitNetResidentLayerStage.AttentionRmsNorm ||
    io.command.step.stage ===
      BitNetResidentLayerStage.AttentionSubRmsNorm ||
    io.command.step.stage === BitNetResidentLayerStage.FfnRmsNorm ||
    io.command.step.stage === BitNetResidentLayerStage.FinalRmsNorm
  val incomingQkv = io.command.step.stage ===
    BitNetResidentLayerStage.QkvProjection
  val incomingResidualMatrix = io.command.step.stage ===
    BitNetResidentLayerStage.AttentionOutputProjection ||
    io.command.step.stage === BitNetResidentLayerStage.DownProjection
  val incomingAttentionStage = io.command.step.stage ===
    BitNetResidentLayerStage.ResidentAttention
  val incomingGateStage = io.command.step.stage ===
    BitNetResidentLayerStage.GateUpProjection
  val incomingLegal =
    (incomingRms && incomingRmsStage && io.command.step.unit ===
      BitNetResidentLayerUnit.RmsNorm) ||
      (incomingMatrix && (incomingQkv || incomingResidualMatrix) &&
        io.command.step.unit === BitNetResidentLayerUnit.MatrixArray) ||
      (incomingAttention && incomingAttentionStage &&
        io.command.step.unit === BitNetResidentLayerUnit.Attention) ||
      (incomingGate && incomingGateStage &&
        io.command.step.unit === BitNetResidentLayerUnit.MatrixArray)

  val incomingClient = BitNetActivationClient()
  incomingClient := BitNetActivationClient.Rms
  when(incomingQkv) {
    incomingClient := BitNetActivationClient.Matrix
  }
  when(incomingResidualMatrix) {
    incomingClient := BitNetActivationClient.Residual
  }
  when(incomingAttentionStage) {
    incomingClient := BitNetActivationClient.Attention
  }
  when(incomingGateStage) {
    incomingClient := BitNetActivationClient.Gate
  }

  val anyCompletionValid = io.rmsCompletion.valid ||
    io.matrixCompletion.valid || io.attentionCompletion.valid
  io.command.ready := state === State.Idle && !responseValid &&
    !anyCompletionValid
  when(io.command.fire) {
    held := io.command.payload
    leaseClient := incomingClient
    when(incomingLegal) {
      state := State.Acquire
    } otherwise {
      response.requestId := io.command.step.requestId
      response.stepId := io.command.step.stepId
      response.stage := io.command.step.stage
      response.layer := io.command.step.layer
      response.success := False
      response.faultCode := B(
        BitNetResidentLayerActivationLeaseFault.InvalidExecutor, 8 bits)
      responseValid := True
      stickyFault := True
      state := State.Report
    }
  }

  // A completion that survived into Idle cannot be authenticated to a future
  // command.  Quarantine before acquiring any new activation owner.
  when(state === State.Idle && anyCompletionValid) {
    stickyFault := True
    state := State.Quarantine
  }

  io.activationAcquire.valid := state === State.Acquire
  io.activationAcquire.client := leaseClient
  io.activationAcquire.requestId := held.step.requestId
  when(io.activationAcquire.fire) { state := State.Dispatch }

  val ownerMatches = io.activationOwnerValid &&
    io.activationOwner === leaseClient &&
    io.activationOwnerRequestId === held.step.requestId
  val ownerLost = (state === State.Dispatch ||
    state === State.WaitCompletion || state === State.Release) &&
    !ownerMatches

  io.rmsCommand.valid := state === State.Dispatch && ownerMatches &&
    held.executor === BitNetResidentLayerEventExecutor.RmsNorm
  io.rmsCommand.payload := held.step
  io.matrixCommand.valid := state === State.Dispatch && ownerMatches &&
    (held.executor === BitNetResidentLayerEventExecutor.MatrixArray ||
      held.executor ===
        BitNetResidentLayerEventExecutor.FfnGatePipeline)
  io.matrixCommand.payload := held
  io.attentionCommand.valid := state === State.Dispatch && ownerMatches &&
    held.executor === BitNetResidentLayerEventExecutor.Attention
  io.attentionCommand.payload := held

  val selectedCommandFire = io.rmsCommand.fire || io.matrixCommand.fire ||
    io.attentionCommand.fire
  when(selectedCommandFire) { state := State.WaitCompletion }

  io.rmsCompletion.ready := state === State.WaitCompletion &&
    held.executor === BitNetResidentLayerEventExecutor.RmsNorm
  io.matrixCompletion.ready := state === State.WaitCompletion &&
    (held.executor === BitNetResidentLayerEventExecutor.MatrixArray ||
      held.executor ===
        BitNetResidentLayerEventExecutor.FfnGatePipeline)
  io.attentionCompletion.ready := state === State.WaitCompletion &&
    held.executor === BitNetResidentLayerEventExecutor.Attention

  val selectedCompletion = BitNetResidentLayerStepCompletion(model)
  selectedCompletion := io.rmsCompletion.payload
  val selectedCompletionFire = Bool()
  selectedCompletionFire := io.rmsCompletion.fire
  when(held.executor === BitNetResidentLayerEventExecutor.MatrixArray ||
      held.executor ===
        BitNetResidentLayerEventExecutor.FfnGatePipeline) {
    selectedCompletion := io.matrixCompletion.payload
    selectedCompletionFire := io.matrixCompletion.fire
  }
  when(held.executor === BitNetResidentLayerEventExecutor.Attention) {
    selectedCompletion := io.attentionCompletion.payload
    selectedCompletionFire := io.attentionCompletion.fire
  }

  val completionIdentity =
    selectedCompletion.requestId === held.step.requestId &&
      selectedCompletion.stepId === held.step.stepId &&
      selectedCompletion.stage === held.step.stage &&
      selectedCompletion.layer === held.step.layer
  when(selectedCompletionFire) {
    response.requestId := held.step.requestId
    response.stepId := held.step.stepId
    response.stage := held.step.stage
    response.layer := held.step.layer
    response.success := completionIdentity && selectedCompletion.success
    response.faultCode := 0
    when(!completionIdentity) {
      response.faultCode := B(
        BitNetResidentLayerActivationLeaseFault.CompletionIdentity,
        8 bits)
      stickyFault := True
    } elsewhen(!selectedCompletion.success) {
      response.faultCode := Mux(selectedCompletion.faultCode.orR,
        selectedCompletion.faultCode,
        B(BitNetResidentLayerActivationLeaseFault.CompletionFailed,
          8 bits))
      stickyFault := True
    }
    state := State.Release
  }

  // Non-owner completions remain backpressured and are diagnosed.
  when(state === State.WaitCompletion &&
      ((io.rmsCompletion.valid && !io.rmsCompletion.ready) ||
        (io.matrixCompletion.valid && !io.matrixCompletion.ready) ||
        (io.attentionCompletion.valid && !io.attentionCompletion.ready))) {
    stickyFault := True
  }

  io.activationRelease.valid := state === State.Release && ownerMatches
  io.activationRelease.client := leaseClient
  io.activationRelease.requestId := held.step.requestId
  when(io.activationRelease.fire) {
    responseValid := True
    state := State.Report
  }

  // Losing the authenticated physical owner is fail-stop: releasing some
  // other client's lease or publishing the captured result would be unsafe.
  when(ownerLost) {
    response.success := False
    response.faultCode := B(
      BitNetResidentLayerActivationLeaseFault.LostActivationOwner,
      8 bits)
    stickyFault := True
    state := State.Quarantine
  }

  io.completion.valid := responseValid && state === State.Report
  io.completion.payload := response
  when(io.completion.fire) {
    responseValid := False
    completedCount := completedCount + 1
    state := State.Idle
  }

  io.busy := state =/= State.Idle || responseValid
  io.protocolFaultSticky := stickyFault
  io.completedEvents := completedCount
}

object GenerateBitNetResidentLayerActivationLeaseCoordinator extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-layer-activation-lease-coordinator")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLayerActivationLeaseCoordinator())
}
