package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentAttentionEventAdapterFault {
  val InvalidCommand = 0xe8
  val CompletionIdentity = 0xe9
  val CompletionFailed = 0xea
}

/**
  * Typed event-to-attention boundary for the resident QK/Softmax/SV engine.
  *
  * Absolute position is carried explicitly: token/KV identity and RoPE query
  * position are both derived from the already-validated LayerStep context,
  * while per-lane causal lengths become the exact attention key counts.
  * No attention score, probability or context activation crosses this block.
  */
class BitNetResidentAttentionEventAdapter(
    cfg: BitNetResidentLayerEventSchedulerConfig =
      BitNetResidentLayerEventSchedulerConfig()) extends Component {
  private val model = cfg.model

  val io = new Bundle {
    val eventCommand = slave(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg)))
    val attentionCommand = master(Stream(
      BitNetAttentionScheduleCommand(model)))
    val attentionCompletion = slave(Stream(
      BitNetAttentionCompletion(model)))
    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val completedEvents = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Issue, WaitAttention, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentLayerEventExecutorCommand(cfg)) init
    BitNetResidentLayerEventExecutorCommand(cfg).getZero
  val response = Reg(BitNetResidentLayerStepCompletion(model)) init
    BitNetResidentLayerStepCompletion(model).getZero
  val responseValid = RegInit(False)
  val stickyFault = RegInit(False)
  val completedCount = Reg(UInt(32 bits)) init 0

  val incomingIdentity =
    io.eventCommand.event.requestId === io.eventCommand.step.requestId &&
      io.eventCommand.event.layer === io.eventCommand.step.layer &&
      io.eventCommand.event.mode === io.eventCommand.step.mode &&
      io.eventCommand.event.tokenBase === io.eventCommand.step.tokenBase &&
      io.eventCommand.event.positionBase ===
        io.eventCommand.step.positionBase &&
      io.eventCommand.event.activeTokenMask ===
        io.eventCommand.step.activeTokenMask
  val incomingLegal =
    io.eventCommand.executor ===
      BitNetResidentLayerEventExecutor.Attention &&
      io.eventCommand.event.stage ===
        BitNetResidentLayerEventStage.AttentionContextReady &&
      io.eventCommand.step.stage ===
        BitNetResidentLayerStage.ResidentAttention &&
      io.eventCommand.step.unit === BitNetResidentLayerUnit.Attention &&
      io.eventCommand.step.tokenBase ===
        io.eventCommand.step.positionBase && incomingIdentity

  io.eventCommand.ready := state === State.Idle && !responseValid
  when(io.eventCommand.fire) {
    held := io.eventCommand.payload
    when(incomingLegal) {
      state := State.Issue
    } otherwise {
      response.requestId := io.eventCommand.step.requestId
      response.stepId := io.eventCommand.step.stepId
      response.stage := io.eventCommand.step.stage
      response.layer := io.eventCommand.step.layer
      response.success := False
      response.faultCode := B(
        BitNetResidentAttentionEventAdapterFault.InvalidCommand, 8 bits)
      responseValid := True
      stickyFault := True
      state := State.Report
    }
  }

  io.attentionCommand.valid := state === State.Issue
  io.attentionCommand.requestId := held.step.requestId
  io.attentionCommand.layer := held.step.layer
  io.attentionCommand.tokenBase := held.step.tokenBase
  io.attentionCommand.mode := held.step.mode
  io.attentionCommand.activeTokenMask := held.step.activeTokenMask
  for (lane <- 0 until model.tokenParallelism) {
    io.attentionCommand.tokenId(lane) :=
      (held.step.tokenBase.resize(model.sequenceWidth) + lane)
        .resize(model.sequenceWidth)
    io.attentionCommand.queryPosition(lane) :=
      (held.step.positionBase.resize(model.sequenceWidth) + lane)
        .resize(model.sequenceWidth)
    io.attentionCommand.keyCount(lane) := held.step.laneCausalLength(lane)
  }
  when(io.attentionCommand.fire) { state := State.WaitAttention }

  val completionIdentity =
    io.attentionCompletion.requestId === held.step.requestId &&
      io.attentionCompletion.layer === held.step.layer &&
      io.attentionCompletion.tokenBase === held.step.tokenBase &&
      io.attentionCompletion.completedLaneMask ===
        held.step.activeTokenMask
  io.attentionCompletion.ready := state === State.WaitAttention &&
    !responseValid
  when(io.attentionCompletion.fire) {
    response.requestId := held.step.requestId
    response.stepId := held.step.stepId
    response.stage := held.step.stage
    response.layer := held.step.layer
    response.success := completionIdentity &&
      io.attentionCompletion.success
    response.faultCode := 0
    when(!completionIdentity) {
      response.faultCode := B(
        BitNetResidentAttentionEventAdapterFault.CompletionIdentity,
        8 bits)
      stickyFault := True
    } elsewhen(!io.attentionCompletion.success) {
      response.faultCode := Mux(io.attentionCompletion.faultCode.orR,
        io.attentionCompletion.faultCode,
        B(BitNetResidentAttentionEventAdapterFault.CompletionFailed,
          8 bits))
      stickyFault := True
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

  io.busy := state =/= State.Idle || responseValid
  io.protocolFaultSticky := stickyFault
  io.completedEvents := completedCount
}

object GenerateBitNetResidentAttentionEventAdapter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-attention-event-adapter")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentAttentionEventAdapter())
}
