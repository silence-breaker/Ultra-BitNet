package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentLayerEventExecutionAdapterFault {
  val LoweringRejected = 0xdc
  val CompletionIdentity = 0xdd
  val CompletionFailed = 0xde
  val UnexpectedCompletion = 0xdf
}

/**
  * Data-commit barrier between the fixed resident event schedule and the
  * physical executors.
  *
  * The adapter owns no arithmetic or resource scheduler.  It lowers one event
  * to its typed executor command, locks that complete event identity until the
  * physical LayerStep completion returns, and only then publishes the matching
  * data commit.  Weight replay and the TP4 lease retire inside MatrixStage;
  * this boundary intentionally has no synthetic prefetch/retirement signals.
  */
class BitNetResidentLayerEventExecutionAdapter(
    cfg: BitNetResidentLayerEventCommandLoweringConfig =
      BitNetResidentLayerEventCommandLoweringConfig()) extends Component {
  private val eventCfg = cfg.event
  private val model = cfg.model

  val io = new Bundle {
    val eventCommand = slave(Stream(
      BitNetResidentLayerEventCommand(eventCfg)))
    val execution = master(Stream(
      BitNetResidentLayerEventExecutorCommand(eventCfg)))
    val executionCompletion = slave(Stream(
      BitNetResidentLayerStepCompletion(model)))
    val dataCommit = master(Stream(BitNetResidentLayerEvent(eventCfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val completedEvents = out UInt(32 bits)
  }

  val lowering = new BitNetResidentLayerEventCommandLowering(cfg)
  lowering.io.eventCommand << io.eventCommand

  object State extends SpinalEnum(binarySequential) {
    val Idle, WaitCompletion, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentLayerEventExecutorCommand(eventCfg)) init
    BitNetResidentLayerEventExecutorCommand(eventCfg).getZero
  val report = Reg(BitNetResidentLayerEvent(eventCfg)) init
    BitNetResidentLayerEvent(eventCfg).getZero
  val reportValid = RegInit(False)
  val stickyFault = RegInit(False)
  val completedCount = Reg(UInt(32 bits)) init 0

  // A completion presented without an outstanding executor transaction is
  // stale.  Keep it quarantined at this boundary and do not allow a following
  // event to make that stale payload look current merely by changing state to
  // WaitCompletion on the next edge.
  val staleCompletionAtIdle = state === State.Idle &&
    io.executionCompletion.valid
  io.execution.valid := state === State.Idle &&
    !staleCompletionAtIdle && lowering.io.executorCommand.valid
  io.execution.payload := lowering.io.executorCommand.payload
  lowering.io.executorCommand.ready := state === State.Idle &&
    !staleCompletionAtIdle && io.execution.ready
  lowering.io.failure.ready := state === State.Idle && !reportValid &&
    !staleCompletionAtIdle

  def copyEventIdentity(
      target: BitNetResidentLayerEvent,
      source: BitNetResidentLayerEventCommand): Unit = {
    target.requestId := source.requestId
    target.layer := source.layer
    target.epoch := source.epoch
    target.stage := source.stage
    target.slot := source.slot
  }

  when(io.execution.fire) {
    held := io.execution.payload
    state := State.WaitCompletion
  }

  when(lowering.io.failure.fire) {
    copyEventIdentity(report, lowering.io.failure.event)
    report.success := False
    report.faultCode := B(
      BitNetResidentLayerEventExecutionAdapterFault.LoweringRejected,
      8 bits)
    reportValid := True
    stickyFault := True
    state := State.Report
  }

  val completionIdentity =
    io.executionCompletion.requestId === held.step.requestId &&
      io.executionCompletion.stepId === held.step.stepId &&
      io.executionCompletion.payload.stage === held.step.stage &&
      io.executionCompletion.layer === held.step.layer
  io.executionCompletion.ready := state === State.WaitCompletion &&
    !reportValid

  when(io.executionCompletion.fire) {
    copyEventIdentity(report, held.event)
    report.success := completionIdentity &&
      io.executionCompletion.success
    report.faultCode := 0
    when(!completionIdentity) {
      report.faultCode := B(
        BitNetResidentLayerEventExecutionAdapterFault.CompletionIdentity,
        8 bits)
      stickyFault := True
    } elsewhen(!io.executionCompletion.success) {
      report.faultCode := Mux(io.executionCompletion.faultCode.orR,
        io.executionCompletion.faultCode,
        B(BitNetResidentLayerEventExecutionAdapterFault.CompletionFailed,
          8 bits))
      stickyFault := True
    }
    reportValid := True
    state := State.Report
  }

  // Unexpected/stale completions are never consumed as the following event.
  when(io.executionCompletion.valid && state =/=
      State.WaitCompletion) {
    stickyFault := True
  }

  io.dataCommit.valid := reportValid
  io.dataCommit.payload := report
  when(io.dataCommit.fire) {
    reportValid := False
    completedCount := completedCount + 1
    state := State.Idle
  }

  io.busy := state =/= State.Idle || reportValid
  io.protocolFaultSticky := stickyFault
  io.completedEvents := completedCount
}

object GenerateBitNetResidentLayerEventExecutionAdapter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-layer-event-execution-adapter")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLayerEventExecutionAdapter())
}
