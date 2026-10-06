package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentMatrixRopeEventBarrierFault {
  val None = 0x00
  val UnexpectedMatrixCompletion = 0xb8
  val UnexpectedRopeCompletion = 0xb9
  val MatrixCompletionIdentity = 0xba
  val RopeCompletionIdentity = 0xbb
  val RopeCompletionFailed = 0xbc
  val MatrixFailedWithRopeOutstanding = 0xbd
}

/** Static type contract for the command fork and terminal join. */
case class BitNetResidentMatrixRopeEventBarrierConfig(
    matrix: BitNetResidentMatrixEventCoordinatorConfig =
      BitNetResidentMatrixEventCoordinatorConfig(),
    rope: BitNetResidentRopeCoefficientStagerConfig =
      BitNetResidentRopeCoefficientStagerConfig()) {
  val event: BitNetResidentLayerEventSchedulerConfig = matrix.event
  val model: BitNetConfig = event.model

  require(model == BitNetConfig.Production)
  require(rope.event == event,
    "matrix and RoPE transactions must use one event identity geometry")
  require(rope.model == model)
  require(rope.stage == matrix.prepared.stage,
    "RoPE must authenticate projections from this matrix stage")
}

/**
  * Atomic resident-QKV command fork and real matrix/RoPE retirement barrier.
  *
  * A QKV command is presented to MatrixEvent and the coefficient stager on the
  * same clock edge.  Their work can then overlap: scale/activation preparation
  * and coefficient-row fetch are independent until the projection seam.  The
  * activation lease above this block is released only after both children
  * return matching, successful terminal identities.
  *
  * A failed matrix completion is a bounded child failure.  It may be reported
  * normally only after the RoPE transaction has already retired successfully
  * (including the same cycle).  Releasing the activation owner while a partial
  * coefficient transaction can still reach Q/K/V would be unsafe, so every
  * stale identity, RoPE failure, or early matrix failure enters reset-only
  * quarantine and suppresses the public completion permanently.
  *
  * Non-QKV matrix events bypass the RoPE fork and retain the normal bounded
  * matrix failure contract.  This component contains control state only.
  */
class BitNetResidentMatrixRopeEventBarrier(
    cfg: BitNetResidentMatrixRopeEventBarrierConfig =
      BitNetResidentMatrixRopeEventBarrierConfig()) extends Component {
  private val model = cfg.model

  val io = new Bundle {
    val command = slave(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg.event)))

    val matrixCommand = master(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg.event)))
    val ropeRound = master(Stream(
      BitNetResidentLayerEventExecutorCommand(cfg.event)))

    val matrixCompletion = slave(Stream(
      BitNetResidentLayerStepCompletion(model)))
    val ropeCompletion = slave(Stream(
      BitNetResidentRopeCoefficientCompletion(cfg.rope)))

    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))

    val busy = out Bool()
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val faultCode = out Bits(8 bits)
    val acceptedEvents = out UInt(32 bits)
    val completedEvents = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Dispatch, WaitChildren, Report, Quarantine = newElement()
  }

  val state = Reg(State()) init State.Idle
  val held = Reg(
    BitNetResidentLayerEventExecutorCommand(cfg.event)) init
      BitNetResidentLayerEventExecutorCommand(cfg.event).getZero
  val heldQkv = RegInit(False)
  val matrixDone = RegInit(False)
  val ropeDone = RegInit(False)
  val response = Reg(BitNetResidentLayerStepCompletion(model)) init
    BitNetResidentLayerStepCompletion(model).getZero
  val stickyFault = RegInit(False)
  val stickyFaultCode = Reg(Bits(8 bits)) init 0
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0

  val anyCompletionValid = io.matrixCompletion.valid ||
    io.ropeCompletion.valid

  // Use all four routing discriminators.  A malformed hybrid command remains
  // a bounded MatrixEvent rejection and never starts an unnecessary RoPE job.
  val incomingQkv =
    io.command.event.stage ===
      BitNetResidentLayerEventStage.QkvResidentReady &&
      io.command.step.stage === BitNetResidentLayerStage.QkvProjection &&
      io.command.step.unit === BitNetResidentLayerUnit.MatrixArray &&
      io.command.executor ===
        BitNetResidentLayerEventExecutor.MatrixArray

  io.command.ready := state === State.Idle && !anyCompletionValid
  when(io.command.fire) {
    held := io.command.payload
    heldQkv := incomingQkv
    matrixDone := False
    ropeDone := False
    acceptedCount := acceptedCount + 1
    state := State.Dispatch
  }

  // An atomic fork: for QKV neither child can fire unless the other child is
  // ready on that same edge.  Completion traffic gates dispatch as well, so a
  // stale terminal token cannot race a newly accepted child transaction.
  val dispatchClean = state === State.Dispatch && !anyCompletionValid
  io.matrixCommand.valid := dispatchClean &&
    (!heldQkv || io.ropeRound.ready)
  io.matrixCommand.payload := held
  io.ropeRound.valid := dispatchClean && heldQkv &&
    io.matrixCommand.ready
  io.ropeRound.payload := held

  val matrixDispatchFire = io.matrixCommand.fire
  val ropeDispatchFire = io.ropeRound.fire
  when(state === State.Dispatch) {
    when(heldQkv) {
      when(matrixDispatchFire && ropeDispatchFire) {
        state := State.WaitChildren
      }
    } otherwise {
      when(matrixDispatchFire) { state := State.WaitChildren }
    }
  }

  io.matrixCompletion.ready := state === State.WaitChildren &&
    !matrixDone
  io.ropeCompletion.ready := state === State.WaitChildren && heldQkv &&
    !ropeDone

  val matrixCompletionFire = io.matrixCompletion.fire
  val ropeCompletionFire = io.ropeCompletion.fire

  val matrixIdentityMatches =
    io.matrixCompletion.requestId === held.step.requestId &&
      io.matrixCompletion.stepId === held.step.stepId &&
      io.matrixCompletion.payload.stage === held.step.stage &&
      io.matrixCompletion.layer === held.step.layer

  val ropeIdentityMatches =
    io.ropeCompletion.requestId === held.step.requestId &&
      io.ropeCompletion.stepId === held.step.stepId &&
      io.ropeCompletion.layer === held.step.layer &&
      io.ropeCompletion.epoch === held.event.epoch &&
      io.ropeCompletion.mode === held.step.mode &&
      io.ropeCompletion.positionBase === held.step.positionBase &&
      io.ropeCompletion.tokenMask === held.step.activeTokenMask

  val ropeTerminalSuccess = ropeIdentityMatches &&
    io.ropeCompletion.success &&
    io.ropeCompletion.emittedCoefficients === cfg.rope.totalCoefficients

  // A second terminal token, a token in Idle/Dispatch/Report, or any RoPE
  // completion attached to a non-QKV event is stale by construction.
  val unexpectedMatrix = io.matrixCompletion.valid &&
    !(state === State.WaitChildren && !matrixDone)
  val unexpectedRope = io.ropeCompletion.valid &&
    !(state === State.WaitChildren && heldQkv && !ropeDone)

  private def fillResponse(success: Bool, childFault: Bits): Unit = {
    response.requestId := held.step.requestId
    response.stepId := held.step.stepId
    response.stage := held.step.stage
    response.layer := held.step.layer
    response.success := success
    response.faultCode := childFault
  }

  private def enterQuarantine(code: Bits): Unit = {
    stickyFault := True
    when(!stickyFault) { stickyFaultCode := code }
    state := State.Quarantine
  }

  when(unexpectedMatrix) {
    enterQuarantine(B(
      BitNetResidentMatrixRopeEventBarrierFault.
        UnexpectedMatrixCompletion, 8 bits))
  } elsewhen(unexpectedRope) {
    enterQuarantine(B(
      BitNetResidentMatrixRopeEventBarrierFault.
        UnexpectedRopeCompletion, 8 bits))
  } elsewhen(state === State.WaitChildren) {
    when(matrixCompletionFire && !matrixIdentityMatches) {
      enterQuarantine(B(
        BitNetResidentMatrixRopeEventBarrierFault.
          MatrixCompletionIdentity, 8 bits))
    } elsewhen(ropeCompletionFire && !ropeIdentityMatches) {
      enterQuarantine(B(
        BitNetResidentMatrixRopeEventBarrierFault.
          RopeCompletionIdentity, 8 bits))
    } elsewhen(ropeCompletionFire && !ropeTerminalSuccess) {
      enterQuarantine(B(
        BitNetResidentMatrixRopeEventBarrierFault.
          RopeCompletionFailed, 8 bits))
    } elsewhen(!heldQkv && matrixCompletionFire) {
      // The child failure is bounded because no coefficient transaction was
      // launched.  Echo only the locked identity, never mutable input fields.
      val reportedFault = Bits(8 bits)
      reportedFault := io.matrixCompletion.faultCode
      when(!io.matrixCompletion.success &&
          !io.matrixCompletion.faultCode.orR) {
        reportedFault := B(
          BitNetResidentMatrixEventFault.PreparedFailed, 8 bits)
      }
      fillResponse(io.matrixCompletion.success, reportedFault)
      state := State.Report
    } elsewhen(heldQkv && matrixCompletionFire &&
        !io.matrixCompletion.success &&
        !(ropeDone || (ropeCompletionFire && ropeTerminalSuccess))) {
      // MatrixEvent can terminate while the coefficient stager still owns a
      // memory/projection transaction.  No normal completion may escape.
      enterQuarantine(B(
        BitNetResidentMatrixRopeEventBarrierFault.
          MatrixFailedWithRopeOutstanding, 8 bits))
    } elsewhen(heldQkv && matrixCompletionFire &&
        !io.matrixCompletion.success) {
      val reportedFault = Bits(8 bits)
      reportedFault := io.matrixCompletion.faultCode
      when(!io.matrixCompletion.faultCode.orR) {
        reportedFault := B(
          BitNetResidentMatrixEventFault.PreparedFailed, 8 bits)
      }
      fillResponse(False, reportedFault)
      state := State.Report
    } elsewhen(heldQkv &&
        ((matrixCompletionFire && io.matrixCompletion.success &&
          (ropeDone || (ropeCompletionFire && ropeTerminalSuccess))) ||
          (ropeCompletionFire && ropeTerminalSuccess && matrixDone))) {
      fillResponse(True, B(0, 8 bits))
      state := State.Report
    } otherwise {
      when(matrixCompletionFire) { matrixDone := True }
      when(ropeCompletionFire) { ropeDone := True }
    }
  }

  // Any stale token blocks the outward valid combinationally on the cycle it
  // is observed.  Thus Report cannot release the activation lease and enter
  // quarantine on the same edge.
  io.completion.valid := state === State.Report && !anyCompletionValid
  io.completion.payload := response
  when(io.completion.fire) {
    completedCount := completedCount + 1
    state := State.Idle
  }

  io.busy := state =/= State.Idle
  io.quarantined := state === State.Quarantine
  io.protocolFaultSticky := stickyFault
  io.faultCode := stickyFaultCode
  io.acceptedEvents := acceptedCount
  io.completedEvents := completedCount
}

object GenerateBitNetResidentMatrixRopeEventBarrier extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-matrix-rope-event-barrier")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentMatrixRopeEventBarrier())
}
