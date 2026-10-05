package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentLmTerminalJoinFault {
  val ScheduleFailed = 0xf8
  val MissingHandoff = 0xf9
  val ScheduleHandoffIdentity = 0xfa
  val LmIdentity = 0xfb
  val LmFailed = 0xfc
  val UnexpectedLmCompletion = 0xfd
}

case class BitNetResidentLmTerminalJoinConfig(
    schedule: BitNetResidentLayerEventSchedulerConfig =
      BitNetResidentLayerEventSchedulerConfig(epochWidth = 8),
    lm: BitNetResidentStreamingLmHeadConfig =
      BitNetResidentStreamingLmHeadConfig()
) {
  val model: BitNetConfig = schedule.model
  require(lm.model == model)
}

/** The first truthful completion of one complete resident transformer round. */
case class BitNetResidentFullRoundCompletion(
    cfg: BitNetResidentLmTerminalJoinConfig) extends Bundle {
  val schedule = BitNetResidentLayerEventRunCompletion(cfg.schedule)
  val handoffValid = Bool()
  val handoff = BitNetResidentLmHeadHandoff(cfg.model)
  val lmValid = Bool()
  val lm = BitNetResidentLmHeadCompletion(cfg.lm)
  val success = Bool()
  val faultCode = Bits(16 bits)
}

/**
  * Terminal ownership join for final RMS -> streaming LM head -> argmax.
  *
  * The layer scheduler is allowed to finish only its schedule after the LM
  * handoff has been accepted.  This component forwards and records that
  * handoff, independently captures the scheduler completion and the real
  * full-vocabulary LM completion, authenticates all three identities, and
  * emits one full-round completion.  A continuing prefill round has no LM
  * handoff and can retire from its successful schedule completion alone.
  */
class BitNetResidentLmTerminalJoin(
    cfg: BitNetResidentLmTerminalJoinConfig =
      BitNetResidentLmTerminalJoinConfig()) extends Component {
  private val model = cfg.model

  val io = new Bundle {
    val schedulerHandoff = slave(Stream(
      BitNetResidentLmHeadHandoff(model)))
    val lmHandoff = master(Stream(BitNetResidentLmHeadHandoff(model)))
    val schedulerCompletion = slave(Stream(
      BitNetResidentLayerEventRunCompletion(cfg.schedule)))
    val lmCompletion = slave(Stream(
      BitNetResidentLmHeadCompletion(cfg.lm)))

    val completion = master(Stream(
      BitNetResidentFullRoundCompletion(cfg)))
    /** A top-level start gate: false while an LM result is still pending. */
    val allowNextRound = out Bool()
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val completedRounds = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, WaitTerminal, Evaluate, Report, Quarantine = newElement()
  }
  val state = Reg(State()) init State.Idle
  val handoff = Reg(BitNetResidentLmHeadHandoff(model)) init
    BitNetResidentLmHeadHandoff(model).getZero
  val schedule = Reg(
    BitNetResidentLayerEventRunCompletion(cfg.schedule)) init
    BitNetResidentLayerEventRunCompletion(cfg.schedule).getZero
  val lm = Reg(BitNetResidentLmHeadCompletion(cfg.lm)) init
    BitNetResidentLmHeadCompletion(cfg.lm).getZero
  val handoffCaptured = RegInit(False)
  val scheduleCaptured = RegInit(False)
  val lmCaptured = RegInit(False)
  val result = Reg(BitNetResidentFullRoundCompletion(cfg)) init
    BitNetResidentFullRoundCompletion(cfg).getZero
  val resultValid = RegInit(False)
  val stickyFault = RegInit(False)
  val completedCount = Reg(UInt(32 bits)) init 0

  // The scheduler cannot leave IssueLmHead until the real LM consumer has
  // accepted this exact payload, so forwarding and capture are atomic.
  val idleStaleLm = state === State.Idle && io.lmCompletion.valid
  io.lmHandoff.valid := state === State.Idle && !idleStaleLm &&
    io.schedulerHandoff.valid
  io.lmHandoff.payload := io.schedulerHandoff.payload
  io.schedulerHandoff.ready := state === State.Idle && !idleStaleLm &&
    io.lmHandoff.ready
  when(io.schedulerHandoff.fire) {
    handoff := io.schedulerHandoff.payload
    handoffCaptured := True
    scheduleCaptured := False
    lmCaptured := False
    state := State.WaitTerminal
  }

  io.schedulerCompletion.ready :=
    (state === State.Idle || state === State.WaitTerminal) &&
      !scheduleCaptured && !idleStaleLm
  when(io.schedulerCompletion.fire) {
    schedule := io.schedulerCompletion.payload
    scheduleCaptured := True
    when(state === State.Idle) {
      // No handoff preceded this completion.  This is legal only for a
      // continuing prefill or an early failed schedule.  A scheduler may
      // present its completion in the same cycle in which the downstream LM
      // consumer accepts the terminal handoff; retain that real handshake
      // and wait for the LM completion instead of evaluating one cycle too
      // early.
      when(io.schedulerHandoff.fire) {
        state := State.WaitTerminal
      } otherwise {
        handoffCaptured := False
        lmCaptured := False
        state := State.Evaluate
      }
    }
  }

  io.lmCompletion.ready := state === State.WaitTerminal &&
    handoffCaptured && !lmCaptured
  when(io.lmCompletion.fire) {
    lm := io.lmCompletion.payload
    lmCaptured := True
  }

  val scheduleAfter = scheduleCaptured || io.schedulerCompletion.fire
  val lmAfter = lmCaptured || io.lmCompletion.fire
  when(state === State.WaitTerminal && scheduleAfter && lmAfter) {
    state := State.Evaluate
  }

  // LM output without a captured handoff is never associated with a future
  // round; leaving ready low makes the failure visible at its producer.
  when(io.lmCompletion.valid &&
      !(state === State.WaitTerminal && handoffCaptured && !lmCaptured)) {
    stickyFault := True
  }
  when(idleStaleLm) {
    state := State.Quarantine
  }

  val scheduleOk = schedule.status ===
    BitNetResidentLayerEventRunStatus.Ok
  val handoffIdentity = handoff.requestId === schedule.requestId &&
    handoff.mode === schedule.mode &&
    handoff.finalLayer === model.layerCount - 1 &&
    handoff.activationBuffer === BitNetPhysicalActivationBuffer.NarrowWork
  val lmIdentity = lm.requestId === handoff.requestId &&
    lm.mode === handoff.mode &&
    lm.tokenPosition === handoff.tokenPosition
  // Decode always produces a token.  The scheduler's handed-off flag remains
  // necessary for a last prompt round, while this mode check prevents a
  // malformed successful Decode completion from retiring as if it were a
  // continuing prefill round.
  val requiresLm = schedule.mode === BitNetMode.Decode ||
    schedule.lmHeadHandedOff

  when(state === State.Evaluate) {
    result.schedule := schedule
    result.handoffValid := handoffCaptured
    result.handoff := handoff
    result.lmValid := handoffCaptured && lmCaptured
    result.lm := lm
    result.success := False
    result.faultCode := 0

    when(!scheduleOk) {
      result.faultCode := schedule.faultCode.resize(16)
      when(!schedule.faultCode.orR) {
        result.faultCode := B(
          BitNetResidentLmTerminalJoinFault.ScheduleFailed, 16 bits)
      }
    } elsewhen(handoffCaptured =/= requiresLm) {
      result.faultCode := B(
        BitNetResidentLmTerminalJoinFault.MissingHandoff, 16 bits)
      stickyFault := True
    } elsewhen(requiresLm && !handoffIdentity) {
      result.faultCode := B(
        BitNetResidentLmTerminalJoinFault.ScheduleHandoffIdentity,
        16 bits)
      stickyFault := True
    } elsewhen(requiresLm && (!lmCaptured || !lmIdentity)) {
      result.faultCode := B(
        BitNetResidentLmTerminalJoinFault.LmIdentity, 16 bits)
      stickyFault := True
    } elsewhen(requiresLm && (!lm.success || lm.fatal)) {
      result.faultCode := Mux(lm.faultCode.orR, lm.faultCode,
        B(BitNetResidentLmTerminalJoinFault.LmFailed, 16 bits))
      stickyFault := True
    } otherwise {
      result.success := True
    }
    resultValid := True
    state := State.Report
  }

  io.completion.valid := resultValid && state === State.Report
  io.completion.payload := result
  when(io.completion.fire) {
    resultValid := False
    handoffCaptured := False
    scheduleCaptured := False
    lmCaptured := False
    completedCount := completedCount + 1
    state := State.Idle
  }

  io.allowNextRound := state === State.Idle && !resultValid &&
    !io.lmCompletion.valid
  io.busy := state =/= State.Idle || resultValid
  io.protocolFaultSticky := stickyFault
  io.completedRounds := completedCount
}

object GenerateBitNetResidentLmTerminalJoin extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-lm-terminal-join")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLmTerminalJoin())
}
