package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Persistent-session phase retained below the PS/PL command boundary. */
object BitNetResidentSessionPhase extends SpinalEnum(binarySequential) {
  val Prefill, Decode = newElement()
}

/**
  * Static ABI of the token/embedding-level resident-session controller.
  *
  * The complete full-layer shell configuration is retained deliberately: this
  * component must not invent a second request, epoch, token, preload or LM
  * identity domain beside the production shell.
  */
case class BitNetResidentSessionRoundControllerConfig(
    shell: BitNetResidentPreloadedHiddenFullLayerShellConfig =
      BitNetResidentPreloadedHiddenFullLayerShellConfig.production(),
    sessionIdWidth: Int = 32) {
  val model: BitNetConfig = shell.model
  val preload: BitNetResidentHiddenPreloadEndpointConfig =
    BitNetResidentHiddenPreloadEndpointConfig(model, shell.layout)
  val terminal: BitNetResidentLmTerminalJoinConfig =
    BitNetResidentLmTerminalJoinConfig(
      schedule = shell.event, lm = shell.lm.resident)

  require(sessionIdWidth >= 16,
    "sessionIdWidth must reject stale host-session descriptors")
  require(shell.event.epochWidth == shell.staging.epochWidth,
    "run and image epoch widths must match the production shell ABI")
  require(model.tokenParallelism == 2 || model.tokenParallelism == 4,
    "the resident outer loop supports the production TP2/TP4 preload ABI")
}

/** One host-authored round descriptor; it contains no PL micro-operations. */
case class BitNetResidentSessionRoundDescriptor(
    cfg: BitNetResidentSessionRoundControllerConfig) extends Bundle {
  val sessionId = UInt(cfg.sessionIdWidth bits)
  val imageEpoch = UInt(cfg.shell.staging.epochWidth bits)
  val mode = BitNetMode()
  val positionBase = UInt(cfg.model.sequenceWidth bits)
  val tokenCount = UInt(cfg.model.roundTokenCountWidth bits)
  val lastPromptRound = Bool()
  val newSession = Bool()
  val psSourceBase = UInt(cfg.model.addressWidth bits)
  val psSourceBytes = UInt((cfg.model.addressWidth + 1) bits)
}

/** Stable host-visible retirement record for exactly one accepted submit. */
case class BitNetResidentSessionRoundResult(
    cfg: BitNetResidentSessionRoundControllerConfig) extends Bundle {
  val sessionId = UInt(cfg.sessionIdWidth bits)
  val roundRequestId = UInt(cfg.model.requestIdWidth bits)
  val runEpoch = UInt(cfg.shell.event.epochWidth bits)
  val success = Bool()
  val faultCode = Bits(16 bits)
  val tokenValid = Bool()
  val tokenId = UInt(cfg.shell.lm.resident.streaming.tokenIdWidth bits)
  val nextPosition = UInt(cfg.model.sequenceWidth bits)
}

/** Stable fault ABI for local rejection and authenticated child failure. */
object BitNetResidentSessionRoundFault {
  val None = 0x0000

  val InvalidTokenCount = 0x1101
  val SequenceOverflow = 0x1102
  val InvalidNewSession = 0x1103
  val SessionUnavailable = 0x1104
  val SessionIdentity = 0x1105
  val PositionGap = 0x1106
  val InvalidPhase = 0x1107

  val PreloadFailed = 0x1201
  val PreloadRetirement = 0x1202
  val FullRoundFailed = 0x1301
  val MissingLmResult = 0x1302
  val FullRoundRetirement = 0x1303

  val UnexpectedPreloadCompletion = 0x1f01
  val PreloadIdentity = 0x1f02
  val UnexpectedFullCompletion = 0x1f03
  val FullCompletionIdentity = 0x1f04
}

/**
  * Strict, single-round outer loop for one persistent KV-cache session.
  *
  * A legal submit is not allowed to reach the full-layer shell until its real
  * hidden preload command has fired and the matching physical preload has
  * retired successfully.  Likewise, a host result cannot appear before the
  * real shell start fires and its complete schedule/LM retirement is
  * authenticated.  Every Stream payload is sourced only from held registers,
  * so arbitrary downstream backpressure cannot mutate an offered command.
  *
  * Locally invalid descriptors are bounded failures and never touch the shell
  * or current session.  Once a real round has started, any authenticated
  * failure invalidates the session because an unknown prefix of KV state may
  * have changed.  A completion with the wrong identity, or any completion in
  * the wrong phase, enters reset-only quarantine; it is never recycled into a
  * later round.
  */
class BitNetResidentSessionRoundController(
    cfg: BitNetResidentSessionRoundControllerConfig =
      BitNetResidentSessionRoundControllerConfig()) extends Component {
  private val model = cfg.model
  private val preloadCfg = cfg.preload
  private val terminalCfg = cfg.terminal

  val io = new Bundle {
    val roundSubmit = slave(Stream(
      BitNetResidentSessionRoundDescriptor(cfg)))

    val preloadCommand = master(Stream(
      BitNetResidentHiddenPreloadCommand(preloadCfg)))
    val preloadCompletion = slave(Stream(
      BitNetResidentHiddenPreloadCompletion(preloadCfg)))

    val start = master(Stream(
      BitNetResidentPreloadedHiddenFullLayerStart(cfg.shell)))
    val completion = slave(Stream(
      BitNetResidentFullRoundCompletion(terminalCfg)))

    val roundResult = master(Stream(
      BitNetResidentSessionRoundResult(cfg)))

    val busy = out Bool()
    val sessionValid = out Bool()
    val sessionId = out UInt(cfg.sessionIdWidth bits)
    val sessionImageEpoch = out UInt(cfg.shell.staging.epochWidth bits)
    val sessionPhase = out(BitNetResidentSessionPhase())
    val sessionNextPosition = out UInt(model.sequenceWidth bits)
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val quarantineFaultCode = out Bits(16 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssuePreload, WaitPreload, IssueStart, WaitCompletion,
        Report, Quarantine = newElement()
  }

  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentSessionRoundDescriptor(cfg)) init
    BitNetResidentSessionRoundDescriptor(cfg).getZero
  val heldMask = Reg(Bits(model.tokenParallelism bits)) init 0
  val activeRequestId = Reg(UInt(model.requestIdWidth bits)) init 0
  val activeRunEpoch = Reg(UInt(cfg.shell.event.epochWidth bits)) init 0

  // Zero remains the power-on/no-work identity.  Each accepted submit,
  // including a locally rejected one, consumes a unique modulo-width pair.
  val nextRequestId = Reg(UInt(model.requestIdWidth bits)) init 1
  val nextRunEpoch = Reg(UInt(cfg.shell.event.epochWidth bits)) init 1

  val residentSessionValid = RegInit(False)
  val residentSessionId = Reg(UInt(cfg.sessionIdWidth bits)) init 0
  val residentImageEpoch =
    Reg(UInt(cfg.shell.staging.epochWidth bits)) init 0
  val residentPhase = Reg(BitNetResidentSessionPhase()) init
    BitNetResidentSessionPhase.Prefill
  val residentNextPosition = Reg(UInt(model.sequenceWidth bits)) init 0

  val resultValid = RegInit(False)
  val result = Reg(BitNetResidentSessionRoundResult(cfg)) init
    BitNetResidentSessionRoundResult(cfg).getZero
  val stickyProtocolFault = RegInit(False)
  val quarantineCode = Reg(Bits(16 bits)) init 0

  // A Stream producer is allowed to deassert VALID only after observing the
  // accepting edge.  Give that already-retired payload one complete cycle to
  // disappear; a VALID which remains beyond the grace cycle is new/unexpected
  // traffic and is quarantined below.
  val preloadRetireGrace = RegInit(False)
  val completionRetireGrace = RegInit(False)

  val submitMask = Bits(model.tokenParallelism bits)
  for (lane <- 0 until model.tokenParallelism) {
    submitMask(lane) := io.roundSubmit.payload.tokenCount > lane
  }
  val heldFinalLane = (held.tokenCount - 1)
    .resize(model.roundTokenCountWidth)
  val heldFinalPosition =
    (held.positionBase + heldFinalLane.resize(model.sequenceWidth))
      .resize(model.sequenceWidth)
  val heldNextPosition =
    (held.positionBase + held.tokenCount.resize(model.sequenceWidth))
      .resize(model.sequenceWidth)
  val requiresLm = held.mode === BitNetMode.Decode ||
    held.lastPromptRound

  val submitPositionEnd = io.roundSubmit.payload.positionBase
    .resize(model.sequenceWidth + 1) +
    io.roundSubmit.payload.tokenCount.resize(model.sequenceWidth + 1)
  val tokenCountLegal = io.roundSubmit.payload.tokenCount =/= 0 &&
    io.roundSubmit.payload.tokenCount <= model.tokenParallelism
  val positionLegal = submitPositionEnd <= model.maxSequenceLength
  val newSessionLegal =
    io.roundSubmit.payload.mode === BitNetMode.Prefill &&
      io.roundSubmit.payload.positionBase === 0
  val sessionIdentityLegal = residentSessionValid &&
    io.roundSubmit.payload.sessionId === residentSessionId &&
    io.roundSubmit.payload.imageEpoch === residentImageEpoch
  val continuationPositionLegal =
    io.roundSubmit.payload.positionBase === residentNextPosition
  val prefillContinuationLegal =
    residentPhase === BitNetResidentSessionPhase.Prefill &&
      io.roundSubmit.payload.mode === BitNetMode.Prefill
  val decodeContinuationLegal =
    residentPhase === BitNetResidentSessionPhase.Decode &&
      io.roundSubmit.payload.mode === BitNetMode.Decode &&
      io.roundSubmit.payload.tokenCount === 1 &&
      !io.roundSubmit.payload.lastPromptRound
  val continuationPhaseLegal = prefillContinuationLegal ||
    decodeContinuationLegal

  val submitFault = Bits(16 bits)
  submitFault := B(BitNetResidentSessionRoundFault.None, 16 bits)
  when(!tokenCountLegal) {
    submitFault := B(
      BitNetResidentSessionRoundFault.InvalidTokenCount, 16 bits)
  } elsewhen(!positionLegal) {
    submitFault := B(
      BitNetResidentSessionRoundFault.SequenceOverflow, 16 bits)
  } elsewhen(io.roundSubmit.payload.newSession && !newSessionLegal) {
    submitFault := B(
      BitNetResidentSessionRoundFault.InvalidNewSession, 16 bits)
  } elsewhen(!io.roundSubmit.payload.newSession &&
      !residentSessionValid) {
    submitFault := B(
      BitNetResidentSessionRoundFault.SessionUnavailable, 16 bits)
  } elsewhen(!io.roundSubmit.payload.newSession &&
      !sessionIdentityLegal) {
    submitFault := B(
      BitNetResidentSessionRoundFault.SessionIdentity, 16 bits)
  } elsewhen(!io.roundSubmit.payload.newSession &&
      !continuationPositionLegal) {
    submitFault := B(
      BitNetResidentSessionRoundFault.PositionGap, 16 bits)
  } elsewhen(!io.roundSubmit.payload.newSession &&
      !continuationPhaseLegal) {
    submitFault := B(
      BitNetResidentSessionRoundFault.InvalidPhase, 16 bits)
  }
  val submitLegal = submitFault ===
    B(BitNetResidentSessionRoundFault.None, 16 bits)

  val stalePreloadCompletion = io.preloadCompletion.valid &&
    state =/= State.WaitPreload && !preloadRetireGrace
  val staleFullCompletion = io.completion.valid &&
    state =/= State.WaitCompletion && !completionRetireGrace
  val staleChildTraffic = stalePreloadCompletion || staleFullCompletion

  io.roundSubmit.ready := state === State.Idle && !staleChildTraffic

  io.preloadCommand.valid := state === State.IssuePreload
  io.preloadCommand.payload.requestId := activeRequestId
  io.preloadCommand.payload.tokenMask := heldMask
  io.preloadCommand.payload.featureCount := model.hiddenSize
  io.preloadCompletion.ready := state === State.WaitPreload

  io.start.valid := state === State.IssueStart
  io.start.payload.run.requestId := activeRequestId
  io.start.payload.run.epoch := activeRunEpoch
  io.start.payload.run.mode := held.mode
  io.start.payload.run.tokenBase := held.positionBase
  io.start.payload.run.tokenCount := held.tokenCount
  io.start.payload.run.positionBase := held.positionBase
  io.start.payload.run.activeTokenMask := heldMask
  io.start.payload.run.resetKv := held.newSession
  io.start.payload.run.lastPromptRound := held.lastPromptRound
  io.start.payload.imageEpoch := held.imageEpoch
  io.start.payload.psSourceBase := held.psSourceBase
  io.start.payload.psSourceBytes := held.psSourceBytes
  io.completion.ready := state === State.WaitCompletion

  io.roundResult.valid := resultValid && state === State.Report
  io.roundResult.payload := result

  io.busy := state =/= State.Idle
  io.sessionValid := residentSessionValid
  io.sessionId := residentSessionId
  io.sessionImageEpoch := residentImageEpoch
  io.sessionPhase := residentPhase
  io.sessionNextPosition := residentNextPosition
  io.quarantined := state === State.Quarantine
  io.protocolFaultSticky := stickyProtocolFault
  io.quarantineFaultCode := quarantineCode

  def prepareResultIdentity(
      sessionId: UInt, requestId: UInt, epoch: UInt): Unit = {
    result.sessionId := sessionId
    result.roundRequestId := requestId
    result.runEpoch := epoch
  }

  def enterQuarantine(code: Int): Unit = {
    stickyProtocolFault := True
    quarantineCode := B(code, 16 bits)
    resultValid := False
    residentSessionValid := False
    state := State.Quarantine
  }

  when(io.roundSubmit.fire) {
    nextRequestId := nextRequestId + 1
    nextRunEpoch := nextRunEpoch + 1
    when(submitLegal) {
      held := io.roundSubmit.payload
      heldMask := submitMask
      activeRequestId := nextRequestId
      activeRunEpoch := nextRunEpoch
      state := State.IssuePreload
    } otherwise {
      prepareResultIdentity(io.roundSubmit.payload.sessionId,
        nextRequestId, nextRunEpoch)
      result.success := False
      result.faultCode := submitFault
      result.tokenValid := False
      result.tokenId := 0
      result.nextPosition := Mux(residentSessionValid &&
        io.roundSubmit.payload.sessionId === residentSessionId,
        residentNextPosition, U(0, model.sequenceWidth bits))
      resultValid := True
      state := State.Report
    }
  }

  when(io.preloadCommand.fire) {
    state := State.WaitPreload
  }

  val preloadIdentityMatches =
    io.preloadCompletion.payload.requestId === activeRequestId &&
      io.preloadCompletion.payload.tokenMask === heldMask &&
      io.preloadCompletion.payload.featureCount === model.hiddenSize
  val preloadRetiredSuccessfully =
    io.preloadCompletion.payload.success &&
      io.preloadCompletion.payload.groupsCommitted === preloadCfg.groupCount &&
      !io.preloadCompletion.payload.faultCode.orR

  when(io.preloadCompletion.fire) {
    preloadRetireGrace := True
    when(!preloadIdentityMatches) {
      enterQuarantine(BitNetResidentSessionRoundFault.PreloadIdentity)
    } elsewhen(!preloadRetiredSuccessfully) {
      prepareResultIdentity(held.sessionId, activeRequestId, activeRunEpoch)
      result.success := False
      result.faultCode := Mux(io.preloadCompletion.payload.faultCode.orR,
        io.preloadCompletion.payload.faultCode.resize(16),
        B(BitNetResidentSessionRoundFault.PreloadFailed, 16 bits))
      result.tokenValid := False
      result.tokenId := 0
      result.nextPosition := 0
      resultValid := True
      residentSessionValid := False
      state := State.Report
    } otherwise {
      state := State.IssueStart
    }
  }
  when(preloadRetireGrace && !io.preloadCompletion.fire) {
    preloadRetireGrace := False
  }

  when(io.start.fire) {
    state := State.WaitCompletion
  }

  val scheduleIdentityMatches =
    io.completion.payload.schedule.requestId === activeRequestId &&
      io.completion.payload.schedule.epoch === activeRunEpoch &&
      io.completion.payload.schedule.mode === held.mode
  val handoffIdentityMatches = !io.completion.payload.handoffValid ||
    (io.completion.payload.handoff.requestId === activeRequestId &&
      io.completion.payload.handoff.mode === held.mode &&
      io.completion.payload.handoff.finalLayer === model.layerCount - 1 &&
      io.completion.payload.handoff.tokenBase === held.positionBase &&
      io.completion.payload.handoff.tokenLane === heldFinalLane &&
      io.completion.payload.handoff.tokenPosition === heldFinalPosition &&
      io.completion.payload.handoff.activationBuffer ===
        BitNetPhysicalActivationBuffer.NarrowWork)
  val lmIdentityMatches = !io.completion.payload.lmValid ||
    (io.completion.payload.lm.requestId === activeRequestId &&
      io.completion.payload.lm.mode === held.mode &&
      io.completion.payload.lm.tokenPosition === heldFinalPosition)
  val fullIdentityMatches = scheduleIdentityMatches &&
    handoffIdentityMatches && lmIdentityMatches

  val scheduleRetiredSuccessfully =
    io.completion.payload.schedule.status ===
      BitNetResidentLayerEventRunStatus.Ok &&
      io.completion.payload.schedule.completedLayers === model.layerCount
  val requiredLmRetiredSuccessfully =
    io.completion.payload.handoffValid &&
      io.completion.payload.schedule.lmHeadHandedOff &&
      io.completion.payload.lmValid &&
      io.completion.payload.lm.success &&
      !io.completion.payload.lm.fatal
  val continuingPrefillRetiredSuccessfully =
    !io.completion.payload.handoffValid &&
      !io.completion.payload.schedule.lmHeadHandedOff &&
      !io.completion.payload.lmValid
  val terminalShapeSuccessfullyRetired =
    Mux(requiresLm, requiredLmRetiredSuccessfully,
      continuingPrefillRetiredSuccessfully)
  val fullRetiredSuccessfully = io.completion.payload.success &&
    !io.completion.payload.faultCode.orR &&
    scheduleRetiredSuccessfully && terminalShapeSuccessfullyRetired

  when(io.completion.fire) {
    completionRetireGrace := True
    when(!fullIdentityMatches) {
      enterQuarantine(
        BitNetResidentSessionRoundFault.FullCompletionIdentity)
    } elsewhen(!fullRetiredSuccessfully) {
      prepareResultIdentity(held.sessionId, activeRequestId, activeRunEpoch)
      result.success := False
      when(io.completion.payload.faultCode.orR) {
        result.faultCode := io.completion.payload.faultCode
      } elsewhen(requiresLm && !requiredLmRetiredSuccessfully) {
        result.faultCode := B(
          BitNetResidentSessionRoundFault.MissingLmResult, 16 bits)
      } otherwise {
        result.faultCode := B(
          BitNetResidentSessionRoundFault.FullRoundFailed, 16 bits)
      }
      result.tokenValid := False
      result.tokenId := 0
      result.nextPosition := 0
      resultValid := True
      residentSessionValid := False
      state := State.Report
    } otherwise {
      prepareResultIdentity(held.sessionId, activeRequestId, activeRunEpoch)
      result.success := True
      result.faultCode := 0
      result.nextPosition := heldNextPosition
      result.tokenValid := requiresLm
      result.tokenId := Mux(requiresLm, io.completion.payload.lm.tokenId,
        U(0, cfg.shell.lm.resident.streaming.tokenIdWidth bits))
      resultValid := True

      residentSessionValid := True
      residentSessionId := held.sessionId
      residentImageEpoch := held.imageEpoch
      residentNextPosition := heldNextPosition
      when(held.mode === BitNetMode.Prefill && held.lastPromptRound) {
        residentPhase := BitNetResidentSessionPhase.Decode
      } elsewhen(held.mode === BitNetMode.Decode) {
        residentPhase := BitNetResidentSessionPhase.Decode
      } otherwise {
        residentPhase := BitNetResidentSessionPhase.Prefill
      }
      state := State.Report
    }
  }
  when(completionRetireGrace && !io.completion.fire) {
    completionRetireGrace := False
  }

  when(io.roundResult.fire) {
    resultValid := False
    state := State.Idle
  }

  // Wrong-phase traffic has priority over every nominal transition above.
  // Its producer remains backpressured and the identity can never be reused.
  when(state =/= State.Quarantine && stalePreloadCompletion) {
    enterQuarantine(
      BitNetResidentSessionRoundFault.UnexpectedPreloadCompletion)
  }
  when(state =/= State.Quarantine && staleFullCompletion) {
    enterQuarantine(
      BitNetResidentSessionRoundFault.UnexpectedFullCompletion)
  }
}

object GenerateBitNetResidentSessionRoundController extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-session-round-controller")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentSessionRoundController())
}
