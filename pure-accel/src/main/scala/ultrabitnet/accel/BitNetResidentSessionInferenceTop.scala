package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Static contract of the session-aware, preloaded-hidden inference top.
  *
  * `shell` remains the sole owner of every physical inference resource.  The
  * session controller adds only the persistent round/session policy above that
  * shell and therefore must use the exact same model, image and event ABIs.
  */
case class BitNetResidentSessionInferenceTopConfig(
    shell: BitNetResidentPreloadedHiddenFullLayerShellConfig =
      BitNetResidentPreloadedHiddenFullLayerShellConfig.production(),
    sessionIdWidth: Int = 32) {
  val session: BitNetResidentSessionRoundControllerConfig =
    BitNetResidentSessionRoundControllerConfig(
      shell = shell, sessionIdWidth = sessionIdWidth)

  require(session.shell == shell,
    "the session controller and physical shell must share one exact ABI")
}

/**
  * Production-shaped composition of the persistent session controller and
  * complete preloaded-hidden thirty-layer shell.
  *
  * This wrapper deliberately keeps the token/embedding boundary visible:
  * `preloadInput` must provide the HiddenResident beats for every submitted
  * round.  Tokenization and embedding lookup are not autonomous PL functions
  * in this top.  Once the preload authenticates, the controller launches the
  * physical shell and returns only its authenticated full-round retirement.
  *
  * The controller is instantiated inside the explicit slow clock domain.  The
  * shell receives the same explicit slow clock plus its existing fast TP4
  * clock, so this composition adds no Stream crossing or implicit clock domain.
  * All five AXI banks and the hidden preload port are slow-domain interfaces.
  *
  * This is intentionally non-canonical until a board-facing command/IRQ ABI,
  * autonomous embedding gather and complete numeric/physical validation exist.
  */
class BitNetResidentSessionInferenceTop(
    cfg: BitNetResidentSessionInferenceTopConfig =
      BitNetResidentSessionInferenceTopConfig(),
    useVendorPrimitive: Boolean = true) extends Component {
  private val shellCfg = cfg.shell
  private val sessionCfg = cfg.session
  private val model = shellCfg.model
  private val preloadCfg = sessionCfg.preload
  private val crossbar = shellCfg.crossbar
  private val readCfg = shellCfg.memory.memory.read(0)
  private val writeCfg = shellCfg.memory.memory.write(0)

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val roundSubmit = slave(Stream(
      BitNetResidentSessionRoundDescriptor(sessionCfg)))
    val roundResult = master(Stream(
      BitNetResidentSessionRoundResult(sessionCfg)))

    // External token/embedding boundary.  The matching preload command and
    // completion are internal between the controller and the physical shell.
    val preloadInput = slave(Stream(
      BitNetResidentHiddenPreloadBeat(preloadCfg)))

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
    val roundControllerBusy = out Bool()
    val shellBusy = out Bool()

    val sessionValid = out Bool()
    val sessionId = out UInt(cfg.sessionIdWidth bits)
    val sessionImageEpoch = out UInt(shellCfg.staging.epochWidth bits)
    val sessionPhase = out(BitNetResidentSessionPhase())
    val sessionNextPosition = out UInt(model.sequenceWidth bits)
    val sessionQuarantined = out Bool()
    val sessionProtocolFaultSticky = out Bool()
    val sessionQuarantineFaultCode = out Bits(16 bits)

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
    val shellFatalFailStop = out Bool()
    val shellProtocolFaultSticky = out Bool()
    val shellFaultBits = out Bits(64 bits)
  }

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  // The controller contains all session state and is therefore explicitly
  // clocked by the same slow domain that owns the shell's control plane.
  val controllerArea = new ClockingArea(slowCd) {
    val controller = new BitNetResidentSessionRoundController(sessionCfg)
  }

  // The shell retains its explicit slow/fast clock ports.  No child interface
  // is adapted, buffered or crossed by this composition.
  val shell = new BitNetResidentPreloadedHiddenFullLayerShell(
    shellCfg, useVendorPrimitive = useVendorPrimitive)
  shell.io.slowClk := io.slowClk
  shell.io.slowResetn := io.slowResetn
  shell.io.fastClk := io.fastClk
  shell.io.fastResetn := io.fastResetn

  val controller = controllerArea.controller
  controller.io.roundSubmit << io.roundSubmit
  io.roundResult << controller.io.roundResult

  shell.io.preloadCommand << controller.io.preloadCommand
  controller.io.preloadCompletion << shell.io.preloadCompletion
  shell.io.start << controller.io.start
  controller.io.completion << shell.io.completion
  shell.io.preloadInput << io.preloadInput

  for (bank <- 0 until crossbar.bankCount) {
    io.axiAr(bank) << shell.io.axiAr(bank)
    shell.io.axiR(bank) << io.axiR(bank)
    io.axiAw(bank) << shell.io.axiAw(bank)
    io.axiW(bank) << shell.io.axiW(bank)
    shell.io.axiB(bank) << io.axiB(bank)
  }

  io.roundControllerBusy := controller.io.busy
  io.shellBusy := shell.io.busy
  io.busy := controller.io.busy || shell.io.busy

  io.sessionValid := controller.io.sessionValid
  io.sessionId := controller.io.sessionId
  io.sessionImageEpoch := controller.io.sessionImageEpoch
  io.sessionPhase := controller.io.sessionPhase
  io.sessionNextPosition := controller.io.sessionNextPosition
  io.sessionQuarantined := controller.io.quarantined
  io.sessionProtocolFaultSticky := controller.io.protocolFaultSticky
  io.sessionQuarantineFaultCode := controller.io.quarantineFaultCode

  io.hiddenReady := shell.io.hiddenReady
  io.hiddenRequestId := shell.io.hiddenRequestId
  io.hiddenTokenMask := shell.io.hiddenTokenMask
  io.activationLeaseSource := shell.io.activationLeaseSource
  io.activationOwnerValid := shell.io.activationOwnerValid
  io.activationOwner := shell.io.activationOwner
  io.linearOwner := shell.io.linearOwner

  io.shellFatalFailStop := shell.io.fatalFailStop
  io.shellProtocolFaultSticky := shell.io.protocolFaultSticky
  io.shellFaultBits := shell.io.faultBits
  io.fatalFailStop := shell.io.fatalFailStop || controller.io.quarantined
  io.protocolFaultSticky := shell.io.protocolFaultSticky ||
    controller.io.protocolFaultSticky
}

object GenerateBitNetResidentSessionInferenceTop extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-resident-session-inference-top")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentSessionInferenceTop())
}
