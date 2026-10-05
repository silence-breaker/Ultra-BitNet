package ultrabitnet.accel

import spinal.core._

/**
  * Registers the system-idle proof used to admit one hidden-state preload.
  *
  * `admissionSafe` may be a wide whole-system combinational predicate.  It is
  * consulted only while reserving/admitting a command and is never allowed to
  * modulate the active preload data path.  Once the command is accepted, the
  * registered window remains open until the operation retires or a registered
  * global fail-stop arrives.  Local port collisions are intentionally not an
  * input here: the parent mux detects and blocks those in the offending cycle.
  */
class BitNetResidentBootstrapWindowGuard extends Component {
  val io = new Bundle {
    val admissionSafe = in Bool()
    val commandPresented = in Bool()
    val commandAccepted = in Bool()
    val operationBusy = in Bool()
    val operationRetired = in Bool()
    val registeredGlobalKill = in Bool()

    /** Gate for the external command Stream.  Reservation deliberately costs
      * one cycle, providing the register cut from the whole-system proof. */
    val commandPermit = out Bool()
    /** Stable lifetime proof consumed by the preload endpoint data path. */
    val windowOpen = out Bool()
    val reserved = out Bool()
  }

  val reserved = RegInit(False)

  // Before a command is accepted, keep the reservation only while the source
  // obeys Stream persistence and the current admission proof remains true.
  // Once busy, admissionSafe is deliberately absent from the lifetime cone.
  when(!io.operationBusy && !io.commandAccepted) {
    reserved := io.admissionSafe && io.commandPresented &&
      !io.registeredGlobalKill
  }
  when(io.commandAccepted) {
    reserved := True
  }
  when(io.operationRetired || io.registeredGlobalKill) {
    reserved := False
  }

  io.commandPermit := reserved && io.admissionSafe &&
    !io.registeredGlobalKill
  io.windowOpen := reserved && !io.registeredGlobalKill
  io.reserved := reserved
}

object GenerateBitNetResidentBootstrapWindowGuard extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-bootstrap-window-guard")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentBootstrapWindowGuard)
}
