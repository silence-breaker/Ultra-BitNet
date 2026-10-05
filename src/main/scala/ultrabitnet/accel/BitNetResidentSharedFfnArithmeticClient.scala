package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Logical owners multiplexed onto physical arithmetic-pool client 2. */
object BitNetResidentSharedFfnArithmeticOwner
    extends SpinalEnum(binarySequential) {
  val GateScaleComposer, GateUpLeasedGate = newElement()
}

/** Identical logical lease boundary used by both resident FFN clients. */
case class BitNetResidentSharedFfnArithmeticLogicalPort(
    cfg: BitNetSharedSpuArithmeticPoolConfig) extends Bundle {
  val acquire = slave(Stream(BitNetSpuArithmeticLeaseRequest(cfg)))
  val operand = slave(Stream(BitNetSpuArithmeticOperandBeat(cfg)))
  val result = master(Stream(BitNetSpuArithmeticResultBeat(cfg)))
  val release = slave(Stream(BitNetSpuArithmeticLeaseRelease(cfg)))
}

/** Single physical FfnGate slot as seen by the logical-client router. */
case class BitNetResidentSharedFfnArithmeticPhysicalPort(
    cfg: BitNetSharedSpuArithmeticPoolConfig) extends Bundle {
  val acquire = master(Stream(BitNetSpuArithmeticLeaseRequest(cfg)))
  val operand = master(Stream(BitNetSpuArithmeticOperandBeat(cfg)))
  val result = slave(Stream(BitNetSpuArithmeticResultBeat(cfg)))
  val release = master(Stream(BitNetSpuArithmeticLeaseRelease(cfg)))
}

/**
  * Multiplexes the scale composer and leased Gate/Up datapath onto the one
  * physical `BitNetSpuArithmeticClient.FfnGate` pool port.
  *
  * The component owns no arithmetic.  A physical acquire handshake fixes the
  * logical owner, tag and epoch until the same client's matching physical
  * release handshake.  At most one operand/result is outstanding, matching
  * both production clients and making sequence authentication explicit.
  * Queued acquires are normal backpressure.  Non-owner or identity-invalid
  * forward traffic is refused and recorded.  A reverse result with the wrong
  * physical owner/tag/epoch/sequence/frame identity enters reset-only
  * quarantine, so it can never be consumed by a later logical owner.
  */
class BitNetResidentSharedFfnArithmeticClient(
    cfg: BitNetSharedSpuArithmeticPoolConfig =
      BitNetSharedSpuArithmeticPoolConfig()) extends Component {
  private val physicalOwner = BitNetSpuArithmeticClient.FfnGate

  val io = new Bundle {
    val gateScaleComposer =
      BitNetResidentSharedFfnArithmeticLogicalPort(cfg)
    val gateUpLeasedGate =
      BitNetResidentSharedFfnArithmeticLogicalPort(cfg)
    val physicalFfnGate =
      BitNetResidentSharedFfnArithmeticPhysicalPort(cfg)

    val ownerValid = out Bool()
    val owner = out(BitNetResidentSharedFfnArithmeticOwner())
    val resultPending = out Bool()
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
  }

  val ownerValid = RegInit(False)
  val owner = Reg(BitNetResidentSharedFfnArithmeticOwner()) init
    BitNetResidentSharedFfnArithmeticOwner.GateScaleComposer
  // True gives GateUp the next simultaneous idle acquire.
  val preferGateUp = RegInit(False)
  val heldTag = Reg(UInt(cfg.tagWidth bits)) init 0
  val heldEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val resultPending = RegInit(False)
  val pendingSequence = Reg(UInt(cfg.sequenceWidth bits)) init 0
  val pendingActive = Reg(Bits(cfg.multiplierCount bits)) init 0
  val pendingFirst = RegInit(False)
  val pendingLast = RegInit(False)
  val protocolFaultSticky = RegInit(False)
  val quarantined = RegInit(False)

  val ownsComposer = ownerValid && owner ===
    BitNetResidentSharedFfnArithmeticOwner.GateScaleComposer
  val ownsGateUp = ownerValid && owner ===
    BitNetResidentSharedFfnArithmeticOwner.GateUpLeasedGate

  /* Fair command-level arbitration; a waiting acquire is never a fault. */
  val chooseComposer = !ownerValid && !quarantined &&
    io.gateScaleComposer.acquire.valid &&
    (!io.gateUpLeasedGate.acquire.valid || !preferGateUp)
  val chooseGateUp = !ownerValid && !quarantined &&
    io.gateUpLeasedGate.acquire.valid &&
    (!io.gateScaleComposer.acquire.valid || preferGateUp)

  io.physicalFfnGate.acquire.valid := chooseComposer || chooseGateUp
  io.physicalFfnGate.acquire.payload :=
    io.gateScaleComposer.acquire.payload
  when(chooseGateUp) {
    io.physicalFfnGate.acquire.payload :=
      io.gateUpLeasedGate.acquire.payload
  }
  io.gateScaleComposer.acquire.ready :=
    chooseComposer && io.physicalFfnGate.acquire.ready
  io.gateUpLeasedGate.acquire.ready :=
    chooseGateUp && io.physicalFfnGate.acquire.ready

  when(io.physicalFfnGate.acquire.fire) {
    ownerValid := True
    resultPending := False
    when(chooseComposer) {
      owner := BitNetResidentSharedFfnArithmeticOwner.GateScaleComposer
      preferGateUp := True
      heldTag := io.gateScaleComposer.acquire.tag
      heldEpoch := io.gateScaleComposer.acquire.epoch
    } otherwise {
      owner := BitNetResidentSharedFfnArithmeticOwner.GateUpLeasedGate
      preferGateUp := False
      heldTag := io.gateUpLeasedGate.acquire.tag
      heldEpoch := io.gateUpLeasedGate.acquire.epoch
    }
  }

  /* Exactly one authenticated operand may be awaiting its reverse result. */
  val composerOperandIdentity =
    io.gateScaleComposer.operand.tag === heldTag &&
      io.gateScaleComposer.operand.epoch === heldEpoch
  val gateUpOperandIdentity =
    io.gateUpLeasedGate.operand.tag === heldTag &&
      io.gateUpLeasedGate.operand.epoch === heldEpoch
  val composerOperandExpected = ownsComposer && !resultPending &&
    !quarantined && composerOperandIdentity
  val gateUpOperandExpected = ownsGateUp && !resultPending &&
    !quarantined && gateUpOperandIdentity

  io.physicalFfnGate.operand.valid :=
    (composerOperandExpected && io.gateScaleComposer.operand.valid) ||
      (gateUpOperandExpected && io.gateUpLeasedGate.operand.valid)
  io.physicalFfnGate.operand.payload :=
    io.gateScaleComposer.operand.payload
  when(ownsGateUp) {
    io.physicalFfnGate.operand.payload :=
      io.gateUpLeasedGate.operand.payload
  }
  io.gateScaleComposer.operand.ready :=
    composerOperandExpected && io.physicalFfnGate.operand.ready
  io.gateUpLeasedGate.operand.ready :=
    gateUpOperandExpected && io.physicalFfnGate.operand.ready

  when(io.physicalFfnGate.operand.fire) {
    resultPending := True
    pendingSequence := io.physicalFfnGate.operand.sequence
    pendingActive := io.physicalFfnGate.operand.active
    pendingFirst := io.physicalFfnGate.operand.first
    pendingLast := io.physicalFfnGate.operand.last
  }

  /* A bad reverse identity is contained before either logical client sees it. */
  val reverseResultIdentity =
    io.physicalFfnGate.result.owner === physicalOwner &&
      io.physicalFfnGate.result.tag === heldTag &&
      io.physicalFfnGate.result.epoch === heldEpoch &&
      io.physicalFfnGate.result.sequence === pendingSequence &&
      io.physicalFfnGate.result.active === pendingActive &&
      io.physicalFfnGate.result.first === pendingFirst &&
      io.physicalFfnGate.result.last === pendingLast
  val reverseResultLegal = ownerValid && resultPending &&
    !quarantined && reverseResultIdentity

  io.gateScaleComposer.result.valid :=
    io.physicalFfnGate.result.valid && reverseResultLegal && ownsComposer
  io.gateUpLeasedGate.result.valid :=
    io.physicalFfnGate.result.valid && reverseResultLegal && ownsGateUp
  io.gateScaleComposer.result.payload := io.physicalFfnGate.result.payload
  io.gateUpLeasedGate.result.payload := io.physicalFfnGate.result.payload
  io.physicalFfnGate.result.ready := reverseResultLegal &&
    Mux(ownsComposer, io.gateScaleComposer.result.ready,
      io.gateUpLeasedGate.result.ready)

  when(io.physicalFfnGate.result.valid && !reverseResultLegal) {
    protocolFaultSticky := True
    quarantined := True
  }
  when(io.physicalFfnGate.result.fire) {
    resultPending := False
  }

  /* Release is the ownership barrier and must match the acquired identity. */
  val composerReleaseIdentity =
    io.gateScaleComposer.release.tag === heldTag &&
      io.gateScaleComposer.release.epoch === heldEpoch
  val gateUpReleaseIdentity =
    io.gateUpLeasedGate.release.tag === heldTag &&
      io.gateUpLeasedGate.release.epoch === heldEpoch
  val composerReleaseExpected = ownsComposer && !resultPending &&
    !quarantined && composerReleaseIdentity
  val gateUpReleaseExpected = ownsGateUp && !resultPending &&
    !quarantined && gateUpReleaseIdentity

  io.physicalFfnGate.release.valid :=
    (composerReleaseExpected && io.gateScaleComposer.release.valid) ||
      (gateUpReleaseExpected && io.gateUpLeasedGate.release.valid)
  io.physicalFfnGate.release.payload :=
    io.gateScaleComposer.release.payload
  when(ownsGateUp) {
    io.physicalFfnGate.release.payload :=
      io.gateUpLeasedGate.release.payload
  }
  io.gateScaleComposer.release.ready :=
    composerReleaseExpected && io.physicalFfnGate.release.ready
  io.gateUpLeasedGate.release.ready :=
    gateUpReleaseExpected && io.physicalFfnGate.release.ready

  when(io.physicalFfnGate.release.fire) {
    ownerValid := False
  }

  /* Non-owner, phase-early, or mismatched forward traffic is refused. */
  when(io.gateScaleComposer.operand.valid && !composerOperandExpected) {
    protocolFaultSticky := True
  }
  when(io.gateUpLeasedGate.operand.valid && !gateUpOperandExpected) {
    protocolFaultSticky := True
  }
  when(io.gateScaleComposer.release.valid && !composerReleaseExpected) {
    protocolFaultSticky := True
  }
  when(io.gateUpLeasedGate.release.valid && !gateUpReleaseExpected) {
    protocolFaultSticky := True
  }

  io.ownerValid := ownerValid
  io.owner := owner
  io.resultPending := resultPending
  io.protocolFaultSticky := protocolFaultSticky
  io.quarantined := quarantined
}

object GenerateBitNetResidentSharedFfnArithmeticClient extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-shared-ffn-arithmetic-client")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentSharedFfnArithmeticClient())
}
