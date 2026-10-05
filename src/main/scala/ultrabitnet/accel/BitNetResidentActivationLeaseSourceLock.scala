package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** The producer identity is orthogonal to the semantic activation client. */
object BitNetResidentActivationLeaseProducer
    extends SpinalEnum(binarySequential) {
  val HiddenPreload, LayerEvent, LmHead = newElement()
}

/** Proof that one source-locked lease reached the physical release port. */
case class BitNetResidentActivationLeaseRetirement(cfg: BitNetConfig)
    extends Bundle {
  val source = BitNetResidentActivationLeaseProducer()
  val lease = BitNetActivationFabricLease(cfg)
}

/**
  * Source-authenticated ingress for the sole resident activation arbiter.
  *
  * `client/requestId` is deliberately not used to recover producer identity:
  * HiddenPreload and LayerEvent can both legally request the Matrix client with
  * the same request id.  The producer selected by the real downstream
  * acquire.fire is retained until its matching downstream release.fire.
  * Every protocol violation enters a sticky, reset-only quarantine.
  */
class BitNetResidentActivationLeaseSourceLock(
    cfg: BitNetConfig = BitNetConfig.Production) extends Component {
  private val sources = BitNetResidentActivationLeaseProducer.elements
  private val sourceCount = sources.size

  val io = new Bundle {
    val producerAcquire = Vec(slave(Stream(
      BitNetActivationFabricLease(cfg))), sourceCount)
    val producerRelease = Vec(slave(Stream(
      BitNetActivationFabricLease(cfg))), sourceCount)

    val fabricAcquire = master(Stream(BitNetActivationFabricLease(cfg)))
    val fabricRelease = master(Stream(BitNetActivationFabricLease(cfg)))
    val fabricOwnerValid = in Bool()
    val fabricOwner = in(BitNetActivationClient())
    val fabricOwnerRequestId = in UInt(cfg.requestIdWidth bits)
    val fabricProtocolFaultSticky = in Bool()

    // Only the locked producer observes the physical owner.
    val producerOwnerValid = out Bits(sourceCount bits)
    val producerOwnerLease = out Vec(
      BitNetActivationFabricLease(cfg), sourceCount)

    val releaseRetired = master(Flow(
      BitNetResidentActivationLeaseRetirement(cfg)))
    val lockedSourceValid = out Bool()
    val lockedSource = out(BitNetResidentActivationLeaseProducer())
    val lockedLease = out(BitNetActivationFabricLease(cfg))
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val sourceLocked = RegInit(False)
  val sourceReg = Reg(BitNetResidentActivationLeaseProducer()) init
    BitNetResidentActivationLeaseProducer.HiddenPreload
  val leaseReg = Reg(BitNetActivationFabricLease(cfg)) init
    BitNetActivationFabricLease(cfg).getZero
  val quarantine = RegInit(False)
  val stickyFault = RegInit(False)

  val acquireValid = Bits(sourceCount bits)
  val releaseValid = Bits(sourceCount bits)
  for (index <- 0 until sourceCount) {
    acquireValid(index) := io.producerAcquire(index).valid
    releaseValid(index) := io.producerRelease(index).valid
  }
  val acquireCollision =
    (acquireValid(0) && acquireValid(1)) ||
      (acquireValid(0) && acquireValid(2)) ||
      (acquireValid(1) && acquireValid(2))
  val releaseCollision =
    (releaseValid(0) && releaseValid(1)) ||
      (releaseValid(0) && releaseValid(2)) ||
      (releaseValid(1) && releaseValid(2))

  val selectedSource = BitNetResidentActivationLeaseProducer()
  selectedSource := BitNetResidentActivationLeaseProducer.HiddenPreload
  when(acquireValid(1)) {
    selectedSource := BitNetResidentActivationLeaseProducer.LayerEvent
  }
  when(acquireValid(2)) {
    selectedSource := BitNetResidentActivationLeaseProducer.LmHead
  }

  val selectedAcquire = BitNetActivationFabricLease(cfg)
  selectedAcquire := BitNetActivationFabricLease(cfg).getZero
  when(acquireValid(0)) { selectedAcquire := io.producerAcquire(0).payload }
  when(acquireValid(1)) { selectedAcquire := io.producerAcquire(1).payload }
  when(acquireValid(2)) { selectedAcquire := io.producerAcquire(2).payload }

  val ownerAuthenticated = sourceLocked && io.fabricOwnerValid &&
    io.fabricOwner === leaseReg.client &&
    io.fabricOwnerRequestId === leaseReg.requestId

  val ownerReleaseValid = Bool()
  val ownerReleaseIdentity = Bool()
  val nonOwnerReleaseValid = Bool()
  ownerReleaseValid := False
  ownerReleaseIdentity := False
  nonOwnerReleaseValid := releaseValid.orR
  for ((source, index) <- sources.zipWithIndex) {
    when(sourceLocked && sourceReg === source) {
      ownerReleaseValid := io.producerRelease(index).valid
      ownerReleaseIdentity :=
        io.producerRelease(index).payload.client === leaseReg.client &&
          io.producerRelease(index).payload.requestId === leaseReg.requestId
      nonOwnerReleaseValid := (releaseValid &
        ~B(BigInt(1) << index, sourceCount bits)).orR
    }
  }

  val ownerStateViolation =
    (!sourceLocked && io.fabricOwnerValid) ||
      (sourceLocked && !ownerAuthenticated)
  val acquireViolation =
    (!sourceLocked && acquireCollision) ||
      (sourceLocked && acquireValid.orR)
  val releaseViolation = releaseCollision ||
    (!sourceLocked && releaseValid.orR) ||
    (sourceLocked && nonOwnerReleaseValid) ||
    (sourceLocked && ownerReleaseValid && !ownerReleaseIdentity)
  val faultNow = io.fabricProtocolFaultSticky || ownerStateViolation ||
    acquireViolation || releaseViolation

  val acquireAllowed = !sourceLocked && !quarantine && !faultNow &&
    acquireValid.orR
  io.fabricAcquire.valid := acquireAllowed
  io.fabricAcquire.payload := selectedAcquire
  for ((source, index) <- sources.zipWithIndex) {
    io.producerAcquire(index).ready := acquireAllowed &&
      selectedSource === source && io.fabricAcquire.ready
  }

  val releaseAllowed = sourceLocked && ownerAuthenticated &&
    ownerReleaseValid && ownerReleaseIdentity && !quarantine && !faultNow
  io.fabricRelease.valid := releaseAllowed
  io.fabricRelease.payload := leaseReg
  for ((source, index) <- sources.zipWithIndex) {
    io.producerRelease(index).ready := releaseAllowed &&
      sourceReg === source && io.fabricRelease.ready
  }

  // Register the retirement proof.  Besides making it stable for the LM
  // gate, this prevents a release response from forming a long combinational
  // path through TerminalJoin/round-idle control back into lease producers.
  val retirementValid = RegInit(False)
  val retirementSource =
    Reg(BitNetResidentActivationLeaseProducer()) init
      BitNetResidentActivationLeaseProducer.HiddenPreload
  val retirementLease = Reg(BitNetActivationFabricLease(cfg)) init
    BitNetActivationFabricLease(cfg).getZero
  retirementValid := io.fabricRelease.fire
  when(io.fabricRelease.fire) {
    retirementSource := sourceReg
    retirementLease := leaseReg
  }
  io.releaseRetired.valid := retirementValid
  io.releaseRetired.payload.source := retirementSource
  io.releaseRetired.payload.lease := retirementLease

  io.producerOwnerValid := 0
  for ((source, index) <- sources.zipWithIndex) {
    io.producerOwnerLease(index).client := io.fabricOwner
    io.producerOwnerLease(index).requestId := io.fabricOwnerRequestId
    io.producerOwnerValid(index) := sourceLocked &&
      sourceReg === source && ownerAuthenticated && !quarantine
  }

  when(!quarantine && faultNow) {
    quarantine := True
    stickyFault := True
  } otherwise {
    when(io.fabricAcquire.fire) {
      sourceLocked := True
      sourceReg := selectedSource
      leaseReg := selectedAcquire
    }
    when(io.fabricRelease.fire) {
      sourceLocked := False
    }
  }

  io.lockedSourceValid := sourceLocked
  io.lockedSource := sourceReg
  io.lockedLease := leaseReg
  io.quarantined := quarantine
  io.protocolFaultSticky := stickyFault
}

/**
  * Holds a real LM result until both release retirement and physical owner
  * clearing have been observed.  Its output payload is a registered Stream
  * and therefore remains stable under TerminalJoin backpressure.
  */
class BitNetResidentLmCompletionAfterReleaseGate(
    cfg: BitNetResidentStreamingLmHeadConfig =
      BitNetResidentStreamingLmHeadConfig()) extends Component {
  private val model = cfg.model

  val io = new Bundle {
    val lmCompletion = slave(Stream(BitNetResidentLmHeadCompletion(cfg)))
    val releaseRetired = slave(Flow(
      BitNetResidentActivationLeaseRetirement(model)))
    val fabricOwnerValid = in Bool()
    val fabricOwner = in(BitNetActivationClient())
    val fabricOwnerRequestId = in UInt(model.requestIdWidth bits)
    val fabricProtocolFaultSticky = in Bool()

    val completion = master(Stream(BitNetResidentLmHeadCompletion(cfg)))
    val completionBuffered = out Bool()
    val releaseObserved = out Bool()
    val ownerClearObserved = out Bool()
    val busy = out Bool()
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, WaitRelease, WaitOwnerClear, Report, Quarantine = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentLmHeadCompletion(cfg)) init
    BitNetResidentLmHeadCompletion(cfg).getZero
  val stickyFault = RegInit(False)

  val lmRetirement = io.releaseRetired.valid &&
    io.releaseRetired.source === BitNetResidentActivationLeaseProducer.LmHead
  val inputOwnerMatches = io.fabricOwnerValid &&
    io.fabricOwner === BitNetActivationClient.LmHead &&
    io.fabricOwnerRequestId === io.lmCompletion.requestId
  val heldOwnerMatches = io.fabricOwnerValid &&
    io.fabricOwner === BitNetActivationClient.LmHead &&
    io.fabricOwnerRequestId === held.requestId
  val retirementMatchesInput = lmRetirement &&
    io.releaseRetired.lease.client === BitNetActivationClient.LmHead &&
    io.releaseRetired.lease.requestId === io.lmCompletion.requestId
  val retirementMatchesHeld = lmRetirement &&
    io.releaseRetired.lease.client === BitNetActivationClient.LmHead &&
    io.releaseRetired.lease.requestId === held.requestId

  io.lmCompletion.ready := state === State.Idle
  io.completion.valid := state === State.Report
  io.completion.payload := held

  val violation = Bool()
  violation := io.fabricProtocolFaultSticky

  switch(state) {
    is(State.Idle) {
      when(lmRetirement && !io.lmCompletion.fire) {
        violation := True
      }
      when(io.lmCompletion.fire) {
        held := io.lmCompletion.payload
        when(!inputOwnerMatches) {
          violation := True
        } elsewhen(io.releaseRetired.valid) {
          when(!retirementMatchesInput) {
            violation := True
          } otherwise {
            state := State.WaitOwnerClear
          }
        } otherwise {
          state := State.WaitRelease
        }
      }
    }

    is(State.WaitRelease) {
      when(io.lmCompletion.valid || !heldOwnerMatches) {
        violation := True
      }
      when(io.releaseRetired.valid) {
        when(!retirementMatchesHeld || !heldOwnerMatches) {
          violation := True
        } otherwise {
          state := State.WaitOwnerClear
        }
      }
    }

    is(State.WaitOwnerClear) {
      when(io.lmCompletion.valid || lmRetirement) {
        violation := True
      }
      when(io.fabricOwnerValid) {
        when(!heldOwnerMatches) { violation := True }
      } otherwise {
        state := State.Report
      }
    }

    is(State.Report) {
      when(io.lmCompletion.valid || lmRetirement) {
        violation := True
      }
      when(io.completion.fire) { state := State.Idle }
    }

    is(State.Quarantine) {
      // Reset is the only exit and all Stream endpoints remain blocked.
    }
  }

  when(violation) {
    state := State.Quarantine
    stickyFault := True
  }

  // Keep the input ready path purely state-derived.  Feeding the current
  // cycle's violation back into ready would create ready -> fire -> violation
  // -> ready.  A violation is still captured synchronously above and may
  // suppress a Report beat in the detecting cycle.
  when(state === State.Quarantine) {
    io.lmCompletion.ready := False
    io.completion.valid := False
  }
  when(violation) {
    io.completion.valid := False
  }

  io.completionBuffered := state =/= State.Idle &&
    state =/= State.Quarantine
  io.releaseObserved := state === State.WaitOwnerClear ||
    state === State.Report
  io.ownerClearObserved := state === State.Report
  io.busy := state =/= State.Idle
  io.quarantined := state === State.Quarantine
  io.protocolFaultSticky := stickyFault
}

object GenerateBitNetResidentActivationLeaseSourceLock extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-activation-lease-source-lock")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentActivationLeaseSourceLock())
}

object GenerateBitNetResidentLmCompletionAfterReleaseGate extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-lm-completion-release-gate")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLmCompletionAfterReleaseGate())
}
