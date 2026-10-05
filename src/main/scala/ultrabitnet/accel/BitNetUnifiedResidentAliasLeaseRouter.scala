package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Reset-only diagnostics for the two-client QueryLmAlias lease router. */
object BitNetUnifiedResidentAliasLeaseRouterFault {
  val AcquireInitialIdentity = 0
  val AcquireSourceDropped = 1
  val AcquireSourceIdentity = 2
  val UnexpectedGrant = 3
  val GrantOwner = 4
  val GrantZeroGeneration = 5
  // Stable diagnostic ABI reservations.  Legal grants are now handshaken
  // into the retained slot immediately, so there is no post-presentation
  // backend source-stability window in which these two faults can arise.
  val GrantSourceDropped = 6
  val GrantSourceIdentity = 7
  val UnexpectedRelease = 8
  val NonOwnerRelease = 9
  val StaleRelease = 10
  val ReleaseSourceDropped = 11
  val ReleaseSourceIdentity = 12
}

case class BitNetUnifiedResidentAliasLeaseRouterConfig(
    scratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()) {
  require(BitNetUnifiedResidentTensorAliasOwner.elements.size == 2)
  require(scratchpad.layout.aliasGenerationWidth >= 2)
}

/**
  * Memory-free Query/LM switch for the unique QueryLmAlias lease endpoint.
  *
  * The two north vector indices are the binary-sequential encodings of
  * [[BitNetUnifiedResidentTensorAliasOwner.Query]] and
  * [[BitNetUnifiedResidentTensorAliasOwner.LmHead]].  Arbitration uses a
  * rotating preference and locks the chosen source before presenting its
  * acquire southbound.  That client remains locked through authenticated grant
  * delivery and the real backend release handshake.
  *
  * A locked acquire/release source must obey the Stream stability contract.
  * A backend grant is authenticated and handshaken into the retained
  * `grantHeld` owner/generation slot before it becomes client-visible.  The
  * physical endpoint is therefore already Owned when the same retained grant
  * is published to Query or LM one slow clock later.  This registered
  * publication boundary keeps backend fail-stop and metadata cones out of the
  * client datapaths without adding another payload register bank.
  *
  * Malformed grants are never published.  Once a legal grant has been
  * captured, it is an in-flight transaction: a newly detected unrelated fault
  * quarantines the router and cleans that exact physical identity on the next
  * edge instead of feeding combinationally into the held client grant.  When a
  * real non-zero backend lease is known, quarantine drives an
  * identity-retained best-effort cleanup release; the router never abandons a
  * granted physical alias merely because its client misbehaved.
  *
  * This component contains no Mem, semantic frontend, clock bridge or physical
  * backend.
  */
class BitNetUnifiedResidentAliasLeaseRouter(
    cfg: BitNetUnifiedResidentAliasLeaseRouterConfig =
      BitNetUnifiedResidentAliasLeaseRouterConfig()) extends Component {
  private val scratchpadCfg = cfg.scratchpad
  private val generationWidth = scratchpadCfg.layout.aliasGenerationWidth
  private val clients = BitNetUnifiedResidentTensorAliasOwner.elements
  private val clientCount = clients.size
  private val queryIndex =
    BitNetUnifiedResidentTensorAliasOwner.Query.position
  private val lmIndex =
    BitNetUnifiedResidentTensorAliasOwner.LmHead.position

  val io = new Bundle {
    val clientAliasAcquire = Vec(
      slave(Stream(BitNetUnifiedResidentAliasAcquire(scratchpadCfg))),
      clientCount)
    val clientAliasGrant = Vec(
      master(Stream(BitNetUnifiedResidentAliasGrant(scratchpadCfg))),
      clientCount)
    val clientAliasRelease = Vec(
      slave(Stream(BitNetUnifiedResidentAliasRelease(scratchpadCfg))),
      clientCount)

    val backendAliasAcquire = master(Stream(
      BitNetUnifiedResidentAliasAcquire(scratchpadCfg)))
    val backendAliasGrant = slave(Stream(
      BitNetUnifiedResidentAliasGrant(scratchpadCfg)))
    val backendAliasRelease = master(Stream(
      BitNetUnifiedResidentAliasRelease(scratchpadCfg)))

    val clientLocked = out Bool()
    val lockedClient = out(BitNetUnifiedResidentTensorAliasOwner())
    val leaseHeld = out Bool()
    val leaseOwner = out(BitNetUnifiedResidentTensorAliasOwner())
    val leaseGeneration = out UInt(generationWidth bits)
    val cleanupPending = out Bool()
    val busy = out Bool()
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(16 bits)
    val acceptedAcquires = out UInt(32 bits)
    val deliveredGrants = out UInt(32 bits)
    val completedReleases = out UInt(32 bits)
    val cleanupReleases = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Acquire, WaitGrant, Granted, Release, Cleanup, Quarantine =
      newElement()
  }

  val state = Reg(State()) init State.Idle
  val preferLm = RegInit(False)
  val lockedValid = RegInit(False)
  val lockedClient = Reg(BitNetUnifiedResidentTensorAliasOwner()) init
    BitNetUnifiedResidentTensorAliasOwner.Query

  val leaseValid = RegInit(False)
  val leaseOwner = Reg(BitNetUnifiedResidentTensorAliasOwner()) init
    BitNetUnifiedResidentTensorAliasOwner.Query
  val leaseGeneration = Reg(UInt(generationWidth bits)) init 0

  val quarantineReg = RegInit(False)
  val stickyFaultBits = Reg(Bits(16 bits)) init 0
  val acceptedAcquireCounter = Reg(UInt(32 bits)) init 0
  val deliveredGrantCounter = Reg(UInt(32 bits)) init 0
  val completedReleaseCounter = Reg(UInt(32 bits)) init 0
  val cleanupReleaseCounter = Reg(UInt(32 bits)) init 0

  // ------------------------------------------------------------------
  // Fair selection and retained source identities.
  // ------------------------------------------------------------------
  val queryAcquireValid = io.clientAliasAcquire(queryIndex).valid
  val lmAcquireValid = io.clientAliasAcquire(lmIndex).valid
  val selectionAny = queryAcquireValid || lmAcquireValid
  val selectLm = lmAcquireValid && (!queryAcquireValid || preferLm)
  val selectedClient = BitNetUnifiedResidentTensorAliasOwner()
  selectedClient := BitNetUnifiedResidentTensorAliasOwner.Query
  when(selectLm) {
    selectedClient := BitNetUnifiedResidentTensorAliasOwner.LmHead
  }
  val selectedAcquireOwner = BitNetUnifiedResidentTensorAliasOwner()
  selectedAcquireOwner := io.clientAliasAcquire(queryIndex).payload.owner
  when(selectLm) {
    selectedAcquireOwner := io.clientAliasAcquire(lmIndex).payload.owner
  }
  val selectedAcquireIdentity = selectedAcquireOwner === selectedClient

  val lockedAcquireValid = Bool()
  val lockedAcquireOwner = BitNetUnifiedResidentTensorAliasOwner()
  lockedAcquireValid := io.clientAliasAcquire(queryIndex).valid
  lockedAcquireOwner := io.clientAliasAcquire(queryIndex).payload.owner
  when(lockedClient === BitNetUnifiedResidentTensorAliasOwner.LmHead) {
    lockedAcquireValid := io.clientAliasAcquire(lmIndex).valid
    lockedAcquireOwner := io.clientAliasAcquire(lmIndex).payload.owner
  }
  val lockedAcquireIdentity = lockedAcquireOwner === lockedClient

  val queryReleaseValid = io.clientAliasRelease(queryIndex).valid
  val lmReleaseValid = io.clientAliasRelease(lmIndex).valid
  val anyRelease = queryReleaseValid || lmReleaseValid
  val releaseCollision = queryReleaseValid && lmReleaseValid
  val lockedReleaseValid = Bool()
  val nonLockedReleaseValid = Bool()
  val lockedReleaseOwner = BitNetUnifiedResidentTensorAliasOwner()
  val lockedReleaseGeneration = UInt(generationWidth bits)
  lockedReleaseValid := queryReleaseValid
  nonLockedReleaseValid := lmReleaseValid
  lockedReleaseOwner := io.clientAliasRelease(queryIndex).payload.owner
  lockedReleaseGeneration :=
    io.clientAliasRelease(queryIndex).payload.generation
  when(lockedClient === BitNetUnifiedResidentTensorAliasOwner.LmHead) {
    lockedReleaseValid := lmReleaseValid
    nonLockedReleaseValid := queryReleaseValid
    lockedReleaseOwner := io.clientAliasRelease(lmIndex).payload.owner
    lockedReleaseGeneration :=
      io.clientAliasRelease(lmIndex).payload.generation
  }
  val lockedReleaseIdentity = leaseValid &&
    lockedReleaseOwner === leaseOwner &&
    lockedReleaseGeneration === leaseGeneration &&
    lockedReleaseGeneration =/= 0

  // Every authenticated backend grant is accepted into this existing one-beat
  // retained slot before publication.  Query and LM see only these registered
  // fields; no backend metadata or live fault reduction reaches their grant
  // ports combinationally.
  val grantHeld = RegInit(False)
  val heldGrantOwner = Reg(BitNetUnifiedResidentTensorAliasOwner()) init
    BitNetUnifiedResidentTensorAliasOwner.Query
  val heldGrantGeneration = Reg(UInt(generationWidth bits)) init 0
  val grantExpected = state === State.WaitGrant && lockedValid && !grantHeld
  val grantOwnerMatches =
    io.backendAliasGrant.payload.owner === lockedClient
  val grantGenerationNonZero =
    io.backendAliasGrant.payload.generation =/= 0
  val currentGrantLegal = grantExpected && grantOwnerMatches &&
    grantGenerationNonZero

  // ------------------------------------------------------------------
  // Raw protocol monitors.  They do not depend on ready, so offending traffic
  // closes new acquire/release and backend-capture work in its detecting
  // cycle.  A grant captured on an earlier edge is intentionally behind the
  // registered publication boundary and remains an in-flight presentation in
  // the detecting cycle; fault cleanup takes priority on the following edge.
  // ------------------------------------------------------------------
  val faultNowBits = Bits(16 bits)
  faultNowBits := 0
  val monitoring = !quarantineReg && state =/= State.Cleanup &&
    state =/= State.Quarantine

  when(monitoring && state === State.Idle && selectionAny &&
      !selectedAcquireIdentity) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.AcquireInitialIdentity) :=
      True
  }
  when(monitoring && state === State.Acquire && !lockedAcquireValid) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.AcquireSourceDropped) :=
      True
  }
  when(monitoring && state === State.Acquire && lockedAcquireValid &&
      !lockedAcquireIdentity) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.AcquireSourceIdentity) :=
      True
  }

  when(monitoring && io.backendAliasGrant.valid && !grantExpected) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.UnexpectedGrant) := True
  }
  when(monitoring && io.backendAliasGrant.valid && grantExpected &&
      !grantOwnerMatches) {
    faultNowBits(BitNetUnifiedResidentAliasLeaseRouterFault.GrantOwner) :=
      True
  }
  when(monitoring && io.backendAliasGrant.valid && grantExpected &&
      !grantGenerationNonZero) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.GrantZeroGeneration) :=
      True
  }
  when(monitoring &&
      (state === State.Idle || state === State.Acquire ||
        state === State.WaitGrant) && anyRelease) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.UnexpectedRelease) := True
  }
  when(monitoring && state === State.Granted &&
      (releaseCollision || nonLockedReleaseValid)) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.NonOwnerRelease) := True
  }
  when(monitoring && state === State.Granted && lockedReleaseValid &&
      !lockedReleaseIdentity) {
    faultNowBits(BitNetUnifiedResidentAliasLeaseRouterFault.StaleRelease) :=
      True
  }
  when(monitoring && state === State.Release && nonLockedReleaseValid) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.NonOwnerRelease) := True
  }
  when(monitoring && state === State.Release && !lockedReleaseValid) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.ReleaseSourceDropped) :=
      True
  }
  when(monitoring && state === State.Release && lockedReleaseValid &&
      !lockedReleaseIdentity) {
    faultNowBits(
      BitNetUnifiedResidentAliasLeaseRouterFault.ReleaseSourceIdentity) :=
      True
  }

  val faultNow = faultNowBits.orR
  val healthyNow = !quarantineReg && !faultNow

  // The cleanup identity preference is: an already delivered lease, the first
  // retained good grant, then a current non-zero grant that is being rejected.
  val currentGrantCanCleanup = io.backendAliasGrant.valid &&
    grantGenerationNonZero
  val heldGrantCanCleanup = grantHeld && heldGrantGeneration =/= 0
  val cleanupNeededNow = leaseValid || heldGrantCanCleanup ||
    currentGrantCanCleanup
  val cleanupOwnerNow = BitNetUnifiedResidentTensorAliasOwner()
  val cleanupGenerationNow = UInt(generationWidth bits)
  cleanupOwnerNow := leaseOwner
  cleanupGenerationNow := leaseGeneration
  when(!leaseValid && heldGrantCanCleanup) {
    cleanupOwnerNow := heldGrantOwner
    cleanupGenerationNow := heldGrantGeneration
  }
  when(!leaseValid && !heldGrantCanCleanup && currentGrantCanCleanup) {
    cleanupOwnerNow := io.backendAliasGrant.payload.owner
    cleanupGenerationNow := io.backendAliasGrant.payload.generation
  }

  // ------------------------------------------------------------------
  // Stream routing.  Defaults close every client; cleanup is the sole path
  // that remains live after quarantine is latched.
  // ------------------------------------------------------------------
  for (index <- 0 until clientCount) {
    io.clientAliasAcquire(index).ready := False
    io.clientAliasGrant(index).valid := False
    io.clientAliasGrant(index).payload :=
      BitNetUnifiedResidentAliasGrant(scratchpadCfg).getZero
    io.clientAliasRelease(index).ready := False
  }
  io.backendAliasAcquire.valid := False
  io.backendAliasAcquire.payload :=
    BitNetUnifiedResidentAliasAcquire(scratchpadCfg).getZero
  io.backendAliasGrant.ready := False
  io.backendAliasRelease.valid := False
  io.backendAliasRelease.payload :=
    BitNetUnifiedResidentAliasRelease(scratchpadCfg).getZero

  when(state === State.Acquire && lockedValid && lockedAcquireValid &&
      lockedAcquireIdentity && healthyNow) {
    io.backendAliasAcquire.valid := True
    io.backendAliasAcquire.payload.owner := lockedClient
    when(lockedClient === BitNetUnifiedResidentTensorAliasOwner.Query) {
      io.clientAliasAcquire(queryIndex).ready :=
        io.backendAliasAcquire.ready
    } otherwise {
      io.clientAliasAcquire(lmIndex).ready := io.backendAliasAcquire.ready
    }
  }

  // The physical grant is captured independently of client backpressure.  Its
  // source may legally withdraw valid or change payload after this handshake.
  when(grantExpected && healthyNow) {
    io.backendAliasGrant.ready := True
  }

  val heldGrantPublishable = state === State.WaitGrant && lockedValid &&
    grantHeld && !quarantineReg
  when(heldGrantPublishable) {
    when(lockedClient === BitNetUnifiedResidentTensorAliasOwner.Query) {
      io.clientAliasGrant(queryIndex).valid := True
      io.clientAliasGrant(queryIndex).payload.owner := heldGrantOwner
      io.clientAliasGrant(queryIndex).payload.generation :=
        heldGrantGeneration
    } otherwise {
      io.clientAliasGrant(lmIndex).valid := True
      io.clientAliasGrant(lmIndex).payload.owner := heldGrantOwner
      io.clientAliasGrant(lmIndex).payload.generation :=
        heldGrantGeneration
    }
  }
  // A malformed/unsolicited non-retained grant is consumed in the detecting
  // cycle so its exact non-zero identity can be cleaned.  Once a physical
  // grant is retained, the backend is already Owned and cannot legally publish
  // another grant before release.
  when(io.backendAliasGrant.valid && faultNow && !grantHeld) {
    io.backendAliasGrant.ready := True
  }

  when(state === State.Release && leaseValid && lockedReleaseValid &&
      lockedReleaseIdentity && healthyNow) {
    io.backendAliasRelease.valid := True
    io.backendAliasRelease.payload.owner := leaseOwner
    io.backendAliasRelease.payload.generation := leaseGeneration
    when(lockedClient === BitNetUnifiedResidentTensorAliasOwner.Query) {
      io.clientAliasRelease(queryIndex).ready :=
        io.backendAliasRelease.ready
    } otherwise {
      io.clientAliasRelease(lmIndex).ready := io.backendAliasRelease.ready
    }
  }

  when(state === State.Cleanup && leaseValid) {
    // Every retained/current non-zero cleanup identity was captured on a real
    // backend grant handshake, so the physical endpoint is already Owned.
    // Waiting for aliasGrant to reappear here would deadlock that endpoint.
    io.backendAliasRelease.valid := True
    io.backendAliasRelease.payload.owner := leaseOwner
    io.backendAliasRelease.payload.generation := leaseGeneration
  }

  // ------------------------------------------------------------------
  // Retained lifecycle and sticky diagnostics.
  // ------------------------------------------------------------------
  when(state =/= State.WaitGrant) {
    grantHeld := False
  }
  val backendGrantCaptureFire = io.backendAliasGrant.fire &&
    currentGrantLegal && healthyNow
  val clientGrantFire = io.clientAliasGrant(queryIndex).fire ||
    io.clientAliasGrant(lmIndex).fire
  when(backendGrantCaptureFire) {
    grantHeld := True
    heldGrantOwner := io.backendAliasGrant.payload.owner
    heldGrantGeneration := io.backendAliasGrant.payload.generation
  }
  when(clientGrantFire || faultNow) {
    grantHeld := False
  }

  switch(state) {
    is(State.Idle) {
      when(selectionAny && selectedAcquireIdentity && healthyNow) {
        lockedValid := True
        lockedClient := selectedClient
        state := State.Acquire
      }
    }
    is(State.Acquire) {
      when(io.backendAliasAcquire.fire) {
        acceptedAcquireCounter := acceptedAcquireCounter + 1
        state := State.WaitGrant
      }
    }
    is(State.WaitGrant) {
      when(clientGrantFire && heldGrantPublishable) {
        leaseValid := True
        leaseOwner := heldGrantOwner
        leaseGeneration := heldGrantGeneration
        deliveredGrantCounter := deliveredGrantCounter + 1
        state := State.Granted
      }
    }
    is(State.Granted) {
      // Selection is explicit and costs one cycle; the release producer must
      // retain its valid/payload until the real backend handshake in Release.
      when(lockedReleaseValid && lockedReleaseIdentity &&
          !nonLockedReleaseValid && healthyNow) {
        state := State.Release
      }
    }
    is(State.Release) {
      when(io.backendAliasRelease.fire) {
        preferLm :=
          lockedClient === BitNetUnifiedResidentTensorAliasOwner.Query
        lockedValid := False
        leaseValid := False
        leaseGeneration := 0
        completedReleaseCounter := completedReleaseCounter + 1
        state := State.Idle
      }
    }
    is(State.Cleanup) {
      when(io.backendAliasRelease.fire) {
        lockedValid := False
        leaseValid := False
        leaseGeneration := 0
        cleanupReleaseCounter := cleanupReleaseCounter + 1
        state := State.Quarantine
      }
    }
    is(State.Quarantine) {
      // Reset is the only exit.
    }
  }

  when(faultNow) {
    quarantineReg := True
    stickyFaultBits := stickyFaultBits | faultNowBits
    grantHeld := False
    when(cleanupNeededNow) {
      leaseValid := True
      leaseOwner := cleanupOwnerNow
      leaseGeneration := cleanupGenerationNow
      state := State.Cleanup
    } otherwise {
      lockedValid := False
      leaseValid := False
      leaseGeneration := 0
      state := State.Quarantine
    }
  }

  io.clientLocked := lockedValid
  io.lockedClient := lockedClient
  io.leaseHeld := leaseValid
  io.leaseOwner := leaseOwner
  io.leaseGeneration := leaseGeneration
  io.cleanupPending := state === State.Cleanup ||
    (faultNow && cleanupNeededNow)
  io.busy := (state =/= State.Idle && state =/= State.Quarantine) ||
    (faultNow && cleanupNeededNow)
  io.quarantined := quarantineReg || faultNow
  io.protocolFaultSticky := quarantineReg || faultNow
  io.faultBits := stickyFaultBits | faultNowBits
  io.acceptedAcquires := acceptedAcquireCounter
  io.deliveredGrants := deliveredGrantCounter
  io.completedReleases := completedReleaseCounter
  io.cleanupReleases := cleanupReleaseCounter
}

object GenerateBitNetUnifiedResidentAliasLeaseRouter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-alias-lease-router")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetUnifiedResidentAliasLeaseRouter())
}
