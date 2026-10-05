package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Fixed geometry of the idle-only resident-hidden bootstrap path. */
case class BitNetResidentHiddenPreloadEndpointConfig(
    model: BitNetConfig = BitNetConfig.Production,
    layout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production)) {
  require(layout.cfg == model,
    "hidden-preload activation layout/config mismatch")
  require(model.hiddenSize == 2560,
    "the production hidden-preload image is exactly 2560 Q16.16 features")
  require(model.tokenParallelism == 2 || model.tokenParallelism == 4,
    "the production hidden-preload stream must be TP2 or TP4")
  require(model.spuCount == 4,
    "the production hidden-preload stream contains four feature shards")
  require(model.hiddenSize % model.spuCount == 0,
    "hidden-preload groups must end on a complete four-shard beat")

  val groupCount: Int = model.hiddenSize / model.spuCount
  val groupWidth: Int = scala.math.max(1, log2Up(groupCount))
  val groupCountWidth: Int = scala.math.max(1, log2Up(groupCount + 1))
  val featureCountWidth: Int = scala.math.max(1, log2Up(model.hiddenSize + 1))
}

/** One idle-bootstrap request.  `featureCount` must be exactly 2560. */
case class BitNetResidentHiddenPreloadCommand(
    cfg: BitNetResidentHiddenPreloadEndpointConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val featureCount = UInt(cfg.featureCountWidth bits)
}

/**
  * One natural activation group.
  *
  * `values(tokenLane)(featureShard)` is signed Q16.16 feature
  * `group * 4 + featureShard`.  The producer must send groups 0 through 639
  * without gaps or reordering and must retain command identity and framing.
  */
case class BitNetResidentHiddenPreloadBeat(
    cfg: BitNetResidentHiddenPreloadEndpointConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val group = UInt(cfg.groupWidth bits)
  val first = Bool()
  val last = Bool()
  val values = Vec(
    Vec(SInt(32 bits), cfg.model.spuCount),
    cfg.model.tokenParallelism)
}

case class BitNetResidentHiddenPreloadCompletion(
    cfg: BitNetResidentHiddenPreloadEndpointConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val featureCount = UInt(cfg.featureCountWidth bits)
  val groupsCommitted = UInt(cfg.groupCountWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

object BitNetResidentHiddenPreloadFault {
  val None = 0x00
  val InvalidCommand = 0x71
  val InputIdentity = 0x72
  val GroupOrder = 0x73
  val Framing = 0x74
  val BootstrapWindowClosed = 0x75
  val ActivationOwner = 0x76
}

/**
  * Idle-only, packet-free initial-hidden writer.
  *
  * The endpoint acquires the logical Matrix client only while the layer
  * scheduler is held idle, then writes the four physical HiddenResident token
  * banks directly.  It owns no RAM, arithmetic or TP4 engine.  Every input
  * group is held in sixteen registers until all active token-lane writes have
  * completed their real activation-fabric handshakes.  The lease is released
  * only after group 639 retires, and normal completion is impossible before
  * the release handshake.
  *
  * `bootstrapWindowOpen` is a system-level exclusion proof: integration must
  * assert it only before scheduler start and must mux this endpoint's lease
  * and Matrix write ports ahead of all inference-event clients.  Losing that
  * proof or the authenticated activation owner is fail-closed.  Malformed
  * commands/beats are reported once and enter reset-only quarantine so that a
  * stale payload can never seed a later request.
  */
class BitNetResidentHiddenPreloadEndpoint(
    cfg: BitNetResidentHiddenPreloadEndpointConfig =
      BitNetResidentHiddenPreloadEndpointConfig()) extends Component {
  private val model = cfg.model
  private val layout = cfg.layout
  private val lanes = model.tokenParallelism
  private val shards = model.spuCount

  val io = new Bundle {
    val command = slave(Stream(BitNetResidentHiddenPreloadCommand(cfg)))
    val input = slave(Stream(BitNetResidentHiddenPreloadBeat(cfg)))

    /** True only while no inference event may acquire or use activations. */
    val bootstrapWindowOpen = in Bool()

    val activationAcquire = master(Stream(BitNetActivationFabricLease(model)))
    val activationOwnerValid = in Bool()
    val activationOwner = in(BitNetActivationClient())
    val activationOwnerRequestId = in UInt(model.requestIdWidth bits)
    val activationWriteData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)
    val activationRelease = master(Stream(BitNetActivationFabricLease(model)))

    val completion = master(Stream(
      BitNetResidentHiddenPreloadCompletion(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Acquire, WaitOwner, AcceptGroup, WriteGroup,
        Release, AbortRelease, Report, Quarantine = newElement()
  }
  val state = Reg(State()) init State.Idle

  val held = Reg(BitNetResidentHiddenPreloadCommand(cfg)) init
    BitNetResidentHiddenPreloadCommand(cfg).getZero
  val expectedGroup = Reg(UInt(cfg.groupWidth bits)) init 0
  val groupsCommitted = Reg(UInt(cfg.groupCountWidth bits)) init 0
  val pendingWrites = Reg(Bits(lanes bits)) init 0
  val groupValues = Vec(
    Vec(Reg(SInt(32 bits)) init 0, shards), lanes)

  val jobFault = RegInit(False)
  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)

  io.busy := state =/= State.Idle
  io.protocolFaultSticky := stickyFault
  io.quarantined := state === State.Quarantine

  val ownerMatches = io.activationOwnerValid &&
    io.activationOwner === BitNetActivationClient.Matrix &&
    io.activationOwnerRequestId === held.requestId

  val commandLegal = io.command.payload.featureCount === model.hiddenSize &&
    io.command.payload.tokenMask.orR

  // A command is accepted only when the external scheduler and the physical
  // activation fabric both prove that bootstrap has exclusive access.
  io.command.ready := state === State.Idle &&
    io.bootstrapWindowOpen && !io.activationOwnerValid
  io.input.ready := state === State.AcceptGroup && ownerMatches &&
    io.bootstrapWindowOpen

  io.activationAcquire.valid := state === State.Acquire &&
    io.bootstrapWindowOpen && !io.activationOwnerValid
  io.activationAcquire.payload.client := BitNetActivationClient.Matrix
  io.activationAcquire.payload.requestId := held.requestId

  io.activationRelease.valid :=
    (state === State.Release || state === State.AbortRelease) && ownerMatches
  io.activationRelease.payload.client := BitNetActivationClient.Matrix
  io.activationRelease.payload.requestId := held.requestId

  val writeFire = Bits(lanes bits)
  for (lane <- 0 until lanes) {
    io.activationWriteData(lane).valid := state === State.WriteGroup &&
      pendingWrites(lane) && ownerMatches && io.bootstrapWindowOpen
    io.activationWriteData(lane).payload.buffer :=
      BitNetPhysicalActivationBuffer.HiddenResident
    io.activationWriteData(lane).payload.group := expectedGroup.resized
    io.activationWriteData(lane).payload.featureValid :=
      B((BigInt(1) << shards) - 1, shards bits)
    for (shard <- 0 until shards) {
      io.activationWriteData(lane).payload.values(shard) :=
        groupValues(lane)(shard)
    }
    writeFire(lane) := io.activationWriteData(lane).fire
  }
  val remainingWrites = pendingWrites & ~writeFire

  io.completion.valid := state === State.Report
  io.completion.payload.requestId := held.requestId
  io.completion.payload.tokenMask := held.tokenMask
  io.completion.payload.featureCount := held.featureCount
  io.completion.payload.groupsCommitted := groupsCommitted
  io.completion.payload.success := !jobFault
  io.completion.payload.faultCode := jobFaultCode

  val expectedFirst = expectedGroup === 0
  val expectedLast = expectedGroup === cfg.groupCount - 1
  val inputIdentityLegal = io.input.payload.requestId === held.requestId &&
    io.input.payload.tokenMask === held.tokenMask
  val inputOrderLegal = io.input.payload.group === expectedGroup
  val inputFramingLegal = io.input.payload.first === expectedFirst &&
    io.input.payload.last === expectedLast

  when(io.command.fire) {
    held := io.command.payload
    expectedGroup := 0
    groupsCommitted := 0
    pendingWrites := 0
    jobFault := !commandLegal
    jobFaultCode := B(BitNetResidentHiddenPreloadFault.None, 8 bits)
    when(commandLegal) {
      state := State.Acquire
    } otherwise {
      jobFaultCode := B(
        BitNetResidentHiddenPreloadFault.InvalidCommand, 8 bits)
      stickyFault := True
      state := State.Report
    }
  }

  switch(state) {
    is(State.Idle) {
      // All per-request state is initialised atomically on command acceptance.
    }

    is(State.Acquire) {
      // Another owner appearing here means bootstrap exclusion was violated.
      when(io.activationOwnerValid) {
        jobFault := True
        jobFaultCode := B(
          BitNetResidentHiddenPreloadFault.ActivationOwner, 8 bits)
        stickyFault := True
        state := State.Report
      } elsewhen(!io.bootstrapWindowOpen) {
        jobFault := True
        jobFaultCode := B(
          BitNetResidentHiddenPreloadFault.BootstrapWindowClosed, 8 bits)
        stickyFault := True
        state := State.Report
      } elsewhen(io.activationAcquire.fire) {
        state := State.WaitOwner
      }
    }

    is(State.WaitOwner) {
      when(io.activationOwnerValid && !ownerMatches) {
        // We cannot safely release a lease whose returned identity is wrong.
        jobFault := True
        jobFaultCode := B(
          BitNetResidentHiddenPreloadFault.ActivationOwner, 8 bits)
        stickyFault := True
        state := State.Quarantine
      } elsewhen(ownerMatches) {
        when(io.bootstrapWindowOpen) {
          state := State.AcceptGroup
        } otherwise {
          jobFault := True
          jobFaultCode := B(
            BitNetResidentHiddenPreloadFault.BootstrapWindowClosed, 8 bits)
          stickyFault := True
          state := State.AbortRelease
        }
      }
    }

    is(State.AcceptGroup) {
      when(!io.activationOwnerValid || !ownerMatches) {
        jobFault := True
        jobFaultCode := B(
          BitNetResidentHiddenPreloadFault.ActivationOwner, 8 bits)
        stickyFault := True
        state := State.Quarantine
      } elsewhen(!io.bootstrapWindowOpen) {
        jobFault := True
        jobFaultCode := B(
          BitNetResidentHiddenPreloadFault.BootstrapWindowClosed, 8 bits)
        stickyFault := True
        state := State.AbortRelease
      } elsewhen(io.input.fire) {
        when(!inputIdentityLegal) {
          jobFault := True
          jobFaultCode := B(
            BitNetResidentHiddenPreloadFault.InputIdentity, 8 bits)
          stickyFault := True
          state := State.AbortRelease
        } elsewhen(!inputOrderLegal) {
          jobFault := True
          jobFaultCode := B(
            BitNetResidentHiddenPreloadFault.GroupOrder, 8 bits)
          stickyFault := True
          state := State.AbortRelease
        } elsewhen(!inputFramingLegal) {
          jobFault := True
          jobFaultCode := B(
            BitNetResidentHiddenPreloadFault.Framing, 8 bits)
          stickyFault := True
          state := State.AbortRelease
        } otherwise {
          for (lane <- 0 until lanes; shard <- 0 until shards) {
            groupValues(lane)(shard) := io.input.payload.values(lane)(shard)
          }
          pendingWrites := held.tokenMask
          state := State.WriteGroup
        }
      }
    }

    is(State.WriteGroup) {
      pendingWrites := remainingWrites
      when(!io.activationOwnerValid || !ownerMatches) {
        jobFault := True
        jobFaultCode := B(
          BitNetResidentHiddenPreloadFault.ActivationOwner, 8 bits)
        stickyFault := True
        state := State.Quarantine
      } elsewhen(!io.bootstrapWindowOpen) {
        pendingWrites := 0
        jobFault := True
        jobFaultCode := B(
          BitNetResidentHiddenPreloadFault.BootstrapWindowClosed, 8 bits)
        stickyFault := True
        state := State.AbortRelease
      } elsewhen(remainingWrites === 0) {
        groupsCommitted := groupsCommitted + 1
        when(expectedGroup === cfg.groupCount - 1) {
          state := State.Release
        } otherwise {
          expectedGroup := expectedGroup + 1
          state := State.AcceptGroup
        }
      }
    }

    is(State.Release) {
      when(!io.activationOwnerValid || !ownerMatches) {
        jobFault := True
        jobFaultCode := B(
          BitNetResidentHiddenPreloadFault.ActivationOwner, 8 bits)
        stickyFault := True
        state := State.Quarantine
      } otherwise {
        when(!io.bootstrapWindowOpen) {
          jobFault := True
          jobFaultCode := B(
            BitNetResidentHiddenPreloadFault.BootstrapWindowClosed, 8 bits)
          stickyFault := True
        }
        when(io.activationRelease.fire) {
          state := State.Report
        }
      }
    }

    is(State.AbortRelease) {
      when(!io.activationOwnerValid || !ownerMatches) {
        // Faulted data is never exposed to inference, even if release cannot
        // be authenticated.  Reset is required to recover the system.
        state := State.Quarantine
      } elsewhen(io.activationRelease.fire) {
        state := State.Report
      }
    }

    is(State.Report) {
      when(io.completion.fire) {
        when(jobFault) {
          state := State.Quarantine
        } otherwise {
          state := State.Idle
        }
      }
    }

    is(State.Quarantine) {
      // Reset-only fail stop.  No lease, data or completion can be emitted.
    }
  }
}

object GenerateBitNetResidentHiddenPreloadEndpoint extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-resident-hidden-preload-endpoint")
  val cfg = BitNetResidentHiddenPreloadEndpointConfig()
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentHiddenPreloadEndpoint(cfg))
}
