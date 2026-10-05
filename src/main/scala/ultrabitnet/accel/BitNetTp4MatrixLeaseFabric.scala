package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Identity carried by acquire/release handshakes.
  *
  * A release must return the tag accepted with its acquire.  Besides catching
  * a stale controller transaction, this makes ownership observable without
  * adding an opcode or packet layer to the matrix streams.
  */
case class BitNetTp4MatrixLeaseCommand(tagWidth: Int = 8) extends Bundle {
  require(tagWidth > 0)
  val tag = UInt(tagWidth bits)
}

/**
  * Atomic two-client arbiter around the one model-wide TP4 matrix cluster.
  *
  * Linear and Attention never own an engine independently: one acquire grants
  * both physical arrays, all ordinary request/return streams, and their common
  * completion state.  Attention additionally owns the DSP-resident SV path.
  * This keeps a complete QK/SV pass, projection, or LM-head scan in PL without
  * a PS-mediated engine hand-off.
  *
  * Lease control and all public matrix streams are in `slowClk`.  `fastClk`
  * remains the related 2x array clock consumed by [[BitNetTp4MatrixCluster]].
  * A release is fail-closed until both arrays are inactive, neither is
  * draining, every return stream is empty, and the owner presents no new
  * request.  Illegal traffic and premature/mismatched releases set a sticky
  * fault but never revoke ownership.
  */
class BitNetTp4MatrixLeaseFabric(
    cfg: BitNetTp4MatrixClusterConfig = BitNetTp4MatrixClusterConfig(),
    useVendorPrimitive: Boolean = true,
    leaseTagWidth: Int = 8
) extends Component {
  require(cfg.logicalEngineCount == 1 || cfg.logicalEngineCount == 2)
  require(cfg.physicalEngineCount == 1 ||
    cfg.physicalEngineCount == cfg.logicalEngineCount)
  require(leaseTagWidth > 0)

  private val m = cfg.matrix
  private val logicalEngines = cfg.logicalEngineCount
  private val physicalEngines = cfg.physicalEngineCount
  private val feederCfg = BitNetContinuousMatrixFeederConfig(matrix = m)

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    // Attention has fixed arbitration priority when both clients acquire in
    // the same idle cycle.  A granted signal denotes persistent ownership,
    // not a one-cycle acknowledgement pulse.
    val linearAcquire = slave(Stream(
      BitNetTp4MatrixLeaseCommand(leaseTagWidth)))
    val linearRelease = slave(Stream(
      BitNetTp4MatrixLeaseCommand(leaseTagWidth)))
    val attentionAcquire = slave(Stream(
      BitNetTp4MatrixLeaseCommand(leaseTagWidth)))
    val attentionRelease = slave(Stream(
      BitNetTp4MatrixLeaseCommand(leaseTagWidth)))

    val linearGranted = out Bool()
    val attentionGranted = out Bool()
    val leaseActive = out Bool()
    val leaseTag = out UInt(leaseTagWidth bits)
    val releaseSafe = out Bool()

    val linearSequence = Vec(slave(Stream(
      BitNetContinuousSequence(m))), logicalEngines)
    val linearMap0Beat = Vec(slave(Stream(
      BitNetMap0PairMatrixBeat(feederCfg))), logicalEngines)
    val linearI8Beat = Vec(slave(Stream(
      BitNetI8MatrixBeat(feederCfg))), logicalEngines)
    val linearDotResult = Vec(master(Stream(
      BitNetProductionCompactTp2DotResultBeat(m))), logicalEngines)
    val linearSvResult = if (!cfg.productionPruneGenericSv) Vec(master(Stream(
      BitNetContinuousSvResultGroup(
        m, cfg.svGroupLanes))), logicalEngines) else null

    val attentionSequence = Vec(slave(Stream(
      BitNetContinuousSequence(m))), logicalEngines)
    /** Production QK enters through the compact typed-I8 mailbox. */
    val attentionI8Beat = Vec(slave(Stream(
      BitNetI8MatrixBeat(feederCfg))), logicalEngines)
    /** Generic QK/SV beat is retained only for explicit research builds. */
    val attentionBeat = if (!cfg.productionPruneLegacyBeat)
      Vec(slave(Stream(BitNetContinuousMatrixBeat(
        m, includeGenericSv = !cfg.productionPruneGenericSv))),
        logicalEngines) else null
    val attentionDotResult = Vec(master(Stream(
      BitNetProductionCompactTp2DotResultBeat(m))), logicalEngines)
    val attentionSvResult = if (!cfg.productionPruneGenericSv)
      Vec(master(Stream(BitNetContinuousSvResultGroup(
        m, cfg.svGroupLanes))), logicalEngines) else null
    val attentionResidentSequence = Vec(slave(Stream(
      BitNetResidentSvSequence(m))), logicalEngines)
    val attentionResidentBeat = Vec(slave(Stream(
      BitNetResidentSvBeat(m))), logicalEngines)
    val attentionResidentResult = Vec(master(Stream(
      BitNetResidentSvResultGroup(
        m, cfg.svGroupLanes))), logicalEngines)

    val sequenceActive = out Bits(logicalEngines bits)
    val draining = out Bits(logicalEngines bits)
    val clusterProtocolFaultSticky = out Bits(logicalEngines bits)
    val leaseFaultSticky = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val cluster = new BitNetTp4MatrixCluster(cfg, useVendorPrimitive)
  cluster.io.slowClk := io.slowClk
  cluster.io.slowResetn := io.slowResetn
  cluster.io.fastClk := io.fastClk
  cluster.io.fastResetn := io.fastResetn

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))
  // Keep a top-level simulation/timing handle for the related 2x clock.  The
  // child cluster consumes these same raw clock/reset pins for its fast area.
  val fastCd = ClockDomain(
    clock = io.fastClk,
    reset = io.fastResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  val slowArea = new ClockingArea(slowCd) {
    val leaseHeld = RegInit(False)
    val attentionOwns = RegInit(False)
    val activeTag = Reg(UInt(leaseTagWidth bits)) init 0
    val leaseFault = RegInit(False)

    val linearOwns = leaseHeld && !attentionOwns
    val attentionOwnsBoth = leaseHeld && attentionOwns

    // R1 keeps the two-engine TP4 ABI but lets one physical TP2 core retire
    // the logical engines as complete waves.  Ownership never changes while
    // a wave is active, so no engine tag is added to the wide matrix payload.
    val serializedWaveHeld = if (physicalEngines < logicalEngines)
      RegInit(False) else False
    val serializedWaveOwner = if (physicalEngines < logicalEngines)
      Reg(UInt(log2Up(logicalEngines) bits)) init 0 else null
    val serializedWaveObservedBusy = if (physicalEngines < logicalEngines)
      RegInit(False) else null
    val serializedPreferEngine1 = if (physicalEngines < logicalEngines)
      RegInit(False) else null

    val anyReturnValid = (0 until physicalEngines).map { engine =>
      cluster.io.dotResult(engine).valid ||
        cluster.io.residentResult(engine).valid ||
        (if (!cfg.productionPruneGenericSv)
          cluster.io.svResult(engine).valid else False)
    }.reduce(_ || _)
    val arraysIdle = !cluster.io.sequenceActive.orR &&
      !cluster.io.draining.orR && !serializedWaveHeld

    val linearRequestValid = (0 until logicalEngines).map { engine =>
      io.linearSequence(engine).valid ||
        io.linearMap0Beat(engine).valid || io.linearI8Beat(engine).valid
    }.reduce(_ || _)
    val attentionRequestValid = (0 until logicalEngines).map { engine =>
      io.attentionSequence(engine).valid ||
        io.attentionI8Beat(engine).valid ||
        (if (!cfg.productionPruneLegacyBeat)
          io.attentionBeat(engine).valid else False) ||
        io.attentionResidentSequence(engine).valid ||
        io.attentionResidentBeat(engine).valid
    }.reduce(_ || _)
    val ownerRequestValid =
      (linearOwns && linearRequestValid) ||
        (attentionOwnsBoth && attentionRequestValid)

    // Requiring the owner's request ports to be empty prevents a release and
    // a newly accepted sequence from crossing on the same slow edge.
    val canRelease = arraysIdle && !anyReturnValid && !ownerRequestValid
    val fabricQuiescent = arraysIdle && !anyReturnValid
    val canAcquire = !leaseHeld && fabricQuiescent

    io.attentionAcquire.ready := canAcquire
    io.linearAcquire.ready := canAcquire && !io.attentionAcquire.valid

    val linearTagMatches = io.linearRelease.payload.tag === activeTag
    val attentionTagMatches = io.attentionRelease.payload.tag === activeTag
    io.linearRelease.ready := linearOwns && linearTagMatches && canRelease
    io.attentionRelease.ready :=
      attentionOwnsBoth && attentionTagMatches && canRelease

    when(io.attentionAcquire.fire) {
      leaseHeld := True
      attentionOwns := True
      activeTag := io.attentionAcquire.payload.tag
    } elsewhen(io.linearAcquire.fire) {
      leaseHeld := True
      attentionOwns := False
      activeTag := io.linearAcquire.payload.tag
    }

    when(io.linearRelease.fire || io.attentionRelease.fire) {
      leaseHeld := False
      attentionOwns := False
      activeTag := 0
    }

    // Matrix requests without the matching whole-cluster lease are never
    // acknowledged.  A held valid is still diagnosed immediately.
    when(linearRequestValid && !linearOwns) {
      leaseFault := True
    }
    when(attentionRequestValid && !attentionOwnsBoth) {
      leaseFault := True
    }

    // Releases from no owner, the other owner, or an old tag are illegal.
    // A correct owner's early release is also a fault, but ownership remains
    // asserted because ready stays low until the complete return drain.
    when(io.linearRelease.valid &&
      (!linearOwns || !linearTagMatches || !canRelease)) {
      leaseFault := True
    }
    when(io.attentionRelease.valid &&
      (!attentionOwnsBoth || !attentionTagMatches || !canRelease)) {
      leaseFault := True
    }

    io.linearGranted := linearOwns
    io.attentionGranted := attentionOwnsBoth
    io.leaseActive := leaseHeld
    io.leaseTag := activeTag
    io.releaseSafe := leaseHeld && canRelease
    io.leaseFaultSticky := leaseFault

    // The serialized variant needs explicit defaults before selecting one
    // logical wave.  The direct E2 variant below connects complete Streams.
    if (physicalEngines < logicalEngines) {
    for (engine <- 0 until physicalEngines) {
      cluster.io.sequence(engine).valid := False
      cluster.io.sequence(engine).payload := BitNetContinuousSequence(m).getZero
      cluster.io.map0Beat(engine).valid := False
      cluster.io.map0Beat(engine).payload :=
        BitNetMap0PairMatrixBeat(feederCfg).getZero
      cluster.io.i8Beat(engine).valid := False
      cluster.io.i8Beat(engine).payload := BitNetI8MatrixBeat(feederCfg).getZero
      if (!cfg.productionPruneLegacyBeat) {
        cluster.io.beat(engine).valid := False
        cluster.io.beat(engine).payload := BitNetContinuousMatrixBeat(
          m, includeGenericSv = !cfg.productionPruneGenericSv).getZero
      }
      cluster.io.residentSequence(engine).valid := False
      cluster.io.residentSequence(engine).payload :=
        BitNetResidentSvSequence(m).getZero
      cluster.io.residentBeat(engine).valid := False
      cluster.io.residentBeat(engine).payload := BitNetResidentSvBeat(m).getZero
      cluster.io.dotResult(engine).ready := False
      cluster.io.residentResult(engine).ready := False
      if (!cfg.productionPruneGenericSv)
        cluster.io.svResult(engine).ready := False
    }

    for (engine <- 0 until logicalEngines) {
      io.linearSequence(engine).ready := False
      io.linearMap0Beat(engine).ready := False
      io.linearI8Beat(engine).ready := False
      io.attentionSequence(engine).ready := False
      io.attentionI8Beat(engine).ready := False
      if (!cfg.productionPruneLegacyBeat)
        io.attentionBeat(engine).ready := False
      io.attentionResidentSequence(engine).ready := False
      io.attentionResidentBeat(engine).ready := False

      io.linearDotResult(engine).valid := False
      io.linearDotResult(engine).payload := cluster.io.dotResult(0).payload
      io.attentionDotResult(engine).valid := False
      io.attentionDotResult(engine).payload := cluster.io.dotResult(0).payload
      if (!cfg.productionPruneGenericSv) {
        io.linearSvResult(engine).valid := False
        io.linearSvResult(engine).payload := cluster.io.svResult(0).payload
        io.attentionSvResult(engine).valid := False
        io.attentionSvResult(engine).payload := cluster.io.svResult(0).payload
      }
      io.attentionResidentResult(engine).valid := False
      io.attentionResidentResult(engine).payload :=
        cluster.io.residentResult(0).payload
    }
    }

    if (physicalEngines == logicalEngines) {
      for (engine <- 0 until logicalEngines) {
        cluster.io.sequence(engine).valid := False
        cluster.io.sequence(engine).payload :=
          BitNetContinuousSequence(m).getZero
        cluster.io.map0Beat(engine).valid := False
        cluster.io.map0Beat(engine).payload :=
          BitNetMap0PairMatrixBeat(feederCfg).getZero
        cluster.io.i8Beat(engine).valid := False
        cluster.io.i8Beat(engine).payload :=
          BitNetI8MatrixBeat(feederCfg).getZero
        if (!cfg.productionPruneLegacyBeat) {
          cluster.io.beat(engine).valid := False
          cluster.io.beat(engine).payload := BitNetContinuousMatrixBeat(
            m, includeGenericSv = !cfg.productionPruneGenericSv).getZero
        }
        cluster.io.residentSequence(engine).valid := False
        cluster.io.residentSequence(engine).payload :=
          BitNetResidentSvSequence(m).getZero
        cluster.io.residentBeat(engine).valid := False
        cluster.io.residentBeat(engine).payload :=
          BitNetResidentSvBeat(m).getZero

        io.linearSequence(engine).ready := False
        io.linearMap0Beat(engine).ready := False
        io.linearI8Beat(engine).ready := False
        io.attentionSequence(engine).ready := False
        io.attentionI8Beat(engine).ready := False
        if (!cfg.productionPruneLegacyBeat)
          io.attentionBeat(engine).ready := False
        io.attentionResidentSequence(engine).ready := False
        io.attentionResidentBeat(engine).ready := False
        cluster.io.dotResult(engine).ready := False
        cluster.io.residentResult(engine).ready := False
        if (!cfg.productionPruneGenericSv)
          cluster.io.svResult(engine).ready := False

        io.linearDotResult(engine).valid := False
        io.linearDotResult(engine).payload := cluster.io.dotResult(engine).payload
        io.attentionDotResult(engine).valid := False
        io.attentionDotResult(engine).payload :=
          cluster.io.dotResult(engine).payload
        if (!cfg.productionPruneGenericSv) {
          io.linearSvResult(engine).valid := False
          io.linearSvResult(engine).payload := cluster.io.svResult(engine).payload
          io.attentionSvResult(engine).valid := False
          io.attentionSvResult(engine).payload :=
            cluster.io.svResult(engine).payload
        }
        io.attentionResidentResult(engine).valid := False
        io.attentionResidentResult(engine).payload :=
          cluster.io.residentResult(engine).payload

        when(linearOwns) {
          cluster.io.sequence(engine).valid := io.linearSequence(engine).valid
          cluster.io.sequence(engine).payload := io.linearSequence(engine).payload
          io.linearSequence(engine).ready := cluster.io.sequence(engine).ready
          cluster.io.map0Beat(engine).valid := io.linearMap0Beat(engine).valid
          cluster.io.map0Beat(engine).payload := io.linearMap0Beat(engine).payload
          io.linearMap0Beat(engine).ready := cluster.io.map0Beat(engine).ready
          cluster.io.i8Beat(engine).valid := io.linearI8Beat(engine).valid
          cluster.io.i8Beat(engine).payload := io.linearI8Beat(engine).payload
          io.linearI8Beat(engine).ready := cluster.io.i8Beat(engine).ready
          io.linearDotResult(engine).valid := cluster.io.dotResult(engine).valid
          cluster.io.dotResult(engine).ready := io.linearDotResult(engine).ready
          if (!cfg.productionPruneGenericSv) {
            io.linearSvResult(engine).valid := cluster.io.svResult(engine).valid
            cluster.io.svResult(engine).ready := io.linearSvResult(engine).ready
          }
        } elsewhen(attentionOwnsBoth) {
          cluster.io.sequence(engine).valid := io.attentionSequence(engine).valid
          cluster.io.sequence(engine).payload :=
            io.attentionSequence(engine).payload
          io.attentionSequence(engine).ready := cluster.io.sequence(engine).ready
          cluster.io.i8Beat(engine).valid := io.attentionI8Beat(engine).valid
          cluster.io.i8Beat(engine).payload := io.attentionI8Beat(engine).payload
          io.attentionI8Beat(engine).ready := cluster.io.i8Beat(engine).ready
          if (!cfg.productionPruneLegacyBeat) {
            cluster.io.beat(engine).valid := io.attentionBeat(engine).valid
            cluster.io.beat(engine).payload := io.attentionBeat(engine).payload
            io.attentionBeat(engine).ready := cluster.io.beat(engine).ready
          }
          io.attentionDotResult(engine).valid := cluster.io.dotResult(engine).valid
          cluster.io.dotResult(engine).ready :=
            io.attentionDotResult(engine).ready
          if (!cfg.productionPruneGenericSv) {
            io.attentionSvResult(engine).valid := cluster.io.svResult(engine).valid
            cluster.io.svResult(engine).ready :=
              io.attentionSvResult(engine).ready
          }
          cluster.io.residentSequence(engine).valid :=
            io.attentionResidentSequence(engine).valid
          cluster.io.residentSequence(engine).payload :=
            io.attentionResidentSequence(engine).payload
          io.attentionResidentSequence(engine).ready :=
            cluster.io.residentSequence(engine).ready
          cluster.io.residentBeat(engine).valid :=
            io.attentionResidentBeat(engine).valid
          cluster.io.residentBeat(engine).payload :=
            io.attentionResidentBeat(engine).payload
          io.attentionResidentBeat(engine).ready :=
            cluster.io.residentBeat(engine).ready
          io.attentionResidentResult(engine).valid :=
            cluster.io.residentResult(engine).valid
          cluster.io.residentResult(engine).ready :=
            io.attentionResidentResult(engine).ready
        }
      }
      io.sequenceActive := cluster.io.sequenceActive
      io.draining := cluster.io.draining
      io.clusterProtocolFaultSticky := cluster.io.protocolFaultSticky
    } else {
      io.sequenceActive := 0
      io.draining := 0
      io.clusterProtocolFaultSticky := 0
      val physicalBusy = cluster.io.sequenceActive.orR ||
        cluster.io.draining.orR || anyReturnValid
      val waveComplete = serializedWaveHeld &&
        serializedWaveObservedBusy && !physicalBusy
      val startValid = Vec(Bool(), logicalEngines)
      for (engine <- 0 until logicalEngines) {
        startValid(engine) :=
          (linearOwns && io.linearSequence(engine).valid) ||
            (attentionOwnsBoth &&
              (io.attentionSequence(engine).valid ||
                io.attentionResidentSequence(engine).valid))
      }

      when(serializedWaveHeld && physicalBusy) {
        serializedWaveObservedBusy := True
      }
      when(waveComplete) {
        serializedWaveHeld := False
        serializedWaveObservedBusy := False
        serializedPreferEngine1 := serializedWaveOwner === 0
      }
      when(!serializedWaveHeld) {
        when(serializedPreferEngine1 && startValid(1)) {
          serializedWaveOwner := 1
          serializedWaveHeld := True
        } elsewhen(startValid(0)) {
          serializedWaveOwner := 0
          serializedWaveHeld := True
        } elsewhen(startValid(1)) {
          serializedWaveOwner := 1
          serializedWaveHeld := True
        }
      }

      when(serializedWaveHeld) {
        for (engine <- 0 until logicalEngines) {
          when(serializedWaveOwner === engine) {
            io.sequenceActive(engine) := cluster.io.sequenceActive(0)
            io.draining(engine) := cluster.io.draining(0)
            when(cluster.io.protocolFaultSticky.orR) {
              io.clusterProtocolFaultSticky(engine) := True
            }

            when(!waveComplete && linearOwns) {
              cluster.io.sequence(0) << io.linearSequence(engine)
              cluster.io.map0Beat(0) << io.linearMap0Beat(engine)
              cluster.io.i8Beat(0) << io.linearI8Beat(engine)
              io.linearDotResult(engine) << cluster.io.dotResult(0)
              if (!cfg.productionPruneGenericSv)
                io.linearSvResult(engine) << cluster.io.svResult(0)
            } elsewhen(!waveComplete && attentionOwnsBoth) {
              cluster.io.sequence(0) << io.attentionSequence(engine)
              cluster.io.i8Beat(0) << io.attentionI8Beat(engine)
              if (!cfg.productionPruneLegacyBeat)
                cluster.io.beat(0) << io.attentionBeat(engine)
              io.attentionDotResult(engine) << cluster.io.dotResult(0)
              if (!cfg.productionPruneGenericSv)
                io.attentionSvResult(engine) << cluster.io.svResult(0)
              cluster.io.residentSequence(0) <<
                io.attentionResidentSequence(engine)
              cluster.io.residentBeat(0) <<
                io.attentionResidentBeat(engine)
              io.attentionResidentResult(engine) <<
                cluster.io.residentResult(0)
            }
          }
        }
      }
    }

    io.protocolFaultSticky := leaseFault ||
      cluster.io.protocolFaultSticky.orR
  }
}

object GenerateBitNetTp4MatrixLeaseFabric extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-tp4-matrix-lease-fabric")
  SpinalConfig(
    targetDirectory = target,
    oneFilePerComponent = false
  ).generateVerilog(new BitNetTp4MatrixLeaseFabric())
}
