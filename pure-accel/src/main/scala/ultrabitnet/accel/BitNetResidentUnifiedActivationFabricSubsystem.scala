package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Drop-in client/lease side of the resident activation fabric, backed by the
  * unified semantic tensor ports instead of a private activation Mem.
  *
  * This component deliberately owns only the existing lease arbiter and the
  * memory-free TP4 compatibility adapter.  A production parent must connect
  * its four semantic Streams to the one shared
  * [[BitNetUnifiedResidentTensorProductionShell]]; instantiating a semantic
  * frontend or physical backend here would create a second tensor image.
  */
class BitNetResidentUnifiedActivationFabricSubsystem(
    cfg: BitNetConfig = BitNetConfig.Production,
    layout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production),
    unified: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig())
    extends Component {
  require(layout.cfg == cfg, "activation subsystem layout/config mismatch")

  private val clientCount = BitNetActivationClient.elements.size

  val io = new Bundle {
    /** Sticky or same-cycle parent fail-stop.  No transaction may cross this
      * subsystem while asserted; reset is the only recovery mechanism. */
    val kill = in Bool()
    val acquire = slave(Stream(BitNetActivationFabricLease(cfg)))
    val release = slave(Stream(BitNetActivationFabricLease(cfg)))

    val clientReadCommand = Vec(
      Vec(slave(Stream(BitNetPhysicalActivationReadCommand(cfg, layout))),
        cfg.tokenParallelism), clientCount)
    val clientReadData = Vec(
      Vec(master(Stream(BitNetPhysicalActivationReadBeat(cfg, layout))),
        cfg.tokenParallelism), clientCount)
    val clientWriteData = Vec(
      Vec(slave(Stream(BitNetPhysicalActivationWriteBeat(cfg, layout))),
        cfg.tokenParallelism), clientCount)

    val unifiedActivationRead = master(Stream(
      BitNetUnifiedResidentActivationGroupRead(unified)))
    val unifiedActivationReadData = slave(Stream(
      BitNetUnifiedResidentActivationGroupReadData(unified)))
    val unifiedActivationWrite = master(Stream(
      BitNetUnifiedResidentActivationGroupWrite(unified)))
    val unifiedActivationWriteFault = slave(Stream(
      BitNetUnifiedResidentActivationGroupWriteFault(unified)))

    val ownerValid = out Bool()
    val owner = out(BitNetActivationClient())
    val ownerRequestId = out UInt(cfg.requestIdWidth bits)
    val readOutstanding = out Bits(cfg.tokenParallelism bits)
    val semanticReadOutstanding = out Bool()
    val semanticWriteOutstanding = out Bool()
    val idle = out Bool()
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
    val readFaultSticky = out Bits(cfg.tokenParallelism bits)
    val writeFaultSticky = out Bits(cfg.tokenParallelism bits)
    val collisionFaultSticky = out Bits(cfg.tokenParallelism bits)
  }

  val arbiter = new BitNetResidentActivationFabricArbiter(cfg, layout)
  val adapter = new BitNetUnifiedActivationFabricAdapter(
    BitNetUnifiedActivationFabricAdapterConfig(
      model = cfg,
      streaming = layout,
      unified = unified))

  // Observe arbiter misuse on the raw south-side inputs.  These predicates do
  // not depend on ready, so they can close the semantic north ports in the
  // same cycle without feeding a gated Stream valid back into itself.
  val rawReleaseMatches = arbiter.io.ownerValid &&
    io.release.payload.client === arbiter.io.owner &&
    io.release.payload.requestId === arbiter.io.ownerRequestId
  val rawNonOwnerTraffic = Bool()
  val rawReadOverlap = Bool()
  rawNonOwnerTraffic := False
  rawReadOverlap := False
  for ((client, clientIndex) <-
      BitNetActivationClient.elements.zipWithIndex;
      lane <- 0 until cfg.tokenParallelism) {
    when(arbiter.io.ownerValid && arbiter.io.owner =/= client &&
      (io.clientReadCommand(clientIndex)(lane).valid ||
        io.clientWriteData(clientIndex)(lane).valid)) {
      rawNonOwnerTraffic := True
    }
    // A response already presented this cycle may legally replace the old
    // outstanding beat.  Otherwise a second command is a protocol overlap.
    when(arbiter.io.ownerValid && arbiter.io.owner === client &&
      io.clientReadCommand(clientIndex)(lane).valid &&
      arbiter.io.readOutstanding(lane) &&
      !arbiter.io.fabricReadData(lane).valid) {
      rawReadOverlap := True
    }
  }
  val rawArbiterFaultNow =
    (io.acquire.valid && arbiter.io.ownerValid) ||
    (io.release.valid && !rawReleaseMatches) ||
    rawNonOwnerTraffic || rawReadOverlap
  val rawArbiterFaultSticky = RegInit(False)
  when(rawArbiterFaultNow) {
    rawArbiterFaultSticky := True
  }

  // Publish only registered quarantine state to a parent.  `io.kill` and the
  // local same-cycle predicates still close this subsystem immediately, but
  // are not reflected combinationally into a parent gate that may have
  // generated the offending client valid.
  val quarantineSticky = RegInit(False)
  when(io.kill || rawArbiterFaultNow || arbiter.io.protocolFaultSticky ||
    adapter.io.protocolFaultSticky || adapter.io.readFaultSticky.orR ||
    adapter.io.writeFaultSticky.orR || adapter.io.failStopNow) {
    quarantineSticky := True
  }

  val commandIngressBlock = io.kill || quarantineSticky ||
    rawArbiterFaultNow || arbiter.io.protocolFaultSticky ||
    adapter.io.protocolFaultSticky || adapter.io.readFaultSticky.orR ||
    adapter.io.writeFaultSticky.orR
  val semanticBlockNow = commandIngressBlock || adapter.io.failStopNow

  arbiter.io.acquire.valid := io.acquire.valid && !semanticBlockNow
  arbiter.io.acquire.payload := io.acquire.payload
  io.acquire.ready := arbiter.io.acquire.ready && !semanticBlockNow
  arbiter.io.release.valid := io.release.valid && !semanticBlockNow
  arbiter.io.release.payload := io.release.payload
  io.release.ready := arbiter.io.release.ready && !semanticBlockNow

  for (client <- 0 until clientCount; lane <- 0 until cfg.tokenParallelism) {
    arbiter.io.clientReadCommand(client)(lane).valid :=
      io.clientReadCommand(client)(lane).valid && !commandIngressBlock
    arbiter.io.clientReadCommand(client)(lane).payload :=
      io.clientReadCommand(client)(lane).payload
    io.clientReadCommand(client)(lane).ready :=
      arbiter.io.clientReadCommand(client)(lane).ready &&
        !commandIngressBlock

    // Quarantine rejects new commands but never strands a response already
    // promised by a completed read handshake.  Fault beats therefore remain
    // stable and drain through the owning client under ordinary backpressure.
    io.clientReadData(client)(lane).valid :=
      arbiter.io.clientReadData(client)(lane).valid
    io.clientReadData(client)(lane).payload :=
      arbiter.io.clientReadData(client)(lane).payload
    arbiter.io.clientReadData(client)(lane).ready :=
      io.clientReadData(client)(lane).ready

    arbiter.io.clientWriteData(client)(lane).valid :=
      io.clientWriteData(client)(lane).valid && !commandIngressBlock
    arbiter.io.clientWriteData(client)(lane).payload :=
      io.clientWriteData(client)(lane).payload
    io.clientWriteData(client)(lane).ready :=
      arbiter.io.clientWriteData(client)(lane).ready &&
        !commandIngressBlock
  }

  for (lane <- 0 until cfg.tokenParallelism) {
    adapter.io.legacyReadCommand(lane) <<
      arbiter.io.fabricReadCommand(lane)
    arbiter.io.fabricReadData(lane) <<
      adapter.io.legacyReadData(lane)
    adapter.io.legacyWriteData(lane) <<
      arbiter.io.fabricWriteData(lane)
  }

  io.unifiedActivationRead.valid :=
    adapter.io.unifiedActivationRead.valid && !semanticBlockNow
  io.unifiedActivationRead.payload :=
    adapter.io.unifiedActivationRead.payload
  adapter.io.unifiedActivationRead.ready :=
    io.unifiedActivationRead.ready && !semanticBlockNow
  adapter.io.unifiedActivationReadData << io.unifiedActivationReadData
  io.unifiedActivationWrite.valid :=
    adapter.io.unifiedActivationWrite.valid && !semanticBlockNow
  io.unifiedActivationWrite.payload :=
    adapter.io.unifiedActivationWrite.payload
  adapter.io.unifiedActivationWrite.ready :=
    io.unifiedActivationWrite.ready && !semanticBlockNow
  adapter.io.unifiedActivationWriteFault <<
    io.unifiedActivationWriteFault

  io.ownerValid := arbiter.io.ownerValid
  io.owner := arbiter.io.owner
  io.ownerRequestId := arbiter.io.ownerRequestId
  io.readOutstanding := arbiter.io.readOutstanding
  io.semanticReadOutstanding := adapter.io.readOutstanding
  io.semanticWriteOutstanding := adapter.io.writeOutstanding
  io.idle := adapter.io.idle && !arbiter.io.readOutstanding.orR
  io.protocolFaultSticky := arbiter.io.protocolFaultSticky ||
    adapter.io.protocolFaultSticky || adapter.io.readFaultSticky.orR ||
    adapter.io.writeFaultSticky.orR || rawArbiterFaultSticky
  io.quarantined := quarantineSticky
  io.readFaultSticky := adapter.io.readFaultSticky
  io.writeFaultSticky := adapter.io.writeFaultSticky

  // The semantic frontend serializes same-format reads and writes before the
  // physical F0/F1 slots.  A backend collision is therefore a global
  // protocol/fail-stop condition reported by the parent tensor shell, never a
  // recoverable per-lane legacy event.
  io.collisionFaultSticky := 0
}

object GenerateBitNetResidentUnifiedActivationFabricSubsystem extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-unified-activation-subsystem")
  val cfg = BitNetConfig.Production
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetResidentUnifiedActivationFabricSubsystem(
    cfg, BitNetStreamingActivationLayout(cfg)))
}
