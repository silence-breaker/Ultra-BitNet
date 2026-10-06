package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetUnifiedResidentTensorProductionShellConfig(
    backend: BitNetUnifiedResidentTensorFastBackendConfig =
      BitNetUnifiedResidentTensorFastBackendConfig(),
    toggleSynchronizerDepth: Int = 2) {
  require(toggleSynchronizerDepth >= 2)

  val semantic: BitNetUnifiedResidentTensorSemanticFrontendConfig =
    BitNetUnifiedResidentTensorSemanticFrontendConfig(backend)
  val physical: BitNetUnifiedResidentTensorRelatedClockBridgeConfig =
    BitNetUnifiedResidentTensorRelatedClockBridgeConfig(
      backend = backend,
      toggleSynchronizerDepth = toggleSynchronizerDepth)
}

/** Production semantic-to-physical composition for the unified tensor image.
  *
  * The semantic guard runs in the slow compute domain and is connected
  * point-to-point to exactly one related-clock physical subsystem.  The
  * latter is therefore the sole owner of the 32 x 128 x 512 RAM image and
  * the sole instance of the dual-pumped backend.  North-side clients never
  * observe physical tags or absolute addresses.
  */
class BitNetUnifiedResidentTensorProductionShell(
    cfg: BitNetUnifiedResidentTensorProductionShellConfig =
      BitNetUnifiedResidentTensorProductionShellConfig())
    extends Component {
  private val scratchpadCfg = cfg.backend.scratchpad
  private val layout = scratchpadCfg.layout

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val activationRead = slave(Stream(
      BitNetUnifiedResidentActivationGroupRead(scratchpadCfg)))
    val activationReadData = master(Stream(
      BitNetUnifiedResidentActivationGroupReadData(scratchpadCfg)))
    val activationWrite = slave(Stream(
      BitNetUnifiedResidentActivationGroupWrite(scratchpadCfg)))
    val activationWriteFault = master(Stream(
      BitNetUnifiedResidentActivationGroupWriteFault(scratchpadCfg)))

    val packedRead = slave(Stream(
      BitNetUnifiedResidentPackedI8Read(scratchpadCfg)))
    val packedReadData = master(
      BitNetUnifiedResidentPackedI8ReadParallelChannel(scratchpadCfg))
    val packedWrite = slave(Stream(
      BitNetUnifiedResidentPackedI8Write(scratchpadCfg)))
    val packedWriteFault = master(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(scratchpadCfg)))

    val aliasAcquire = slave(Stream(
      BitNetUnifiedResidentAliasAcquire(scratchpadCfg)))
    val aliasGrant = master(Stream(
      BitNetUnifiedResidentAliasGrant(scratchpadCfg)))
    val aliasRelease = slave(Stream(
      BitNetUnifiedResidentAliasRelease(scratchpadCfg)))

    val aliasOwnerValid = out Bool()
    val aliasOwner = out(BitNetUnifiedResidentTensorAliasOwner())
    val aliasGeneration = out UInt(layout.aliasGenerationWidth bits)
    val activationReadOutstanding = out Bool()
    val packedReadOutstanding = out Bool()
    val activationWriteOutstanding = out Bool()
    val packedWriteOutstanding = out Bool()
    val semanticReadResponsePending = out Bool()
    val semanticWriteFaultPending = out Bool()
    val physicalCommandPending = out Bits(4 bits)
    val physicalResponsePending = out Bits(4 bits)

    val formatFaultSticky = out Bool()
    val rangeFaultSticky = out Bool()
    val aliasFaultSticky = out Bool()
    val semanticCollisionFaultSticky = out Bool()
    val bridgeProtocolFaultSticky = out Bool()
    val physicalCollisionFaultSticky = out Bool()
    val protocolFaultSticky = out Bool()
    val collisionFaultSticky = out Bool()
    val quarantined = out Bool()

    val acceptedReads = out UInt(32 bits)
    val acceptedWrites = out UInt(32 bits)
    val forwardedReads = out UInt(32 bits)
    val forwardedWrites = out UInt(32 bits)
    val authenticatedReadResponses = out UInt(32 bits)
    val authenticatedWriteCommits = out UInt(32 bits)

    val fastPhaseF0 = out Bool()
    val fastReadPipelineValid = out Bool()
    val fastWritePipelineValid = out Bool()
  }

  // This is intentionally the only physical tensor subsystem in the shell.
  val physical =
    new BitNetUnifiedResidentTensorPhysicalSubsystem(cfg.physical)
  physical.io.slowClk := io.slowClk
  physical.io.slowResetn := io.slowResetn
  physical.io.fastClk := io.fastClk
  physical.io.fastResetn := io.fastResetn

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))
  // Keep a top-level descriptor for simulation and board-wrapper binding;
  // reaching through the cloned child subsystem leaves no stable simulator
  // signal handle after hierarchy transforms.
  val fastCd = ClockDomain(
    clock = io.fastClk,
    reset = io.fastResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  val slowArea = new ClockingArea(slowCd) {
    // The physical collision flag is generated in the 2x domain.  It is
    // sticky there, but still has to cross through synchronizer flops before
    // it can participate in slow-domain fail-stop and status logic.
    val physicalCollisionFaultSync = BufferCC(
      physical.io.fastCollisionFaultSticky,
      init = False,
      bufferDepth = Some(cfg.toggleSynchronizerDepth),
      inputAttributes = Seq(crossClockDomain))
    val semantic =
      new BitNetUnifiedResidentTensorSemanticFrontend(
        cfg.semantic,
        productionPackedFullRowOnly = true,
        productionActivationGroupTransport = true)
    val activationGroupAdapter =
      new BitNetUnifiedResidentActivationGroupSemanticAdapter(
        scratchpadCfg)

    semantic.io.activationRead << activationGroupAdapter.io.wideRead
    activationGroupAdapter.io.wideReadData << semantic.io.activationReadData
    semantic.io.activationWrite << activationGroupAdapter.io.wideWrite
    activationGroupAdapter.io.wideWriteFault <<
      semantic.io.activationWriteFault

    physical.io.packedRead << semantic.io.fastPackedRead
    semantic.io.fastPackedReadParallelData.absoluteRow :=
      physical.io.packedReadData.absoluteRow
    semantic.io.fastPackedReadParallelData.tag :=
      physical.io.packedReadData.tag
    semantic.io.fastPackedReadParallelData.bankMask :=
      physical.io.packedReadData.bankMask
    semantic.io.fastPackedReadParallelData.laneMask :=
      physical.io.packedReadData.laneMask
    semantic.io.fastPackedReadParallelData.segmentValid :=
      physical.io.packedReadData.segmentValid
    for (lane <- 0 until layout.packedTransportSegmentCount) {
      semantic.io.fastPackedReadParallelData.segments(lane) :=
        physical.io.packedReadData.segments(lane)
    }
    physical.io.packedReadData.segmentReady :=
      semantic.io.fastPackedReadParallelData.segmentReady
    physical.io.activationWrite << semantic.io.fastActivationWrite
    semantic.io.fastActivationWriteCommit <<
      physical.io.activationWriteCommit
    physical.io.activationRead << semantic.io.fastActivationRead
    semantic.io.fastActivationReadData << physical.io.activationReadData
    physical.io.packedWrite << semantic.io.fastPackedWrite
    semantic.io.fastPackedWriteCommit << physical.io.packedWriteCommit
    semantic.io.downstreamProtocolFaultSticky :=
      physical.io.bridgeProtocolFaultSticky || physicalCollisionFaultSync
  }

  slowArea.activationGroupAdapter.io.groupRead << io.activationRead
  io.activationReadData << slowArea.activationGroupAdapter.io.groupReadData
  slowArea.activationGroupAdapter.io.groupWrite << io.activationWrite
  io.activationWriteFault <<
    slowArea.activationGroupAdapter.io.groupWriteFault
  slowArea.semantic.io.packedRead << io.packedRead
  io.packedReadData.region :=
    slowArea.semantic.io.packedReadParallelData.region
  io.packedReadData.localRow :=
    slowArea.semantic.io.packedReadParallelData.localRow
  io.packedReadData.bankMask :=
    slowArea.semantic.io.packedReadParallelData.bankMask
  io.packedReadData.laneMask :=
    slowArea.semantic.io.packedReadParallelData.laneMask
  io.packedReadData.fault :=
    slowArea.semantic.io.packedReadParallelData.fault
  io.packedReadData.faultCode :=
    slowArea.semantic.io.packedReadParallelData.faultCode
  io.packedReadData.segmentValid :=
    slowArea.semantic.io.packedReadParallelData.segmentValid
  for (lane <- 0 until layout.packedTransportSegmentCount) {
    io.packedReadData.segments(lane) :=
      slowArea.semantic.io.packedReadParallelData.segments(lane)
  }
  slowArea.semantic.io.packedReadParallelData.segmentReady :=
    io.packedReadData.segmentReady
  slowArea.semantic.io.packedWrite << io.packedWrite
  io.packedWriteFault << slowArea.semantic.io.packedWriteFault
  slowArea.semantic.io.aliasAcquire << io.aliasAcquire
  io.aliasGrant << slowArea.semantic.io.aliasGrant
  slowArea.semantic.io.aliasRelease << io.aliasRelease

  io.aliasOwnerValid := slowArea.semantic.io.aliasOwnerValid
  io.aliasOwner := slowArea.semantic.io.aliasOwner
  io.aliasGeneration := slowArea.semantic.io.aliasGeneration
  io.activationReadOutstanding :=
    slowArea.semantic.io.activationReadOutstanding
  io.packedReadOutstanding := slowArea.semantic.io.packedReadOutstanding
  io.activationWriteOutstanding :=
    slowArea.semantic.io.activationWriteOutstanding
  io.packedWriteOutstanding :=
    slowArea.semantic.io.packedWriteOutstanding
  io.semanticReadResponsePending :=
    slowArea.semantic.io.readResponsePending
  io.semanticWriteFaultPending := slowArea.semantic.io.writeFaultPending
  io.physicalCommandPending := physical.io.commandPending
  io.physicalResponsePending := physical.io.responsePending

  io.formatFaultSticky := slowArea.semantic.io.formatFaultSticky
  io.rangeFaultSticky := slowArea.semantic.io.rangeFaultSticky
  io.aliasFaultSticky := slowArea.semantic.io.aliasFaultSticky
  io.semanticCollisionFaultSticky :=
    slowArea.semantic.io.collisionFaultSticky
  io.bridgeProtocolFaultSticky :=
    physical.io.bridgeProtocolFaultSticky
  io.physicalCollisionFaultSticky :=
    slowArea.physicalCollisionFaultSync
  io.protocolFaultSticky := slowArea.semantic.io.protocolFaultSticky ||
    physical.io.bridgeProtocolFaultSticky ||
      slowArea.physicalCollisionFaultSync
  io.collisionFaultSticky :=
    slowArea.semantic.io.collisionFaultSticky ||
      slowArea.physicalCollisionFaultSync
  io.quarantined := slowArea.semantic.io.quarantined

  io.acceptedReads := slowArea.semantic.io.acceptedReads
  io.acceptedWrites := slowArea.semantic.io.acceptedWrites
  io.forwardedReads := slowArea.semantic.io.forwardedReads
  io.forwardedWrites := slowArea.semantic.io.forwardedWrites
  io.authenticatedReadResponses :=
    slowArea.semantic.io.authenticatedReadResponses
  io.authenticatedWriteCommits :=
    slowArea.semantic.io.authenticatedWriteCommits

  io.fastPhaseF0 := physical.io.fastPhaseF0
  io.fastReadPipelineValid := physical.io.fastReadPipelineValid
  io.fastWritePipelineValid := physical.io.fastWritePipelineValid
}

object GenerateBitNetUnifiedResidentTensorProductionShell extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-tensor-production-shell")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedResidentTensorProductionShell())
}
