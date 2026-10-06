package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** One physical unified tensor image behind the related 1x/2x clock bridge.
  *
  * Semantic range/format/alias authentication deliberately lives above this
  * component.  This shell only owns clock-domain transfer and the single
  * `32 x 128 x 512` physical image; production integration must instantiate
  * it exactly once and attach every Activation/Projection/Q/LM adapter to its
  * slow-domain streams.
  */
class BitNetUnifiedResidentTensorPhysicalSubsystem(
    cfg: BitNetUnifiedResidentTensorRelatedClockBridgeConfig =
      BitNetUnifiedResidentTensorRelatedClockBridgeConfig())
    extends Component {
  private val backendCfg = cfg.backend
  private val layout = backendCfg.scratchpad.layout

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val packedRead = slave(Stream(
      BitNetUnifiedResidentTensorFastReadCommand(backendCfg)))
    val packedReadData = master(
      BitNetUnifiedResidentTensorFastPackedReadParallelChannel(backendCfg))
    val activationWrite = slave(Stream(
      BitNetUnifiedResidentTensorFastActivationWriteCommand(backendCfg)))
    val activationWriteCommit = master(Stream(
      BitNetUnifiedResidentTensorFastWriteCommit(backendCfg)))

    val activationRead = slave(Stream(
      BitNetUnifiedResidentTensorFastReadCommand(backendCfg)))
    val activationReadData = master(Stream(
      BitNetUnifiedResidentTensorFastActivationReadResponse(backendCfg)))
    val packedWrite = slave(Stream(
      BitNetUnifiedResidentTensorFastPackedWriteCommand(backendCfg)))
    val packedWriteCommit = master(Stream(
      BitNetUnifiedResidentTensorFastPackedWriteCommit(backendCfg)))

    val commandPending = out Bits(4 bits)
    val responsePending = out Bits(4 bits)
    val bridgeProtocolFaultSticky = out Bool()

    // Fast-domain observability.  A slow-domain consumer must synchronize
    // these status bits before using them in retirement or reset logic.
    val fastPhaseF0 = out Bool()
    val fastReadPipelineValid = out Bool()
    val fastWritePipelineValid = out Bool()
    val fastCollisionFaultSticky = out Bool()
  }

  // Keep both top-level clock-domain descriptors public.  Besides making the
  // related-clock contract explicit at this integration boundary, this lets
  // simulation drive the real top-level clock/reset ports without reaching
  // through a cloned child-component ClockDomain handle.
  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))
  val fastCd = ClockDomain(
    clock = io.fastClk,
    reset = io.fastResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  val bridge =
    new BitNetUnifiedResidentTensorScratchpadRelatedClockBridge(cfg)
  bridge.io.slowClk := io.slowClk
  bridge.io.slowResetn := io.slowResetn
  bridge.io.fastClk := io.fastClk
  bridge.io.fastResetn := io.fastResetn

  bridge.io.packedRead << io.packedRead
  // The physical production boundary is natively four-lane parallel.  All
  // 4096 data bits remain direct ownership-held wires from the backend's one
  // retained row; only the four ready bits travel back toward the bridge.
  io.packedReadData.absoluteRow := bridge.io.packedReadData.absoluteRow
  io.packedReadData.tag := bridge.io.packedReadData.tag
  io.packedReadData.bankMask := bridge.io.packedReadData.bankMask
  io.packedReadData.laneMask := bridge.io.packedReadData.laneMask
  io.packedReadData.segmentValid := bridge.io.packedReadData.segmentValid
  for (lane <- 0 until layout.packedTransportSegmentCount) {
    io.packedReadData.segments(lane) :=
      bridge.io.packedReadData.segments(lane)
  }
  bridge.io.packedReadData.segmentReady := io.packedReadData.segmentReady
  bridge.io.activationWrite << io.activationWrite
  io.activationWriteCommit << bridge.io.activationWriteCommit
  bridge.io.activationRead << io.activationRead
  io.activationReadData << bridge.io.activationReadData
  bridge.io.packedWrite << io.packedWrite
  io.packedWriteCommit << bridge.io.packedWriteCommit

  val fastArea = new ClockingArea(fastCd) {
    val backend =
      new BitNetUnifiedResidentTensorScratchpadFastBackend(backendCfg)

    backend.io.packedRead << bridge.io.backendPackedRead
    bridge.io.backendPackedReadData << backend.io.packedReadData
    backend.io.activationWrite << bridge.io.backendActivationWrite
    bridge.io.backendActivationWriteCommit <<
      backend.io.activationWriteCommit
    backend.io.activationRead << bridge.io.backendActivationRead
    bridge.io.backendActivationReadData << backend.io.activationReadData
    backend.io.packedWrite << bridge.io.backendPackedWrite
    bridge.io.backendPackedWriteCommit << backend.io.packedWriteCommit
  }

  io.commandPending := bridge.io.commandPending
  io.responsePending := bridge.io.responsePending
  io.bridgeProtocolFaultSticky := bridge.io.protocolFaultSticky
  io.fastPhaseF0 := fastArea.backend.io.phaseF0
  io.fastReadPipelineValid := fastArea.backend.io.readPipelineValid
  io.fastWritePipelineValid := fastArea.backend.io.writePipelineValid
  io.fastCollisionFaultSticky :=
    fastArea.backend.io.collisionFaultSticky
}

object GenerateBitNetUnifiedResidentTensorPhysicalSubsystem extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-tensor-physical-subsystem")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedResidentTensorPhysicalSubsystem())
}
