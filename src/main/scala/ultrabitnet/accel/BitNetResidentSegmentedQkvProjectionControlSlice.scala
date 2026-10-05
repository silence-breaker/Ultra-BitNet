package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Resident control slice for the three segmented Query/Key/Value matrices.
  *
  * The stage expander still owns Q -> K -> V sequencing.  For each part this
  * slice atomically arms three independent physical owners:
  *
  *   1. the typed segmented-QKV arithmetic bridge and sole external TP4 path;
  *   2. the shared resident five-bank weight service; and
  *   3. the compute/weight retirement barrier.
  *
  * Query and Key completions retire only their own part and allow the stage
  * expander to issue the next part.  The bridge withholds Value's compute
  * completion until the resident Q/RoPE/KV-cache controller reports success;
  * consequently the stage expander cannot emit the whole QKV-stage
  * completion before all three matrices and resident post-processing retire.
  *
  * This component contains no matrix array, atomic composite-QKV assembler,
  * AXI master, GEMV route, or output-sized QKV RAM.  Decode ordering uses the
  * segmented bridge's single bounded 5 KiB far-half memory.
  */
class BitNetResidentSegmentedQkvProjectionControlSlice(
    weightCfg: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig(),
    bridgeCfg: BitNetSegmentedQkvProductionBridgeConfig =
      BitNetSegmentedQkvProductionBridgeConfig()) extends Component {
  private val stageCfg = weightCfg.stage
  private val endpointCfg = bridgeCfg.endpoint

  require(bridgeCfg.stage == stageCfg)
  require(endpointCfg.feeder == stageCfg.feeder)
  require(bridgeCfg.model == stageCfg.model)

  val io = new Bundle {
    /** Direct output of ResidentMatrixStageLaunchExpander for QKV. */
    val projection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(stageCfg)))

    /** The only model-wide TP4 linear transaction boundary. */
    val linearLaunch = master(Stream(
      BitNetResidentLinearLaunch(endpointCfg)))
    val linearResult = slave(Stream(
      BitNetResidentFinalFeaturePair(endpointCfg)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(endpointCfg)))

    /** Existing resident QKV/RoPE/KV-cache streaming contract. */
    val ropeCoefficient = slave(Stream(
      BitNetQkvFeatureRopeCoefficient(bridgeCfg.coefficient)))
    val qkvRound = master(Stream(
      BitNetResidentQkvRoundCommand(bridgeCfg.resident)))
    val qkvDot = master(Stream(BitNetQkvTp4DotBeat(bridgeCfg.qkv)))
    val qkvCompletion = slave(Stream(
      BitNetResidentQkvRoundCompletion(bridgeCfg.resident)))

    /** Client boundary of the sole resident Matrix weight service. */
    val weightProjection = master(Stream(
      BitNetResidentMatrixProjectionLaunch(stageCfg)))
    val weightRetirement = slave(Stream(
      BitNetResidentMatrixWeightRetirement(weightCfg)))

    /** Per-part completion consumed directly by the stage expander. */
    val projectionCompletion = master(Stream(
      BitNetResidentMatrixProjectionCompletion(stageCfg)))

    val supported = out Bool()
    val activePart = out(BitNetResidentMatrixProjectionPart())
    val busy = out Bool()
    val dependentFailureDrainActive = out Bool()
    val fatalRoundPoisoned = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val bridge = new BitNetSegmentedQkvProductionBridge(bridgeCfg)
  val barrier = new BitNetResidentMatrixProjectionBarrier(weightCfg)
  val activePartReg = Reg(BitNetResidentMatrixProjectionPart()) init
    BitNetResidentMatrixProjectionPart.Query

  // No owner may observe a projection unless all three capture the exact
  // stage-expander payload on the same clock edge.
  bridge.io.projection.payload := io.projection.payload
  io.weightProjection.payload := io.projection.payload
  barrier.io.begin.payload := io.projection.payload
  val supported = bridge.io.supported
  val forkReady = bridge.io.projection.ready &&
    io.weightProjection.ready && barrier.io.begin.ready
  io.projection.ready := supported && forkReady
  bridge.io.projection.valid := io.projection.valid && supported &&
    io.weightProjection.ready && barrier.io.begin.ready
  io.weightProjection.valid := io.projection.valid && supported &&
    bridge.io.projection.ready && barrier.io.begin.ready
  barrier.io.begin.valid := io.projection.valid && supported &&
    bridge.io.projection.ready && io.weightProjection.ready
  when(io.projection.fire) { activePartReg := io.projection.part }

  io.linearLaunch << bridge.io.linearLaunch
  bridge.io.linearResult << io.linearResult
  bridge.io.linearCompletion << io.linearCompletion
  bridge.io.ropeCoefficient << io.ropeCoefficient
  io.qkvRound << bridge.io.qkvRound
  io.qkvDot << bridge.io.qkvDot
  bridge.io.qkvCompletion << io.qkvCompletion

  barrier.io.computeCompletion << bridge.io.projectionCompletion
  barrier.io.weightRetirement << io.weightRetirement
  io.projectionCompletion << barrier.io.completion

  // A failed physical side has already been fully retired by `barrier`.  If
  // it belongs to Q or K, the resident QKV controller is nevertheless still
  // open and has no abort command.  Poison it before another part/round can be
  // admitted.  Value normally closes the resident round before this point,
  // so a weight-only Value failure remains recoverable.
  bridge.io.externalPoison := barrier.io.completion.valid &&
    !barrier.io.completion.success

  io.supported := supported
  io.activePart := activePartReg
  io.busy := bridge.io.busy || barrier.io.busy
  io.dependentFailureDrainActive := bridge.io.failureDrainActive
  io.fatalRoundPoisoned := bridge.io.fatalRoundPoisoned
  io.protocolFaultSticky := bridge.io.protocolFaultSticky ||
    barrier.io.protocolFaultSticky
}

object GenerateBitNetResidentSegmentedQkvProjectionControlSlice
    extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-segmented-qkv-control-slice")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(
      new BitNetResidentSegmentedQkvProjectionControlSlice())
}
