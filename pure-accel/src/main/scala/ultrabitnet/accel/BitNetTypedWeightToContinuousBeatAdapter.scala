package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Stateless compatibility shim for the current byte-wide continuous core.
  *
  * Map0 performs only `{4'b0000, raw4}`; the ternary codes are not decoded,
  * sign-extended, multiplied or cached here.  I8 bytes pass unchanged.  The
  * two typed inputs are mutually exclusive by sequence contract.  If both
  * are nevertheless asserted, Map0 is accepted first and `formatCollision`
  * reports the violation while I8 remains backpressured and lossless.
  */
class BitNetTypedWeightToContinuousBeatAdapter(
    cfg: BitNetContinuousMatrixFeederConfig =
      BitNetContinuousMatrixFeederConfig(),
    includeGenericSv: Boolean = true
) extends Component {
  private val matrix = cfg.matrix
  private val base = matrix.base

  val io = new Bundle {
    val map0 = slave(Stream(BitNetMap0PairMatrixBeat(cfg)))
    val i8 = slave(Stream(BitNetI8MatrixBeat(cfg)))
    val beat = master(Stream(BitNetContinuousMatrixBeat(
      matrix, includeGenericSv)))
    val formatCollision = out Bool()
  }

  val selectMap0 = io.map0.valid
  val selectI8 = !selectMap0 && io.i8.valid
  io.formatCollision := io.map0.valid && io.i8.valid

  io.beat.valid := selectMap0 || selectI8
  io.map0.ready := io.beat.ready
  io.i8.ready := io.beat.ready && !selectMap0

  val selectedMeta = BitNetTypedMatrixBeatMeta(cfg)
  selectedMeta := io.map0.payload.meta
  when(selectI8) { selectedMeta := io.i8.payload.meta }

  for (site <- 0 until base.siteCount) {
    io.beat.payload.rawWeights(site) :=
      B(0, 4 bits) ## io.map0.payload.rawWeights(site)
    when(selectI8) {
      io.beat.payload.rawWeights(site) := io.i8.payload.rawWeights(site)
    }
    for (lane <- 0 until matrix.tokenLanesPerEngine) {
      io.beat.payload.activations(lane)(site) :=
        selectedMeta.activations(lane)(site)
    }
    if (includeGenericSv) io.beat.payload.svValue(site) := 0
  }

  io.beat.payload.laneMask := selectedMeta.laneMask
  for (lane <- 0 until matrix.tokenLanesPerEngine) {
    io.beat.payload.accumulatorTag(lane) :=
      selectedMeta.accumulatorTag(lane)
    io.beat.payload.firstK(lane) := selectedMeta.firstK(lane)
    io.beat.payload.lastK(lane) := selectedMeta.lastK(lane)
    io.beat.payload.taskId(lane) := selectedMeta.taskId(lane)
    io.beat.payload.tokenId(lane) := selectedMeta.tokenId(lane)
    io.beat.payload.outputBase(lane) := selectedMeta.outputBase(lane)
  }

  // This adapter is a dot-only compatibility boundary.  Every SV field is a
  // compile-time zero, so it cannot manufacture an attention transaction.
  if (includeGenericSv) {
    io.beat.payload.svFactorRaw := 0
    io.beat.payload.svFactorSigned := False
    io.beat.payload.svFactorFractionBits := 0
    io.beat.payload.svLimbShift := 0
    io.beat.payload.contextBank := 0
    io.beat.payload.contextSlot := 0
    io.beat.payload.headLane := 0
  }
}

object GenerateBitNetTypedWeightToContinuousBeatAdapter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-typed-weight-adapter")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetTypedWeightToContinuousBeatAdapter())
}
