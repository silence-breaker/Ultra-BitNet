package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Physical meaning of the four matrix/SPU lanes during attention. */
object BitNetAttentionLaneRole extends SpinalEnum(binarySequential) {
  /** Prefill: each lane is a different prompt token, one query head per pass. */
  val TokenParallel = newElement()
  /** Decode: all lanes are one token's four GQA query heads. */
  val HeadParallel = newElement()
}

case class BitNetAttentionScheduleCommand(cfg: BitNetConfig)
    extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val layer = UInt(cfg.layerWidth bits)
  /** Workload-controller round base; retained verbatim through completion. */
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val mode = BitNetMode()
  val activeTokenMask = Bits(cfg.tokenParallelism bits)
  val tokenId = Vec(UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  val queryPosition = Vec(
    UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  /** Causal key count for each logical token, from one through 4096. */
  val keyCount = Vec(UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
}

/**
  * One complete QK -> buffered-softmax -> resident-SV pass.
  *
  * BitNet deliberately changes lane ownership by workload mode.  During
  * prefill two TP2 engines are four token lanes, and the 20 query heads are
  * visited in 20 passes.  During decode the same four physical lanes become
  * the four query heads of one GQA group, so only five passes are required.
  * Making that distinction explicit prevents an apparently fast decode
  * mapping from silently destroying TP4 prefill throughput.
  */
case class BitNetAttentionPass(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val layer = UInt(cfg.layerWidth bits)
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val mode = BitNetMode()
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(log2Up(cfg.queryHeadCount) bits)
  val kvHead = UInt(log2Up(cfg.kvHeadCount) bits)
  val laneMask = Bits(cfg.tokenParallelism bits)
  val tokenId = Vec(UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  val queryHeadId = Vec(
    UInt(log2Up(cfg.queryHeadCount) bits), cfg.tokenParallelism)
  val queryPosition = Vec(
    UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  val keyCount = Vec(UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  /** Number of physical key rows fetched; inactive/shorter lanes mask locally. */
  val maxKeyCount = UInt(cfg.sequenceWidth bits)
  val first = Bool()
  val last = Bool()
}

/**
  * Fixed 20Q/5KV attention pass generator.
  *
  * This is the model-resident equivalent of BitNet's head/round control;
  * it is not a host-authored microprogram.  No K/V address or probability
  * payload crosses to PS.  A downstream KV scheduler expands each pass into
  * head-local key reads and feeds the buffered-softmax/resident-DSP path.
  */
class BitNetAttentionPassScheduler(
    cfg: BitNetConfig = BitNetConfig.Production) extends Component {
  require(cfg.tokenParallelism == 2 || cfg.tokenParallelism == 4)
  require(cfg.queryHeadCount == 20 && cfg.kvHeadCount == 5)
  require(cfg.gqaGroupSize == 4)

  val io = new Bundle {
    val command = slave(Stream(BitNetAttentionScheduleCommand(cfg)))
    val pass = master(Stream(BitNetAttentionPass(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedCommands = out UInt(32 bits)
    val emittedPasses = out UInt(32 bits)
  }

  val active = RegInit(False)
  val held = Reg(BitNetAttentionScheduleCommand(cfg)) init (
    BitNetAttentionScheduleCommand(cfg).getZero)
  val passIndex = Reg(UInt(log2Up(cfg.queryHeadCount) bits)) init 0
  val fault = RegInit(False)
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val emittedCount = Reg(UInt(32 bits)) init 0

  val prefillMaskLegal = (1 to cfg.tokenParallelism).map { count =>
    io.command.activeTokenMask === B((BigInt(1) << count) - 1,
      cfg.tokenParallelism bits)
  }.reduce(_ || _)
  val decodeMaskLegal = io.command.activeTokenMask ===
    B(1, cfg.tokenParallelism bits)
  val keyCountsLegal = Vec(Bool(), cfg.tokenParallelism)
  for (lane <- 0 until cfg.tokenParallelism) {
    keyCountsLegal(lane) := !io.command.activeTokenMask(lane) ||
      (io.command.keyCount(lane) =/= 0 &&
        io.command.keyCount(lane) <= cfg.maxSequenceLength &&
        io.command.queryPosition(lane) < cfg.maxSequenceLength &&
        io.command.keyCount(lane) ===
          io.command.queryPosition(lane).resize(cfg.sequenceWidth) + 1)
  }
  val commandLegal = io.command.layer < cfg.layerCount &&
    keyCountsLegal.asBits.andR &&
    ((io.command.mode === BitNetMode.Prefill && prefillMaskLegal) ||
      (io.command.mode === BitNetMode.Decode && decodeMaskLegal))

  io.command.ready := !active
  when(io.command.fire) {
    when(commandLegal) {
      held := io.command.payload
      passIndex := 0
      active := True
      acceptedCount := acceptedCount + 1
    } otherwise {
      fault := True
    }
  }

  val isPrefill = held.mode === BitNetMode.Prefill
  val lastPrefillPass = passIndex === cfg.queryHeadCount - 1
  val decodePassCount = cfg.queryHeadCount / cfg.tokenParallelism
  val decodePassesPerKvHead = cfg.gqaGroupSize / cfg.tokenParallelism
  val lastDecodePass = passIndex === decodePassCount - 1
  val currentLast = Mux(isPrefill, lastPrefillPass, lastDecodePass)

  io.pass.valid := active
  io.pass.requestId := held.requestId
  io.pass.layer := held.layer
  io.pass.tokenBase := held.tokenBase
  io.pass.mode := held.mode
  io.pass.laneRole := BitNetAttentionLaneRole.TokenParallel
  io.pass.passIndex := passIndex
  io.pass.kvHead := (passIndex >> log2Up(cfg.gqaGroupSize)).resized
  io.pass.laneMask := held.activeTokenMask
  io.pass.first := passIndex === 0
  io.pass.last := currentLast

  val activeKeyCount = Vec(UInt(cfg.sequenceWidth bits),
    cfg.tokenParallelism)
  for (lane <- 0 until cfg.tokenParallelism) {
    io.pass.tokenId(lane) := held.tokenId(lane)
    io.pass.queryPosition(lane) := held.queryPosition(lane)
    io.pass.keyCount(lane) := held.keyCount(lane)
    io.pass.queryHeadId(lane) := passIndex
    activeKeyCount(lane) := Mux(held.activeTokenMask(lane),
      held.keyCount(lane), U(0, cfg.sequenceWidth bits))
  }
  val maxKey = activeKeyCount.drop(1).foldLeft(activeKeyCount(0)) {
    (maximum, candidate) => Mux(maximum > candidate, maximum, candidate)
  }
  io.pass.maxKeyCount := maxKey

  when(!isPrefill) {
    io.pass.laneRole := BitNetAttentionLaneRole.HeadParallel
    io.pass.kvHead := (passIndex / decodePassesPerKvHead).resized
    io.pass.laneMask.setAll()
    io.pass.maxKeyCount := held.keyCount(0)
    for (lane <- 0 until cfg.tokenParallelism) {
      io.pass.tokenId(lane) := held.tokenId(0)
      io.pass.queryPosition(lane) := held.queryPosition(0)
      io.pass.keyCount(lane) := held.keyCount(0)
      val decodeKvHead = passIndex / decodePassesPerKvHead
      val decodeSubpass = passIndex % decodePassesPerKvHead
      io.pass.queryHeadId(lane) := ((decodeKvHead <<
        log2Up(cfg.gqaGroupSize)) +
        decodeSubpass * cfg.tokenParallelism + lane)
        .resize(log2Up(cfg.queryHeadCount))
    }
  }

  when(io.pass.fire) {
    emittedCount := emittedCount + 1
    when(currentLast) {
      active := False
      passIndex := 0
    } otherwise {
      passIndex := passIndex + 1
    }
  }

  io.busy := active
  io.protocolFaultSticky := fault
  io.acceptedCommands := acceptedCount
  io.emittedPasses := emittedCount
}

object BitNetAttentionPassMath {
  case class Pass(
      laneRole: String,
      kvHead: Int,
      queryHeads: Seq[Int],
      tokenIds: Seq[Int],
      keyCounts: Seq[Int],
      laneMask: Int)

  def schedule(
      prefill: Boolean,
      activeTokenMask: Int,
      tokenIds: Seq[Int],
      keyCounts: Seq[Int]): Seq[Pass] = {
    val lanes = tokenIds.length
    require((lanes == 2 || lanes == 4) && keyCounts.length == lanes)
    require(4 % lanes == 0)
    if (prefill) {
      (0 until 20).map { queryHead =>
        Pass("token", queryHead / 4, Seq.fill(lanes)(queryHead),
          tokenIds, keyCounts, activeTokenMask)
      }
    } else {
      require(activeTokenMask == 1)
      val passesPerKvHead = 4 / lanes
      (0 until 20 / lanes).map { passIndex =>
        val kvHead = passIndex / passesPerKvHead
        val subpass = passIndex % passesPerKvHead
        Pass("head", kvHead,
          (0 until lanes).map(kvHead * 4 + subpass * lanes + _),
          Seq.fill(lanes)(tokenIds.head),
          Seq.fill(lanes)(keyCounts.head), (1 << lanes) - 1)
      }
    }
  }
}
