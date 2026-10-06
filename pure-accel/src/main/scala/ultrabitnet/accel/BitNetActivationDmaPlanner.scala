package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Layer-boundary traffic only; intermediate tensors never leave PL scratch. */
object BitNetActivationTransfer extends SpinalEnum {
  val LoadLayerInput, StoreLayerOutput, LoadFinalToken = newElement()
}

/**
  * Fixed BitNet ping/pong activation arena in bank 0 (PL DDR).
  *
  * The resident model image ends before this arena.  Linux writes prompt
  * embeddings into ping once; PL then alternates complete layer hidden states
  * between ping and pong.  Attention scores, probabilities, QKV, FFN wide
  * intermediates and logits remain on chip and are never represented here.
  */
case class BitNetActivationArenaLayout(
    cfg: BitNetConfig,
    pingBase: BigInt,
    pongBase: BigInt,
    bufferBytes: BigInt,
    bankCapacity: BigInt,
    bank: Int = 0
) {
  require(bank == 0, "production activation ping/pong belongs to PL DDR bank 0")
  require(bufferBytes == BigInt(cfg.maxPrefillTokens) * cfg.hiddenSize * 2)
  require((pingBase & 0x7f) == 0 && (pongBase & 0x7f) == 0)
  require(pingBase >= 0 && pongBase >= pingBase + bufferBytes)
  require(pongBase + bufferBytes <= bankCapacity)

  val tokenBytes: Int = cfg.hiddenSize * 2
  val reservedEnd: BigInt = pongBase + bufferBytes
}

object BitNetActivationArenaLayout {
  private def align(value: BigInt, alignment: BigInt = 128): BigInt =
    (value + alignment - 1) & ~(alignment - 1)

  def production(
      cfg: BitNetConfig = BitNetConfig.Production,
      resident: BitNetFiveBankResidentLayout = BitNetFiveBankResidentLayout()
  ): BitNetActivationArenaLayout = {
    require(cfg == BitNetConfig.Production && resident.cfg == cfg)
    val bytes = BigInt(cfg.maxPrefillTokens) * cfg.hiddenSize * 2
    val ping = align(resident.usedBytes.head)
    val pong = align(ping + bytes)
    BitNetActivationArenaLayout(
      cfg = cfg,
      pingBase = ping,
      pongBase = pong,
      bufferBytes = bytes,
      bankCapacity = resident.bankCapacities.head
    )
  }
}

case class BitNetActivationDmaPlannerConfig(
    model: BitNetConfig = BitNetConfig.Production,
    arena: BitNetActivationArenaLayout =
      BitNetActivationArenaLayout.production(),
    dataWidth: Int = 128,
    maxBurstBeats: Int = 256
) {
  require(arena.cfg == model)
  require(dataWidth == 128)
  require(maxBurstBeats == 256)
  val beatBytes: Int = dataWidth / 8
  val beatShift: Int = log2Up(beatBytes)
  val maxRoundBytes: Int = model.tokenParallelism * arena.tokenBytes
  val maxRoundBeats: Int = maxRoundBytes / beatBytes
  // AXI's legal maximum is 256 beats, which itself needs nine bits when this
  // field carries an actual count rather than LEN.  Tiny test/model shapes may
  // have a smaller complete round, but the comparison constant must still fit.
  val remainingWidth: Int = scala.math.max(9, log2Up(maxRoundBeats + 1))
  val burstCountWidth: Int = log2Up((maxRoundBeats + maxBurstBeats - 1) /
    maxBurstBeats + 2)
}

case class BitNetActivationBoundaryCommand(cfg: BitNetActivationDmaPlannerConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val mode = BitNetMode()
  val transfer = BitNetActivationTransfer()
  val layer = UInt(cfg.model.layerWidth bits)
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenCount = UInt(cfg.model.roundTokenCountWidth bits)
}

case class BitNetActivationDmaBurst(cfg: BitNetActivationDmaPlannerConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val transfer = BitNetActivationTransfer()
  val layer = UInt(cfg.model.layerWidth bits)
  val bank = UInt(3 bits)
  val write = Bool()
  val localAddress = UInt(cfg.model.addressWidth bits)
  /** Actual beat count, not AXI LEN. */
  val beats = UInt(9 bits)
  val logicalByteOffset = UInt(cfg.model.addressWidth bits)
  val first = Bool()
  val last = Bool()
  /** Final-token reload is placed in physical token lane zero. */
  val remapToLaneZero = Bool()
}

object BitNetActivationPlanFault extends SpinalEnum {
  val None, InvalidMode, InvalidLayer, InvalidTokenRange, ArenaBounds = newElement()
}

case class BitNetActivationPlanCompletion(cfg: BitNetActivationDmaPlannerConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val transfer = BitNetActivationTransfer()
  val success = Bool()
  val fault = BitNetActivationPlanFault()
  val burstCount = UInt(cfg.burstCountWidth bits)
}

/** Pure oracle used by package tooling and RTL tests. */
object BitNetActivationDmaOracle {
  case class Burst(address: BigInt, beats: Int, byteOffset: BigInt)

  def split(address: BigInt, bytes: BigInt, beatBytes: Int = 16,
            maxBurstBeats: Int = 256): Vector[Burst] = {
    require(address >= 0 && bytes > 0)
    require(address % beatBytes == 0 && bytes % beatBytes == 0)
    val result = Vector.newBuilder[Burst]
    var cursor = address
    var remaining = bytes / beatBytes
    var offset = BigInt(0)
    while (remaining > 0) {
      val pageBeats = (4096 - (cursor & 0xfff).toInt) / beatBytes
      val take = remaining.min(BigInt(maxBurstBeats)).min(BigInt(pageBeats)).toInt
      require(take > 0)
      result += Burst(cursor, take, offset)
      cursor += take * beatBytes
      offset += take * beatBytes
      remaining -= take
    }
    result.result()
  }

  def bases(layout: BitNetActivationArenaLayout, layer: Int):
      (BigInt, BigInt) = {
    require(layer >= 0 && layer < layout.cfg.layerCount)
    if ((layer & 1) == 0) layout.pingBase -> layout.pongBase
    else layout.pongBase -> layout.pingBase
  }
}

/**
  * Lossless layer-boundary burst planner.
  *
  * One accepted semantic command is expanded into legal 128-bit bursts.  Each
  * burst is at most 256 beats and never crosses 4 KiB, so the bank-local AXI
  * adapter has no hidden long-transfer assumption.  `completion` means all
  * burst descriptors were accepted by the downstream DMA coordinator; actual
  * R/B retirement remains that coordinator's barrier.
  */
class BitNetActivationDmaPlanner(
    cfg: BitNetActivationDmaPlannerConfig =
      BitNetActivationDmaPlannerConfig()
) extends Component {
  private val model = cfg.model
  private val calculationWidth = model.addressWidth + 1

  // Keep address geometry in carry chains.  Production tokenBytes=5120 is
  // exactly (1<<12)+(1<<10); inferring a generic multiplier wastes one of the
  // only 360 DSP48E2 sites needed by the two unified matrix arrays.
  private def multiplyByConstant(value: UInt, constant: Int): UInt = {
    require(constant > 0)
    val wide = value.resize(calculationWidth)
    val terms = (0 until 31).filter(bit => ((constant >> bit) & 1) != 0)
      .map(bit => (wide << bit).resize(calculationWidth))
    val result = UInt(calculationWidth bits)
    result := terms.reduce((left, right) =>
      (left + right).resize(calculationWidth))
    result
  }

  val io = new Bundle {
    val command = slave(Stream(BitNetActivationBoundaryCommand(cfg)))
    val burst = master(Stream(BitNetActivationDmaBurst(cfg)))
    val completion = master(Stream(BitNetActivationPlanCompletion(cfg)))
    val busy = out Bool()
    val faultSticky = out Bool()
  }

  val active = RegInit(False)
  val heldRequest = Reg(UInt(model.requestIdWidth bits)) init 0
  val heldTransfer = Reg(BitNetActivationTransfer()) init
    BitNetActivationTransfer.LoadLayerInput
  val heldLayer = Reg(UInt(model.layerWidth bits)) init 0
  val heldRemap = RegInit(False)
  val address = Reg(UInt(model.addressWidth bits)) init 0
  val remaining = Reg(UInt(cfg.remainingWidth bits)) init 0
  val logicalOffset = Reg(UInt(model.addressWidth bits)) init 0
  val burstIndex = Reg(UInt(cfg.burstCountWidth bits)) init 0

  val completionValid = RegInit(False)
  val completionPayload = Reg(BitNetActivationPlanCompletion(cfg))
  io.completion.valid := completionValid
  io.completion.payload := completionPayload
  when(io.completion.fire) { completionValid := False }

  val commandLayerValid = if ((1 << model.layerWidth) == model.layerCount) {
    True
  } else {
    io.command.payload.layer < U(model.layerCount, model.layerWidth bits)
  }
  val tokenEnd = io.command.payload.tokenBase.resize(model.sequenceWidth + 1) +
    io.command.payload.tokenCount.resize(model.sequenceWidth + 1)
  val tokenRangeValid = io.command.payload.tokenCount =/= 0 &&
    io.command.payload.tokenCount <= model.tokenParallelism &&
    tokenEnd <= model.maxPrefillTokens
  val alignedRound = if (model.tokenParallelism == 1) True else
    io.command.payload.tokenBase(log2Up(model.tokenParallelism) - 1 downto 0) === 0
  val isLoad = io.command.payload.transfer ===
    BitNetActivationTransfer.LoadLayerInput
  val isStore = io.command.payload.transfer ===
    BitNetActivationTransfer.StoreLayerOutput
  val isFinal = io.command.payload.transfer ===
    BitNetActivationTransfer.LoadFinalToken
  val prefillContract = io.command.payload.mode === BitNetMode.Prefill &&
    ((isFinal && io.command.payload.layer === model.layerCount - 1 &&
      io.command.payload.tokenCount === 1) ||
      (!isFinal && alignedRound))
  val decodeContract = io.command.payload.mode === BitNetMode.Decode && isLoad &&
    io.command.payload.layer === 0 && io.command.payload.tokenBase === 0 &&
    io.command.payload.tokenCount === 1
  val modeValid = prefillContract || decodeContract

  val pingBase = U(cfg.arena.pingBase, model.addressWidth bits)
  val pongBase = U(cfg.arena.pongBase, model.addressWidth bits)
  val layerOdd = io.command.payload.layer(0)
  val layerInputBase = Mux(layerOdd, pongBase, pingBase)
  val layerOutputBase = Mux(layerOdd, pingBase, pongBase)
  val selectedBase = Mux(isStore || isFinal, layerOutputBase, layerInputBase)
  val selectedTokenCount = Mux(isFinal,
    U(1, model.roundTokenCountWidth bits), io.command.payload.tokenCount)
  val byteOffsetWide = multiplyByConstant(
    io.command.payload.tokenBase, cfg.arena.tokenBytes)
  val transferBytesWide = multiplyByConstant(
    selectedTokenCount, cfg.arena.tokenBytes)
  val startWide = selectedBase.resize(calculationWidth) + byteOffsetWide
  val endWide = startWide + transferBytesWide
  val selectedArenaEnd = selectedBase.resize(calculationWidth) +
    U(cfg.arena.bufferBytes, calculationWidth bits)
  val arenaValid = endWide <= selectedArenaEnd &&
    endWide <= cfg.arena.bankCapacity && !startWide.msb

  val commandValid = commandLayerValid && tokenRangeValid && modeValid && arenaValid
  val commandFault = BitNetActivationPlanFault()
  commandFault := BitNetActivationPlanFault.None
  when(!modeValid) {
    commandFault := BitNetActivationPlanFault.InvalidMode
  } elsewhen (!commandLayerValid) {
    commandFault := BitNetActivationPlanFault.InvalidLayer
  } elsewhen (!tokenRangeValid) {
    commandFault := BitNetActivationPlanFault.InvalidTokenRange
  } elsewhen (!arenaValid) {
    commandFault := BitNetActivationPlanFault.ArenaBounds
  }

  io.command.ready := !active && !completionValid
  when(io.command.fire) {
    heldRequest := io.command.payload.requestId
    heldTransfer := io.command.payload.transfer
    heldLayer := io.command.payload.layer
    heldRemap := isFinal
    address := startWide.resize(model.addressWidth)
    remaining := (transferBytesWide >> cfg.beatShift).resize(cfg.remainingWidth)
    logicalOffset := 0
    burstIndex := 0
    when(commandValid) {
      active := True
    } otherwise {
      completionPayload.requestId := io.command.payload.requestId
      completionPayload.transfer := io.command.payload.transfer
      completionPayload.success := False
      completionPayload.fault := commandFault
      completionPayload.burstCount := 0
      completionValid := True
    }
  }

  val pageBytes = U(4096, 13 bits) - address(11 downto 0).resize(13)
  val pageBeats = (pageBytes >> cfg.beatShift).resize(cfg.remainingWidth)
  val cappedBeats = UInt(cfg.remainingWidth bits)
  cappedBeats := remaining
  when(remaining > cfg.maxBurstBeats) {
    cappedBeats := cfg.maxBurstBeats
  }
  val burstBeats = Mux(pageBeats < cappedBeats, pageBeats, cappedBeats)
  val lastBurst = remaining === burstBeats

  io.burst.valid := active
  io.burst.payload.requestId := heldRequest
  io.burst.payload.transfer := heldTransfer
  io.burst.payload.layer := heldLayer
  io.burst.payload.bank := cfg.arena.bank
  io.burst.payload.write := heldTransfer ===
    BitNetActivationTransfer.StoreLayerOutput
  io.burst.payload.localAddress := address
  io.burst.payload.beats := burstBeats.resize(9)
  io.burst.payload.logicalByteOffset := logicalOffset
  io.burst.payload.first := burstIndex === 0
  io.burst.payload.last := lastBurst
  io.burst.payload.remapToLaneZero := heldRemap

  when(io.burst.fire) {
    val advanceBytes = (burstBeats.resize(model.addressWidth) << cfg.beatShift)
      .resize(model.addressWidth)
    address := address + advanceBytes
    logicalOffset := logicalOffset + advanceBytes
    remaining := remaining - burstBeats
    burstIndex := burstIndex + 1
    when(lastBurst) {
      active := False
      completionPayload.requestId := heldRequest
      completionPayload.transfer := heldTransfer
      completionPayload.success := True
      completionPayload.fault := BitNetActivationPlanFault.None
      completionPayload.burstCount := burstIndex + 1
      completionValid := True
    }
  }

  val sticky = RegInit(False)
  when(io.command.fire && !commandValid) { sticky := True }
  io.faultSticky := sticky
  io.busy := active || completionValid
}

object GenerateBitNetActivationDmaPlanner extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-activation-dma-planner")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetActivationDmaPlanner())
}
