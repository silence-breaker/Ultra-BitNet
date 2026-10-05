package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Raw AXI4 read-address payload.  A thin board wrapper maps this Stream to an
  * AXI4 AR channel and supplies the fixed INCR/128-bit attributes.
  */
case class BitNetAxiReadAddress(cfg: BitNetAxiBankReadAdapterConfig)
    extends Bundle {
  val id = UInt(cfg.axiIdWidth bits)
  val address = UInt(cfg.axiAddressWidth bits)
  /** AXI encodes burst length as beats minus one. */
  val len = UInt(8 bits)
}

case class BitNetAxiReadData(cfg: BitNetAxiBankReadAdapterConfig)
    extends Bundle {
  val id = UInt(cfg.axiIdWidth bits)
  val data = Bits(cfg.crossbar.dataWidth bits)
  val resp = Bits(2 bits)
  val last = Bool()
}

case class BitNetAxiBankReadAdapterConfig(
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    bankId: Int = 0,
    physicalBase: BigInt = 0,
    axiAddressWidth: Int = 40,
    axiIdWidth: Int = 4,
    maxOutstanding: Int = 16
) {
  require(bankId >= 0 && bankId < crossbar.bankCount)
  require(physicalBase >= 0 && physicalBase < (BigInt(1) << axiAddressWidth))
  require((physicalBase & 0xf) == 0, "physical AXI base must be beat aligned")
  require(crossbar.dataWidth == 128)
  require(maxOutstanding >= 2 && isPow2(maxOutstanding))
  require(axiIdWidth >= log2Up(maxOutstanding))
  require(maxOutstanding <= (1 << axiIdWidth))
  require(crossbar.maxBurstBeats == 256)

  val slotWidth: Int = log2Up(maxOutstanding)
  val physicalCalculationWidth: Int = axiAddressWidth + 1
  val beatBytes: Int = crossbar.dataWidth / 8
  val beatShift: Int = log2Up(beatBytes)
}

/** Pure-Scala oracle for build-time and simulation checks. */
object BitNetAxiBurstPlanner {
  case class Burst(address: BigInt, beats: Int)

  def split(address: BigInt, beats: Int, beatBytes: Int = 16): Vector[Burst] = {
    require(address >= 0 && (address % beatBytes) == 0)
    require(beats >= 1 && beats <= 256)
    val result = Vector.newBuilder[Burst]
    var cursor = address
    var remaining = beats
    while (remaining > 0) {
      val bytesToBoundary = 4096 - (cursor & 0xfff).toInt
      val boundaryBeats = bytesToBoundary / beatBytes
      val take = scala.math.min(remaining, scala.math.min(256, boundaryBeats))
      require(take > 0)
      result += Burst(cursor, take)
      cursor += BigInt(take * beatBytes)
      remaining -= take
    }
    result.result()
  }
}

/**
  * One bank's lossless read-command to AXI4 adapter.
  *
  * It performs the protocol work that cannot be delegated to SmartConnect:
  * every logical command is split at 4 KiB boundaries, up to sixteen logical
  * commands remain outstanding with distinct AXI IDs, responses may return in
  * any ID order, and AXI sub-burst RLAST is converted back into one logical
  * `last`.  An invalid command returns a synthetic DECERR completion instead
  * of hanging its parent task.  No PS software participates in this path.
  */
class BitNetAxiBankReadAdapter(
    cfg: BitNetAxiBankReadAdapterConfig =
      BitNetAxiBankReadAdapterConfig()
) extends Component {
  private val xcfg = cfg.crossbar

  val io = new Bundle {
    val command = slave(Stream(BitNetBankReadCommand(xcfg)))
    val axiAr = master(Stream(BitNetAxiReadAddress(cfg)))
    val axiR = slave(Stream(BitNetAxiReadData(cfg)))
    val response = master(Stream(BitNetBankReadResponse(xcfg)))

    val outstanding = out UInt(log2Up(cfg.maxOutstanding + 1) bits)
    val commandExpansionBusy = out Bool()
    val commandFaultSticky = out Bool()
    val responseProtocolFaultSticky = out Bool()
    val axiResponseFaultSticky = out Bool()
  }

  val slotValid = Vec.fill(cfg.maxOutstanding)(Reg(Bool()) init False)
  val slotClient = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.clientWidth bits)) init 0)
  val slotTransaction = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.transactionIdWidth bits)) init 0)
  val slotTag = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.tagWidth bits)) init 0)
  val slotTotalBeats = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.beatCountWidth bits)) init 0)
  val slotBeatIndex = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.beatCountWidth bits)) init 0)
  val slotSubRemaining = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.beatCountWidth bits)) init 0)
  val slotSecondBeats = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.beatCountWidth bits)) init 0)

  val freeExists = !slotValid.asBits.andR
  val freeSlot = UInt(cfg.slotWidth bits)
  freeSlot := 0
  // Reversed assignment priority makes the lowest free slot win.
  for (slot <- (0 until cfg.maxOutstanding).reverse) {
    when(!slotValid(slot)) { freeSlot := slot }
  }

  val commandPhysical = (U(cfg.physicalBase,
    cfg.physicalCalculationWidth bits) +
    io.command.payload.localAddress.resize(cfg.physicalCalculationWidth))
  val commandBytes = io.command.payload.beats.resize(
    cfg.physicalCalculationWidth) << cfg.beatShift
  val commandEnd = commandPhysical + commandBytes
  val commandAligned =
    io.command.payload.localAddress(cfg.beatShift - 1 downto 0) === 0
  val commandBankValid = io.command.payload.bank === cfg.bankId
  // `beats` is deliberately one bit wider than an AXI LEN field so that 256
  // can be represented without wrapping to zero.  That also means malformed
  // producers can present 257..511; reject those locally instead of silently
  // expanding a logical command that exceeds the bank-crossbar contract.
  val commandBeatsValid = io.command.payload.beats >= 1 &&
    io.command.payload.beats <= xcfg.maxBurstBeats
  val commandAddressValid = commandPhysical.msb === False &&
    commandEnd <= (BigInt(1) << cfg.axiAddressWidth)
  val commandValid = commandBankValid && commandAligned &&
    commandBeatsValid && commandAddressValid

  val rejectValid = Reg(Bool()) init False
  val rejectPayload = Reg(BitNetBankReadResponse(xcfg))

  // One command is expanded at a time, but already-issued slots continue to
  // return independently. A 256-beat command needs at most two AR transfers.
  val expandActive = Reg(Bool()) init False
  val expandSlot = Reg(UInt(cfg.slotWidth bits)) init 0
  val expandAddress = Reg(UInt(cfg.axiAddressWidth bits)) init 0
  val expandRemaining = Reg(UInt(xcfg.beatCountWidth bits)) init 0
  io.commandExpansionBusy := expandActive

  io.command.ready := !expandActive && !rejectValid &&
    (!commandValid || freeExists)

  val startPageOffset = commandPhysical(11 downto 0)
  val startPageBytes = U(4096, 13 bits) - startPageOffset.resize(13)
  val startPageBeats = (startPageBytes >> cfg.beatShift)
    .resize(xcfg.beatCountWidth)
  val startFirstBeats = UInt(xcfg.beatCountWidth bits)
  startFirstBeats := Mux(io.command.payload.beats < startPageBeats,
    io.command.payload.beats, startPageBeats)
  val startSecondBeats =
    (io.command.payload.beats - startFirstBeats).resize(xcfg.beatCountWidth)

  when(io.command.fire) {
    when(commandValid) {
      val slot = freeSlot
      slotValid(slot) := True
      slotClient(slot) := io.command.payload.client
      slotTransaction(slot) := io.command.payload.transactionId
      slotTag(slot) := io.command.payload.tag
      slotTotalBeats(slot) := io.command.payload.beats
      slotBeatIndex(slot) := U(0, xcfg.beatCountWidth bits)
      slotSubRemaining(slot) := startFirstBeats
      slotSecondBeats(slot) := startSecondBeats

      expandActive := True
      expandSlot := slot
      expandAddress := commandPhysical.resize(cfg.axiAddressWidth)
      expandRemaining := io.command.payload.beats
    } otherwise {
      rejectValid := True
      rejectPayload.client := io.command.payload.client
      rejectPayload.transactionId := io.command.payload.transactionId
      rejectPayload.tag := io.command.payload.tag
      rejectPayload.beatIndex := U(0, xcfg.beatIndexWidth bits)
      rejectPayload.data := B(0, xcfg.dataWidth bits)
      rejectPayload.last := True
      rejectPayload.rresp := B"11" // DECERR
    }
  }

  val pageOffset = expandAddress(11 downto 0)
  val pageBytes = U(4096, 13 bits) - pageOffset.resize(13)
  val pageBeats = (pageBytes >> cfg.beatShift).resize(xcfg.beatCountWidth)
  val currentBurstBeats = UInt(xcfg.beatCountWidth bits)
  currentBurstBeats := Mux(expandRemaining < pageBeats,
    expandRemaining, pageBeats)

  io.axiAr.valid := expandActive
  io.axiAr.payload.id := expandSlot.resized
  io.axiAr.payload.address := expandAddress
  io.axiAr.payload.len := (currentBurstBeats - 1).resize(8)

  when(io.axiAr.fire) {
    expandAddress := (expandAddress +
      (currentBurstBeats.resize(cfg.axiAddressWidth) << cfg.beatShift))
      .resize(cfg.axiAddressWidth)
    expandRemaining := expandRemaining - currentBurstBeats
    when(expandRemaining === currentBurstBeats) { expandActive := False }
  }

  val rIdInRange = if (cfg.maxOutstanding == (1 << cfg.axiIdWidth)) {
    True
  } else {
    io.axiR.payload.id < U(cfg.maxOutstanding, cfg.axiIdWidth bits)
  }
  val safeRSlot = io.axiR.payload.id(cfg.slotWidth - 1 downto 0)
  val rSlotValid = rIdInRange && slotValid(safeRSlot)
  val rLogicalLast = slotBeatIndex(safeRSlot) + 1 ===
    slotTotalBeats(safeRSlot)
  val rSubLast = slotSubRemaining(safeRSlot) === 1

  io.response.valid := rejectValid || (io.axiR.valid && rSlotValid)
  io.response.payload := rejectPayload
  when(!rejectValid) {
    io.response.payload.client := slotClient(safeRSlot)
    io.response.payload.transactionId := slotTransaction(safeRSlot)
    io.response.payload.tag := slotTag(safeRSlot)
    io.response.payload.beatIndex := slotBeatIndex(safeRSlot).resized
    io.response.payload.data := io.axiR.payload.data
    io.response.payload.last := rLogicalLast
    io.response.payload.rresp := io.axiR.payload.resp
  }

  io.axiR.ready := Mux(rejectValid, False,
    Mux(rSlotValid, io.response.ready, True))

  when(rejectValid && io.response.fire) { rejectValid := False }

  val rFire = io.axiR.fire
  val acceptedRFire = rFire && rSlotValid
  val badRFire = rFire && !rSlotValid
  val badLastFire = acceptedRFire && (io.axiR.payload.last =/= rSubLast)

  when(acceptedRFire) {
    val slot = safeRSlot
    slotBeatIndex(slot) := slotBeatIndex(slot) + 1
    when(rSubLast) {
      when(rLogicalLast) {
        slotValid(slot) := False
        slotSubRemaining(slot) := U(0, xcfg.beatCountWidth bits)
      } otherwise {
        slotSubRemaining(slot) := slotSecondBeats(slot)
        slotSecondBeats(slot) := U(0, xcfg.beatCountWidth bits)
      }
    } otherwise {
      slotSubRemaining(slot) := slotSubRemaining(slot) - 1
    }
  }

  val commandFault = Reg(Bool()) init False
  val protocolFault = Reg(Bool()) init False
  val axiFault = Reg(Bool()) init False
  when(io.command.fire && !commandValid) { commandFault := True }
  when(badRFire || badLastFire) { protocolFault := True }
  when(acceptedRFire && io.axiR.payload.resp =/= B"00") { axiFault := True }
  io.commandFaultSticky := commandFault
  io.responseProtocolFaultSticky := protocolFault
  io.axiResponseFaultSticky := axiFault
  io.outstanding := CountOne(slotValid.asBits)
    .resize(log2Up(cfg.maxOutstanding + 1))
}

object GenerateBitNetAxiBankReadAdapter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-axi-bank-read-adapter")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetAxiBankReadAdapter())
}
