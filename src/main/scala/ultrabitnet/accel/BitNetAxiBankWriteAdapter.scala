package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Raw AXI4 write-address payload; the board wrapper supplies INCR attributes. */
case class BitNetAxiWriteAddress(cfg: BitNetAxiBankWriteAdapterConfig)
    extends Bundle {
  val id = UInt(cfg.axiIdWidth bits)
  val address = UInt(cfg.axiAddressWidth bits)
  /** AXI encodes burst length as beats minus one. */
  val len = UInt(8 bits)
}

/** Raw AXI4 write-data payload. AXI4 deliberately has no WID. */
case class BitNetAxiWriteData(cfg: BitNetAxiBankWriteAdapterConfig)
    extends Bundle {
  val data = Bits(cfg.crossbar.dataWidth bits)
  val strobe = Bits(cfg.crossbar.strobeWidth bits)
  val last = Bool()
}

/** Raw AXI4 write response. */
case class BitNetAxiWriteResponse(cfg: BitNetAxiBankWriteAdapterConfig)
    extends Bundle {
  val id = UInt(cfg.axiIdWidth bits)
  val resp = Bits(2 bits)
}

case class BitNetAxiBankWriteAdapterConfig(
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

/** Pure-Scala oracle used by exporters, unit tests and trace checkers. */
object BitNetAxiWriteBurstPlanner {
  case class Burst(address: BigInt, firstBeat: Int, beats: Int)

  def split(address: BigInt, beats: Int, beatBytes: Int = 16): Vector[Burst] = {
    require(address >= 0 && (address % beatBytes) == 0)
    require(beats >= 1 && beats <= 256)
    val result = Vector.newBuilder[Burst]
    var cursor = address
    var firstBeat = 0
    var remaining = beats
    while (remaining > 0) {
      val bytesToBoundary = 4096 - (cursor & 0xfff).toInt
      val boundaryBeats = bytesToBoundary / beatBytes
      val take = scala.math.min(remaining, scala.math.min(256, boundaryBeats))
      require(take > 0)
      result += Burst(cursor, firstBeat, take)
      cursor += BigInt(take * beatBytes)
      firstBeat += take
      remaining -= take
    }
    result.result()
  }
}

/**
  * One bank's lossless write-command/data to AXI4 adapter.
  *
  * A logical write is split at every 4 KiB boundary. Its sub-bursts use one
  * unique AXI ID and their (possibly out-of-order across IDs) B responses are
  * reduced to a single bank-local response. W remains non-interleaved as AXI4
  * requires, while up to `maxOutstanding` completed data streams may await B.
  *
  * Invalid commands consume their declared input stream and complete locally
  * with DECERR. If an otherwise valid producer asserts logical `last` early,
  * zero-strobe padding finishes the already-advertised AXI burst(s); this both
  * records a sticky protocol fault and avoids deadlocking the memory fabric.
  */
class BitNetAxiBankWriteAdapter(
    cfg: BitNetAxiBankWriteAdapterConfig =
      BitNetAxiBankWriteAdapterConfig()
) extends Component {
  private val xcfg = cfg.crossbar

  val io = new Bundle {
    val command = slave(Stream(BitNetBankWriteCommand(xcfg)))
    val writeData = slave(Stream(BitNetBankWriteData(xcfg)))
    val axiAw = master(Stream(BitNetAxiWriteAddress(cfg)))
    val axiW = master(Stream(BitNetAxiWriteData(cfg)))
    val axiB = slave(Stream(BitNetAxiWriteResponse(cfg)))
    val response = master(Stream(BitNetLocalBankWriteResponse(xcfg)))

    val outstanding = out UInt(log2Up(cfg.maxOutstanding + 1) bits)
    val commandFaultSticky = out Bool()
    val dataProtocolFaultSticky = out Bool()
    val responseProtocolFaultSticky = out Bool()
    val axiResponseFaultSticky = out Bool()
  }

  // A slot remains live until the last B response and downstream response
  // handshake. This prevents an AXI ID from being reused prematurely.
  val slotValid = Vec.fill(cfg.maxOutstanding)(Reg(Bool()) init False)
  val slotClient = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.clientWidth bits)) init 0)
  val slotTransaction = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.transactionIdWidth bits)) init 0)
  val slotTag = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(xcfg.tagWidth bits)) init 0)
  val slotPendingB = Vec.fill(cfg.maxOutstanding)(Reg(UInt(2 bits)) init 0)
  // Only a B whose WLAST has already handshaken may be consumed.
  val slotCompletedAwaitingB = Vec.fill(cfg.maxOutstanding)(
    Reg(UInt(2 bits)) init 0)
  val slotWorstResp = Vec.fill(cfg.maxOutstanding)(Reg(Bits(2 bits)) init 0)

  val freeExists = !slotValid.asBits.andR
  val freeSlot = UInt(cfg.slotWidth bits)
  freeSlot := U(0, cfg.slotWidth bits)
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
  val commandClientValid = io.command.payload.client < xcfg.clientCount
  val commandBeatsValid = io.command.payload.beats =/= 0 &&
    io.command.payload.beats <= xcfg.maxBurstBeats
  val commandAddressValid = commandPhysical.msb === False &&
    commandEnd <= (BigInt(1) << cfg.axiAddressWidth)
  val commandValid = commandBankValid && commandClientValid &&
    commandAligned && commandBeatsValid && commandAddressValid

  // One logical data stream is active because AXI4 removed WID. Other slots
  // can independently wait for B, so memory response latency remains hidden.
  val dataActive = Reg(Bool()) init False
  val rejectMode = Reg(Bool()) init False
  val padMode = Reg(Bool()) init False
  val activeSlot = Reg(UInt(cfg.slotWidth bits)) init 0
  val activeClient = Reg(UInt(xcfg.clientWidth bits)) init 0
  val activeTransaction = Reg(UInt(xcfg.transactionIdWidth bits)) init 0
  val activeTag = Reg(UInt(xcfg.tagWidth bits)) init 0
  val activeTotalBeats = Reg(UInt(xcfg.beatCountWidth bits)) init 0
  val logicalRemaining = Reg(UInt(xcfg.beatCountWidth bits)) init 0
  val burstAddress = Reg(UInt(cfg.axiAddressWidth bits)) init 0
  val burstLength = Reg(UInt(xcfg.beatCountWidth bits)) init 0
  val burstRemaining = Reg(UInt(xcfg.beatCountWidth bits)) init 0
  val secondBurstBeats = Reg(UInt(xcfg.beatCountWidth bits)) init 0
  val awPending = Reg(Bool()) init False
  val dataPhase = Reg(Bool()) init False

  val rejectValid = Reg(Bool()) init False
  val rejectPayload = Reg(BitNetLocalBankWriteResponse(xcfg))

  io.command.ready := !dataActive && !rejectValid &&
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
    activeClient := io.command.payload.client
    activeTransaction := io.command.payload.transactionId
    activeTag := io.command.payload.tag
    activeTotalBeats := io.command.payload.beats
    logicalRemaining := io.command.payload.beats
    rejectPayload.client := io.command.payload.client
    rejectPayload.transactionId := io.command.payload.transactionId
    rejectPayload.tag := io.command.payload.tag
    rejectPayload.bresp := B"11"

    when(commandValid) {
      val slot = freeSlot
      slotValid(slot) := True
      slotClient(slot) := io.command.payload.client
      slotTransaction(slot) := io.command.payload.transactionId
      slotTag(slot) := io.command.payload.tag
      slotPendingB(slot) := Mux(startSecondBeats === 0,
        U(1, 2 bits), U(2, 2 bits))
      slotCompletedAwaitingB(slot) := U(0, 2 bits)
      slotWorstResp(slot) := B"00"

      dataActive := True
      rejectMode := False
      padMode := False
      activeSlot := slot
      burstAddress := commandPhysical.resize(cfg.axiAddressWidth)
      burstLength := startFirstBeats
      burstRemaining := startFirstBeats
      secondBurstBeats := startSecondBeats
      awPending := True
      dataPhase := False
    } otherwise {
      padMode := False
      awPending := False
      dataPhase := False
      // A zero-beat command has no data to drain.
      when(io.command.payload.beats === 0) {
        dataActive := False
        rejectMode := False
        rejectValid := True
      } otherwise {
        dataActive := True
        rejectMode := True
      }
    }
  }

  io.axiAw.valid := dataActive && !rejectMode && awPending
  io.axiAw.payload.id := activeSlot.resized
  io.axiAw.payload.address := burstAddress
  io.axiAw.payload.len := (burstLength - 1).resize(8)
  when(io.axiAw.fire) {
    awPending := False
    dataPhase := True
  }

  io.axiW.valid := dataActive && !rejectMode && dataPhase &&
    (padMode || io.writeData.valid)
  io.axiW.payload.data := Mux(padMode, B(0, xcfg.dataWidth bits),
    io.writeData.payload.data)
  io.axiW.payload.strobe := Mux(padMode, B(0, xcfg.strobeWidth bits),
    io.writeData.payload.strobe)
  // WLAST is a sub-burst boundary, not the logical stream's last bit.
  io.axiW.payload.last := burstRemaining === 1

  io.writeData.ready := dataActive && Mux(rejectMode, True,
    dataPhase && !padMode && io.axiW.ready)

  val expectedBeatIndex =
    (activeTotalBeats - logicalRemaining).resize(xcfg.beatIndexWidth)
  val expectedLogicalLast = logicalRemaining === 1
  val inputDataFire = io.writeData.fire
  val inputMetadataBad = io.writeData.payload.client =/= activeClient ||
    io.writeData.payload.transactionId =/= activeTransaction ||
    io.writeData.payload.tag =/= activeTag ||
    io.writeData.payload.beatIndex =/= expectedBeatIndex
  val inputLastBad = io.writeData.payload.last =/= expectedLogicalLast

  // Invalid commands are consumed locally. Early last terminates the drain
  // because the crossbar also releases its bank lock at that point.
  when(inputDataFire && rejectMode) {
    logicalRemaining := logicalRemaining - 1
    when(io.writeData.payload.last || expectedLogicalLast) {
      dataActive := False
      rejectMode := False
      rejectValid := True
    }
  }

  val validWriteFire = io.axiW.fire
  val completingSubBurst = validWriteFire && burstRemaining === 1
  when(validWriteFire) {
    logicalRemaining := logicalRemaining - 1
    burstRemaining := burstRemaining - 1

    when(!padMode && io.writeData.payload.last && !expectedLogicalLast) {
      padMode := True
    }

    when(burstRemaining === 1) {
      dataPhase := False
      slotCompletedAwaitingB(activeSlot) :=
        slotCompletedAwaitingB(activeSlot) + 1
      when(expectedLogicalLast) {
        dataActive := False
        padMode := False
      } otherwise {
        burstAddress := (burstAddress +
          (burstLength.resize(cfg.axiAddressWidth) << cfg.beatShift))
          .resize(cfg.axiAddressWidth)
        burstLength := secondBurstBeats
        burstRemaining := secondBurstBeats
        secondBurstBeats := U(0, xcfg.beatCountWidth bits)
        awPending := True
      }
    }
  }

  val bIdInRange = if (cfg.maxOutstanding == (1 << cfg.axiIdWidth)) {
    True
  } else {
    io.axiB.payload.id < U(cfg.maxOutstanding, cfg.axiIdWidth bits)
  }
  val safeBSlot = io.axiB.payload.id(cfg.slotWidth - 1 downto 0)
  val bSlotValid = bIdInRange && slotValid(safeBSlot)
  val bHasCompletedBurst = bSlotValid &&
    slotCompletedAwaitingB(safeBSlot) =/= 0
  val bIsLogicalFinal = bSlotValid && slotPendingB(safeBSlot) === 1
  val sameSlotCompleting = completingSubBurst &&
    safeBSlot === activeSlot

  def worseResp(previous: Bits, current: Bits): Bits =
    Mux(previous.asUInt >= current.asUInt, previous, current)

  val accumulatedBResp = worseResp(slotWorstResp(safeBSlot),
    io.axiB.payload.resp)
  io.response.valid := rejectValid ||
    (io.axiB.valid && bHasCompletedBurst && bIsLogicalFinal)
  io.response.payload := rejectPayload
  when(!rejectValid) {
    io.response.payload.client := slotClient(safeBSlot)
    io.response.payload.transactionId := slotTransaction(safeBSlot)
    io.response.payload.tag := slotTag(safeBSlot)
    io.response.payload.bresp := accumulatedBResp
  }

  // A same-cycle WLAST is made visible to B on the following cycle. This
  // removes a read/modify/write collision in the completed-burst counter.
  val bCanConsumeWithoutResponse = !bSlotValid ||
    (bHasCompletedBurst && !bIsLogicalFinal)
  val bCanConsumeFinal = bHasCompletedBurst && bIsLogicalFinal &&
    !rejectValid && io.response.ready
  io.axiB.ready := !sameSlotCompleting &&
    (bCanConsumeWithoutResponse || bCanConsumeFinal)

  when(rejectValid && io.response.fire) { rejectValid := False }

  val bFire = io.axiB.fire
  val acceptedBFire = bFire && bHasCompletedBurst
  val badBFire = bFire && !bHasCompletedBurst
  when(acceptedBFire) {
    val slot = safeBSlot
    slotCompletedAwaitingB(slot) := slotCompletedAwaitingB(slot) - 1
    slotPendingB(slot) := slotPendingB(slot) - 1
    slotWorstResp(slot) := accumulatedBResp
    when(bIsLogicalFinal) {
      slotValid(slot) := False
      slotPendingB(slot) := U(0, 2 bits)
      slotCompletedAwaitingB(slot) := U(0, 2 bits)
    }
  }

  val commandFault = Reg(Bool()) init False
  val dataFault = Reg(Bool()) init False
  val responseFault = Reg(Bool()) init False
  val axiFault = Reg(Bool()) init False
  when(io.command.fire && !commandValid) { commandFault := True }
  when(inputDataFire && (inputMetadataBad || inputLastBad)) { dataFault := True }
  when(io.writeData.valid && !dataActive) { dataFault := True }
  // Observing B before its WLAST is an AXI protocol error. Do not consume it;
  // a compliant slave will hold BVALID until the adapter becomes ready.
  when((io.axiB.valid && bSlotValid && !bHasCompletedBurst &&
      !sameSlotCompleting) || badBFire) {
    responseFault := True
  }
  when(acceptedBFire && io.axiB.payload.resp =/= B"00") { axiFault := True }
  io.commandFaultSticky := commandFault
  io.dataProtocolFaultSticky := dataFault
  io.responseProtocolFaultSticky := responseFault
  io.axiResponseFaultSticky := axiFault
  io.outstanding := CountOne(slotValid.asBits)
    .resize(log2Up(cfg.maxOutstanding + 1))
}

object GenerateBitNetAxiBankWriteAdapter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-axi-bank-write-adapter")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetAxiBankWriteAdapter())
}
