package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentProjectionScaleFault {
  val None = 0x00
  val InvalidCommand = 0xe0
  val ReadProtocol = 0xe1
  val Conversion = 0xe2
}

case class BitNetResidentProjectionScaleStagerConfig(
    model: BitNetConfig = BitNetConfig.Production,
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    bank: Int = 0,
    projectionScaleBase: BigInt =
      BitNetFiveBankStripedLayout().auxBases.head +
        BitNetFiveBankStripedLayout.ProjectionScaleOffset,
    epochWidth: Int = 8
) {
  require(model == BitNetConfig.Production)
  require(crossbar.dataWidth == 128)
  require(crossbar.maxBurstBeats >= 2)
  require(bank >= 0 && bank < crossbar.bankCount)
  require(epochWidth >= 4)
  require((projectionScaleBase & 1) == 0)
  require(projectionScaleBase >= 0)
  require(projectionScaleBase + BigInt(model.layerCount) * 14 <=
    (BigInt(1) << crossbar.localAddressWidth))

  val scaleCount: Int = BitNetResidentMatrixScaleIndex.Count
  val scaleIndexWidth: Int = log2Up(scaleCount)
  val windowBeats: Int = 2
  val windowBytes: Int = windowBeats * crossbar.dataWidth / 8
}

/** One layer-scale transaction generated and consumed entirely in PL. */
case class BitNetResidentProjectionScaleRequest(
    cfg: BitNetResidentProjectionScaleStagerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
}

case class BitNetResidentProjectionScaleSet(
    cfg: BitNetResidentProjectionScaleStagerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  /** Q,K,V,O,Gate,Up,Down, in the order used by MatrixStageCommand. */
  val valueQ16 = Vec(UInt(32 bits), cfg.scaleCount)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

object BitNetResidentProjectionScaleStagerMath {
  final case class Window(address: BigInt, firstWord: Int)

  def window(base: BigInt, layer: Int): Window = {
    require(base >= 0 && (base & 1) == 0)
    require(layer >= 0)
    val exact = base + BigInt(layer) * 14
    Window(exact & ~BigInt(15), ((exact & 15) / 2).toInt)
  }
}

/**
  * Fetches and converts the seven BF16 projection scales of one resident
  * layer.  The 14-byte rows are intentionally not padded in AUX.BIN, so a row
  * can straddle a 128-bit memory beat.  Two aligned beats form a bounded
  * 32-byte window; seven words are selected and passed through the existing
  * bit-accurate BF16-to-Q16.16 converter over seven cycles.
  *
  * The result is held as one tagged set and can be reused by Q/K/V/O/Gate/Up/
  * Down.  No CPU descriptor, floating-point instruction, AXI callback or
  * per-projection scale read exists on this path.
  */
class BitNetResidentProjectionScaleStager(
    cfg: BitNetResidentProjectionScaleStagerConfig =
      BitNetResidentProjectionScaleStagerConfig()) extends Component {
  private val model = cfg.model
  private val crossbar = cfg.crossbar

  val io = new Bundle {
    val request = slave(Stream(
      BitNetResidentProjectionScaleRequest(cfg)))
    val readRequest = master(Stream(
      BitNetClientBankReadRequest(crossbar)))
    val readResponse = slave(Stream(
      BitNetClientBankReadResponse(crossbar)))
    val scales = master(Stream(BitNetResidentProjectionScaleSet(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedRequests = out UInt(32 bits)
    val completedRequests = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssueRead, ReceiveRead, Convert, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentProjectionScaleRequest(cfg)) init
    BitNetResidentProjectionScaleRequest(cfg).getZero
  val alignedAddress = Reg(UInt(crossbar.localAddressWidth bits)) init 0
  val firstWord = Reg(UInt(4 bits)) init 0
  val expectedBeat = Reg(UInt(crossbar.beatIndexWidth bits)) init 0
  val readTag = Reg(UInt(crossbar.tagWidth bits)) init 0
  val transactionId = Reg(UInt(crossbar.transactionIdWidth bits)) init 0
  val readWindow = Vec(Reg(Bits(crossbar.dataWidth bits)) init 0,
    cfg.windowBeats)
  val inputIndex = Reg(UInt(4 bits)) init 0
  val outputIndex = Reg(UInt(4 bits)) init 0
  val converted = Vec(Reg(UInt(32 bits)) init 0, cfg.scaleCount)
  val jobFault = RegInit(False)
  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0

  io.busy := state =/= State.Idle
  io.protocolFaultSticky := stickyFault
  io.acceptedRequests := acceptedCount
  io.completedRequests := completedCount

  val layerLegal = Bool()
  if ((1 << model.layerWidth) == model.layerCount) {
    layerLegal := True
  } else {
    layerLegal := io.request.payload.layer < model.layerCount
  }
  val exactAddress = UInt(crossbar.localAddressWidth bits)
  exactAddress := (U(cfg.projectionScaleBase,
    crossbar.localAddressWidth bits) +
    io.request.payload.layer.resize(crossbar.localAddressWidth) * 14).resized
  val alignedCandidate = UInt(crossbar.localAddressWidth bits)
  alignedCandidate := exactAddress
  alignedCandidate(3 downto 0) := 0

  io.request.ready := state === State.Idle
  when(io.request.fire) {
    held := io.request.payload
    alignedAddress := alignedCandidate
    firstWord := (exactAddress(3 downto 1)).resized
    expectedBeat := 0
    inputIndex := 0
    outputIndex := 0
    converted.foreach(_ := 0)
    jobFault := !layerLegal
    jobFaultCode := 0
    transactionId := (io.request.payload.requestId.asBits ^
      io.request.payload.epoch.resize(model.requestIdWidth).asBits)
      .asUInt.resize(crossbar.transactionIdWidth)
    acceptedCount := acceptedCount + 1
    when(layerLegal) {
      state := State.IssueRead
    } otherwise {
      jobFaultCode := B(
        BitNetResidentProjectionScaleFault.InvalidCommand, 8 bits)
      stickyFault := True
      state := State.Report
    }
  }

  io.readRequest.valid := state === State.IssueRead
  io.readRequest.payload.bank := cfg.bank
  io.readRequest.payload.transactionId := transactionId
  io.readRequest.payload.tag :=
    (held.requestId.resize(crossbar.tagWidth) ^
      held.epoch.resize(crossbar.tagWidth) ^
      held.layer.resize(crossbar.tagWidth)).resized
  io.readRequest.payload.localAddress := alignedAddress
  io.readRequest.payload.beats := cfg.windowBeats

  when(io.readRequest.fire) {
    readTag := io.readRequest.payload.tag
    expectedBeat := 0
    state := State.ReceiveRead
  }

  io.readResponse.ready := state === State.ReceiveRead
  val responseExpectedLast = expectedBeat === cfg.windowBeats - 1
  val responseLegal =
    io.readResponse.payload.bank === cfg.bank &&
      io.readResponse.payload.transactionId === transactionId &&
      io.readResponse.payload.tag === readTag &&
      io.readResponse.payload.beatIndex === expectedBeat &&
      io.readResponse.payload.last === responseExpectedLast &&
      !io.readResponse.payload.fault &&
      io.readResponse.payload.rresp === B"2'b00"

  when(io.readResponse.fire) {
    readWindow(expectedBeat.resized) := io.readResponse.payload.data
    when(!responseLegal) {
      when(!jobFault) {
        jobFaultCode := B(
          BitNetResidentProjectionScaleFault.ReadProtocol, 8 bits)
      }
      jobFault := True
      stickyFault := True
    }
    when(responseExpectedLast) {
      inputIndex := 0
      outputIndex := 0
      state := State.Convert
    } otherwise {
      expectedBeat := expectedBeat + 1
    }
  }

  val packedWindow = readWindow(1) ## readWindow(0)
  val words = packedWindow.subdivideIn(16 bits)
  val selectedWord = words((firstWord + inputIndex).resized)
  val converter = new NativeBf16Q16Staging
  converter.io.input.valid := state === State.Convert &&
    inputIndex < cfg.scaleCount
  converter.io.input.payload.raw := selectedWord
  converter.io.input.payload.first := inputIndex === 0
  converter.io.input.payload.last := inputIndex === cfg.scaleCount - 1
  when(converter.io.input.fire) {
    inputIndex := inputIndex + 1
  }

  converter.io.output.ready := state === State.Convert
  val conversionFault = converter.io.output.payload.special ||
    converter.io.output.payload.overflow ||
    converter.io.output.payload.value <= 0
  when(converter.io.output.fire) {
    converted(outputIndex.resized) := converter.io.output.payload.value.asUInt
    when(conversionFault) {
      when(!jobFault) {
        jobFaultCode := B(
          BitNetResidentProjectionScaleFault.Conversion, 8 bits)
      }
      jobFault := True
      stickyFault := True
    }
    when(outputIndex === cfg.scaleCount - 1) {
      state := State.Report
    } otherwise {
      outputIndex := outputIndex + 1
    }
  }

  io.scales.valid := state === State.Report
  io.scales.payload.requestId := held.requestId
  io.scales.payload.layer := held.layer
  io.scales.payload.epoch := held.epoch
  for (index <- 0 until cfg.scaleCount) {
    io.scales.payload.valueQ16(index) := Mux(jobFault,
      U(0, 32 bits), converted(index))
  }
  io.scales.payload.success := !jobFault
  io.scales.payload.faultCode := jobFaultCode
  when(io.scales.fire) {
    completedCount := completedCount + 1
    state := State.Idle
  }
}

object GenerateBitNetResidentProjectionScaleStager extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-projection-scale-stager")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentProjectionScaleStager())
}
