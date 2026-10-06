package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Fixed physical boundary of the one BitNet memory complex. */
case class BitNetFiveBankAxiMemoryComplexConfig(
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    physicalBase: Vector[BigInt] = Vector.fill(5)(BigInt(0)),
    axiAddressWidth: Int = 40,
    axiIdWidth: Int = 4,
    maxOutstanding: Int = 16
) {
  require(physicalBase.length == crossbar.bankCount)
  require(physicalBase.forall(base => base >= 0 && (base & 0xf) == 0))

  def read(bank: Int): BitNetAxiBankReadAdapterConfig =
    BitNetAxiBankReadAdapterConfig(
      crossbar = crossbar,
      bankId = bank,
      physicalBase = physicalBase(bank),
      axiAddressWidth = axiAddressWidth,
      axiIdWidth = axiIdWidth,
      maxOutstanding = maxOutstanding)

  def write(bank: Int): BitNetAxiBankWriteAdapterConfig =
    BitNetAxiBankWriteAdapterConfig(
      crossbar = crossbar,
      bankId = bank,
      physicalBase = physicalBase(bank),
      axiAddressWidth = axiAddressWidth,
      axiIdWidth = axiIdWidth,
      maxOutstanding = maxOutstanding)
}

/**
  * The sole six-client by five-bank AXI memory complex.
  *
  * Bank 0 is the PL DDR window and banks 1..4 are the four logical PS HP
  * windows.  They remain independent AXI masters here, but banks 1..4 still
  * share the board's one physical PS DDR controller; this hierarchy does not
  * pretend they are four separate DRAM devices.  All request arbitration,
  * AXI-ID ownership, 4-KiB splitting and response routing are in PL.
  *
  * The exposed channels are intentionally raw AXI payload Streams.  A board
  * wrapper supplies the fixed AXI4 INCR/SIZE/cache/protection attributes and
  * any required width/clock conversion.  No PS callback or per-layer DMA
  * command is present in this component.
  */
class BitNetFiveBankAxiMemoryComplex(
    cfg: BitNetFiveBankAxiMemoryComplexConfig =
      BitNetFiveBankAxiMemoryComplexConfig()) extends Component {
  private val xcfg = cfg.crossbar
  private val commonRead = cfg.read(0)
  private val commonWrite = cfg.write(0)

  val io = new Bundle {
    val clientReadRequest = Vec(
      slave(Stream(BitNetClientBankReadRequest(xcfg))), xcfg.clientCount)
    val directMatrixReadRequest = Vec(
      slave(Stream(BitNetClientBankReadRequest(xcfg))), xcfg.bankCount)
    val directLmReadRequest = Vec(
      slave(Stream(BitNetClientBankReadRequest(xcfg))), xcfg.bankCount)
    val clientReadResponse = Vec(
      master(Stream(BitNetClientBankReadResponse(xcfg))), xcfg.clientCount)
    val clientWriteRequest = Vec(
      slave(Stream(BitNetClientBankWriteRequest(xcfg))), xcfg.clientCount)
    val clientWriteData = Vec(
      slave(Stream(BitNetClientBankWriteData(xcfg))), xcfg.clientCount)
    val clientWriteResponse = Vec(
      master(Stream(BitNetClientBankWriteResponse(xcfg))), xcfg.clientCount)

    val axiAr = Vec(master(Stream(BitNetAxiReadAddress(commonRead))),
      xcfg.bankCount)
    val axiR = Vec(slave(Stream(BitNetAxiReadData(commonRead))),
      xcfg.bankCount)
    val axiAw = Vec(master(Stream(BitNetAxiWriteAddress(commonWrite))),
      xcfg.bankCount)
    val axiW = Vec(master(Stream(BitNetAxiWriteData(commonWrite))),
      xcfg.bankCount)
    val axiB = Vec(slave(Stream(BitNetAxiWriteResponse(commonWrite))),
      xcfg.bankCount)

    val readOutstanding = Vec(
      out UInt(log2Up(cfg.maxOutstanding + 1) bits), xcfg.bankCount)
    val readCommandExpansionBusy = Vec(out Bool(), xcfg.bankCount)
    val writeOutstanding = Vec(
      out UInt(log2Up(cfg.maxOutstanding + 1) bits), xcfg.bankCount)
    val readFaultSticky = out Bits(xcfg.bankCount bits)
    val writeFaultSticky = out Bits(xcfg.bankCount bits)
    val routingFaultSticky = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val crossbar = new BitNetBankLocalCrossbar(xcfg)
  for (client <- 0 until xcfg.clientCount) {
    crossbar.io.clientReadRequest(client) << io.clientReadRequest(client)
    io.clientReadResponse(client) << crossbar.io.clientReadResponse(client)
    crossbar.io.clientWriteRequest(client) << io.clientWriteRequest(client)
    crossbar.io.clientWriteData(client) << io.clientWriteData(client)
    io.clientWriteResponse(client) << crossbar.io.clientWriteResponse(client)
  }
  for (bank <- 0 until xcfg.bankCount) {
    crossbar.io.directMatrixReadRequest(bank) <<
      io.directMatrixReadRequest(bank)
    crossbar.io.directLmReadRequest(bank) << io.directLmReadRequest(bank)
  }

  val readAdapters = Array.tabulate(xcfg.bankCount) { bank =>
    new BitNetAxiBankReadAdapter(cfg.read(bank))
  }
  val writeAdapters = Array.tabulate(xcfg.bankCount) { bank =>
    new BitNetAxiBankWriteAdapter(cfg.write(bank))
  }

  val adapterProtocolFault = Bits(xcfg.bankCount bits)
  for (bank <- 0 until xcfg.bankCount) {
    val read = readAdapters(bank)
    val write = writeAdapters(bank)

    read.io.command << crossbar.io.bankReadCommand(bank)
    crossbar.io.bankReadResponse(bank) << read.io.response
    io.axiAr(bank) << read.io.axiAr
    read.io.axiR << io.axiR(bank)

    write.io.command << crossbar.io.bankWriteCommand(bank)
    write.io.writeData << crossbar.io.bankWriteData(bank)
    crossbar.io.bankWriteResponse(bank) << write.io.response
    io.axiAw(bank) << write.io.axiAw
    io.axiW(bank) << write.io.axiW
    write.io.axiB << io.axiB(bank)

    io.readOutstanding(bank) := read.io.outstanding
    io.readCommandExpansionBusy(bank) := read.io.commandExpansionBusy
    io.writeOutstanding(bank) := write.io.outstanding
    adapterProtocolFault(bank) :=
      read.io.commandFaultSticky ||
        read.io.responseProtocolFaultSticky ||
        write.io.commandFaultSticky ||
        write.io.dataProtocolFaultSticky ||
        write.io.responseProtocolFaultSticky
  }

  io.readFaultSticky := crossbar.io.readResponseFaultSticky
  io.writeFaultSticky := crossbar.io.writeResponseFaultSticky
  io.routingFaultSticky := crossbar.io.responseRoutingFaultSticky
  io.protocolFaultSticky := adapterProtocolFault.orR ||
    crossbar.io.responseRoutingFaultSticky ||
    crossbar.io.readResponseFaultSticky.orR ||
    crossbar.io.writeResponseFaultSticky.orR
}

object GenerateBitNetFiveBankAxiMemoryComplex extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-five-bank-axi-memory-complex")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetFiveBankAxiMemoryComplex())
}
