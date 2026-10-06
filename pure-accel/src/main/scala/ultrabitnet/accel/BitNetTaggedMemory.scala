package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Tagged five-bank memory protocol used between native sequencers and AXI
  * adapters.  Addresses on the logical side are byte addresses in the
  * compute-visible interleaved image.  One logical beat contains one 128-bit
  * beat from PL DDR and one from each of the four PS HP ports.
  *
  * The adapters below deliberately stop at a bank-local protocol.  A board
  * top adds the 40-bit PS high-DDR bases and translates commands/responses to
  * AXI4.  Keeping the bank-local address here prevents a 34-bit model offset
  * from being confused with a physical 0x8_0000_0000 PS address.
  */
case class BitNetMemoryConfig(
    addressWidth: Int = 34,
    transactionIdWidth: Int = 8,
    bankCount: Int = 5,
    bankDataWidth: Int = 128,
    maxBurstBeats: Int = 256,
    responseFifoDepth: Int = 16
) {
  require(addressWidth >= 32)
  require(transactionIdWidth >= 4)
  require(bankCount == 5,
    "AXU3EGB native production fabric is one MIG plus four HP banks")
  require(bankDataWidth == 128,
    "all five AXU3EGB native bank ports use 128-bit beats")
  require(maxBurstBeats > 0 && isPow2(maxBurstBeats))
  require(responseFifoDepth >= 2 && isPow2(responseFifoDepth))

  val bankBeatBytes: Int = bankDataWidth / 8
  val logicalDataWidth: Int = bankDataWidth * bankCount
  val logicalBeatBytes: Int = logicalDataWidth / 8
  val beatCountWidth: Int = log2Up(maxBurstBeats + 1)
  val beatIndexWidth: Int = log2Up(maxBurstBeats)
}

case class BitNetLogicalBurst(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val address = UInt(cfg.addressWidth bits)
  /** Number of complete 640-bit logical beats; zero is illegal. */
  val beats = UInt(cfg.beatCountWidth bits)
}

case class BitNetBankBurst(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  /** Bank-local byte address. Physical PS/PL base addition is board-specific. */
  val address = UInt(cfg.addressWidth bits)
  val beats = UInt(cfg.beatCountWidth bits)
}

case class BitNetBankReadBeat(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val beatIndex = UInt(cfg.beatIndexWidth bits)
  val data = Bits(cfg.bankDataWidth bits)
  val last = Bool()
  /** AXI RRESP was not OKAY. */
  val error = Bool()
}

case class BitNetLogicalReadBeat(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val beatIndex = UInt(cfg.beatIndexWidth bits)
  val data = Bits(cfg.logicalDataWidth bits)
  val last = Bool()
  val error = Bool()
  val tagMismatch = Bool()
}

case class BitNetLogicalWriteBeat(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val beatIndex = UInt(cfg.beatIndexWidth bits)
  val data = Bits(cfg.logicalDataWidth bits)
  val strobe = Bits(cfg.logicalBeatBytes bits)
  val last = Bool()
}

case class BitNetBankWriteBeat(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val beatIndex = UInt(cfg.beatIndexWidth bits)
  val data = Bits(cfg.bankDataWidth bits)
  val strobe = Bits(cfg.bankBeatBytes bits)
  val last = Bool()
}

case class BitNetBankWriteResponse(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  /** AXI BRESP was not OKAY. */
  val error = Bool()
}

case class BitNetLogicalWriteResponse(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val error = Bool()
  val tagMismatch = Bool()
}

case class BitNetCommandFault(cfg: BitNetMemoryConfig) extends Bundle {
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val address = UInt(cfg.addressWidth bits)
  val zeroLength = Bool()
  val misaligned = Bool()
  val tooLong = Bool()
  val addressOverflow = Bool()
}

/** Pure-Scala address/layout oracle used by image linkers and tests. */
object BitNetFiveBankLayout {
  def bankAddress(cfg: BitNetMemoryConfig, logicalAddress: BigInt): BigInt = {
    require(logicalAddress >= 0)
    require(logicalAddress % cfg.logicalBeatBytes == 0,
      s"logical address must be ${cfg.logicalBeatBytes}-byte aligned")
    logicalAddress / cfg.bankCount
  }

  def join(bankBeats: Seq[Seq[Byte]]): Seq[Byte] = {
    require(bankBeats.length == 5)
    require(bankBeats.forall(_.length == 16))
    bankBeats.head.indices.flatMap(byte => bankBeats.map(_(byte)))
  }

  def split(logicalBeat: Seq[Byte]): Seq[Seq[Byte]] = {
    require(logicalBeat.length == 80)
    (0 until 5).map(bank => logicalBeat.indices.collect {
      case index if index % 5 == bank => logicalBeat(index)
    })
  }
}

/**
  * Lockstep command mapper plus tagged response assembler.
  *
  * Multiple bursts may be outstanding. Each AXI adapter must issue them with
  * the supplied transaction id and preserve beat order within that id. The
  * per-bank FIFOs absorb latency skew; id/index/last comparisons make any
  * cross-bank ordering violation fail visibly instead of silently mixing data.
  */
class BitNetFiveBankReadFabric(
    cfg: BitNetMemoryConfig = BitNetMemoryConfig()) extends Component {
  val io = new Bundle {
    val logicalCommand = slave(Stream(BitNetLogicalBurst(cfg)))
    val bankCommand = Vec(master(Stream(BitNetBankBurst(cfg))), cfg.bankCount)
    val bankData = Vec(slave(Stream(BitNetBankReadBeat(cfg))), cfg.bankCount)
    val logicalData = master(Stream(BitNetLogicalReadBeat(cfg)))
    val commandFault = master(Flow(BitNetCommandFault(cfg)))
    val responseFaultSticky = out Bool()
  }

  private val calculationWidth = cfg.addressWidth + cfg.beatCountWidth + 1
  val zeroLength = io.logicalCommand.payload.beats === 0
  val tooLong = io.logicalCommand.payload.beats > cfg.maxBurstBeats
  val misaligned =
    (io.logicalCommand.payload.address % cfg.logicalBeatBytes) =/= 0
  val logicalBytes =
    io.logicalCommand.payload.beats.resize(calculationWidth) * cfg.logicalBeatBytes
  val logicalEnd = io.logicalCommand.payload.address.resize(calculationWidth) + logicalBytes
  val addressOverflow = logicalEnd > U(BigInt(1) << cfg.addressWidth, calculationWidth bits)
  val commandOk = !zeroLength && !tooLong && !misaligned && !addressOverflow

  val commandFork = new StreamFork(BitNetLogicalBurst(cfg), cfg.bankCount,
    synchronous = true)
  commandFork.io.input.valid := io.logicalCommand.valid && commandOk
  commandFork.io.input.payload := io.logicalCommand.payload
  io.logicalCommand.ready := Mux(commandOk, commandFork.io.input.ready, True)

  val logicalBeatAddress = io.logicalCommand.payload.address / cfg.logicalBeatBytes
  val bankAddress =
    (logicalBeatAddress << log2Up(cfg.bankBeatBytes)).resize(cfg.addressWidth)
  for (bank <- 0 until cfg.bankCount) {
    io.bankCommand(bank).arbitrationFrom(commandFork.io.outputs(bank))
    io.bankCommand(bank).payload.transactionId :=
      commandFork.io.outputs(bank).payload.transactionId
    io.bankCommand(bank).payload.address := bankAddress
    io.bankCommand(bank).payload.beats := commandFork.io.outputs(bank).payload.beats
  }

  io.commandFault.valid := io.logicalCommand.fire && !commandOk
  io.commandFault.payload.transactionId := io.logicalCommand.payload.transactionId
  io.commandFault.payload.address := io.logicalCommand.payload.address
  io.commandFault.payload.zeroLength := zeroLength
  io.commandFault.payload.misaligned := misaligned
  io.commandFault.payload.tooLong := tooLong
  io.commandFault.payload.addressOverflow := addressOverflow

  val responseFifos = Array.fill(cfg.bankCount)(
    StreamFifo(BitNetBankReadBeat(cfg), cfg.responseFifoDepth)
  )
  for (bank <- 0 until cfg.bankCount) {
    responseFifos(bank).io.push << io.bankData(bank)
  }
  val joined = StreamJoin(Vec(responseFifos.map(_.io.pop.toEvent())))
  io.logicalData.arbitrationFrom(joined)

  val reference = responseFifos.head.io.pop.payload
  val sameId = responseFifos.tail
    .map(_.io.pop.payload.transactionId === reference.transactionId)
    .reduce(_ && _)
  val sameIndex = responseFifos.tail
    .map(_.io.pop.payload.beatIndex === reference.beatIndex)
    .reduce(_ && _)
  val sameLast = responseFifos.tail
    .map(_.io.pop.payload.last === reference.last)
    .reduce(_ && _)
  val responseError = responseFifos
    .map(_.io.pop.payload.error)
    .reduce(_ || _)

  val bankBytes = responseFifos.map(_.io.pop.payload.data.subdivideIn(8 bits))
  val logicalBytesOut = Vec(Bits(8 bits), cfg.logicalBeatBytes)
  for (byte <- 0 until cfg.bankBeatBytes; bank <- 0 until cfg.bankCount) {
    logicalBytesOut(byte * cfg.bankCount + bank) := bankBytes(bank)(byte)
  }
  io.logicalData.payload.transactionId := reference.transactionId
  io.logicalData.payload.beatIndex := reference.beatIndex
  io.logicalData.payload.data := logicalBytesOut.asBits
  io.logicalData.payload.last := reference.last
  io.logicalData.payload.tagMismatch := !sameId || !sameIndex || !sameLast
  io.logicalData.payload.error := responseError || io.logicalData.payload.tagMismatch

  val responseFaultSticky = Reg(Bool()) init False
  when(io.logicalData.fire && io.logicalData.payload.error) {
    responseFaultSticky := True
  }
  io.responseFaultSticky := responseFaultSticky
}

/** Write-side command mapper, byte de-interleaver and tagged BRESP join. */
class BitNetFiveBankWriteFabric(
    cfg: BitNetMemoryConfig = BitNetMemoryConfig()) extends Component {
  val io = new Bundle {
    val logicalCommand = slave(Stream(BitNetLogicalBurst(cfg)))
    val logicalData = slave(Stream(BitNetLogicalWriteBeat(cfg)))
    val bankCommand = Vec(master(Stream(BitNetBankBurst(cfg))), cfg.bankCount)
    val bankData = Vec(master(Stream(BitNetBankWriteBeat(cfg))), cfg.bankCount)
    val bankResponse = Vec(slave(Stream(BitNetBankWriteResponse(cfg))), cfg.bankCount)
    val logicalResponse = master(Stream(BitNetLogicalWriteResponse(cfg)))
    val commandFault = master(Flow(BitNetCommandFault(cfg)))
    val responseFaultSticky = out Bool()
  }

  private val calculationWidth = cfg.addressWidth + cfg.beatCountWidth + 1
  val zeroLength = io.logicalCommand.payload.beats === 0
  val tooLong = io.logicalCommand.payload.beats > cfg.maxBurstBeats
  val misaligned =
    (io.logicalCommand.payload.address % cfg.logicalBeatBytes) =/= 0
  val logicalBytes =
    io.logicalCommand.payload.beats.resize(calculationWidth) * cfg.logicalBeatBytes
  val logicalEnd = io.logicalCommand.payload.address.resize(calculationWidth) + logicalBytes
  val addressOverflow = logicalEnd > U(BigInt(1) << cfg.addressWidth, calculationWidth bits)
  val commandOk = !zeroLength && !tooLong && !misaligned && !addressOverflow

  val commandFork = new StreamFork(BitNetLogicalBurst(cfg), cfg.bankCount,
    synchronous = true)
  commandFork.io.input.valid := io.logicalCommand.valid && commandOk
  commandFork.io.input.payload := io.logicalCommand.payload
  io.logicalCommand.ready := Mux(commandOk, commandFork.io.input.ready, True)
  val logicalBeatAddress = io.logicalCommand.payload.address / cfg.logicalBeatBytes
  val bankAddress =
    (logicalBeatAddress << log2Up(cfg.bankBeatBytes)).resize(cfg.addressWidth)
  for (bank <- 0 until cfg.bankCount) {
    io.bankCommand(bank).arbitrationFrom(commandFork.io.outputs(bank))
    io.bankCommand(bank).payload.transactionId :=
      commandFork.io.outputs(bank).payload.transactionId
    io.bankCommand(bank).payload.address := bankAddress
    io.bankCommand(bank).payload.beats := commandFork.io.outputs(bank).payload.beats
  }

  io.commandFault.valid := io.logicalCommand.fire && !commandOk
  io.commandFault.payload.transactionId := io.logicalCommand.payload.transactionId
  io.commandFault.payload.address := io.logicalCommand.payload.address
  io.commandFault.payload.zeroLength := zeroLength
  io.commandFault.payload.misaligned := misaligned
  io.commandFault.payload.tooLong := tooLong
  io.commandFault.payload.addressOverflow := addressOverflow

  val dataFork = new StreamFork(NoData(), cfg.bankCount, synchronous = true)
  dataFork.io.input.arbitrationFrom(io.logicalData)
  val logicalDataBytes = io.logicalData.payload.data.subdivideIn(8 bits)
  for (bank <- 0 until cfg.bankCount) {
    io.bankData(bank).arbitrationFrom(dataFork.io.outputs(bank))
    io.bankData(bank).payload.transactionId := io.logicalData.payload.transactionId
    io.bankData(bank).payload.beatIndex := io.logicalData.payload.beatIndex
    io.bankData(bank).payload.last := io.logicalData.payload.last
    val bytes = Vec(Bits(8 bits), cfg.bankBeatBytes)
    val strobes = Bits(cfg.bankBeatBytes bits)
    for (byte <- 0 until cfg.bankBeatBytes) {
      bytes(byte) := logicalDataBytes(byte * cfg.bankCount + bank)
      strobes(byte) := io.logicalData.payload.strobe(byte * cfg.bankCount + bank)
    }
    io.bankData(bank).payload.data := bytes.asBits
    io.bankData(bank).payload.strobe := strobes
  }

  val responseFifos = Array.fill(cfg.bankCount)(
    StreamFifo(BitNetBankWriteResponse(cfg), cfg.responseFifoDepth)
  )
  for (bank <- 0 until cfg.bankCount) {
    responseFifos(bank).io.push << io.bankResponse(bank)
  }
  val joined = StreamJoin(Vec(responseFifos.map(_.io.pop.toEvent())))
  io.logicalResponse.arbitrationFrom(joined)
  val reference = responseFifos.head.io.pop.payload
  val sameId = responseFifos.tail
    .map(_.io.pop.payload.transactionId === reference.transactionId)
    .reduce(_ && _)
  val responseError = responseFifos
    .map(_.io.pop.payload.error)
    .reduce(_ || _)
  io.logicalResponse.payload.transactionId := reference.transactionId
  io.logicalResponse.payload.tagMismatch := !sameId
  io.logicalResponse.payload.error := responseError || !sameId

  val responseFaultSticky = Reg(Bool()) init False
  when(io.logicalResponse.fire && io.logicalResponse.payload.error) {
    responseFaultSticky := True
  }
  io.responseFaultSticky := responseFaultSticky
}
