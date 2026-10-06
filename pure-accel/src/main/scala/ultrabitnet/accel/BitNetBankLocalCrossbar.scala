package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Six-client/five-bank boundary used by the BitNet datapath.
  * The default clients are two matrix engines followed by four SPUs.  Physical
  * AXI address-window translation remains outside this bank-local crossbar.
  */
case class BitNetBankLocalCrossbarConfig(
    matrixEngineClients: Int = 2,
    spuClients: Int = 4,
    bankCount: Int = 5,
    localAddressWidth: Int = 34,
    transactionIdWidth: Int = 8,
    tagWidth: Int = 20,
    dataWidth: Int = 128,
    maxBurstBeats: Int = 256,
    responseFifoDepth: Int = 8
) {
  require(matrixEngineClients > 0 && spuClients > 0)
  require(bankCount == 5,
    "native AXU3EG memory is one PL MIG bank plus four PS HP banks")
  require(localAddressWidth >= 32)
  require(transactionIdWidth >= 4 && tagWidth >= transactionIdWidth)
  require(dataWidth == 128, "bank-local AXU3EG datapaths are 128 bits")
  require(maxBurstBeats > 0 && isPow2(maxBurstBeats))
  require(responseFifoDepth >= 2 && isPow2(responseFifoDepth))

  val clientCount: Int = matrixEngineClients + spuClients
  val clientWidth: Int = scala.math.max(1, log2Up(clientCount))
  val bankWidth: Int = scala.math.max(1, log2Up(bankCount))
  val beatCountWidth: Int = log2Up(maxBurstBeats + 1)
  val beatIndexWidth: Int = scala.math.max(1, log2Up(maxBurstBeats))
  val strobeWidth: Int = dataWidth / 8
}

/** Client-side read request; the address is already local to the named bank. */
case class BitNetClientBankReadRequest(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val bank = UInt(cfg.bankWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val localAddress = UInt(cfg.localAddressWidth bits)
  val beats = UInt(cfg.beatCountWidth bits)
}

/** Command handed to a future bank-specific AXI AR adapter. */
case class BitNetBankReadCommand(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val bank = UInt(cfg.bankWidth bits)
  val client = UInt(cfg.clientWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val localAddress = UInt(cfg.localAddressWidth bits)
  val beats = UInt(cfg.beatCountWidth bits)
}

/** Read beat returned by one bank-specific AXI R adapter. */
case class BitNetBankReadResponse(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val client = UInt(cfg.clientWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val beatIndex = UInt(cfg.beatIndexWidth bits)
  val data = Bits(cfg.dataWidth bits)
  val last = Bool()
  /** Raw AXI RRESP; anything other than OKAY (00) is propagated as a fault. */
  val rresp = Bits(2 bits)
}

case class BitNetClientBankReadResponse(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val bank = UInt(cfg.bankWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val beatIndex = UInt(cfg.beatIndexWidth bits)
  val data = Bits(cfg.dataWidth bits)
  val last = Bool()
  val rresp = Bits(2 bits)
  val fault = Bool()
}

/** Client-side write address request. */
case class BitNetClientBankWriteRequest(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val bank = UInt(cfg.bankWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val localAddress = UInt(cfg.localAddressWidth bits)
  val beats = UInt(cfg.beatCountWidth bits)
}

case class BitNetBankWriteCommand(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val bank = UInt(cfg.bankWidth bits)
  val client = UInt(cfg.clientWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val localAddress = UInt(cfg.localAddressWidth bits)
  val beats = UInt(cfg.beatCountWidth bits)
}

/**
  * Write data names its bank and tag explicitly. A client may therefore have
  * independent bursts active on different banks and interleave their beats.
  */
case class BitNetClientBankWriteData(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val bank = UInt(cfg.bankWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val beatIndex = UInt(cfg.beatIndexWidth bits)
  val data = Bits(cfg.dataWidth bits)
  val strobe = Bits(cfg.strobeWidth bits)
  val last = Bool()
}

case class BitNetBankWriteData(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val client = UInt(cfg.clientWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val beatIndex = UInt(cfg.beatIndexWidth bits)
  val data = Bits(cfg.dataWidth bits)
  val strobe = Bits(cfg.strobeWidth bits)
  val last = Bool()
}

case class BitNetLocalBankWriteResponse(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val client = UInt(cfg.clientWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  /** Raw AXI BRESP; anything other than OKAY (00) is a fault. */
  val bresp = Bits(2 bits)
}

case class BitNetClientBankWriteResponse(cfg: BitNetBankLocalCrossbarConfig)
    extends Bundle {
  val bank = UInt(cfg.bankWidth bits)
  val transactionId = UInt(cfg.transactionIdWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val bresp = Bits(2 bits)
  val fault = Bool()
}

/**
  * Independent 6x5 read/write bank-local crossbar.
  *
  * Commands arbitrate independently at every bank, so an SPU using KV bank 4
  * cannot block an engine streaming weights from bank 0. Read and write
  * responses have one FIFO per bank and are then routed by the echoed client
  * plus transaction/tag. Banks may return in any order. No five-bank lockstep
  * join exists here.
  *
  * Write data is locked to the client whose command won that bank until the
  * declared beat count (or an asserted last) is consumed. A bank can service
  * one write-data burst at a time while other banks proceed independently.
  */
class BitNetBankLocalCrossbar(
    cfg: BitNetBankLocalCrossbarConfig = BitNetBankLocalCrossbarConfig()
) extends Component {
  val io = new Bundle {
    val clientReadRequest = Vec(
      slave(Stream(BitNetClientBankReadRequest(cfg))), cfg.clientCount)
    // Production Matrix/LM requests retain their static bank identity up to
    // the physical arbiter.  They reuse logical response client IDs 0/1.
    val directMatrixReadRequest = Vec(
      slave(Stream(BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val directLmReadRequest = Vec(
      slave(Stream(BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val clientReadResponse = Vec(
      master(Stream(BitNetClientBankReadResponse(cfg))), cfg.clientCount)
    val bankReadCommand = Vec(
      master(Stream(BitNetBankReadCommand(cfg))), cfg.bankCount)
    val bankReadResponse = Vec(
      slave(Stream(BitNetBankReadResponse(cfg))), cfg.bankCount)

    val clientWriteRequest = Vec(
      slave(Stream(BitNetClientBankWriteRequest(cfg))), cfg.clientCount)
    val clientWriteData = Vec(
      slave(Stream(BitNetClientBankWriteData(cfg))), cfg.clientCount)
    val clientWriteResponse = Vec(
      master(Stream(BitNetClientBankWriteResponse(cfg))), cfg.clientCount)
    val bankWriteCommand = Vec(
      master(Stream(BitNetBankWriteCommand(cfg))), cfg.bankCount)
    val bankWriteData = Vec(
      master(Stream(BitNetBankWriteData(cfg))), cfg.bankCount)
    val bankWriteResponse = Vec(
      slave(Stream(BitNetLocalBankWriteResponse(cfg))), cfg.bankCount)

    val readResponseFaultSticky = out Bits(cfg.bankCount bits)
    val writeResponseFaultSticky = out Bits(cfg.bankCount bits)
    /** Invalid echoed client ids are drained and reported instead of deadlocking. */
    val responseRoutingFaultSticky = out Bool()
  }

  // -------------------------------------------------------------------------
  // Independent read-command round-robin arbiters, one per bank.
  // -------------------------------------------------------------------------
  val readCandidates = Array.tabulate(cfg.bankCount, cfg.clientCount) {
    (_, _) => Stream(BitNetBankReadCommand(cfg))
  }
  val directMatrixReadCandidates = Array.fill(cfg.bankCount)(
    Stream(BitNetBankReadCommand(cfg)))
  val directLmReadCandidates = Array.fill(cfg.bankCount)(
    Stream(BitNetBankReadCommand(cfg)))
  for (bank <- 0 until cfg.bankCount) {
    for (client <- 0 until cfg.clientCount) {
      val candidate = readCandidates(bank)(client)
      val source = io.clientReadRequest(client)
      candidate.valid := source.valid && source.payload.bank === bank
      candidate.payload.bank := bank
      candidate.payload.client := client
      candidate.payload.transactionId := source.payload.transactionId
      candidate.payload.tag := source.payload.tag
      candidate.payload.localAddress := source.payload.localAddress
      candidate.payload.beats := source.payload.beats
    }

    def connectDirect(
        candidate: Stream[BitNetBankReadCommand],
        source: Stream[BitNetClientBankReadRequest],
        client: Int): Unit = {
      candidate.valid := source.valid
      candidate.payload.bank := bank
      candidate.payload.client := client
      candidate.payload.transactionId := source.payload.transactionId
      candidate.payload.tag := source.payload.tag
      candidate.payload.localAddress := source.payload.localAddress
      candidate.payload.beats := source.payload.beats
    }
    connectDirect(directMatrixReadCandidates(bank),
      io.directMatrixReadRequest(bank), 0)
    connectDirect(directLmReadCandidates(bank),
      io.directLmReadRequest(bank), 1)

    val selected = StreamArbiterFactory.roundRobin.transactionLock
      .on(Vec(readCandidates(bank) ++ Array(
        directMatrixReadCandidates(bank), directLmReadCandidates(bank))))
    io.bankReadCommand(bank) << selected
    io.directMatrixReadRequest(bank).ready :=
      directMatrixReadCandidates(bank).ready
    io.directLmReadRequest(bank).ready := directLmReadCandidates(bank).ready
  }
  for (client <- 0 until cfg.clientCount) {
    io.clientReadRequest(client).ready := (0 until cfg.bankCount)
      .map(bank => readCandidates(bank)(client).ready &&
        io.clientReadRequest(client).payload.bank === bank)
      .reduce(_ || _)
  }

  // -------------------------------------------------------------------------
  // Write-address arbitration and per-bank data locks.
  // -------------------------------------------------------------------------
  val writeActive = Vec.fill(cfg.bankCount)(Reg(Bool()) init False)
  val writeOwner = Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.clientWidth bits)) init 0)
  val writeTransaction = Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.transactionIdWidth bits)) init 0)
  val writeTag = Vec.fill(cfg.bankCount)(Reg(UInt(cfg.tagWidth bits)) init 0)
  val writeBeats = Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.beatCountWidth bits)) init 0)
  val writeBeatCount = Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.beatCountWidth bits)) init 0)

  val writeCandidates = Array.tabulate(cfg.bankCount, cfg.clientCount) {
    (_, _) => Stream(BitNetBankWriteCommand(cfg))
  }
  for (bank <- 0 until cfg.bankCount) {
    for (client <- 0 until cfg.clientCount) {
      val candidate = writeCandidates(bank)(client)
      val source = io.clientWriteRequest(client)
      candidate.valid := source.valid && source.payload.bank === bank
      candidate.payload.bank := bank
      candidate.payload.client := client
      candidate.payload.transactionId := source.payload.transactionId
      candidate.payload.tag := source.payload.tag
      candidate.payload.localAddress := source.payload.localAddress
      candidate.payload.beats := source.payload.beats
    }
    val selected = StreamArbiterFactory.roundRobin.noLock
      .on(Vec(writeCandidates(bank)))
    io.bankWriteCommand(bank).valid := selected.valid && !writeActive(bank)
    io.bankWriteCommand(bank).payload := selected.payload
    selected.ready := io.bankWriteCommand(bank).ready && !writeActive(bank)

    when(io.bankWriteCommand(bank).fire) {
      writeActive(bank) := True
      writeOwner(bank) := io.bankWriteCommand(bank).payload.client
      writeTransaction(bank) := io.bankWriteCommand(bank).payload.transactionId
      writeTag(bank) := io.bankWriteCommand(bank).payload.tag
      writeBeats(bank) := io.bankWriteCommand(bank).payload.beats
      writeBeatCount(bank) := 0
    }
  }
  for (client <- 0 until cfg.clientCount) {
    io.clientWriteRequest(client).ready := (0 until cfg.bankCount)
      .map(bank => writeCandidates(bank)(client).ready &&
        io.clientWriteRequest(client).payload.bank === bank)
      .reduce(_ || _)
  }

  for (bank <- 0 until cfg.bankCount) {
    val output = io.bankWriteData(bank)
    output.valid := False
    output.payload.client := writeOwner(bank)
    output.payload.transactionId := writeTransaction(bank)
    output.payload.tag := writeTag(bank)
    output.payload.beatIndex := 0
    output.payload.data := 0
    output.payload.strobe := 0
    output.payload.last := False
    for (client <- 0 until cfg.clientCount) {
      val source = io.clientWriteData(client)
      when(writeActive(bank) && writeOwner(bank) === client && source.valid &&
        source.payload.bank === bank &&
        source.payload.transactionId === writeTransaction(bank) &&
        source.payload.tag === writeTag(bank)) {
        output.valid := True
        output.payload.client := client
        output.payload.transactionId := source.payload.transactionId
        output.payload.tag := source.payload.tag
        output.payload.beatIndex := source.payload.beatIndex
        output.payload.data := source.payload.data
        output.payload.strobe := source.payload.strobe
        output.payload.last := source.payload.last
      }
    }

    when(output.fire) {
      writeBeatCount(bank) := writeBeatCount(bank) + 1
      when(output.payload.last ||
        writeBeatCount(bank) + 1 >= writeBeats(bank)) {
        writeActive(bank) := False
      }
    }
  }
  for (client <- 0 until cfg.clientCount) {
    val source = io.clientWriteData(client)
    io.clientWriteData(client).ready := (0 until cfg.bankCount).map { bank =>
      writeActive(bank) && writeOwner(bank) === client &&
        source.payload.bank === bank &&
        source.payload.transactionId === writeTransaction(bank) &&
        source.payload.tag === writeTag(bank) && io.bankWriteData(bank).ready
    }.reduce(_ || _)
  }

  // -------------------------------------------------------------------------
  // Per-bank response FIFOs, followed by lossless client/tag routing.
  // -------------------------------------------------------------------------
  // Bank read responses are wide (the 128-bit data beat plus routing
  // metadata).  The default asynchronous StreamFifo maps every bank into
  // distributed RAM and creates a dense local routing hot spot around the
  // five-bank crossbar.  A synchronous head keeps one-beat-per-cycle
  // throughput while making block-RAM inference legal; the extra first-beat
  // cycle is fully hidden by the existing Stream handshake.
  val readResponseFifos = Array.fill(cfg.bankCount)(
    new StreamFifo(
      BitNetBankReadResponse(cfg),
      cfg.responseFifoDepth,
      withAsyncRead = false))
  readResponseFifos.foreach(
    _.logic.ram.addAttribute("ram_style", "block"))
  val writeResponseFifos = Array.fill(cfg.bankCount)(
    StreamFifo(BitNetLocalBankWriteResponse(cfg), cfg.responseFifoDepth))
  for (bank <- 0 until cfg.bankCount) {
    readResponseFifos(bank).io.push << io.bankReadResponse(bank)
    writeResponseFifos(bank).io.push << io.bankWriteResponse(bank)
  }

  val readResponseCandidates = Array.tabulate(cfg.clientCount, cfg.bankCount) {
    (_, _) => Stream(BitNetClientBankReadResponse(cfg))
  }
  for (client <- 0 until cfg.clientCount) {
    for (bank <- 0 until cfg.bankCount) {
      val source = readResponseFifos(bank).io.pop
      val candidate = readResponseCandidates(client)(bank)
      candidate.valid := source.valid && source.payload.client === client
      candidate.payload.bank := bank
      candidate.payload.transactionId := source.payload.transactionId
      candidate.payload.tag := source.payload.tag
      candidate.payload.beatIndex := source.payload.beatIndex
      candidate.payload.data := source.payload.data
      candidate.payload.last := source.payload.last
      candidate.payload.rresp := source.payload.rresp
      candidate.payload.fault := source.payload.rresp =/= B"00"
    }
    val selected = StreamArbiterFactory.roundRobin.noLock
      .on(Vec(readResponseCandidates(client)))
    io.clientReadResponse(client) << selected
  }

  val writeResponseCandidates = Array.tabulate(cfg.clientCount, cfg.bankCount) {
    (_, _) => Stream(BitNetClientBankWriteResponse(cfg))
  }
  for (client <- 0 until cfg.clientCount) {
    for (bank <- 0 until cfg.bankCount) {
      val source = writeResponseFifos(bank).io.pop
      val candidate = writeResponseCandidates(client)(bank)
      candidate.valid := source.valid && source.payload.client === client
      candidate.payload.bank := bank
      candidate.payload.transactionId := source.payload.transactionId
      candidate.payload.tag := source.payload.tag
      candidate.payload.bresp := source.payload.bresp
      candidate.payload.fault := source.payload.bresp =/= B"00"
    }
    val selected = StreamArbiterFactory.roundRobin.noLock
      .on(Vec(writeResponseCandidates(client)))
    io.clientWriteResponse(client) << selected
  }

  val routingFaultSticky = Reg(Bool()) init False
  for (bank <- 0 until cfg.bankCount) {
    val readPop = readResponseFifos(bank).io.pop
    val readClientValid = readPop.payload.client < cfg.clientCount
    val readReady = (0 until cfg.clientCount)
      .map(client => readResponseCandidates(client)(bank).ready &&
        readPop.payload.client === client)
      .reduce(_ || _)
    readPop.ready := Mux(readClientValid, readReady, True)

    val writePop = writeResponseFifos(bank).io.pop
    val writeClientValid = writePop.payload.client < cfg.clientCount
    val writeReady = (0 until cfg.clientCount)
      .map(client => writeResponseCandidates(client)(bank).ready &&
        writePop.payload.client === client)
      .reduce(_ || _)
    writePop.ready := Mux(writeClientValid, writeReady, True)

    when((readPop.fire && !readClientValid) ||
      (writePop.fire && !writeClientValid)) {
      routingFaultSticky := True
    }
  }
  io.responseRoutingFaultSticky := routingFaultSticky

  val readFaultSticky = Vec.fill(cfg.bankCount)(Reg(Bool()) init False)
  val writeFaultSticky = Vec.fill(cfg.bankCount)(Reg(Bool()) init False)
  for (bank <- 0 until cfg.bankCount) {
    when(io.bankReadResponse(bank).fire &&
      io.bankReadResponse(bank).payload.rresp =/= B"00") {
      readFaultSticky(bank) := True
    }
    when(io.bankWriteResponse(bank).fire &&
      io.bankWriteResponse(bank).payload.bresp =/= B"00") {
      writeFaultSticky(bank) := True
    }
  }
  io.readResponseFaultSticky := readFaultSticky.asBits
  io.writeResponseFaultSticky := writeFaultSticky.asBits
}
