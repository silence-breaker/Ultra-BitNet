package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Fixed fast-clock slots used by the dual-pumped tensor RAM backend.
  *
  * F0 owns packed reads and activation writes.  F1 owns activation reads and
  * packed writes.  One read and one write may therefore be accepted on every
  * fast clock while each typed client sees one opportunity per slow clock.
  */
object BitNetUnifiedResidentTensorFastPhase
    extends SpinalEnum(binarySequential) {
  val F0PackedReadActivationWrite, F1ActivationReadPackedWrite = newElement()
}

case class BitNetUnifiedResidentTensorFastBackendConfig(
    scratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig(),
    tagWidth: Int = 12) {
  require(tagWidth > 0)
  require(scratchpad.layout.bankCount == 32)
  require(scratchpad.layout.wordWidth ==
    scratchpad.layout.tokenLanes * scratchpad.layout.activationWidth)
  require(scratchpad.layout.rowCount == 512)
}

/** A physical read accepted after region, format and alias authentication.
  * The fast backend deliberately does not repeat those semantic checks.
  */
case class BitNetUnifiedResidentTensorFastReadCommand(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val layout = cfg.scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  val tag = UInt(cfg.tagWidth bits)
}

case class BitNetUnifiedResidentTensorFastActivationWriteCommand(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val layout = cfg.scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  /** Per-bank S32 lane enables. */
  val elementMask = Vec(Bits(layout.tokenLanes bits), layout.bankCount)
  val tag = UInt(cfg.tagWidth bits)
  val words = Vec(
    BitNetUnifiedResidentActivationWord(cfg.scratchpad), layout.bankCount)
}

case class BitNetUnifiedResidentTensorFastPackedWriteCommand(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val layout = cfg.scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val segmentIndex = UInt(log2Up(layout.packedTransportSegmentCount) bits)
  /** Native 32-bit BRAM symbols updated in every selected bank. */
  val symbolMask = Bits(layout.nativeSymbolCount bits)
  val tag = UInt(cfg.tagWidth bits)
  val segmentWords = Vec(
    Bits(layout.wordWidth bits), layout.packedTransportWordsPerSegment)
}

case class BitNetUnifiedResidentTensorFastActivationReadResponse(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val layout = cfg.scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  val tag = UInt(cfg.tagWidth bits)
  val words = Vec(
    BitNetUnifiedResidentActivationWord(cfg.scratchpad), layout.bankCount)
}

case class BitNetUnifiedResidentTensorFastPackedReadResponse(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val layout = cfg.scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val bankMask = Bits(layout.bankCount bits)
  val tag = UInt(cfg.tagWidth bits)
  val words = Vec(
    BitNetUnifiedResidentPackedI8Word(cfg.scratchpad), layout.bankCount)
}

/** Completion for the write which committed on the preceding active edge.
  * A set bit in `collisionMask` was deliberately not written.  Words outside
  * `committedBankMask` are likewise unchanged.
  */
case class BitNetUnifiedResidentTensorFastWriteCommit(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val layout = cfg.scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val requestedBankMask = Bits(layout.bankCount bits)
  val committedBankMask = Bits(layout.bankCount bits)
  val collisionMask = Bits(layout.bankCount bits)
  val tag = UInt(cfg.tagWidth bits)
}

/** Compact authenticated completion for one fixed eight-bank packed segment.
  * All eight words are requested; only the local masks need to cross back to
  * the slow clock domain.
  */
case class BitNetUnifiedResidentTensorFastPackedWriteCommit(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig) extends Bundle {
  private val layout = cfg.scratchpad.layout
  val absoluteRow = UInt(layout.rowAddressWidth bits)
  val segmentIndex = UInt(log2Up(layout.packedTransportSegmentCount) bits)
  val committedWordMask = Bits(layout.packedTransportWordsPerSegment bits)
  val collisionWordMask = Bits(layout.packedTransportWordsPerSegment bits)
  val tag = UInt(cfg.tagWidth bits)
}

/** Timing-oriented 2x-clock physical backend for the unified tensor image.
  *
  * This component must be instantiated in the fast clock domain.  Its four
  * ingress streams are already in that domain: a related-clock mailbox or an
  * asynchronous FIFO is required before this block when their producers run
  * on the slow clock.  Reset establishes F0, after which the phase alternates
  * unconditionally.  Holding Stream.valid until ready therefore aligns a
  * command without exposing the phase bit to the physical RAM ports.
  *
  * The ingress command is first captured in a bank-local register stage.  The
  * registered enable, address, data and S32 mask directly feed the fixed read
  * and write ports of one simple-dual-port memory per bank.  No arbitration or
  * global legality bit remains between those registers and RAM ports.  A
  * command accepted on edge N accesses RAM on edge N+1.  Writes commit after
  * N+1; reads are retained in a two-entry packed/activation response slot and
  * become valid after N+2.  Flow consumers must accept every pulse.  Only
  * banks selected by bankMask carry defined read data.
  *
  * Exactly 32 memories of `wordWidth` x 512 bits are inferred.  The bank-local
  * keep/dont_touch attributes prevent Vivado from
  * merging the replicated address and enable launch registers back into a
  * high-fanout source.
  */
class BitNetUnifiedResidentTensorScratchpadFastBackend(
    cfg: BitNetUnifiedResidentTensorFastBackendConfig =
      BitNetUnifiedResidentTensorFastBackendConfig()) extends Component {
  private val layout = cfg.scratchpad.layout

  val io = new Bundle {
    val packedRead = slave(Stream(
      BitNetUnifiedResidentTensorFastReadCommand(cfg)))
    val packedReadData = master(Flow(
      BitNetUnifiedResidentTensorFastPackedReadParallelPayload(cfg)))
    val activationWrite = slave(Stream(
      BitNetUnifiedResidentTensorFastActivationWriteCommand(cfg)))
    val activationWriteCommit = master(Flow(
      BitNetUnifiedResidentTensorFastWriteCommit(cfg)))

    val activationRead = slave(Stream(
      BitNetUnifiedResidentTensorFastReadCommand(cfg)))
    val activationReadData = master(Flow(
      BitNetUnifiedResidentTensorFastActivationReadResponse(cfg)))
    val packedWrite = slave(Stream(
      BitNetUnifiedResidentTensorFastPackedWriteCommand(cfg)))
    val packedWriteCommit = master(Flow(
      BitNetUnifiedResidentTensorFastPackedWriteCommit(cfg)))

    val phase = out(BitNetUnifiedResidentTensorFastPhase())
    val phaseF0 = out Bool()
    val readPipelineValid = out Bool()
    val writePipelineValid = out Bool()
    val collisionFaultSticky = out Bool()
  }

  val phaseReg = Reg(BitNetUnifiedResidentTensorFastPhase()) init
    BitNetUnifiedResidentTensorFastPhase.F0PackedReadActivationWrite
  when(phaseReg ===
      BitNetUnifiedResidentTensorFastPhase.F0PackedReadActivationWrite) {
    phaseReg :=
      BitNetUnifiedResidentTensorFastPhase.F1ActivationReadPackedWrite
  } otherwise {
    phaseReg :=
      BitNetUnifiedResidentTensorFastPhase.F0PackedReadActivationWrite
  }
  val phaseF0 = phaseReg ===
    BitNetUnifiedResidentTensorFastPhase.F0PackedReadActivationWrite
  io.phase := phaseReg
  io.phaseF0 := phaseF0

  // Fixed slots replace both preferPacked arbiters in the reference backend.
  io.packedRead.ready := phaseF0
  io.activationWrite.ready := phaseF0
  io.activationRead.ready := !phaseF0
  io.packedWrite.ready := !phaseF0

  val selectedReadFire = Mux(
    phaseF0, io.packedRead.fire, io.activationRead.fire)
  val selectedReadPacked = phaseF0
  val selectedReadRow = Mux(
    phaseF0,
    io.packedRead.absoluteRow,
    io.activationRead.absoluteRow)
  val selectedReadMask = Mux(
    phaseF0,
    io.packedRead.bankMask,
    io.activationRead.bankMask)
  val selectedReadTag = Mux(
    phaseF0,
    io.packedRead.tag,
    io.activationRead.tag)

  val selectedWriteFire = Mux(
    phaseF0, io.activationWrite.fire, io.packedWrite.fire)
  val selectedWritePacked = !phaseF0
  val selectedWriteRow = Mux(
    phaseF0,
    io.activationWrite.absoluteRow,
    io.packedWrite.absoluteRow)
  val packedSegmentMasks = Vec((0 until
    layout.packedTransportSegmentCount).map { segment =>
      B(BigInt(0xff) <<
        (segment * layout.packedTransportWordsPerSegment),
        layout.bankCount bits)
    })
  val selectedPackedWriteMask =
    packedSegmentMasks(io.packedWrite.segmentIndex)
  val selectedWriteRequestedMask = Mux(
    phaseF0,
    io.activationWrite.bankMask,
    selectedPackedWriteMask)
  val activationElementPresentMask = Bits(layout.bankCount bits)
  for (bank <- 0 until layout.bankCount) {
    activationElementPresentMask(bank) :=
      io.activationWrite.elementMask(bank).orR
  }
  // A selected activation bank with no enabled token is a completed no-op.
  // Preserve the raw request in the commit, but exclude it from every
  // physical enable, collision and committed-bank indication.
  val selectedWriteMask = Mux(
    phaseF0,
    io.activationWrite.bankMask & activationElementPresentMask,
    Mux(io.packedWrite.symbolMask.orR,
      selectedPackedWriteMask, B(0, layout.bankCount bits)))
  val selectedWriteTag = Mux(
    phaseF0,
    io.activationWrite.tag,
    io.packedWrite.tag)

  // Same-edge read/write commands execute together on the following edge.
  // Resolve collision policy here, before the bank-local WE registers.
  val selectedRowsCollide = selectedReadFire && selectedWriteFire &&
    selectedReadRow === selectedWriteRow
  val selectedCollisionMask = Bits(layout.bankCount bits)
  for (bank <- 0 until layout.bankCount) {
    selectedCollisionMask(bank) := selectedRowsCollide &&
      selectedReadMask(bank) && selectedWriteMask(bank)
  }

  val memories = Array.tabulate(layout.bankCount) { bank =>
    val memory = Mem(Bits(layout.wordWidth bits), layout.rowCount)
    memory.setName(s"fastResidentTensorBank${bank}")
    memory.addAttribute("ram_style", "block")
    memory
  }

  // One read-only and one write-only launch port let Vivado use RAMB36 SDP
  // width rather than the 36-bit TDP limit.  Packed and activation clients
  // own alternating fast slots; the response slot memories below retain the
  // two typed read results independently for the related-clock bridge.
  val readPortEnableRegs = Array.tabulate(layout.bankCount) { bank =>
    val value = RegInit(False)
    value.setName(s"fastBank${bank}ReadPortEnable")
    value.addAttribute("keep", "true")
    value.addAttribute("dont_touch", "true")
    value
  }
  val readPortAddressRegs = Array.tabulate(layout.bankCount) { bank =>
    val value = Reg(UInt(layout.rowAddressWidth bits)) init 0
    value.setName(s"fastBank${bank}ReadPortAddress")
    value.addAttribute("keep", "true")
    value.addAttribute("dont_touch", "true")
    value
  }
  val writePortEnableRegs = Array.tabulate(layout.bankCount) { bank =>
    val value = RegInit(False)
    value.setName(s"fastBank${bank}WritePortEnable")
    value.addAttribute("keep", "true")
    value.addAttribute("dont_touch", "true")
    value
  }
  val writePortAddressRegs = Array.tabulate(layout.bankCount) { bank =>
    val value = Reg(UInt(layout.rowAddressWidth bits)) init 0
    value.setName(s"fastBank${bank}WritePortAddress")
    value.addAttribute("keep", "true")
    value.addAttribute("dont_touch", "true")
    value
  }
  val writePortDataRegs = Array.tabulate(layout.bankCount) { bank =>
    val value = Reg(Bits(layout.wordWidth bits)) init 0
    value.setName(s"fastBank${bank}WritePortData")
    value.addAttribute("keep", "true")
    value.addAttribute("dont_touch", "true")
    value
  }
  // Keep one native 32-bit symbol per activation lane.  Packed symbols are
  // complete groups of adjacent i8 features across all physical token lanes.
  // This preserves partial packed writes without inferring 16 byte memories.
  val writePortElementMaskRegs =
    Array.tabulate(layout.bankCount) { bank =>
      val value = Reg(Bits(layout.nativeSymbolCount bits)) init 0
      value.setName(s"fastBank${bank}WritePortElementMask")
      value.addAttribute("keep", "true")
      value.addAttribute("dont_touch", "true")
      value
    }

  val readStageValid = RegInit(False)
  val readStagePacked = RegInit(False)
  val readStageRow = Reg(UInt(layout.rowAddressWidth bits)) init 0
  val readStageMask = Reg(Bits(layout.bankCount bits)) init 0
  val readStageTag = Reg(UInt(cfg.tagWidth bits)) init 0

  val writeStageValid = RegInit(False)
  val writeStagePacked = RegInit(False)
  val writeStageRow = Reg(UInt(layout.rowAddressWidth bits)) init 0
  val writeStageRequestedMask = Reg(Bits(layout.bankCount bits)) init 0
  val writeStageCommittedMask = Reg(Bits(layout.bankCount bits)) init 0
  val writeStageCollisionMask = Reg(Bits(layout.bankCount bits)) init 0
  val writeStageTag = Reg(UInt(cfg.tagWidth bits)) init 0
  val writeStagePackedSegmentIndex = Reg(
    UInt(log2Up(layout.packedTransportSegmentCount) bits)) init 0

  readStageValid := selectedReadFire
  writeStageValid := selectedWriteFire
  when(selectedReadFire) {
    readStagePacked := selectedReadPacked
    readStageRow := selectedReadRow
    readStageMask := selectedReadMask
    readStageTag := selectedReadTag
  }
  when(selectedWriteFire) {
    writeStagePacked := selectedWritePacked
    writeStageRow := selectedWriteRow
    writeStageRequestedMask := selectedWriteRequestedMask
    writeStageCommittedMask := selectedWriteMask & ~selectedCollisionMask
    writeStageCollisionMask := selectedCollisionMask
    writeStageTag := selectedWriteTag
    when(selectedWritePacked) {
      writeStagePackedSegmentIndex := io.packedWrite.segmentIndex
    }
  }

  for (bank <- 0 until layout.bankCount) {
    readPortEnableRegs(bank) :=
      selectedReadFire && selectedReadMask(bank)
    when(selectedReadFire) {
      readPortAddressRegs(bank) := selectedReadRow
    }

    writePortEnableRegs(bank) := selectedWriteFire &&
      selectedWriteMask(bank) && !selectedCollisionMask(bank)
    when(selectedWriteFire) {
      writePortAddressRegs(bank) := selectedWriteRow
      writePortDataRegs(bank) := Mux(
        phaseF0,
        io.activationWrite.words(bank).asBits,
        io.packedWrite.segmentWords(
          bank % layout.packedTransportWordsPerSegment))
      writePortElementMaskRegs(bank) := Mux(
        phaseF0,
        io.activationWrite.elementMask(bank),
        io.packedWrite.symbolMask)
    }
  }

  val memoryReadData = memories.zipWithIndex.map {
    case (memory, bank) =>
      val data = memory.readSync(
        address = readPortAddressRegs(bank),
        enable = readPortEnableRegs(bank))
      memory.write(
        address = writePortAddressRegs(bank),
        data = writePortDataRegs(bank),
        enable = writePortEnableRegs(bank),
        mask = writePortElementMaskRegs(bank))
      data
  }

  // The SDP read appears one edge after the bank launch registers.  Retain it
  // in one packed/activation slot per bank on the following edge so a slow
  // consumer can acknowledge either type without holding the main BRAM port.
  val responseStoreValid = RegNext(readStageValid) init False
  val responseStorePacked = RegInit(False)
  val responseStoreRow = Reg(UInt(layout.rowAddressWidth bits)) init 0
  val responseStoreMask = Reg(Bits(layout.bankCount bits)) init 0
  val responseStoreTag = Reg(UInt(cfg.tagWidth bits)) init 0
  when(readStageValid) {
    responseStorePacked := readStagePacked
    responseStoreRow := readStageRow
    responseStoreMask := readStageMask
    responseStoreTag := readStageTag
  }

  // A depth-two distributed RAM wastes one LUT per response bit on this
  // device.  Two explicit local row registers trade 8192 plentiful FFs for
  // more than 4600 scarce LUTRAMs while preserving the zero-copy bridge
  // contract.  The packed row is retained exactly once here; its four token
  // lanes below are only fixed wiring views of these same 4096 bits.
  val packedResponseRegs = Array.tabulate(layout.bankCount) { bank =>
    val value = Reg(Bits(layout.wordWidth bits)) init 0
    value.setName(s"fastBank${bank}PackedResponse")
    value
  }
  val activationResponseRegs = Array.tabulate(layout.bankCount) { bank =>
    val value = Reg(Bits(layout.wordWidth bits)) init 0
    value.setName(s"fastBank${bank}ActivationResponse")
    value
  }
  when(responseStoreValid) {
    when(responseStorePacked) {
      for (bank <- 0 until layout.bankCount) {
        packedResponseRegs(bank) := memoryReadData(bank)
      }
    } otherwise {
      for (bank <- 0 until layout.bankCount) {
        activationResponseRegs(bank) := memoryReadData(bank)
      }
    }
  }

  val responseValid = RegNext(responseStoreValid) init False
  val responsePacked = RegInit(False)
  val responseRow = Reg(UInt(layout.rowAddressWidth bits)) init 0
  val responseMask = Reg(Bits(layout.bankCount bits)) init 0
  val responseTag = Reg(UInt(cfg.tagWidth bits)) init 0
  when(responseStoreValid) {
    responsePacked := responseStorePacked
    responseRow := responseStoreRow
    responseMask := responseStoreMask
    responseTag := responseStoreTag
  }
  io.packedReadData.valid := responseValid && responsePacked
  io.activationReadData.valid := responseValid && !responsePacked
  io.packedReadData.absoluteRow := responseRow
  io.packedReadData.bankMask := responseMask
  io.packedReadData.tag := responseTag
  io.packedReadData.laneMask := B(
    (BigInt(1) << layout.packedTransportSegmentCount) - 1,
    layout.packedTransportSegmentCount bits)
  io.activationReadData.absoluteRow := responseRow
  io.activationReadData.bankMask := responseMask
  io.activationReadData.tag := responseTag
  for (bank <- 0 until layout.bankCount) {
    // Data outside responseMask is intentionally unspecified.  Avoiding a
    // 4096-bit zeroing mux is part of the fast backend resource contract.
    val packedResponseWord =
      BitNetUnifiedResidentPackedI8Word(cfg.scratchpad)
    packedResponseWord.assignFromBits(packedResponseRegs(bank))
    for (feature <- 0 until layout.packedFeaturesPerWord;
         lane <- 0 until layout.packedTransportSegmentCount) {
      val site = bank * layout.packedFeaturesPerWord + feature
      io.packedReadData.segments(lane).value(site) :=
        packedResponseWord.feature(feature)(lane)
    }
    io.activationReadData.words(bank).assignFromBits(
      activationResponseRegs(bank))
  }

  val commitValid = RegNext(writeStageValid) init False
  val commitPacked = RegInit(False)
  val commitRow = Reg(UInt(layout.rowAddressWidth bits)) init 0
  val commitRequestedMask = Reg(Bits(layout.bankCount bits)) init 0
  val commitBankMask = Reg(Bits(layout.bankCount bits)) init 0
  val commitCollisionMask = Reg(Bits(layout.bankCount bits)) init 0
  val commitTag = Reg(UInt(cfg.tagWidth bits)) init 0
  val commitPackedSegmentIndex = Reg(
    UInt(log2Up(layout.packedTransportSegmentCount) bits)) init 0
  when(writeStageValid) {
    commitPacked := writeStagePacked
    commitRow := writeStageRow
    commitRequestedMask := writeStageRequestedMask
    commitBankMask := writeStageCommittedMask
    commitCollisionMask := writeStageCollisionMask
    commitTag := writeStageTag
    commitPackedSegmentIndex := writeStagePackedSegmentIndex
  }

  io.activationWriteCommit.valid := commitValid && !commitPacked
  io.packedWriteCommit.valid := commitValid && commitPacked
  io.activationWriteCommit.absoluteRow := commitRow
  io.activationWriteCommit.requestedBankMask := commitRequestedMask
  io.activationWriteCommit.committedBankMask := commitBankMask
  io.activationWriteCommit.collisionMask := commitCollisionMask
  io.activationWriteCommit.tag := commitTag
  io.packedWriteCommit.absoluteRow := commitRow
  io.packedWriteCommit.segmentIndex := commitPackedSegmentIndex
  for (word <- 0 until layout.packedTransportWordsPerSegment) {
    val bankIndex = commitPackedSegmentIndex ##
      U(word, log2Up(layout.packedTransportWordsPerSegment) bits)
    io.packedWriteCommit.committedWordMask(word) :=
      commitBankMask(bankIndex.asUInt)
    io.packedWriteCommit.collisionWordMask(word) :=
      commitCollisionMask(bankIndex.asUInt)
  }
  io.packedWriteCommit.tag := commitTag

  val collisionFault = RegInit(False)
  // Retire the sticky flag from the already registered write-stage mask.
  // The collision decision itself still gates the physical write on the
  // command edge; only observability moves by one fast cycle.  This avoids a
  // long bridge-valid -> bank decode -> 32-bank reduction path at 532 MHz.
  when(writeStageValid && writeStageCollisionMask.orR) {
    collisionFault := True
  }

  io.readPipelineValid :=
    readStageValid || responseStoreValid || responseValid
  io.writePipelineValid := writeStageValid || commitValid
  io.collisionFaultSticky := collisionFault
}

object GenerateBitNetUnifiedResidentTensorScratchpadFastBackend extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-tensor-fast-backend")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedResidentTensorScratchpadFastBackend())
}
