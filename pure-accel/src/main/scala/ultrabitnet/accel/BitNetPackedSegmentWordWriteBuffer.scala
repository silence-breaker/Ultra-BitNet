package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Geometry of the production packed-segment word-write ring.
  *
  * One 1,024-bit segment is physically sixteen independently byte-writeable
  * 64-bit SDP RAM banks.  The deliberately shallow ring replaces the two
  * fabric-register segment assemblies used by scalar Query and Projection
  * producers without introducing another wide elastic register.
  */
case class BitNetPackedSegmentWordWriteBufferConfig(
    segmentWidth: Int = 1024,
    bankWidth: Int = 64,
    ringDepth: Int = 2,
    metadataWidth: Int = 32) {
  require(segmentWidth == 1024,
    "the production packed transport segment is exactly 1,024 bits")
  require(bankWidth == 64,
    "one physical bank must use the native 64-bit SDP width")
  require(segmentWidth % bankWidth == 0)
  require(ringDepth == 2 || ringDepth == 4,
    "the packed segment staging ring is intentionally two or four rows")
  require(metadataWidth > 0)

  val bankCount: Int = segmentWidth / bankWidth
  val bytesPerBank: Int = bankWidth / 8
  val bankIndexWidth: Int = log2Up(bankCount)
  val slotAddressWidth: Int = log2Up(ringDepth)
  val occupancyWidth: Int = log2Up(ringDepth + 1)

  require(bankCount == 16)
  require(bytesPerBank == 8)
}

/** One masked native-word update to the current ring tail.
  *
  * Query producers normally assert one byte-enable bit.  Projection producers
  * normally assert either the low or high four byte-enable bits so all four
  * token lanes of a feature are written together.  Other non-overlapping masks
  * remain legal; every byte must be covered exactly once before `commit`.
  */
case class BitNetPackedSegmentWordWrite(
    cfg: BitNetPackedSegmentWordWriteBufferConfig) extends Bundle {
  val data = Bits(cfg.bankWidth bits)
  val byteMask = Bits(cfg.bytesPerBank bits)
  val bankIndex = UInt(cfg.bankIndexWidth bits)
  val slotAddress = UInt(cfg.slotAddressWidth bits)
  val commit = Bool()
  val metadata = Bits(cfg.metadataWidth bits)
  /** Upstream semantic/framing poison.  A fired poisoned beat enters the same
    * reset-only fail-stop state as a malformed local write protocol.
    */
  val fault = Bool()
}

/** One complete segment in commit order. */
case class BitNetPackedSegmentWordBuffered(
    cfg: BitNetPackedSegmentWordWriteBufferConfig) extends Bundle {
  val data = Bits(cfg.segmentWidth bits)
  val metadata = Bits(cfg.metadataWidth bits)
  val slotAddress = UInt(cfg.slotAddressWidth bits)
}

/** Native byte-writeable common-clock SDP bank used in production. */
class BitNetPackedSegmentWordXpmSdpram64(depth: Int) extends BlackBox {
  require(depth == 2 || depth == 4)
  private val addressWidth = log2Up(depth)

  setDefinitionName("xpm_memory_sdpram")
  addGeneric("ADDR_WIDTH_A", addressWidth)
  addGeneric("ADDR_WIDTH_B", addressWidth)
  addGeneric("AUTO_SLEEP_TIME", 0)
  addGeneric("BYTE_WRITE_WIDTH_A", 8)
  addGeneric("CASCADE_HEIGHT", 0)
  addGeneric("CLOCKING_MODE", "common_clock")
  addGeneric("ECC_MODE", "no_ecc")
  addGeneric("MEMORY_INIT_FILE", "none")
  addGeneric("MEMORY_INIT_PARAM", "0")
  addGeneric("MEMORY_OPTIMIZATION", "true")
  addGeneric("MEMORY_PRIMITIVE", "block")
  addGeneric("MEMORY_SIZE", 64 * depth)
  addGeneric("MESSAGE_CONTROL", 0)
  addGeneric("READ_DATA_WIDTH_B", 64)
  addGeneric("READ_LATENCY_B", 1)
  addGeneric("READ_RESET_VALUE_B", "0")
  addGeneric("RST_MODE_B", "SYNC")
  addGeneric("SIM_ASSERT_CHK", 0)
  addGeneric("USE_EMBEDDED_CONSTRAINT", 0)
  addGeneric("USE_MEM_INIT", 0)
  addGeneric("WAKEUP_TIME", "disable_sleep")
  addGeneric("WRITE_DATA_WIDTH_A", 64)
  addGeneric("WRITE_MODE_B", "no_change")

  val dbiterrb = out Bool()
  val doutb = out Bits (64 bits)
  val sbiterrb = out Bool()
  val addra = in UInt (addressWidth bits)
  val addrb = in UInt (addressWidth bits)
  val clka = in Bool()
  val clkb = in Bool()
  val dina = in Bits (64 bits)
  val ena = in Bool()
  val enb = in Bool()
  val injectdbiterra = in Bool()
  val injectsbiterra = in Bool()
  val regceb = in Bool()
  val rstb = in Bool()
  val sleep = in Bool()
  val wea = in Bits (8 bits)

  noIoPrefix()
}

/** Single-clock masked-word assembly ring for 1,024-bit packed segments.
  *
  * The producer may write one 64-bit bank per cycle with arbitrary disjoint
  * byte enables.  A segment becomes visible only when the firing beat both
  * supplies the final missing bytes and asserts `commit`.  Slot, metadata,
  * duplicate-byte and early/missing-commit violations quarantine the buffer
  * until reset; an uncommitted partial row can therefore never escape.
  *
  * Sixteen XPM SDP `doutb` ports form the only wide holding stage.  While the
  * output Stream is stalled no new read is issued, so data and sidecar metadata
  * remain stable without a 1,024-bit fabric Reg or Mem.  A full ring never
  * accepts a same-address replacement on the cycle its oldest row is read;
  * the freed slot reopens on the following cycle.
  */
class BitNetPackedSegmentWordWriteBuffer(
    cfg: BitNetPackedSegmentWordWriteBufferConfig =
      BitNetPackedSegmentWordWriteBufferConfig(),
    useVendorByteRam: Boolean = true) extends Component {
  val io = new Bundle {
    val wordWrite = slave(Stream(BitNetPackedSegmentWordWrite(cfg)))
    val segment = master(Stream(BitNetPackedSegmentWordBuffered(cfg)))

    val full = out Bool()
    val empty = out Bool()
    val partial = out Bool()
    val usedSlots = out UInt(cfg.occupancyWidth bits)
    val protocolFaultSticky = out Bool()
  }

  // Each native bank contributes exactly 64 bits to the BRAM-backed output.
  // The portable alternative exists solely for open-source simulation.
  val vendorDataBanks = if (useVendorByteRam)
    Array.tabulate(cfg.bankCount) { bank =>
      val memory = new BitNetPackedSegmentWordXpmSdpram64(cfg.ringDepth)
      memory.setName(s"packedSegmentWordDataBank${bank}")
      memory
    }
  else null
  val portableDataBanks = if (!useVendorByteRam)
    Array.tabulate(cfg.bankCount) { bank =>
      val memory = Mem(Bits(cfg.bankWidth bits), cfg.ringDepth)
      memory.setName(s"packedSegmentWordDataBank${bank}Portable")
      memory
    }
  else null

  val metadataMemory = Mem(Bits(cfg.metadataWidth bits), cfg.ringDepth)
  metadataMemory.setName("packedSegmentWordMetadataMemory")
  metadataMemory.addAttribute("ram_style", "distributed")

  val stickyFault = RegInit(False)
  val writeAddress = Reg(UInt(cfg.slotAddressWidth bits)) init 0
  val readAddress = Reg(UInt(cfg.slotAddressWidth bits)) init 0
  val occupancy = Reg(UInt(cfg.occupancyWidth bits)) init 0

  // Only the current tail may be partial.  Keeping byte ownership rather than
  // payload data in fabric makes both the one-byte Query order and four-byte
  // Projection order independently auditable.
  val partialActive = RegInit(False)
  val partialMetadata = Reg(Bits(cfg.metadataWidth bits)) init 0
  val byteCoverage = Vec(
    Reg(Bits(cfg.bytesPerBank bits)) init 0, cfg.bankCount)

  val outputValid = RegInit(False)
  // Once a segment has been presented under backpressure, even a later local
  // producer fault must not retract that Stream.  It is the sole exception to
  // sticky quarantine: the already-promised row drains, while no new row can
  // become visible or be written.
  val outputOffered = RegInit(False)
  io.segment.valid := outputValid && (!stickyFault || outputOffered)
  io.segment.payload.slotAddress := readAddress
  val outputFire = io.segment.fire
  val outputStalled = io.segment.valid && !io.segment.ready
  val outputMustDrain = outputOffered || outputStalled

  when(outputStalled) {
    outputOffered := True
  }
  when(outputFire) {
    outputOffered := False
  }

  val readAddressAfterFire = UInt(cfg.slotAddressWidth bits)
  readAddressAfterFire := readAddress
  when(outputFire) {
    readAddressAfterFire := readAddress + 1
  }

  val occupancyAfterFire = UInt(cfg.occupancyWidth bits)
  occupancyAfterFire := occupancy
  when(outputFire) {
    occupancyAfterFire := occupancy - 1
  }

  val outputMayAdvance = !outputValid || io.segment.ready
  val issueReadCandidate = outputMayAdvance && occupancyAfterFire =/= 0 &&
    !stickyFault
  val issueAddress = readAddressAfterFire

  // This conflict is unreachable for a legal non-full ring, but the explicit
  // priority also prevents an illegal externally addressed beat from ever
  // toggling the write port against the active synchronous read address.
  val writeReadAddressConflict = issueReadCandidate && io.wordWrite.valid &&
    io.wordWrite.payload.slotAddress === issueAddress
  val hasWriteCapacity = occupancy =/=
    U(cfg.ringDepth, cfg.occupancyWidth bits)
  io.wordWrite.ready := hasWriteCapacity && !stickyFault &&
    !writeReadAddressConflict

  val selectedCoverage = byteCoverage(io.wordWrite.payload.bankIndex)
  val nonEmptyMask = io.wordWrite.payload.byteMask.orR
  val disjointMask =
    !(selectedCoverage & io.wordWrite.payload.byteMask).orR

  val coverageAfterBeat = Vec(Bits(cfg.bytesPerBank bits), cfg.bankCount)
  for (bank <- 0 until cfg.bankCount) {
    coverageAfterBeat(bank) := byteCoverage(bank)
    when(io.wordWrite.payload.bankIndex === bank) {
      coverageAfterBeat(bank) :=
        byteCoverage(bank) | io.wordWrite.payload.byteMask
    }
  }
  val rowCompleteAfterBeat = coverageAfterBeat.map(_.andR).reduce(_ && _)
  val metadataLegal = !partialActive ||
    io.wordWrite.payload.metadata === partialMetadata
  val slotLegal = io.wordWrite.payload.slotAddress === writeAddress
  // Commit must coincide with the exact beat which fills the final missing
  // bytes: neither an early commit nor a complete-but-uncommitted row is legal.
  val commitLegal = io.wordWrite.payload.commit === rowCompleteAfterBeat
  val beatLegal = !io.wordWrite.payload.fault && nonEmptyMask &&
    disjointMask && metadataLegal && slotLegal && commitLegal

  val writeFire = io.wordWrite.fire
  val protocolFaultNow = writeFire && !beatLegal
  val memoryWrite = writeFire && beatLegal
  val commitFire = memoryWrite && io.wordWrite.payload.commit
  // A simultaneous illegal write may still perform a harmless private RAM
  // read.  The later protocolFaultNow branch has priority over outputValid,
  // and stickyFault masks the segment interface, so no data can escape.  Do
  // not feed the full byte-coverage/commit checker into all sixteen BRAM read
  // enables; that creates a high-fanout ten-LUT timing cone for no safety gain.
  val issueRead = issueReadCandidate

  val readBanks = if (useVendorByteRam) {
    vendorDataBanks.zipWithIndex.map { case (memory, bank) =>
      val bankSelected = io.wordWrite.payload.bankIndex === bank
      memory.clka := ClockDomain.current.readClockWire
      memory.clkb := ClockDomain.current.readClockWire
      memory.addra := io.wordWrite.payload.slotAddress
      memory.addrb := issueAddress
      memory.dina := io.wordWrite.payload.data
      memory.ena := memoryWrite && bankSelected
      memory.enb := issueRead
      memory.injectdbiterra := False
      memory.injectsbiterra := False
      memory.regceb := True
      memory.rstb := False
      memory.sleep := False
      memory.wea := io.wordWrite.payload.byteMask
      memory.doutb
    }
  } else {
    portableDataBanks.zipWithIndex.map { case (memory, bank) =>
      val bankSelected = io.wordWrite.payload.bankIndex === bank
      memory.write(
        address = io.wordWrite.payload.slotAddress,
        data = io.wordWrite.payload.data,
        enable = memoryWrite && bankSelected,
        mask = io.wordWrite.payload.byteMask)
      memory.readSync(
        address = issueAddress,
        enable = issueRead)
    }
  }

  val readData = Bits(cfg.segmentWidth bits)
  for (bank <- 0 until cfg.bankCount) {
    readData((bank + 1) * cfg.bankWidth - 1 downto bank * cfg.bankWidth) :=
      readBanks(bank)
  }
  io.segment.payload.data := readData

  metadataMemory.write(
    address = io.wordWrite.payload.slotAddress,
    data = io.wordWrite.payload.metadata,
    enable = commitFire)
  val readMetadata = metadataMemory.readSync(
    address = issueAddress,
    enable = issueRead)
  io.segment.payload.metadata := readMetadata

  when(outputMayAdvance) {
    outputValid := issueRead
  }
  when(outputFire) {
    readAddress := readAddressAfterFire
  }

  when(memoryWrite) {
    when(!partialActive) {
      partialMetadata := io.wordWrite.payload.metadata
    }
    when(commitFire) {
      partialActive := False
      for (bank <- 0 until cfg.bankCount) {
        byteCoverage(bank) := 0
      }
      writeAddress := writeAddress + 1
    } otherwise {
      partialActive := True
      byteCoverage(io.wordWrite.payload.bankIndex) :=
        selectedCoverage | io.wordWrite.payload.byteMask
    }
  }

  when(commitFire && !outputFire) {
    occupancy := occupancy + 1
  } elsewhen (!commitFire && outputFire) {
    occupancy := occupancy - 1
  }

  when(protocolFaultNow) {
    stickyFault := True
    when(!outputMustDrain) {
      outputValid := False
      outputOffered := False
    }
    partialActive := False
    for (bank <- 0 until cfg.bankCount) {
      byteCoverage(bank) := 0
    }
  }

  io.full := occupancy === U(cfg.ringDepth, cfg.occupancyWidth bits)
  io.empty := occupancy === 0
  io.partial := partialActive
  io.usedSlots := occupancy
  io.protocolFaultSticky := stickyFault
}
