package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Geometry of the LM-only packed segment staging buffer.
  *
  * The production configuration is deliberately small and wide: two or four
  * 1,024-bit rows, physically banked as sixteen 64-bit simple-dual-port
  * memories.  It replaces a pair of fabric-register 4,096-bit images;
  * `ram_style = block` keeps the storage decision explicit for Vivado.
  */
case class BitNetLmPackedSegmentBufferConfig(
    segmentWidth: Int = 1024,
    segmentCount: Int = 4,
    ringDepth: Int = 4,
    metadataWidth: Int = 32,
    captureCdcDepth: Int = 8,
    synchronizerDepth: Int = 2,
    scatterCapture: Boolean = false) {
  require(segmentWidth == 1024,
    "the production LM transport uses 1,024-bit segments")
  require(segmentWidth % 8 == 0)
  require(segmentCount == 2 || segmentCount == 4,
    "the production LM staging ring supports TP2 or TP4 segment groups")
  require(ringDepth >= segmentCount && isPow2(ringDepth),
    "the physical ring must hold at least one complete capture group")
  require(metadataWidth > 0)
  require(captureCdcDepth >= 4 && isPow2(captureCdcDepth))
  require(synchronizerDepth >= 2)

  val segmentBytes: Int = segmentWidth / 8
  val bankWidth: Int = 64
  val bankCount: Int = segmentWidth / bankWidth
  val bytesPerBank: Int = bankWidth / 8
  val byteIndexWidth: Int = log2Up(segmentBytes)
  val bankIndexWidth: Int = log2Up(bankCount)
  val slotIndexWidth: Int = log2Up(segmentCount)
  val addressWidth: Int = log2Up(ringDepth)
  val pointerWidth: Int = addressWidth + 1
  val captureCountWidth: Int = log2Up(captureCdcDepth * 2)

  require(bankCount == 16)
  require(bytesPerBank == 8)
}

/** One byte of a slow-domain capture segment.
  *
  * Metadata must remain identical for all 128 bytes.  `first`, `last`, and
  * `byteIndex` are checked before a row is made visible to the reader.
  */
case class BitNetLmPackedCaptureByte(
    cfg: BitNetLmPackedSegmentBufferConfig) extends Bundle {
  val metadata = Bits(cfg.metadataWidth bits)
  /** Logical segment slot selected by an interleaved capture producer.  The
    * scatter writer rotates it by the current physical ring base; sequential
    * mode requires zero and otherwise ignores this field.
    */
  val slotIndex = UInt(cfg.slotIndexWidth bits)
  val byteIndex = UInt(cfg.byteIndexWidth bits)
  val data = Bits(8 bits)
  val first = Bool()
  val last = Bool()
  val fault = Bool()
}

/** A complete segment supplied by a fast-domain packed-read backend. */
case class BitNetLmPackedReplaySegment(
    cfg: BitNetLmPackedSegmentBufferConfig) extends Bundle {
  val metadata = Bits(cfg.metadataWidth bits)
  val data = Bits(cfg.segmentWidth bits)
}

/** One committed segment in slow-domain consumption order. */
case class BitNetLmPackedBufferedSegment(
    cfg: BitNetLmPackedSegmentBufferConfig) extends Bundle {
  val metadata = Bits(cfg.metadataWidth bits)
  val data = Bits(cfg.segmentWidth bits)
  /** False for a byte-assembled capture, true for a fast replay push. */
  val fromReplay = Bool()
}

/** One native-width bank of the production LM segment store.
  *
  * XPM is a Vivado built-in macro library.  The generated production RTL is
  * therefore self-contained as a synthesis input even though the simulator
  * model is not copied into that file.  Verilator configurations instantiate
  * the portable fallback in [[BitNetLmPackedSegmentBuffer]] instead.
  */
class BitNetLmXpmByteWriteSdpram64(depth: Int) extends BlackBox {
  require(depth >= 4 && isPow2(depth))
  private val addressWidth = log2Up(depth)

  setDefinitionName("xpm_memory_sdpram")
  addGeneric("ADDR_WIDTH_A", addressWidth)
  addGeneric("ADDR_WIDTH_B", addressWidth)
  addGeneric("AUTO_SLEEP_TIME", 0)
  addGeneric("BYTE_WRITE_WIDTH_A", 8)
  addGeneric("CASCADE_HEIGHT", 0)
  addGeneric("CLOCKING_MODE", "independent_clock")
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

/** Shared capture/replay staging RAM for the LM activation endpoint.
  *
  * Slow scalar capture beats cross through a narrow asynchronous FIFO and are
  * byte-written into the current fast-side RAM row.  The write pointer moves
  * only after all 128 framing-checked bytes have arrived, so a partial or
  * malformed row can never become visible.  A fast packed-read result writes
  * the same RAM port in one cycle.  Capture has priority and locks the port
  * until its current segment is complete; callers normally keep capture and
  * replay job-mutually-exclusive.
  *
  * The wide storage is a conventional dual-clock simple-dual-port RAM: fast
  * unmasked 64-bit bank writes and slow synchronous reads.  Gray-coded
  * ownership pointers are the only state crossing between the RAM clock
  * domains.  The
  * slow RAM output is itself the elastic output stage, so backpressure adds no
  * second 1,024-bit register bank and a ready consumer receives one segment
  * per slow clock after initial RAM latency.
  *
  * Both resets must be asserted for a new session.  Independently resetting
  * one side while traffic is live is outside the component contract.
  */
class BitNetLmPackedSegmentBuffer(
    cfg: BitNetLmPackedSegmentBufferConfig =
      BitNetLmPackedSegmentBufferConfig(),
    useVendorByteRam: Boolean = true,
    /** Reuse the enclosing component's clock domain for capture-only
      * integration.  The explicit clock/reset ports remain for a stable
      * standalone interface but are unused in this mode.  Production
      * dual-clock probes and replay integrations keep the default `false`.
      */
    useCurrentClockDomain: Boolean = false) extends Component {
  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    // Slow-domain scalar producer and segment consumer.
    val captureByte = slave(Stream(BitNetLmPackedCaptureByte(cfg)))
    val segment = master(Stream(BitNetLmPackedBufferedSegment(cfg)))

    // Fast-domain packed-read producer.
    val replaySegment = slave(Stream(BitNetLmPackedReplaySegment(cfg)))

    // Clock-domain-local diagnostics: writeFull is fast, readEmpty and the
    // synchronized sticky fault are slow.
    val writeFull = out Bool()
    val readEmpty = out Bool()
    /** Slow-domain terminal: no accepted capture byte remains in the CDC,
      * no interleaved group is partial, and no capture segment remains in the
      * wide ring/output stage.
      */
    val captureIdle = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val slowCd = if (useCurrentClockDomain) ClockDomain.current else
    ClockDomain(
      clock = io.slowClk,
      reset = io.slowResetn,
      config = ClockDomainConfig(
        clockEdge = RISING,
        resetKind = ASYNC,
        resetActiveLevel = LOW))
  val fastCd = if (useCurrentClockDomain) ClockDomain.current else
    ClockDomain(
      clock = io.fastClk,
      reset = io.fastResetn,
      config = ClockDomainConfig(
        clockEdge = RISING,
        resetKind = ASYNC,
        resetActiveLevel = LOW))

  // Sixteen independently byte-writeable 64-bit banks map one-for-one to
  // native RAMB36 SDP width.  Do not replace this with one masked 1,024-bit
  // Spinal Mem: that form has been measured to infer 128 RAMB18 instances.
  val vendorDataBanks = if (useVendorByteRam)
    Array.tabulate(cfg.bankCount) { bank =>
      val memory = new BitNetLmXpmByteWriteSdpram64(cfg.ringDepth)
      memory.setName(s"lmSegmentDataBank${bank}")
      memory
    }
  else null
  val portableDataBanks = if (!useVendorByteRam)
    Array.tabulate(cfg.bankCount) { bank =>
      val memory = Mem(Bits(cfg.bankWidth bits), cfg.ringDepth)
      memory.setName(s"lmSegmentDataBank${bank}Portable")
      memory
    }
  else null
  val segmentMetadataMemory = Mem(
    Bits((cfg.metadataWidth + 1) bits), cfg.ringDepth)
  segmentMetadataMemory.setName("lmSegmentMetadataMemory")
  segmentMetadataMemory.addAttribute("ram_style", "distributed")

  // A real dual-clock instance confines capture to a narrow asynchronous
  // FIFO; no 1,024-bit payload crosses through registers.  Capture-only codec
  // integration already shares one clock, so a direct retained Stream avoids
  // synthesizing redundant Gray pointers and hundreds of CDC LUTs.
  val capturePush = Stream(BitNetLmPackedCaptureByte(cfg))
  val capturePop = Stream(BitNetLmPackedCaptureByte(cfg))
  if (useCurrentClockDomain) {
    capturePop.valid := capturePush.valid
    capturePop.payload := capturePush.payload
    capturePush.ready := capturePop.ready
  } else {
    val captureCdc = StreamFifoCC(
      dataType = HardType(BitNetLmPackedCaptureByte(cfg)),
      depth = cfg.captureCdcDepth,
      pushClock = slowCd,
      popClock = fastCd)
    captureCdc.setName("lmCaptureByteCdc")
    captureCdc.ram.addAttribute("ram_style", "distributed")
    captureCdc.io.push.valid := capturePush.valid
    captureCdc.io.push.payload := capturePush.payload
    capturePush.ready := captureCdc.io.push.ready
    capturePop.valid := captureCdc.io.pop.valid
    capturePop.payload := captureCdc.io.pop.payload
    captureCdc.io.pop.ready := capturePop.ready
  }

  val writeGrayX = Bits(cfg.pointerWidth bits)
  val readGrayX = Bits(cfg.pointerWidth bits)
  val fastFaultX = Bool()
  val captureConsumedGrayX = Bits(cfg.captureCountWidth bits)
  val captureGroupActiveX = Bool()
  Seq(writeGrayX, readGrayX, fastFaultX, captureConsumedGrayX,
    captureGroupActiveX).foreach(
    _.addTag(crossClockDomain))

  private def binaryToGray(value: UInt): Bits =
    (value ^ (value |>> 1)).asBits

  private def grayToBinary(value: Bits): UInt = {
    val binary = Bits(value.getWidth bits)
    for (bit <- value.high downto 0) {
      binary(bit) := value(value.high downto bit).xorR
    }
    binary.asUInt
  }

  val slowArea = new ClockingArea(slowCd) {
    val fastFaultSync = BufferCC(
      fastFaultX,
      init = False,
      bufferDepth = Some(cfg.synchronizerDepth),
      inputAttributes = Seq(crossClockDomain))

    capturePush.valid := io.captureByte.valid && !fastFaultSync
    capturePush.payload := io.captureByte.payload
    io.captureByte.ready := capturePush.ready && !fastFaultSync
    val captureAcceptedCount = Reg(UInt(cfg.captureCountWidth bits)) init 0
    when(io.captureByte.fire) {
      captureAcceptedCount := captureAcceptedCount + 1
    }
    val captureConsumedGraySync = BufferCC(
      captureConsumedGrayX,
      init = B(0, cfg.captureCountWidth bits),
      bufferDepth = Some(cfg.synchronizerDepth),
      inputAttributes = Seq(crossClockDomain))
    val captureGroupActiveSync = BufferCC(
      captureGroupActiveX,
      init = False,
      bufferDepth = Some(cfg.synchronizerDepth),
      inputAttributes = Seq(crossClockDomain))
    val captureAcceptedGray = binaryToGray(captureAcceptedCount)

    val writeGraySync = BufferCC(
      writeGrayX,
      init = B(0, cfg.pointerWidth bits),
      bufferDepth = Some(cfg.synchronizerDepth),
      inputAttributes = Seq(crossClockDomain))
    val readBinary = Reg(UInt(cfg.pointerWidth bits)) init 0
    val readGray = Reg(Bits(cfg.pointerWidth bits)) init 0
    readGrayX := readGray

    // The memory output is the only wide slow-domain holding stage.
    val outputValid = RegInit(False)
    val outputFire = outputValid && io.segment.ready
    val readBinaryAfterFire = UInt(cfg.pointerWidth bits)
    readBinaryAfterFire := readBinary
    when(outputFire) {
      readBinaryAfterFire := readBinary + 1
    }
    val readGrayAfterFire = binaryToGray(readBinaryAfterFire)
    val entryAvailableAfterFire = readGrayAfterFire =/= writeGraySync
    val outputMayAdvance = !outputValid || io.segment.ready
    val issueRead = outputMayAdvance && entryAvailableAfterFire
    val issueAddress = readBinaryAfterFire(
      cfg.addressWidth - 1 downto 0)

    val readBanks = if (useVendorByteRam) {
      vendorDataBanks.map { memory =>
        memory.clkb := io.slowClk
        memory.addrb := issueAddress
        memory.enb := issueRead
        memory.regceb := True
        memory.rstb := False
        memory.doutb
      }
    } else {
      portableDataBanks.map(_.readSync(
        address = issueAddress,
        enable = issueRead,
        clockCrossing = true))
    }
    val readData = Bits(cfg.segmentWidth bits)
    for (bank <- 0 until cfg.bankCount) {
      readData((bank + 1) * cfg.bankWidth - 1 downto
        bank * cfg.bankWidth) := readBanks(bank)
    }
    val readMetadata = segmentMetadataMemory.readSync(
      address = issueAddress,
      enable = issueRead,
      clockCrossing = true)

    when(outputMayAdvance) {
      outputValid := issueRead
    }
    when(outputFire) {
      readBinary := readBinaryAfterFire
      readGray := readGrayAfterFire
    }

    io.segment.valid := outputValid
    io.segment.payload.data := readData
    io.segment.payload.metadata :=
      readMetadata(cfg.metadataWidth - 1 downto 0)
    io.segment.payload.fromReplay := readMetadata(cfg.metadataWidth)

    val ringEmpty = !outputValid && readGray === writeGraySync
    io.readEmpty := ringEmpty
    // Require the complete idle predicate on two consecutive slow edges.  It
    // prevents synchronizer skew between the consumed-byte count and the
    // committed-ring pointer from making an abort completion one cycle early.
    val captureIdleCandidate =
      captureAcceptedGray === captureConsumedGraySync &&
        !captureGroupActiveSync && ringEmpty
    val captureIdleCandidateDelayed = RegNext(captureIdleCandidate) init False
    io.captureIdle := captureIdleCandidate && captureIdleCandidateDelayed
    io.protocolFaultSticky := fastFaultSync
  }

  val fastArea = new ClockingArea(fastCd) {
    val readGraySync = BufferCC(
      readGrayX,
      init = B(0, cfg.pointerWidth bits),
      bufferDepth = Some(cfg.synchronizerDepth),
      inputAttributes = Seq(crossClockDomain))
    val writeBinary = Reg(UInt(cfg.pointerWidth bits)) init 0
    val writeGray = Reg(Bits(cfg.pointerWidth bits)) init 0
    writeGrayX := writeGray

    // With a power-of-two asynchronous ring, full is the read Gray pointer
    // with its two most-significant bits inverted.
    val fullTarget = Bits(cfg.pointerWidth bits)
    fullTarget := readGraySync
    fullTarget(cfg.pointerWidth - 1 downto cfg.pointerWidth - 2) :=
      ~readGraySync(cfg.pointerWidth - 1 downto cfg.pointerWidth - 2)
    // Register the capacity decision in the fast domain.  In particular,
    // synchronized read-pointer bits never feed the sixteen BRAM write
    // enables combinationally.  Entering full uses the pointer after the
    // accepted commit.  Leaving full is deliberately conservative: first
    // observe a synchronized read-pointer move, then wait one further fast
    // edge before reopening the write port.
    val fullReg = RegInit(False)
    val fullReadGray = Reg(Bits(cfg.pointerWidth bits)) init 0
    val fullReleasePending = RegInit(False)
    io.writeFull := fullReg

    val stickyFault = RegInit(False)
    fastFaultX := stickyFault
    val captureConsumedCount = Reg(UInt(cfg.captureCountWidth bits)) init 0
    when(capturePop.fire) {
      captureConsumedCount := captureConsumedCount + 1
    }
    captureConsumedGrayX := binaryToGray(captureConsumedCount)

    val captureBeatLegal = Bool()
    val captureWrite = Bool()
    val captureCommit = Bool()
    val captureMemoryWrite = Bool()
    val captureGroupBusy = Bool()
    val captureWriteAddress = UInt(cfg.addressWidth bits)
    val captureWriteBank = UInt(cfg.bankIndexWidth bits)
    val captureWriteData = Bits(cfg.bankWidth bits)
    val captureWriteMask = Bits(cfg.bytesPerBank bits)
    val captureCommitMetadata = Bits(cfg.metadataWidth bits)
    Seq(captureBeatLegal, captureWrite, captureCommit,
      captureMemoryWrite, captureGroupBusy, captureWriteAddress,
      captureWriteBank, captureWriteData, captureWriteMask,
      captureCommitMetadata).foreach(_.allowOverride)
    captureBeatLegal := False
    captureWrite := False
    captureCommit := False
    captureMemoryWrite := False
    captureGroupBusy := False
    captureWriteAddress := 0
    captureWriteBank := 0
    captureWriteData := 0
    captureWriteMask := 0
    captureCommitMetadata := capturePop.payload.metadata

    if (cfg.scatterCapture) {
      // Physical packed segments are filled concurrently.  Quantized features
      // arrive tile-major; each slot therefore sees one contiguous feature
      // chunk per dense tile.  Transpose that sequence into the generic packed
      // word order: word index, four-feature index, then token/tile lane.
      val scatterGroupActive = RegInit(False)
      // Empty-ring groups may start at any physical address after replay or
      // prior capture traffic.  Preserve the first beat's write pointer as
      // the base for all four logical slots, including the wrapped slot.
      val scatterBaseAddress =
        Reg(UInt(cfg.addressWidth bits)) init 0
      val scatterGlobalCount =
        Reg(UInt(log2Up(cfg.segmentBytes * cfg.segmentCount) bits)) init 0
      val scatterSlotCount = Vec(
        Reg(UInt(cfg.byteIndexWidth bits)) init 0, cfg.segmentCount)
      val scatterSlotMetadata = Vec(
        Reg(Bits(cfg.metadataWidth bits)) init 0, cfg.segmentCount)
      val selectedSlotCount =
        scatterSlotCount(capturePop.payload.slotIndex)
      val selectedSlotMetadata =
        scatterSlotMetadata(capturePop.payload.slotIndex)
      val scatterGroupBase = Mux(
        scatterGroupActive,
        scatterBaseAddress,
        writeBinary(cfg.addressWidth - 1 downto 0))
      val scatterPhysicalAddress = UInt(cfg.addressWidth bits)
      scatterPhysicalAddress :=
        scatterGroupBase +
          capturePop.payload.slotIndex.resize(cfg.addressWidth)
      val scatterChunkBytes = cfg.segmentBytes / cfg.segmentCount
      val scatterChunkBits = log2Up(scatterChunkBytes)
      val scatterTokenBits = log2Up(cfg.segmentCount)
      val expectedSlot = (scatterGlobalCount |>> scatterChunkBits)
        .resize(cfg.slotIndexWidth)
      val scatterElement = selectedSlotCount(
        scatterChunkBits - 1 downto 0)
      val scatterWord = (scatterElement |>> 2)
      val scatterFeature = scatterElement(1 downto 0)
      val scatterToken = (selectedSlotCount |>> scatterChunkBits)
        .resize(scatterTokenBits)
      val expectedPhysicalByte = UInt(cfg.byteIndexWidth bits)
      expectedPhysicalByte := (
        (scatterWord.resize(cfg.byteIndexWidth) <<
          (2 + scatterTokenBits)) +
        (scatterFeature.resize(cfg.byteIndexWidth) << scatterTokenBits) +
        scatterToken.resize(cfg.byteIndexWidth)).resized
      val expectedFirst = selectedSlotCount === 0
      val expectedLast = selectedSlotCount === cfg.segmentBytes - 1
      val metadataLegal = expectedFirst ||
        capturePop.payload.metadata === selectedSlotMetadata
      val commitOrderLegal = !expectedLast ||
        writeBinary(cfg.addressWidth - 1 downto 0) ===
          scatterPhysicalAddress
      val synchronizedReadBinary = grayToBinary(readGraySync)
      val ringUsed = (writeBinary - synchronizedReadBinary).resize(
        cfg.pointerWidth)
      val groupCapacityAvailable = ringUsed <= U(
        cfg.ringDepth - cfg.segmentCount, cfg.pointerWidth bits)

      // A scatter group reserves one physical row per segment.  A deeper
      // physical ring may admit the next group while the final row of the
      // previous one is still completing its packed-write audit.  Depth eight
      // costs no extra RAMB36 for these tiny 64-bit banks and removes the
      // otherwise accumulating row-boundary bubble in continuous capture.
      capturePop.ready := !fullReg && !stickyFault &&
        (scatterGroupActive || groupCapacityAvailable)
      captureGroupBusy := scatterGroupActive
      captureGroupActiveX := scatterGroupActive
      captureBeatLegal :=
        capturePop.payload.slotIndex === expectedSlot &&
          capturePop.payload.byteIndex === expectedPhysicalByte &&
          capturePop.payload.first === expectedFirst &&
          capturePop.payload.last === expectedLast &&
          metadataLegal && commitOrderLegal &&
          !capturePop.payload.fault
      captureWrite := capturePop.fire && captureBeatLegal
      captureCommit := captureWrite && expectedLast
      captureMemoryWrite := captureWrite
      captureWriteAddress := scatterPhysicalAddress
      captureWriteBank := capturePop.payload.byteIndex(
        cfg.byteIndexWidth - 1 downto log2Up(cfg.bytesPerBank))
        .resize(cfg.bankIndexWidth)
      val scatterByteInBank = capturePop.payload.byteIndex(
        log2Up(cfg.bytesPerBank) - 1 downto 0)
      for (byte <- 0 until cfg.bytesPerBank) {
        when(scatterByteInBank === byte) {
          captureWriteData((byte + 1) * 8 - 1 downto byte * 8) :=
            capturePop.payload.data
          captureWriteMask(byte) := True
        }
      }
      captureCommitMetadata := selectedSlotMetadata

      when(captureWrite) {
        when(!scatterGroupActive) {
          scatterGroupActive := True
          scatterBaseAddress :=
            writeBinary(cfg.addressWidth - 1 downto 0)
        }
        when(expectedFirst) {
          scatterSlotMetadata(capturePop.payload.slotIndex) :=
            capturePop.payload.metadata
        }
        when(expectedLast) {
          scatterSlotCount(capturePop.payload.slotIndex) :=
            U(0, cfg.byteIndexWidth bits)
        } otherwise {
          scatterSlotCount(capturePop.payload.slotIndex) :=
            selectedSlotCount + 1
        }
        when(scatterGlobalCount ===
            cfg.segmentBytes * cfg.segmentCount - 1) {
          scatterGlobalCount := 0
          scatterGroupActive := False
        } otherwise {
          scatterGlobalCount := scatterGlobalCount + 1
        }
      }
    } else {
      val capturePartial = RegInit(False)
      val captureExpectedIndex =
        Reg(UInt(cfg.byteIndexWidth bits)) init 0
      val captureMetadata = Reg(Bits(cfg.metadataWidth bits)) init 0
      // Only the current eight-byte RAM word is assembled in fabric.  Every
      // eighth legal byte writes one complete native-width word.
      val captureChunk = Reg(Bits(cfg.bankWidth bits)) init 0
      val captureExpectedFirst = !capturePartial
      val captureExpectedLast =
        captureExpectedIndex === cfg.segmentBytes - 1
      val captureMetadataLegal = !capturePartial ||
        capturePop.payload.metadata === captureMetadata
      capturePop.ready := !fullReg && !stickyFault
      captureGroupBusy := capturePartial
      captureGroupActiveX := capturePartial
      captureBeatLegal :=
        capturePop.payload.slotIndex === 0 &&
          capturePop.payload.byteIndex === captureExpectedIndex &&
          capturePop.payload.first === captureExpectedFirst &&
          capturePop.payload.last === captureExpectedLast &&
          captureMetadataLegal && !capturePop.payload.fault
      captureWrite := capturePop.fire && captureBeatLegal
      captureCommit := captureWrite && captureExpectedLast
      captureWriteAddress :=
        writeBinary(cfg.addressWidth - 1 downto 0)
      captureWriteBank := captureExpectedIndex(
        cfg.byteIndexWidth - 1 downto log2Up(cfg.bytesPerBank))
        .resize(cfg.bankIndexWidth)
      val captureByteInBank = captureExpectedIndex(
        log2Up(cfg.bytesPerBank) - 1 downto 0)
      val nextCaptureChunk = Bits(cfg.bankWidth bits)
      nextCaptureChunk := captureChunk
      when(captureByteInBank === 0) {
        nextCaptureChunk := 0
      }
      for (byte <- 0 until cfg.bytesPerBank) {
        when(captureByteInBank === byte) {
          nextCaptureChunk((byte + 1) * 8 - 1 downto byte * 8) :=
            capturePop.payload.data
        }
      }
      val captureChunkComplete = captureByteInBank.andR
      captureMemoryWrite := captureWrite && captureChunkComplete
      captureWriteData := nextCaptureChunk
      captureWriteMask := B(
        (BigInt(1) << cfg.bytesPerBank) - 1,
        cfg.bytesPerBank bits)
      captureCommitMetadata := captureMetadata

      when(captureWrite) {
        when(captureChunkComplete) {
          captureChunk := 0
        } otherwise {
          captureChunk := nextCaptureChunk
        }
        when(!capturePartial) {
          captureMetadata := capturePop.payload.metadata
        }
        when(captureExpectedLast) {
          capturePartial := False
          captureExpectedIndex := 0
        } otherwise {
          capturePartial := True
          captureExpectedIndex := captureExpectedIndex + 1
        }
      }
    }

    val captureFire = capturePop.valid && capturePop.ready

    // Framing and metadata validation is intentionally kept off the native
    // RAM write-enable path.  The scatter validator contains several wide
    // comparisons; driving all sixteen BRAM byte enables directly from that
    // logic missed the 532 MHz fast-domain target.  One shared native-word
    // stage cuts that path without cloning a 1,024-bit payload register and
    // still accepts one capture byte on every fast edge.
    val captureMemoryWriteStage = RegNext(captureMemoryWrite) init False
    val captureWriteAddressStage = RegNext(captureWriteAddress) init 0
    val captureWriteBankStage = RegNext(captureWriteBank) init 0
    val captureWriteDataStage = RegNext(captureWriteData) init 0
    val captureWriteMaskStage = RegNext(captureWriteMask) init 0
    val captureCommitStage = RegNext(captureCommit) init False
    val captureCommitMetadataStage = RegNext(captureCommitMetadata) init 0

    io.replaySegment.ready := !fullReg && !stickyFault &&
      !captureGroupBusy && !capturePop.valid && !captureMemoryWriteStage
    val replayWrite = io.replaySegment.fire

    val writeAddress = writeBinary(
      cfg.addressWidth - 1 downto 0)

    for (bank <- 0 until cfg.bankCount) {
      val bankWriteData = Bits(cfg.bankWidth bits)
      val bankWriteMask = Bits(cfg.bytesPerBank bits)
      val bankWriteAddress = UInt(cfg.addressWidth bits)
      val bankWriteEnable = Bool()
      bankWriteData := captureWriteDataStage
      bankWriteMask := captureWriteMaskStage
      bankWriteAddress := captureWriteAddressStage
      bankWriteEnable := captureMemoryWriteStage &&
        captureWriteBankStage === bank
      when(replayWrite) {
        bankWriteData := io.replaySegment.payload.data(
          (bank + 1) * cfg.bankWidth - 1 downto bank * cfg.bankWidth)
        bankWriteMask := B(
          (BigInt(1) << cfg.bytesPerBank) - 1,
          cfg.bytesPerBank bits)
        bankWriteAddress := writeAddress
        bankWriteEnable := True
      }
      if (useVendorByteRam) {
        val memory = vendorDataBanks(bank)
        memory.clka := io.fastClk
        memory.addra := bankWriteAddress
        memory.dina := bankWriteData
        memory.ena := bankWriteEnable
        memory.wea := bankWriteMask
        memory.injectdbiterra := False
        memory.injectsbiterra := False
        memory.sleep := False
      } else {
        portableDataBanks(bank).write(
          address = bankWriteAddress,
          data = bankWriteData,
          enable = bankWriteEnable,
          mask = bankWriteMask)
      }
    }

    val committedMetadata = Bits((cfg.metadataWidth + 1) bits)
    committedMetadata := 0
    committedMetadata(cfg.metadataWidth - 1 downto 0) :=
      captureCommitMetadataStage
    committedMetadata(cfg.metadataWidth) := False
    when(replayWrite) {
      committedMetadata(cfg.metadataWidth - 1 downto 0) :=
        io.replaySegment.payload.metadata
      committedMetadata(cfg.metadataWidth) := True
    }
    segmentMetadataMemory.write(
      address = Mux(replayWrite, writeAddress, captureWriteAddressStage),
      data = committedMetadata,
      enable = captureCommitStage || replayWrite)

    when(captureFire && !captureBeatLegal) {
      // The current physical row was never committed: stop permanently and
      // require a session reset instead of exposing partial/stale bytes.
      stickyFault := True
    }

    val segmentCommit = captureCommit || replayWrite
    when(segmentCommit) {
      val nextWriteBinary = writeBinary + 1
      val nextWriteGray = binaryToGray(nextWriteBinary)
      writeBinary := nextWriteBinary
      writeGray := nextWriteGray
      fullReg := nextWriteGray === fullTarget
      fullReleasePending := False
      when(nextWriteGray === fullTarget) {
        fullReadGray := readGraySync
      }
    }

    when(fullReg) {
      when(fullReleasePending) {
        fullReg := False
        fullReleasePending := False
      } elsewhen(readGraySync =/= fullReadGray) {
        fullReleasePending := True
      }
    } otherwise {
      // A read-pointer move cannot reduce free space.  When no commit occurs,
      // the registered non-full permit therefore remains open.
      when(!segmentCommit) {
        fullReleasePending := False
      }
    }
  }
}
