package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetUnifiedResidentTensorScratchpadConfig(
    layout: BitNetUnifiedResidentTensorScratchpadLayout =
      BitNetUnifiedResidentTensorScratchpadLayout()) {
  require(layout.bankCount == 32)
  require(layout.wordWidth == layout.tokenLanes * layout.activationWidth)
  require(layout.rowCount == 512)
}

/** One S32 activation feature across all physical token lanes.  It must never
  * be narrowed to the packed-i8 word type.
  */
case class BitNetUnifiedResidentActivationWord(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val token = Vec(
    SInt(cfg.layout.activationWidth bits), cfg.layout.tokenLanes)
}

/** Four adjacent i8 features across all physical token lanes. */
case class BitNetUnifiedResidentPackedI8Word(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val feature = Vec(
    Vec(SInt(cfg.layout.packedWidth bits), cfg.layout.tokenLanes),
    cfg.layout.packedFeaturesPerWord)
}

case class BitNetUnifiedResidentActivationRead(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankMask = Bits(cfg.layout.bankCount bits)
}

case class BitNetUnifiedResidentActivationReadData(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankMask = Bits(cfg.layout.bankCount bits)
  val words = Vec(
    BitNetUnifiedResidentActivationWord(cfg), cfg.layout.bankCount)
  val fault = Bool()
  val faultCode = Bits(4 bits)
}

case class BitNetUnifiedResidentActivationWrite(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankMask = Bits(cfg.layout.bankCount bits)
  /** Per-bank S32 lane enables. */
  val elementMask = Vec(
    Bits(cfg.layout.tokenLanes bits), cfg.layout.bankCount)
  val words = Vec(
    BitNetUnifiedResidentActivationWord(cfg), cfg.layout.bankCount)
}

case class BitNetUnifiedResidentPackedI8Read(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankMask = Bits(cfg.layout.bankCount bits)
  /** Ignored outside QueryLmAlias. */
  val aliasOwner = BitNetUnifiedResidentTensorAliasOwner()
  /** Ignored outside QueryLmAlias. */
  val aliasGeneration = UInt(cfg.layout.aliasGenerationWidth bits)
}

case class BitNetUnifiedResidentPackedI8ReadData(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankMask = Bits(cfg.layout.bankCount bits)
  val words = Vec(
    BitNetUnifiedResidentPackedI8Word(cfg), cfg.layout.bankCount)
  val fault = Bool()
  val faultCode = Bits(4 bits)
}

case class BitNetUnifiedResidentPackedI8Write(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  private val layout = cfg.layout
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(layout.localRowWidth bits)
  /** Selects one fixed, contiguous bank group. */
  val segmentIndex = UInt(log2Up(layout.packedTransportSegmentCount) bits)
  /** Native 32-bit BRAM symbols updated in every selected bank.  The owning
    * endpoint defines the four-byte layout inside a symbol; the physical RAM
    * always treats it as one indivisible write unit.  A zero mask is illegal.
    */
  val symbolMask = Bits(layout.nativeSymbolCount bits)
  /** Raw physical words for the selected bank group.  Inactive 32-bit symbols
    * are preserved without decomposing the RAM into byte-write memories.
    */
  val segmentWords = Vec(
    Bits(layout.wordWidth bits), layout.packedTransportWordsPerSegment)
  /** Ignored outside QueryLmAlias. */
  val aliasOwner = BitNetUnifiedResidentTensorAliasOwner()
  /** Ignored outside QueryLmAlias. */
  val aliasGeneration = UInt(layout.aliasGenerationWidth bits)
}

case class BitNetUnifiedResidentAliasAcquire(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val owner = BitNetUnifiedResidentTensorAliasOwner()
}

case class BitNetUnifiedResidentAliasGrant(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val owner = BitNetUnifiedResidentTensorAliasOwner()
  val generation = UInt(cfg.layout.aliasGenerationWidth bits)
}

case class BitNetUnifiedResidentAliasRelease(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val owner = BitNetUnifiedResidentTensorAliasOwner()
  val generation = UInt(cfg.layout.aliasGenerationWidth bits)
}

object BitNetUnifiedResidentTensorFault {
  val WrongFormat = 0
  val RowOutOfRange = 1
  val AliasOwnership = 2
  val ReadWriteCollision = 3
}

object BitNetUnifiedResidentAliasLeaseState
    extends SpinalEnum(binarySequential) {
  val Idle, Grant, Owned = newElement()
}

/**
  * Physical backend for all on-chip resident tensor images.
  *
  * Exactly 32 memories of `wordWidth` x 512 rows are inferred.  The activation
  * and packed-i8 ports are typed separately and arbitrate one synchronous read
  * plus one masked write per bank per clock.  A later fast-clock integration
  * wrapper may dual-pump these ports; this component intentionally specifies
  * correctness, address ownership and collision behavior first.
  *
  * Projection ping/pong regions are independent.  Query and LM use the same
  * final 20-row region, so their accesses additionally carry a private lease
  * generation.  A stale or non-owner access is accepted as a fault response
  * (reads) or dropped (writes); it can never modify the shared rows.
  */
class BitNetUnifiedResidentTensorScratchpad(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()) extends Component {
  private val layout = cfg.layout

  val io = new Bundle {
    val activationRead = slave(Stream(
      BitNetUnifiedResidentActivationRead(cfg)))
    val activationReadData = master(Stream(
      BitNetUnifiedResidentActivationReadData(cfg)))
    val activationWrite = slave(Stream(
      BitNetUnifiedResidentActivationWrite(cfg)))

    val packedRead = slave(Stream(
      BitNetUnifiedResidentPackedI8Read(cfg)))
    val packedReadData = master(Stream(
      BitNetUnifiedResidentPackedI8ReadData(cfg)))
    val packedWrite = slave(Stream(
      BitNetUnifiedResidentPackedI8Write(cfg)))

    val aliasAcquire = slave(Stream(
      BitNetUnifiedResidentAliasAcquire(cfg)))
    val aliasGrant = master(Stream(
      BitNetUnifiedResidentAliasGrant(cfg)))
    val aliasRelease = slave(Stream(
      BitNetUnifiedResidentAliasRelease(cfg)))

    val aliasOwnerValid = out Bool()
    val aliasOwner = out(BitNetUnifiedResidentTensorAliasOwner())
    val aliasGeneration = out UInt(layout.aliasGenerationWidth bits)
    val readResponsePending = out Bool()
    val formatFaultSticky = out Bool()
    val rangeFaultSticky = out Bool()
    val aliasFaultSticky = out Bool()
    val collisionFaultSticky = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedReads = out UInt(32 bits)
    val acceptedWrites = out UInt(32 bits)
    val committedWrites = out UInt(32 bits)
  }

  val memories = Array.tabulate(layout.bankCount) { bank =>
    val memory = Mem(Bits(layout.wordWidth bits), layout.rowCount)
    memory.setName(s"residentTensorBank${bank}")
    memory.addAttribute("ram_style", "block")
    memory
  }

  private val regionCount = BitNetUnifiedResidentTensorRegion.elements.size
  val regionBase = Vec(UInt(layout.rowAddressWidth bits), regionCount)
  val regionRows = Vec(UInt((layout.localRowWidth + 1) bits), regionCount)
  val regionIsActivation = Vec(Bool(), regionCount)
  val regionIsPacked = Vec(Bool(), regionCount)
  BitNetUnifiedResidentTensorRegion.elements.zipWithIndex.foreach {
    case (region, encoding) =>
      val descriptor = layout.descriptor(region)
      regionBase(encoding) := U(descriptor.baseRow,
        layout.rowAddressWidth bits)
      regionRows(encoding) := U(descriptor.rowCount,
        (layout.localRowWidth + 1) bits)
      regionIsActivation(encoding) :=
        (if (descriptor.format ==
          BitNetUnifiedResidentTensorWordFormat.ActivationS32x4) True
        else False)
      regionIsPacked(encoding) :=
        (if (descriptor.format ==
          BitNetUnifiedResidentTensorWordFormat.PackedI8x16) True
        else False)
  }

  // ---------------------------------------------------------------------
  // Private lease for the Q/LM physical alias.
  // ---------------------------------------------------------------------
  val aliasState = Reg(BitNetUnifiedResidentAliasLeaseState()) init
    BitNetUnifiedResidentAliasLeaseState.Idle
  val aliasOwnerReg = Reg(BitNetUnifiedResidentTensorAliasOwner()) init
    BitNetUnifiedResidentTensorAliasOwner.Query
  val aliasGenerationReg = Reg(UInt(layout.aliasGenerationWidth bits)) init 0
  val nextAliasGeneration = Reg(UInt(layout.aliasGenerationWidth bits)) init 1
  val aliasOwnerValidReg = RegInit(False)

  io.aliasOwnerValid := aliasOwnerValidReg
  io.aliasOwner := aliasOwnerReg
  io.aliasGeneration := aliasGenerationReg

  io.aliasAcquire.ready :=
    aliasState === BitNetUnifiedResidentAliasLeaseState.Idle
  when(io.aliasAcquire.fire) {
    aliasOwnerReg := io.aliasAcquire.owner
    aliasGenerationReg := nextAliasGeneration
    when(nextAliasGeneration.andR) {
      nextAliasGeneration := 1
    } otherwise {
      nextAliasGeneration := nextAliasGeneration + 1
    }
    aliasState := BitNetUnifiedResidentAliasLeaseState.Grant
  }

  io.aliasGrant.valid :=
    aliasState === BitNetUnifiedResidentAliasLeaseState.Grant
  io.aliasGrant.owner := aliasOwnerReg
  io.aliasGrant.generation := aliasGenerationReg
  when(io.aliasGrant.fire) {
    aliasOwnerValidReg := True
    aliasState := BitNetUnifiedResidentAliasLeaseState.Owned
  }

  // ---------------------------------------------------------------------
  // One elastic synchronous-read stage shared by the two typed views.
  // ---------------------------------------------------------------------
  val responseValid = RegInit(False)
  val responsePacked = RegInit(False)
  val responseRegion = Reg(BitNetUnifiedResidentTensorRegion()) init
    BitNetUnifiedResidentTensorRegion.Hidden
  val responseLocalRow = Reg(UInt(layout.localRowWidth bits)) init 0
  val responseBankMask = Reg(Bits(layout.bankCount bits)) init 0
  val responseLegal = RegInit(False)
  val responseFaultCode = Reg(Bits(4 bits)) init 0

  io.activationReadData.valid := responseValid && !responsePacked
  io.packedReadData.valid := responseValid && responsePacked
  io.activationReadData.region := responseRegion
  io.activationReadData.localRow := responseLocalRow
  io.activationReadData.bankMask := responseBankMask
  io.activationReadData.fault := !responseLegal
  io.activationReadData.faultCode := responseFaultCode
  io.packedReadData.region := responseRegion
  io.packedReadData.localRow := responseLocalRow
  io.packedReadData.bankMask := responseBankMask
  io.packedReadData.fault := !responseLegal
  io.packedReadData.faultCode := responseFaultCode

  val selectedResponseReady = Mux(responsePacked,
    io.packedReadData.ready, io.activationReadData.ready)
  val readCanIssue = !responseValid || selectedResponseReady
  val preferPackedRead = RegInit(False)
  val choosePackedRead = io.packedRead.valid &&
    (!io.activationRead.valid || preferPackedRead)
  val chooseActivationRead = io.activationRead.valid && !choosePackedRead
  io.activationRead.ready := readCanIssue && chooseActivationRead
  io.packedRead.ready := readCanIssue && choosePackedRead
  val selectedReadFire = io.activationRead.fire || io.packedRead.fire
  val selectedReadPacked = io.packedRead.fire

  val selectedReadRegion = BitNetUnifiedResidentTensorRegion()
  val selectedReadLocalRow = UInt(layout.localRowWidth bits)
  val selectedReadBankMask = Bits(layout.bankCount bits)
  val selectedReadAliasOwner = BitNetUnifiedResidentTensorAliasOwner()
  val selectedReadAliasGeneration =
    UInt(layout.aliasGenerationWidth bits)
  selectedReadRegion := io.activationRead.region
  selectedReadLocalRow := io.activationRead.localRow
  selectedReadBankMask := io.activationRead.bankMask
  selectedReadAliasOwner := BitNetUnifiedResidentTensorAliasOwner.Query
  selectedReadAliasGeneration := 0
  when(selectedReadPacked) {
    selectedReadRegion := io.packedRead.region
    selectedReadLocalRow := io.packedRead.localRow
    selectedReadBankMask := io.packedRead.bankMask
    selectedReadAliasOwner := io.packedRead.aliasOwner
    selectedReadAliasGeneration := io.packedRead.aliasGeneration
  }

  val selectedReadRegionIndex = selectedReadRegion.asBits.asUInt
  val selectedReadFormatLegal = Mux(selectedReadPacked,
    regionIsPacked(selectedReadRegionIndex),
    regionIsActivation(selectedReadRegionIndex))
  val selectedReadRowLegal =
    selectedReadLocalRow.resize(layout.localRowWidth + 1) <
      regionRows(selectedReadRegionIndex)
  val selectedReadTargetsAlias = selectedReadPacked &&
    selectedReadRegion === BitNetUnifiedResidentTensorRegion.QueryLmAlias
  val selectedReadAliasLegal = !selectedReadTargetsAlias ||
    (aliasState === BitNetUnifiedResidentAliasLeaseState.Owned &&
      aliasOwnerValidReg && selectedReadAliasOwner === aliasOwnerReg &&
      selectedReadAliasGeneration === aliasGenerationReg)
  val selectedReadLegal = selectedReadFormatLegal &&
    selectedReadRowLegal && selectedReadAliasLegal
  val selectedReadAbsoluteRow =
    (regionBase(selectedReadRegionIndex).resize(layout.rowAddressWidth + 1) +
      selectedReadLocalRow.resize(layout.rowAddressWidth + 1))
      .resize(layout.rowAddressWidth)

  val selectedReadFaultCode = Bits(4 bits)
  selectedReadFaultCode := 0
  selectedReadFaultCode(BitNetUnifiedResidentTensorFault.WrongFormat) :=
    !selectedReadFormatLegal
  selectedReadFaultCode(BitNetUnifiedResidentTensorFault.RowOutOfRange) :=
    !selectedReadRowLegal
  selectedReadFaultCode(BitNetUnifiedResidentTensorFault.AliasOwnership) :=
    !selectedReadAliasLegal
  selectedReadFaultCode(
    BitNetUnifiedResidentTensorFault.ReadWriteCollision) := False

  val memoryReadData = memories.zipWithIndex.map { case (memory, bank) =>
    memory.readSync(
      address = selectedReadAbsoluteRow,
      enable = selectedReadFire && selectedReadLegal &&
        selectedReadBankMask(bank))
  }

  when(readCanIssue) {
    responseValid := selectedReadFire
    when(selectedReadFire) {
      responsePacked := selectedReadPacked
      responseRegion := selectedReadRegion
      responseLocalRow := selectedReadLocalRow
      responseBankMask := selectedReadBankMask
      responseLegal := selectedReadLegal
      responseFaultCode := selectedReadFaultCode
      preferPackedRead := !selectedReadPacked
    }
  }

  for (bank <- 0 until layout.bankCount) {
    val returnedWord = Bits(layout.wordWidth bits)
    returnedWord := B(0, layout.wordWidth bits)
    when(responseLegal && responseBankMask(bank)) {
      returnedWord := memoryReadData(bank)
    }
    io.activationReadData.words(bank).assignFromBits(returnedWord)
    io.packedReadData.words(bank).assignFromBits(returnedWord)
  }

  // ---------------------------------------------------------------------
  // One masked write port per physical bank, shared by both typed views.
  // ---------------------------------------------------------------------
  val preferPackedWrite = RegInit(False)
  val choosePackedWrite = io.packedWrite.valid &&
    (!io.activationWrite.valid || preferPackedWrite)
  val chooseActivationWrite = io.activationWrite.valid && !choosePackedWrite
  io.activationWrite.ready := chooseActivationWrite
  io.packedWrite.ready := choosePackedWrite
  val selectedWriteFire = io.activationWrite.fire || io.packedWrite.fire
  val selectedWritePacked = io.packedWrite.fire

  val selectedWriteRegion = BitNetUnifiedResidentTensorRegion()
  val selectedWriteLocalRow = UInt(layout.localRowWidth bits)
  val selectedWriteBankMask = Bits(layout.bankCount bits)
  val selectedWriteAliasOwner = BitNetUnifiedResidentTensorAliasOwner()
  val selectedWriteAliasGeneration =
    UInt(layout.aliasGenerationWidth bits)
  val selectedWriteWords = Vec(Bits(layout.wordWidth bits), layout.bankCount)
  val selectedWriteElementMasks = Vec(
    Bits(layout.nativeSymbolCount bits), layout.bankCount)
  selectedWriteRegion := io.activationWrite.region
  selectedWriteLocalRow := io.activationWrite.localRow
  // A bank selected without any token lane is a legal no-op, not a physical
  // commit.  Folding elementMask.orR into the effective mask prevents both a
  // false committedWrites increment and a zero-byte RAM write transaction.
  for (bank <- 0 until layout.bankCount) {
    selectedWriteBankMask(bank) := io.activationWrite.bankMask(bank) &&
      io.activationWrite.elementMask(bank).orR
  }
  selectedWriteAliasOwner := BitNetUnifiedResidentTensorAliasOwner.Query
  selectedWriteAliasGeneration := 0
  for (bank <- 0 until layout.bankCount) {
    selectedWriteWords(bank) := io.activationWrite.words(bank).asBits
    selectedWriteElementMasks(bank) :=
      io.activationWrite.elementMask(bank)
  }
  when(selectedWritePacked) {
    selectedWriteRegion := io.packedWrite.region
    selectedWriteLocalRow := io.packedWrite.localRow
    selectedWriteAliasOwner := io.packedWrite.aliasOwner
    selectedWriteAliasGeneration := io.packedWrite.aliasGeneration
    for (bank <- 0 until layout.bankCount) {
      val segment = bank / layout.packedTransportWordsPerSegment
      selectedWriteBankMask(bank) :=
        io.packedWrite.segmentIndex === segment &&
          io.packedWrite.symbolMask.orR
      selectedWriteWords(bank) := io.packedWrite.segmentWords(
        bank % layout.packedTransportWordsPerSegment)
      selectedWriteElementMasks(bank) := io.packedWrite.symbolMask
    }
  }

  val selectedWriteRegionIndex = selectedWriteRegion.asBits.asUInt
  // symbolMask has exactly nativeSymbolCount bits, so its type excludes
  // out-of-range native symbols.  Zero is a packed-write format error.
  val selectedWriteFormatLegal = Mux(selectedWritePacked,
    regionIsPacked(selectedWriteRegionIndex) && io.packedWrite.symbolMask.orR,
    regionIsActivation(selectedWriteRegionIndex))
  val selectedWriteRowLegal =
    selectedWriteLocalRow.resize(layout.localRowWidth + 1) <
      regionRows(selectedWriteRegionIndex)
  val selectedWriteTargetsAlias = selectedWritePacked &&
    selectedWriteRegion === BitNetUnifiedResidentTensorRegion.QueryLmAlias
  val selectedWriteAliasLegal = !selectedWriteTargetsAlias ||
    (aliasState === BitNetUnifiedResidentAliasLeaseState.Owned &&
      aliasOwnerValidReg && selectedWriteAliasOwner === aliasOwnerReg &&
      selectedWriteAliasGeneration === aliasGenerationReg)
  val selectedWriteLegal = selectedWriteFormatLegal &&
    selectedWriteRowLegal && selectedWriteAliasLegal
  val selectedWriteAbsoluteRow =
    (regionBase(selectedWriteRegionIndex).resize(layout.rowAddressWidth + 1) +
      selectedWriteLocalRow.resize(layout.rowAddressWidth + 1))
      .resize(layout.rowAddressWidth)

  when(selectedWriteFire) {
    preferPackedWrite := !selectedWritePacked
  }

  val collisionBits = Bits(layout.bankCount bits)
  for (bank <- 0 until layout.bankCount) {
    collisionBits(bank) := selectedReadFire && selectedReadLegal &&
      selectedWriteFire && selectedWriteLegal &&
      selectedReadAbsoluteRow === selectedWriteAbsoluteRow &&
      selectedReadBankMask(bank) && selectedWriteBankMask(bank)
    memories(bank).write(
      address = selectedWriteAbsoluteRow,
      data = selectedWriteWords(bank),
      enable = selectedWriteFire && selectedWriteLegal &&
        selectedWriteBankMask(bank) && !collisionBits(bank),
      mask = selectedWriteElementMasks(bank))
  }

  // An alias cannot retire while an alias response or request is live.  A
  // mismatched release is consumed, reported and leaves the lease unchanged.
  val aliasResponsePending = responseValid &&
    responseRegion === BitNetUnifiedResidentTensorRegion.QueryLmAlias
  val aliasAccessPresented =
    (io.packedRead.valid &&
      io.packedRead.region ===
        BitNetUnifiedResidentTensorRegion.QueryLmAlias) ||
    (io.packedWrite.valid &&
      io.packedWrite.region ===
        BitNetUnifiedResidentTensorRegion.QueryLmAlias)
  io.aliasRelease.ready :=
    aliasState === BitNetUnifiedResidentAliasLeaseState.Owned &&
      !aliasResponsePending && !aliasAccessPresented
  val aliasReleaseLegal = io.aliasRelease.owner === aliasOwnerReg &&
    io.aliasRelease.generation === aliasGenerationReg
  when(io.aliasRelease.fire && aliasReleaseLegal) {
    aliasOwnerValidReg := False
    aliasState := BitNetUnifiedResidentAliasLeaseState.Idle
  }

  // ---------------------------------------------------------------------
  // Sticky diagnostics and small accounting counters.
  // ---------------------------------------------------------------------
  val formatFault = RegInit(False)
  val rangeFault = RegInit(False)
  val aliasFault = RegInit(False)
  val collisionFault = RegInit(False)
  val acceptedReadCount = Reg(UInt(32 bits)) init 0
  val acceptedWriteCount = Reg(UInt(32 bits)) init 0
  val committedWriteCount = Reg(UInt(32 bits)) init 0

  when(selectedReadFire) {
    acceptedReadCount := acceptedReadCount + 1
    when(!selectedReadFormatLegal) { formatFault := True }
    when(!selectedReadRowLegal) { rangeFault := True }
    when(!selectedReadAliasLegal) { aliasFault := True }
  }
  when(selectedWriteFire) {
    acceptedWriteCount := acceptedWriteCount + 1
    when(!selectedWriteFormatLegal) { formatFault := True }
    when(!selectedWriteRowLegal) { rangeFault := True }
    when(!selectedWriteAliasLegal) { aliasFault := True }
    when(selectedWriteLegal && (selectedWriteBankMask & ~collisionBits).orR) {
      committedWriteCount := committedWriteCount + 1
    }
  }
  when(collisionBits.orR) { collisionFault := True }
  when(io.aliasRelease.fire && !aliasReleaseLegal) { aliasFault := True }

  io.readResponsePending := responseValid
  io.formatFaultSticky := formatFault
  io.rangeFaultSticky := rangeFault
  io.aliasFaultSticky := aliasFault
  io.collisionFaultSticky := collisionFault
  io.protocolFaultSticky :=
    formatFault || rangeFault || aliasFault || collisionFault
  io.acceptedReads := acceptedReadCount
  io.acceptedWrites := acceptedWriteCount
  io.committedWrites := committedWriteCount
}

object GenerateBitNetUnifiedResidentTensorScratchpad extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-tensor-scratchpad")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedResidentTensorScratchpad())
}
