package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Geometry of the BRAM-staged QueryLmAlias client. */
case class BitNetUnifiedResidentQueryAdapterConfig(
    resident: BitNetQResidentBufferConfig =
      BitNetQResidentBufferConfig(),
    scratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()) {
  require(resident.model.tokenParallelism == 2)
  require(resident.model.queryHeadCount == 20)
  require(resident.model.gqaGroupSize == 4)
  require(resident.model.headSize == 128)
  require(scratchpad.layout.tokenLanes == resident.model.tokenParallelism)
  require(scratchpad.layout.bankCount == 32)
  require(scratchpad.layout.packedFeaturesPerWord == 4)
  require(scratchpad.layout.nativeSymbolCount == 2)
  require(scratchpad.layout.queryLmAlias.rowCount ==
    resident.model.queryHeadCount)

  val factorIndexWidth: Int =
    log2Up(resident.model.queryHeadCount * resident.model.tokenParallelism)
}

/**
  * Identity-authenticated retirement of one resident Q image.
  *
  * The complete load identity is repeated deliberately.  A late controller
  * command from an older request, even if it happens to reuse the same layer
  * or token base, therefore cannot release the current QueryLmAlias lease.
  */
case class BitNetUnifiedResidentQueryRetire(
    cfg: BitNetUnifiedResidentQueryAdapterConfig) extends Bundle {
  val requestId = UInt(cfg.resident.model.requestIdWidth bits)
  val layer = UInt(cfg.resident.model.layerWidth bits)
  val mode = BitNetMode()
  val tokenBase = UInt(cfg.resident.model.sequenceWidth bits)
  val tokenId = Vec(
    UInt(cfg.resident.model.sequenceWidth bits),
    cfg.resident.model.tokenParallelism)
  val tokenMask = Bits(cfg.resident.model.tokenParallelism bits)
  val aliasGeneration = UInt(
    cfg.scratchpad.layout.aliasGenerationWidth bits)
}

/**
  * BRAM-staged resident-query adapter for the unified tensor image.
  *
  * Query owns a slot-major physical layout inside QueryLmAlias.  Each 32-bit
  * native BRAM symbol contains four adjacent query elements.  Prefill stores
  * token lane `k` in the packed feature/lane position of query-head row `h`;
  * decode stores query head `h` in symbol `h & 1` of subpass row `h >> 1`.
  * Each scalar sample writes
  * one byte into a private segment buffer, while the semantic packed write
  * commits either both prefill symbols or exactly one decode-head symbol.
  * Complete segments remain Stream-stable until the semantic frontend reports
  * an authenticated write commit.  Consequently a successful load audits
  * exactly twenty logical heads and never exposes a partial image.
  *
  * Both prefill and decode fetch exactly one full row per pass, then recover
  * query lane `k` from physical symbol `k`.  The packed response itself remains
  * owned by the semantic frontend while QK borrows it; this adapter retains
  * only factor/identity metadata and a one-bit registered release boundary.
  * The write side uses a shallow native byte-write BRAM ring rather than a pair
  * of 1,024-bit fabric-register segment assemblies.
  *
  * `semanticPacked*Outstanding` are the frontend's authoritative status
  * signals.  Fault, retire, and abort prevent every new packed transaction;
  * an already-retained terminal write remains valid until its real commit or
  * fault, then both outstanding signals must clear before alias release.
  */
class BitNetUnifiedResidentQueryAdapter(
    cfg: BitNetUnifiedResidentQueryAdapterConfig =
      BitNetUnifiedResidentQueryAdapterConfig(),
    useVendorSegmentRam: Boolean = true) extends Component {
  private val residentCfg = cfg.resident
  private val scratchpadCfg = cfg.scratchpad
  private val model = residentCfg.model
  private val layout = scratchpadCfg.layout
  private val lanes = model.tokenParallelism
  private val heads = model.queryHeadCount
  private val allBanks = B(
    (BigInt(1) << layout.bankCount) - 1, layout.bankCount bits)

  val io = new Bundle {
    // North interface: field-for-field replacement for QResidentBuffer.
    val load = slave(Stream(BitNetQResidentLoadCommand(residentCfg)))
    val sample = slave(Stream(BitNetQkvQuantizedSample(residentCfg.qkv)))
    val loadCompletion = master(Stream(
      BitNetQResidentLoadCompletion(residentCfg)))
    val lookup = slave(Stream(BitNetAttentionPass(model)))
    // The QK assembler borrows the retained packed response in place.  It keeps
    // ready low for the full QK pass and fires only to release the payload after
    // the final raw score; abort revokes the payload directly.
    val block = master(Stream(BitNetQkQueryBlock(residentCfg.qk)))

    // A normal lifetime end is identity checked.  Abort is the explicit
    // cleanup path for an upstream QKV/KV failure which may not have a useful
    // completion identity left to present.
    val retire = slave(Stream(
      BitNetUnifiedResidentQueryRetire(cfg)))
    val abort = in Bool()

    // South interface: direct connection to the single semantic frontend.
    val packedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(scratchpadCfg)))
    val packedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(scratchpadCfg)))
    val packedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(scratchpadCfg)))
    val packedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(scratchpadCfg)))
    val aliasAcquire = master(Stream(
      BitNetUnifiedResidentAliasAcquire(scratchpadCfg)))
    val aliasGrant = slave(Stream(
      BitNetUnifiedResidentAliasGrant(scratchpadCfg)))
    val aliasRelease = master(Stream(
      BitNetUnifiedResidentAliasRelease(scratchpadCfg)))
    // Asserted by the shared alias router from the moment it has locked this
    // client, including the interval before the backend acquire can fire.
    // A direct (unrouted) frontend may tie this low: before its Stream fire
    // there is then no retained downstream source obligation to drain.
    val aliasAcquireOwned = in Bool()
    val semanticPackedReadOutstanding = in Bool()
    val semanticPackedWriteOutstanding = in Bool()

    val loaded = out Bool()
    val busy = out Bool()
    val leaseHeld = out Bool()
    val aliasGeneration = out UInt(layout.aliasGenerationWidth bits)
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val writeFaultMetadataSticky = out Bool()
    val acceptedLoads = out UInt(32 bits)
    val completedLoads = out UInt(32 bits)
    val completedLookups = out UInt(32 bits)
    val committedRows = out UInt(32 bits)
    val issuedReads = out UInt(32 bits)
  }

  // ------------------------------------------------------------------
  // Lifetime, lease, completion, and fail-close state.
  // ------------------------------------------------------------------
  val activeLoad = Reg(BitNetQResidentLoadCommand(residentCfg)) init
    BitNetQResidentLoadCommand(residentCfg).getZero
  val loadInFlight = RegInit(False)
  val loadActive = RegInit(False)
  val imageLoaded = RegInit(False)
  val acquirePending = RegInit(False)
  val waitingGrant = RegInit(False)
  val cancelWaitingGrant = RegInit(False)
  val leaseHeldReg = RegInit(False)
  val leaseGeneration = Reg(UInt(layout.aliasGenerationWidth bits)) init 0

  val releasePending = RegInit(False)
  val releaseOwner = Reg(BitNetUnifiedResidentTensorAliasOwner()) init
    BitNetUnifiedResidentTensorAliasOwner.Query
  val releaseGeneration = Reg(UInt(layout.aliasGenerationWidth bits)) init 0

  // Two alternating BRAM rows let scalar Q loading continue while the
  // preceding 1,024-bit segment is committed and audited.  The wide output is
  // formed directly from the sixteen native 64-bit RAM dout ports.
  private val packedSegmentCount = layout.packedTransportSegmentCount
  private val packedSegmentWordCount = layout.bankCount / packedSegmentCount
  require(packedSegmentCount == 2)
  require(packedSegmentWordCount == 16)
  private val segmentMetadataSegmentLow = 0
  private val segmentMetadataRowLow =
    segmentMetadataSegmentLow + log2Up(packedSegmentCount)
  private val segmentMetadataSymbolMaskLow =
    segmentMetadataRowLow + layout.localRowWidth
  private val segmentMetadataGenerationLow =
    segmentMetadataSymbolMaskLow + layout.nativeSymbolCount
  private val segmentMetadataWidth =
    segmentMetadataGenerationLow + layout.aliasGenerationWidth
  private val segmentBufferCfg =
    BitNetPackedSegmentWordWriteBufferConfig(
      segmentWidth = layout.packedTransportSegmentWidth,
      bankWidth = 64,
      ringDepth = 2,
      metadataWidth = segmentMetadataWidth)
  val segmentBuffer = new BitNetPackedSegmentWordWriteBuffer(
    segmentBufferCfg, useVendorByteRam = useVendorSegmentRam)
  segmentBuffer.setName("querySegmentBuffer")
  // This pointer follows actual physical commits.  In particular, a partial
  // row completed only for abort cleanup still consumes its real ring slot.
  val segmentWriteSlot =
    Reg(UInt(segmentBufferCfg.slotAddressWidth bits)) init 0
  val retainedWriteLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val retainedWriteAliasGeneration =
    Reg(UInt(layout.aliasGenerationWidth bits)) init 0
  val retainedWriteSegment =
    Reg(UInt(log2Up(packedSegmentCount) bits)) init 0
  val retainedWriteSymbolMask =
    Reg(Bits(layout.nativeSymbolCount bits)) init 0
  val writeAuditActive = RegInit(False)
  val writeOffered = RegInit(False)
  val segmentBufferDrained = segmentBuffer.io.protocolFaultSticky ||
    (segmentBuffer.io.empty && !segmentBuffer.io.partial &&
      !segmentBuffer.io.segment.valid)
  val packedWritePending = !segmentBufferDrained || writeAuditActive ||
    writeOffered
  val abortWriteOwnershipProbe = RegInit(False)
  // Abort is allowed to cancel an acquire/read only after one complete cycle
  // in which the corresponding shared router had an opportunity to expose a
  // just-selected owner.  If ownership appears, the adapter retains valid to
  // the real southbound fire and drains the resulting grant/read response.
  val abortAcquireOwnershipProbe = RegInit(False)
  val abortReadOwnershipProbe = RegInit(False)
  val abortReadDrain = RegInit(False)
  val discardReadResponse = RegInit(False)

  val stickyFault = RegInit(False)
  val quarantineReg = RegInit(False)
  val writeFaultMetadataFault = RegInit(False)

  val completionValid = RegInit(False)
  val completionPayload = Reg(
    BitNetQResidentLoadCompletion(residentCfg)) init
      BitNetQResidentLoadCompletion(residentCfg).getZero
  io.loadCompletion.valid := completionValid
  io.loadCompletion.payload := completionPayload
  // An immediately rejected semantic write reports its registered fault one
  // cycle after north fire.  If that fire completed row twenty, success is
  // already pending; mask it combinationally in the fault cycle so an always-
  // ready consumer can never observe a false success.
  when(io.packedWriteFault.valid) {
    io.loadCompletion.payload.success := False
  }
  when(io.loadCompletion.fire) {
    completionValid := False
  }

  val acceptedLoadCount = Reg(UInt(32 bits)) init 0
  val completedLoadCount = Reg(UInt(32 bits)) init 0
  val completedLookupCount = Reg(UInt(32 bits)) init 0
  val committedRowCount = Reg(UInt(32 bits)) init 0
  val issuedReadCount = Reg(UInt(32 bits)) init 0
  io.acceptedLoads := acceptedLoadCount
  io.completedLoads := completedLoadCount
  io.completedLookups := completedLookupCount
  io.committedRows := committedRowCount
  io.issuedReads := issuedReadCount

  val legalPrefillLoad = io.load.mode === BitNetMode.Prefill &&
    (1 to lanes).map { count =>
      io.load.tokenMask === B((BigInt(1) << count) - 1, lanes bits)
    }.reduce(_ || _)
  val legalDecodeLoad = io.load.mode === BitNetMode.Decode &&
    io.load.tokenMask === B(1, lanes bits)
  val loadCommandLegal = io.load.layer < model.layerCount &&
    (legalPrefillLoad || legalDecodeLoad)

  val noSemanticOutstanding = !io.semanticPackedReadOutstanding &&
    !io.semanticPackedWriteOutstanding
  val operationalIdle = !loadInFlight && !loadActive && !imageLoaded &&
    !acquirePending && !waitingGrant && !leaseHeldReg &&
    !releasePending && !packedWritePending
  io.load.ready := operationalIdle && !completionValid &&
    !quarantineReg && !io.abort

  when(io.load.fire) {
    activeLoad := io.load.payload
    acceptedLoadCount := acceptedLoadCount + 1
    when(loadCommandLegal) {
      loadInFlight := True
      acquirePending := True
    } otherwise {
      stickyFault := True
      quarantineReg := True
      completionValid := True
      completionPayload.requestId := io.load.requestId
      completionPayload.layer := io.load.layer
      completionPayload.tokenBase := io.load.tokenBase
      completionPayload.tokenMask := io.load.tokenMask
      completionPayload.success := False
      completedLoadCount := completedLoadCount + 1
    }
  }

  val cancelUnownedAcquireNow = abortAcquireOwnershipProbe &&
    !io.aliasAcquireOwned
  io.aliasAcquire.valid := acquirePending && !quarantineReg &&
    !cancelUnownedAcquireNow
  io.aliasAcquire.owner := BitNetUnifiedResidentTensorAliasOwner.Query
  when(io.aliasAcquire.fire) {
    acquirePending := False
    waitingGrant := True
    abortAcquireOwnershipProbe := False
  }

  // Grants are always drained.  This prevents a malformed/late grant from
  // wedging a shared frontend, while every unexpected grant still fail-closes
  // this client.
  io.aliasGrant.ready := True
  val expectedGrant = waitingGrant &&
    io.aliasGrant.owner === BitNetUnifiedResidentTensorAliasOwner.Query &&
    io.aliasGrant.generation =/= 0
  val grantFaultNow = io.aliasGrant.valid && !expectedGrant
  when(io.aliasGrant.fire) {
    when(expectedGrant) {
      waitingGrant := False
      leaseHeldReg := True
      leaseGeneration := io.aliasGrant.generation
      when(cancelWaitingGrant || io.abort) {
        cancelWaitingGrant := False
        releasePending := True
        releaseOwner := BitNetUnifiedResidentTensorAliasOwner.Query
        releaseGeneration := io.aliasGrant.generation
      } otherwise {
        loadActive := True
      }
    } otherwise {
      stickyFault := True
      quarantineReg := True
      waitingGrant := False
      acquirePending := False
      loadActive := False
      imageLoaded := False
      when(loadInFlight && !completionValid) {
        loadInFlight := False
        completionValid := True
        completionPayload.requestId := activeLoad.requestId
        completionPayload.layer := activeLoad.layer
        completionPayload.tokenBase := activeLoad.tokenBase
        completionPayload.tokenMask := activeLoad.tokenMask
        completionPayload.success := False
        completedLoadCount := completedLoadCount + 1
      }
      // A non-zero malformed grant may represent a real frontend lease.  Use
      // its own owner/generation for a best-effort release.  Generation zero
      // is never emitted on release.
      when(io.aliasGrant.generation =/= 0) {
        releasePending := True
        releaseOwner := io.aliasGrant.owner
        releaseGeneration := io.aliasGrant.generation
      }
    }
  }

  // ------------------------------------------------------------------
  // Scalar Q load -> four authenticated packed segments per logical head.
  // ------------------------------------------------------------------
  val factors = Vec((0 until heads * lanes).map { _ =>
    Reg(UInt(32 bits)) init 0
  })
  val headValid = Reg(Bits(heads bits)) init 0
  val headCaptured = Reg(Bits(heads bits)) init 0
  val frameActive = RegInit(False)
  val activeHead = Reg(UInt(residentCfg.qkv.headWidth bits)) init 0
  val expectedElement =
    Reg(UInt(residentCfg.qkv.elementWidth bits)) init 0
  val expectedLane =
    Reg(UInt(residentCfg.qkv.tokenLaneWidth bits)) init 0

  when(io.aliasGrant.fire && expectedGrant &&
      !cancelWaitingGrant && !io.abort) {
    headValid := 0
    headCaptured := 0
    frameActive := False
    expectedElement := 0
    expectedLane := 0
    writeAuditActive := False
    writeOffered := False
    abortWriteOwnershipProbe := False
  }

  val expectedFirst = !frameActive && expectedElement === 0 &&
    expectedLane === 0
  val expectedLast = frameActive &&
    expectedElement === model.headSize - 1 &&
    expectedLane === lanes - 1
  val sampleHeadInRange = io.sample.head < heads
  val sampleHeadOneHot = UIntToOh(io.sample.head, heads)
  val sampleHeadDuplicate = (headCaptured & sampleHeadOneHot).orR
  val sampleFactorIndex =
    (io.sample.head.resize(cfg.factorIndexWidth) * lanes +
      io.sample.tokenLane.resize(cfg.factorIndexWidth))
      .resize(cfg.factorIndexWidth)
  val expectedTokenId = activeLoad.tokenId(io.sample.tokenLane)
  val expectedTokenActive = activeLoad.tokenMask(io.sample.tokenLane)
  val sampleOrderLegal = io.sample.element === expectedElement &&
    io.sample.tokenLane === expectedLane
  val sampleFrameLegal = Mux(expectedFirst,
    io.sample.first && !sampleHeadDuplicate,
    !io.sample.first && io.sample.head === activeHead)
  val sampleEndLegal = io.sample.last === expectedLast
  val sampleFactorLegal = io.sample.element === 0 ||
    !io.sample.tokenActive ||
    io.sample.dequantFactorQ16 === factors(sampleFactorIndex)
  val sampleLegal = sampleHeadInRange && sampleOrderLegal &&
    sampleFrameLegal && sampleEndLegal && sampleFactorLegal &&
    io.sample.tokenId.resize(model.sequenceWidth) === expectedTokenId &&
    io.sample.tokenActive === expectedTokenActive &&
    (!io.sample.tokenActive || io.sample.dequantFactorQ16.orR)

  val samplePhysicalBank = expectedElement(6 downto 2)
  val sampleSegmentIndex = samplePhysicalBank(4)
  val sampleLocalWord = samplePhysicalBank(3 downto 0)
  val sampleWordFeature = expectedElement(1 downto 0)
  val sampleAtSegmentEnd = expectedElement(5 downto 0).andR &&
    expectedLane === lanes - 1

  val bufferedMetadata = segmentBuffer.io.segment.payload.metadata
  val issueSegmentIndex = bufferedMetadata(
    segmentMetadataRowLow - 1 downto
      segmentMetadataSegmentLow).asUInt
  val issueLocalRow = bufferedMetadata(
    segmentMetadataSymbolMaskLow - 1 downto
      segmentMetadataRowLow).asUInt
  val issueSymbolMask = bufferedMetadata(
    segmentMetadataGenerationLow - 1 downto
      segmentMetadataSymbolMaskLow)
  val issueGeneration = bufferedMetadata(
    segmentMetadataWidth - 1 downto
      segmentMetadataGenerationLow).asUInt
  val issueSegmentFull = segmentBuffer.io.segment.valid

  val writeFaultNow = io.packedWriteFault.valid
  // A returned fault is authenticated against the retained transaction, not
  // against packedWrite.valid: fault priority deliberately suppresses valid
  // in the same cycle.  The authoritative semantic outstanding input also
  // covers the router's one-cycle post-fire fault-routing window.
  val writeFaultMatchesCurrent = issueSegmentFull &&
    io.packedWriteFault.region ===
      BitNetUnifiedResidentTensorRegion.QueryLmAlias &&
    io.packedWriteFault.localRow === issueLocalRow &&
    io.packedWriteFault.segmentIndex === issueSegmentIndex
  val writeFaultMatchesAudit = writeAuditActive &&
      io.packedWriteFault.region ===
        BitNetUnifiedResidentTensorRegion.QueryLmAlias &&
      io.packedWriteFault.localRow === retainedWriteLocalRow &&
      io.packedWriteFault.segmentIndex === retainedWriteSegment
  val writeFaultMatches = writeFaultMatchesCurrent ||
    writeFaultMatchesAudit
  val writeFaultMetadataMismatch = writeFaultNow && !writeFaultMatches
  io.packedWriteFault.ready := True

  val sampleFaultNow = loadActive &&
    io.sample.valid && !sampleLegal
  val revokeNow = quarantineReg || io.abort || io.retire.valid ||
    grantFaultNow ||
    writeFaultNow || sampleFaultNow ||
    segmentBuffer.io.protocolFaultSticky
  val atHeadEnd = expectedLast && sampleLegal

  // Query contributes exactly one byte per sample.  The slot-major physical
  // ABI places four adjacent elements inside one native 32-bit symbol.  For
  // prefill the destination symbol is the token lane.  Decode uses
  // `headSlot XOR tokenLane`: active lane zero lands in the head's one-hot
  // symbol, while the three inactive zero lanes still cover every private
  // segment-buffer byte exactly once.  The one-hot semantic symbol mask then
  // preserves the other three query heads already resident in the GQA row.
  val sampleByte = Mux(io.sample.tokenActive, io.sample.value.asBits,
    B(0, layout.packedWidth bits))
  val sampleWordData = Bits(segmentBufferCfg.bankWidth bits)
  for (byte <- 0 until segmentBufferCfg.bytesPerBank) {
    sampleWordData((byte + 1) * layout.packedWidth - 1 downto
      byte * layout.packedWidth) := sampleByte
  }
  val sampleHead = Mux(expectedFirst, io.sample.head, activeHead)
  val sampleHeadSlot = sampleHead(
    log2Up(model.gqaGroupSize) - 1 downto 0)
  val sampleDestinationSymbol = UInt(
    log2Up(layout.nativeSymbolCount) bits)
  sampleDestinationSymbol := expectedLane.resize(
    log2Up(layout.nativeSymbolCount))
  when(activeLoad.mode === BitNetMode.Decode) {
    sampleDestinationSymbol := sampleHeadSlot(0).asUInt ^ expectedLane.resize(
      log2Up(layout.nativeSymbolCount))
  }
  val prefillByteInBank =
    (sampleWordFeature.asBits ## expectedLane.asBits).asUInt
  val decodeByteInBank =
    (sampleDestinationSymbol.asBits ## sampleWordFeature.asBits).asUInt
  val sampleByteInBank = Mux(
    activeLoad.mode === BitNetMode.Decode,
    decodeByteInBank,
    prefillByteInBank)
  val sampleLocalRow = UInt(layout.localRowWidth bits)
  sampleLocalRow := sampleHead.resize(layout.localRowWidth)
  when(activeLoad.mode === BitNetMode.Decode) {
    sampleLocalRow := (sampleHead >> 1)
      .resize(layout.localRowWidth)
  }
  val sampleSymbolMask = Bits(layout.nativeSymbolCount bits)
  sampleSymbolMask := B(
    (BigInt(1) << layout.nativeSymbolCount) - 1,
    layout.nativeSymbolCount bits)
  when(activeLoad.mode === BitNetMode.Decode) {
    sampleSymbolMask := UIntToOh(sampleHeadSlot(0).asUInt,
      layout.nativeSymbolCount)
  }
  val sampleSegmentMetadata = Bits(segmentMetadataWidth bits)
  sampleSegmentMetadata := 0
  sampleSegmentMetadata(
    segmentMetadataRowLow - 1 downto segmentMetadataSegmentLow) :=
      sampleSegmentIndex.asBits
  sampleSegmentMetadata(
    segmentMetadataSymbolMaskLow - 1 downto segmentMetadataRowLow) :=
      sampleLocalRow.asBits
  sampleSegmentMetadata(
    segmentMetadataGenerationLow - 1 downto
      segmentMetadataSymbolMaskLow) := sampleSymbolMask
  sampleSegmentMetadata(
    segmentMetadataWidth - 1 downto segmentMetadataGenerationLow) :=
      leaseGeneration.asBits

  // Abort/fault may stop the scalar producer immediately.  Complete only the
  // already-partial private row with zero bytes so the BRAM ring can retire;
  // no padded segment is ever authenticated as resident Q data.
  val cleanupWriteActive = releasePending || quarantineReg || io.abort ||
    io.retire.valid || grantFaultNow || writeFaultNow || sampleFaultNow
  val paddingPartial = cleanupWriteActive && segmentBuffer.io.partial &&
    !segmentBuffer.io.protocolFaultSticky
  val normalSampleWrite = loadActive && io.sample.valid && !revokeNow
  segmentBuffer.io.wordWrite.valid := normalSampleWrite || paddingPartial
  segmentBuffer.io.wordWrite.payload.data := Mux(
    paddingPartial, B(0, segmentBufferCfg.bankWidth bits), sampleWordData)
  segmentBuffer.io.wordWrite.payload.byteMask :=
    UIntToOh(sampleByteInBank, segmentBufferCfg.bytesPerBank)
  segmentBuffer.io.wordWrite.payload.bankIndex := sampleLocalWord
  segmentBuffer.io.wordWrite.payload.slotAddress := segmentWriteSlot
  segmentBuffer.io.wordWrite.payload.commit := sampleAtSegmentEnd
  segmentBuffer.io.wordWrite.payload.metadata := sampleSegmentMetadata
  segmentBuffer.io.wordWrite.payload.fault := False
  val paddingWordFire = segmentBuffer.io.wordWrite.fire && paddingPartial
  when(segmentBuffer.io.wordWrite.fire && sampleAtSegmentEnd) {
    segmentWriteSlot := segmentWriteSlot + 1
  }

  // A pending write survives abort/quarantine and drains only through the real
  // semantic ready or an explicit fault.  On the ownership-probe decision
  // cycle, suppress valid before clearing pending: a Router/frontend whose
  // ready arrives on that exact edge therefore cannot capture a transaction
  // based on the old outstanding value.  Fault has priority over commit.
  val cancelUnownedWriteNow = abortWriteOwnershipProbe &&
    !io.semanticPackedWriteOutstanding && !writeOffered
  val nonAbortWriteRevoke = releasePending || quarantineReg || io.retire.valid ||
    grantFaultNow || sampleFaultNow || writeFaultNow ||
    segmentBuffer.io.protocolFaultSticky
  io.packedWrite.valid := issueSegmentFull && !writeAuditActive &&
    !writeFaultNow && !cancelUnownedWriteNow &&
    (writeOffered || !nonAbortWriteRevoke)
  io.packedWrite.region :=
    BitNetUnifiedResidentTensorRegion.QueryLmAlias
  io.packedWrite.localRow := issueLocalRow
  io.packedWrite.segmentIndex := issueSegmentIndex
  io.packedWrite.symbolMask := issueSymbolMask
  io.packedWrite.aliasOwner :=
    BitNetUnifiedResidentTensorAliasOwner.Query
  io.packedWrite.aliasGeneration := issueGeneration
  for (word <- 0 until packedSegmentWordCount) {
    io.packedWrite.segmentWords(word) :=
      segmentBuffer.io.segment.payload.data(
        (word + 1) * layout.wordWidth - 1 downto word * layout.wordWidth)
  }
  io.sample.ready := loadActive && !revokeNow && !paddingPartial &&
    segmentBuffer.io.wordWrite.ready

  // Never reclaim a row on the cycle abort first observes it.  The probe gives
  // a shared router one edge to expose ownership; a Stream already offered
  // while stalled remains stable even when a direct frontend ties ownership
  // low.  Other cleanup rows have never been offered and may be discarded.
  val cleanupBufferedWrites = releasePending || quarantineReg ||
    io.retire.valid || grantFaultNow || sampleFaultNow || writeFaultNow
  val currentWriteOwned = writeOffered ||
    (io.semanticPackedWriteOutstanding && !writeAuditActive)
  val discardBufferedSegment = issueSegmentFull && !currentWriteOwned &&
    ((cleanupBufferedWrites && !abortWriteOwnershipProbe) ||
      cancelUnownedWriteNow)
  segmentBuffer.io.segment.ready := io.packedWrite.fire ||
    (io.packedWriteFault.fire && writeOffered) || discardBufferedSegment

  when(io.packedWrite.valid && !io.packedWrite.ready) {
    writeOffered := True
  }
  when(io.packedWrite.fire) {
    writeOffered := False
    writeAuditActive := True
    retainedWriteLocalRow := io.packedWrite.localRow
    retainedWriteAliasGeneration := io.packedWrite.aliasGeneration
    retainedWriteSegment := io.packedWrite.segmentIndex
    retainedWriteSymbolMask := io.packedWrite.symbolMask
    abortWriteOwnershipProbe := False
  }
  when(io.packedWriteFault.fire) {
    writeOffered := False
    writeAuditActive := False
    abortWriteOwnershipProbe := False
  }

  when(writeAuditActive && !writeFaultNow) {
    writeAuditActive := False
    when(!releasePending && !quarantineReg && !io.abort &&
        retainedWriteSegment === packedSegmentCount - 1) {
      val retainedWriteHeadSlot = OHToUInt(retainedWriteSymbolMask)
      val authenticatedHeadIndex = UInt(residentCfg.qkv.headWidth bits)
      authenticatedHeadIndex := retainedWriteLocalRow.resize(
        residentCfg.qkv.headWidth)
      when(activeLoad.mode === BitNetMode.Decode) {
        authenticatedHeadIndex :=
          ((retainedWriteLocalRow.resize(residentCfg.qkv.headWidth) << 1) +
            retainedWriteHeadSlot.resize(residentCfg.qkv.headWidth))
            .resize(residentCfg.qkv.headWidth)
      }
      val authenticatedHead = UIntToOh(authenticatedHeadIndex, heads)
      val headsAfter = headValid | authenticatedHead
      headValid := headsAfter
      committedRowCount := committedRowCount + 1
      when(headsAfter.andR) {
        loadInFlight := False
        imageLoaded := True
        completionValid := True
        completionPayload.requestId := activeLoad.requestId
        completionPayload.layer := activeLoad.layer
        completionPayload.tokenBase := activeLoad.tokenBase
        completionPayload.tokenMask := activeLoad.tokenMask
        completionPayload.success := True
        completedLoadCount := completedLoadCount + 1
      }
    }
  }

  // Outstanding becomes authoritative one cycle after either the packed
  // router or the direct semantic frontend captures a valid.  Give that
  // boundary one complete cycle to claim an aborting write.  If nobody claims
  // it, cancellation is safe only if the Stream was never offered while
  // stalled; either ownership or `writeOffered` retains it to a real terminal.
  when(io.abort && io.packedWrite.valid &&
      !io.semanticPackedWriteOutstanding) {
    abortWriteOwnershipProbe := True
  }
  when(abortWriteOwnershipProbe) {
    when(io.semanticPackedWriteOutstanding || writeOffered) {
      abortWriteOwnershipProbe := False
    } otherwise {
      abortWriteOwnershipProbe := False
    }
  }
  // A real terminal always wins over a same-cycle abort probe.
  when(io.packedWrite.fire || io.packedWriteFault.fire) {
    abortWriteOwnershipProbe := False
  }

  when(io.sample.fire) {
    when(expectedFirst) {
      activeHead := io.sample.head
      frameActive := True
    }
    when(io.sample.element === 0) {
      factors(sampleFactorIndex) := Mux(io.sample.tokenActive,
        io.sample.dequantFactorQ16, U(0, 32 bits))
    }
    when(atHeadEnd) {
      val capturedAfter = headCaptured | sampleHeadOneHot
      headCaptured := capturedAfter
      frameActive := False
      when(capturedAfter.andR) {
        loadActive := False
      }
    }
  }

  // Normal input and private zero padding share the same strictly sequential
  // byte cursor.  This makes the final padding beat the buffer's genuine
  // coverage-completing commit rather than an out-of-band flush.
  when(io.sample.fire || paddingWordFire) {
    when(expectedLane === lanes - 1) {
      expectedLane := 0
      when(expectedElement === model.headSize - 1) {
        expectedElement := 0
      } otherwise {
        expectedElement := expectedElement + 1
      }
    } otherwise {
      expectedLane := expectedLane + 1
    }
  }

  // ------------------------------------------------------------------
  // Lookup authentication and packed-row reconstruction.
  // ------------------------------------------------------------------
  val activeLookup = Reg(BitNetAttentionPass(model)) init
    BitNetAttentionPass(model).getZero
  val readActive = RegInit(False)
  val adapterReadOutstanding = RegInit(False)
  val expectedReadRow = Reg(UInt(layout.localRowWidth bits)) init 0
  // QK observes the response without consuming it.  Its terminal block.fire
  // sets this compact boundary; the packed response drains on the following
  // slow cycle, avoiding a score.ready -> router/CDC ready path.
  val responseReleasePending = RegInit(False)
  val outputFactors = Vec(Reg(UInt(32 bits)) init 0, lanes)

  val lookupPrefill = io.lookup.mode === BitNetMode.Prefill &&
    io.lookup.laneRole === BitNetAttentionLaneRole.TokenParallel
  val lookupDecode = io.lookup.mode === BitNetMode.Decode &&
    io.lookup.laneRole === BitNetAttentionLaneRole.HeadParallel
  val prefillHeadInRange = io.lookup.passIndex < heads
  val decodeHeadInRange = io.lookup.kvHead < model.kvHeadCount
  val prefillMetadataMatches = io.lookup.laneMask === activeLoad.tokenMask &&
    io.lookup.tokenId.asBits === activeLoad.tokenId.asBits
  val decodeTokenMatches = io.lookup.laneMask.andR &&
    activeLoad.tokenMask === B(1, lanes bits) &&
    io.lookup.tokenId.map(_ === activeLoad.tokenId(0)).andR
  val lookupLegal = imageLoaded && leaseHeldReg &&
    leaseGeneration =/= 0 &&
    io.lookup.requestId === activeLoad.requestId &&
    io.lookup.layer === activeLoad.layer &&
    io.lookup.tokenBase === activeLoad.tokenBase &&
    io.lookup.mode === activeLoad.mode &&
    ((lookupPrefill && prefillHeadInRange && prefillMetadataMatches) ||
      (lookupDecode && decodeHeadInRange && decodeTokenMatches))

  io.lookup.ready := imageLoaded && !readActive &&
    !adapterReadOutstanding && !responseReleasePending &&
    !releasePending && !quarantineReg && !io.abort
  val badLookupNow = io.lookup.fire && !lookupLegal
  when(io.lookup.fire) {
    activeLookup := io.lookup.payload
    discardReadResponse := False
    for (lane <- 0 until lanes) {
      val prefillFactorIndex =
        (io.lookup.passIndex.resize(cfg.factorIndexWidth) * lanes + lane)
          .resize(cfg.factorIndexWidth)
      val decodeHead = io.lookup.queryHeadId(lane)
      val decodeFactorIndex =
        (decodeHead.resize(cfg.factorIndexWidth) * lanes)
          .resize(cfg.factorIndexWidth)
      outputFactors(lane) := Mux(!lookupLegal || !io.lookup.laneMask(lane),
        U(0, 32 bits),
        Mux(lookupPrefill, factors(prefillFactorIndex),
          factors(decodeFactorIndex)))
    }
    when(lookupLegal) {
      readActive := True
    }
  }

  val lookupIsPrefill =
    activeLookup.mode === BitNetMode.Prefill
  val requestedReadRow = UInt(layout.localRowWidth bits)
  requestedReadRow := activeLookup.passIndex.resize(layout.localRowWidth)
  when(!lookupIsPrefill) {
    requestedReadRow := activeLookup.passIndex.resize(layout.localRowWidth)
  }

  val readRevokeNow = quarantineReg || io.abort || io.retire.valid ||
    releasePending || io.packedWriteFault.valid
  val cancelUnownedReadNow = abortReadOwnershipProbe &&
    !io.semanticPackedReadOutstanding
  val preserveAbortingReadCommand = io.abort || abortReadOwnershipProbe ||
    abortReadDrain || io.semanticPackedReadOutstanding
  io.packedRead.valid := readActive && !adapterReadOutstanding &&
    !cancelUnownedReadNow && (!readRevokeNow || preserveAbortingReadCommand)
  io.packedRead.region :=
    BitNetUnifiedResidentTensorRegion.QueryLmAlias
  io.packedRead.localRow := requestedReadRow
  io.packedRead.bankMask := allBanks
  io.packedRead.aliasOwner :=
    BitNetUnifiedResidentTensorAliasOwner.Query
  io.packedRead.aliasGeneration := leaseGeneration

  when(io.packedRead.fire) {
    adapterReadOutstanding := True
    expectedReadRow := requestedReadRow
    issuedReadCount := issuedReadCount + 1
    when(io.abort || abortReadOwnershipProbe || abortReadDrain) {
      // The router accepted a command that abort had already revoked at the
      // lookup boundary.  It is still a real transaction and its response is
      // drained, but it must never recreate a query block.
      readActive := False
      abortReadOwnershipProbe := False
      abortReadDrain := False
      discardReadResponse := True
    }
  }

  val readResponseMetadataMatches = adapterReadOutstanding &&
    io.packedReadData.region ===
      BitNetUnifiedResidentTensorRegion.QueryLmAlias &&
    io.packedReadData.localRow === expectedReadRow &&
    io.packedReadData.bankMask === allBanks
  val readResponseFault = io.packedReadData.fault ||
    !readResponseMetadataMatches
  val readFaultNow = io.packedReadData.valid && readResponseFault
  // Every current-cycle revoke suppresses block.valid before it can create a
  // QK side effect.  Fault/abort cleanup drains independently of block.ready;
  // only a healthy response is held for the multi-cycle QK borrow.
  val readFailCloseNow = readRevokeNow || grantFaultNow || sampleFaultNow ||
    badLookupNow || segmentBuffer.io.protocolFaultSticky
  val responseBorrowable = adapterReadOutstanding &&
    !discardReadResponse && !responseReleasePending &&
    !readFailCloseNow && !readResponseFault
  io.packedReadData.ready := !adapterReadOutstanding ||
    discardReadResponse || responseReleasePending || readFailCloseNow ||
    readFaultNow

  when(io.packedReadData.fire) {
    when(adapterReadOutstanding) {
      adapterReadOutstanding := False
      readActive := False
    }
  }

  io.block.valid := io.packedReadData.valid && responseBorrowable
  io.block.requestId := activeLookup.requestId
  io.block.layer := activeLookup.layer
  io.block.laneRole := activeLookup.laneRole
  io.block.passIndex := activeLookup.passIndex
  io.block.kvHead := activeLookup.kvHead
  io.block.laneMask := activeLookup.laneMask
  io.block.tokenId := activeLookup.tokenId
  io.block.queryHeadId := activeLookup.queryHeadId
  io.block.queryPosition := activeLookup.queryPosition
  io.block.keyCount := activeLookup.keyCount
  // QueryLmAlias is intentionally slot-major.  The first packed-word index is
  // the Query symbol/head slot and the second is the adjacent element byte;
  // swapping these equal-sized dimensions would silently corrupt decode.
  for (bank <- 0 until layout.bankCount;
       wordFeature <- 0 until layout.packedFeaturesPerWord;
       lane <- 0 until lanes) {
    val element = bank * layout.packedFeaturesPerWord + wordFeature
    val prefillByte = wordFeature * lanes + lane
    val decodeByte = lane * layout.packedFeaturesPerWord + wordFeature
    val prefillValue = io.packedReadData.words(bank).asBits(
      (prefillByte + 1) * layout.packedWidth - 1 downto
        prefillByte * layout.packedWidth).asSInt
    val decodeValue = io.packedReadData.words(bank).asBits(
      (decodeByte + 1) * layout.packedWidth - 1 downto
        decodeByte * layout.packedWidth).asSInt
    io.block.q(lane)(element) := Mux(
      lookupIsPrefill, prefillValue, decodeValue)
  }
  io.block.qFactorQ16 := outputFactors
  when(io.block.fire) {
    responseReleasePending := True
    completedLookupCount := completedLookupCount + 1
  }

  // ------------------------------------------------------------------
  // Identity-checked normal retirement and unconditional abort cleanup.
  // ------------------------------------------------------------------
  val retireIdentityMatches = leaseHeldReg && imageLoaded &&
    io.retire.requestId === activeLoad.requestId &&
    io.retire.layer === activeLoad.layer &&
    io.retire.mode === activeLoad.mode &&
    io.retire.tokenBase === activeLoad.tokenBase &&
    io.retire.tokenId.asBits === activeLoad.tokenId.asBits &&
    io.retire.tokenMask === activeLoad.tokenMask &&
    io.retire.aliasGeneration === leaseGeneration &&
    io.retire.aliasGeneration =/= 0
  io.retire.ready := !releasePending && !waitingGrant &&
    !acquirePending && !io.abort
  val retireFaultNow = io.retire.fire && !retireIdentityMatches

  when(io.retire.fire) {
    imageLoaded := False
    loadActive := False
    readActive := False
    discardReadResponse := adapterReadOutstanding
    when(retireIdentityMatches) {
      releasePending := True
      releaseOwner := BitNetUnifiedResidentTensorAliasOwner.Query
      releaseGeneration := leaseGeneration
    } otherwise {
      stickyFault := True
      quarantineReg := True
      when(leaseHeldReg && leaseGeneration =/= 0) {
        releasePending := True
        releaseOwner := BitNetUnifiedResidentTensorAliasOwner.Query
        releaseGeneration := leaseGeneration
      }
    }
  }

  val abortHasWork = acquirePending || waitingGrant || leaseHeldReg ||
    loadInFlight || loadActive || imageLoaded || readActive ||
    adapterReadOutstanding || responseReleasePending || packedWritePending ||
    abortAcquireOwnershipProbe || abortReadOwnershipProbe || abortReadDrain
  when(io.abort && abortHasWork) {
    imageLoaded := False
    loadActive := False
    when(adapterReadOutstanding) {
      readActive := False
      discardReadResponse := True
    }
    // Keep a presented acquire/read for one owner-observation cycle.  This
    // covers the exact edge on which a router locks the source.  A genuinely
    // unowned command is then suppressed combinationally before cancellation;
    // an owned command remains stable until its real terminal.
    when(acquirePending) {
      cancelWaitingGrant := True
      when(!io.aliasAcquireOwned && !abortAcquireOwnershipProbe) {
        abortAcquireOwnershipProbe := True
      }
    }
    when(readActive && !adapterReadOutstanding) {
      when(io.semanticPackedReadOutstanding) {
        abortReadDrain := True
        abortReadOwnershipProbe := False
      } elsewhen(!abortReadOwnershipProbe) {
        abortReadOwnershipProbe := True
      }
    }
    when(loadInFlight && !completionValid) {
      loadInFlight := False
      completionValid := True
      completionPayload.requestId := activeLoad.requestId
      completionPayload.layer := activeLoad.layer
      completionPayload.tokenBase := activeLoad.tokenBase
      completionPayload.tokenMask := activeLoad.tokenMask
      completionPayload.success := False
      completedLoadCount := completedLoadCount + 1
    }
    when(waitingGrant) {
      cancelWaitingGrant := True
    } elsewhen (leaseHeldReg && leaseGeneration =/= 0) {
      releasePending := True
      releaseOwner := BitNetUnifiedResidentTensorAliasOwner.Query
      releaseGeneration := leaseGeneration
    }
  }

  // Ownership probes resolve one cycle after abort.  The cancel predicates
  // above already deassert valid in this decision cycle, so a router/direct
  // frontend cannot claim the transaction on the edge that removes it.
  when(abortAcquireOwnershipProbe) {
    when(io.aliasAcquireOwned) {
      abortAcquireOwnershipProbe := False
    } otherwise {
      acquirePending := False
      abortAcquireOwnershipProbe := False
      cancelWaitingGrant := False
    }
  }
  when(abortReadOwnershipProbe) {
    when(io.semanticPackedReadOutstanding) {
      abortReadOwnershipProbe := False
      abortReadDrain := True
    } otherwise {
      readActive := False
      abortReadOwnershipProbe := False
      abortReadDrain := False
      discardReadResponse := False
    }
  }
  // Real handshakes win over a probe that was observing the same boundary.
  when(io.aliasAcquire.fire) {
    abortAcquireOwnershipProbe := False
  }
  when(io.packedRead.fire) {
    abortReadOwnershipProbe := False
    abortReadDrain := False
  }

  // All protocol failures revoke data commands in their observation cycle.
  // The following registered cleanup starts only after the offending semantic
  // response/fault has been drained.
  val protocolFaultNow = grantFaultNow || sampleFaultNow || writeFaultNow ||
    badLookupNow || readFaultNow || retireFaultNow ||
    segmentBuffer.io.protocolFaultSticky
  when(protocolFaultNow) {
    stickyFault := True
    quarantineReg := True
    imageLoaded := False
    loadActive := False
    readActive := False
    discardReadResponse := adapterReadOutstanding
    acquirePending := False
    when(loadInFlight && !completionValid) {
      loadInFlight := False
      completionValid := True
      completionPayload.requestId := activeLoad.requestId
      completionPayload.layer := activeLoad.layer
      completionPayload.tokenBase := activeLoad.tokenBase
      completionPayload.tokenMask := activeLoad.tokenMask
      completionPayload.success := False
      completedLoadCount := completedLoadCount + 1
    }
    when(leaseHeldReg && leaseGeneration =/= 0) {
      releasePending := True
      releaseOwner := BitNetUnifiedResidentTensorAliasOwner.Query
      releaseGeneration := leaseGeneration
    }
  }
  when(writeFaultMetadataMismatch) {
    writeFaultMetadataFault := True
  }
  when(writeFaultNow && completionValid) {
    completionPayload.success := False
  }

  // A real response terminal wins over abort/retire/fault bookkeeping which
  // observed the old outstanding bit on the same edge.  In particular, do not
  // leave a stale discard flag that could poison a later authenticated lookup.
  when(io.packedReadData.fire) {
    discardReadResponse := False
    responseReleasePending := False
  }

  // packedWritePending covers committed and partial BRAM rows plus the
  // post-fire audit window.  Therefore packedWrite.valid implies pending, and
  // testing the current valid again is redundant.
  // Keeping release dependent only on registered ownership/pending state also
  // prevents an alias-grant fault from feeding combinationally back through
  // packedWrite.valid into the shared alias router's release monitor.
  io.aliasRelease.valid := releasePending &&
    releaseGeneration =/= 0 && noSemanticOutstanding &&
    !readActive && !adapterReadOutstanding && !responseReleasePending &&
    !io.packedRead.valid && !packedWritePending
  io.aliasRelease.owner := releaseOwner
  io.aliasRelease.generation := releaseGeneration
  when(io.aliasRelease.fire) {
    releasePending := False
    leaseHeldReg := False
    leaseGeneration := 0
    releaseGeneration := 0
    cancelWaitingGrant := False
    abortAcquireOwnershipProbe := False
  }

  io.loaded := imageLoaded
  io.leaseHeld := leaseHeldReg
  io.aliasGeneration := leaseGeneration
  io.quarantined := quarantineReg
  io.protocolFaultSticky := stickyFault ||
    segmentBuffer.io.protocolFaultSticky
  io.writeFaultMetadataSticky := writeFaultMetadataFault
  // A successfully loaded/leased image is quiescent, matching the legacy
  // QResidentBuffer contract: `loaded` denotes residency while `busy` denotes
  // an active protocol transaction.  This distinction lets attention start
  // without first retiring the lease it is about to consume.
  io.busy := loadInFlight || loadActive || acquirePending || waitingGrant ||
    releasePending || completionValid || readActive ||
    adapterReadOutstanding || responseReleasePending || packedWritePending ||
      abortWriteOwnershipProbe || abortAcquireOwnershipProbe ||
      abortReadOwnershipProbe || abortReadDrain
}

object GenerateBitNetUnifiedResidentQueryAdapter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-query-adapter")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedResidentQueryAdapter())
}
