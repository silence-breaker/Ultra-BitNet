package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetUnifiedResidentTensorSemanticFrontendConfig(
    backend: BitNetUnifiedResidentTensorFastBackendConfig =
      BitNetUnifiedResidentTensorFastBackendConfig()) {
  require(backend.tagWidth > 0)
  require(backend.scratchpad.layout.bankCount == 32)
  require(backend.scratchpad.layout.wordWidth ==
    backend.scratchpad.layout.tokenLanes *
      backend.scratchpad.layout.activationWidth)
  require(backend.scratchpad.layout.rowCount == 512)
}

/** A semantic write has no completion channel in the reference scratchpad
  * interface.  This retained sideband reports only locally rejected writes
  * and authenticated-commit failures, without changing that command bundle.
  */
case class BitNetUnifiedResidentTensorSemanticWriteFault(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankMask = Bits(cfg.layout.bankCount bits)
  val faultCode = Bits(4 bits)
}

/** Retained fault identity for the fixed 4 x 1024-bit packed-write ABI.
  * The bank mask is deliberately not transported: it is a pure function of
  * segmentIndex, so carrying and comparing another 32 global bits would only
  * recreate the metadata fanout this interface is intended to remove.
  */
case class BitNetUnifiedResidentPackedI8WriteFault(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val segmentIndex =
    UInt(log2Up(cfg.layout.packedTransportSegmentCount) bits)
  val faultCode = Bits(4 bits)
}

object BitNetUnifiedResidentTensorSemanticFrontendFault {
  /** The legacy semantic response has only four fault bits.  Bit three is
    * also the conservative fail-stop indication for an unauthenticated
    * backend response; protocolFaultSticky distinguishes it from an ordinary
    * read/write collision.
    */
  val BackendProtocolMask: Int =
    1 << BitNetUnifiedResidentTensorFault.ReadWriteCollision
}

/**
  * Memory-free semantic guard in front of the related-clock tensor backend.
  *
  * The north side deliberately reuses the typed region/local-row bundles of
  * [[BitNetUnifiedResidentTensorScratchpad]].  The south side is the four
  * slow-domain command/response Stream pairs exposed by
  * [[BitNetUnifiedResidentTensorScratchpadRelatedClockBridge]].  This block
  * owns no Mem and instantiates neither the bridge nor the physical backend.
  * Wide write payloads are forwarded directly while the producer holds its
  * Stream stable; wide read payloads are likewise returned directly under
  * downstream backpressure.
  *
  * There is at most one accepted command of each of the four types.  Every
  * downstream read response is authenticated by type, private tag, absolute
  * row and bank mask.  Every write commit additionally has to match the full
  * requested, committed and collision masks.  The F0 pair and F1 pair are
  * serialized before the bridge, so a legitimate completion has a zero
  * collision mask and a completely predictable committed mask.  Any
  * unsolicited or mismatched response permanently quarantines the frontend;
  * only reset can make it issue another physical command.
  */
class BitNetUnifiedResidentTensorSemanticFrontend(
    cfg: BitNetUnifiedResidentTensorSemanticFrontendConfig =
      BitNetUnifiedResidentTensorSemanticFrontendConfig(),
    productionPackedFullRowOnly: Boolean = false,
    productionActivationGroupTransport: Boolean = false)
    extends Component {
  private val backendCfg = cfg.backend
  private val scratchpadCfg = backendCfg.scratchpad
  private val layout = scratchpadCfg.layout
  private val regionCount =
    BitNetUnifiedResidentTensorRegion.elements.size
  private val packedReadLaneCount = layout.packedTransportSegmentCount
  private val packedReadFullLaneMask =
    (BigInt(1) << packedReadLaneCount) - 1

  val io = new Bundle {
    // Reference semantic interface.
    val activationRead = slave(Stream(
      BitNetUnifiedResidentActivationRead(scratchpadCfg)))
    val activationReadData = master(Stream(
      BitNetUnifiedResidentActivationReadData(scratchpadCfg)))
    val activationWrite = slave(Stream(
      BitNetUnifiedResidentActivationWrite(scratchpadCfg)))
    val activationWriteFault = master(Stream(
      BitNetUnifiedResidentTensorSemanticWriteFault(scratchpadCfg)))

    val packedRead = slave(Stream(
      BitNetUnifiedResidentPackedI8Read(scratchpadCfg)))
    /** Generic/reference return, pruned from production elaborations. */
    val packedReadData = if (!productionPackedFullRowOnly)
      master(Stream(
        BitNetUnifiedResidentPackedI8ReadData(scratchpadCfg))) else null
    /** Production return: four independently acknowledged 1,024-bit lanes. */
    val packedReadParallelData = if (productionPackedFullRowOnly)
      master(BitNetUnifiedResidentPackedI8ReadParallelChannel(
        scratchpadCfg)) else null
    val packedWrite = slave(Stream(
      BitNetUnifiedResidentPackedI8Write(scratchpadCfg)))
    val packedWriteFault = master(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(scratchpadCfg)))

    val aliasAcquire = slave(Stream(
      BitNetUnifiedResidentAliasAcquire(scratchpadCfg)))
    val aliasGrant = master(Stream(
      BitNetUnifiedResidentAliasGrant(scratchpadCfg)))
    val aliasRelease = slave(Stream(
      BitNetUnifiedResidentAliasRelease(scratchpadCfg)))

    // Slow-domain side of the related-clock bridge.  Naming these ports
    // `fast*` denotes their physical command format, not their clock domain.
    val fastPackedRead = master(Stream(
      BitNetUnifiedResidentTensorFastReadCommand(backendCfg)))
    /** Generic/reference physical return, absent from production RTL. */
    val fastPackedReadData = if (!productionPackedFullRowOnly)
      slave(Stream(
        BitNetUnifiedResidentTensorFastPackedReadResponse(
          backendCfg))) else null
    /** Native production physical return from the related-clock bridge. */
    val fastPackedReadParallelData = if (productionPackedFullRowOnly)
      slave(BitNetUnifiedResidentTensorFastPackedReadParallelChannel(
        backendCfg)) else null
    val fastActivationWrite = master(Stream(
      BitNetUnifiedResidentTensorFastActivationWriteCommand(backendCfg)))
    val fastActivationWriteCommit = slave(Stream(
      BitNetUnifiedResidentTensorFastWriteCommit(backendCfg)))

    val fastActivationRead = master(Stream(
      BitNetUnifiedResidentTensorFastReadCommand(backendCfg)))
    val fastActivationReadData = slave(Stream(
      BitNetUnifiedResidentTensorFastActivationReadResponse(backendCfg)))
    val fastPackedWrite = master(Stream(
      BitNetUnifiedResidentTensorFastPackedWriteCommand(backendCfg)))
    val fastPackedWriteCommit = slave(Stream(
      BitNetUnifiedResidentTensorFastPackedWriteCommit(backendCfg)))
    val downstreamProtocolFaultSticky = in Bool()

    val aliasOwnerValid = out Bool()
    val aliasOwner = out(BitNetUnifiedResidentTensorAliasOwner())
    val aliasGeneration = out UInt(layout.aliasGenerationWidth bits)
    val activationReadOutstanding = out Bool()
    val packedReadOutstanding = out Bool()
    val activationWriteOutstanding = out Bool()
    val packedWriteOutstanding = out Bool()
    val readResponsePending = out Bool()
    val writeFaultPending = out Bool()
    val formatFaultSticky = out Bool()
    val rangeFaultSticky = out Bool()
    val aliasFaultSticky = out Bool()
    val collisionFaultSticky = out Bool()
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
    val acceptedReads = out UInt(32 bits)
    val acceptedWrites = out UInt(32 bits)
    val forwardedReads = out UInt(32 bits)
    val forwardedWrites = out UInt(32 bits)
    val authenticatedReadResponses = out UInt(32 bits)
    val authenticatedWriteCommits = out UInt(32 bits)
  }

  // ---------------------------------------------------------------------
  // Compile-time layout materialized as small legality/address muxes.
  // ---------------------------------------------------------------------
  val regionBase = Vec(UInt(layout.rowAddressWidth bits), regionCount)
  val regionRows = Vec(UInt((layout.localRowWidth + 1) bits), regionCount)
  val regionIsActivation = Vec(Bool(), regionCount)
  val regionIsPacked = Vec(Bool(), regionCount)
  BitNetUnifiedResidentTensorRegion.elements.zipWithIndex.foreach {
    case (region, encoding) =>
      val descriptor = layout.descriptor(region)
      regionBase(encoding) := U(
        descriptor.baseRow, layout.rowAddressWidth bits)
      regionRows(encoding) := U(
        descriptor.rowCount, (layout.localRowWidth + 1) bits)
      regionIsActivation(encoding) :=
        (if (descriptor.format ==
          BitNetUnifiedResidentTensorWordFormat.ActivationS32x4) True
        else False)
      regionIsPacked(encoding) :=
        (if (descriptor.format ==
          BitNetUnifiedResidentTensorWordFormat.PackedI8x16) True
        else False)
  }

  private def encodingLegal(
      region: BitNetUnifiedResidentTensorRegion.C): Bool =
    region.asBits.asUInt < U(regionCount, region.asBits.getWidth bits)

  private def safeRegionIndex(
      region: BitNetUnifiedResidentTensorRegion.C): UInt = {
    val raw = region.asBits.asUInt
    Mux(encodingLegal(region), raw, U(0, raw.getWidth bits))
  }

  private def activationFormatLegal(
      region: BitNetUnifiedResidentTensorRegion.C): Bool = {
    val index = safeRegionIndex(region)
    encodingLegal(region) && regionIsActivation(index)
  }

  private def packedFormatLegal(
      region: BitNetUnifiedResidentTensorRegion.C): Bool = {
    val index = safeRegionIndex(region)
    encodingLegal(region) && regionIsPacked(index)
  }

  private def rowLegal(
      region: BitNetUnifiedResidentTensorRegion.C,
      localRow: UInt): Bool = {
    val index = safeRegionIndex(region)
    encodingLegal(region) &&
      localRow.resize(layout.localRowWidth + 1) < regionRows(index)
  }

  private def absoluteRow(
      region: BitNetUnifiedResidentTensorRegion.C,
      localRow: UInt): UInt = {
    val index = safeRegionIndex(region)
    val translated = (regionBase(index).resize(layout.rowAddressWidth + 1) +
      localRow.resize(layout.rowAddressWidth + 1))
      .resize(layout.rowAddressWidth)
    Mux(encodingLegal(region), translated,
      U(0, layout.rowAddressWidth bits))
  }

  private def semanticFaultCode(
      formatLegal: Bool,
      rangeLegal: Bool,
      aliasLegal: Bool): Bits = {
    val result = Bits(4 bits)
    result := 0
    result(BitNetUnifiedResidentTensorFault.WrongFormat) := !formatLegal
    result(BitNetUnifiedResidentTensorFault.RowOutOfRange) := !rangeLegal
    result(BitNetUnifiedResidentTensorFault.AliasOwnership) := !aliasLegal
    result(BitNetUnifiedResidentTensorFault.ReadWriteCollision) := False
    result
  }

  // ---------------------------------------------------------------------
  // Private, non-zero Q/LM alias lease.
  // ---------------------------------------------------------------------
  val aliasState = Reg(BitNetUnifiedResidentAliasLeaseState()) init
    BitNetUnifiedResidentAliasLeaseState.Idle
  val aliasOwnerReg = Reg(BitNetUnifiedResidentTensorAliasOwner()) init
    BitNetUnifiedResidentTensorAliasOwner.Query
  val aliasGenerationReg =
    Reg(UInt(layout.aliasGenerationWidth bits)) init 0
  val nextAliasGeneration =
    Reg(UInt(layout.aliasGenerationWidth bits)) init 1
  val aliasOwnerValidReg = RegInit(False)

  val protocolFault = RegInit(False)

  def aliasAccessLegal(
      owner: BitNetUnifiedResidentTensorAliasOwner.C,
      generation: UInt): Bool =
    aliasState === BitNetUnifiedResidentAliasLeaseState.Owned &&
      aliasOwnerValidReg && generation =/= 0 &&
      owner === aliasOwnerReg && generation === aliasGenerationReg

  // ---------------------------------------------------------------------
  // Four semantic legality checks and absolute-row translations.
  // ---------------------------------------------------------------------
  val activationReadFormatLegal =
    activationFormatLegal(io.activationRead.region)
  val activationReadRowLegal =
    rowLegal(io.activationRead.region, io.activationRead.localRow)
  val activationReadLegal =
    activationReadFormatLegal && activationReadRowLegal
  val activationReadAbsoluteRow =
    absoluteRow(io.activationRead.region, io.activationRead.localRow)
  val activationReadFaultCode = semanticFaultCode(
    activationReadFormatLegal, activationReadRowLegal, True)

  val activationWriteFormatLegal =
    activationFormatLegal(io.activationWrite.region)
  val activationWriteRowLegal =
    rowLegal(io.activationWrite.region, io.activationWrite.localRow)
  val activationWriteLegal =
    activationWriteFormatLegal && activationWriteRowLegal
  val activationWriteAbsoluteRow =
    absoluteRow(io.activationWrite.region, io.activationWrite.localRow)
  val activationWriteFaultCode = semanticFaultCode(
    activationWriteFormatLegal, activationWriteRowLegal, True)

  // Projection, Query and LM are full-row clients in the production shell.
  // Keeping the generic masked-read behavior as the default preserves the
  // reference frontend, while the production specialization removes the
  // 4,096-bit per-bank return-zeroing mux from the synthesized datapath.
  val packedReadMaskLegal =
    if (productionPackedFullRowOnly) io.packedRead.bankMask.andR else True
  val packedReadFormatLegal =
    packedFormatLegal(io.packedRead.region) && packedReadMaskLegal
  val packedReadRowLegal =
    rowLegal(io.packedRead.region, io.packedRead.localRow)
  val packedReadTargetsAlias =
    io.packedRead.region ===
      BitNetUnifiedResidentTensorRegion.QueryLmAlias
  val packedReadAliasLegal = !packedReadTargetsAlias ||
    aliasAccessLegal(
      io.packedRead.aliasOwner, io.packedRead.aliasGeneration)
  val packedReadLegal =
    packedReadFormatLegal && packedReadRowLegal && packedReadAliasLegal
  val packedReadAbsoluteRow =
    absoluteRow(io.packedRead.region, io.packedRead.localRow)
  val packedReadFaultCode = semanticFaultCode(
    packedReadFormatLegal, packedReadRowLegal, packedReadAliasLegal)

  // symbolMask is width-bounded by the typed command; only the empty subset is
  // illegal.  Treat it as a format fault so it never reaches physical RAM.
  val packedWriteSymbolMaskLegal = io.packedWrite.symbolMask.orR
  val packedWriteFormatLegal = packedFormatLegal(io.packedWrite.region) &&
    packedWriteSymbolMaskLegal
  val packedWriteRowLegal =
    rowLegal(io.packedWrite.region, io.packedWrite.localRow)
  val packedWriteTargetsAlias =
    io.packedWrite.region ===
      BitNetUnifiedResidentTensorRegion.QueryLmAlias
  val packedWriteAliasLegal = !packedWriteTargetsAlias ||
    aliasAccessLegal(
      io.packedWrite.aliasOwner, io.packedWrite.aliasGeneration)
  val packedWriteLegal =
    packedWriteFormatLegal && packedWriteRowLegal && packedWriteAliasLegal
  val packedWriteAbsoluteRow =
    absoluteRow(io.packedWrite.region, io.packedWrite.localRow)
  val packedWriteFaultCode = semanticFaultCode(
    packedWriteFormatLegal, packedWriteRowLegal, packedWriteAliasLegal)

  // ---------------------------------------------------------------------
  // One retained transaction context for each physical command type.
  // ---------------------------------------------------------------------
  val activationReadBusy = RegInit(False)
  val activationReadAwaiting = RegInit(False)
  val activationReadLocalFault = RegInit(False)
  val activationReadExpectedRegion =
    Reg(BitNetUnifiedResidentTensorRegion()) init
      BitNetUnifiedResidentTensorRegion.Hidden
  val activationReadExpectedLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val activationReadExpectedAbsoluteRow =
    Reg(UInt(layout.rowAddressWidth bits)) init 0
  val activationReadExpectedMask =
    Reg(Bits(layout.bankCount bits)) init 0
  val activationReadExpectedTag =
    Reg(UInt(backendCfg.tagWidth bits)) init 0
  val activationReadLocalFaultCode = Reg(Bits(4 bits)) init 0
  val nextActivationReadTag =
    Reg(UInt(backendCfg.tagWidth bits)) init 1

  val packedReadBusy = RegInit(False)
  val packedReadAwaiting = RegInit(False)
  val packedReadLocalFault = RegInit(False)
  val packedReadExpectedRegion =
    Reg(BitNetUnifiedResidentTensorRegion()) init
      BitNetUnifiedResidentTensorRegion.ProjectionSlot0
  val packedReadExpectedLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val packedReadExpectedAbsoluteRow =
    Reg(UInt(layout.rowAddressWidth bits)) init 0
  val packedReadExpectedMask =
    Reg(Bits(layout.bankCount bits)) init 0
  val packedReadExpectedTag =
    Reg(UInt(backendCfg.tagWidth bits)) init 0
  val packedReadLocalFaultCode = Reg(Bits(4 bits)) init 0
  val nextPackedReadTag = Reg(UInt(backendCfg.tagWidth bits)) init 1
  // Production retains only lane ownership.  The 4096-bit response payload
  // remains owned by the physical bridge and crosses this frontend as wiring.
  val packedReadResponseSegmentPending =
    if (productionPackedFullRowOnly)
      Reg(Bits(packedReadLaneCount bits)) init 0 else null

  val activationWriteBusy = RegInit(False)
  val activationWriteAwaiting = RegInit(False)
  val activationWriteCommittedAck = RegInit(False)
  val activationWriteLocalFault = RegInit(False)
  val activationWriteExpectedRegion =
    Reg(BitNetUnifiedResidentTensorRegion()) init
      BitNetUnifiedResidentTensorRegion.Hidden
  val activationWriteExpectedLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val activationWriteExpectedAbsoluteRow =
    Reg(UInt(layout.rowAddressWidth bits)) init 0
  val activationWriteExpectedRequestedMask =
    Reg(Bits(layout.bankCount bits)) init 0
  val activationWriteExpectedCommittedMask =
    Reg(Bits(layout.bankCount bits)) init 0
  val activationWriteExpectedTag =
    Reg(UInt(backendCfg.tagWidth bits)) init 0
  val activationWriteLocalFaultCode = Reg(Bits(4 bits)) init 0
  val nextActivationWriteTag =
    Reg(UInt(backendCfg.tagWidth bits)) init 1

  val packedWriteBusy = RegInit(False)
  val packedWriteAwaiting = RegInit(False)
  val packedWriteCommittedAck = RegInit(False)
  val packedWriteLocalFault = RegInit(False)
  val packedWriteExpectedRegion =
    Reg(BitNetUnifiedResidentTensorRegion()) init
      BitNetUnifiedResidentTensorRegion.ProjectionSlot0
  val packedWriteExpectedLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val packedWriteExpectedAbsoluteRow =
    Reg(UInt(layout.rowAddressWidth bits)) init 0
  val packedWriteExpectedSegmentIndex = Reg(
    UInt(log2Up(layout.packedTransportSegmentCount) bits)) init 0
  val packedWriteExpectedTag =
    Reg(UInt(backendCfg.tagWidth bits)) init 0
  val packedWriteLocalFaultCode = Reg(Bits(4 bits)) init 0
  val nextPackedWriteTag = Reg(UInt(backendCfg.tagWidth bits)) init 1

  private def incrementNonZero(tag: UInt): Unit = {
    when(tag.andR) {
      tag := 1
    } otherwise {
      tag := tag + 1
    }
  }

  val activationWriteEffectiveMask = Bits(layout.bankCount bits)
  for (bank <- 0 until layout.bankCount) {
    activationWriteEffectiveMask(bank) :=
      io.activationWrite.bankMask(bank) &&
        io.activationWrite.elementMask(bank).orR
  }

  // ---------------------------------------------------------------------
  // Authenticate every return before using it to derive same-cycle health.
  // `healthyNow` closes all north/south command and lease interfaces in the
  // very cycle in which any channel violates its retained transaction.
  // ---------------------------------------------------------------------
  val activationReadResponseExpectedRaw =
    activationReadBusy && activationReadAwaiting
  val activationReadResponseMatchesRaw =
    activationReadResponseExpectedRaw &&
      io.fastActivationReadData.tag === activationReadExpectedTag &&
      io.fastActivationReadData.absoluteRow ===
        activationReadExpectedAbsoluteRow &&
      io.fastActivationReadData.bankMask === activationReadExpectedMask
  val activationReadResponseMismatchNow =
    io.fastActivationReadData.valid && !activationReadResponseMatchesRaw

  val packedReadResponseExpectedRaw = packedReadBusy && packedReadAwaiting
  val packedReadParallelHeaderMatchesRaw =
    if (productionPackedFullRowOnly)
      packedReadResponseExpectedRaw &&
        io.fastPackedReadParallelData.tag === packedReadExpectedTag &&
        io.fastPackedReadParallelData.absoluteRow ===
          packedReadExpectedAbsoluteRow &&
        io.fastPackedReadParallelData.bankMask === packedReadExpectedMask &&
        io.fastPackedReadParallelData.laneMask === B(
          packedReadFullLaneMask, packedReadLaneCount bits)
    else null
  val packedReadParallelExpectedValidSet =
    if (productionPackedFullRowOnly)
      Mux(
        packedReadResponseSegmentPending.orR,
        packedReadResponseSegmentPending,
        B(packedReadFullLaneMask, packedReadLaneCount bits))
    else null
  val packedReadResponseMatchesRaw = if (productionPackedFullRowOnly) {
    packedReadParallelHeaderMatchesRaw &&
      io.fastPackedReadParallelData.segmentValid.orR &&
      io.fastPackedReadParallelData.segmentValid ===
        packedReadParallelExpectedValidSet
  } else {
    packedReadResponseExpectedRaw &&
      io.fastPackedReadData.tag === packedReadExpectedTag &&
      io.fastPackedReadData.absoluteRow === packedReadExpectedAbsoluteRow &&
      io.fastPackedReadData.bankMask === packedReadExpectedMask
  }
  val packedReadResponseMismatchNow = if (productionPackedFullRowOnly) {
    val fastResponsePresent =
      io.fastPackedReadParallelData.segmentValid.orR
    val initialMismatch = !packedReadResponseSegmentPending.orR &&
      fastResponsePresent &&
      (!packedReadParallelHeaderMatchesRaw ||
        io.fastPackedReadParallelData.segmentValid =/=
          B(packedReadFullLaneMask, packedReadLaneCount bits))
    val retainedMismatch = packedReadResponseExpectedRaw &&
      packedReadResponseSegmentPending.orR &&
      (!packedReadParallelHeaderMatchesRaw ||
        io.fastPackedReadParallelData.segmentValid =/=
          packedReadResponseSegmentPending)
    initialMismatch || retainedMismatch
  } else {
    io.fastPackedReadData.valid && !packedReadResponseMatchesRaw
  }

  val activationWriteCommitExpectedRaw =
    activationWriteBusy && activationWriteAwaiting
  val activationWriteCommitMatchesRaw =
    activationWriteCommitExpectedRaw &&
      io.fastActivationWriteCommit.tag === activationWriteExpectedTag &&
      io.fastActivationWriteCommit.absoluteRow ===
        activationWriteExpectedAbsoluteRow &&
      io.fastActivationWriteCommit.requestedBankMask ===
        activationWriteExpectedRequestedMask &&
      io.fastActivationWriteCommit.committedBankMask ===
        activationWriteExpectedCommittedMask &&
      io.fastActivationWriteCommit.collisionMask === 0
  val activationWriteCommitMismatchNow =
    io.fastActivationWriteCommit.valid && !activationWriteCommitMatchesRaw

  val packedWriteCommitExpectedRaw = packedWriteBusy && packedWriteAwaiting
  // The compact backend commit authenticates this already-accepted command by
  // tag plus physical address.  symbolMask is checked before command issue and
  // deliberately is not echoed on the compact return path.
  val packedWriteCommitMatchesRaw = packedWriteCommitExpectedRaw &&
    io.fastPackedWriteCommit.tag === packedWriteExpectedTag &&
    io.fastPackedWriteCommit.absoluteRow ===
      packedWriteExpectedAbsoluteRow &&
    io.fastPackedWriteCommit.segmentIndex ===
      packedWriteExpectedSegmentIndex &&
    io.fastPackedWriteCommit.committedWordMask.andR &&
    io.fastPackedWriteCommit.collisionWordMask === 0
  val packedWriteCommitMismatchNow =
    io.fastPackedWriteCommit.valid && !packedWriteCommitMatchesRaw

  val protocolFaultNow = io.downstreamProtocolFaultSticky ||
    activationReadResponseMismatchNow || packedReadResponseMismatchNow ||
    activationWriteCommitMismatchNow || packedWriteCommitMismatchNow
  val healthyNow = !protocolFault && !protocolFaultNow
  io.protocolFaultSticky := protocolFault || protocolFaultNow
  io.quarantined := protocolFault || protocolFaultNow

  // A detector still closes every new north/south command in its own cycle
  // through healthyNow.  Do not, however, let a fault on one retained return
  // combinationally invalidate an unrelated return channel: that previously
  // made an activation-write commit comparison feed through the packed-read
  // router, LM codec, matrix feeder and TP2 phase control in one slow cycle.
  // The offending return is rejected locally below; protocolFault publishes
  // the cross-channel quarantine on the following edge.  Every accepted
  // transaction remains retained and is converted to the existing local
  // fault/drain path by the protocolFaultNow state update.
  val registeredHealthy = !protocolFault
  val activationReadReturnHealthy =
    registeredHealthy && !activationReadResponseMismatchNow
  val packedReadReturnHealthy =
    registeredHealthy && !packedReadResponseMismatchNow

  io.aliasAcquire.ready :=
    aliasState === BitNetUnifiedResidentAliasLeaseState.Idle &&
      healthyNow
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
    aliasState === BitNetUnifiedResidentAliasLeaseState.Grant &&
      healthyNow
  io.aliasGrant.owner := aliasOwnerReg
  io.aliasGrant.generation := aliasGenerationReg
  when(io.aliasGrant.fire) {
    aliasOwnerValidReg := True
    aliasState := BitNetUnifiedResidentAliasLeaseState.Owned
  }

  io.aliasOwnerValid := aliasOwnerValidReg && healthyNow
  io.aliasOwner := aliasOwnerReg
  io.aliasGeneration := Mux(
    healthyNow, aliasGenerationReg,
    U(0, layout.aliasGenerationWidth bits))

  // ---------------------------------------------------------------------
  // Same-format round-robin arbitration.  Activation rows and packed rows
  // occupy disjoint physical ranges, so one activation operation and one
  // packed operation may proceed together.  A read and write of the same
  // format are serialized until the earlier response/commit retires, which
  // removes adjacent-fast-phase RAW/WAR ambiguity.
  // ---------------------------------------------------------------------
  val preferActivationRead = RegInit(True)
  val activationReadCandidate = io.activationRead.valid &&
    activationReadLegal && !activationReadBusy && !activationWriteBusy &&
    healthyNow
  val activationWriteCandidate = io.activationWrite.valid &&
    activationWriteLegal && !activationWriteBusy && !activationReadBusy &&
    healthyNow
  val chooseActivationRead = activationReadCandidate &&
    (!activationWriteCandidate || preferActivationRead)
  val chooseActivationWrite =
    activationWriteCandidate && !chooseActivationRead

  val preferPackedRead = RegInit(True)
  val packedReadCandidate = io.packedRead.valid && packedReadLegal &&
    !packedReadBusy && !packedWriteBusy && healthyNow
  val packedWriteCandidate = io.packedWrite.valid && packedWriteLegal &&
    !packedWriteBusy && !packedReadBusy && healthyNow
  val choosePackedRead = packedReadCandidate &&
    (!packedWriteCandidate || preferPackedRead)
  val choosePackedWrite = packedWriteCandidate && !choosePackedRead

  io.fastPackedRead.valid := choosePackedRead
  io.fastPackedRead.absoluteRow := packedReadAbsoluteRow
  io.fastPackedRead.bankMask := io.packedRead.bankMask
  io.fastPackedRead.tag := nextPackedReadTag

  io.fastActivationWrite.valid := chooseActivationWrite
  io.fastActivationWrite.absoluteRow := activationWriteAbsoluteRow
  io.fastActivationWrite.bankMask := io.activationWrite.bankMask
  io.fastActivationWrite.tag := nextActivationWriteTag
  for (bank <- 0 until layout.bankCount) {
    io.fastActivationWrite.elementMask(bank) :=
      io.activationWrite.elementMask(bank)
    io.fastActivationWrite.words(bank) := io.activationWrite.words(bank)
  }

  io.fastActivationRead.valid := chooseActivationRead
  io.fastActivationRead.absoluteRow := activationReadAbsoluteRow
  io.fastActivationRead.bankMask := io.activationRead.bankMask
  io.fastActivationRead.tag := nextActivationReadTag

  io.fastPackedWrite.valid := choosePackedWrite
  io.fastPackedWrite.absoluteRow := packedWriteAbsoluteRow
  io.fastPackedWrite.segmentIndex := io.packedWrite.segmentIndex
  io.fastPackedWrite.symbolMask := io.packedWrite.symbolMask
  io.fastPackedWrite.tag := nextPackedWriteTag
  for (word <- 0 until layout.packedTransportWordsPerSegment) {
    io.fastPackedWrite.segmentWords(word) :=
      io.packedWrite.segmentWords(word)
  }

  // Invalid reads/writes retire locally.  Legal reads retire when their
  // physical command is accepted.  A legal write deliberately keeps north
  // ready low until a fully authenticated commit reaches CommittedAck.
  io.activationRead.ready := !activationReadBusy && healthyNow &&
    Mux(activationReadLegal,
      chooseActivationRead && io.fastActivationRead.ready, True)
  io.packedRead.ready := !packedReadBusy && healthyNow &&
    Mux(packedReadLegal,
      choosePackedRead && io.fastPackedRead.ready, True)
  io.activationWrite.ready := healthyNow &&
    ((!activationWriteBusy && !activationWriteLegal) ||
      (activationWriteBusy && activationWriteCommittedAck))
  io.packedWrite.ready := healthyNow &&
    ((!packedWriteBusy && !packedWriteLegal) ||
      (packedWriteBusy && packedWriteCommittedAck))

  when(io.activationRead.fire) {
    activationReadBusy := True
    activationReadExpectedRegion := io.activationRead.region
    activationReadExpectedLocalRow := io.activationRead.localRow
    activationReadExpectedAbsoluteRow := activationReadAbsoluteRow
    activationReadExpectedMask := io.activationRead.bankMask
    activationReadLocalFaultCode := activationReadFaultCode
    when(activationReadLegal) {
      activationReadExpectedTag := nextActivationReadTag
      activationReadAwaiting := True
      incrementNonZero(nextActivationReadTag)
      preferActivationRead := False
    } otherwise {
      activationReadLocalFault := True
    }
  }

  when(io.packedRead.fire) {
    packedReadBusy := True
    packedReadExpectedRegion := io.packedRead.region
    packedReadExpectedLocalRow := io.packedRead.localRow
    packedReadExpectedAbsoluteRow := packedReadAbsoluteRow
    packedReadExpectedMask := io.packedRead.bankMask
    packedReadLocalFaultCode := packedReadFaultCode
    when(packedReadLegal) {
      packedReadExpectedTag := nextPackedReadTag
      packedReadAwaiting := True
      incrementNonZero(nextPackedReadTag)
      preferPackedRead := False
    } otherwise {
      packedReadLocalFault := True
    }
  }

  when(io.activationWrite.fire) {
    when(activationWriteCommittedAck) {
      activationWriteBusy := False
      activationWriteCommittedAck := False
    } otherwise {
      // The only idle-state north fire is an illegal semantic write.
      activationWriteBusy := True
      activationWriteExpectedRegion := io.activationWrite.region
      activationWriteExpectedLocalRow := io.activationWrite.localRow
      activationWriteExpectedAbsoluteRow := activationWriteAbsoluteRow
      activationWriteExpectedRequestedMask := io.activationWrite.bankMask
      activationWriteExpectedCommittedMask := activationWriteEffectiveMask
      activationWriteLocalFaultCode := activationWriteFaultCode
      activationWriteLocalFault := True
    }
  }

  when(io.fastActivationWrite.fire) {
    activationWriteBusy := True
    activationWriteAwaiting := True
    activationWriteCommittedAck := False
    activationWriteLocalFault := False
    activationWriteExpectedRegion := io.activationWrite.region
    activationWriteExpectedLocalRow := io.activationWrite.localRow
    activationWriteExpectedAbsoluteRow := activationWriteAbsoluteRow
    activationWriteExpectedRequestedMask := io.activationWrite.bankMask
    activationWriteExpectedCommittedMask := activationWriteEffectiveMask
    activationWriteExpectedTag := nextActivationWriteTag
    activationWriteLocalFaultCode := activationWriteFaultCode
    incrementNonZero(nextActivationWriteTag)
    preferActivationRead := True
  }

  when(io.packedWrite.fire) {
    when(packedWriteCommittedAck) {
      packedWriteBusy := False
      packedWriteCommittedAck := False
    } otherwise {
      packedWriteBusy := True
      packedWriteExpectedRegion := io.packedWrite.region
      packedWriteExpectedLocalRow := io.packedWrite.localRow
      packedWriteExpectedAbsoluteRow := packedWriteAbsoluteRow
      packedWriteExpectedSegmentIndex := io.packedWrite.segmentIndex
      packedWriteLocalFaultCode := packedWriteFaultCode
      packedWriteLocalFault := True
    }
  }

  when(io.fastPackedWrite.fire) {
    packedWriteBusy := True
    packedWriteAwaiting := True
    packedWriteCommittedAck := False
    packedWriteLocalFault := False
    packedWriteExpectedRegion := io.packedWrite.region
    packedWriteExpectedLocalRow := io.packedWrite.localRow
    packedWriteExpectedAbsoluteRow := packedWriteAbsoluteRow
    packedWriteExpectedSegmentIndex := io.packedWrite.segmentIndex
    packedWriteExpectedTag := nextPackedWriteTag
    packedWriteLocalFaultCode := packedWriteFaultCode
    incrementNonZero(nextPackedWriteTag)
    preferPackedRead := True
  }

  // ---------------------------------------------------------------------
  // Full read-response authentication and zero-copy semantic return.
  // ---------------------------------------------------------------------
  val activationReadResponseMatches =
    activationReadResponseMatchesRaw && activationReadReturnHealthy

  io.activationReadData.valid := activationReadLocalFault ||
    (io.fastActivationReadData.valid && activationReadResponseMatches)
  io.activationReadData.region := activationReadExpectedRegion
  io.activationReadData.localRow := activationReadExpectedLocalRow
  io.activationReadData.bankMask := activationReadExpectedMask
  io.activationReadData.fault := activationReadLocalFault
  io.activationReadData.faultCode := activationReadLocalFaultCode
  for (bank <- 0 until layout.bankCount) {
    val returnedWord = Bits(layout.wordWidth bits)
    if (productionActivationGroupTransport) {
      // The production compact adapter exposes only the four requested words
      // and fault data is explicitly undefined.  Returning the authenticated
      // backend row directly removes 4096 bits of per-bank zero muxes from the
      // 266 MHz domain; the adapter performs four aligned 8:1 selections.
      returnedWord := io.fastActivationReadData.words(bank).asBits
    } else {
      returnedWord := 0
      when(!activationReadLocalFault &&
        activationReadExpectedMask(bank)) {
        returnedWord := io.fastActivationReadData.words(bank).asBits
      }
    }
    io.activationReadData.words(bank).assignFromBits(returnedWord)
  }
  io.fastActivationReadData.ready := !activationReadReturnHealthy ||
    (activationReadResponseMatchesRaw && io.activationReadData.ready)

  when(io.activationReadData.fire) {
    activationReadBusy := False
    activationReadAwaiting := False
    activationReadLocalFault := False
  }
  val packedReadResponseMatches =
    packedReadResponseMatchesRaw && packedReadReturnHealthy
  val packedReadAuthenticatedEvent = Bool()

  if (productionPackedFullRowOnly) {
    // The canonical semantic header is the retained request context.  The
    // physical header is used only for authentication and never reaches the
    // north side.  Fault data is explicitly undefined, which avoids a wide
    // zero/fault mux while preserving fail-stop identity.
    io.packedReadParallelData.region := packedReadExpectedRegion
    io.packedReadParallelData.localRow := packedReadExpectedLocalRow
    io.packedReadParallelData.bankMask := packedReadExpectedMask
    io.packedReadParallelData.laneMask := B(
      packedReadFullLaneMask, packedReadLaneCount bits)
    io.packedReadParallelData.fault := packedReadLocalFault
    io.packedReadParallelData.faultCode := packedReadLocalFaultCode
    for (lane <- 0 until packedReadLaneCount) {
      io.packedReadParallelData.segments(lane) :=
        io.fastPackedReadParallelData.segments(lane)
    }

    val packedReadOutputValidSet = Bits(packedReadLaneCount bits)
    packedReadOutputValidSet := 0
    when(packedReadLocalFault) {
      packedReadOutputValidSet := packedReadParallelExpectedValidSet
    } elsewhen(packedReadResponseMatches) {
      packedReadOutputValidSet :=
        io.fastPackedReadParallelData.segmentValid
    }
    io.packedReadParallelData.segmentValid := packedReadOutputValidSet

    // Corrupt/unsolicited physical responses are drained immediately so the
    // bridge can release ownership, while healthy lanes propagate independent
    // north-side ready bits without a payload snapshot.
    io.fastPackedReadParallelData.segmentReady := 0
    when(!packedReadReturnHealthy) {
      io.fastPackedReadParallelData.segmentReady :=
        io.fastPackedReadParallelData.segmentValid
    } elsewhen(packedReadResponseMatchesRaw && !packedReadLocalFault) {
      io.fastPackedReadParallelData.segmentReady :=
        io.packedReadParallelData.segmentReady & packedReadOutputValidSet
    }

    val packedReadOutputFire =
      packedReadOutputValidSet & io.packedReadParallelData.segmentReady
    val packedReadOutputPendingAfter =
      packedReadOutputValidSet & ~packedReadOutputFire
    val packedReadOutputCompletes = packedReadOutputValidSet.orR &&
      packedReadOutputFire.orR && !packedReadOutputPendingAfter.orR
    packedReadAuthenticatedEvent := packedReadOutputCompletes &&
      !packedReadLocalFault && packedReadResponseMatches
    when(packedReadOutputFire.orR) {
      packedReadResponseSegmentPending := packedReadOutputPendingAfter
      when(!packedReadOutputPendingAfter.orR) {
        packedReadBusy := False
        packedReadAwaiting := False
        packedReadLocalFault := False
      }
    }
  } else {
    io.packedReadData.valid := packedReadLocalFault ||
      (io.fastPackedReadData.valid && packedReadResponseMatches)
    io.packedReadData.region := packedReadExpectedRegion
    io.packedReadData.localRow := packedReadExpectedLocalRow
    io.packedReadData.bankMask := packedReadExpectedMask
    io.packedReadData.fault := packedReadLocalFault
    io.packedReadData.faultCode := packedReadLocalFaultCode
    for (bank <- 0 until layout.bankCount) {
      val returnedWord = Bits(layout.wordWidth bits)
      returnedWord := 0
      when(!packedReadLocalFault && packedReadExpectedMask(bank)) {
        returnedWord := io.fastPackedReadData.words(bank).asBits
      }
      io.packedReadData.words(bank).assignFromBits(returnedWord)
    }
    io.fastPackedReadData.ready := !packedReadReturnHealthy ||
      (packedReadResponseMatchesRaw && io.packedReadData.ready)
    packedReadAuthenticatedEvent :=
      io.fastPackedReadData.fire && packedReadResponseMatches

    when(io.packedReadData.fire) {
      packedReadBusy := False
      packedReadAwaiting := False
      packedReadLocalFault := False
    }
  }
  // ---------------------------------------------------------------------
  // Full write-commit authentication.  Correct commits retire internally;
  // rejected semantic writes and corrupt commits use retained fault Streams.
  // ---------------------------------------------------------------------
  io.activationWriteFault.valid := activationWriteLocalFault
  io.activationWriteFault.region := activationWriteExpectedRegion
  io.activationWriteFault.localRow := activationWriteExpectedLocalRow
  io.activationWriteFault.bankMask :=
    activationWriteExpectedRequestedMask
  io.activationWriteFault.faultCode := activationWriteLocalFaultCode
  when(io.activationWriteFault.fire) {
    activationWriteBusy := False
    activationWriteAwaiting := False
    activationWriteCommittedAck := False
    activationWriteLocalFault := False
  }

  val activationWriteCommitMatches =
    activationWriteCommitMatchesRaw && healthyNow
  io.fastActivationWriteCommit.ready := True
  when(io.fastActivationWriteCommit.fire && activationWriteCommitMatches) {
    activationWriteAwaiting := False
    activationWriteCommittedAck := True
  }

  io.packedWriteFault.valid := packedWriteLocalFault
  io.packedWriteFault.region := packedWriteExpectedRegion
  io.packedWriteFault.localRow := packedWriteExpectedLocalRow
  io.packedWriteFault.segmentIndex := packedWriteExpectedSegmentIndex
  io.packedWriteFault.faultCode := packedWriteLocalFaultCode
  when(io.packedWriteFault.fire) {
    packedWriteBusy := False
    packedWriteAwaiting := False
    packedWriteCommittedAck := False
    packedWriteLocalFault := False
  }

  val packedWriteCommitMatches = packedWriteCommitMatchesRaw && healthyNow
  io.fastPackedWriteCommit.ready := True
  when(io.fastPackedWriteCommit.fire && packedWriteCommitMatches) {
    packedWriteAwaiting := False
    packedWriteCommittedAck := True
  }

  // An alias lease cannot retire until every access bearing that lease has
  // either produced its authenticated response/commit or a local fault.
  val aliasAccessOutstanding =
    (packedReadBusy &&
      packedReadExpectedRegion ===
        BitNetUnifiedResidentTensorRegion.QueryLmAlias) ||
    (packedWriteBusy &&
      packedWriteExpectedRegion ===
        BitNetUnifiedResidentTensorRegion.QueryLmAlias)
  val aliasAccessPresented =
    (io.packedRead.valid &&
      io.packedRead.region ===
        BitNetUnifiedResidentTensorRegion.QueryLmAlias) ||
    (io.packedWrite.valid &&
      io.packedWrite.region ===
        BitNetUnifiedResidentTensorRegion.QueryLmAlias)
  io.aliasRelease.ready :=
    aliasState === BitNetUnifiedResidentAliasLeaseState.Owned &&
      !aliasAccessOutstanding && !aliasAccessPresented && healthyNow
  val aliasReleaseLegal = io.aliasRelease.owner === aliasOwnerReg &&
    io.aliasRelease.generation =/= 0 &&
    io.aliasRelease.generation === aliasGenerationReg
  when(io.aliasRelease.fire && aliasReleaseLegal) {
    aliasOwnerValidReg := False
    aliasGenerationReg := 0
    aliasState := BitNetUnifiedResidentAliasLeaseState.Idle
  }

  // ---------------------------------------------------------------------
  // Fail-closed diagnostics and accounting.
  // ---------------------------------------------------------------------
  val formatFault = RegInit(False)
  val rangeFault = RegInit(False)
  val aliasFault = RegInit(False)
  val collisionFault = RegInit(False)
  val acceptedReadCount = Reg(UInt(32 bits)) init 0
  val acceptedWriteCount = Reg(UInt(32 bits)) init 0
  val forwardedReadCount = Reg(UInt(32 bits)) init 0
  val forwardedWriteCount = Reg(UInt(32 bits)) init 0
  val authenticatedReadResponseCount = Reg(UInt(32 bits)) init 0
  val authenticatedWriteCommitCount = Reg(UInt(32 bits)) init 0

  val acceptedReadEvents = Cat(
    io.packedRead.fire, io.activationRead.fire)
  val acceptedWriteEvents = Cat(
    io.packedWrite.fire, io.activationWrite.fire)
  val forwardedReadEvents = Cat(
    io.fastPackedRead.fire, io.fastActivationRead.fire)
  val forwardedWriteEvents = Cat(
    io.fastPackedWrite.fire, io.fastActivationWrite.fire)
  val authenticatedReadEvents = Cat(
    packedReadAuthenticatedEvent,
    io.fastActivationReadData.fire && activationReadResponseMatches)
  val authenticatedWriteEvents = Cat(
    io.fastPackedWriteCommit.fire && packedWriteCommitMatches,
    io.fastActivationWriteCommit.fire && activationWriteCommitMatches)

  when(acceptedReadEvents.orR) {
    acceptedReadCount := acceptedReadCount +
      CountOne(acceptedReadEvents).resize(32)
  }
  when(acceptedWriteEvents.orR) {
    acceptedWriteCount := acceptedWriteCount +
      CountOne(acceptedWriteEvents).resize(32)
  }
  when(forwardedReadEvents.orR) {
    forwardedReadCount := forwardedReadCount +
      CountOne(forwardedReadEvents).resize(32)
  }
  when(forwardedWriteEvents.orR) {
    forwardedWriteCount := forwardedWriteCount +
      CountOne(forwardedWriteEvents).resize(32)
  }
  when(authenticatedReadEvents.orR) {
    authenticatedReadResponseCount := authenticatedReadResponseCount +
      CountOne(authenticatedReadEvents).resize(32)
  }
  when(authenticatedWriteEvents.orR) {
    authenticatedWriteCommitCount := authenticatedWriteCommitCount +
      CountOne(authenticatedWriteEvents).resize(32)
  }

  when(io.activationRead.fire) {
    when(!activationReadFormatLegal) { formatFault := True }
    when(!activationReadRowLegal) { rangeFault := True }
  }
  when(io.packedRead.fire) {
    when(!packedReadFormatLegal) { formatFault := True }
    when(!packedReadRowLegal) { rangeFault := True }
    when(!packedReadAliasLegal) { aliasFault := True }
  }
  when(io.activationWrite.fire) {
    when(!activationWriteFormatLegal) { formatFault := True }
    when(!activationWriteRowLegal) { rangeFault := True }
  }
  when(io.packedWrite.fire) {
    when(!packedWriteFormatLegal) { formatFault := True }
    when(!packedWriteRowLegal) { rangeFault := True }
    when(!packedWriteAliasLegal) { aliasFault := True }
  }
  when(io.aliasRelease.fire && !aliasReleaseLegal) {
    aliasFault := True
  }

  when(protocolFaultNow) {
    protocolFault := True
    if (productionPackedFullRowOnly) {
      // Any partially exposed good row is abandoned.  The canonical fault row
      // restarts at 1111 while the corrupt physical valid set is drained.
      when(!protocolFault && !packedReadLocalFault) {
        packedReadResponseSegmentPending := 0
      }
    }
    when(activationReadBusy && !activationReadLocalFault) {
      activationReadAwaiting := False
      activationReadLocalFault := True
      activationReadLocalFaultCode := B(
        BitNetUnifiedResidentTensorSemanticFrontendFault
          .BackendProtocolMask, 4 bits)
    }
    when(packedReadBusy && !packedReadLocalFault) {
      packedReadAwaiting := False
      packedReadLocalFault := True
      packedReadLocalFaultCode := B(
        BitNetUnifiedResidentTensorSemanticFrontendFault
          .BackendProtocolMask, 4 bits)
    }
    when(activationWriteBusy && !activationWriteLocalFault) {
      activationWriteAwaiting := False
      activationWriteCommittedAck := False
      activationWriteLocalFault := True
      activationWriteLocalFaultCode := B(
        BitNetUnifiedResidentTensorSemanticFrontendFault
          .BackendProtocolMask, 4 bits)
    }
    when(packedWriteBusy && !packedWriteLocalFault) {
      packedWriteAwaiting := False
      packedWriteCommittedAck := False
      packedWriteLocalFault := True
      packedWriteLocalFaultCode := B(
        BitNetUnifiedResidentTensorSemanticFrontendFault
          .BackendProtocolMask, 4 bits)
    }
  }
  when((io.fastActivationWriteCommit.valid &&
      io.fastActivationWriteCommit.collisionMask.orR) ||
    (io.fastPackedWriteCommit.valid &&
      io.fastPackedWriteCommit.collisionWordMask.orR)) {
    collisionFault := True
  }

  io.activationReadOutstanding := activationReadBusy
  io.packedReadOutstanding := packedReadBusy
  io.activationWriteOutstanding := activationWriteBusy
  io.packedWriteOutstanding := packedWriteBusy
  io.readResponsePending := io.activationReadData.valid ||
    (if (productionPackedFullRowOnly)
      io.packedReadParallelData.segmentValid.orR
    else io.packedReadData.valid)
  io.writeFaultPending :=
    io.activationWriteFault.valid || io.packedWriteFault.valid
  io.formatFaultSticky := formatFault
  io.rangeFaultSticky := rangeFault
  io.aliasFaultSticky := aliasFault
  io.collisionFaultSticky := collisionFault
  io.acceptedReads := acceptedReadCount
  io.acceptedWrites := acceptedWriteCount
  io.forwardedReads := forwardedReadCount
  io.forwardedWrites := forwardedWriteCount
  io.authenticatedReadResponses := authenticatedReadResponseCount
  io.authenticatedWriteCommits := authenticatedWriteCommitCount
}

object GenerateBitNetUnifiedResidentTensorSemanticFrontend extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-resident-tensor-semantic-frontend")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedResidentTensorSemanticFrontend())
}
