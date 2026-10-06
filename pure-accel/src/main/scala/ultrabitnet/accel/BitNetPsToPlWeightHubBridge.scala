package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Static attachment of the page-copy engine to the production memory hub.
  *
  * Bank 1 is the first PS-DDR high-performance bank and bank 0 is the PL MIG.
  * The copy engine deliberately transfers one aligned 4 KiB page as a single
  * 256-beat logical transaction.  This bridge slices that page into bounded
  * client-native sub-bursts so staging cannot retain a physical bank owner for
  * the whole page.
  */
case class BitNetPsToPlWeightHubBridgeConfig(
    copy: BitNetWeightCopyAxiConfig = BitNetWeightCopyAxiConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    sourceBank: Int = 1,
    destinationBank: Int = 0,
    subBurstBeats: Int = 32) {
  require(sourceBank >= 0 && sourceBank < crossbar.bankCount)
  require(destinationBank >= 0 && destinationBank < crossbar.bankCount)
  require(sourceBank != destinationBank)
  require(copy.dataWidth == crossbar.dataWidth)
  require(copy.staging.model.addressWidth <= crossbar.localAddressWidth)
  require(copy.staging.beatsPerPage == 256)
  require(copy.staging.pageBytes == 4096)
  require(subBurstBeats > 0 && isPow2(subBurstBeats))
  require(copy.staging.beatsPerPage % subBurstBeats == 0)
  require(subBurstBeats <= crossbar.maxBurstBeats)
  require(crossbar.transactionIdWidth >= 2)
  require(crossbar.tagWidth >= crossbar.transactionIdWidth)

  val subBurstCount: Int = copy.staging.beatsPerPage / subBurstBeats
  require(BigInt(subBurstCount) <= (BigInt(1) << crossbar.transactionIdWidth),
    "transaction IDs must not repeat within one raw page")
  require(BigInt(subBurstCount) <= (BigInt(1) << crossbar.tagWidth),
    "tags must not repeat within one raw page")
}

/**
  * Control-only protocol bridge between the existing raw page-copy channels
  * and one client-native read/write slot of
  * [[BitNetProductionPlMemoryOwnershipHub]].
  *
  * A raw AR/AW is captured before it is exposed to the hub.  Generated
  * transaction/tag identities roll at every hub sub-burst, while the raw page
  * owner remains live through the final accepted R/B.  Returned bank,
  * transaction, tag, beat, last and redundant status fields are authenticated
  * before the response is converted back to the copy engine's fixed AXI ID.
  * Stream payloads therefore remain stable under arbitrary hub or copy-engine
  * backpressure, and a stale response can never be relabelled as a later page.
  *
  * Well-formed non-OKAY RRESP/BRESP values are returned to the copy engine so
  * its existing workload completion reports the real memory failure.  A
  * metadata/status-encoding violation is converted to SLVERR (and, where
  * applicable, the raw ID-mismatch sentinel), then puts this bridge into a
  * reset-only quarantine after the current owner is retired.  The component
  * contains no page RAM and no model arithmetic datapath.
  */
class BitNetPsToPlWeightHubBridge(
    cfg: BitNetPsToPlWeightHubBridgeConfig =
      BitNetPsToPlWeightHubBridgeConfig()) extends Component {
  private val copy = cfg.copy
  private val xcfg = cfg.crossbar
  private val pageBeats = copy.staging.beatsPerPage
  private val pageAlignmentBits = log2Up(copy.staging.pageBytes)
  private val subBurstBeats = cfg.subBurstBeats
  private val subBurstCount = cfg.subBurstCount
  private val subBurstBytes = subBurstBeats * (copy.dataWidth / 8)
  private val subBurstIndexWidth = scala.math.max(1, log2Up(subBurstCount))

  val io = new Bundle {
    // Existing BitNetPsToPlWeightStagingSubsystem raw copy boundary.
    val psAr = slave(Stream(BitNetWeightCopyReadAddress(copy)))
    val psR = master(Stream(BitNetWeightCopyReadData(copy)))
    val plAw = slave(Stream(BitNetWeightCopyWriteAddress(copy)))
    val plW = slave(Stream(BitNetWeightCopyWriteData(copy)))
    val plB = master(Stream(BitNetWeightCopyWriteResponse(copy)))

    // Connect these six ports to the production hub's reserved client.
    val hubReadRequest = master(Stream(
      BitNetClientBankReadRequest(xcfg)))
    val hubReadResponse = slave(Stream(
      BitNetClientBankReadResponse(xcfg)))
    val hubWriteRequest = master(Stream(
      BitNetClientBankWriteRequest(xcfg)))
    val hubWriteData = master(Stream(
      BitNetClientBankWriteData(xcfg)))
    val hubWriteResponse = slave(Stream(
      BitNetClientBankWriteResponse(xcfg)))

    val readOwnerValid = out Bool()
    val writeOwnerValid = out Bool()
    val busy = out Bool()
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val readResponseFaultSticky = out Bool()
    val writeResponseFaultSticky = out Bool()
    val acceptedReadBursts = out UInt(32 bits)
    val completedReadBursts = out UInt(32 bits)
    val acceptedWriteBursts = out UInt(32 bits)
    val completedWriteBursts = out UInt(32 bits)
  }

  val quarantine = RegInit(False)
  val protocolFault = RegInit(False)
  val readResponseFault = RegInit(False)
  val writeResponseFault = RegInit(False)

  val acceptedReadCount = Reg(UInt(32 bits)) init 0
  val completedReadCount = Reg(UInt(32 bits)) init 0
  val acceptedWriteCount = Reg(UInt(32 bits)) init 0
  val completedWriteCount = Reg(UInt(32 bits)) init 0

  val mismatchRawId = U(copy.axiId ^ 1, copy.axiIdWidth bits)

  // ----------------------------------------------------------------------
  // PS-DDR read: raw AR -> fixed bank-1 client request -> checked raw R.
  // ----------------------------------------------------------------------
  val readOwnerValid = RegInit(False)
  val readCommandPending = RegInit(False)
  val readRejectResponseValid = RegInit(False)
  val readPoisoned = RegInit(False)
  val readAddress = Reg(UInt(xcfg.localAddressWidth bits)) init 0
  val readTransaction = Reg(UInt(xcfg.transactionIdWidth bits)) init 0
  val readTag = Reg(UInt(xcfg.tagWidth bits)) init 0
  val readSubBurst = Reg(UInt(subBurstIndexWidth bits)) init 0
  val readExpectedBeat = Reg(UInt(xcfg.beatIndexWidth bits)) init 0

  val nextReadTransaction = Reg(UInt(xcfg.transactionIdWidth bits)) init 0
  val nextReadTag = Reg(UInt(xcfg.tagWidth bits)) init 0

  val readRawAddressLegal =
    io.psAr.payload.id === copy.axiId &&
      io.psAr.payload.len === pageBeats - 1 &&
      io.psAr.payload.localAddress(pageAlignmentBits - 1 downto 0) === 0

  // A response without an authenticated owner has priority over a new AR.
  // It is drained and permanently closes the bridge before reuse.
  io.psAr.ready := !readOwnerValid && !quarantine &&
    !io.hubReadResponse.valid
  when(io.psAr.fire) {
    readOwnerValid := True
    readCommandPending := readRawAddressLegal
    readRejectResponseValid := !readRawAddressLegal
    readPoisoned := !readRawAddressLegal
    readAddress := io.psAr.payload.localAddress.resized
    readTransaction := nextReadTransaction
    readTag := nextReadTag
    readSubBurst := 0
    readExpectedBeat := 0
    nextReadTransaction := nextReadTransaction + 1
    nextReadTag := nextReadTag + 1
    acceptedReadCount := acceptedReadCount + 1
    when(!readRawAddressLegal) {
      protocolFault := True
    }
  }

  io.hubReadRequest.valid := readOwnerValid && readCommandPending &&
    !readRejectResponseValid
  io.hubReadRequest.payload.bank := cfg.sourceBank
  io.hubReadRequest.payload.transactionId := readTransaction
  io.hubReadRequest.payload.tag := readTag
  io.hubReadRequest.payload.localAddress := readAddress
  io.hubReadRequest.payload.beats := subBurstBeats
  when(io.hubReadRequest.fire) {
    readCommandPending := False
  }

  val readResponseOwned = readOwnerValid && !readCommandPending &&
    !readRejectResponseValid
  val readExpectedLast =
    readExpectedBeat.resize(xcfg.beatCountWidth) === subBurstBeats - 1
  val readExpectedPageLast =
    readSubBurst === subBurstCount - 1 && readExpectedLast
  val readIdentityLegal = readResponseOwned &&
    io.hubReadResponse.payload.bank === cfg.sourceBank &&
    io.hubReadResponse.payload.transactionId === readTransaction &&
    io.hubReadResponse.payload.tag === readTag &&
    io.hubReadResponse.payload.beatIndex === readExpectedBeat
  val readLastLegal =
    io.hubReadResponse.payload.last === readExpectedLast
  val readStatusEncodingLegal =
    io.hubReadResponse.payload.fault ===
      (io.hubReadResponse.payload.rresp =/= B"2'b00")
  val readProtocolLegal =
    readIdentityLegal && readLastLegal && readStatusEncodingLegal
  val readSubBurstTerminal =
    io.hubReadResponse.payload.last || readExpectedLast
  val readRawTerminal = readSubBurstTerminal &&
    (readExpectedPageLast || readPoisoned || !readProtocolLegal)

  io.psR.valid := readRejectResponseValid ||
    (io.hubReadResponse.valid && readResponseOwned)
  io.psR.payload.id := copy.axiId
  io.psR.payload.data := 0
  io.psR.payload.resp := B"2'b11"
  io.psR.payload.last := True
  when(!readRejectResponseValid) {
    io.psR.payload.id := Mux(readIdentityLegal,
      U(copy.axiId, copy.axiIdWidth bits), mismatchRawId)
    io.psR.payload.data := io.hubReadResponse.payload.data
    io.psR.payload.resp := Mux(readProtocolLegal,
      io.hubReadResponse.payload.rresp, B"2'b10")
    // Hub LAST is local to each bounded sub-burst.  The raw copy engine sees
    // one continuous page and receives LAST only at page beat 255, or at the
    // end of a poisoned sub-burst to fail-stop deterministically.
    io.psR.payload.last := readRawTerminal
  }

  io.hubReadResponse.ready := False
  when(readResponseOwned) {
    io.hubReadResponse.ready := io.psR.ready
  } elsewhen(!readRejectResponseValid) {
    // Ownerless/pre-command data is stale.  Drain it exactly once, then close.
    io.hubReadResponse.ready := True
  }

  when(io.hubReadResponse.fire && readResponseOwned) {
    when(!readProtocolLegal) {
      readPoisoned := True
      protocolFault := True
    }
    when(io.hubReadResponse.payload.rresp =/= B"2'b00" ||
      io.hubReadResponse.payload.fault) {
      readResponseFault := True
    }
    when(readSubBurstTerminal) {
      readExpectedBeat := 0
      when(readPoisoned || !readProtocolLegal) {
        readOwnerValid := False
        readCommandPending := False
        completedReadCount := completedReadCount + 1
        quarantine := True
      } elsewhen(readExpectedPageLast) {
        readOwnerValid := False
        readCommandPending := False
        completedReadCount := completedReadCount + 1
      } otherwise {
        // Allocate a fresh identity only after this sub-burst's final R beat
        // retires; no response can be relabelled as the next slice.
        readSubBurst := readSubBurst + 1
        readAddress := readAddress + subBurstBytes
        readTransaction := nextReadTransaction
        readTag := nextReadTag
        nextReadTransaction := nextReadTransaction + 1
        nextReadTag := nextReadTag + 1
        readCommandPending := True
      }
    } otherwise {
      readExpectedBeat := readExpectedBeat + 1
    }
  }

  // An illegal raw AR is completed locally with one DECERR/last beat.  It is
  // deliberately not sent to the memory hub.
  when(readRejectResponseValid && io.psR.fire) {
    readRejectResponseValid := False
    readOwnerValid := False
    completedReadCount := completedReadCount + 1
    quarantine := True
  }

  when(io.hubReadResponse.fire && !readResponseOwned) {
    protocolFault := True
    quarantine := True
    // If an owner was waiting for its request, surface the stale-return fault
    // to the raw engine rather than silently wedging it.
    when(readOwnerValid) {
      readCommandPending := False
      readRejectResponseValid := True
      readPoisoned := True
    }
  }

  // ----------------------------------------------------------------------
  // PL-DDR write: raw AW/W -> fixed bank-0 request/data -> checked raw B.
  // ----------------------------------------------------------------------
  val writeOwnerValid = RegInit(False)
  val writeCommandPending = RegInit(False)
  val writeDataDone = RegInit(False)
  val writeRejectDrain = RegInit(False)
  val writeSyntheticResponseValid = RegInit(False)
  val writeSyntheticResponseResp = Reg(Bits(2 bits)) init 0
  val writeSyntheticResponseIdMismatch = RegInit(False)
  val writePrematureResponseObserved = RegInit(False)
  val writePoisoned = RegInit(False)
  val writeRawPacketDone = RegInit(False)
  val writeAddress = Reg(UInt(xcfg.localAddressWidth bits)) init 0
  val writeTransaction = Reg(UInt(xcfg.transactionIdWidth bits)) init 0
  val writeTag = Reg(UInt(xcfg.tagWidth bits)) init 0
  val writeSubBurst = Reg(UInt(subBurstIndexWidth bits)) init 0
  val writeExpectedBeat = Reg(UInt(xcfg.beatIndexWidth bits)) init 0
  val writeRawBeat = Reg(UInt(copy.beatIndexWidth bits)) init 0
  val writeAggregatedResponse = Reg(Bits(2 bits)) init 0

  val nextWriteTransaction = Reg(UInt(xcfg.transactionIdWidth bits)) init
    (BigInt(1) << (xcfg.transactionIdWidth - 1))
  val nextWriteTag = Reg(UInt(xcfg.tagWidth bits)) init
    (BigInt(1) << (xcfg.tagWidth - 1))

  val writeRawAddressLegal =
    io.plAw.payload.id === copy.axiId &&
      io.plAw.payload.len === pageBeats - 1 &&
      io.plAw.payload.localAddress(pageAlignmentBits - 1 downto 0) === 0

  io.plAw.ready := !writeOwnerValid && !quarantine &&
    !io.hubWriteResponse.valid
  when(io.plAw.fire) {
    writeOwnerValid := True
    writeCommandPending := writeRawAddressLegal
    writeDataDone := False
    writeRejectDrain := !writeRawAddressLegal
    writeSyntheticResponseValid := False
    writeSyntheticResponseResp := B"2'b11"
    writeSyntheticResponseIdMismatch := False
    writePrematureResponseObserved := False
    writePoisoned := !writeRawAddressLegal
    writeRawPacketDone := False
    writeAddress := io.plAw.payload.localAddress.resized
    writeTransaction := nextWriteTransaction
    writeTag := nextWriteTag
    writeSubBurst := 0
    writeExpectedBeat := 0
    writeRawBeat := 0
    writeAggregatedResponse := B"2'b00"
    nextWriteTransaction := nextWriteTransaction + 1
    nextWriteTag := nextWriteTag + 1
    acceptedWriteCount := acceptedWriteCount + 1
    when(!writeRawAddressLegal) {
      protocolFault := True
    }
  }

  io.hubWriteRequest.valid := writeOwnerValid && writeCommandPending &&
    !writeRejectDrain
  io.hubWriteRequest.payload.bank := cfg.destinationBank
  io.hubWriteRequest.payload.transactionId := writeTransaction
  io.hubWriteRequest.payload.tag := writeTag
  io.hubWriteRequest.payload.localAddress := writeAddress
  io.hubWriteRequest.payload.beats := subBurstBeats
  when(io.hubWriteRequest.fire) {
    writeCommandPending := False
  }

  val writeExpectedSubBurstLast =
    writeExpectedBeat.resize(xcfg.beatCountWidth) === subBurstBeats - 1
  val writeExpectedRawLast = writeRawBeat === pageBeats - 1
  val writeRawLastLegal = io.plW.payload.last === writeExpectedRawLast
  val writeDataActive = writeOwnerValid && !writeCommandPending &&
    !writeDataDone && !writeRejectDrain &&
    !writeSyntheticResponseValid

  io.hubWriteData.valid := io.plW.valid && writeDataActive
  io.hubWriteData.payload.bank := cfg.destinationBank
  io.hubWriteData.payload.transactionId := writeTransaction
  io.hubWriteData.payload.tag := writeTag
  io.hubWriteData.payload.beatIndex := writeExpectedBeat
  io.hubWriteData.payload.data := io.plW.payload.data
  io.hubWriteData.payload.strobe := io.plW.payload.strobe
  // The hub sees a terminal W every sub-burst.  An illegal early raw LAST is
  // also forwarded so the already-issued hub owner can retire safely.
  io.hubWriteData.payload.last :=
    writeExpectedSubBurstLast || io.plW.payload.last

  io.plW.ready := False
  when(writeRejectDrain) {
    io.plW.ready := True
  } elsewhen(writeDataActive) {
    io.plW.ready := io.hubWriteData.ready
  }

  when(io.plW.fire && writeDataActive) {
    when(!writeRawLastLegal) {
      writePoisoned := True
      protocolFault := True
    }
    when(io.plW.payload.last || writeExpectedRawLast) {
      writeRawPacketDone := True
    } otherwise {
      writeRawBeat := writeRawBeat + 1
    }
    when(io.plW.payload.last || writeExpectedSubBurstLast) {
      writeDataDone := True
    } otherwise {
      writeExpectedBeat := writeExpectedBeat + 1
    }
  }

  // Invalid AW commands, and the raw suffix after a poisoned hub sub-burst,
  // are drained locally.  No malformed suffix can create an orphan hub W.
  when(io.plW.fire && writeRejectDrain) {
    when(io.plW.payload.last || writeExpectedRawLast) {
      writeRejectDrain := False
      writeDataDone := True
      writeRawPacketDone := True
      writeSyntheticResponseValid := True
    } otherwise {
      writeRawBeat := writeRawBeat + 1
    }
  }
  val writeResponseOwned = writeOwnerValid && !writeCommandPending &&
    writeDataDone && !writeRejectDrain &&
    !writeSyntheticResponseValid

  // A B response is not owned until the hub request and every W beat have
  // retired.  If the hub advertises B early, remember the protocol violation
  // but deliberately hold ready low.  The request and W stream must complete
  // normally; only then may the still-stable B be consumed and poisoned.
  when(io.hubWriteResponse.valid && writeOwnerValid &&
    !writeRejectDrain && !writeSyntheticResponseValid &&
    !writeResponseOwned) {
    writePrematureResponseObserved := True
    writePoisoned := True
  }

  val writeIdentityLegal = writeResponseOwned &&
    io.hubWriteResponse.payload.bank === cfg.destinationBank &&
    io.hubWriteResponse.payload.transactionId === writeTransaction &&
    io.hubWriteResponse.payload.tag === writeTag
  val writeStatusEncodingLegal =
    io.hubWriteResponse.payload.fault ===
      (io.hubWriteResponse.payload.bresp =/= B"2'b00")
  val writeProtocolLegal = writeIdentityLegal &&
    writeStatusEncodingLegal && !writePrematureResponseObserved
  val writeLastSubBurst = writeSubBurst === subBurstCount - 1
  val writeResponseTerminatesRawPage =
    writeRawPacketDone || writeLastSubBurst

  val writePageResponse = Bits(2 bits)
  writePageResponse := io.hubWriteResponse.payload.bresp
  when(writeAggregatedResponse =/= B"2'b00") {
    writePageResponse := writeAggregatedResponse
  }

  io.plB.valid := (writeSyntheticResponseValid && writeOwnerValid &&
    writeDataDone) || (io.hubWriteResponse.valid && writeResponseOwned &&
    writeResponseTerminatesRawPage)
  io.plB.payload.id := Mux(writeSyntheticResponseIdMismatch,
    mismatchRawId, U(copy.axiId, copy.axiIdWidth bits))
  io.plB.payload.resp := writeSyntheticResponseResp
  when(!writeSyntheticResponseValid) {
    io.plB.payload.id := Mux(writeIdentityLegal,
      U(copy.axiId, copy.axiIdWidth bits), mismatchRawId)
    io.plB.payload.resp := Mux(writeProtocolLegal && !writePoisoned,
      writePageResponse, B"2'b10")
  }

  io.hubWriteResponse.ready := False
  when(writeResponseOwned) {
    // Intermediate B responses retire internally.  The eighth (or a response
    // after an early raw packet end) is held until the one raw B can retire.
    io.hubWriteResponse.ready := Mux(writeResponseTerminatesRawPage,
      io.plB.ready, True)
  } elsewhen((!writeOwnerValid || writeRejectDrain) &&
    !writeSyntheticResponseValid) {
    // A genuinely ownerless B is stale.  Drain it once and fail closed.
    io.hubWriteResponse.ready := True
  }

  when(io.hubWriteResponse.fire && writeResponseOwned) {
    when(!writeProtocolLegal) {
      writePoisoned := True
      protocolFault := True
    }
    when(io.hubWriteResponse.payload.bresp =/= B"2'b00" ||
      io.hubWriteResponse.payload.fault) {
      writeResponseFault := True
      when(writeAggregatedResponse === B"2'b00") {
        writeAggregatedResponse := io.hubWriteResponse.payload.bresp
      }
    }

    when(writeResponseTerminatesRawPage) {
      // This same handshake is the raw B handshake because ready is coupled
      // above.  Counts remain page-level rather than sub-burst-level.
      writeOwnerValid := False
      writeDataDone := False
      completedWriteCount := completedWriteCount + 1
      when(writePoisoned || !writeProtocolLegal) {
        quarantine := True
      }
    } elsewhen(writePoisoned || !writeProtocolLegal) {
      // Finish and authenticate the current sub-burst, then stop issuing hub
      // requests.  The rest of the bounded raw page is consumed locally.
      writeDataDone := False
      writeRejectDrain := True
      writeSyntheticResponseResp := B"2'b10"
      writeSyntheticResponseIdMismatch := !writeIdentityLegal
      writePrematureResponseObserved := False
    } otherwise {
      // A fresh command identity is allocated only after the preceding B.
      // Consequently the second sub-burst can never overtake the first B.
      writeSubBurst := writeSubBurst + 1
      writeAddress := writeAddress + subBurstBytes
      writeTransaction := nextWriteTransaction
      writeTag := nextWriteTag
      nextWriteTransaction := nextWriteTransaction + 1
      nextWriteTag := nextWriteTag + 1
      writeExpectedBeat := 0
      writeDataDone := False
      writeCommandPending := True
      writePrematureResponseObserved := False
    }
  }

  when(writeSyntheticResponseValid && io.plB.fire) {
    writeSyntheticResponseValid := False
    writeOwnerValid := False
    writeDataDone := False
    completedWriteCount := completedWriteCount + 1
    quarantine := True
  }

  when(io.hubWriteResponse.fire && (!writeOwnerValid || writeRejectDrain)) {
    protocolFault := True
    quarantine := True
  }

  io.readOwnerValid := readOwnerValid
  io.writeOwnerValid := writeOwnerValid
  io.busy := readOwnerValid || writeOwnerValid ||
    readRejectResponseValid || writeSyntheticResponseValid
  io.quarantined := quarantine
  io.protocolFaultSticky := protocolFault
  io.readResponseFaultSticky := readResponseFault
  io.writeResponseFaultSticky := writeResponseFault
  io.acceptedReadBursts := acceptedReadCount
  io.completedReadBursts := completedReadCount
  io.acceptedWriteBursts := acceptedWriteCount
  io.completedWriteBursts := completedWriteCount
}

object GenerateBitNetPsToPlWeightHubBridge extends App {
  SpinalConfig(targetDirectory = "generated/bitnet-weight-hub-bridge")
    .generateVerilog(new BitNetPsToPlWeightHubBridge())
}
