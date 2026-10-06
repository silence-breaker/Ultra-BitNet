package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Static users of the one packed-i8 semantic port.  The encoding is also
  * used as the index of the three north-side Stream vectors.
  */
object BitNetUnifiedPackedSemanticClient
    extends SpinalEnum(binarySequential) {
  val Projection, Query, LmHead = newElement()
}

/** Router-local protocol diagnostics.  Return-side faults quarantine the
  * router immediately.  Source-side faults denote a producer violating the
  * Stream rule that valid and identity remain stable until ready.
  */
object BitNetUnifiedPackedSemanticRouterFault {
  val UnexpectedReadData = 0
  val ReadDataIdentity = 1
  val UnexpectedWriteFault = 2
  val WriteFaultIdentity = 3
  val ReadSourceDropped = 4
  val ReadSourceIdentity = 5
  val WriteSourceDropped = 6
  val WriteSourceIdentity = 7
}

/** Stable structural marker consumed by production elaboration tests.
  *
  * The router retains only compact write owner/identity state.  The selected
  * Stream producer remains responsible for all packed data bits until the
  * client and backend handshakes occur in the same cycle.
  */
object BitNetUnifiedPackedSemanticClientRouterStorageContract {
  val RetainedPackedReadPayloadSnapshots: Int = 0
  val RetainedPackedReadPayloadBits: Int = 0
  val RetainedPackedWritePayloadSnapshots: Int = 0
  val RetainedPackedWritePayloadBits: Int = 0
  val DirectLockedClientWritePayload: Boolean = true
}

case class BitNetUnifiedPackedSemanticClientRouterConfig(
    scratchpad: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()) {
  require(BitNetUnifiedPackedSemanticClient.elements.size == 3)
  require(scratchpad.layout.bankCount == 32)
  require(scratchpad.layout.wordWidth ==
    scratchpad.layout.tokenLanes * scratchpad.layout.activationWidth)
}

/**
  * Memory-free three-client switch for the unique packed semantic frontend.
  *
  * Read and write arbitration are independent.  Each uses a rotating
  * priority pointer and locks the first selected producer before exposing it
  * southbound.  A read owner remains locked from command selection through
  * consumption of its response.  A write owner remains locked through the
  * semantic frontend's deliberately deferred `ready`, or through consumption
  * of its retained fault response.  Consequently neither long frontend
  * backpressure nor a competing client can change a command in flight.
  *
  * Returned data/faults are routed solely by the retained owner, never by
  * return metadata.  Region, row and mask are authenticated against the
  * retained command.  An unsolicited or mismatched return sets an irreversible
  * quarantine before another command can be selected; an already promised
  * response is nevertheless allowed to drain to its retained owner.
  *
  * This component intentionally owns no Mem, frontend, bridge, backend or
  * packed-data snapshot.  Selection is not acceptance: the locked producer
  * must obey the Stream contract and retain valid plus its complete payload
  * until the client and semantic frontend fire together.  A producer which
  * withdraws or mutates identity before that terminal fails closed without
  * forwarding invalid data.  After fire, compact identity remains for the
  * semantic frontend's one-cycle delayed-fault window.
  */
class BitNetUnifiedPackedSemanticClientRouter(
    cfg: BitNetUnifiedPackedSemanticClientRouterConfig =
      BitNetUnifiedPackedSemanticClientRouterConfig()) extends Component {
  private val scratchpadCfg = cfg.scratchpad
  private val layout = scratchpadCfg.layout
  private val clients = BitNetUnifiedPackedSemanticClient.elements
  private val clientCount = clients.size
  private val clientIndexWidth = log2Up(clientCount)
  private val packedReadLaneCount =
    layout.packedTransportSegmentCount
  private val packedReadFullLaneMask = B(
    (BigInt(1) << packedReadLaneCount) - 1,
    packedReadLaneCount bits)

  // Public Scala-level elaboration markers: the structural test reads these
  // from the actual generated production component rather than guessing from
  // tool-dependent Verilog signal names.
  val retainedPackedWritePayloadSnapshots: Int =
    BitNetUnifiedPackedSemanticClientRouterStorageContract
      .RetainedPackedWritePayloadSnapshots
  val retainedPackedWritePayloadBits: Int =
    BitNetUnifiedPackedSemanticClientRouterStorageContract
      .RetainedPackedWritePayloadBits
  val directLockedClientWritePayload: Boolean =
    BitNetUnifiedPackedSemanticClientRouterStorageContract
      .DirectLockedClientWritePayload
  val retainedPackedReadPayloadSnapshots: Int =
    BitNetUnifiedPackedSemanticClientRouterStorageContract
      .RetainedPackedReadPayloadSnapshots
  val retainedPackedReadPayloadBits: Int =
    BitNetUnifiedPackedSemanticClientRouterStorageContract
      .RetainedPackedReadPayloadBits
  require(retainedPackedReadPayloadSnapshots == 0)
  require(retainedPackedReadPayloadBits == 0)
  require(retainedPackedWritePayloadSnapshots == 0)
  require(retainedPackedWritePayloadBits == 0)
  require(directLockedClientWritePayload)

  val io = new Bundle {
    val clientPackedRead = Vec(
      slave(Stream(BitNetUnifiedResidentPackedI8Read(scratchpadCfg))),
      clientCount)
    val clientPackedReadData = Vec(
      master(BitNetUnifiedResidentPackedI8ReadParallelChannel(
        scratchpadCfg)),
      clientCount)
    val clientPackedWrite = Vec(
      slave(Stream(BitNetUnifiedResidentPackedI8Write(scratchpadCfg))),
      clientCount)
    val clientPackedWriteFault = Vec(
      master(Stream(
        BitNetUnifiedResidentPackedI8WriteFault(scratchpadCfg))),
      clientCount)

    val packedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(scratchpadCfg)))
    val packedReadData = slave(
      BitNetUnifiedResidentPackedI8ReadParallelChannel(scratchpadCfg))
    val packedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(scratchpadCfg)))
    val packedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(scratchpadCfg)))

    val readOwnerValid = out Bool()
    val readOwner = out(BitNetUnifiedPackedSemanticClient())
    val readCommandHeld = out Bool()
    val readResponseOutstanding = out Bool()
    val writeOwnerValid = out Bool()
    val writeOwner = out(BitNetUnifiedPackedSemanticClient())
    val writeCommandHeld = out Bool()
    val busy = out Bool()
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(8 bits)
    val acceptedReads = out UInt(32 bits)
    val completedReadResponses = out UInt(32 bits)
    val completedWrites = out UInt(32 bits)
    val completedWriteFaults = out UInt(32 bits)
  }

  private def enumFromIndex(index: UInt):
      BitNetUnifiedPackedSemanticClient.C = {
    val result = BitNetUnifiedPackedSemanticClient()
    result := BitNetUnifiedPackedSemanticClient.Projection
    switch(index) {
      is(1) { result := BitNetUnifiedPackedSemanticClient.Query }
      is(2) { result := BitNetUnifiedPackedSemanticClient.LmHead }
    }
    result
  }

  private def nextClient(index: UInt): UInt = {
    val result = UInt(clientIndexWidth bits)
    result := index + 1
    when(index === clientCount - 1) {
      result := 0
    }
    result
  }

  private def readIdentityMatches(
      payload: BitNetUnifiedResidentPackedI8Read,
      region: BitNetUnifiedResidentTensorRegion.C,
      localRow: UInt,
      bankMask: Bits,
      aliasOwner: BitNetUnifiedResidentTensorAliasOwner.C,
      aliasGeneration: UInt): Bool =
    payload.region === region &&
      payload.localRow === localRow &&
      payload.bankMask === bankMask &&
      payload.aliasOwner === aliasOwner &&
      payload.aliasGeneration === aliasGeneration

  private def writeIdentityMatches(
      payload: BitNetUnifiedResidentPackedI8Write,
      region: BitNetUnifiedResidentTensorRegion.C,
      localRow: UInt,
      segmentIndex: UInt,
      symbolMask: Bits,
      aliasOwner: BitNetUnifiedResidentTensorAliasOwner.C,
      aliasGeneration: UInt): Bool =
    payload.region === region &&
      payload.localRow === localRow &&
      payload.segmentIndex === segmentIndex &&
      payload.symbolMask === symbolMask &&
      payload.aliasOwner === aliasOwner &&
      payload.aliasGeneration === aliasGeneration

  private def returnIdentityMatches(
      region: BitNetUnifiedResidentTensorRegion.C,
      localRow: UInt,
      bankMask: Bits,
      expectedRegion: BitNetUnifiedResidentTensorRegion.C,
      expectedLocalRow: UInt,
      expectedBankMask: Bits): Bool =
    region === expectedRegion &&
      localRow === expectedLocalRow &&
      bankMask === expectedBankMask

  // Independent rotating-priority cursors.  Reset serves Projection first;
  // every completed transaction starts the next search at its successor.
  val readRoundRobin = Reg(UInt(clientIndexWidth bits)) init 0
  val writeRoundRobin = Reg(UInt(clientIndexWidth bits)) init 0

  val readCommandHeldReg = RegInit(False)
  val readResponseOutstandingReg = RegInit(False)
  val readResponsePendingMaskReg =
    Reg(Bits(layout.packedTransportSegmentCount bits)) init 0
  val readResponseMetadataSeenReg = RegInit(False)
  val readResponseExpectedFaultReg = RegInit(False)
  val readResponseExpectedFaultCodeReg = Reg(Bits(4 bits)) init 0
  val readProtocolFaultPendingReg = RegInit(False)
  val readProtocolFaultCodeReg = Reg(Bits(4 bits)) init 0
  val readOwnerIndex = Reg(UInt(clientIndexWidth bits)) init 0
  val readExpectedRegion =
    Reg(BitNetUnifiedResidentTensorRegion()) init
      BitNetUnifiedResidentTensorRegion.ProjectionSlot0
  val readExpectedLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val readExpectedBankMask = Reg(Bits(layout.bankCount bits)) init 0
  val readExpectedAliasOwner =
    Reg(BitNetUnifiedResidentTensorAliasOwner()) init
      BitNetUnifiedResidentTensorAliasOwner.Query
  val readExpectedAliasGeneration =
    Reg(UInt(layout.aliasGenerationWidth bits)) init 0

  val writeCommandHeldReg = RegInit(False)
  // Keep the owner for the cycle following southbound fire.  Legal writes use
  // fire as their committed acknowledgement; an immediately rejected semantic
  // write instead raises its registered local fault in that following cycle.
  val writePostFireFaultWindowReg = RegInit(False)
  val writeOwnerIndex = Reg(UInt(clientIndexWidth bits)) init 0
  val writeExpectedRegion =
    Reg(BitNetUnifiedResidentTensorRegion()) init
      BitNetUnifiedResidentTensorRegion.ProjectionSlot0
  val writeExpectedLocalRow =
    Reg(UInt(layout.localRowWidth bits)) init 0
  val writeExpectedSegmentIndex = Reg(
    UInt(log2Up(layout.packedTransportSegmentCount) bits)) init 0
  val writeExpectedSymbolMask =
    Reg(Bits(layout.nativeSymbolCount bits)) init 0
  val writeExpectedAliasOwner =
    Reg(BitNetUnifiedResidentTensorAliasOwner()) init
      BitNetUnifiedResidentTensorAliasOwner.Query
  val writeExpectedAliasGeneration =
    Reg(UInt(layout.aliasGenerationWidth bits)) init 0

  val quarantineReg = RegInit(False)
  val faultBitsReg = Reg(Bits(8 bits)) init 0
  val acceptedReadsReg = Reg(UInt(32 bits)) init 0
  val completedReadResponsesReg = Reg(UInt(32 bits)) init 0
  val completedWritesReg = Reg(UInt(32 bits)) init 0
  val completedWriteFaultsReg = Reg(UInt(32 bits)) init 0

  // -------------------------------------------------------------------
  // Retained-source checks.  Non-selected clients are intentionally free to
  // change; only the locked Stream producer owes stability until ready.
  // -------------------------------------------------------------------
  val lockedReadSourceValid =
    io.clientPackedRead(readOwnerIndex).valid
  val lockedReadSourceIdentityMatches = readIdentityMatches(
    io.clientPackedRead(readOwnerIndex).payload,
    readExpectedRegion,
    readExpectedLocalRow,
    readExpectedBankMask,
    readExpectedAliasOwner,
    readExpectedAliasGeneration)
  val readSourceDroppedNow = readCommandHeldReg &&
    !lockedReadSourceValid
  val readSourceIdentityNow = readCommandHeldReg &&
    lockedReadSourceValid && !lockedReadSourceIdentityMatches

  val lockedWriteSourceValid =
    io.clientPackedWrite(writeOwnerIndex).valid
  val lockedWriteSourceIdentityMatches = writeIdentityMatches(
    io.clientPackedWrite(writeOwnerIndex).payload,
    writeExpectedRegion,
    writeExpectedLocalRow,
    writeExpectedSegmentIndex,
    writeExpectedSymbolMask,
    writeExpectedAliasOwner,
    writeExpectedAliasGeneration)
  // A semantic fault is itself the retained write's terminal response.  Once
  // it is present, source withdrawal/mutation is no longer diagnosed as a
  // second producer fault; matching and attribution belong exclusively to the
  // retained fault identity checks below.
  val writeSourceDroppedNow = writeCommandHeldReg &&
    !io.packedWriteFault.valid && !lockedWriteSourceValid
  val writeSourceIdentityNow = writeCommandHeldReg &&
    !io.packedWriteFault.valid && lockedWriteSourceValid &&
    !lockedWriteSourceIdentityMatches

  // -------------------------------------------------------------------
  // Authenticate southbound returns before they affect arbitration.  The
  // production semantic frontend cannot return before the slow cycle after a
  // read issue.  Therefore only the registered outstanding state authorizes a
  // response; data concurrent with issue is unsolicited and is drained while
  // quarantining the router.  The registered quarantine is the only protocol
  // state allowed to gate an already locked request, so a current-cycle return
  // detector never feeds back through packedRead.valid/ready.
  // -------------------------------------------------------------------
  val retainedReadRequestPermit = !quarantineReg
  val readReturnExpected = readResponseOutstandingReg
  val readReturnIdentityMatches = returnIdentityMatches(
    io.packedReadData.region,
    io.packedReadData.localRow,
    io.packedReadData.bankMask,
    readExpectedRegion,
    readExpectedLocalRow,
    readExpectedBankMask)
  val readReturnOffered = io.packedReadData.segmentValid.orR
  val readReturnLaneMaskMatches =
    io.packedReadData.laneMask === packedReadFullLaneMask
  val readReturnValidSetMatches =
    io.packedReadData.segmentValid === readResponsePendingMaskReg
  val readReturnFaultCoherent =
    io.packedReadData.fault === io.packedReadData.faultCode.orR
  val readReturnBaseProtocolMatches = readReturnIdentityMatches &&
    readReturnLaneMaskMatches && readReturnValidSetMatches &&
    readReturnFaultCoherent
  val readReturnSharedMetadataMatches =
    !readResponseMetadataSeenReg ||
      (io.packedReadData.fault === readResponseExpectedFaultReg &&
        io.packedReadData.faultCode ===
          readResponseExpectedFaultCodeReg)
  val readReturnProtocolMatches = readReturnBaseProtocolMatches &&
    readReturnSharedMetadataMatches
  val unexpectedReadDataNow = readReturnOffered &&
    !readReturnExpected
  val readDataIdentityNow = readReturnOffered &&
    readReturnExpected && !readProtocolFaultPendingReg &&
    !readReturnProtocolMatches

  val writeFaultExpected = writeCommandHeldReg ||
    writePostFireFaultWindowReg
  val writeFaultIdentityMatches =
    io.packedWriteFault.payload.region === writeExpectedRegion &&
      io.packedWriteFault.payload.localRow === writeExpectedLocalRow &&
      io.packedWriteFault.payload.segmentIndex === writeExpectedSegmentIndex
  val unexpectedWriteFaultNow = io.packedWriteFault.valid &&
    !writeFaultExpected
  val writeFaultIdentityNow = io.packedWriteFault.valid &&
    writeFaultExpected && !writeFaultIdentityMatches

  val protocolFaultNow = unexpectedReadDataNow || readDataIdentityNow ||
    unexpectedWriteFaultNow || writeFaultIdentityNow ||
    readSourceDroppedNow || readSourceIdentityNow ||
    writeSourceDroppedNow || writeSourceIdentityNow
  // Current-cycle detectors can suppress acquisition of a new owner, but do
  // not form part of the combinational request handshake of the retained one.
  val healthyNow = !quarantineReg && !protocolFaultNow

  // -------------------------------------------------------------------
  // Read request selection and locked command forwarding.
  // -------------------------------------------------------------------
  val readChoiceValid = Bool()
  val readChoice = UInt(clientIndexWidth bits)
  readChoiceValid := False
  readChoice := readRoundRobin
  switch(readRoundRobin) {
    is(0) {
      when(io.clientPackedRead(0).valid) {
        readChoiceValid := True; readChoice := 0
      } elsewhen(io.clientPackedRead(1).valid) {
        readChoiceValid := True; readChoice := 1
      } elsewhen(io.clientPackedRead(2).valid) {
        readChoiceValid := True; readChoice := 2
      }
    }
    is(1) {
      when(io.clientPackedRead(1).valid) {
        readChoiceValid := True; readChoice := 1
      } elsewhen(io.clientPackedRead(2).valid) {
        readChoiceValid := True; readChoice := 2
      } elsewhen(io.clientPackedRead(0).valid) {
        readChoiceValid := True; readChoice := 0
      }
    }
    default {
      when(io.clientPackedRead(2).valid) {
        readChoiceValid := True; readChoice := 2
      } elsewhen(io.clientPackedRead(0).valid) {
        readChoiceValid := True; readChoice := 0
      } elsewhen(io.clientPackedRead(1).valid) {
        readChoiceValid := True; readChoice := 1
      }
    }
  }

  when(!readCommandHeldReg && !readResponseOutstandingReg &&
      healthyNow && readChoiceValid) {
    readCommandHeldReg := True
    readOwnerIndex := readChoice
    readExpectedRegion := io.clientPackedRead(readChoice).region
    readExpectedLocalRow := io.clientPackedRead(readChoice).localRow
    readExpectedBankMask := io.clientPackedRead(readChoice).bankMask
    readExpectedAliasOwner := io.clientPackedRead(readChoice).aliasOwner
    readExpectedAliasGeneration :=
      io.clientPackedRead(readChoice).aliasGeneration
  }

  io.packedRead.valid := readCommandHeldReg && retainedReadRequestPermit &&
    lockedReadSourceValid && lockedReadSourceIdentityMatches
  // Metadata is retained explicitly so non-selected clients can never perturb
  // the visible command while the physical frontend applies backpressure.
  io.packedRead.region := readExpectedRegion
  io.packedRead.localRow := readExpectedLocalRow
  io.packedRead.bankMask := readExpectedBankMask
  io.packedRead.aliasOwner := readExpectedAliasOwner
  io.packedRead.aliasGeneration := readExpectedAliasGeneration

  for (client <- 0 until clientCount) {
    io.clientPackedRead(client).ready := False
    when(readCommandHeldReg && readOwnerIndex === client &&
        retainedReadRequestPermit && lockedReadSourceIdentityMatches) {
      io.clientPackedRead(client).ready := io.packedRead.ready
    }
  }

  val readIssueFire = io.packedRead.fire
  when(readIssueFire) {
    readCommandHeldReg := False
    readResponseOutstandingReg := True
    readResponsePendingMaskReg := packedReadFullLaneMask
    readResponseMetadataSeenReg := False
    readProtocolFaultPendingReg := False
    readProtocolFaultCodeReg := 0
    readRoundRobin := nextClient(readOwnerIndex)
    acceptedReadsReg := acceptedReadsReg + 1
  }

  // -------------------------------------------------------------------
  // Read data is routed as four simultaneously available token-lane
  // segments.  Only compact owner/header/pending state is retained here; the
  // 4096 data bits remain owned by the semantic frontend until their
  // individual ready bits fire.  Malformed returns are drained and converted
  // into a canonical retained-header fault.  Fault payload data is undefined
  // by contract, so this conversion needs no wide snapshot or zeroing mux.
  // -------------------------------------------------------------------
  io.packedReadData.segmentReady := 0
  val readClientFireMask = Bits(packedReadLaneCount bits)
  readClientFireMask := 0

  for (client <- 0 until clientCount) {
    val clientData = io.clientPackedReadData(client)
    clientData.region := readExpectedRegion
    clientData.localRow := readExpectedLocalRow
    clientData.bankMask := readExpectedBankMask
    clientData.laneMask := packedReadFullLaneMask
    clientData.fault := False
    clientData.faultCode := 0
    clientData.segmentValid := 0
    for (lane <- 0 until packedReadLaneCount) {
      clientData.segments(lane) := io.packedReadData.segments(lane)
    }

    when(readReturnExpected && readOwnerIndex === client) {
      when(readProtocolFaultPendingReg) {
        clientData.fault := True
        clientData.faultCode := readProtocolFaultCodeReg
        clientData.segmentValid := readResponsePendingMaskReg
      } elsewhen(readReturnOffered && !readDataIdentityNow) {
        clientData.fault := io.packedReadData.fault
        clientData.faultCode := io.packedReadData.faultCode
        clientData.segmentValid := io.packedReadData.segmentValid
        io.packedReadData.segmentReady := clientData.segmentReady
      }
      readClientFireMask := clientData.segmentValid &
        clientData.segmentReady
    }
  }

  // Unsolicited or malformed physical lanes must never hold the shared
  // frontend hostage.  During local fault delivery, any late residual lanes
  // are likewise drained independently of the retained client's readiness.
  when(unexpectedReadDataNow || readDataIdentityNow ||
      readProtocolFaultPendingReg) {
    io.packedReadData.segmentReady := io.packedReadData.segmentValid
  }

  when(readDataIdentityNow) {
    readProtocolFaultPendingReg := True
    readProtocolFaultCodeReg := io.packedReadData.faultCode |
      B(BitNetUnifiedResidentTensorSemanticFrontendFault
        .BackendProtocolMask, 4 bits)
  }
  when(readReturnOffered && readReturnExpected &&
      !readProtocolFaultPendingReg && !readResponseMetadataSeenReg &&
      readReturnBaseProtocolMatches) {
    readResponseMetadataSeenReg := True
    readResponseExpectedFaultReg := io.packedReadData.fault
    readResponseExpectedFaultCodeReg := io.packedReadData.faultCode
  }

  val readResponsePendingAfter = readResponsePendingMaskReg &
    ~readClientFireMask
  when(readClientFireMask.orR) {
    readResponsePendingMaskReg := readResponsePendingAfter
    when(!readResponsePendingAfter.orR) {
      readResponseOutstandingReg := False
      readResponseMetadataSeenReg := False
      readProtocolFaultPendingReg := False
      completedReadResponsesReg := completedReadResponsesReg + 1
    }
  }
  // -------------------------------------------------------------------
  // Write request selection and retained owner.  The write payload continues
  // to come from that one locked Stream; only identity is locally retained.
  // -------------------------------------------------------------------
  val writeChoiceValid = Bool()
  val writeChoice = UInt(clientIndexWidth bits)
  writeChoiceValid := False
  writeChoice := writeRoundRobin
  switch(writeRoundRobin) {
    is(0) {
      when(io.clientPackedWrite(0).valid) {
        writeChoiceValid := True; writeChoice := 0
      } elsewhen(io.clientPackedWrite(1).valid) {
        writeChoiceValid := True; writeChoice := 1
      } elsewhen(io.clientPackedWrite(2).valid) {
        writeChoiceValid := True; writeChoice := 2
      }
    }
    is(1) {
      when(io.clientPackedWrite(1).valid) {
        writeChoiceValid := True; writeChoice := 1
      } elsewhen(io.clientPackedWrite(2).valid) {
        writeChoiceValid := True; writeChoice := 2
      } elsewhen(io.clientPackedWrite(0).valid) {
        writeChoiceValid := True; writeChoice := 0
      }
    }
    default {
      when(io.clientPackedWrite(2).valid) {
        writeChoiceValid := True; writeChoice := 2
      } elsewhen(io.clientPackedWrite(0).valid) {
        writeChoiceValid := True; writeChoice := 0
      } elsewhen(io.clientPackedWrite(1).valid) {
        writeChoiceValid := True; writeChoice := 1
      }
    }
  }

  when(!writeCommandHeldReg && !writePostFireFaultWindowReg &&
      healthyNow && writeChoiceValid) {
    writeCommandHeldReg := True
    writeOwnerIndex := writeChoice
    writeExpectedRegion := io.clientPackedWrite(writeChoice).region
    writeExpectedLocalRow := io.clientPackedWrite(writeChoice).localRow
    writeExpectedSegmentIndex :=
      io.clientPackedWrite(writeChoice).segmentIndex
    writeExpectedSymbolMask := io.clientPackedWrite(writeChoice).symbolMask
    writeExpectedAliasOwner := io.clientPackedWrite(writeChoice).aliasOwner
    writeExpectedAliasGeneration :=
      io.clientPackedWrite(writeChoice).aliasGeneration
  }

  // Selection locks only owner and identity.  The selected producer owns the
  // wide payload until the same-cycle north/south fire.  If it violates the
  // Stream contract, valid is suppressed before the backend can accept its
  // withdrawn or mutated bus.  A retained fault response has priority over
  // the deferred ready terminal.
  io.packedWrite.valid := writeCommandHeldReg &&
    !io.packedWriteFault.valid && lockedWriteSourceValid &&
    lockedWriteSourceIdentityMatches
  io.packedWrite.payload :=
    io.clientPackedWrite(writeOwnerIndex).payload

  for (client <- 0 until clientCount) {
    io.clientPackedWrite(client).ready := False
    when(writeCommandHeldReg && writeOwnerIndex === client &&
        lockedWriteSourceValid && lockedWriteSourceIdentityMatches &&
        !io.packedWriteFault.valid) {
      // The selected client is accepted iff the backend accepts the exact
      // same payload in this cycle; selection alone never raises ready.
      io.clientPackedWrite(client).ready :=
        io.packedWrite.ready && io.packedWrite.valid
    }
  }

  val writeCommandFire = io.packedWrite.fire
  when(writeCommandFire) {
    writeCommandHeldReg := False
    writePostFireFaultWindowReg := True
    writeRoundRobin := nextClient(writeOwnerIndex)
    completedWritesReg := completedWritesReg + 1
  }

  // With the production semantic frontend, an immediate-reject fault is
  // registered in the cycle directly after fire.  Absence of a fault in that
  // complete window makes the committed fire terminal.
  when(writePostFireFaultWindowReg && !io.packedWriteFault.valid) {
    writePostFireFaultWindowReg := False
  }

  io.packedWriteFault.ready := unexpectedWriteFaultNow
  for (client <- 0 until clientCount) {
    io.clientPackedWriteFault(client).valid := False
    io.clientPackedWriteFault(client).payload := io.packedWriteFault.payload
    when(writeFaultIdentityNow) {
      io.clientPackedWriteFault(client).faultCode :=
        io.packedWriteFault.faultCode |
          B(BitNetUnifiedResidentTensorSemanticFrontendFault
            .BackendProtocolMask, 4 bits)
    }
    when(writeFaultExpected && writeOwnerIndex === client) {
      io.clientPackedWriteFault(client).valid := io.packedWriteFault.valid
      io.packedWriteFault.ready :=
        io.clientPackedWriteFault(client).ready
    }
  }

  val writeFaultFire = io.packedWriteFault.fire && writeFaultExpected
  when(writeFaultFire) {
    writeCommandHeldReg := False
    writePostFireFaultWindowReg := False
    writeRoundRobin := nextClient(writeOwnerIndex)
    completedWriteFaultsReg := completedWriteFaultsReg + 1
  }

  // -------------------------------------------------------------------
  // Same-cycle fail-stop for future selections.  A selected read which has
  // not reached the frontend is cancelled.  A write-source violation also
  // cancels its unaccepted lock: there is deliberately no private payload to
  // forward after the producer broke Stream stability.  Promised return and
  // post-fire fault paths remain live so accepted transactions can drain.
  // -------------------------------------------------------------------
  when(protocolFaultNow) {
    quarantineReg := True
    when(readCommandHeldReg && !readIssueFire) {
      readCommandHeldReg := False
    }
    when(writeCommandHeldReg && !writeCommandFire &&
        (writeSourceDroppedNow || writeSourceIdentityNow)) {
      writeCommandHeldReg := False
    }
  }
  when(unexpectedReadDataNow) {
    faultBitsReg(
      BitNetUnifiedPackedSemanticRouterFault.UnexpectedReadData) := True
  }
  when(readDataIdentityNow) {
    faultBitsReg(
      BitNetUnifiedPackedSemanticRouterFault.ReadDataIdentity) := True
  }
  when(unexpectedWriteFaultNow) {
    faultBitsReg(
      BitNetUnifiedPackedSemanticRouterFault.UnexpectedWriteFault) := True
  }
  when(writeFaultIdentityNow) {
    faultBitsReg(
      BitNetUnifiedPackedSemanticRouterFault.WriteFaultIdentity) := True
  }
  when(readSourceDroppedNow) {
    faultBitsReg(
      BitNetUnifiedPackedSemanticRouterFault.ReadSourceDropped) := True
  }
  when(readSourceIdentityNow) {
    faultBitsReg(
      BitNetUnifiedPackedSemanticRouterFault.ReadSourceIdentity) := True
  }
  when(writeSourceDroppedNow) {
    faultBitsReg(
      BitNetUnifiedPackedSemanticRouterFault.WriteSourceDropped) := True
  }
  when(writeSourceIdentityNow) {
    faultBitsReg(
      BitNetUnifiedPackedSemanticRouterFault.WriteSourceIdentity) := True
  }

  io.readOwnerValid := readCommandHeldReg || readResponseOutstandingReg
  io.readOwner := enumFromIndex(readOwnerIndex)
  io.readCommandHeld := readCommandHeldReg
  io.readResponseOutstanding := readResponseOutstandingReg
  io.writeOwnerValid := writeCommandHeldReg ||
    writePostFireFaultWindowReg
  io.writeOwner := enumFromIndex(writeOwnerIndex)
  io.writeCommandHeld := writeCommandHeldReg
  io.busy := readCommandHeldReg || readResponseOutstandingReg ||
    writeCommandHeldReg || writePostFireFaultWindowReg
  io.quarantined := quarantineReg || protocolFaultNow
  io.protocolFaultSticky := faultBitsReg.orR || protocolFaultNow
  io.faultBits := faultBitsReg
  io.acceptedReads := acceptedReadsReg
  io.completedReadResponses := completedReadResponsesReg
  io.completedWrites := completedWritesReg
  io.completedWriteFaults := completedWriteFaultsReg
}

object GenerateBitNetUnifiedPackedSemanticClientRouter extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-unified-packed-semantic-client-router")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetUnifiedPackedSemanticClientRouter())
}
