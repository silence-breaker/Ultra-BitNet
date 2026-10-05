package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentAuxReadOwner {
  val LayerRmsGamma = 0
  val ProjectionScale = 1
  val PostGateGamma = 2
  val Rope = 3
  val Count = 4
}

/**
  * Burst-granular owner mux for the production resident AUX read port.
  *
  * Layer RMS gamma, projection scale, post-gate gamma and RoPE coefficients
  * are independent logical clients, but all four images live behind the one
  * `rmsAuxRead` client of [[BitNetProductionPlMemoryOwnershipHub]].  This
  * component admits exactly one request and then retains that logical owner
  * until the declared final response beat is accepted by the same client.
  * No RAM, arithmetic core or TP4 resource is instantiated here.
  *
  * Every returned beat is checked against request-time bank, transaction and
  * tag metadata, a monotonically increasing beat index, the declared burst
  * length, and the redundant RRESP/fault encoding.  A malformed or ownerless
  * beat is drained without being exposed to any client and places the mux in
  * reset-only quarantine.  A well-formed AXI error beat is exposed to the
  * owning client so that its job can terminate deterministically; ownership
  * is still retained through the real final beat and the mux quarantines at
  * that boundary.  Consequently a stale response can never be relabelled as
  * another logical client's response.
  */
class BitNetResidentAuxReadOwnerMux(
    cfg: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    auxBank: Int = 0) extends Component {
  require(auxBank >= 0 && auxBank < cfg.bankCount)

  val io = new Bundle {
    val layerRmsGammaRequest = slave(Stream(
      BitNetClientBankReadRequest(cfg)))
    val layerRmsGammaResponse = master(Stream(
      BitNetClientBankReadResponse(cfg)))
    val projectionScaleRequest = slave(Stream(
      BitNetClientBankReadRequest(cfg)))
    val projectionScaleResponse = master(Stream(
      BitNetClientBankReadResponse(cfg)))
    val postGateGammaRequest = slave(Stream(
      BitNetClientBankReadRequest(cfg)))
    val postGateGammaResponse = master(Stream(
      BitNetClientBankReadResponse(cfg)))
    val ropeRequest = slave(Stream(
      BitNetClientBankReadRequest(cfg)))
    val ropeResponse = master(Stream(
      BitNetClientBankReadResponse(cfg)))

    val downstreamRequest = master(Stream(
      BitNetClientBankReadRequest(cfg)))
    val downstreamResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg)))

    val busy = out Bool()
    val ownerValid = out Bool()
    val activeOwner = out UInt(2 bits)
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedBursts = out UInt(32 bits)
    val completedBursts = out UInt(32 bits)
  }

  val ownerValid = RegInit(False)
  val commandPending = RegInit(False)
  val activeOwner = Reg(UInt(2 bits)) init 0
  val roundRobinStart = Reg(UInt(2 bits)) init 0
  val quarantine = RegInit(False)
  val protocolFault = RegInit(False)
  val transactionFaultSeen = RegInit(False)

  val heldBank = Reg(UInt(cfg.bankWidth bits)) init 0
  val heldTransactionId = Reg(UInt(cfg.transactionIdWidth bits)) init 0
  val heldTag = Reg(UInt(cfg.tagWidth bits)) init 0
  val heldLocalAddress = Reg(UInt(cfg.localAddressWidth bits)) init 0
  val heldBeats = Reg(UInt(cfg.beatCountWidth bits)) init 0
  val expectedBeat = Reg(UInt(cfg.beatIndexWidth bits)) init 0

  val acceptedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0

  val requestValids = Bits(BitNetResidentAuxReadOwner.Count bits)
  requestValids(0) := io.layerRmsGammaRequest.valid
  requestValids(1) := io.projectionScaleRequest.valid
  requestValids(2) := io.postGateGammaRequest.valid
  requestValids(3) := io.ropeRequest.valid

  /* Four-entry rotating priority encoder.  The selected client command is
   * captured before it is exposed downstream, so physical backpressure cannot
   * change a valid Stream payload. */
  val selectedValid = Bool()
  val selectedOwner = UInt(2 bits)
  selectedValid := requestValids.orR
  switch(roundRobinStart) {
    is(0) {
      when(requestValids(0)) { selectedOwner := 0 }
        .elsewhen(requestValids(1)) { selectedOwner := 1 }
        .elsewhen(requestValids(2)) { selectedOwner := 2 }
        .otherwise { selectedOwner := 3 }
    }
    is(1) {
      when(requestValids(1)) { selectedOwner := 1 }
        .elsewhen(requestValids(2)) { selectedOwner := 2 }
        .elsewhen(requestValids(3)) { selectedOwner := 3 }
        .otherwise { selectedOwner := 0 }
    }
    is(2) {
      when(requestValids(2)) { selectedOwner := 2 }
        .elsewhen(requestValids(3)) { selectedOwner := 3 }
        .elsewhen(requestValids(0)) { selectedOwner := 0 }
        .otherwise { selectedOwner := 1 }
    }
    default {
      when(requestValids(3)) { selectedOwner := 3 }
        .elsewhen(requestValids(0)) { selectedOwner := 0 }
        .elsewhen(requestValids(1)) { selectedOwner := 1 }
        .otherwise { selectedOwner := 2 }
    }
  }

  val selectedRequest = BitNetClientBankReadRequest(cfg)
  selectedRequest := io.layerRmsGammaRequest.payload
  switch(selectedOwner) {
    is(BitNetResidentAuxReadOwner.ProjectionScale) {
      selectedRequest := io.projectionScaleRequest.payload
    }
    is(BitNetResidentAuxReadOwner.PostGateGamma) {
      selectedRequest := io.postGateGammaRequest.payload
    }
    is(BitNetResidentAuxReadOwner.Rope) {
      selectedRequest := io.ropeRequest.payload
    }
  }

  val selectedRequestLegal =
    selectedRequest.bank === auxBank &&
      selectedRequest.beats =/= 0 &&
      selectedRequest.beats <= cfg.maxBurstBeats
  // An ownerless response has priority over admitting a new transaction.  It
  // is stale by construction and must be quarantined before any client can be
  // selected for the same cycle.
  val canSelectRequest = !ownerValid && !quarantine &&
    !io.downstreamResponse.valid

  // The selected client command is first captured into the transaction owner
  // registers.  This one-entry command holding stage keeps the downstream
  // Stream payload stable for arbitrary AR-side backpressure.
  io.downstreamRequest.valid := ownerValid && commandPending && !quarantine
  io.downstreamRequest.payload.bank := heldBank
  io.downstreamRequest.payload.transactionId := heldTransactionId
  io.downstreamRequest.payload.tag := heldTag
  io.downstreamRequest.payload.localAddress := heldLocalAddress
  io.downstreamRequest.payload.beats := heldBeats

  io.layerRmsGammaRequest.ready := False
  io.projectionScaleRequest.ready := False
  io.postGateGammaRequest.ready := False
  io.ropeRequest.ready := False
  val selectedReady = canSelectRequest
  when(selectedValid) {
    switch(selectedOwner) {
      is(BitNetResidentAuxReadOwner.LayerRmsGamma) {
        io.layerRmsGammaRequest.ready := selectedReady
      }
      is(BitNetResidentAuxReadOwner.ProjectionScale) {
        io.projectionScaleRequest.ready := selectedReady
      }
      is(BitNetResidentAuxReadOwner.PostGateGamma) {
        io.postGateGammaRequest.ready := selectedReady
      }
      default {
        io.ropeRequest.ready := selectedReady
      }
    }
  }

  val selectedInputFire =
    (io.layerRmsGammaRequest.fire || io.projectionScaleRequest.fire ||
      io.postGateGammaRequest.fire || io.ropeRequest.fire)
  when(selectedInputFire && !selectedRequestLegal) {
    quarantine := True
    protocolFault := True
  }

  when(selectedInputFire && selectedRequestLegal) {
    ownerValid := True
    commandPending := True
    activeOwner := selectedOwner
    heldBank := selectedRequest.bank
    heldTransactionId := selectedRequest.transactionId
    heldTag := selectedRequest.tag
    heldLocalAddress := selectedRequest.localAddress
    heldBeats := selectedRequest.beats
    expectedBeat := 0
    transactionFaultSeen := False
    roundRobinStart := selectedOwner + 1
    acceptedCount := acceptedCount + 1
  }

  when(io.downstreamRequest.fire) {
    commandPending := False
  }

  val expectedLast =
    expectedBeat.resize(cfg.beatCountWidth) === heldBeats - 1
  val responseIdentityLegal = ownerValid && !commandPending &&
    io.downstreamResponse.payload.bank === heldBank &&
    io.downstreamResponse.payload.transactionId === heldTransactionId &&
    io.downstreamResponse.payload.tag === heldTag &&
    io.downstreamResponse.payload.beatIndex === expectedBeat &&
    io.downstreamResponse.payload.last === expectedLast
  val responseStatusEncodingLegal =
    io.downstreamResponse.payload.fault ===
      (io.downstreamResponse.payload.rresp =/= B"2'b00")
  val responseForwardable = !quarantine && responseIdentityLegal &&
    responseStatusEncodingLegal

  io.layerRmsGammaResponse.valid := io.downstreamResponse.valid &&
    responseForwardable &&
    activeOwner === BitNetResidentAuxReadOwner.LayerRmsGamma
  io.projectionScaleResponse.valid := io.downstreamResponse.valid &&
    responseForwardable &&
    activeOwner === BitNetResidentAuxReadOwner.ProjectionScale
  io.postGateGammaResponse.valid := io.downstreamResponse.valid &&
    responseForwardable &&
    activeOwner === BitNetResidentAuxReadOwner.PostGateGamma
  io.ropeResponse.valid := io.downstreamResponse.valid &&
    responseForwardable &&
    activeOwner === BitNetResidentAuxReadOwner.Rope
  io.layerRmsGammaResponse.payload := io.downstreamResponse.payload
  io.projectionScaleResponse.payload := io.downstreamResponse.payload
  io.postGateGammaResponse.payload := io.downstreamResponse.payload
  io.ropeResponse.payload := io.downstreamResponse.payload

  val activeClientReady = Bool()
  switch(activeOwner) {
    is(BitNetResidentAuxReadOwner.LayerRmsGamma) {
      activeClientReady := io.layerRmsGammaResponse.ready
    }
    is(BitNetResidentAuxReadOwner.ProjectionScale) {
      activeClientReady := io.projectionScaleResponse.ready
    }
    is(BitNetResidentAuxReadOwner.PostGateGamma) {
      activeClientReady := io.postGateGammaResponse.ready
    }
    default {
      activeClientReady := io.ropeResponse.ready
    }
  }

  // Invalid/ownerless beats are consumed locally and can never leak to a
  // client.  Valid beats inherit only the locked owner's backpressure.
  io.downstreamResponse.ready := quarantine || !ownerValid ||
    (ownerValid && !responseForwardable) ||
    (responseForwardable && activeClientReady)

  when(io.downstreamResponse.fire) {
    when(!ownerValid || !responseIdentityLegal ||
        !responseStatusEncodingLegal) {
      ownerValid := False
      commandPending := False
      quarantine := True
      protocolFault := True
    } otherwise {
      val responseHasError = io.downstreamResponse.payload.fault ||
        io.downstreamResponse.payload.rresp =/= B"2'b00"
      when(responseHasError) {
        transactionFaultSeen := True
        protocolFault := True
      }
      when(expectedLast) {
        ownerValid := False
        commandPending := False
        completedCount := completedCount + 1
        when(transactionFaultSeen || responseHasError) {
          quarantine := True
        }
      } otherwise {
        expectedBeat := expectedBeat + 1
      }
    }
  }

  io.busy := ownerValid || quarantine
  io.ownerValid := ownerValid
  io.activeOwner := activeOwner
  io.quarantined := quarantine
  io.protocolFaultSticky := protocolFault
  io.acceptedBursts := acceptedCount
  io.completedBursts := completedCount
}

object GenerateBitNetResidentAuxReadOwnerMux extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-aux-read-owner-mux")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentAuxReadOwnerMux())
}
