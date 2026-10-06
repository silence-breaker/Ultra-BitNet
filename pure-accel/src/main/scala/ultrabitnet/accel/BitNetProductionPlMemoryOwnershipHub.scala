package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Stable client ownership of the only production five-bank crossbar. */
object BitNetProductionMemoryClient {
  final val MatrixRead = 0
  final val LmHeadRead = 1
  final val RmsAuxRead = 2
  final val KvRead = 3
  final val KvWrite = 4
  /** Low-priority PS-to-PL resident-weight staging owner. */
  final val WeightStage = 5
  /** Stable ABI alias retained for the existing reserved* top-level ports. */
  final val Reserved = WeightStage
  final val Count = 6
}

case class BitNetProductionPlMemoryOwnershipHubConfig(
    memory: BitNetFiveBankAxiMemoryComplexConfig =
      BitNetFiveBankAxiMemoryComplexConfig(),
    stagingReadComputeGrantQuota: Int = 8,
    stagingWriteComputeGrantQuota: Int = 8) {
  require(memory.crossbar.bankCount == 5)
  require(memory.crossbar.clientCount == BitNetProductionMemoryClient.Count,
    "the production ownership ABI is exactly six clients")
  require(stagingReadComputeGrantQuota >= 1,
    "read staging progress quota must be positive")
  require(stagingWriteComputeGrantQuota >= 1,
    "write staging progress quota must be positive")
}

/**
  * Fail-closed boundary for one fixed read client.
  *
  * An illegal bank is completed locally with DECERR and is never presented to
  * the shared crossbar.  A malformed bank identity returning from the memory
  * complex is still drained through this fixed port, but is forced to DECERR
  * so it cannot be mistaken for valid resident data.
  */
private[accel] class BitNetFixedReadClientGuard(
    cfg: BitNetBankLocalCrossbarConfig,
    allowedBanks: Vector[Int]) extends Component {
  require(allowedBanks.nonEmpty)
  require(allowedBanks.distinct == allowedBanks)
  require(allowedBanks.forall(bank => bank >= 0 && bank < cfg.bankCount))

  val io = new Bundle {
    val request = slave(Stream(BitNetClientBankReadRequest(cfg)))
    val response = master(Stream(BitNetClientBankReadResponse(cfg)))
    val downstreamRequest = master(Stream(
      BitNetClientBankReadRequest(cfg)))
    val downstreamResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
  }

  def bankAllowed(bank: UInt): Bool =
    allowedBanks.map(candidate => bank === candidate).reduce(_ || _)

  val rejectValid = RegInit(False)
  val rejectPayload = Reg(BitNetClientBankReadResponse(cfg)) init
    BitNetClientBankReadResponse(cfg).getZero
  val faultSticky = RegInit(False)

  val requestLegal = bankAllowed(io.request.payload.bank)
  io.downstreamRequest.valid := io.request.valid && requestLegal && !rejectValid
  io.downstreamRequest.payload := io.request.payload
  io.request.ready := !rejectValid && Mux(requestLegal,
    io.downstreamRequest.ready, True)

  when(io.request.fire && !requestLegal) {
    rejectValid := True
    rejectPayload.bank := io.request.payload.bank
    rejectPayload.transactionId := io.request.payload.transactionId
    rejectPayload.tag := io.request.payload.tag
    rejectPayload.beatIndex := 0
    rejectPayload.data := 0
    rejectPayload.last := True
    rejectPayload.rresp := B"2'b11"
    rejectPayload.fault := True
    faultSticky := True
  }

  val returnedBankLegal = bankAllowed(io.downstreamResponse.payload.bank)
  io.response.valid := rejectValid ||
    (io.downstreamResponse.valid && !rejectValid)
  io.response.payload := io.downstreamResponse.payload
  when(rejectValid) {
    io.response.payload := rejectPayload
  } elsewhen(io.downstreamResponse.valid && !returnedBankLegal) {
    io.response.payload.rresp := B"2'b11"
    io.response.payload.fault := True
  }
  io.downstreamResponse.ready := !rejectValid && io.response.ready

  when(rejectValid && io.response.fire) { rejectValid := False }
  when(io.downstreamResponse.fire && !returnedBankLegal) {
    faultSticky := True
  }

  io.busy := rejectValid || io.downstreamRequest.valid ||
    io.downstreamResponse.valid
  io.protocolFaultSticky := faultSticky
}

/**
  * Fail-closed boundary for one fixed write client.
  *
  * Illegal commands never reach AW.  Their declared data stream is consumed
  * locally up to `last`/the declared beat count, then one DECERR is returned.
  * This keeps a bad producer from wedging unrelated bank owners.
  */
private[accel] class BitNetFixedWriteClientGuard(
    cfg: BitNetBankLocalCrossbarConfig,
    allowedBanks: Vector[Int]) extends Component {
  require(allowedBanks.nonEmpty)
  require(allowedBanks.distinct == allowedBanks)
  require(allowedBanks.forall(bank => bank >= 0 && bank < cfg.bankCount))

  val io = new Bundle {
    val request = slave(Stream(BitNetClientBankWriteRequest(cfg)))
    val data = slave(Stream(BitNetClientBankWriteData(cfg)))
    val response = master(Stream(BitNetClientBankWriteResponse(cfg)))
    val downstreamRequest = master(Stream(
      BitNetClientBankWriteRequest(cfg)))
    val downstreamData = master(Stream(BitNetClientBankWriteData(cfg)))
    val downstreamResponse = slave(Stream(
      BitNetClientBankWriteResponse(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
  }

  def bankAllowed(bank: UInt): Bool =
    allowedBanks.map(candidate => bank === candidate).reduce(_ || _)

  val rejectDrain = RegInit(False)
  val rejectResponseValid = RegInit(False)
  val rejectRemaining = Reg(UInt(cfg.beatCountWidth bits)) init 0
  val rejectPayload = Reg(BitNetClientBankWriteResponse(cfg)) init
    BitNetClientBankWriteResponse(cfg).getZero
  val faultSticky = RegInit(False)
  // Serialize this fixed write owner until its B response is consumed.  The
  // packed-KV writer is already command/response sequential, and serialization
  // prevents an invalid later command from stealing an earlier burst's W data.
  val validOutstanding = RegInit(False)
  val activeBank = Reg(UInt(cfg.bankWidth bits)) init 0
  val activeTransaction = Reg(UInt(cfg.transactionIdWidth bits)) init 0
  val activeTag = Reg(UInt(cfg.tagWidth bits)) init 0
  val activeBeat = Reg(UInt(cfg.beatIndexWidth bits)) init 0
  val activeRemaining = Reg(UInt(cfg.beatCountWidth bits)) init 0
  val activeDataFault = RegInit(False)
  val abortDrain = RegInit(False)
  val abortRemaining = Reg(UInt(cfg.beatCountWidth bits)) init 0
  val rejecting = rejectDrain || rejectResponseValid

  val requestLegal = bankAllowed(io.request.payload.bank)
  io.downstreamRequest.valid := io.request.valid && requestLegal &&
    !rejecting && !validOutstanding
  io.downstreamRequest.payload := io.request.payload
  io.request.ready := !rejecting && !validOutstanding && Mux(requestLegal,
    io.downstreamRequest.ready, True)

  when(io.downstreamRequest.fire) {
    validOutstanding := True
    activeBank := io.downstreamRequest.payload.bank
    activeTransaction := io.downstreamRequest.payload.transactionId
    activeTag := io.downstreamRequest.payload.tag
    activeBeat := 0
    activeRemaining := io.downstreamRequest.payload.beats
    activeDataFault := False
    abortDrain := False
    abortRemaining := 0
  }

  when(io.request.fire && !requestLegal) {
    rejectPayload.bank := io.request.payload.bank
    rejectPayload.transactionId := io.request.payload.transactionId
    rejectPayload.tag := io.request.payload.tag
    rejectPayload.bresp := B"2'b11"
    rejectPayload.fault := True
    rejectRemaining := io.request.payload.beats
    rejectDrain := io.request.payload.beats =/= 0
    rejectResponseValid := io.request.payload.beats === 0
    faultSticky := True
  }

  val hasExpectedData = validOutstanding && activeRemaining =/= 0 &&
    !abortDrain
  val expectedLast = activeRemaining === 1
  val dataMetadataLegal = io.data.payload.bank === activeBank &&
    io.data.payload.transactionId === activeTransaction &&
    io.data.payload.tag === activeTag &&
    io.data.payload.beatIndex === activeBeat &&
    io.data.payload.last === expectedLast

  io.downstreamData.valid := io.data.valid && !rejecting && hasExpectedData
  io.downstreamData.payload := io.data.payload
  // A malformed beat is converted into a zero-strobe early logical last with
  // the *captured* command identity.  The bank adapter then pads every
  // advertised AXI beat, so AWLEN is honoured without exposing corrupt data.
  when(hasExpectedData && !dataMetadataLegal) {
    io.downstreamData.payload.bank := activeBank
    io.downstreamData.payload.transactionId := activeTransaction
    io.downstreamData.payload.tag := activeTag
    io.downstreamData.payload.beatIndex := activeBeat
    io.downstreamData.payload.data := 0
    io.downstreamData.payload.strobe := 0
    io.downstreamData.payload.last := True
  }

  io.data.ready := False
  when(rejectDrain) {
    io.data.ready := True
  } elsewhen(abortDrain) {
    // Drain the producer's residual logical packet locally while the physical
    // adapter completes its zero-strobe padding and waits for B.
    io.data.ready := True
  } elsewhen(!rejectResponseValid && hasExpectedData) {
    io.data.ready := io.downstreamData.ready
  } elsewhen(!rejecting && validOutstanding && activeRemaining === 0) {
    // Late surplus data never reaches the crossbar and cannot hold B hostage.
    io.data.ready := True
  }

  when(rejectDrain && io.data.fire) {
    rejectRemaining := rejectRemaining - 1
    when(io.data.payload.last || rejectRemaining === 1) {
      rejectDrain := False
      rejectResponseValid := True
    }
  }

  when(!rejecting && hasExpectedData && io.downstreamData.fire) {
    when(dataMetadataLegal) {
      activeRemaining := activeRemaining - 1
      activeBeat := activeBeat + 1
    } otherwise {
      activeDataFault := True
      faultSticky := True
      activeRemaining := 0
      abortRemaining := activeRemaining - 1
      abortDrain := !io.data.payload.last && activeRemaining > 1
    }
  }

  when(abortDrain && io.data.fire) {
    abortRemaining := abortRemaining - 1
    when(io.data.payload.last || abortRemaining === 1) {
      abortDrain := False
    }
  }

  when(!rejecting && validOutstanding && activeRemaining === 0 &&
    !abortDrain && io.data.fire) {
    activeDataFault := True
    faultSticky := True
  }

  val returnedBankLegal = bankAllowed(io.downstreamResponse.payload.bank)
  val returnedIdentityLegal = returnedBankLegal && validOutstanding &&
    io.downstreamResponse.payload.bank === activeBank &&
    io.downstreamResponse.payload.transactionId === activeTransaction &&
    io.downstreamResponse.payload.tag === activeTag
  io.response.valid := rejectResponseValid ||
    (io.downstreamResponse.valid && !rejectResponseValid && !abortDrain)
  io.response.payload := io.downstreamResponse.payload
  when(rejectResponseValid) {
    io.response.payload := rejectPayload
  } elsewhen(io.downstreamResponse.valid &&
    (!returnedIdentityLegal || activeDataFault)) {
    io.response.payload.bresp := B"2'b11"
    io.response.payload.fault := True
  }
  io.downstreamResponse.ready := !rejectResponseValid && !abortDrain &&
    io.response.ready

  when(rejectResponseValid && io.response.fire) {
    rejectResponseValid := False
  }
  when(io.downstreamResponse.fire &&
    (!returnedIdentityLegal || activeDataFault)) {
    faultSticky := True
  }
  when(io.downstreamResponse.fire) {
    validOutstanding := False
    activeRemaining := 0
    activeDataFault := False
  }
  when(io.data.valid && !rejecting && !validOutstanding) {
    faultSticky := True
  }

  io.busy := rejecting || validOutstanding || abortDrain ||
    io.downstreamRequest.valid ||
    io.downstreamData.valid || io.downstreamResponse.valid
  io.protocolFaultSticky := faultSticky
}

/**
  * Transparent, fail-stable admission gate for low-priority weight reads.
  *
  * A blocked request is backpressured at its source and is never accepted into
  * a private queue.  Once VALID has first been presented downstream,
  * `offeredLocked` deliberately ignores later blockers until that same request
  * fires.  The source Stream contract keeps its payload stable during this
  * interval, while input and downstream handshakes remain exactly identical.
  */
private[accel] class BitNetLowPriorityStagingReadAdmission(
    cfg: BitNetBankLocalCrossbarConfig) extends Component {
  val io = new Bundle {
    val request = slave(Stream(BitNetClientBankReadRequest(cfg)))
    val downstreamRequest = master(Stream(
      BitNetClientBankReadRequest(cfg)))
    val blockedBanks = in Bits(cfg.bankCount bits)
    val busy = out Bool()
  }

  val offeredLocked = RegInit(False)

  val requestBankBlocked = Bool()
  requestBankBlocked := True
  for (bank <- 0 until cfg.bankCount) {
    when(io.request.payload.bank === bank) {
      requestBankBlocked := io.blockedBanks(bank)
    }
  }

  val gateOpen = offeredLocked || !requestBankBlocked
  io.downstreamRequest.valid := io.request.valid && gateOpen
  io.downstreamRequest.payload := io.request.payload
  io.request.ready := io.downstreamRequest.ready && gateOpen

  when(io.downstreamRequest.valid && !io.downstreamRequest.ready) {
    offeredLocked := True
  }
  when(io.downstreamRequest.fire) {
    offeredLocked := False
  }

  io.busy := io.request.valid || offeredLocked
}

/** Write-command counterpart of the locked low-priority read admission. */
private[accel] class BitNetLowPriorityStagingWriteAdmission(
    cfg: BitNetBankLocalCrossbarConfig) extends Component {
  val io = new Bundle {
    val request = slave(Stream(BitNetClientBankWriteRequest(cfg)))
    val downstreamRequest = master(Stream(
      BitNetClientBankWriteRequest(cfg)))
    val blockedBanks = in Bits(cfg.bankCount bits)
    val busy = out Bool()
  }

  val offeredLocked = RegInit(False)

  val requestBankBlocked = Bool()
  requestBankBlocked := True
  for (bank <- 0 until cfg.bankCount) {
    when(io.request.payload.bank === bank) {
      requestBankBlocked := io.blockedBanks(bank)
    }
  }

  val gateOpen = offeredLocked || !requestBankBlocked
  io.downstreamRequest.valid := io.request.valid && gateOpen
  io.downstreamRequest.payload := io.request.payload
  io.request.ready := io.downstreamRequest.ready && gateOpen

  when(io.downstreamRequest.valid && !io.downstreamRequest.ready) {
    offeredLocked := True
  }
  when(io.downstreamRequest.fire) {
    offeredLocked := False
  }

  io.busy := io.request.valid || offeredLocked
}

/**
  * Per-bank bounded-progress state for one WeightStage command direction.
  *
  * Counters exist only while client 5 continuously holds a request for that
  * bank.  Resident compute therefore keeps its normal first-choice policy,
  * but after `computeGrantQuota` real crossbar admissions the selected bank
  * raises `forceAdmission` until exactly that staging request fires.  The
  * counter then clears, so the following staging command yields to a fresh
  * compute quota.  This component never owns a memory transaction: the
  * existing crossbar remains the sole read/write owner authority.
  */
private[accel] class BitNetBoundedStagingGrantQuota(
    cfg: BitNetBankLocalCrossbarConfig,
    computeGrantQuota: Int) extends Component {
  require(computeGrantQuota >= 1)

  private val countWidth = log2Up(computeGrantQuota + 1)

  val io = new Bundle {
    val stagingRequestValid = in Bool()
    val stagingRequestBank = in UInt(cfg.bankWidth bits)
    val stagingRequestFire = in Bool()
    val computeGrantBanks = in Bits(cfg.bankCount bits)
    val forceAdmissionBanks = out Bits(cfg.bankCount bits)
  }

  val consecutiveComputeGrants = Vec.fill(cfg.bankCount)(
    Reg(UInt(countWidth bits)) init 0)
  io.forceAdmissionBanks := 0

  for (bank <- 0 until cfg.bankCount) {
    val requestTargetsBank = io.stagingRequestValid &&
      io.stagingRequestBank === bank
    val quotaReached = consecutiveComputeGrants(bank) ===
      U(computeGrantQuota, countWidth bits)
    io.forceAdmissionBanks(bank) := requestTargetsBank && quotaReached

    when(!requestTargetsBank) {
      consecutiveComputeGrants(bank) := 0
    } elsewhen(io.stagingRequestFire) {
      consecutiveComputeGrants(bank) := 0
    } elsewhen(io.computeGrantBanks(bank) && !quotaReached) {
      consecutiveComputeGrants(bank) := consecutiveComputeGrants(bank) + 1
    }
  }
}

/**
  * Per-bank, backpressure-aware admission for the two multi-bank read owners.
  *
  * Each owner still occupies exactly one concentrator/crossbar client, so it
  * may issue at most one command per cycle.  Unlike feeding all five ports
  * directly into a no-lock concentrator, this scheduler never selects a bank
  * whose AXI command expander is known busy.  A selected port is locked only
  * if its actual Stream handshake stalls, which keeps payload stable without
  * letting an AR-stalled bank hide ready work in the other four banks.
  */
private[accel] class BitNetMatrixLmPerBankAdmission(
    cfg: BitNetBankLocalCrossbarConfig) extends Component {
  val io = new Bundle {
    val matrixRequest = Vec(slave(Stream(
      BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val lmRequest = Vec(slave(Stream(
      BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val matrixOutput = Vec(master(Stream(
      BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val lmOutput = Vec(master(Stream(
      BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val bankUnavailable = in Bits(cfg.bankCount bits)
    val protocolFaultSticky = out Bool()
  }

  val matrixNextBank = Reg(UInt(cfg.bankWidth bits)) init 0
  val lmNextBank = Reg(UInt(cfg.bankWidth bits)) init 0
  val matrixHeld = RegInit(False)
  val lmHeld = RegInit(False)
  val matrixHeldBank = Reg(UInt(cfg.bankWidth bits)) init 0
  val lmHeldBank = Reg(UInt(cfg.bankWidth bits)) init 0
  val preferLm = Vec.fill(cfg.bankCount)(RegInit(False))

  val matrixEligible = (0 until cfg.bankCount).map { bank =>
    io.matrixRequest(bank).valid && !io.bankUnavailable(bank)
  }
  val lmEligible = (0 until cfg.bankCount).map { bank =>
    io.lmRequest(bank).valid && !io.bankUnavailable(bank)
  }

  def choose(eligible: Seq[Bool], start: UInt,
             excludeValid: Bool, excludeBank: UInt): (Bool, UInt) = {
    val found = Bool()
    val selected = UInt(cfg.bankWidth bits)
    found := False
    selected := 0
    switch(start) {
      for (startBank <- 0 until cfg.bankCount) {
        is(startBank) {
          var seen: Bool = False
          for (offset <- 0 until cfg.bankCount) {
            val bank = (startBank + offset) % cfg.bankCount
            val candidate = eligible(bank) &&
              (!excludeValid || excludeBank =/= bank)
            when(candidate && !seen) {
              found := True
              selected := bank
            }
            seen = seen || candidate
          }
        }
      }
    }
    (found, selected)
  }

  val noExclude = False
  val zeroBank = U(0, cfg.bankWidth bits)
  val matrixFirst = choose(matrixEligible, matrixNextBank,
    Mux(lmHeld, True, noExclude), lmHeldBank)
  val lmFirst = choose(lmEligible, lmNextBank,
    Mux(matrixHeld, True, noExclude), matrixHeldBank)

  val firstCollision = !matrixHeld && !lmHeld && matrixFirst._1 &&
    lmFirst._1 && matrixFirst._2 === lmFirst._2
  val collisionPrefersLm = Bool()
  collisionPrefersLm := False
  for (bank <- 0 until cfg.bankCount) {
    when(matrixFirst._2 === bank) { collisionPrefersLm := preferLm(bank) }
  }
  val matrixAlternate = choose(matrixEligible, matrixNextBank,
    True, matrixFirst._2)
  val lmAlternate = choose(lmEligible, lmNextBank, True, lmFirst._2)

  val matrixSelectedValid = Bool()
  val matrixSelectedBank = UInt(cfg.bankWidth bits)
  val lmSelectedValid = Bool()
  val lmSelectedBank = UInt(cfg.bankWidth bits)
  matrixSelectedValid := Mux(matrixHeld, True, matrixFirst._1)
  matrixSelectedBank := Mux(matrixHeld, matrixHeldBank, matrixFirst._2)
  lmSelectedValid := Mux(lmHeld, True, lmFirst._1)
  lmSelectedBank := Mux(lmHeld, lmHeldBank, lmFirst._2)

  when(firstCollision) {
    when(collisionPrefersLm) {
      matrixSelectedValid := matrixAlternate._1
      matrixSelectedBank := matrixAlternate._2
    } otherwise {
      lmSelectedValid := lmAlternate._1
      lmSelectedBank := lmAlternate._2
    }
  }

  for (bank <- 0 until cfg.bankCount) {
    io.matrixRequest(bank).ready := False
    io.lmRequest(bank).ready := False
    io.matrixOutput(bank).valid := False
    io.matrixOutput(bank).payload := io.matrixRequest(bank).payload
    io.lmOutput(bank).valid := False
    io.lmOutput(bank).payload := io.lmRequest(bank).payload

    when(matrixSelectedValid && matrixSelectedBank === bank) {
      io.matrixOutput(bank).valid := io.matrixRequest(bank).valid
      io.matrixRequest(bank).ready := io.matrixOutput(bank).ready
    }
    when(lmSelectedValid && lmSelectedBank === bank) {
      io.lmOutput(bank).valid := io.lmRequest(bank).valid
      io.lmRequest(bank).ready := io.lmOutput(bank).ready
    }
  }

  val matrixSelectedFire = (0 until cfg.bankCount)
    .map(bank => io.matrixOutput(bank).fire).reduce(_ || _)
  val lmSelectedFire = (0 until cfg.bankCount)
    .map(bank => io.lmOutput(bank).fire).reduce(_ || _)

  def advance(bank: UInt): UInt = {
    val next = UInt(cfg.bankWidth bits)
    next := bank + 1
    when(bank === cfg.bankCount - 1) { next := 0 }
    next
  }

  when(matrixSelectedValid && !matrixSelectedFire &&
    io.matrixRequest(matrixSelectedBank).valid) {
    matrixHeld := True
    matrixHeldBank := matrixSelectedBank
  }
  when(lmSelectedValid && !lmSelectedFire &&
    io.lmRequest(lmSelectedBank).valid) {
    lmHeld := True
    lmHeldBank := lmSelectedBank
  }

  when(matrixSelectedFire) {
    matrixHeld := False
    matrixNextBank := advance(matrixSelectedBank)
  }
  when(lmSelectedFire) {
    lmHeld := False
    lmNextBank := advance(lmSelectedBank)
  }

  for (bank <- 0 until cfg.bankCount) {
    when(io.matrixOutput(bank).fire) {
      preferLm(bank) := True
    }
    when(io.lmOutput(bank).fire) {
      preferLm(bank) := False
    }
  }

  val sourceFault = RegInit(False)
  when(matrixHeld && !io.matrixRequest(matrixHeldBank).valid) {
    matrixHeld := False
    sourceFault := True
  }
  when(lmHeld && !io.lmRequest(lmHeldBank).valid) {
    lmHeld := False
    sourceFault := True
  }
  io.protocolFaultSticky := sourceFault
}

/**
  * Production PL-local ownership hub for all persistent model memory.
  *
  * There is exactly one six-client/five-bank crossbar and exactly one AXI
  * read plus one AXI write adapter per physical bank.  Matrix and LM retain
  * five bank-local request ports through the physical bank arbiters while
  * responses still use stable logical client IDs 0/1.  RMS AUX, packed-KV
  * read and packed-KV append are
  * fixed clients.  ABI client 5 is the low-priority WeightStage owner: its
  * reads yield to resident compute demand and its writes yield to KV append
  * before first presentation, subject to independent bounded-progress quotas.
  * A stalled VALID is never retracted and an accepted burst is never
  * preempted.  No command, activation or completion is routed through the
  * ARM/PS software.
  */
class BitNetProductionPlMemoryOwnershipHub(
    cfg: BitNetProductionPlMemoryOwnershipHubConfig =
      BitNetProductionPlMemoryOwnershipHubConfig()) extends Component {
  private val xcfg = cfg.memory.crossbar
  private val readCfg = cfg.memory.read(0)
  private val writeCfg = cfg.memory.write(0)
  private val allBanks = Vector.range(0, xcfg.bankCount)

  val io = new Bundle {
    val matrixReadRequest = Vec(slave(Stream(
      BitNetClientBankReadRequest(xcfg))), xcfg.bankCount)
    val matrixReadResponse = Vec(master(Stream(
      BitNetClientBankReadResponse(xcfg))), xcfg.bankCount)

    val lmReadRequest = Vec(slave(Stream(
      BitNetClientBankReadRequest(xcfg))), xcfg.bankCount)
    val lmReadResponse = Vec(master(Stream(
      BitNetClientBankReadResponse(xcfg))), xcfg.bankCount)

    val rmsAuxReadRequest = slave(Stream(
      BitNetClientBankReadRequest(xcfg)))
    val rmsAuxReadResponse = master(Stream(
      BitNetClientBankReadResponse(xcfg)))
    val kvReadRequest = slave(Stream(
      BitNetClientBankReadRequest(xcfg)))
    val kvReadResponse = master(Stream(
      BitNetClientBankReadResponse(xcfg)))
    val kvWriteRequest = slave(Stream(
      BitNetClientBankWriteRequest(xcfg)))
    val kvWriteData = slave(Stream(
      BitNetClientBankWriteData(xcfg)))
    val kvWriteResponse = master(Stream(
      BitNetClientBankWriteResponse(xcfg)))

    /** ABI-retained client-5 ports, assigned to low-priority WeightStage. */
    val reservedReadRequest = slave(Stream(
      BitNetClientBankReadRequest(xcfg)))
    val reservedReadResponse = master(Stream(
      BitNetClientBankReadResponse(xcfg)))
    val reservedWriteRequest = slave(Stream(
      BitNetClientBankWriteRequest(xcfg)))
    val reservedWriteData = slave(Stream(
      BitNetClientBankWriteData(xcfg)))
    val reservedWriteResponse = master(Stream(
      BitNetClientBankWriteResponse(xcfg)))

    val axiAr = Vec(master(Stream(BitNetAxiReadAddress(readCfg))),
      xcfg.bankCount)
    val axiR = Vec(slave(Stream(BitNetAxiReadData(readCfg))),
      xcfg.bankCount)
    val axiAw = Vec(master(Stream(BitNetAxiWriteAddress(writeCfg))),
      xcfg.bankCount)
    val axiW = Vec(master(Stream(BitNetAxiWriteData(writeCfg))),
      xcfg.bankCount)
    val axiB = Vec(slave(Stream(BitNetAxiWriteResponse(writeCfg))),
      xcfg.bankCount)

    val readOutstanding = Vec(out UInt(
      log2Up(cfg.memory.maxOutstanding + 1) bits), xcfg.bankCount)
    val writeOutstanding = Vec(out UInt(
      log2Up(cfg.memory.maxOutstanding + 1) bits), xcfg.bankCount)
    val memoryReadFaultSticky = out Bits(xcfg.bankCount bits)
    val memoryWriteFaultSticky = out Bits(xcfg.bankCount bits)
    val routingFaultSticky = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val memory = new BitNetFiveBankAxiMemoryComplex(cfg.memory)
  val matrix = new BitNetFiveBankDirectReadClientEdge(xcfg)
  val lm = new BitNetFiveBankDirectReadClientEdge(xcfg)
  val rmsAux = new BitNetFixedReadClientGuard(xcfg, Vector(0))
  val kvRead = new BitNetFixedReadClientGuard(xcfg, allBanks)
  val kvWrite = new BitNetFixedWriteClientGuard(xcfg, allBanks)
  val reservedRead = new BitNetFixedReadClientGuard(xcfg, allBanks)
  val reservedWrite = new BitNetFixedWriteClientGuard(xcfg, allBanks)
  val stagingReadAdmission =
    new BitNetLowPriorityStagingReadAdmission(xcfg)
  val stagingWriteAdmission =
    new BitNetLowPriorityStagingWriteAdmission(xcfg)
  val stagingReadQuota = new BitNetBoundedStagingGrantQuota(
    xcfg, cfg.stagingReadComputeGrantQuota)
  val stagingWriteQuota = new BitNetBoundedStagingGrantQuota(
    xcfg, cfg.stagingWriteComputeGrantQuota)

  // The LM fetcher owns one logical burst per physical bank.  Do not let the
  // fetcher's bankPhase state observe the global admission/crossbar/AXI
  // command-ready cone: accepting a request here transfers it losslessly into
  // a two-entry bank-local mailbox.  In particular, push ready depends only on
  // the registered tail occupancy, never on the current downstream ready.
  //
  // The head/tail form also supports one pop plus one replacement push in the
  // same cycle while preserving a stable head under downstream backpressure.
  // There is deliberately no abort/fail-stop flush; an accepted command must
  // continue through the memory complex and return its tagged completion.
  val lmIngressHeadValid = Vec.fill(xcfg.bankCount)(RegInit(False))
  val lmIngressTailValid = Vec.fill(xcfg.bankCount)(RegInit(False))
  val lmIngressHeadPayload = Vec.fill(xcfg.bankCount)(
    Reg(BitNetClientBankReadRequest(xcfg)))
  val lmIngressTailPayload = Vec.fill(xcfg.bankCount)(
    Reg(BitNetClientBankReadRequest(xcfg)))
  val lmIngress = Vec.fill(xcfg.bankCount)(
    Stream(BitNetClientBankReadRequest(xcfg)))

  for (bank <- 0 until xcfg.bankCount) {
    matrix.io.bankRequest(bank) << io.matrixReadRequest(bank)
    memory.io.directMatrixReadRequest(bank) <<
      matrix.io.directRequest(bank)
    io.matrixReadResponse(bank) << matrix.io.bankResponse(bank)

    val lmPush = io.lmReadRequest(bank)
    val lmPop = lmIngress(bank)
    lmPush.ready := !lmIngressTailValid(bank)
    lmPop.valid := lmIngressHeadValid(bank)
    lmPop.payload := lmIngressHeadPayload(bank)

    val lmPushFire = lmPush.fire
    val lmPopFire = lmPop.fire
    when(lmPushFire && lmPopFire) {
      // A legal simultaneous push/pop can only occur with an empty tail.
      // Replace the retiring head directly and keep one command available.
      lmIngressHeadValid(bank) := True
      lmIngressHeadPayload(bank) := lmPush.payload
    } elsewhen(lmPushFire) {
      when(!lmIngressHeadValid(bank)) {
        lmIngressHeadValid(bank) := True
        lmIngressHeadPayload(bank) := lmPush.payload
      } otherwise {
        lmIngressTailValid(bank) := True
        lmIngressTailPayload(bank) := lmPush.payload
      }
    } elsewhen(lmPopFire) {
      when(lmIngressTailValid(bank)) {
        lmIngressHeadValid(bank) := True
        lmIngressHeadPayload(bank) := lmIngressTailPayload(bank)
        lmIngressTailValid(bank) := False
      } otherwise {
        lmIngressHeadValid(bank) := False
      }
    }

    lm.io.bankRequest(bank) << lmPop
    memory.io.directLmReadRequest(bank) << lm.io.directRequest(bank)
    io.lmReadResponse(bank) << lm.io.bankResponse(bank)
  }

  memory.io.clientReadRequest(BitNetProductionMemoryClient.MatrixRead).valid :=
    False
  memory.io.clientReadRequest(BitNetProductionMemoryClient.MatrixRead).payload :=
    BitNetClientBankReadRequest(xcfg).getZero
  matrix.io.clientResponse <<
    memory.io.clientReadResponse(BitNetProductionMemoryClient.MatrixRead)
  memory.io.clientReadRequest(BitNetProductionMemoryClient.LmHeadRead).valid :=
    False
  memory.io.clientReadRequest(BitNetProductionMemoryClient.LmHeadRead).payload :=
    BitNetClientBankReadRequest(xcfg).getZero
  lm.io.clientResponse <<
    memory.io.clientReadResponse(BitNetProductionMemoryClient.LmHeadRead)

  rmsAux.io.request << io.rmsAuxReadRequest
  io.rmsAuxReadResponse << rmsAux.io.response
  memory.io.clientReadRequest(BitNetProductionMemoryClient.RmsAuxRead) <<
    rmsAux.io.downstreamRequest
  rmsAux.io.downstreamResponse <<
    memory.io.clientReadResponse(BitNetProductionMemoryClient.RmsAuxRead)

  // Break the packed-KV state -> admission/crossbar/AXI ready cone.  Accepted
  // commands remain lossless and ordered; the register changes first-command
  // latency by one cycle without reducing the one-command-per-cycle capacity.
  val kvReadIngress = io.kvReadRequest.m2sPipe()
  kvRead.io.request << kvReadIngress
  io.kvReadResponse << kvRead.io.response
  memory.io.clientReadRequest(BitNetProductionMemoryClient.KvRead) <<
    kvRead.io.downstreamRequest
  kvRead.io.downstreamResponse <<
    memory.io.clientReadResponse(BitNetProductionMemoryClient.KvRead)

  // Client 4 has a write owner only. Its unused read direction is sealed.
  memory.io.clientReadRequest(BitNetProductionMemoryClient.KvWrite).valid :=
    False
  memory.io.clientReadRequest(BitNetProductionMemoryClient.KvWrite).payload :=
    BitNetClientBankReadRequest(xcfg).getZero
  memory.io.clientReadResponse(BitNetProductionMemoryClient.KvWrite).ready :=
    True

  reservedRead.io.request << io.reservedReadRequest
  io.reservedReadResponse << reservedRead.io.response
  stagingReadAdmission.io.request << reservedRead.io.downstreamRequest
  memory.io.clientReadRequest(BitNetProductionMemoryClient.WeightStage) <<
    stagingReadAdmission.io.downstreamRequest
  reservedRead.io.downstreamResponse <<
    memory.io.clientReadResponse(BitNetProductionMemoryClient.WeightStage)

  // Clients 0..3 have no production write ownership.
  for (client <- 0 to BitNetProductionMemoryClient.KvRead) {
    memory.io.clientWriteRequest(client).valid := False
    memory.io.clientWriteRequest(client).payload :=
      BitNetClientBankWriteRequest(xcfg).getZero
    memory.io.clientWriteData(client).valid := False
    memory.io.clientWriteData(client).payload :=
      BitNetClientBankWriteData(xcfg).getZero
    memory.io.clientWriteResponse(client).ready := True
  }

  kvWrite.io.request << io.kvWriteRequest
  kvWrite.io.data << io.kvWriteData
  io.kvWriteResponse << kvWrite.io.response
  memory.io.clientWriteRequest(BitNetProductionMemoryClient.KvWrite) <<
    kvWrite.io.downstreamRequest
  memory.io.clientWriteData(BitNetProductionMemoryClient.KvWrite) <<
    kvWrite.io.downstreamData
  kvWrite.io.downstreamResponse <<
    memory.io.clientWriteResponse(BitNetProductionMemoryClient.KvWrite)

  reservedWrite.io.request << io.reservedWriteRequest
  reservedWrite.io.data << io.reservedWriteData
  io.reservedWriteResponse << reservedWrite.io.response
  stagingWriteAdmission.io.request << reservedWrite.io.downstreamRequest
  memory.io.clientWriteRequest(BitNetProductionMemoryClient.WeightStage) <<
    stagingWriteAdmission.io.downstreamRequest
  memory.io.clientWriteData(BitNetProductionMemoryClient.WeightStage) <<
    reservedWrite.io.downstreamData
  reservedWrite.io.downstreamResponse <<
    memory.io.clientWriteResponse(BitNetProductionMemoryClient.WeightStage)

  // Consume the adapter's registered command-expansion state directly.  It
  // covers the interval from a legal logical-command handshake through the
  // final 4-KiB-safe AR, so no second address validator or AR counter is
  // needed at this hierarchy.
  val readExpandBusy = memory.io.readCommandExpansionBusy
  val readSlotsFull = Bits(xcfg.bankCount bits)
  for (bank <- 0 until xcfg.bankCount) {
    readSlotsFull(bank) := memory.io.readOutstanding(bank) ===
      cfg.memory.maxOutstanding
  }

  // Client 5 normally waits until every real compute reader for the same bank
  // has gone idle.  A per-bank quota counts actual compute commands accepted
  // by the crossbar while the same staging request remains pending.  At the
  // threshold only that one request may ignore new compute demand; physical
  // expansion/full conditions remain hard blockers.  The admission gate then
  // keeps a first-presented stalled VALID fail-stable, and the crossbar retains
  // all transaction ownership.
  val computeReadDemand = Bits(xcfg.bankCount bits)
  val computeReadGrant = Bits(xcfg.bankCount bits)
  val stagingReadBlocked = Bits(xcfg.bankCount bits)
  for (bank <- 0 until xcfg.bankCount) {
    computeReadDemand(bank) :=
      (io.matrixReadRequest(bank).valid &&
        io.matrixReadRequest(bank).payload.bank === bank) ||
      (io.lmReadRequest(bank).valid &&
        io.lmReadRequest(bank).payload.bank === bank) ||
      (lmIngress(bank).valid &&
        lmIngress(bank).payload.bank === bank) ||
      (rmsAux.io.downstreamRequest.valid &&
        rmsAux.io.downstreamRequest.payload.bank === bank) ||
      (kvRead.io.downstreamRequest.valid &&
        kvRead.io.downstreamRequest.payload.bank === bank)
    computeReadGrant(bank) :=
      memory.io.directMatrixReadRequest(bank).fire ||
      memory.io.directLmReadRequest(bank).fire ||
      (BitNetProductionMemoryClient.RmsAuxRead to
      BitNetProductionMemoryClient.KvRead).map { client =>
      memory.io.clientReadRequest(client).fire &&
        memory.io.clientReadRequest(client).payload.bank === bank
    }.reduce(_ || _)
    stagingReadBlocked(bank) :=
      (computeReadDemand(bank) &&
        !stagingReadQuota.io.forceAdmissionBanks(bank)) ||
      readExpandBusy(bank) || readSlotsFull(bank)
  }
  stagingReadQuota.io.stagingRequestValid :=
    stagingReadAdmission.io.request.valid
  stagingReadQuota.io.stagingRequestBank :=
    stagingReadAdmission.io.request.payload.bank
  stagingReadQuota.io.stagingRequestFire :=
    stagingReadAdmission.io.downstreamRequest.fire
  stagingReadQuota.io.computeGrantBanks := computeReadGrant
  stagingReadAdmission.io.blockedBanks := stagingReadBlocked

  // Weight staging writes target PL-DDR bank 0 in production, while all five
  // ABI banks use the same safe rule.  KV append normally wins first
  // presentation.  After a bounded number of actual KV command grants, one
  // continuously pending staging command may contend; an outstanding bank
  // transaction remains a hard blocker.  Once either command fires, the
  // existing crossbar's per-bank write owner/data lock carries its whole burst
  // without preemption.
  val computeWriteGrant = Bits(xcfg.bankCount bits)
  val stagingWriteBlocked = Bits(xcfg.bankCount bits)
  for (bank <- 0 until xcfg.bankCount) {
    val computeWriteDemand = kvWrite.io.downstreamRequest.valid &&
      kvWrite.io.downstreamRequest.payload.bank === bank
    computeWriteGrant(bank) :=
      memory.io.clientWriteRequest(
        BitNetProductionMemoryClient.KvWrite).fire &&
      memory.io.clientWriteRequest(
        BitNetProductionMemoryClient.KvWrite).payload.bank === bank
    stagingWriteBlocked(bank) :=
      (computeWriteDemand &&
        !stagingWriteQuota.io.forceAdmissionBanks(bank)) ||
      memory.io.writeOutstanding(bank) =/= 0
  }
  stagingWriteQuota.io.stagingRequestValid :=
    stagingWriteAdmission.io.request.valid
  stagingWriteQuota.io.stagingRequestBank :=
    stagingWriteAdmission.io.request.payload.bank
  stagingWriteQuota.io.stagingRequestFire :=
    stagingWriteAdmission.io.downstreamRequest.fire
  stagingWriteQuota.io.computeGrantBanks := computeWriteGrant
  stagingWriteAdmission.io.blockedBanks := stagingWriteBlocked

  for (bank <- 0 until xcfg.bankCount) {
    io.axiAr(bank) << memory.io.axiAr(bank)
    memory.io.axiR(bank) << io.axiR(bank)
    io.axiAw(bank) << memory.io.axiAw(bank)
    io.axiW(bank) << memory.io.axiW(bank)
    memory.io.axiB(bank) << io.axiB(bank)
    io.readOutstanding(bank) := memory.io.readOutstanding(bank)
    io.writeOutstanding(bank) := memory.io.writeOutstanding(bank)
  }

  io.memoryReadFaultSticky := memory.io.readFaultSticky
  io.memoryWriteFaultSticky := memory.io.writeFaultSticky
  io.routingFaultSticky := memory.io.routingFaultSticky
  io.protocolFaultSticky := memory.io.protocolFaultSticky ||
    matrix.io.requestBankFaultSticky ||
    matrix.io.responseBankFaultSticky ||
    lm.io.requestBankFaultSticky ||
    lm.io.responseBankFaultSticky ||
    rmsAux.io.protocolFaultSticky ||
    kvRead.io.protocolFaultSticky ||
    kvWrite.io.protocolFaultSticky ||
    reservedRead.io.protocolFaultSticky ||
    reservedWrite.io.protocolFaultSticky
}

object GenerateBitNetProductionPlMemoryOwnershipHub extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-production-pl-memory-ownership-hub")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetProductionPlMemoryOwnershipHub())
}
