package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Collapse one subsystem's five bank-local read ports into one crossbar
  * client without collapsing the five physical memories.
  *
  * A subsystem such as the K-striped matrix prefetcher or the streaming LM
  * head naturally owns one request/response stream per bank.  The global
  * memory crossbar, however, identifies the subsystem as one logical client.
  * This adapter performs that shape conversion: at most one command is
  * admitted per cycle, while responses from all five independent banks are
  * demultiplexed concurrently by their echoed bank identity.
  *
  * A request presented on the wrong static bank port is consumed locally and
  * returned as one synthetic DECERR beat.  It is never allowed to address a
  * different physical bank.  An invalid response bank is drained and latched
  * as a protocol fault so a corrupt AXI/client tag cannot deadlock the fabric.
  */
class BitNetFiveBankReadClientConcentrator(
    cfg: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig()) extends Component {
  val io = new Bundle {
    val bankRequest = Vec(
      slave(Stream(BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val bankResponse = Vec(
      master(Stream(BitNetClientBankReadResponse(cfg))), cfg.bankCount)

    val clientRequest = master(Stream(
      BitNetClientBankReadRequest(cfg)))
    val clientResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg)))

    val busy = out Bool()
    val requestBankFaultSticky = out Bool()
    val responseBankFaultSticky = out Bool()
    val forwardedRequests = out UInt(32 bits)
    val returnedResponses = out UInt(32 bits)
    val rejectedRequests = out UInt(32 bits)
  }

  val legalCandidates = Array.tabulate(cfg.bankCount) { bank =>
    val candidate = Stream(BitNetClientBankReadRequest(cfg))
    candidate.valid := io.bankRequest(bank).valid &&
      io.bankRequest(bank).payload.bank === bank
    candidate.payload := io.bankRequest(bank).payload
    candidate
  }
  val selected = StreamArbiterFactory.roundRobin.noLock
    .on(Vec(legalCandidates))
  io.clientRequest.valid := selected.valid
  io.clientRequest.payload := selected.payload
  selected.ready := io.clientRequest.ready

  // One synthetic response slot is sufficient because malformed requests
  // are local protocol failures, not normal high-throughput traffic.
  val rejectValid = RegInit(False)
  val rejectPayload = Reg(BitNetClientBankReadResponse(cfg)) init
    BitNetClientBankReadResponse(cfg).getZero

  val badRequest = Bits(cfg.bankCount bits)
  for (bank <- 0 until cfg.bankCount) {
    badRequest(bank) := io.bankRequest(bank).valid &&
      io.bankRequest(bank).payload.bank =/= bank
  }
  val badFound = badRequest.orR
  val badIndex = UInt(cfg.bankWidth bits)
  badIndex := 0
  // Reversed assignment priority makes the lowest numbered bad port win.
  for (bank <- (0 until cfg.bankCount).reverse) {
    when(badRequest(bank)) { badIndex := bank }
  }

  for (bank <- 0 until cfg.bankCount) {
    io.bankRequest(bank).ready :=
      legalCandidates(bank).ready &&
        io.bankRequest(bank).payload.bank === bank
    when(!rejectValid && badFound && badIndex === bank) {
      io.bankRequest(bank).ready := True
    }
  }

  val rejectedFire = badFound && !rejectValid &&
    io.bankRequest(badIndex).fire
  when(rejectedFire) {
    rejectValid := True
    rejectPayload.bank := badIndex
    rejectPayload.transactionId :=
      io.bankRequest(badIndex).payload.transactionId
    rejectPayload.tag := io.bankRequest(badIndex).payload.tag
    rejectPayload.beatIndex := 0
    rejectPayload.data := 0
    rejectPayload.last := True
    rejectPayload.rresp := B"11"
    rejectPayload.fault := True
  }

  val responseBankValid = io.clientResponse.payload.bank < cfg.bankCount
  val rejectBank = rejectPayload.bank
  for (bank <- 0 until cfg.bankCount) {
    val rejectForBank = rejectValid && rejectBank === bank
    val realForBank = io.clientResponse.valid && responseBankValid &&
      io.clientResponse.payload.bank === bank

    io.bankResponse(bank).valid := rejectForBank ||
      (realForBank && !rejectForBank)
    io.bankResponse(bank).payload := io.clientResponse.payload
    when(rejectForBank) {
      io.bankResponse(bank).payload := rejectPayload
    }
  }

  io.clientResponse.ready := !responseBankValid
  for (bank <- 0 until cfg.bankCount) {
    when(responseBankValid && io.clientResponse.payload.bank === bank) {
      // A local DECERR for this same bank has priority for one cycle/hold.
      io.clientResponse.ready := io.bankResponse(bank).ready &&
        !(rejectValid && rejectBank === bank)
    }
  }

  val rejectReady = Bool()
  rejectReady := False
  for (bank <- 0 until cfg.bankCount) {
    when(rejectBank === bank) {
      rejectReady := io.bankResponse(bank).ready
    }
  }
  when(rejectValid && rejectReady) { rejectValid := False }

  val requestFault = RegInit(False)
  val responseFault = RegInit(False)
  val requestCount = Reg(UInt(32 bits)) init 0
  val responseCount = Reg(UInt(32 bits)) init 0
  val rejectedCount = Reg(UInt(32 bits)) init 0

  when(io.clientRequest.fire) { requestCount := requestCount + 1 }
  when(io.clientResponse.fire && responseBankValid) {
    responseCount := responseCount + 1
  }
  when(rejectedFire) {
    requestFault := True
    rejectedCount := rejectedCount + 1
  }
  when(io.clientResponse.fire && !responseBankValid) {
    responseFault := True
  }

  io.busy := rejectValid || io.clientRequest.valid || io.clientResponse.valid
  io.requestBankFaultSticky := requestFault
  io.responseBankFaultSticky := responseFault
  io.forwardedRequests := requestCount
  io.returnedResponses := responseCount
  io.rejectedRequests := rejectedCount
}

/**
  * Fail-closed edge that keeps legal requests bank-local all the way into the
  * physical bank arbiters.  Only malformed requests share a reject slot;
  * therefore a stalled legal bank can never hide work on another bank.
  */
class BitNetFiveBankDirectReadClientEdge(
    cfg: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig()) extends Component {
  val io = new Bundle {
    val bankRequest = Vec(
      slave(Stream(BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val directRequest = Vec(
      master(Stream(BitNetClientBankReadRequest(cfg))), cfg.bankCount)
    val bankResponse = Vec(
      master(Stream(BitNetClientBankReadResponse(cfg))), cfg.bankCount)
    val clientResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg)))
    val requestBankFaultSticky = out Bool()
    val responseBankFaultSticky = out Bool()
  }

  val rejectValid = RegInit(False)
  val rejectPayload = Reg(BitNetClientBankReadResponse(cfg)) init
    BitNetClientBankReadResponse(cfg).getZero

  val badRequest = Bits(cfg.bankCount bits)
  for (bank <- 0 until cfg.bankCount) {
    val legal = io.bankRequest(bank).payload.bank === bank
    io.directRequest(bank).valid := io.bankRequest(bank).valid && legal
    io.directRequest(bank).payload := io.bankRequest(bank).payload
    io.bankRequest(bank).ready := io.directRequest(bank).ready && legal
    badRequest(bank) := io.bankRequest(bank).valid && !legal
  }

  val badFound = badRequest.orR
  val badIndex = UInt(cfg.bankWidth bits)
  badIndex := 0
  for (bank <- (0 until cfg.bankCount).reverse) {
    when(badRequest(bank)) { badIndex := bank }
  }
  for (bank <- 0 until cfg.bankCount) {
    when(!rejectValid && badFound && badIndex === bank) {
      io.bankRequest(bank).ready := True
    }
  }

  val rejectedFire = badFound && !rejectValid &&
    io.bankRequest(badIndex).fire
  when(rejectedFire) {
    rejectValid := True
    rejectPayload.bank := badIndex
    rejectPayload.transactionId :=
      io.bankRequest(badIndex).payload.transactionId
    rejectPayload.tag := io.bankRequest(badIndex).payload.tag
    rejectPayload.beatIndex := 0
    rejectPayload.data := 0
    rejectPayload.last := True
    rejectPayload.rresp := B"11"
    rejectPayload.fault := True
  }

  val responseBankValid = io.clientResponse.payload.bank < cfg.bankCount
  val rejectBank = rejectPayload.bank
  for (bank <- 0 until cfg.bankCount) {
    val rejectForBank = rejectValid && rejectBank === bank
    val realForBank = io.clientResponse.valid && responseBankValid &&
      io.clientResponse.payload.bank === bank
    io.bankResponse(bank).valid := rejectForBank ||
      (realForBank && !rejectForBank)
    io.bankResponse(bank).payload := io.clientResponse.payload
    when(rejectForBank) {
      io.bankResponse(bank).payload := rejectPayload
    }
  }

  io.clientResponse.ready := !responseBankValid
  for (bank <- 0 until cfg.bankCount) {
    when(responseBankValid && io.clientResponse.payload.bank === bank) {
      io.clientResponse.ready := io.bankResponse(bank).ready &&
        !(rejectValid && rejectBank === bank)
    }
  }

  val rejectReady = Bool()
  rejectReady := False
  for (bank <- 0 until cfg.bankCount) {
    when(rejectBank === bank) {
      rejectReady := io.bankResponse(bank).ready
    }
  }
  when(rejectValid && rejectReady) { rejectValid := False }

  val requestFault = RegInit(False)
  val responseFault = RegInit(False)
  when(rejectedFire) { requestFault := True }
  when(io.clientResponse.fire && !responseBankValid) {
    responseFault := True
  }
  io.requestBankFaultSticky := requestFault
  io.responseBankFaultSticky := responseFault
}

object GenerateBitNetFiveBankReadClientConcentrator extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-five-bank-read-client-concentrator")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetFiveBankReadClientConcentrator())
}
