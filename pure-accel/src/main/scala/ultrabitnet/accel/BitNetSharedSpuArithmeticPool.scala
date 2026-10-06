package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Fixed client numbering for the single resident nonlinear arithmetic bank.
  *
  * QK score scaling is intentionally absent.  Its independent sixteen-DSP
  * lane bank is required to sustain one key per slow clock during attention.
  */
object BitNetSpuArithmeticClient {
  val ProjectionScale = 0
  val RmsNorm = 1
  val FfnGate = 2
  val DynamicQuant = 3
  val Rope = 4
  val HeadQuant = 5
  val Count = 6

  val Names: Seq[String] = Seq(
    "projection-scale",
    "rmsnorm",
    "ffn-gate",
    "dynamic-quant",
    "rope",
    "head-quant")
}

/** Physical geometry of the BitNet shared SPU multiplier bank.
  *
  * A 26-bit unsigned magnitude by an 18-bit unsigned limb maps to one
  * DSP48E2 on Zynq UltraScale+.  Eight such sites are sufficient for every
  * mutually exclusive nonlinear phase:
  *
  *   - four lanes x two limbs for RMSNorm and FFN gating;
  *   - four RoPE products x two magnitude limbs for one token lane;
  *   - four reciprocal limbs plus two threshold limbs for quantisation;
  *   - four factor limbs plus four raw-dot limbs for projection scaling.
  */
case class BitNetSharedSpuArithmeticPoolConfig(
    clientCount: Int = BitNetSpuArithmeticClient.Count,
    multiplierCount: Int = 8,
    operandAWidth: Int = 26,
    operandBWidth: Int = 18,
    tagWidth: Int = 20,
    epochWidth: Int = 8,
    sequenceWidth: Int = 16
) {
  require(clientCount == BitNetSpuArithmeticClient.Count,
    "the production lease table has six fixed clients")
  require(multiplierCount == 8,
    "the production nonlinear arithmetic ceiling is eight DSP48E2s")
  require(operandAWidth == 26 && operandBWidth == 18,
    "unsigned magnitudes must fit one signed DSP48E2 operand pair")
  require(tagWidth >= 1 && epochWidth >= 1 && sequenceWidth >= 1)

  val clientWidth: Int = Math.max(1, log2Up(clientCount))
  val productWidth: Int = operandAWidth + operandBWidth
}

/** One command-level ownership request.  A client keeps the lease across as
  * many limb-operation beats as its algorithm requires.
  */
case class BitNetSpuArithmeticLeaseRequest(
    cfg: BitNetSharedSpuArithmeticPoolConfig) extends Bundle {
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
}

/** Eight independent unsigned magnitude-limb products in one slow clock. */
case class BitNetSpuArithmeticOperandBeat(
    cfg: BitNetSharedSpuArithmeticPoolConfig) extends Bundle {
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val sequence = UInt(cfg.sequenceWidth bits)
  val active = Bits(cfg.multiplierCount bits)
  val operandA = Vec(UInt(cfg.operandAWidth bits), cfg.multiplierCount)
  val operandB = Vec(UInt(cfg.operandBWidth bits), cfg.multiplierCount)
  val first = Bool()
  val last = Bool()
}

case class BitNetSpuArithmeticResultBeat(
    cfg: BitNetSharedSpuArithmeticPoolConfig) extends Bundle {
  val owner = UInt(cfg.clientWidth bits)
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val sequence = UInt(cfg.sequenceWidth bits)
  val active = Bits(cfg.multiplierCount bits)
  val product = Vec(UInt(cfg.productWidth bits), cfg.multiplierCount)
  val first = Bool()
  val last = Bool()
}

/** Release is accepted only after every legal result has drained.  A stale
  * tag/epoch is consumed to avoid deadlock, but raises the sticky fault.
  */
case class BitNetSpuArithmeticLeaseRelease(
    cfg: BitNetSharedSpuArithmeticPoolConfig) extends Bundle {
  val tag = UInt(cfg.tagWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
}

/**
  * Command-exclusive, eight-DSP arithmetic lease for all resident SPU phases.
  *
  * The global layer sequencer already guarantees that RMSNorm, projection
  * scaling, RoPE/head quantisation, FFN gating and activation quantisation are
  * mutually exclusive.  This block turns that schedule invariant into one
  * physical multiplier bank instead of merely relying on mutually exclusive
  * clock enables across separately instantiated DSPs.
  *
  * Arbitration is deterministic, lowest client index first.  Once acquired,
  * ownership cannot change until a matching release is consumed after the
  * elastic result stage drains.  Operand beats with stale tag/epoch are
  * consumed without a result and set `protocolFaultSticky`; this fail-closed
  * rule prevents a delayed client from corrupting the next phase.  The legal
  * path accepts one eight-product beat per clock and remains stable under
  * arbitrary result backpressure.
  */
class BitNetSharedSpuArithmeticPool(
    cfg: BitNetSharedSpuArithmeticPoolConfig =
      BitNetSharedSpuArithmeticPoolConfig()) extends Component {
  val io = new Bundle {
    val acquire = Vec(
      slave(Stream(BitNetSpuArithmeticLeaseRequest(cfg))),
      cfg.clientCount)
    val operand = Vec(
      slave(Stream(BitNetSpuArithmeticOperandBeat(cfg))),
      cfg.clientCount)
    val result = Vec(
      master(Stream(BitNetSpuArithmeticResultBeat(cfg))),
      cfg.clientCount)
    val release = Vec(
      slave(Stream(BitNetSpuArithmeticLeaseRelease(cfg))),
      cfg.clientCount)

    val clearFault = in Bool()
    val busy = out Bool()
    val owned = out Bits(cfg.clientCount bits)
    val activeOwner = out UInt(cfg.clientWidth bits)
    val activeTag = out UInt(cfg.tagWidth bits)
    val activeEpoch = out UInt(cfg.epochWidth bits)
    val protocolFaultSticky = out Bool()
    val acceptedLeases = out UInt(32 bits)
    val acceptedOperations = out UInt(32 bits)
    val emittedResults = out UInt(32 bits)
    val releasedLeases = out UInt(32 bits)
    val droppedOperations = out UInt(32 bits)
  }

  val leaseActive = RegInit(False)
  val ownerReg = Reg(UInt(cfg.clientWidth bits)) init 0
  val tagReg = Reg(UInt(cfg.tagWidth bits)) init 0
  val epochReg = Reg(UInt(cfg.epochWidth bits)) init 0
  val fault = RegInit(False)
  val acceptedLeaseCount = Reg(UInt(32 bits)) init 0
  val acceptedOperationCount = Reg(UInt(32 bits)) init 0
  val emittedResultCount = Reg(UInt(32 bits)) init 0
  val releasedLeaseCount = Reg(UInt(32 bits)) init 0
  val droppedOperationCount = Reg(UInt(32 bits)) init 0

  when(io.clearFault) { fault := False }

  /* Lowest-numbered request wins only while the bank is unowned. */
  val acquireFound = Bool()
  val acquireSelect = UInt(cfg.clientWidth bits)
  acquireFound := False
  acquireSelect := 0
  for (client <- (0 until cfg.clientCount).reverse) {
    when(io.acquire(client).valid) {
      acquireFound := True
      acquireSelect := client
    }
  }
  for (client <- 0 until cfg.clientCount) {
    io.acquire(client).ready := !leaseActive && acquireFound &&
      acquireSelect === client
    when(io.acquire(client).fire) {
      leaseActive := True
      ownerReg := client
      tagReg := io.acquire(client).tag
      epochReg := io.acquire(client).epoch
      acceptedLeaseCount := acceptedLeaseCount + 1
    }
  }

  /* Select only the active owner's operand stream. */
  val selectedOperand = Stream(BitNetSpuArithmeticOperandBeat(cfg))
  selectedOperand.valid := False
  selectedOperand.payload := BitNetSpuArithmeticOperandBeat(cfg).getZero
  for (client <- 0 until cfg.clientCount) {
    io.operand(client).ready := False
    when(leaseActive && ownerReg === client) {
      selectedOperand.valid := io.operand(client).valid
      selectedOperand.payload := io.operand(client).payload
      io.operand(client).ready := selectedOperand.ready
    }
  }

  val operationLegal = selectedOperand.tag === tagReg &&
    selectedOperand.epoch === epochReg
  val productStage = Stream(BitNetSpuArithmeticResultBeat(cfg))
  productStage.valid := selectedOperand.valid && operationLegal
  selectedOperand.ready := Mux(operationLegal, productStage.ready, True)
  productStage.owner := ownerReg
  productStage.tag := selectedOperand.tag
  productStage.epoch := selectedOperand.epoch
  productStage.sequence := selectedOperand.sequence
  productStage.active := selectedOperand.active
  productStage.first := selectedOperand.first
  productStage.last := selectedOperand.last

  for (site <- 0 until cfg.multiplierCount) {
    val limbProduct = (
      selectedOperand.operandA(site) * selectedOperand.operandB(site)
    ).resize(cfg.productWidth)
    limbProduct.addAttribute("use_dsp", "yes")
    productStage.product(site) := 0
    when(selectedOperand.active(site)) {
      productStage.product(site) := limbProduct
    }
  }

  when(selectedOperand.fire) {
    when(operationLegal) {
      acceptedOperationCount := acceptedOperationCount + 1
    } otherwise {
      droppedOperationCount := droppedOperationCount + 1
      fault := True
    }
  }

  val registeredProduct = productStage.m2sPipe()
  registeredProduct.ready := False
  for (client <- 0 until cfg.clientCount) {
    io.result(client).valid := registeredProduct.valid &&
      registeredProduct.owner === client
    io.result(client).payload := registeredProduct.payload
    when(registeredProduct.owner === client) {
      registeredProduct.ready := io.result(client).ready
    }
    when(io.result(client).fire) {
      emittedResultCount := emittedResultCount + 1
    }
  }

  /* Release only after the owner's operand and result paths are empty. */
  for (client <- 0 until cfg.clientCount) {
    io.release(client).ready := leaseActive && ownerReg === client &&
      !selectedOperand.valid && !registeredProduct.valid
    when(io.release(client).fire) {
      when(io.release(client).tag =/= tagReg ||
          io.release(client).epoch =/= epochReg) {
        fault := True
      }
      leaseActive := False
      releasedLeaseCount := releasedLeaseCount + 1
    }
  }

  io.busy := leaseActive
  io.owned := 0
  when(leaseActive) { io.owned(ownerReg) := True }
  io.activeOwner := ownerReg
  io.activeTag := tagReg
  io.activeEpoch := epochReg
  io.protocolFaultSticky := fault
  io.acceptedLeases := acceptedLeaseCount
  io.acceptedOperations := acceptedOperationCount
  io.emittedResults := emittedResultCount
  io.releasedLeases := releasedLeaseCount
  io.droppedOperations := droppedOperationCount
}

object GenerateBitNetSharedSpuArithmeticPool extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-shared-spu-arithmetic-pool")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetSharedSpuArithmeticPool())
}
