package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentGateScaleComposerFault {
  val None = 0x00
  val InvalidCommand = 0xe8
  val PoolMetadata = 0xe9
  val ScaleOverflow = 0xea
  val ZeroScale = 0xeb
}

case class BitNetResidentGateScaleComposerConfig(
    model: BitNetConfig = BitNetConfig.Production,
    gate: BitNetSharedSpuFfnGateConfig =
      BitNetSharedSpuFfnGateConfig(),
    arithmetic: BitNetSharedSpuArithmeticPoolConfig =
      BitNetSharedSpuArithmeticPoolConfig(),
    epochWidth: Int = 8
) {
  require(model == BitNetConfig.Production)
  require(gate.model == model)
  require((gate.tokenLanes == 2 || gate.tokenLanes == 4) &&
    gate.scaleWidth == 26)
  require(arithmetic.clientCount == BitNetSpuArithmeticClient.Count)
  require(arithmetic.multiplierCount == 8)
  require(arithmetic.operandAWidth >= 16 && arithmetic.operandBWidth >= 16)
  require(arithmetic.productWidth >= 32)
  require(arithmetic.sequenceWidth >= 2)
  require(epochWidth == arithmetic.epochWidth)
}

/** Resident inputs needed to derive the exact combined Gate/Up scales. */
case class BitNetResidentGateScaleComposeCommand(
    cfg: BitNetResidentGateScaleComposerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val activationFactorQ16 = Vec(UInt(32 bits), cfg.model.tokenParallelism)
  val gateWeightScaleQ16 = UInt(32 bits)
  val upWeightScaleQ16 = UInt(32 bits)
}

/**
  * A successful result can be wired field-for-field to the leased Gate core.
  * Failed results deliberately carry zero scales and must not be published as
  * a gate command by the surrounding transaction binder.
  */
case class BitNetResidentGateScaleComposeResult(
    cfg: BitNetResidentGateScaleComposerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val gateScaleQ16 = Vec(UInt(cfg.gate.scaleWidth bits),
    cfg.model.tokenParallelism)
  val upScaleQ16 = Vec(UInt(cfg.gate.scaleWidth bits),
    cfg.model.tokenParallelism)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

object BitNetResidentGateScaleComposerMath {
  val Maximum: BigInt = (BigInt(1) << 26) - 1

  /** Round a positive Q16.16-by-Q16.16 product back to Q16.16. */
  def compose(activation: BigInt, weight: BigInt): Either[Int, BigInt] = {
    require(activation >= 0 && activation < (BigInt(1) << 32))
    require(weight >= 0 && weight < (BigInt(1) << 32))
    if (activation == 0 || weight == 0) {
      Left(BitNetResidentGateScaleComposerFault.ZeroScale)
    } else {
      val rounded = (activation * weight + (BigInt(1) << 15)) >> 16
      if (rounded == 0)
        Left(BitNetResidentGateScaleComposerFault.ZeroScale)
      else if (rounded > Maximum)
        Left(BitNetResidentGateScaleComposerFault.ScaleOverflow)
      else Right(rounded)
    }
  }
}

/**
  * DSP-free resident Gate/Up combined-scale composer.
  *
  * The matrix endpoint carries the activation dequantisation factor and the
  * AUX stager carries the two BF16 row scales independently.  The exact gate
  * consumes their positive Q16.16 products.  One command therefore borrows
  * the existing FfnGate arithmetic-pool client for four cycles, evaluating
  * one token lane's Gate and Up products in parallel.  Each U32xU32 product is
  * reconstructed from four 16x16 limbs, so all eight physical pool sites are
  * used and this component owns no multiplier or second arithmetic bank.
  *
  * Inactive lanes are emitted as zero.  Active zero, overflow and bad pool
  * identity are reported fail-closed; no partially valid scale set escapes.
  */
class BitNetResidentGateScaleComposer(
    cfg: BitNetResidentGateScaleComposerConfig =
      BitNetResidentGateScaleComposerConfig()) extends Component {
  private val lanes = cfg.model.tokenParallelism
  private val pool = cfg.arithmetic
  private val gate = cfg.gate
  private val poolOwner = BitNetSpuArithmeticClient.FfnGate
  private val maximum = BitNetResidentGateScaleComposerMath.Maximum

  val io = new Bundle {
    val command = slave(Stream(
      BitNetResidentGateScaleComposeCommand(cfg)))
    val result = master(Stream(
      BitNetResidentGateScaleComposeResult(cfg)))

    /** Direct client-2 boundary of the one model-wide SPU pool. */
    val poolAcquire = master(Stream(
      BitNetSpuArithmeticLeaseRequest(pool)))
    val poolOperand = master(Stream(
      BitNetSpuArithmeticOperandBeat(pool)))
    val poolResult = slave(Stream(
      BitNetSpuArithmeticResultBeat(pool)))
    val poolRelease = master(Stream(
      BitNetSpuArithmeticLeaseRelease(pool)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
    val acceptedCommands = out UInt(32 bits)
    val completedCommands = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Acquire, Issue, WaitResult, Release, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentGateScaleComposeCommand(cfg)) init
    BitNetResidentGateScaleComposeCommand(cfg).getZero
  val lane = Reg(UInt(log2Up(lanes) bits)) init 0
  val leaseTag = Reg(UInt(pool.tagWidth bits)) init 0
  val gateScales = Vec(Reg(UInt(gate.scaleWidth bits)) init 0, lanes)
  val upScales = Vec(Reg(UInt(gate.scaleWidth bits)) init 0, lanes)
  val jobFault = RegInit(False)
  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)
  val stickyOverflow = RegInit(False)
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0

  val layerLegal = Bool()
  if ((1 << cfg.model.layerWidth) == cfg.model.layerCount) {
    layerLegal := True
  } else {
    layerLegal := io.command.layer < cfg.model.layerCount
  }
  val prefixMask = (1 to lanes).map { count =>
    io.command.tokenMask === B((BigInt(1) << count) - 1, lanes bits)
  }.reduce(_ || _)
  val activeFactorsPositive = Vec(Bool(), lanes)
  for (index <- 0 until lanes) {
    activeFactorsPositive(index) := !io.command.tokenMask(index) ||
      io.command.activationFactorQ16(index) =/= 0
  }
  val commandLegal = layerLegal && prefixMask &&
    activeFactorsPositive.asBits.andR &&
    io.command.gateWeightScaleQ16 =/= 0 &&
    io.command.upWeightScaleQ16 =/= 0

  io.command.ready := state === State.Idle
  when(io.command.fire) {
    held := io.command.payload
    lane := 0
    leaseTag := (io.command.requestId.resize(pool.tagWidth) ^
      io.command.stepId.resize(pool.tagWidth) ^
      io.command.epoch.resize(pool.tagWidth)).resized
    gateScales.foreach(_ := 0)
    upScales.foreach(_ := 0)
    jobFault := !commandLegal
    jobFaultCode := 0
    acceptedCount := acceptedCount + 1
    when(commandLegal) {
      state := State.Acquire
    } otherwise {
      jobFaultCode := B(
        BitNetResidentGateScaleComposerFault.InvalidCommand, 8 bits)
      stickyFault := True
      state := State.Report
    }
  }

  io.poolAcquire.valid := state === State.Acquire
  io.poolAcquire.tag := leaseTag
  io.poolAcquire.epoch := held.epoch
  when(io.poolAcquire.fire) { state := State.Issue }

  io.poolOperand.valid := state === State.Issue
  io.poolOperand.tag := leaseTag
  io.poolOperand.epoch := held.epoch
  io.poolOperand.sequence := lane.resized
  io.poolOperand.active := B"8'hff"
  io.poolOperand.first := lane === 0
  io.poolOperand.last := lane === lanes - 1

  def driveU32xU32(base: Int, a: UInt, b: UInt): Unit = {
    io.poolOperand.operandA(base + 0) := a(15 downto 0).resized
    io.poolOperand.operandB(base + 0) := b(15 downto 0).resized
    io.poolOperand.operandA(base + 1) := a(15 downto 0).resized
    io.poolOperand.operandB(base + 1) := b(31 downto 16).resized
    io.poolOperand.operandA(base + 2) := a(31 downto 16).resized
    io.poolOperand.operandB(base + 2) := b(15 downto 0).resized
    io.poolOperand.operandA(base + 3) := a(31 downto 16).resized
    io.poolOperand.operandB(base + 3) := b(31 downto 16).resized
  }
  driveU32xU32(0, held.activationFactorQ16(lane),
    held.gateWeightScaleQ16)
  driveU32xU32(4, held.activationFactorQ16(lane),
    held.upWeightScaleQ16)

  when(io.poolOperand.fire) { state := State.WaitResult }

  io.poolResult.ready := state === State.WaitResult
  val resultMetadataLegal =
    io.poolResult.owner === poolOwner &&
      io.poolResult.tag === leaseTag &&
      io.poolResult.epoch === held.epoch &&
      io.poolResult.sequence === lane.resize(pool.sequenceWidth) &&
      io.poolResult.active === B"8'hff" &&
      io.poolResult.first === (lane === 0) &&
      io.poolResult.last === (lane === lanes - 1)

  def reconstructU32xU32(base: Int): UInt = {
    val lowLow = io.poolResult.product(base + 0).resize(65)
    val lowHigh = io.poolResult.product(base + 1).resize(65)
    val highLow = io.poolResult.product(base + 2).resize(65)
    val highHigh = io.poolResult.product(base + 3).resize(65)
    (lowLow + ((lowHigh + highLow).resize(65) |<< 16) +
      (highHigh |<< 32)).resize(65)
  }

  val gateProduct = reconstructU32xU32(0)
  val upProduct = reconstructU32xU32(4)
  val gateRoundedWide = (gateProduct + U(BigInt(1) << 15, 65 bits)) |>> 16
  val upRoundedWide = (upProduct + U(BigInt(1) << 15, 65 bits)) |>> 16
  val activeLane = held.tokenMask(lane)
  val gateOverflow = gateRoundedWide > U(maximum, gateRoundedWide.getWidth bits)
  val upOverflow = upRoundedWide > U(maximum, upRoundedWide.getWidth bits)
  val gateZero = gateRoundedWide === 0
  val upZero = upRoundedWide === 0

  when(io.poolResult.fire) {
    when(!resultMetadataLegal) {
      jobFault := True
      jobFaultCode := B(
        BitNetResidentGateScaleComposerFault.PoolMetadata, 8 bits)
      stickyFault := True
    } elsewhen(activeLane && (gateOverflow || upOverflow)) {
      jobFault := True
      jobFaultCode := B(
        BitNetResidentGateScaleComposerFault.ScaleOverflow, 8 bits)
      stickyFault := True
      stickyOverflow := True
    } elsewhen(activeLane && (gateZero || upZero)) {
      jobFault := True
      jobFaultCode := B(
        BitNetResidentGateScaleComposerFault.ZeroScale, 8 bits)
      stickyFault := True
    } otherwise {
      when(activeLane) {
        gateScales(lane) := gateRoundedWide.resized
        upScales(lane) := upRoundedWide.resized
      } otherwise {
        gateScales(lane) := U(0, gate.scaleWidth bits)
        upScales(lane) := U(0, gate.scaleWidth bits)
      }
    }

    when(lane === lanes - 1) {
      state := State.Release
    } otherwise {
      lane := lane + 1
      state := State.Issue
    }
  }

  io.poolRelease.valid := state === State.Release
  io.poolRelease.tag := leaseTag
  io.poolRelease.epoch := held.epoch
  when(io.poolRelease.fire) { state := State.Report }

  io.result.valid := state === State.Report
  io.result.requestId := held.requestId
  io.result.stepId := held.stepId
  io.result.layer := held.layer
  io.result.epoch := held.epoch
  io.result.tokenMask := held.tokenMask
  for (index <- 0 until lanes) {
    io.result.gateScaleQ16(index) := Mux(jobFault,
      U(0, gate.scaleWidth bits), gateScales(index))
    io.result.upScaleQ16(index) := Mux(jobFault,
      U(0, gate.scaleWidth bits), upScales(index))
  }
  io.result.success := !jobFault
  io.result.faultCode := jobFaultCode
  when(io.result.fire) {
    completedCount := completedCount + 1
    state := State.Idle
  }

  io.busy := state =/= State.Idle
  io.protocolFaultSticky := stickyFault
  io.overflowSticky := stickyOverflow
  io.acceptedCommands := acceptedCount
  io.completedCommands := completedCount
}

object GenerateBitNetResidentGateScaleComposer extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-gate-scale-composer")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentGateScaleComposer())
}
