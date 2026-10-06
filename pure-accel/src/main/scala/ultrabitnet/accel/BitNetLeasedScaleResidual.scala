package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * DSP-free projection scale/residual client for the global SPU arithmetic
  * lease.
  *
  * The public input/output contract is exactly [[NativeScaleResidual]].  Wide
  * products are expressed as five eight-limb pool beats:
  *
  *   0. activationFactor(0) x weightScale;
  *   1..3. accumulator(n) x factor(n), in parallel with factor(n+1);
  *   4. accumulator(3) x factor(3).
  *
  * U32 x U32 and U33 x U32 are each reconstructed from four unsigned limbs.
  * The client owns no multiplier and accepts no new TP4 vector until its
  * result has been reassembled and the command-level lease released.  This is
  * still faster than the twenty slow clocks between production QKV final
  * features, so it cannot reduce the matrix/DDR roofline.
  */
class BitNetLeasedScaleResidual(
    cfg: NativeScaleResidualConfig = NativeScaleResidualConfig(),
    poolCfg: BitNetSharedSpuArithmeticPoolConfig =
      BitNetSharedSpuArithmeticPoolConfig()) extends Component {
  require(cfg.lanes == 2 || cfg.lanes == 4,
    "the production scale endpoint must be TP2 or TP4")
  require(poolCfg.clientCount == BitNetSpuArithmeticClient.Count)
  require(poolCfg.multiplierCount == 8 &&
    poolCfg.operandAWidth >= 17 && poolCfg.operandBWidth >= 16)

  val io = new Bundle {
    val input = slave(Stream(NativeScaleResidualBeat(cfg)))
    val output = master(Stream(NativeScaleResidualResult(cfg)))

    val poolAcquire = master(Stream(
      BitNetSpuArithmeticLeaseRequest(poolCfg)))
    val poolOperand = master(Stream(
      BitNetSpuArithmeticOperandBeat(poolCfg)))
    val poolResult = slave(Stream(
      BitNetSpuArithmeticResultBeat(poolCfg)))
    val poolRelease = master(Stream(
      BitNetSpuArithmeticLeaseRelease(poolCfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedVectors = out UInt(32 bits)
    val emittedVectors = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Acquire, IssueFactor0, WaitFactor0, IssueDotAndFactor,
        WaitDotAndFactor, IssueFinalDot, WaitFinalDot, Release,
        Output = newElement()
  }
  val state = Reg(State()) init State.Idle

  private val projectionOwner = BitNetSpuArithmeticClient.ProjectionScale
  private val positiveMax = BigInt("7fffffff", 16)
  private val negativeMagnitudeMax = BigInt(1) << (cfg.valueWidth - 1)

  val held = Reg(NativeScaleResidualBeat(cfg)) init (
    NativeScaleResidualBeat(cfg).getZero)
  val activeTag = Reg(UInt(poolCfg.tagWidth bits)) init 0
  val activeEpoch = Reg(UInt(poolCfg.epochWidth bits)) init 0
  val nextTag = Reg(UInt(poolCfg.tagWidth bits)) init 0
  val nextEpoch = Reg(UInt(poolCfg.epochWidth bits)) init 0
  val laneReg = Reg(UInt(log2Up(cfg.lanes) bits)) init 0
  val deltaFactorReg = Reg(UInt(cfg.factorWidth bits)) init 0
  val outputDelta = Vec(Reg(SInt(cfg.valueWidth bits)) init 0, cfg.lanes)
  val outputValue = Vec(Reg(SInt(cfg.valueWidth bits)) init 0, cfg.lanes)
  val fault = RegInit(False)
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val emittedCount = Reg(UInt(32 bits)) init 0

  io.input.ready := state === State.Idle
  when(io.input.fire) {
    held := io.input.payload
    activeTag := nextTag
    activeEpoch := nextEpoch
    when(nextTag.andR) {
      nextTag := 0
      nextEpoch := nextEpoch + 1
    } otherwise {
      nextTag := nextTag + 1
    }
    laneReg := 0
    deltaFactorReg := 0
    for (lane <- 0 until cfg.lanes) {
      outputDelta(lane) := 0
      outputValue(lane) := 0
    }
    acceptedCount := acceptedCount + 1
    state := State.Acquire
  }

  io.poolAcquire.valid := state === State.Acquire
  io.poolAcquire.tag := activeTag
  io.poolAcquire.epoch := activeEpoch
  when(io.poolAcquire.fire) { state := State.IssueFactor0 }

  io.poolOperand.valid := state === State.IssueFactor0 ||
    state === State.IssueDotAndFactor || state === State.IssueFinalDot
  io.poolOperand.tag := activeTag
  io.poolOperand.epoch := activeEpoch
  io.poolOperand.sequence := 0
  io.poolOperand.active := 0
  io.poolOperand.operandA.foreach(_ := 0)
  io.poolOperand.operandB.foreach(_ := 0)
  io.poolOperand.first := False
  io.poolOperand.last := False

  /** Map one U32 x U32 product onto four pool sites. */
  def driveU32xU32(base: Int, a: UInt, b: UInt): Unit = {
    io.poolOperand.active(base + 0) := True
    io.poolOperand.active(base + 1) := True
    io.poolOperand.active(base + 2) := True
    io.poolOperand.active(base + 3) := True
    io.poolOperand.operandA(base + 0) := a(15 downto 0).resized
    io.poolOperand.operandB(base + 0) := b(15 downto 0).resized
    io.poolOperand.operandA(base + 1) := a(15 downto 0).resized
    io.poolOperand.operandB(base + 1) := b(31 downto 16).resized
    io.poolOperand.operandA(base + 2) := a(31 downto 16).resized
    io.poolOperand.operandB(base + 2) := b(15 downto 0).resized
    io.poolOperand.operandA(base + 3) := a(31 downto 16).resized
    io.poolOperand.operandB(base + 3) := b(31 downto 16).resized
  }

  /** Map one U33 x U32 product onto four pool sites. */
  def driveU33xU32(base: Int, a: UInt, b: UInt): Unit = {
    io.poolOperand.active(base + 0) := True
    io.poolOperand.active(base + 1) := True
    io.poolOperand.active(base + 2) := True
    io.poolOperand.active(base + 3) := True
    io.poolOperand.operandA(base + 0) := a(16 downto 0).resized
    io.poolOperand.operandB(base + 0) := b(15 downto 0).resized
    io.poolOperand.operandA(base + 1) := a(16 downto 0).resized
    io.poolOperand.operandB(base + 1) := b(31 downto 16).resized
    io.poolOperand.operandA(base + 2) := a(32 downto 17).resized
    io.poolOperand.operandB(base + 2) := b(15 downto 0).resized
    io.poolOperand.operandA(base + 3) := a(32 downto 17).resized
    io.poolOperand.operandB(base + 3) := b(31 downto 16).resized
  }

  val selectedAccumulator = held.accumulator(laneReg)
    .resize(cfg.magnitudeWidth)
  val selectedMagnitude = UInt(cfg.magnitudeWidth bits)
  selectedMagnitude := selectedAccumulator.asUInt
  when(selectedAccumulator.msb) {
    selectedMagnitude := (-selectedAccumulator).asUInt
  }

  when(state === State.IssueFactor0) {
    io.poolOperand.sequence := 0
    io.poolOperand.first := True
    driveU32xU32(0, held.activationFactor(0), held.weightScale)
  }
  when(state === State.IssueDotAndFactor) {
    io.poolOperand.sequence := (laneReg + 1).resized
    driveU33xU32(0, selectedMagnitude, deltaFactorReg)
    driveU32xU32(4, held.activationFactor(laneReg + 1), held.weightScale)
  }
  when(state === State.IssueFinalDot) {
    io.poolOperand.sequence := 4
    io.poolOperand.last := True
    driveU33xU32(0, selectedMagnitude, deltaFactorReg)
  }

  when(io.poolOperand.fire) {
    switch(state) {
      is(State.IssueFactor0) { state := State.WaitFactor0 }
      is(State.IssueDotAndFactor) { state := State.WaitDotAndFactor }
      is(State.IssueFinalDot) { state := State.WaitFinalDot }
      default { }
    }
  }

  io.poolResult.ready := state === State.WaitFactor0 ||
    state === State.WaitDotAndFactor || state === State.WaitFinalDot
  val expectedSequence = UInt(poolCfg.sequenceWidth bits)
  expectedSequence := 0
  when(state === State.WaitDotAndFactor) {
    expectedSequence := (laneReg + 1).resized
  }
  when(state === State.WaitFinalDot) { expectedSequence := 4 }
  val resultMetadataValid =
    io.poolResult.owner === projectionOwner &&
      io.poolResult.tag === activeTag &&
      io.poolResult.epoch === activeEpoch &&
      io.poolResult.sequence === expectedSequence

  /** Reconstruct four 16x16 limbs into the exact U64 factor product. */
  def reconstructU32xU32(base: Int): UInt = {
    val lowLow = io.poolResult.product(base + 0).resize(64)
    val cross = io.poolResult.product(base + 1).resize(45) +
      io.poolResult.product(base + 2).resize(45)
    val highHigh = io.poolResult.product(base + 3).resize(64)
    (lowLow + (cross.resize(64) |<< 16) +
      (highHigh |<< 32)).resize(64)
  }

  /** Reconstruct 17/16 by 16/16 limbs into the exact U65 dot product. */
  def reconstructU33xU32(base: Int): UInt = {
    val lowLow = io.poolResult.product(base + 0).resize(65)
    val lowHigh = io.poolResult.product(base + 1).resize(65)
    val highLow = io.poolResult.product(base + 2).resize(65)
    val highHigh = io.poolResult.product(base + 3).resize(65)
    (lowLow + (lowHigh |<< 16) + (highLow |<< 17) +
      (highHigh |<< 33)).resize(65)
  }

  def roundedFactor(product: UInt): UInt = {
    val rounded = product.resize(65) +
      U(BigInt(1) << (cfg.fractionBits - 1), 65 bits)
    val shifted = rounded |>> cfg.fractionBits
    val result = UInt(cfg.factorWidth bits)
    when(shifted > U(BigInt("ffffffff", 16), shifted.getWidth bits)) {
      result := U(BigInt("ffffffff", 16), cfg.factorWidth bits)
    } otherwise {
      result := shifted.resized
    }
    result
  }

  // Metadata is a control guard, not part of the arithmetic cone.  Keeping
  // it out of every reconstructed product removes a high-fanout tag compare
  // from the 65-bit carry path; an invalid response is consumed, faults, and
  // leaves the FSM waiting for a correctly tagged response.
  val factor0Product = reconstructU32xU32(0)
  val nextFactorProduct = reconstructU32xU32(4)
  val dotProduct = reconstructU33xU32(0)

  val scaledDelta = SInt(cfg.valueWidth bits)
  scaledDelta := 0
  when(selectedAccumulator.msb) {
    when(dotProduct >= U(negativeMagnitudeMax, 65 bits)) {
      scaledDelta := S(-negativeMagnitudeMax, cfg.valueWidth bits)
    } otherwise {
      scaledDelta := (-dotProduct.resize(cfg.valueWidth + 1).asSInt)
        .resize(cfg.valueWidth)
    }
  } otherwise {
    when(dotProduct > U(positiveMax, 65 bits)) {
      scaledDelta := S(positiveMax, cfg.valueWidth bits)
    } otherwise {
      scaledDelta := dotProduct.resize(cfg.valueWidth).asSInt
    }
  }

  val maskedDelta = Mux(held.laneMask(laneReg), scaledDelta,
    S(0, cfg.valueWidth bits))
  val residualSum = held.residual(laneReg).resize(cfg.valueWidth + 1) +
    maskedDelta.resize(cfg.valueWidth + 1)
  val saturatedResidual = SInt(cfg.valueWidth bits)
  when(residualSum > S(positiveMax, cfg.valueWidth + 1 bits)) {
    saturatedResidual := S(positiveMax, cfg.valueWidth bits)
  } elsewhen (residualSum <
      S(-negativeMagnitudeMax, cfg.valueWidth + 1 bits)) {
    saturatedResidual := S(-negativeMagnitudeMax, cfg.valueWidth bits)
  } otherwise {
    saturatedResidual := residualSum.resized
  }
  val selectedValue = Mux(
    held.residualEnable, saturatedResidual, maskedDelta)

  when(io.poolResult.fire) {
    when(!resultMetadataValid) {
      fault := True
    } otherwise {
      switch(state) {
        is(State.WaitFactor0) {
          deltaFactorReg := roundedFactor(factor0Product)
          laneReg := 0
          state := State.IssueDotAndFactor
        }
        is(State.WaitDotAndFactor) {
          outputDelta(laneReg) := maskedDelta
          outputValue(laneReg) := Mux(
            held.laneMask(laneReg), selectedValue,
            S(0, cfg.valueWidth bits))
          deltaFactorReg := roundedFactor(nextFactorProduct)
          laneReg := laneReg + 1
          when(laneReg === cfg.lanes - 2) {
            state := State.IssueFinalDot
          } otherwise {
            state := State.IssueDotAndFactor
          }
        }
        is(State.WaitFinalDot) {
          outputDelta(laneReg) := maskedDelta
          outputValue(laneReg) := Mux(
            held.laneMask(laneReg), selectedValue,
            S(0, cfg.valueWidth bits))
          state := State.Release
        }
        default { }
      }
    }
  }

  io.poolRelease.valid := state === State.Release
  io.poolRelease.tag := activeTag
  io.poolRelease.epoch := activeEpoch
  when(io.poolRelease.fire) { state := State.Output }

  io.output.valid := state === State.Output
  io.output.scaledDelta := outputDelta
  io.output.value := outputValue
  io.output.laneMask := held.laneMask
  io.output.first := held.first
  io.output.last := held.last
  when(io.output.fire) {
    emittedCount := emittedCount + 1
    state := State.Idle
  }

  io.busy := state =/= State.Idle
  io.protocolFaultSticky := fault
  io.acceptedVectors := acceptedCount
  io.emittedVectors := emittedCount
}
