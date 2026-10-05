package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentFfnDownInputFault {
  val None = 0x00
  val InvalidBinding = 0xf0
  val PostGateIdentity = 0xf1
  val PostGateFailed = 0xf2
  val ActivationIdentity = 0xf3
  val ActivationFailed = 0xf4
}

case class BitNetResidentFfnDownInputCoordinatorConfig(
    postGate: BitNetResidentFfnPostGateRmsSubsystemConfig =
      BitNetResidentFfnPostGateRmsSubsystemConfig(),
    activation: BitNetResidentProjectionActivationSubsystemConfig =
      BitNetResidentProjectionActivationSubsystemConfig()
) {
  val model: BitNetConfig = postGate.model
  val rms: BitNetSharedSpuRmsNormConfig = postGate.arithmetic
  val quant: BitNetSharedDynamicI8QuantizerConfig = activation.quant
  val tile: BitNetResidentActivationTileBufferConfig = activation.tile

  require(activation.model == model)
  require(activation.activationLayout == postGate.activationLayout)
  require(model == BitNetConfig.Production)
  require(model.ffnSize == 6912)
  require(rms.maxLength == model.ffnSize)
  require(quant.tagWidth >=
    BitNetResidentLayerGeometry.stepIdWidth(model))
  require(tile.epochWidth == quant.epochWidth)
}

/** Resident context that turns the gate publication into a Down-ready image. */
case class BitNetResidentFfnDownInputContext(
    cfg: BitNetResidentFfnDownInputCoordinatorConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val mode = BitNetMode()
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val imageEpoch = UInt(cfg.quant.epochWidth bits)
  val imageSlot = UInt(cfg.tile.slotWidth bits)
}

/** Authenticated publication consumed by the Down activation replay binder. */
case class BitNetResidentFfnDownInputReady(
    cfg: BitNetResidentFfnDownInputCoordinatorConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val mode = BitNetMode()
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val imageTag = UInt(cfg.quant.tagWidth bits)
  val imageEpoch = UInt(cfg.quant.epochWidth bits)
  val imageSlot = UInt(cfg.tile.slotWidth bits)
  val featureCount = UInt(cfg.tile.featureCountWidth bits)
  val activationFactorQ16 = Vec(UInt(32 bits),
    cfg.model.tokenParallelism)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/**
  * Gate publication -> FfnSubNorm -> dynamic-i8 image transaction coordinator.
  *
  * This block owns no RMS core, quantizer, activation RAM, matrix array or
  * memory port.  It atomically binds the GateUp pipeline's typed RMS command
  * to resident layer context, waits for the in-place 6912-feature FfnSubNorm
  * commit, then asks the one model-wide activation builder to fill a selected
  * ping/pong slot from WideWork.  A successful public completion therefore
  * means that both the exact gate result and its normalized dynamic-i8 image
  * are ready for Down; there is no PS-visible tensor between the stages.
  */
class BitNetResidentFfnDownInputCoordinator(
    cfg: BitNetResidentFfnDownInputCoordinatorConfig =
      BitNetResidentFfnDownInputCoordinatorConfig()) extends Component {
  private val model = cfg.model
  private val rms = cfg.rms
  private val activation = cfg.activation

  val io = new Bundle {
    /** These inputs are accepted on exactly the same edge. */
    val context = slave(Stream(BitNetResidentFfnDownInputContext(cfg)))
    val gateRmsCommand = slave(Stream(BitNetSharedRmsCommand(rms)))

    /** Atomic start boundary of the unique post-gate RMS client. */
    val postGateContext = master(Stream(
      BitNetResidentFfnPostGateRmsContext(model)))
    val postGateRmsCommand = master(Stream(BitNetSharedRmsCommand(rms)))
    val postGateCompletion = slave(Stream(
      BitNetResidentFfnPostGateRmsCompletion(model)))

    /** Command boundary of the unique projection activation builder. */
    val activationCommand = master(Stream(
      BitNetResidentProjectionActivationCommand(activation)))
    val activationCompletion = slave(Stream(
      BitNetResidentProjectionActivationCompletion(activation)))

    val ready = master(Stream(BitNetResidentFfnDownInputReady(cfg)))

    val supported = out Bool()
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val completedInputs = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssuePostGate, WaitPostGate, IssueActivation,
        WaitActivation, Report = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentFfnDownInputContext(cfg)) init
    BitNetResidentFfnDownInputContext(cfg).getZero
  val heldRms = Reg(BitNetSharedRmsCommand(rms)) init
    BitNetSharedRmsCommand(rms).getZero
  val resultFactors = Vec(Reg(UInt(32 bits)) init 0,
    model.tokenParallelism)
  val jobFault = RegInit(False)
  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)
  val completedCount = Reg(UInt(32 bits)) init 0

  val layerLegal = Bool()
  if ((1 << model.layerWidth) == model.layerCount) {
    layerLegal := True
  } else {
    layerLegal := io.context.layer < model.layerCount
  }
  val prefixMask = (1 to model.tokenParallelism).map { count =>
    io.context.tokenMask === B((BigInt(1) << count) - 1,
      model.tokenParallelism bits)
  }.reduce(_ || _)
  val incomingBindingLegal = layerLegal && prefixMask &&
    io.gateRmsCommand.length === model.ffnSize &&
    io.gateRmsCommand.tokenMask === io.context.tokenMask &&
    io.gateRmsCommand.sourceBuffer ===
      BitNetPhysicalActivationBuffer.WideWork &&
    io.gateRmsCommand.tag === io.context.stepId.resize(rms.tagWidth)
  io.supported := incomingBindingLegal

  val atomicInput = state === State.Idle && io.context.valid &&
    io.gateRmsCommand.valid
  io.context.ready := state === State.Idle && io.gateRmsCommand.valid
  io.gateRmsCommand.ready := state === State.Idle && io.context.valid
  when(atomicInput) {
    held := io.context.payload
    heldRms := io.gateRmsCommand.payload
    resultFactors.foreach(_ := 0)
    jobFault := !incomingBindingLegal
    jobFaultCode := 0
    when(incomingBindingLegal) {
      state := State.IssuePostGate
    } otherwise {
      jobFaultCode := B(BitNetResidentFfnDownInputFault.InvalidBinding,
        8 bits)
      stickyFault := True
      state := State.Report
    }
  }

  // Neither child observes half of the post-gate transaction.
  io.postGateContext.valid := state === State.IssuePostGate
  io.postGateRmsCommand.valid := state === State.IssuePostGate
  io.postGateContext.requestId := held.requestId
  io.postGateContext.stepId := held.stepId
  io.postGateContext.layer := held.layer
  io.postGateContext.mode := held.mode
  io.postGateContext.tokenMask := held.tokenMask
  io.postGateRmsCommand.payload := heldRms
  when(io.postGateContext.fire && io.postGateRmsCommand.fire) {
    state := State.WaitPostGate
  }

  val postGateIdentityLegal =
    io.postGateCompletion.requestId === held.requestId &&
      io.postGateCompletion.stepId === held.stepId &&
      io.postGateCompletion.layer === held.layer &&
      io.postGateCompletion.mode === held.mode &&
      io.postGateCompletion.tokenMask === held.tokenMask &&
      io.postGateCompletion.measuredFeatures === model.ffnSize &&
      io.postGateCompletion.replayedFeatures === model.ffnSize &&
      io.postGateCompletion.writtenFeatures === model.ffnSize
  io.postGateCompletion.ready := state === State.WaitPostGate
  when(io.postGateCompletion.fire) {
    when(!postGateIdentityLegal) {
      jobFault := True
      jobFaultCode := B(BitNetResidentFfnDownInputFault.PostGateIdentity,
        8 bits)
      stickyFault := True
      state := State.Report
    } elsewhen(!io.postGateCompletion.success) {
      jobFault := True
      jobFaultCode := Mux(io.postGateCompletion.faultCode.orR,
        io.postGateCompletion.faultCode,
        B(BitNetResidentFfnDownInputFault.PostGateFailed, 8 bits))
      stickyFault := True
      state := State.Report
    } otherwise {
      state := State.IssueActivation
    }
  }

  io.activationCommand.valid := state === State.IssueActivation
  io.activationCommand.requestId := held.requestId
  io.activationCommand.tag := held.stepId.resize(cfg.quant.tagWidth)
  io.activationCommand.epoch := held.imageEpoch
  io.activationCommand.slot := held.imageSlot
  io.activationCommand.sourceBuffer :=
    BitNetPhysicalActivationBuffer.WideWork
  io.activationCommand.featureCount := model.ffnSize
  io.activationCommand.tokenMask := held.tokenMask
  when(io.activationCommand.fire) { state := State.WaitActivation }

  val activationIdentityLegal =
    io.activationCompletion.requestId === held.requestId &&
      io.activationCompletion.tag ===
        held.stepId.resize(cfg.quant.tagWidth) &&
      io.activationCompletion.epoch === held.imageEpoch &&
      io.activationCompletion.slot === held.imageSlot &&
      io.activationCompletion.sourceBuffer ===
        BitNetPhysicalActivationBuffer.WideWork &&
      io.activationCompletion.featureCount === model.ffnSize &&
      io.activationCompletion.acceptedFeatures === model.ffnSize &&
      io.activationCompletion.tokenMask === held.tokenMask
  io.activationCompletion.ready := state === State.WaitActivation
  when(io.activationCompletion.fire) {
    when(!activationIdentityLegal) {
      jobFault := True
      jobFaultCode := B(
        BitNetResidentFfnDownInputFault.ActivationIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.activationCompletion.success) {
      jobFault := True
      jobFaultCode := B(
        BitNetResidentFfnDownInputFault.ActivationFailed, 8 bits)
      stickyFault := True
    } otherwise {
      for (lane <- 0 until model.tokenParallelism) {
        resultFactors(lane) :=
          io.activationCompletion.dequantFactorQ16(lane)
      }
    }
    state := State.Report
  }

  io.ready.valid := state === State.Report
  io.ready.requestId := held.requestId
  io.ready.stepId := held.stepId
  io.ready.layer := held.layer
  io.ready.mode := held.mode
  io.ready.tokenMask := held.tokenMask
  io.ready.imageTag := held.stepId.resize(cfg.quant.tagWidth)
  io.ready.imageEpoch := held.imageEpoch
  io.ready.imageSlot := held.imageSlot
  io.ready.featureCount := model.ffnSize
  for (lane <- 0 until model.tokenParallelism) {
    io.ready.activationFactorQ16(lane) := Mux(jobFault,
      U(0, 32 bits), resultFactors(lane))
  }
  io.ready.success := !jobFault
  io.ready.faultCode := jobFaultCode
  when(io.ready.fire) {
    completedCount := completedCount + 1
    state := State.Idle
  }

  io.busy := state =/= State.Idle
  io.protocolFaultSticky := stickyFault
  io.completedInputs := completedCount
}

object GenerateBitNetResidentFfnDownInputCoordinator extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-ffn-down-input-coordinator")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentFfnDownInputCoordinator())
}
