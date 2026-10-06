package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetResidentFusedResidualEndpointConfig(
    projection: BitNetResidentProjectionFeatureAssemblerConfig =
      BitNetResidentProjectionFeatureAssemblerConfig(),
    activationLayout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production)
) {
  val model: BitNetConfig = projection.model
  val endpoint: BitNetResidentLinearEndpointConfig = projection.endpoint
  val tokenLanes: Int = projection.tokenLanes
  val scale: NativeScaleResidualConfig =
    NativeScaleResidualConfig(lanes = tokenLanes)
  val shardWidth: Int = log2Up(model.spuCount)

  require(activationLayout.cfg == model)
  require((tokenLanes == 2 || tokenLanes == 4) &&
    model.spuCount == 4 && isPow2(model.spuCount))
  require(activationLayout.features(
    BitNetPhysicalActivationBuffer.HiddenResident) >= model.hiddenSize)
}

case class BitNetResidentFusedResidualCompletion(
    cfg: BitNetResidentFusedResidualEndpointConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val kind = BitNetResidentLinearKind()
  val sequenceId = UInt(cfg.endpoint.matrix.sequenceIdWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
  val acceptedFeatures = UInt(32 bits)
  val writtenFeatures = UInt(32 bits)
}

object BitNetResidentFusedResidualFault {
  val None = 0x00
  val InvalidLaunch = 0xf1
  val FeatureMetadata = 0xf2
  val ReadMetadata = 0xf3
  val ScaleMetadata = 0xf4
  val FeatureCount = 0xf5
}

/**
  * Streaming O/Down scale + residual endpoint.
  *
  * A logical TP4 projection feature causes one read from each active token's
  * HiddenResident stripe.  The raw dot, Q16.16 factors and those four residual
  * values are sent to the compute island's *single* projection-scale adapter.
  * Its saturated result is written back to only the selected shard.  The
  * component owns no multiplier and buffers only one feature, so matrix,
  * scaling and activation-BRAM traffic remain overlapped.
  *
  * `completion` is the destination-epoch commit barrier: it is emitted only
  * after the final active-lane write handshake.  A protocol error raises the
  * sticky fault and returns failure; the enclosing island fail-stops before a
  * partially written physical image can become a visible semantic epoch.
  */
class BitNetResidentFusedResidualEndpoint(
    cfg: BitNetResidentFusedResidualEndpointConfig =
      BitNetResidentFusedResidualEndpointConfig()) extends Component {
  private val lanes = cfg.tokenLanes
  private val matrix = cfg.endpoint.matrix

  val io = new Bundle {
    val launch = slave(Stream(BitNetResidentLinearLaunch(cfg.endpoint)))
    val input = slave(Stream(BitNetResidentProjectionFeature(cfg.projection)))

    val scaleInput = master(Stream(NativeScaleResidualBeat(cfg.scale)))
    val scaleOutput = slave(Stream(NativeScaleResidualResult(cfg.scale)))

    val readCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(cfg.model, cfg.activationLayout))),
      lanes)
    val readData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(cfg.model, cfg.activationLayout))),
      lanes)
    val writeData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(cfg.model, cfg.activationLayout))),
      lanes)

    val completion = master(Stream(
      BitNetResidentFusedResidualCompletion(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedFrames = out UInt(32 bits)
    val completedFrames = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val WaitFeature, IssueReads, WaitReads, IssueScale, WaitScale,
        IssueWrites = newElement()
  }
  val state = Reg(State()) init State.WaitFeature
  val active = RegInit(False)
  val descriptor = Reg(BitNetResidentLinearLaunch(cfg.endpoint)) init
    BitNetResidentLinearLaunch(cfg.endpoint).getZero
  val held = Reg(BitNetResidentProjectionFeature(cfg.projection)) init
    BitNetResidentProjectionFeature(cfg.projection).getZero
  val scaled = Reg(NativeScaleResidualResult(cfg.scale)) init
    NativeScaleResidualResult(cfg.scale).getZero
  val residual = Vec(Reg(SInt(32 bits)) init 0, lanes)
  val readCommandPending = Reg(Bits(lanes bits)) init 0
  val readResponsePending = Reg(Bits(lanes bits)) init 0
  val writePending = Reg(Bits(lanes bits)) init 0
  val expectedFeatures = Reg(UInt(32 bits)) init 0
  val acceptedFeatures = Reg(UInt(32 bits)) init 0
  val writtenFeatures = Reg(UInt(32 bits)) init 0
  val stickyFault = RegInit(False)
  val acceptedFrameCount = Reg(UInt(32 bits)) init 0
  val completedFrameCount = Reg(UInt(32 bits)) init 0

  val completionValid = RegInit(False)
  val completionReg = Reg(
    BitNetResidentFusedResidualCompletion(cfg)) init
    BitNetResidentFusedResidualCompletion(cfg).getZero
  io.completion.valid := completionValid
  io.completion.payload := completionReg
  when(io.completion.fire) { completionValid := False }

  val launchFusedKind =
    io.launch.kind === BitNetResidentLinearKind.AttentionOutput ||
      io.launch.kind === BitNetResidentLinearKind.Down
  val launchLegal = launchFusedKind &&
    BitNetResidentLinearProtocol.launchLegal(io.launch.payload, cfg.endpoint)
  val launchSpan = Vec(UInt(32 bits), cfg.endpoint.engineCount)
  for (engine <- 0 until cfg.endpoint.engineCount) {
    launchSpan(engine) :=
      io.launch.matrixTask.outputEndExclusive(engine).resize(32) -
        io.launch.matrixTask.outputBase(engine).resize(32)
  }
  val launchFeatureCount = UInt(32 bits)
  launchFeatureCount := launchSpan.reduce(_ + _)
  when(io.launch.mode === BitNetMode.Prefill) {
    launchFeatureCount := launchSpan(0)
  }

  io.launch.ready := !active && !completionValid
  when(io.launch.fire) {
    descriptor := io.launch.payload
    expectedFeatures := launchFeatureCount
    acceptedFeatures := 0
    writtenFeatures := 0
    readCommandPending := 0
    readResponsePending := 0
    writePending := 0
    state := State.WaitFeature
    when(launchLegal) {
      active := True
      acceptedFrameCount := acceptedFrameCount + 1
    } otherwise {
      active := False
      stickyFault := True
      completionReg.requestId := io.launch.requestId
      completionReg.layer := io.launch.layer
      completionReg.kind := io.launch.kind
      completionReg.sequenceId := io.launch.sequenceId
      completionReg.success := False
      completionReg.faultCode :=
        B(BitNetResidentFusedResidualFault.InvalidLaunch, 8 bits)
      completionReg.acceptedFeatures := 0
      completionReg.writtenFeatures := 0
      completionValid := True
      completedFrameCount := completedFrameCount + 1
    }
  }

  val inputIdentityLegal =
    io.input.requestId === descriptor.requestId &&
      io.input.layer === descriptor.layer &&
      io.input.kind === descriptor.kind &&
      io.input.mode === descriptor.mode &&
      io.input.sequenceId === descriptor.sequenceId &&
      io.input.tokenMask === descriptor.tokenMask
  val inputFramingLegal =
    io.input.first === (acceptedFeatures === 0) &&
      io.input.last === (acceptedFeatures === expectedFeatures - 1) &&
      acceptedFeatures < expectedFeatures &&
      io.input.featureIndex < cfg.model.hiddenSize
  val inputLegal = inputIdentityLegal && inputFramingLegal

  io.input.ready := active && state === State.WaitFeature &&
    !completionValid
  when(io.input.fire) {
    when(inputLegal) {
      held := io.input.payload
      acceptedFeatures := acceptedFeatures + 1
      readCommandPending := io.input.tokenMask
      readResponsePending := io.input.tokenMask
      residual.foreach(_ := 0)
      state := State.IssueReads
    } otherwise {
      stickyFault := True
      active := False
      completionReg.requestId := descriptor.requestId
      completionReg.layer := descriptor.layer
      completionReg.kind := descriptor.kind
      completionReg.sequenceId := descriptor.sequenceId
      completionReg.success := False
      completionReg.faultCode :=
        B(BitNetResidentFusedResidualFault.FeatureMetadata, 8 bits)
      completionReg.acceptedFeatures := acceptedFeatures
      completionReg.writtenFeatures := writtenFeatures
      completionValid := True
      completedFrameCount := completedFrameCount + 1
    }
  }

  val heldGroup = (held.featureIndex |>> cfg.shardWidth)
    .resize(cfg.activationLayout.groupIndexWidth)
  val heldShard = held.featureIndex(cfg.shardWidth - 1 downto 0)

  val readCommandFire = Bits(lanes bits)
  readCommandFire := 0
  for (lane <- 0 until lanes) {
    io.readCommand(lane).valid := state === State.IssueReads &&
      readCommandPending(lane)
    io.readCommand(lane).buffer :=
      BitNetPhysicalActivationBuffer.HiddenResident
    io.readCommand(lane).group := heldGroup
    readCommandFire(lane) := io.readCommand(lane).fire
  }
  when(state === State.IssueReads) {
    when(!(readCommandPending & ~readCommandFire).orR) {
      readCommandPending := 0
      state := State.WaitReads
    } otherwise {
      readCommandPending := readCommandPending & ~readCommandFire
    }
  }

  val readResponseFire = Bits(lanes bits)
  val readResponseFault = Bits(lanes bits)
  readResponseFire := 0
  readResponseFault := 0
  for (lane <- 0 until lanes) {
    io.readData(lane).ready := state === State.WaitReads &&
      readResponsePending(lane)
    readResponseFire(lane) := io.readData(lane).fire
    readResponseFault(lane) := io.readData(lane).fire &&
      (io.readData(lane).fault ||
        io.readData(lane).buffer =/=
          BitNetPhysicalActivationBuffer.HiddenResident ||
        io.readData(lane).group =/= heldGroup ||
        !io.readData(lane).featureValid(heldShard))
    when(io.readData(lane).fire) {
      residual(lane) := io.readData(lane).values(heldShard)
    }
  }
  val readsRetire = state === State.WaitReads &&
    !(readResponsePending & ~readResponseFire).orR
  when(state === State.WaitReads) {
    readResponsePending := readResponsePending & ~readResponseFire
    when(readResponseFault.orR) {
      stickyFault := True
    }
    when(readsRetire) {
      readResponsePending := 0
      when(readResponseFault.orR) {
        active := False
        completionReg.requestId := descriptor.requestId
        completionReg.layer := descriptor.layer
        completionReg.kind := descriptor.kind
        completionReg.sequenceId := descriptor.sequenceId
        completionReg.success := False
        completionReg.faultCode :=
          B(BitNetResidentFusedResidualFault.ReadMetadata, 8 bits)
        completionReg.acceptedFeatures := acceptedFeatures
        completionReg.writtenFeatures := writtenFeatures
        completionValid := True
        completedFrameCount := completedFrameCount + 1
        state := State.WaitFeature
      } otherwise {
        state := State.IssueScale
      }
    }
  }

  io.scaleInput.valid := active && state === State.IssueScale
  io.scaleInput.accumulator := held.raw
  io.scaleInput.residual := residual
  io.scaleInput.activationFactor := held.activationFactorQ16
  io.scaleInput.weightScale := held.weightScaleQ16
  io.scaleInput.laneMask := held.tokenMask
  io.scaleInput.residualEnable := True
  io.scaleInput.first := held.first
  io.scaleInput.last := held.last
  when(io.scaleInput.fire) { state := State.WaitScale }

  io.scaleOutput.ready := active && state === State.WaitScale
  when(io.scaleOutput.fire) {
    val scaleMetadataLegal =
      io.scaleOutput.laneMask === held.tokenMask &&
        io.scaleOutput.first === held.first &&
        io.scaleOutput.last === held.last
    when(scaleMetadataLegal) {
      scaled := io.scaleOutput.payload
      writePending := held.tokenMask
      state := State.IssueWrites
    } otherwise {
      stickyFault := True
      active := False
      completionReg.requestId := descriptor.requestId
      completionReg.layer := descriptor.layer
      completionReg.kind := descriptor.kind
      completionReg.sequenceId := descriptor.sequenceId
      completionReg.success := False
      completionReg.faultCode :=
        B(BitNetResidentFusedResidualFault.ScaleMetadata, 8 bits)
      completionReg.acceptedFeatures := acceptedFeatures
      completionReg.writtenFeatures := writtenFeatures
      completionValid := True
      completedFrameCount := completedFrameCount + 1
      state := State.WaitFeature
    }
  }

  val writeFire = Bits(lanes bits)
  writeFire := 0
  for (lane <- 0 until lanes) {
    io.writeData(lane).valid := active && state === State.IssueWrites &&
      writePending(lane)
    io.writeData(lane).buffer :=
      BitNetPhysicalActivationBuffer.HiddenResident
    io.writeData(lane).group := heldGroup
    io.writeData(lane).values.foreach(_ := 0)
    io.writeData(lane).featureValid := 0
    for (shard <- 0 until cfg.model.spuCount) {
      when(heldShard === shard) {
        io.writeData(lane).values(shard) := scaled.value(lane)
        io.writeData(lane).featureValid(shard) := True
      }
    }
    writeFire(lane) := io.writeData(lane).fire
  }
  val writesRetire = state === State.IssueWrites &&
    !(writePending & ~writeFire).orR
  when(state === State.IssueWrites) {
    writePending := writePending & ~writeFire
    when(writesRetire) {
      writePending := 0
      writtenFeatures := writtenFeatures + 1
      when(held.last) {
        val countLegal = writtenFeatures + 1 === expectedFeatures
        active := False
        completionReg.requestId := descriptor.requestId
        completionReg.layer := descriptor.layer
        completionReg.kind := descriptor.kind
        completionReg.sequenceId := descriptor.sequenceId
        completionReg.success := countLegal
        completionReg.faultCode :=
          B(BitNetResidentFusedResidualFault.None, 8 bits)
        when(!countLegal) {
          stickyFault := True
          completionReg.faultCode :=
            B(BitNetResidentFusedResidualFault.FeatureCount, 8 bits)
        }
        completionReg.acceptedFeatures := acceptedFeatures
        completionReg.writtenFeatures := writtenFeatures + 1
        completionValid := True
        completedFrameCount := completedFrameCount + 1
        state := State.WaitFeature
      } otherwise {
        state := State.WaitFeature
      }
    }
  }

  io.busy := active || state =/= State.WaitFeature || completionValid
  io.protocolFaultSticky := stickyFault
  io.acceptedFrames := acceptedFrameCount
  io.completedFrames := completedFrameCount
}

object GenerateBitNetResidentFusedResidualEndpoint extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-fused-residual-endpoint")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentFusedResidualEndpoint())
}
