package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** The PL-resident consumers of the single dynamic-i8 datapath. */
object BitNetResidentSharedDynamicQuantOwner
    extends SpinalEnum(binarySequential) {
  val ProjectionImage, QkvHead, LmHead = newElement()
}

/** Client-side view of the model-wide dynamic-i8 service. */
case class BitNetResidentSharedDynamicQuantClientPort(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val command = slave(Stream(BitNetSharedDynamicQuantCommand(cfg)))
  val measure = slave(Stream(BitNetSharedDynamicQuantMeasureBeat(cfg)))
  val replayRequest = master(Stream(
    BitNetSharedDynamicQuantReplayRequest(cfg)))
  val replay = slave(Stream(BitNetSharedDynamicQuantReplayBeat(cfg)))
  val output = master(Stream(BitNetSharedDynamicQuantResultBeat(cfg)))
  val done = master(Stream(BitNetSharedDynamicQuantDone(cfg)))
}

/** Direction of the one physical quantizer as seen by its owner router. */
case class BitNetResidentSharedDynamicQuantArithmeticPort(
    cfg: BitNetSharedDynamicI8QuantizerConfig) extends Bundle {
  val command = master(Stream(BitNetSharedDynamicQuantCommand(cfg)))
  val measure = master(Stream(BitNetSharedDynamicQuantMeasureBeat(cfg)))
  val replayRequest = slave(Stream(
    BitNetSharedDynamicQuantReplayRequest(cfg)))
  val replay = master(Stream(BitNetSharedDynamicQuantReplayBeat(cfg)))
  val output = slave(Stream(BitNetSharedDynamicQuantResultBeat(cfg)))
  val done = slave(Stream(BitNetSharedDynamicQuantDone(cfg)))
}

/**
  * Whole-command owner router for resident projection-image and LM-head
  * quantization.  Ownership starts at the physical command handshake and is
  * released only when the same client's authenticated `done` is consumed.
  * A queued command is harmless; non-owner data can never reach the core.
  */
class BitNetResidentSharedDynamicQuantOwnerRouter(
    cfg: BitNetSharedDynamicI8QuantizerConfig =
      BitNetSharedDynamicI8QuantizerConfig()) extends Component {
  val io = new Bundle {
    val projectionImage = BitNetResidentSharedDynamicQuantClientPort(cfg)
    val qkvHead = BitNetResidentSharedDynamicQuantClientPort(cfg)
    val lmHead = BitNetResidentSharedDynamicQuantClientPort(cfg)
    val arithmetic = BitNetResidentSharedDynamicQuantArithmeticPort(cfg)

    val ownerValid = out Bool()
    val owner = out(BitNetResidentSharedDynamicQuantOwner())
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
  }

  val ownerValid = RegInit(False)
  val owner = Reg(BitNetResidentSharedDynamicQuantOwner()) init
    BitNetResidentSharedDynamicQuantOwner.ProjectionImage
  // After a projection grant, LM wins the next simultaneous idle request.
  val preferLmHead = RegInit(False)
  val stickyFault = RegInit(False)
  val quarantined = RegInit(False)

  val heldTag = Reg(UInt(cfg.tagWidth bits)) init 0
  val heldEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val heldLength = Reg(UInt(cfg.lengthWidth bits)) init 0
  val heldMask = Reg(Bits(cfg.tokenLanes bits)) init 0
  val outputCount = Reg(UInt(cfg.lengthWidth bits)) init 0

  val ownsProjection = ownerValid &&
    owner === BitNetResidentSharedDynamicQuantOwner.ProjectionImage
  val ownsQkvHead = ownerValid &&
    owner === BitNetResidentSharedDynamicQuantOwner.QkvHead
  val ownsLmHead = ownerValid &&
    owner === BitNetResidentSharedDynamicQuantOwner.LmHead

  val chooseQkvHead = !ownerValid && !quarantined &&
    io.qkvHead.command.valid
  val chooseProjection = !ownerValid && !quarantined && !chooseQkvHead &&
    io.projectionImage.command.valid &&
    (!io.lmHead.command.valid || !preferLmHead)
  val chooseLmHead = !ownerValid && !quarantined && !chooseQkvHead &&
    io.lmHead.command.valid &&
    (!io.projectionImage.command.valid || preferLmHead)

  io.arithmetic.command.valid :=
    chooseProjection || chooseQkvHead || chooseLmHead
  io.arithmetic.command.payload := io.projectionImage.command.payload
  when(chooseQkvHead) {
    io.arithmetic.command.payload := io.qkvHead.command.payload
  } elsewhen(chooseLmHead) {
    io.arithmetic.command.payload := io.lmHead.command.payload
  }
  io.projectionImage.command.ready :=
    chooseProjection && io.arithmetic.command.ready
  io.qkvHead.command.ready := chooseQkvHead && io.arithmetic.command.ready
  io.lmHead.command.ready := chooseLmHead && io.arithmetic.command.ready

  when(io.arithmetic.command.fire) {
    ownerValid := True
    heldTag := io.arithmetic.command.tag
    heldEpoch := io.arithmetic.command.epoch
    heldLength := io.arithmetic.command.vectorLength
    heldMask := io.arithmetic.command.tokenMask
    outputCount := 0
    when(chooseQkvHead) {
      owner := BitNetResidentSharedDynamicQuantOwner.QkvHead
    } elsewhen(chooseProjection) {
      owner := BitNetResidentSharedDynamicQuantOwner.ProjectionImage
      preferLmHead := True
    } otherwise {
      owner := BitNetResidentSharedDynamicQuantOwner.LmHead
      preferLmHead := False
    }
  }

  // The owner is the only source which can advance either forward input.
  io.arithmetic.measure.valid := !quarantined &&
    ((ownsProjection && io.projectionImage.measure.valid) ||
      (ownsQkvHead && io.qkvHead.measure.valid) ||
      (ownsLmHead && io.lmHead.measure.valid))
  io.arithmetic.measure.payload := io.projectionImage.measure.payload
  when(ownsQkvHead) {
    io.arithmetic.measure.payload := io.qkvHead.measure.payload
  } elsewhen(ownsLmHead) {
    io.arithmetic.measure.payload := io.lmHead.measure.payload
  }
  io.projectionImage.measure.ready := ownsProjection && !quarantined &&
    io.arithmetic.measure.ready
  io.qkvHead.measure.ready := ownsQkvHead && !quarantined &&
    io.arithmetic.measure.ready
  io.lmHead.measure.ready := ownsLmHead && !quarantined &&
    io.arithmetic.measure.ready

  io.arithmetic.replay.valid := !quarantined &&
    ((ownsProjection && io.projectionImage.replay.valid) ||
      (ownsQkvHead && io.qkvHead.replay.valid) ||
      (ownsLmHead && io.lmHead.replay.valid))
  io.arithmetic.replay.payload := io.projectionImage.replay.payload
  when(ownsQkvHead) {
    io.arithmetic.replay.payload := io.qkvHead.replay.payload
  } elsewhen(ownsLmHead) {
    io.arithmetic.replay.payload := io.lmHead.replay.payload
  }
  io.projectionImage.replay.ready := ownsProjection && !quarantined &&
    io.arithmetic.replay.ready
  io.qkvHead.replay.ready := ownsQkvHead && !quarantined &&
    io.arithmetic.replay.ready
  io.lmHead.replay.ready := ownsLmHead && !quarantined &&
    io.arithmetic.replay.ready

  // Reverse traffic is identity-checked before it becomes client-visible.
  val replayRequestIdentity = ownerValid &&
    io.arithmetic.replayRequest.tag === heldTag &&
    io.arithmetic.replayRequest.epoch === heldEpoch &&
    io.arithmetic.replayRequest.vectorLength === heldLength &&
    io.arithmetic.replayRequest.tokenMask === heldMask
  val replayRequestLegal = !quarantined && replayRequestIdentity
  io.projectionImage.replayRequest.valid :=
    io.arithmetic.replayRequest.valid && replayRequestLegal && ownsProjection
  io.qkvHead.replayRequest.valid :=
    io.arithmetic.replayRequest.valid && replayRequestLegal && ownsQkvHead
  io.lmHead.replayRequest.valid :=
    io.arithmetic.replayRequest.valid && replayRequestLegal && ownsLmHead
  io.projectionImage.replayRequest.payload :=
    io.arithmetic.replayRequest.payload
  io.qkvHead.replayRequest.payload := io.arithmetic.replayRequest.payload
  io.lmHead.replayRequest.payload := io.arithmetic.replayRequest.payload
  io.arithmetic.replayRequest.ready := replayRequestLegal && Mux(
    ownsProjection,
    io.projectionImage.replayRequest.ready,
    Mux(ownsQkvHead, io.qkvHead.replayRequest.ready,
      io.lmHead.replayRequest.ready))

  val expectedOutputFirst = outputCount === 0
  val expectedOutputLast = outputCount === (heldLength - 1).resized
  val outputIdentity = ownerValid &&
    io.arithmetic.output.tag === heldTag &&
    io.arithmetic.output.epoch === heldEpoch &&
    io.arithmetic.output.featureIndex === outputCount.resized &&
    io.arithmetic.output.tokenMask === heldMask &&
    io.arithmetic.output.first === expectedOutputFirst &&
    io.arithmetic.output.last === expectedOutputLast
  val outputLegal = !quarantined && outputIdentity
  io.projectionImage.output.valid :=
    io.arithmetic.output.valid && outputLegal && ownsProjection
  io.qkvHead.output.valid :=
    io.arithmetic.output.valid && outputLegal && ownsQkvHead
  io.lmHead.output.valid :=
    io.arithmetic.output.valid && outputLegal && ownsLmHead
  io.projectionImage.output.payload := io.arithmetic.output.payload
  io.qkvHead.output.payload := io.arithmetic.output.payload
  io.lmHead.output.payload := io.arithmetic.output.payload
  io.arithmetic.output.ready := outputLegal && Mux(
    ownsProjection,
    io.projectionImage.output.ready,
    Mux(ownsQkvHead, io.qkvHead.output.ready, io.lmHead.output.ready))
  when(io.arithmetic.output.fire) {
    outputCount := outputCount + 1
  }

  val doneIdentity = ownerValid &&
    io.arithmetic.done.tag === heldTag &&
    io.arithmetic.done.epoch === heldEpoch &&
    io.arithmetic.done.vectorLength === heldLength &&
    io.arithmetic.done.emittedFeatures <= heldLength &&
    (!io.arithmetic.done.success ||
      (io.arithmetic.done.emittedFeatures === heldLength &&
        outputCount === heldLength))
  val doneLegal = !quarantined && doneIdentity
  io.projectionImage.done.valid :=
    io.arithmetic.done.valid && doneLegal && ownsProjection
  io.qkvHead.done.valid :=
    io.arithmetic.done.valid && doneLegal && ownsQkvHead
  io.lmHead.done.valid :=
    io.arithmetic.done.valid && doneLegal && ownsLmHead
  io.projectionImage.done.payload := io.arithmetic.done.payload
  io.qkvHead.done.payload := io.arithmetic.done.payload
  io.lmHead.done.payload := io.arithmetic.done.payload
  io.arithmetic.done.ready := doneLegal && Mux(
    ownsProjection,
    io.projectionImage.done.ready,
    Mux(ownsQkvHead, io.qkvHead.done.ready, io.lmHead.done.ready))
  when(io.arithmetic.done.fire) {
    ownerValid := False
  }

  // Bad reverse identity is reset-only quarantined: it cannot be attributed
  // to the following command.  Benign non-owner source traffic is refused and
  // diagnosed, while the real owner is still allowed to drain.
  when(io.arithmetic.replayRequest.valid && !replayRequestLegal) {
    stickyFault := True
    quarantined := True
  }
  when(io.arithmetic.output.valid && !outputLegal) {
    stickyFault := True
    quarantined := True
  }
  when(io.arithmetic.done.valid && !doneLegal) {
    stickyFault := True
    quarantined := True
  }
  when(io.projectionImage.measure.valid && !ownsProjection) {
    stickyFault := True
  }
  when(io.qkvHead.measure.valid && !ownsQkvHead) { stickyFault := True }
  when(io.lmHead.measure.valid && !ownsLmHead) {
    stickyFault := True
  }
  when(io.projectionImage.replay.valid && !ownsProjection) {
    stickyFault := True
  }
  when(io.qkvHead.replay.valid && !ownsQkvHead) { stickyFault := True }
  when(io.lmHead.replay.valid && !ownsLmHead) {
    stickyFault := True
  }

  io.ownerValid := ownerValid
  io.owner := owner
  io.protocolFaultSticky := stickyFault
  io.quarantined := quarantined
}

/** The model-wide service which physically owns the only dynamic-i8 core. */
class BitNetResidentSharedDynamicQuantService(
    cfg: BitNetSharedDynamicI8QuantizerConfig =
      BitNetSharedDynamicI8QuantizerConfig()) extends Component {
  val io = new Bundle {
    val projectionImage = BitNetResidentSharedDynamicQuantClientPort(cfg)
    val qkvHead = BitNetResidentSharedDynamicQuantClientPort(cfg)
    val lmHead = BitNetResidentSharedDynamicQuantClientPort(cfg)

    val ownerValid = out Bool()
    val owner = out(BitNetResidentSharedDynamicQuantOwner())
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
    val completedCommands = out UInt(32 bits)
  }

  val router = new BitNetResidentSharedDynamicQuantOwnerRouter(cfg)
  val sharedQuantizer = new BitNetSharedDynamicI8Quantizer(cfg)

  router.io.projectionImage.command << io.projectionImage.command
  router.io.projectionImage.measure << io.projectionImage.measure
  io.projectionImage.replayRequest <<
    router.io.projectionImage.replayRequest
  router.io.projectionImage.replay << io.projectionImage.replay
  io.projectionImage.output << router.io.projectionImage.output
  io.projectionImage.done << router.io.projectionImage.done

  router.io.qkvHead.command << io.qkvHead.command
  router.io.qkvHead.measure << io.qkvHead.measure
  io.qkvHead.replayRequest << router.io.qkvHead.replayRequest
  router.io.qkvHead.replay << io.qkvHead.replay
  io.qkvHead.output << router.io.qkvHead.output
  io.qkvHead.done << router.io.qkvHead.done

  router.io.lmHead.command << io.lmHead.command
  router.io.lmHead.measure << io.lmHead.measure
  io.lmHead.replayRequest << router.io.lmHead.replayRequest
  router.io.lmHead.replay << io.lmHead.replay
  io.lmHead.output << router.io.lmHead.output
  io.lmHead.done << router.io.lmHead.done

  sharedQuantizer.io.command << router.io.arithmetic.command
  sharedQuantizer.io.measureInput << router.io.arithmetic.measure
  router.io.arithmetic.replayRequest << sharedQuantizer.io.replayRequest
  sharedQuantizer.io.replayInput << router.io.arithmetic.replay
  router.io.arithmetic.output << sharedQuantizer.io.output
  router.io.arithmetic.done << sharedQuantizer.io.done

  io.ownerValid := router.io.ownerValid
  io.owner := router.io.owner
  io.busy := router.io.ownerValid || sharedQuantizer.io.busy
  io.protocolFaultSticky := router.io.protocolFaultSticky ||
    sharedQuantizer.io.protocolFaultSticky
  io.quarantined := router.io.quarantined
  io.completedCommands := sharedQuantizer.io.completedCommands
}

object GenerateBitNetResidentSharedDynamicQuantService extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-shared-dynamic-quant-service")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentSharedDynamicQuantService())
}
