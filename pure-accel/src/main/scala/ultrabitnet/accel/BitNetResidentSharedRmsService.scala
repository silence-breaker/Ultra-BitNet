package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** The two resident activation coordinators which share one physical RMS. */
object BitNetResidentSharedRmsOwner extends SpinalEnum(binarySequential) {
  val LayerRms, FfnPostGate = newElement()
}

/**
  * Client-side view of the model-wide RMS arithmetic service.
  *
  * Both resident coordinators use exactly this type.  Commands, measure data
  * and replay data flow into the service; replay requests, normalized results
  * and completion flow back to the command owner.
  */
case class BitNetResidentSharedRmsClientPort(
    cfg: BitNetSharedSpuRmsNormConfig) extends Bundle {
  val command = slave(Stream(BitNetSharedRmsCommand(cfg)))
  val measure = slave(Stream(BitNetSharedRmsMeasureBeat(cfg)))
  val replayRequest = master(Stream(BitNetSharedRmsReplayRequest(cfg)))
  val replay = slave(Stream(BitNetSharedRmsReplayBeat(cfg)))
  val result = master(Stream(BitNetSharedRmsResultBeat(cfg)))
  val done = master(Stream(BitNetSharedRmsDone(cfg)))
}

/** Direction of the single physical arithmetic core as seen by the router. */
case class BitNetResidentSharedRmsArithmeticPort(
    cfg: BitNetSharedSpuRmsNormConfig) extends Bundle {
  val command = master(Stream(BitNetSharedRmsCommand(cfg)))
  val measure = master(Stream(BitNetSharedRmsMeasureBeat(cfg)))
  val replayRequest = slave(Stream(BitNetSharedRmsReplayRequest(cfg)))
  val replay = master(Stream(BitNetSharedRmsReplayBeat(cfg)))
  val result = slave(Stream(BitNetSharedRmsResultBeat(cfg)))
  val done = slave(Stream(BitNetSharedRmsDone(cfg)))
}

/**
  * Command-granular owner router for the unique RMS arithmetic instance.
  *
  * Ownership is acquired on the physical command handshake and is not
  * released until that owner's done beat is accepted.  A malformed or early
  * arithmetic-side reverse beat enters a reset-only quarantine: it is never
  * exposed to either client, no later command is admitted, and therefore the
  * error cannot be mistaken for the following client's transaction.
  */
class BitNetResidentSharedRmsOwnerRouter(
    cfg: BitNetSharedSpuRmsNormConfig =
      BitNetSharedSpuRmsNormConfig()) extends Component {
  val io = new Bundle {
    val layerRms = BitNetResidentSharedRmsClientPort(cfg)
    val ffnPostGate = BitNetResidentSharedRmsClientPort(cfg)
    val arithmetic = BitNetResidentSharedRmsArithmeticPort(cfg)

    val ownerValid = out Bool()
    val owner = out(BitNetResidentSharedRmsOwner())
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
  }

  object Phase extends SpinalEnum(binarySequential) {
    val Idle, Measure, WaitReplayRequest, Replay, WaitDone = newElement()
  }

  val phase = Reg(Phase()) init Phase.Idle
  val ownerValid = RegInit(False)
  val owner = Reg(BitNetResidentSharedRmsOwner()) init
    BitNetResidentSharedRmsOwner.LayerRms
  // True means that FfnPostGate wins the next simultaneous idle request.
  val preferFfnPostGate = RegInit(False)
  val protocolFaultSticky = RegInit(False)
  val quarantined = RegInit(False)

  val heldLength = Reg(UInt(cfg.lengthWidth bits)) init 0
  val heldTokenMask = Reg(Bits(cfg.tokenLanes bits)) init 0
  val heldSource = Reg(BitNetPhysicalActivationBuffer()) init
    BitNetPhysicalActivationBuffer.HiddenResident
  val heldTag = Reg(UInt(cfg.tagWidth bits)) init 0
  val measureCount = Reg(UInt(cfg.lengthWidth bits)) init 0
  val replayCount = Reg(UInt(cfg.lengthWidth bits)) init 0
  val resultCount = Reg(UInt(cfg.lengthWidth bits)) init 0

  val ownsLayer = ownerValid &&
    owner === BitNetResidentSharedRmsOwner.LayerRms
  val ownsFfn = ownerValid &&
    owner === BitNetResidentSharedRmsOwner.FfnPostGate

  /* Fair idle-only command arbitration.  A queued command is not an error. */
  val chooseLayer = !ownerValid && !quarantined &&
    io.layerRms.command.valid &&
    (!io.ffnPostGate.command.valid || !preferFfnPostGate)
  val chooseFfn = !ownerValid && !quarantined &&
    io.ffnPostGate.command.valid &&
    (!io.layerRms.command.valid || preferFfnPostGate)

  io.arithmetic.command.valid := chooseLayer || chooseFfn
  io.arithmetic.command.payload := io.layerRms.command.payload
  when(chooseFfn) {
    io.arithmetic.command.payload := io.ffnPostGate.command.payload
  }
  io.layerRms.command.ready := chooseLayer && io.arithmetic.command.ready
  io.ffnPostGate.command.ready := chooseFfn && io.arithmetic.command.ready

  val selectedCommand = BitNetSharedRmsCommand(cfg)
  selectedCommand := io.layerRms.command.payload
  when(chooseFfn) { selectedCommand := io.ffnPostGate.command.payload }
  val selectedCommandLegal = selectedCommand.length =/= 0 &&
    selectedCommand.length <= cfg.maxLength &&
    selectedCommand.tokenMask.orR

  when(io.arithmetic.command.fire) {
    ownerValid := True
    heldLength := selectedCommand.length
    heldTokenMask := selectedCommand.tokenMask
    heldSource := selectedCommand.sourceBuffer
    heldTag := selectedCommand.tag
    measureCount := 0
    replayCount := 0
    resultCount := 0
    when(chooseLayer) {
      owner := BitNetResidentSharedRmsOwner.LayerRms
      preferFfnPostGate := True
    } otherwise {
      owner := BitNetResidentSharedRmsOwner.FfnPostGate
      preferFfnPostGate := False
    }
    when(selectedCommandLegal) {
      phase := Phase.Measure
    } otherwise {
      // The physical core reports an invalid command directly through done.
      phase := Phase.WaitDone
    }
  }

  /* Client -> core measure path. */
  val layerMeasureExpected = ownsLayer && phase === Phase.Measure &&
    !quarantined
  val ffnMeasureExpected = ownsFfn && phase === Phase.Measure &&
    !quarantined
  io.arithmetic.measure.valid :=
    (layerMeasureExpected && io.layerRms.measure.valid) ||
      (ffnMeasureExpected && io.ffnPostGate.measure.valid)
  io.arithmetic.measure.payload := io.layerRms.measure.payload
  when(ownsFfn) {
    io.arithmetic.measure.payload := io.ffnPostGate.measure.payload
  }
  io.layerRms.measure.ready :=
    layerMeasureExpected && io.arithmetic.measure.ready
  io.ffnPostGate.measure.ready :=
    ffnMeasureExpected && io.arithmetic.measure.ready

  val expectedMeasureFirst = measureCount === 0
  val expectedMeasureLast =
    measureCount === (heldLength - 1).resized
  when(io.arithmetic.measure.fire) {
    when(io.arithmetic.measure.tokenMask =/= heldTokenMask ||
        io.arithmetic.measure.first =/= expectedMeasureFirst ||
        io.arithmetic.measure.last =/= expectedMeasureLast) {
      protocolFaultSticky := True
    }
    when(expectedMeasureLast) {
      phase := Phase.WaitReplayRequest
    } otherwise {
      measureCount := measureCount + 1
    }
  }

  /* Core -> owner replay-request path, including full metadata isolation. */
  val replayRequestPhaseLegal = ownerValid &&
    phase === Phase.WaitReplayRequest && !quarantined
  val replayRequestMetadataLegal =
    io.arithmetic.replayRequest.length === heldLength &&
      io.arithmetic.replayRequest.tokenMask === heldTokenMask &&
      io.arithmetic.replayRequest.sourceBuffer === heldSource &&
      io.arithmetic.replayRequest.tag === heldTag &&
      io.arithmetic.replayRequest.inPlace
  val replayRequestLegal = replayRequestPhaseLegal &&
    replayRequestMetadataLegal

  io.layerRms.replayRequest.valid :=
    io.arithmetic.replayRequest.valid && replayRequestLegal && ownsLayer
  io.ffnPostGate.replayRequest.valid :=
    io.arithmetic.replayRequest.valid && replayRequestLegal && ownsFfn
  io.layerRms.replayRequest.payload := io.arithmetic.replayRequest.payload
  io.ffnPostGate.replayRequest.payload := io.arithmetic.replayRequest.payload
  io.arithmetic.replayRequest.ready := replayRequestLegal &&
    Mux(ownsLayer, io.layerRms.replayRequest.ready,
      io.ffnPostGate.replayRequest.ready)

  when(io.arithmetic.replayRequest.valid && !replayRequestLegal) {
    protocolFaultSticky := True
    quarantined := True
  }
  when(io.arithmetic.replayRequest.fire) {
    phase := Phase.Replay
  }

  /* Owner -> core replay path. */
  val replayCapacity = replayCount < heldLength
  val layerReplayExpected = ownsLayer && phase === Phase.Replay &&
    replayCapacity && !quarantined
  val ffnReplayExpected = ownsFfn && phase === Phase.Replay &&
    replayCapacity && !quarantined
  io.arithmetic.replay.valid :=
    (layerReplayExpected && io.layerRms.replay.valid) ||
      (ffnReplayExpected && io.ffnPostGate.replay.valid)
  io.arithmetic.replay.payload := io.layerRms.replay.payload
  when(ownsFfn) {
    io.arithmetic.replay.payload := io.ffnPostGate.replay.payload
  }
  io.layerRms.replay.ready :=
    layerReplayExpected && io.arithmetic.replay.ready
  io.ffnPostGate.replay.ready :=
    ffnReplayExpected && io.arithmetic.replay.ready

  val expectedReplayFirst = replayCount === 0
  val expectedReplayLast = replayCount === (heldLength - 1).resized
  when(io.arithmetic.replay.fire) {
    when(io.arithmetic.replay.tokenMask =/= heldTokenMask ||
        io.arithmetic.replay.first =/= expectedReplayFirst ||
        io.arithmetic.replay.last =/= expectedReplayLast) {
      protocolFaultSticky := True
    }
    replayCount := replayCount + 1
  }

  /* Core -> owner result path.  Bad identity is quarantined, never forwarded. */
  val resultPhaseLegal = ownerValid && phase === Phase.Replay &&
    resultCount < heldLength && replayCount > resultCount && !quarantined
  val expectedResultFirst = resultCount === 0
  val expectedResultLast = resultCount === (heldLength - 1).resized
  val resultMetadataLegal =
    io.arithmetic.result.tokenMask === heldTokenMask &&
      io.arithmetic.result.featureIndex === resultCount.resized &&
      io.arithmetic.result.first === expectedResultFirst &&
      io.arithmetic.result.last === expectedResultLast &&
      io.arithmetic.result.tag === heldTag
  val resultLegal = resultPhaseLegal && resultMetadataLegal

  io.layerRms.result.valid :=
    io.arithmetic.result.valid && resultLegal && ownsLayer
  io.ffnPostGate.result.valid :=
    io.arithmetic.result.valid && resultLegal && ownsFfn
  io.layerRms.result.payload := io.arithmetic.result.payload
  io.ffnPostGate.result.payload := io.arithmetic.result.payload
  io.arithmetic.result.ready := resultLegal &&
    Mux(ownsLayer, io.layerRms.result.ready, io.ffnPostGate.result.ready)

  when(io.arithmetic.result.valid && !resultLegal) {
    protocolFaultSticky := True
    quarantined := True
  }
  when(io.arithmetic.result.fire) {
    when(expectedResultLast) {
      phase := Phase.WaitDone
    } otherwise {
      resultCount := resultCount + 1
    }
  }

  /* Core -> owner done path.  This edge is the ownership release barrier. */
  val donePhaseLegal = ownerValid && phase === Phase.WaitDone &&
    !quarantined
  val doneMetadataLegal = io.arithmetic.done.tag === heldTag
  val doneLegal = donePhaseLegal && doneMetadataLegal
  io.layerRms.done.valid :=
    io.arithmetic.done.valid && doneLegal && ownsLayer
  io.ffnPostGate.done.valid :=
    io.arithmetic.done.valid && doneLegal && ownsFfn
  io.layerRms.done.payload := io.arithmetic.done.payload
  io.ffnPostGate.done.payload := io.arithmetic.done.payload
  io.arithmetic.done.ready := doneLegal &&
    Mux(ownsLayer, io.layerRms.done.ready, io.ffnPostGate.done.ready)

  when(io.arithmetic.done.valid && !doneLegal) {
    protocolFaultSticky := True
    quarantined := True
  }
  when(io.arithmetic.done.fire) {
    ownerValid := False
    phase := Phase.Idle
  }

  /* Non-owner or phase-early client data is never consumed. */
  when(io.layerRms.measure.valid && !layerMeasureExpected) {
    protocolFaultSticky := True
  }
  when(io.ffnPostGate.measure.valid && !ffnMeasureExpected) {
    protocolFaultSticky := True
  }
  when(io.layerRms.replay.valid && !layerReplayExpected) {
    protocolFaultSticky := True
  }
  when(io.ffnPostGate.replay.valid && !ffnReplayExpected) {
    protocolFaultSticky := True
  }

  io.ownerValid := ownerValid
  io.owner := owner
  io.protocolFaultSticky := protocolFaultSticky
  io.quarantined := quarantined
}

/**
  * Model-wide physical RMS service.  This is the only component in the
  * resident composition which instantiates RMS arithmetic.
  */
class BitNetResidentSharedRmsService(
    cfg: BitNetSharedSpuRmsNormConfig =
      BitNetSharedSpuRmsNormConfig()) extends Component {
  val io = new Bundle {
    val layerRms = BitNetResidentSharedRmsClientPort(cfg)
    val ffnPostGate = BitNetResidentSharedRmsClientPort(cfg)

    val ownerValid = out Bool()
    val owner = out(BitNetResidentSharedRmsOwner())
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
    val quarantined = out Bool()
  }

  val router = new BitNetResidentSharedRmsOwnerRouter(cfg)
  val sharedRms = new BitNetSharedSpuRmsNorm(cfg)

  router.io.layerRms.command << io.layerRms.command
  router.io.layerRms.measure << io.layerRms.measure
  io.layerRms.replayRequest << router.io.layerRms.replayRequest
  router.io.layerRms.replay << io.layerRms.replay
  io.layerRms.result << router.io.layerRms.result
  io.layerRms.done << router.io.layerRms.done

  router.io.ffnPostGate.command << io.ffnPostGate.command
  router.io.ffnPostGate.measure << io.ffnPostGate.measure
  io.ffnPostGate.replayRequest << router.io.ffnPostGate.replayRequest
  router.io.ffnPostGate.replay << io.ffnPostGate.replay
  io.ffnPostGate.result << router.io.ffnPostGate.result
  io.ffnPostGate.done << router.io.ffnPostGate.done

  sharedRms.io.command << router.io.arithmetic.command
  sharedRms.io.measureInput << router.io.arithmetic.measure
  router.io.arithmetic.replayRequest << sharedRms.io.replayRequest
  sharedRms.io.replayInput << router.io.arithmetic.replay
  router.io.arithmetic.result << sharedRms.io.output
  router.io.arithmetic.done << sharedRms.io.done

  io.ownerValid := router.io.ownerValid
  io.owner := router.io.owner
  io.busy := router.io.ownerValid || sharedRms.io.busy
  io.protocolFaultSticky :=
    router.io.protocolFaultSticky || sharedRms.io.protocolFaultSticky
  io.overflowSticky := sharedRms.io.overflowSticky
  io.quarantined := router.io.quarantined
}

object GenerateBitNetResidentSharedRmsService extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-shared-rms-service")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentSharedRmsService())
}
