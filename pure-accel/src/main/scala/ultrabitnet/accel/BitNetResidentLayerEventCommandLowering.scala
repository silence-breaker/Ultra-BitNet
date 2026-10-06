package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Physical production executor selected by one fused layer event. */
object BitNetResidentLayerEventExecutor extends SpinalEnum(
    binarySequential) {
  val RmsNorm, MatrixArray, Attention, FfnGatePipeline = newElement()
}

/** A lowered command keeps the complete source command as its immutable
  * event identity/context, beside the legacy-complete LayerStep contract.
  */
case class BitNetResidentLayerEventExecutorCommand(
    cfg: BitNetResidentLayerEventSchedulerConfig) extends Bundle {
  val event = BitNetResidentLayerEventCommand(cfg)
  val step = BitNetResidentLayerStep(cfg.model)
  val executor = BitNetResidentLayerEventExecutor()
}

object BitNetResidentLayerEventLoweringFailureReason extends SpinalEnum(
    binarySequential) {
  val InvalidStage, LayerOutOfRange, FinalNormLayerMismatch,
      ResetScopeViolation = newElement()
}

/** Typed, backpressure-safe rejection; an invalid input never appears on the
  * executor stream.
  */
case class BitNetResidentLayerEventLoweringFailure(
    cfg: BitNetResidentLayerEventSchedulerConfig) extends Bundle {
  val event = BitNetResidentLayerEventCommand(cfg)
  val reason = BitNetResidentLayerEventLoweringFailureReason()
}

case class BitNetResidentLayerEventCommandLoweringConfig(
    event: BitNetResidentLayerEventSchedulerConfig =
      BitNetResidentLayerEventSchedulerConfig()) {
  val model: BitNetConfig = event.model
}

/**
  * Stateless production lowering from the eight fused per-layer commits plus
  * final norm into the typed physical-executor contract.
  *
  * This component deliberately contains no arithmetic, storage or scheduler.
  * Backpressure is returned directly from exactly one of `executorCommand`
  * and `failure`.  Consequently an input transaction can neither duplicate
  * across the two outputs nor disappear while its selected consumer stalls.
  */
class BitNetResidentLayerEventCommandLowering(
    cfg: BitNetResidentLayerEventCommandLoweringConfig =
      BitNetResidentLayerEventCommandLoweringConfig()) extends Component {
  private val eventCfg = cfg.event
  private val model = cfg.model
  private val stepIdWidth = BitNetResidentLayerGeometry.stepIdWidth(model)

  val io = new Bundle {
    val eventCommand = slave(Stream(
      BitNetResidentLayerEventCommand(eventCfg)))
    val executorCommand = master(Stream(
      BitNetResidentLayerEventExecutorCommand(eventCfg)))
    val failure = master(Stream(
      BitNetResidentLayerEventLoweringFailure(eventCfg)))
  }

  val stageKnown =
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.AttentionNormReady ||
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.QkvResidentReady ||
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.AttentionContextReady ||
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.AttentionSubNormReady ||
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.AttentionBranchCommit ||
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.FfnNormReady ||
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.FfnDownInputReady ||
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.LayerCommit ||
    io.eventCommand.payload.stage ===
      BitNetResidentLayerEventStage.FinalNormReady
  val layerInRange = io.eventCommand.layer < model.layerCount
  val finalNorm = io.eventCommand.payload.stage ===
    BitNetResidentLayerEventStage.FinalNormReady
  val finalLayerMatches = !finalNorm ||
    io.eventCommand.layer === model.layerCount - 1
  // The event scheduler scopes a requested KV reset to precisely the first
  // layer's fused QKV event.  Lowering rejects any independently fabricated
  // reset marker outside that transaction.
  val resetScopeLegal = !io.eventCommand.resetKv ||
    (io.eventCommand.layer === 0 &&
      io.eventCommand.payload.stage ===
        BitNetResidentLayerEventStage.QkvResidentReady)
  val commandLegal = stageKnown && layerInRange && finalLayerMatches &&
    resetScopeLegal

  io.executorCommand.valid := io.eventCommand.valid && commandLegal
  io.failure.valid := io.eventCommand.valid && !commandLegal
  io.eventCommand.ready := Mux(commandLegal,
    io.executorCommand.ready, io.failure.ready)

  io.executorCommand.payload.event := io.eventCommand.payload
  io.failure.payload.event := io.eventCommand.payload

  val step = io.executorCommand.payload.step
  step.requestId := io.eventCommand.requestId
  val layerTimesEight = (io.eventCommand.layer.resize(stepIdWidth) << 3)
    .resize(stepIdWidth)
  val eventOrdinal =
    io.eventCommand.payload.stage.asBits.asUInt.resize(stepIdWidth)
  step.stepId := (layerTimesEight + eventOrdinal).resize(stepIdWidth)
  when(finalNorm) {
    step.stepId := U(model.layerCount *
      BitNetResidentLayerEventStage.LayerStageCount, stepIdWidth bits)
  }
  step.mode := io.eventCommand.mode
  step.stage.allowOverride
  step.unit.allowOverride
  step.stage := BitNetResidentLayerStage.AttentionRmsNorm
  step.unit := BitNetResidentLayerUnit.RmsNorm
  step.layer := io.eventCommand.layer
  step.tokenBase := io.eventCommand.tokenBase
  step.tokenCount := io.eventCommand.tokenCount
  step.positionBase := io.eventCommand.positionBase
  for (lane <- 0 until model.tokenParallelism) {
    step.laneCausalLength(lane) :=
      io.eventCommand.laneCausalLength(lane)
  }
  step.activeTokenMask := io.eventCommand.activeTokenMask
  step.resetKv := io.eventCommand.resetKv
  step.firstLayer := io.eventCommand.layer === 0
  step.lastLayer := io.eventCommand.layer === model.layerCount - 1

  io.executorCommand.payload.executor.allowOverride
  io.executorCommand.payload.executor :=
    BitNetResidentLayerEventExecutor.RmsNorm
  switch(io.eventCommand.payload.stage) {
    is(BitNetResidentLayerEventStage.AttentionNormReady) {
      step.stage := BitNetResidentLayerStage.AttentionRmsNorm
      step.unit := BitNetResidentLayerUnit.RmsNorm
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.RmsNorm
    }
    is(BitNetResidentLayerEventStage.QkvResidentReady) {
      step.stage := BitNetResidentLayerStage.QkvProjection
      step.unit := BitNetResidentLayerUnit.MatrixArray
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.MatrixArray
    }
    is(BitNetResidentLayerEventStage.AttentionContextReady) {
      step.stage := BitNetResidentLayerStage.ResidentAttention
      step.unit := BitNetResidentLayerUnit.Attention
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.Attention
    }
    is(BitNetResidentLayerEventStage.AttentionSubNormReady) {
      step.stage := BitNetResidentLayerStage.AttentionSubRmsNorm
      step.unit := BitNetResidentLayerUnit.RmsNorm
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.RmsNorm
    }
    is(BitNetResidentLayerEventStage.AttentionBranchCommit) {
      step.stage := BitNetResidentLayerStage.AttentionOutputProjection
      step.unit := BitNetResidentLayerUnit.MatrixArray
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.MatrixArray
    }
    is(BitNetResidentLayerEventStage.FfnNormReady) {
      step.stage := BitNetResidentLayerStage.FfnRmsNorm
      step.unit := BitNetResidentLayerUnit.RmsNorm
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.RmsNorm
    }
    is(BitNetResidentLayerEventStage.FfnDownInputReady) {
      step.stage := BitNetResidentLayerStage.GateUpProjection
      step.unit := BitNetResidentLayerUnit.MatrixArray
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.FfnGatePipeline
    }
    is(BitNetResidentLayerEventStage.LayerCommit) {
      step.stage := BitNetResidentLayerStage.DownProjection
      step.unit := BitNetResidentLayerUnit.MatrixArray
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.MatrixArray
    }
    is(BitNetResidentLayerEventStage.FinalNormReady) {
      step.stage := BitNetResidentLayerStage.FinalRmsNorm
      step.unit := BitNetResidentLayerUnit.RmsNorm
      io.executorCommand.payload.executor :=
        BitNetResidentLayerEventExecutor.RmsNorm
    }
  }

  io.failure.payload.reason :=
    BitNetResidentLayerEventLoweringFailureReason.ResetScopeViolation
  when(!stageKnown) {
    io.failure.payload.reason :=
      BitNetResidentLayerEventLoweringFailureReason.InvalidStage
  } elsewhen(!layerInRange) {
    io.failure.payload.reason :=
      BitNetResidentLayerEventLoweringFailureReason.LayerOutOfRange
  } elsewhen(!finalLayerMatches) {
    io.failure.payload.reason :=
      BitNetResidentLayerEventLoweringFailureReason.
        FinalNormLayerMismatch
  }
}

object GenerateBitNetResidentLayerEventCommandLowering extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-layer-event-command-lowering")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLayerEventCommandLowering())
}
