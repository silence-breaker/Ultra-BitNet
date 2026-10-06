package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Semantic stages of one resident transformer layer.
  *
  * This is deliberately a control contract, not an assertion that every
  * corresponding arithmetic datapath is already integrated.  In particular,
  * ExactRelu2Gate names the deployed Ultra-BitNet operation
  * `g > 0 ? g*g*u : 0`; BitNet's Qwen SiLU is not numerically
  * interchangeable with this model contract.
  */
object BitNetResidentLayerStage extends SpinalEnum(binarySequential) {
  val AttentionRmsNorm,
      QkvProjection,
      RopeKvAppend,
      ResidentAttention,
      AttentionSubRmsNorm,
      AttentionOutputProjection,
      AttentionResidual,
      FfnRmsNorm,
      GateUpProjection,
      ExactRelu2Gate,
      DownProjection,
      FfnResidual,
      FinalRmsNorm = newElement()

  val LayerOrder: Vector[E] = Vector(
    AttentionRmsNorm,
    QkvProjection,
    RopeKvAppend,
    ResidentAttention,
    AttentionSubRmsNorm,
    AttentionOutputProjection,
    AttentionResidual,
    FfnRmsNorm,
    GateUpProjection,
    ExactRelu2Gate,
    DownProjection,
    FfnResidual
  )

  val LayerStageCount: Int = LayerOrder.length
}

/** Physical owner expected to accept a resident-layer step. */
object BitNetResidentLayerUnit extends SpinalEnum(binarySequential) {
  val RmsNorm,
      MatrixArray,
      QkvPostprocess,
      Attention,
      Residual,
      Relu2Gate = newElement()
}

object BitNetResidentLayerRunStatus extends SpinalEnum(binarySequential) {
  val Ok,
      InvalidCommand,
      StepFault,
      CompletionMismatch = newElement()
}

object BitNetResidentLayerGeometry {
  def stepIdWidth(cfg: BitNetConfig): Int =
    scala.math.max(1, log2Up(cfg.layerCount *
      BitNetResidentLayerStage.LayerStageCount + 1))

  def completedLayerWidth(cfg: BitNetConfig): Int =
    scala.math.max(1, log2Up(cfg.layerCount + 1))
}

/**
  * One resident token-round submission.
  *
  * Prefill rounds contain one to tokenParallelism contiguous low lanes.
  * Decode contains exactly lane zero.  `positionBase` is the absolute sequence
  * position of lane zero; causal lengths are derived in PL, so software cannot
  * accidentally expose a later prompt lane.  Decode always requests an LM
  * handoff.  Prefill requests it only for `lastPromptRound`.
  */
case class BitNetResidentLayerRun(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val mode = BitNetMode()
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val tokenCount = UInt(cfg.roundTokenCountWidth bits)
  val positionBase = UInt(cfg.sequenceWidth bits)
  val activeTokenMask = Bits(cfg.tokenParallelism bits)
  val resetKv = Bool()
  val lastPromptRound = Bool()
}

/** One completion-tagged operation issued by the 30-layer sequencer. */
case class BitNetResidentLayerStep(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg) bits)
  val mode = BitNetMode()
  val stage = BitNetResidentLayerStage()
  val unit = BitNetResidentLayerUnit()
  val layer = UInt(cfg.layerWidth bits)
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val tokenCount = UInt(cfg.roundTokenCountWidth bits)
  val positionBase = UInt(cfg.sequenceWidth bits)
  val laneCausalLength = Vec(UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  val activeTokenMask = Bits(cfg.tokenParallelism bits)
  val resetKv = Bool()
  val firstLayer = Bool()
  val lastLayer = Bool()
}

/** A unit must echo this identity exactly before the next step can issue. */
case class BitNetResidentLayerStepCompletion(cfg: BitNetConfig)
    extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg) bits)
  val stage = BitNetResidentLayerStage()
  val layer = UInt(cfg.layerWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/**
  * Ownership transfer after layer 29.  The LM-head implementation remains a
  * separate streamed consumer and must report its own eventual completion at
  * the top level; accepting this handoff does not claim that logits exist.
  */
case class BitNetResidentLmHeadHandoff(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val mode = BitNetMode()
  val finalLayer = UInt(cfg.layerWidth bits)
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val tokenLane = UInt(cfg.roundTokenCountWidth bits)
  val tokenPosition = UInt(cfg.sequenceWidth bits)
  /** Final normalized activation remains in this PL-resident scratch buffer. */
  val activationBuffer = BitNetPhysicalActivationBuffer()
}

/** Completion of sequencing only, not completion of the handed-off LM head. */
case class BitNetResidentLayerRunCompletion(cfg: BitNetConfig)
    extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val mode = BitNetMode()
  val status = BitNetResidentLayerRunStatus()
  val completedLayers = UInt(
    BitNetResidentLayerGeometry.completedLayerWidth(cfg) bits)
  val failedLayer = UInt(cfg.layerWidth bits)
  val failedStage = BitNetResidentLayerStage()
  val lmHeadHandedOff = Bool()
  val faultCode = Bits(8 bits)
}

/** Pure-Scala schedule oracle shared by tests and integration tooling. */
case class BitNetResidentLayerPlanStep(
    layer: Int,
    stage: BitNetResidentLayerStage.E,
    stepId: Int,
    resetKv: Boolean,
    firstLayer: Boolean,
    lastLayer: Boolean
)

object BitNetResidentLayerOracle {
  def plan(layerCount: Int, resetKv: Boolean): Vector[BitNetResidentLayerPlanStep] = {
    require(layerCount > 0)
    (0 until layerCount).toVector.flatMap { layer =>
      BitNetResidentLayerStage.LayerOrder.zipWithIndex.map {
        case (stage, ordinal) =>
          BitNetResidentLayerPlanStep(
            layer = layer,
            stage = stage,
            stepId = layer * BitNetResidentLayerStage.LayerStageCount + ordinal,
            resetKv = resetKv && layer == 0 &&
              stage == BitNetResidentLayerStage.RopeKvAppend,
            firstLayer = layer == 0,
            lastLayer = layer == layerCount - 1
          )
      }
    }
  }

  def requiresLmHead(decode: Boolean, lastPromptRound: Boolean): Boolean =
    decode || lastPromptRound
}

/**
  * Completion-driven 30-layer resident micro-sequencer.
  *
  * Exactly one semantic step may be outstanding.  The next step is selected
  * only after a matching successful completion, which makes residual and
  * activation liveness barriers explicit.  A failed or mismatched completion
  * immediately suppresses all further layer/LM requests and produces one
  * backpressure-safe failed run completion.
  */
class BitNetResidentLayerSequencer(
    cfg: BitNetConfig = BitNetConfig.Production
) extends Component {
  // Geometry is fixed at elaboration just like BitNet's generated model
  // controller.  Keeping cfg parametric permits a small bit-accurate full-flow
  // test image without reintroducing a runtime microprogram or descriptor.

  val io = new Bundle {
    val command = slave(Stream(BitNetResidentLayerRun(cfg)))
    val step = master(Stream(BitNetResidentLayerStep(cfg)))
    val stepCompletion = slave(
      Stream(BitNetResidentLayerStepCompletion(cfg)))
    val lmHeadHandoff = master(Stream(BitNetResidentLmHeadHandoff(cfg)))
    val completion = master(Stream(BitNetResidentLayerRunCompletion(cfg)))
    val busy = out Bool()
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssueStep, WaitStep, IssueLmHead, Respond = newElement()
  }

  val state = Reg(State()) init State.Idle
  val requestId = Reg(UInt(cfg.requestIdWidth bits)) init 0
  val mode = Reg(BitNetMode()) init BitNetMode.Decode
  val tokenBase = Reg(UInt(cfg.sequenceWidth bits)) init 0
  val tokenCount = Reg(UInt(cfg.roundTokenCountWidth bits)) init 0
  val positionBase = Reg(UInt(cfg.sequenceWidth bits)) init 0
  val activeTokenMask = Reg(Bits(cfg.tokenParallelism bits)) init 0
  val resetKv = Reg(Bool()) init False
  val lastPromptRound = Reg(Bool()) init False

  val layer = Reg(UInt(cfg.layerWidth bits)) init 0
  val stage = Reg(BitNetResidentLayerStage()) init
    BitNetResidentLayerStage.AttentionRmsNorm
  val stepId = Reg(UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg) bits)) init 0
  val completedLayers = Reg(UInt(
    BitNetResidentLayerGeometry.completedLayerWidth(cfg) bits)) init 0

  val responseStatus = Reg(BitNetResidentLayerRunStatus()) init
    BitNetResidentLayerRunStatus.Ok
  val responseCompletedLayers = Reg(UInt(
    BitNetResidentLayerGeometry.completedLayerWidth(cfg) bits)) init 0
  val responseFailedLayer = Reg(UInt(cfg.layerWidth bits)) init 0
  val responseFailedStage = Reg(BitNetResidentLayerStage()) init
    BitNetResidentLayerStage.AttentionRmsNorm
  val responseLmHeadHandedOff = Reg(Bool()) init False
  val responseFaultCode = Reg(Bits(8 bits)) init 0

  io.busy := state =/= State.Idle
  io.command.ready := state === State.Idle

  io.step.valid := state === State.IssueStep
  io.step.payload.requestId := requestId
  io.step.payload.stepId := stepId
  io.step.payload.mode := mode
  io.step.payload.stage := stage
  io.step.payload.unit.allowOverride
  io.step.payload.unit := BitNetResidentLayerUnit.Residual
  io.step.payload.layer := layer
  io.step.payload.tokenBase := tokenBase
  io.step.payload.tokenCount := tokenCount
  io.step.payload.positionBase := positionBase
  for (laneIndex <- 0 until cfg.tokenParallelism) {
    io.step.payload.laneCausalLength(laneIndex) := 0
    when(activeTokenMask(laneIndex)) {
      io.step.payload.laneCausalLength(laneIndex) :=
        (positionBase + U(laneIndex + 1, cfg.sequenceWidth bits))
          .resize(cfg.sequenceWidth)
    }
  }
  io.step.payload.activeTokenMask := activeTokenMask
  io.step.payload.resetKv := resetKv && layer === 0 &&
    stage === BitNetResidentLayerStage.RopeKvAppend
  io.step.payload.firstLayer := layer === 0
  io.step.payload.lastLayer := layer === cfg.layerCount - 1

  switch(stage) {
    is(BitNetResidentLayerStage.AttentionRmsNorm,
       BitNetResidentLayerStage.AttentionSubRmsNorm,
       BitNetResidentLayerStage.FfnRmsNorm,
       BitNetResidentLayerStage.FinalRmsNorm) {
      io.step.payload.unit := BitNetResidentLayerUnit.RmsNorm
    }
    is(BitNetResidentLayerStage.QkvProjection,
       BitNetResidentLayerStage.AttentionOutputProjection,
       BitNetResidentLayerStage.GateUpProjection,
       BitNetResidentLayerStage.DownProjection) {
      io.step.payload.unit := BitNetResidentLayerUnit.MatrixArray
    }
    is(BitNetResidentLayerStage.RopeKvAppend) {
      io.step.payload.unit := BitNetResidentLayerUnit.QkvPostprocess
    }
    is(BitNetResidentLayerStage.ResidentAttention) {
      io.step.payload.unit := BitNetResidentLayerUnit.Attention
    }
    is(BitNetResidentLayerStage.AttentionResidual,
       BitNetResidentLayerStage.FfnResidual) {
      io.step.payload.unit := BitNetResidentLayerUnit.Residual
    }
    is(BitNetResidentLayerStage.ExactRelu2Gate) {
      io.step.payload.unit := BitNetResidentLayerUnit.Relu2Gate
    }
  }

  val finalTokenLane = UInt(cfg.roundTokenCountWidth bits)
  finalTokenLane := 0
  when(mode === BitNetMode.Prefill) {
    finalTokenLane := tokenCount - 1
  }

  io.lmHeadHandoff.valid := state === State.IssueLmHead
  io.lmHeadHandoff.payload.requestId := requestId
  io.lmHeadHandoff.payload.mode := mode
  io.lmHeadHandoff.payload.finalLayer := layer
  io.lmHeadHandoff.payload.tokenBase := tokenBase
  io.lmHeadHandoff.payload.tokenLane := finalTokenLane
  io.lmHeadHandoff.payload.tokenPosition :=
    (positionBase + finalTokenLane.resize(cfg.sequenceWidth))
      .resize(cfg.sequenceWidth)
  io.lmHeadHandoff.payload.activationBuffer :=
    BitNetPhysicalActivationBuffer.NarrowWork

  io.stepCompletion.ready := state === State.WaitStep

  io.completion.valid := state === State.Respond
  io.completion.payload.requestId := requestId
  io.completion.payload.mode := mode
  io.completion.payload.status := responseStatus
  io.completion.payload.completedLayers := responseCompletedLayers
  io.completion.payload.failedLayer := responseFailedLayer
  io.completion.payload.failedStage := responseFailedStage
  io.completion.payload.lmHeadHandedOff := responseLmHeadHandedOff
  io.completion.payload.faultCode := responseFaultCode

  val expectedMask = Bits(cfg.tokenParallelism bits)
  for (laneIndex <- 0 until cfg.tokenParallelism) {
    expectedMask(laneIndex) := io.command.payload.tokenCount > laneIndex
  }
  val positionEnd = io.command.payload.positionBase
    .resize(cfg.sequenceWidth + 1) +
    io.command.payload.tokenCount.resize(cfg.sequenceWidth + 1)
  val commonCommandValid = io.command.payload.tokenCount =/= 0 &&
    io.command.payload.tokenCount <= cfg.tokenParallelism &&
    io.command.payload.activeTokenMask === expectedMask &&
    positionEnd <= cfg.maxSequenceLength &&
    (!io.command.payload.resetKv || io.command.payload.positionBase === 0)
  val modeCommandValid =
    (io.command.payload.mode === BitNetMode.Decode &&
      io.command.payload.tokenCount === 1 &&
      io.command.payload.activeTokenMask === B(1, cfg.tokenParallelism bits)) ||
    (io.command.payload.mode === BitNetMode.Prefill)
  val commandValid = commonCommandValid && modeCommandValid

  def prepareResponse(
      statusValue: BitNetResidentLayerRunStatus.E,
      completedValue: UInt,
      lmHeadValue: Bool,
      faultValue: Bits
  ): Unit = {
    responseStatus := statusValue
    responseCompletedLayers := completedValue.resized
    responseFailedLayer := layer
    responseFailedStage := stage
    responseLmHeadHandedOff := lmHeadValue
    responseFaultCode := faultValue
  }

  when(io.command.fire) {
    requestId := io.command.payload.requestId
    mode := io.command.payload.mode
    tokenBase := io.command.payload.tokenBase
    tokenCount := io.command.payload.tokenCount
    positionBase := io.command.payload.positionBase
    activeTokenMask := io.command.payload.activeTokenMask
    resetKv := io.command.payload.resetKv
    lastPromptRound := io.command.payload.lastPromptRound
    layer := 0
    stage := BitNetResidentLayerStage.AttentionRmsNorm
    stepId := 0
    completedLayers := 0
    responseStatus := BitNetResidentLayerRunStatus.Ok
    responseCompletedLayers := 0
    responseFailedLayer := 0
    responseFailedStage := BitNetResidentLayerStage.AttentionRmsNorm
    responseLmHeadHandedOff := False
    responseFaultCode := 0
    when(commandValid) {
      state := State.IssueStep
    } otherwise {
      responseStatus := BitNetResidentLayerRunStatus.InvalidCommand
      responseFaultCode := B(0xe0, 8 bits)
      state := State.Respond
    }
  }

  when(state === State.IssueStep) {
    when(io.step.fire) {
      state := State.WaitStep
    }
  }

  val completionMatches =
    io.stepCompletion.payload.requestId === requestId &&
      io.stepCompletion.payload.stepId === stepId &&
      io.stepCompletion.payload.stage === stage &&
      io.stepCompletion.payload.layer === layer

  when(state === State.WaitStep && io.stepCompletion.fire) {
    when(!completionMatches) {
      prepareResponse(
        BitNetResidentLayerRunStatus.CompletionMismatch,
        completedLayers,
        False,
        Mux(io.stepCompletion.payload.faultCode.orR,
          io.stepCompletion.payload.faultCode, B(0xf0, 8 bits))
      )
      state := State.Respond
    } elsewhen (!io.stepCompletion.payload.success) {
      prepareResponse(
        BitNetResidentLayerRunStatus.StepFault,
        completedLayers,
        False,
        io.stepCompletion.payload.faultCode
      )
      state := State.Respond
    } otherwise {
      switch(stage) {
        is(BitNetResidentLayerStage.AttentionRmsNorm) {
          stage := BitNetResidentLayerStage.QkvProjection
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.QkvProjection) {
          stage := BitNetResidentLayerStage.RopeKvAppend
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.RopeKvAppend) {
          stage := BitNetResidentLayerStage.ResidentAttention
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.ResidentAttention) {
          stage := BitNetResidentLayerStage.AttentionSubRmsNorm
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.AttentionSubRmsNorm) {
          stage := BitNetResidentLayerStage.AttentionOutputProjection
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.AttentionOutputProjection) {
          stage := BitNetResidentLayerStage.AttentionResidual
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.AttentionResidual) {
          stage := BitNetResidentLayerStage.FfnRmsNorm
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.FfnRmsNorm) {
          stage := BitNetResidentLayerStage.GateUpProjection
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.GateUpProjection) {
          stage := BitNetResidentLayerStage.ExactRelu2Gate
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.ExactRelu2Gate) {
          stage := BitNetResidentLayerStage.DownProjection
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.DownProjection) {
          stage := BitNetResidentLayerStage.FfnResidual
          stepId := stepId + 1
          state := State.IssueStep
        }
        is(BitNetResidentLayerStage.FfnResidual) {
          completedLayers := completedLayers + 1
          when(layer === cfg.layerCount - 1) {
            when(mode === BitNetMode.Decode || lastPromptRound) {
              stage := BitNetResidentLayerStage.FinalRmsNorm
              stepId := stepId + 1
              state := State.IssueStep
            } otherwise {
              prepareResponse(
                BitNetResidentLayerRunStatus.Ok,
                U(cfg.layerCount,
                  BitNetResidentLayerGeometry.completedLayerWidth(cfg) bits),
                False,
                B(0, 8 bits)
              )
              state := State.Respond
            }
          } otherwise {
            layer := layer + 1
            stage := BitNetResidentLayerStage.AttentionRmsNorm
            stepId := stepId + 1
            state := State.IssueStep
          }
        }
        is(BitNetResidentLayerStage.FinalRmsNorm) {
          state := State.IssueLmHead
        }
      }
    }
  }

  when(state === State.IssueLmHead) {
    when(io.lmHeadHandoff.fire) {
      prepareResponse(
        BitNetResidentLayerRunStatus.Ok,
        completedLayers,
        True,
        B(0, 8 bits)
      )
      state := State.Respond
    }
  }

  when(io.completion.fire) {
    state := State.Idle
  }
}

object GenerateBitNetResidentLayerSequencer extends App {
  SpinalConfig(
    targetDirectory = "build/bitnet-resident-layer-sequencer",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetResidentLayerSequencer())
}
