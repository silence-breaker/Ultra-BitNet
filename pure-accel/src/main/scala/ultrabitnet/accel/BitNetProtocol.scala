package ultrabitnet.accel

import spinal.core._

import scala.language.postfixOps

/** Prefill and decode share the physical engines but use different schedules. */
object BitNetMode extends SpinalEnum {
  val Prefill, Decode = newElement()
}

/**
  * Model stages are statically known, just as BitNet's inst_gen tables
  * are.  These are semantic stages, not software supplied opcodes.
  */
object BitNetStage extends SpinalEnum {
  val LayerInputNorm,
      QkvProjection,
      RopeKvAppend,
      AttentionFused,
      AttentionSubNorm,
      AttentionOutputProjection,
      AttentionResidual,
      PostAttentionNorm,
      GateUpProjection,
      FfnGateSubNorm,
      DownProjection,
      FfnResidual,
      FinalNorm,
      LmHeadProjection,
      LmHeadReduce = newElement()

  val LayerOrder: Vector[E] = Vector(
    LayerInputNorm,
    QkvProjection,
    RopeKvAppend,
    AttentionFused,
    AttentionSubNorm,
    AttentionOutputProjection,
    AttentionResidual,
    PostAttentionNorm,
    GateUpProjection,
    FfnGateSubNorm,
    DownProjection,
    FfnResidual
  )

  val FinalOrder: Vector[E] = Vector(FinalNorm, LmHeadProjection, LmHeadReduce)
  val FullLayerStageCount: Int = LayerOrder.size
}

object BitNetEngineOp extends SpinalEnum {
  val QkvProjection,
      FusedAttention,
      AttentionOutputProjection,
      GateUpProjection,
      DownProjection,
      LmHeadProjection = newElement()
}

object BitNetSpuOp extends SpinalEnum {
  val AttentionRmsNorm,
      RopeKvAppend,
      OnlineAttention,
      AttentionSubRmsNorm,
      AttentionResidual,
      FfnRmsNorm,
      ExactFfnGateSubRmsNorm,
      FfnResidual,
      FinalRmsNorm,
      LmHeadReduce = newElement()
}

/** Named activation-buffer locations; physical banking remains a top-level choice. */
object BitNetTensorSlot extends SpinalEnum {
  val Hidden,
      Normalized,
      Qkv,
      AttentionContext,
      Projection,
      GateUp,
      FfnActivated,
      Down,
      LogitPartial,
      TokenResult = newElement()
}

object BitNetStatus extends SpinalEnum {
  val Ok, InvalidDescriptor, EngineFault, SpuFault, AttentionFault,
      CompletionMismatch = newElement()
}

/** Pure-Scala schedule accounting used by build-time checks and tooling. */
object BitNetSchedule {
  def tokenRounds(cfg: BitNetConfig, tokens: Int): Int = {
    require(tokens > 0 && tokens <= cfg.maxPrefillTokens)
    (tokens + cfg.tokenParallelism - 1) / cfg.tokenParallelism
  }

  def decodeBarrierCount(cfg: BitNetConfig): Int =
    cfg.layerCount * BitNetStage.FullLayerStageCount + BitNetStage.FinalOrder.size

  def prefillBarrierCount(cfg: BitNetConfig, tokens: Int): Int =
    cfg.layerCount * BitNetStage.FullLayerStageCount * tokenRounds(cfg, tokens) +
      BitNetStage.FinalOrder.size

  /** Decode shards one token across every physical engine and SPU. */
  def decodeEngineMask(cfg: BitNetConfig): BigInt =
    (BigInt(1) << cfg.engineCount) - 1

  def decodeSpuMask(cfg: BitNetConfig): BigInt =
    (BigInt(1) << cfg.spuCount) - 1
}

/** Written once by Linux before workloads are submitted; all bases are PL-visible. */
case class BitNetResidentModel(cfg: BitNetConfig) extends Bundle {
  val transformerBase = UInt(cfg.addressWidth bits)
  val auxiliaryBase = UInt(cfg.addressWidth bits)
  val ropeBase = UInt(cfg.addressWidth bits)
  val lmHeadBase = UInt(cfg.addressWidth bits)
  val kvBase = UInt(cfg.addressWidth bits)
  val activationBase = UInt(cfg.addressWidth bits)
}

/** One host submission.  A complete prefill block or one decode token is one workload. */
case class BitNetWorkload(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val mode = BitNetMode()
  val tokenCount = UInt(cfg.sequenceWidth bits)
  val pastTokenCount = UInt(cfg.sequenceWidth bits)
  val inputAddress = UInt(cfg.addressWidth bits)
  val outputAddress = UInt(cfg.addressWidth bits)
  val resetKv = Bool()
}

/** Command delivered to the native ternary/attention engine array. */
case class BitNetEngineCommand(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val mode = BitNetMode()
  val stage = BitNetStage()
  val operation = BitNetEngineOp()
  val layer = UInt(cfg.layerWidth bits)
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val tokenCount = UInt(cfg.roundTokenCountWidth bits)
  val positionBase = UInt(cfg.sequenceWidth bits)
  val sequenceLength = UInt(cfg.sequenceWidth bits)
  /** Per-token causal limit; unlike sequenceLength this never exposes a later lane. */
  val laneSequenceLength = Vec(UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  val activeTokenMask = Bits(cfg.tokenParallelism bits)
  val activeEngineMask = Bits(cfg.engineCount bits)
  val sourceA = BitNetTensorSlot()
  val sourceB = BitNetTensorSlot()
  val destination = BitNetTensorSlot()
  val inputAddress = UInt(cfg.addressWidth bits)
  val outputAddress = UInt(cfg.addressWidth bits)
  val firstRound = Bool()
  val lastRound = Bool()
  val lastLayer = Bool()
}

/** Command delivered to the replicated nonlinear/scratch/KV processors. */
case class BitNetSpuCommand(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val mode = BitNetMode()
  val stage = BitNetStage()
  val operation = BitNetSpuOp()
  val layer = UInt(cfg.layerWidth bits)
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val tokenCount = UInt(cfg.roundTokenCountWidth bits)
  val positionBase = UInt(cfg.sequenceWidth bits)
  val sequenceLength = UInt(cfg.sequenceWidth bits)
  val laneSequenceLength = Vec(UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  val activeTokenMask = Bits(cfg.tokenParallelism bits)
  /** Physical SPU shards. Decode deliberately enables all shards for one token. */
  val activeSpuMask = Bits(cfg.spuCount bits)
  val sourceA = BitNetTensorSlot()
  val sourceB = BitNetTensorSlot()
  val destination = BitNetTensorSlot()
  val inputAddress = UInt(cfg.addressWidth bits)
  val outputAddress = UInt(cfg.addressWidth bits)
  val resetKv = Bool()
  val firstRound = Bool()
  val lastRound = Bool()
  val lastLayer = Bool()
}

case class BitNetEngineCompletion(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val stage = BitNetStage()
  val layer = UInt(cfg.layerWidth bits)
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val completedMask = Bits(cfg.engineCount bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

case class BitNetSpuCompletion(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val stage = BitNetStage()
  val layer = UInt(cfg.layerWidth bits)
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val completedMask = Bits(cfg.spuCount bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

/**
  * Completion of one complete resident attention invocation for a token
  * round.  QK, full-score softmax and DSP-resident SV are deliberately one
  * barrier here; exposing their internal phases to the layer controller would
  * recreate the host-style per-operator schedule that this datapath removes.
  */
case class BitNetAttentionCompletion(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val layer = UInt(cfg.layerWidth bits)
  val tokenBase = UInt(cfg.sequenceWidth bits)
  val completedLaneMask = Bits(cfg.tokenParallelism bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

case class BitNetWorkloadCompletion(cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val mode = BitNetMode()
  val status = BitNetStatus()
  val producedTokenCount = UInt(cfg.sequenceWidth bits)
  val nextKvLength = UInt(cfg.sequenceWidth bits)
  val faultCode = Bits(8 bits)
  val elapsedCycles = UInt(64 bits)
}
