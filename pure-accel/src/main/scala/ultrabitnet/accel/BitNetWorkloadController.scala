package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Model-specific PL scheduler inspired by BitNet's inst_gen.
  *
  * There is no host-authored microprogram and no legacy GEMV packet parser in
  * this block.  A workload is expanded into a fixed transformer schedule.  In
  * Prefill follows BitNet's bounded-scratch traversal: one four-token
  * round completes the whole attention/FFN layer before the next round reuses
  * the physical activation lanes.  Hidden states are ping/pong offloaded at
  * the layer boundary by the downstream dataflow.  This ordering is essential:
  * a stage-major traversal would overwrite an earlier round in the four-lane
  * scratchpad before its QKV/attention/FFN consumers run.  Decode follows the
  * same layer body with one logical token sharded across all physical units.
  *
  * Engine and SPU completion masks are explicit barriers.  This permits the
  * eventual top level to replicate either unit independently and to complete a
  * command in multiple events without changing the host ABI.
  */
class BitNetWorkloadController(cfg: BitNetConfig = BitNetConfig.Production)
    extends Component {

  val io = new Bundle {
    val workload = slave(Stream(BitNetWorkload(cfg)))
    val engineCommand = master(Stream(BitNetEngineCommand(cfg)))
    val spuCommand = master(Stream(BitNetSpuCommand(cfg)))
    val attentionCommand = master(
      Stream(BitNetAttentionScheduleCommand(cfg)))
    val engineCompletion = slave(Stream(BitNetEngineCompletion(cfg)))
    val spuCompletion = slave(Stream(BitNetSpuCompletion(cfg)))
    val attentionCompletion = slave(
      Stream(BitNetAttentionCompletion(cfg)))
    val completion = master(Stream(BitNetWorkloadCompletion(cfg)))
    val busy = out Bool()
  }

  object ControlState extends SpinalEnum {
    val Idle, Issue, WaitForUnits, Respond = newElement()
  }

  val state = Reg(ControlState()) init ControlState.Idle
  val requestId = Reg(UInt(cfg.requestIdWidth bits)) init 0
  val mode = Reg(BitNetMode()) init BitNetMode.Decode
  val tokenCount = Reg(UInt(cfg.sequenceWidth bits)) init 0
  val pastTokenCount = Reg(UInt(cfg.sequenceWidth bits)) init 0
  val inputAddress = Reg(UInt(cfg.addressWidth bits)) init 0
  val outputAddress = Reg(UInt(cfg.addressWidth bits)) init 0
  val resetKv = Reg(Bool()) init False

  val stage = Reg(BitNetStage()) init BitNetStage.LayerInputNorm
  val layer = Reg(UInt(cfg.layerWidth bits)) init 0
  val tokenBase = Reg(UInt(cfg.sequenceWidth bits)) init 0

  val engineIssued = Reg(Bool()) init False
  val spuIssued = Reg(Bool()) init False
  val attentionIssued = Reg(Bool()) init False
  val engineDoneMask = Reg(Bits(cfg.engineCount bits)) init 0
  val spuDoneMask = Reg(Bits(cfg.spuCount bits)) init 0
  val attentionDoneMask = Reg(Bits(cfg.tokenParallelism bits)) init 0
  val faultSeen = Reg(Bool()) init False
  val faultStatus = Reg(BitNetStatus()) init BitNetStatus.Ok
  val faultCode = Reg(Bits(8 bits)) init 0

  val cycleCounter = Reg(UInt(64 bits)) init 0
  val responseStatus = Reg(BitNetStatus()) init BitNetStatus.Ok
  val responseFaultCode = Reg(Bits(8 bits)) init 0
  val responseCycles = Reg(UInt(64 bits)) init 0
  val responseProducedTokens = Reg(UInt(cfg.sequenceWidth bits)) init 0
  val responseNextKvLength = Reg(UInt(cfg.sequenceWidth bits)) init 0

  io.busy := state =/= ControlState.Idle
  io.workload.ready := state === ControlState.Idle

  io.completion.valid := state === ControlState.Respond
  io.completion.payload.requestId := requestId
  io.completion.payload.mode := mode
  io.completion.payload.status := responseStatus
  io.completion.payload.producedTokenCount := responseProducedTokens
  io.completion.payload.nextKvLength := responseNextKvLength
  io.completion.payload.faultCode := responseFaultCode
  io.completion.payload.elapsedCycles := responseCycles

  when(state =/= ControlState.Idle && state =/= ControlState.Respond) {
    cycleCounter := cycleCounter + 1
  }

  // Static stage classification.  No runtime opcode table is involved.
  val needsEngine = Bool()
  val needsSpu = Bool()
  val needsAttention = Bool()
  needsEngine := False
  needsSpu := False
  needsAttention := False
  switch(stage) {
    is(BitNetStage.QkvProjection,
       BitNetStage.AttentionOutputProjection,
       BitNetStage.GateUpProjection,
       BitNetStage.DownProjection,
       BitNetStage.LmHeadProjection) {
      needsEngine := True
    }
    is(BitNetStage.AttentionFused) {
      // One command owns QK, complete-score softmax and DSP-resident SV.  The
      // layer controller never exposes or independently barriers those phases.
      needsAttention := True
    }
    default {
      needsSpu := True
    }
  }

  // Final norm and LM head only consume the last prompt token.  Transformer
  // layer stages consume every prefill token in tokenParallelism-sized rounds.
  val finalTokenOnly = stage === BitNetStage.FinalNorm ||
    stage === BitNetStage.LmHeadProjection ||
    stage === BitNetStage.LmHeadReduce

  val remainingTokens = UInt(cfg.sequenceWidth bits)
  remainingTokens := tokenCount - tokenBase
  when(finalTokenOnly) {
    remainingTokens := 1
  }

  val roundTokenCountWide = UInt(cfg.sequenceWidth bits)
  roundTokenCountWide := remainingTokens
  when(remainingTokens > cfg.tokenParallelism) {
    roundTokenCountWide := cfg.tokenParallelism
  }

  val activeTokenMask = Bits(cfg.tokenParallelism bits)
  for (lane <- 0 until cfg.tokenParallelism) {
    activeTokenMask(lane) := remainingTokens > lane
  }

  val activeEngineMask = Bits(cfg.engineCount bits)
  // Engines own output-row/head/vocabulary shards, not token lanes. Both
  // prefill (including a one-token tail round) and decode use every engine.
  activeEngineMask.setAll()

  val activeSpuMask = Bits(cfg.spuCount bits)
  for (spu <- 0 until cfg.spuCount) {
    val assignedLanes = (0 until cfg.tokenParallelism).filter(_ % cfg.spuCount == spu)
    if (assignedLanes.nonEmpty) {
      activeSpuMask(spu) := assignedLanes.map(activeTokenMask(_)).reduce(_ || _)
    } else {
      activeSpuMask(spu) := False
    }
  }
  when(mode === BitNetMode.Decode) {
    // Four physical SPUs shard heads/vector ranges for the active token.
    activeSpuMask.setAll()
  }

  val extendedRoundEnd = tokenBase.resize(cfg.sequenceWidth + 1) +
    U(cfg.tokenParallelism, cfg.sequenceWidth + 1 bits)
  val hasMoreRounds = !finalTokenOnly &&
    extendedRoundEnd < tokenCount.resize(cfg.sequenceWidth + 1)
  val firstRound = tokenBase === 0
  val lastRound = !hasMoreRounds
  val lastLayer = layer === cfg.layerCount - 1
  val positionBase = (pastTokenCount + tokenBase).resize(cfg.sequenceWidth)
  val sequenceLength =
    (pastTokenCount + tokenBase + roundTokenCountWide).resize(cfg.sequenceWidth)
  val laneSequenceLength = Vec(UInt(cfg.sequenceWidth bits), cfg.tokenParallelism)
  for (lane <- 0 until cfg.tokenParallelism) {
    laneSequenceLength(lane) :=
      (positionBase + U(lane + 1, cfg.sequenceWidth bits)).resize(cfg.sequenceWidth)
  }

  // Common engine-command metadata.
  io.engineCommand.valid := state === ControlState.Issue && needsEngine && !engineIssued
  io.engineCommand.payload.requestId := requestId
  io.engineCommand.payload.mode := mode
  io.engineCommand.payload.stage := stage
  io.engineCommand.payload.operation := BitNetEngineOp.QkvProjection
  io.engineCommand.payload.layer := layer
  io.engineCommand.payload.tokenBase := tokenBase
  io.engineCommand.payload.tokenCount := roundTokenCountWide.resized
  io.engineCommand.payload.positionBase := positionBase
  io.engineCommand.payload.sequenceLength := sequenceLength
  io.engineCommand.payload.laneSequenceLength := laneSequenceLength
  io.engineCommand.payload.activeTokenMask := activeTokenMask
  io.engineCommand.payload.activeEngineMask := activeEngineMask
  io.engineCommand.payload.sourceA := BitNetTensorSlot.Hidden
  io.engineCommand.payload.sourceB := BitNetTensorSlot.Hidden
  io.engineCommand.payload.destination := BitNetTensorSlot.Hidden
  io.engineCommand.payload.inputAddress := inputAddress
  io.engineCommand.payload.outputAddress := outputAddress
  io.engineCommand.payload.firstRound := firstRound
  io.engineCommand.payload.lastRound := lastRound
  io.engineCommand.payload.lastLayer := lastLayer

  switch(stage) {
    is(BitNetStage.QkvProjection) {
      io.engineCommand.payload.operation := BitNetEngineOp.QkvProjection
      io.engineCommand.payload.sourceA := BitNetTensorSlot.Normalized
      io.engineCommand.payload.destination := BitNetTensorSlot.Qkv
    }
    is(BitNetStage.AttentionOutputProjection) {
      io.engineCommand.payload.operation := BitNetEngineOp.AttentionOutputProjection
      io.engineCommand.payload.sourceA := BitNetTensorSlot.Normalized
      io.engineCommand.payload.destination := BitNetTensorSlot.Projection
    }
    is(BitNetStage.GateUpProjection) {
      io.engineCommand.payload.operation := BitNetEngineOp.GateUpProjection
      io.engineCommand.payload.sourceA := BitNetTensorSlot.Normalized
      io.engineCommand.payload.destination := BitNetTensorSlot.GateUp
    }
    is(BitNetStage.DownProjection) {
      io.engineCommand.payload.operation := BitNetEngineOp.DownProjection
      io.engineCommand.payload.sourceA := BitNetTensorSlot.FfnActivated
      io.engineCommand.payload.destination := BitNetTensorSlot.Down
    }
    is(BitNetStage.LmHeadProjection) {
      io.engineCommand.payload.operation := BitNetEngineOp.LmHeadProjection
      io.engineCommand.payload.sourceA := BitNetTensorSlot.Normalized
      io.engineCommand.payload.destination := BitNetTensorSlot.LogitPartial
    }
  }

  // Common SPU-command metadata.
  io.spuCommand.valid := state === ControlState.Issue && needsSpu && !spuIssued
  io.spuCommand.payload.requestId := requestId
  io.spuCommand.payload.mode := mode
  io.spuCommand.payload.stage := stage
  io.spuCommand.payload.operation := BitNetSpuOp.AttentionRmsNorm
  io.spuCommand.payload.layer := layer
  io.spuCommand.payload.tokenBase := tokenBase
  io.spuCommand.payload.tokenCount := roundTokenCountWide.resized
  io.spuCommand.payload.positionBase := positionBase
  io.spuCommand.payload.sequenceLength := sequenceLength
  io.spuCommand.payload.laneSequenceLength := laneSequenceLength
  io.spuCommand.payload.activeTokenMask := activeTokenMask
  io.spuCommand.payload.activeSpuMask := activeSpuMask
  io.spuCommand.payload.sourceA := BitNetTensorSlot.Hidden
  io.spuCommand.payload.sourceB := BitNetTensorSlot.Hidden
  io.spuCommand.payload.destination := BitNetTensorSlot.Hidden
  io.spuCommand.payload.inputAddress := inputAddress
  io.spuCommand.payload.outputAddress := outputAddress
  io.spuCommand.payload.resetKv := resetKv && layer === 0 && tokenBase === 0
  io.spuCommand.payload.firstRound := firstRound
  io.spuCommand.payload.lastRound := lastRound
  io.spuCommand.payload.lastLayer := lastLayer

  switch(stage) {
    is(BitNetStage.LayerInputNorm) {
      io.spuCommand.payload.operation := BitNetSpuOp.AttentionRmsNorm
      io.spuCommand.payload.sourceA := BitNetTensorSlot.Hidden
      io.spuCommand.payload.destination := BitNetTensorSlot.Normalized
    }
    is(BitNetStage.RopeKvAppend) {
      io.spuCommand.payload.operation := BitNetSpuOp.RopeKvAppend
      io.spuCommand.payload.sourceA := BitNetTensorSlot.Qkv
      io.spuCommand.payload.destination := BitNetTensorSlot.Qkv
    }
    is(BitNetStage.AttentionSubNorm) {
      io.spuCommand.payload.operation := BitNetSpuOp.AttentionSubRmsNorm
      io.spuCommand.payload.sourceA := BitNetTensorSlot.AttentionContext
      io.spuCommand.payload.destination := BitNetTensorSlot.Normalized
    }
    is(BitNetStage.AttentionResidual) {
      io.spuCommand.payload.operation := BitNetSpuOp.AttentionResidual
      io.spuCommand.payload.sourceA := BitNetTensorSlot.Hidden
      io.spuCommand.payload.sourceB := BitNetTensorSlot.Projection
      io.spuCommand.payload.destination := BitNetTensorSlot.Hidden
    }
    is(BitNetStage.PostAttentionNorm) {
      io.spuCommand.payload.operation := BitNetSpuOp.FfnRmsNorm
      io.spuCommand.payload.sourceA := BitNetTensorSlot.Hidden
      io.spuCommand.payload.destination := BitNetTensorSlot.Normalized
    }
    is(BitNetStage.FfnGateSubNorm) {
      io.spuCommand.payload.operation := BitNetSpuOp.ExactFfnGateSubRmsNorm
      io.spuCommand.payload.sourceA := BitNetTensorSlot.GateUp
      io.spuCommand.payload.sourceB := BitNetTensorSlot.GateUp
      io.spuCommand.payload.destination := BitNetTensorSlot.FfnActivated
    }
    is(BitNetStage.FfnResidual) {
      io.spuCommand.payload.operation := BitNetSpuOp.FfnResidual
      io.spuCommand.payload.sourceA := BitNetTensorSlot.Hidden
      io.spuCommand.payload.sourceB := BitNetTensorSlot.Down
      io.spuCommand.payload.destination := BitNetTensorSlot.Hidden
    }
    is(BitNetStage.FinalNorm) {
      io.spuCommand.payload.operation := BitNetSpuOp.FinalRmsNorm
      io.spuCommand.payload.sourceA := BitNetTensorSlot.Hidden
      io.spuCommand.payload.destination := BitNetTensorSlot.Normalized
    }
    is(BitNetStage.LmHeadReduce) {
      io.spuCommand.payload.operation := BitNetSpuOp.LmHeadReduce
      io.spuCommand.payload.sourceA := BitNetTensorSlot.LogitPartial
      io.spuCommand.payload.destination := BitNetTensorSlot.TokenResult
    }
  }

  // Resident-attention command.  The absolute sequence position is the
  // packet-free token/context tag used throughout QK, softmax, KV and SV.
  // Inactive tail lanes are zeroed rather than exposing an out-of-range
  // position beyond the submitted prompt.
  io.attentionCommand.valid := state === ControlState.Issue &&
    needsAttention && !attentionIssued
  io.attentionCommand.payload.requestId := requestId
  io.attentionCommand.payload.layer := layer
  io.attentionCommand.payload.tokenBase := tokenBase
  io.attentionCommand.payload.mode := mode
  io.attentionCommand.payload.activeTokenMask := activeTokenMask
  for (lane <- 0 until cfg.tokenParallelism) {
    io.attentionCommand.payload.tokenId(lane) := 0
    io.attentionCommand.payload.queryPosition(lane) := 0
    io.attentionCommand.payload.keyCount(lane) := 0
    when(activeTokenMask(lane)) {
      io.attentionCommand.payload.tokenId(lane) :=
        (positionBase + lane).resize(cfg.sequenceWidth)
      io.attentionCommand.payload.queryPosition(lane) :=
        (positionBase + lane).resize(cfg.sequenceWidth)
      io.attentionCommand.payload.keyCount(lane) := laneSequenceLength(lane)
    }
  }

  val incomingEnd = io.workload.payload.pastTokenCount.resize(cfg.sequenceWidth + 1) +
    io.workload.payload.tokenCount.resize(cfg.sequenceWidth + 1)
  val descriptorValid = io.workload.payload.tokenCount =/= 0 &&
    incomingEnd <= cfg.maxSequenceLength &&
    (!io.workload.payload.resetKv || io.workload.payload.pastTokenCount === 0) &&
    ((io.workload.payload.mode === BitNetMode.Decode && io.workload.payload.tokenCount === 1) ||
      (io.workload.payload.mode === BitNetMode.Prefill &&
        io.workload.payload.tokenCount <= cfg.maxPrefillTokens))

  when(io.workload.fire) {
    requestId := io.workload.payload.requestId
    mode := io.workload.payload.mode
    tokenCount := io.workload.payload.tokenCount
    pastTokenCount := io.workload.payload.pastTokenCount
    inputAddress := io.workload.payload.inputAddress
    outputAddress := io.workload.payload.outputAddress
    resetKv := io.workload.payload.resetKv
    layer := 0
    stage := BitNetStage.LayerInputNorm
    tokenBase := 0
    engineIssued := False
    spuIssued := False
    attentionIssued := False
    engineDoneMask := 0
    spuDoneMask := 0
    attentionDoneMask := 0
    faultSeen := False
    faultStatus := BitNetStatus.Ok
    faultCode := 0
    cycleCounter := 0
    responseCycles := 0
    responseFaultCode := 0
    responseProducedTokens := 0
    responseNextKvLength := io.workload.payload.pastTokenCount

    when(descriptorValid) {
      state := ControlState.Issue
    } otherwise {
      responseStatus := BitNetStatus.InvalidDescriptor
      state := ControlState.Respond
    }
  }

  val engineAccepted = io.engineCommand.fire
  val spuAccepted = io.spuCommand.fire
  val attentionAccepted = io.attentionCommand.fire
  when(state === ControlState.Issue) {
    when(engineAccepted) {
      engineIssued := True
    }
    when(spuAccepted) {
      spuIssued := True
    }
    when(attentionAccepted) {
      attentionIssued := True
    }

    val engineIssueComplete = !needsEngine || engineIssued || engineAccepted
    val spuIssueComplete = !needsSpu || spuIssued || spuAccepted
    val attentionIssueComplete = !needsAttention || attentionIssued ||
      attentionAccepted
    when(engineIssueComplete && spuIssueComplete && attentionIssueComplete) {
      engineDoneMask := 0
      spuDoneMask := 0
      attentionDoneMask := 0
      state := ControlState.WaitForUnits
    }
  }

  io.engineCompletion.ready := state === ControlState.WaitForUnits
  io.spuCompletion.ready := state === ControlState.WaitForUnits
  io.attentionCompletion.ready := state === ControlState.WaitForUnits

  val engineCompletionMatches = io.engineCompletion.payload.requestId === requestId &&
    io.engineCompletion.payload.stage === stage &&
    io.engineCompletion.payload.layer === layer &&
    io.engineCompletion.payload.tokenBase === tokenBase
  val spuCompletionMatches = io.spuCompletion.payload.requestId === requestId &&
    io.spuCompletion.payload.stage === stage &&
    io.spuCompletion.payload.layer === layer &&
    io.spuCompletion.payload.tokenBase === tokenBase
  val attentionCompletionMatches =
    io.attentionCompletion.payload.requestId === requestId &&
      io.attentionCompletion.payload.layer === layer &&
      io.attentionCompletion.payload.tokenBase === tokenBase &&
      stage === BitNetStage.AttentionFused
  val matchingEngineFire = io.engineCompletion.fire && engineCompletionMatches
  val matchingSpuFire = io.spuCompletion.fire && spuCompletionMatches
  val matchingAttentionFire = io.attentionCompletion.fire &&
    attentionCompletionMatches
  val engineMismatch = io.engineCompletion.fire && !engineCompletionMatches
  val spuMismatch = io.spuCompletion.fire && !spuCompletionMatches
  val attentionMismatch = io.attentionCompletion.fire &&
    !attentionCompletionMatches

  val engineMaskAfter = engineDoneMask |
    Mux(matchingEngineFire, io.engineCompletion.payload.completedMask, B(0, cfg.engineCount bits))
  val spuMaskAfter = spuDoneMask |
    Mux(matchingSpuFire, io.spuCompletion.payload.completedMask, B(0, cfg.spuCount bits))
  val attentionMaskAfter = attentionDoneMask |
    Mux(matchingAttentionFire,
      io.attentionCompletion.payload.completedLaneMask,
      B(0, cfg.tokenParallelism bits))
  val engineBarrierDone = !needsEngine ||
    (engineMaskAfter & activeEngineMask) === activeEngineMask
  val spuBarrierDone = !needsSpu ||
    (spuMaskAfter & activeSpuMask) === activeSpuMask
  val attentionBarrierDone = !needsAttention ||
    (attentionMaskAfter & activeTokenMask) === activeTokenMask
  val engineFaultNow = matchingEngineFire && !io.engineCompletion.payload.success
  val spuFaultNow = matchingSpuFire && !io.spuCompletion.payload.success
  val attentionFaultNow = matchingAttentionFire &&
    !io.attentionCompletion.payload.success

  when(state === ControlState.WaitForUnits) {
    when(matchingEngineFire) {
      engineDoneMask := engineMaskAfter
      when(!io.engineCompletion.payload.success) {
        faultSeen := True
        faultStatus := BitNetStatus.EngineFault
        faultCode := io.engineCompletion.payload.faultCode
      }
    }
    when(matchingSpuFire) {
      spuDoneMask := spuMaskAfter
      when(!io.spuCompletion.payload.success) {
        faultSeen := True
        faultStatus := BitNetStatus.SpuFault
        faultCode := io.spuCompletion.payload.faultCode
      }
    }
    when(matchingAttentionFire) {
      attentionDoneMask := attentionMaskAfter
      when(!io.attentionCompletion.payload.success) {
        faultSeen := True
        faultStatus := BitNetStatus.AttentionFault
        faultCode := io.attentionCompletion.payload.faultCode
      }
    }

    when(engineMismatch || spuMismatch || attentionMismatch) {
      responseStatus := BitNetStatus.CompletionMismatch
      responseFaultCode := Mux(
        engineMismatch,
        io.engineCompletion.payload.faultCode,
        Mux(spuMismatch, io.spuCompletion.payload.faultCode,
          io.attentionCompletion.payload.faultCode))
      responseCycles := cycleCounter + 1
      responseProducedTokens := 0
      responseNextKvLength := pastTokenCount
      state := ControlState.Respond
    } elsewhen (engineBarrierDone && spuBarrierDone && attentionBarrierDone) {
      when(faultSeen || engineFaultNow || spuFaultNow || attentionFaultNow) {
        responseStatus := Mux(
          engineFaultNow,
          BitNetStatus.EngineFault,
          Mux(spuFaultNow, BitNetStatus.SpuFault,
            Mux(attentionFaultNow, BitNetStatus.AttentionFault,
              faultStatus))
        )
        responseFaultCode := Mux(
          engineFaultNow,
          io.engineCompletion.payload.faultCode,
          Mux(spuFaultNow, io.spuCompletion.payload.faultCode,
            Mux(attentionFaultNow,
              io.attentionCompletion.payload.faultCode, faultCode))
        )
        responseCycles := cycleCounter + 1
        responseProducedTokens := 0
        responseNextKvLength := pastTokenCount
        state := ControlState.Respond
      } otherwise {
        engineIssued := False
        spuIssued := False
        attentionIssued := False
        engineDoneMask := 0
        spuDoneMask := 0
        attentionDoneMask := 0

        // A token round stays resident in the four physical activation lanes
        // for the complete layer body.  Only FfnResidual advances the round;
        // every other stage advances the semantic operator while preserving
        // tokenBase.  This is the same layer->round traversal used by
        // BitNet's prefill round/layer counters.
        switch(stage) {
            is(BitNetStage.LayerInputNorm) {
              stage := BitNetStage.QkvProjection
              state := ControlState.Issue
            }
            is(BitNetStage.QkvProjection) {
              stage := BitNetStage.RopeKvAppend
              state := ControlState.Issue
            }
            is(BitNetStage.RopeKvAppend) {
              stage := BitNetStage.AttentionFused
              state := ControlState.Issue
            }
            is(BitNetStage.AttentionFused) {
              stage := BitNetStage.AttentionSubNorm
              state := ControlState.Issue
            }
            is(BitNetStage.AttentionSubNorm) {
              stage := BitNetStage.AttentionOutputProjection
              state := ControlState.Issue
            }
            is(BitNetStage.AttentionOutputProjection) {
              stage := BitNetStage.AttentionResidual
              state := ControlState.Issue
            }
            is(BitNetStage.AttentionResidual) {
              stage := BitNetStage.PostAttentionNorm
              state := ControlState.Issue
            }
            is(BitNetStage.PostAttentionNorm) {
              stage := BitNetStage.GateUpProjection
              state := ControlState.Issue
            }
            is(BitNetStage.GateUpProjection) {
              stage := BitNetStage.FfnGateSubNorm
              state := ControlState.Issue
            }
            is(BitNetStage.FfnGateSubNorm) {
              stage := BitNetStage.DownProjection
              state := ControlState.Issue
            }
            is(BitNetStage.DownProjection) {
              stage := BitNetStage.FfnResidual
              state := ControlState.Issue
            }
            is(BitNetStage.FfnResidual) {
              when(hasMoreRounds) {
                tokenBase := (tokenBase + cfg.tokenParallelism)
                  .resize(cfg.sequenceWidth)
                stage := BitNetStage.LayerInputNorm
              } otherwise {
                tokenBase := 0
                when(lastLayer) {
                  stage := BitNetStage.FinalNorm
                  when(mode === BitNetMode.Prefill) {
                    // The final-token load/normalization coordinator maps this
                    // absolute token index back into lane zero.
                    tokenBase := tokenCount - 1
                  }
                } otherwise {
                  layer := layer + 1
                  stage := BitNetStage.LayerInputNorm
                }
              }
              state := ControlState.Issue
            }
            is(BitNetStage.FinalNorm) {
              stage := BitNetStage.LmHeadProjection
              when(mode === BitNetMode.Prefill) {
                tokenBase := tokenCount - 1
              }
              state := ControlState.Issue
            }
            is(BitNetStage.LmHeadProjection) {
              stage := BitNetStage.LmHeadReduce
              when(mode === BitNetMode.Prefill) {
                tokenBase := tokenCount - 1
              }
              state := ControlState.Issue
            }
            is(BitNetStage.LmHeadReduce) {
              responseStatus := BitNetStatus.Ok
              responseFaultCode := 0
              responseCycles := cycleCounter + 1
              responseProducedTokens := 1
              responseNextKvLength :=
                (pastTokenCount + tokenCount).resize(cfg.sequenceWidth)
              state := ControlState.Respond
            }
        }
      }
    }
  }

  when(io.completion.fire) {
    state := ControlState.Idle
  }
}
