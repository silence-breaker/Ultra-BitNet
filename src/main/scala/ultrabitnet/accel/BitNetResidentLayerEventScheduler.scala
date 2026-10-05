package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * The production-visible activation publications of one resident layer.
  *
  * Fused implementation details deliberately do not appear here:
  * QkvResidentReady includes Q/K RoPE and packed K/V append,
  * AttentionBranchCommit includes O projection plus residual writeback,
  * FfnDownInputReady includes GateUp, exact ReLU2 gating and post-gate
  * normalization/quantization, and LayerCommit includes Down plus residual.
  */
object BitNetResidentLayerEventStage extends SpinalEnum(binarySequential) {
  val AttentionNormReady,
      QkvResidentReady,
      AttentionContextReady,
      AttentionSubNormReady,
      AttentionBranchCommit,
      FfnNormReady,
      FfnDownInputReady,
      LayerCommit,
      FinalNormReady = newElement()

  val LayerOrder: Vector[E] = Vector(
    AttentionNormReady,
    QkvResidentReady,
    AttentionContextReady,
    AttentionSubNormReady,
    AttentionBranchCommit,
    FfnNormReady,
    FfnDownInputReady,
    LayerCommit
  )

  val LayerStageCount: Int = LayerOrder.length
}

case class BitNetResidentLayerEventSchedulerConfig(
    model: BitNetConfig = BitNetConfig.Production,
    epochWidth: Int = 16,
    slotCount: Int = 2
) {
  require(epochWidth >= 4, "epochWidth must authenticate stale work")
  require(slotCount >= 2 && (slotCount & (slotCount - 1)) == 0,
    "slotCount must be a power of two for activation-image ping-pong")

  val slotWidth: Int = log2Up(slotCount)
  val completedLayerWidth: Int =
    scala.math.max(1, log2Up(model.layerCount + 1))
}

/** Starts one fixed resident transformer image. */
case class BitNetResidentLayerEventRun(
    cfg: BitNetResidentLayerEventSchedulerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val mode = BitNetMode()
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenCount = UInt(cfg.model.roundTokenCountWidth bits)
  val positionBase = UInt(cfg.model.sequenceWidth bits)
  val activeTokenMask = Bits(cfg.model.tokenParallelism bits)
  val resetKv = Bool()
  val lastPromptRound = Bool()
}

/**
  * One strictly ordered production operation.  `slot` is only the activation
  * image ping-pong ordinal.  It is not a weight-prefetch or matrix lease slot.
  */
case class BitNetResidentLayerEventCommand(
    cfg: BitNetResidentLayerEventSchedulerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val stage = BitNetResidentLayerEventStage()
  val slot = UInt(cfg.slotWidth bits)
  val mode = BitNetMode()
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenCount = UInt(cfg.model.roundTokenCountWidth bits)
  val positionBase = UInt(cfg.model.sequenceWidth bits)
  val laneCausalLength = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.model.tokenParallelism)
  val activeTokenMask = Bits(cfg.model.tokenParallelism bits)
  val resetKv = Bool()
  val lastPromptRound = Bool()
}

/** Authenticated publication/failure for exactly one accepted data command. */
case class BitNetResidentLayerEvent(
    cfg: BitNetResidentLayerEventSchedulerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val stage = BitNetResidentLayerEventStage()
  val slot = UInt(cfg.slotWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
}

object BitNetResidentLayerEventRunStatus extends SpinalEnum(binarySequential) {
  // Resource-specific values remain encoded for software compatibility.  The
  // scheduler now produces only Ok, InvalidCommand, ProtocolFault and
  // DataCommitFault because weight replay is owned below this interface.
  val Ok,
      InvalidCommand,
      ProtocolFault,
      DataCommitFault,
      MatrixLeaseFault,
      WeightRetirementFault,
      PrefetchFault = newElement()
}

case class BitNetResidentLayerEventRunCompletion(
    cfg: BitNetResidentLayerEventSchedulerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val mode = BitNetMode()
  val status = BitNetResidentLayerEventRunStatus()
  val completedLayers = UInt(cfg.completedLayerWidth bits)
  val failedLayer = UInt(cfg.model.layerWidth bits)
  val failedStage = BitNetResidentLayerEventStage()
  val failedSlot = UInt(cfg.slotWidth bits)
  val lmHeadHandedOff = Bool()
  val faultCode = Bits(8 bits)
}

case class BitNetResidentLayerEventPlanEntry(
    layer: Int,
    stage: BitNetResidentLayerEventStage.E,
    slot: Int
)

/** Pure-Scala schedule oracle used by integration tooling and focused tests. */
object BitNetResidentLayerEventOracle {
  def plan(layerCount: Int, slotCount: Int = 2,
      includeFinalNorm: Boolean = true)
      : Vector[BitNetResidentLayerEventPlanEntry] = {
    require(layerCount > 0)
    require(slotCount >= 2 && (slotCount & (slotCount - 1)) == 0)
    val layerEntries = (0 until layerCount).toVector.flatMap { layer =>
      BitNetResidentLayerEventStage.LayerOrder.map(stage => (layer, stage))
    }
    val allEntries = if (includeFinalNorm)
      layerEntries :+ (layerCount - 1,
        BitNetResidentLayerEventStage.FinalNormReady)
    else layerEntries
    allEntries.zipWithIndex.map {
      case ((layer, stage), ordinal) =>
        BitNetResidentLayerEventPlanEntry(
          layer = layer,
          stage = stage,
          slot = ordinal % slotCount)
    }
  }
}

object BitNetResidentLayerEventFault {
  val DataProtocol = 0xd1
  val MatrixLeaseProtocol = 0xd2
  val WeightRetirementProtocol = 0xd3
  val PrefetchProtocol = 0xd4
  val DataFailed = 0xe1
  val MatrixLeaseFailed = 0xe2
  val WeightRetirementFailed = 0xe3
  val PrefetchFailed = 0xe4
}

/**
  * Strict activation-publication barrier for the resident production flow.
  *
  * Exactly one dataCommand may be outstanding.  Its matching successful
  * dataCommit is the only event that advances the schedule.  Matrix weight
  * expansion, output-group fill and automatic replay/retirement remain inside
  * the matrix executor and therefore have no scheduler-level acknowledgement.
  * This deliberately prevents a synthetic prefetch acknowledgement from
  * publishing an activation before the real computation has retired.
  *
  * Continuing prefill executes `layerCount * 8` operations.  Decode and the
  * last prefill round additionally execute FinalNormReady, then hold one LM
  * head handoff until accepted.  Successful scheduler completion after that
  * handoff means ownership transferred to the LM-head engine; it does not mean
  * that the LM-head token result has completed.
  *
  * `resetKv` is asserted only for layer-zero QKV.  The downstream KV writer
  * overwrites from position zero and the causal key-count masks any stale tail;
  * resetting a run does not require clearing the full KV DDR allocation.
  */
class BitNetResidentLayerEventScheduler(
    cfg: BitNetResidentLayerEventSchedulerConfig =
      BitNetResidentLayerEventSchedulerConfig()) extends Component {
  private val m = cfg.model

  val io = new Bundle {
    val start = slave(Stream(BitNetResidentLayerEventRun(cfg)))
    val dataCommand = master(Stream(BitNetResidentLayerEventCommand(cfg)))
    val dataCommit = slave(Stream(BitNetResidentLayerEvent(cfg)))
    val lmHeadHandoff = master(
      Stream(BitNetResidentLmHeadHandoff(m)))
    val completion = master(
      Stream(BitNetResidentLayerEventRunCompletion(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, Active, IssueLmHead, Respond = newElement()
  }

  val state = Reg(State()) init State.Idle
  val requestId = Reg(UInt(m.requestIdWidth bits)) init 0
  val epoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val mode = Reg(BitNetMode()) init BitNetMode.Decode
  val tokenBase = Reg(UInt(m.sequenceWidth bits)) init 0
  val tokenCount = Reg(UInt(m.roundTokenCountWidth bits)) init 0
  val positionBase = Reg(UInt(m.sequenceWidth bits)) init 0
  val activeTokenMask = Reg(Bits(m.tokenParallelism bits)) init 0
  val resetKv = RegInit(False)
  val lastPromptRound = RegInit(False)

  val dataLayer = Reg(UInt(m.layerWidth bits)) init 0
  val dataStage = Reg(BitNetResidentLayerEventStage()) init
    BitNetResidentLayerEventStage.AttentionNormReady
  val dataSlot = Reg(UInt(cfg.slotWidth bits)) init 0
  val dataOutstanding = RegInit(False)
  val activeDataRequestId = Reg(UInt(m.requestIdWidth bits)) init 0
  val activeDataLayer = Reg(UInt(m.layerWidth bits)) init 0
  val activeDataEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val activeDataStage = Reg(BitNetResidentLayerEventStage()) init
    BitNetResidentLayerEventStage.AttentionNormReady
  val activeDataSlot = Reg(UInt(cfg.slotWidth bits)) init 0

  val completedLayers = Reg(UInt(cfg.completedLayerWidth bits)) init 0
  val responseStatus = Reg(BitNetResidentLayerEventRunStatus()) init
    BitNetResidentLayerEventRunStatus.Ok
  val responseCompletedLayers = Reg(UInt(cfg.completedLayerWidth bits)) init 0
  val responseFailedLayer = Reg(UInt(m.layerWidth bits)) init 0
  val responseFailedStage = Reg(BitNetResidentLayerEventStage()) init
    BitNetResidentLayerEventStage.AttentionNormReady
  val responseFailedSlot = Reg(UInt(cfg.slotWidth bits)) init 0
  val responseLmHeadHandedOff = RegInit(False)
  val responseFaultCode = Reg(Bits(8 bits)) init 0
  val stickyProtocolFault = RegInit(False)

  io.start.ready := state === State.Idle
  io.busy := state =/= State.Idle
  io.protocolFaultSticky := stickyProtocolFault

  // Accept an early/duplicate event while active so it is consumed and turned
  // into a fail-closed protocol response instead of deadlocking the producer.
  io.dataCommit.ready := state === State.Active

  val dataIdentityMatches = dataOutstanding &&
    io.dataCommit.payload.requestId === activeDataRequestId &&
    io.dataCommit.payload.layer === activeDataLayer &&
    io.dataCommit.payload.epoch === activeDataEpoch &&
    io.dataCommit.payload.stage === activeDataStage &&
    io.dataCommit.payload.slot === activeDataSlot
  val incomingCommitFault = io.dataCommit.fire &&
    (!dataIdentityMatches || !io.dataCommit.payload.success)

  def assignCommand(payload: BitNetResidentLayerEventCommand): Unit = {
    payload.requestId := requestId
    payload.layer := dataLayer
    payload.epoch := epoch
    payload.stage := dataStage
    payload.slot := dataSlot
    payload.mode := mode
    payload.tokenBase := tokenBase
    payload.tokenCount := tokenCount
    payload.positionBase := positionBase
    for (lane <- 0 until m.tokenParallelism) {
      payload.laneCausalLength(lane) := 0
      when(activeTokenMask(lane)) {
        payload.laneCausalLength(lane) :=
          (positionBase + U(lane + 1, m.sequenceWidth bits))
            .resize(m.sequenceWidth)
      }
    }
    payload.activeTokenMask := activeTokenMask
    payload.resetKv := resetKv && dataLayer === 0 &&
      dataStage === BitNetResidentLayerEventStage.QkvResidentReady
    payload.lastPromptRound := lastPromptRound
  }

  io.dataCommand.valid := state === State.Active && !dataOutstanding &&
    !incomingCommitFault
  assignCommand(io.dataCommand.payload)

  io.completion.valid := state === State.Respond
  io.completion.payload.requestId := requestId
  io.completion.payload.epoch := epoch
  io.completion.payload.mode := mode
  io.completion.payload.status := responseStatus
  io.completion.payload.completedLayers := responseCompletedLayers
  io.completion.payload.failedLayer := responseFailedLayer
  io.completion.payload.failedStage := responseFailedStage
  io.completion.payload.failedSlot := responseFailedSlot
  io.completion.payload.lmHeadHandedOff := responseLmHeadHandedOff
  io.completion.payload.faultCode := responseFaultCode

  val finalTokenLane = UInt(m.roundTokenCountWidth bits)
  finalTokenLane := 0
  when(mode === BitNetMode.Prefill) {
    finalTokenLane := tokenCount - 1
  }
  io.lmHeadHandoff.valid := state === State.IssueLmHead
  io.lmHeadHandoff.payload.requestId := requestId
  io.lmHeadHandoff.payload.mode := mode
  io.lmHeadHandoff.payload.finalLayer := m.layerCount - 1
  io.lmHeadHandoff.payload.tokenBase := tokenBase
  io.lmHeadHandoff.payload.tokenLane := finalTokenLane
  io.lmHeadHandoff.payload.tokenPosition :=
    (positionBase + finalTokenLane.resize(m.sequenceWidth))
      .resize(m.sequenceWidth)
  io.lmHeadHandoff.payload.activationBuffer :=
    BitNetPhysicalActivationBuffer.NarrowWork

  def faultOrDefault(provided: Bits, defaultCode: Int): Bits =
    Mux(provided.orR, provided, B(defaultCode, 8 bits))

  def prepareFailure(
      statusValue: BitNetResidentLayerEventRunStatus.E,
      event: BitNetResidentLayerEvent,
      faultValue: Bits): Unit = {
    responseStatus := statusValue
    responseCompletedLayers := completedLayers
    responseFailedLayer := event.layer
    responseFailedStage := event.stage
    responseFailedSlot := event.slot
    responseLmHeadHandedOff := False
    responseFaultCode := faultValue
    state := State.Respond
  }

  val expectedStartMask = Bits(m.tokenParallelism bits)
  for (lane <- 0 until m.tokenParallelism) {
    expectedStartMask(lane) := io.start.payload.tokenCount > lane
  }
  val startPositionEnd = io.start.payload.positionBase
    .resize(m.sequenceWidth + 1) +
    io.start.payload.tokenCount.resize(m.sequenceWidth + 1)
  val commonStartLegal = io.start.payload.tokenCount =/= 0 &&
    io.start.payload.tokenCount <= m.tokenParallelism &&
    io.start.payload.activeTokenMask === expectedStartMask &&
    startPositionEnd <= m.maxSequenceLength &&
    io.start.payload.tokenBase === io.start.payload.positionBase &&
    (!io.start.payload.resetKv || io.start.payload.positionBase === 0)
  val modeStartLegal =
    (io.start.payload.mode === BitNetMode.Decode &&
      io.start.payload.tokenCount === 1 &&
      io.start.payload.activeTokenMask ===
        B(1, m.tokenParallelism bits)) ||
      io.start.payload.mode === BitNetMode.Prefill
  val startLegal = commonStartLegal && modeStartLegal

  when(io.start.fire) {
    requestId := io.start.payload.requestId
    epoch := io.start.payload.epoch
    mode := io.start.payload.mode
    tokenBase := io.start.payload.tokenBase
    tokenCount := io.start.payload.tokenCount
    positionBase := io.start.payload.positionBase
    activeTokenMask := io.start.payload.activeTokenMask
    resetKv := io.start.payload.resetKv
    lastPromptRound := io.start.payload.lastPromptRound
    dataLayer := 0
    dataStage := BitNetResidentLayerEventStage.AttentionNormReady
    dataSlot := 0
    dataOutstanding := False
    completedLayers := 0
    responseStatus := BitNetResidentLayerEventRunStatus.Ok
    responseCompletedLayers := 0
    responseFailedLayer := 0
    responseFailedStage :=
      BitNetResidentLayerEventStage.AttentionNormReady
    responseFailedSlot := 0
    responseLmHeadHandedOff := False
    responseFaultCode := 0
    when(startLegal) {
      state := State.Active
    } otherwise {
      responseStatus := BitNetResidentLayerEventRunStatus.InvalidCommand
      responseFaultCode := B(0xe0, 8 bits)
      state := State.Respond
    }
  }

  when(io.dataCommand.fire) {
    dataOutstanding := True
    activeDataRequestId := io.dataCommand.payload.requestId
    activeDataLayer := io.dataCommand.payload.layer
    activeDataEpoch := io.dataCommand.payload.epoch
    activeDataStage := io.dataCommand.payload.stage
    activeDataSlot := io.dataCommand.payload.slot
  }

  val requiresLmHead = mode === BitNetMode.Decode || lastPromptRound
  val nextCompletedLayers = (completedLayers + 1)
    .resize(cfg.completedLayerWidth)

  when(io.dataCommit.fire) {
    when(!dataIdentityMatches) {
      stickyProtocolFault := True
      prepareFailure(
        BitNetResidentLayerEventRunStatus.ProtocolFault,
        io.dataCommit.payload,
        faultOrDefault(io.dataCommit.payload.faultCode,
          BitNetResidentLayerEventFault.DataProtocol))
    } elsewhen(!io.dataCommit.payload.success) {
      prepareFailure(
        BitNetResidentLayerEventRunStatus.DataCommitFault,
        io.dataCommit.payload,
        faultOrDefault(io.dataCommit.payload.faultCode,
          BitNetResidentLayerEventFault.DataFailed))
    } otherwise {
      dataOutstanding := False
      when(dataStage === BitNetResidentLayerEventStage.LayerCommit) {
        completedLayers := nextCompletedLayers
        when(dataLayer === m.layerCount - 1) {
          when(requiresLmHead) {
            dataStage := BitNetResidentLayerEventStage.FinalNormReady
            dataSlot := (dataSlot + 1).resize(cfg.slotWidth)
          } otherwise {
            responseStatus := BitNetResidentLayerEventRunStatus.Ok
            responseCompletedLayers := nextCompletedLayers
            responseFailedLayer := dataLayer
            responseFailedStage := dataStage
            responseFailedSlot := dataSlot
            responseLmHeadHandedOff := False
            responseFaultCode := 0
            state := State.Respond
          }
        } otherwise {
          dataLayer := dataLayer + 1
          dataStage :=
            BitNetResidentLayerEventStage.AttentionNormReady
          dataSlot := (dataSlot + 1).resize(cfg.slotWidth)
        }
      } elsewhen(
          dataStage === BitNetResidentLayerEventStage.FinalNormReady) {
        responseStatus := BitNetResidentLayerEventRunStatus.Ok
        responseCompletedLayers := completedLayers
        responseFailedLayer := dataLayer
        responseFailedStage := dataStage
        responseFailedSlot := dataSlot
        responseLmHeadHandedOff := False
        responseFaultCode := 0
        state := State.IssueLmHead
      } otherwise {
        dataSlot := (dataSlot + 1).resize(cfg.slotWidth)
        switch(dataStage) {
          is(BitNetResidentLayerEventStage.AttentionNormReady) {
            dataStage := BitNetResidentLayerEventStage.QkvResidentReady
          }
          is(BitNetResidentLayerEventStage.QkvResidentReady) {
            dataStage := BitNetResidentLayerEventStage.AttentionContextReady
          }
          is(BitNetResidentLayerEventStage.AttentionContextReady) {
            dataStage := BitNetResidentLayerEventStage.AttentionSubNormReady
          }
          is(BitNetResidentLayerEventStage.AttentionSubNormReady) {
            dataStage := BitNetResidentLayerEventStage.AttentionBranchCommit
          }
          is(BitNetResidentLayerEventStage.AttentionBranchCommit) {
            dataStage := BitNetResidentLayerEventStage.FfnNormReady
          }
          is(BitNetResidentLayerEventStage.FfnNormReady) {
            dataStage := BitNetResidentLayerEventStage.FfnDownInputReady
          }
          is(BitNetResidentLayerEventStage.FfnDownInputReady) {
            dataStage := BitNetResidentLayerEventStage.LayerCommit
          }
        }
      }
    }
  }

  when(io.lmHeadHandoff.fire) {
    responseLmHeadHandedOff := True
    state := State.Respond
  }

  when(io.completion.fire) {
    state := State.Idle
  }
}

object GenerateBitNetResidentLayerEventScheduler extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-resident-layer-event-scheduler")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLayerEventScheduler())
}
