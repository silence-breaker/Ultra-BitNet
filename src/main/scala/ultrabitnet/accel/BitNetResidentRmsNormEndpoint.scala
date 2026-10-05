package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Static wiring of the resident layer RMSNorm executor. */
case class BitNetResidentRmsNormEndpointConfig(
    model: BitNetConfig = BitNetConfig.Production,
    activationLayout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production),
    arithmetic: BitNetSharedSpuRmsNormConfig =
      BitNetSharedSpuRmsNormConfig(),
    /** Production exposes one model-wide arithmetic owner; standalone OOC
      * keeps the historical self-contained default. */
    externalArithmetic: Boolean = false
) {
  require(activationLayout.cfg == model,
    "RMSNorm endpoint activation layout/config mismatch")
  require(model.tokenParallelism == arithmetic.tokenLanes,
    "one RMS arithmetic lane is required per TP4 token lane")
  require(model.spuCount == 4,
    "the resident activation fabric is striped four features per group")
  require(model.hiddenSize <= arithmetic.maxLength,
    "hidden RMSNorm length exceeds the shared arithmetic core")
  require(model.hiddenSize % model.spuCount == 0,
    "production hidden RMSNorm must end on a complete activation group")

  val groupCount: Int = model.hiddenSize / model.spuCount
}

/**
  * PL-owned request for one gamma vector.  A DDR/PLDDR staging block may start
  * filling its ping-pong window while the endpoint performs the measure pass.
  */
case class BitNetResidentRmsGammaRequest(
    cfg: BitNetResidentRmsNormEndpointConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val stage = BitNetResidentLayerStage()
  val layer = UInt(cfg.model.layerWidth bits)
  val length = UInt(cfg.arithmetic.lengthWidth bits)
}

/** One signed-Q16.16 gamma value returned by the resident parameter feeder. */
case class BitNetResidentRmsGammaBeat(
    cfg: BitNetResidentRmsNormEndpointConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val stage = BitNetResidentLayerStage()
  val layer = UInt(cfg.model.layerWidth bits)
  val featureIndex = UInt(cfg.arithmetic.indexWidth bits)
  val gammaQ16 = SInt(cfg.arithmetic.dataWidth bits)
  val first = Bool()
  val last = Bool()
  /** Conversion, AUX address or AXI failure reported by the PL stager. */
  val fault = Bool()
}

/**
  * Packet-free layer RMSNorm endpoint.
  *
  * The endpoint is the complete activation-fabric coordinator around the
  * already bit-accurate shared RMS arithmetic block.  It gathers one group
  * from each active token bank, serialises the four feature stripes into the
  * measure/replay datapath, and commits every output feature to NarrowWork.
  * No activation or partial statistic leaves PL.
  *
  * Gamma fetching is deliberately a separate typed stream.  Its request is
  * issued in parallel with the measure pass, allowing the five-bank resident
  * parameter complex to hide PS-DDR/PLDDR latency without creating a host
  * callback between the two RMS passes.
  */
class BitNetResidentRmsNormEndpoint(
    cfg: BitNetResidentRmsNormEndpointConfig =
      BitNetResidentRmsNormEndpointConfig()) extends Component {
  private val model = cfg.model
  private val layout = cfg.activationLayout
  private val arithmetic = cfg.arithmetic
  private val lanes = model.tokenParallelism
  private val stripes = model.spuCount

  val io = new Bundle {
    val command = slave(Stream(BitNetResidentLayerStep(model)))
    val gammaRequest = master(Stream(BitNetResidentRmsGammaRequest(cfg)))
    val gammaInput = slave(Stream(BitNetResidentRmsGammaBeat(cfg)))

    val activationReadCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val activationReadData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)
    val activationWriteData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    /** Present only in the production shared-arithmetic elaboration. */
    val rmsCommand = if (cfg.externalArithmetic)
      master(Stream(BitNetSharedRmsCommand(arithmetic))) else null
    val rmsMeasure = if (cfg.externalArithmetic)
      master(Stream(BitNetSharedRmsMeasureBeat(arithmetic))) else null
    val rmsReplayRequest = if (cfg.externalArithmetic)
      slave(Stream(BitNetSharedRmsReplayRequest(arithmetic))) else null
    val rmsReplay = if (cfg.externalArithmetic)
      master(Stream(BitNetSharedRmsReplayBeat(arithmetic))) else null
    val rmsResult = if (cfg.externalArithmetic)
      slave(Stream(BitNetSharedRmsResultBeat(arithmetic))) else null
    val rmsDone = if (cfg.externalArithmetic)
      slave(Stream(BitNetSharedRmsDone(arithmetic))) else null

    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
  }

  val localRms = if (!cfg.externalArithmetic)
    new BitNetSharedSpuRmsNorm(arithmetic) else null
  if (!cfg.externalArithmetic) localRms.setName("rms")

  // Direction-neutral internal wires keep the coordinator identical in
  // standalone and production-shared elaborations.
  val arithmeticCommand = Stream(BitNetSharedRmsCommand(arithmetic))
  val arithmeticMeasure = Stream(BitNetSharedRmsMeasureBeat(arithmetic))
  val arithmeticReplayRequest = Stream(
    BitNetSharedRmsReplayRequest(arithmetic))
  val arithmeticReplay = Stream(BitNetSharedRmsReplayBeat(arithmetic))
  val arithmeticResult = Stream(BitNetSharedRmsResultBeat(arithmetic))
  val arithmeticDone = Stream(BitNetSharedRmsDone(arithmetic))

  if (cfg.externalArithmetic) {
    io.rmsCommand << arithmeticCommand
    io.rmsMeasure << arithmeticMeasure
    arithmeticReplayRequest << io.rmsReplayRequest
    io.rmsReplay << arithmeticReplay
    arithmeticResult << io.rmsResult
    arithmeticDone << io.rmsDone
  } else {
    localRms.io.command << arithmeticCommand
    localRms.io.measureInput << arithmeticMeasure
    arithmeticReplayRequest << localRms.io.replayRequest
    localRms.io.replayInput << arithmeticReplay
    arithmeticResult << localRms.io.output
    arithmeticDone << localRms.io.done
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, StartArithmetic,
        MeasureIssue, MeasureCollect, MeasureEmit,
        WaitReplayRequest,
        ReplayIssue, ReplayCollect, ReplayEmit,
        CaptureOutput, WriteOutput, WaitDone, Complete = newElement()
  }
  val state = Reg(State()) init State.Idle

  val held = Reg(BitNetResidentLayerStep(model)) init
    BitNetResidentLayerStep(model).getZero
  val sourceBuffer = Reg(BitNetPhysicalActivationBuffer()) init
    BitNetPhysicalActivationBuffer.HiddenResident
  val destinationBuffer = Reg(BitNetPhysicalActivationBuffer()) init
    BitNetPhysicalActivationBuffer.NarrowWork

  val gammaRequestPending = RegInit(False)
  val readIssuePending = Reg(Bits(lanes bits)) init 0
  val readResponsePending = Reg(Bits(lanes bits)) init 0
  val writePending = Reg(Bits(lanes bits)) init 0
  val groupIndex = Reg(UInt(layout.groupIndexWidth bits)) init 0
  val stripeIndex = Reg(UInt(log2Up(stripes) bits)) init 0

  val groupValues = Vec(Vec(Reg(SInt(32 bits)) init 0, stripes), lanes)
  val resultValues = Vec(Reg(SInt(32 bits)) init 0, lanes)

  val jobProtocolFault = RegInit(False)
  val jobActivationFault = RegInit(False)
  val jobGammaFault = RegInit(False)
  val jobCoreFault = RegInit(False)
  val jobOverflow = RegInit(False)
  val globalProtocolFault = RegInit(False)
  val globalOverflow = RegInit(False)
  val localRmsProtocolFault = if (cfg.externalArithmetic) False
    else localRms.io.protocolFaultSticky
  val localRmsOverflow = if (cfg.externalArithmetic) False
    else localRms.io.overflowSticky
  io.protocolFaultSticky := globalProtocolFault || localRmsProtocolFault
  io.overflowSticky := globalOverflow || localRmsOverflow
  io.busy := state =/= State.Idle

  val rmsStageLegal =
    io.command.payload.stage === BitNetResidentLayerStage.AttentionRmsNorm ||
      io.command.payload.stage ===
        BitNetResidentLayerStage.AttentionSubRmsNorm ||
      io.command.payload.stage === BitNetResidentLayerStage.FfnRmsNorm ||
      io.command.payload.stage === BitNetResidentLayerStage.FinalRmsNorm
  val layerInRange = Bool()
  if ((1 << model.layerWidth) == model.layerCount) {
    layerInRange := True
  } else {
    layerInRange := io.command.payload.layer <
      U(model.layerCount, model.layerWidth bits)
  }
  val commandLegal = rmsStageLegal &&
    io.command.payload.unit === BitNetResidentLayerUnit.RmsNorm &&
    layerInRange &&
    io.command.payload.activeTokenMask.orR

  io.command.ready := state === State.Idle

  io.gammaRequest.valid := gammaRequestPending
  io.gammaRequest.payload.requestId := held.requestId
  io.gammaRequest.payload.stepId := held.stepId
  io.gammaRequest.payload.stage := held.stage
  io.gammaRequest.payload.layer := held.layer
  io.gammaRequest.payload.length := model.hiddenSize
  when(io.gammaRequest.fire) {
    gammaRequestPending := False
  }

  arithmeticCommand.valid := state === State.StartArithmetic
  arithmeticCommand.payload.length := model.hiddenSize
  arithmeticCommand.payload.tokenMask := held.activeTokenMask
  arithmeticCommand.payload.sourceBuffer := sourceBuffer
  arithmeticCommand.payload.tag := held.stepId.resized

  val currentFeature = UInt(arithmetic.indexWidth bits)
  currentFeature :=
    ((groupIndex.resize(arithmetic.indexWidth) |<< log2Up(stripes)) +
      stripeIndex.resize(arithmetic.indexWidth)).resized
  val currentFirst = currentFeature === 0
  val currentLast = currentFeature === model.hiddenSize - 1

  val readCommandFire = Bits(lanes bits)
  val readResponseFire = Bits(lanes bits)
  val writeFire = Bits(lanes bits)

  for (lane <- 0 until lanes) {
    io.activationReadCommand(lane).valid :=
      (state === State.MeasureIssue || state === State.ReplayIssue) &&
        readIssuePending(lane)
    io.activationReadCommand(lane).payload.buffer := sourceBuffer
    io.activationReadCommand(lane).payload.group := groupIndex
    readCommandFire(lane) := io.activationReadCommand(lane).fire

    io.activationReadData(lane).ready :=
      (state === State.MeasureCollect || state === State.ReplayCollect) &&
        readResponsePending(lane)
    readResponseFire(lane) := io.activationReadData(lane).fire

    io.activationWriteData(lane).valid :=
      state === State.WriteOutput && writePending(lane)
    io.activationWriteData(lane).payload.buffer := destinationBuffer
    io.activationWriteData(lane).payload.group := groupIndex
    io.activationWriteData(lane).payload.featureValid :=
      UIntToOh(stripeIndex, stripes).asBits
    for (stripe <- 0 until stripes) {
      io.activationWriteData(lane).payload.values(stripe) := 0
    }
    io.activationWriteData(lane).payload.values(stripeIndex) :=
      resultValues(lane)
    writeFire(lane) := io.activationWriteData(lane).fire
  }

  val remainingReadCommands = readIssuePending & ~readCommandFire
  val remainingReadResponses = readResponsePending & ~readResponseFire
  val remainingWrites = writePending & ~writeFire

  arithmeticMeasure.valid := state === State.MeasureEmit
  arithmeticMeasure.payload.tokenMask := held.activeTokenMask
  arithmeticMeasure.payload.first := currentFirst
  arithmeticMeasure.payload.last := currentLast
  for (lane <- 0 until lanes) {
    arithmeticMeasure.payload.value(lane) := groupValues(lane)(stripeIndex)
  }

  arithmeticReplayRequest.ready :=
    state === State.WaitReplayRequest && !gammaRequestPending

  val gammaIdentityMatches =
    io.gammaInput.payload.requestId === held.requestId &&
      io.gammaInput.payload.stepId === held.stepId &&
      io.gammaInput.payload.stage === held.stage &&
      io.gammaInput.payload.layer === held.layer &&
      io.gammaInput.payload.featureIndex === currentFeature &&
      io.gammaInput.payload.first === currentFirst &&
      io.gammaInput.payload.last === currentLast &&
      !io.gammaInput.payload.fault

  arithmeticReplay.valid := state === State.ReplayEmit &&
    io.gammaInput.valid
  arithmeticReplay.payload.tokenMask := held.activeTokenMask
  arithmeticReplay.payload.first := currentFirst
  arithmeticReplay.payload.last := currentLast
  arithmeticReplay.payload.gamma := io.gammaInput.payload.gammaQ16
  for (lane <- 0 until lanes) {
    arithmeticReplay.payload.value(lane) := groupValues(lane)(stripeIndex)
  }
  io.gammaInput.ready := state === State.ReplayEmit &&
    arithmeticReplay.ready

  arithmeticResult.ready := state === State.CaptureOutput
  arithmeticDone.ready := state === State.WaitDone

  val outputIdentityMatches =
    arithmeticResult.payload.tag === held.stepId.resized &&
      arithmeticResult.payload.featureIndex === currentFeature &&
      arithmeticResult.payload.tokenMask === held.activeTokenMask &&
      arithmeticResult.payload.first === currentFirst &&
      arithmeticResult.payload.last === currentLast

  io.completion.valid := state === State.Complete
  io.completion.payload.requestId := held.requestId
  io.completion.payload.stepId := held.stepId
  io.completion.payload.stage := held.stage
  io.completion.payload.layer := held.layer
  io.completion.payload.success := !(jobProtocolFault ||
    jobActivationFault || jobGammaFault || jobCoreFault || jobOverflow)
  io.completion.payload.faultCode := 0
  when(jobProtocolFault) {
    io.completion.payload.faultCode := B(0x31, 8 bits)
  } elsewhen(jobActivationFault) {
    io.completion.payload.faultCode := B(0x32, 8 bits)
  } elsewhen(jobGammaFault) {
    io.completion.payload.faultCode := B(0x33, 8 bits)
  } elsewhen(jobCoreFault) {
    io.completion.payload.faultCode := B(0x34, 8 bits)
  } elsewhen(jobOverflow) {
    io.completion.payload.faultCode := B(0x35, 8 bits)
  }

  when(io.command.fire) {
    held := io.command.payload
    sourceBuffer := BitNetPhysicalActivationBuffer.HiddenResident
    when(io.command.payload.stage ===
      BitNetResidentLayerStage.AttentionSubRmsNorm) {
      sourceBuffer := BitNetPhysicalActivationBuffer.WideWork
    }
    destinationBuffer := BitNetPhysicalActivationBuffer.NarrowWork
    gammaRequestPending := commandLegal
    groupIndex := 0
    stripeIndex := 0
    readIssuePending := 0
    readResponsePending := 0
    writePending := 0
    jobProtocolFault := !commandLegal
    jobActivationFault := False
    jobGammaFault := False
    jobCoreFault := False
    jobOverflow := False
    when(commandLegal) {
      state := State.StartArithmetic
    } otherwise {
      globalProtocolFault := True
      state := State.Complete
    }
  }

  switch(state) {
    is(State.Idle) {
      // All job state is initialised atomically on command acceptance.
    }

    is(State.StartArithmetic) {
      when(arithmeticCommand.fire) {
        readIssuePending := held.activeTokenMask
        for (lane <- 0 until lanes; stripe <- 0 until stripes) {
          groupValues(lane)(stripe) := 0
        }
        state := State.MeasureIssue
      }
    }

    is(State.MeasureIssue, State.ReplayIssue) {
      readIssuePending := remainingReadCommands
      when(remainingReadCommands === 0) {
        readResponsePending := held.activeTokenMask
        when(state === State.MeasureIssue) {
          state := State.MeasureCollect
        } otherwise {
          state := State.ReplayCollect
        }
      }
    }

    is(State.MeasureCollect, State.ReplayCollect) {
      readResponsePending := remainingReadResponses
      for (lane <- 0 until lanes) {
        when(readResponseFire(lane)) {
          for (stripe <- 0 until stripes) {
            groupValues(lane)(stripe) :=
              io.activationReadData(lane).payload.values(stripe)
          }
          val metadataFault =
            io.activationReadData(lane).payload.fault ||
              io.activationReadData(lane).payload.buffer =/= sourceBuffer ||
              io.activationReadData(lane).payload.group =/= groupIndex ||
              io.activationReadData(lane).payload.featureValid =/=
                B((BigInt(1) << stripes) - 1, stripes bits)
          when(metadataFault) {
            jobActivationFault := True
            globalProtocolFault := True
          }
        }
      }
      when(remainingReadResponses === 0) {
        when(state === State.MeasureCollect) {
          state := State.MeasureEmit
        } otherwise {
          state := State.ReplayEmit
        }
      }
    }

    is(State.MeasureEmit) {
      when(arithmeticMeasure.fire) {
        when(currentLast) {
          groupIndex := 0
          stripeIndex := 0
          state := State.WaitReplayRequest
        } elsewhen(stripeIndex === stripes - 1) {
          groupIndex := groupIndex + 1
          stripeIndex := 0
          readIssuePending := held.activeTokenMask
          for (lane <- 0 until lanes; stripe <- 0 until stripes) {
            groupValues(lane)(stripe) := 0
          }
          state := State.MeasureIssue
        } otherwise {
          stripeIndex := stripeIndex + 1
        }
      }
    }

    is(State.WaitReplayRequest) {
      when(arithmeticReplayRequest.fire) {
        when(arithmeticReplayRequest.payload.tag =/= held.stepId.resized ||
          arithmeticReplayRequest.payload.length =/= model.hiddenSize ||
          arithmeticReplayRequest.payload.tokenMask =/=
            held.activeTokenMask ||
          arithmeticReplayRequest.payload.sourceBuffer =/= sourceBuffer) {
          jobCoreFault := True
          globalProtocolFault := True
        }
        readIssuePending := held.activeTokenMask
        for (lane <- 0 until lanes; stripe <- 0 until stripes) {
          groupValues(lane)(stripe) := 0
        }
        state := State.ReplayIssue
      }
    }

    is(State.ReplayEmit) {
      when(arithmeticReplay.fire) {
        when(!gammaIdentityMatches) {
          jobGammaFault := True
          globalProtocolFault := True
        }
        state := State.CaptureOutput
      }
    }

    is(State.CaptureOutput) {
      when(arithmeticResult.fire) {
        for (lane <- 0 until lanes) {
          resultValues(lane) := arithmeticResult.payload.value(lane)
        }
        when(!outputIdentityMatches) {
          jobCoreFault := True
          globalProtocolFault := True
        }
        writePending := held.activeTokenMask
        state := State.WriteOutput
      }
    }

    is(State.WriteOutput) {
      writePending := remainingWrites
      when(remainingWrites === 0) {
        when(currentLast) {
          state := State.WaitDone
        } otherwise {
          when(stripeIndex === stripes - 1) {
            groupIndex := groupIndex + 1
            stripeIndex := 0
            readIssuePending := held.activeTokenMask
            for (lane <- 0 until lanes; stripe <- 0 until stripes) {
              groupValues(lane)(stripe) := 0
            }
            state := State.ReplayIssue
          } otherwise {
            stripeIndex := stripeIndex + 1
            state := State.ReplayEmit
          }
        }
      }
    }

    is(State.WaitDone) {
      when(arithmeticDone.fire) {
        when(arithmeticDone.payload.tag =/= held.stepId.resized ||
          arithmeticDone.payload.protocolFault) {
          jobCoreFault := True
          globalProtocolFault := True
        }
        when(arithmeticDone.payload.overflow) {
          jobOverflow := True
          globalOverflow := True
        }
        state := State.Complete
      }
    }

    is(State.Complete) {
      when(io.completion.fire) {
        state := State.Idle
      }
    }
  }
}

object GenerateBitNetResidentRmsNormEndpoint extends App {
  SpinalConfig(
    targetDirectory = "build/bitnet-resident-rmsnorm-endpoint",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetResidentRmsNormEndpoint())
}
