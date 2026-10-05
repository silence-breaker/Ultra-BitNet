package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetResidentFfnPostGateRmsContext(
    cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg) bits)
  val layer = UInt(cfg.layerWidth bits)
  val mode = BitNetMode()
  val tokenMask = Bits(cfg.tokenParallelism bits)
}

case class BitNetResidentFfnPostGateRmsCompletion(
    cfg: BitNetConfig) extends Bundle {
  val requestId = UInt(cfg.requestIdWidth bits)
  val stepId = UInt(BitNetResidentLayerGeometry.stepIdWidth(cfg) bits)
  val layer = UInt(cfg.layerWidth bits)
  val mode = BitNetMode()
  val tokenMask = Bits(cfg.tokenParallelism bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
  val measuredFeatures = UInt(32 bits)
  val replayedFeatures = UInt(32 bits)
  val writtenFeatures = UInt(32 bits)
}

object BitNetResidentFfnPostGateRmsFault {
  val InvalidBinding = 0xd1
  val ActivationMetadata = 0xd2
  val GammaMetadata = 0xd3
  val ReplayRequest = 0xd4
  val RmsResult = 0xd5
  val RmsDone = 0xd6
  val Overflow = 0xd7
  val FeatureCount = 0xd8
}

case class BitNetResidentFfnSubNormGammaRequest(
    cfg: BitNetResidentFfnPostGateRmsSubsystemConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(
    BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val length = UInt(cfg.arithmetic.lengthWidth bits)
}

case class BitNetResidentFfnSubNormGammaBeat(
    cfg: BitNetResidentFfnPostGateRmsSubsystemConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val stepId = UInt(
    BitNetResidentLayerGeometry.stepIdWidth(cfg.model) bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val featureIndex = UInt(cfg.arithmetic.indexWidth bits)
  val gammaQ16 = SInt(cfg.arithmetic.dataWidth bits)
  val first = Bool()
  val last = Bool()
  val fault = Bool()
}

case class BitNetResidentFfnPostGateRmsSubsystemConfig(
    model: BitNetConfig = BitNetConfig.Production,
    activationLayout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production),
    arithmetic: BitNetSharedSpuRmsNormConfig =
      BitNetSharedSpuRmsNormConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    bank: Int = 0,
    gammaAddresses: BitNetResidentRmsGammaAddressLayout =
      BitNetResidentRmsGammaAddressLayout.production()) {
  require(model == BitNetConfig.Production)
  require(activationLayout.cfg == model)
  require(model.ffnSize == 6912 && model.spuCount == 4 &&
    (model.tokenParallelism == 2 || model.tokenParallelism == 4))
  require(arithmetic.maxLength == model.ffnSize &&
    arithmetic.tokenLanes == model.tokenParallelism)
  require(arithmetic.dataWidth == 32)
  require(activationLayout.features(
    BitNetPhysicalActivationBuffer.WideWork) >= model.ffnSize)
  require(model.ffnSize % model.spuCount == 0)
  require(crossbar.dataWidth == 128 && crossbar.maxBurstBeats == 256)
  require(bank >= 0 && bank < crossbar.bankCount)
  gammaAddresses.validate(model, crossbar.localAddressWidth)

  val wordsPerBeat: Int = crossbar.dataWidth / 16
  val gammaBeats: Int = model.ffnSize / wordsPerBeat
  val gammaBeatCountWidth: Int = log2Up(gammaBeats + 1)
  val gammaIndexWidth: Int = log2Up(model.ffnSize)
  val gammaBurstCount: Int =
    (gammaBeats + crossbar.maxBurstBeats - 1) / crossbar.maxBurstBeats
  val gammaBurstWidth: Int = scala.math.max(1, log2Up(gammaBurstCount + 1))
  val groupCount: Int = model.ffnSize / model.spuCount
}

/**
  * Dedicated FfnSubNorm AUX prefetch window.
  *
  * It starts with the RMS measure pass, converts one BF16 word per cycle and
  * stores exactly 6912 signed Q16.16 values.  The window is a parameter cache,
  * not an activation copy; replay remains entirely PL-local.
  */
class BitNetResidentFfnSubNormGammaStager(
    cfg: BitNetResidentFfnPostGateRmsSubsystemConfig =
      BitNetResidentFfnPostGateRmsSubsystemConfig()) extends Component {
  private val model = cfg.model
  private val crossbar = cfg.crossbar
  private val wordsPerBeat = cfg.wordsPerBeat

  val io = new Bundle {
    val request = slave(Stream(
      BitNetResidentFfnSubNormGammaRequest(cfg)))
    val readRequest = master(Stream(
      BitNetClientBankReadRequest(crossbar)))
    val readResponse = slave(Stream(
      BitNetClientBankReadResponse(crossbar)))
    val gamma = master(Stream(BitNetResidentFfnSubNormGammaBeat(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssueBurst, ReceiveBurst, WaitConversion, Emit = newElement()
  }
  val state = Reg(State()) init State.Idle
  val held = Reg(BitNetResidentFfnSubNormGammaRequest(cfg)) init
    BitNetResidentFfnSubNormGammaRequest(cfg).getZero
  val vectorBase = Reg(UInt(crossbar.localAddressWidth bits)) init 0
  val jobFault = RegInit(False)
  val stickyFault = RegInit(False)

  val layerInRange = if ((1 << model.layerWidth) == model.layerCount) True
    else io.request.layer < model.layerCount
  val requestLegal = layerInRange &&
    io.request.length === model.ffnSize
  val layerBases = Vec(UInt(crossbar.localAddressWidth bits), model.layerCount)
  for (layer <- 0 until model.layerCount) {
    layerBases(layer) := U(
      cfg.gammaAddresses.ffnSubNormBase +
        BigInt(layer) * model.ffnSize * 2,
      crossbar.localAddressWidth bits)
  }

  val totalAcceptedBeats = Reg(UInt(cfg.gammaBeatCountWidth bits)) init 0
  val burstBeats = Reg(UInt(crossbar.beatCountWidth bits)) init 0
  val expectedBeat = Reg(UInt(crossbar.beatIndexWidth bits)) init 0
  val burstOrdinal = Reg(UInt(cfg.gammaBurstWidth bits)) init 0
  val burstTag = Reg(UInt(crossbar.tagWidth bits)) init 0
  val transactionId = Reg(UInt(crossbar.transactionIdWidth bits)) init 0

  val remaining = U(cfg.gammaBeats, cfg.gammaBeatCountWidth bits) -
    totalAcceptedBeats
  val nextBurst = UInt(crossbar.beatCountWidth bits)
  nextBurst := remaining.resized
  when(remaining > crossbar.maxBurstBeats) {
    nextBurst := crossbar.maxBurstBeats
  }

  io.request.ready := state === State.Idle
  io.readRequest.valid := state === State.IssueBurst
  io.readRequest.bank := cfg.bank
  io.readRequest.transactionId := transactionId
  io.readRequest.tag :=
    (held.stepId.resize(crossbar.tagWidth) ^
      held.requestId.resize(crossbar.tagWidth) ^
      burstOrdinal.resize(crossbar.tagWidth)).resized
  io.readRequest.localAddress :=
    (vectorBase +
      (totalAcceptedBeats.resize(crossbar.localAddressWidth) |<< 4)).resized
  io.readRequest.beats := nextBurst

  val beatHeld = RegInit(False)
  val beatData = Reg(Bits(crossbar.dataWidth bits)) init 0
  val beatWord = Reg(UInt(log2Up(wordsPerBeat) bits)) init 0
  val heldBeatLast = RegInit(False)
  val words = beatData.subdivideIn(16 bits)
  io.readResponse.ready := state === State.ReceiveBurst && !beatHeld
  val responseLegal =
    io.readResponse.bank === cfg.bank &&
      io.readResponse.transactionId === transactionId &&
      io.readResponse.tag === burstTag &&
      io.readResponse.beatIndex === expectedBeat &&
      io.readResponse.last ===
        (expectedBeat.resize(crossbar.beatCountWidth) === burstBeats - 1) &&
      !io.readResponse.fault && io.readResponse.rresp === B"2'b00"

  val converter = new NativeBf16Q16Staging
  val acceptedWord = Reg(UInt(cfg.gammaIndexWidth bits)) init 0
  converter.io.input.valid := beatHeld
  converter.io.input.raw := words(beatWord)
  converter.io.input.first := acceptedWord === 0
  converter.io.input.last := acceptedWord === model.ffnSize - 1

  val memory = Mem(SInt(32 bits), model.ffnSize)
    .addAttribute("ram_style", "block")
  val convertedWord = Reg(UInt(cfg.gammaIndexWidth bits)) init 0
  converter.io.output.ready := state =/= State.Emit
  when(converter.io.output.fire) {
    memory.write(convertedWord, converter.io.output.value)
    when(converter.io.output.special || converter.io.output.overflow) {
      jobFault := True
      stickyFault := True
    }
    when(!converter.io.output.last) {
      convertedWord := convertedWord + 1
    }
  }

  when(converter.io.input.fire) {
    when(!converter.io.input.last) { acceptedWord := acceptedWord + 1 }
    when(beatWord === wordsPerBeat - 1) {
      beatHeld := False
      beatWord := 0
      when(heldBeatLast) {
        when(totalAcceptedBeats === cfg.gammaBeats) {
          state := State.WaitConversion
        } otherwise {
          burstOrdinal := burstOrdinal + 1
          state := State.IssueBurst
        }
      }
    } otherwise {
      beatWord := beatWord + 1
    }
  }

  val emitIndex = Reg(UInt(cfg.gammaIndexWidth bits)) init 0
  val readsComplete = RegInit(False)
  val gammaValid = RegInit(False)
  val gammaIndex = Reg(UInt(cfg.arithmetic.indexWidth bits)) init 0
  val gammaFirst = RegInit(False)
  val gammaLast = RegInit(False)
  val gammaCanAdvance = !gammaValid || io.gamma.ready
  val issueGammaRead = state === State.Emit && !readsComplete &&
    gammaCanAdvance
  val gammaValue = memory.readSync(emitIndex, enable = issueGammaRead)
  when(gammaCanAdvance) {
    gammaValid := issueGammaRead
    when(issueGammaRead) {
      gammaIndex := emitIndex.resized
      gammaFirst := emitIndex === 0
      gammaLast := emitIndex === model.ffnSize - 1
      when(emitIndex === model.ffnSize - 1) {
        readsComplete := True
      } otherwise {
        emitIndex := emitIndex + 1
      }
    }
  }

  io.gamma.valid := gammaValid
  io.gamma.requestId := held.requestId
  io.gamma.stepId := held.stepId
  io.gamma.layer := held.layer
  io.gamma.featureIndex := gammaIndex
  io.gamma.gammaQ16 := Mux(jobFault, S(0, 32 bits), gammaValue)
  io.gamma.first := gammaFirst
  io.gamma.last := gammaLast
  io.gamma.fault := jobFault

  when(io.request.fire) {
    held := io.request.payload
    vectorBase := layerBases(io.request.layer)
    jobFault := !requestLegal
    totalAcceptedBeats := 0
    burstOrdinal := 0
    expectedBeat := 0
    beatHeld := False
    beatWord := 0
    acceptedWord := 0
    convertedWord := 0
    emitIndex := 0
    readsComplete := False
    gammaValid := False
    transactionId := io.request.stepId.resized
    when(requestLegal) {
      state := State.IssueBurst
    } otherwise {
      stickyFault := True
      state := State.Emit
    }
  }

  when(io.readRequest.fire) {
    burstBeats := nextBurst
    expectedBeat := 0
    burstTag := io.readRequest.tag
    state := State.ReceiveBurst
  }

  when(io.readResponse.fire) {
    beatData := io.readResponse.data
    beatHeld := True
    beatWord := 0
    heldBeatLast :=
      expectedBeat.resize(crossbar.beatCountWidth) === burstBeats - 1
    totalAcceptedBeats := totalAcceptedBeats + 1
    when(!responseLegal) {
      jobFault := True
      stickyFault := True
    }
    when(!io.readResponse.last) { expectedBeat := expectedBeat + 1 }
  }

  when(state === State.WaitConversion && converter.io.output.fire &&
      converter.io.output.last) {
    emitIndex := 0
    readsComplete := False
    gammaValid := False
    state := State.Emit
  }

  when(io.gamma.fire && io.gamma.last) {
    gammaValid := False
    state := State.Idle
  }

  io.busy := state =/= State.Idle
  io.protocolFaultSticky := stickyFault
}

/**
  * Real post-gate FfnSubNorm activation coordinator.
  *
  * The component owns address/control state and one AUX gamma window only.  It
  * does not instantiate RMS arithmetic: all typed RMS streams terminate at
  * the public boundary so the model-wide shared RMS owner remains unique.
  */
class BitNetResidentFfnPostGateRmsSubsystem(
    cfg: BitNetResidentFfnPostGateRmsSubsystemConfig =
      BitNetResidentFfnPostGateRmsSubsystemConfig()) extends Component {
  private val model = cfg.model
  private val layout = cfg.activationLayout
  private val rms = cfg.arithmetic
  private val lanes = model.tokenParallelism
  private val stripes = model.spuCount

  val io = new Bundle {
    /** Atomic association of the gate publication and its resident layer. */
    val context = slave(Stream(
      BitNetResidentFfnPostGateRmsContext(model)))
    val gateRmsCommand = slave(Stream(BitNetSharedRmsCommand(rms)))

    /** The sole external shared RMS arithmetic boundary. */
    val rmsCommand = master(Stream(BitNetSharedRmsCommand(rms)))
    val rmsMeasure = master(Stream(BitNetSharedRmsMeasureBeat(rms)))
    val rmsReplayRequest = slave(Stream(
      BitNetSharedRmsReplayRequest(rms)))
    val rmsReplay = master(Stream(BitNetSharedRmsReplayBeat(rms)))
    val rmsResult = slave(Stream(BitNetSharedRmsResultBeat(rms)))
    val rmsDone = slave(Stream(BitNetSharedRmsDone(rms)))

    val activationReadCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val activationReadData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)
    val activationWriteData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    val auxReadRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar)))
    val auxReadResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar)))

    val completion = master(Stream(
      BitNetResidentFfnPostGateRmsCompletion(model)))
    val supported = out Bool()
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
  }

  val gamma = new BitNetResidentFfnSubNormGammaStager(cfg)
  io.auxReadRequest << gamma.io.readRequest
  gamma.io.readResponse << io.auxReadResponse

  object State extends SpinalEnum(binarySequential) {
    val Idle, StartOwners,
        MeasureIssue, MeasureCollect, MeasureEmit,
        WaitReplayRequest,
        ReplayIssue, ReplayCollect, ReplayEmit,
        CaptureResult, WriteResult, WaitDone, Complete = newElement()
  }
  val state = Reg(State()) init State.Idle
  val heldContext = Reg(BitNetResidentFfnPostGateRmsContext(model)) init
    BitNetResidentFfnPostGateRmsContext(model).getZero
  val heldCommand = Reg(BitNetSharedRmsCommand(rms)) init
    BitNetSharedRmsCommand(rms).getZero
  val readIssuePending = Reg(Bits(lanes bits)) init 0
  val readResponsePending = Reg(Bits(lanes bits)) init 0
  val writePending = Reg(Bits(lanes bits)) init 0
  val groupIndex = Reg(UInt(layout.groupIndexWidth bits)) init 0
  val stripeIndex = Reg(UInt(log2Up(stripes) bits)) init 0
  val groupValues = Vec(Vec(Reg(SInt(32 bits)) init 0, stripes), lanes)
  val resultValues = Vec(Reg(SInt(32 bits)) init 0, lanes)

  val measured = Reg(UInt(32 bits)) init 0
  val replayed = Reg(UInt(32 bits)) init 0
  val written = Reg(UInt(32 bits)) init 0
  val jobBindingFault = RegInit(False)
  val jobActivationFault = RegInit(False)
  val jobGammaFault = RegInit(False)
  val jobReplayRequestFault = RegInit(False)
  val jobResultFault = RegInit(False)
  val jobDoneFault = RegInit(False)
  val jobOverflow = RegInit(False)
  val stickyFault = RegInit(False)
  val stickyOverflow = RegInit(False)

  val layerInRange = if ((1 << model.layerWidth) == model.layerCount) True
    else io.context.layer < model.layerCount
  val bindingLegal = layerInRange && io.context.tokenMask.orR &&
    io.gateRmsCommand.length === model.ffnSize &&
    io.gateRmsCommand.tokenMask === io.context.tokenMask &&
    io.gateRmsCommand.sourceBuffer ===
      BitNetPhysicalActivationBuffer.WideWork &&
    io.gateRmsCommand.tag === io.context.stepId.resize(rms.tagWidth)
  val atomicReady = state === State.Idle && io.context.valid &&
    io.gateRmsCommand.valid
  io.context.ready := state === State.Idle && io.gateRmsCommand.valid
  io.gateRmsCommand.ready := state === State.Idle && io.context.valid
  io.supported := bindingLegal

  when(atomicReady) {
    heldContext := io.context.payload
    heldCommand := io.gateRmsCommand.payload
    measured := 0
    replayed := 0
    written := 0
    groupIndex := 0
    stripeIndex := 0
    readIssuePending := 0
    readResponsePending := 0
    writePending := 0
    jobBindingFault := !bindingLegal
    jobActivationFault := False
    jobGammaFault := False
    jobReplayRequestFault := False
    jobResultFault := False
    jobDoneFault := False
    jobOverflow := False
    when(bindingLegal) {
      state := State.StartOwners
    } otherwise {
      stickyFault := True
      state := State.Complete
    }
  }

  gamma.io.request.valid := state === State.StartOwners &&
    io.rmsCommand.ready
  gamma.io.request.requestId := heldContext.requestId
  gamma.io.request.stepId := heldContext.stepId
  gamma.io.request.layer := heldContext.layer
  gamma.io.request.length := model.ffnSize
  io.rmsCommand.valid := state === State.StartOwners &&
    gamma.io.request.ready
  io.rmsCommand.payload := heldCommand
  when(gamma.io.request.fire && io.rmsCommand.fire) {
    readIssuePending := heldContext.tokenMask
    state := State.MeasureIssue
  }

  val currentFeature = UInt(rms.indexWidth bits)
  currentFeature :=
    ((groupIndex.resize(rms.indexWidth) |<< log2Up(stripes)) +
      stripeIndex.resize(rms.indexWidth)).resized
  val currentFirst = currentFeature === 0
  val currentLast = currentFeature === model.ffnSize - 1

  val readCommandFire = Bits(lanes bits)
  val readResponseFire = Bits(lanes bits)
  val writeFire = Bits(lanes bits)
  readCommandFire := 0
  readResponseFire := 0
  writeFire := 0
  for (lane <- 0 until lanes) {
    io.activationReadCommand(lane).valid :=
      (state === State.MeasureIssue || state === State.ReplayIssue) &&
        readIssuePending(lane)
    io.activationReadCommand(lane).buffer :=
      BitNetPhysicalActivationBuffer.WideWork
    io.activationReadCommand(lane).group := groupIndex
    readCommandFire(lane) := io.activationReadCommand(lane).fire

    io.activationReadData(lane).ready :=
      (state === State.MeasureCollect || state === State.ReplayCollect) &&
        readResponsePending(lane)
    readResponseFire(lane) := io.activationReadData(lane).fire

    io.activationWriteData(lane).valid := state === State.WriteResult &&
      writePending(lane)
    io.activationWriteData(lane).buffer :=
      BitNetPhysicalActivationBuffer.WideWork
    io.activationWriteData(lane).group := groupIndex
    io.activationWriteData(lane).featureValid :=
      UIntToOh(stripeIndex, stripes).asBits
    io.activationWriteData(lane).values.foreach(_ := 0)
    io.activationWriteData(lane).values(stripeIndex) := resultValues(lane)
    writeFire(lane) := io.activationWriteData(lane).fire
  }

  val remainingReadCommands = readIssuePending & ~readCommandFire
  val remainingReadResponses = readResponsePending & ~readResponseFire
  val remainingWrites = writePending & ~writeFire

  io.rmsMeasure.valid := state === State.MeasureEmit
  io.rmsMeasure.tokenMask := heldContext.tokenMask
  io.rmsMeasure.first := currentFirst
  io.rmsMeasure.last := currentLast
  for (lane <- 0 until lanes) {
    io.rmsMeasure.value(lane) := groupValues(lane)(stripeIndex)
  }

  io.rmsReplayRequest.ready := state === State.WaitReplayRequest
  val replayRequestLegal =
    io.rmsReplayRequest.length === model.ffnSize &&
      io.rmsReplayRequest.tokenMask === heldContext.tokenMask &&
      io.rmsReplayRequest.sourceBuffer ===
        BitNetPhysicalActivationBuffer.WideWork &&
      io.rmsReplayRequest.tag === heldContext.stepId.resize(rms.tagWidth) &&
      io.rmsReplayRequest.inPlace

  val gammaIdentityLegal =
    gamma.io.gamma.requestId === heldContext.requestId &&
      gamma.io.gamma.stepId === heldContext.stepId &&
      gamma.io.gamma.layer === heldContext.layer &&
      gamma.io.gamma.featureIndex === currentFeature &&
      gamma.io.gamma.first === currentFirst &&
      gamma.io.gamma.last === currentLast && !gamma.io.gamma.fault
  io.rmsReplay.valid := state === State.ReplayEmit && gamma.io.gamma.valid
  io.rmsReplay.tokenMask := heldContext.tokenMask
  io.rmsReplay.first := currentFirst
  io.rmsReplay.last := currentLast
  io.rmsReplay.gamma := Mux(gammaIdentityLegal,
    gamma.io.gamma.gammaQ16, S(0, rms.dataWidth bits))
  for (lane <- 0 until lanes) {
    io.rmsReplay.value(lane) := groupValues(lane)(stripeIndex)
  }
  gamma.io.gamma.ready := state === State.ReplayEmit && io.rmsReplay.ready

  io.rmsResult.ready := state === State.CaptureResult
  val resultIdentityLegal =
    io.rmsResult.tag === heldContext.stepId.resize(rms.tagWidth) &&
      io.rmsResult.featureIndex === currentFeature &&
      io.rmsResult.tokenMask === heldContext.tokenMask &&
      io.rmsResult.first === currentFirst &&
      io.rmsResult.last === currentLast
  io.rmsDone.ready := state === State.WaitDone

  val jobFailed = jobBindingFault || jobActivationFault || jobGammaFault ||
    jobReplayRequestFault || jobResultFault || jobDoneFault || jobOverflow ||
    measured =/= model.ffnSize || replayed =/= model.ffnSize ||
    written =/= model.ffnSize
  io.completion.valid := state === State.Complete
  io.completion.requestId := heldContext.requestId
  io.completion.stepId := heldContext.stepId
  io.completion.layer := heldContext.layer
  io.completion.mode := heldContext.mode
  io.completion.tokenMask := heldContext.tokenMask
  io.completion.success := !jobFailed
  io.completion.faultCode := 0
  when(jobBindingFault) {
    io.completion.faultCode := B(
      BitNetResidentFfnPostGateRmsFault.InvalidBinding, 8 bits)
  } elsewhen(jobActivationFault) {
    io.completion.faultCode := B(
      BitNetResidentFfnPostGateRmsFault.ActivationMetadata, 8 bits)
  } elsewhen(jobGammaFault) {
    io.completion.faultCode := B(
      BitNetResidentFfnPostGateRmsFault.GammaMetadata, 8 bits)
  } elsewhen(jobReplayRequestFault) {
    io.completion.faultCode := B(
      BitNetResidentFfnPostGateRmsFault.ReplayRequest, 8 bits)
  } elsewhen(jobResultFault) {
    io.completion.faultCode := B(
      BitNetResidentFfnPostGateRmsFault.RmsResult, 8 bits)
  } elsewhen(jobDoneFault) {
    io.completion.faultCode := B(
      BitNetResidentFfnPostGateRmsFault.RmsDone, 8 bits)
  } elsewhen(jobOverflow) {
    io.completion.faultCode := B(
      BitNetResidentFfnPostGateRmsFault.Overflow, 8 bits)
  } elsewhen(measured =/= model.ffnSize || replayed =/= model.ffnSize ||
      written =/= model.ffnSize) {
    io.completion.faultCode := B(
      BitNetResidentFfnPostGateRmsFault.FeatureCount, 8 bits)
  }
  io.completion.measuredFeatures := measured
  io.completion.replayedFeatures := replayed
  io.completion.writtenFeatures := written

  switch(state) {
    is(State.MeasureIssue, State.ReplayIssue) {
      readIssuePending := remainingReadCommands
      when(remainingReadCommands === 0) {
        readResponsePending := heldContext.tokenMask
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
            groupValues(lane)(stripe) := io.activationReadData(lane).values(stripe)
          }
          when(io.activationReadData(lane).fault ||
              io.activationReadData(lane).buffer =/=
                BitNetPhysicalActivationBuffer.WideWork ||
              io.activationReadData(lane).group =/= groupIndex ||
              io.activationReadData(lane).featureValid =/= B"4'b1111") {
            jobActivationFault := True
            stickyFault := True
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
      when(io.rmsMeasure.fire) {
        measured := measured + 1
        when(currentLast) {
          groupIndex := 0
          stripeIndex := 0
          state := State.WaitReplayRequest
        } elsewhen(stripeIndex === stripes - 1) {
          groupIndex := groupIndex + 1
          stripeIndex := 0
          readIssuePending := heldContext.tokenMask
          state := State.MeasureIssue
        } otherwise {
          stripeIndex := stripeIndex + 1
        }
      }
    }

    is(State.WaitReplayRequest) {
      when(io.rmsReplayRequest.fire) {
        when(!replayRequestLegal) {
          jobReplayRequestFault := True
          stickyFault := True
        }
        groupIndex := 0
        stripeIndex := 0
        readIssuePending := heldContext.tokenMask
        state := State.ReplayIssue
      }
    }

    is(State.ReplayEmit) {
      when(io.rmsReplay.fire) {
        replayed := replayed + 1
        when(!gammaIdentityLegal) {
          jobGammaFault := True
          stickyFault := True
        }
        state := State.CaptureResult
      }
    }

    is(State.CaptureResult) {
      when(io.rmsResult.fire) {
        for (lane <- 0 until lanes) {
          resultValues(lane) := Mux(resultIdentityLegal,
            io.rmsResult.value(lane), S(0, 32 bits))
        }
        when(!resultIdentityLegal) {
          jobResultFault := True
          stickyFault := True
        }
        writePending := heldContext.tokenMask
        state := State.WriteResult
      }
    }

    is(State.WriteResult) {
      writePending := remainingWrites
      when(remainingWrites === 0) {
        written := written + 1
        when(currentLast) {
          state := State.WaitDone
        } elsewhen(stripeIndex === stripes - 1) {
          groupIndex := groupIndex + 1
          stripeIndex := 0
          readIssuePending := heldContext.tokenMask
          state := State.ReplayIssue
        } otherwise {
          stripeIndex := stripeIndex + 1
          state := State.ReplayEmit
        }
      }
    }

    is(State.WaitDone) {
      when(io.rmsDone.fire) {
        when(io.rmsDone.tag =/= heldContext.stepId.resize(rms.tagWidth) ||
            io.rmsDone.protocolFault) {
          jobDoneFault := True
          stickyFault := True
        }
        when(io.rmsDone.overflow) {
          jobOverflow := True
          stickyOverflow := True
        }
        state := State.Complete
      }
    }

    is(State.Complete) {
      when(io.completion.fire) { state := State.Idle }
    }
  }

  io.busy := state =/= State.Idle || gamma.io.busy
  io.protocolFaultSticky := stickyFault || gamma.io.protocolFaultSticky
  io.overflowSticky := stickyOverflow
}

object GenerateBitNetResidentFfnPostGateRmsSubsystem extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-ffn-post-gate-rms-subsystem")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentFfnPostGateRmsSubsystem())
}
