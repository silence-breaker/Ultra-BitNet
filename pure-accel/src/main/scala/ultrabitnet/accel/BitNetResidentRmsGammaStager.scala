package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Bank-local addresses of the five layer/final norm arrays. */
case class BitNetResidentRmsGammaAddressLayout(
    finalNormBase: BigInt,
    inputNormBase: BigInt,
    postAttentionNormBase: BigInt,
    attentionSubNormBase: BigInt,
    ffnSubNormBase: BigInt
) {
  def validate(model: BitNetConfig, addressWidth: Int):
      BitNetResidentRmsGammaAddressLayout = {
    val limit = BigInt(1) << addressWidth
    val hiddenBytes = BigInt(model.hiddenSize) * 2
    val ffnBytes = BigInt(model.ffnSize) * 2
    val ranges = Seq(
      finalNormBase -> hiddenBytes,
      inputNormBase -> (BigInt(model.layerCount) * hiddenBytes),
      postAttentionNormBase -> (BigInt(model.layerCount) * hiddenBytes),
      attentionSubNormBase -> (BigInt(model.layerCount) * hiddenBytes),
      ffnSubNormBase -> (BigInt(model.layerCount) * ffnBytes))
    require(ranges.forall { case (base, bytes) =>
      base >= 0 && (base & 1) == 0 && bytes > 0 && base + bytes <= limit
    }, "RMS gamma arrays escape the selected bank-local address window")
    this
  }
}

object BitNetResidentRmsGammaAddressLayout {
  /** AUX.BIN is replicated in every five-bank image; bank zero is PL DDR. */
  def production(bank: Int = 0): BitNetResidentRmsGammaAddressLayout = {
    require(bank >= 0 && bank < BitNetFiveBankStripedLayout.BankCount)
    val resident = BitNetFiveBankStripedLayout()
    val aux = resident.auxBases(bank)
    BitNetResidentRmsGammaAddressLayout(
      finalNormBase = aux + BitNetFiveBankStripedLayout.FinalNormOffset,
      inputNormBase = aux + BitNetFiveBankStripedLayout.InputNormOffset,
      postAttentionNormBase = aux + BitNetFiveBankStripedLayout.PostNormOffset,
      attentionSubNormBase =
        aux + BitNetFiveBankStripedLayout.AttentionSubNormOffset,
      ffnSubNormBase = aux + BitNetFiveBankStripedLayout.FfnSubNormOffset)
  }
}

case class BitNetResidentRmsGammaStagerConfig(
    endpoint: BitNetResidentRmsNormEndpointConfig =
      BitNetResidentRmsNormEndpointConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    bank: Int = 0,
    addresses: BitNetResidentRmsGammaAddressLayout =
      BitNetResidentRmsGammaAddressLayout.production()
) {
  val model: BitNetConfig = endpoint.model
  require(crossbar.dataWidth == 128)
  require(bank >= 0 && bank < crossbar.bankCount)
  require(model.hiddenSize % 8 == 0,
    "one 128-bit AUX beat must contain eight complete BF16 gamma words")
  require(crossbar.maxBurstBeats == 256,
    "the production stager splits gamma reads at the AXI 256-beat limit")
  addresses.validate(model, crossbar.localAddressWidth)

  val wordsPerBeat: Int = crossbar.dataWidth / 16
  val totalBeats: Int = model.hiddenSize / wordsPerBeat
  val totalBeatWidth: Int = log2Up(totalBeats + 1)
  val bufferIndexWidth: Int = log2Up(model.hiddenSize)
  val burstOrdinalWidth: Int = scala.math.max(1,
    log2Up((totalBeats + crossbar.maxBurstBeats - 1) /
      crossbar.maxBurstBeats + 1))
}

/**
  * Resident AUX.BIN gamma prefetch and format boundary.
  *
  * The request is accepted at the beginning of the RMS measure pass.  The
  * complete BF16 vector is fetched from the replicated AUX image, converted
  * to signed Q16.16 and placed in one BRAM window before the arithmetic core
  * asks for replay.  Consequently no AXI burst remains open while RMS output
  * is backpressured, and the conversion/read latency is hidden by measure,
  * mean, sqrt and reciprocal work.
  *
  * One 128-bit response beat is intentionally unpacked over eight cycles.  A
  * hidden-size vector still fills faster than the shared RMS arithmetic can
  * finish its first pass, while the narrow converter and single-port BRAM
  * cost much less than an eight-way floating-point boundary.
  */
class BitNetResidentRmsGammaStager(
    cfg: BitNetResidentRmsGammaStagerConfig =
      BitNetResidentRmsGammaStagerConfig()) extends Component {
  private val model = cfg.model
  private val endpoint = cfg.endpoint
  private val crossbar = cfg.crossbar
  private val wordsPerBeat = cfg.wordsPerBeat

  val io = new Bundle {
    val request = slave(Stream(BitNetResidentRmsGammaRequest(endpoint)))
    val readRequest = master(Stream(BitNetClientBankReadRequest(crossbar)))
    val readResponse = slave(Stream(BitNetClientBankReadResponse(crossbar)))
    val gamma = master(Stream(BitNetResidentRmsGammaBeat(endpoint)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedRequests = out UInt(32 bits)
    val acceptedReadBeats = out UInt(32 bits)
    val emittedGammaWords = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, IssueBurst, ReceiveBurst, WaitConversion, Emit = newElement()
  }
  val state = Reg(State()) init State.Idle
  io.busy := state =/= State.Idle

  val held = Reg(BitNetResidentRmsGammaRequest(endpoint)) init
    BitNetResidentRmsGammaRequest(endpoint).getZero
  val vectorBase = Reg(UInt(crossbar.localAddressWidth bits)) init 0
  val jobFault = RegInit(False)
  val globalFault = RegInit(False)
  io.protocolFaultSticky := globalFault

  val acceptedRequestCount = Reg(UInt(32 bits)) init 0
  val acceptedReadBeatCount = Reg(UInt(32 bits)) init 0
  val emittedGammaCount = Reg(UInt(32 bits)) init 0
  io.acceptedRequests := acceptedRequestCount
  io.acceptedReadBeats := acceptedReadBeatCount
  io.emittedGammaWords := emittedGammaCount

  val layerInRange = Bool()
  if ((1 << model.layerWidth) == model.layerCount) {
    layerInRange := True
  } else {
    layerInRange := io.request.payload.layer <
      U(model.layerCount, model.layerWidth bits)
  }
  val stageLegal =
    io.request.payload.stage === BitNetResidentLayerStage.AttentionRmsNorm ||
      io.request.payload.stage ===
        BitNetResidentLayerStage.AttentionSubRmsNorm ||
      io.request.payload.stage === BitNetResidentLayerStage.FfnRmsNorm ||
      io.request.payload.stage === BitNetResidentLayerStage.FinalRmsNorm
  val requestLegal = stageLegal && layerInRange &&
    io.request.payload.length === model.hiddenSize

  val finalBase = U(cfg.addresses.finalNormBase,
    crossbar.localAddressWidth bits)
  val inputBases = Vec(UInt(crossbar.localAddressWidth bits), model.layerCount)
  val postBases = Vec(UInt(crossbar.localAddressWidth bits), model.layerCount)
  val attentionSubBases =
    Vec(UInt(crossbar.localAddressWidth bits), model.layerCount)
  for (layer <- 0 until model.layerCount) {
    val stride = BigInt(layer) * model.hiddenSize * 2
    inputBases(layer) := U(cfg.addresses.inputNormBase + stride,
      crossbar.localAddressWidth bits)
    postBases(layer) := U(cfg.addresses.postAttentionNormBase + stride,
      crossbar.localAddressWidth bits)
    attentionSubBases(layer) := U(
      cfg.addresses.attentionSubNormBase + stride,
      crossbar.localAddressWidth bits)
  }

  val selectedBase = UInt(crossbar.localAddressWidth bits)
  selectedBase := finalBase
  switch(io.request.payload.stage) {
    is(BitNetResidentLayerStage.AttentionRmsNorm) {
      selectedBase := inputBases(io.request.payload.layer)
    }
    is(BitNetResidentLayerStage.AttentionSubRmsNorm) {
      selectedBase := attentionSubBases(io.request.payload.layer)
    }
    is(BitNetResidentLayerStage.FfnRmsNorm) {
      selectedBase := postBases(io.request.payload.layer)
    }
    is(BitNetResidentLayerStage.FinalRmsNorm) {
      selectedBase := finalBase
    }
  }

  io.request.ready := state === State.Idle

  val totalAcceptedBeats = Reg(UInt(cfg.totalBeatWidth bits)) init 0
  val burstBeats = Reg(UInt(crossbar.beatCountWidth bits)) init 0
  val expectedBeatIndex = Reg(UInt(crossbar.beatIndexWidth bits)) init 0
  val burstOrdinal = Reg(UInt(cfg.burstOrdinalWidth bits)) init 0
  val burstTag = Reg(UInt(crossbar.tagWidth bits)) init 0
  val transactionId = Reg(UInt(crossbar.transactionIdWidth bits)) init 0

  val remainingBeats = UInt(cfg.totalBeatWidth bits)
  remainingBeats := U(cfg.totalBeats, cfg.totalBeatWidth bits) -
    totalAcceptedBeats
  val nextBurstBeats = UInt(crossbar.beatCountWidth bits)
  nextBurstBeats := remainingBeats.resized
  if (cfg.totalBeats > crossbar.maxBurstBeats) {
    when(remainingBeats > crossbar.maxBurstBeats) {
      nextBurstBeats := crossbar.maxBurstBeats
    }
  }

  io.readRequest.valid := state === State.IssueBurst
  io.readRequest.payload.bank := cfg.bank
  io.readRequest.payload.transactionId := transactionId
  io.readRequest.payload.tag :=
    (held.stepId.resize(crossbar.tagWidth) ^
      held.requestId.resize(crossbar.tagWidth) ^
      burstOrdinal.resize(crossbar.tagWidth)).resized
  io.readRequest.payload.localAddress :=
    (vectorBase +
      (totalAcceptedBeats.resize(crossbar.localAddressWidth) |<< 4)).resized
  io.readRequest.payload.beats := nextBurstBeats

  val beatHeld = RegInit(False)
  val beatData = Reg(Bits(crossbar.dataWidth bits)) init 0
  val beatWord = Reg(UInt(log2Up(wordsPerBeat) bits)) init 0
  val heldBeatLastInBurst = RegInit(False)
  val words = beatData.subdivideIn(16 bits)

  io.readResponse.ready := state === State.ReceiveBurst && !beatHeld
  val responseMetadataLegal =
    io.readResponse.payload.bank === cfg.bank &&
      io.readResponse.payload.transactionId === transactionId &&
      io.readResponse.payload.tag === burstTag &&
      io.readResponse.payload.beatIndex === expectedBeatIndex &&
      io.readResponse.payload.last ===
        (expectedBeatIndex.resize(crossbar.beatCountWidth) === burstBeats - 1) &&
      !io.readResponse.payload.fault &&
      io.readResponse.payload.rresp === B"2'b00"

  val converter = new NativeBf16Q16Staging
  val acceptedWordIndex = Reg(UInt(cfg.bufferIndexWidth bits)) init 0
  converter.io.input.valid := beatHeld
  converter.io.input.payload.raw := words(beatWord)
  converter.io.input.payload.first := acceptedWordIndex === 0
  converter.io.input.payload.last := acceptedWordIndex === model.hiddenSize - 1

  val gammaMemory = Mem(SInt(32 bits), model.hiddenSize)
    .addAttribute("ram_style", "block")
  val convertedWordIndex = Reg(UInt(cfg.bufferIndexWidth bits)) init 0
  converter.io.output.ready := state =/= State.Emit
  when(converter.io.output.fire) {
    gammaMemory.write(convertedWordIndex, converter.io.output.payload.value)
    when(converter.io.output.payload.special ||
      converter.io.output.payload.overflow) {
      jobFault := True
      globalFault := True
    }
    when(!converter.io.output.payload.last) {
      convertedWordIndex := convertedWordIndex + 1
    }
  }

  when(converter.io.input.fire) {
    when(!converter.io.input.payload.last) {
      acceptedWordIndex := acceptedWordIndex + 1
    }
    when(beatWord === wordsPerBeat - 1) {
      beatHeld := False
      beatWord := 0
      when(heldBeatLastInBurst) {
        when(totalAcceptedBeats === cfg.totalBeats) {
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

  val emitIndex = Reg(UInt(cfg.bufferIndexWidth bits)) init 0
  val allEmitReadsIssued = RegInit(False)
  val gammaValid = RegInit(False)
  val gammaIndex = Reg(UInt(endpoint.arithmetic.indexWidth bits)) init 0
  val gammaFirst = RegInit(False)
  val gammaLast = RegInit(False)
  val gammaReadCanAdvance = !gammaValid || io.gamma.ready
  val gammaReadIssue = state === State.Emit &&
    !allEmitReadsIssued && gammaReadCanAdvance
  val gammaReadValue = gammaMemory.readSync(emitIndex,
    enable = gammaReadIssue)

  when(gammaReadCanAdvance) {
    gammaValid := gammaReadIssue
    when(gammaReadIssue) {
      gammaIndex := emitIndex.resized
      gammaFirst := emitIndex === 0
      gammaLast := emitIndex === model.hiddenSize - 1
      when(emitIndex === model.hiddenSize - 1) {
        allEmitReadsIssued := True
      } otherwise {
        emitIndex := emitIndex + 1
      }
    }
  }

  io.gamma.valid := gammaValid
  io.gamma.payload.requestId := held.requestId
  io.gamma.payload.stepId := held.stepId
  io.gamma.payload.stage := held.stage
  io.gamma.payload.layer := held.layer
  io.gamma.payload.featureIndex := gammaIndex
  io.gamma.payload.gammaQ16 := Mux(jobFault, S(0, 32 bits), gammaReadValue)
  io.gamma.payload.first := gammaFirst
  io.gamma.payload.last := gammaLast
  io.gamma.payload.fault := jobFault

  when(io.request.fire) {
    held := io.request.payload
    vectorBase := selectedBase
    jobFault := !requestLegal
    totalAcceptedBeats := 0
    burstOrdinal := 0
    expectedBeatIndex := 0
    beatHeld := False
    beatWord := 0
    acceptedWordIndex := 0
    convertedWordIndex := 0
    emitIndex := 0
    allEmitReadsIssued := False
    gammaValid := False
    transactionId := io.request.payload.stepId.resized
    acceptedRequestCount := acceptedRequestCount + 1
    when(requestLegal) {
      state := State.IssueBurst
    } otherwise {
      globalFault := True
      // Drain a deterministic all-zero, fault-marked vector so an upstream
      // endpoint can retire its tagged stage instead of hanging forever.
      state := State.Emit
    }
  }

  when(state === State.IssueBurst && io.readRequest.fire) {
    burstBeats := nextBurstBeats
    expectedBeatIndex := 0
    burstTag := io.readRequest.payload.tag
    state := State.ReceiveBurst
  }

  when(io.readResponse.fire) {
    acceptedReadBeatCount := acceptedReadBeatCount + 1
    beatData := io.readResponse.payload.data
    beatHeld := True
    beatWord := 0
    // Advance by the declared request geometry even when malformed AXI
    // metadata is reported, so the tagged job drains deterministically.
    heldBeatLastInBurst :=
      expectedBeatIndex.resize(crossbar.beatCountWidth) === burstBeats - 1
    totalAcceptedBeats := totalAcceptedBeats + 1
    when(!responseMetadataLegal) {
      jobFault := True
      globalFault := True
    }
    when(!io.readResponse.payload.last) {
      expectedBeatIndex := expectedBeatIndex + 1
    }
  }

  when(state === State.WaitConversion && converter.io.output.fire &&
    converter.io.output.payload.last) {
    emitIndex := 0
    allEmitReadsIssued := False
    gammaValid := False
    state := State.Emit
  }

  when(io.gamma.fire) {
    emittedGammaCount := emittedGammaCount + 1
    when(io.gamma.payload.last) {
      gammaValid := False
      state := State.Idle
    }
  }
}

object GenerateBitNetResidentRmsGammaStager extends App {
  SpinalConfig(
    targetDirectory = "build/bitnet-resident-rms-gamma-stager",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetResidentRmsGammaStager())
}
