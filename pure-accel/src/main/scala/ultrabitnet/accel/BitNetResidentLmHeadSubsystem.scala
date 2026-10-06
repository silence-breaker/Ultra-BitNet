package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Memory-side geometry for one resident LM row-pair fetch. */
case class BitNetResidentLmFetchConfig(
    resident: BitNetResidentStreamingLmHeadConfig =
      BitNetResidentStreamingLmHeadConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    epochWidth: Int = 8
) {
  val model: BitNetConfig = resident.model
  val lm: BitNetStreamingLmHeadConfig = resident.streaming
  val feeder: BitNetContinuousMatrixFeederConfig = resident.linear.feeder
  val bankCount: Int = lm.bankCount
  val engineCount: Int = lm.engineCount
  val slotCount: Int = 2
  val slotWidth: Int = log2Up(slotCount)
  val beatBytes: Int = crossbar.dataWidth / 8
  val segmentsPerTile: Int = lm.tileFeatures / beatBytes
  val maxStripeFeatures: Int = lm.stripeInputFeatures.max
  val maxStripeBeats: Int = maxStripeFeatures / beatBytes
  val responseBeatWidth: Int = log2Up(maxStripeBeats + 1)
  val expectedPairWidth: Int = log2Up(lm.pairCount + 1)
  val encodedTagWidth: Int =
    epochWidth + lm.requestIdWidth + 1 + lm.engineWidth + lm.bankWidth +
      slotWidth
  val encodedTransactionWidth: Int =
    1 + lm.engineWidth + lm.bankWidth + slotWidth
  val bytesPerRowPair: Int = engineCount * lm.hiddenSize
  val totalWeightBufferBytes: Int = slotCount * bytesPerRowPair

  require(bankCount == 5 && (engineCount == 1 || engineCount == 2))
  require(slotCount == 2)
  require(crossbar.bankCount == bankCount && crossbar.dataWidth == 128)
  require(crossbar.localAddressWidth == lm.addressWidth)
  require((lm.tileFeatures == 64 && segmentsPerTile == 4) ||
    (lm.tileFeatures == 128 && segmentsPerTile == 8))
  require(lm.stripeInputStarts.zip(lm.stripeInputFeatures).forall {
    case (start, features) =>
      start % lm.tileFeatures == 0 && features % lm.tileFeatures == 0
  })
  require(maxStripeFeatures % beatBytes == 0)
  require(maxStripeBeats <= crossbar.maxBurstBeats)
  require(epochWidth >= 4)
  require(encodedTagWidth <= crossbar.tagWidth,
    s"LM fetch identity needs $encodedTagWidth tag bits")
  require(encodedTransactionWidth <= crossbar.transactionIdWidth)
}

case class BitNetResidentLmFetchJob(cfg: BitNetResidentLmFetchConfig)
    extends Bundle {
  val sequenceId = UInt(cfg.resident.linear.matrix.sequenceIdWidth bits)
  val epoch = UInt(cfg.epochWidth bits)
  val cacheGeneration = UInt(16 bits)
}

object BitNetResidentLmFetchFault {
  val JobProtocol = 0
  val RowDescriptor = 1
  val ResponseIdentity = 2
  val AxiResponse = 3
  val UnexpectedResponse = 4
}

object BitNetResidentLmFetchKind extends Enumeration {
  val Weight, Scale = Value
}

/** Pure-Scala address oracle shared by directed tests and image audits. */
object BitNetResidentLmFetchMath {
  case class Read(
      kind: BitNetResidentLmFetchKind.Value,
      engine: Int,
      tokenId: Int,
      requestId: Int,
      bank: Int,
      localAddress: BigInt,
      beats: Int,
      scaleWordLane: Int)

  def pair(
      cfg: BitNetStreamingLmHeadConfig,
      pairOrdinal: Int,
      requestIdBase: Int = 0): Vector[Read] = {
    require(pairOrdinal >= 0 && pairOrdinal < cfg.pairCount)
    val reads = Vector.newBuilder[Read]
    for (engine <- 0 until cfg.engineCount) {
      val token = pairOrdinal * cfg.engineCount + engine
      if (token < cfg.vocabularySize) {
        val request = (requestIdBase + engine) &
          ((1 << cfg.requestIdWidth) - 1)
        for (bank <- 0 until cfg.bankCount) {
          reads += Read(
            BitNetResidentLmFetchKind.Weight,
            engine,
            token,
            request,
            bank,
            cfg.stripeWeightBases(bank) +
              BigInt(token) * cfg.stripeInputFeatures(bank),
            cfg.stripeInputFeatures(bank) / 16,
            0)
        }
        val owner = cfg.scaleRowStarts.zip(cfg.scaleRows).indexWhere {
          case (start, rows) => token >= start && token < start + rows
        }
        require(owner >= 0)
        val scaleAddress = cfg.scaleBases(owner) +
          BigInt(token - cfg.scaleRowStarts(owner)) * 4
        reads += Read(
          BitNetResidentLmFetchKind.Scale,
          engine,
          token,
          request,
          owner,
          scaleAddress & ~BigInt(15),
          1,
          ((scaleAddress & 15) / 4).toInt)
      }
    }
    reads.result()
  }
}

object BitNetResidentLmBankPhase extends SpinalEnum(binarySequential) {
  val Idle, IssueWeight0, ReceiveWeight0, IssueWeight1, ReceiveWeight1,
      IssueScale0, ReceiveScale0, IssueScale1, ReceiveScale1, Done =
    newElement()
}

/**
  * Five-bank LM row-pair fetcher.
  *
  * One logical burst is outstanding per bank.  The five banks run in
  * parallel; within a bank the even row, odd row and the locally-owned scale
  * word(s) are ordered.  This deliberately avoids relying on AXI response
  * interleaving for correctness and makes a corrupt transaction/tag fail
  * closed without losing the burst that must be drained.
  *
  * Each engine owns eight rotated 128-bit memories.  Five simultaneous bank
  * responses therefore use at most one write port per memory when aligned;
  * skew collisions are backpressured.  Replay reads the eight memories once
  * and restores one natural 128-byte typed-I8 tile.  No vocabulary-sized
  * weight or logit store exists here.
  */
private[accel] class BitNetResidentLmWeightScaleSingleSlotReference(
    cfg: BitNetResidentLmFetchConfig = BitNetResidentLmFetchConfig())
    extends Component {
  private val lm = cfg.lm
  private val matrix = cfg.resident.linear.matrix
  private val crossbar = cfg.crossbar

  val io = new Bundle {
    val jobStart = slave(Flow(BitNetResidentLmFetchJob(cfg)))
    val jobRetire = in Bool()
    val rowPair = slave(Stream(BitNetLmRowPairRequest(lm)))

    val readRequest = Vec(master(Stream(
      BitNetClientBankReadRequest(crossbar))), cfg.bankCount)
    val readResponse = Vec(slave(Stream(
      BitNetClientBankReadResponse(crossbar))), cfg.bankCount)

    val i8Weight = master(Stream(
      BitNetI8WeightBeat(cfg.resident.linear.feeder)))
    val rowScale = master(Stream(
      BitNetResidentLmRowScaleResponse(cfg.resident)))

    val canStartJob = out Bool()
    val jobActive = out Bool()
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(8 bits)
    val issuedReadRequests = out UInt(32 bits)
    val acceptedReadResponses = out UInt(32 bits)
    val emittedWeightTiles = out UInt(32 bits)
    val emittedScales = out UInt(32 bits)
  }

  val activeJob = RegInit(False)
  val activeSequenceId = Reg(UInt(matrix.sequenceIdWidth bits)) init 0
  val activeEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val expectedPair = Reg(UInt(cfg.expectedPairWidth bits)) init 0
  val nextWeightBeatOrdinal = Reg(UInt(matrix.beatCountWidth bits)) init 0
  val pairActive = RegInit(False)
  val pairReg = Reg(BitNetLmRowPairRequest(lm)) init
    BitNetLmRowPairRequest(lm).getZero
  val pairFault = RegInit(False)

  val stickyFault = RegInit(False)
  val faultBitsReg = Reg(Bits(8 bits)) init 0
  val issuedReadRequestCount = Reg(UInt(32 bits)) init 0
  val acceptedReadResponseCount = Reg(UInt(32 bits)) init 0
  val emittedWeightTileCount = Reg(UInt(32 bits)) init 0
  val emittedScaleCount = Reg(UInt(32 bits)) init 0

  io.canStartJob := !activeJob && !pairActive
  io.jobActive := activeJob
  io.busy := pairActive
  io.protocolFaultSticky := stickyFault
  io.faultBits := faultBitsReg
  io.issuedReadRequests := issuedReadRequestCount
  io.acceptedReadResponses := acceptedReadResponseCount
  io.emittedWeightTiles := emittedWeightTileCount
  io.emittedScales := emittedScaleCount

  when(io.jobStart.valid) {
    when(!activeJob && !pairActive) {
      activeJob := True
      activeSequenceId := io.jobStart.sequenceId
      activeEpoch := io.jobStart.epoch
      expectedPair := 0
      nextWeightBeatOrdinal := 0
    } otherwise {
      stickyFault := True
      faultBitsReg(BitNetResidentLmFetchFault.JobProtocol) := True
    }
  }

  when(io.jobRetire) {
    when(!activeJob || pairActive || expectedPair =/= lm.pairCount) {
      stickyFault := True
      faultBitsReg(BitNetResidentLmFetchFault.JobProtocol) := True
    }
    activeJob := False
  }

  val expectedEngineMask = Bits(cfg.engineCount bits)
  val expectedToken = Vec(UInt(lm.tokenIdWidth bits), cfg.engineCount)
  expectedEngineMask := 0
  for (engine <- 0 until cfg.engineCount) {
    val row = io.rowPair.pairOrdinal.resize(lm.tokenIdWidth + 1) + engine
    expectedToken(engine) := row.resize(lm.tokenIdWidth)
    expectedEngineMask(engine) := row < lm.vocabularySize
  }

  val descriptorChecks = scala.collection.mutable.ArrayBuffer[Bool](
    activeJob,
    io.rowPair.pairOrdinal.resize(cfg.expectedPairWidth) === expectedPair,
    io.rowPair.engineMask === expectedEngineMask,
    io.rowPair.firstPair === (io.rowPair.pairOrdinal === 0),
    io.rowPair.lastPair ===
      (io.rowPair.pairOrdinal === lm.pairCount - 1))
  for (bank <- 0 until cfg.bankCount) {
    descriptorChecks +=
      (io.rowPair.inputStart(bank) === lm.stripeInputStarts(bank))
    descriptorChecks +=
      (io.rowPair.inputFeatures(bank) === lm.stripeInputFeatures(bank))
    for (engine <- 0 until cfg.engineCount) {
      val weightEnd =
        io.rowPair.engine(engine).weightAddress(bank).resize(
          lm.addressWidth + 1) +
          io.rowPair.inputFeatures(bank).resize(lm.addressWidth + 1)
      descriptorChecks += (!expectedEngineMask(engine) ||
        (
          io.rowPair.engine(engine).tokenId === expectedToken(engine) &&
          io.rowPair.engine(engine).weightAddress(bank)(6 downto 0) === 0 &&
          !weightEnd.msb))
    }
  }
  for (engine <- 0 until cfg.engineCount) {
    descriptorChecks += (!expectedEngineMask(engine) ||
      (
        io.rowPair.engine(engine).scaleBank < cfg.bankCount &&
        io.rowPair.engine(engine).scaleAddress(1 downto 0) === 0))
  }
  descriptorChecks += (!expectedEngineMask.andR ||
      io.rowPair.engine(0).requestId =/=
        io.rowPair.engine(1).requestId)
  val descriptorLegal = descriptorChecks.reduce(_ && _)

  io.rowPair.ready := activeJob && !pairActive

  val bankPhase = Vec.fill(cfg.bankCount)(
    Reg(BitNetResidentLmBankPhase()) init BitNetResidentLmBankPhase.Idle)
  val bankResponseBeat = Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.responseBeatWidth bits)) init 0)

  val scaleValid = Vec.fill(cfg.engineCount)(RegInit(False))
  val scaleSent = Vec.fill(cfg.engineCount)(RegInit(False))
  val scaleFault = Vec.fill(cfg.engineCount)(RegInit(False))
  val scaleValue = Vec.fill(cfg.engineCount)(Reg(Bits(32 bits)) init 0)

  val replayTile = Reg(UInt(lm.tileIndexWidth bits)) init 0
  val replayAllIssued = RegInit(False)
  val weightTilesDone = RegInit(False)

  when(io.rowPair.fire) {
    pairReg := io.rowPair.payload
    pairActive := True
    pairFault := !descriptorLegal
    replayTile := 0
    replayAllIssued := False
    weightTilesDone := False
    for (engine <- 0 until cfg.engineCount) {
      scaleValid(engine) := !descriptorLegal && expectedEngineMask(engine)
      scaleSent(engine) := False
      scaleFault(engine) := !descriptorLegal && expectedEngineMask(engine)
      scaleValue(engine) := 0
    }
    for (bank <- 0 until cfg.bankCount) {
      bankResponseBeat(bank) := 0
      bankPhase(bank) := BitNetResidentLmBankPhase.IssueWeight0
      when(!descriptorLegal) {
        bankPhase(bank) := BitNetResidentLmBankPhase.Done
      }
    }
    when(!descriptorLegal) {
      stickyFault := True
      faultBitsReg(BitNetResidentLmFetchFault.RowDescriptor) := True
    }
  }

  def encodedTag(
      requestId: UInt,
      scale: Bool,
      engine: UInt,
      bank: Int): UInt = {
    val body = (activeEpoch.asBits ## requestId.asBits ## scale.asBits ##
      engine.asBits ## U(bank, lm.bankWidth bits).asBits).asUInt
    body.resize(crossbar.tagWidth)
  }

  def encodedTransaction(scale: Bool, engine: UInt, bank: Int): UInt = {
    val body = (scale.asBits ## engine.asBits ##
      U(bank, lm.bankWidth bits).asBits).asUInt
    body.resize(crossbar.transactionIdWidth)
  }

  val requestFire = Bits(cfg.bankCount bits)
  for (bank <- 0 until cfg.bankCount) {
    val phase = bankPhase(bank)
    val issueWeight0 = phase === BitNetResidentLmBankPhase.IssueWeight0
    val issueWeight1 = phase === BitNetResidentLmBankPhase.IssueWeight1
    val issueScale0 = phase === BitNetResidentLmBankPhase.IssueScale0
    val issueScale1 = phase === BitNetResidentLmBankPhase.IssueScale1
    val issue = issueWeight0 || issueWeight1 || issueScale0 || issueScale1
    val engine = UInt(lm.engineWidth bits)
    val scale = Bool()
    engine := 0
    scale := False
    when(issueWeight1 || issueScale1) { engine := 1 }
    when(issueScale0 || issueScale1) { scale := True }

    val requestId = UInt(lm.requestIdWidth bits)
    val address = UInt(crossbar.localAddressWidth bits)
    requestId := pairReg.engine(0).requestId
    address := pairReg.engine(0).weightAddress(bank).resized
    when(engine === 1) {
      requestId := pairReg.engine(1).requestId
      address := pairReg.engine(1).weightAddress(bank).resized
    }
    when(scale) {
      val scaleAddress = UInt(lm.addressWidth bits)
      scaleAddress := pairReg.engine(0).scaleAddress
      when(engine === 1) { scaleAddress := pairReg.engine(1).scaleAddress }
      address := (scaleAddress &
        U(((BigInt(1) << lm.addressWidth) - 1) & ~BigInt(15),
          lm.addressWidth bits)).resized
    }

    io.readRequest(bank).valid := pairActive && issue
    io.readRequest(bank).bank := bank
    io.readRequest(bank).transactionId :=
      encodedTransaction(scale, engine, bank)
    io.readRequest(bank).tag := encodedTag(requestId, scale, engine, bank)
    io.readRequest(bank).localAddress := address
    io.readRequest(bank).beats :=
      (pairReg.inputFeatures(bank) >> 4).resize(crossbar.beatCountWidth)
    when(scale) { io.readRequest(bank).beats := 1 }
    requestFire(bank) := io.readRequest(bank).fire

    when(io.readRequest(bank).fire) {
      bankResponseBeat(bank) := 0
      switch(phase) {
        is(BitNetResidentLmBankPhase.IssueWeight0) {
          bankPhase(bank) := BitNetResidentLmBankPhase.ReceiveWeight0
        }
        is(BitNetResidentLmBankPhase.IssueWeight1) {
          bankPhase(bank) := BitNetResidentLmBankPhase.ReceiveWeight1
        }
        is(BitNetResidentLmBankPhase.IssueScale0) {
          bankPhase(bank) := BitNetResidentLmBankPhase.ReceiveScale0
        }
        is(BitNetResidentLmBankPhase.IssueScale1) {
          bankPhase(bank) := BitNetResidentLmBankPhase.ReceiveScale1
        }
      }
    }
  }
  when(requestFire.orR) {
    issuedReadRequestCount := issuedReadRequestCount +
      CountOne(requestFire).resize(32)
  }

  val receiving = Vec(Bool(), cfg.bankCount)
  val receivingScale = Vec(Bool(), cfg.bankCount)
  val responseEngine = Vec(UInt(lm.engineWidth bits), cfg.bankCount)
  val responseLegal = Vec(Bool(), cfg.bankCount)
  val responseExpectedLast = Vec(Bool(), cfg.bankCount)
  val responseDesiredMemory = Vec(
    UInt(log2Up(cfg.segmentsPerTile) bits), cfg.bankCount)
  val responseGlobalTile = Vec(UInt(lm.tileIndexWidth bits), cfg.bankCount)

  for (bank <- 0 until cfg.bankCount) {
    val phase = bankPhase(bank)
    receiving(bank) :=
      phase === BitNetResidentLmBankPhase.ReceiveWeight0 ||
      phase === BitNetResidentLmBankPhase.ReceiveWeight1 ||
      phase === BitNetResidentLmBankPhase.ReceiveScale0 ||
      phase === BitNetResidentLmBankPhase.ReceiveScale1
    receivingScale(bank) :=
      phase === BitNetResidentLmBankPhase.ReceiveScale0 ||
      phase === BitNetResidentLmBankPhase.ReceiveScale1
    responseEngine(bank) := 0
    when(phase === BitNetResidentLmBankPhase.ReceiveWeight1 ||
      phase === BitNetResidentLmBankPhase.ReceiveScale1) {
      responseEngine(bank) := 1
    }

    val expectedRequestId = UInt(lm.requestIdWidth bits)
    expectedRequestId := pairReg.engine(0).requestId
    when(responseEngine(bank) === 1) {
      expectedRequestId := pairReg.engine(1).requestId
    }
    val totalBeats = UInt(crossbar.beatCountWidth bits)
    totalBeats := (pairReg.inputFeatures(bank) >> 4)
      .resize(crossbar.beatCountWidth)
    when(receivingScale(bank)) { totalBeats := 1 }
    responseExpectedLast(bank) :=
      bankResponseBeat(bank).resize(crossbar.beatCountWidth) === totalBeats - 1

    val metadataLegal =
      io.readResponse(bank).bank === bank &&
      io.readResponse(bank).transactionId === encodedTransaction(
        receivingScale(bank), responseEngine(bank), bank) &&
      io.readResponse(bank).tag === encodedTag(
        expectedRequestId, receivingScale(bank), responseEngine(bank), bank) &&
      io.readResponse(bank).beatIndex === bankResponseBeat(bank).resized &&
      io.readResponse(bank).last === responseExpectedLast(bank)
    val axiLegal = !io.readResponse(bank).fault &&
      io.readResponse(bank).rresp === B"2'b00"
    responseLegal(bank) := receiving(bank) && metadataLegal && axiLegal

    val globalBeat = (
      (pairReg.inputStart(bank) >> 4).resize(lm.featureIndexWidth + 1) +
        bankResponseBeat(bank).resize(lm.featureIndexWidth + 1)
    ).resize(lm.featureIndexWidth + 1)
    val naturalSegment = globalBeat(2 downto 0)
    responseDesiredMemory(bank) :=
      (naturalSegment.resize(4) + U(bank, 4 bits))(2 downto 0)
    responseGlobalTile(bank) :=
      (globalBeat >> 3).resize(lm.tileIndexWidth)
  }

  val responseWinnerValid = Vec.fill(cfg.engineCount)(
    Vec.fill(cfg.segmentsPerTile)(Bool()))
  val responseWinnerBank = Vec.fill(cfg.engineCount)(
    Vec.fill(cfg.segmentsPerTile)(UInt(lm.bankWidth bits)))
  for (engine <- 0 until cfg.engineCount;
       memory <- 0 until cfg.segmentsPerTile) {
    responseWinnerValid(engine)(memory) := False
    responseWinnerBank(engine)(memory) := 0
    for (bank <- (0 until cfg.bankCount).reverse) {
      when(io.readResponse(bank).valid && receiving(bank) &&
        !receivingScale(bank) && responseEngine(bank) === engine &&
        responseDesiredMemory(bank) === memory) {
        responseWinnerValid(engine)(memory) := True
        responseWinnerBank(engine)(memory) := bank
      }
    }
  }

  for (bank <- 0 until cfg.bankCount) {
    io.readResponse(bank).ready := !receiving(bank) || receivingScale(bank)
    for (engine <- 0 until cfg.engineCount;
         memory <- 0 until cfg.segmentsPerTile) {
      when(receiving(bank) && !receivingScale(bank) &&
        responseEngine(bank) === engine &&
        responseDesiredMemory(bank) === memory &&
        responseWinnerValid(engine)(memory) &&
        responseWinnerBank(engine)(memory) === bank) {
        io.readResponse(bank).ready := True
      }
    }
  }

  val weightMemory = Array.fill(cfg.engineCount, cfg.segmentsPerTile)(
    Mem(Bits(crossbar.dataWidth bits), lm.tileCount))
  for (engine <- 0 until cfg.engineCount;
       memory <- 0 until cfg.segmentsPerTile) {
    weightMemory(engine)(memory).addAttribute("ram_style", "block")
    val winner = responseWinnerBank(engine)(memory)
    val winnerFire = responseWinnerValid(engine)(memory) &&
      io.readResponse(winner).fire &&
      responseEngine(winner) === engine && !receivingScale(winner)
    weightMemory(engine)(memory).write(
      address = responseGlobalTile(winner),
      data = io.readResponse(winner).data,
      enable = winnerFire && responseLegal(winner) && !pairFault)
  }

  val responseFire = Bits(cfg.bankCount bits)
  for (bank <- 0 until cfg.bankCount) {
    responseFire(bank) := io.readResponse(bank).fire
    when(io.readResponse(bank).fire) {
      when(!receiving(bank)) {
        stickyFault := True
        faultBitsReg(BitNetResidentLmFetchFault.UnexpectedResponse) := True
        when(pairActive) { pairFault := True }
      } otherwise {
        when(!responseLegal(bank)) {
          stickyFault := True
          pairFault := True
          faultBitsReg(BitNetResidentLmFetchFault.ResponseIdentity) := True
          when(io.readResponse(bank).fault ||
            io.readResponse(bank).rresp =/= B"2'b00") {
            faultBitsReg(BitNetResidentLmFetchFault.AxiResponse) := True
          }
        }

        when(receivingScale(bank)) {
          val words = Vec(Bits(32 bits), 4)
          for (word <- 0 until 4) {
            words(word) := io.readResponse(bank).data(
              (word + 1) * 32 - 1 downto word * 32)
          }
          val scaleAddress = UInt(lm.addressWidth bits)
          scaleAddress := pairReg.engine(0).scaleAddress
          when(responseEngine(bank) === 1) {
            scaleAddress := pairReg.engine(1).scaleAddress
          }
          scaleValue(responseEngine(bank)) := words(scaleAddress(3 downto 2))
          scaleValid(responseEngine(bank)) := True
          scaleFault(responseEngine(bank)) := !responseLegal(bank)
        }

        val finishing = responseExpectedLast(bank) ||
          io.readResponse(bank).last
        when(finishing) {
          bankResponseBeat(bank) := 0
          switch(bankPhase(bank)) {
            is(BitNetResidentLmBankPhase.ReceiveWeight0) {
              when(pairReg.engineMask(1)) {
                bankPhase(bank) := BitNetResidentLmBankPhase.IssueWeight1
              } elsewhen(pairReg.engine(0).scaleBank === bank) {
                bankPhase(bank) := BitNetResidentLmBankPhase.IssueScale0
              } otherwise {
                bankPhase(bank) := BitNetResidentLmBankPhase.Done
              }
            }
            is(BitNetResidentLmBankPhase.ReceiveWeight1) {
              when(pairReg.engine(0).scaleBank === bank) {
                bankPhase(bank) := BitNetResidentLmBankPhase.IssueScale0
              } elsewhen(pairReg.engineMask(1) &&
                pairReg.engine(1).scaleBank === bank) {
                bankPhase(bank) := BitNetResidentLmBankPhase.IssueScale1
              } otherwise {
                bankPhase(bank) := BitNetResidentLmBankPhase.Done
              }
            }
            is(BitNetResidentLmBankPhase.ReceiveScale0) {
              when(pairReg.engineMask(1) &&
                pairReg.engine(1).scaleBank === bank) {
                bankPhase(bank) := BitNetResidentLmBankPhase.IssueScale1
              } otherwise {
                bankPhase(bank) := BitNetResidentLmBankPhase.Done
              }
            }
            is(BitNetResidentLmBankPhase.ReceiveScale1) {
              bankPhase(bank) := BitNetResidentLmBankPhase.Done
            }
          }
        } otherwise {
          bankResponseBeat(bank) := bankResponseBeat(bank) + 1
        }
      }
    }
  }
  when(responseFire.orR) {
    acceptedReadResponseCount := acceptedReadResponseCount +
      CountOne(responseFire).resize(32)
  }

  val allBanksDone = bankPhase.map(
    _ === BitNetResidentLmBankPhase.Done).asBits.andR
  val selectedScaleEngine = UInt(lm.engineWidth bits)
  val scaleAvailable = Bool()
  scaleAvailable := False
  selectedScaleEngine := 0
  for (engine <- (0 until cfg.engineCount).reverse) {
    when(pairReg.engineMask(engine) && scaleValid(engine) &&
      !scaleSent(engine)) {
      scaleAvailable := True
      selectedScaleEngine := engine
    }
  }
  // Hold scales until the complete row pair is known good/bad.  A late fault
  // on another bank can therefore never let an earlier optimistic scale
  // escape without the pair-level fail-closed bit.
  io.rowScale.valid := pairActive && allBanksDone && scaleAvailable
  io.rowScale.requestId := pairReg.engine(selectedScaleEngine).requestId
  io.rowScale.engine := selectedScaleEngine
  io.rowScale.tokenId := pairReg.engine(selectedScaleEngine).tokenId
  io.rowScale.scaleBank := pairReg.engine(selectedScaleEngine).scaleBank
  io.rowScale.rowScaleF32 := scaleValue(selectedScaleEngine)
  io.rowScale.fault := pairFault || scaleFault(selectedScaleEngine)
  when(io.rowScale.fire) {
    scaleSent(selectedScaleEngine) := True
    emittedScaleCount := emittedScaleCount + 1
  }

  val replayValid = RegInit(False)
  val replayTileReg = Reg(UInt(lm.tileIndexWidth bits)) init 0
  val replayBeatOrdinal = Reg(UInt(matrix.beatCountWidth bits)) init 0
  val replayFirst = RegInit(False)
  val replayLast = RegInit(False)
  val replayBankReg = Reg(UInt(lm.bankWidth bits)) init 0

  val replayStageReady = !replayValid || io.i8Weight.ready
  val replayIssue = pairActive && allBanksDone &&
    !replayAllIssued && replayStageReady
  val replayBank = UInt(lm.bankWidth bits)
  replayBank := 0
  for (bank <- 0 until cfg.bankCount) {
    val tileStart = lm.stripeInputStarts(bank) / lm.tileFeatures
    val tileEnd = tileStart +
      lm.stripeInputFeatures(bank) / lm.tileFeatures
    when(replayTile >= tileStart && replayTile < tileEnd) {
      replayBank := bank
    }
  }

  val replayMemoryData = Array.fill(cfg.engineCount, cfg.segmentsPerTile)(
    Bits(crossbar.dataWidth bits))
  for (engine <- 0 until cfg.engineCount;
       memory <- 0 until cfg.segmentsPerTile) {
    replayMemoryData(engine)(memory) :=
      weightMemory(engine)(memory).readSync(
        address = replayTile,
        enable = replayIssue)
  }

  when(replayIssue) {
    replayValid := True
    replayTileReg := replayTile
    replayBeatOrdinal := nextWeightBeatOrdinal
    nextWeightBeatOrdinal := nextWeightBeatOrdinal + 1
    replayFirst := replayTile === 0
    replayLast := replayTile === lm.tileCount - 1
    replayBankReg := replayBank
    when(replayTile === lm.tileCount - 1) {
      replayAllIssued := True
    } otherwise {
      replayTile := replayTile + 1
    }
  } elsewhen(io.i8Weight.fire) {
    replayValid := False
  }

  io.i8Weight.valid := replayValid
  io.i8Weight.sequenceId := activeSequenceId
  io.i8Weight.beatOrdinal := replayBeatOrdinal
  io.i8Weight.use := BitNetI8WeightUse.LmHead
  io.i8Weight.engineOutputBase(0) := pairReg.pairOrdinal.resized
  io.i8Weight.engineOutputBase(1) := pairReg.pairOrdinal.resized
  io.i8Weight.firstK := replayFirst
  io.i8Weight.lastK := replayLast
  io.i8Weight.fault := pairFault
  for (engine <- 0 until cfg.engineCount; site <- 0 until lm.tileFeatures) {
    val segment = site / cfg.beatBytes
    val byte = site % cfg.beatBytes
    val rotatedMemory = (
      replayBankReg.resize(4) + U(segment, 4 bits))(2 downto 0)
    val memoryVector = Vec(Bits(crossbar.dataWidth bits), cfg.segmentsPerTile)
    for (memory <- 0 until cfg.segmentsPerTile) {
      memoryVector(memory) := replayMemoryData(engine)(memory)
    }
    io.i8Weight.engineSiteWeight(engine)(site) :=
      memoryVector(rotatedMemory)(byte * 8 + 7 downto byte * 8)
    when(pairFault || !pairReg.engineMask(engine)) {
      io.i8Weight.engineSiteWeight(engine)(site) := 0
    }
  }
  when(io.i8Weight.fire) {
    emittedWeightTileCount := emittedWeightTileCount + 1
    when(replayLast) { weightTilesDone := True }
  }

  val weightDoneAfter = weightTilesDone ||
    (io.i8Weight.fire && replayLast)
  val scaleDoneAfter = Bits(cfg.engineCount bits)
  for (engine <- 0 until cfg.engineCount) {
    scaleDoneAfter(engine) := !pairReg.engineMask(engine) ||
      scaleSent(engine) ||
      (io.rowScale.fire && selectedScaleEngine === engine)
  }
  when(pairActive && weightDoneAfter && scaleDoneAfter.andR) {
    pairActive := False
    expectedPair := expectedPair + 1
    for (bank <- 0 until cfg.bankCount) {
      bankPhase(bank) := BitNetResidentLmBankPhase.Idle
    }
  }
}

/**
  * Production-facing resident LM-head subsystem.
  *
  * It owns the LM weight/scale fetcher, the small resident coordinator and the
  * LmHead side of the QueryLmAlias lease.  It deliberately owns neither the
  * shared dynamic quantizer, a second tensor backend nor a matrix array.  The
  * outward ports are direct clients of NarrowWork, the shared quantizer, the
  * unified packed semantic router, the five-bank crossbar and the sole TP4
  * compute island.
  */
class BitNetResidentLmHeadSubsystem(
    cfg: BitNetResidentLmFetchConfig = BitNetResidentLmFetchConfig(),
    externalWeightTileCache: Boolean = false)
    extends Component {
  private val residentCfg = cfg.resident
  private val model = cfg.model
  private val q = residentCfg.quant
  private val linear = residentCfg.linear
  private val scratchpadCfg = residentCfg.unifiedScratchpad
  private val scratchpadLayout = scratchpadCfg.layout
  private val tileCacheCfg = BitNetResidentSharedWeightTileCacheConfig(
    dataWidth = cfg.crossbar.dataWidth,
    poolCount = cfg.engineCount,
    memoryCount = cfg.segmentsPerTile,
    depth = if (cfg.resident.streaming.tileFeatures == 64) 216 else 108,
    sequenceIdWidth = linear.matrix.sequenceIdWidth,
    sourceEpochWidth = cfg.epochWidth)

  val io = new Bundle {
    val handoff = slave(Stream(BitNetResidentLmHeadHandoff(model)))

    // Whole-operation ownership of the single physical activation fabric.
    // A top connects these streams to BitNetResidentActivationFabricArbiter;
    // the client identity is fixed to BitNetActivationClient.LmHead.
    val activationAcquire = master(Stream(
      BitNetActivationFabricLease(model)))
    val activationRelease = master(Stream(
      BitNetActivationFabricLease(model)))

    val activationReadCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(
        model, residentCfg.activationLayout))), model.tokenParallelism)
    val activationReadData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(
        model, residentCfg.activationLayout))), model.tokenParallelism)

    val quantCommand = master(Stream(BitNetSharedDynamicQuantCommand(q)))
    val quantMeasureInput = master(Stream(
      BitNetSharedDynamicQuantMeasureBeat(q)))
    val quantReplayRequest = slave(Stream(
      BitNetSharedDynamicQuantReplayRequest(q)))
    val quantReplayInput = master(Stream(
      BitNetSharedDynamicQuantReplayBeat(q)))
    val quantOutput = slave(Stream(BitNetSharedDynamicQuantResultBeat(q)))
    val quantDone = slave(Stream(BitNetSharedDynamicQuantDone(q)))

    // QueryLmAlias is leased for the complete capture plus vocabulary scan.
    // The top-level two-client router is the sole connection to the unique
    // semantic frontend and physical tensor backend.
    val aliasAcquire = master(Stream(
      BitNetUnifiedResidentAliasAcquire(scratchpadCfg)))
    val aliasGrant = slave(Stream(
      BitNetUnifiedResidentAliasGrant(scratchpadCfg)))
    val aliasRelease = master(Stream(
      BitNetUnifiedResidentAliasRelease(scratchpadCfg)))
    val packedRead = master(Stream(
      BitNetUnifiedResidentPackedI8Read(scratchpadCfg)))
    val packedReadData = slave(Stream(
      BitNetUnifiedResidentPackedI8ReadData(scratchpadCfg)))
    val packedWrite = master(Stream(
      BitNetUnifiedResidentPackedI8Write(scratchpadCfg)))
    val packedWriteFault = slave(Stream(
      BitNetUnifiedResidentPackedI8WriteFault(scratchpadCfg)))
    val semanticPackedReadOutstanding = in Bool()
    val semanticPackedWriteOutstanding = in Bool()
    val abort = in Bool()

    val lmReadRequest = Vec(master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar))), cfg.bankCount)
    val lmReadResponse = Vec(slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar))), cfg.bankCount)

    val cacheWrite = Vec.fill(cfg.engineCount)(Vec.fill(
      cfg.segmentsPerTile)(master(Stream(
        BitNetResidentWeightTileCacheWrite(tileCacheCfg)))))
    val cacheRead = master(Stream(
      BitNetResidentWeightTileCacheRead(tileCacheCfg)))
    val cacheReadData = Vec.fill(cfg.engineCount)(Vec.fill(
      cfg.segmentsPerTile)(in Bits(cfg.crossbar.dataWidth bits)))
    val cacheAcquire = master(Stream(
      BitNetResidentWeightTileCacheAcquire(tileCacheCfg)))
    val cacheGrant = slave(Stream(
      BitNetResidentWeightTileCacheGrant(tileCacheCfg)))
    // Authenticate the Stream payload against the independently observed
    // physical-cache generation.  Private-cache builds have no lease port and
    // therefore intentionally elaborate without this sideband.
    val cacheActiveGeneration = if (externalWeightTileCache)
      in UInt(tileCacheCfg.generationWidth bits) else null
    val cacheRelease = master(Stream(
      BitNetResidentWeightTileCacheRelease(tileCacheCfg)))

    val linearLaunch = master(Stream(BitNetResidentLinearLaunch(linear)))
    val matrixI8Weight = master(Stream(
      BitNetI8WeightBeat(linear.feeder)))
    val matrixActivation = master(Stream(
      BitNetNormalizedActivationTile(linear.feeder)))
    val linearResult = slave(Stream(
      BitNetResidentFinalFeaturePair(linear)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(linear)))

    val completion = master(Stream(
      BitNetResidentLmHeadCompletion(residentCfg)))
    val busy = out Bool()
    /**
      * A malformed shared-cache grant permanently fail-stops this LM client.
      * The accepted handoff is retained for diagnosis and reset is required.
      */
    val weightLeaseQuarantined = out Bool()
    val aliasLeaseQuarantined = out Bool()
    val aliasLeaseHeld = out Bool()
    val aliasGeneration = out UInt(
      scratchpadLayout.aliasGenerationWidth bits)
    val protocolFaultSticky = out Bool()
    val residentFaultBits = out Bits(16 bits)
    val fetchFaultBits = out Bits(8 bits)
    val issuedLmReadRequests = out UInt(32 bits)
    val acceptedLmReadResponses = out UInt(32 bits)
  }

  val resident = new BitNetResidentStreamingLmHead(residentCfg)
  val fetcher = new BitNetResidentLmWeightScaleFetcher(
    cfg, externalWeightTileCache = externalWeightTileCache)
  val nextFetchEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val pendingHandoffValid = RegInit(False)
  val pendingHandoff = Reg(BitNetResidentLmHeadHandoff(model)) init
    BitNetResidentLmHeadHandoff(model).getZero
  val activationLeaseHeld = RegInit(False)
  val activationReleasePending = RegInit(False)
  val activationLeaseRequestId = Reg(UInt(model.requestIdWidth bits)) init 0
  val residentJobStarted = RegInit(False)
  val pendingFetchSequence = Reg(UInt(
    linear.matrix.sequenceIdWidth bits)) init 0
  val pendingFetchEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val weightAcquireIssued = RegInit(False)
  val weightLeaseHeld = RegInit(False)
  val weightReleasePending = RegInit(False)
  val weightLeaseGeneration = Reg(UInt(
    tileCacheCfg.generationWidth bits)) init 0
  val fetchRetireIssued = RegInit(False)
  val weightLeaseFaultSticky = RegInit(False)
  val weightLeaseQuarantine = RegInit(False)
  val weightInvalidGrantReleasePending = RegInit(False)
  val weightInvalidGrantGeneration = Reg(UInt(
    tileCacheCfg.generationWidth bits)) init 0
  val weightInvalidGrantSequence = Reg(UInt(
    linear.matrix.sequenceIdWidth bits)) init 0
  val weightInvalidGrantEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val aliasAcquireIssued = RegInit(False)
  val aliasLeaseHeldReg = RegInit(False)
  val aliasLeaseGeneration = Reg(UInt(
    scratchpadLayout.aliasGenerationWidth bits)) init 0
  val aliasReleasePending = RegInit(False)
  val aliasReleaseOwner = Reg(
    BitNetUnifiedResidentTensorAliasOwner()) init
      BitNetUnifiedResidentTensorAliasOwner.LmHead
  val aliasReleaseGeneration = Reg(UInt(
    scratchpadLayout.aliasGenerationWidth bits)) init 0
  val aliasLeaseFaultSticky = RegInit(False)
  val aliasLeaseQuarantine = RegInit(False)
  val abortCleanupPending = RegInit(False)
  io.weightLeaseQuarantined := weightLeaseQuarantine
  io.aliasLeaseQuarantined := aliasLeaseQuarantine
  io.aliasLeaseHeld := aliasLeaseHeldReg
  io.aliasGeneration := aliasLeaseGeneration

  val operationOwned = pendingHandoffValid || activationLeaseHeld ||
    activationReleasePending || residentJobStarted ||
    weightAcquireIssued || weightLeaseHeld || weightReleasePending ||
    weightInvalidGrantReleasePending || aliasAcquireIssued ||
    aliasLeaseHeldReg || aliasReleasePending ||
    io.semanticPackedReadOutstanding || io.semanticPackedWriteOutstanding
  // A level-high board-wide abort while this LM client is genuinely idle is
  // inert.  Only a job whose handoff/lease was already accepted is cancelled.
  val activeAbort = abortCleanupPending || (io.abort && operationOwned)
  when(io.abort && operationOwned) { abortCleanupPending := True }

  if (!externalWeightTileCache) {
    io.cacheAcquire.valid := False
    io.cacheAcquire.payload := BitNetResidentWeightTileCacheAcquire(
      tileCacheCfg).getZero
    io.cacheGrant.ready := False
    io.cacheRelease.valid := False
    io.cacheRelease.payload := BitNetResidentWeightTileCacheRelease(
      tileCacheCfg).getZero
  }

  // Buffer the controller handoff while the explicit LmHead activation lease
  // is acquired.  The resident core cannot issue a read command before the
  // arbiter has acknowledged ownership.
  io.handoff.ready := !pendingHandoffValid && !activationLeaseHeld &&
    !activationReleasePending && resident.io.handoff.ready &&
    fetcher.io.canStartJob && !weightAcquireIssued && !weightLeaseHeld &&
    !weightReleasePending && !weightInvalidGrantReleasePending &&
    !weightLeaseQuarantine && !aliasAcquireIssued && !aliasLeaseHeldReg &&
    !aliasReleasePending && !aliasLeaseQuarantine && !activeAbort &&
    !io.abort
  when(io.handoff.fire) {
    pendingHandoffValid := True
    pendingHandoff := io.handoff.payload
    activationLeaseRequestId := io.handoff.requestId
    pendingFetchSequence := (io.handoff.requestId.asBits ^
      io.handoff.tokenPosition.resize(model.requestIdWidth).asBits)
      .asUInt.resize(linear.matrix.sequenceIdWidth)
    pendingFetchEpoch := nextFetchEpoch
    nextFetchEpoch := nextFetchEpoch + 1
    fetchRetireIssued := False
  }

  io.activationAcquire.valid := pendingHandoffValid &&
    !activationLeaseHeld && !weightLeaseQuarantine &&
    !aliasLeaseQuarantine && !activeAbort
  io.activationAcquire.client := BitNetActivationClient.LmHead
  io.activationAcquire.requestId := activationLeaseRequestId
  when(io.activationAcquire.fire) { activationLeaseHeld := True }

  io.aliasAcquire.valid := pendingHandoffValid && !aliasAcquireIssued &&
    !aliasLeaseHeldReg && !aliasReleasePending && !aliasLeaseQuarantine &&
    !weightLeaseQuarantine && !activeAbort
  io.aliasAcquire.owner := BitNetUnifiedResidentTensorAliasOwner.LmHead
  when(io.aliasAcquire.fire) { aliasAcquireIssued := True }

  // Always drain grants.  An unsolicited, wrong-owner, or zero-generation
  // grant is a reset-only fail-stop, but a real non-zero identity is retained
  // long enough for a best-effort release.
  io.aliasGrant.ready := True
  val aliasGrantLegal = aliasAcquireIssued &&
    io.aliasGrant.owner === BitNetUnifiedResidentTensorAliasOwner.LmHead &&
    io.aliasGrant.generation =/= 0 && !aliasLeaseQuarantine &&
    !weightLeaseQuarantine && !activeAbort
  val aliasGrantForCancelledAcquire = aliasAcquireIssued && activeAbort &&
    io.aliasGrant.owner === BitNetUnifiedResidentTensorAliasOwner.LmHead &&
    io.aliasGrant.generation =/= 0
  when(io.aliasGrant.fire) {
    aliasAcquireIssued := False
    when(aliasGrantLegal) {
      aliasLeaseHeldReg := True
      aliasLeaseGeneration := io.aliasGrant.generation
    } elsewhen(aliasGrantForCancelledAcquire) {
      // The request was already accepted before abort.  Drain its real grant
      // and return that exact identity without turning cancellation into a
      // reset-only protocol quarantine.
      aliasReleasePending := True
      aliasReleaseOwner := io.aliasGrant.owner
      aliasReleaseGeneration := io.aliasGrant.generation
    } otherwise {
      aliasLeaseHeldReg := False
      aliasLeaseGeneration := 0
      aliasLeaseFaultSticky := True
      aliasLeaseQuarantine := True
      when(io.aliasGrant.generation =/= 0) {
        aliasReleasePending := True
        aliasReleaseOwner := io.aliasGrant.owner
        aliasReleaseGeneration := io.aliasGrant.generation
      }
      when(activationLeaseHeld || io.activationAcquire.fire) {
        activationReleasePending := True
      }
    }
  }

  if (externalWeightTileCache) {
    io.cacheAcquire.valid := pendingHandoffValid && !weightAcquireIssued &&
      !weightLeaseHeld && !weightReleasePending &&
      !weightInvalidGrantReleasePending && !weightLeaseQuarantine &&
      !aliasLeaseQuarantine && !activeAbort
    io.cacheAcquire.sequenceId := pendingFetchSequence
    io.cacheAcquire.sourceEpoch := pendingFetchEpoch
    when(io.cacheAcquire.fire) { weightAcquireIssued := True }

    // Never lock the shared grant return.  Unsolicited/malformed grants are
    // consumed fail-closed, and their returned identity is retained for one
    // best-effort cleanup release when generation is non-zero.
    io.cacheGrant.ready := True
    val weightGrantLegal =
      weightAcquireIssued && !weightLeaseHeld && !weightReleasePending &&
        !weightInvalidGrantReleasePending && !weightLeaseQuarantine &&
        !activeAbort &&
        io.cacheGrant.sequenceId === pendingFetchSequence &&
        io.cacheGrant.sourceEpoch === pendingFetchEpoch &&
        io.cacheGrant.generation =/= 0 &&
        io.cacheGrant.generation === io.cacheActiveGeneration
    val weightGrantForCancelledAcquire = weightAcquireIssued && activeAbort &&
      io.cacheGrant.generation =/= 0
    when(io.cacheGrant.fire) {
      weightAcquireIssued := False
      when(weightGrantLegal) {
        weightLeaseHeld := True
        weightLeaseGeneration := io.cacheGrant.generation
      } elsewhen(weightGrantForCancelledAcquire) {
        weightInvalidGrantReleasePending := True
        weightInvalidGrantSequence := io.cacheGrant.sequenceId
        weightInvalidGrantEpoch := io.cacheGrant.sourceEpoch
        weightInvalidGrantGeneration := io.cacheGrant.generation
      } otherwise {
        // The bad generation is never installed as an authenticated lease.
        // If it is non-zero, retain it only long enough to make one
        // best-effort release using the identity actually returned.
        weightLeaseHeld := False
        weightLeaseGeneration := 0
        weightLeaseQuarantine := True
        weightLeaseFaultSticky := True
        when(io.cacheGrant.generation =/= 0) {
          weightInvalidGrantReleasePending := True
          weightInvalidGrantSequence := io.cacheGrant.sequenceId
          weightInvalidGrantEpoch := io.cacheGrant.sourceEpoch
          weightInvalidGrantGeneration := io.cacheGrant.generation
        }
        // Activation ownership may have arrived before, or in the same cycle
        // as, the malformed weight grant.  Drain it instead of leaking the
        // sole PL-resident activation fabric while the LM frontend fail-stops.
        when(activationLeaseHeld || io.activationAcquire.fire) {
          activationReleasePending := True
        }
        when(aliasLeaseHeldReg && aliasLeaseGeneration =/= 0) {
          aliasReleasePending := True
          aliasReleaseOwner :=
            BitNetUnifiedResidentTensorAliasOwner.LmHead
          aliasReleaseGeneration := aliasLeaseGeneration
        }
      }
    }
  }

  // These registers only participate in the external-cache lease protocol.
  // Drive them to their reset-safe state in the legacy/private-cache build so
  // Spinal does not report deliberately unassigned state during elaboration.
  if (!externalWeightTileCache) {
    weightAcquireIssued := False
    weightLeaseHeld := False
    weightReleasePending := False
    weightLeaseFaultSticky := False
    weightLeaseQuarantine := False
    weightInvalidGrantReleasePending := False
    weightInvalidGrantGeneration := 0
    weightInvalidGrantSequence := 0
    weightInvalidGrantEpoch := 0
  }

  val weightLeaseReady = if (externalWeightTileCache) weightLeaseHeld else True
  resident.io.handoff.valid := pendingHandoffValid && activationLeaseHeld &&
    weightLeaseReady && aliasLeaseHeldReg && !weightLeaseQuarantine &&
    !aliasLeaseQuarantine && !activeAbort
  resident.io.handoff.payload := pendingHandoff
  when(resident.io.handoff.fire) {
    pendingHandoffValid := False
    residentJobStarted := True
  }

  fetcher.io.jobStart.valid := resident.io.handoff.fire
  fetcher.io.jobStart.sequenceId := pendingFetchSequence
  fetcher.io.jobStart.epoch := pendingFetchEpoch
  fetcher.io.jobStart.cacheGeneration := weightLeaseGeneration

  resident.io.aliasLeaseValid := aliasLeaseHeldReg &&
    !aliasLeaseQuarantine
  resident.io.aliasLeaseGeneration := aliasLeaseGeneration
  resident.io.semanticPackedReadOutstanding :=
    io.semanticPackedReadOutstanding
  resident.io.semanticPackedWriteOutstanding :=
    io.semanticPackedWriteOutstanding
  resident.io.abort := activeAbort || aliasLeaseQuarantine ||
    weightLeaseQuarantine
  io.packedRead << resident.io.packedRead
  resident.io.packedReadData << io.packedReadData
  io.packedWrite << resident.io.packedWrite
  resident.io.packedWriteFault << io.packedWriteFault

  when(resident.io.completion.valid && activationLeaseHeld) {
    activationReleasePending := True
  }
  val residentFailStop = resident.io.protocolFaultSticky ||
    resident.io.codecQuarantined
  when(residentFailStop) {
    aliasLeaseFaultSticky := True
    aliasLeaseQuarantine := True
    when(aliasLeaseHeldReg && aliasLeaseGeneration =/= 0) {
      aliasReleasePending := True
      aliasReleaseOwner := BitNetUnifiedResidentTensorAliasOwner.LmHead
      aliasReleaseGeneration := aliasLeaseGeneration
    }
    when(activationLeaseHeld) { activationReleasePending := True }
  }
  when(weightLeaseQuarantine && aliasLeaseHeldReg &&
      aliasLeaseGeneration =/= 0) {
    aliasLeaseFaultSticky := True
    aliasLeaseQuarantine := True
    aliasReleasePending := True
    aliasReleaseOwner := BitNetUnifiedResidentTensorAliasOwner.LmHead
    aliasReleaseGeneration := aliasLeaseGeneration
  }
  when(activeAbort) {
    // Stop a buffered, not-yet-dispatched handoff.  Accepted acquire requests
    // remain marked issued until their grants arrive and are then released.
    pendingHandoffValid := False
    when(aliasLeaseHeldReg && aliasLeaseGeneration =/= 0) {
      aliasReleasePending := True
      aliasReleaseOwner := BitNetUnifiedResidentTensorAliasOwner.LmHead
      aliasReleaseGeneration := aliasLeaseGeneration
    }
    when(weightLeaseHeld) { weightReleasePending := True }
    when(activationLeaseHeld || io.activationAcquire.fire) {
      activationReleasePending := True
    }
  }
  when(resident.io.completion.valid && aliasLeaseHeldReg &&
      aliasLeaseGeneration =/= 0) {
    aliasReleasePending := True
    aliasReleaseOwner := BitNetUnifiedResidentTensorAliasOwner.LmHead
    aliasReleaseGeneration := aliasLeaseGeneration
  }

  val noPackedOutstanding = !io.semanticPackedReadOutstanding &&
    !io.semanticPackedWriteOutstanding && !resident.io.packedRead.valid &&
    !resident.io.packedWrite.valid && !resident.io.codecBusy
  io.aliasRelease.valid := aliasReleasePending &&
    aliasReleaseGeneration =/= 0 && noPackedOutstanding
  io.aliasRelease.owner := aliasReleaseOwner
  io.aliasRelease.generation := aliasReleaseGeneration
  when(io.aliasRelease.fire) {
    aliasReleasePending := False
    aliasLeaseHeldReg := False
    aliasLeaseGeneration := 0
    aliasReleaseGeneration := 0
  }
  val activationDrainSafe = !residentJobStarted ||
    resident.io.completion.valid
  io.activationRelease.valid := activationReleasePending &&
    activationDrainSafe
  io.activationRelease.client := BitNetActivationClient.LmHead
  io.activationRelease.requestId := activationLeaseRequestId
  when(io.activationRelease.fire) {
    activationReleasePending := False
    activationLeaseHeld := False
  }

  for (lane <- 0 until model.tokenParallelism) {
    io.activationReadCommand(lane) << resident.io.activationReadCommand(lane)
    resident.io.activationReadData(lane) << io.activationReadData(lane)
  }

  io.quantCommand << resident.io.quantCommand
  io.quantMeasureInput << resident.io.quantMeasureInput
  resident.io.quantReplayRequest << io.quantReplayRequest
  io.quantReplayInput << resident.io.quantReplayInput
  resident.io.quantOutput << io.quantOutput
  resident.io.quantDone << io.quantDone

  fetcher.io.rowPair << resident.io.rowPair
  resident.io.i8WeightInput << fetcher.io.i8Weight
  resident.io.rowScale << fetcher.io.rowScale
  for (bank <- 0 until cfg.bankCount) {
    io.lmReadRequest(bank) << fetcher.io.readRequest(bank)
    fetcher.io.readResponse(bank) << io.lmReadResponse(bank)
  }
  for (engine <- 0 until cfg.engineCount;
       memory <- 0 until cfg.segmentsPerTile) {
    io.cacheWrite(engine)(memory) << fetcher.io.cacheWrite(engine)(memory)
    fetcher.io.cacheReadData(engine)(memory) :=
      io.cacheReadData(engine)(memory)
  }
  io.cacheRead << fetcher.io.cacheRead

  io.linearLaunch << resident.io.linearLaunch
  io.matrixI8Weight << resident.io.matrixI8Weight
  io.matrixActivation << resident.io.matrixActivation
  resident.io.linearResult << io.linearResult
  resident.io.linearCompletion << io.linearCompletion

  if (externalWeightTileCache) {
    val retireFetcherNow = (resident.io.completion.valid || residentFailStop) &&
      !fetchRetireIssued && !fetcher.io.busy
    fetcher.io.jobRetire := retireFetcherNow
    when(retireFetcherNow) {
      fetchRetireIssued := True
      weightReleasePending := True
    }
    when((aliasLeaseQuarantine || activeAbort) && weightLeaseHeld) {
      weightReleasePending := True
    }

    val normalWeightRelease = weightReleasePending && weightLeaseHeld &&
      fetcher.io.canStartJob && !fetcher.io.jobActive && !fetcher.io.busy
    io.cacheRelease.valid := weightInvalidGrantReleasePending ||
      normalWeightRelease
    io.cacheRelease.sequenceId := Mux(weightInvalidGrantReleasePending,
      weightInvalidGrantSequence, pendingFetchSequence)
    io.cacheRelease.sourceEpoch := Mux(weightInvalidGrantReleasePending,
      weightInvalidGrantEpoch, pendingFetchEpoch)
    io.cacheRelease.generation := Mux(weightInvalidGrantReleasePending,
      weightInvalidGrantGeneration, weightLeaseGeneration)
    when(io.cacheRelease.fire) {
      when(weightInvalidGrantReleasePending) {
        weightInvalidGrantReleasePending := False
        weightInvalidGrantSequence := 0
        weightInvalidGrantEpoch := 0
        weightInvalidGrantGeneration := 0
      } otherwise {
        weightReleasePending := False
        weightLeaseHeld := False
        weightLeaseGeneration := 0
      }
    }

    val weightRetired = fetchRetireIssued && !weightAcquireIssued &&
      !weightLeaseHeld && !weightReleasePending
    val aliasRetired = !aliasAcquireIssued && !aliasLeaseHeldReg &&
      !aliasReleasePending && noPackedOutstanding
    val activationRetired = !activationLeaseHeld &&
      !activationReleasePending
    io.completion.valid := resident.io.completion.valid && weightRetired &&
      aliasRetired && activationRetired
    io.completion.payload := resident.io.completion.payload
    resident.io.completion.ready := io.completion.ready && weightRetired &&
      aliasRetired && activationRetired
  } else {
    val aliasRetired = !aliasAcquireIssued && !aliasLeaseHeldReg &&
      !aliasReleasePending && noPackedOutstanding
    val activationRetired = !activationLeaseHeld &&
      !activationReleasePending
    io.completion.valid := resident.io.completion.valid && aliasRetired &&
      activationRetired
    io.completion.payload := resident.io.completion.payload
    resident.io.completion.ready := io.completion.ready && aliasRetired &&
      activationRetired
    fetcher.io.jobRetire := resident.io.completion.fire ||
      (residentFailStop && fetcher.io.jobActive && !fetcher.io.busy)
  }

  io.busy := resident.io.busy || fetcher.io.busy ||
    pendingHandoffValid || activationLeaseHeld || activationReleasePending ||
    weightAcquireIssued || weightLeaseHeld || weightReleasePending ||
    weightInvalidGrantReleasePending || weightLeaseQuarantine ||
    aliasAcquireIssued || aliasLeaseHeldReg || aliasReleasePending ||
    aliasLeaseQuarantine || abortCleanupPending ||
    io.semanticPackedReadOutstanding ||
    io.semanticPackedWriteOutstanding
  io.protocolFaultSticky := resident.io.protocolFaultSticky ||
    fetcher.io.protocolFaultSticky || weightLeaseFaultSticky ||
    aliasLeaseFaultSticky
  io.residentFaultBits := resident.io.faultBits
  io.fetchFaultBits := fetcher.io.faultBits
  io.issuedLmReadRequests := fetcher.io.issuedReadRequests
  io.acceptedLmReadResponses := fetcher.io.acceptedReadResponses
  when(resident.io.completion.fire) { residentJobStarted := False }
  val abortCleanupDrained = !pendingHandoffValid && !activationLeaseHeld &&
    !activationReleasePending && !residentJobStarted &&
    !weightAcquireIssued && !weightLeaseHeld && !weightReleasePending &&
    !weightInvalidGrantReleasePending && !aliasAcquireIssued &&
    !aliasLeaseHeldReg && !aliasReleasePending &&
    !io.semanticPackedReadOutstanding &&
    !io.semanticPackedWriteOutstanding
  when(abortCleanupPending && abortCleanupDrained) {
    abortCleanupPending := False
  }
}

object GenerateBitNetResidentLmHeadSubsystem extends App {
  val output = if (args.nonEmpty) args(0)
    else "build/bitnet-resident-lm-head-subsystem"
  SpinalConfig(targetDirectory = output, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLmHeadSubsystem())
}
