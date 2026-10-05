package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Two-slot resident LM weight/scale fetcher.
  *
  * Slot N may replay its twenty typed matrix beats while slot N+1 is filled
  * from all five banks.  There is still at most one logical burst outstanding
  * per physical bank.  The slot bit is part of both the transaction and full
  * response tag, so a late or corrupt response cannot overwrite the other
  * ping-pong image.
  */
class BitNetResidentLmWeightScaleFetcher(
    cfg: BitNetResidentLmFetchConfig = BitNetResidentLmFetchConfig(),
    externalWeightTileCache: Boolean = false)
    extends Component {
  private val lm = cfg.lm
  private val matrix = cfg.resident.linear.matrix
  private val crossbar = cfg.crossbar
  private val segmentIndexWidth = log2Up(cfg.segmentsPerTile)
  private val tileCacheCfg = BitNetResidentSharedWeightTileCacheConfig(
    dataWidth = crossbar.dataWidth,
    poolCount = cfg.engineCount,
    memoryCount = cfg.segmentsPerTile,
    depth = if (cfg.lm.tileFeatures == 64) 216 else 108,
    sequenceIdWidth = matrix.sequenceIdWidth,
    sourceEpochWidth = cfg.epochWidth)

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

    // In the production shell, pool is the LM engine and address is
    // `slot * 20 + kTile`.  Default standalone instances retain their private
    // slot memories for the focused numerical/protocol regression.
    val cacheWrite = Vec.fill(cfg.engineCount)(Vec.fill(
      cfg.segmentsPerTile)(master(Stream(
        BitNetResidentWeightTileCacheWrite(tileCacheCfg)))))
    val cacheRead = master(Stream(
      BitNetResidentWeightTileCacheRead(tileCacheCfg)))
    val cacheReadData = Vec.fill(cfg.engineCount)(Vec.fill(
      cfg.segmentsPerTile)(in Bits(crossbar.dataWidth bits)))

    val canStartJob = out Bool()
    val jobActive = out Bool()
    val busy = out Bool()
    val slotOccupied = out Bits(cfg.slotCount bits)
    val replayActive = out Bool()
    val replaySlot = out UInt(cfg.slotWidth bits)
    val replayPairOrdinal = out UInt(lm.pairOrdinalWidth bits)
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(8 bits)
    val issuedReadRequests = out UInt(32 bits)
    val acceptedReadResponses = out UInt(32 bits)
    val emittedWeightTiles = out UInt(32 bits)
    val emittedScales = out UInt(32 bits)
    val readRequestsDuringReplay = out UInt(32 bits)
  }

  if (!externalWeightTileCache) {
    for (engine <- 0 until cfg.engineCount;
         memory <- 0 until cfg.segmentsPerTile) {
      io.cacheWrite(engine)(memory).valid := False
      io.cacheWrite(engine)(memory).payload :=
        BitNetResidentWeightTileCacheWrite(tileCacheCfg).getZero
    }
    io.cacheRead.valid := False
    io.cacheRead.payload := BitNetResidentWeightTileCacheRead(
      tileCacheCfg).getZero
  }

  val activeJob = RegInit(False)
  val activeSequenceId = Reg(UInt(matrix.sequenceIdWidth bits)) init 0
  val activeEpoch = Reg(UInt(cfg.epochWidth bits)) init 0
  val activeCacheGeneration = Reg(UInt(
    tileCacheCfg.generationWidth bits)) init 0
  val nextAcceptedPair = Reg(UInt(cfg.expectedPairWidth bits)) init 0
  val nextReplayPair = Reg(UInt(cfg.expectedPairWidth bits)) init 0
  val nextWeightBeatOrdinal = Reg(UInt(matrix.beatCountWidth bits)) init 0

  val occupied = Vec.fill(cfg.slotCount)(RegInit(False))
  val pairReg = Vec.fill(cfg.slotCount)(
    Reg(BitNetLmRowPairRequest(lm)) init
      BitNetLmRowPairRequest(lm).getZero)
  val pairFault = Vec.fill(cfg.slotCount)(RegInit(False))
  val slotWeightDone = Vec.fill(cfg.slotCount)(RegInit(False))

  val bankPhase = Vec.fill(cfg.slotCount)(Vec.fill(cfg.bankCount)(
    Reg(BitNetResidentLmBankPhase()) init
      BitNetResidentLmBankPhase.Idle))
  // At most one slot is in a Receive phase for a given bank, so the bank-local
  // counter is intentionally shared across the two slots.
  val bankResponseBeat = Vec.fill(cfg.bankCount)(
    Reg(UInt(cfg.responseBeatWidth bits)) init 0)

  val scaleValid = Vec.fill(cfg.slotCount)(Vec.fill(cfg.engineCount)(
    RegInit(False)))
  val scaleSent = Vec.fill(cfg.slotCount)(Vec.fill(cfg.engineCount)(
    RegInit(False)))
  val scaleFault = Vec.fill(cfg.slotCount)(Vec.fill(cfg.engineCount)(
    RegInit(False)))
  val scaleValue = Vec.fill(cfg.slotCount)(Vec.fill(cfg.engineCount)(
    Reg(Bits(32 bits)) init 0))

  val stickyFault = RegInit(False)
  val faultBitsReg = Reg(Bits(8 bits)) init 0
  val issuedReadRequestCount = Reg(UInt(32 bits)) init 0
  val acceptedReadResponseCount = Reg(UInt(32 bits)) init 0
  val emittedWeightTileCount = Reg(UInt(32 bits)) init 0
  val emittedScaleCount = Reg(UInt(32 bits)) init 0
  val readRequestsDuringReplayCount = Reg(UInt(32 bits)) init 0

  val replaying = RegInit(False)
  val replaySlotReg = Reg(UInt(cfg.slotWidth bits)) init 0
  val replayTile = Reg(UInt(lm.tileIndexWidth bits)) init 0
  val replayAllIssued = RegInit(False)

  io.canStartJob := !activeJob && !occupied.asBits.orR && !replaying
  io.jobActive := activeJob
  io.busy := occupied.asBits.orR || replaying
  io.slotOccupied := occupied.asBits
  io.replayActive := replaying
  io.replaySlot := replaySlotReg
  io.replayPairOrdinal := pairReg(replaySlotReg).pairOrdinal
  io.protocolFaultSticky := stickyFault
  io.faultBits := faultBitsReg
  io.issuedReadRequests := issuedReadRequestCount
  io.acceptedReadResponses := acceptedReadResponseCount
  io.emittedWeightTiles := emittedWeightTileCount
  io.emittedScales := emittedScaleCount
  io.readRequestsDuringReplay := readRequestsDuringReplayCount

  when(io.jobStart.valid) {
    when(!activeJob && !occupied.asBits.orR && !replaying) {
      activeJob := True
      activeSequenceId := io.jobStart.sequenceId
      activeEpoch := io.jobStart.epoch
      activeCacheGeneration := io.jobStart.cacheGeneration
      nextAcceptedPair := 0
      nextReplayPair := 0
      nextWeightBeatOrdinal := 0
      for (slot <- 0 until cfg.slotCount;
           bank <- 0 until cfg.bankCount) {
        bankPhase(slot)(bank) := BitNetResidentLmBankPhase.Idle
      }
    } otherwise {
      stickyFault := True
      faultBitsReg(BitNetResidentLmFetchFault.JobProtocol) := True
    }
  }

  when(io.jobRetire) {
    when(!activeJob || occupied.asBits.orR || replaying ||
      nextAcceptedPair =/= lm.pairCount ||
      nextReplayPair =/= lm.pairCount) {
      stickyFault := True
      faultBitsReg(BitNetResidentLmFetchFault.JobProtocol) := True
    }
    activeJob := False
  }

  // -----------------------------------------------------------------------
  // Ordered row descriptor acceptance into alternating physical slots.
  // -----------------------------------------------------------------------
  val acceptSlot = nextAcceptedPair(cfg.slotWidth - 1 downto 0)
  val expectedEngineMask = Bits(cfg.engineCount bits)
  val expectedToken = Vec(UInt(lm.tokenIdWidth bits), cfg.engineCount)
  for (engine <- 0 until cfg.engineCount) {
    val row = (io.rowPair.pairOrdinal.asBits ##
      U(engine, lm.engineWidth bits).asBits).asUInt
    expectedToken(engine) := row.resize(lm.tokenIdWidth)
    expectedEngineMask(engine) := row < lm.vocabularySize
  }

  val descriptorChecks = scala.collection.mutable.ArrayBuffer[Bool](
    activeJob,
    nextAcceptedPair < lm.pairCount,
    io.rowPair.pairOrdinal.resize(cfg.expectedPairWidth) === nextAcceptedPair,
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
        (io.rowPair.engine(engine).tokenId === expectedToken(engine) &&
          io.rowPair.engine(engine).weightAddress(bank)(6 downto 0) === 0 &&
          !weightEnd.msb))
    }
  }
  for (engine <- 0 until cfg.engineCount) {
    descriptorChecks += (!expectedEngineMask(engine) ||
      (io.rowPair.engine(engine).scaleBank < cfg.bankCount &&
        io.rowPair.engine(engine).scaleAddress(1 downto 0) === 0))
  }
  if (cfg.engineCount > 1) {
    descriptorChecks += (!expectedEngineMask.andR ||
      io.rowPair.engine(0).requestId =/=
        io.rowPair.engine(1).requestId)
  }
  val descriptorLegal = descriptorChecks.reduce(_ && _)

  io.rowPair.ready := activeJob && nextAcceptedPair < lm.pairCount &&
    !occupied(acceptSlot)
  when(io.rowPair.fire) {
    val slot = acceptSlot
    pairReg(slot) := io.rowPair.payload
    pairFault(slot) := !descriptorLegal
    occupied(slot) := True
    slotWeightDone(slot) := False
    for (engine <- 0 until cfg.engineCount) {
      scaleValid(slot)(engine) :=
        !descriptorLegal && expectedEngineMask(engine)
      scaleSent(slot)(engine) := False
      scaleFault(slot)(engine) :=
        !descriptorLegal && expectedEngineMask(engine)
      scaleValue(slot)(engine) := B(0, 32 bits)
    }
    for (bank <- 0 until cfg.bankCount) {
      bankPhase(slot)(bank) := BitNetResidentLmBankPhase.IssueWeight0
      when(!descriptorLegal) {
        bankPhase(slot)(bank) := BitNetResidentLmBankPhase.Done
      }
    }
    nextAcceptedPair := nextAcceptedPair + 1
    when(!descriptorLegal) {
      stickyFault := True
      faultBitsReg(BitNetResidentLmFetchFault.RowDescriptor) := True
    }
  }

  def encodedTag(
      requestId: UInt,
      scale: Bool,
      engine: UInt,
      bank: Int,
      slot: UInt): UInt = {
    val body = (activeEpoch.asBits ## requestId.asBits ## scale.asBits ##
      engine.asBits ## U(bank, lm.bankWidth bits).asBits ##
      slot.asBits).asUInt
    body.resize(crossbar.tagWidth)
  }

  def encodedTransaction(
      scale: Bool,
      engine: UInt,
      bank: Int,
      slot: UInt): UInt = {
    val body = (scale.asBits ## engine.asBits ##
      U(bank, lm.bankWidth bits).asBits ## slot.asBits).asUInt
    body.resize(crossbar.transactionIdWidth)
  }

  // -----------------------------------------------------------------------
  // Per-bank request scheduling.  A bank may fill either slot but never owns
  // two outstanding logical bursts.
  // -----------------------------------------------------------------------
  val bankReceiving = Vec(Bool(), cfg.bankCount)
  val bankReceiveSlot = Vec(UInt(cfg.slotWidth bits), cfg.bankCount)
  val requestFire = Bits(cfg.bankCount bits)

  for (bank <- 0 until cfg.bankCount) {
    val receiveCandidate = Vec(Bool(), cfg.slotCount)
    val issueCandidate = Vec(Bool(), cfg.slotCount)
    for (slot <- 0 until cfg.slotCount) {
      val phase = bankPhase(slot)(bank)
      receiveCandidate(slot) := occupied(slot) && (
        phase === BitNetResidentLmBankPhase.ReceiveWeight0 ||
        phase === BitNetResidentLmBankPhase.ReceiveWeight1 ||
        phase === BitNetResidentLmBankPhase.ReceiveScale0 ||
        phase === BitNetResidentLmBankPhase.ReceiveScale1)
      issueCandidate(slot) := occupied(slot) && (
        phase === BitNetResidentLmBankPhase.IssueWeight0 ||
        phase === BitNetResidentLmBankPhase.IssueWeight1 ||
        phase === BitNetResidentLmBankPhase.IssueScale0 ||
        phase === BitNetResidentLmBankPhase.IssueScale1)
    }

    bankReceiving(bank) := receiveCandidate.asBits.orR
    bankReceiveSlot(bank) := 0
    when(receiveCandidate(0)) { bankReceiveSlot(bank) := 0 }
    when(receiveCandidate(1) && (!receiveCandidate(0) ||
      pairReg(1).pairOrdinal < pairReg(0).pairOrdinal)) {
      bankReceiveSlot(bank) := 1
    }
    when(receiveCandidate.asBits.andR) {
      stickyFault := True
      pairFault(0) := True
      pairFault(1) := True
      faultBitsReg(BitNetResidentLmFetchFault.JobProtocol) := True
    }

    val requestExists = issueCandidate.asBits.orR && !bankReceiving(bank)
    val requestSlot = UInt(cfg.slotWidth bits)
    requestSlot := 0
    when(issueCandidate(1) && (!issueCandidate(0) ||
      pairReg(1).pairOrdinal < pairReg(0).pairOrdinal)) {
      requestSlot := 1
    }
    val phaseVector = Vec(
      bankPhase(0)(bank), bankPhase(1)(bank))
    val phase = phaseVector(requestSlot)
    val issueWeight1 = phase === BitNetResidentLmBankPhase.IssueWeight1
    val issueScale0 = phase === BitNetResidentLmBankPhase.IssueScale0
    val issueScale1 = phase === BitNetResidentLmBankPhase.IssueScale1
    val scale = issueScale0 || issueScale1
    val engine = UInt(lm.engineWidth bits)
    engine := 0
    when(issueWeight1 || issueScale1) { engine := 1 }

    val selectedPair = pairReg(requestSlot)
    val requestId = UInt(lm.requestIdWidth bits)
    val address = UInt(crossbar.localAddressWidth bits)
    requestId := selectedPair.engine(0).requestId
    address := selectedPair.engine(0).weightAddress(bank).resized
    if (cfg.engineCount > 1) {
      when(engine === 1) {
        requestId := selectedPair.engine(1).requestId
        address := selectedPair.engine(1).weightAddress(bank).resized
      }
    }
    when(scale) {
      val scaleAddress = UInt(lm.addressWidth bits)
      scaleAddress := selectedPair.engine(0).scaleAddress
      if (cfg.engineCount > 1) {
        when(engine === 1) {
          scaleAddress := selectedPair.engine(1).scaleAddress
        }
      }
      address := (scaleAddress &
        U(((BigInt(1) << lm.addressWidth) - 1) & ~BigInt(15),
          lm.addressWidth bits)).resized
    }

    io.readRequest(bank).valid := requestExists
    io.readRequest(bank).bank := bank
    io.readRequest(bank).transactionId :=
      encodedTransaction(scale, engine, bank, requestSlot)
    io.readRequest(bank).tag :=
      encodedTag(requestId, scale, engine, bank, requestSlot)
    io.readRequest(bank).localAddress := address
    io.readRequest(bank).beats :=
      (selectedPair.inputFeatures(bank) >> 4)
        .resize(crossbar.beatCountWidth)
    when(scale) { io.readRequest(bank).beats := 1 }
    requestFire(bank) := io.readRequest(bank).fire

    when(io.readRequest(bank).fire) {
      bankResponseBeat(bank) := 0
      switch(phase) {
        is(BitNetResidentLmBankPhase.IssueWeight0) {
          bankPhase(requestSlot)(bank) :=
            BitNetResidentLmBankPhase.ReceiveWeight0
        }
        is(BitNetResidentLmBankPhase.IssueWeight1) {
          bankPhase(requestSlot)(bank) :=
            BitNetResidentLmBankPhase.ReceiveWeight1
        }
        is(BitNetResidentLmBankPhase.IssueScale0) {
          bankPhase(requestSlot)(bank) :=
            BitNetResidentLmBankPhase.ReceiveScale0
        }
        is(BitNetResidentLmBankPhase.IssueScale1) {
          bankPhase(requestSlot)(bank) :=
            BitNetResidentLmBankPhase.ReceiveScale1
        }
      }
    }
  }
  when(requestFire.orR) {
    issuedReadRequestCount := issuedReadRequestCount +
      CountOne(requestFire).resize(32)
    when(replaying) {
      readRequestsDuringReplayCount := readRequestsDuringReplayCount +
        CountOne(requestFire).resize(32)
    }
  }

  // -----------------------------------------------------------------------
  // Tagged response validation and rotated BRAM fill.
  // -----------------------------------------------------------------------
  val receivingScale = Vec(Bool(), cfg.bankCount)
  val responseEngine = Vec(UInt(lm.engineWidth bits), cfg.bankCount)
  val responseLegal = Vec(Bool(), cfg.bankCount)
  val responseExpectedLast = Vec(Bool(), cfg.bankCount)
  val responseDesiredMemory = Vec(
    UInt(log2Up(cfg.segmentsPerTile) bits), cfg.bankCount)
  val responseGlobalTile = Vec(UInt(lm.tileIndexWidth bits), cfg.bankCount)

  for (bank <- 0 until cfg.bankCount) {
    val slot = bankReceiveSlot(bank)
    val phaseVector = Vec(
      bankPhase(0)(bank), bankPhase(1)(bank))
    val phase = phaseVector(slot)
    receivingScale(bank) := bankReceiving(bank) && (
      phase === BitNetResidentLmBankPhase.ReceiveScale0 ||
      phase === BitNetResidentLmBankPhase.ReceiveScale1)
    responseEngine(bank) := 0
    when(phase === BitNetResidentLmBankPhase.ReceiveWeight1 ||
      phase === BitNetResidentLmBankPhase.ReceiveScale1) {
      responseEngine(bank) := 1
    }

    val selectedPair = pairReg(slot)
    val expectedRequestId = UInt(lm.requestIdWidth bits)
    expectedRequestId := selectedPair.engine(0).requestId
    if (cfg.engineCount > 1) {
      when(responseEngine(bank) === 1) {
        expectedRequestId := selectedPair.engine(1).requestId
      }
    }
    val totalBeats = UInt(crossbar.beatCountWidth bits)
    totalBeats := (selectedPair.inputFeatures(bank) >> 4)
      .resize(crossbar.beatCountWidth)
    when(receivingScale(bank)) { totalBeats := 1 }
    responseExpectedLast(bank) := bankReceiving(bank) &&
      bankResponseBeat(bank).resize(crossbar.beatCountWidth) === totalBeats - 1

    val metadataLegal =
      io.readResponse(bank).bank === bank &&
      io.readResponse(bank).transactionId === encodedTransaction(
        receivingScale(bank), responseEngine(bank), bank, slot) &&
      io.readResponse(bank).tag === encodedTag(
        expectedRequestId, receivingScale(bank),
        responseEngine(bank), bank, slot) &&
      io.readResponse(bank).beatIndex === bankResponseBeat(bank).resized &&
      io.readResponse(bank).last === responseExpectedLast(bank)
    val axiLegal = !io.readResponse(bank).fault &&
      io.readResponse(bank).rresp === B"2'b00"
    responseLegal(bank) := bankReceiving(bank) && metadataLegal && axiLegal

    val globalBeat = (
      (selectedPair.inputStart(bank) >> 4)
        .resize(lm.featureIndexWidth + 1) +
        bankResponseBeat(bank).resize(lm.featureIndexWidth + 1)
    ).resize(lm.featureIndexWidth + 1)
    val naturalSegment = globalBeat(segmentIndexWidth - 1 downto 0)
    responseDesiredMemory(bank) :=
      (naturalSegment.resize(segmentIndexWidth + 1) +
        U(bank, segmentIndexWidth + 1 bits))(
          segmentIndexWidth - 1 downto 0)
    responseGlobalTile(bank) :=
      (globalBeat >> segmentIndexWidth).resize(lm.tileIndexWidth)
  }

  val responseWinnerValid = Vec.fill(cfg.slotCount)(
    Vec.fill(cfg.engineCount)(
      Vec.fill(cfg.segmentsPerTile)(Bool())))
  val responseWinnerBank = Vec.fill(cfg.slotCount)(
    Vec.fill(cfg.engineCount)(
      Vec.fill(cfg.segmentsPerTile)(UInt(lm.bankWidth bits))))
  for (slot <- 0 until cfg.slotCount;
       engine <- 0 until cfg.engineCount;
       memory <- 0 until cfg.segmentsPerTile) {
    responseWinnerValid(slot)(engine)(memory) := False
    responseWinnerBank(slot)(engine)(memory) := 0
    for (bank <- (0 until cfg.bankCount).reverse) {
      when(io.readResponse(bank).valid && bankReceiving(bank) &&
        bankReceiveSlot(bank) === slot && !receivingScale(bank) &&
        responseEngine(bank) === engine &&
        responseDesiredMemory(bank) === memory) {
        responseWinnerValid(slot)(engine)(memory) := True
        responseWinnerBank(slot)(engine)(memory) := bank
      }
    }
  }

  // A shared physical cache has one write port per engine/memory.  Collapse
  // the two logical slots before accepting a response; a losing slot is
  // backpressured, while malformed data is still drained without a write.
  val externalWinnerValid = Vec.fill(cfg.engineCount)(
    Vec.fill(cfg.segmentsPerTile)(Bool()))
  val externalWinnerSlot = Vec.fill(cfg.engineCount)(
    Vec.fill(cfg.segmentsPerTile)(UInt(cfg.slotWidth bits)))
  val externalWinnerBank = Vec.fill(cfg.engineCount)(
    Vec.fill(cfg.segmentsPerTile)(UInt(lm.bankWidth bits)))
  for (engine <- 0 until cfg.engineCount;
       memory <- 0 until cfg.segmentsPerTile) {
    externalWinnerValid(engine)(memory) := False
    externalWinnerSlot(engine)(memory) := 0
    externalWinnerBank(engine)(memory) := 0
    for (slot <- (0 until cfg.slotCount).reverse;
         bank <- (0 until cfg.bankCount).reverse) {
      when(io.readResponse(bank).valid && bankReceiving(bank) &&
        bankReceiveSlot(bank) === slot && !receivingScale(bank) &&
        responseEngine(bank) === engine &&
        responseDesiredMemory(bank) === memory) {
        externalWinnerValid(engine)(memory) := True
        externalWinnerSlot(engine)(memory) := slot
        externalWinnerBank(engine)(memory) := bank
      }
    }
  }

  for (bank <- 0 until cfg.bankCount) {
    io.readResponse(bank).ready :=
      !bankReceiving(bank) || receivingScale(bank)
    for (slot <- 0 until cfg.slotCount;
         engine <- 0 until cfg.engineCount;
         memory <- 0 until cfg.segmentsPerTile) {
      when(bankReceiving(bank) && bankReceiveSlot(bank) === slot &&
        !receivingScale(bank) && responseEngine(bank) === engine &&
        responseDesiredMemory(bank) === memory &&
        responseWinnerValid(slot)(engine)(memory) &&
        responseWinnerBank(slot)(engine)(memory) === bank) {
        if (externalWeightTileCache) {
          val selectedByPhysicalPort =
            externalWinnerValid(engine)(memory) &&
              externalWinnerSlot(engine)(memory) === slot &&
              externalWinnerBank(engine)(memory) === bank
          val cacheWriteRequired = responseLegal(bank) && !pairFault(slot)
          io.readResponse(bank).ready := !cacheWriteRequired ||
            (selectedByPhysicalPort && io.cacheWrite(engine)(memory).ready)
        } else {
          io.readResponse(bank).ready := True
        }
      }
    }
  }

  val weightMemory = if (externalWeightTileCache) None else Some(
    Array.fill(cfg.slotCount, cfg.engineCount, cfg.segmentsPerTile)(
      Mem(Bits(crossbar.dataWidth bits), lm.tileCount)))
  if (externalWeightTileCache) {
    for (engine <- 0 until cfg.engineCount;
         memory <- 0 until cfg.segmentsPerTile) {
      val slot = externalWinnerSlot(engine)(memory)
      val bank = externalWinnerBank(engine)(memory)
      val cacheAddress = UInt(tileCacheCfg.addressWidth bits)
      cacheAddress := responseGlobalTile(bank).resized
      when(slot === 1) {
        cacheAddress := (U(lm.tileCount, tileCacheCfg.addressWidth bits) +
          responseGlobalTile(bank).resize(tileCacheCfg.addressWidth)).resized
      }
      io.cacheWrite(engine)(memory).valid :=
        externalWinnerValid(engine)(memory) && responseLegal(bank) &&
          !pairFault(slot)
      io.cacheWrite(engine)(memory).generation := activeCacheGeneration
      io.cacheWrite(engine)(memory).address := cacheAddress
      io.cacheWrite(engine)(memory).data := io.readResponse(bank).data
    }
  } else {
    for (slot <- 0 until cfg.slotCount;
         engine <- 0 until cfg.engineCount;
         memory <- 0 until cfg.segmentsPerTile) {
      weightMemory.get(slot)(engine)(memory).addAttribute(
        "ram_style", "block")
      val winner = responseWinnerBank(slot)(engine)(memory)
      val winnerFire = responseWinnerValid(slot)(engine)(memory) &&
        io.readResponse(winner).fire &&
        bankReceiveSlot(winner) === slot &&
        responseEngine(winner) === engine && !receivingScale(winner)
      weightMemory.get(slot)(engine)(memory).write(
        address = responseGlobalTile(winner),
        data = io.readResponse(winner).data,
        enable = winnerFire && responseLegal(winner) && !pairFault(slot))
    }
  }

  val responseFire = Bits(cfg.bankCount bits)
  for (bank <- 0 until cfg.bankCount) {
    responseFire(bank) := io.readResponse(bank).fire
    when(io.readResponse(bank).fire) {
      when(!bankReceiving(bank)) {
        stickyFault := True
        faultBitsReg(BitNetResidentLmFetchFault.UnexpectedResponse) := True
      } otherwise {
        val slot = bankReceiveSlot(bank)
        val selectedPair = pairReg(slot)
        when(!responseLegal(bank)) {
          stickyFault := True
          pairFault(slot) := True
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
          scaleAddress := selectedPair.engine(0).scaleAddress
          if (cfg.engineCount > 1) {
            when(responseEngine(bank) === 1) {
              scaleAddress := selectedPair.engine(1).scaleAddress
            }
          }
          if (cfg.engineCount == 1) {
            scaleValue(slot)(0) := words(scaleAddress(3 downto 2))
            scaleValid(slot)(0) := True
            scaleFault(slot)(0) := !responseLegal(bank)
          } else {
            scaleValue(slot)(responseEngine(bank)) :=
              words(scaleAddress(3 downto 2))
            scaleValid(slot)(responseEngine(bank)) := True
            scaleFault(slot)(responseEngine(bank)) := !responseLegal(bank)
          }
        }

        val finishing = responseExpectedLast(bank) ||
          io.readResponse(bank).last
        when(finishing) {
          bankResponseBeat(bank) := 0
          val phaseVector = Vec(
            bankPhase(0)(bank), bankPhase(1)(bank))
          switch(phaseVector(slot)) {
            is(BitNetResidentLmBankPhase.ReceiveWeight0) {
              if (cfg.engineCount > 1) {
                when(selectedPair.engineMask(1)) {
                  bankPhase(slot)(bank) :=
                    BitNetResidentLmBankPhase.IssueWeight1
                } elsewhen(selectedPair.engine(0).scaleBank === bank) {
                  bankPhase(slot)(bank) :=
                    BitNetResidentLmBankPhase.IssueScale0
                } otherwise {
                  bankPhase(slot)(bank) := BitNetResidentLmBankPhase.Done
                }
              } else {
                when(selectedPair.engine(0).scaleBank === bank) {
                  bankPhase(slot)(bank) :=
                    BitNetResidentLmBankPhase.IssueScale0
                } otherwise {
                  bankPhase(slot)(bank) := BitNetResidentLmBankPhase.Done
                }
              }
            }
            is(BitNetResidentLmBankPhase.ReceiveWeight1) {
              when(selectedPair.engine(0).scaleBank === bank) {
                bankPhase(slot)(bank) :=
                  BitNetResidentLmBankPhase.IssueScale0
              } elsewhen((if (cfg.engineCount == 1) False else
                  selectedPair.engineMask(1) &&
                    selectedPair.engine(1).scaleBank === bank)) {
                bankPhase(slot)(bank) := BitNetResidentLmBankPhase.IssueScale1
              } otherwise {
                bankPhase(slot)(bank) := BitNetResidentLmBankPhase.Done
              }
            }
            is(BitNetResidentLmBankPhase.ReceiveScale0) {
              when(if (cfg.engineCount == 1) False else
                  selectedPair.engineMask(1) &&
                    selectedPair.engine(1).scaleBank === bank) {
                bankPhase(slot)(bank) := BitNetResidentLmBankPhase.IssueScale1
              } otherwise {
                bankPhase(slot)(bank) := BitNetResidentLmBankPhase.Done
              }
            }
            is(BitNetResidentLmBankPhase.ReceiveScale1) {
              bankPhase(slot)(bank) := BitNetResidentLmBankPhase.Done
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

  val slotBanksDone = Vec(Bool(), cfg.slotCount)
  for (slot <- 0 until cfg.slotCount) {
    slotBanksDone(slot) := bankPhase(slot).map(
      _ === BitNetResidentLmBankPhase.Done).asBits.andR
  }

  // -----------------------------------------------------------------------
  // Ordered replay.  A younger ready slot cannot pass the next pair ordinal.
  // -----------------------------------------------------------------------
  val desiredReplaySlot =
    nextReplayPair(cfg.slotWidth - 1 downto 0)
  val desiredReplayReady = activeJob && nextReplayPair < lm.pairCount &&
    occupied(desiredReplaySlot) && slotBanksDone(desiredReplaySlot) &&
    pairReg(desiredReplaySlot).pairOrdinal.resize(cfg.expectedPairWidth) ===
      nextReplayPair
  when(!replaying && desiredReplayReady) {
    replaying := True
    replaySlotReg := desiredReplaySlot
    replayTile := 0
    replayAllIssued := False
  }

  val selectedScaleEngine = UInt(lm.engineWidth bits)
  val scaleAvailable = Bool()
  val replayPair = pairReg(replaySlotReg)
  scaleAvailable := False
  selectedScaleEngine := 0
  for (engine <- (0 until cfg.engineCount).reverse) {
    when(replayPair.engineMask(engine) &&
      scaleValid(replaySlotReg)(engine) &&
      !scaleSent(replaySlotReg)(engine)) {
      scaleAvailable := True
      selectedScaleEngine := engine
    }
  }
  io.rowScale.valid := replaying && scaleAvailable
  io.rowScale.requestId := replayPair.engine(0).requestId
  io.rowScale.engine := selectedScaleEngine
  io.rowScale.tokenId := replayPair.engine(0).tokenId
  io.rowScale.scaleBank := replayPair.engine(0).scaleBank
  io.rowScale.rowScaleF32 := scaleValue(replaySlotReg)(0)
  io.rowScale.fault := pairFault(replaySlotReg) ||
    scaleFault(replaySlotReg)(0)
  for (engine <- 1 until cfg.engineCount) {
    when(selectedScaleEngine === engine) {
      io.rowScale.requestId := replayPair.engine(engine).requestId
      io.rowScale.tokenId := replayPair.engine(engine).tokenId
      io.rowScale.scaleBank := replayPair.engine(engine).scaleBank
      io.rowScale.rowScaleF32 := scaleValue(replaySlotReg)(engine)
      io.rowScale.fault := pairFault(replaySlotReg) ||
        scaleFault(replaySlotReg)(engine)
    }
  }
  when(io.rowScale.fire) {
    for (engine <- 0 until cfg.engineCount) {
      when(selectedScaleEngine === engine) {
        scaleSent(replaySlotReg)(engine) := True
      }
    }
    emittedScaleCount := emittedScaleCount + 1
  }

  val replayValid = RegInit(False)
  val replayOutputSlot = Reg(UInt(cfg.slotWidth bits)) init 0
  val replayBeatOrdinal = Reg(UInt(matrix.beatCountWidth bits)) init 0
  val replayFirst = RegInit(False)
  val replayLast = RegInit(False)
  val replayBankReg = Reg(UInt(lm.bankWidth bits)) init 0

  val replayStageReady = !replayValid || io.i8Weight.ready
  val replayCandidate = replaying && !replayAllIssued && replayStageReady
  val replayIssue = Bool()
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

  if (externalWeightTileCache) {
    val cacheAddress = UInt(tileCacheCfg.addressWidth bits)
    cacheAddress := replayTile.resized
    when(replaySlotReg === 1) {
      cacheAddress := (U(lm.tileCount, tileCacheCfg.addressWidth bits) +
        replayTile.resize(tileCacheCfg.addressWidth)).resized
    }
    io.cacheRead.valid := replayCandidate
    io.cacheRead.generation := activeCacheGeneration
    io.cacheRead.address := cacheAddress
    io.cacheRead.poolMask := replayPair.engineMask
    replayIssue := io.cacheRead.fire
  } else {
    replayIssue := replayCandidate
  }

  val replayMemoryData = Array.fill(
    cfg.slotCount, cfg.engineCount, cfg.segmentsPerTile)(
      Bits(crossbar.dataWidth bits))
  for (slot <- 0 until cfg.slotCount;
       engine <- 0 until cfg.engineCount;
       memory <- 0 until cfg.segmentsPerTile) {
    if (externalWeightTileCache) {
      replayMemoryData(slot)(engine)(memory) :=
        io.cacheReadData(engine)(memory)
    } else {
      replayMemoryData(slot)(engine)(memory) :=
        weightMemory.get(slot)(engine)(memory).readSync(
          address = replayTile,
          enable = replayIssue && replaySlotReg === slot)
    }
  }

  when(replayIssue) {
    replayValid := True
    replayOutputSlot := replaySlotReg
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

  val outputPair = pairReg(replayOutputSlot)
  io.i8Weight.valid := replayValid
  io.i8Weight.sequenceId := activeSequenceId
  io.i8Weight.beatOrdinal := replayBeatOrdinal
  io.i8Weight.use := BitNetI8WeightUse.LmHead
  for (engine <- 0 until cfg.engineCount) {
    io.i8Weight.engineOutputBase(engine) := outputPair.pairOrdinal.resized
  }
  io.i8Weight.firstK := replayFirst
  io.i8Weight.lastK := replayLast
  io.i8Weight.fault := pairFault(replayOutputSlot)
  for (engine <- 0 until cfg.engineCount; site <- 0 until lm.tileFeatures) {
    val segment = site / cfg.beatBytes
    val byte = site % cfg.beatBytes
    val rotatedMemory = (
      replayBankReg.resize(segmentIndexWidth + 1) +
        U(segment, segmentIndexWidth + 1 bits))(
          segmentIndexWidth - 1 downto 0)
    val slotByte = Vec(Bits(8 bits), cfg.slotCount)
    for (slot <- 0 until cfg.slotCount) {
      val memoryVector = Vec(
        Bits(crossbar.dataWidth bits), cfg.segmentsPerTile)
      for (memory <- 0 until cfg.segmentsPerTile) {
        memoryVector(memory) := replayMemoryData(slot)(engine)(memory)
      }
      slotByte(slot) :=
        memoryVector(rotatedMemory)(byte * 8 + 7 downto byte * 8)
    }
    io.i8Weight.engineSiteWeight(engine)(site) :=
      slotByte(replayOutputSlot)
    when(pairFault(replayOutputSlot) ||
      !outputPair.engineMask(engine)) {
      io.i8Weight.engineSiteWeight(engine)(site) := 0
    }
  }
  when(io.i8Weight.fire) {
    emittedWeightTileCount := emittedWeightTileCount + 1
    when(replayLast) { slotWeightDone(replayOutputSlot) := True }
  }

  val weightDoneAfter = slotWeightDone(replaySlotReg) ||
    (io.i8Weight.fire && replayLast &&
      replayOutputSlot === replaySlotReg)
  val scaleDoneAfter = Bits(cfg.engineCount bits)
  for (engine <- 0 until cfg.engineCount) {
    scaleDoneAfter(engine) := !replayPair.engineMask(engine) ||
      scaleSent(replaySlotReg)(engine) ||
      (io.rowScale.fire && selectedScaleEngine === engine)
  }
  when(replaying && weightDoneAfter && scaleDoneAfter.andR) {
    val slot = replaySlotReg
    occupied(slot) := False
    replaying := False
    nextReplayPair := nextReplayPair + 1
    for (bank <- 0 until cfg.bankCount) {
      bankPhase(slot)(bank) := BitNetResidentLmBankPhase.Idle
    }
  }
}
