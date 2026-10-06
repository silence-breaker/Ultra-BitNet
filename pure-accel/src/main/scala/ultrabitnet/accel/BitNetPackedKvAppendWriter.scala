package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetPackedKvAppendWriterConfig(
    model: BitNetConfig = BitNetConfig.Production,
    qkv: BitNetQkvPostprocessConfig = BitNetQkvPostprocessConfig(),
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig()
) {
  require(model == BitNetConfig.Production)
  require(qkv.tokenLanes == model.tokenParallelism)
  require(qkv.kvHeadCount == model.kvHeadCount)
  require(qkv.headSize == model.headSize)
  require((model.tokenParallelism == 2 || model.tokenParallelism == 4) &&
    model.kvHeadCount == 5)
  require(model.headSize == 128 && crossbar.dataWidth == 128)
  require(BitNetFiveBankStripedLayout.KvScalePackNum == 4)
  require(BitNetFiveBankStripedLayout.KvPackedBlockBytes == 528)

  val rowsPerBlock: Int = BitNetFiveBankStripedLayout.KvScalePackNum
  val bytesPerBeat: Int = crossbar.dataWidth / 8
  val segmentsPerRow: Int = model.headSize / bytesPerBeat
  val beatsPerBlock: Int = 1 + rowsPerBlock * segmentsPerRow
  val blockIndexWidth: Int = log2Up(model.maxSequenceLength / rowsPerBlock)
  val kvHeadWidth: Int = log2Up(model.kvHeadCount)
}

/** One QKV projection round owns all five K and five V frames. */
case class BitNetPackedKvAppendCommand(
    cfg: BitNetPackedKvAppendWriterConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val mode = BitNetMode()
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenId = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.model.tokenParallelism)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
}

case class BitNetPackedKvAppendCompletion(
    cfg: BitNetPackedKvAppendWriterConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val success = Bool()
}

object BitNetPackedKvAppendWriterMath {
  case class BlockBeat(data: BigInt, strobe: Int)

  /** Pure mirror of the partial-block WDATA mapping. */
  def blockBeats(
      block: Int,
      positions: Seq[Int],
      active: Seq[Boolean],
      factors: Seq[Long],
      rows: Seq[Seq[Int]]): Seq[BlockBeat] = {
    require(positions.length == 4 && active.length == 4)
    require(factors.length == 4 && rows.length == 4)
    require(rows.forall(row => row.length == 128))
    val scaleData = (0 until 4).foldLeft(BigInt(0)) { (data, row) =>
      val lane = (0 until 4).find { lane =>
        active(lane) && positions(lane) / 4 == block &&
          positions(lane) % 4 == row
      }
      lane match {
        case Some(index) => data |
          ((BigInt(factors(index)) & 0xffffffffL) << (row * 32))
        case None => data
      }
    }
    val scaleStrobe = (0 until 4).foldLeft(0) { (mask, row) =>
      if ((0 until 4).exists(lane => active(lane) &&
        positions(lane) / 4 == block && positions(lane) % 4 == row))
        mask | (0xf << (row * 4))
      else mask
    }
    val vector = for {
      row <- 0 until 4
      segment <- 0 until 8
    } yield {
      val lane = (0 until 4).find { lane =>
        active(lane) && positions(lane) / 4 == block &&
          positions(lane) % 4 == row
      }
      lane match {
        case Some(index) =>
          val data = (0 until 16).foldLeft(BigInt(0)) { (word, byte) =>
            word | ((BigInt(rows(index)(segment * 16 + byte)) & 0xff) <<
              (byte * 8))
          }
          BlockBeat(data, 0xffff)
        case None => BlockBeat(0, 0)
      }
    }
    BlockBeat(scaleData, scaleStrobe) +: vector
  }
}

/**
  * Bank-local packed K/V append owner.
  *
  * K and V scalar frames are captured directly from QKV post-processing.
  * Every touched four-position block is emitted as one regular 33-beat burst:
  * a four-scale channel followed by four rows.  WSTRB preserves untouched
  * rows, so decode updates one cache entry without a read/modify/write through
  * the PS.  A TP4 prefill round aligned to four positions naturally becomes a
  * fully populated contiguous burst.
  */
class BitNetPackedKvAppendWriter(
    cfg: BitNetPackedKvAppendWriterConfig =
      BitNetPackedKvAppendWriterConfig(),
    layout: BitNetFiveBankStripedLayout =
      BitNetFiveBankStripedLayout()) extends Component {
  require(layout.cfg == cfg.model)
  layout.validate()
  private val lanes = cfg.model.tokenParallelism
  private val segments = cfg.segmentsPerRow
  private val heads = cfg.model.kvHeadCount

  val io = new Bundle {
    val command = slave(Stream(BitNetPackedKvAppendCommand(cfg)))
    val key = slave(Stream(BitNetQkvQuantizedSample(cfg.qkv)))
    val value = slave(Stream(BitNetQkvQuantizedSample(cfg.qkv)))
    val writeRequest = master(Stream(
      BitNetClientBankWriteRequest(cfg.crossbar)))
    val writeData = master(Stream(
      BitNetClientBankWriteData(cfg.crossbar)))
    val writeResponse = slave(Stream(
      BitNetClientBankWriteResponse(cfg.crossbar)))
    val completion = master(Stream(BitNetPackedKvAppendCompletion(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedCommands = out UInt(32 bits)
    val acceptedFrames = out UInt(32 bits)
    val issuedBlocks = out UInt(32 bits)
    val emittedWriteBeats = out UInt(32 bits)
    val completedCommands = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Idle, Capture, IssueBlock, SendBlock, WaitResponse = newElement()
  }
  val state = Reg(State()) init State.Idle
  val active = Reg(BitNetPackedKvAppendCommand(cfg)) init (
    BitNetPackedKvAppendCommand(cfg).getZero)
  val commandFault = RegInit(False)
  val faultSticky = RegInit(False)
  val keyHeads = Reg(Bits(heads bits)) init 0
  val valueHeads = Reg(Bits(heads bits)) init 0

  val frameActive = RegInit(False)
  val frameKind = Reg(BitNetPackedKvKind()) init BitNetPackedKvKind.Key
  val frameHead = Reg(UInt(cfg.kvHeadWidth bits)) init 0
  val expectedElement = Reg(UInt(cfg.qkv.elementWidth bits)) init 0
  val expectedLane = Reg(UInt(cfg.qkv.tokenLaneWidth bits)) init 0
  val frameFactors = Vec(Reg(UInt(32 bits)) init 0, lanes)
  val segmentAccumulator = Vec(Reg(Bits(128 bits)) init 0, lanes)
  val frameSegments = Vec((0 until lanes).map { _ =>
    Vec(Reg(Bits(128 bits)) init 0, segments)
  })

  val emitBlock = Reg(UInt(cfg.blockIndexWidth bits)) init 0
  val finalBlock = Reg(UInt(cfg.blockIndexWidth bits)) init 0
  val writeBeat = Reg(UInt(log2Up(cfg.beatsPerBlock) bits)) init 0

  val completionValid = RegInit(False)
  val completionPayload = Reg(BitNetPackedKvAppendCompletion(cfg)) init (
    BitNetPackedKvAppendCompletion(cfg).getZero)
  io.completion.valid := completionValid
  io.completion.payload := completionPayload
  when(io.completion.fire) { completionValid := False }

  val acceptedCommandCount = Reg(UInt(32 bits)) init 0
  val acceptedFrameCount = Reg(UInt(32 bits)) init 0
  val issuedBlockCount = Reg(UInt(32 bits)) init 0
  val emittedBeatCount = Reg(UInt(32 bits)) init 0
  val completedCommandCount = Reg(UInt(32 bits)) init 0
  io.acceptedCommands := acceptedCommandCount
  io.acceptedFrames := acceptedFrameCount
  io.issuedBlocks := issuedBlockCount
  io.emittedWriteBeats := emittedBeatCount
  io.completedCommands := completedCommandCount
  io.protocolFaultSticky := faultSticky

  val legalPrefillMask = (1 to lanes).map { count =>
    io.command.tokenMask === B((BigInt(1) << count) - 1, lanes bits)
  }.reduce(_ || _)
  val legalPrefill = io.command.mode === BitNetMode.Prefill &&
    legalPrefillMask
  val legalDecode = io.command.mode === BitNetMode.Decode &&
    io.command.tokenMask === B(1, lanes bits)
  val activePositionsInRange = (0 until lanes).map { lane =>
    !io.command.tokenMask(lane) ||
      io.command.tokenId(lane) < cfg.model.maxSequenceLength
  }.andR
  val positionsConsecutive = (1 until lanes).map { lane =>
    !io.command.tokenMask(lane) ||
      io.command.tokenId(lane) === io.command.tokenId(0) + lane
  }.andR
  val commandLegal = io.command.layer < cfg.model.layerCount &&
    (legalPrefill || legalDecode) && activePositionsInRange &&
    positionsConsecutive

  io.command.ready := state === State.Idle && !completionValid
  when(io.command.fire) {
    active := io.command.payload
    commandFault := !commandLegal
    keyHeads := 0
    valueHeads := 0
    frameActive := False
    expectedElement := 0
    expectedLane := 0
    state := State.Capture
    acceptedCommandCount := acceptedCommandCount + 1
    when(!commandLegal) { faultSticky := True }
  }

  val captureReady = state === State.Capture
  io.key.ready := captureReady
  io.value.ready := captureReady
  val bothInputs = io.key.valid && io.value.valid
  val selectedValid = io.key.valid || io.value.valid
  val selected = BitNetQkvQuantizedSample(cfg.qkv)
  selected := io.value.payload
  when(io.key.valid) { selected := io.key.payload }
  val selectedKind = BitNetPackedKvKind()
  selectedKind := BitNetPackedKvKind.Value
  when(io.key.valid) { selectedKind := BitNetPackedKvKind.Key }
  val selectedFire = captureReady && selectedValid

  val expectedFirst = !frameActive && expectedElement === 0 &&
    expectedLane === 0
  val expectedLast = frameActive &&
    expectedElement === cfg.model.headSize - 1 && expectedLane === lanes - 1
  val selectedHeadInRange = selected.head < heads
  val selectedHead = selected.head.resize(cfg.kvHeadWidth)
  val selectedHeadOneHot = UIntToOh(selectedHead, heads)
  val selectedDuplicate = Mux(selectedKind === BitNetPackedKvKind.Key,
    (keyHeads & selectedHeadOneHot).orR,
    (valueHeads & selectedHeadOneHot).orR)
  val expectedTokenId = active.tokenId(selected.tokenLane)
  val expectedTokenActive = active.tokenMask(selected.tokenLane)
  val sampleOrderLegal = selected.element === expectedElement &&
    selected.tokenLane === expectedLane
  val sampleFrameLegal = Mux(expectedFirst,
    selected.first && !selectedDuplicate,
    !selected.first && selectedKind === frameKind &&
      selectedHead === frameHead)
  val sampleEndLegal = selected.last === expectedLast
  val sampleFactorLegal = selected.element === 0 ||
    !selected.tokenActive ||
    selected.dequantFactorQ16 === frameFactors(selected.tokenLane)
  val sampleLegal = !bothInputs && selectedHeadInRange &&
    sampleOrderLegal && sampleFrameLegal && sampleEndLegal &&
    sampleFactorLegal &&
    selected.tokenId.resize(cfg.model.sequenceWidth) === expectedTokenId &&
    selected.tokenActive === expectedTokenActive

  when(selectedFire) {
    when(!sampleLegal) {
      commandFault := True
      faultSticky := True
    }
    when(expectedFirst) {
      frameActive := True
      frameKind := selectedKind
      frameHead := selectedHead
      for (lane <- 0 until lanes) {
        segmentAccumulator(lane) := B(0, 128 bits)
      }
    }
    when(selected.element === 0) {
      frameFactors(selected.tokenLane) := Mux(selected.tokenActive,
        selected.dequantFactorQ16, U(0, 32 bits))
      when(selected.tokenActive && !selected.dequantFactorQ16.orR) {
        commandFault := True
        faultSticky := True
      }
    }

    val byteInSegment = selected.element(3 downto 0)
    val segmentIndex = (selected.element >> 4).resize(log2Up(segments))
    val assembledSegment = Bits(128 bits)
    assembledSegment := segmentAccumulator(selected.tokenLane)
    switch(byteInSegment) {
      for (byte <- 0 until cfg.bytesPerBeat) {
        is(byte) {
          assembledSegment(byte * 8 + 7 downto byte * 8) := Mux(
            selected.tokenActive, selected.value.asBits, B(0, 8 bits))
        }
      }
    }
    segmentAccumulator(selected.tokenLane) := assembledSegment
    when(byteInSegment === cfg.bytesPerBeat - 1) {
      frameSegments(selected.tokenLane)(segmentIndex) := assembledSegment
      segmentAccumulator(selected.tokenLane) := B(0, 128 bits)
    }

    when(expectedLane === lanes - 1) {
      expectedLane := 0
      when(expectedElement === cfg.model.headSize - 1) {
        expectedElement := 0
      } otherwise {
        expectedElement := expectedElement + 1
      }
    } otherwise {
      expectedLane := expectedLane + 1
    }

    when(expectedLast) {
      frameActive := False
      emitBlock := (active.tokenId(0) >> 2).resize(cfg.blockIndexWidth)
      val lastPosition = UInt(cfg.model.sequenceWidth bits)
      lastPosition := active.tokenId(0)
      for (lane <- 1 until lanes) {
        when(active.tokenMask(lane)) { lastPosition := active.tokenId(lane) }
      }
      finalBlock := (lastPosition >> 2).resize(cfg.blockIndexWidth)
      acceptedFrameCount := acceptedFrameCount + 1
      state := State.IssueBlock
    }
  }

  val blocksPerKind = cfg.model.maxSequenceLength / cfg.rowsPerBlock
  val kindIndex = frameKind.asBits.asUInt.resize(cfg.model.addressWidth + 1)
  val packedBlock = ((active.layer.resize(cfg.model.addressWidth + 1) * 2 +
    kindIndex) * blocksPerKind +
    emitBlock.resize(cfg.model.addressWidth + 1))
    .resize(cfg.model.addressWidth + 1)
  val packedOffset = ((packedBlock << 9) + (packedBlock << 4))
    .resize(cfg.model.addressWidth + 1)
  val kvBases = Vec(layout.kvShards.map(shard =>
    U(shard.packedBase, cfg.model.addressWidth bits)))
  val blockAddress = (kvBases(frameHead).resize(cfg.model.addressWidth + 1) +
    packedOffset).resize(cfg.model.addressWidth)

  val expectedTransaction = ((frameKind.asBits.asUInt.resize(
    cfg.crossbar.transactionIdWidth) << 7).resize(
      cfg.crossbar.transactionIdWidth) ^
    (frameHead.resize(cfg.crossbar.transactionIdWidth) << 4).resize(
      cfg.crossbar.transactionIdWidth) ^
    emitBlock.resize(cfg.crossbar.transactionIdWidth))
    .resize(cfg.crossbar.transactionIdWidth)
  val expectedTag = (active.requestId.resize(cfg.crossbar.tagWidth) ^
    (active.layer.resize(cfg.crossbar.tagWidth) << 14).resize(
      cfg.crossbar.tagWidth) ^
    (frameKind.asBits.asUInt.resize(cfg.crossbar.tagWidth) << 13).resize(
      cfg.crossbar.tagWidth) ^
    (frameHead.resize(cfg.crossbar.tagWidth) << 10).resize(
      cfg.crossbar.tagWidth) ^
    emitBlock.resize(cfg.crossbar.tagWidth)).resize(cfg.crossbar.tagWidth)

  io.writeRequest.valid := state === State.IssueBlock
  io.writeRequest.bank := frameHead.resize(cfg.crossbar.bankWidth)
  io.writeRequest.transactionId := expectedTransaction
  io.writeRequest.tag := expectedTag
  io.writeRequest.localAddress := blockAddress.resize(
    cfg.crossbar.localAddressWidth)
  io.writeRequest.beats := cfg.beatsPerBlock
  when(io.writeRequest.fire) {
    writeBeat := 0
    issuedBlockCount := issuedBlockCount + 1
    state := State.SendBlock
  }

  val dataOrdinal = (writeBeat - 1).resize(log2Up(
    cfg.rowsPerBlock * segments))
  val beatRow = (dataOrdinal >> log2Up(segments)).resize(2)
  val beatSegment = dataOrdinal(log2Up(segments) - 1 downto 0)
    .resize(log2Up(segments))
  val selectedLane = UInt(log2Up(lanes) bits)
  val rowOwned = Bool()
  selectedLane := 0
  rowOwned := False
  for (lane <- (0 until lanes).reverse) {
    val laneBlock = (active.tokenId(lane) >> 2).resize(cfg.blockIndexWidth)
    val laneRow = active.tokenId(lane)(1 downto 0)
    when(active.tokenMask(lane) && laneBlock === emitBlock &&
      laneRow === beatRow) {
      selectedLane := lane
      rowOwned := True
    }
  }

  val scaleData = Bits(cfg.crossbar.dataWidth bits)
  val scaleStrobe = Bits(cfg.crossbar.strobeWidth bits)
  scaleData := 0
  scaleStrobe := 0
  for (row <- 0 until cfg.rowsPerBlock) {
    val scaleLane = UInt(log2Up(lanes) bits)
    val scaleOwned = Bool()
    scaleLane := 0
    scaleOwned := False
    for (lane <- (0 until lanes).reverse) {
      val laneBlock = (active.tokenId(lane) >> 2).resize(cfg.blockIndexWidth)
      when(active.tokenMask(lane) && laneBlock === emitBlock &&
        active.tokenId(lane)(1 downto 0) === row) {
        scaleLane := lane
        scaleOwned := True
      }
    }
    when(scaleOwned) {
      scaleData(row * 32 + 31 downto row * 32) := frameFactors(scaleLane).asBits
      scaleStrobe(row * 4 + 3 downto row * 4) := B"4'b1111"
    }
  }

  io.writeData.valid := state === State.SendBlock
  io.writeData.bank := frameHead.resize(cfg.crossbar.bankWidth)
  io.writeData.transactionId := expectedTransaction
  io.writeData.tag := expectedTag
  io.writeData.beatIndex := writeBeat.resize(cfg.crossbar.beatIndexWidth)
  io.writeData.data := 0
  io.writeData.strobe := 0
  io.writeData.last := writeBeat === cfg.beatsPerBlock - 1
  when(writeBeat === 0) {
    io.writeData.data := scaleData
    io.writeData.strobe := scaleStrobe
  } otherwise {
    io.writeData.data := Mux(rowOwned,
      frameSegments(selectedLane)(beatSegment), B(0, cfg.crossbar.dataWidth bits))
    io.writeData.strobe := Mux(rowOwned,
      B((BigInt(1) << cfg.crossbar.strobeWidth) - 1,
        cfg.crossbar.strobeWidth bits),
      B(0, cfg.crossbar.strobeWidth bits))
  }
  when(io.writeData.fire) {
    emittedBeatCount := emittedBeatCount + 1
    when(writeBeat === cfg.beatsPerBlock - 1) {
      state := State.WaitResponse
    } otherwise {
      writeBeat := writeBeat + 1
    }
  }

  io.writeResponse.ready := state === State.WaitResponse
  val responseLegal = io.writeResponse.bank ===
    frameHead.resize(cfg.crossbar.bankWidth) &&
    io.writeResponse.transactionId === expectedTransaction &&
    io.writeResponse.tag === expectedTag &&
    !io.writeResponse.fault && io.writeResponse.bresp === B"2'b00"
  when(io.writeResponse.fire) {
    when(!responseLegal) {
      commandFault := True
      faultSticky := True
    }
    when(emitBlock =/= finalBlock) {
      emitBlock := emitBlock + 1
      state := State.IssueBlock
    } otherwise {
      val frameOneHot = UIntToOh(frameHead, heads)
      val keyAfter = keyHeads | Mux(
        frameKind === BitNetPackedKvKind.Key,
        frameOneHot, B(0, heads bits))
      val valueAfter = valueHeads | Mux(
        frameKind === BitNetPackedKvKind.Value,
        frameOneHot, B(0, heads bits))
      keyHeads := keyAfter
      valueHeads := valueAfter
      when(keyAfter.andR && valueAfter.andR) {
        completionPayload.requestId := active.requestId
        completionPayload.layer := active.layer
        completionPayload.tokenBase := active.tokenBase
        completionPayload.tokenMask := active.tokenMask
        completionPayload.success := !commandFault && responseLegal
        completionValid := True
        completedCommandCount := completedCommandCount + 1
        state := State.Idle
      } otherwise {
        state := State.Capture
      }
    }
  }

  io.busy := state =/= State.Idle || completionValid
}

object GenerateBitNetPackedKvAppendWriter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-packed-kv-append-writer")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetPackedKvAppendWriter())
}
