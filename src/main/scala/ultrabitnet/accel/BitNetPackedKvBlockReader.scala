package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetPackedKvKind extends SpinalEnum(binarySequential) {
  val Key, Value = newElement()
}

case class BitNetPackedKvBlockReaderConfig(
    model: BitNetConfig = BitNetConfig.Production,
    crossbar: BitNetBankLocalCrossbarConfig =
      BitNetBankLocalCrossbarConfig(),
    maxSeq: Int = 4096
) {
  require(model == BitNetConfig.Production,
    "the packed KV ABI is bound to the production resident image")
  require(maxSeq >= 4 && maxSeq <= model.maxSequenceLength && isPow2(maxSeq))
  require(model.kvHeadCount == crossbar.bankCount)
  require(model.headSize == 128 && crossbar.dataWidth == 128)
  require(BitNetFiveBankStripedLayout.KvScalePackNum == 4)
  require(BitNetFiveBankStripedLayout.KvScaleChannelBytes == 16)
  require(BitNetFiveBankStripedLayout.KvPackedBlockBytes == 528)

  val rowsPerBlock: Int = BitNetFiveBankStripedLayout.KvScalePackNum
  val segmentsPerRow: Int = model.headSize / (crossbar.dataWidth / 8)
  val beatsPerBlock: Int = 1 + rowsPerBlock * segmentsPerRow
  val blockIndexWidth: Int = log2Up(model.maxSequenceLength / rowsPerBlock)
  val passIndexWidth: Int = log2Up(model.queryHeadCount)
  val kvHeadWidth: Int = log2Up(model.kvHeadCount)
  val rowSlotCount: Int = 2
}

case class BitNetPackedKvSweepCommand(
    cfg: BitNetPackedKvBlockReaderConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val kind = BitNetPackedKvKind()
  /** Ordered rows to replay, from one through maxSeq. */
  val keyCount = UInt(cfg.model.sequenceWidth bits)
}

case class BitNetPackedKvRowBeat(
    cfg: BitNetPackedKvBlockReaderConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val kind = BitNetPackedKvKind()
  val row = Vec(SInt(8 bits), cfg.model.headSize)
  val factorQ16 = UInt(32 bits)
  val keyPosition = UInt(cfg.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
  val success = Bool()
}

object BitNetPackedKvBlockReaderMath {
  def blockLocalAddress(
      layout: BitNetFiveBankStripedLayout,
      layer: Int,
      kvHead: Int,
      valueKind: Boolean,
      block: Int): BigInt = {
    val cfg = layout.cfg
    require(layer >= 0 && layer < cfg.layerCount)
    require(kvHead >= 0 && kvHead < cfg.kvHeadCount)
    val blocksPerKind = cfg.maxSequenceLength /
      BitNetFiveBankStripedLayout.KvScalePackNum
    require(block >= 0 && block < blocksPerKind)
    val kind = if (valueKind) 1 else 0
    layout.kvShards(kvHead).packedBase +
      BigInt((layer * 2 + kind) * blocksPerKind + block) *
        BitNetFiveBankStripedLayout.KvPackedBlockBytes
  }

  def parseScaleChannel(bytes: Seq[Int]): Seq[Long] = {
    require(bytes.length == 16 && bytes.forall(v => v >= 0 && v <= 255))
    (0 until 4).map { row =>
      (0 until 4).foldLeft(0L) { case (value, byte) =>
        value | (bytes(row * 4 + byte).toLong << (byte * 8))
      } & 0xffffffffL
    }
  }
}

/**
  * BitNet packed KV sweep reader.
  *
  * One bank-local command reads a 16-byte four-scale channel followed by four
  * 128-byte rows.  Two 128-byte ping-pong row slots overlap AXI reception and
  * matrix consumption; if both rows are retained by downstream backpressure,
  * the response stream is backpressured before either slot can be overwritten.
  * Only one block is outstanding, which makes arbitrary AXI ID reordering
  * irrelevant while retaining a continuous 33-beat burst per four rows.
  */
class BitNetPackedKvBlockReader(
    cfg: BitNetPackedKvBlockReaderConfig =
      BitNetPackedKvBlockReaderConfig(),
    layout: BitNetFiveBankStripedLayout =
      BitNetFiveBankStripedLayout()) extends Component {
  require(layout.cfg == cfg.model)
  layout.validate()
  private val rowsPerBlock = cfg.rowsPerBlock
  private val segmentsPerRow = cfg.segmentsPerRow
  private val bytesPerSegment = cfg.crossbar.dataWidth / 8

  val io = new Bundle {
    val command = slave(Stream(BitNetPackedKvSweepCommand(cfg)))
    val readRequest = master(Stream(
      BitNetClientBankReadRequest(cfg.crossbar)))
    val readResponse = slave(Stream(
      BitNetClientBankReadResponse(cfg.crossbar)))
    val row = master(Stream(BitNetPackedKvRowBeat(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedCommands = out UInt(32 bits)
    val issuedBlocks = out UInt(32 bits)
    val acceptedReadBeats = out UInt(32 bits)
    val emittedRows = out UInt(32 bits)
    val completedCommands = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Idle, IssueBlock, ReceiveBlock = newElement()
  }
  val state = Reg(State()) init State.Idle
  val active = Reg(BitNetPackedKvSweepCommand(cfg)) init (
    BitNetPackedKvSweepCommand(cfg).getZero)
  val blockIndex = Reg(UInt(cfg.blockIndexWidth bits)) init 0
  val responseBeat = Reg(UInt(log2Up(cfg.beatsPerBlock) bits)) init 0
  val responseDone = RegInit(False)
  val rowsInBlock = Reg(UInt(log2Up(rowsPerBlock + 1) bits)) init 0
  val emittedInBlock = Reg(UInt(log2Up(rowsPerBlock + 1) bits)) init 0
  val nextEmitKey = Reg(UInt(cfg.model.sequenceWidth bits)) init 0
  val blockFault = RegInit(False)

  val scale = Vec(Reg(UInt(32 bits)) init 0, rowsPerBlock)
  val slotValid = Vec(RegInit(False), cfg.rowSlotCount)
  val slotKeyIndex = Vec(
    Reg(UInt(cfg.model.sequenceWidth bits)) init 0, cfg.rowSlotCount)
  val slotFactor = Vec(Reg(UInt(32 bits)) init 0, cfg.rowSlotCount)
  val slotSuccess = Vec(RegInit(False), cfg.rowSlotCount)
  val rowSegments = Vec((0 until cfg.rowSlotCount).map { _ =>
    Vec(Reg(Bits(cfg.crossbar.dataWidth bits)) init 0, segmentsPerRow)
  })

  val fault = RegInit(False)
  val acceptedCommands = Reg(UInt(32 bits)) init 0
  val issuedBlocks = Reg(UInt(32 bits)) init 0
  val acceptedReadBeats = Reg(UInt(32 bits)) init 0
  val emittedRows = Reg(UInt(32 bits)) init 0
  val completedCommands = Reg(UInt(32 bits)) init 0
  io.busy := state =/= State.Idle
  io.protocolFaultSticky := fault
  io.acceptedCommands := acceptedCommands
  io.issuedBlocks := issuedBlocks
  io.acceptedReadBeats := acceptedReadBeats
  io.emittedRows := emittedRows
  io.completedCommands := completedCommands

  val commandLegal = io.command.layer < cfg.model.layerCount &&
    io.command.kvHead < cfg.model.kvHeadCount &&
    io.command.keyCount =/= 0 && io.command.keyCount <= cfg.maxSeq
  io.command.ready := state === State.Idle
  when(io.command.fire) {
    when(commandLegal) {
      active := io.command.payload
      blockIndex := 0
      nextEmitKey := 0
      responseDone := False
      emittedInBlock := 0
      slotValid.foreach(_ := False)
      acceptedCommands := acceptedCommands + 1
      state := State.IssueBlock
    } otherwise {
      fault := True
    }
  }

  val blocksPerKind = cfg.model.maxSequenceLength / rowsPerBlock
  val kindIndex = active.kind.asBits.asUInt.resize(cfg.model.addressWidth + 1)
  val packedBlock = ((active.layer.resize(cfg.model.addressWidth + 1) * 2 +
    kindIndex) * blocksPerKind +
    blockIndex.resize(cfg.model.addressWidth + 1))
    .resize(cfg.model.addressWidth + 1)
  val packedBlockOffset = ((packedBlock << 9) + (packedBlock << 4))
    .resize(cfg.model.addressWidth + 1)
  val kvBases = Vec(layout.kvShards.map(shard =>
    U(shard.packedBase, cfg.model.addressWidth bits)))
  val blockAddress = (kvBases(active.kvHead).resize(
    cfg.model.addressWidth + 1) + packedBlockOffset)
    .resize(cfg.model.addressWidth)

  val expectedTransaction = blockIndex.resize(
    cfg.crossbar.transactionIdWidth)
  val expectedTag = (active.requestId.resize(cfg.crossbar.tagWidth) ^
    (active.layer.resize(cfg.crossbar.tagWidth) << 12).resize(cfg.crossbar.tagWidth) ^
    (active.passIndex.resize(cfg.crossbar.tagWidth) << 7).resize(cfg.crossbar.tagWidth) ^
    (active.kind.asBits.asUInt.resize(cfg.crossbar.tagWidth) << 6)
      .resize(cfg.crossbar.tagWidth) ^
    blockIndex.resize(cfg.crossbar.tagWidth)).resize(cfg.crossbar.tagWidth)

  io.readRequest.valid := state === State.IssueBlock
  io.readRequest.bank := active.kvHead.resize(cfg.crossbar.bankWidth)
  io.readRequest.transactionId := expectedTransaction
  io.readRequest.tag := expectedTag
  io.readRequest.localAddress := blockAddress.resize(
    cfg.crossbar.localAddressWidth)
  io.readRequest.beats := cfg.beatsPerBlock
  when(io.readRequest.fire) {
    val remaining = active.keyCount -
      (blockIndex.resize(cfg.model.sequenceWidth) << log2Up(rowsPerBlock))
    rowsInBlock := Mux(remaining > rowsPerBlock,
      U(rowsPerBlock, rowsInBlock.getWidth bits),
      remaining.resize(rowsInBlock.getWidth))
    responseBeat := 0
    responseDone := False
    emittedInBlock := 0
    blockFault := False
    issuedBlocks := issuedBlocks + 1
    state := State.ReceiveBlock
  }

  val dataBeat = responseBeat =/= 0
  val dataOrdinal = (responseBeat - 1).resize(log2Up(
    rowsPerBlock * segmentsPerRow))
  val receiveRow = (dataOrdinal >> log2Up(segmentsPerRow))
    .resize(log2Up(rowsPerBlock))
  val receiveSegment = dataOrdinal(log2Up(segmentsPerRow) - 1 downto 0)
    .resize(log2Up(segmentsPerRow))
  val receiveSlot = receiveRow(0).asUInt
  val activeReceiveRow = receiveRow.resize(rowsInBlock.getWidth) < rowsInBlock
  val startingRow = receiveSegment === 0
  val slotAvailable = !activeReceiveRow || !startingRow ||
    !slotValid(receiveSlot)
  io.readResponse.ready := state === State.ReceiveBlock &&
    !responseDone && slotAvailable

  val responseMetadataLegal =
    io.readResponse.bank === active.kvHead.resize(cfg.crossbar.bankWidth) &&
    io.readResponse.transactionId === expectedTransaction &&
    io.readResponse.tag === expectedTag &&
    io.readResponse.beatIndex === responseBeat.resize(
      cfg.crossbar.beatIndexWidth) &&
    io.readResponse.last === (responseBeat === cfg.beatsPerBlock - 1) &&
    !io.readResponse.fault && io.readResponse.rresp === B"2'b00"

  when(io.readResponse.fire) {
    acceptedReadBeats := acceptedReadBeats + 1
    when(!responseMetadataLegal) {
      blockFault := True
      fault := True
    }
    when(!dataBeat) {
      for (row <- 0 until rowsPerBlock) {
        scale(row) := io.readResponse.data(row * 32 + 31 downto row * 32)
          .asUInt
      }
    } otherwise {
      when(activeReceiveRow) {
        rowSegments(receiveSlot)(receiveSegment) := io.readResponse.data
        when(receiveSegment === segmentsPerRow - 1) {
          val globalKey = (blockIndex.resize(cfg.model.sequenceWidth) <<
            log2Up(rowsPerBlock)) + receiveRow.resize(
              cfg.model.sequenceWidth)
          slotKeyIndex(receiveSlot) := globalKey.resized
          slotFactor(receiveSlot) := scale(receiveRow)
          slotSuccess(receiveSlot) := !blockFault && responseMetadataLegal
          slotValid(receiveSlot) := True
        }
      }
    }
    when(responseBeat === cfg.beatsPerBlock - 1) {
      responseDone := True
    } otherwise {
      responseBeat := responseBeat + 1
    }
  }

  val emitSelect = UInt(log2Up(cfg.rowSlotCount) bits)
  emitSelect := 0
  val emitExists = Bool()
  emitExists := False
  for (slot <- (0 until cfg.rowSlotCount).reverse) {
    when(slotValid(slot) && slotKeyIndex(slot) === nextEmitKey) {
      emitSelect := slot
      emitExists := True
    }
  }
  io.row.valid := emitExists
  io.row.requestId := active.requestId
  io.row.layer := active.layer
  io.row.passIndex := active.passIndex
  io.row.kvHead := active.kvHead
  io.row.kind := active.kind
  io.row.factorQ16 := slotFactor(emitSelect)
  io.row.keyPosition := slotKeyIndex(emitSelect)
  io.row.keyIndex := slotKeyIndex(emitSelect)
  io.row.firstInSequence := slotKeyIndex(emitSelect) === 0
  io.row.lastInSequence := slotKeyIndex(emitSelect) === active.keyCount - 1
  io.row.success := slotSuccess(emitSelect)
  for (feature <- 0 until cfg.model.headSize) {
    val segment = feature / bytesPerSegment
    val byte = feature % bytesPerSegment
    io.row.row(feature) := rowSegments(emitSelect)(segment)(
      byte * 8 + 7 downto byte * 8).asSInt
  }

  when(io.row.fire) {
    slotValid(emitSelect) := False
    nextEmitKey := nextEmitKey + 1
    emittedInBlock := emittedInBlock + 1
    emittedRows := emittedRows + 1
  }

  val blockDrained = responseDone &&
    emittedInBlock === rowsInBlock && !slotValid.asBits.orR
  when(state === State.ReceiveBlock && blockDrained) {
    val nextBlockBase = ((blockIndex.resize(cfg.model.sequenceWidth) + 1) <<
      log2Up(rowsPerBlock)).resize(cfg.model.sequenceWidth)
    when(nextBlockBase >= active.keyCount) {
      completedCommands := completedCommands + 1
      state := State.Idle
    } otherwise {
      blockIndex := blockIndex + 1
      state := State.IssueBlock
    }
  }
}

object GenerateBitNetPackedKvBlockReader extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-packed-kv-block-reader")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetPackedKvBlockReader())
}
