package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetBankedWeightShard(
    target: BitNetAddressTarget.E,
    layer: Int,
    bank: Int,
    localBase: BigInt,
    bytes: BigInt,
    inputFeatures: Int,
    outputStart: Int,
    outputFeatures: Int
)

case class BitNetBankedLmShard(
    bank: Int,
    rowStart: Int,
    rows: Int,
    weightBase: BigInt,
    scaleBase: BigInt
)

case class BitNetBankedKvShard(
    bank: Int,
    head: Int,
    packedBase: BigInt,
    packedBytes: BigInt
)

/** Scala-time mirror of tools/plan_bitnet_five_bank.py. */
case class BitNetFiveBankResidentLayout(
    cfg: BitNetConfig,
    weightShards: Vector[BitNetBankedWeightShard],
    auxBases: Vector[BigInt],
    ropeBases: Vector[BigInt],
    lmShards: Vector[BitNetBankedLmShard],
    kvShards: Vector[BitNetBankedKvShard],
    usedBytes: Vector[BigInt],
    bankCapacities: Vector[BigInt]
) {
  val bankCount: Int = 5

  def weight(target: BitNetAddressTarget.E, layer: Int, bank: Int):
      BitNetBankedWeightShard =
    weightShards.find(shard => shard.target == target && shard.layer == layer &&
      shard.bank == bank).getOrElse(
      throw new IllegalArgumentException(
        s"missing native weight shard target=$target layer=$layer bank=$bank"))

  def validate(): BitNetFiveBankResidentLayout = {
    require(cfg.layerCount == 30 && cfg.hiddenSize == 2560 && cfg.ffnSize == 6912)
    require(cfg.vocabularySize == 128256 && cfg.kvHeadCount == 5 && cfg.headSize == 128)
    require(auxBases.size == bankCount && ropeBases.size == bankCount)
    require(lmShards.map(_.bank) == (0 until bankCount))
    require(kvShards.map(_.bank) == (0 until bankCount))
    require(usedBytes.size == bankCount && bankCapacities.size == bankCount)
    require(usedBytes.zip(bankCapacities).forall { case (used, capacity) =>
      used > 0 && used <= capacity
    })
    val targets = BitNetFiveBankResidentLayout.WeightGeometry.map(_._1)
    require(weightShards.size == cfg.layerCount * targets.size * bankCount)
    for (layer <- 0 until cfg.layerCount; target <- targets) {
      val shards = (0 until bankCount).map(weight(target, layer, _))
      require(shards.map(_.outputFeatures).sum ==
        BitNetFiveBankResidentLayout.outputFeatures(cfg, target))
      require(shards.map(_.bytes).sum ==
        BitNetFiveBankResidentLayout.weightBytes(cfg, target))
      require(shards.sliding(2).forall {
        case Seq(left, right) => left.outputStart + left.outputFeatures == right.outputStart
        case _ => true
      })
    }
    require(kvShards.zipWithIndex.forall { case (shard, index) =>
      shard.head == index && shard.bank == index &&
        shard.packedBase % BitNetFiveBankResidentLayout.Alignment == 0 &&
        shard.packedBytes == BigInt(cfg.layerCount) * 2 *
          (cfg.maxSequenceLength /
            BitNetFiveBankResidentLayout.KvScalePackNum) *
          BitNetFiveBankResidentLayout.KvPackedBlockBytes
    })
    this
  }
}

object BitNetFiveBankResidentLayout {
  val BankCount = 5
  val Alignment: BigInt = 128
  val PlCapacity: BigInt = BigInt(1) << 30
  val PsCapacity: BigInt = BigInt(0x10000000L)
  val AuxHeaderBytes: BigInt = 64
  val AuxBytes: BigInt = 881124
  val RopeBytes: BigInt = 1048624
  val FinalNormOffset: BigInt = 64
  val InputNormOffset: BigInt = 5184
  val PostNormOffset: BigInt = 158784
  val AttentionSubNormOffset: BigInt = 312384
  val FfnSubNormOffset: BigInt = 465984
  val ProjectionScaleOffset: BigInt = 880704
  /** One 128-bit scale channel precedes four 128-byte KV rows. */
  val KvScalePackNum: Int = 4
  val KvScaleChannelBytes: Int = KvScalePackNum * 4
  val KvPackedBlockBytes: Int = KvScaleChannelBytes +
    KvScalePackNum * BitNetConfig.Production.headSize

  case class KvPackedOffsets(scale: BigInt, vector: BigInt)

  /** Byte offsets inside one head-local packed KV allocation. */
  def kvPackedOffsets(
      cfg: BitNetConfig,
      layer: Int,
      valueKind: Boolean,
      position: Int): KvPackedOffsets = {
    require(cfg == BitNetConfig.Production)
    require(layer >= 0 && layer < cfg.layerCount)
    require(position >= 0 && position < cfg.maxSequenceLength)
    val blocksPerKind = cfg.maxSequenceLength / KvScalePackNum
    val kind = if (valueKind) 1 else 0
    val block = (layer * 2 + kind) * blocksPerKind +
      position / KvScalePackNum
    val row = position % KvScalePackNum
    val blockBase = BigInt(block) * KvPackedBlockBytes
    KvPackedOffsets(
      scale = blockBase + row * 4,
      vector = blockBase + KvScaleChannelBytes + row * cfg.headSize)
  }

  val WeightGeometry: Vector[(BitNetAddressTarget.E, String)] = Vector(
    BitNetAddressTarget.QWeight -> "q",
    BitNetAddressTarget.KWeight -> "k",
    BitNetAddressTarget.VWeight -> "v",
    BitNetAddressTarget.OWeight -> "o",
    BitNetAddressTarget.GateWeight -> "gate",
    BitNetAddressTarget.UpWeight -> "up",
    BitNetAddressTarget.DownWeight -> "down"
  )

  private def align(value: BigInt): BigInt =
    (value + Alignment - 1) & ~(Alignment - 1)

  private def partition(total: Int, bank: Int): (Int, Int) = {
    val quotient = total / BankCount
    val remainder = total % BankCount
    val count = quotient + (if (bank < remainder) 1 else 0)
    val start = bank * quotient + scala.math.min(bank, remainder)
    start -> count
  }

  def inputFeatures(cfg: BitNetConfig, target: BitNetAddressTarget.E): Int =
    if (target == BitNetAddressTarget.DownWeight) cfg.ffnSize else cfg.hiddenSize

  def outputFeatures(cfg: BitNetConfig, target: BitNetAddressTarget.E): Int =
    target match {
      case BitNetAddressTarget.KWeight | BitNetAddressTarget.VWeight =>
        cfg.kvHeadCount * cfg.headSize
      case BitNetAddressTarget.GateWeight | BitNetAddressTarget.UpWeight =>
        cfg.ffnSize
      case _ => cfg.hiddenSize
    }

  def weightBytes(cfg: BitNetConfig, target: BitNetAddressTarget.E): BigInt =
    BigInt(inputFeatures(cfg, target)) * outputFeatures(cfg, target) / 4

  def apply(cfg: BitNetConfig = BitNetConfig.Production):
      BitNetFiveBankResidentLayout = {
    require(cfg == BitNetConfig.Production,
      "five-bank production layout is bound to the deployed 30-layer model")
    val cursor = Array.fill[BigInt](BankCount)(0)
    def allocate(bank: Int, bytes: BigInt): BigInt = {
      require(bytes > 0)
      val base = align(cursor(bank))
      cursor(bank) = base + bytes
      base
    }

    val weights = Vector.newBuilder[BitNetBankedWeightShard]
    for (layer <- 0 until cfg.layerCount; (target, _) <- WeightGeometry;
         bank <- 0 until BankCount) {
      val out = outputFeatures(cfg, target)
      val in = inputFeatures(cfg, target)
      require(out % 4 == 0)
      val (groupStart, groupCount) = partition(out / 4, bank)
      val bytes = BigInt(groupCount) * in
      weights += BitNetBankedWeightShard(
        target = target,
        layer = layer,
        bank = bank,
        localBase = allocate(bank, bytes),
        bytes = bytes,
        inputFeatures = in,
        outputStart = groupStart * 4,
        outputFeatures = groupCount * 4
      )
    }

    val aux = Vector.tabulate(BankCount)(bank => allocate(bank, AuxBytes))
    val rope = Vector.tabulate(BankCount)(bank => allocate(bank, RopeBytes))

    val lm = Vector.tabulate(BankCount) { bank =>
      val (rowStart, rows) = partition(cfg.vocabularySize, bank)
      val weightBase = allocate(bank, BigInt(rows) * cfg.hiddenSize)
      val scaleBase = allocate(bank, BigInt(rows) * 4)
      BitNetBankedLmShard(bank, rowStart, rows, weightBase, scaleBase)
    }

    require(cfg.maxSequenceLength % KvScalePackNum == 0)
    val kvBlocksPerKind = cfg.maxSequenceLength / KvScalePackNum
    val kvPackedBytes = BigInt(cfg.layerCount) * 2 * kvBlocksPerKind *
      KvPackedBlockBytes
    val kv = Vector.tabulate(BankCount) { bank =>
      BitNetBankedKvShard(
        bank = bank,
        head = bank,
        packedBase = allocate(bank, kvPackedBytes),
        packedBytes = kvPackedBytes
      )
    }
    val used = cursor.map(align).toVector
    val capacities = Vector(PlCapacity) ++ Vector.fill(BankCount - 1)(PsCapacity)
    BitNetFiveBankResidentLayout(
      cfg, weights.result(), aux, rope, lm, kv, used, capacities
    ).validate()
  }
}

object BitNetBankedAddressFault extends SpinalEnum {
  val None,
      UnknownTarget,
      InvalidBank,
      InvalidLayer,
      InvalidTokenLane,
      InvalidPosition,
      InvalidLmRow,
      ZeroLength,
      Misaligned,
      TargetBounds,
      BankBounds = newElement()
}

case class BitNetBankedAddressRequest(
    cfg: BitNetConfig,
    bankCount: Int = 5) extends Bundle {
  val target = BitNetAddressTarget()
  val bank = UInt(log2Up(bankCount) bits)
  val layer = UInt(cfg.layerWidth bits)
  val mode = BitNetMode()
  val prefillBasePosition = UInt(cfg.sequenceWidth bits)
  val prefillRound = UInt(cfg.sequenceWidth bits)
  val tokenLane = UInt(cfg.roundTokenCountWidth bits)
  val decodePosition = UInt(cfg.sequenceWidth bits)
  /** Global vocabulary row, used only by LM weight/scale targets. */
  val lmRow = UInt(BitNetAddressGeometry.lmRowIndexWidth(cfg) bits)
  val byteOffset = UInt(cfg.addressWidth bits)
  val transferBytes = UInt(cfg.addressWidth bits)
}

case class BitNetBankedAddressResponse(
    cfg: BitNetConfig,
    bankCount: Int = 5) extends Bundle {
  val target = BitNetAddressTarget()
  val bank = UInt(log2Up(bankCount) bits)
  val localAddress = UInt(cfg.addressWidth bits)
  val targetBytes = UInt(cfg.addressWidth bits)
  val requiredAlignmentBytes = UInt(8 bits)
  val effectivePosition = UInt(cfg.sequenceWidth bits)
  val outputStart = UInt(BitNetAddressGeometry.lmRowIndexWidth(cfg) bits)
  val outputFeatures = UInt(BitNetAddressGeometry.lmRowIndexWidth(cfg) bits)
  val ok = Bool()
  val fault = BitNetBankedAddressFault()
}

/**
  * Bank-local production AGU matching the five-bank linker.
  *
  * Transformer rows and LM rows are true bank shards, not byte-interleaved
  * lockstep beats.  K/V head n is native to bank n.  AUX/RoPE are replicated,
  * so the caller may choose the least-contended bank for irregular small reads.
  */
class BitNetBankedAddressGenerator(
    cfg: BitNetConfig = BitNetConfig.Production,
    layout: BitNetFiveBankResidentLayout = BitNetFiveBankResidentLayout()
) extends Component {
  require(layout.cfg == cfg)
  layout.validate()
  private val bankCount = layout.bankCount
  private val bankWidth = log2Up(bankCount)
  private val tableIndexWidth = log2Up(cfg.layerCount * bankCount)
  private val calculationWidth = cfg.addressWidth + 1

  val io = new Bundle {
    val request = slave(Stream(BitNetBankedAddressRequest(cfg, bankCount)))
    val response = master(Stream(BitNetBankedAddressResponse(cfg, bankCount)))
  }

  private def uintTable(values: Seq[BigInt], width: Int): Vec[UInt] =
    Vec(values.map(value => U(value, width bits)))

  val bankValid = io.request.bank < bankCount
  val safeBank = UInt(bankWidth bits)
  safeBank := Mux(bankValid, io.request.bank, U(0, bankWidth bits))
  val layerValid = io.request.layer < cfg.layerCount
  val safeLayer = UInt(cfg.layerWidth bits)
  safeLayer := Mux(layerValid, io.request.layer, U(0, cfg.layerWidth bits))
  val tableIndex = (
    safeLayer.resize(tableIndexWidth) * bankCount +
      safeBank.resize(tableIndexWidth)
  ).resize(tableIndexWidth)

  case class WeightTables(
      bases: Vec[UInt], bytes: Vec[UInt], starts: Vec[UInt], features: Vec[UInt])
  val weightTables: Map[BitNetAddressTarget.E, WeightTables] =
    BitNetFiveBankResidentLayout.WeightGeometry.map {
    case (target, _) =>
      val shards = for (layer <- 0 until cfg.layerCount; bank <- 0 until bankCount)
        yield layout.weight(target, layer, bank)
      target -> WeightTables(
        uintTable(shards.map(_.localBase), cfg.addressWidth),
        uintTable(shards.map(_.bytes), cfg.addressWidth),
        uintTable(shards.map(shard => BigInt(shard.outputStart)),
          BitNetAddressGeometry.lmRowIndexWidth(cfg)),
        uintTable(shards.map(shard => BigInt(shard.outputFeatures)),
          BitNetAddressGeometry.lmRowIndexWidth(cfg))
      )
    }.toMap

  val auxBases = uintTable(layout.auxBases, cfg.addressWidth)
  val ropeBases = uintTable(layout.ropeBases, cfg.addressWidth)
  val lmWeightBases = uintTable(layout.lmShards.map(_.weightBase), cfg.addressWidth)
  val lmScaleBases = uintTable(layout.lmShards.map(_.scaleBase), cfg.addressWidth)
  val lmRowStarts = uintTable(layout.lmShards.map(shard => BigInt(shard.rowStart)),
    BitNetAddressGeometry.lmRowIndexWidth(cfg))
  val lmRowCounts = uintTable(layout.lmShards.map(shard => BigInt(shard.rows)),
    BitNetAddressGeometry.lmRowIndexWidth(cfg))
  val kvPackedBases = uintTable(
    layout.kvShards.map(_.packedBase), cfg.addressWidth)
  val bankCapacities = uintTable(layout.bankCapacities, cfg.addressWidth + 1)

  val prefillPosition = UInt(calculationWidth bits)
  prefillPosition := (
    io.request.prefillBasePosition.resize(calculationWidth) +
      (io.request.prefillRound.resize(calculationWidth) *
        cfg.tokenParallelism).resize(calculationWidth) +
      io.request.tokenLane.resize(calculationWidth)
  ).resize(calculationWidth)
  val effectivePosition = UInt(calculationWidth bits)
  effectivePosition := io.request.decodePosition.resize(calculationWidth)
  when(io.request.mode === BitNetMode.Prefill) {
    effectivePosition := prefillPosition
  }

  val selectedBase = UInt(cfg.addressWidth bits)
  val selectedBytes = UInt(cfg.addressWidth bits)
  val alignment = UInt(8 bits)
  val outputStart = UInt(BitNetAddressGeometry.lmRowIndexWidth(cfg) bits)
  val outputFeatures = UInt(BitNetAddressGeometry.lmRowIndexWidth(cfg) bits)
  val known = Bool()
  val requiresLayer = Bool()
  val requiresPosition = Bool()
  val requiresLmRow = Bool()

  selectedBase.allowOverride
  selectedBytes.allowOverride
  alignment.allowOverride
  outputStart.allowOverride
  outputFeatures.allowOverride
  known.allowOverride
  requiresLayer.allowOverride
  requiresPosition.allowOverride
  requiresLmRow.allowOverride

  selectedBase := 0
  selectedBytes := 0
  alignment := 1
  outputStart := 0
  outputFeatures := 0
  known := False
  requiresLayer := False
  requiresPosition := False
  requiresLmRow := False

  private def selectWeight(target: BitNetAddressTarget.E): Unit = {
    val tables = weightTables(target)
    selectedBase := tables.bases(tableIndex)
    selectedBytes := tables.bytes(tableIndex)
    alignment := 128
    outputStart := tables.starts(tableIndex)
    outputFeatures := tables.features(tableIndex)
    known := True
    requiresLayer := True
  }

  private def selectAux(offset: BigInt, bytes: BigInt,
                        layerStride: BigInt = 0): Unit = {
    val layerOffset = safeLayer.resize(calculationWidth) *
      U(layerStride, calculationWidth bits)
    selectedBase := (auxBases(safeBank).resize(calculationWidth) +
      U(offset, calculationWidth bits) + layerOffset).resize(cfg.addressWidth)
    selectedBytes := U(bytes, cfg.addressWidth bits)
    alignment := 2
    known := True
    requiresLayer := (if (layerStride != 0) True else False)
  }

  switch(io.request.target) {
    is(BitNetAddressTarget.QWeight) { selectWeight(BitNetAddressTarget.QWeight) }
    is(BitNetAddressTarget.KWeight) { selectWeight(BitNetAddressTarget.KWeight) }
    is(BitNetAddressTarget.VWeight) { selectWeight(BitNetAddressTarget.VWeight) }
    is(BitNetAddressTarget.OWeight) { selectWeight(BitNetAddressTarget.OWeight) }
    is(BitNetAddressTarget.GateWeight) { selectWeight(BitNetAddressTarget.GateWeight) }
    is(BitNetAddressTarget.UpWeight) { selectWeight(BitNetAddressTarget.UpWeight) }
    is(BitNetAddressTarget.DownWeight) { selectWeight(BitNetAddressTarget.DownWeight) }
    // GateUpWeight is intentionally absent: this older output-row-sharded
    // image stores distinct Gate/Up objects and must reject the fused ABI.

    is(BitNetAddressTarget.QProjectionScale) {
      selectAux(BitNetFiveBankResidentLayout.ProjectionScaleOffset, 2, 14)
    }
    is(BitNetAddressTarget.KProjectionScale) {
      selectAux(BitNetFiveBankResidentLayout.ProjectionScaleOffset + 2, 2, 14)
    }
    is(BitNetAddressTarget.VProjectionScale) {
      selectAux(BitNetFiveBankResidentLayout.ProjectionScaleOffset + 4, 2, 14)
    }
    is(BitNetAddressTarget.OProjectionScale) {
      selectAux(BitNetFiveBankResidentLayout.ProjectionScaleOffset + 6, 2, 14)
    }
    is(BitNetAddressTarget.GateProjectionScale) {
      selectAux(BitNetFiveBankResidentLayout.ProjectionScaleOffset + 8, 2, 14)
    }
    is(BitNetAddressTarget.UpProjectionScale) {
      selectAux(BitNetFiveBankResidentLayout.ProjectionScaleOffset + 10, 2, 14)
    }
    is(BitNetAddressTarget.DownProjectionScale) {
      selectAux(BitNetFiveBankResidentLayout.ProjectionScaleOffset + 12, 2, 14)
    }
    is(BitNetAddressTarget.InputNorm) {
      selectAux(BitNetFiveBankResidentLayout.InputNormOffset,
        BigInt(cfg.hiddenSize) * 2, BigInt(cfg.hiddenSize) * 2)
    }
    is(BitNetAddressTarget.PostAttentionNorm) {
      selectAux(BitNetFiveBankResidentLayout.PostNormOffset,
        BigInt(cfg.hiddenSize) * 2, BigInt(cfg.hiddenSize) * 2)
    }
    is(BitNetAddressTarget.AttentionSubNorm) {
      selectAux(BitNetFiveBankResidentLayout.AttentionSubNormOffset,
        BigInt(cfg.hiddenSize) * 2, BigInt(cfg.hiddenSize) * 2)
    }
    is(BitNetAddressTarget.FfnSubNorm) {
      selectAux(BitNetFiveBankResidentLayout.FfnSubNormOffset,
        BigInt(cfg.ffnSize) * 2, BigInt(cfg.ffnSize) * 2)
    }
    is(BitNetAddressTarget.FinalNorm) {
      selectAux(BitNetFiveBankResidentLayout.FinalNormOffset,
        BigInt(cfg.hiddenSize) * 2)
    }

    is(BitNetAddressTarget.RopeRow) {
      val rowOffset = effectivePosition * (cfg.headSize / 2 * 4)
      selectedBase := (ropeBases(safeBank).resize(calculationWidth) + 48 +
        rowOffset).resize(cfg.addressWidth)
      selectedBytes := U(cfg.headSize / 2 * 4, cfg.addressWidth bits)
      alignment := 2
      known := True
      requiresPosition := True
    }

    is(BitNetAddressTarget.KvKey, BitNetAddressTarget.KvValue) {
      val kind = UInt(1 bits)
      kind := (io.request.target === BitNetAddressTarget.KvValue).asUInt
      val blockInKind = (effectivePosition >> log2Up(
        BitNetFiveBankResidentLayout.KvScalePackNum)).resize(
          calculationWidth)
      val rowInBlock = effectivePosition(
        log2Up(BitNetFiveBankResidentLayout.KvScalePackNum) - 1 downto 0)
        .resize(calculationWidth)
      val packedBlock = ((safeLayer.resize(calculationWidth) * 2 + kind) *
        (cfg.maxSequenceLength /
          BitNetFiveBankResidentLayout.KvScalePackNum) + blockInKind)
        .resize(calculationWidth)
      // 528 bytes = 512 + 16; keep the address path shift/add only.
      val blockOffset = ((packedBlock << 9) + (packedBlock << 4))
        .resize(calculationWidth)
      val vectorOffset = (rowInBlock << log2Up(cfg.headSize))
        .resize(calculationWidth)
      selectedBase := (kvPackedBases(safeBank).resize(calculationWidth) +
        blockOffset + BitNetFiveBankResidentLayout.KvScaleChannelBytes +
        vectorOffset).resize(cfg.addressWidth)
      selectedBytes := U(cfg.headSize, cfg.addressWidth bits)
      // The leading 16-byte scale channel keeps every row AXI-beat aligned;
      // a row is intentionally 16 (not 128) bytes from the packed block base.
      alignment := 16
      outputStart := (safeBank.resize(outputStart.getWidth) * cfg.headSize)
        .resize(outputStart.getWidth)
      outputFeatures := cfg.headSize
      known := True
      requiresLayer := True
      requiresPosition := True
    }

    is(BitNetAddressTarget.KvKeyScale, BitNetAddressTarget.KvValueScale) {
      val kind = UInt(1 bits)
      kind := (io.request.target === BitNetAddressTarget.KvValueScale).asUInt
      val blockInKind = (effectivePosition >> log2Up(
        BitNetFiveBankResidentLayout.KvScalePackNum)).resize(
          calculationWidth)
      val rowInBlock = effectivePosition(
        log2Up(BitNetFiveBankResidentLayout.KvScalePackNum) - 1 downto 0)
        .resize(calculationWidth)
      val packedBlock = ((safeLayer.resize(calculationWidth) * 2 + kind) *
        (cfg.maxSequenceLength /
          BitNetFiveBankResidentLayout.KvScalePackNum) + blockInKind)
        .resize(calculationWidth)
      val blockOffset = ((packedBlock << 9) + (packedBlock << 4))
        .resize(calculationWidth)
      selectedBase := (kvPackedBases(safeBank).resize(calculationWidth) +
        blockOffset + (rowInBlock << 2)).resize(cfg.addressWidth)
      selectedBytes := 4
      alignment := 4
      known := True
      requiresLayer := True
      requiresPosition := True
    }

    is(BitNetAddressTarget.LmWeightTile, BitNetAddressTarget.LmScaleTile) {
      val rowStart = lmRowStarts(safeBank)
      val rows = lmRowCounts(safeBank)
      val rowEnd = rowStart.resize(rowStart.getWidth + 1) +
        rows.resize(rowStart.getWidth + 1)
      val rowValid = io.request.lmRow.resize(rowEnd.getWidth) >=
        rowStart.resize(rowEnd.getWidth) &&
        io.request.lmRow.resize(rowEnd.getWidth) < rowEnd
      val localRow = (io.request.lmRow.resize(rowEnd.getWidth) -
        rowStart.resize(rowEnd.getWidth)).resize(rowStart.getWidth)
      val remainingRows = (rowEnd - io.request.lmRow.resize(rowEnd.getWidth))
      when(io.request.target === BitNetAddressTarget.LmWeightTile) {
        selectedBase := (lmWeightBases(safeBank).resize(calculationWidth) +
          localRow.resize(calculationWidth) * cfg.hiddenSize).resize(cfg.addressWidth)
        selectedBytes := (remainingRows.resize(calculationWidth) * cfg.hiddenSize)
          .resize(cfg.addressWidth)
        alignment := 128
      } otherwise {
        selectedBase := (lmScaleBases(safeBank).resize(calculationWidth) +
          localRow.resize(calculationWidth) * 4).resize(cfg.addressWidth)
        selectedBytes := (remainingRows.resize(calculationWidth) * 4)
          .resize(cfg.addressWidth)
        alignment := 4
      }
      outputStart := io.request.lmRow
      outputFeatures := remainingRows.resize(outputFeatures.getWidth)
      known := rowValid
      requiresLmRow := True
    }
  }

  val alignmentMask = (alignment.resize(cfg.addressWidth) - 1)
    .resize(cfg.addressWidth)
  val candidate = (selectedBase.resize(calculationWidth) +
    io.request.byteOffset.resize(calculationWidth))
  val relativeEnd = io.request.byteOffset.resize(calculationWidth) +
    io.request.transferBytes.resize(calculationWidth)
  val absoluteEnd = candidate + io.request.transferBytes.resize(calculationWidth)
  val invalidLayer = requiresLayer && !layerValid
  val invalidTokenLane = requiresPosition &&
    io.request.mode === BitNetMode.Prefill &&
    io.request.tokenLane >= cfg.tokenParallelism
  val invalidPosition = requiresPosition && effectivePosition >= cfg.maxSequenceLength
  val invalidLmRow = requiresLmRow && !known
  val zeroLength = io.request.transferBytes === 0
  val misaligned = (candidate.resize(cfg.addressWidth) & alignmentMask).orR ||
    (io.request.transferBytes & alignmentMask).orR
  val targetBounds = relativeEnd > selectedBytes.resize(calculationWidth)
  val bankBounds = absoluteEnd > bankCapacities(safeBank).resize(calculationWidth)
  val anyFault = !known || !bankValid || invalidLayer || invalidTokenLane ||
    invalidPosition || invalidLmRow || zeroLength || misaligned || targetBounds || bankBounds

  val fault = BitNetBankedAddressFault()
  fault := BitNetBankedAddressFault.None
  when(!bankValid) {
    fault := BitNetBankedAddressFault.InvalidBank
  } elsewhen (!known && !requiresLmRow) {
    fault := BitNetBankedAddressFault.UnknownTarget
  } elsewhen (invalidLayer) {
    fault := BitNetBankedAddressFault.InvalidLayer
  } elsewhen (invalidTokenLane) {
    fault := BitNetBankedAddressFault.InvalidTokenLane
  } elsewhen (invalidPosition) {
    fault := BitNetBankedAddressFault.InvalidPosition
  } elsewhen (invalidLmRow) {
    fault := BitNetBankedAddressFault.InvalidLmRow
  } elsewhen (zeroLength) {
    fault := BitNetBankedAddressFault.ZeroLength
  } elsewhen (misaligned) {
    fault := BitNetBankedAddressFault.Misaligned
  } elsewhen (targetBounds) {
    fault := BitNetBankedAddressFault.TargetBounds
  } elsewhen (bankBounds) {
    fault := BitNetBankedAddressFault.BankBounds
  }

  val computed = Stream(BitNetBankedAddressResponse(cfg, bankCount))
  computed.valid := io.request.valid
  io.request.ready := computed.ready
  computed.target := io.request.target
  computed.bank := io.request.bank
  computed.localAddress := candidate.resize(cfg.addressWidth)
  computed.targetBytes := selectedBytes
  computed.requiredAlignmentBytes := alignment
  computed.effectivePosition := effectivePosition.resize(cfg.sequenceWidth)
  computed.outputStart := outputStart
  computed.outputFeatures := outputFeatures
  computed.ok := !anyFault
  computed.fault := fault
  io.response << computed.m2sPipe()
}
