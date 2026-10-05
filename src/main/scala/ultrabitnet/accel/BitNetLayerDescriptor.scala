package ultrabitnet.accel

import spinal.core._

import scala.language.postfixOps

/**
  * One fixed dense-transformer layer in the native dataflow.
  *
  * These entries replace the old per-operation resident microprogram.  The
  * layer controller always executes the same dense body and changes only this
  * descriptor index.  Explicit addresses also allow a future linker to place
  * individual tensors on different PL-visible DDR windows without changing
  * the controller.
  */
case class BitNetLayerDescriptor(cfg: BitNetConfig) extends Bundle {
  val qWeightBase = UInt(cfg.addressWidth bits)
  val kWeightBase = UInt(cfg.addressWidth bits)
  val vWeightBase = UInt(cfg.addressWidth bits)
  val oWeightBase = UInt(cfg.addressWidth bits)
  val gateWeightBase = UInt(cfg.addressWidth bits)
  val upWeightBase = UInt(cfg.addressWidth bits)
  val downWeightBase = UInt(cfg.addressWidth bits)

  val inputNormBase = UInt(cfg.addressWidth bits)
  val postAttentionNormBase = UInt(cfg.addressWidth bits)
  val attentionSubNormBase = UInt(cfg.addressWidth bits)
  val ffnSubNormBase = UInt(cfg.addressWidth bits)

  /** Seven consecutive BF16 projection scales in Q,K,V,O,Gate,Up,Down order. */
  val projectionScaleBase = UInt(cfg.addressWidth bits)
}

/** Elaboration-time form of [[BitNetLayerDescriptor]]. */
case class BitNetLayerDescriptorValue(
    qWeightBase: BigInt,
    kWeightBase: BigInt,
    vWeightBase: BigInt,
    oWeightBase: BigInt,
    gateWeightBase: BigInt,
    upWeightBase: BigInt,
    downWeightBase: BigInt,
    inputNormBase: BigInt,
    postAttentionNormBase: BigInt,
    attentionSubNormBase: BigInt,
    ffnSubNormBase: BigInt,
    projectionScaleBase: BigInt
) {
  def weightBases: Vector[BigInt] = Vector(
    qWeightBase,
    kWeightBase,
    vWeightBase,
    oWeightBase,
    gateWeightBase,
    upWeightBase,
    downWeightBase
  )

  def normBases: Vector[BigInt] = Vector(
    inputNormBase,
    postAttentionNormBase,
    attentionSubNormBase,
    ffnSubNormBase
  )
}

/** One streaming LM-head tile; logits are never materialised as a full vector. */
case class BitNetLmTileDescriptor(cfg: BitNetConfig) extends Bundle {
  val rowStart = UInt(BitNetAddressGeometry.lmRowIndexWidth(cfg) bits)
  val rows = UInt(BitNetAddressGeometry.lmTileRowCountWidth bits)
  val weightBase = UInt(cfg.addressWidth bits)
  val scaleBase = UInt(cfg.addressWidth bits)
}

case class BitNetLmTileDescriptorValue(
    rowStart: Int,
    rows: Int,
    weightBase: BigInt,
    scaleBase: BigInt
)

/** Static tensor sizes and bus alignment used by the package linker. */
object BitNetAddressGeometry {
  val BusAlignmentBytes: BigInt = 32
  val ScalarBf16Bytes: BigInt = 2
  val ScalarF32Bytes: BigInt = 4
  val RopeHeaderBytes: BigInt = 48
  val LmShardRows: Int = 8192
  val LmTileRows: Int = 4096
  val ProjectionCount: Int = 7
  val lmTileRowCountWidth: Int = log2Up(LmTileRows + 1)

  def lmTileCount(cfg: BitNetConfig): Int =
    (cfg.vocabularySize + LmTileRows - 1) / LmTileRows

  def lmTileIndexWidth(cfg: BitNetConfig): Int =
    math.max(1, log2Up(lmTileCount(cfg) + 1))

  def lmRowIndexWidth(cfg: BitNetConfig): Int =
    math.max(1, log2Up(cfg.vocabularySize + 1))

  def qWeightBytes(cfg: BitNetConfig): BigInt =
    BigInt(cfg.hiddenSize) * cfg.hiddenSize / 4

  def kWeightBytes(cfg: BitNetConfig): BigInt =
    BigInt(cfg.kvHeadCount) * cfg.headSize * cfg.hiddenSize / 4

  def vWeightBytes(cfg: BitNetConfig): BigInt = kWeightBytes(cfg)

  def oWeightBytes(cfg: BitNetConfig): BigInt = qWeightBytes(cfg)

  def gateWeightBytes(cfg: BitNetConfig): BigInt =
    BigInt(cfg.ffnSize) * cfg.hiddenSize / 4

  def upWeightBytes(cfg: BitNetConfig): BigInt = gateWeightBytes(cfg)

  def downWeightBytes(cfg: BitNetConfig): BigInt =
    BigInt(cfg.hiddenSize) * cfg.ffnSize / 4

  def weightBytes(cfg: BitNetConfig): Vector[BigInt] = Vector(
    qWeightBytes(cfg),
    kWeightBytes(cfg),
    vWeightBytes(cfg),
    oWeightBytes(cfg),
    gateWeightBytes(cfg),
    upWeightBytes(cfg),
    downWeightBytes(cfg)
  )

  def hiddenNormBytes(cfg: BitNetConfig): BigInt = BigInt(cfg.hiddenSize) * 2
  def ffnNormBytes(cfg: BitNetConfig): BigInt = BigInt(cfg.ffnSize) * 2
  def kvSize(cfg: BitNetConfig): BigInt = BigInt(cfg.kvHeadCount) * cfg.headSize
  def ropeRowBytes(cfg: BitNetConfig): BigInt = BigInt(cfg.headSize / 2) * 4
}

/**
  * Complete native address map.  Layer descriptors are deliberately separate
  * from the global RoPE/KV/LM metadata: the former are indexed once per dense
  * layer body, while the latter are indexed by token position or LM tile.
  */
case class BitNetResidentAddressLayout(
    cfg: BitNetConfig,
    layers: Vector[BitNetLayerDescriptorValue],
    finalNormBase: BigInt,
    ropeAssetBase: BigInt,
    ropeTableBase: BigInt,
    ropeRowBytes: BigInt,
    ropeTableBytes: BigInt,
    kvDataBase: BigInt,
    kvDataBytes: BigInt,
    kvDataLayerStride: BigInt,
    kvDataTokenStride: BigInt,
    kvDataKindStride: BigInt,
    kvScaleBase: BigInt,
    kvScaleBytes: BigInt,
    kvScaleLayerStride: BigInt,
    kvScaleKindStride: BigInt,
    kvScaleTokenStride: BigInt,
    lmTiles: Vector[BitNetLmTileDescriptorValue],
    residentBytes: BigInt
) {
  import BitNetAddressGeometry._

  private case class Range(name: String, base: BigInt, bytes: BigInt) {
    val end: BigInt = base + bytes
  }

  /** Fail elaboration if a generated/linker-supplied map can escape or overlap. */
  def validate(): BitNetResidentAddressLayout = {
    val addressLimit = BigInt(1) << cfg.addressWidth
    require(layers.length == cfg.layerCount,
      s"native descriptor table must contain exactly ${cfg.layerCount} entries")
    require(lmTiles.length == lmTileCount(cfg),
      s"native LM table must contain exactly ${lmTileCount(cfg)} tiles")
    require(residentBytes > 0 && residentBytes <= addressLimit,
      "resident byte count is outside the native address space")

    def aligned(name: String, value: BigInt, alignment: BigInt): Unit = {
      require(value >= 0 && value < addressLimit, s"$name is outside the address space")
      require((value & (alignment - 1)) == 0,
        s"$name must be $alignment-byte aligned")
    }

    val weightSizes = weightBytes(cfg)
    val ranges = scala.collection.mutable.ArrayBuffer.empty[Range]
    layers.zipWithIndex.foreach { case (layer, index) =>
      layer.weightBases.zip(weightSizes).zipWithIndex.foreach {
        case ((base, bytes), projection) =>
          aligned(s"layer $index weight $projection", base, BusAlignmentBytes)
          ranges += Range(s"layer.$index.weight.$projection", base, bytes)
      }
      val normSizes = Vector(
        hiddenNormBytes(cfg),
        hiddenNormBytes(cfg),
        hiddenNormBytes(cfg),
        ffnNormBytes(cfg)
      )
      layer.normBases.zip(normSizes).zipWithIndex.foreach {
        case ((base, bytes), norm) =>
          aligned(s"layer $index norm $norm", base, ScalarBf16Bytes)
          ranges += Range(s"layer.$index.norm.$norm", base, bytes)
      }
      aligned(s"layer $index projection scales", layer.projectionScaleBase,
        ScalarBf16Bytes)
      ranges += Range(
        s"layer.$index.projection-scales",
        layer.projectionScaleBase,
        ProjectionCount * ScalarBf16Bytes
      )
    }

    aligned("final norm", finalNormBase, ScalarBf16Bytes)
    ranges += Range("final-norm", finalNormBase, hiddenNormBytes(cfg))

    aligned("RoPE asset", ropeAssetBase, BusAlignmentBytes)
    aligned("RoPE table", ropeTableBase, ScalarBf16Bytes)
    require(ropeTableBase == ropeAssetBase + RopeHeaderBytes,
      "RoPE table must immediately follow the 48-byte production header")
    require(ropeRowBytes == BitNetAddressGeometry.ropeRowBytes(cfg),
      "RoPE row stride does not match head size")
    require(ropeTableBytes == BigInt(cfg.maxSequenceLength) * ropeRowBytes,
      "RoPE table does not cover the configured sequence length exactly")
    ranges += Range("rope-asset", ropeAssetBase, RopeHeaderBytes + ropeTableBytes)

    aligned("KV data", kvDataBase, BusAlignmentBytes)
    aligned("KV scales", kvScaleBase, BusAlignmentBytes)
    require(kvDataTokenStride == kvSize(cfg) * 2,
      "KV data token stride must contain one K and one V vector")
    require(kvDataKindStride == kvSize(cfg),
      "KV data kind stride must equal one compact i8 vector")
    require(kvDataLayerStride == BigInt(cfg.maxSequenceLength) * kvDataTokenStride,
      "KV data layer stride is invalid")
    require(kvDataBytes == BigInt(cfg.layerCount) * kvDataLayerStride,
      "KV data byte count is invalid")
    require(kvScaleTokenStride == ScalarF32Bytes,
      "KV scale token stride must be one F32 value")
    require(kvScaleKindStride == BigInt(cfg.maxSequenceLength) * ScalarF32Bytes,
      "KV scale kind stride is invalid")
    require(kvScaleLayerStride == kvScaleKindStride * 2,
      "KV scale layer stride must contain K and V scales")
    require(kvScaleBytes == BigInt(cfg.layerCount) * kvScaleLayerStride,
      "KV scale byte count is invalid")
    ranges += Range("kv-data", kvDataBase, kvDataBytes)
    ranges += Range("kv-scales", kvScaleBase, kvScaleBytes)

    var nextRow = 0
    lmTiles.zipWithIndex.foreach { case (tile, index) =>
      require(tile.rowStart == nextRow, s"LM tile $index is not row-contiguous")
      require(tile.rows > 0 && tile.rows <= LmTileRows && tile.rows % 4 == 0,
        s"LM tile $index has an invalid row count")
      aligned(s"LM tile $index weights", tile.weightBase, BusAlignmentBytes)
      aligned(s"LM tile $index scales", tile.scaleBase, ScalarF32Bytes)
      ranges += Range(
        s"lm.$index.weights",
        tile.weightBase,
        BigInt(tile.rows) * cfg.hiddenSize
      )
      ranges += Range(s"lm.$index.scales", tile.scaleBase, BigInt(tile.rows) * 4)
      nextRow += tile.rows
    }
    require(nextRow == cfg.vocabularySize,
      "LM tile rows do not cover the vocabulary exactly")

    ranges.foreach { range =>
      require(range.bytes > 0 && range.base >= 0 && range.end <= addressLimit,
        s"${range.name} is outside the native address space")
    }
    val sorted = ranges.sortBy(_.base)
    sorted.sliding(2).foreach {
      case Seq(previous, current) =>
        require(previous.end <= current.base,
          s"native address ranges overlap: ${previous.name} and ${current.name}")
      case _ =>
    }
    this
  }
}

/**
  * Exact 30-layer production map emitted by tools/plan_plddr_layout.py.
  *
  * The arithmetic is intentionally repeated here instead of embedding opaque
  * hexadecimal constants.  It is therefore straightforward to compare this
  * table against a linker manifest, and any shape drift fails elaboration.
  */
object BitNetProductionAddressLayout {
  import BitNetAddressGeometry._

  private def alignUp(value: BigInt, alignment: BigInt = BusAlignmentBytes): BigInt = {
    require(alignment > 0 && (alignment & (alignment - 1)) == 0)
    (value + alignment - 1) & -alignment
  }

  def apply(cfg: BitNetConfig = BitNetConfig.Production): BitNetResidentAddressLayout = {
    require(cfg.layerCount == 30, "production package contains 30 dense layers")
    require(cfg.hiddenSize == 2560, "production package hidden size is 2560")
    require(cfg.ffnSize == 6912, "production package FFN size is 6912")
    require(cfg.vocabularySize == 128256, "production package vocabulary is 128256")
    require(cfg.kvHeadCount * cfg.headSize == 640,
      "production package compact KV width is 640")
    require(cfg.headSize == 128, "production RoPE head size is 128")
    require(cfg.maxSequenceLength == 4096,
      "production RoPE/KV assets contain 4096 positions")

    val weightSizes = weightBytes(cfg)
    var weightCursor = BigInt(0)
    val weightBases = Vector.tabulate(cfg.layerCount) { _ =>
      val bases = weightSizes.map { bytes =>
        val base = alignUp(weightCursor)
        weightCursor = base + bytes
        base
      }
      bases
    }

    val auxiliaryBase = alignUp(weightCursor)
    val finalNormOffset = BigInt(64)
    val inputNormOffset = finalNormOffset + hiddenNormBytes(cfg)
    val postAttentionNormOffset =
      inputNormOffset + BigInt(cfg.layerCount) * hiddenNormBytes(cfg)
    val attentionSubNormOffset =
      postAttentionNormOffset + BigInt(cfg.layerCount) * hiddenNormBytes(cfg)
    val ffnSubNormOffset =
      attentionSubNormOffset + BigInt(cfg.layerCount) * hiddenNormBytes(cfg)
    val projectionScaleOffset =
      ffnSubNormOffset + BigInt(cfg.layerCount) * ffnNormBytes(cfg)
    val auxiliaryBytes = alignUp(
      projectionScaleOffset +
        BigInt(cfg.layerCount) * ProjectionCount * ScalarBf16Bytes,
      4
    )

    val layers = Vector.tabulate(cfg.layerCount) { layer =>
      val bases = weightBases(layer)
      BitNetLayerDescriptorValue(
        qWeightBase = bases(0),
        kWeightBase = bases(1),
        vWeightBase = bases(2),
        oWeightBase = bases(3),
        gateWeightBase = bases(4),
        upWeightBase = bases(5),
        downWeightBase = bases(6),
        inputNormBase = auxiliaryBase + inputNormOffset +
          BigInt(layer) * hiddenNormBytes(cfg),
        postAttentionNormBase = auxiliaryBase + postAttentionNormOffset +
          BigInt(layer) * hiddenNormBytes(cfg),
        attentionSubNormBase = auxiliaryBase + attentionSubNormOffset +
          BigInt(layer) * hiddenNormBytes(cfg),
        ffnSubNormBase = auxiliaryBase + ffnSubNormOffset +
          BigInt(layer) * ffnNormBytes(cfg),
        projectionScaleBase = auxiliaryBase + projectionScaleOffset +
          BigInt(layer) * ProjectionCount * ScalarBf16Bytes
      )
    }

    val ropeAssetBase = alignUp(auxiliaryBase + auxiliaryBytes)
    val ropeStride = BitNetAddressGeometry.ropeRowBytes(cfg)
    val ropeBytes = BigInt(cfg.maxSequenceLength) * ropeStride
    val ropeTableBase = ropeAssetBase + RopeHeaderBytes

    var lmCursor = alignUp(ropeAssetBase + RopeHeaderBytes + ropeBytes)
    val lmTiles = scala.collection.mutable.ArrayBuffer.empty[BitNetLmTileDescriptorValue]
    var shardRowStart = 0
    while (shardRowStart < cfg.vocabularySize) {
      val shardRows = math.min(LmShardRows, cfg.vocabularySize - shardRowStart)
      val shardBase = alignUp(lmCursor)
      var tileOffset = 0
      while (tileOffset < shardRows) {
        val rows = math.min(LmTileRows, shardRows - tileOffset)
        lmTiles += BitNetLmTileDescriptorValue(
          rowStart = shardRowStart + tileOffset,
          rows = rows,
          weightBase = shardBase + BigInt(tileOffset) * cfg.hiddenSize,
          scaleBase = shardBase + BigInt(shardRows) * cfg.hiddenSize +
            BigInt(tileOffset) * ScalarF32Bytes
        )
        tileOffset += rows
      }
      lmCursor = shardBase + BigInt(shardRows) *
        (cfg.hiddenSize + ScalarF32Bytes.toInt)
      shardRowStart += shardRows
    }

    val kvDataBase = alignUp(lmCursor)
    val kvKindStride = kvSize(cfg)
    val kvTokenStride = kvKindStride * 2
    val kvLayerStride = BigInt(cfg.maxSequenceLength) * kvTokenStride
    val kvDataBytes = BigInt(cfg.layerCount) * kvLayerStride
    val kvScaleBase = alignUp(kvDataBase + kvDataBytes)
    val kvScaleTokenStride = ScalarF32Bytes
    val kvScaleKindStride = BigInt(cfg.maxSequenceLength) * kvScaleTokenStride
    val kvScaleLayerStride = kvScaleKindStride * 2
    val kvScaleBytes = BigInt(cfg.layerCount) * kvScaleLayerStride
    val residentBytes = alignUp(kvScaleBase + kvScaleBytes)

    BitNetResidentAddressLayout(
      cfg = cfg,
      layers = layers,
      finalNormBase = auxiliaryBase + finalNormOffset,
      ropeAssetBase = ropeAssetBase,
      ropeTableBase = ropeTableBase,
      ropeRowBytes = ropeStride,
      ropeTableBytes = ropeBytes,
      kvDataBase = kvDataBase,
      kvDataBytes = kvDataBytes,
      kvDataLayerStride = kvLayerStride,
      kvDataTokenStride = kvTokenStride,
      kvDataKindStride = kvKindStride,
      kvScaleBase = kvScaleBase,
      kvScaleBytes = kvScaleBytes,
      kvScaleLayerStride = kvScaleLayerStride,
      kvScaleKindStride = kvScaleKindStride,
      kvScaleTokenStride = kvScaleTokenStride,
      lmTiles = lmTiles.toVector,
      residentBytes = residentBytes
    ).validate()
  }
}

/** Asynchronous 30-entry descriptor lookup for the fixed dense layer body. */
class BitNetLayerDescriptorTable(
    cfg: BitNetConfig = BitNetConfig.Production,
    layout: BitNetResidentAddressLayout = BitNetProductionAddressLayout()
) extends Component {
  require(layout.cfg == cfg, "descriptor layout/configuration mismatch")
  layout.validate()

  val io = new Bundle {
    val layer = in UInt(cfg.layerWidth bits)
    val descriptor = out(BitNetLayerDescriptor(cfg))
    val valid = out Bool()
  }

  val entries = Vec(BitNetLayerDescriptor(cfg), cfg.layerCount)
  layout.layers.zipWithIndex.foreach { case (value, index) =>
    entries(index).qWeightBase := U(value.qWeightBase, cfg.addressWidth bits)
    entries(index).kWeightBase := U(value.kWeightBase, cfg.addressWidth bits)
    entries(index).vWeightBase := U(value.vWeightBase, cfg.addressWidth bits)
    entries(index).oWeightBase := U(value.oWeightBase, cfg.addressWidth bits)
    entries(index).gateWeightBase := U(value.gateWeightBase, cfg.addressWidth bits)
    entries(index).upWeightBase := U(value.upWeightBase, cfg.addressWidth bits)
    entries(index).downWeightBase := U(value.downWeightBase, cfg.addressWidth bits)
    entries(index).inputNormBase := U(value.inputNormBase, cfg.addressWidth bits)
    entries(index).postAttentionNormBase :=
      U(value.postAttentionNormBase, cfg.addressWidth bits)
    entries(index).attentionSubNormBase :=
      U(value.attentionSubNormBase, cfg.addressWidth bits)
    entries(index).ffnSubNormBase := U(value.ffnSubNormBase, cfg.addressWidth bits)
    entries(index).projectionScaleBase :=
      U(value.projectionScaleBase, cfg.addressWidth bits)
  }

  io.valid := io.layer < cfg.layerCount
  val safeLayer = Mux(io.valid, io.layer, U(0, cfg.layerWidth bits))
  io.descriptor := entries(safeLayer)
}

/** Small independent descriptor lookup for the 32 streaming LM-head tiles. */
class BitNetLmTileDescriptorTable(
    cfg: BitNetConfig = BitNetConfig.Production,
    layout: BitNetResidentAddressLayout = BitNetProductionAddressLayout()
) extends Component {
  require(layout.cfg == cfg, "LM layout/configuration mismatch")
  layout.validate()

  private val tileCount = BitNetAddressGeometry.lmTileCount(cfg)
  private val tableIndexWidth = math.max(1, log2Up(tileCount))

  val io = new Bundle {
    val tile = in UInt(BitNetAddressGeometry.lmTileIndexWidth(cfg) bits)
    val descriptor = out(BitNetLmTileDescriptor(cfg))
    val valid = out Bool()
  }

  val entries = Vec(BitNetLmTileDescriptor(cfg), tileCount)
  layout.lmTiles.zipWithIndex.foreach { case (value, index) =>
    entries(index).rowStart := U(value.rowStart,
      BitNetAddressGeometry.lmRowIndexWidth(cfg) bits)
    entries(index).rows := U(value.rows,
      BitNetAddressGeometry.lmTileRowCountWidth bits)
    entries(index).weightBase := U(value.weightBase, cfg.addressWidth bits)
    entries(index).scaleBase := U(value.scaleBase, cfg.addressWidth bits)
  }

  io.valid := io.tile < tileCount
  val safeTile = Mux(
    io.valid,
    io.tile,
    U(0, BitNetAddressGeometry.lmTileIndexWidth(cfg) bits)
  )
  io.descriptor := entries(safeTile.resize(tableIndexWidth))
}
