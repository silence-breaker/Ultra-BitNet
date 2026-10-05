package ultrabitnet.accel

/**
  * One K-dimension stripe of a compact group-of-four ternary matrix.
  *
  * The source matrix is group-major: one byte represents four output rows at
  * one input feature.  Every bank stores the same output groups, but only its
  * contiguous, 128-feature-aligned K slice.  Five banks can therefore fill a
  * complete group concurrently without changing the natural output order.
  */
case class BitNetStripedWeightShard(
    target: BitNetAddressTarget.E,
    layer: Int,
    bank: Int,
    localBase: BigInt,
    bytes: BigInt,
    inputStart: Int,
    inputFeatures: Int,
    outputFeatures: Int
) {
  require(inputStart >= 0 && inputFeatures > 0 && outputFeatures > 0)
  require((inputStart & 127) == 0 && (inputFeatures & 127) == 0)
  require((outputFeatures & 3) == 0)

  val outputGroups: Int = outputFeatures / 4
  val groupStrideBytes: Int = inputFeatures

  def groupLocalBase(group: Int): BigInt = {
    require(group >= 0 && group < outputGroups)
    localBase + BigInt(group) * groupStrideBytes
  }
}

/** LM rows are K-striped as well; F32 scales remain row-sharded. */
case class BitNetStripedLmShard(
    bank: Int,
    inputStart: Int,
    inputFeatures: Int,
    weightBase: BigInt,
    weightBytes: BigInt,
    scaleRowStart: Int,
    scaleRows: Int,
    scaleBase: BigInt,
    scaleBytes: BigInt
) {
  require(bank >= 0 && bank < BitNetFiveBankStripedLayout.BankCount)
  require(inputStart >= 0 && inputFeatures > 0)
  require((inputStart & 127) == 0 && (inputFeatures & 127) == 0)
  require(scaleRowStart >= 0 && scaleRows > 0)
}

/** One packed K/V head owned by a logical bank-local address window. */
case class BitNetStripedKvShard(
    bank: Int,
    head: Int,
    packedBase: BigInt,
    packedBytes: BigInt
) {
  require(bank >= 0 && bank < BitNetFiveBankStripedLayout.BankCount)
  require(head >= 0)
  require(packedBase >= 0 && packedBytes > 0)
}

/**
  * Bandwidth-correct resident image for one PL-DDR and four PS-DDR HP ports.
  *
  * The older [[BitNetFiveBankResidentLayout]] partitions output groups.  A
  * group then lives in one bank and a matrix engine can consume at most one
  * 128-bit bank response per cycle.  This production layout partitions K
  * tiles instead.  All five banks participate in every Transformer group and
  * every LM row, matching BitNet's wide weight synchronisation while
  * retaining the intended 5x128-bit PL-side port geometry.  Banks 1..4 are
  * logical HP windows sharing one physical PS-DDR controller; only bank 0 is
  * a separate physical PL DDR channel.
  */
case class BitNetFiveBankStripedLayout(
    cfg: BitNetConfig,
    weightShards: Vector[BitNetStripedWeightShard],
    auxBases: Vector[BigInt],
    ropeBases: Vector[BigInt],
    lmShards: Vector[BitNetStripedLmShard],
    kvShards: Vector[BitNetStripedKvShard],
    usedBytes: Vector[BigInt],
    bankCapacities: Vector[BigInt]
) {
  import BitNetFiveBankStripedLayout._

  def weight(
      target: BitNetAddressTarget.E,
      layer: Int,
      bank: Int): BitNetStripedWeightShard =
    weightShards.find(shard =>
      shard.target == target && shard.layer == layer && shard.bank == bank)
      .getOrElse(throw new IllegalArgumentException(
        s"missing striped weight target=$target layer=$layer bank=$bank"))

  def validate(): BitNetFiveBankStripedLayout = {
    require(cfg == BitNetConfig.Production,
      "the striped five-bank image is bound to the deployed model")
    require(auxBases.size == BankCount && ropeBases.size == BankCount)
    require(lmShards.map(_.bank) == (0 until BankCount))
    require(kvShards.map(_.bank) == (0 until BankCount))
    require(usedBytes.size == BankCount && bankCapacities.size == BankCount)
    require(usedBytes.zip(bankCapacities).forall { case (used, capacity) =>
      used > 0 && used <= capacity
    })
    require(auxBases.zip(bankCapacities).forall { case (base, capacity) =>
      base % Alignment == 0 && base + AuxBytes <= capacity
    })
    require(ropeBases.zip(bankCapacities).forall { case (base, capacity) =>
      base % Alignment == 0 && base + RopeBytes <= capacity
    })
    require(FinalNormOffset + BigInt(cfg.hiddenSize) * 2 <= AuxBytes)
    require(InputNormOffset + BigInt(cfg.layerCount) * cfg.hiddenSize * 2 <=
      AuxBytes)
    require(PostNormOffset + BigInt(cfg.layerCount) * cfg.hiddenSize * 2 <=
      AuxBytes)
    require(AttentionSubNormOffset +
      BigInt(cfg.layerCount) * cfg.hiddenSize * 2 <= AuxBytes)
    require(FfnSubNormOffset + BigInt(cfg.layerCount) * cfg.ffnSize * 2 <=
      AuxBytes)
    require(ProjectionScaleOffset + BigInt(cfg.layerCount) * 14 == AuxBytes)
    require(BitNetAddressGeometry.RopeHeaderBytes +
      BigInt(cfg.maxSequenceLength) * (cfg.headSize / 2) * 4 == RopeBytes)

    val targets = WeightGeometry.map(_._1)
    require(weightShards.size == cfg.layerCount * targets.size * BankCount)
    for (layer <- 0 until cfg.layerCount; target <- targets) {
      val shards = (0 until BankCount).map(weight(target, layer, _))
      val expectedInput = inputFeatures(cfg, target)
      val expectedOutput = outputFeatures(cfg, target)
      require(shards.forall(_.outputFeatures == expectedOutput))
      require(shards.head.inputStart == 0)
      require(shards.sliding(2).forall {
        case Seq(left, right) =>
          left.inputStart + left.inputFeatures == right.inputStart
        case _ => true
      })
      require(shards.last.inputStart + shards.last.inputFeatures ==
        expectedInput)
      require(shards.map(_.bytes).sum == weightBytes(cfg, target))
      require(shards.forall(shard =>
        shard.bytes == BigInt(expectedOutput / 4) * shard.inputFeatures))
    }

    require(lmShards.head.inputStart == 0)
    require(lmShards.sliding(2).forall {
      case Seq(left, right) =>
        left.inputStart + left.inputFeatures == right.inputStart
      case _ => true
    })
    require(lmShards.last.inputStart + lmShards.last.inputFeatures ==
      cfg.hiddenSize)
    require(lmShards.forall(shard =>
      shard.weightBytes == BigInt(cfg.vocabularySize) * shard.inputFeatures &&
        shard.scaleBytes == BigInt(shard.scaleRows) * 4))
    require(lmShards.map(_.weightBytes).sum ==
      BigInt(cfg.vocabularySize) * cfg.hiddenSize)
    require(lmShards.head.scaleRowStart == 0)
    require(lmShards.sliding(2).forall {
      case Seq(left, right) =>
        left.scaleRowStart + left.scaleRows == right.scaleRowStart
      case _ => true
    })
    require(lmShards.last.scaleRowStart + lmShards.last.scaleRows ==
      cfg.vocabularySize)

    require(kvShards.zipWithIndex.forall { case (shard, bank) =>
      shard.bank == bank && shard.head == bank &&
        shard.packedBase % Alignment == 0 &&
        shard.packedBytes == BigInt(cfg.layerCount) * 2 *
          (cfg.maxSequenceLength / KvScalePackNum) * KvPackedBlockBytes
    })
    this
  }
}

object BitNetFiveBankStripedLayout {
  val BankCount: Int = 5
  val Alignment: BigInt = 128
  val TileFeatures: Int = 128
  val PlCapacity: BigInt = BigInt(1) << 30
  val PsCapacity: BigInt = BigInt(0x10000000L)
  // These offsets are part of the deployed AUX.BIN ABI.  They live here as
  // production-layout constants instead of inheriting the obsolete
  // output-row-sharded linker.
  val AuxHeaderBytes: BigInt = 64
  val AuxBytes: BigInt = 881124
  val RopeBytes: BigInt = 1048624
  val FinalNormOffset: BigInt = 64
  val InputNormOffset: BigInt = 5184
  val PostNormOffset: BigInt = 158784
  val AttentionSubNormOffset: BigInt = 312384
  val FfnSubNormOffset: BigInt = 465984
  val ProjectionScaleOffset: BigInt = 880704

  /** Four Q16.16 factors followed by four 128-byte i8 rows. */
  val KvScalePackNum: Int = 4
  val KvScaleChannelBytes: Int = KvScalePackNum * 4
  val KvPackedBlockBytes: Int =
    KvScaleChannelBytes + KvScalePackNum * BitNetConfig.Production.headSize

  val WeightGeometry: Vector[(BitNetAddressTarget.E, String)] = Vector(
    BitNetAddressTarget.QWeight -> "q",
    BitNetAddressTarget.KWeight -> "k",
    BitNetAddressTarget.VWeight -> "v",
    BitNetAddressTarget.OWeight -> "o",
    BitNetAddressTarget.GateUpWeight -> "gate_up",
    BitNetAddressTarget.DownWeight -> "down"
  )

  private def align(value: BigInt): BigInt =
    (value + Alignment - 1) & ~(Alignment - 1)

  /** Partition only whole 128-feature K tiles. */
  def partitionInput(features: Int, bank: Int): (Int, Int) = {
    require(features > 0 && features % TileFeatures == 0)
    require(bank >= 0 && bank < BankCount)
    val tiles = features / TileFeatures
    val quotient = tiles / BankCount
    val remainder = tiles % BankCount
    val tileCount = quotient + (if (bank < remainder) 1 else 0)
    val tileStart = bank * quotient + scala.math.min(bank, remainder)
    (tileStart * TileFeatures, tileCount * TileFeatures)
  }

  private def partitionRows(rows: Int, bank: Int): (Int, Int) = {
    val quotient = rows / BankCount
    val remainder = rows % BankCount
    val count = quotient + (if (bank < remainder) 1 else 0)
    val start = bank * quotient + scala.math.min(bank, remainder)
    (start, count)
  }

  def inputFeatures(
      cfg: BitNetConfig,
      target: BitNetAddressTarget.E): Int =
    if (target == BitNetAddressTarget.DownWeight) cfg.ffnSize
    else cfg.hiddenSize

  def outputFeatures(
      cfg: BitNetConfig,
      target: BitNetAddressTarget.E): Int = target match {
    case BitNetAddressTarget.KWeight | BitNetAddressTarget.VWeight =>
      cfg.kvHeadCount * cfg.headSize
    // This is a physical output space.  Each adjacent pair of complete
    // four-feature groups is Gate[n] followed by Up[n], exactly matching the
    // source_interleaved image ABI.  No full-FFN intermediate is allocated.
    case BitNetAddressTarget.GateUpWeight => 2 * cfg.ffnSize
    case BitNetAddressTarget.QWeight | BitNetAddressTarget.OWeight |
         BitNetAddressTarget.DownWeight => cfg.hiddenSize
    case _ => throw new IllegalArgumentException(
      s"target $target is not a Transformer weight")
  }

  def weightBytes(
      cfg: BitNetConfig,
      target: BitNetAddressTarget.E): BigInt =
    BigInt(inputFeatures(cfg, target)) * outputFeatures(cfg, target) / 4

  /** Physical group for Gate/Up semantic group `n` in G[n], U[n] order. */
  def gateUpPhysicalGroup(semanticGroup: Int, up: Boolean): Int = {
    require(semanticGroup >= 0)
    semanticGroup * 2 + (if (up) 1 else 0)
  }

  /** Semantic four-feature group carried by an interleaved physical group. */
  def gateUpSemanticGroup(physicalGroup: Int): Int = {
    require(physicalGroup >= 0)
    physicalGroup / 2
  }

  /** True only for the Up member of an interleaved Gate/Up group pair. */
  def gateUpIsUpGroup(physicalGroup: Int): Boolean = {
    require(physicalGroup >= 0)
    (physicalGroup & 1) != 0
  }

  def apply(
      cfg: BitNetConfig = BitNetConfig.Production):
      BitNetFiveBankStripedLayout = {
    require(cfg == BitNetConfig.Production)
    val cursor = Array.fill[BigInt](BankCount)(0)
    def allocate(bank: Int, bytes: BigInt): BigInt = {
      require(bytes > 0)
      val base = align(cursor(bank))
      cursor(bank) = base + bytes
      base
    }

    val weights = Vector.newBuilder[BitNetStripedWeightShard]
    for (layer <- 0 until cfg.layerCount;
         (target, _) <- WeightGeometry;
         bank <- 0 until BankCount) {
      val in = inputFeatures(cfg, target)
      val out = outputFeatures(cfg, target)
      val (inputStart, inputCount) = partitionInput(in, bank)
      val bytes = BigInt(out / 4) * inputCount
      weights += BitNetStripedWeightShard(
        target = target,
        layer = layer,
        bank = bank,
        localBase = allocate(bank, bytes),
        bytes = bytes,
        inputStart = inputStart,
        inputFeatures = inputCount,
        outputFeatures = out)
    }

    val aux = Vector.tabulate(BankCount)(bank => allocate(bank, AuxBytes))
    val rope = Vector.tabulate(BankCount)(bank => allocate(bank, RopeBytes))

    val lm = Vector.tabulate(BankCount) { bank =>
      val (inputStart, inputCount) = partitionInput(cfg.hiddenSize, bank)
      val weightBytes = BigInt(cfg.vocabularySize) * inputCount
      val weightBase = allocate(bank, weightBytes)
      val (scaleStart, scaleRows) = partitionRows(cfg.vocabularySize, bank)
      val scaleBytes = BigInt(scaleRows) * 4
      val scaleBase = allocate(bank, scaleBytes)
      BitNetStripedLmShard(
        bank, inputStart, inputCount, weightBase, weightBytes,
        scaleStart, scaleRows, scaleBase, scaleBytes)
    }

    require(cfg.maxSequenceLength % KvScalePackNum == 0)
    val blocksPerKind = cfg.maxSequenceLength / KvScalePackNum
    val packedBytes = BigInt(cfg.layerCount) * 2 * blocksPerKind *
      KvPackedBlockBytes
    val kv = Vector.tabulate(BankCount) { bank =>
      BitNetStripedKvShard(
        bank = bank,
        head = bank,
        packedBase = allocate(bank, packedBytes),
        packedBytes = packedBytes)
    }

    val used = cursor.map(align).toVector
    val capacities = Vector(PlCapacity) ++
      Vector.fill(BankCount - 1)(PsCapacity)
    BitNetFiveBankStripedLayout(
      cfg, weights.result(), aux, rope, lm, kv, used, capacities).validate()
  }
}
