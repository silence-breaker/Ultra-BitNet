package ultrabitnet.accel

import spinal.core._

/** The six addressable lifetime regions in the unified resident tensor RAM. */
object BitNetUnifiedResidentTensorRegion
    extends SpinalEnum(binarySequential) {
  val Hidden, Narrow, Wide, ProjectionSlot0, ProjectionSlot1, QueryLmAlias =
    newElement()
}

/** Q and LM never own the aliased tail region at the same time. */
object BitNetUnifiedResidentTensorAliasOwner
    extends SpinalEnum(binarySequential) {
  val Query, LmHead = newElement()
}

/** A physical resident word has one of two deliberately non-interchangeable
  * interpretations.  This is a Scala layout property rather than a hardware
  * tag stored beside every word.
  */
sealed trait BitNetUnifiedResidentTensorWordFormat
object BitNetUnifiedResidentTensorWordFormat {
  /** One signed 32-bit value per physical token lane. */
  case object ActivationS32x4
      extends BitNetUnifiedResidentTensorWordFormat
  /** Four adjacent features x the physical token lanes, all signed i8. */
  case object PackedI8x16
      extends BitNetUnifiedResidentTensorWordFormat
}

case class BitNetUnifiedResidentTensorRegionDescriptor(
    region: BitNetUnifiedResidentTensorRegion.E,
    name: String,
    baseRow: Int,
    rowCount: Int,
    format: BitNetUnifiedResidentTensorWordFormat) {
  require(baseRow >= 0)
  require(rowCount > 0)

  val lastRow: Int = baseRow + rowCount - 1
  def containsAbsoluteRow(row: Int): Boolean =
    row >= baseRow && row <= lastRow
}

case class BitNetUnifiedResidentTensorWordLocation(
    bank: Int,
    absoluteRow: Int,
    elementInWord: Int,
    bitOffset: Int)

/**
 * Fixed production layout for 32 logical banks of TP-scaled x 512 storage.
  *
 * S32 regions use one bank word for all token values of one feature:
  * `bank = feature % 32`, `row = base + feature / 32`.  Packed-i8 regions use
  * the layout already expected by the resident TP4 feeder: one word contains
  * four adjacent features x the physical tokens, and 32 banks form a
  * 128-feature tile.
  * The final eight rows have no semantic region and are therefore impossible
  * to address through the typed scratchpad ports.
  */
case class BitNetUnifiedResidentTensorScratchpadLayout(
    bankCount: Int = 32,
    wordWidth: Int = BitNetConfig.Production.tokenParallelism * 32,
    rowCount: Int = 512,
    tokenLanes: Int = BitNetConfig.Production.tokenParallelism,
    activationWidth: Int = 32,
    packedWidth: Int = 8,
    packedFeaturesPerWord: Int = 4
) {
  import BitNetUnifiedResidentTensorRegion._
  import BitNetUnifiedResidentTensorWordFormat._

  require(bankCount == 32)
  require(wordWidth == tokenLanes * 32)
  require(rowCount == 512)
  require(tokenLanes == 2 || tokenLanes == 4)
  require(activationWidth == 32)
  require(packedWidth == 8)
  require(packedFeaturesPerWord == 4)
  require(tokenLanes * activationWidth == wordWidth)
  require(tokenLanes * packedFeaturesPerWord * packedWidth == wordWidth)

  val hidden = BitNetUnifiedResidentTensorRegionDescriptor(
    Hidden, "Hidden", 0, 80, ActivationS32x4)
  val narrow = BitNetUnifiedResidentTensorRegionDescriptor(
    Narrow, "Narrow", 80, 80, ActivationS32x4)
  val wide = BitNetUnifiedResidentTensorRegionDescriptor(
    Wide, "Wide", 160, 216, ActivationS32x4)
  val projectionSlot0 = BitNetUnifiedResidentTensorRegionDescriptor(
    ProjectionSlot0, "ProjectionSlot0", 376, 54, PackedI8x16)
  val projectionSlot1 = BitNetUnifiedResidentTensorRegionDescriptor(
    ProjectionSlot1, "ProjectionSlot1", 430, 54, PackedI8x16)
  val queryLmAlias = BitNetUnifiedResidentTensorRegionDescriptor(
    QueryLmAlias, "QueryLmAlias", 484, 20, PackedI8x16)

  val regions: Vector[BitNetUnifiedResidentTensorRegionDescriptor] = Vector(
    hidden, narrow, wide, projectionSlot0, projectionSlot1, queryLmAlias)
  val activationRegions: Vector[BitNetUnifiedResidentTensorRegionDescriptor] =
    regions.filter(_.format == ActivationS32x4)
  val packedRegions: Vector[BitNetUnifiedResidentTensorRegionDescriptor] =
    regions.filter(_.format == PackedI8x16)

  val guardBaseRow: Int = 504
  val guardRowCount: Int = 8
  val guardLastRow: Int = guardBaseRow + guardRowCount - 1
  val addressableRowCount: Int = guardBaseRow
  val rowAddressWidth: Int = log2Up(rowCount)
  val bankWidth: Int = log2Up(bankCount)
  val localRowWidth: Int = log2Up(regions.map(_.rowCount).max)
  val aliasGenerationWidth: Int = 16
  val nativeSymbolWidth: Int = 32
  val nativeSymbolCount: Int = wordWidth / nativeSymbolWidth
  val packedFeaturesPerSymbol: Int =
    packedFeaturesPerWord / nativeSymbolCount
  val packedTransportSegmentCount: Int = tokenLanes
  val packedTransportWordsPerSegment: Int =
    bankCount / packedTransportSegmentCount
  val packedTransportSegmentWidth: Int =
    packedTransportWordsPerSegment * wordWidth

  require(bankCount % packedTransportSegmentCount == 0)
  require(nativeSymbolCount == tokenLanes)
  require(packedFeaturesPerWord % nativeSymbolCount == 0)
  require(packedTransportSegmentWidth == 1024)

  require(regions.map(_.region).distinct.size == regions.size)
  require(regions.head.baseRow == 0)
  require(regions.sliding(2).forall {
    case Vector(left, right) => left.lastRow + 1 == right.baseRow
    case _ => true
  })
  require(regions.last.lastRow + 1 == guardBaseRow)
  require(guardLastRow == rowCount - 1)

  private val descriptorByRegion = regions.map(d => d.region -> d).toMap

  def descriptor(region: BitNetUnifiedResidentTensorRegion.E):
      BitNetUnifiedResidentTensorRegionDescriptor =
    descriptorByRegion(region)

  def regionAtAbsoluteRow(row: Int):
      Option[BitNetUnifiedResidentTensorRegionDescriptor] = {
    require(row >= 0 && row < rowCount)
    regions.find(_.containsAbsoluteRow(row))
  }

  def absoluteRow(
      region: BitNetUnifiedResidentTensorRegion.E,
      localRow: Int): Int = {
    val d = descriptor(region)
    require(localRow >= 0 && localRow < d.rowCount,
      s"local row $localRow is outside ${d.name}[0:${d.rowCount - 1}]")
    d.baseRow + localRow
  }

  def activationFeatureCapacity(
      region: BitNetUnifiedResidentTensorRegion.E): Int = {
    val d = descriptor(region)
    require(d.format == ActivationS32x4,
      s"${d.name} is not an SInt32 activation region")
    d.rowCount * bankCount
  }

  def packedFeatureCapacity(
      region: BitNetUnifiedResidentTensorRegion.E): Int = {
    val d = descriptor(region)
    require(d.format == PackedI8x16,
      s"${d.name} is not a packed-i8 region")
    d.rowCount * bankCount * packedFeaturesPerWord
  }

  def activationLocation(
      region: BitNetUnifiedResidentTensorRegion.E,
      feature: Int,
      tokenLane: Int): BitNetUnifiedResidentTensorWordLocation = {
    require(tokenLane >= 0 && tokenLane < tokenLanes)
    require(feature >= 0 && feature < activationFeatureCapacity(region))
    BitNetUnifiedResidentTensorWordLocation(
      bank = feature % bankCount,
      absoluteRow = absoluteRow(region, feature / bankCount),
      elementInWord = tokenLane,
      bitOffset = tokenLane * activationWidth)
  }

  def packedI8Location(
      region: BitNetUnifiedResidentTensorRegion.E,
      feature: Int,
      tokenLane: Int): BitNetUnifiedResidentTensorWordLocation = {
    require(tokenLane >= 0 && tokenLane < tokenLanes)
    require(feature >= 0 && feature < packedFeatureCapacity(region))
    val featureWord = feature / packedFeaturesPerWord
    val featureInWord = feature % packedFeaturesPerWord
    val element = featureInWord * tokenLanes + tokenLane
    BitNetUnifiedResidentTensorWordLocation(
      bank = featureWord % bankCount,
      absoluteRow = absoluteRow(region, featureWord / bankCount),
      elementInWord = element,
      bitOffset = element * packedWidth)
  }
}
