package ultrabitnet.accel

import spinal.core.log2Up

/**
  * Static shape of the BitNet BitNet datapath.
  *
  * This configuration deliberately describes a fixed model/dataflow rather
  * than a generic GEMV packet processor.  The controller can therefore emit
  * short, typed commands while the address generators and compute engines are
  * specialised for the deployed model at elaboration time.
  */
case class BitNetConfig(
    layerCount: Int = 30,
    hiddenSize: Int = 2560,
    ffnSize: Int = 6912,
    vocabularySize: Int = 128256,
    queryHeadCount: Int = 20,
    kvHeadCount: Int = 5,
    headSize: Int = 128,
    maxSequenceLength: Int = 4096,
    maxPrefillTokens: Int = 4096,
    tokenParallelism: Int = 4,
    engineCount: Int = 2,
    physicalMatrixEngineCount: Int = 2,
    spuCount: Int = 4,
    addressWidth: Int = 34,
    requestIdWidth: Int = 32
) {
  require(layerCount > 0, "layerCount must be positive")
  require(hiddenSize > 0 && ffnSize > 0 && vocabularySize > 0,
    "model dimensions must be positive")
  require(queryHeadCount > 0 && kvHeadCount > 0 && headSize > 0,
    "attention dimensions must be positive")
  require(queryHeadCount * headSize == hiddenSize,
    "queryHeadCount * headSize must equal hiddenSize")
  require(queryHeadCount % kvHeadCount == 0,
    "queryHeadCount must be an integer multiple of kvHeadCount for GQA")
  require(maxSequenceLength > 1, "maxSequenceLength must exceed one token")
  require(maxPrefillTokens > 0 && maxPrefillTokens <= maxSequenceLength,
    "maxPrefillTokens must be within the sequence capacity")
  require(tokenParallelism > 0 && (tokenParallelism & (tokenParallelism - 1)) == 0,
    "tokenParallelism must be a power of two")
  require(engineCount > 0, "engineCount must be positive")
  require(physicalMatrixEngineCount > 0 &&
    physicalMatrixEngineCount <= engineCount,
    "physicalMatrixEngineCount must be within the logical engine count")
  require(spuCount > 0, "spuCount must be positive")
  require(addressWidth >= 32, "addressWidth must cover the resident DDR map")
  require(requestIdWidth >= 8, "requestIdWidth is too small for safe completion matching")

  val sequenceWidth: Int = log2Up(maxSequenceLength + 1)
  val prefillWidth: Int = log2Up(maxPrefillTokens + 1)
  val layerWidth: Int = math.max(1, log2Up(layerCount))
  val roundTokenCountWidth: Int = log2Up(tokenParallelism + 1)
  val gqaGroupSize: Int = queryHeadCount / kvHeadCount
}

object BitNetConfig {
  /**
    * AXU3EG production baseline: two token lanes, one physical matrix engine
    * and four SPUs. Board implementation uses related 50/100 MHz clocks.
    */
  val Production: BitNetConfig = BitNetConfig(
    tokenParallelism = 2,
    engineCount = 1,
    physicalMatrixEngineCount = 1,
    spuCount = 4)

  /** Eight token lanes/four engines for exploration after synthesis feedback. */
  val Aggressive: BitNetConfig =
    BitNetConfig(
      tokenParallelism = 8,
      engineCount = 4,
      physicalMatrixEngineCount = 4,
      spuCount = 8)
}
