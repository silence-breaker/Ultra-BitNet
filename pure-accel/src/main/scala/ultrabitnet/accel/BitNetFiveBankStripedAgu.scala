package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetFiveBankStripedAguFault extends SpinalEnum(binarySequential) {
  val None, UnknownTarget, InvalidLayer, InvalidGroup, InvalidLmRow,
      InvalidPosition,
      BankRange = newElement()
}

case class BitNetFiveBankStripedAguConfig(
    model: BitNetConfig = BitNetConfig.Production,
    layout: BitNetFiveBankStripedLayout =
      BitNetFiveBankStripedLayout()
) {
  require(model == BitNetConfig.Production)
  require(layout.cfg == model)
  layout.validate()

  val bankCount: Int = BitNetFiveBankStripedLayout.BankCount
  val bankWidth: Int = log2Up(bankCount)
  // GateUp is a physical G0,U0,G1,U1... stream and therefore has twice the
  // FFN output groups of either legacy matrix on its own.
  val maxOutputGroups: Int = scala.math.max(
    model.hiddenSize / 4, (2 * model.ffnSize) / 4)
  val outputGroupWidth: Int = log2Up(maxOutputGroups)
  val inputFeatureWidth: Int = log2Up(model.ffnSize + 1)
  val outputFeatureWidth: Int =
    BitNetAddressGeometry.lmRowIndexWidth(model)
  val lmRowWidth: Int = BitNetAddressGeometry.lmRowIndexWidth(model)
}

case class BitNetStripedTransformerGroupRequest(
    cfg: BitNetFiveBankStripedAguConfig) extends Bundle {
  val target = BitNetAddressTarget()
  val layer = UInt(cfg.model.layerWidth bits)
  val outputGroup = UInt(cfg.outputGroupWidth bits)
}

case class BitNetStripedTransformerGroupResponse(
    cfg: BitNetFiveBankStripedAguConfig) extends Bundle {
  val target = BitNetAddressTarget()
  val layer = UInt(cfg.model.layerWidth bits)
  val outputGroup = UInt(cfg.outputGroupWidth bits)
  val outputBase = UInt(cfg.outputFeatureWidth bits)
  val localAddress = Vec(UInt(cfg.model.addressWidth bits), cfg.bankCount)
  val inputStart = Vec(UInt(cfg.inputFeatureWidth bits), cfg.bankCount)
  val inputFeatures = Vec(UInt(cfg.inputFeatureWidth bits), cfg.bankCount)
  val ok = Bool()
  val fault = BitNetFiveBankStripedAguFault()
}

case class BitNetStripedLmRowRequest(
    cfg: BitNetFiveBankStripedAguConfig) extends Bundle {
  val row = UInt(cfg.lmRowWidth bits)
}

case class BitNetStripedLmRowResponse(
    cfg: BitNetFiveBankStripedAguConfig) extends Bundle {
  val row = UInt(cfg.lmRowWidth bits)
  val weightAddress = Vec(UInt(cfg.model.addressWidth bits), cfg.bankCount)
  val inputStart = Vec(UInt(cfg.inputFeatureWidth bits), cfg.bankCount)
  val inputFeatures = Vec(UInt(cfg.inputFeatureWidth bits), cfg.bankCount)
  val scaleBank = UInt(cfg.bankWidth bits)
  val scaleAddress = UInt(cfg.model.addressWidth bits)
  val ok = Bool()
  val fault = BitNetFiveBankStripedAguFault()
}

object BitNetFiveBankStripedAguOracle {
  case class TransformerGroup(
      outputBase: Int,
      addresses: Vector[BigInt],
      inputStarts: Vector[Int],
      inputFeatures: Vector[Int])

  case class LmRow(
      addresses: Vector[BigInt],
      inputStarts: Vector[Int],
      inputFeatures: Vector[Int],
      scaleBank: Int,
      scaleAddress: BigInt)

  def transformer(
      layout: BitNetFiveBankStripedLayout,
      target: BitNetAddressTarget.E,
      layer: Int,
      outputGroup: Int): TransformerGroup = {
    val shards = (0 until layout.bankCapacities.size).map(
      layout.weight(target, layer, _)).toVector
    require(outputGroup >= 0 && outputGroup < shards.head.outputGroups)
    TransformerGroup(
      outputBase = outputGroup * 4,
      addresses = shards.map(_.groupLocalBase(outputGroup)),
      inputStarts = shards.map(_.inputStart),
      inputFeatures = shards.map(_.inputFeatures))
  }

  def lm(
      layout: BitNetFiveBankStripedLayout,
      row: Int): LmRow = {
    require(row >= 0 && row < layout.cfg.vocabularySize)
    val owner = layout.lmShards.indexWhere(shard =>
      row >= shard.scaleRowStart && row < shard.scaleRowStart + shard.scaleRows)
    require(owner >= 0)
    val scale = layout.lmShards(owner)
    LmRow(
      addresses = layout.lmShards.map(shard =>
        shard.weightBase + BigInt(row) * shard.inputFeatures),
      inputStarts = layout.lmShards.map(_.inputStart),
      inputFeatures = layout.lmShards.map(_.inputFeatures),
      scaleBank = owner,
      scaleAddress = scale.scaleBase + BigInt(row - scale.scaleRowStart) * 4)
  }
}

/**
  * One-entry elastic AGU for a K-striped Transformer group.
  *
  * A response names all five bank-local slices of the same output group.  It
  * never emits a partially valid address vector: an unknown target, invalid
  * layer/group, arithmetic overflow or bank-window escape returns one typed
  * failed response for the fixed controller to retire fail-closed.
  */
class BitNetFiveBankStripedTransformerAgu(
    cfg: BitNetFiveBankStripedAguConfig =
      BitNetFiveBankStripedAguConfig()) extends Component {
  private val model = cfg.model
  private val bankCount = cfg.bankCount
  private val tableEntries = model.layerCount * bankCount
  private val tableIndexWidth = log2Up(tableEntries)
  private val calculationWidth = model.addressWidth + 1
  private val targets = BitNetFiveBankStripedLayout.WeightGeometry.map(_._1)

  val io = new Bundle {
    val request = slave(Stream(BitNetStripedTransformerGroupRequest(cfg)))
    val response = master(Stream(BitNetStripedTransformerGroupResponse(cfg)))
  }

  private def tableFor(
      target: BitNetAddressTarget.E,
      select: BitNetStripedWeightShard => BigInt): Vec[UInt] =
    Vec((for (layer <- 0 until model.layerCount; bank <- 0 until bankCount)
      yield U(select(cfg.layout.weight(target, layer, bank)),
        model.addressWidth bits)))

  private def intTableFor(
      target: BitNetAddressTarget.E,
      select: BitNetStripedWeightShard => Int): Vec[UInt] =
    Vec((for (layer <- 0 until model.layerCount; bank <- 0 until bankCount)
      yield U(select(cfg.layout.weight(target, layer, bank)),
        cfg.inputFeatureWidth bits)))

  val baseTables = targets.map(target => target ->
    tableFor(target, _.localBase)).toMap
  val startTables = targets.map(target => target ->
    intTableFor(target, _.inputStart)).toMap
  val strideTables = targets.map(target => target ->
    intTableFor(target, _.inputFeatures)).toMap

  val layerLegal = io.request.layer < model.layerCount
  val safeLayer = UInt(model.layerWidth bits)
  safeLayer := Mux(layerLegal, io.request.layer, U(0, model.layerWidth bits))
  val targetKnown = targets.map(io.request.target === _).reduce(_ || _)

  val selectedGroups = UInt((cfg.outputGroupWidth + 1) bits)
  selectedGroups := 0
  for (target <- targets) {
    when(io.request.target === target) {
      selectedGroups := U(
        BitNetFiveBankStripedLayout.outputFeatures(model, target) / 4,
        selectedGroups.getWidth bits)
    }
  }
  val groupLegal = io.request.outputGroup.resize(selectedGroups.getWidth) <
    selectedGroups

  val selectedBase = Vec(UInt(model.addressWidth bits), bankCount)
  val selectedStart = Vec(UInt(cfg.inputFeatureWidth bits), bankCount)
  val selectedStride = Vec(UInt(cfg.inputFeatureWidth bits), bankCount)
  val bankRangeLegal = Vec(Bool(), bankCount)
  val calculatedAddress = Vec(UInt(model.addressWidth bits), bankCount)
  for (bank <- 0 until bankCount) {
    val index = (safeLayer.resize(tableIndexWidth) * bankCount + bank)
      .resize(tableIndexWidth)
    selectedBase(bank) := 0
    selectedStart(bank) := 0
    selectedStride(bank) := 0
    for (target <- targets) {
      when(io.request.target === target) {
        selectedBase(bank) := baseTables(target)(index)
        selectedStart(bank) := startTables(target)(index)
        selectedStride(bank) := strideTables(target)(index)
      }
    }
    val offset = (io.request.outputGroup.resize(calculationWidth) *
      selectedStride(bank).resize(calculationWidth)).resize(calculationWidth)
    val start = (selectedBase(bank).resize(calculationWidth) + offset)
      .resize(calculationWidth)
    val end = (start + selectedStride(bank).resize(calculationWidth))
      .resize(calculationWidth)
    calculatedAddress(bank) := start.resize(model.addressWidth)
    bankRangeLegal(bank) := !start.msb && !end.msb &&
      end <= U(cfg.layout.bankCapacities(bank), calculationWidth bits) &&
      start(6 downto 0) === 0
  }
  val allBanksInRange = bankRangeLegal.asBits.andR
  val requestLegal = targetKnown && layerLegal && groupLegal && allBanksInRange

  val responseValid = RegInit(False)
  val responsePayload = Reg(BitNetStripedTransformerGroupResponse(cfg)) init
    BitNetStripedTransformerGroupResponse(cfg).getZero
  io.response.valid := responseValid
  io.response.payload := responsePayload
  io.request.ready := !responseValid || io.response.ready

  when(io.request.fire) {
    responseValid := True
    responsePayload.target := io.request.target
    responsePayload.layer := io.request.layer
    responsePayload.outputGroup := io.request.outputGroup
    responsePayload.outputBase :=
      (io.request.outputGroup.resize(cfg.outputFeatureWidth) << 2).resized
    responsePayload.localAddress := calculatedAddress
    responsePayload.inputStart := selectedStart
    responsePayload.inputFeatures := selectedStride
    responsePayload.ok := requestLegal
    responsePayload.fault := BitNetFiveBankStripedAguFault.None
    when(!targetKnown) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.UnknownTarget
    } elsewhen (!layerLegal) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.InvalidLayer
    } elsewhen (!groupLegal) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.InvalidGroup
    } elsewhen (!allBanksInRange) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.BankRange
    }
  } elsewhen (io.response.fire) {
    responseValid := False
  }
}

/** One-entry elastic AGU for a five-bank K-striped LM row and its scale. */
class BitNetFiveBankStripedLmAgu(
    cfg: BitNetFiveBankStripedAguConfig =
      BitNetFiveBankStripedAguConfig()) extends Component {
  private val model = cfg.model
  private val calculationWidth = model.addressWidth + 1

  val io = new Bundle {
    val request = slave(Stream(BitNetStripedLmRowRequest(cfg)))
    val response = master(Stream(BitNetStripedLmRowResponse(cfg)))
  }

  val rowLegal = io.request.row < model.vocabularySize
  val weightAddress = Vec(UInt(model.addressWidth bits), cfg.bankCount)
  val inputStart = Vec(UInt(cfg.inputFeatureWidth bits), cfg.bankCount)
  val inputFeatures = Vec(UInt(cfg.inputFeatureWidth bits), cfg.bankCount)
  val bankLegal = Vec(Bool(), cfg.bankCount)
  for (bank <- 0 until cfg.bankCount) {
    val shard = cfg.layout.lmShards(bank)
    inputStart(bank) := U(shard.inputStart, cfg.inputFeatureWidth bits)
    inputFeatures(bank) := U(shard.inputFeatures, cfg.inputFeatureWidth bits)
    val offset = (io.request.row.resize(calculationWidth) *
      shard.inputFeatures).resize(calculationWidth)
    val start = (U(shard.weightBase, calculationWidth bits) + offset)
      .resize(calculationWidth)
    val end = (start + shard.inputFeatures).resize(calculationWidth)
    weightAddress(bank) := start.resize(model.addressWidth)
    bankLegal(bank) := !start.msb && !end.msb &&
      end <= U(cfg.layout.bankCapacities(bank), calculationWidth bits) &&
      start(6 downto 0) === 0
  }

  val scaleOwner = UInt(cfg.bankWidth bits)
  val scaleAddress = UInt(model.addressWidth bits)
  val scaleFound = Bool()
  scaleOwner := 0
  scaleAddress := 0
  scaleFound := False
  for (bank <- 0 until cfg.bankCount) {
    val shard = cfg.layout.lmShards(bank)
    when(io.request.row >= shard.scaleRowStart &&
        io.request.row < shard.scaleRowStart + shard.scaleRows) {
      scaleOwner := bank
      scaleAddress := (U(shard.scaleBase, model.addressWidth bits) +
        ((io.request.row - shard.scaleRowStart).resize(model.addressWidth) << 2))
        .resize(model.addressWidth)
      scaleFound := True
    }
  }
  val requestLegal = rowLegal && bankLegal.asBits.andR && scaleFound

  val responseValid = RegInit(False)
  val responsePayload = Reg(BitNetStripedLmRowResponse(cfg)) init
    BitNetStripedLmRowResponse(cfg).getZero
  io.response.valid := responseValid
  io.response.payload := responsePayload
  io.request.ready := !responseValid || io.response.ready

  when(io.request.fire) {
    responseValid := True
    responsePayload.row := io.request.row
    responsePayload.weightAddress := weightAddress
    responsePayload.inputStart := inputStart
    responsePayload.inputFeatures := inputFeatures
    responsePayload.scaleBank := scaleOwner
    responsePayload.scaleAddress := scaleAddress
    responsePayload.ok := requestLegal
    responsePayload.fault := BitNetFiveBankStripedAguFault.None
    when(!rowLegal) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.InvalidLmRow
    } elsewhen (!bankLegal.asBits.andR || !scaleFound) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.BankRange
    }
  } elsewhen (io.response.fire) {
    responseValid := False
  }
}

object GenerateBitNetFiveBankStripedAgus extends App {
  SpinalConfig(
    targetDirectory = "build/bitnet-five-bank-striped-agus",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetFiveBankStripedTransformerAgu())
  SpinalConfig(
    targetDirectory = "build/bitnet-five-bank-striped-lm-agu",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetFiveBankStripedLmAgu())
}
