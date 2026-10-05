package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetFiveBankStripedAuxAguConfig(
    model: BitNetConfig = BitNetConfig.Production,
    layout: BitNetFiveBankStripedLayout =
      BitNetFiveBankStripedLayout()
) {
  require(model == BitNetConfig.Production)
  require(layout.cfg == model)
  layout.validate()

  val bankCount: Int = BitNetFiveBankStripedLayout.BankCount
}

/**
  * One replicated-constant request.  All addresses are bank-local: the board
  * adapters add either the MIG window or the corresponding PS high-DDR base.
  */
case class BitNetStripedAuxRequest(
    cfg: BitNetFiveBankStripedAuxAguConfig) extends Bundle {
  val target = BitNetAddressTarget()
  val layer = UInt(cfg.model.layerWidth bits)
  val position = UInt(cfg.model.sequenceWidth bits)
}

case class BitNetStripedAuxResponse(
    cfg: BitNetFiveBankStripedAuxAguConfig) extends Bundle {
  val target = BitNetAddressTarget()
  val layer = UInt(cfg.model.layerWidth bits)
  val position = UInt(cfg.model.sequenceWidth bits)
  val localAddress = Vec(UInt(cfg.model.addressWidth bits), cfg.bankCount)
  val transferBytes = UInt(cfg.model.addressWidth bits)
  val requiredAlignmentBytes = UInt(6 bits)
  val ok = Bool()
  val fault = BitNetFiveBankStripedAguFault()
}

object BitNetFiveBankStripedAuxAguOracle {
  final case class Range(addresses: Vector[BigInt], bytes: BigInt,
                         alignment: Int)

  private def layerRange(
      layout: BitNetFiveBankStripedLayout,
      offset: BigInt,
      stride: BigInt,
      layer: Int,
      bytes: BigInt): Range = {
    require(layer >= 0 && layer < layout.cfg.layerCount)
    Range(layout.auxBases.map(_ + offset + BigInt(layer) * stride),
      bytes, 2)
  }

  def apply(
      layout: BitNetFiveBankStripedLayout,
      target: BitNetAddressTarget.E,
      layer: Int = 0,
      position: Int = 0): Range = {
    val cfg = layout.cfg
    import BitNetFiveBankStripedLayout._
    target match {
      case BitNetAddressTarget.InputNorm =>
        layerRange(layout, InputNormOffset,
          BigInt(cfg.hiddenSize) * 2, layer,
          BigInt(cfg.hiddenSize) * 2)
      case BitNetAddressTarget.PostAttentionNorm =>
        layerRange(layout, PostNormOffset,
          BigInt(cfg.hiddenSize) * 2, layer,
          BigInt(cfg.hiddenSize) * 2)
      case BitNetAddressTarget.AttentionSubNorm =>
        layerRange(layout, AttentionSubNormOffset,
          BigInt(cfg.hiddenSize) * 2, layer,
          BigInt(cfg.hiddenSize) * 2)
      case BitNetAddressTarget.FfnSubNorm =>
        layerRange(layout, FfnSubNormOffset,
          BigInt(cfg.ffnSize) * 2, layer,
          BigInt(cfg.ffnSize) * 2)
      case BitNetAddressTarget.FinalNorm =>
        Range(layout.auxBases.map(_ + FinalNormOffset),
          BigInt(cfg.hiddenSize) * 2, 2)
      case BitNetAddressTarget.QProjectionScale |
           BitNetAddressTarget.KProjectionScale |
           BitNetAddressTarget.VProjectionScale |
           BitNetAddressTarget.OProjectionScale |
           BitNetAddressTarget.GateProjectionScale |
           BitNetAddressTarget.UpProjectionScale |
           BitNetAddressTarget.DownProjectionScale =>
        val ordinal = Vector(
          BitNetAddressTarget.QProjectionScale,
          BitNetAddressTarget.KProjectionScale,
          BitNetAddressTarget.VProjectionScale,
          BitNetAddressTarget.OProjectionScale,
          BitNetAddressTarget.GateProjectionScale,
          BitNetAddressTarget.UpProjectionScale,
          BitNetAddressTarget.DownProjectionScale).indexOf(target)
        require(ordinal >= 0)
        layerRange(layout, ProjectionScaleOffset + ordinal * 2,
          14, layer, 2)
      case BitNetAddressTarget.RopeRow =>
        require(position >= 0 && position < cfg.maxSequenceLength)
        val rowBytes = BigInt(cfg.headSize / 2) * 4
        Range(layout.ropeBases.map(_ + BitNetAddressGeometry.RopeHeaderBytes +
          BigInt(position) * rowBytes), rowBytes, 2)
      case _ => throw new IllegalArgumentException(
        s"target $target is not a replicated AUX/RoPE object")
    }
  }
}

/** Elastic five-bank AGU for replicated AUX constants and RoPE rows. */
class BitNetFiveBankStripedAuxAgu(
    cfg: BitNetFiveBankStripedAuxAguConfig =
      BitNetFiveBankStripedAuxAguConfig()) extends Component {
  private val model = cfg.model
  private val layout = cfg.layout
  private val calculationWidth = model.addressWidth + 1

  val io = new Bundle {
    val request = slave(Stream(BitNetStripedAuxRequest(cfg)))
    val response = master(Stream(BitNetStripedAuxResponse(cfg)))
  }

  val layerTargets = Vector(
    BitNetAddressTarget.InputNorm,
    BitNetAddressTarget.PostAttentionNorm,
    BitNetAddressTarget.AttentionSubNorm,
    BitNetAddressTarget.FfnSubNorm,
    BitNetAddressTarget.QProjectionScale,
    BitNetAddressTarget.KProjectionScale,
    BitNetAddressTarget.VProjectionScale,
    BitNetAddressTarget.OProjectionScale,
    BitNetAddressTarget.GateProjectionScale,
    BitNetAddressTarget.UpProjectionScale,
    BitNetAddressTarget.DownProjectionScale)
  val allTargets = layerTargets ++ Vector(
    BitNetAddressTarget.FinalNorm, BitNetAddressTarget.RopeRow)

  val targetKnown = allTargets.map(io.request.target === _).reduce(_ || _)
  val requiresLayer = layerTargets.map(io.request.target === _).reduce(_ || _)
  val requiresPosition = io.request.target === BitNetAddressTarget.RopeRow
  val layerLegal = !requiresLayer || io.request.layer < model.layerCount
  val positionLegal = !requiresPosition ||
    io.request.position < model.maxSequenceLength
  val safeLayer = Mux(io.request.layer < model.layerCount,
    io.request.layer, U(0, model.layerWidth bits))
  val safePosition = Mux(io.request.position < model.maxSequenceLength,
    io.request.position, U(0, model.sequenceWidth bits))

  val offset = UInt(model.addressWidth bits)
  val transferBytes = UInt(model.addressWidth bits)
  val alignment = UInt(6 bits)
  offset := 0
  transferBytes := 0
  alignment := 2

  def selectLayerRange(baseOffset: BigInt, stride: BigInt,
                       bytes: BigInt): Unit = {
    offset := (U(baseOffset, model.addressWidth bits) +
      safeLayer.resize(model.addressWidth) * stride).resized
    transferBytes := U(bytes, model.addressWidth bits)
  }

  switch(io.request.target) {
    is(BitNetAddressTarget.InputNorm) {
      selectLayerRange(BitNetFiveBankStripedLayout.InputNormOffset,
        BigInt(model.hiddenSize) * 2, BigInt(model.hiddenSize) * 2)
    }
    is(BitNetAddressTarget.PostAttentionNorm) {
      selectLayerRange(BitNetFiveBankStripedLayout.PostNormOffset,
        BigInt(model.hiddenSize) * 2, BigInt(model.hiddenSize) * 2)
    }
    is(BitNetAddressTarget.AttentionSubNorm) {
      selectLayerRange(BitNetFiveBankStripedLayout.AttentionSubNormOffset,
        BigInt(model.hiddenSize) * 2, BigInt(model.hiddenSize) * 2)
    }
    is(BitNetAddressTarget.FfnSubNorm) {
      selectLayerRange(BitNetFiveBankStripedLayout.FfnSubNormOffset,
        BigInt(model.ffnSize) * 2, BigInt(model.ffnSize) * 2)
    }
    is(BitNetAddressTarget.FinalNorm) {
      offset := U(BitNetFiveBankStripedLayout.FinalNormOffset,
        model.addressWidth bits)
      transferBytes := U(BigInt(model.hiddenSize) * 2,
        model.addressWidth bits)
    }
    is(BitNetAddressTarget.QProjectionScale,
       BitNetAddressTarget.KProjectionScale,
       BitNetAddressTarget.VProjectionScale,
       BitNetAddressTarget.OProjectionScale,
       BitNetAddressTarget.GateProjectionScale,
       BitNetAddressTarget.UpProjectionScale,
       BitNetAddressTarget.DownProjectionScale) {
      val scaleOrdinal = UInt(3 bits)
      scaleOrdinal := 0
      when(io.request.target === BitNetAddressTarget.KProjectionScale) {
        scaleOrdinal := 1
      } elsewhen(io.request.target === BitNetAddressTarget.VProjectionScale) {
        scaleOrdinal := 2
      } elsewhen(io.request.target === BitNetAddressTarget.OProjectionScale) {
        scaleOrdinal := 3
      } elsewhen(io.request.target === BitNetAddressTarget.GateProjectionScale) {
        scaleOrdinal := 4
      } elsewhen(io.request.target === BitNetAddressTarget.UpProjectionScale) {
        scaleOrdinal := 5
      } elsewhen(io.request.target === BitNetAddressTarget.DownProjectionScale) {
        scaleOrdinal := 6
      }
      offset := (U(BitNetFiveBankStripedLayout.ProjectionScaleOffset,
        model.addressWidth bits) +
        safeLayer.resize(model.addressWidth) * 14 +
        (scaleOrdinal.resize(model.addressWidth) << 1)).resized
      transferBytes := 2
    }
    is(BitNetAddressTarget.RopeRow) {
      val rowBytes = model.headSize / 2 * 4
      offset := (U(BitNetAddressGeometry.RopeHeaderBytes,
        model.addressWidth bits) +
        safePosition.resize(model.addressWidth) * rowBytes).resized
      transferBytes := rowBytes
    }
  }

  val localAddress = Vec(UInt(model.addressWidth bits), cfg.bankCount)
  val bankRangeLegal = Vec(Bool(), cfg.bankCount)
  for (bank <- 0 until cfg.bankCount) {
    val objectBase = UInt(model.addressWidth bits)
    val objectBytes = UInt(model.addressWidth bits)
    objectBase := U(layout.auxBases(bank), model.addressWidth bits)
    objectBytes := U(BitNetFiveBankStripedLayout.AuxBytes,
      model.addressWidth bits)
    when(requiresPosition) {
      objectBase := U(layout.ropeBases(bank), model.addressWidth bits)
      objectBytes := U(BitNetFiveBankStripedLayout.RopeBytes,
        model.addressWidth bits)
    }
    val start = objectBase.resize(calculationWidth) +
      offset.resize(calculationWidth)
    val endWithinObject = offset.resize(calculationWidth) +
      transferBytes.resize(calculationWidth)
    val endWithinBank = start + transferBytes.resize(calculationWidth)
    localAddress(bank) := start.resize(model.addressWidth)
    bankRangeLegal(bank) := !start.msb && !endWithinObject.msb &&
      !endWithinBank.msb &&
      endWithinObject <= objectBytes.resize(calculationWidth) &&
      endWithinBank <= U(layout.bankCapacities(bank), calculationWidth bits) &&
      !start(0)
  }
  val rangesLegal = bankRangeLegal.asBits.andR
  val requestLegal = targetKnown && layerLegal && positionLegal && rangesLegal

  val responseValid = RegInit(False)
  val responsePayload = Reg(BitNetStripedAuxResponse(cfg)) init
    BitNetStripedAuxResponse(cfg).getZero
  io.response.valid := responseValid
  io.response.payload := responsePayload
  io.request.ready := !responseValid || io.response.ready

  when(io.request.fire) {
    responseValid := True
    responsePayload.target := io.request.target
    responsePayload.layer := io.request.layer
    responsePayload.position := io.request.position
    responsePayload.localAddress := localAddress
    responsePayload.transferBytes := transferBytes
    responsePayload.requiredAlignmentBytes := alignment
    responsePayload.ok := requestLegal
    responsePayload.fault := BitNetFiveBankStripedAguFault.None
    when(!targetKnown) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.UnknownTarget
    } elsewhen(!layerLegal) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.InvalidLayer
    } elsewhen(!positionLegal) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.InvalidPosition
    } elsewhen(!rangesLegal) {
      responsePayload.fault := BitNetFiveBankStripedAguFault.BankRange
    }
  } elsewhen(io.response.fire) {
    responseValid := False
  }
}

object GenerateBitNetFiveBankStripedAuxAgu extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-five-bank-striped-aux-agu")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetFiveBankStripedAuxAgu())
}
