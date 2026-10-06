package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Every external tensor address consumed by the fixed native layer body. */
object BitNetAddressTarget extends SpinalEnum {
  val QWeight,
      KWeight,
      VWeight,
      OWeight,
      GateWeight,
      UpWeight,
      DownWeight,
      QProjectionScale,
      KProjectionScale,
      VProjectionScale,
      OProjectionScale,
      GateProjectionScale,
      UpProjectionScale,
      DownProjectionScale,
      InputNorm,
      PostAttentionNorm,
      AttentionSubNorm,
      FfnSubNorm,
      FinalNorm,
      RopeRow,
      KvKey,
      KvValue,
      KvKeyScale,
      KvValueScale,
      LmWeightTile,
      LmScaleTile,
      // Production five-bank images fuse Gate and Up into one physical
      // group stream.  Keep this element last so the research/legacy target
      // encodings above remain stable; the non-striped AGU deliberately has
      // no descriptor for it and therefore rejects it as UnknownTarget.
      GateUpWeight = newElement()
}

/** First failing rule for one address request.  Individual flags remain visible. */
object BitNetAddressFault extends SpinalEnum {
  val None,
      UnknownTarget,
      InvalidLayer,
      InvalidTokenLane,
      InvalidPosition,
      InvalidLmTile,
      ZeroLength,
      Misaligned,
      TargetBounds,
      AddressOverflow = newElement()
}

/**
  * One native memory-range request.
  *
  * Prefill positions are derived in PL as
  *   prefillBasePosition + prefillRound * tokenParallelism + tokenLane.
  * Decode uses decodePosition directly.  byteOffset and transferBytes then
  * select one aligned burst range within the chosen tensor/row/tile.
  */
case class BitNetAddressRequest(cfg: BitNetConfig) extends Bundle {
  val target = BitNetAddressTarget()
  val layer = UInt(cfg.layerWidth bits)
  val mode = BitNetMode()
  val prefillBasePosition = UInt(cfg.sequenceWidth bits)
  val prefillRound = UInt(cfg.sequenceWidth bits)
  val tokenLane = UInt(cfg.roundTokenCountWidth bits)
  val decodePosition = UInt(cfg.sequenceWidth bits)
  val lmTile = UInt(BitNetAddressGeometry.lmTileIndexWidth(cfg) bits)
  val byteOffset = UInt(cfg.addressWidth bits)
  val transferBytes = UInt(cfg.addressWidth bits)
}

case class BitNetAddressResponse(cfg: BitNetConfig) extends Bundle {
  val target = BitNetAddressTarget()
  val address = UInt(cfg.addressWidth bits)
  val targetBytes = UInt(cfg.addressWidth bits)
  val requiredAlignmentBytes = UInt(6 bits)
  val effectivePosition = UInt(cfg.sequenceWidth bits)
  val lmRowStart = UInt(BitNetAddressGeometry.lmRowIndexWidth(cfg) bits)
  val lmRows = UInt(BitNetAddressGeometry.lmTileRowCountWidth bits)

  val ok = Bool()
  val fault = BitNetAddressFault()
  val unknownTarget = Bool()
  val invalidLayer = Bool()
  val invalidTokenLane = Bool()
  val invalidPosition = Bool()
  val invalidLmTile = Bool()
  val zeroLength = Bool()
  val misaligned = Bool()
  val targetOutOfBounds = Bool()
  val addressOverflow = Bool()
}

/**
  * Pipelined 34-bit native address generator.
  *
  * No old GEMV packet parser or InstructionSequencer is present.  A projection
  * target selects one address from the 30-entry descriptor table; the fixed
  * controller supplies only layer, token-round/position and local burst range.
  * The single output register is both a timing boundary and a lossless Stream
  * skid point under memory backpressure.
  */
class BitNetAddressGenerator(
    cfg: BitNetConfig = BitNetConfig.Production,
    layout: BitNetResidentAddressLayout = BitNetProductionAddressLayout()
) extends Component {
  require(layout.cfg == cfg, "address-generator layout/configuration mismatch")
  layout.validate()

  val io = new Bundle {
    val request = slave(Stream(BitNetAddressRequest(cfg)))
    val response = master(Stream(BitNetAddressResponse(cfg)))
  }

  private val calculationWidth = cfg.addressWidth + 1
  private val tokenLaneShift = log2Up(cfg.tokenParallelism)

  val layerTable = new BitNetLayerDescriptorTable(cfg, layout)
  layerTable.io.layer := io.request.payload.layer

  val lmTable = new BitNetLmTileDescriptorTable(cfg, layout)
  lmTable.io.tile := io.request.payload.lmTile

  // Position generation stays in PL.  The widened path prevents an invalid
  // prefill round from wrapping back into the legal 0..4095 range.
  val prefillRoundOffset = UInt(calculationWidth bits)
  prefillRoundOffset := (
    io.request.payload.prefillRound.resize(calculationWidth) << tokenLaneShift
  ).resize(calculationWidth)
  val prefillPosition = UInt(calculationWidth bits)
  prefillPosition := io.request.payload.prefillBasePosition.resize(calculationWidth) +
    prefillRoundOffset + io.request.payload.tokenLane.resize(calculationWidth)
  val effectivePositionWide = UInt(calculationWidth bits)
  effectivePositionWide := io.request.payload.decodePosition.resize(calculationWidth)
  when(io.request.payload.mode === BitNetMode.Prefill) {
    effectivePositionWide := prefillPosition
  }

  val selectedBase = UInt(cfg.addressWidth bits)
  val selectedBytes = UInt(cfg.addressWidth bits)
  val alignmentBytes = UInt(6 bits)
  val alignmentMask = UInt(cfg.addressWidth bits)
  val targetKnown = Bool()
  val requiresLayer = Bool()
  val requiresPosition = Bool()
  val requiresLmTile = Bool()

  // All legal enum elements assign these fields.  Keep the defaults for the
  // unused enum encodings as a fail-closed path and explicitly permit the
  // exhaustive switch to replace them.
  selectedBase.allowOverride
  selectedBytes.allowOverride
  alignmentBytes.allowOverride
  alignmentMask.allowOverride
  targetKnown.allowOverride

  selectedBase := 0
  selectedBytes := 0
  alignmentBytes := 1
  alignmentMask := 0
  targetKnown := False
  requiresLayer := False
  requiresPosition := False
  requiresLmTile := False

  private def selectLayerRange(base: UInt, bytes: BigInt, alignment: Int): Unit = {
    selectedBase := base
    selectedBytes := U(bytes, cfg.addressWidth bits)
    alignmentBytes := U(alignment, 6 bits)
    alignmentMask := U(alignment - 1, cfg.addressWidth bits)
    targetKnown := True
    requiresLayer := True
  }

  private def selectGlobalRange(base: BigInt, bytes: BigInt, alignment: Int): Unit = {
    selectedBase := U(base, cfg.addressWidth bits)
    selectedBytes := U(bytes, cfg.addressWidth bits)
    alignmentBytes := U(alignment, 6 bits)
    alignmentMask := U(alignment - 1, cfg.addressWidth bits)
    targetKnown := True
  }

  switch(io.request.payload.target) {
    is(BitNetAddressTarget.QWeight) {
      selectLayerRange(
        layerTable.io.descriptor.qWeightBase,
        BitNetAddressGeometry.qWeightBytes(cfg),
        32
      )
    }
    is(BitNetAddressTarget.KWeight) {
      selectLayerRange(
        layerTable.io.descriptor.kWeightBase,
        BitNetAddressGeometry.kWeightBytes(cfg),
        32
      )
    }
    is(BitNetAddressTarget.VWeight) {
      selectLayerRange(
        layerTable.io.descriptor.vWeightBase,
        BitNetAddressGeometry.vWeightBytes(cfg),
        32
      )
    }
    is(BitNetAddressTarget.OWeight) {
      selectLayerRange(
        layerTable.io.descriptor.oWeightBase,
        BitNetAddressGeometry.oWeightBytes(cfg),
        32
      )
    }
    is(BitNetAddressTarget.GateWeight) {
      selectLayerRange(
        layerTable.io.descriptor.gateWeightBase,
        BitNetAddressGeometry.gateWeightBytes(cfg),
        32
      )
    }
    is(BitNetAddressTarget.UpWeight) {
      selectLayerRange(
        layerTable.io.descriptor.upWeightBase,
        BitNetAddressGeometry.upWeightBytes(cfg),
        32
      )
    }
    is(BitNetAddressTarget.DownWeight) {
      selectLayerRange(
        layerTable.io.descriptor.downWeightBase,
        BitNetAddressGeometry.downWeightBytes(cfg),
        32
      )
    }

    is(BitNetAddressTarget.QProjectionScale) {
      selectLayerRange(layerTable.io.descriptor.projectionScaleBase, 2, 2)
    }
    is(BitNetAddressTarget.KProjectionScale) {
      selectLayerRange(layerTable.io.descriptor.projectionScaleBase + 2, 2, 2)
    }
    is(BitNetAddressTarget.VProjectionScale) {
      selectLayerRange(layerTable.io.descriptor.projectionScaleBase + 4, 2, 2)
    }
    is(BitNetAddressTarget.OProjectionScale) {
      selectLayerRange(layerTable.io.descriptor.projectionScaleBase + 6, 2, 2)
    }
    is(BitNetAddressTarget.GateProjectionScale) {
      selectLayerRange(layerTable.io.descriptor.projectionScaleBase + 8, 2, 2)
    }
    is(BitNetAddressTarget.UpProjectionScale) {
      selectLayerRange(layerTable.io.descriptor.projectionScaleBase + 10, 2, 2)
    }
    is(BitNetAddressTarget.DownProjectionScale) {
      selectLayerRange(layerTable.io.descriptor.projectionScaleBase + 12, 2, 2)
    }

    is(BitNetAddressTarget.InputNorm) {
      selectLayerRange(
        layerTable.io.descriptor.inputNormBase,
        BitNetAddressGeometry.hiddenNormBytes(cfg),
        2
      )
    }
    is(BitNetAddressTarget.PostAttentionNorm) {
      selectLayerRange(
        layerTable.io.descriptor.postAttentionNormBase,
        BitNetAddressGeometry.hiddenNormBytes(cfg),
        2
      )
    }
    is(BitNetAddressTarget.AttentionSubNorm) {
      selectLayerRange(
        layerTable.io.descriptor.attentionSubNormBase,
        BitNetAddressGeometry.hiddenNormBytes(cfg),
        2
      )
    }
    is(BitNetAddressTarget.FfnSubNorm) {
      selectLayerRange(
        layerTable.io.descriptor.ffnSubNormBase,
        BitNetAddressGeometry.ffnNormBytes(cfg),
        2
      )
    }
    is(BitNetAddressTarget.FinalNorm) {
      selectGlobalRange(
        layout.finalNormBase,
        BitNetAddressGeometry.hiddenNormBytes(cfg),
        2
      )
    }

    is(BitNetAddressTarget.RopeRow) {
      val rowOffset = (
        effectivePositionWide * U(layout.ropeRowBytes, calculationWidth bits)
      ).resize(calculationWidth)
      selectedBase := (
        U(layout.ropeTableBase, calculationWidth bits) + rowOffset
      ).resize(cfg.addressWidth)
      selectedBytes := U(layout.ropeRowBytes, cfg.addressWidth bits)
      alignmentBytes := 2
      alignmentMask := 1
      targetKnown := True
      requiresPosition := True
    }

    is(BitNetAddressTarget.KvKey, BitNetAddressTarget.KvValue) {
      val layerOffset = (
        io.request.payload.layer.resize(calculationWidth) *
          U(layout.kvDataLayerStride, calculationWidth bits)
      ).resize(calculationWidth)
      val tokenOffset = (
        effectivePositionWide * U(layout.kvDataTokenStride, calculationWidth bits)
      ).resize(calculationWidth)
      val kindOffset = UInt(calculationWidth bits)
      kindOffset := 0
      when(io.request.payload.target === BitNetAddressTarget.KvValue) {
        kindOffset := U(layout.kvDataKindStride, calculationWidth bits)
      }
      selectedBase := (
        U(layout.kvDataBase, calculationWidth bits) + layerOffset +
          tokenOffset + kindOffset
      ).resize(cfg.addressWidth)
      selectedBytes := U(layout.kvDataKindStride, cfg.addressWidth bits)
      alignmentBytes := 32
      alignmentMask := 31
      targetKnown := True
      requiresLayer := True
      requiresPosition := True
    }

    is(BitNetAddressTarget.KvKeyScale, BitNetAddressTarget.KvValueScale) {
      val layerOffset = (
        io.request.payload.layer.resize(calculationWidth) *
          U(layout.kvScaleLayerStride, calculationWidth bits)
      ).resize(calculationWidth)
      val tokenOffset = (
        effectivePositionWide * U(layout.kvScaleTokenStride, calculationWidth bits)
      ).resize(calculationWidth)
      val kindOffset = UInt(calculationWidth bits)
      kindOffset := 0
      when(io.request.payload.target === BitNetAddressTarget.KvValueScale) {
        kindOffset := U(layout.kvScaleKindStride, calculationWidth bits)
      }
      selectedBase := (
        U(layout.kvScaleBase, calculationWidth bits) + layerOffset +
          kindOffset + tokenOffset
      ).resize(cfg.addressWidth)
      selectedBytes := U(layout.kvScaleTokenStride, cfg.addressWidth bits)
      alignmentBytes := 4
      alignmentMask := 3
      targetKnown := True
      requiresLayer := True
      requiresPosition := True
    }

    is(BitNetAddressTarget.LmWeightTile) {
      selectedBase := lmTable.io.descriptor.weightBase
      selectedBytes := (
        lmTable.io.descriptor.rows.resize(cfg.addressWidth) *
          U(cfg.hiddenSize, cfg.addressWidth bits)
      ).resize(cfg.addressWidth)
      alignmentBytes := 32
      alignmentMask := 31
      targetKnown := True
      requiresLmTile := True
    }
    is(BitNetAddressTarget.LmScaleTile) {
      selectedBase := lmTable.io.descriptor.scaleBase
      selectedBytes := (
        lmTable.io.descriptor.rows.resize(cfg.addressWidth) * 4
      ).resize(cfg.addressWidth)
      alignmentBytes := 4
      alignmentMask := 3
      targetKnown := True
      requiresLmTile := True
    }
  }

  val relativeEnd = UInt(calculationWidth bits)
  relativeEnd := io.request.payload.byteOffset.resize(calculationWidth) +
    io.request.payload.transferBytes.resize(calculationWidth)
  val addressWide = UInt(calculationWidth bits)
  addressWide := selectedBase.resize(calculationWidth) +
    io.request.payload.byteOffset.resize(calculationWidth)
  val addressEndWide = UInt(calculationWidth bits)
  addressEndWide := addressWide +
    io.request.payload.transferBytes.resize(calculationWidth)

  val candidateAddress = addressWide.resize(cfg.addressWidth)
  val invalidLayer = requiresLayer && !layerTable.io.valid
  val invalidTokenLane = requiresPosition &&
    io.request.payload.mode === BitNetMode.Prefill &&
    io.request.payload.tokenLane >= cfg.tokenParallelism
  val invalidPosition = requiresPosition &&
    effectivePositionWide >= cfg.maxSequenceLength
  val invalidLmTile = requiresLmTile && !lmTable.io.valid
  val zeroLength = io.request.payload.transferBytes === 0
  val misaligned =
    (candidateAddress & alignmentMask).orR ||
      (io.request.payload.transferBytes & alignmentMask).orR
  val targetOutOfBounds = relativeEnd > selectedBytes.resize(calculationWidth)
  val addressOverflow = addressEndWide >
    U(BigInt(1) << cfg.addressWidth, calculationWidth bits)

  val anyFault = !targetKnown || invalidLayer || invalidTokenLane ||
    invalidPosition || invalidLmTile || zeroLength || misaligned ||
    targetOutOfBounds || addressOverflow

  val fault = BitNetAddressFault()
  fault := BitNetAddressFault.None
  when(!targetKnown) {
    fault := BitNetAddressFault.UnknownTarget
  } elsewhen (invalidLayer) {
    fault := BitNetAddressFault.InvalidLayer
  } elsewhen (invalidTokenLane) {
    fault := BitNetAddressFault.InvalidTokenLane
  } elsewhen (invalidPosition) {
    fault := BitNetAddressFault.InvalidPosition
  } elsewhen (invalidLmTile) {
    fault := BitNetAddressFault.InvalidLmTile
  } elsewhen (zeroLength) {
    fault := BitNetAddressFault.ZeroLength
  } elsewhen (misaligned) {
    fault := BitNetAddressFault.Misaligned
  } elsewhen (targetOutOfBounds) {
    fault := BitNetAddressFault.TargetBounds
  } elsewhen (addressOverflow) {
    fault := BitNetAddressFault.AddressOverflow
  }

  val computed = Stream(BitNetAddressResponse(cfg))
  computed.valid := io.request.valid
  io.request.ready := computed.ready
  computed.payload.target := io.request.payload.target
  computed.payload.address := candidateAddress
  computed.payload.targetBytes := selectedBytes
  computed.payload.requiredAlignmentBytes := alignmentBytes
  computed.payload.effectivePosition := effectivePositionWide.resize(cfg.sequenceWidth)
  computed.payload.lmRowStart := lmTable.io.descriptor.rowStart
  computed.payload.lmRows := lmTable.io.descriptor.rows
  computed.payload.ok := !anyFault
  computed.payload.fault := fault
  computed.payload.unknownTarget := !targetKnown
  computed.payload.invalidLayer := invalidLayer
  computed.payload.invalidTokenLane := invalidTokenLane
  computed.payload.invalidPosition := invalidPosition
  computed.payload.invalidLmTile := invalidLmTile
  computed.payload.zeroLength := zeroLength
  computed.payload.misaligned := misaligned
  computed.payload.targetOutOfBounds := targetOutOfBounds
  computed.payload.addressOverflow := addressOverflow

  io.response << computed.m2sPipe()
}
