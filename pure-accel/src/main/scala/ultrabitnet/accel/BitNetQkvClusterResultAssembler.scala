package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Geometry of the lossless cluster-result to QKV normalization boundary. */
case class BitNetQkvClusterResultAssemblerConfig(
    model: BitNetConfig = BitNetConfig.Production,
    continuous: BitNetContinuousMatrixConfig =
      BitNetContinuousMatrixConfig(),
    qkv: BitNetQkvPostprocessConfig = BitNetQkvPostprocessConfig(),
    kTileCountWidth: Int = 16
) {
  require(model.engineCount >= 1 && model.engineCount <= 2)
  require(model.queryHeadCount == 20 && model.kvHeadCount == 5)
  require(model.headSize >= 4 && isPow2(model.headSize))
  require(continuous.tokenLanesPerEngine == 2)
  require(model.tokenParallelism ==
    model.engineCount * continuous.tokenLanesPerEngine)
  require(continuous.base.outputLanes == 2)
  require(continuous.base.accumulatorWidth == 32)
  require(continuous.base.tokenIdWidth == qkv.tokenIdWidth)
  require(qkv.tokenLanes == model.tokenParallelism)
  require(qkv.queryHeadCount == model.queryHeadCount)
  require(qkv.kvHeadCount == model.kvHeadCount)
  require(qkv.headSize == model.headSize)
  require(kTileCountWidth >= 2)

  val engineCount: Int = model.engineCount
  val tokenLanes: Int = model.tokenParallelism
  val queryFeatures: Int = model.queryHeadCount * model.headSize
  val keyFeatures: Int = model.kvHeadCount * model.headSize
  val valueFeatures: Int = model.kvHeadCount * model.headSize
  val totalFeatures: Int = queryFeatures + keyFeatures + valueFeatures
  val packedRows: Int = continuous.base.outputLanes
  val prefillGroups: Int = totalFeatures / packedRows
  val decodeFeaturesPerEngine: Int = totalFeatures / engineCount
  val decodeGroupsPerEngine: Int = decodeFeaturesPerEngine / packedRows
  val decodeGroupWidth: Int = Math.max(1, log2Up(decodeGroupsPerEngine))

  require(totalFeatures == 30 * model.headSize)
  require(totalFeatures % (engineCount * packedRows) == 0)
  require(totalFeatures <= (1 << continuous.base.outputIndexWidth))
}

/**
  * Descriptor forked from the QKV projection task when the shared cluster is
  * leased.  It contains every invariant needed to reject a result without
  * asking Linux to reconstruct or authenticate a QKV payload.
  *
  * Prefill owns the complete [0,totalFeatures) row interval on both engines.
  * Decode owns two equal, contiguous row shards; their physical engine order
  * may be swapped, but together they must cover the complete interval.
  */
case class BitNetQkvClusterRoundDescriptor(
    cfg: BitNetQkvClusterResultAssemblerConfig) extends Bundle {
  private val m = cfg.continuous
  private val b = m.base

  val scheduleMode = BitNetMode()
  val sequenceId = UInt(m.sequenceIdWidth bits)
  val taskIdBase = UInt(m.taskIdWidth bits)
  val expectedBeats = UInt(m.beatCountWidth bits)
  val kTilesPerOutput = UInt(cfg.kTileCountWidth bits)
  val activeTokenMask = Bits(cfg.tokenLanes bits)
  val tokenId = Vec(UInt(b.tokenIdWidth bits), cfg.tokenLanes)
  val activationFactorQ16 = Vec(UInt(32 bits), cfg.tokenLanes)
  val weightScaleQ16 = Vec(UInt(32 bits), cfg.engineCount)
  val outputBase = Vec(UInt(b.outputIndexWidth bits), cfg.engineCount)
  val outputEndExclusive =
    Vec(UInt(b.outputIndexWidth bits), cfg.engineCount)
}

/** One naturally ordered QKV feature's position-specific RoPE coefficients. */
case class BitNetQkvFeatureRopeCoefficient(
    cfg: BitNetQkvClusterResultAssemblerConfig) extends Bundle {
  private val m = cfg.continuous
  val sequenceId = UInt(m.sequenceIdWidth bits)
  val globalFeature = UInt(m.base.outputIndexWidth bits)
  val cosine = Vec(SInt(16 bits), cfg.tokenLanes)
  val sine = Vec(SInt(16 bits), cfg.tokenLanes)
}

/** First architecturally visible reason why one QKV round was rejected. */
object BitNetQkvAssemblerFaultCode extends SpinalEnum(binarySequential) {
  val None, InvalidDescriptor, ResultMetadata, RopeMetadata,
      UnexpectedResult = newElement()
}

/** Backpressure-safe terminal event for one accepted round descriptor. */
case class BitNetQkvAssemblerCompletion(
    cfg: BitNetQkvClusterResultAssemblerConfig) extends Bundle {
  val sequenceId = UInt(cfg.continuous.sequenceIdWidth bits)
  val success = Bool()
  val faultCode = BitNetQkvAssemblerFaultCode()
  /** Number of feature beats actually accepted by the downstream consumer. */
  val emittedFeatures = UInt(32 bits)
}

/** Pure mapping oracle shared by schedule construction and tests. */
object BitNetQkvClusterResultAssemblerOracle {
  sealed trait Kind
  case object Query extends Kind
  case object Key extends Kind
  case object Value extends Kind

  final case class Feature(kind: Kind, head: Int, element: Int)
  final case class DecodeSource(
      globalFeature: Int,
      engine: Int,
      pairIndex: Int,
      packedLane: Int)

  def mapFeature(
      globalFeature: Int,
      queryHeads: Int = 20,
      kvHeads: Int = 5,
      headSize: Int = 128): Feature = {
    val queryFeatures = queryHeads * headSize
    val keyFeatures = kvHeads * headSize
    val totalFeatures = queryFeatures + 2 * keyFeatures
    require(globalFeature >= 0 && globalFeature < totalFeatures)
    if (globalFeature < queryFeatures) {
      Feature(Query, globalFeature / headSize, globalFeature % headSize)
    } else if (globalFeature < queryFeatures + keyFeatures) {
      val local = globalFeature - queryFeatures
      Feature(Key, local / headSize, local % headSize)
    } else {
      val local = globalFeature - queryFeatures - keyFeatures
      Feature(Value, local / headSize, local % headSize)
    }
  }

  /** Natural replay route for one full-range or two non-overlapping shards. */
  def decodeSources(
      outputBase: Seq[Int],
      outputEndExclusive: Seq[Int]): Seq[DecodeSource] = {
    val engines = outputBase.length
    require((engines == 1 || engines == 2) &&
      outputEndExclusive.length == engines)
    val total = outputEndExclusive.max
    require(outputBase.min == 0)
    require((0 until engines).forall { engine =>
      outputBase(engine) >= 0 &&
        outputEndExclusive(engine) > outputBase(engine) &&
        (outputBase(engine) & 1) == 0 &&
        (outputEndExclusive(engine) & 1) == 0
    })
    require(engines == 1 ||
      (outputEndExclusive(0) == outputBase(1) && outputBase(0) == 0) ||
      (outputEndExclusive(1) == outputBase(0) && outputBase(1) == 0))
    (0 until total).map { feature =>
      val engine = (0 until engines).find { candidate =>
        feature >= outputBase(candidate) &&
          feature < outputEndExclusive(candidate)
      }.get
      val local = feature - outputBase(engine)
      DecodeSource(feature, engine, local / 2, local & 1)
    }
  }
}

/**
  * Converts the only TP4 cluster's two TP2 projection returns into the single
  * QKV postprocess stream.
  *
  * In prefill both engines refer to the same packed output rows and represent
  * token lanes 0/1 and 2/3.  A final-K pair is held while packed values 0 and 1
  * are emitted as consecutive natural features.  In decode each engine uses
  * lane 0 for an independent output-row shard.  Final pairs are written to two
  * local BRAMs and replayed in global feature order.  This bounded PL reorder
  * is required because the feeder advances two far-separated decode shards in
  * parallel; forwarding arrival order would mix Q/K/V head frames.
  *
  * Every result is checked against the resident descriptor.  Invalid data is
  * consumed only to drain the failed round, never marked valid downstream.
  */
class BitNetQkvClusterResultAssembler(
    cfg: BitNetQkvClusterResultAssemblerConfig =
      BitNetQkvClusterResultAssemblerConfig()) extends Component {
  private val m = cfg.continuous
  private val b = m.base
  private val engines = cfg.engineCount
  private val lanes = cfg.tokenLanes
  private val locals = m.tokenLanesPerEngine

  val io = new Bundle {
    val round = slave(Stream(BitNetQkvClusterRoundDescriptor(cfg)))
    val engineResult = Vec(
      slave(Stream(BitNetProductionCompactTp2DotResultBeat(m))), engines)
    val ropeCoefficient =
      slave(Stream(BitNetQkvFeatureRopeCoefficient(cfg)))
    val output = master(Stream(BitNetQkvTp4DotBeat(cfg.qkv)))
    val completion = master(Stream(BitNetQkvAssemblerCompletion(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedRounds = out UInt(32 bits)
    val completedRounds = out UInt(32 bits)
    val failedRounds = out UInt(32 bits)
    val emittedFeatures = out UInt(32 bits)
  }

  object State extends SpinalEnum(binarySequential) {
    val Idle, PrefillCollect, PrefillEmit, DecodeCollect, DecodeIssue,
        DecodeWait, DecodeEmit, FaultDrain = newElement()
  }

  val state = Reg(State()) init State.Idle
  val descriptor = Reg(BitNetQkvClusterRoundDescriptor(cfg)) init
    BitNetQkvClusterRoundDescriptor(cfg).getZero
  val activeEngineMask = Reg(Bits(engines bits)) init 0
  val resultBeatCount = Vec(Reg(UInt(m.beatCountWidth bits)) init 0, engines)
  val kTileIndex =
    Vec(Reg(UInt(cfg.kTileCountWidth bits)) init 0, engines)
  val outputGroup =
    Vec(Reg(UInt(b.outputIndexWidth bits)) init 0, engines)
  val coefficientCount = Reg(UInt(b.outputIndexWidth bits)) init 0
  val roundFailed = RegInit(False)
  val activeFaultCode = Reg(BitNetQkvAssemblerFaultCode()) init
    BitNetQkvAssemblerFaultCode.None
  val roundEmittedFeatures = Reg(UInt(32 bits)) init 0

  val stickyFault = RegInit(False)
  val acceptedCount = Reg(UInt(32 bits)) init 0
  val completedCount = Reg(UInt(32 bits)) init 0
  val failedCount = Reg(UInt(32 bits)) init 0
  val emittedCount = Reg(UInt(32 bits)) init 0

  val completionValid = RegInit(False)
  val completionReg = Reg(BitNetQkvAssemblerCompletion(cfg)) init
    BitNetQkvAssemblerCompletion(cfg).getZero
  io.completion.valid := completionValid
  io.completion.payload := completionReg
  when(io.completion.fire) { completionValid := False }

  io.busy := state =/= State.Idle || completionValid
  io.protocolFaultSticky := stickyFault
  io.acceptedRounds := acceptedCount
  io.completedRounds := completedCount
  io.failedRounds := failedCount
  io.emittedFeatures := emittedCount

  val prefillMaskLegal = (1 to lanes).map { count =>
    io.round.activeTokenMask === B((BigInt(1) << count) - 1, lanes bits)
  }.reduce(_ || _)
  val inputPrefill = io.round.scheduleMode === BitNetMode.Prefill
  val inputDecode = io.round.scheduleMode === BitNetMode.Decode
  val totalFeaturesU = U(cfg.totalFeatures, b.outputIndexWidth bits)
  val halfFeaturesU = U(cfg.decodeFeaturesPerEngine,
    b.outputIndexWidth bits)
  val prefillLayoutLegal = (0 until engines).map { engine =>
    io.round.outputBase(engine) === 0 &&
      io.round.outputEndExclusive(engine) === totalFeaturesU &&
      io.round.weightScaleQ16(engine) === io.round.weightScaleQ16(0)
  }.reduce(_ && _)
  val decodeLayoutLegal = if (engines == 1) {
    io.round.outputBase(0) === 0 &&
      io.round.outputEndExclusive(0) === totalFeaturesU
  } else {
    val forward = io.round.outputBase(0) === 0 &&
      io.round.outputEndExclusive(0) === halfFeaturesU &&
      io.round.outputBase(1) === halfFeaturesU &&
      io.round.outputEndExclusive(1) === totalFeaturesU
    val reverse = io.round.outputBase(1) === 0 &&
      io.round.outputEndExclusive(1) === halfFeaturesU &&
      io.round.outputBase(0) === halfFeaturesU &&
      io.round.outputEndExclusive(0) === totalFeaturesU
    forward || reverse
  }
  val expectedPrefillBeats =
    (io.round.kTilesPerOutput.resize(m.beatCountWidth) *
      U(cfg.prefillGroups, m.beatCountWidth bits)).resize(m.beatCountWidth)
  val expectedDecodeBeats =
    (io.round.kTilesPerOutput.resize(m.beatCountWidth) *
      U(cfg.decodeGroupsPerEngine, m.beatCountWidth bits)).resize(
        m.beatCountWidth)
  val descriptorLegal = io.round.kTilesPerOutput =/= 0 &&
    ((inputPrefill && prefillMaskLegal && prefillLayoutLegal &&
      io.round.expectedBeats === expectedPrefillBeats) ||
      (inputDecode && io.round.activeTokenMask === B(1, lanes bits) &&
        decodeLayoutLegal &&
        io.round.expectedBeats === expectedDecodeBeats))

  io.round.ready := state === State.Idle && !completionValid
  when(io.round.fire) {
    when(descriptorLegal) {
      descriptor := io.round.payload
      acceptedCount := acceptedCount + 1
      resultBeatCount.foreach(_ := 0)
      kTileIndex.foreach(_ := 0)
      outputGroup.foreach(_ := 0)
      coefficientCount := 0
      roundFailed := False
      activeFaultCode := BitNetQkvAssemblerFaultCode.None
      roundEmittedFeatures := 0
      when(inputPrefill) {
        for (engine <- 0 until engines) {
          activeEngineMask(engine) := io.round.activeTokenMask(
            engine * locals + locals - 1 downto engine * locals).orR
        }
        state := State.PrefillCollect
      } otherwise {
        activeEngineMask := B((BigInt(1) << engines) - 1, engines bits)
        state := State.DecodeCollect
      }
    } otherwise {
      stickyFault := True
      failedCount := failedCount + 1
      completionReg.sequenceId := io.round.sequenceId
      completionReg.success := False
      completionReg.faultCode :=
        BitNetQkvAssemblerFaultCode.InvalidDescriptor
      completionReg.emittedFeatures := 0
      completionValid := True
    }
  }

  val prefillRaw = Vec((0 until b.outputLanes).map { _ =>
    Vec((0 until lanes).map { _ => Reg(SInt(32 bits)) init 0 })
  })
  val prefillPackedLane = Reg(UInt(log2Up(b.outputLanes) bits)) init 0

  val decodeMemories = Array.fill(engines)(
    Mem(Bits((b.outputLanes * b.accumulatorWidth) bits),
      cfg.decodeGroupsPerEngine))

  for (engine <- 0 until engines) {
    io.engineResult(engine).ready := False
  }
  io.ropeCoefficient.ready := False

  val allActivePrefillResultsValid = (0 until engines).map { engine =>
    !activeEngineMask(engine) || io.engineResult(engine).valid
  }.reduce(_ && _)
  val unexpectedDisabledPrefillValid = (0 until engines).map { engine =>
    !activeEngineMask(engine) && io.engineResult(engine).valid
  }.reduce(_ || _)
  val prefillPairLegal = Vec(Bool(), engines)
  for (engine <- 0 until engines) {
    val pair = io.engineResult(engine).payload
    val expectedMask = descriptor.activeTokenMask(
      engine * locals + locals - 1 downto engine * locals)
    val expectedBaseTag =
      (outputGroup(engine).resize(m.accumulatorTagWidth) << 1)
        .resize(m.accumulatorTagWidth)
    val rowsCanonical = Vec(Bool(), locals)
    for (local <- 0 until locals) {
      rowsCanonical(local) := True
      when(!expectedMask(local)) {
        rowsCanonical(local) := pair.rawRows(local).asBits === 0
      }
    }
    prefillPairLegal(engine) := !activeEngineMask(engine) ||
      (pair.validMask === expectedMask && pair.validMask =/= B"2'b00" &&
        pair.sequenceId === descriptor.sequenceId &&
        pair.baseAccumulatorTag === expectedBaseTag &&
        (!expectedMask(1) ||
          (pair.baseAccumulatorTag + 1).resize(m.accumulatorTagWidth) ===
            (expectedBaseTag + 1).resize(m.accumulatorTagWidth)) &&
        pair.firstK === (kTileIndex(engine) === 0) &&
        pair.lastK ===
          (kTileIndex(engine) === descriptor.kTilesPerOutput - 1) &&
        pair.formatClass ===
          BitNetProductionCompactDotFormatClass.Map0KMajor &&
        rowsCanonical.asBits.andR)
  }
  val allPrefillMetadataLegal = prefillPairLegal.asBits.andR
  val currentPrefillLastK = kTileIndex(0) ===
    descriptor.kTilesPerOutput - 1

  when(state === State.PrefillCollect) {
    for (engine <- 0 until engines) {
      when(activeEngineMask(engine)) {
        // A disabled-engine violation wins the cycle atomically.  Keep every
        // active result held upstream so FaultDrain can account for it on the
        // following cycle instead of consuming it without incrementing the
        // bounded drain count.
        io.engineResult(engine).ready := allActivePrefillResultsValid &&
          !unexpectedDisabledPrefillValid
      } otherwise {
        // Unexpected output from a disabled tail engine is consumed but can
        // never enter the normalized QKV stream.
        io.engineResult(engine).ready := True
        when(io.engineResult(engine).fire) {
          stickyFault := True
          roundFailed := True
          when(!roundFailed) {
            activeFaultCode := BitNetQkvAssemblerFaultCode.UnexpectedResult
          }
          state := State.FaultDrain
        }
      }
    }

    val joinedFire = allActivePrefillResultsValid &&
      !unexpectedDisabledPrefillValid &&
      (0 until engines).map { engine =>
        !activeEngineMask(engine) || io.engineResult(engine).ready
      }.reduce(_ && _)
    when(joinedFire) {
      for (engine <- 0 until engines) {
        when(activeEngineMask(engine)) {
          resultBeatCount(engine) := resultBeatCount(engine) + 1
        }
      }
      when(!allPrefillMetadataLegal) {
        stickyFault := True
        roundFailed := True
        when(!roundFailed) {
          activeFaultCode := BitNetQkvAssemblerFaultCode.ResultMetadata
        }
        state := State.FaultDrain
      } otherwise {
        when(currentPrefillLastK) {
          for (packed <- 0 until b.outputLanes;
               lane <- 0 until lanes) {
            val engine = lane / locals
            val local = lane % locals
            prefillRaw(packed)(lane) := 0
            when(activeEngineMask(engine)) {
              prefillRaw(packed)(lane) :=
                io.engineResult(engine).payload.rawRows(local)(packed)
            }
          }
          prefillPackedLane := 0
          state := State.PrefillEmit
        }
      }

      for (engine <- 0 until engines) {
        when(activeEngineMask(engine)) {
          when(kTileIndex(engine) === descriptor.kTilesPerOutput - 1) {
            kTileIndex(engine) := 0
            outputGroup(engine) := outputGroup(engine) + 1
          } otherwise {
            kTileIndex(engine) := kTileIndex(engine) + 1
          }
        }
      }
    }
  }

  val outputValidReg = RegInit(False)
  val outputReg = Reg(BitNetQkvTp4DotBeat(cfg.qkv)) init
    BitNetQkvTp4DotBeat(cfg.qkv).getZero
  val outputEndsRound = RegInit(False)
  io.output.valid := outputValidReg
  io.output.payload := outputReg
  when(io.output.fire) {
    outputValidReg := False
    emittedCount := emittedCount + 1
    roundEmittedFeatures := roundEmittedFeatures + 1
  }

  def assignMappedFeature(
      target: BitNetQkvTp4DotBeat,
      globalFeature: UInt): Unit = {
    val queryEnd = U(cfg.queryFeatures, b.outputIndexWidth bits)
    val keyEnd = U(cfg.queryFeatures + cfg.keyFeatures,
      b.outputIndexWidth bits)
    val element = globalFeature(log2Up(cfg.model.headSize) - 1 downto 0)
    target.kind := BitNetQkvKind.Query
    target.head := (globalFeature >> log2Up(cfg.model.headSize))
      .resize(cfg.qkv.headWidth)
    target.element := element.resize(cfg.qkv.elementWidth)
    when(globalFeature >= queryEnd && globalFeature < keyEnd) {
      val local = globalFeature - queryEnd
      target.kind := BitNetQkvKind.Key
      target.head := (local >> log2Up(cfg.model.headSize))
        .resize(cfg.qkv.headWidth)
      target.element := local(log2Up(cfg.model.headSize) - 1 downto 0)
        .resize(cfg.qkv.elementWidth)
    } elsewhen(globalFeature >= keyEnd) {
      val local = globalFeature - keyEnd
      target.kind := BitNetQkvKind.Value
      target.head := (local >> log2Up(cfg.model.headSize))
        .resize(cfg.qkv.headWidth)
      target.element := local(log2Up(cfg.model.headSize) - 1 downto 0)
        .resize(cfg.qkv.elementWidth)
    }
    // Every Q/K/V boundary is head-size aligned, so the low coordinate bits
    // are also the natural element after subtracting either constant base.
    // Use that combinational value here; reading target.element would observe
    // the register's previous value in the sequential output-fill block.
    target.first := element === 0
    target.last := element === cfg.model.headSize - 1
  }

  val coefficientMatches =
    io.ropeCoefficient.sequenceId === descriptor.sequenceId &&
      io.ropeCoefficient.globalFeature === coefficientCount

  when(state === State.PrefillEmit && !outputValidReg) {
    io.ropeCoefficient.ready := True
    when(io.ropeCoefficient.fire) {
      coefficientCount := coefficientCount + 1
      when(coefficientMatches) {
        for (lane <- 0 until lanes) {
          outputReg.rawDot(lane) := prefillRaw(prefillPackedLane)(lane)
          outputReg.activationFactorQ16(lane) :=
            descriptor.activationFactorQ16(lane)
          outputReg.tokenId(lane) := descriptor.tokenId(lane)
          outputReg.cosine(lane) := io.ropeCoefficient.cosine(lane)
          outputReg.sine(lane) := io.ropeCoefficient.sine(lane)
        }
        outputReg.weightScaleQ16 := descriptor.weightScaleQ16(0)
        outputReg.tokenMask := descriptor.activeTokenMask
        assignMappedFeature(outputReg, coefficientCount)
        outputEndsRound := coefficientCount === cfg.totalFeatures - 1
        outputValidReg := True
      } otherwise {
        stickyFault := True
        roundFailed := True
        when(!roundFailed) {
          activeFaultCode := BitNetQkvAssemblerFaultCode.RopeMetadata
        }
        state := State.FaultDrain
      }
    }
  }

  when(state === State.PrefillEmit && io.output.fire) {
    when(outputEndsRound) {
      completedCount := completedCount + 1
      completionReg.sequenceId := descriptor.sequenceId
      completionReg.success := True
      completionReg.faultCode := BitNetQkvAssemblerFaultCode.None
      completionReg.emittedFeatures := roundEmittedFeatures + 1
      completionValid := True
      state := State.Idle
    } elsewhen(prefillPackedLane === b.outputLanes - 1) {
      prefillPackedLane := 0
      state := State.PrefillCollect
    } otherwise {
      prefillPackedLane := prefillPackedLane + 1
    }
  }

  val decodePairLegal = Vec(Bool(), engines)
  for (engine <- 0 until engines) {
    val pair = io.engineResult(engine).payload
    val expectedBaseTag = outputGroup(engine).resize(m.accumulatorTagWidth)
    decodePairLegal(engine) :=
      pair.validMask === B"2'b01" &&
        pair.sequenceId === descriptor.sequenceId &&
        pair.baseAccumulatorTag === expectedBaseTag &&
        pair.firstK === (kTileIndex(engine) === 0) &&
        pair.lastK ===
          (kTileIndex(engine) === descriptor.kTilesPerOutput - 1) &&
        pair.formatClass ===
          BitNetProductionCompactDotFormatClass.Map0KMajor &&
        pair.rawRows(1).asBits === 0
  }

  when(state === State.DecodeCollect) {
    for (engine <- 0 until engines) {
      io.engineResult(engine).ready :=
        resultBeatCount(engine) < descriptor.expectedBeats
      when(io.engineResult(engine).fire) {
        resultBeatCount(engine) := resultBeatCount(engine) + 1
        when(!decodePairLegal(engine)) {
          stickyFault := True
          roundFailed := True
          when(!roundFailed) {
            activeFaultCode := BitNetQkvAssemblerFaultCode.ResultMetadata
          }
        } otherwise {
          when(kTileIndex(engine) === descriptor.kTilesPerOutput - 1) {
            val packed =
              io.engineResult(engine).payload.rawRows(0)(1).asBits ##
                io.engineResult(engine).payload.rawRows(0)(0).asBits
            decodeMemories(engine).write(
              outputGroup(engine).resize(cfg.decodeGroupWidth), packed)
          }
        }
        when(kTileIndex(engine) === descriptor.kTilesPerOutput - 1) {
          kTileIndex(engine) := 0
          outputGroup(engine) := outputGroup(engine) + 1
        } otherwise {
          kTileIndex(engine) := kTileIndex(engine) + 1
        }
      }
    }

    val enginesDoneNext = Vec(Bool(), engines)
    for (engine <- 0 until engines) {
      enginesDoneNext(engine) :=
        resultBeatCount(engine) === descriptor.expectedBeats ||
          (io.engineResult(engine).fire &&
            resultBeatCount(engine) === descriptor.expectedBeats - 1)
    }
    when(enginesDoneNext.asBits.andR) {
      state := State.DecodeIssue
    }
  }

  val engineIndexWidth = scala.math.max(1, log2Up(engines))
  val replayOwner = UInt(engineIndexWidth bits)
  replayOwner := 0
  for (engine <- 1 until engines) {
    when(coefficientCount >= descriptor.outputBase(engine) &&
        coefficientCount < descriptor.outputEndExclusive(engine)) {
      replayOwner := engine
    }
  }
  val replayLocalFeature = Vec(UInt(b.outputIndexWidth bits), engines)
  val replayReadAddress = Vec(UInt(cfg.decodeGroupWidth bits), engines)
  for (engine <- 0 until engines) {
    replayLocalFeature(engine) :=
      (coefficientCount - descriptor.outputBase(engine)).resized
    replayReadAddress(engine) :=
      (replayLocalFeature(engine) >> 1).resize(cfg.decodeGroupWidth)
  }
  val decodeReadEnable = Vec(Bool(), engines)
  for (engine <- 0 until engines) {
    decodeReadEnable(engine) := state === State.DecodeIssue &&
      !outputValidReg && io.ropeCoefficient.valid && coefficientMatches &&
      replayOwner === engine
  }
  val decodeReadData = (0 until engines).map { engine =>
    decodeMemories(engine).readSync(
      replayReadAddress(engine), decodeReadEnable(engine))
  }
  val readOwner = Reg(UInt(engineIndexWidth bits)) init 0
  val readPackedLane = Reg(UInt(log2Up(b.outputLanes) bits)) init 0
  val readGlobalFeature = Reg(UInt(b.outputIndexWidth bits)) init 0
  val readCosine = Vec(Reg(SInt(16 bits)) init 0, lanes)
  val readSine = Vec(Reg(SInt(16 bits)) init 0, lanes)

  when(state === State.DecodeIssue) {
    when(roundFailed) {
      state := State.FaultDrain
    } elsewhen(!outputValidReg) {
      io.ropeCoefficient.ready := True
      when(io.ropeCoefficient.fire) {
        coefficientCount := coefficientCount + 1
        when(coefficientMatches) {
          readOwner := replayOwner
          readPackedLane := (if (engines == 1)
            replayLocalFeature(0)(0).asUInt
          else replayLocalFeature(replayOwner)(0).asUInt)
          readGlobalFeature := coefficientCount
          readCosine := io.ropeCoefficient.cosine
          readSine := io.ropeCoefficient.sine
          state := State.DecodeWait
        } otherwise {
          stickyFault := True
          roundFailed := True
          when(!roundFailed) {
            activeFaultCode := BitNetQkvAssemblerFaultCode.RopeMetadata
          }
          state := State.FaultDrain
        }
      }
    }
  }

  // readSync data becomes valid in this state, one cycle after DecodeIssue.
  when(state === State.DecodeWait) {
    state := State.DecodeEmit
  }

  when(state === State.DecodeEmit && !outputValidReg) {
    val selectedWord = Bits((b.outputLanes * b.accumulatorWidth) bits)
    selectedWord := decodeReadData(0)
    for (engine <- 1 until engines) {
      when(readOwner === engine) { selectedWord := decodeReadData(engine) }
    }
    val selectedRaw = SInt(b.accumulatorWidth bits)
    selectedRaw := selectedWord(b.accumulatorWidth - 1 downto 0).asSInt
    when(readPackedLane === 1) {
      selectedRaw := selectedWord(2 * b.accumulatorWidth - 1 downto
        b.accumulatorWidth).asSInt
    }
    for (lane <- 0 until lanes) {
      if (lane == 0) {
        outputReg.rawDot(lane) := selectedRaw
        outputReg.activationFactorQ16(lane) :=
          descriptor.activationFactorQ16(0)
        outputReg.tokenId(lane) := descriptor.tokenId(0)
      } else {
        outputReg.rawDot(lane) := 0
        outputReg.activationFactorQ16(lane) := 0
        outputReg.tokenId(lane) := 0
      }
      outputReg.cosine(lane) := readCosine(lane)
      outputReg.sine(lane) := readSine(lane)
    }
    outputReg.weightScaleQ16 := (if (engines == 1)
      descriptor.weightScaleQ16(0)
    else descriptor.weightScaleQ16(readOwner))
    outputReg.tokenMask := B(1, lanes bits)
    assignMappedFeature(outputReg, readGlobalFeature)
    outputEndsRound := readGlobalFeature === cfg.totalFeatures - 1
    outputValidReg := True
  }

  when(state === State.DecodeEmit && io.output.fire) {
    when(outputEndsRound) {
      completedCount := completedCount + 1
      completionReg.sequenceId := descriptor.sequenceId
      completionReg.success := True
      completionReg.faultCode := BitNetQkvAssemblerFaultCode.None
      completionReg.emittedFeatures := roundEmittedFeatures + 1
      completionValid := True
      state := State.Idle
    } otherwise {
      state := State.DecodeIssue
    }
  }

  // A failed round is drained according to its descriptor.  No failed result
  // or coefficient can set outputValidReg, so the boundary is fail-closed.
  when(state === State.FaultDrain) {
    outputValidReg := False
    for (engine <- 0 until engines) {
      val expected = Mux(activeEngineMask(engine),
        descriptor.expectedBeats, U(0, m.beatCountWidth bits))
      io.engineResult(engine).ready := resultBeatCount(engine) < expected ||
        !activeEngineMask(engine)
      when(io.engineResult(engine).fire &&
          resultBeatCount(engine) < expected) {
        resultBeatCount(engine) := resultBeatCount(engine) + 1
      }
    }
    io.ropeCoefficient.ready :=
      coefficientCount < U(cfg.totalFeatures, b.outputIndexWidth bits)
    when(io.ropeCoefficient.fire) {
      coefficientCount := coefficientCount + 1
    }

    val resultDrainDone = (0 until engines).map { engine =>
      !activeEngineMask(engine) ||
        resultBeatCount(engine) === descriptor.expectedBeats ||
        (io.engineResult(engine).fire &&
          resultBeatCount(engine) === descriptor.expectedBeats - 1)
    }.reduce(_ && _)
    val coefficientDrainDone =
      coefficientCount === U(cfg.totalFeatures, b.outputIndexWidth bits) ||
        (io.ropeCoefficient.fire &&
          coefficientCount === cfg.totalFeatures - 1)
    when(resultDrainDone && coefficientDrainDone) {
      failedCount := failedCount + 1
      completionReg.sequenceId := descriptor.sequenceId
      completionReg.success := False
      completionReg.faultCode := activeFaultCode
      completionReg.emittedFeatures := roundEmittedFeatures
      completionValid := True
      state := State.Idle
    }
  }

  // Stray payloads outside an owned round are accepted and discarded.  This
  // prevents an upstream mailbox from deadlocking while making the violation
  // permanently observable.
  when(state === State.Idle) {
    for (engine <- 0 until engines) {
      io.engineResult(engine).ready := True
      when(io.engineResult(engine).fire) { stickyFault := True }
    }
    io.ropeCoefficient.ready := True
    when(io.ropeCoefficient.fire) { stickyFault := True }
  }
}

object GenerateBitNetQkvClusterResultAssembler extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-qkv-cluster-result-assembler")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetQkvClusterResultAssembler())
}
