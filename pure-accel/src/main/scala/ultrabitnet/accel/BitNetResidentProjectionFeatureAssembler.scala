package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Configuration of the O/Down final-pair to TP4-feature boundary.
  *
  * The two physical arrays have different meanings in the two schedules:
  * prefill shards token lanes while decode shards output rows.  Downstream
  * residual/scaling arithmetic must not have to rediscover that distinction.
  */
case class BitNetResidentProjectionFeatureAssemblerConfig(
    endpoint: BitNetResidentLinearEndpointConfig =
      BitNetResidentLinearEndpointConfig()) {
  val model: BitNetConfig = endpoint.model
  val matrix: BitNetContinuousMatrixConfig = endpoint.matrix
  val engineCount: Int = endpoint.engineCount
  val tokenLanes: Int = endpoint.tokenLanes
  val localTokenLanes: Int = endpoint.localTokenLanes

  require((engineCount == 1 || engineCount == 2) &&
    tokenLanes == engineCount * localTokenLanes && localTokenLanes == 2)
  require(matrix.base.outputLanes == 2)
}

/** One logical feature carrying every active token lane's raw projection. */
case class BitNetResidentProjectionFeature(
    cfg: BitNetResidentProjectionFeatureAssemblerConfig) extends Bundle {
  private val b = cfg.matrix.base

  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val kind = BitNetResidentLinearKind()
  val mode = BitNetMode()
  val sequenceId = UInt(cfg.matrix.sequenceIdWidth bits)
  val featureIndex = UInt(b.outputIndexWidth bits)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val raw = Vec(SInt(b.accumulatorWidth bits), cfg.tokenLanes)
  val activationFactorQ16 = Vec(UInt(32 bits), cfg.tokenLanes)
  val weightScaleQ16 = UInt(32 bits)
  val first = Bool()
  val last = Bool()
}

case class BitNetResidentProjectionAssemblyCompletion(
    cfg: BitNetResidentProjectionFeatureAssemblerConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val kind = BitNetResidentLinearKind()
  val sequenceId = UInt(cfg.matrix.sequenceIdWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
  val acceptedPairs = UInt(32 bits)
  val emittedFeatures = UInt(32 bits)
}

object BitNetResidentProjectionAssemblyFault {
  val None: Int = 0x00
  val InvalidLaunch: Int = 0xe1
  val PairMetadata: Int = 0xe2
  val PairOrder: Int = 0xe3
  val FeatureCount: Int = 0xe4
  val StrayPair: Int = 0xe5
}

/**
  * Reconstitutes a logical TP4 feature stream from the sole matrix island.
  *
  * Prefill waits for the matching result pair from both token-sharded
  * engines, then emits row 0 and row 1 with lanes [0..3].  Decode consumes the
  * two independent output shards without joining them and places the one
  * active token in lane zero.  At most one final-K pair per engine is held;
  * no output-sized reorder RAM or PS-visible packet is introduced.
  *
  * O/Down features feed the fused residual path.  Fused GateUp physical rows
  * retain their 0..2*FFN-1 physical index here and feed the small resident
  * Gate/Up pairing window; this boundary never allocates an output-sized
  * reorder memory.
  *
  * A malformed frame is drained according to the authenticated launch shape
  * and completes with failure.  Partial writes are safe because the enclosing
  * stage controller changes the destination activation-buffer epoch only
  * after a successful completion.
  */
class BitNetResidentProjectionFeatureAssembler(
    cfg: BitNetResidentProjectionFeatureAssemblerConfig =
      BitNetResidentProjectionFeatureAssemblerConfig()) extends Component {
  private val b = cfg.matrix.base
  private val engines = cfg.engineCount
  private val locals = cfg.localTokenLanes

  val io = new Bundle {
    val launch = slave(Stream(BitNetResidentLinearLaunch(cfg.endpoint)))
    val input = slave(Stream(BitNetResidentFinalFeaturePair(cfg.endpoint)))
    val output = master(Stream(BitNetResidentProjectionFeature(cfg)))
    val completion = master(Stream(
      BitNetResidentProjectionAssemblyCompletion(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedFrames = out UInt(32 bits)
    val completedFrames = out UInt(32 bits)
  }

  val descriptor = Reg(BitNetResidentLinearLaunch(cfg.endpoint)) init
    BitNetResidentLinearLaunch(cfg.endpoint).getZero
  val active = RegInit(False)
  val jobFault = RegInit(False)
  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)

  val nextBase = Vec(Reg(UInt(b.outputIndexWidth bits)) init 0, engines)
  val expectedPairs = Vec(Reg(UInt(b.outputIndexWidth bits)) init 0, engines)
  val acceptedPairs = Vec(Reg(UInt(b.outputIndexWidth bits)) init 0, engines)
  val expectedFeatures = Reg(UInt((b.outputIndexWidth + 1) bits)) init 0
  val emittedFeatures = Reg(UInt((b.outputIndexWidth + 1) bits)) init 0

  val pendingValid = Vec(RegInit(False), engines)
  val pendingPayload = Vec(
    Reg(BitNetResidentFinalFeaturePair(cfg.endpoint)) init
      BitNetResidentFinalFeaturePair(cfg.endpoint).getZero,
    engines)
  val emitActive = RegInit(False)
  val engineIndexWidth = Math.max(1, log2Up(engines))
  val emitEngine = Reg(UInt(engineIndexWidth bits)) init 0
  val emitRow = Reg(UInt(log2Up(b.outputLanes) bits)) init 0
  val decodeRoundRobin = Reg(UInt(engineIndexWidth bits)) init 0

  val acceptedFrameCount = Reg(UInt(32 bits)) init 0
  val completedFrameCount = Reg(UInt(32 bits)) init 0
  val acceptedPairCount = Reg(UInt(32 bits)) init 0

  val completionValid = RegInit(False)
  val completionReg = Reg(
    BitNetResidentProjectionAssemblyCompletion(cfg)) init
      BitNetResidentProjectionAssemblyCompletion(cfg).getZero
  io.completion.valid := completionValid
  io.completion.payload := completionReg
  when(io.completion.fire) { completionValid := False }

  val launchKindLegal =
    io.launch.kind === BitNetResidentLinearKind.AttentionOutput ||
      io.launch.kind === BitNetResidentLinearKind.GateUp ||
      io.launch.kind === BitNetResidentLinearKind.Down
  val launchProtocolLegal = BitNetResidentLinearProtocol.launchLegal(
    io.launch.payload, cfg.endpoint)
  val launchLegal = launchKindLegal && launchProtocolLegal
  io.launch.ready := !active && !completionValid

  val launchSpans = Vec(UInt((b.outputIndexWidth + 1) bits), engines)
  val launchPairCounts = Vec(UInt(b.outputIndexWidth bits), engines)
  for (engine <- 0 until engines) {
    launchSpans(engine) :=
      io.launch.matrixTask.outputEndExclusive(engine).resize(
        b.outputIndexWidth + 1) -
        io.launch.matrixTask.outputBase(engine).resize(
          b.outputIndexWidth + 1)
    launchPairCounts(engine) :=
      ((launchSpans(engine) + 1) >> 1).resize(b.outputIndexWidth)
  }
  val launchExpectedFeatures = UInt((b.outputIndexWidth + 1) bits)
  launchExpectedFeatures := launchSpans.reduce(_ + _)
  when(io.launch.mode === BitNetMode.Prefill) {
    launchExpectedFeatures := launchSpans(0)
  }

  when(io.launch.fire) {
    descriptor := io.launch.payload
    active := launchLegal
    jobFault := False
    jobFaultCode := 0
    emittedFeatures := 0
    acceptedPairCount := 0
    expectedFeatures := launchExpectedFeatures
    emitActive := False
    emitEngine := 0
    emitRow := 0
    decodeRoundRobin := 0
    for (engine <- 0 until engines) {
      nextBase(engine) := io.launch.matrixTask.outputBase(engine)
      expectedPairs(engine) := launchPairCounts(engine)
      acceptedPairs(engine) := 0
      pendingValid(engine) := False
    }
    when(launchLegal) {
      acceptedFrameCount := acceptedFrameCount + 1
    } otherwise {
      stickyFault := True
      completionReg.requestId := io.launch.requestId
      completionReg.layer := io.launch.layer
      completionReg.kind := io.launch.kind
      completionReg.sequenceId := io.launch.sequenceId
      completionReg.success := False
      completionReg.faultCode :=
        B(BitNetResidentProjectionAssemblyFault.InvalidLaunch, 8 bits)
      completionReg.acceptedPairs := 0
      completionReg.emittedFeatures := 0
      completionValid := True
    }
  }

  val prefill = descriptor.mode === BitNetMode.Prefill
  val activeEngineMask = Bits(engines bits)
  activeEngineMask := B((BigInt(1) << engines) - 1, engines bits)
  when(prefill) {
    for (engine <- 0 until engines) {
      activeEngineMask(engine) := descriptor.tokenMask(
        engine * locals + locals - 1 downto engine * locals).orR
    }
  }

  val inputEngine = io.input.engine
  val inputEngineActive = Bool()
  inputEngineActive := activeEngineMask(0)
  for (engine <- 1 until engines) {
    when(inputEngine === engine) {
      inputEngineActive := activeEngineMask(engine)
    }
  }
  val expectedLocalMask = Bits(cfg.tokenLanes bits)
  expectedLocalMask := B(1, cfg.tokenLanes bits)
  when(prefill) {
    for (engine <- 0 until engines) {
      when(inputEngine === engine) {
        expectedLocalMask := (descriptor.tokenMask(
          engine * locals + locals - 1 downto engine * locals)
          .resize(cfg.tokenLanes) << (engine * locals)).resize(cfg.tokenLanes)
      }
    }
  }

  val expectedRowMask = Bits(b.outputLanes bits)
  val selectedNextBase = UInt(b.outputIndexWidth bits)
  val selectedEndExclusive = UInt(b.outputIndexWidth bits)
  val selectedWeightScale = UInt(32 bits)
  val selectedAcceptedPairs = UInt(b.outputIndexWidth bits)
  val selectedExpectedPairs = UInt(b.outputIndexWidth bits)
  selectedNextBase := nextBase(0)
  selectedEndExclusive := descriptor.matrixTask.outputEndExclusive(0)
  selectedWeightScale := descriptor.matrixTask.weightScaleQ16(0)
  selectedAcceptedPairs := acceptedPairs(0)
  selectedExpectedPairs := expectedPairs(0)
  for (engine <- 1 until engines) {
    when(inputEngine === engine) {
      selectedNextBase := nextBase(engine)
      selectedEndExclusive := descriptor.matrixTask.outputEndExclusive(engine)
      selectedWeightScale := descriptor.matrixTask.weightScaleQ16(engine)
      selectedAcceptedPairs := acceptedPairs(engine)
      selectedExpectedPairs := expectedPairs(engine)
    }
  }
  for (row <- 0 until b.outputLanes) {
    expectedRowMask(row) :=
      selectedNextBase.resize(b.outputIndexWidth + 1) + row <
        selectedEndExclusive.resize(b.outputIndexWidth + 1)
  }
  val inputIdentityLegal =
    io.input.requestId === descriptor.requestId &&
      io.input.layer === descriptor.layer &&
      io.input.kind === descriptor.kind &&
      io.input.mode === descriptor.mode &&
      io.input.sequenceId === descriptor.sequenceId &&
      io.input.weightScaleQ16 === selectedWeightScale &&
      io.input.globalTokenLaneMask === expectedLocalMask &&
      io.input.outputLaneMask === expectedRowMask
  val inputOrderLegal =
    io.input.physicalOutputBase === selectedNextBase &&
      selectedAcceptedPairs < selectedExpectedPairs
  val inputLegal = inputEngineActive && inputIdentityLegal && inputOrderLegal

  io.input.ready := False
  when(!active) {
    // Never let a stale island result deadlock a shared result path.
    io.input.ready := True
  } elsewhen(jobFault) {
    io.input.ready := True
  } otherwise {
    io.input.ready := !pendingValid(0)
    for (engine <- 1 until engines) {
      when(inputEngine === engine) {
        io.input.ready := !pendingValid(engine)
      }
    }
  }

  when(io.input.fire) {
    when(!active) {
      stickyFault := True
    } otherwise {
      acceptedPairCount := acceptedPairCount + 1
      when(inputEngineActive &&
          selectedAcceptedPairs < selectedExpectedPairs) {
        for (engine <- 0 until engines) {
          when(inputEngine === engine) {
            acceptedPairs(engine) := acceptedPairs(engine) + 1
            nextBase(engine) := nextBase(engine) + b.outputLanes
          }
        }
      }
      when(!jobFault && inputLegal) {
        for (engine <- 0 until engines) {
          when(inputEngine === engine) {
            pendingPayload(engine) := io.input.payload
            pendingValid(engine) := True
          }
        }
      } elsewhen(!jobFault) {
        jobFault := True
        stickyFault := True
        jobFaultCode := B(
          BitNetResidentProjectionAssemblyFault.PairMetadata, 8 bits)
        when(inputIdentityLegal && !inputOrderLegal) {
          jobFaultCode := B(
            BitNetResidentProjectionAssemblyFault.PairOrder, 8 bits)
        }
        pendingValid.foreach(_ := False)
        emitActive := False
      }
    }
  }

  val allPrefillPending = (0 until engines).map { engine =>
    !activeEngineMask(engine) || pendingValid(engine)
  }.reduce(_ && _)
  val prefillPairMatches = if (engines == 1) True else
    !activeEngineMask(1) ||
      (pendingPayload(0).physicalOutputBase ===
        pendingPayload(1).physicalOutputBase &&
        pendingPayload(0).outputLaneMask === pendingPayload(1).outputLaneMask &&
        pendingPayload(0).weightScaleQ16 === pendingPayload(1).weightScaleQ16)

  when(active && !jobFault && !emitActive) {
    when(prefill && allPrefillPending) {
      when(prefillPairMatches) {
        emitActive := True
        emitEngine := 0
        emitRow := 0
      } otherwise {
        jobFault := True
        stickyFault := True
        jobFaultCode := B(
          BitNetResidentProjectionAssemblyFault.PairOrder, 8 bits)
        pendingValid.foreach(_ := False)
      }
    } elsewhen(!prefill && pendingValid.asBits.orR) {
      emitEngine := 0
      if (engines > 1) {
        when(!pendingValid(0) ||
            (pendingValid(1) && decodeRoundRobin === 1)) {
          emitEngine := 1
        }
      }
      emitRow := 0
      emitActive := True
    }
  }

  val selectedPair = BitNetResidentFinalFeaturePair(cfg.endpoint)
  selectedPair := pendingPayload(0)
  for (engine <- 1 until engines) {
    when(emitEngine === engine) { selectedPair := pendingPayload(engine) }
  }
  val selectedBase = UInt(b.outputIndexWidth bits)
  selectedBase := selectedPair.physicalOutputBase
  when(prefill) { selectedBase := pendingPayload(0).physicalOutputBase }

  io.output.valid := active && !jobFault && emitActive &&
    selectedPair.outputLaneMask(emitRow)
  io.output.requestId := descriptor.requestId
  io.output.layer := descriptor.layer
  io.output.kind := descriptor.kind
  io.output.mode := descriptor.mode
  io.output.sequenceId := descriptor.sequenceId
  io.output.featureIndex := selectedBase + emitRow
  io.output.tokenMask := descriptor.tokenMask
  io.output.raw.foreach(_ := 0)
  io.output.activationFactorQ16.foreach(_ := 0)
  io.output.weightScaleQ16 := selectedPair.weightScaleQ16
  io.output.first := emittedFeatures === 0
  io.output.last := emittedFeatures === expectedFeatures - 1

  for (globalLane <- 0 until cfg.tokenLanes) {
    val engine = globalLane / locals
    val local = globalLane % locals
    when(prefill && descriptor.tokenMask(globalLane)) {
      io.output.raw(globalLane) := pendingPayload(engine).rawRows(local)(emitRow)
      io.output.activationFactorQ16(globalLane) :=
        pendingPayload(engine).activationFactorQ16(local)
    }
  }
  when(!prefill) {
    io.output.raw(0) := selectedPair.rawRows(0)(emitRow)
    io.output.activationFactorQ16(0) := selectedPair.activationFactorQ16(0)
  }

  when(io.output.fire) {
    emittedFeatures := emittedFeatures + 1
    val hasSecondRow = emitRow === 0 && selectedPair.outputLaneMask(1)
    when(hasSecondRow) {
      emitRow := 1
    } otherwise {
      emitActive := False
      when(prefill) {
        for (engine <- 0 until engines) {
          when(activeEngineMask(engine)) { pendingValid(engine) := False }
        }
      } otherwise {
        for (engine <- 0 until engines) {
          when(emitEngine === engine) { pendingValid(engine) := False }
        }
        decodeRoundRobin := emitEngine + 1
      }
    }
  }

  val allPairsAccepted = (0 until engines).map { engine =>
    !activeEngineMask(engine) || acceptedPairs(engine) === expectedPairs(engine)
  }.reduce(_ && _)
  val frameDrained = allPairsAccepted && !pendingValid.asBits.orR &&
    !emitActive
  when(active && frameDrained && !completionValid) {
    completionReg.requestId := descriptor.requestId
    completionReg.layer := descriptor.layer
    completionReg.kind := descriptor.kind
    completionReg.sequenceId := descriptor.sequenceId
    completionReg.success := !jobFault &&
      emittedFeatures === expectedFeatures
    completionReg.faultCode := jobFaultCode
    when(!jobFault && emittedFeatures =/= expectedFeatures) {
      completionReg.faultCode := B(
        BitNetResidentProjectionAssemblyFault.FeatureCount, 8 bits)
      stickyFault := True
    }
    completionReg.acceptedPairs := acceptedPairCount
    completionReg.emittedFeatures := emittedFeatures.resized
    completionValid := True
    completedFrameCount := completedFrameCount + 1
    active := False
  }

  io.busy := active || completionValid || emitActive || pendingValid.asBits.orR
  io.protocolFaultSticky := stickyFault
  io.acceptedFrames := acceptedFrameCount
  io.completedFrames := completedFrameCount
}

object GenerateBitNetResidentProjectionFeatureAssembler extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-projection-feature-assembler")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentProjectionFeatureAssembler())
}
