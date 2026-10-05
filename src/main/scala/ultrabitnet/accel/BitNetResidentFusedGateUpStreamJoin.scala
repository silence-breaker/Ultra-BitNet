package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Configuration of the resident, compact Gate/Up pairing boundary.
  *
  * The production image stores complete four-row Map0 groups in the order
  *
  *   Gate[0..3], Up[0..3], Gate[4..7], Up[4..7], ...
  *
  * and therefore contains exactly twice the semantic FFN feature count.  Two
  * four-feature windows are sufficient: prefill uses one monotonically, while
  * decode may interleave the independently monotonic lower and upper physical
  * halves.  No activation-sized RAM is owned by this block.
  */
case class BitNetResidentFusedGateUpStreamJoinConfig(
    projection: BitNetResidentProjectionFeatureAssemblerConfig =
      BitNetResidentProjectionFeatureAssemblerConfig(),
    gate: BitNetSharedSpuFfnGateConfig =
      BitNetSharedSpuFfnGateConfig()) {
  val model: BitNetConfig = projection.model
  val endpoint: BitNetResidentLinearEndpointConfig = projection.endpoint
  val tokenLanes: Int = model.tokenParallelism
  val semanticFeatures: Int = model.ffnSize
  val physicalFeatures: Int = 2 * semanticFeatures
  val rowsPerGroup: Int = 4
  val semanticGroups: Int = semanticFeatures / rowsPerGroup
  val windowCount: Int = 2
  val physicalCountWidth: Int = log2Up(physicalFeatures + 1)
  val semanticGroupWidth: Int = log2Up(semanticGroups)

  require(gate.model == model,
    "the projection boundary and FFN gate must share one model geometry")
  require(gate.maxLength == semanticFeatures)
  require((tokenLanes == 2 || tokenLanes == 4) &&
    rowsPerGroup == 4 && windowCount == 2)
  require(semanticFeatures == 6912 && physicalFeatures == 13824,
    "the production fused GateUp image is fixed at 13824 physical rows")
  require(semanticFeatures % rowsPerGroup == 0)
  require(projection.matrix.base.accumulatorWidth == gate.dataWidth)
  require(projection.matrix.base.outputIndexWidth >=
    log2Up(physicalFeatures + 1))
}

case class BitNetResidentFusedGateUpJoinCompletion(
    cfg: BitNetResidentFusedGateUpStreamJoinConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val mode = BitNetMode()
  val sequenceId = UInt(cfg.projection.matrix.sequenceIdWidth bits)
  val success = Bool()
  val faultCode = Bits(8 bits)
  val acceptedPhysicalFeatures = UInt(32 bits)
  val emittedSemanticFeatures = UInt(32 bits)
}

object BitNetResidentFusedGateUpJoinFault {
  val None: Int = 0x00
  val InvalidLaunch: Int = 0xd1
  val Identity: Int = 0xd2
  val PhysicalRange: Int = 0xd3
  val PhysicalOrder: Int = 0xd4
  val Duplicate: Int = 0xd5
  val Framing: Int = 0xd6
  val FeatureCount: Int = 0xd7
  val StrayFeature: Int = 0xd8
}

/**
  * Pairs the interleaved resident Gate/Up physical result stream.
  *
  * Physical row `p` maps to semantic row
  * `((p >> 3) << 2) | p(1 downto 0)`; `p(2)` selects Gate/Up.  The output
  * address is non-monotonic in decode, but follows one fixed schedule:
  * lower group 0, upper group 0, lower group 1, upper group 1, ... .  Its
  * first/last flags describe that authenticated output ordinal.  The fixed
  * permutation lets the production gate prove one-to-one coverage without a
  * feature-sized seen RAM.
  *
  * Once a legal launch has been accepted, any malformed feature moves the
  * block into a fail-closed drain.  Exactly the launch-authenticated 13824
  * physical transfers are retired before failure completion, so a corrupt
  * producer cannot strand the model-wide matrix result path.
  */
class BitNetResidentFusedGateUpStreamJoin(
    cfg: BitNetResidentFusedGateUpStreamJoinConfig =
      BitNetResidentFusedGateUpStreamJoinConfig()) extends Component {
  private val p = cfg.projection
  private val b = p.matrix.base
  private val rows = cfg.rowsPerGroup
  private val windows = cfg.windowCount

  val io = new Bundle {
    val launch = slave(Stream(BitNetResidentLinearLaunch(cfg.endpoint)))
    val input = slave(Stream(BitNetResidentProjectionFeature(p)))
    val output = master(Stream(BitNetFusedGateUpBeat(cfg.gate)))
    val completion = master(Stream(
      BitNetResidentFusedGateUpJoinCompletion(cfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedFrames = out UInt(32 bits)
    val completedFrames = out UInt(32 bits)
    val failedFrames = out UInt(32 bits)
    val acceptedPhysicalFeatures = out UInt(32 bits)
    val emittedSemanticFeatures = out UInt(32 bits)
  }

  val descriptor = Reg(BitNetResidentLinearLaunch(cfg.endpoint)) init
    BitNetResidentLinearLaunch(cfg.endpoint).getZero
  val active = RegInit(False)
  val jobFault = RegInit(False)
  val jobFaultCode = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)

  val acceptedPhysical = Reg(UInt(cfg.physicalCountWidth bits)) init 0
  val emittedSemantic = Reg(UInt(cfg.gate.lengthWidth bits)) init 0
  // The two counters authenticate the independent decode halves.  Prefill
  // uses only counter zero over the complete physical range.
  val nextPhysical = Vec(
    Reg(UInt(b.outputIndexWidth bits)) init 0,
    cfg.endpoint.engineCount)

  val windowValid = Vec(RegInit(False), windows)
  val windowGroup = Vec(
    Reg(UInt(cfg.semanticGroupWidth bits)) init 0, windows)
  val windowGateValid = Vec(Reg(Bits(rows bits)) init 0, windows)
  val windowUpValid = Vec(Reg(Bits(rows bits)) init 0, windows)
  val windowGateRaw = Vec((0 until windows).map { _ =>
    Vec((0 until rows).map { _ =>
      Vec((0 until cfg.tokenLanes).map { _ =>
        Reg(SInt(cfg.gate.dataWidth bits)) init 0
      })
    })
  })
  val windowUpRaw = Vec((0 until windows).map { _ =>
    Vec((0 until rows).map { _ =>
      Vec((0 until cfg.tokenLanes).map { _ =>
        Reg(SInt(cfg.gate.dataWidth bits)) init 0
      })
    })
  })

  val emitActive = RegInit(False)
  val emitWindow = Reg(UInt(log2Up(windows) bits)) init 0
  val emitLane = Reg(UInt(log2Up(rows) bits)) init 0

  val acceptedFrameCount = Reg(UInt(32 bits)) init 0
  val completedFrameCount = Reg(UInt(32 bits)) init 0
  val failedFrameCount = Reg(UInt(32 bits)) init 0
  val acceptedPhysicalCount = Reg(UInt(32 bits)) init 0
  val emittedSemanticCount = Reg(UInt(32 bits)) init 0

  val completionValid = RegInit(False)
  val completionReg = Reg(
    BitNetResidentFusedGateUpJoinCompletion(cfg)) init
      BitNetResidentFusedGateUpJoinCompletion(cfg).getZero
  io.completion.valid := completionValid
  io.completion.payload := completionReg
  when(io.completion.fire) { completionValid := False }

  val task = io.launch.matrixTask
  val launchPrefill = io.launch.mode === BitNetMode.Prefill
  val launchDecode = io.launch.mode === BitNetMode.Decode
  val exactPrefillSpan = (0 until cfg.endpoint.engineCount).map { engine =>
    task.outputBase(engine) === 0 &&
      task.outputEndExclusive(engine) === cfg.physicalFeatures
  }.reduce(_ && _)
  val exactDecodeSpan = if (cfg.endpoint.engineCount == 1) {
    task.outputBase(0) === 0 &&
      task.outputEndExclusive(0) === cfg.physicalFeatures
  } else {
    task.outputBase(0) === 0 &&
      task.outputEndExclusive(0) === cfg.semanticFeatures &&
      task.outputBase(1) === cfg.semanticFeatures &&
      task.outputEndExclusive(1) === cfg.physicalFeatures
  }
  val launchLegal =
    io.launch.kind === BitNetResidentLinearKind.GateUp &&
      BitNetResidentLinearProtocol.launchLegal(
        io.launch.payload, cfg.endpoint) &&
      ((launchPrefill && exactPrefillSpan) ||
        (launchDecode && exactDecodeSpan))

  io.launch.ready := !active && !completionValid
  when(io.launch.fire) {
    when(launchLegal) {
      descriptor := io.launch.payload
      active := True
      jobFault := False
      jobFaultCode := 0
      acceptedPhysical := 0
      emittedSemantic := 0
      nextPhysical(0) := 0
      if (cfg.endpoint.engineCount > 1)
        nextPhysical(1) := U(cfg.semanticFeatures, b.outputIndexWidth bits)
      windowValid.foreach(_ := False)
      windowGateValid.foreach(_ := B(0, rows bits))
      windowUpValid.foreach(_ := B(0, rows bits))
      emitActive := False
      emitWindow := 0
      emitLane := 0
      acceptedFrameCount := acceptedFrameCount + 1
    } otherwise {
      stickyFault := True
      failedFrameCount := failedFrameCount + 1
      completionReg.requestId := io.launch.requestId
      completionReg.layer := io.launch.layer
      completionReg.mode := io.launch.mode
      completionReg.sequenceId := io.launch.sequenceId
      completionReg.success := False
      completionReg.faultCode :=
        B(BitNetResidentFusedGateUpJoinFault.InvalidLaunch, 8 bits)
      completionReg.acceptedPhysicalFeatures := 0
      completionReg.emittedSemanticFeatures := 0
      completionValid := True
    }
  }

  val physicalIndex = io.input.featureIndex
  val physicalInRange = physicalIndex < cfg.physicalFeatures
  val physicalUpperHalf = physicalIndex >= cfg.semanticFeatures
  val physicalStream = UInt(
    Math.max(1, log2Up(cfg.endpoint.engineCount)) bits)
  physicalStream := 0
  if (cfg.endpoint.engineCount > 1) {
    when(descriptor.mode === BitNetMode.Decode && physicalUpperHalf) {
      physicalStream := 1
    }
  }
  val semanticGroup = (physicalIndex >> 3).resize(
    cfg.semanticGroupWidth)
  val physicalLane = physicalIndex(1 downto 0)
  val isUp = physicalIndex(2)

  val matchWindow = Bits(windows bits)
  val freeWindow = Bits(windows bits)
  for (window <- 0 until windows) {
    matchWindow(window) := windowValid(window) &&
      windowGroup(window) === semanticGroup
    freeWindow(window) := !windowValid(window)
  }
  val selectedWindow = UInt(log2Up(windows) bits)
  selectedWindow := 0
  when(!matchWindow(0) &&
      (matchWindow(1) || (!freeWindow(0) && freeWindow(1)))) {
    selectedWindow := 1
  }
  val hasWindow = matchWindow.orR || freeWindow.orR
  val selectedMatches = matchWindow(selectedWindow)
  val selectedEmitting = emitActive && emitWindow === selectedWindow
  val duplicate = selectedMatches &&
    ((!isUp && windowGateValid(selectedWindow)(physicalLane)) ||
      (isUp && windowUpValid(selectedWindow)(physicalLane)))

  val expectedTokenMask = descriptor.tokenMask
  val expectedWeightScale = UInt(32 bits)
  expectedWeightScale := descriptor.matrixTask.weightScaleQ16(0)
  if (cfg.endpoint.engineCount > 1) {
    when(descriptor.mode === BitNetMode.Decode && physicalUpperHalf) {
      expectedWeightScale := descriptor.matrixTask.weightScaleQ16(1)
    }
  }
  val laneMetadataLegal = Vec(Bool(), cfg.tokenLanes)
  for (lane <- 0 until cfg.tokenLanes) {
    laneMetadataLegal(lane) :=
      (expectedTokenMask(lane) &&
        io.input.activationFactorQ16(lane) ===
          descriptor.matrixTask.activationFactorQ16(lane)) ||
      (!expectedTokenMask(lane) &&
        io.input.activationFactorQ16(lane) === 0 &&
        io.input.raw(lane) === 0)
  }
  val identityLegal =
    io.input.requestId === descriptor.requestId &&
      io.input.layer === descriptor.layer &&
      io.input.kind === BitNetResidentLinearKind.GateUp &&
      io.input.mode === descriptor.mode &&
      io.input.sequenceId === descriptor.sequenceId &&
      io.input.tokenMask === expectedTokenMask &&
      io.input.weightScaleQ16 === expectedWeightScale &&
      laneMetadataLegal.asBits.andR
  val selectedNextPhysical = UInt(b.outputIndexWidth bits)
  selectedNextPhysical := nextPhysical(0)
  if (cfg.endpoint.engineCount > 1) {
    when(physicalStream === 1) { selectedNextPhysical := nextPhysical(1) }
  }
  val physicalOrderLegal = physicalInRange &&
    physicalIndex === selectedNextPhysical
  val framingLegal =
    io.input.first === (acceptedPhysical === 0) &&
      io.input.last ===
        (acceptedPhysical === cfg.physicalFeatures - 1)
  val inputLegal = identityLegal && physicalInRange &&
    physicalOrderLegal && framingLegal && !duplicate

  // Illegal traffic is always accepted into the bounded failure drain.  This
  // is important when both small windows are occupied by partial groups: a
  // malformed third group must report a fault rather than deadlock upstream.
  val storageAvailable = hasWindow && !selectedEmitting
  io.input.ready := False
  when(!active) {
    io.input.ready := True
  } elsewhen(acceptedPhysical < cfg.physicalFeatures) {
    io.input.ready := jobFault || !inputLegal || storageAvailable
  }

  when(io.input.fire) {
    when(!active) {
      stickyFault := True
    } otherwise {
      acceptedPhysical := acceptedPhysical + 1
      acceptedPhysicalCount := acceptedPhysicalCount + 1
      when(!jobFault && inputLegal) {
        for (engine <- 0 until cfg.endpoint.engineCount) {
          when(physicalStream === engine) {
            nextPhysical(engine) := nextPhysical(engine) + 1
          }
        }

        when(!selectedMatches) {
          windowValid(selectedWindow) := True
          windowGroup(selectedWindow) := semanticGroup
          windowGateValid(selectedWindow) := B(0, rows bits)
          windowUpValid(selectedWindow) := B(0, rows bits)
        }
        when(isUp) {
          windowUpValid(selectedWindow)(physicalLane) := True
          for (lane <- 0 until cfg.tokenLanes) {
            windowUpRaw(selectedWindow)(physicalLane)(lane) :=
              io.input.raw(lane)
          }
        } otherwise {
          windowGateValid(selectedWindow)(physicalLane) := True
          for (lane <- 0 until cfg.tokenLanes) {
            windowGateRaw(selectedWindow)(physicalLane)(lane) :=
              io.input.raw(lane)
          }
        }
      } elsewhen(!jobFault) {
        jobFault := True
        stickyFault := True
        windowValid.foreach(_ := False)
        windowGateValid.foreach(_ := B(0, rows bits))
        windowUpValid.foreach(_ := B(0, rows bits))
        emitActive := False
        jobFaultCode :=
          B(BitNetResidentFusedGateUpJoinFault.Identity, 8 bits)
        when(identityLegal && !physicalInRange) {
          jobFaultCode :=
            B(BitNetResidentFusedGateUpJoinFault.PhysicalRange, 8 bits)
        } elsewhen(identityLegal && duplicate) {
          jobFaultCode :=
            B(BitNetResidentFusedGateUpJoinFault.Duplicate, 8 bits)
        } elsewhen(identityLegal && physicalInRange &&
            !physicalOrderLegal) {
          jobFaultCode :=
            B(BitNetResidentFusedGateUpJoinFault.PhysicalOrder, 8 bits)
        } elsewhen(identityLegal && physicalInRange &&
            physicalOrderLegal && !framingLegal) {
          jobFaultCode :=
            B(BitNetResidentFusedGateUpJoinFault.Framing, 8 bits)
        }
      }
    }
  }

  val windowComplete = Bits(windows bits)
  for (window <- 0 until windows) {
    windowComplete(window) := windowValid(window) &&
      windowGateValid(window).andR && windowUpValid(window).andR
  }

  val semanticOrdinalGroup = (emittedSemantic >> 2).resize(
    cfg.semanticGroupWidth)
  val expectedEmitGroup = UInt(cfg.semanticGroupWidth bits)
  expectedEmitGroup := semanticOrdinalGroup
  if (cfg.endpoint.engineCount > 1) {
    when(descriptor.mode === BitNetMode.Decode) {
      expectedEmitGroup := (semanticOrdinalGroup >> 1).resized
      when(semanticOrdinalGroup(0)) {
        expectedEmitGroup := ((semanticOrdinalGroup >> 1) +
          U(cfg.semanticGroups / 2, cfg.semanticGroupWidth bits)).resized
      }
    }
  }
  val expectedWindowComplete = Bits(windows bits)
  for (window <- 0 until windows) {
    expectedWindowComplete(window) := windowComplete(window) &&
      windowGroup(window) === expectedEmitGroup
  }
  when(active && !jobFault && !emitActive &&
    expectedWindowComplete.orR) {
    emitActive := True
    emitWindow := 0
    when(!expectedWindowComplete(0) && expectedWindowComplete(1)) {
      emitWindow := 1
    }
    emitLane := 0
  }

  io.output.valid := active && !jobFault && emitActive
  io.output.tokenMask := descriptor.tokenMask
  io.output.featureIndex :=
    ((windowGroup(emitWindow) << 2) + emitLane).resize(cfg.gate.indexWidth)
  io.output.first := emittedSemantic === 0
  io.output.last := emittedSemantic === cfg.semanticFeatures - 1
  for (lane <- 0 until cfg.tokenLanes) {
    io.output.gateRaw(lane) := windowGateRaw(emitWindow)(emitLane)(lane)
    io.output.upRaw(lane) := windowUpRaw(emitWindow)(emitLane)(lane)
  }

  when(io.output.fire) {
    emittedSemantic := emittedSemantic + 1
    emittedSemanticCount := emittedSemanticCount + 1
    when(emitLane === rows - 1) {
      windowValid(emitWindow) := False
      windowGateValid(emitWindow) := B(0, rows bits)
      windowUpValid(emitWindow) := B(0, rows bits)
      emitActive := False
      emitLane := 0
    } otherwise {
      emitLane := emitLane + 1
    }
  }

  val windowsEmpty = !windowValid.asBits.orR && !emitActive
  val physicalFrameAccepted =
    acceptedPhysical === cfg.physicalFeatures
  when(active && physicalFrameAccepted && !completionValid) {
    when(jobFault) {
      completionReg.requestId := descriptor.requestId
      completionReg.layer := descriptor.layer
      completionReg.mode := descriptor.mode
      completionReg.sequenceId := descriptor.sequenceId
      completionReg.success := False
      completionReg.faultCode := jobFaultCode
      completionReg.acceptedPhysicalFeatures := acceptedPhysical.resized
      completionReg.emittedSemanticFeatures := emittedSemantic.resized
      completionValid := True
      failedFrameCount := failedFrameCount + 1
      active := False
    } elsewhen(windowsEmpty) {
      completionReg.requestId := descriptor.requestId
      completionReg.layer := descriptor.layer
      completionReg.mode := descriptor.mode
      completionReg.sequenceId := descriptor.sequenceId
      completionReg.success := emittedSemantic === cfg.semanticFeatures
      completionReg.faultCode := 0
      when(emittedSemantic =/= cfg.semanticFeatures) {
        completionReg.faultCode :=
          B(BitNetResidentFusedGateUpJoinFault.FeatureCount, 8 bits)
        stickyFault := True
        failedFrameCount := failedFrameCount + 1
      } otherwise {
        completedFrameCount := completedFrameCount + 1
      }
      completionReg.acceptedPhysicalFeatures := acceptedPhysical.resized
      completionReg.emittedSemanticFeatures := emittedSemantic.resized
      completionValid := True
      active := False
    }
  }

  io.busy := active || completionValid || emitActive ||
    windowValid.asBits.orR
  io.protocolFaultSticky := stickyFault
  io.acceptedFrames := acceptedFrameCount
  io.completedFrames := completedFrameCount
  io.failedFrames := failedFrameCount
  io.acceptedPhysicalFeatures := acceptedPhysicalCount
  io.emittedSemanticFeatures := emittedSemanticCount
}

object GenerateBitNetResidentFusedGateUpStreamJoin extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-fused-gate-up-stream-join")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentFusedGateUpStreamJoin())
}
