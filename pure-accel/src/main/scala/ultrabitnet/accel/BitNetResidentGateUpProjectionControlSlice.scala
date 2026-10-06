package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Fault namespace for the three arithmetic retirements of fused GateUp. */
object BitNetResidentGateUpComputeBarrierFault {
  val LinearIdentity = 0xb1
  val LinearFailed = 0xb2
  val AssemblyIdentity = 0xb3
  val AssemblyFailed = 0xb4
  val JoinIdentity = 0xb5
  val JoinFailed = 0xb6
}

/**
  * Authenticates and joins the three ordered arithmetic retirements which
  * belong to one fused GateUp matrix transaction.
  *
  * The matrix endpoint may finish before the feature assembler and the small
  * Gate/Up pairing windows have drained.  Conversely, a downstream fault must
  * not release the matrix lease or the resident weight slots early.  This
  * scoreboard therefore consumes every side exactly once and reports only
  * after all three sides have retired, even if the first side already failed.
  */
class BitNetResidentGateUpComputeBarrier(
    weightCfg: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig(),
    joinCfg: BitNetResidentFusedGateUpStreamJoinConfig =
      BitNetResidentFusedGateUpStreamJoinConfig()) extends Component {
  private val stageCfg = weightCfg.stage
  private val assemblerCfg = joinCfg.projection

  require(assemblerCfg.model == stageCfg.model)
  require(assemblerCfg.endpoint.feeder == stageCfg.feeder)

  val io = new Bundle {
    val begin = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(stageCfg)))
    val linearCompletion = slave(Stream(
      BitNetResidentMatrixProjectionCompletion(stageCfg)))
    val assemblyCompletion = slave(Stream(
      BitNetResidentProjectionAssemblyCompletion(assemblerCfg)))
    val joinCompletion = slave(Stream(
      BitNetResidentFusedGateUpJoinCompletion(joinCfg)))
    val completion = master(Stream(
      BitNetResidentMatrixProjectionCompletion(stageCfg)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val active = RegInit(False)
  val held = Reg(BitNetResidentMatrixProjectionLaunch(stageCfg)) init
    BitNetResidentMatrixProjectionLaunch(stageCfg).getZero

  val linearDone = RegInit(False)
  val assemblyDone = RegInit(False)
  val joinDone = RegInit(False)
  val linearOk = RegInit(False)
  val assemblyOk = RegInit(False)
  val joinOk = RegInit(False)
  val linearFault = Reg(Bits(8 bits)) init 0
  val assemblyFault = Reg(Bits(8 bits)) init 0
  val joinFault = Reg(Bits(8 bits)) init 0
  val stickyFault = RegInit(False)

  val completionValid = RegInit(False)
  val completionReg = Reg(
    BitNetResidentMatrixProjectionCompletion(stageCfg)) init
      BitNetResidentMatrixProjectionCompletion(stageCfg).getZero

  io.begin.ready := !active && !completionValid
  when(io.begin.fire) {
    held := io.begin.payload
    active := True
    linearDone := False
    assemblyDone := False
    joinDone := False
    linearOk := False
    assemblyOk := False
    joinOk := False
    linearFault := 0
    assemblyFault := 0
    joinFault := 0
  }

  val linearIdentityMatches =
    io.linearCompletion.payload.requestId === held.requestId &&
      io.linearCompletion.payload.stepId === held.stepId &&
      io.linearCompletion.payload.stage === held.stage &&
      io.linearCompletion.payload.layer === held.layer &&
      io.linearCompletion.payload.part === held.part &&
      io.linearCompletion.payload.sequenceId === held.sequenceId
  io.linearCompletion.ready := active && !linearDone
  when(io.linearCompletion.fire) {
    linearDone := True
    linearOk := io.linearCompletion.payload.success &&
      linearIdentityMatches
    linearFault := io.linearCompletion.payload.faultCode
    when(!linearIdentityMatches) {
      linearFault := B(
        BitNetResidentGateUpComputeBarrierFault.LinearIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.linearCompletion.payload.success) {
      when(io.linearCompletion.payload.faultCode === 0) {
        linearFault := B(
          BitNetResidentGateUpComputeBarrierFault.LinearFailed, 8 bits)
      }
      stickyFault := True
    }
  }

  val assemblyIdentityMatches =
    io.assemblyCompletion.payload.requestId === held.requestId &&
      io.assemblyCompletion.payload.layer === held.layer &&
      io.assemblyCompletion.payload.kind === BitNetResidentLinearKind.GateUp &&
      io.assemblyCompletion.payload.sequenceId === held.sequenceId
  io.assemblyCompletion.ready := active && !assemblyDone
  when(io.assemblyCompletion.fire) {
    assemblyDone := True
    assemblyOk := io.assemblyCompletion.payload.success &&
      assemblyIdentityMatches
    assemblyFault := io.assemblyCompletion.payload.faultCode
    when(!assemblyIdentityMatches) {
      assemblyFault := B(
        BitNetResidentGateUpComputeBarrierFault.AssemblyIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.assemblyCompletion.payload.success) {
      when(io.assemblyCompletion.payload.faultCode === 0) {
        assemblyFault := B(
          BitNetResidentGateUpComputeBarrierFault.AssemblyFailed, 8 bits)
      }
      stickyFault := True
    }
  }

  val joinIdentityMatches =
    io.joinCompletion.payload.requestId === held.requestId &&
      io.joinCompletion.payload.layer === held.layer &&
      io.joinCompletion.payload.mode === held.mode &&
      io.joinCompletion.payload.sequenceId === held.sequenceId
  io.joinCompletion.ready := active && !joinDone
  when(io.joinCompletion.fire) {
    joinDone := True
    joinOk := io.joinCompletion.payload.success && joinIdentityMatches
    joinFault := io.joinCompletion.payload.faultCode
    when(!joinIdentityMatches) {
      joinFault := B(
        BitNetResidentGateUpComputeBarrierFault.JoinIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.joinCompletion.payload.success) {
      when(io.joinCompletion.payload.faultCode === 0) {
        joinFault := B(
          BitNetResidentGateUpComputeBarrierFault.JoinFailed, 8 bits)
      }
      stickyFault := True
    }
  }

  val linearDoneAfter = linearDone || io.linearCompletion.fire
  val assemblyDoneAfter = assemblyDone || io.assemblyCompletion.fire
  val joinDoneAfter = joinDone || io.joinCompletion.fire
  val linearOkAfter = Mux(io.linearCompletion.fire,
    io.linearCompletion.payload.success && linearIdentityMatches, linearOk)
  val assemblyOkAfter = Mux(io.assemblyCompletion.fire,
    io.assemblyCompletion.payload.success && assemblyIdentityMatches,
    assemblyOk)
  val joinOkAfter = Mux(io.joinCompletion.fire,
    io.joinCompletion.payload.success && joinIdentityMatches, joinOk)

  val linearFaultAfter = Bits(8 bits)
  linearFaultAfter := linearFault
  when(io.linearCompletion.fire) {
    linearFaultAfter := io.linearCompletion.payload.faultCode
    when(!linearIdentityMatches) {
      linearFaultAfter := B(
        BitNetResidentGateUpComputeBarrierFault.LinearIdentity, 8 bits)
    } elsewhen(!io.linearCompletion.payload.success &&
        io.linearCompletion.payload.faultCode === 0) {
      linearFaultAfter := B(
        BitNetResidentGateUpComputeBarrierFault.LinearFailed, 8 bits)
    }
  }
  val assemblyFaultAfter = Bits(8 bits)
  assemblyFaultAfter := assemblyFault
  when(io.assemblyCompletion.fire) {
    assemblyFaultAfter := io.assemblyCompletion.payload.faultCode
    when(!assemblyIdentityMatches) {
      assemblyFaultAfter := B(
        BitNetResidentGateUpComputeBarrierFault.AssemblyIdentity, 8 bits)
    } elsewhen(!io.assemblyCompletion.payload.success &&
        io.assemblyCompletion.payload.faultCode === 0) {
      assemblyFaultAfter := B(
        BitNetResidentGateUpComputeBarrierFault.AssemblyFailed, 8 bits)
    }
  }
  val joinFaultAfter = Bits(8 bits)
  joinFaultAfter := joinFault
  when(io.joinCompletion.fire) {
    joinFaultAfter := io.joinCompletion.payload.faultCode
    when(!joinIdentityMatches) {
      joinFaultAfter := B(
        BitNetResidentGateUpComputeBarrierFault.JoinIdentity, 8 bits)
    } elsewhen(!io.joinCompletion.payload.success &&
        io.joinCompletion.payload.faultCode === 0) {
      joinFaultAfter := B(
        BitNetResidentGateUpComputeBarrierFault.JoinFailed, 8 bits)
    }
  }

  when(active && linearDoneAfter && assemblyDoneAfter && joinDoneAfter &&
      !completionValid) {
    completionReg.requestId := held.requestId
    completionReg.stepId := held.stepId
    completionReg.stage := held.stage
    completionReg.layer := held.layer
    completionReg.part := held.part
    completionReg.sequenceId := held.sequenceId
    completionReg.success := linearOkAfter && assemblyOkAfter && joinOkAfter
    completionReg.faultCode := 0
    when(!linearOkAfter) {
      completionReg.faultCode := linearFaultAfter
    } elsewhen(!assemblyOkAfter) {
      completionReg.faultCode := assemblyFaultAfter
    } elsewhen(!joinOkAfter) {
      completionReg.faultCode := joinFaultAfter
    }
    completionValid := True
    active := False
  }

  io.completion.valid := completionValid
  io.completion.payload := completionReg
  when(io.completion.fire) { completionValid := False }

  io.busy := active || completionValid
  io.protocolFaultSticky := stickyFault
}

/**
  * Production fused GateUp control slice around the sole TP4 compute island.
  *
  * This block owns control only: it instantiates no matrix array, FFN gate,
  * or output-sized memory.  One projection launch is atomically forked to the
  * linear adapter, shared five-bank weight path, arithmetic scoreboard and final
  * weight barrier.  The resulting linear descriptor is then atomically forked
  * to the external island endpoint, feature assembler and compact Gate/Up
  * join.  The exposed fused beats are intended for an already-armed exact
  * Relu2 gate instance.
  */
class BitNetResidentGateUpProjectionControlSlice(
    cfg: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig(),
    joinCfg: BitNetResidentFusedGateUpStreamJoinConfig =
      BitNetResidentFusedGateUpStreamJoinConfig()) extends Component {
  private val stageCfg = cfg.stage
  private val model = stageCfg.model
  private val endpointCfg = joinCfg.endpoint
  private val b = endpointCfg.matrix.base

  require(joinCfg.projection.model == model)
  require(endpointCfg.feeder == stageCfg.feeder)

  val io = new Bundle {
    val projection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(stageCfg)))

    /** Sole model-wide TP4 island transaction. */
    val linearLaunch = master(Stream(
      BitNetResidentLinearLaunch(endpointCfg)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(endpointCfg)))
    val linearResult = slave(Stream(
      BitNetResidentFinalFeaturePair(endpointCfg)))

    /** Client boundary of the sole resident Matrix weight service. */
    val weightProjection = master(Stream(
      BitNetResidentMatrixProjectionLaunch(stageCfg)))
    val weightRetirement = slave(Stream(
      BitNetResidentMatrixWeightRetirement(cfg)))

    /** Raw, paired Gate/Up features for the pre-armed exact Relu2 gate. */
    val fusedGateUp = master(Stream(BitNetFusedGateUpBeat(joinCfg.gate)))

    val projectionCompletion = master(Stream(
      BitNetResidentMatrixProjectionCompletion(stageCfg)))
    val supported = out Bool()
    val busy = out Bool()
    /** True only while bounded synthetic traffic finishes a failed dependant. */
    val dependentFailureDrainActive = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val linear = new BitNetResidentMatrixLinearTransactionAdapter(stageCfg)
  val assembler = new BitNetResidentProjectionFeatureAssembler(
    joinCfg.projection)
  val join = new BitNetResidentFusedGateUpStreamJoin(joinCfg)
  val computeBarrier = new BitNetResidentGateUpComputeBarrier(cfg, joinCfg)
  val projectionBarrier = new BitNetResidentMatrixProjectionBarrier(cfg)

  // A local copy is used only to finish bounded failure drains.  It is not a
  // software-visible descriptor and owns no activation or result RAM.
  val held = Reg(BitNetResidentMatrixProjectionLaunch(stageCfg)) init
    BitNetResidentMatrixProjectionLaunch(stageCfg).getZero
  val expectedAssemblerPairs = Vec(
    Reg(UInt(b.outputIndexWidth bits)) init 0, model.engineCount)
  val acceptedAssemblerPairs = Vec(
    Reg(UInt(b.outputIndexWidth bits)) init 0, model.engineCount)
  val acceptedJoinFeatures = Reg(
    UInt(joinCfg.physicalCountWidth bits)) init 0
  val linearFailureDrain = RegInit(False)
  val assemblyFailureDrain = RegInit(False)
  val recoveryFaultSticky = RegInit(False)

  val launch = io.projection.payload
  val candidateLinear = BitNetResidentLinearLaunch(endpointCfg)
  candidateLinear.requestId := launch.requestId
  candidateLinear.layer := launch.layer
  candidateLinear.kind := launch.kind
  candidateLinear.mode := launch.mode
  candidateLinear.tokenMask := launch.tokenMask
  candidateLinear.sequenceId := launch.sequenceId
  candidateLinear.matrixTask := launch.matrixTask
  val exactPhysicalSpan = if (model.engineCount == 1) {
    launch.matrixTask.outputBase(0) === 0 &&
      launch.matrixTask.outputEndExclusive(0) === 2 * model.ffnSize
  } else {
    (launch.mode === BitNetMode.Prefill &&
      launch.matrixTask.outputBase(0) === 0 &&
      launch.matrixTask.outputBase(1) === 0 &&
      launch.matrixTask.outputEndExclusive(0) === 2 * model.ffnSize &&
      launch.matrixTask.outputEndExclusive(1) === 2 * model.ffnSize) ||
    (launch.mode === BitNetMode.Decode &&
      launch.matrixTask.outputBase(0) === 0 &&
      launch.matrixTask.outputEndExclusive(0) === model.ffnSize &&
      launch.matrixTask.outputBase(1) === model.ffnSize &&
      launch.matrixTask.outputEndExclusive(1) === 2 * model.ffnSize)
  }
  val expectedGroupsPerEngine = UInt(b.outputIndexWidth bits)
  expectedGroupsPerEngine := model.ffnSize / 2
  when(launch.mode === BitNetMode.Decode) {
    expectedGroupsPerEngine := model.ffnSize / (2 * model.engineCount)
  }
  val supported =
    launch.stage === BitNetResidentLayerStage.GateUpProjection &&
      launch.part === BitNetResidentMatrixProjectionPart.FusedGateUp &&
      launch.partOrdinal === 0 && launch.partCount === 1 &&
      launch.firstPart && launch.lastPart &&
      launch.kind === BitNetResidentLinearKind.GateUp &&
      launch.weightTarget === BitNetAddressTarget.GateUpWeight &&
      launch.scaleTarget === BitNetAddressTarget.GateProjectionScale &&
      launch.hasSecondaryScale &&
      launch.secondaryScaleTarget === BitNetAddressTarget.UpProjectionScale &&
      launch.inputFeatures === model.hiddenSize &&
      launch.outputFeatures === 2 * model.ffnSize &&
      launch.groupsPerEngine === expectedGroupsPerEngine &&
      launch.directLinearCompatible &&
      !launch.directStageCompletionCompatible && exactPhysicalSpan &&
      BitNetResidentLinearProtocol.launchLegal(candidateLinear, endpointCfg)

  // The semantic projection is accepted only when every independent owner can
  // capture the exact same descriptor on this clock edge.
  val projectionForkReady = linear.io.projection.ready &&
    io.weightProjection.ready && computeBarrier.io.begin.ready &&
    projectionBarrier.io.begin.ready
  io.projection.ready := supported && projectionForkReady
  linear.io.projection.valid := io.projection.valid && supported &&
    io.weightProjection.ready && computeBarrier.io.begin.ready &&
    projectionBarrier.io.begin.ready
  io.weightProjection.valid := io.projection.valid && supported &&
    linear.io.projection.ready && computeBarrier.io.begin.ready &&
    projectionBarrier.io.begin.ready
  computeBarrier.io.begin.valid := io.projection.valid && supported &&
    linear.io.projection.ready && io.weightProjection.ready &&
    projectionBarrier.io.begin.ready
  projectionBarrier.io.begin.valid := io.projection.valid && supported &&
    linear.io.projection.ready && io.weightProjection.ready &&
    computeBarrier.io.begin.ready
  linear.io.projection.payload := launch
  io.weightProjection.payload := launch
  computeBarrier.io.begin.payload := launch
  projectionBarrier.io.begin.payload := launch

  when(io.projection.fire) {
    held := launch
    acceptedAssemblerPairs.foreach(_ := 0)
    acceptedJoinFeatures := 0
    linearFailureDrain := False
    assemblyFailureDrain := False
    for (engine <- 0 until model.engineCount) {
      val span = launch.matrixTask.outputEndExclusive(engine).resize(
        b.outputIndexWidth + 1) -
        launch.matrixTask.outputBase(engine).resize(b.outputIndexWidth + 1)
      expectedAssemblerPairs(engine) := ((span + 1) >> 1).resized
      when(launch.mode === BitNetMode.Prefill &&
          !launch.tokenMask(engine * endpointCfg.localTokenLanes +
            endpointCfg.localTokenLanes - 1 downto
            engine * endpointCfg.localTokenLanes).orR) {
        expectedAssemblerPairs(engine) := 0
      }
    }
  }

  // The adapter's concrete linear descriptor arms all consumers atomically;
  // the TP4 island can therefore never return a pair before both resident
  // downstream boundaries own the matching identity and shape.
  val linearForkReady = io.linearLaunch.ready && assembler.io.launch.ready &&
    join.io.launch.ready
  linear.io.linearLaunch.ready := linearForkReady
  io.linearLaunch.valid := linear.io.linearLaunch.valid &&
    assembler.io.launch.ready && join.io.launch.ready
  assembler.io.launch.valid := linear.io.linearLaunch.valid &&
    io.linearLaunch.ready && join.io.launch.ready
  join.io.launch.valid := linear.io.linearLaunch.valid &&
    io.linearLaunch.ready && assembler.io.launch.ready
  io.linearLaunch.payload := linear.io.linearLaunch.payload
  assembler.io.launch.payload := linear.io.linearLaunch.payload
  join.io.launch.payload := linear.io.linearLaunch.payload

  linear.io.linearCompletion.valid := io.linearCompletion.valid
  linear.io.linearCompletion.payload := io.linearCompletion.payload
  io.linearCompletion.ready := linear.io.linearCompletion.ready

  val rawLinearIdentityMatches =
    io.linearCompletion.requestId === held.requestId &&
      io.linearCompletion.layer === held.layer &&
      io.linearCompletion.kind === BitNetResidentLinearKind.GateUp &&
      io.linearCompletion.sequenceId === held.sequenceId
  val assemblerPairsMissing = (0 until model.engineCount).map { engine =>
    acceptedAssemblerPairs(engine) < expectedAssemblerPairs(engine)
  }.reduce(_ || _)
  when(io.linearCompletion.fire &&
      (!io.linearCompletion.success || !rawLinearIdentityMatches) &&
      assemblerPairsMissing) {
    // The endpoint has drained the physical TP4 returns.  If it suppressed
    // final pairs while doing so, synthesize only the missing frame ordinals
    // into the assembler's failure drain; no arithmetic payload is trusted.
    linearFailureDrain := True
    recoveryFaultSticky := True
  }

  val recoveryPairEngine = UInt(
    Math.max(1, log2Up(model.engineCount)) bits)
  recoveryPairEngine := 0
  if (model.engineCount > 1) {
    when(acceptedAssemblerPairs(0) >= expectedAssemblerPairs(0)) {
      recoveryPairEngine := 1
    }
  }
  val recoveryPair = BitNetResidentFinalFeaturePair(endpointCfg)
  recoveryPair.requestId := held.requestId ^ 1
  recoveryPair.layer := held.layer
  recoveryPair.kind := BitNetResidentLinearKind.GateUp
  recoveryPair.mode := held.mode
  recoveryPair.sequenceId := held.sequenceId
  recoveryPair.engine := recoveryPairEngine
  recoveryPair.physicalOutputBase := (held.matrixTask.outputBase(0) +
    (acceptedAssemblerPairs(0) << 1)).resized
  if (model.engineCount > 1) {
    when(recoveryPairEngine === 1) {
      recoveryPair.physicalOutputBase := (held.matrixTask.outputBase(1) +
        (acceptedAssemblerPairs(1) << 1)).resized
    }
  }
  recoveryPair.globalTokenLaneMask := B(1, model.tokenParallelism bits)
  when(held.mode === BitNetMode.Prefill) {
    recoveryPair.globalTokenLaneMask :=
      held.tokenMask(endpointCfg.localTokenLanes - 1 downto 0)
        .resize(model.tokenParallelism)
    if (model.engineCount > 1) {
      when(recoveryPairEngine === 1) {
        recoveryPair.globalTokenLaneMask := (held.tokenMask(
          2 * endpointCfg.localTokenLanes - 1 downto
            endpointCfg.localTokenLanes).resize(model.tokenParallelism) <<
          endpointCfg.localTokenLanes).resize(model.tokenParallelism)
      }
    }
  }
  recoveryPair.outputLaneMask := B"2'b11"
  recoveryPair.weightScaleQ16 := held.matrixTask.weightScaleQ16(0)
  if (model.engineCount > 1) {
    when(recoveryPairEngine === 1) {
      recoveryPair.weightScaleQ16 := held.matrixTask.weightScaleQ16(1)
    }
  }
  recoveryPair.physicalVectorFirst := False
  recoveryPair.physicalVectorLast := False
  for (local <- 0 until endpointCfg.localTokenLanes) {
    recoveryPair.taskId(local) := 0
    recoveryPair.tokenId(local) := 0
    recoveryPair.activationFactorQ16(local) := 0
    for (row <- 0 until b.outputLanes) {
      recoveryPair.rawRows(local)(row) := 0
    }
  }
  assembler.io.input.valid := io.linearResult.valid
  assembler.io.input.payload := io.linearResult.payload
  io.linearResult.ready := assembler.io.input.ready
  when(linearFailureDrain) {
    assembler.io.input.valid := assemblerPairsMissing
    assembler.io.input.payload := recoveryPair
    io.linearResult.ready := True
    when(io.linearResult.valid) { recoveryFaultSticky := True }
  }
  when(assembler.io.input.fire) {
    for (engine <- 0 until model.engineCount) {
      when(assembler.io.input.engine === engine &&
          acceptedAssemblerPairs(engine) < expectedAssemblerPairs(engine)) {
        acceptedAssemblerPairs(engine) := acceptedAssemblerPairs(engine) + 1
      }
    }
  }

  val assemblyIdentityMatches =
    assembler.io.completion.requestId === held.requestId &&
      assembler.io.completion.layer === held.layer &&
      assembler.io.completion.kind === BitNetResidentLinearKind.GateUp &&
      assembler.io.completion.sequenceId === held.sequenceId
  when(assembler.io.completion.fire) {
    linearFailureDrain := False
    when((!assembler.io.completion.success || !assemblyIdentityMatches) &&
        acceptedJoinFeatures < joinCfg.physicalFeatures) {
      // A failed assembler has already consumed all final pairs.  Complete the
      // join's fixed-size failure drain with identity-invalid placeholders so
      // its bounded scoreboard can retire instead of stranding the lease.
      assemblyFailureDrain := True
      recoveryFaultSticky := True
    }
  }

  join.io.input.valid := assembler.io.output.valid
  join.io.input.payload := assembler.io.output.payload
  assembler.io.output.ready := join.io.input.ready
  val recoveryFeature = BitNetResidentProjectionFeature(joinCfg.projection)
  recoveryFeature.requestId := held.requestId ^ 1
  recoveryFeature.layer := held.layer
  recoveryFeature.kind := BitNetResidentLinearKind.GateUp
  recoveryFeature.mode := held.mode
  recoveryFeature.sequenceId := held.sequenceId
  recoveryFeature.featureIndex := acceptedJoinFeatures.resized
  recoveryFeature.tokenMask := held.tokenMask
  recoveryFeature.weightScaleQ16 := held.matrixTask.weightScaleQ16(0)
  recoveryFeature.first := acceptedJoinFeatures === 0
  recoveryFeature.last :=
    acceptedJoinFeatures === joinCfg.physicalFeatures - 1
  for (lane <- 0 until model.tokenParallelism) {
    recoveryFeature.raw(lane) := 0
    recoveryFeature.activationFactorQ16(lane) := 0
  }
  when(assemblyFailureDrain) {
    join.io.input.valid := acceptedJoinFeatures < joinCfg.physicalFeatures
    join.io.input.payload := recoveryFeature
    assembler.io.output.ready := True
    when(assembler.io.output.valid) { recoveryFaultSticky := True }
  }
  when(join.io.input.fire &&
      acceptedJoinFeatures < joinCfg.physicalFeatures) {
    acceptedJoinFeatures := acceptedJoinFeatures + 1
  }
  when(join.io.completion.fire) { assemblyFailureDrain := False }

  io.fusedGateUp << join.io.output

  computeBarrier.io.linearCompletion << linear.io.completion
  computeBarrier.io.assemblyCompletion << assembler.io.completion
  computeBarrier.io.joinCompletion << join.io.completion
  projectionBarrier.io.computeCompletion << computeBarrier.io.completion
  projectionBarrier.io.weightRetirement << io.weightRetirement
  io.projectionCompletion << projectionBarrier.io.completion

  io.supported := supported
  io.busy := linear.io.active || assembler.io.busy || join.io.busy ||
    computeBarrier.io.busy || projectionBarrier.io.busy ||
    linearFailureDrain || assemblyFailureDrain
  io.dependentFailureDrainActive :=
    linearFailureDrain || assemblyFailureDrain
  io.protocolFaultSticky := linear.io.protocolFaultSticky ||
    assembler.io.protocolFaultSticky || join.io.protocolFaultSticky ||
    computeBarrier.io.protocolFaultSticky ||
    projectionBarrier.io.protocolFaultSticky || recoveryFaultSticky
}

object GenerateBitNetResidentGateUpProjectionControlSlice extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-gate-up-projection-control-slice")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentGateUpProjectionControlSlice())
}
