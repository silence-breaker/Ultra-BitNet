package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentGateUpLeasedGatePipelineFault {
  val ProjectionIdentity = 0xe1
  val ProjectionFailed = 0xe2
  val GateIdentity = 0xe3
  val GateFailed = 0xe4
}

/** Static composition contract for the resident GateUp projection and gate. */
case class BitNetResidentGateUpLeasedGatePipelineConfig(
    projection: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig(),
    join: BitNetResidentFusedGateUpStreamJoinConfig =
      BitNetResidentFusedGateUpStreamJoinConfig(),
    gate: BitNetLeasedFfnGateConfig =
      BitNetLeasedFfnGateConfig()
) {
  require(projection.stage.model == join.model)
  require(join.gate == gate.gate,
    "the fused semantic boundary and leased gate must share one contract")
  require(projection.stage.model == gate.gate.model)
  require(gate.gate.maxLength == projection.stage.model.ffnSize)
}

/**
  * Independent production composition of resident GateUp and the leased FFN
  * gate.  It deliberately owns neither a matrix island nor an arithmetic
  * pool: the existing linear/shared-weight boundaries and the fixed FfnGate lease
  * ports remain visible at the component boundary.
  *
  * A projection launch and its pre-staged combined-scale command are accepted
  * on the same clock edge.  Length, token identity, scatter authority and the
  * step tag are checked before either child can observe the transaction.  The
  * combined-scale values themselves are produced by the separate scale
  * staging path; this wrapper performs no multiplication.
  *
  * Projection retirement is captured only after the control slice's compute
  * and resident-weight barriers have both completed.  RMS publication is held
  * until that successful retirement, and the public projection completion and
  * gate done streams are raised only after the leased gate has also released
  * its arithmetic lease and completed.  On projection failure, an identity-
  * invalid fixed-length semantic drain drives the already-armed gate into its
  * fail-closed drain state.  No synthetic beat can write WideWork or issue RMS.
  */
class BitNetResidentGateUpLeasedGatePipeline(
    cfg: BitNetResidentGateUpLeasedGatePipelineConfig =
      BitNetResidentGateUpLeasedGatePipelineConfig()) extends Component {
  private val projectionCfg = cfg.projection
  private val stageCfg = projectionCfg.stage
  private val joinCfg = cfg.join
  private val gateCfg = cfg.gate.gate
  private val poolCfg = cfg.gate.arithmetic
  private val model = stageCfg.model
  private val endpointCfg = joinCfg.endpoint
  private val layout = BitNetStreamingActivationLayout(model)

  val io = new Bundle {
    /** These two inputs form one atomic transaction. */
    val projection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(stageCfg)))
    val gateCommand = slave(Stream(BitNetSharedFfnGateCommand(gateCfg)))

    /** Existing sole-TP4-island boundary, unchanged. */
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
      BitNetResidentMatrixWeightRetirement(projectionCfg)))

    val wideWrite = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))),
      gateCfg.tokenLanes)
    val rmsCommand = master(Stream(
      BitNetSharedRmsCommand(gateCfg.rmsConfig)))

    /** Both streams become valid only after the dual completion barrier. */
    val projectionCompletion = master(Stream(
      BitNetResidentMatrixProjectionCompletion(stageCfg)))
    val done = master(Stream(BitNetSharedFfnGateDone(gateCfg)))

    /** Direct client-2 boundary of the single model-wide arithmetic pool. */
    val poolAcquire = master(Stream(
      BitNetSpuArithmeticLeaseRequest(poolCfg)))
    val poolOperand = master(Stream(
      BitNetSpuArithmeticOperandBeat(poolCfg)))
    val poolResult = slave(Stream(
      BitNetSpuArithmeticResultBeat(poolCfg)))
    val poolRelease = master(Stream(
      BitNetSpuArithmeticLeaseRelease(poolCfg)))

    val supported = out Bool()
    val busy = out Bool()
    val dependentFailureDrainActive = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedFusedFeatures = out UInt(32 bits)
    val emittedWideFeatures = out UInt(32 bits)
  }

  val control = new BitNetResidentGateUpProjectionControlSlice(
    projectionCfg, joinCfg)
  val gate = new BitNetLeasedFfnGate(cfg.gate)

  /* ------------------------------------------------------------------
   * Atomic projection + combined-scale command binding.
   * ------------------------------------------------------------------ */
  val active = RegInit(False)
  val heldProjection = Reg(
    BitNetResidentMatrixProjectionLaunch(stageCfg)) init
      BitNetResidentMatrixProjectionLaunch(stageCfg).getZero
  val heldGateCommand = Reg(BitNetSharedFfnGateCommand(gateCfg)) init
    BitNetSharedFfnGateCommand(gateCfg).getZero
  val associationFaultSticky = RegInit(False)

  val expectedGateTag = io.projection.payload.stepId.resize(
    gateCfg.tagWidth)
  val gateCommandAssociationLegal =
    io.gateCommand.payload.length === model.ffnSize &&
      io.gateCommand.payload.tokenMask === io.projection.payload.tokenMask &&
      io.gateCommand.payload.authenticatedScatter &&
      io.gateCommand.payload.tag === expectedGateTag

  control.io.projection.payload := io.projection.payload
  gate.io.command.payload := io.gateCommand.payload
  val atomicCandidate = !active && io.projection.valid &&
    io.gateCommand.valid && gateCommandAssociationLegal
  control.io.projection.valid := atomicCandidate && gate.io.command.ready
  gate.io.command.valid := atomicCandidate && control.io.projection.ready
  io.projection.ready := !active && io.gateCommand.valid &&
    gateCommandAssociationLegal && control.io.projection.ready &&
    gate.io.command.ready
  io.gateCommand.ready := !active && io.projection.valid &&
    gateCommandAssociationLegal && control.io.projection.ready &&
    gate.io.command.ready

  val atomicLaunchFire = control.io.projection.fire && gate.io.command.fire
  when(io.projection.valid && io.gateCommand.valid &&
      control.io.supported && !gateCommandAssociationLegal) {
    associationFaultSticky := True
  }
  when(atomicLaunchFire) {
    active := True
    heldProjection := io.projection.payload
    heldGateCommand := io.gateCommand.payload
  }

  /* ------------------------------------------------------------------
   * Existing linear and shared resident-weight boundaries are passthroughs.
   * ------------------------------------------------------------------ */
  io.linearLaunch << control.io.linearLaunch
  control.io.linearCompletion << io.linearCompletion
  control.io.linearResult << io.linearResult
  io.weightProjection << control.io.weightProjection
  control.io.weightRetirement << io.weightRetirement

  /* ------------------------------------------------------------------
   * Raw fused semantic path and bounded projection-failure drain.
   * ------------------------------------------------------------------ */
  val gateAcceptedInJob = Reg(UInt(gateCfg.lengthWidth bits)) init 0
  val gateFailureDrain = RegInit(False)
  val projectionCaptured = RegInit(False)
  val projectionIdentityOk = RegInit(False)
  val projectionReg = Reg(
    BitNetResidentMatrixProjectionCompletion(stageCfg)) init
      BitNetResidentMatrixProjectionCompletion(stageCfg).getZero

  gate.io.fusedInput.valid := control.io.fusedGateUp.valid
  gate.io.fusedInput.payload := control.io.fusedGateUp.payload
  control.io.fusedGateUp.ready := gate.io.fusedInput.ready

  when(gateFailureDrain) {
    gate.io.fusedInput.valid := gateAcceptedInJob < model.ffnSize
    gate.io.fusedInput.payload.tokenMask :=
      heldGateCommand.tokenMask ^ B(1, gateCfg.tokenLanes bits)
    gate.io.fusedInput.payload.featureIndex := gateAcceptedInJob.resized
    gate.io.fusedInput.payload.first := gateAcceptedInJob === 0
    gate.io.fusedInput.payload.last :=
      gateAcceptedInJob === model.ffnSize - 1
    for (lane <- 0 until gateCfg.tokenLanes) {
      gate.io.fusedInput.payload.gateRaw(lane) := 0
      gate.io.fusedInput.payload.upRaw(lane) := 0
    }
    control.io.fusedGateUp.ready := False
  }

  when(atomicLaunchFire) {
    gateAcceptedInJob := 0
    gateFailureDrain := False
    projectionCaptured := False
    projectionIdentityOk := False
  }
  when(active && gate.io.fusedInput.fire &&
      gateAcceptedInJob < model.ffnSize) {
    gateAcceptedInJob := gateAcceptedInJob + 1
    when(gateFailureDrain && gateAcceptedInJob === model.ffnSize - 1) {
      gateFailureDrain := False
    }
  }
  val gateAcceptedAfterCurrentTransfer =
    gateAcceptedInJob.resize(gateCfg.lengthWidth + 1) +
      Mux(gate.io.fusedInput.fire,
        U(1, (gateCfg.lengthWidth + 1) bits),
        U(0, (gateCfg.lengthWidth + 1) bits))

  val projectionCompletionIdentityMatches =
    control.io.projectionCompletion.payload.requestId ===
      heldProjection.requestId &&
      control.io.projectionCompletion.payload.stepId ===
        heldProjection.stepId &&
      control.io.projectionCompletion.payload.stage ===
        heldProjection.stage &&
      control.io.projectionCompletion.payload.layer ===
        heldProjection.layer &&
      control.io.projectionCompletion.payload.part ===
        heldProjection.part &&
      control.io.projectionCompletion.payload.sequenceId ===
        heldProjection.sequenceId
  control.io.projectionCompletion.ready := active && !projectionCaptured
  when(control.io.projectionCompletion.fire) {
    projectionCaptured := True
    projectionReg := control.io.projectionCompletion.payload
    projectionIdentityOk := projectionCompletionIdentityMatches
    when(!projectionCompletionIdentityMatches ||
        !control.io.projectionCompletion.payload.success) {
      associationFaultSticky := True
      when(gateAcceptedAfterCurrentTransfer < model.ffnSize) {
        gateFailureDrain := True
      }
    }
  }

  val projectionSucceeded = projectionCaptured &&
    projectionIdentityOk && projectionReg.success

  /* Writes are speculative until RMS publication.  Once a projection failure
   * is known, any in-flight gate write is consumed internally and every
   * synthetic drain beat is identity-invalid, so no further valid WideWork
   * transfer can escape the component. */
  val discardGateOutput = projectionCaptured && !projectionSucceeded
  for (lane <- 0 until gateCfg.tokenLanes) {
    io.wideWrite(lane).valid := gate.io.wideWrite(lane).valid &&
      !discardGateOutput
    io.wideWrite(lane).payload := gate.io.wideWrite(lane).payload
    gate.io.wideWrite(lane).ready := Mux(discardGateOutput,
      True, io.wideWrite(lane).ready)
  }

  // RMS is the publication barrier for speculative WideWork.  Holding it is
  // safe because the FFN arithmetic lease has already been released.
  io.rmsCommand.valid := gate.io.rmsCommand.valid && projectionSucceeded
  io.rmsCommand.payload := gate.io.rmsCommand.payload
  gate.io.rmsCommand.ready := projectionCaptured &&
    Mux(projectionSucceeded, io.rmsCommand.ready, True)

  /* ------------------------------------------------------------------
   * Direct FfnGate arithmetic-pool boundary.
   * ------------------------------------------------------------------ */
  io.poolAcquire << gate.io.poolAcquire
  io.poolOperand << gate.io.poolOperand
  gate.io.poolResult << io.poolResult
  io.poolRelease << gate.io.poolRelease

  /* ------------------------------------------------------------------
   * Gate completion capture and dual, independently backpressured report.
   * ------------------------------------------------------------------ */
  val gateCaptured = RegInit(False)
  val gateIdentityOk = RegInit(False)
  val gateDoneReg = Reg(BitNetSharedFfnGateDone(gateCfg)) init
    BitNetSharedFfnGateDone(gateCfg).getZero
  when(atomicLaunchFire) {
    gateCaptured := False
    gateIdentityOk := False
  }

  val gateDoneIdentityMatches =
    gate.io.done.payload.tag === heldGateCommand.tag &&
      gate.io.done.payload.length === heldGateCommand.length
  gate.io.done.ready := active && !gateCaptured
  when(gate.io.done.fire) {
    gateCaptured := True
    gateDoneReg := gate.io.done.payload
    gateIdentityOk := gateDoneIdentityMatches
    when(!gateDoneIdentityMatches || gate.io.done.payload.protocolFault) {
      associationFaultSticky := True
    }
  }

  val reporting = RegInit(False)
  val projectionReportPending = RegInit(False)
  val doneReportPending = RegInit(False)
  when(atomicLaunchFire) {
    reporting := False
    projectionReportPending := False
    doneReportPending := False
  }
  when(active && projectionCaptured && gateCaptured && !reporting) {
    reporting := True
    projectionReportPending := True
    doneReportPending := True
  }

  val combinedSuccess = projectionIdentityOk && projectionReg.success &&
    gateIdentityOk && !gateDoneReg.protocolFault &&
    gateDoneReg.rmsCommandIssued
  val combinedFaultCode = Bits(8 bits)
  combinedFaultCode := 0
  when(!projectionIdentityOk) {
    combinedFaultCode := B(
      BitNetResidentGateUpLeasedGatePipelineFault.ProjectionIdentity,
      8 bits)
  } elsewhen(!projectionReg.success) {
    combinedFaultCode := projectionReg.faultCode
    when(projectionReg.faultCode === 0) {
      combinedFaultCode := B(
        BitNetResidentGateUpLeasedGatePipelineFault.ProjectionFailed,
        8 bits)
    }
  } elsewhen(!gateIdentityOk) {
    combinedFaultCode := B(
      BitNetResidentGateUpLeasedGatePipelineFault.GateIdentity, 8 bits)
  } elsewhen(gateDoneReg.protocolFault ||
      !gateDoneReg.rmsCommandIssued) {
    combinedFaultCode := B(
      BitNetResidentGateUpLeasedGatePipelineFault.GateFailed, 8 bits)
  }

  io.projectionCompletion.valid := reporting && projectionReportPending
  io.projectionCompletion.payload.requestId := heldProjection.requestId
  io.projectionCompletion.payload.stepId := heldProjection.stepId
  io.projectionCompletion.payload.stage := heldProjection.stage
  io.projectionCompletion.payload.layer := heldProjection.layer
  io.projectionCompletion.payload.part := heldProjection.part
  io.projectionCompletion.payload.sequenceId := heldProjection.sequenceId
  io.projectionCompletion.payload.success := combinedSuccess
  io.projectionCompletion.payload.faultCode := combinedFaultCode

  io.done.valid := reporting && doneReportPending
  io.done.payload.tag := heldGateCommand.tag
  io.done.payload.length := heldGateCommand.length
  io.done.payload.protocolFault := !combinedSuccess
  io.done.payload.overflow := gateDoneReg.overflow
  io.done.payload.rmsCommandIssued := combinedSuccess

  val projectionPendingAfter = projectionReportPending &&
    !io.projectionCompletion.fire
  val donePendingAfter = doneReportPending && !io.done.fire
  when(reporting) {
    projectionReportPending := projectionPendingAfter
    doneReportPending := donePendingAfter
    when(!projectionPendingAfter && !donePendingAfter) {
      reporting := False
      active := False
      projectionCaptured := False
      gateCaptured := False
    }
  }

  io.supported := control.io.supported && gateCommandAssociationLegal
  io.busy := active || control.io.busy || gate.io.busy || reporting
  io.dependentFailureDrainActive :=
    control.io.dependentFailureDrainActive || gateFailureDrain
  io.protocolFaultSticky := associationFaultSticky ||
    control.io.protocolFaultSticky || gate.io.protocolFaultSticky
  io.acceptedFusedFeatures := gate.io.acceptedFeatures
  io.emittedWideFeatures := gate.io.emittedFeatures
}

object GenerateBitNetResidentGateUpLeasedGatePipeline extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-gate-up-leased-gate-pipeline")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentGateUpLeasedGatePipeline())
}
