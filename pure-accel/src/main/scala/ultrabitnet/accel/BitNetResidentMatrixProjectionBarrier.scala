package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

object BitNetResidentMatrixProjectionBarrierFault {
  val ComputeIdentity = 0xa1
  val ComputeFailed = 0xa2
  val WeightIdentity = 0xa3
  val WeightFailed = 0xa4
}

/**
  * Joins arithmetic retirement with five-bank replay retirement.
  *
  * A failed side does not release the transaction early: the other physical
  * side is still drained and authenticated before the projection completion is
  * exposed to the semantic stage expander.
  */
class BitNetResidentMatrixProjectionBarrier(
    cfg: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig()) extends Component {
  private val s = cfg.stage

  val io = new Bundle {
    val begin = slave(Stream(BitNetResidentMatrixProjectionLaunch(s)))
    val computeCompletion = slave(Stream(
      BitNetResidentMatrixProjectionCompletion(s)))
    val weightRetirement = slave(Stream(
      BitNetResidentMatrixWeightRetirement(cfg)))
    val completion = master(Stream(
      BitNetResidentMatrixProjectionCompletion(s)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val active = RegInit(False)
  val held = Reg(BitNetResidentMatrixProjectionLaunch(s)) init
    BitNetResidentMatrixProjectionLaunch(s).getZero
  val computeDone = RegInit(False)
  val weightDone = RegInit(False)
  val computeOk = RegInit(False)
  val weightOk = RegInit(False)
  val computeFault = Reg(Bits(8 bits)) init 0
  val weightFault = Reg(Bits(8 bits)) init 0
  val completionValid = RegInit(False)
  val completionReg = Reg(
    BitNetResidentMatrixProjectionCompletion(s)) init
    BitNetResidentMatrixProjectionCompletion(s).getZero
  val stickyFault = RegInit(False)

  io.begin.ready := !active && !completionValid
  when(io.begin.fire) {
    held := io.begin.payload
    active := True
    computeDone := False
    weightDone := False
    computeOk := False
    weightOk := False
    computeFault := 0
    weightFault := 0
  }

  val computeIdentityMatches =
    io.computeCompletion.payload.requestId === held.requestId &&
    io.computeCompletion.payload.stepId === held.stepId &&
    io.computeCompletion.payload.stage === held.stage &&
    io.computeCompletion.payload.layer === held.layer &&
    io.computeCompletion.payload.part === held.part &&
    io.computeCompletion.payload.sequenceId === held.sequenceId
  io.computeCompletion.ready := active && !computeDone
  when(io.computeCompletion.fire) {
    computeDone := True
    computeOk := io.computeCompletion.payload.success &&
      computeIdentityMatches
    computeFault := io.computeCompletion.payload.faultCode
    when(!computeIdentityMatches) {
      computeFault := B(
        BitNetResidentMatrixProjectionBarrierFault.ComputeIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.computeCompletion.payload.success) {
      when(io.computeCompletion.payload.faultCode === 0) {
        computeFault := B(
          BitNetResidentMatrixProjectionBarrierFault.ComputeFailed, 8 bits)
      }
      stickyFault := True
    }
  }

  val weightIdentityMatches =
    io.weightRetirement.payload.requestId === held.requestId &&
    io.weightRetirement.payload.stepId === held.stepId &&
    io.weightRetirement.payload.stage === held.stage &&
    io.weightRetirement.payload.layer === held.layer &&
    io.weightRetirement.payload.part === held.part &&
    io.weightRetirement.payload.sequenceId === held.sequenceId &&
    io.weightRetirement.payload.target === held.weightTarget
  io.weightRetirement.ready := active && !weightDone
  when(io.weightRetirement.fire) {
    weightDone := True
    weightOk := io.weightRetirement.payload.success &&
      weightIdentityMatches
    weightFault := io.weightRetirement.payload.faultCode
    when(!weightIdentityMatches) {
      weightFault := B(
        BitNetResidentMatrixProjectionBarrierFault.WeightIdentity, 8 bits)
      stickyFault := True
    } elsewhen(!io.weightRetirement.payload.success) {
      when(io.weightRetirement.payload.faultCode === 0) {
        weightFault := B(
          BitNetResidentMatrixProjectionBarrierFault.WeightFailed, 8 bits)
      }
      stickyFault := True
    }
  }

  val computeDoneAfter = computeDone || io.computeCompletion.fire
  val weightDoneAfter = weightDone || io.weightRetirement.fire
  val computeOkAfter = Mux(io.computeCompletion.fire,
    io.computeCompletion.payload.success && computeIdentityMatches, computeOk)
  val weightOkAfter = Mux(io.weightRetirement.fire,
    io.weightRetirement.payload.success && weightIdentityMatches, weightOk)
  val computeFaultAfter = Bits(8 bits)
  computeFaultAfter := computeFault
  when(io.computeCompletion.fire) {
    computeFaultAfter := io.computeCompletion.payload.faultCode
    when(!computeIdentityMatches) {
      computeFaultAfter := B(
        BitNetResidentMatrixProjectionBarrierFault.ComputeIdentity, 8 bits)
    } elsewhen(!io.computeCompletion.payload.success &&
        io.computeCompletion.payload.faultCode === 0) {
      computeFaultAfter := B(
        BitNetResidentMatrixProjectionBarrierFault.ComputeFailed, 8 bits)
    }
  }
  val weightFaultAfter = Bits(8 bits)
  weightFaultAfter := weightFault
  when(io.weightRetirement.fire) {
    weightFaultAfter := io.weightRetirement.payload.faultCode
    when(!weightIdentityMatches) {
      weightFaultAfter := B(
        BitNetResidentMatrixProjectionBarrierFault.WeightIdentity, 8 bits)
    } elsewhen(!io.weightRetirement.payload.success &&
        io.weightRetirement.payload.faultCode === 0) {
      weightFaultAfter := B(
        BitNetResidentMatrixProjectionBarrierFault.WeightFailed, 8 bits)
    }
  }

  when(active && computeDoneAfter && weightDoneAfter && !completionValid) {
    completionReg.requestId := held.requestId
    completionReg.stepId := held.stepId
    completionReg.stage := held.stage
    completionReg.layer := held.layer
    completionReg.part := held.part
    completionReg.sequenceId := held.sequenceId
    completionReg.success := computeOkAfter && weightOkAfter
    completionReg.faultCode := 0
    when(!computeOkAfter) {
      completionReg.faultCode := computeFaultAfter
    } elsewhen(!weightOkAfter) {
      completionReg.faultCode := weightFaultAfter
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
  * Direct control slice for the two projections already compatible with the
  * existing island: AttentionOutput and Down.  It atomically forks one launch
  * to the real linear transaction adapter, the shared two-AGU weight service
  * and the retirement barrier.  QKV and GateUp are not accepted here.
  */
class BitNetResidentMatrixDirectProjectionControlSlice(
    cfg: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig()) extends Component {
  private val s = cfg.stage
  private val linearCfg = BitNetResidentLinearEndpointConfig(
    model = s.model, feeder = s.feeder)

  val io = new Bundle {
    val projection = slave(Stream(BitNetResidentMatrixProjectionLaunch(s)))
    val linearLaunch = master(Stream(BitNetResidentLinearLaunch(linearCfg)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(linearCfg)))
    val weightProjection = master(Stream(
      BitNetResidentMatrixProjectionLaunch(s)))
    val weightRetirement = slave(Stream(
      BitNetResidentMatrixWeightRetirement(cfg)))
    val completion = master(Stream(
      BitNetResidentMatrixProjectionCompletion(s)))
    val supported = out Bool()
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val linear = new BitNetResidentMatrixLinearTransactionAdapter(s)
  val barrier = new BitNetResidentMatrixProjectionBarrier(cfg)

  val supported = io.projection.payload.directLinearCompatible &&
    io.projection.payload.directStageCompletionCompatible &&
    (io.projection.payload.part ===
      BitNetResidentMatrixProjectionPart.AttentionOutput ||
      io.projection.payload.part === BitNetResidentMatrixProjectionPart.Down)
  val allReady = linear.io.projection.ready && io.weightProjection.ready &&
    barrier.io.begin.ready
  io.projection.ready := supported && allReady
  linear.io.projection.valid := io.projection.valid && supported &&
    io.weightProjection.ready && barrier.io.begin.ready
  io.weightProjection.valid := io.projection.valid && supported &&
    linear.io.projection.ready && barrier.io.begin.ready
  barrier.io.begin.valid := io.projection.valid && supported &&
    linear.io.projection.ready && io.weightProjection.ready
  linear.io.projection.payload := io.projection.payload
  io.weightProjection.payload := io.projection.payload
  barrier.io.begin.payload := io.projection.payload

  io.linearLaunch << linear.io.linearLaunch
  linear.io.linearCompletion << io.linearCompletion
  barrier.io.computeCompletion << linear.io.completion
  barrier.io.weightRetirement << io.weightRetirement
  io.completion << barrier.io.completion

  io.supported := supported
  io.busy := linear.io.active || barrier.io.busy
  io.protocolFaultSticky := linear.io.protocolFaultSticky ||
    barrier.io.protocolFaultSticky
}

object GenerateBitNetResidentMatrixDirectProjectionControlSlice extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-resident-matrix-direct-projection-control-slice")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentMatrixDirectProjectionControlSlice())
}
