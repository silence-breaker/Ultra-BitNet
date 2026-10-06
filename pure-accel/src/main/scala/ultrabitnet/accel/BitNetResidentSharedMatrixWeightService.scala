package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** The mutually-exclusive client which owns the resident matrix weight path. */
object BitNetResidentMatrixWeightServiceOwner
    extends SpinalEnum(binarySequential) {
  val Idle, SegmentedQkv, Direct, FusedGateUp = newElement()
}

/**
  * One resident weight-command engine shared by all MatrixArray projections.
  *
  * The matrix-stage scheduler permits only one projection owner at a time, so
  * duplicating the expander, its two AGUs and its two-slot retirement
  * scoreboard in every control slice cannot increase throughput.  This
  * service accepts the one-hot client launch, locks that client until the
  * complete weight retirement is consumed, and returns the retirement only to
  * the locked owner.  The underlying subsystem retains the existing two-AGU
  * issue cadence, slot overlap, completion authentication and failure drain.
  */
class BitNetResidentSharedMatrixWeightService(
    cfg: BitNetResidentMatrixWeightCommandExpanderConfig =
      BitNetResidentMatrixWeightCommandExpanderConfig()) extends Component {
  private val s = cfg.stage
  private val p = cfg.prefetch

  val io = new Bundle {
    val qkvProjection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(s)))
    val qkvRetirement = master(Stream(
      BitNetResidentMatrixWeightRetirement(cfg)))

    val directProjection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(s)))
    val directRetirement = master(Stream(
      BitNetResidentMatrixWeightRetirement(cfg)))

    val gateProjection = slave(Stream(
      BitNetResidentMatrixProjectionLaunch(s)))
    val gateRetirement = master(Stream(
      BitNetResidentMatrixWeightRetirement(cfg)))

    val command = master(Stream(
      BitNetFiveBankStripedWeightCommand(p)))
    val prefetchCompletion = slave(Stream(
      BitNetFiveBankStripedWeightCompletion(p)))

    val activeOwner = out(BitNetResidentMatrixWeightServiceOwner())
    val busy = out Bool()
    val slotBusy = out Bits(p.slotCount bits)
    val clientCollisionSticky = out Bool()
    val protocolFaultSticky = out Bool()
  }

  val weights = new BitNetResidentMatrixWeightCommandSubsystem(cfg)
  io.command << weights.io.command
  weights.io.prefetchCompletion << io.prefetchCompletion

  val owner = Reg(BitNetResidentMatrixWeightServiceOwner()) init
    BitNetResidentMatrixWeightServiceOwner.Idle
  val collisionSticky = RegInit(False)
  val serviceSticky = RegInit(False)
  val ownerIdle = owner === BitNetResidentMatrixWeightServiceOwner.Idle

  val qkvOnly = io.qkvProjection.valid &&
    !io.directProjection.valid && !io.gateProjection.valid
  val directOnly = io.directProjection.valid &&
    !io.qkvProjection.valid && !io.gateProjection.valid
  val gateOnly = io.gateProjection.valid &&
    !io.qkvProjection.valid && !io.directProjection.valid
  val clientCollision =
    (io.qkvProjection.valid && io.directProjection.valid) ||
      (io.qkvProjection.valid && io.gateProjection.valid) ||
      (io.directProjection.valid && io.gateProjection.valid)

  when(clientCollision) { collisionSticky := True }

  weights.io.projection.valid := ownerIdle &&
    (qkvOnly || directOnly || gateOnly)
  weights.io.projection.payload := io.qkvProjection.payload
  when(directOnly) {
    weights.io.projection.payload := io.directProjection.payload
  } elsewhen(gateOnly) {
    weights.io.projection.payload := io.gateProjection.payload
  }

  io.qkvProjection.ready := ownerIdle && qkvOnly &&
    weights.io.projection.ready
  io.directProjection.ready := ownerIdle && directOnly &&
    weights.io.projection.ready
  io.gateProjection.ready := ownerIdle && gateOnly &&
    weights.io.projection.ready

  when(weights.io.projection.fire) {
    when(qkvOnly) {
      owner := BitNetResidentMatrixWeightServiceOwner.SegmentedQkv
    } elsewhen(directOnly) {
      owner := BitNetResidentMatrixWeightServiceOwner.Direct
    } otherwise {
      owner := BitNetResidentMatrixWeightServiceOwner.FusedGateUp
    }
  }

  io.qkvRetirement.valid := False
  io.qkvRetirement.payload := weights.io.retirement.payload
  io.directRetirement.valid := False
  io.directRetirement.payload := weights.io.retirement.payload
  io.gateRetirement.valid := False
  io.gateRetirement.payload := weights.io.retirement.payload
  weights.io.retirement.ready :=
    (owner === BitNetResidentMatrixWeightServiceOwner.Idle) ||
      ((owner === BitNetResidentMatrixWeightServiceOwner.SegmentedQkv) &&
        io.qkvRetirement.ready) ||
      ((owner === BitNetResidentMatrixWeightServiceOwner.Direct) &&
        io.directRetirement.ready) ||
      ((owner === BitNetResidentMatrixWeightServiceOwner.FusedGateUp) &&
        io.gateRetirement.ready)

  switch(owner) {
    is(BitNetResidentMatrixWeightServiceOwner.Idle) {
      // This state is unreachable for a conforming underlying subsystem.  A
      // defensive drain prevents an internal stale retirement from wedging the
      // sole resident weight endpoint.
      when(weights.io.retirement.fire) { serviceSticky := True }
    }
    is(BitNetResidentMatrixWeightServiceOwner.SegmentedQkv) {
      io.qkvRetirement.valid := weights.io.retirement.valid
    }
    is(BitNetResidentMatrixWeightServiceOwner.Direct) {
      io.directRetirement.valid := weights.io.retirement.valid
    }
    is(BitNetResidentMatrixWeightServiceOwner.FusedGateUp) {
      io.gateRetirement.valid := weights.io.retirement.valid
    }
  }

  when(weights.io.retirement.fire && owner =/=
      BitNetResidentMatrixWeightServiceOwner.Idle) {
    owner := BitNetResidentMatrixWeightServiceOwner.Idle
  }

  io.activeOwner := owner
  io.busy := owner =/=
    BitNetResidentMatrixWeightServiceOwner.Idle || weights.io.busy
  io.slotBusy := weights.io.slotBusy
  io.clientCollisionSticky := collisionSticky
  io.protocolFaultSticky := collisionSticky || serviceSticky ||
    weights.io.protocolFaultSticky
}

object GenerateBitNetResidentSharedMatrixWeightService extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-shared-matrix-weight-service")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentSharedMatrixWeightService())
}
