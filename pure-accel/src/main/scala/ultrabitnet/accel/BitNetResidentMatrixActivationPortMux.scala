package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Internal phase that owns an outstanding synchronous activation read. */
object BitNetResidentMatrixActivationReadOwner
    extends SpinalEnum(binarySequential) {
  val ProjectionImage, PostGateRms, FusedResidual = newElement()
}

/**
  * Lossless per-lane port mux for one whole matrix-event activation lease.
  *
  * Projection-image measure/replay, Gate post-normalisation and O/Down fused
  * residual execute in mutually exclusive phases of the same outer event.
  * This block lets them use one physical activation-fabric client without a
  * mid-event lease handoff.  A synchronous read owner is retained until its
  * response handshake; write arbitration has no hidden buffering.  Any phase
  * overlap is a real protocol error and enters reset-only quarantine rather
  * than silently selecting a corrupt tensor source.
  */
class BitNetResidentMatrixActivationPortMux(
    model: BitNetConfig = BitNetConfig.Production,
    layout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production))
    extends Component {
  require(layout.cfg == model)
  private val lanes = model.tokenParallelism

  val io = new Bundle {
    val projectionReadCommand = Vec(slave(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val projectionReadData = Vec(master(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)

    val gateWriteData = Vec(slave(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    val postGateReadCommand = Vec(slave(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val postGateReadData = Vec(master(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)
    val postGateWriteData = Vec(slave(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    val residualReadCommand = Vec(slave(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val residualReadData = Vec(master(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)
    val residualWriteData = Vec(slave(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    val fabricReadCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val fabricReadData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)
    val fabricWriteData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    val readOutstanding = out Bits(lanes bits)
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
  }

  val readOutstanding = Reg(Bits(lanes bits)) init 0
  val readOwner = Vec(
    Reg(BitNetResidentMatrixActivationReadOwner()) init
      BitNetResidentMatrixActivationReadOwner.ProjectionImage,
    lanes)
  val stickyFault = RegInit(False)
  val quarantine = RegInit(False)

  for (lane <- 0 until lanes) {
    val projectionRead = io.projectionReadCommand(lane).valid
    val postGateRead = io.postGateReadCommand(lane).valid
    val residualRead = io.residualReadCommand(lane).valid
    val readCollision =
      (projectionRead && postGateRead) ||
        (projectionRead && residualRead) ||
        (postGateRead && residualRead)

    io.fabricReadCommand(lane).valid := False
    io.fabricReadCommand(lane).payload :=
      io.projectionReadCommand(lane).payload
    io.projectionReadCommand(lane).ready := False
    io.postGateReadCommand(lane).ready := False
    io.residualReadCommand(lane).ready := False

    when(!quarantine && !readOutstanding(lane) && !readCollision) {
      when(projectionRead) {
        io.fabricReadCommand(lane).valid := True
        io.fabricReadCommand(lane).payload :=
          io.projectionReadCommand(lane).payload
        io.projectionReadCommand(lane).ready :=
          io.fabricReadCommand(lane).ready
      } elsewhen(postGateRead) {
        io.fabricReadCommand(lane).valid := True
        io.fabricReadCommand(lane).payload :=
          io.postGateReadCommand(lane).payload
        io.postGateReadCommand(lane).ready :=
          io.fabricReadCommand(lane).ready
      } elsewhen(residualRead) {
        io.fabricReadCommand(lane).valid := True
        io.fabricReadCommand(lane).payload :=
          io.residualReadCommand(lane).payload
        io.residualReadCommand(lane).ready :=
          io.fabricReadCommand(lane).ready
      }
    }

    when(io.fabricReadCommand(lane).fire) {
      readOutstanding(lane) := True
      when(postGateRead) {
        readOwner(lane) :=
          BitNetResidentMatrixActivationReadOwner.PostGateRms
      } elsewhen(residualRead) {
        readOwner(lane) :=
          BitNetResidentMatrixActivationReadOwner.FusedResidual
      } otherwise {
        readOwner(lane) :=
          BitNetResidentMatrixActivationReadOwner.ProjectionImage
      }
    }

    io.projectionReadData(lane).valid := False
    io.projectionReadData(lane).payload := io.fabricReadData(lane).payload
    io.postGateReadData(lane).valid := False
    io.postGateReadData(lane).payload := io.fabricReadData(lane).payload
    io.residualReadData(lane).valid := False
    io.residualReadData(lane).payload := io.fabricReadData(lane).payload
    io.fabricReadData(lane).ready := quarantine ||
      !readOutstanding(lane)

    when(!quarantine && readOutstanding(lane)) {
      switch(readOwner(lane)) {
        is(BitNetResidentMatrixActivationReadOwner.ProjectionImage) {
          io.projectionReadData(lane).valid :=
            io.fabricReadData(lane).valid
          io.fabricReadData(lane).ready :=
            io.projectionReadData(lane).ready
        }
        is(BitNetResidentMatrixActivationReadOwner.PostGateRms) {
          io.postGateReadData(lane).valid := io.fabricReadData(lane).valid
          io.fabricReadData(lane).ready :=
            io.postGateReadData(lane).ready
        }
        is(BitNetResidentMatrixActivationReadOwner.FusedResidual) {
          io.residualReadData(lane).valid := io.fabricReadData(lane).valid
          io.fabricReadData(lane).ready :=
            io.residualReadData(lane).ready
        }
      }
    }
    when(io.fabricReadData(lane).fire) {
      readOutstanding(lane) := False
    }

    val gateWrite = io.gateWriteData(lane).valid
    val postGateWrite = io.postGateWriteData(lane).valid
    val residualWrite = io.residualWriteData(lane).valid
    val writeCollision =
      (gateWrite && postGateWrite) ||
        (gateWrite && residualWrite) ||
        (postGateWrite && residualWrite)

    io.fabricWriteData(lane).valid := False
    io.fabricWriteData(lane).payload := io.gateWriteData(lane).payload
    io.gateWriteData(lane).ready := False
    io.postGateWriteData(lane).ready := False
    io.residualWriteData(lane).ready := False
    when(!quarantine && !writeCollision) {
      when(gateWrite) {
        io.fabricWriteData(lane).valid := True
        io.fabricWriteData(lane).payload := io.gateWriteData(lane).payload
        io.gateWriteData(lane).ready := io.fabricWriteData(lane).ready
      } elsewhen(postGateWrite) {
        io.fabricWriteData(lane).valid := True
        io.fabricWriteData(lane).payload :=
          io.postGateWriteData(lane).payload
        io.postGateWriteData(lane).ready :=
          io.fabricWriteData(lane).ready
      } elsewhen(residualWrite) {
        io.fabricWriteData(lane).valid := True
        io.fabricWriteData(lane).payload :=
          io.residualWriteData(lane).payload
        io.residualWriteData(lane).ready :=
          io.fabricWriteData(lane).ready
      }
    }

    val unexpectedReadReturn = io.fabricReadData(lane).valid &&
      !readOutstanding(lane)
    when(readCollision || writeCollision || unexpectedReadReturn) {
      stickyFault := True
      quarantine := True
    }
  }

  io.readOutstanding := readOutstanding
  io.protocolFaultSticky := stickyFault
  io.quarantined := quarantine
}

object GenerateBitNetResidentMatrixActivationPortMux extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-matrix-activation-port-mux")
  val model = BitNetConfig.Production
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentMatrixActivationPortMux(
      model, BitNetStreamingActivationLayout(model)))
}
