package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** Static clients of the one physical resident-activation fabric. */
object BitNetActivationClient extends SpinalEnum(binarySequential) {
  val Rms, Matrix, Attention, Residual, Gate, LmHead = newElement()
}

case class BitNetActivationFabricLease(cfg: BitNetConfig) extends Bundle {
  val client = BitNetActivationClient()
  val requestId = UInt(cfg.requestIdWidth bits)
}

/**
  * PL-local, whole-operation ownership switch for resident activations.
  *
  * The fixed layer controller runs only one semantic owner at a time.  This
  * arbiter turns that fact into a physical invariant: all clients share the
  * same four-lane [[BitNetStreamingActivationFabric]], and ownership cannot
  * be released while any accepted synchronous read still has an unconsumed
  * response.  Writes are retired only by their real fabric handshake.
  *
  * This component contains no activation memory, packet parser, DMA or host
  * callback.  It is only a tagged Stream switch and completion barrier.
  */
class BitNetResidentActivationFabricArbiter(
    cfg: BitNetConfig = BitNetConfig.Production,
    layout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production)
) extends Component {
  require(layout.cfg == cfg, "activation arbiter layout/config mismatch")

  private val clients = BitNetActivationClient.elements
  private val clientCount = clients.size

  val io = new Bundle {
    val acquire = slave(Stream(BitNetActivationFabricLease(cfg)))
    val release = slave(Stream(BitNetActivationFabricLease(cfg)))

    val clientReadCommand = Vec(
      Vec(slave(Stream(BitNetPhysicalActivationReadCommand(cfg, layout))),
        cfg.tokenParallelism),
      clientCount)
    val clientReadData = Vec(
      Vec(master(Stream(BitNetPhysicalActivationReadBeat(cfg, layout))),
        cfg.tokenParallelism),
      clientCount)
    val clientWriteData = Vec(
      Vec(slave(Stream(BitNetPhysicalActivationWriteBeat(cfg, layout))),
        cfg.tokenParallelism),
      clientCount)

    val fabricReadCommand = Vec(
      master(Stream(BitNetPhysicalActivationReadCommand(cfg, layout))),
      cfg.tokenParallelism)
    val fabricReadData = Vec(
      slave(Stream(BitNetPhysicalActivationReadBeat(cfg, layout))),
      cfg.tokenParallelism)
    val fabricWriteData = Vec(
      master(Stream(BitNetPhysicalActivationWriteBeat(cfg, layout))),
      cfg.tokenParallelism)

    val ownerValid = out Bool()
    val owner = out(BitNetActivationClient())
    val ownerRequestId = out UInt(cfg.requestIdWidth bits)
    val readOutstanding = out Bits(cfg.tokenParallelism bits)
    val protocolFaultSticky = out Bool()
  }

  val ownerValid = RegInit(False)
  val owner = Reg(BitNetActivationClient()) init BitNetActivationClient.Rms
  val ownerRequestId = Reg(UInt(cfg.requestIdWidth bits)) init 0
  val readOutstanding = Reg(Bits(cfg.tokenParallelism bits)) init 0
  val protocolFault = RegInit(False)

  io.ownerValid := ownerValid
  io.owner := owner
  io.ownerRequestId := ownerRequestId
  io.readOutstanding := readOutstanding
  io.protocolFaultSticky := protocolFault

  io.acquire.ready := !ownerValid

  val ownerHasPendingCommand = Bool()
  ownerHasPendingCommand := False
  for ((client, clientIndex) <- clients.zipWithIndex) {
    when(ownerValid && owner === client) {
      for (lane <- 0 until cfg.tokenParallelism) {
        when(io.clientReadCommand(clientIndex)(lane).valid ||
          io.clientWriteData(clientIndex)(lane).valid) {
          ownerHasPendingCommand := True
        }
      }
    }
  }

  val releaseMatches = ownerValid &&
    io.release.payload.client === owner &&
    io.release.payload.requestId === ownerRequestId
  io.release.ready := releaseMatches && !readOutstanding.orR &&
    !ownerHasPendingCommand

  when(io.acquire.fire) {
    ownerValid := True
    owner := io.acquire.payload.client
    ownerRequestId := io.acquire.payload.requestId
  }
  when(io.acquire.valid && ownerValid) {
    protocolFault := True
  }
  when(io.release.valid && !releaseMatches) {
    protocolFault := True
  }
  when(io.release.fire) {
    ownerValid := False
    readOutstanding := 0
  }

  for (lane <- 0 until cfg.tokenParallelism) {
    io.fabricReadCommand(lane).valid := False
    io.fabricReadCommand(lane).payload :=
      BitNetPhysicalActivationReadCommand(cfg, layout).getZero
    io.fabricReadData(lane).ready := False
    io.fabricWriteData(lane).valid := False
    io.fabricWriteData(lane).payload :=
      BitNetPhysicalActivationWriteBeat(cfg, layout).getZero

    for ((client, clientIndex) <- clients.zipWithIndex) {
      io.clientReadCommand(clientIndex)(lane).ready := False
      io.clientReadData(clientIndex)(lane).valid := False
      io.clientReadData(clientIndex)(lane).payload :=
        io.fabricReadData(lane).payload
      io.clientWriteData(clientIndex)(lane).ready := False

      when(ownerValid && owner === client) {
        io.fabricReadCommand(lane).valid :=
          io.clientReadCommand(clientIndex)(lane).valid
        io.fabricReadCommand(lane).payload :=
          io.clientReadCommand(clientIndex)(lane).payload
        io.clientReadCommand(clientIndex)(lane).ready :=
          io.fabricReadCommand(lane).ready

        io.clientReadData(clientIndex)(lane).valid :=
          io.fabricReadData(lane).valid
        io.fabricReadData(lane).ready :=
          io.clientReadData(clientIndex)(lane).ready

        io.fabricWriteData(lane).valid :=
          io.clientWriteData(clientIndex)(lane).valid
        io.fabricWriteData(lane).payload :=
          io.clientWriteData(clientIndex)(lane).payload
        io.clientWriteData(clientIndex)(lane).ready :=
          io.fabricWriteData(lane).ready
      }

      // An inactive owner is never allowed to contend for the physical RAM.
      when(ownerValid && owner =/= client &&
        (io.clientReadCommand(clientIndex)(lane).valid ||
          io.clientWriteData(clientIndex)(lane).valid)) {
        protocolFault := True
      }
    }

    val readIssued = io.fabricReadCommand(lane).fire
    val readReturned = io.fabricReadData(lane).fire
    when(readIssued && readOutstanding(lane) && !readReturned) {
      // The physical fabric is a one-response elastic port per token lane.
      protocolFault := True
    }
    when(readIssued =/= readReturned) {
      readOutstanding(lane) := readIssued
    }
  }
}

object GenerateBitNetResidentActivationFabricArbiter extends App {
  val target = args.headOption.getOrElse(
    "build/bitnet-resident-activation-fabric-arbiter")
  val cfg = BitNetConfig.Production
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentActivationFabricArbiter(
      cfg, BitNetStreamingActivationLayout(cfg)))
}
