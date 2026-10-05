package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Physical activation-port wiring behind the whole-event lease controller.
  *
  * Layer RMS, attention-context publication and LM-head reads have fixed
  * logical clients.  Projection-image, post-gate and fused-residual traffic
  * instead share one physical operator port and are routed to Matrix, Gate or
  * Residual according to the already-authenticated activation owner.  The
  * owner cannot change while a synchronous read is outstanding because that
  * invariant is enforced by [[BitNetResidentActivationFabricArbiter]].
  *
  * This block owns no activation storage and performs no arithmetic.  It is a
  * fail-stop identity interconnect: traffic before ownership, traffic from an
  * inactive semantic source, or a response from a non-owner client enters a
  * reset-only quarantine so that a stale beat can never be associated with a
  * later transformer event.
  */
class BitNetResidentActivationClientInterconnect(
    model: BitNetConfig = BitNetConfig.Production,
    layout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production))
    extends Component {
  require(layout.cfg == model,
    "activation-client interconnect layout/config mismatch")

  private val lanes = model.tokenParallelism
  private val clients = BitNetActivationClient.elements
  private val clientCount = clients.size
  private val rmsIndex = BitNetActivationClient.Rms.position
  private val attentionIndex = BitNetActivationClient.Attention.position
  private val lmIndex = BitNetActivationClient.LmHead.position

  val io = new Bundle {
    val ownerValid = in Bool()
    val owner = in(BitNetActivationClient())

    /** Private layer-RMS endpoint ports. */
    val rmsReadCommand = Vec(slave(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val rmsReadData = Vec(master(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)
    val rmsWriteData = Vec(slave(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    /**
      * Output of BitNetResidentMatrixActivationPortMux.  The same wires are
      * valid for Matrix, Gate and Residual leases, but never for two at once.
      */
    val matrixReadCommand = Vec(slave(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val matrixReadData = Vec(master(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)
    val matrixWriteData = Vec(slave(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    /** Attention publishes the completed context; it does not read here. */
    val attentionWriteData = Vec(slave(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))), lanes)

    /** Streaming LM head only reads the final NarrowWork activation. */
    val lmReadCommand = Vec(slave(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))), lanes)
    val lmReadData = Vec(master(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))), lanes)

    /** Direct connection to the one activation-fabric subsystem. */
    val clientReadCommand = Vec(
      Vec(master(Stream(
        BitNetPhysicalActivationReadCommand(model, layout))), lanes),
      clientCount)
    val clientReadData = Vec(
      Vec(slave(Stream(
        BitNetPhysicalActivationReadBeat(model, layout))), lanes),
      clientCount)
    val clientWriteData = Vec(
      Vec(master(Stream(
        BitNetPhysicalActivationWriteBeat(model, layout))), lanes),
      clientCount)

    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
  }

  val stickyFault = RegInit(False)
  val quarantine = RegInit(False)

  val rmsSelected = io.ownerValid &&
    io.owner === BitNetActivationClient.Rms
  val attentionSelected = io.ownerValid &&
    io.owner === BitNetActivationClient.Attention
  val lmSelected = io.ownerValid &&
    io.owner === BitNetActivationClient.LmHead
  val matrixSelected = io.ownerValid && (
    io.owner === BitNetActivationClient.Matrix ||
      io.owner === BitNetActivationClient.Gate ||
      io.owner === BitNetActivationClient.Residual)

  for (client <- 0 until clientCount; lane <- 0 until lanes) {
    io.clientReadCommand(client)(lane).valid := False
    io.clientReadCommand(client)(lane).payload :=
      BitNetPhysicalActivationReadCommand(model, layout).getZero
    io.clientReadData(client)(lane).ready := quarantine
    io.clientWriteData(client)(lane).valid := False
    io.clientWriteData(client)(lane).payload :=
      BitNetPhysicalActivationWriteBeat(model, layout).getZero
  }

  for (lane <- 0 until lanes) {
    io.rmsReadCommand(lane).ready := False
    io.rmsReadData(lane).valid := False
    io.rmsReadData(lane).payload := io.clientReadData(rmsIndex)(lane).payload
    io.rmsWriteData(lane).ready := False

    io.matrixReadCommand(lane).ready := False
    io.matrixReadData(lane).valid := False
    io.matrixReadData(lane).payload :=
      BitNetPhysicalActivationReadBeat(model, layout).getZero
    io.matrixWriteData(lane).ready := False

    io.attentionWriteData(lane).ready := False

    io.lmReadCommand(lane).ready := False
    io.lmReadData(lane).valid := False
    io.lmReadData(lane).payload := io.clientReadData(lmIndex)(lane).payload

    when(!quarantine && rmsSelected) {
      io.clientReadCommand(rmsIndex)(lane).valid :=
        io.rmsReadCommand(lane).valid
      io.clientReadCommand(rmsIndex)(lane).payload :=
        io.rmsReadCommand(lane).payload
      io.rmsReadCommand(lane).ready :=
        io.clientReadCommand(rmsIndex)(lane).ready

      io.rmsReadData(lane).valid :=
        io.clientReadData(rmsIndex)(lane).valid
      io.clientReadData(rmsIndex)(lane).ready :=
        io.rmsReadData(lane).ready

      io.clientWriteData(rmsIndex)(lane).valid :=
        io.rmsWriteData(lane).valid
      io.clientWriteData(rmsIndex)(lane).payload :=
        io.rmsWriteData(lane).payload
      io.rmsWriteData(lane).ready :=
        io.clientWriteData(rmsIndex)(lane).ready
    }

    when(!quarantine && attentionSelected) {
      io.clientWriteData(attentionIndex)(lane).valid :=
        io.attentionWriteData(lane).valid
      io.clientWriteData(attentionIndex)(lane).payload :=
        io.attentionWriteData(lane).payload
      io.attentionWriteData(lane).ready :=
        io.clientWriteData(attentionIndex)(lane).ready
    }

    when(!quarantine && lmSelected) {
      io.clientReadCommand(lmIndex)(lane).valid :=
        io.lmReadCommand(lane).valid
      io.clientReadCommand(lmIndex)(lane).payload :=
        io.lmReadCommand(lane).payload
      io.lmReadCommand(lane).ready :=
        io.clientReadCommand(lmIndex)(lane).ready

      io.lmReadData(lane).valid :=
        io.clientReadData(lmIndex)(lane).valid
      io.clientReadData(lmIndex)(lane).ready :=
        io.lmReadData(lane).ready
    }

    // Matrix/Gate/Residual are three logical lease identities sharing the
    // same physical operator-side wires.  Route both command and response by
    // the stable activation owner.
    for (client <- Vector(
        BitNetActivationClient.Matrix,
        BitNetActivationClient.Gate,
        BitNetActivationClient.Residual)) {
      val index = client.position
      when(!quarantine && io.ownerValid && io.owner === client) {
        io.clientReadCommand(index)(lane).valid :=
          io.matrixReadCommand(lane).valid
        io.clientReadCommand(index)(lane).payload :=
          io.matrixReadCommand(lane).payload
        io.matrixReadCommand(lane).ready :=
          io.clientReadCommand(index)(lane).ready

        io.matrixReadData(lane).valid :=
          io.clientReadData(index)(lane).valid
        io.matrixReadData(lane).payload :=
          io.clientReadData(index)(lane).payload
        io.clientReadData(index)(lane).ready :=
          io.matrixReadData(lane).ready

        io.clientWriteData(index)(lane).valid :=
          io.matrixWriteData(lane).valid
        io.clientWriteData(index)(lane).payload :=
          io.matrixWriteData(lane).payload
        io.matrixWriteData(lane).ready :=
          io.clientWriteData(index)(lane).ready
      }
    }

    val rmsTraffic = io.rmsReadCommand(lane).valid ||
      io.rmsWriteData(lane).valid
    val matrixTraffic = io.matrixReadCommand(lane).valid ||
      io.matrixWriteData(lane).valid
    val attentionTraffic = io.attentionWriteData(lane).valid
    val lmTraffic = io.lmReadCommand(lane).valid

    val selectedReadReturn =
      (rmsSelected && io.clientReadData(rmsIndex)(lane).valid) ||
        (lmSelected && io.clientReadData(lmIndex)(lane).valid) ||
        (matrixSelected && (
          (io.owner === BitNetActivationClient.Matrix &&
            io.clientReadData(
              BitNetActivationClient.Matrix.position)(lane).valid) ||
          (io.owner === BitNetActivationClient.Gate &&
            io.clientReadData(
              BitNetActivationClient.Gate.position)(lane).valid) ||
          (io.owner === BitNetActivationClient.Residual &&
            io.clientReadData(
              BitNetActivationClient.Residual.position)(lane).valid)))
    val readReturnValid = Bits(clientCount bits)
    for (client <- 0 until clientCount) {
      readReturnValid(client) := io.clientReadData(client)(lane).valid
    }
    val anyReadReturn = readReturnValid.orR

    when(!quarantine && (
        (rmsTraffic && !rmsSelected) ||
          (matrixTraffic && !matrixSelected) ||
          (attentionTraffic && !attentionSelected) ||
          (lmTraffic && !lmSelected) ||
          (anyReadReturn && !selectedReadReturn))) {
      stickyFault := True
      quarantine := True
    }
  }

  io.protocolFaultSticky := stickyFault
  io.quarantined := quarantine
}

object GenerateBitNetResidentActivationClientInterconnect extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-activation-client-interconnect")
  val model = BitNetConfig.Production
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentActivationClientInterconnect(
      model, BitNetStreamingActivationLayout(model)))
}
