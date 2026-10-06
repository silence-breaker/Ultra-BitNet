package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** Complete resident RMS layer unit: AUX staging plus activation arithmetic. */
case class BitNetResidentRmsNormSubsystemConfig(
    endpoint: BitNetResidentRmsNormEndpointConfig =
      BitNetResidentRmsNormEndpointConfig(),
    gamma: BitNetResidentRmsGammaStagerConfig =
      BitNetResidentRmsGammaStagerConfig()
) {
  require(gamma.endpoint == endpoint,
    "RMS gamma stager and arithmetic endpoint must share one typed contract")

  val externalArithmetic: Boolean = endpoint.externalArithmetic
}

/**
  * Production boundary of the RmsNorm owner selected by the layer router.
  *
  * The gamma request/response streams are private wires here: the public
  * boundary contains the semantic layer step, activation-bank ports and one
  * bank-local AUX read client.  With the default endpoint configuration the
  * RMS arithmetic remains private.  Production composition selects
  * `externalArithmetic=true`, exposing the same typed measure/replay streams
  * so layer RMS and post-gate RMS can share exactly one physical core.  This
  * remains a PL-local wire boundary; software cannot insert a callback.
  */
class BitNetResidentRmsNormSubsystem(
    cfg: BitNetResidentRmsNormSubsystemConfig =
      BitNetResidentRmsNormSubsystemConfig()) extends Component {
  private val model = cfg.endpoint.model
  private val layout = cfg.endpoint.activationLayout
  private val crossbar = cfg.gamma.crossbar

  val io = new Bundle {
    val command = slave(Stream(BitNetResidentLayerStep(model)))
    val completion = master(Stream(
      BitNetResidentLayerStepCompletion(model)))

    val activationReadCommand = Vec(master(Stream(
      BitNetPhysicalActivationReadCommand(model, layout))),
      model.tokenParallelism)
    val activationReadData = Vec(slave(Stream(
      BitNetPhysicalActivationReadBeat(model, layout))),
      model.tokenParallelism)
    val activationWriteData = Vec(master(Stream(
      BitNetPhysicalActivationWriteBeat(model, layout))),
      model.tokenParallelism)

    val auxReadRequest = master(Stream(
      BitNetClientBankReadRequest(crossbar)))
    val auxReadResponse = slave(Stream(
      BitNetClientBankReadResponse(crossbar)))

    /** Optional model-wide RMS service boundary. */
    val rmsCommand = if (cfg.externalArithmetic)
      master(Stream(BitNetSharedRmsCommand(cfg.endpoint.arithmetic)))
    else null
    val rmsMeasure = if (cfg.externalArithmetic)
      master(Stream(BitNetSharedRmsMeasureBeat(cfg.endpoint.arithmetic)))
    else null
    val rmsReplayRequest = if (cfg.externalArithmetic)
      slave(Stream(BitNetSharedRmsReplayRequest(cfg.endpoint.arithmetic)))
    else null
    val rmsReplay = if (cfg.externalArithmetic)
      master(Stream(BitNetSharedRmsReplayBeat(cfg.endpoint.arithmetic)))
    else null
    val rmsResult = if (cfg.externalArithmetic)
      slave(Stream(BitNetSharedRmsResultBeat(cfg.endpoint.arithmetic)))
    else null
    val rmsDone = if (cfg.externalArithmetic)
      slave(Stream(BitNetSharedRmsDone(cfg.endpoint.arithmetic)))
    else null

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
  }

  val endpoint = new BitNetResidentRmsNormEndpoint(cfg.endpoint)
  val gammaStager = new BitNetResidentRmsGammaStager(cfg.gamma)

  endpoint.io.command << io.command
  io.completion << endpoint.io.completion

  gammaStager.io.request << endpoint.io.gammaRequest
  endpoint.io.gammaInput << gammaStager.io.gamma
  io.auxReadRequest << gammaStager.io.readRequest
  gammaStager.io.readResponse << io.auxReadResponse

  if (cfg.externalArithmetic) {
    io.rmsCommand << endpoint.io.rmsCommand
    io.rmsMeasure << endpoint.io.rmsMeasure
    endpoint.io.rmsReplayRequest << io.rmsReplayRequest
    io.rmsReplay << endpoint.io.rmsReplay
    endpoint.io.rmsResult << io.rmsResult
    endpoint.io.rmsDone << io.rmsDone
  }

  for (lane <- 0 until model.tokenParallelism) {
    io.activationReadCommand(lane) <<
      endpoint.io.activationReadCommand(lane)
    endpoint.io.activationReadData(lane) << io.activationReadData(lane)
    io.activationWriteData(lane) << endpoint.io.activationWriteData(lane)
  }

  io.busy := endpoint.io.busy || gammaStager.io.busy
  io.protocolFaultSticky := endpoint.io.protocolFaultSticky ||
    gammaStager.io.protocolFaultSticky
  io.overflowSticky := endpoint.io.overflowSticky
}

object GenerateBitNetResidentRmsNormSubsystem extends App {
  SpinalConfig(
    targetDirectory = "build/bitnet-resident-rmsnorm-subsystem",
    oneFilePerComponent = false
  ).generateVerilog(new BitNetResidentRmsNormSubsystem())
}
