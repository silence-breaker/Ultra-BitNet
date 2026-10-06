package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** The only two legal clients of the production resident linear plane. */
object BitNetResidentLinearPlaneOwner extends SpinalEnum(binarySequential) {
  val Idle, LayerMatrix, LmHead = newElement()
}

/** Sticky protocol diagnostics.  Identity faults on the physical return
  * boundary additionally put the router into irreversible quarantine until
  * reset; feeder-side faults merely stop the offending producer.
  */
object BitNetResidentLinearPlaneRouterFault {
  val InvalidLayerLaunch = 0
  val InvalidLmHeadLaunch = 1
  val NonOwnerLayerFeeder = 2
  val NonOwnerLmHeadFeeder = 3
  val LayerFeederIdentity = 4
  val LmHeadFeederIdentity = 5
  val DownstreamResultIdentity = 6
  val DownstreamCompletionIdentity = 7
}

case class BitNetResidentLinearPlaneOwnerRouterConfig(
    datapath: BitNetResidentSharedTp4ComputeDatapathConfig =
      BitNetResidentSharedTp4ComputeDatapathConfig()) {
  val model: BitNetConfig = datapath.model
  val feeder: BitNetContinuousMatrixFeederConfig = datapath.feeder
  val endpoint: BitNetResidentLinearEndpointConfig =
    datapath.linearEndpoint

  require(endpoint.model == model)
  require(endpoint.feeder == feeder)
}

/**
  * Production integration seam in front of the sole resident TP4 plane.
  *
  * This is intentionally only an ownership and authentication component.  It
  * owns no feeder, matrix array, arithmetic pool, RAM or quantizer.  A legal
  * launch is selected with round-robin arbitration and the selected client is
  * then locked until its matching downstream completion is accepted by that
  * same client.  A launch left asserted while another client owns the plane is
  * simply queued by backpressure and is not a protocol error.
  *
  * The two feeder interfaces retain their native production types: Map0 is a
  * layer-matrix-only stream and I8 is an LM-head-only stream.  Request/layer/
  * kind are therefore implicit in the locked launch while every feeder beat
  * authenticates its explicit sequence (and Map0 mode).  Result/completion
  * returns carry and authenticate the complete transaction identity.
  */
class BitNetResidentLinearPlaneOwnerRouter(
    cfg: BitNetResidentLinearPlaneOwnerRouterConfig =
      BitNetResidentLinearPlaneOwnerRouterConfig()) extends Component {
  private val model = cfg.model
  private val feeder = cfg.feeder
  private val endpoint = cfg.endpoint

  val io = new Bundle {
    // Layer MatrixStageControl + ActivationReplayBinder client.
    val layerMatrixLaunch = slave(Stream(
      BitNetResidentLinearProgram(endpoint)))
    val layerMatrixMap0RawGroup = slave(Stream(
      BitNetMap0RawGroupTile(feeder, model.ffnSize)))
    val layerMatrixActivationTile = slave(Stream(
      BitNetNormalizedActivationTile(feeder)))
    val layerMatrixResult = master(Stream(
      BitNetResidentFinalFeaturePair(endpoint)))
    val layerMatrixCompletion = master(Stream(
      BitNetResidentLinearCompletion(endpoint)))

    // ResidentLmHeadSubsystem client.
    val lmHeadLaunch = slave(Stream(
      BitNetResidentLinearLaunch(endpoint)))
    val lmHeadI8WeightBeat = slave(Stream(BitNetI8WeightBeat(feeder)))
    val lmHeadActivationTile = slave(Stream(
      BitNetNormalizedActivationTile(feeder)))
    val lmHeadResult = master(Stream(
      BitNetResidentFinalFeaturePair(endpoint)))
    val lmHeadCompletion = master(Stream(
      BitNetResidentLinearCompletion(endpoint)))

    // Unique BitNetResidentSharedTp4ComputeDatapath linear/feeder ports.
    val linearLaunch = master(Stream(
      BitNetResidentLinearProgram(endpoint)))
    val map0RawGroup = master(Stream(
      BitNetMap0RawGroupTile(feeder, model.ffnSize)))
    val i8WeightBeat = master(Stream(BitNetI8WeightBeat(feeder)))
    val activationTile = master(Stream(
      BitNetNormalizedActivationTile(feeder)))
    val linearResult = slave(Stream(
      BitNetResidentFinalFeaturePair(endpoint)))
    val linearCompletion = slave(Stream(
      BitNetResidentLinearCompletion(endpoint)))

    val activeOwner = out(BitNetResidentLinearPlaneOwner())
    val busy = out Bool()
    val roundRobinPreferLmHead = out Bool()
    val quarantined = out Bool()
    val protocolFaultSticky = out Bool()
    val faultBits = out Bits(16 bits)
  }

  val owner = Reg(BitNetResidentLinearPlaneOwner()) init
    BitNetResidentLinearPlaneOwner.Idle
  val heldProgram = Reg(BitNetResidentLinearProgram(endpoint)) init
    BitNetResidentLinearProgram(endpoint).getZero
  // Layer wins the first tie.  Every accepted launch gives the next tie to
  // the other client, independent of transaction duration or backpressure.
  val preferLmHead = RegInit(False)
  val quarantine = RegInit(False)
  val faultBitsReg = Reg(Bits(16 bits)) init 0

  val ownerIdle = owner === BitNetResidentLinearPlaneOwner.Idle
  val ownerLayer = owner === BitNetResidentLinearPlaneOwner.LayerMatrix
  val ownerLmHead = owner === BitNetResidentLinearPlaneOwner.LmHead
  val ownerActive = ownerLayer || ownerLmHead

  /* ------------------------------------------------------------------
   * Legal-only, starvation-free launch arbitration.
   * ------------------------------------------------------------------ */
  // Layer launches were fully authenticated at the Binder's only
  // legacy-to-compact boundary.  Keep the hot registered seam compact and
  // apply only canonical-program checks here; LM remains a legacy client and
  // therefore retains its complete launch validator below.
  val layerLaunchLegal = BitNetResidentLinearProgramCodec.programLegal(
    io.layerMatrixLaunch.payload, endpoint) &&
    io.layerMatrixLaunch.kind =/= BitNetResidentLinearKind.LmHead
  val lmHeadLaunchLegal = BitNetResidentLinearProtocol.launchLegal(
    io.lmHeadLaunch.payload, endpoint) &&
    BitNetResidentLinearProgramCodec.launchEncodable(
      io.lmHeadLaunch.payload, endpoint) &&
    io.lmHeadLaunch.kind === BitNetResidentLinearKind.LmHead
  val layerEligible = io.layerMatrixLaunch.valid && layerLaunchLegal
  val lmHeadEligible = io.lmHeadLaunch.valid && lmHeadLaunchLegal
  val grantLayer = ownerIdle && !quarantine && layerEligible &&
    (!lmHeadEligible || !preferLmHead)
  val grantLmHead = ownerIdle && !quarantine && lmHeadEligible &&
    (!layerEligible || preferLmHead)

  val lmHeadProgram = BitNetResidentLinearProgramCodec.encode(
    io.lmHeadLaunch.payload, endpoint)
  io.linearLaunch.valid := grantLayer || grantLmHead
  io.linearLaunch.payload := io.layerMatrixLaunch.payload
  when(grantLmHead) {
    io.linearLaunch.payload := lmHeadProgram
  }
  io.layerMatrixLaunch.ready := grantLayer && io.linearLaunch.ready
  io.lmHeadLaunch.ready := grantLmHead && io.linearLaunch.ready

  when(io.layerMatrixLaunch.valid && !layerLaunchLegal) {
    faultBitsReg(BitNetResidentLinearPlaneRouterFault.InvalidLayerLaunch) :=
      True
  }
  when(io.lmHeadLaunch.valid && !lmHeadLaunchLegal) {
    faultBitsReg(BitNetResidentLinearPlaneRouterFault.InvalidLmHeadLaunch) :=
      True
  }

  when(io.linearLaunch.fire) {
    heldProgram := io.linearLaunch.payload
    when(grantLayer) {
      owner := BitNetResidentLinearPlaneOwner.LayerMatrix
      preferLmHead := True
    } otherwise {
      owner := BitNetResidentLinearPlaneOwner.LmHead
      preferLmHead := False
    }
  }

  /* ------------------------------------------------------------------
   * Owner- and identity-authenticated native feeder routing.
   * ------------------------------------------------------------------ */
  io.map0RawGroup.valid := False
  io.map0RawGroup.payload := io.layerMatrixMap0RawGroup.payload
  io.layerMatrixMap0RawGroup.ready := False
  io.i8WeightBeat.valid := False
  io.i8WeightBeat.payload := io.lmHeadI8WeightBeat.payload
  io.lmHeadI8WeightBeat.ready := False
  io.activationTile.valid := False
  io.activationTile.payload := io.layerMatrixActivationTile.payload
  io.layerMatrixActivationTile.ready := False
  io.lmHeadActivationTile.ready := False

  val layerMap0Identity =
    io.layerMatrixMap0RawGroup.sequenceId === heldProgram.sequenceId &&
      io.layerMatrixMap0RawGroup.scheduleMode === heldProgram.mode
  val layerActivationIdentity =
    io.layerMatrixActivationTile.sequenceId === heldProgram.sequenceId
  val lmHeadI8Identity =
    io.lmHeadI8WeightBeat.sequenceId === heldProgram.sequenceId &&
      io.lmHeadI8WeightBeat.use === BitNetI8WeightUse.LmHead
  val lmHeadActivationIdentity =
    io.lmHeadActivationTile.sequenceId === heldProgram.sequenceId

  when(ownerLayer && !quarantine) {
    io.map0RawGroup.valid := io.layerMatrixMap0RawGroup.valid &&
      layerMap0Identity
    io.layerMatrixMap0RawGroup.ready := io.map0RawGroup.ready &&
      layerMap0Identity
    io.activationTile.valid := io.layerMatrixActivationTile.valid &&
      layerActivationIdentity
    io.activationTile.payload := io.layerMatrixActivationTile.payload
    io.layerMatrixActivationTile.ready := io.activationTile.ready &&
      layerActivationIdentity

    when(io.layerMatrixMap0RawGroup.valid && !layerMap0Identity) {
      faultBitsReg(
        BitNetResidentLinearPlaneRouterFault.LayerFeederIdentity) := True
    }
    when(io.layerMatrixActivationTile.valid && !layerActivationIdentity) {
      faultBitsReg(
        BitNetResidentLinearPlaneRouterFault.LayerFeederIdentity) := True
    }
    when(io.lmHeadI8WeightBeat.valid || io.lmHeadActivationTile.valid) {
      faultBitsReg(
        BitNetResidentLinearPlaneRouterFault.NonOwnerLmHeadFeeder) := True
    }
  } elsewhen(ownerLmHead && !quarantine) {
    io.i8WeightBeat.valid := io.lmHeadI8WeightBeat.valid &&
      lmHeadI8Identity
    io.lmHeadI8WeightBeat.ready := io.i8WeightBeat.ready &&
      lmHeadI8Identity
    io.activationTile.valid := io.lmHeadActivationTile.valid &&
      lmHeadActivationIdentity
    io.activationTile.payload := io.lmHeadActivationTile.payload
    io.lmHeadActivationTile.ready := io.activationTile.ready &&
      lmHeadActivationIdentity

    when(io.lmHeadI8WeightBeat.valid && !lmHeadI8Identity) {
      faultBitsReg(
        BitNetResidentLinearPlaneRouterFault.LmHeadFeederIdentity) := True
    }
    when(io.lmHeadActivationTile.valid && !lmHeadActivationIdentity) {
      faultBitsReg(
        BitNetResidentLinearPlaneRouterFault.LmHeadFeederIdentity) := True
    }
    when(io.layerMatrixMap0RawGroup.valid ||
        io.layerMatrixActivationTile.valid) {
      faultBitsReg(
        BitNetResidentLinearPlaneRouterFault.NonOwnerLayerFeeder) := True
    }
  } otherwise {
    when(io.layerMatrixMap0RawGroup.valid ||
        io.layerMatrixActivationTile.valid) {
      faultBitsReg(
        BitNetResidentLinearPlaneRouterFault.NonOwnerLayerFeeder) := True
    }
    when(io.lmHeadI8WeightBeat.valid || io.lmHeadActivationTile.valid) {
      faultBitsReg(
        BitNetResidentLinearPlaneRouterFault.NonOwnerLmHeadFeeder) := True
    }
  }

  /* ------------------------------------------------------------------
   * Complete-identity return authentication and fail-stop quarantine.
   * ------------------------------------------------------------------ */
  val resultIdentityMatches = ownerActive &&
    io.linearResult.requestId === heldProgram.requestId &&
    io.linearResult.layer === heldProgram.layer &&
    io.linearResult.kind === heldProgram.kind &&
    io.linearResult.mode === heldProgram.mode &&
    io.linearResult.sequenceId === heldProgram.sequenceId
  val completionIdentityMatches = ownerActive &&
    io.linearCompletion.requestId === heldProgram.requestId &&
    io.linearCompletion.layer === heldProgram.layer &&
    io.linearCompletion.kind === heldProgram.kind &&
    io.linearCompletion.sequenceId === heldProgram.sequenceId
  val badResultIdentity = io.linearResult.valid && !resultIdentityMatches
  val badCompletionIdentity = io.linearCompletion.valid &&
    !completionIdentityMatches
  // A malformed return wins over every nominal handshake in that cycle.
  val healthyReturnCycle = !quarantine && !badResultIdentity &&
    !badCompletionIdentity

  io.layerMatrixResult.valid := False
  io.layerMatrixResult.payload := io.linearResult.payload
  io.lmHeadResult.valid := False
  io.lmHeadResult.payload := io.linearResult.payload
  io.linearResult.ready := quarantine || badResultIdentity ||
    badCompletionIdentity

  io.layerMatrixCompletion.valid := False
  io.layerMatrixCompletion.payload := io.linearCompletion.payload
  io.lmHeadCompletion.valid := False
  io.lmHeadCompletion.payload := io.linearCompletion.payload
  io.linearCompletion.ready := quarantine || badResultIdentity ||
    badCompletionIdentity

  when(healthyReturnCycle && ownerLayer) {
    io.layerMatrixResult.valid := io.linearResult.valid
    io.linearResult.ready := io.layerMatrixResult.ready
    io.layerMatrixCompletion.valid := io.linearCompletion.valid
    io.linearCompletion.ready := io.layerMatrixCompletion.ready
  } elsewhen(healthyReturnCycle && ownerLmHead) {
    io.lmHeadResult.valid := io.linearResult.valid
    io.linearResult.ready := io.lmHeadResult.ready
    io.lmHeadCompletion.valid := io.linearCompletion.valid
    io.linearCompletion.ready := io.lmHeadCompletion.ready
  }

  when(healthyReturnCycle && io.linearCompletion.fire) {
    // `healthyReturnCycle` proves both owner and full completion identity.
    owner := BitNetResidentLinearPlaneOwner.Idle
  }

  when(badResultIdentity) {
    faultBitsReg(
      BitNetResidentLinearPlaneRouterFault.DownstreamResultIdentity) := True
    quarantine := True
  }
  when(badCompletionIdentity) {
    faultBitsReg(
      BitNetResidentLinearPlaneRouterFault.DownstreamCompletionIdentity) :=
      True
    quarantine := True
  }

  io.activeOwner := owner
  io.busy := ownerActive || quarantine
  io.roundRobinPreferLmHead := preferLmHead
  io.quarantined := quarantine
  io.faultBits := faultBitsReg
  io.protocolFaultSticky := faultBitsReg.orR
}

object GenerateBitNetResidentLinearPlaneOwnerRouter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-resident-linear-plane-owner-router")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentLinearPlaneOwnerRouter())
}
