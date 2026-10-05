package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Packet-free bridge from the bank-local KV/context scheduler to one
  * production continuous matrix-array lane.
  *
  * The KV coordinator was intentionally written against a mode-locked
  * job/phase boundary so that it did not own a second multiplier array.  This
  * bridge preserves that boundary while translating every phase into a beat
  * of [[BitNetContinuousMatrixArrayCore]].  It contains no arithmetic and no
  * context storage: alpha/context scaling and probability/V AXPY still execute
  * in the same 143-DSP array used by projection, QK and LM-head dot products.
  */
class BitNetContinuousSvAdapter(
    cfg: BitNetContinuousMatrixConfig = BitNetContinuousMatrixConfig()
) extends Component {
  private val b = cfg.base

  val io = new Bundle {
    val job = slave(Stream(BitNetUnifiedJob(b)))
    val phase = slave(Stream(BitNetUnifiedMappedPhase(b)))

    val sequence = master(Stream(BitNetContinuousSequence(cfg)))
    val beat = master(Stream(BitNetContinuousMatrixBeat(cfg)))
    val continuousResult =
      slave(Stream(BitNetContinuousSvRawResult(cfg)))
    val result = master(Stream(BitNetUnifiedSvRawResult(b)))

    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedPhases = out UInt(32 bits)
    val returnedResults = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Idle, IssueSequence, Run = newElement()
  }
  val state = Reg(State()) init State.Idle

  val activeMode = Reg(BitNetUnifiedMatrixMode()) init
    BitNetUnifiedMatrixMode.SvContextScaleQ15
  val activeFormat = Reg(BitNetUnifiedOperandFormat()) init
    BitNetUnifiedOperandFormat.Q15TileAlpha
  val expectedPhases = Reg(UInt(b.phaseCountWidth bits)) init 0
  val phaseCount = Reg(UInt(b.phaseCountWidth bits)) init 0
  val resultCount = Reg(UInt(b.phaseCountWidth bits)) init 0
  val sequenceId = Reg(UInt(cfg.sequenceIdWidth bits)) init 0
  val sequenceCounter = Reg(UInt(cfg.sequenceIdWidth bits)) init 0

  val fault = RegInit(False)
  val accepted = Reg(UInt(32 bits)) init 0
  val returned = Reg(UInt(32 bits)) init 0
  io.busy := state =/= State.Idle
  io.protocolFaultSticky := fault
  io.acceptedPhases := accepted
  io.returnedResults := returned

  val legalMode = io.job.mode ===
    BitNetUnifiedMatrixMode.SvContextScaleQ15 ||
    io.job.mode === BitNetUnifiedMatrixMode.SvAxpyQ15
  val legalFormat =
    (io.job.mode === BitNetUnifiedMatrixMode.SvContextScaleQ15 &&
      (io.job.format === BitNetUnifiedOperandFormat.Q15TileAlpha ||
        io.job.format ===
          BitNetUnifiedOperandFormat.Q15FinalReciprocal)) ||
    (io.job.mode === BitNetUnifiedMatrixMode.SvAxpyQ15 &&
      io.job.format === BitNetUnifiedOperandFormat.Q15Probability)
  val legalJob = legalMode && legalFormat && io.job.phaseCount =/= 0

  // An illegal job is deliberately not consumed: accepting a dot-mode job on
  // an SV-only bridge would make the coordinator wait forever for an SV
  // result.  The sticky fault makes the malformed command fail visibly.
  io.job.ready := state === State.Idle && legalJob
  when(state === State.Idle && io.job.valid && !legalJob) {
    fault := True
  }
  when(io.job.fire) {
    activeMode := io.job.mode
    activeFormat := io.job.format
    expectedPhases := io.job.phaseCount
    phaseCount := 0
    resultCount := 0
    sequenceId := sequenceCounter
    sequenceCounter := sequenceCounter + 1
    state := State.IssueSequence
  }

  io.sequence.valid := state === State.IssueSequence
  io.sequence.mode := activeMode
  io.sequence.format := activeFormat
  io.sequence.sequenceId := sequenceId
  io.sequence.expectedBeats := expectedPhases.resized
  io.sequence.activationFactorQ16.foreach(_ := 0)
  io.sequence.weightScaleQ16 := 0
  io.sequence.outputFeatureCount := 0
  io.sequence.routeFinalToDynamicQuant := False
  io.sequence.residualEnable := False
  when(io.sequence.fire) {
    state := State.Run
  }

  io.phase.ready := state === State.Run &&
    phaseCount < expectedPhases && io.beat.ready
  io.beat.valid := state === State.Run &&
    phaseCount < expectedPhases && io.phase.valid
  io.beat.rawWeights.foreach(_ := 0)
  for (lane <- 0 until cfg.tokenLanesPerEngine) {
    io.beat.activations(lane).foreach(_ := 0)
    io.beat.accumulatorTag(lane) := 0
    io.beat.firstK(lane) := False
    io.beat.lastK(lane) := False
    io.beat.taskId(lane) := sequenceId.resized
    io.beat.tokenId(lane) := io.phase.tokenId
    io.beat.outputBase(lane) := 0
  }
  io.beat.laneMask := B(1, cfg.laneMaskWidth bits)
  io.beat.svValue := io.phase.svValue
  io.beat.svFactorRaw := io.phase.svFactorQ15.asBits.resize(
    cfg.factorWidth)
  io.beat.svFactorSigned := False
  io.beat.svFactorFractionBits := U(15, cfg.factorFractionWidth bits)
  io.beat.svLimbShift := io.phase.svLimbShift
  io.beat.contextBank := io.phase.contextBank
  io.beat.contextSlot := io.phase.contextSlot
  io.beat.headLane := io.phase.headLane

  when(io.beat.fire) {
    phaseCount := phaseCount + 1
    accepted := accepted + 1
  }

  val resultMetadataMatches =
    io.continuousResult.mode === activeMode &&
    io.continuousResult.format === activeFormat &&
    io.continuousResult.sequenceId === sequenceId &&
    !io.continuousResult.svFactorSigned &&
    io.continuousResult.svFactorFractionBits === 15

  io.result.valid := state === State.Run && io.continuousResult.valid
  io.result.mode := io.continuousResult.mode
  io.result.format := io.continuousResult.format
  io.result.contextBank := io.continuousResult.contextBank
  io.result.contextSlot := io.continuousResult.contextSlot
  io.result.headLane := io.continuousResult.headLane
  io.result.svLimbShift := io.continuousResult.svLimbShift
  io.result.tokenId := io.continuousResult.tokenId
  io.result.rawP := io.continuousResult.rawP

  // Unexpected results are drained outside Run to avoid wedging the shared
  // array.  During Run, ordinary downstream backpressure is propagated.
  io.continuousResult.ready := Mux(state === State.Run, io.result.ready, True)
  when(io.continuousResult.fire && state =/= State.Run) {
    fault := True
  }
  when(io.result.fire) {
    returned := returned + 1
    when(!resultMetadataMatches || resultCount >= expectedPhases) {
      fault := True
    }
    when(resultCount === expectedPhases - 1) {
      when(phaseCount =/= expectedPhases) {
        fault := True
      }
      resultCount := 0
      state := State.Idle
    } otherwise {
      resultCount := resultCount + 1
    }
  }
}

object GenerateBitNetContinuousSvAdapter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-continuous-sv-adapter")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetContinuousSvAdapter())
}
