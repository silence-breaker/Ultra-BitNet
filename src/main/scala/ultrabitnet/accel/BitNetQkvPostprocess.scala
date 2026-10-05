package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Architectural destination of one QKV projection head. */
object BitNetQkvKind extends SpinalEnum(binarySequential) {
  val Query, Key, Value = newElement()
}

/**
  * Geometry of the typed QKV postprocess boundary.
  *
  * The production continuous matrix cluster returns one token lane and two
  * output rows per engine result.  Prefill assigns engines by token while
  * decode assigns them by row, so that mode-aware assembler belongs beside
  * the cluster scheduler.  This component starts at its single lossless
  * normalization boundary: one output feature across all four token lanes.
  */
case class BitNetQkvPostprocessConfig(
    tokenLanes: Int = BitNetConfig.Production.tokenParallelism,
    queryHeadCount: Int = 20,
    kvHeadCount: Int = 5,
    headSize: Int = 128,
    tokenIdWidth: Int = 16,
    metaFifoDepth: Int = 8,
    externalDynamicQuant: Boolean = false,
    arithmetic: BitNetSharedSpuArithmeticPoolConfig =
      BitNetSharedSpuArithmeticPoolConfig()
) {
  require(tokenLanes == 2 || tokenLanes == 4,
    "the native matrix/postprocess boundary must be TP2 or TP4")
  require(queryHeadCount == kvHeadCount * 4,
    "BitNet uses four query heads for each KV head")
  require(headSize >= 4 && isPow2(headSize) && headSize % 2 == 0)
  require(metaFifoDepth >= 4 && isPow2(metaFifoDepth))

  val headWidth: Int = Math.max(1, log2Up(queryHeadCount))
  val elementWidth: Int = log2Up(headSize)
  val halfSize: Int = headSize / 2
  val halfIndexWidth: Int = log2Up(halfSize)
  val tokenLaneWidth: Int = log2Up(tokenLanes)
}

/** QKV client of the model-wide dynamic-i8 arithmetic core.
  *
  * The adapter keeps only the head-local replay image and scalar lane output
  * contract.  Max/divide/multiply/correction arithmetic lives in the existing
  * shared core.  A command is acquired before the first measure beat, and the
  * retained Vec4 image is replayed only after the authenticated request.
  */
class BitNetQkvSharedQuantClientAdapter(
    headCfg: NativeDynamicI8QuantizerConfig,
    sharedCfg: BitNetSharedDynamicI8QuantizerConfig)
    extends Component {
  require(headCfg.tokenLanes == sharedCfg.tokenLanes)
  require(headCfg.vectorLength <= sharedCfg.maxVectorLength)
  require(headCfg.inputWidth == sharedCfg.inputWidth)

  val io = new Bundle {
    val prepare = slave(Stream(Bits(headCfg.tokenLanes bits)))
    val input = slave(Stream(NativeDynamicQuantBeat(headCfg)))
    val output = master(Stream(BitNetHeadQuantSample(headCfg)))
    val command = master(Stream(BitNetSharedDynamicQuantCommand(sharedCfg)))
    val measure = master(Stream(BitNetSharedDynamicQuantMeasureBeat(sharedCfg)))
    val replayRequest = slave(Stream(
      BitNetSharedDynamicQuantReplayRequest(sharedCfg)))
    val replay = master(Stream(BitNetSharedDynamicQuantReplayBeat(sharedCfg)))
    val result = slave(Stream(BitNetSharedDynamicQuantResultBeat(sharedCfg)))
    val done = slave(Stream(BitNetSharedDynamicQuantDone(sharedCfg)))
    val busy = out Bool()
    val protocolFault = out Bool()
  }

  object State extends SpinalEnum(binarySequential) {
    val AwaitCommand, Measure, WaitReplay, Replay, DrainResult, WaitDone =
      newElement()
  }
  val state = Reg(State()) init State.AwaitCommand
  val storage = Mem(Bits(headCfg.replayEntryWidth bits), headCfg.vectorLength)
  storage.addAttribute("ram_style", "block")
  storage.setName("qkvSharedQuantReplayMemory")
  val tagCounter = Reg(UInt(sharedCfg.tagWidth bits)) init 1
  val epochCounter = Reg(UInt(sharedCfg.epochWidth bits)) init 1
  val activeTag = Reg(UInt(sharedCfg.tagWidth bits)) init 0
  val activeEpoch = Reg(UInt(sharedCfg.epochWidth bits)) init 0
  val activeMask = Reg(Bits(sharedCfg.tokenLanes bits)) init 0
  val measureIndex = Reg(UInt(headCfg.indexWidth bits)) init 0
  val replayReadIndex = Reg(UInt(headCfg.indexWidth bits)) init 0
  val allReplayReadsIssued = RegInit(False)
  val outputLane = Reg(UInt(log2Up(headCfg.tokenLanes) bits)) init 0
  val fault = RegInit(False)

  io.command.valid := state === State.AwaitCommand && io.prepare.valid
  io.command.tag := tagCounter
  io.command.epoch := epochCounter
  io.command.vectorLength := headCfg.vectorLength
  io.command.tokenMask := io.prepare.payload
  io.prepare.ready := state === State.AwaitCommand && io.command.ready
  io.input.ready := False

  when(io.command.fire) {
    activeTag := tagCounter
    activeEpoch := epochCounter
    activeMask := io.prepare.payload
    tagCounter := tagCounter + 1
    epochCounter := epochCounter + 1
    measureIndex := 0
    fault := !io.prepare.payload.orR
    state := State.Measure
  }

  io.measure.valid := state === State.Measure && io.input.valid
  io.measure.value := io.input.value
  io.measure.tag := activeTag
  io.measure.epoch := activeEpoch
  io.measure.featureIndex := measureIndex.resized
  io.measure.tokenMask := io.input.tokenMask
  io.measure.first := measureIndex === 0
  io.measure.last := measureIndex === headCfg.vectorLength - 1
  when(state === State.Measure) { io.input.ready := io.measure.ready }

  val packedInput = Bits(headCfg.replayEntryWidth bits)
  for (lane <- 0 until headCfg.tokenLanes) {
    packedInput((lane + 1) * headCfg.inputWidth - 1 downto
      lane * headCfg.inputWidth) := io.input.value(lane).asBits
  }
  when(io.measure.fire) {
    storage.write(measureIndex, packedInput, enable = io.measure.fire)
    when(io.input.tokenMask =/= activeMask ||
      io.input.first =/= (measureIndex === 0) ||
      io.input.last =/= (measureIndex === headCfg.vectorLength - 1)) {
      fault := True
    }
    when(measureIndex === headCfg.vectorLength - 1) {
      state := State.WaitReplay
    } otherwise {
      measureIndex := measureIndex + 1
    }
  }

  val replayIdentity = io.replayRequest.tag === activeTag &&
    io.replayRequest.epoch === activeEpoch &&
    io.replayRequest.vectorLength === headCfg.vectorLength &&
    io.replayRequest.tokenMask === activeMask
  io.replayRequest.ready := state === State.WaitReplay
  when(io.replayRequest.fire) {
    when(!replayIdentity) { fault := True }
    replayReadIndex := 0
    allReplayReadsIssued := False
    state := State.Replay
  }

  val replayFifo = StreamFifo(
    NativeDynamicQuantReplay(headCfg), headCfg.replayFifoDepth)
  val replayReadIssue = state === State.Replay &&
    !allReplayReadsIssued && replayFifo.io.availability > 1
  val replayReadData = storage.readSync(replayReadIndex, replayReadIssue)
  val replayResponseValid = RegNext(replayReadIssue) init False
  val replayResponseIndex = RegNextWhen(replayReadIndex, replayReadIssue) init 0
  replayFifo.io.push.valid := replayResponseValid
  replayFifo.io.push.data := replayReadData
  replayFifo.io.push.index := replayResponseIndex
  when(replayReadIssue) {
    when(replayReadIndex === headCfg.vectorLength - 1) {
      allReplayReadsIssued := True
    } otherwise {
      replayReadIndex := replayReadIndex + 1
    }
  }

  val replayValues = Vec(SInt(headCfg.inputWidth bits), headCfg.tokenLanes)
  for (lane <- 0 until headCfg.tokenLanes) {
    replayValues(lane) := replayFifo.io.pop.data(
      (lane + 1) * headCfg.inputWidth - 1 downto lane * headCfg.inputWidth).asSInt
  }
  io.replay.valid := state === State.Replay && replayFifo.io.pop.valid
  io.replay.value := replayValues
  io.replay.tag := activeTag
  io.replay.epoch := activeEpoch
  io.replay.featureIndex := replayFifo.io.pop.index.resized
  io.replay.tokenMask := activeMask
  io.replay.first := replayFifo.io.pop.index === 0
  io.replay.last := replayFifo.io.pop.index === headCfg.vectorLength - 1
  replayFifo.io.pop.ready := io.replay.ready && state === State.Replay
  when(io.replay.fire && io.replay.last) {
    state := State.DrainResult
    outputLane := 0
  }

  io.output.valid := (state === State.Replay || state === State.DrainResult) &&
    io.result.valid
  io.output.value := io.result.value(outputLane)
  io.output.dequantFactor := io.result.dequantFactorQ16(outputLane)
  io.output.tokenLane := outputLane
  io.output.element := io.result.featureIndex.resized
  io.output.tokenActive := io.result.tokenMask(outputLane)
  io.output.first := io.result.first && outputLane === 0
  io.output.last := io.result.last &&
    outputLane === headCfg.tokenLanes - 1
  io.result.ready := io.output.ready &&
    outputLane === headCfg.tokenLanes - 1
  when(io.output.fire) {
    when(outputLane === headCfg.tokenLanes - 1) {
      outputLane := 0
      when(io.result.last) { state := State.WaitDone }
    } otherwise {
      outputLane := outputLane + 1
    }
  }

  val doneIdentity = io.done.tag === activeTag &&
    io.done.epoch === activeEpoch &&
    io.done.vectorLength === headCfg.vectorLength &&
    io.done.emittedFeatures === headCfg.vectorLength && io.done.success
  io.done.ready := state === State.WaitDone
  when(io.done.fire) {
    when(!doneIdentity) { fault := True }
    state := State.AwaitCommand
  }

  io.busy := state =/= State.AwaitCommand || io.prepare.valid || io.input.valid
  io.protocolFault := fault
}

/**
  * TP4-assembled form of final (`lastK`) continuous matrix dot results.
  *
  * `rawDot`, activation factor, weight scale, and token ids are copied without
  * arithmetic reinterpretation from [[BitNetContinuousDotResult]].  The
  * cluster-side assembler only changes geometry (two engines/two rows into
  * one feature/four tokens); it must not scale or spill this beat to DDR.
  */
case class BitNetQkvTp4DotBeat(cfg: BitNetQkvPostprocessConfig)
    extends Bundle {
  val rawDot = Vec(SInt(32 bits), cfg.tokenLanes)
  val activationFactorQ16 = Vec(UInt(32 bits), cfg.tokenLanes)
  val weightScaleQ16 = UInt(32 bits)
  val tokenId = Vec(UInt(cfg.tokenIdWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val kind = BitNetQkvKind()
  /** Query uses 0..19; K/V use 0..4. */
  val head = UInt(cfg.headWidth bits)
  /** Natural (non-interleaved) head coordinate, 0..127 in production. */
  val element = UInt(cfg.elementWidth bits)
  /** Used only for Q/K upper-half beats; one coefficient per token position. */
  val cosine = Vec(SInt(16 bits), cfg.tokenLanes)
  val sine = Vec(SInt(16 bits), cfg.tokenLanes)
  val first = Bool()
  val last = Bool()
}

case class BitNetQkvPostprocessMeta(cfg: BitNetQkvPostprocessConfig)
    extends Bundle {
  val tokenId = Vec(UInt(cfg.tokenIdWidth bits), cfg.tokenLanes)
  val tokenMask = Bits(cfg.tokenLanes bits)
  val kind = BitNetQkvKind()
  val head = UInt(cfg.headWidth bits)
  val element = UInt(cfg.elementWidth bits)
  val cosine = Vec(SInt(16 bits), cfg.tokenLanes)
  val sine = Vec(SInt(16 bits), cfg.tokenLanes)
  val first = Bool()
  val last = Bool()
}

/** Scalar replay sample consumed by attention or the bank-local KV writer. */
case class BitNetQkvQuantizedSample(cfg: BitNetQkvPostprocessConfig)
    extends Bundle {
  val value = SInt(8 bits)
  val dequantFactorQ16 = UInt(32 bits)
  val tokenId = UInt(cfg.tokenIdWidth bits)
  val tokenLane = UInt(cfg.tokenLaneWidth bits)
  val tokenActive = Bool()
  val head = UInt(cfg.headWidth bits)
  val element = UInt(cfg.elementWidth bits)
  val first = Bool()
  val last = Bool()
}

/**
  * Resident QKV scale/RoPE/quantise scheduler.
  *
  * One [[BitNetLeasedScaleResidual]] converts raw projection dots to signed
  * Q16.16 using the compute-island-global arithmetic pool.
  * Q and K then share one [[BitNetTimeMuxRope]] and one
  * [[BitNetHeadI8Quantizer]]; V bypasses only RoPE and uses that same
  * quantiser.  The split-half RoPE input requires 64 lower values to be held
  * until their matching upper values arrive.  Rotated upper values use a
  * second local memory while rotated lower values stream into the quantiser;
  * upper values are then replayed in natural element order.  Consequently Q,
  * K and V all leave with the same 0..127 coordinate convention.
  *
  * Frames are intentionally exclusive.  A new head cannot enter until the
  * previous quantised replay has drained, which makes ownership of the two
  * shared nonlinear blocks explicit and prevents Q/K/V mixing under arbitrary
  * backpressure.  No activation, QKV, or score intermediate has a DDR route.
  */
class BitNetQkvPostprocess(
    cfg: BitNetQkvPostprocessConfig = BitNetQkvPostprocessConfig())
    extends Component {
  private val scaleCfg = NativeScaleResidualConfig(lanes = cfg.tokenLanes)
  private val ropeCfg = NativeRopeConfig(
    tokenLanes = cfg.tokenLanes,
    headCount = cfg.queryHeadCount,
    headSize = cfg.headSize)
  private val quantCfg = NativeDynamicI8QuantizerConfig(
    tokenLanes = cfg.tokenLanes,
    vectorLength = cfg.headSize,
    replayFifoDepth = 4)
  private val sharedQuantCfg = BitNetSharedDynamicI8QuantizerConfig()

  val io = new Bundle {
    val input = slave(Stream(BitNetQkvTp4DotBeat(cfg)))
    val query = master(Stream(BitNetQkvQuantizedSample(cfg)))
    val key = master(Stream(BitNetQkvQuantizedSample(cfg)))
    val value = master(Stream(BitNetQkvQuantizedSample(cfg)))

    // The sole projection-scale adapter is also used by O/Down after their
    // final-pair assembler.  The global stage schedule makes the two routes
    // mutually exclusive; this second typed route prevents a duplicate
    // scaler/client from appearing elsewhere in the compute island.
    val linearScaleInput = slave(Stream(NativeScaleResidualBeat(scaleCfg)))
    val linearScaleOutput = master(Stream(
      NativeScaleResidualResult(scaleCfg)))

    val poolAcquire = master(Stream(
      BitNetSpuArithmeticLeaseRequest(cfg.arithmetic)))
    val poolOperand = master(Stream(
      BitNetSpuArithmeticOperandBeat(cfg.arithmetic)))
    val poolResult = slave(Stream(
      BitNetSpuArithmeticResultBeat(cfg.arithmetic)))
    val poolRelease = master(Stream(
      BitNetSpuArithmeticLeaseRelease(cfg.arithmetic)))
    val poolFault = in Bool()

    val quantCommand = master(Stream(
      BitNetSharedDynamicQuantCommand(sharedQuantCfg)))
    val quantMeasure = master(Stream(
      BitNetSharedDynamicQuantMeasureBeat(sharedQuantCfg)))
    val quantReplayRequest = slave(Stream(
      BitNetSharedDynamicQuantReplayRequest(sharedQuantCfg)))
    val quantReplay = master(Stream(
      BitNetSharedDynamicQuantReplayBeat(sharedQuantCfg)))
    val quantResult = slave(Stream(
      BitNetSharedDynamicQuantResultBeat(sharedQuantCfg)))
    val quantDone = slave(Stream(
      BitNetSharedDynamicQuantDone(sharedQuantCfg)))

    val busy = out Bool()
    val protocolFault = out Bool()
    val acceptedFrames = out UInt(32 bits)
    val emittedFrames = out UInt(32 bits)
  }

  val scaler = new BitNetLeasedScaleResidual(scaleCfg, cfg.arithmetic)
  val rope = new BitNetTimeMuxRope(ropeCfg)
  val internalQuantizer = if (!cfg.externalDynamicQuant)
    new BitNetHeadI8Quantizer(quantCfg) else null
  val sharedQuantAdapter = if (cfg.externalDynamicQuant)
    new BitNetQkvSharedQuantClientAdapter(quantCfg, sharedQuantCfg) else null
  val quantInput = Stream(NativeDynamicQuantBeat(quantCfg))
  val quantOutput = Stream(BitNetHeadQuantSample(quantCfg))
  val quantPrepare = Stream(Bits(cfg.tokenLanes bits))
  val quantBusy = Bool()
  val quantFault = Bool()
  if (cfg.externalDynamicQuant) {
    sharedQuantAdapter.io.prepare << quantPrepare
    sharedQuantAdapter.io.input << quantInput
    quantOutput << sharedQuantAdapter.io.output
    io.quantCommand << sharedQuantAdapter.io.command
    io.quantMeasure << sharedQuantAdapter.io.measure
    sharedQuantAdapter.io.replayRequest << io.quantReplayRequest
    io.quantReplay << sharedQuantAdapter.io.replay
    sharedQuantAdapter.io.result << io.quantResult
    sharedQuantAdapter.io.done << io.quantDone
    quantBusy := sharedQuantAdapter.io.busy
    quantFault := sharedQuantAdapter.io.protocolFault
  } else {
    quantPrepare.ready := True
    internalQuantizer.io.input << quantInput
    quantOutput << internalQuantizer.io.output
    io.quantCommand.valid := False
    io.quantCommand.payload := BitNetSharedDynamicQuantCommand(
      sharedQuantCfg).getZero
    io.quantMeasure.valid := False
    io.quantMeasure.payload := BitNetSharedDynamicQuantMeasureBeat(
      sharedQuantCfg).getZero
    io.quantReplayRequest.ready := True
    io.quantReplay.valid := False
    io.quantReplay.payload := BitNetSharedDynamicQuantReplayBeat(
      sharedQuantCfg).getZero
    io.quantResult.ready := True
    io.quantDone.ready := True
    quantBusy := internalQuantizer.io.busy
    quantFault := internalQuantizer.io.protocolFault
  }
  val metaFifo = StreamFifo(
    BitNetQkvPostprocessMeta(cfg), cfg.metaFifoDepth)

  io.poolAcquire << scaler.io.poolAcquire
  io.poolOperand << scaler.io.poolOperand
  scaler.io.poolResult << io.poolResult
  io.poolRelease << scaler.io.poolRelease

  val qkvScaleInput = Stream(NativeScaleResidualBeat(scaleCfg))
  val qkvScaleOutput = Stream(NativeScaleResidualResult(scaleCfg))
  val scaleOwnerLinear = RegInit(False)

  // QKV has deterministic priority if a malformed outer schedule presents
  // both routes together.  Since the scaler accepts only one in-flight TP4
  // vector, a single owner bit is sufficient and cannot be overwritten.
  scaler.io.input.valid := qkvScaleInput.valid || io.linearScaleInput.valid
  scaler.io.input.payload := io.linearScaleInput.payload
  when(qkvScaleInput.valid) {
    scaler.io.input.payload := qkvScaleInput.payload
  }
  qkvScaleInput.ready := scaler.io.input.ready
  io.linearScaleInput.ready := scaler.io.input.ready && !qkvScaleInput.valid
  when(scaler.io.input.fire) {
    scaleOwnerLinear := !qkvScaleInput.valid
  }

  qkvScaleOutput.valid := scaler.io.output.valid && !scaleOwnerLinear
  qkvScaleOutput.payload := scaler.io.output.payload
  io.linearScaleOutput.valid := scaler.io.output.valid && scaleOwnerLinear
  io.linearScaleOutput.payload := scaler.io.output.payload
  scaler.io.output.ready := Mux(
    scaleOwnerLinear, io.linearScaleOutput.ready, qkvScaleOutput.ready)

  val ingressOpen = RegInit(True)
  val ingressFrame = RegInit(False)
  val expectedElement = Reg(UInt(cfg.elementWidth bits)) init 0
  val activeKind = Reg(BitNetQkvKind()) init BitNetQkvKind.Query
  val activeHead = Reg(UInt(cfg.headWidth bits)) init 0
  val activeTokenId = Vec(
    Reg(UInt(cfg.tokenIdWidth bits)) init 0, cfg.tokenLanes)
  val activeTokenMask = Reg(Bits(cfg.tokenLanes bits)) init 0
  val fault = RegInit(False)
  val acceptedFrameCount = Reg(UInt(32 bits)) init 0
  val emittedFrameCount = Reg(UInt(32 bits)) init 0

  val needsQuantPrepare = !ingressFrame
  val ingressBarrierReady = qkvScaleInput.ready && metaFifo.io.push.ready &&
    (!needsQuantPrepare || quantPrepare.ready)
  io.input.ready := ingressOpen && ingressBarrierReady
  qkvScaleInput.valid := ingressOpen && io.input.valid &&
    metaFifo.io.push.ready && (!needsQuantPrepare || quantPrepare.ready)
  metaFifo.io.push.valid := ingressOpen && io.input.valid &&
    qkvScaleInput.ready && (!needsQuantPrepare || quantPrepare.ready)
  quantPrepare.valid := ingressOpen && !ingressFrame && io.input.valid &&
    qkvScaleInput.ready && metaFifo.io.push.ready
  quantPrepare.payload := io.input.tokenMask

  qkvScaleInput.residual.foreach(_ := 0)
  qkvScaleInput.accumulator := io.input.rawDot
  qkvScaleInput.activationFactor := io.input.activationFactorQ16
  qkvScaleInput.weightScale := io.input.weightScaleQ16
  qkvScaleInput.laneMask := io.input.tokenMask
  qkvScaleInput.residualEnable := False
  qkvScaleInput.first := io.input.first
  qkvScaleInput.last := io.input.last

  metaFifo.io.push.tokenId := io.input.tokenId
  metaFifo.io.push.tokenMask := io.input.tokenMask
  metaFifo.io.push.kind := io.input.kind
  metaFifo.io.push.head := io.input.head
  metaFifo.io.push.element := io.input.element
  metaFifo.io.push.cosine := io.input.cosine
  metaFifo.io.push.sine := io.input.sine
  metaFifo.io.push.first := io.input.first
  metaFifo.io.push.last := io.input.last

  when(io.input.fire) {
    when(!ingressFrame) {
      when(!io.input.first || io.input.element =/= 0) { fault := True }
      ingressFrame := True
      expectedElement := 1
      activeKind := io.input.kind
      activeHead := io.input.head
      activeTokenId := io.input.tokenId
      activeTokenMask := io.input.tokenMask
      when(io.input.kind =/= BitNetQkvKind.Query &&
          io.input.head >= cfg.kvHeadCount) {
        fault := True
      }
      when(io.input.kind === BitNetQkvKind.Query &&
          io.input.head >= cfg.queryHeadCount) {
        fault := True
      }
    } otherwise {
      when(io.input.first || io.input.element =/= expectedElement ||
          io.input.kind =/= activeKind || io.input.head =/= activeHead ||
          io.input.tokenMask =/= activeTokenMask) {
        fault := True
      }
      for (lane <- 0 until cfg.tokenLanes) {
        when(io.input.tokenId(lane) =/= activeTokenId(lane)) {
          fault := True
        }
      }
      expectedElement := expectedElement + 1
    }

    when(io.input.last) {
      when(io.input.element =/= cfg.headSize - 1) { fault := True }
      ingressFrame := False
      ingressOpen := False
      acceptedFrameCount := acceptedFrameCount + 1
    } elsewhen (io.input.element === cfg.headSize - 1) {
      fault := True
    }
  }

  // Atomic join: scale output and its typed metadata can move only together.
  val scaledValid = qkvScaleOutput.valid && metaFifo.io.pop.valid
  val scaledReady = Bool()
  scaledReady := False
  qkvScaleOutput.ready := scaledReady && metaFifo.io.pop.valid
  metaFifo.io.pop.ready := scaledReady && qkvScaleOutput.valid
  val scaledFire = scaledValid && scaledReady

  val lowerMemory = Mem(Bits((cfg.tokenLanes * 32) bits), cfg.halfSize)
  val rotatedUpperMemory = Mem(
    Bits((cfg.tokenLanes * 32) bits), cfg.halfSize)

  object State extends SpinalEnum(binarySequential) {
    val CollectScaled, ReplayRotatedUpper, DrainQuantizer = newElement()
  }
  val state = Reg(State()) init State.CollectScaled
  val replayIndex = Reg(UInt(cfg.halfIndexWidth bits)) init 0

  // Defaults for the two shared nonlinear blocks.
  rope.io.input.valid := False
  rope.io.input.lower.foreach(_ := 0)
  rope.io.input.upper.foreach(_ := 0)
  rope.io.input.cosine.foreach(_ := 0)
  rope.io.input.sine.foreach(_ := 0)
  rope.io.input.tokenMask := activeTokenMask
  rope.io.input.head := activeHead.resized
  rope.io.input.pair := 0
  rope.io.input.first := False
  rope.io.input.last := False
  rope.io.output.ready := False

  quantInput.valid := False
  quantInput.value.foreach(_ := 0)
  quantInput.tokenMask := activeTokenMask
  quantInput.first := False
  quantInput.last := False

  val currentMeta = metaFifo.io.pop.payload
  val isQk = activeKind === BitNetQkvKind.Query ||
    activeKind === BitNetQkvKind.Key
  val inLowerHalf = currentMeta.element < cfg.halfSize
  val pairIndex = (currentMeta.element - cfg.halfSize)
    .resize(cfg.halfIndexWidth)
  val lowerReadBits = lowerMemory.readAsync(pairIndex)
  val lowerRead = Vec(SInt(32 bits), cfg.tokenLanes)
  lowerRead.assignFromBits(lowerReadBits)

  val rotatedUpperReadBits = rotatedUpperMemory.readAsync(replayIndex)
  val rotatedUpperRead = Vec(SInt(32 bits), cfg.tokenLanes)
  rotatedUpperRead.assignFromBits(rotatedUpperReadBits)

  when(state === State.CollectScaled) {
    when(isQk) {
      // Lower half is local state only.  Each upper beat then joins its lower
      // partner and enters the single shared RoPE instance.
      when(inLowerHalf) {
        scaledReady := True
        when(scaledFire) {
          lowerMemory.write(
            currentMeta.element.resize(cfg.halfIndexWidth),
            qkvScaleOutput.value.asBits)
        }
      } otherwise {
        rope.io.input.valid := scaledValid
        scaledReady := rope.io.input.ready
        rope.io.input.lower := lowerRead
        rope.io.input.upper := qkvScaleOutput.value
        rope.io.input.cosine := currentMeta.cosine
        rope.io.input.sine := currentMeta.sine
        rope.io.input.tokenMask := currentMeta.tokenMask
        rope.io.input.head := currentMeta.head.resized
        rope.io.input.pair := pairIndex
        rope.io.input.first := pairIndex === 0
        rope.io.input.last := pairIndex === cfg.halfSize - 1
      }

      // Quantise the rotated lower half now and retain upper values for a
      // natural-order second half replay.
      quantInput.valid := rope.io.output.valid
      rope.io.output.ready := quantInput.ready
      quantInput.value := rope.io.output.lower
      quantInput.tokenMask := rope.io.output.tokenMask
      quantInput.first := rope.io.output.first
      quantInput.last := False
      when(rope.io.output.fire) {
        rotatedUpperMemory.write(
          rope.io.output.pair.resize(cfg.halfIndexWidth),
          rope.io.output.upper.asBits)
        when(rope.io.output.last) {
          state := State.ReplayRotatedUpper
          replayIndex := 0
        }
      }
    } otherwise {
      // V has no positional rotation, but shares the same exact head-local
      // dynamic quantiser and therefore the same Q16.16/i8 contract.
      quantInput.valid := scaledValid
      scaledReady := quantInput.ready
      quantInput.value := qkvScaleOutput.value
      quantInput.tokenMask := currentMeta.tokenMask
      quantInput.first := currentMeta.first
      quantInput.last := currentMeta.last
      when(scaledFire && currentMeta.last) {
        state := State.DrainQuantizer
      }
    }
  }

  when(state === State.ReplayRotatedUpper) {
    quantInput.valid := True
    quantInput.value := rotatedUpperRead
    quantInput.tokenMask := activeTokenMask
    quantInput.first := False
    quantInput.last := replayIndex === cfg.halfSize - 1
    when(quantInput.fire) {
      when(replayIndex === cfg.halfSize - 1) {
        state := State.DrainQuantizer
      } otherwise {
        replayIndex := replayIndex + 1
      }
    }
  }

  val routedSample = BitNetQkvQuantizedSample(cfg)
  routedSample.value := quantOutput.value
  routedSample.dequantFactorQ16 := quantOutput.dequantFactor
  routedSample.tokenLane := quantOutput.tokenLane
  routedSample.tokenId := activeTokenId(quantOutput.tokenLane)
  routedSample.tokenActive := quantOutput.tokenActive
  routedSample.head := activeHead
  routedSample.element := quantOutput.element.resized
  routedSample.first := quantOutput.first
  routedSample.last := quantOutput.last

  io.query.valid := quantOutput.valid &&
    activeKind === BitNetQkvKind.Query
  io.key.valid := quantOutput.valid &&
    activeKind === BitNetQkvKind.Key
  io.value.valid := quantOutput.valid &&
    activeKind === BitNetQkvKind.Value
  io.query.payload := routedSample
  io.key.payload := routedSample
  io.value.payload := routedSample

  switch(activeKind) {
    is(BitNetQkvKind.Query) { quantOutput.ready := io.query.ready }
    is(BitNetQkvKind.Key) { quantOutput.ready := io.key.ready }
    is(BitNetQkvKind.Value) { quantOutput.ready := io.value.ready }
  }

  when(quantOutput.fire && quantOutput.last) {
    when(state =/= State.DrainQuantizer) { fault := True }
    state := State.CollectScaled
    ingressOpen := True
    emittedFrameCount := emittedFrameCount + 1
  }

  io.protocolFault := fault || scaler.io.protocolFaultSticky ||
    quantFault || io.poolFault
  io.acceptedFrames := acceptedFrameCount
  io.emittedFrames := emittedFrameCount
  io.busy := !ingressOpen || ingressFrame ||
    state =/= State.CollectScaled || scaler.io.busy ||
    metaFifo.io.pop.valid || rope.io.busy || quantBusy
}

object GenerateBitNetQkvPostprocess extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-qkv-postprocess")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetQkvPostprocess())
}
