package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Fixed production contract between four physical attention/SPU lanes and
  * the two TP2 resident-SV matrix engines.
  *
  * Lane ownership is mode dependent, exactly as in BitNet:
  *
  *   - prefill: four lanes are four tokens and a pass visits one Q head;
  *   - decode: four lanes are the four Q heads of one GQA group/token.
  *
  * In either mode a raw V row is fetched once.  Four final probabilities are
  * multiplied by its Q16.16 scale at one key/cycle, then the common V row and
  * two factors are sent to each resident matrix engine.
  */
case class BitNetResidentAttentionConfig(
    model: BitNetConfig = BitNetConfig.Production,
    continuous: BitNetContinuousMatrixConfig =
      BitNetContinuousMatrixConfig(),
    maxSeq: Int = 4096,
    probabilityWidth: Int = 16,
    valueWidth: Int = 8,
    valueScaleWidth: Int = 32,
    factorWidth: Int = 18,
    factorFractionBits: Int = 15,
    preparedFifoDepth: Int = 16
) {
  require(maxSeq >= 64 && maxSeq <= model.maxSequenceLength && isPow2(maxSeq))
  require(probabilityWidth == 16,
    "the buffered SPU emits final unsigned Q1.15 probabilities")
  require(model.gqaGroupSize == 4)
  require(model.engineCount >= 1 && model.engineCount <= 2)
  require(continuous.tokenLanesPerEngine == 2)
  require(model.tokenParallelism ==
    model.engineCount * continuous.tokenLanesPerEngine)
  require(model.headSize % continuous.base.siteCount == 0)
  require(continuous.base.contextBanks >= model.kvHeadCount)
  require(continuous.base.tokenIdWidth >= model.sequenceWidth)
  require(valueWidth == 8 && valueScaleWidth == 32)
  require(factorWidth == 18 && factorFractionBits == 15)
  require(continuous.factorWidth == factorWidth)
  require(preparedFifoDepth >= 4 && isPow2(preparedFifoDepth))

  val physicalLanes: Int = model.tokenParallelism
  val engineCount: Int = model.engineCount
  val lanesPerEngine: Int = continuous.tokenLanesPerEngine
  val headDim: Int = model.headSize
  val queryHeadWidth: Int = log2Up(model.queryHeadCount)
  val kvHeadWidth: Int = log2Up(model.kvHeadCount)
  val passIndexWidth: Int = log2Up(model.queryHeadCount)
}

/**
  * Four final, fully normalised probabilities for one physical key row.
  * The repeated pass/lane metadata makes accidental cross-pass joins fail
  * closed before either resident DSP context can be modified.
  */
case class BitNetResidentLaneProbabilityBeat(
    cfg: BitNetResidentAttentionConfig
) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val laneMask = Bits(cfg.physicalLanes bits)
  val tokenId = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val queryHeadId = Vec(
    UInt(cfg.queryHeadWidth bits), cfg.physicalLanes)
  val queryPosition = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val keyCount = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val weight = Vec(UInt(cfg.probabilityWidth bits), cfg.physicalLanes)
  val allMasked = Bits(cfg.physicalLanes bits)
  val keyPosition = UInt(cfg.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
}

/**
  * One ordered raw-V row.  DDR request tagging/reordering is upstream; this
  * boundary deliberately contains neither a V tile RAM nor context BRAM.
  */
case class BitNetResidentValueBeat(cfg: BitNetResidentAttentionConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val value = Vec(SInt(cfg.valueWidth bits), cfg.headDim)
  val valueScaleQ16 = UInt(cfg.valueScaleWidth bits)
  val keyPosition = UInt(cfg.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
  val success = Bool()
}

/** One-copy representation before the two-engine synchronous broadcast. */
case class BitNetResidentPreparedBeat(
    cfg: BitNetResidentAttentionConfig
) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val laneMask = Bits(cfg.physicalLanes bits)
  val tokenId = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val queryHeadId = Vec(
    UInt(cfg.queryHeadWidth bits), cfg.physicalLanes)
  val queryPosition = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val keyCount = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val value = Vec(SInt(cfg.valueWidth bits), cfg.headDim)
  val factorRaw = Vec(UInt(cfg.factorWidth bits), cfg.physicalLanes)
  val factorSaturated = Bits(cfg.physicalLanes bits)
  val allMasked = Bits(cfg.physicalLanes bits)
  val keyPosition = UInt(cfg.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
}

private[accel] case class BitNetResidentFactorProduct(
    cfg: BitNetResidentAttentionConfig
) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val laneRole = BitNetAttentionLaneRole()
  val passIndex = UInt(cfg.passIndexWidth bits)
  val kvHead = UInt(cfg.kvHeadWidth bits)
  val laneMask = Bits(cfg.physicalLanes bits)
  val tokenId = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val queryHeadId = Vec(
    UInt(cfg.queryHeadWidth bits), cfg.physicalLanes)
  val queryPosition = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val keyCount = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.physicalLanes)
  val value = Vec(SInt(cfg.valueWidth bits), cfg.headDim)
  val product = Vec(UInt((cfg.probabilityWidth +
    cfg.valueScaleWidth) bits), cfg.physicalLanes)
  val enabled = Bits(cfg.physicalLanes bits)
  val allMasked = Bits(cfg.physicalLanes bits)
  val keyPosition = UInt(cfg.model.sequenceWidth bits)
  val keyIndex = UInt(cfg.model.sequenceWidth bits)
  val firstInSequence = Bool()
  val lastInSequence = Bool()
}

object BitNetResidentAttentionMath {
  val FactorFractionBits: Int = 15
  val FactorMaximum: BigInt = (BigInt(1) << 18) - 1

  case class EffectiveFactor(rawUq3_15: BigInt, saturated: Boolean)

  /** Unsigned Q1.15 x unsigned Q16.16 -> rounded/saturated UQ3.15. */
  def effectiveFactor(probQ15: BigInt,
                      valueScaleQ16: BigInt): EffectiveFactor = {
    require(probQ15 >= 0 && probQ15 < (BigInt(1) << 16))
    require(valueScaleQ16 >= 0 && valueScaleQ16 < (BigInt(1) << 32))
    val rounded = (probQ15 * valueScaleQ16 + (BigInt(1) << 15)) >> 16
    EffectiveFactor(rounded.min(FactorMaximum), rounded > FactorMaximum)
  }

  /** rawV*UQ3.15 becomes resident Q16.16 by one exact left shift. */
  def contextIncrementQ16(rawValue: Int, factorRaw: BigInt): BigInt = {
    require(rawValue >= -128 && rawValue <= 127)
    require(factorRaw >= 0 && factorRaw <= FactorMaximum)
    BigInt(rawValue) * factorRaw * 2
  }
}

/** One-beat-per-cycle synchronous-read queue for the wide prepared SV beat.
  *
  * The synchronous BRAM output is itself the elastic head slot.  A stalled
  * head suppresses further reads; a pop may issue the next address on the
  * same edge, so a continuously-ready consumer still receives one beat per
  * cycle.  Public occupancy includes memory plus that head and is therefore
  * exactly `preparedFifoDepth`, matching the former distributed StreamFifo.
  */
private[accel] class BitNetResidentPreparedBramFifo(
    cfg: BitNetResidentAttentionConfig) extends Component {
  private val depth = cfg.preparedFifoDepth
  private val pointerWidth = log2Up(depth)
  private val occupancyWidth = log2Up(depth + 1)

  val io = new Bundle {
    val push = slave(Stream(BitNetResidentPreparedBeat(cfg)))
    val pop = master(Stream(BitNetResidentPreparedBeat(cfg)))
    val occupancy = out UInt(occupancyWidth bits)
  }

  val memory = Mem(BitNetResidentPreparedBeat(cfg), depth)
  memory.addAttribute("ram_style", "block")
  memory.setName("preparedMemory")
  val writePointer = Reg(UInt(pointerWidth bits)) init 0
  val readPointer = Reg(UInt(pointerWidth bits)) init 0
  val memoryCount = Reg(UInt(occupancyWidth bits)) init 0
  val readValid = RegInit(False)

  val totalOccupancy = UInt(occupancyWidth bits)
  totalOccupancy := (memoryCount +
    readValid.asUInt.resize(occupancyWidth)).resized
  io.occupancy := totalOccupancy
  io.push.ready := totalOccupancy < depth || io.pop.fire

  // The BRAM output may be replaced exactly when its current beat is being
  // consumed.  This is the one-address-per-cycle steady-state path.
  val issueRead = memoryCount =/= 0 && (!readValid || io.pop.ready)
  val readData = memory.readSync(readPointer, issueRead)
  io.pop.valid := readValid
  io.pop.payload := readData

  when(io.push.fire) {
    memory.write(writePointer, io.push.payload)
    writePointer := writePointer + 1
  }
  when(issueRead) {
    readPointer := readPointer + 1
  }
  when(io.push.fire =/= issueRead) {
    when(io.push.fire) {
      memoryCount := memoryCount + 1
    } otherwise {
      memoryCount := memoryCount - 1
    }
  }

  when(issueRead) {
    readValid := True
  } elsewhen(io.pop.fire) {
    readValid := False
  }
}

/**
  * Fully elastic one-key/cycle probability/value join and factor pipeline.
  * Four parallel 16x32 products are the speed-first implementation: sharing
  * one multiplier would take four slow clocks/key and starve both DSP arrays.
  */
class BitNetResidentProbabilityValueJoin(
    cfg: BitNetResidentAttentionConfig = BitNetResidentAttentionConfig()
) extends Component {
  private val lanes = cfg.physicalLanes
  private val productWidth = cfg.probabilityWidth + cfg.valueScaleWidth

  val io = new Bundle {
    val clearFault = in Bool()
    val probability = slave(Stream(
      BitNetResidentLaneProbabilityBeat(cfg)))
    val value = slave(Stream(BitNetResidentValueBeat(cfg)))
    val prepared = master(Stream(BitNetResidentPreparedBeat(cfg)))
    val protocolFault = out Bool()
    val factorSaturationFault = out Bool()
    val acceptedKeys = out UInt(32 bits)
    val emittedKeys = out UInt(32 bits)
  }

  val protocolFault = RegInit(False)
  val saturationFault = RegInit(False)
  val acceptedKeys = Reg(UInt(32 bits)) init 0
  val emittedKeys = Reg(UInt(32 bits)) init 0
  io.protocolFault := protocolFault
  io.factorSaturationFault := saturationFault
  io.acceptedKeys := acceptedKeys
  io.emittedKeys := emittedKeys

  when(io.clearFault) {
    protocolFault := False
    saturationFault := False
    acceptedKeys := 0
    emittedKeys := 0
  }

  val productStage = Stream(BitNetResidentFactorProduct(cfg))
  productStage.valid := io.probability.valid && io.value.valid
  io.probability.ready := productStage.ready && io.value.valid
  io.value.ready := productStage.ready && io.probability.valid

  val metadataMatches = io.probability.requestId === io.value.requestId &&
    io.probability.layer === io.value.layer &&
    io.probability.passIndex === io.value.passIndex &&
    io.probability.kvHead === io.value.kvHead &&
    io.probability.keyPosition === io.value.keyPosition &&
    io.probability.keyIndex === io.value.keyIndex &&
    io.probability.firstInSequence === io.value.firstInSequence &&
    io.probability.lastInSequence === io.value.lastInSequence &&
    io.value.success

  productStage.payload.requestId := io.probability.requestId
  productStage.payload.layer := io.probability.layer
  productStage.payload.laneRole := io.probability.laneRole
  productStage.payload.passIndex := io.probability.passIndex
  productStage.payload.kvHead := io.probability.kvHead
  productStage.payload.laneMask := io.probability.laneMask
  productStage.payload.tokenId := io.probability.tokenId
  productStage.payload.queryHeadId := io.probability.queryHeadId
  productStage.payload.queryPosition := io.probability.queryPosition
  productStage.payload.keyCount := io.probability.keyCount
  productStage.payload.value := io.value.value
  productStage.payload.allMasked := io.probability.allMasked
  productStage.payload.keyPosition := io.probability.keyPosition
  productStage.payload.keyIndex := io.probability.keyIndex
  productStage.payload.firstInSequence := io.probability.firstInSequence
  productStage.payload.lastInSequence := io.probability.lastInSequence
  for (lane <- 0 until lanes) {
    productStage.payload.product(lane) :=
      io.probability.weight(lane) * io.value.valueScaleQ16
    val causalLane = io.probability.keyIndex <
      io.probability.keyCount(lane) &&
      io.probability.keyPosition <= io.probability.queryPosition(lane)
    productStage.payload.enabled(lane) := metadataMatches &&
      io.probability.laneMask(lane) && !io.probability.allMasked(lane) &&
      causalLane
  }

  when(productStage.fire) {
    acceptedKeys := acceptedKeys + 1
    when(!metadataMatches) {
      protocolFault := True
    }
  }

  val registeredProduct = productStage.m2sPipe()
  val preparedInput = Stream(BitNetResidentPreparedBeat(cfg))
  preparedInput.valid := registeredProduct.valid
  registeredProduct.ready := preparedInput.ready
  preparedInput.payload.requestId := registeredProduct.requestId
  preparedInput.payload.layer := registeredProduct.layer
  preparedInput.payload.laneRole := registeredProduct.laneRole
  preparedInput.payload.passIndex := registeredProduct.passIndex
  preparedInput.payload.kvHead := registeredProduct.kvHead
  preparedInput.payload.laneMask := registeredProduct.laneMask
  preparedInput.payload.tokenId := registeredProduct.tokenId
  preparedInput.payload.queryHeadId := registeredProduct.queryHeadId
  preparedInput.payload.queryPosition := registeredProduct.queryPosition
  preparedInput.payload.keyCount := registeredProduct.keyCount
  preparedInput.payload.value := registeredProduct.value
  preparedInput.payload.allMasked := registeredProduct.allMasked
  preparedInput.payload.keyPosition := registeredProduct.keyPosition
  preparedInput.payload.keyIndex := registeredProduct.keyIndex
  preparedInput.payload.firstInSequence :=
    registeredProduct.firstInSequence
  preparedInput.payload.lastInSequence := registeredProduct.lastInSequence

  val maximum = U((BigInt(1) << cfg.factorWidth) - 1,
    cfg.factorWidth bits)
  val saturation = Bits(lanes bits)
  for (lane <- 0 until lanes) {
    val rounded = (registeredProduct.product(lane).resize(
      productWidth + 1) + U(BigInt(1) << 15,
      (productWidth + 1) bits)) >> 16
    val overflow = rounded > maximum.resize(rounded.getWidth)
    saturation(lane) := registeredProduct.enabled(lane) && overflow
    preparedInput.payload.factorRaw(lane) := 0
    when(registeredProduct.enabled(lane)) {
      when(overflow) {
        preparedInput.payload.factorRaw(lane) := maximum
      } otherwise {
        preparedInput.payload.factorRaw(lane) :=
          rounded.resize(cfg.factorWidth)
      }
    }
  }
  preparedInput.payload.factorSaturated := saturation
  when(preparedInput.fire && saturation.orR) {
    saturationFault := True
  }

  val fifo = new BitNetResidentPreparedBramFifo(cfg)
  fifo.io.push << preparedInput
  io.prepared << fifo.io.pop
  when(io.prepared.fire) {
    emittedKeys := emittedKeys + 1
  }
}

/**
  * Pass-driven production scheduler.  Its upstream command is exactly
  * [[BitNetAttentionPass]], so the 20-pass prefill and five-pass decode
  * schedules cannot silently reinterpret the four physical lanes.
  */
class BitNetResidentAttentionScheduler(
    cfg: BitNetResidentAttentionConfig = BitNetResidentAttentionConfig()
) extends Component {
  private val lanes = cfg.physicalLanes
  private val engines = cfg.engineCount
  private val c = cfg.continuous
  private val b = c.base

  val io = new Bundle {
    val pass = slave(Stream(BitNetAttentionPass(cfg.model)))
    val featureHalf = in Bool()
    val probability = slave(Stream(
      BitNetResidentLaneProbabilityBeat(cfg)))
    val value = slave(Stream(BitNetResidentValueBeat(cfg)))
    val residentSequence = Vec(
      master(Stream(BitNetResidentSvSequence(c))), engines)
    val residentBeat = Vec(
      master(Stream(BitNetResidentSvBeat(c))), engines)
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val factorSaturationFault = out Bool()
    val acceptedKeys = out UInt(32 bits)
    val submittedKeys = out UInt(32 bits)
    val completedPasses = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Idle, IssueSequences, StreamKeys = newElement()
  }
  val state = Reg(State()) init State.Idle
  val activePass = Reg(BitNetAttentionPass(cfg.model)) init (
    BitNetAttentionPass(cfg.model).getZero)
  val activeEngineMask = Reg(Bits(engines bits)) init 0
  val sequencePending = Reg(Bits(engines bits)) init 0
  val ingressIndex = Reg(UInt(cfg.model.sequenceWidth bits)) init 0
  val ingressComplete = RegInit(False)
  val egressIndex = Reg(UInt(cfg.model.sequenceWidth bits)) init 0
  val fault = RegInit(False)
  val submittedKeys = Reg(UInt(32 bits)) init 0
  val completedPasses = Reg(UInt(32 bits)) init 0
  val activeFeatureHalf = RegInit(False)

  io.busy := state =/= State.Idle
  io.submittedKeys := submittedKeys
  io.completedPasses := completedPasses

  val activeKeyCount = Vec(UInt(cfg.model.sequenceWidth bits), lanes)
  for (lane <- 0 until lanes) {
    activeKeyCount(lane) := Mux(io.pass.laneMask(lane),
      io.pass.keyCount(lane), U(0, cfg.model.sequenceWidth bits))
  }
  val derivedMaxKeyCount = activeKeyCount.drop(1).foldLeft(
    activeKeyCount(0)) { (maximum, candidate) =>
      Mux(maximum > candidate, maximum, candidate)
    }

  val prefillRole = io.pass.mode === BitNetMode.Prefill &&
    io.pass.laneRole === BitNetAttentionLaneRole.TokenParallel
  val decodeRole = io.pass.mode === BitNetMode.Decode &&
    io.pass.laneRole === BitNetAttentionLaneRole.HeadParallel
  val laneMetadataLegal = Vec(Bool(), lanes)
  val decodePassesPerKv = cfg.model.gqaGroupSize / lanes
  for (lane <- 0 until lanes) {
    val decodeSubpass = io.pass.passIndex % decodePassesPerKv
    val expectedDecodeHead =
      ((io.pass.kvHead.resize(cfg.queryHeadWidth) << 2) +
        decodeSubpass * lanes + lane)
        .resize(cfg.queryHeadWidth)
    val expectedPrefillHead = io.pass.passIndex.resize(cfg.queryHeadWidth)
    laneMetadataLegal(lane) := !io.pass.laneMask(lane) ||
      (io.pass.keyCount(lane) =/= 0 &&
        io.pass.keyCount(lane) <= U(cfg.maxSeq,
          cfg.model.sequenceWidth bits) &&
        ((prefillRole &&
          io.pass.queryHeadId(lane) === expectedPrefillHead) ||
         (decodeRole &&
          io.pass.queryHeadId(lane) === expectedDecodeHead &&
          io.pass.tokenId(lane) === io.pass.tokenId(0))))
  }
  val roleShapeLegal = Bool()
  roleShapeLegal := False
  when(prefillRole) {
    roleShapeLegal := io.pass.kvHead ===
      (io.pass.passIndex >> log2Up(cfg.model.gqaGroupSize)).resized
  }
  when(decodeRole) {
    roleShapeLegal := io.pass.laneMask.andR &&
      (io.pass.passIndex / decodePassesPerKv).resize(cfg.kvHeadWidth) ===
        io.pass.kvHead
  }
  val passLegal = io.pass.kvHead < cfg.model.kvHeadCount &&
    io.pass.layer < cfg.model.layerCount &&
    io.pass.laneMask.orR && laneMetadataLegal.asBits.andR &&
    roleShapeLegal && io.pass.maxKeyCount === derivedMaxKeyCount &&
    io.pass.maxKeyCount =/= 0 &&
    io.pass.maxKeyCount <= U(cfg.maxSeq,
      cfg.model.sequenceWidth bits)

  val incomingEngineMask = Bits(engines bits)
  for (engine <- 0 until engines) {
    val firstLane = engine * cfg.lanesPerEngine
    incomingEngineMask(engine) := io.pass.laneMask(
      firstLane + cfg.lanesPerEngine - 1 downto firstLane).orR
  }

  io.pass.ready := state === State.Idle
  when(io.pass.fire) {
    when(passLegal) {
      activePass := io.pass.payload
      activeEngineMask := incomingEngineMask
      sequencePending := incomingEngineMask
      ingressIndex := 0
      ingressComplete := False
      egressIndex := 0
      submittedKeys := 0
      activeFeatureHalf := io.featureHalf
      state := State.IssueSequences
    } otherwise {
      fault := True
    }
  }

  val activeSequenceId = (activePass.requestId.resize(
    c.sequenceIdWidth) + activePass.passIndex.resize(
    c.sequenceIdWidth)).resize(c.sequenceIdWidth)
  val activeSchedule = BitNetResidentSvSchedule()
  activeSchedule := BitNetResidentSvSchedule.PrefillTokenParallel
  when(activePass.laneRole === BitNetAttentionLaneRole.HeadParallel) {
    activeSchedule := BitNetResidentSvSchedule.DecodeHeadParallel
  }
  val activeLaneTokenTag = Vec(UInt(c.taskIdWidth bits), lanes)
  for (lane <- 0 until lanes) {
    activeLaneTokenTag(lane) := activePass.requestId.resize(c.taskIdWidth)
    when(activePass.laneRole === BitNetAttentionLaneRole.TokenParallel) {
      activeLaneTokenTag(lane) := (activePass.requestId.resize(
        c.taskIdWidth) + U(lane, c.taskIdWidth bits)).resized
    }
  }

  val sequenceFire = Bits(engines bits)
  for (engine <- 0 until engines) {
    val firstLane = engine * cfg.lanesPerEngine
    val sequence = io.residentSequence(engine)
    sequence.valid := state === State.IssueSequences &&
      sequencePending(engine)
    sequence.payload.schedule := activeSchedule
    sequence.payload.sequenceId := activeSequenceId
    sequence.payload.expectedKeys :=
      activePass.maxKeyCount.resize(c.beatCountWidth)
    sequence.payload.kvHead := activePass.kvHead.resize(
      b.contextBankWidth)
    sequence.payload.featureHalf := activeFeatureHalf
    sequence.payload.queryMask := activePass.laneMask(
      firstLane + cfg.lanesPerEngine - 1 downto firstLane)
    for (localLane <- 0 until cfg.lanesPerEngine) {
      val lane = firstLane + localLane
      sequence.payload.tokenTag(localLane) :=
        activeLaneTokenTag(lane)
      sequence.payload.tokenId(localLane) :=
        activePass.tokenId(lane).resize(b.tokenIdWidth)
      sequence.payload.queryHeadId(localLane) :=
        activePass.queryHeadId(lane).resize(
          sequence.payload.queryHeadId(localLane).getWidth)
      sequence.payload.contextSlot(localLane) :=
        U(lane, b.contextSlotWidth bits)
    }
    sequenceFire(engine) := sequence.fire
  }
  val pendingAfter = sequencePending & ~sequenceFire
  when(state === State.IssueSequences) {
    sequencePending := pendingAfter
    when(pendingAfter === 0) {
      state := State.StreamKeys
    }
  }

  val join = new BitNetResidentProbabilityValueJoin(cfg)
  join.io.clearFault := io.pass.fire
  val ingressOpen = state === State.StreamKeys && !ingressComplete
  join.io.probability.valid := io.probability.valid && ingressOpen
  join.io.probability.payload := io.probability.payload
  io.probability.ready := join.io.probability.ready && ingressOpen
  join.io.value.valid := io.value.valid && ingressOpen
  join.io.value.payload := io.value.payload
  io.value.ready := join.io.value.ready && ingressOpen

  val ingressFire = join.io.probability.fire && join.io.value.fire
  when(ingressFire) {
    when(io.probability.keyIndex =/= ingressIndex ||
        io.probability.firstInSequence =/= (ingressIndex === 0) ||
        io.probability.lastInSequence =/=
          (ingressIndex === activePass.maxKeyCount - 1)) {
      fault := True
    }
    when(ingressIndex === activePass.maxKeyCount - 1) {
      ingressComplete := True
    } otherwise {
      ingressIndex := ingressIndex + 1
    }
  }

  val preparedFork = new StreamFork(
    BitNetResidentPreparedBeat(cfg), engines, synchronous = true)
  preparedFork.io.input << join.io.prepared

  val expectedFirst = egressIndex === 0
  val expectedLast = egressIndex === activePass.maxKeyCount - 1
  val metadataMatchesPass = join.io.prepared.requestId ===
    activePass.requestId &&
    join.io.prepared.layer === activePass.layer &&
    join.io.prepared.laneRole === activePass.laneRole &&
    join.io.prepared.passIndex === activePass.passIndex &&
    join.io.prepared.kvHead === activePass.kvHead &&
    join.io.prepared.laneMask === activePass.laneMask &&
    join.io.prepared.tokenId.asBits === activePass.tokenId.asBits &&
    join.io.prepared.queryHeadId.asBits ===
      activePass.queryHeadId.asBits &&
    join.io.prepared.queryPosition.asBits ===
      activePass.queryPosition.asBits &&
    join.io.prepared.keyCount.asBits === activePass.keyCount.asBits &&
    join.io.prepared.keyIndex === egressIndex &&
    join.io.prepared.firstInSequence === expectedFirst &&
    join.io.prepared.lastInSequence === expectedLast

  when(preparedFork.io.input.fire) {
    when(!metadataMatchesPass) {
      fault := True
    }
    submittedKeys := submittedKeys + 1
    when(expectedLast) {
      completedPasses := completedPasses + 1
      state := State.Idle
    } otherwise {
      egressIndex := egressIndex + 1
    }
  }

  for (engine <- 0 until engines) {
    val firstLane = engine * cfg.lanesPerEngine
    val source = preparedFork.io.outputs(engine)
    val beat = io.residentBeat(engine)
    /* A disabled prefill-tail engine is an always-ready sink inside the
     * synchronous fork.  It neither receives a sequence nor exposes a beat,
     * so the active engine can advance without an illegal queryMask=00 peer. */
    beat.valid := source.valid && activeEngineMask(engine)
    source.ready := Mux(activeEngineMask(engine), beat.ready, True)
    beat.payload.sequenceId := activeSequenceId
    beat.payload.keyIndex := egressIndex.resize(c.beatCountWidth)
    beat.payload.firstKey := expectedFirst
    beat.payload.lastKey := expectedLast
    for (site <- 0 until b.siteCount) {
      val sourceFeature = UInt(log2Up(cfg.headDim) bits)
      sourceFeature := (activeFeatureHalf.asUInt * b.siteCount + site)
        .resize(sourceFeature.getWidth)
      beat.payload.rawV(site) := source.value(sourceFeature)
    }
    for (localLane <- 0 until cfg.lanesPerEngine) {
      val lane = firstLane + localLane
      beat.payload.tokenTag(localLane) :=
        activeLaneTokenTag(lane)
      beat.payload.tokenId(localLane) :=
        activePass.tokenId(lane).resize(b.tokenIdWidth)
      beat.payload.factorRaw(localLane) := 0
      val laneActiveForKey = activePass.laneMask(lane) &&
        egressIndex < activePass.keyCount(lane)
      when(metadataMatchesPass && laneActiveForKey) {
        beat.payload.factorRaw(localLane) := source.factorRaw(lane)
      }
    }
  }

  io.protocolFaultSticky := fault || join.io.protocolFault
  io.factorSaturationFault := join.io.factorSaturationFault
  io.acceptedKeys := join.io.acceptedKeys
}
