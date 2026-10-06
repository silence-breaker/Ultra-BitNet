package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetResidentContextWriterConfig(
    model: BitNetConfig = BitNetConfig.Production,
    continuous: BitNetContinuousMatrixConfig =
      BitNetContinuousMatrixConfig(),
    groupLanes: Int = 8
) {
  require(model.engineCount >= 1 && model.engineCount <= 2 &&
    model.tokenParallelism == model.engineCount * continuous.tokenLanesPerEngine)
  require(model.queryHeadCount == 20 && model.kvHeadCount == 5)
  require(model.gqaGroupSize == 4 && model.headSize == 128)
  require(model.spuCount == 4)
  require(continuous.tokenLanesPerEngine == 2)
  require(model.headSize % continuous.base.siteCount == 0)
  require(continuous.base.dspPWidth == 48)
  require(groupLanes == 8 && model.headSize % groupLanes == 0)

  val engineCount: Int = model.engineCount
  val laneCount: Int = model.tokenParallelism
  val lanesPerEngine: Int = continuous.tokenLanesPerEngine
  val groupCount: Int = model.headSize / groupLanes
  val groupIndexWidth: Int = log2Up(groupCount)
  val queryHeadWidth: Int = log2Up(model.queryHeadCount)
  val kvHeadWidth: Int = log2Up(model.kvHeadCount)
  val passIndexWidth: Int = log2Up(model.queryHeadCount)
}

/** Completion is emitted only after the last Vec4 has entered WideWork. */
case class BitNetAttentionPassDone(cfg: BitNetResidentContextWriterConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val passIndex = UInt(cfg.passIndexWidth bits)
  val laneMask = Bits(cfg.laneCount bits)
  val first = Bool()
  val last = Bool()
  val success = Bool()
  val fault = Bool()
}

private[accel] case class BitNetResidentContextSlot(
    cfg: BitNetResidentContextWriterConfig) extends Bundle {
  val residentLane = UInt(log2Up(cfg.lanesPerEngine) bits)
  val globalLane = UInt(log2Up(cfg.laneCount) bits)
  val queryHeadId = UInt(cfg.queryHeadWidth bits)
  val logicalGroup = UInt(cfg.groupIndexWidth bits)
  val protocolValid = Bool()
  val rawContext = Vec(
    SInt(cfg.continuous.base.dspPWidth bits), cfg.groupLanes)
}

object BitNetResidentContextWriterMath {
  val S32Maximum: BigInt = (BigInt(1) << 31) - 1
  val S32Minimum: BigInt = -(BigInt(1) << 31)

  case class Converted(valueQ16: BigInt, overflow: Boolean)

  /** Resident raw context is Q*.15; one exact shift produces Q16.16. */
  def toQ16(rawContext: BigInt): Converted = {
    require(rawContext >= -(BigInt(1) << 47) &&
      rawContext < (BigInt(1) << 47))
    val shifted = rawContext << 1
    if (shifted > S32Maximum) Converted(S32Maximum, overflow = true)
    else if (shifted < S32Minimum) Converted(S32Minimum, overflow = true)
    else Converted(shifted, overflow = false)
  }

  /** WideWork stores four consecutive features in each physical group. */
  def wideWorkGroup(queryHead: Int, residentGroup: Int, half: Int): Int = {
    require(queryHead >= 0 && queryHead < 20)
    require(residentGroup >= 0 && residentGroup < 16)
    require(half == 0 || half == 1)
    queryHead * 32 + residentGroup * 2 + half
  }
}

/**
  * Converts grouped resident DSP contexts into the physical activation fabric.
  *
  * Each Vec8 result is retained in one small per-engine slot and written as
  * two Vec4 WideWork beats.  Prefill engines target disjoint token ports and
  * can progress together.  Decode maps every Q head to token port zero, where
  * a round-robin arbiter serialises the two asynchronous engine streams.
  */
class BitNetResidentContextWriter(
    cfg: BitNetResidentContextWriterConfig =
      BitNetResidentContextWriterConfig(),
    layout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production))
    extends Component {
  require(layout.cfg == cfg.model)
  private val engines = cfg.engineCount
  private val lanes = cfg.laneCount
  private val m = cfg.continuous
  private val b = m.base

  val io = new Bundle {
    val pass = slave(Stream(BitNetAttentionPass(cfg.model)))
    val result = Vec(
      slave(Stream(BitNetResidentSvResultGroup(m, cfg.groupLanes))),
      engines)
    val writeData = Vec(
      master(Stream(BitNetPhysicalActivationWriteBeat(cfg.model, layout))),
      lanes)
    val passDone = master(Stream(BitNetAttentionPassDone(cfg)))
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val overflowSticky = out Bool()
    val acceptedGroups = out UInt(32 bits)
    val emittedWrites = out UInt(32 bits)
    val completedPasses = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val Idle, Collect, EmitDone = newElement()
  }
  val state = Reg(State()) init State.Idle
  val activePass = Reg(BitNetAttentionPass(cfg.model)) init (
    BitNetAttentionPass(cfg.model).getZero)
  val activeSchedule = BitNetResidentSvSchedule()
  activeSchedule := BitNetResidentSvSchedule.PrefillTokenParallel
  when(activePass.laneRole === BitNetAttentionLaneRole.HeadParallel) {
    activeSchedule := BitNetResidentSvSchedule.DecodeHeadParallel
  }
  val activeSequenceId = (activePass.requestId.resize(m.sequenceIdWidth) +
    activePass.passIndex.resize(m.sequenceIdWidth)).resize(m.sequenceIdWidth)

  val globalProtocolFault = RegInit(False)
  val globalOverflow = RegInit(False)
  val passProtocolFault = RegInit(False)
  val passOverflow = RegInit(False)
  val acceptedGroupCount = Reg(UInt(32 bits)) init 0
  val emittedWriteCount = Reg(UInt(32 bits)) init 0
  val completedPassCount = Reg(UInt(32 bits)) init 0
  io.busy := state =/= State.Idle
  io.protocolFaultSticky := globalProtocolFault
  io.overflowSticky := globalOverflow
  io.acceptedGroups := acceptedGroupCount
  io.emittedWrites := emittedWriteCount
  io.completedPasses := completedPassCount

  val acceptedPerLane = Vec(
    Reg(UInt(cfg.groupIndexWidth bits)) init 0, lanes)
  val acceptedDone = Reg(Bits(lanes bits)) init 0
  val writtenDone = Reg(Bits(lanes bits)) init 0

  val prefillRole = io.pass.mode === BitNetMode.Prefill &&
    io.pass.laneRole === BitNetAttentionLaneRole.TokenParallel
  val decodeRole = io.pass.mode === BitNetMode.Decode &&
    io.pass.laneRole === BitNetAttentionLaneRole.HeadParallel
  val legalPrefillMask = (1 to lanes).map { count =>
    io.pass.laneMask === B((BigInt(1) << count) - 1, lanes bits)
  }.reduce(_ || _)
  val decodePassesPerKv = cfg.model.gqaGroupSize / lanes
  val laneLegal = Vec(Bool(), lanes)
  for (lane <- 0 until lanes) {
    val decodeSubpass = io.pass.passIndex % decodePassesPerKv
    val decodeHead = ((io.pass.kvHead.resize(cfg.queryHeadWidth) << 2) +
      decodeSubpass * lanes + lane).resize(cfg.queryHeadWidth)
    laneLegal(lane) := !io.pass.laneMask(lane) ||
      ((prefillRole && io.pass.queryHeadId(lane) ===
        io.pass.passIndex.resize(cfg.queryHeadWidth)) ||
       (decodeRole && io.pass.queryHeadId(lane) === decodeHead &&
        io.pass.tokenId(lane) === io.pass.tokenId(0)))
  }
  val roleLegal = (prefillRole && legalPrefillMask &&
    io.pass.kvHead ===
      (io.pass.passIndex >> log2Up(cfg.model.gqaGroupSize)).resized) ||
    (decodeRole && io.pass.laneMask.andR &&
      (io.pass.passIndex / decodePassesPerKv).resize(cfg.kvHeadWidth) ===
        io.pass.kvHead)
  val passLegal = io.pass.layer < cfg.model.layerCount &&
    io.pass.kvHead < cfg.model.kvHeadCount && roleLegal &&
    laneLegal.asBits.andR

  io.pass.ready := state === State.Idle
  when(io.pass.fire) {
    activePass := io.pass.payload
    passProtocolFault := !passLegal
    passOverflow := False
    acceptedGroupCount := 0
    emittedWriteCount := 0
    acceptedDone := 0
    writtenDone := 0
    for (lane <- 0 until lanes) { acceptedPerLane(lane) := 0 }
    when(passLegal) {
      state := State.Collect
    } otherwise {
      globalProtocolFault := True
      state := State.EmitDone
    }
  }

  val slotValid = Vec(RegInit(False), engines)
  val slotHalf = Vec(RegInit(False), engines)
  val slots = Vec((0 until engines).map { _ =>
    Reg(BitNetResidentContextSlot(cfg)) init (
      BitNetResidentContextSlot(cfg).getZero)
  })

  val resultFire = Bits(engines bits)
  for (engine <- 0 until engines) {
    val input = io.result(engine)
    resultFire(engine) := input.fire
    input.ready := state === State.Collect && !slotValid(engine)
    val globalLane = (U(engine * cfg.lanesPerEngine,
      log2Up(lanes) bits) + input.residentLane.resize(log2Up(lanes)))
      .resize(log2Up(lanes))
    val mappedActive = activePass.laneMask(globalLane) &&
      !acceptedDone(globalLane)
    val expectedTag = UInt(m.taskIdWidth bits)
    expectedTag := activePass.requestId.resize(m.taskIdWidth)
    when(activeSchedule === BitNetResidentSvSchedule.PrefillTokenParallel) {
      expectedTag := (activePass.requestId.resize(m.taskIdWidth) +
        globalLane.resize(m.taskIdWidth)).resized
    }
    val groupsPerHalf = b.siteCount / cfg.groupLanes
    val logicalGroup = (input.featureHalf.asUInt * groupsPerHalf +
      input.groupIndex.resize(cfg.groupIndexWidth)).resized
    val expectedFirst = input.groupIndex === 0
    val expectedLast = input.groupIndex === groupsPerHalf - 1
    val metadataLegal = mappedActive &&
      input.schedule === activeSchedule &&
      input.sequenceId === activeSequenceId &&
      input.tokenTag === expectedTag &&
      input.tokenId === activePass.tokenId(globalLane).resize(b.tokenIdWidth) &&
      input.kvHead === activePass.kvHead.resize(b.contextBankWidth) &&
      input.queryHeadId === activePass.queryHeadId(globalLane).resize(
        input.queryHeadId.getWidth) &&
      input.contextSlot === globalLane.resize(b.contextSlotWidth) &&
      input.residentLane === U(0, input.residentLane.getWidth bits) +
        (globalLane - U(engine * cfg.lanesPerEngine,
          globalLane.getWidth bits)).resize(input.residentLane.getWidth) &&
      logicalGroup === acceptedPerLane(globalLane) &&
      input.first === expectedFirst && input.last === expectedLast

    when(input.fire) {
      when(mappedActive) {
        slotValid(engine) := True
        slotHalf(engine) := False
        slots(engine).residentLane := input.residentLane
        slots(engine).globalLane := globalLane
        slots(engine).queryHeadId :=
          activePass.queryHeadId(globalLane)
        slots(engine).logicalGroup := logicalGroup
        slots(engine).protocolValid := metadataLegal
        slots(engine).rawContext := input.rawContext
        when(logicalGroup === cfg.groupCount - 1) {
          acceptedDone(globalLane) := True
        } otherwise {
          acceptedPerLane(globalLane) :=
            acceptedPerLane(globalLane) + 1
        }
        when(!metadataLegal) {
          passProtocolFault := True
          globalProtocolFault := True
        }
      } otherwise {
        passProtocolFault := True
        globalProtocolFault := True
      }
    }
  }
  when(resultFire.orR) {
    acceptedGroupCount := acceptedGroupCount +
      CountOne(resultFire).resize(acceptedGroupCount.getWidth)
  }

  val candidate = Vec(
    Stream(BitNetPhysicalActivationWriteBeat(cfg.model, layout)), engines)
  val candidateOverflow = Vec(Bits(cfg.model.spuCount bits), engines)
  for (engine <- 0 until engines) {
    candidate(engine).valid := slotValid(engine)
    candidate(engine).payload.buffer :=
      BitNetPhysicalActivationBuffer.WideWork
    val featureBase = slots(engine).queryHeadId.resize(
      layout.groupIndexWidth + 3) * cfg.model.headSize +
      slots(engine).logicalGroup.resize(layout.groupIndexWidth + 3) *
        cfg.groupLanes +
      slotHalf(engine).asUInt.resize(layout.groupIndexWidth + 3) *
        cfg.model.spuCount
    candidate(engine).payload.group :=
      (featureBase >> log2Up(cfg.model.spuCount))
        .resize(layout.groupIndexWidth)
    candidate(engine).payload.featureValid.setAll()
    candidateOverflow(engine) := 0
    for (shard <- 0 until cfg.model.spuCount) {
      val sourceIndex = UInt(log2Up(cfg.groupLanes) bits)
      sourceIndex := (slotHalf(engine).asUInt.resize(
        sourceIndex.getWidth) * cfg.model.spuCount + shard)
        .resize(sourceIndex.getWidth)
      val shifted = (slots(engine).rawContext(sourceIndex).resize(49) << 1)
        .resize(49)
      val tooHigh = shifted > S(BitNetResidentContextWriterMath.S32Maximum,
        49 bits)
      val tooLow = shifted < S(BitNetResidentContextWriterMath.S32Minimum,
        49 bits)
      candidateOverflow(engine)(shard) :=
        slots(engine).protocolValid && (tooHigh || tooLow)
      candidate(engine).payload.values(shard) := 0
      when(slots(engine).protocolValid) {
        when(tooHigh) {
          candidate(engine).payload.values(shard) :=
            S(BitNetResidentContextWriterMath.S32Maximum, 32 bits)
        } elsewhen(tooLow) {
          candidate(engine).payload.values(shard) :=
            S(BitNetResidentContextWriterMath.S32Minimum, 32 bits)
        } otherwise {
          candidate(engine).payload.values(shard) := shifted.resize(32)
        }
      }
    }
  }

  for (port <- 0 until lanes) {
    io.writeData(port).valid := False
    io.writeData(port).payload :=
      BitNetPhysicalActivationWriteBeat(cfg.model, layout).getZero
  }
  candidate.foreach(_.ready := False)

  val decodeRoundRobin = RegInit(False)
  val decodeSelect = Bool()
  decodeSelect := False
  if (engines > 1) {
    when(candidate(1).valid && (!candidate(0).valid || decodeRoundRobin)) {
      decodeSelect := True
    }
  }
  when(activePass.laneRole === BitNetAttentionLaneRole.HeadParallel) {
    io.writeData(0).valid := candidate(0).valid
    io.writeData(0).payload := candidate(0).payload
    candidate(0).ready := io.writeData(0).ready && !decodeSelect
    if (engines > 1) {
      io.writeData(0).valid := Mux(decodeSelect,
        candidate(1).valid, candidate(0).valid)
      when(decodeSelect) { io.writeData(0).payload := candidate(1).payload }
      candidate(1).ready := io.writeData(0).ready && decodeSelect
    }
    when(io.writeData(0).fire) { decodeRoundRobin := !decodeSelect }
  } otherwise {
    for (engine <- 0 until engines) {
      val lowPort = engine * cfg.lanesPerEngine
      val highPort = lowPort + 1
      val selectHigh = slots(engine).residentLane.asBool
      io.writeData(lowPort).valid := candidate(engine).valid && !selectHigh
      io.writeData(lowPort).payload := candidate(engine).payload
      io.writeData(highPort).valid := candidate(engine).valid && selectHigh
      io.writeData(highPort).payload := candidate(engine).payload
      candidate(engine).ready := Mux(selectHigh,
        io.writeData(highPort).ready, io.writeData(lowPort).ready)
    }
  }

  val finalWrite = Bits(lanes bits)
  finalWrite := 0
  val candidateFire = Bits(engines bits)
  for (engine <- 0 until engines) {
    candidateFire(engine) := candidate(engine).fire
    when(candidate(engine).fire) {
      when(candidateOverflow(engine).orR) {
        passOverflow := True
        globalOverflow := True
      }
      when(slotHalf(engine)) {
        slotValid(engine) := False
        slotHalf(engine) := False
        when(slots(engine).logicalGroup === cfg.groupCount - 1) {
          writtenDone(slots(engine).globalLane) := True
          finalWrite(slots(engine).globalLane) := True
        }
      } otherwise {
        slotHalf(engine) := True
      }
    }
  }
  when(candidateFire.orR) {
    emittedWriteCount := emittedWriteCount +
      CountOne(candidateFire).resize(emittedWriteCount.getWidth)
  }

  val writtenAfter = writtenDone | finalWrite
  when(state === State.Collect &&
      (writtenAfter & activePass.laneMask) === activePass.laneMask) {
    state := State.EmitDone
  }

  io.passDone.valid := state === State.EmitDone
  io.passDone.requestId := activePass.requestId
  io.passDone.layer := activePass.layer
  io.passDone.tokenBase := activePass.tokenBase
  io.passDone.passIndex := activePass.passIndex
  io.passDone.laneMask := activePass.laneMask
  io.passDone.first := activePass.first
  io.passDone.last := activePass.last
  io.passDone.fault := passProtocolFault || passOverflow
  io.passDone.success := !(passProtocolFault || passOverflow)
  when(io.passDone.fire) {
    completedPassCount := completedPassCount + 1
    state := State.Idle
  }
}
