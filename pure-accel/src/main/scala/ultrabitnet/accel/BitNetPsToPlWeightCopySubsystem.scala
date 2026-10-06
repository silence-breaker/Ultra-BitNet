package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetWeightCopyAxiConfig(
    staging: BitNetPsToPlWeightStagingConfig =
      BitNetPsToPlWeightStagingConfig(),
    dataWidth: Int = 128,
    axiIdWidth: Int = 4,
    axiId: Int = 0,
    commandQueueDepth: Int = 8,
    replayFifoDepth: Int = 4) {
  require(dataWidth == 128)
  require(axiIdWidth >= 1 && axiId >= 0 && axiId < (1 << axiIdWidth))
  require(commandQueueDepth == staging.maxOutstanding,
    "the copy queue must absorb every controller-owned page")
  require(commandQueueDepth >= 2 &&
    (commandQueueDepth & (commandQueueDepth - 1)) == 0)
  require(replayFifoDepth >= 2 &&
    (replayFifoDepth & (replayFifoDepth - 1)) == 0)
  require(staging.pageBytes == 4096 && staging.beatsPerPage == 256)

  val beatBytes: Int = dataWidth / 8
  val beatIndexWidth: Int = log2Up(staging.beatsPerPage)
}

/** Board wrapper maps this to the PS-DDR AXI4 AR channel. */
case class BitNetWeightCopyReadAddress(cfg: BitNetWeightCopyAxiConfig)
    extends Bundle {
  val id = UInt(cfg.axiIdWidth bits)
  val localAddress = UInt(cfg.staging.model.addressWidth bits)
  /** AXI LEN, so one 4 KiB page is 255. */
  val len = UInt(8 bits)
}

/** PS-DDR AXI4 R boundary. */
case class BitNetWeightCopyReadData(cfg: BitNetWeightCopyAxiConfig)
    extends Bundle {
  val id = UInt(cfg.axiIdWidth bits)
  val data = Bits(cfg.dataWidth bits)
  val resp = Bits(2 bits)
  val last = Bool()
}

/** Board wrapper maps this to the PL-DDR AXI4 AW channel. */
case class BitNetWeightCopyWriteAddress(cfg: BitNetWeightCopyAxiConfig)
    extends Bundle {
  val id = UInt(cfg.axiIdWidth bits)
  val localAddress = UInt(cfg.staging.model.addressWidth bits)
  val len = UInt(8 bits)
}

/** PL-DDR AXI4 W boundary; AXI4 deliberately has no WID. */
case class BitNetWeightCopyWriteData(cfg: BitNetWeightCopyAxiConfig)
    extends Bundle {
  val data = Bits(cfg.dataWidth bits)
  val strobe = Bits((cfg.dataWidth / 8) bits)
  val last = Bool()
}

/** PL-DDR AXI4 B boundary. */
case class BitNetWeightCopyWriteResponse(cfg: BitNetWeightCopyAxiConfig)
    extends Bundle {
  val id = UInt(cfg.axiIdWidth bits)
  val resp = Bits(2 bits)
}

object BitNetWeightCopyFaultCode {
  val None: Int = 0x00
  val InvalidCommand: Int = 0xe0
  val ReadIdMismatch: Int = 0xe1
  val ReadResponse: Int = 0xe2
  val ReadLastMismatch: Int = 0xe3
  val WriteIdMismatch: Int = 0xe4
  val WriteResponse: Int = 0xe5
}

object BitNetWeightCopyState extends SpinalEnum(binarySequential) {
  val Idle, ReadAddress, ReadData, ReadDrain, WriteAddress, WriteData,
      WriteResponse, EmitCompletion = newElement()
}

/**
  * One-page-at-a-time, lossless PS-DDR read to PL-DDR write engine.
  *
  * Up to eight controller commands wait in the typed command FIFO.  A page is
  * read completely into one 256x128-bit block RAM and checked before AW is
  * issued, so a bad/short/long/error R transaction never contaminates PL DDR.
  * Writes remain AXI4 ordered, and success is returned only after the matching
  * B response is accepted with OKAY.
  */
class BitNetPsToPlWeightCopyEngine(
    cfg: BitNetWeightCopyAxiConfig = BitNetWeightCopyAxiConfig())
    extends Component {
  private val scfg = cfg.staging
  private val model = scfg.model
  private val pageBeats = scfg.beatsPerPage

  val io = new Bundle {
    val command = slave(Stream(BitNetWeightStagingDmaCommand(scfg)))
    val completion = master(Stream(BitNetWeightStagingDmaCompletion(scfg)))

    val psAr = master(Stream(BitNetWeightCopyReadAddress(cfg)))
    val psR = slave(Stream(BitNetWeightCopyReadData(cfg)))
    val plAw = master(Stream(BitNetWeightCopyWriteAddress(cfg)))
    val plW = master(Stream(BitNetWeightCopyWriteData(cfg)))
    val plB = slave(Stream(BitNetWeightCopyWriteResponse(cfg)))

    val busy = out Bool()
    val queued = out UInt(log2Up(cfg.commandQueueDepth + 1) bits)
    val commandFaultSticky = out Bool()
    val readProtocolFaultSticky = out Bool()
    val readResponseFaultSticky = out Bool()
    val writeProtocolFaultSticky = out Bool()
    val writeResponseFaultSticky = out Bool()
  }

  val commandFifo = StreamFifo(
    BitNetWeightStagingDmaCommand(scfg), cfg.commandQueueDepth)
  commandFifo.io.push << io.command
  io.queued := commandFifo.io.occupancy.resized

  val state = Reg(BitNetWeightCopyState()) init BitNetWeightCopyState.Idle
  val held = Reg(BitNetWeightStagingDmaCommand(scfg)) init
    BitNetWeightStagingDmaCommand(scfg).getZero

  val completionValid = RegInit(False)
  val completionPayload = Reg(BitNetWeightStagingDmaCompletion(scfg)) init
    BitNetWeightStagingDmaCompletion(scfg).getZero
  io.completion.valid := completionValid
  io.completion.payload := completionPayload
  when(io.completion.fire) {
    completionValid := False
    state := BitNetWeightCopyState.Idle
  }

  io.busy := state =/= BitNetWeightCopyState.Idle ||
    commandFifo.io.occupancy =/= 0 || completionValid

  val commandFaultSticky = RegInit(False)
  val readProtocolFaultSticky = RegInit(False)
  val readResponseFaultSticky = RegInit(False)
  val writeProtocolFaultSticky = RegInit(False)
  val writeResponseFaultSticky = RegInit(False)
  io.commandFaultSticky := commandFaultSticky
  io.readProtocolFaultSticky := readProtocolFaultSticky
  io.readResponseFaultSticky := readResponseFaultSticky
  io.writeProtocolFaultSticky := writeProtocolFaultSticky
  io.writeResponseFaultSticky := writeResponseFaultSticky

  // -----------------------------------------------------------------------
  // Fail-closed semantic command validation before either AXI address fires.
  // -----------------------------------------------------------------------
  val queuedCommand = commandFifo.io.pop.payload
  val layerLegal = queuedCommand.layer < model.layerCount
  val pageLegal = queuedCommand.page < scfg.layerPages
  val safeLayer = UInt(model.layerWidth bits)
  val safePage = UInt(scfg.pageIndexWidth bits)
  safeLayer := Mux(layerLegal, queuedCommand.layer, U(0, model.layerWidth bits))
  safePage := Mux(pageLegal, queuedCommand.page,
    U(0, scfg.pageIndexWidth bits))

  val expectedTag = UInt(scfg.dmaTagWidth bits)
  expectedTag := (queuedCommand.generation.asBits ## queuedCommand.layer.asBits ##
    queuedCommand.slot.asBits ## queuedCommand.page.asBits).asUInt
  val pageOffset =
    (safePage.resize(model.addressWidth) << log2Up(scfg.pageBytes)).resized
  val destinationBases = Vec(scfg.staging.layers.map(layer =>
    U(layer.destinationBase, model.addressWidth bits)))
  val expectedDestination = destinationBases(safeLayer) + pageOffset

  val expectedTarget = BitNetAddressTarget()
  val expectedFirstTargetPage = Bool()
  val expectedLastTargetPage = Bool()
  expectedTarget := BitNetFiveBankStripedLayout.WeightGeometry.head._1
  expectedFirstTargetPage := safePage === 0
  expectedLastTargetPage := False
  var targetPageStart = 0
  for ((target, index) <-
      BitNetFiveBankStripedLayout.WeightGeometry.map(_._1).zipWithIndex) {
    val targetPages =
      (scfg.staging.layers.head.targets(index).bytes / scfg.pageBytes).toInt
    val targetPageEnd = targetPageStart + targetPages
    when(safePage >= targetPageStart && safePage < targetPageEnd) {
      expectedTarget := target
      expectedFirstTargetPage := safePage === targetPageStart
      expectedLastTargetPage := safePage === targetPageEnd - 1
    }
    targetPageStart = targetPageEnd
  }

  val addressWidth = scfg.addressCalculationWidth
  val sourceEnd = queuedCommand.sourceLocalAddress.resize(addressWidth) +
    U(scfg.pageBytes, addressWidth bits)
  val destinationEnd =
    queuedCommand.destinationLocalAddress.resize(addressWidth) +
      U(scfg.pageBytes, addressWidth bits)
  val addressesAligned =
    queuedCommand.sourceLocalAddress(log2Up(scfg.pageBytes) - 1 downto 0) === 0 &&
      queuedCommand.destinationLocalAddress(
        log2Up(scfg.pageBytes) - 1 downto 0) === 0
  val addressesInRange = !sourceEnd.msb && !destinationEnd.msb &&
    sourceEnd <= U(scfg.psDdrCapacityBytes, addressWidth bits) &&
    destinationEnd <= U(scfg.plDdrCapacityBytes, addressWidth bits)
  val metadataValid = queuedCommand.bytes === scfg.pageBytes &&
    queuedCommand.beats === pageBeats && queuedCommand.tag === expectedTag &&
    queuedCommand.destinationLocalAddress === expectedDestination &&
    queuedCommand.target === expectedTarget &&
    queuedCommand.firstLayerPage === (safePage === 0) &&
    queuedCommand.lastLayerPage === (safePage === scfg.layerPages - 1) &&
    queuedCommand.firstTargetPage === expectedFirstTargetPage &&
    queuedCommand.lastTargetPage === expectedLastTargetPage
  val commandLegal = layerLegal && pageLegal && addressesAligned &&
    addressesInRange && metadataValid

  commandFifo.io.pop.ready := state === BitNetWeightCopyState.Idle &&
    !completionValid
  when(commandFifo.io.pop.fire) {
    held := queuedCommand
    completionPayload.tag := queuedCommand.tag
    completionPayload.success := False
    completionPayload.faultCode := BitNetWeightCopyFaultCode.InvalidCommand
    when(commandLegal) {
      state := BitNetWeightCopyState.ReadAddress
    } otherwise {
      completionValid := True
      state := BitNetWeightCopyState.EmitCompletion
      commandFaultSticky := True
    }
  }

  // -----------------------------------------------------------------------
  // PS AXI read.  The address is exactly one aligned 4 KiB/256-beat burst.
  // -----------------------------------------------------------------------
  io.psAr.valid := state === BitNetWeightCopyState.ReadAddress
  io.psAr.id := cfg.axiId
  io.psAr.localAddress := held.sourceLocalAddress
  io.psAr.len := 255

  val readBeatIndex = Reg(UInt(cfg.beatIndexWidth bits)) init 0
  val readFault = RegInit(False)
  val readFaultCode = Reg(Bits(8 bits)) init 0
  val pageMemory = Mem(Bits(cfg.dataWidth bits), pageBeats)
  pageMemory.addAttribute("ram_style", "block")

  when(io.psAr.fire) {
    readBeatIndex := 0
    readFault := False
    readFaultCode := 0
    state := BitNetWeightCopyState.ReadData
  }

  val expectedReadLast = readBeatIndex === pageBeats - 1
  val currentReadIdFault = io.psR.id =/= cfg.axiId
  val currentReadResponseFault = io.psR.resp =/= B"00"
  val currentReadLastFault = io.psR.last =/= expectedReadLast
  val currentReadFault = currentReadIdFault || currentReadResponseFault ||
    currentReadLastFault
  val currentReadFaultCode = Bits(8 bits)
  currentReadFaultCode := BitNetWeightCopyFaultCode.ReadLastMismatch
  when(currentReadResponseFault) {
    currentReadFaultCode := BitNetWeightCopyFaultCode.ReadResponse
  }
  when(currentReadIdFault) {
    currentReadFaultCode := BitNetWeightCopyFaultCode.ReadIdMismatch
  }

  io.psR.ready := state === BitNetWeightCopyState.ReadData ||
    state === BitNetWeightCopyState.ReadDrain
  val readDataFire = io.psR.fire &&
    state === BitNetWeightCopyState.ReadData
  pageMemory.write(
    address = readBeatIndex,
    data = io.psR.data,
    enable = readDataFire && !readFault && !currentReadFault)

  when(readDataFire) {
    when(currentReadIdFault || currentReadLastFault) {
      readProtocolFaultSticky := True
    }
    when(currentReadResponseFault) { readResponseFaultSticky := True }
    when(currentReadFault && !readFault) {
      readFault := True
      readFaultCode := currentReadFaultCode
    }

    when(io.psR.last && !expectedReadLast) {
      completionPayload.success := False
      completionPayload.faultCode := Mux(readFault,
        readFaultCode, currentReadFaultCode)
      completionValid := True
      state := BitNetWeightCopyState.EmitCompletion
    } elsewhen (expectedReadLast) {
      when(io.psR.last) {
        when(readFault || currentReadFault) {
          completionPayload.success := False
          completionPayload.faultCode := Mux(readFault,
            readFaultCode, currentReadFaultCode)
          completionValid := True
          state := BitNetWeightCopyState.EmitCompletion
        } otherwise {
          state := BitNetWeightCopyState.WriteAddress
        }
      } otherwise {
        readFault := True
        readFaultCode := BitNetWeightCopyFaultCode.ReadLastMismatch
        state := BitNetWeightCopyState.ReadDrain
      }
    } otherwise {
      readBeatIndex := readBeatIndex + 1
    }
  }

  when(io.psR.fire && state === BitNetWeightCopyState.ReadDrain) {
    when(io.psR.id =/= cfg.axiId || !io.psR.last) {
      readProtocolFaultSticky := True
    }
    when(io.psR.resp =/= B"00") { readResponseFaultSticky := True }
    when(io.psR.last) {
      completionPayload.success := False
      completionPayload.faultCode := readFaultCode
      completionValid := True
      state := BitNetWeightCopyState.EmitCompletion
    }
  }

  // -----------------------------------------------------------------------
  // PL AXI write.  Synchronous BRAM replay plus a small elastic FIFO holds W
  // stable under arbitrary backpressure and prevents page-buffer overwrite.
  // -----------------------------------------------------------------------
  io.plAw.valid := state === BitNetWeightCopyState.WriteAddress
  io.plAw.id := cfg.axiId
  io.plAw.localAddress := held.destinationLocalAddress
  io.plAw.len := 255

  val replayReadIndex = Reg(UInt(cfg.beatIndexWidth bits)) init 0
  val allReplayReadsIssued = RegInit(False)
  val writeBeatIndex = Reg(UInt(cfg.beatIndexWidth bits)) init 0
  val replayFifo = StreamFifo(Bits(cfg.dataWidth bits), cfg.replayFifoDepth)
  val replayReadIssue = state === BitNetWeightCopyState.WriteData &&
    !allReplayReadsIssued && replayFifo.io.availability > 1
  val replayReadData = pageMemory.readSync(replayReadIndex, replayReadIssue)
  val replayResponseValid = RegNext(replayReadIssue) init False
  replayFifo.io.push.valid := replayResponseValid
  replayFifo.io.push.payload := replayReadData

  when(io.plAw.fire) {
    replayReadIndex := 0
    allReplayReadsIssued := False
    writeBeatIndex := 0
    state := BitNetWeightCopyState.WriteData
  }
  when(replayReadIssue) {
    when(replayReadIndex === pageBeats - 1) {
      allReplayReadsIssued := True
    } otherwise {
      replayReadIndex := replayReadIndex + 1
    }
  }

  io.plW.valid := state === BitNetWeightCopyState.WriteData &&
    replayFifo.io.pop.valid
  io.plW.data := replayFifo.io.pop.payload
  io.plW.strobe := B((BigInt(1) << (cfg.dataWidth / 8)) - 1,
    (cfg.dataWidth / 8) bits)
  io.plW.last := writeBeatIndex === pageBeats - 1
  replayFifo.io.pop.ready := state === BitNetWeightCopyState.WriteData &&
    io.plW.ready

  when(io.plW.fire) {
    when(writeBeatIndex === pageBeats - 1) {
      state := BitNetWeightCopyState.WriteResponse
    } otherwise {
      writeBeatIndex := writeBeatIndex + 1
    }
  }

  // Only B retirement can create a successful completion.
  io.plB.ready := state === BitNetWeightCopyState.WriteResponse
  when(io.plB.fire) {
    val idOk = io.plB.id === cfg.axiId
    val responseOk = io.plB.resp === B"00"
    completionPayload.success := idOk && responseOk
    completionPayload.faultCode := BitNetWeightCopyFaultCode.None
    when(!idOk) {
      completionPayload.faultCode := BitNetWeightCopyFaultCode.WriteIdMismatch
      writeProtocolFaultSticky := True
    } elsewhen (!responseOk) {
      completionPayload.faultCode := BitNetWeightCopyFaultCode.WriteResponse
      writeResponseFaultSticky := True
    }
    completionValid := True
    state := BitNetWeightCopyState.EmitCompletion
  }

  // Unexpected channel traffic is never consumed and is made visible.
  when(io.psR.valid && state =/= BitNetWeightCopyState.ReadData &&
    state =/= BitNetWeightCopyState.ReadDrain) {
    readProtocolFaultSticky := True
  }
  when(io.plB.valid && state =/= BitNetWeightCopyState.WriteResponse) {
    writeProtocolFaultSticky := True
  }
}

/** Controller plus the real page copy datapath; board AXI wiring stays outside. */
class BitNetPsToPlWeightStagingSubsystem(
    stagingCfg: BitNetPsToPlWeightStagingConfig =
      BitNetPsToPlWeightStagingConfig(),
    copyCfg: BitNetWeightCopyAxiConfig = BitNetWeightCopyAxiConfig())
    extends Component {
  require(copyCfg.staging == stagingCfg)

  val io = new Bundle {
    val start = slave(Stream(BitNetWeightStagingStart(stagingCfg)))
    val computeGrant =
      master(Stream(BitNetWeightStagingComputeGrant(stagingCfg)))
    val computeCompletion =
      slave(Stream(BitNetWeightStagingComputeCompletion(stagingCfg)))
    val completion =
      master(Stream(BitNetWeightStagingWorkloadCompletion(stagingCfg)))

    val psAr = master(Stream(BitNetWeightCopyReadAddress(copyCfg)))
    val psR = slave(Stream(BitNetWeightCopyReadData(copyCfg)))
    val plAw = master(Stream(BitNetWeightCopyWriteAddress(copyCfg)))
    val plW = master(Stream(BitNetWeightCopyWriteData(copyCfg)))
    val plB = slave(Stream(BitNetWeightCopyWriteResponse(copyCfg)))

    val busy = out Bool()
    val controllerOutstanding = out UInt(
      log2Up(stagingCfg.maxOutstanding + 1) bits)
    val copyQueued = out UInt(log2Up(copyCfg.commandQueueDepth + 1) bits)
    val stagingFaultSticky = out Bool()
    val copyFaultSticky = out Bool()
    /** Malformed command/ID/last traffic; reset-only integration fault. */
    val copyProtocolFaultSticky = out Bool()
    /** Well-formed non-OKAY RRESP/BRESP; ordinary workload/data failure. */
    val copyResponseFaultSticky = out Bool()
  }

  val controller = new BitNetPsToPlWeightStagingController(stagingCfg)
  val copy = new BitNetPsToPlWeightCopyEngine(copyCfg)

  controller.io.start << io.start
  io.computeGrant << controller.io.computeGrant
  controller.io.computeCompletion << io.computeCompletion
  io.completion << controller.io.completion

  copy.io.command << controller.io.dmaCommand
  controller.io.dmaCompletion << copy.io.completion

  io.psAr << copy.io.psAr
  copy.io.psR << io.psR
  io.plAw << copy.io.plAw
  io.plW << copy.io.plW
  copy.io.plB << io.plB

  io.busy := controller.io.busy || copy.io.busy
  io.controllerOutstanding := controller.io.outstanding
  io.copyQueued := copy.io.queued
  io.stagingFaultSticky := controller.io.faultSticky
  io.copyProtocolFaultSticky := copy.io.commandFaultSticky ||
    copy.io.readProtocolFaultSticky || copy.io.writeProtocolFaultSticky
  io.copyResponseFaultSticky := copy.io.readResponseFaultSticky ||
    copy.io.writeResponseFaultSticky
  io.copyFaultSticky := io.copyProtocolFaultSticky ||
    io.copyResponseFaultSticky
}

object GenerateBitNetPsToPlWeightStagingSubsystem extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-ps-to-pl-weight-staging-subsystem")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetPsToPlWeightStagingSubsystem())
}
