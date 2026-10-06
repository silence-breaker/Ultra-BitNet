package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Query-cache geometry between QKV post-processing and resident attention.
  *
  * Four 128-bit memories retain one complete Q projection round.  Prefill
  * reads the same head from four token memories in parallel; decode reuses the
  * single active-token memory and reads the four heads of one GQA group over
  * 32 cycles.  The latter cost is negligible beside a KV sweep and avoids a
  * second copy of the 20x4x128-byte query image.
  */
case class BitNetQResidentBufferConfig(
    model: BitNetConfig = BitNetConfig.Production,
    qkv: BitNetQkvPostprocessConfig = BitNetQkvPostprocessConfig(),
    qk: BitNetQkTp4AssemblerConfig = BitNetQkTp4AssemblerConfig()
) {
  require(model.tokenParallelism == 2 || model.tokenParallelism == 4)
  require(model.queryHeadCount == 20 && model.gqaGroupSize == 4)
  require(model.headSize == 128)
  require(qkv.tokenLanes == model.tokenParallelism)
  require(qkv.queryHeadCount == model.queryHeadCount)
  require(qkv.headSize == model.headSize)
  require(qk.model == model)

  val bytesPerSegment: Int = 16
  val segmentsPerHead: Int = model.headSize / bytesPerSegment
  val memoryDepth: Int = model.queryHeadCount * segmentsPerHead
  val memoryAddressWidth: Int = log2Up(memoryDepth)
  val readIndexWidth: Int = log2Up(model.gqaGroupSize * segmentsPerHead)
}

/** Starts ownership of the one-round resident Q image. */
case class BitNetQResidentLoadCommand(cfg: BitNetQResidentBufferConfig)
    extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val mode = BitNetMode()
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenId = Vec(
    UInt(cfg.model.sequenceWidth bits), cfg.model.tokenParallelism)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
}

case class BitNetQResidentLoadCompletion(
    cfg: BitNetQResidentBufferConfig) extends Bundle {
  val requestId = UInt(cfg.model.requestIdWidth bits)
  val layer = UInt(cfg.model.layerWidth bits)
  val tokenBase = UInt(cfg.model.sequenceWidth bits)
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val success = Bool()
}

/**
  * Packet-free Q resident buffer.
  *
  * The scalar ingress is the direct output of [[BitNetQkvPostprocess]].  It
  * is transposed into 128-bit segments as it arrives, so no Q tensor is ever
  * written to DDR.  A pass lookup rebuilds the single wide
  * [[BitNetQkQueryBlock]] consumed by the physical QK array.
  */
class BitNetQResidentBuffer(
    cfg: BitNetQResidentBufferConfig = BitNetQResidentBufferConfig())
    extends Component {
  private val model = cfg.model
  private val lanes = model.tokenParallelism
  private val heads = model.queryHeadCount
  private val segments = cfg.segmentsPerHead

  val io = new Bundle {
    val load = slave(Stream(BitNetQResidentLoadCommand(cfg)))
    val sample = slave(Stream(BitNetQkvQuantizedSample(cfg.qkv)))
    val loadCompletion = master(Stream(BitNetQResidentLoadCompletion(cfg)))
    val lookup = slave(Stream(BitNetAttentionPass(model)))
    val block = master(Stream(BitNetQkQueryBlock(cfg.qk)))

    val loaded = out Bool()
    val busy = out Bool()
    val protocolFaultSticky = out Bool()
    val acceptedLoads = out UInt(32 bits)
    val completedLoads = out UInt(32 bits)
    val completedLookups = out UInt(32 bits)
  }

  val memories = Array.fill(lanes)(
    Mem(Bits(128 bits), cfg.memoryDepth)
      .addAttribute("ram_style", "block")
  )
  val factors = Vec((0 until heads * lanes).map { _ =>
    Reg(UInt(32 bits)) init 0
  })
  val segmentAccumulator = Vec(
    Reg(Bits(128 bits)) init 0, lanes)

  val activeLoad = Reg(BitNetQResidentLoadCommand(cfg)) init (
    BitNetQResidentLoadCommand(cfg).getZero)
  val loadActive = RegInit(False)
  val imageLoaded = RegInit(False)
  val loadFault = RegInit(False)
  val faultSticky = RegInit(False)
  val frameActive = RegInit(False)
  val activeHead = Reg(UInt(cfg.qkv.headWidth bits)) init 0
  val expectedElement = Reg(UInt(cfg.qkv.elementWidth bits)) init 0
  val expectedLane = Reg(UInt(cfg.qkv.tokenLaneWidth bits)) init 0
  val headValid = Reg(Bits(heads bits)) init 0

  val completionValid = RegInit(False)
  val completionPayload = Reg(BitNetQResidentLoadCompletion(cfg)) init (
    BitNetQResidentLoadCompletion(cfg).getZero)
  io.loadCompletion.valid := completionValid
  io.loadCompletion.payload := completionPayload
  when(io.loadCompletion.fire) { completionValid := False }

  val acceptedLoadCount = Reg(UInt(32 bits)) init 0
  val completedLoadCount = Reg(UInt(32 bits)) init 0
  val completedLookupCount = Reg(UInt(32 bits)) init 0
  io.acceptedLoads := acceptedLoadCount
  io.completedLoads := completedLoadCount
  io.completedLookups := completedLookupCount
  io.loaded := imageLoaded
  io.protocolFaultSticky := faultSticky

  val outputValid = RegInit(False)
  val activeLookup = Reg(BitNetAttentionPass(model)) init (
    BitNetAttentionPass(model).getZero)
  val lookupPoison = RegInit(False)
  val outputSegments = Vec((0 until lanes).map { _ =>
    Vec(Reg(Bits(128 bits)) init 0, segments)
  })
  val outputFactors = Vec(Reg(UInt(32 bits)) init 0, lanes)

  val readActive = RegInit(False)
  val readIssuedAll = RegInit(False)
  val readIndex = Reg(UInt(cfg.readIndexWidth bits)) init 0

  io.load.ready := !loadActive && !readActive && !outputValid &&
    !completionValid
  val legalPrefillLoad = io.load.mode === BitNetMode.Prefill &&
    (io.load.tokenMask === B"4'b0001" ||
      io.load.tokenMask === B"4'b0011" ||
      io.load.tokenMask === B"4'b0111" ||
      io.load.tokenMask === B"4'b1111")
  val legalDecodeLoad = io.load.mode === BitNetMode.Decode &&
    io.load.tokenMask === B"4'b0001"
  val loadCommandLegal = io.load.layer < model.layerCount &&
    (legalPrefillLoad || legalDecodeLoad)

  when(io.load.fire) {
    activeLoad := io.load.payload
    loadActive := True
    imageLoaded := False
    loadFault := !loadCommandLegal
    headValid := 0
    frameActive := False
    expectedElement := 0
    expectedLane := 0
    for (lane <- 0 until lanes) {
      segmentAccumulator(lane) := B(0, 128 bits)
    }
    acceptedLoadCount := acceptedLoadCount + 1
    when(!loadCommandLegal) { faultSticky := True }
  }

  val expectedFirst = !frameActive && expectedElement === 0 &&
    expectedLane === 0
  val expectedLast = frameActive &&
    expectedElement === model.headSize - 1 && expectedLane === lanes - 1
  val sampleHeadInRange = io.sample.head < heads
  val sampleHeadOneHot = UIntToOh(io.sample.head, heads)
  val sampleHeadDuplicate = (headValid & sampleHeadOneHot).orR
  val sampleFactorIndex = (io.sample.head.resize(log2Up(heads * lanes)) *
    lanes + io.sample.tokenLane.resize(log2Up(heads * lanes)))
    .resize(log2Up(heads * lanes))
  val expectedTokenId = activeLoad.tokenId(io.sample.tokenLane)
  val expectedTokenActive = activeLoad.tokenMask(io.sample.tokenLane)
  val sampleOrderLegal = io.sample.element === expectedElement &&
    io.sample.tokenLane === expectedLane
  val sampleFrameLegal = Mux(expectedFirst,
    io.sample.first && !sampleHeadDuplicate,
    !io.sample.first && io.sample.head === activeHead)
  val sampleEndLegal = io.sample.last === expectedLast
  val sampleFactorLegal = io.sample.element === 0 ||
    !io.sample.tokenActive ||
    io.sample.dequantFactorQ16 === factors(sampleFactorIndex)
  val sampleLegal = sampleHeadInRange && sampleOrderLegal &&
    sampleFrameLegal && sampleEndLegal && sampleFactorLegal &&
    io.sample.tokenId.resize(model.sequenceWidth) === expectedTokenId &&
    io.sample.tokenActive === expectedTokenActive

  io.sample.ready := loadActive && !completionValid
  when(io.sample.fire) {
    when(!sampleLegal) {
      loadFault := True
      faultSticky := True
    }

    when(expectedFirst) {
      activeHead := io.sample.head
      frameActive := True
      for (lane <- 0 until lanes) {
        segmentAccumulator(lane) := B(0, 128 bits)
      }
    }

    when(io.sample.element === 0) {
      factors(sampleFactorIndex) := Mux(io.sample.tokenActive,
        io.sample.dequantFactorQ16, U(0, 32 bits))
      when(io.sample.tokenActive && !io.sample.dequantFactorQ16.orR) {
        loadFault := True
        faultSticky := True
      }
    }

    val byteInSegment = io.sample.element(3 downto 0)
    val segmentIndex = (io.sample.element >> 4).resize(
      log2Up(segments))
    val assembledSegment = Bits(128 bits)
    assembledSegment := segmentAccumulator(io.sample.tokenLane)
    switch(byteInSegment) {
      for (byte <- 0 until cfg.bytesPerSegment) {
        is(byte) {
          assembledSegment(byte * 8 + 7 downto byte * 8) := Mux(
            io.sample.tokenActive,
            io.sample.value.asBits,
            B(0, 8 bits))
        }
      }
    }
    segmentAccumulator(io.sample.tokenLane) := assembledSegment

    when(byteInSegment === cfg.bytesPerSegment - 1) {
      val writeAddress = ((io.sample.head.resize(cfg.memoryAddressWidth) <<
        log2Up(segments)) + segmentIndex.resize(cfg.memoryAddressWidth))
        .resize(cfg.memoryAddressWidth)
      for (lane <- 0 until lanes) {
        when(io.sample.tokenLane === lane) {
          memories(lane).write(writeAddress, assembledSegment)
        }
      }
      segmentAccumulator(io.sample.tokenLane) := B(0, 128 bits)
    }

    when(expectedLane === lanes - 1) {
      expectedLane := 0
      when(expectedElement === model.headSize - 1) {
        expectedElement := 0
      } otherwise {
        expectedElement := expectedElement + 1
      }
    } otherwise {
      expectedLane := expectedLane + 1
    }

    when(expectedLast) {
      val headsAfter = headValid | sampleHeadOneHot
      headValid := headsAfter
      frameActive := False
      when(headsAfter.andR) {
        loadActive := False
        imageLoaded := True
        completionValid := True
        completionPayload.requestId := activeLoad.requestId
        completionPayload.layer := activeLoad.layer
        completionPayload.tokenBase := activeLoad.tokenBase
        completionPayload.tokenMask := activeLoad.tokenMask
        completionPayload.success := !loadFault && sampleLegal &&
          !(io.sample.tokenActive && !io.sample.dequantFactorQ16.orR)
        completedLoadCount := completedLoadCount + 1
      }
    }
  }

  val lookupPrefill = io.lookup.mode === BitNetMode.Prefill &&
    io.lookup.laneRole === BitNetAttentionLaneRole.TokenParallel
  val lookupDecode = io.lookup.mode === BitNetMode.Decode &&
    io.lookup.laneRole === BitNetAttentionLaneRole.HeadParallel
  val prefillMetadataMatches = io.lookup.laneMask === activeLoad.tokenMask &&
    io.lookup.tokenId.asBits === activeLoad.tokenId.asBits
  val decodeTokenMatches = io.lookup.laneMask.andR &&
    activeLoad.tokenMask === B"4'b0001" &&
    io.lookup.tokenId.map(_ === activeLoad.tokenId(0)).andR
  val lookupLegal = imageLoaded &&
    io.lookup.requestId === activeLoad.requestId &&
    io.lookup.layer === activeLoad.layer &&
    io.lookup.tokenBase === activeLoad.tokenBase &&
    io.lookup.mode === activeLoad.mode &&
    ((lookupPrefill && prefillMetadataMatches) ||
      (lookupDecode && decodeTokenMatches))

  io.lookup.ready := imageLoaded && !readActive && !outputValid
  when(io.lookup.fire) {
    activeLookup := io.lookup.payload
    lookupPoison := !lookupLegal
    readActive := True
    readIssuedAll := False
    readIndex := 0
    for (lane <- 0 until lanes; segment <- 0 until segments) {
      outputSegments(lane)(segment) := 0
    }
    for (lane <- 0 until lanes) {
      val prefillFactorIndex =
        (io.lookup.passIndex.resize(log2Up(heads * lanes)) * lanes + lane)
          .resize(log2Up(heads * lanes))
      val decodeHead = ((io.lookup.kvHead.resize(cfg.qkv.headWidth) << 2) +
        lane).resize(cfg.qkv.headWidth)
      val decodeFactorIndex =
        (decodeHead.resize(log2Up(heads * lanes)) * lanes)
          .resize(log2Up(heads * lanes))
      outputFactors(lane) := Mux(!lookupLegal || !io.lookup.laneMask(lane),
        U(0, 32 bits),
        Mux(lookupPrefill, factors(prefillFactorIndex),
          factors(decodeFactorIndex)))
    }
    when(!lookupLegal) { faultSticky := True }
  }

  val activePrefill = activeLookup.mode === BitNetMode.Prefill
  val requestedLane = (readIndex >> log2Up(segments)).resize(2)
  val requestedSegment = readIndex(log2Up(segments) - 1 downto 0)
  val requestedHead = UInt(cfg.qkv.headWidth bits)
  requestedHead := activeLookup.passIndex.resize(cfg.qkv.headWidth)
  when(!activePrefill) {
    requestedHead := ((activeLookup.kvHead.resize(cfg.qkv.headWidth) << 2) +
      requestedLane).resize(cfg.qkv.headWidth)
  }
  val readAddress = ((requestedHead.resize(cfg.memoryAddressWidth) <<
    log2Up(segments)) + requestedSegment.resize(cfg.memoryAddressWidth))
    .resize(cfg.memoryAddressWidth)
  val readIssue = readActive && !readIssuedAll
  val finalReadIndex = Mux(activePrefill,
    U(segments - 1, cfg.readIndexWidth bits),
    U(model.gqaGroupSize * segments - 1, cfg.readIndexWidth bits))
  val memoryReadData = memories.zipWithIndex.map { case (memory, lane) =>
    val laneReadEnable = if (lane == 0) readIssue
      else readIssue && activePrefill
    memory.readSync(readAddress, laneReadEnable)
  }
  val responseValid = RegNext(readIssue) init False
  val responsePrefill = RegNextWhen(activePrefill, readIssue) init False
  val responseLane = RegNextWhen(requestedLane, readIssue) init 0
  val responseSegment = RegNextWhen(requestedSegment, readIssue) init 0
  val responseLast = RegNextWhen(readIndex === finalReadIndex, readIssue) init False

  when(readIssue) {
    when(readIndex === finalReadIndex) {
      readIssuedAll := True
    } otherwise {
      readIndex := readIndex + 1
    }
  }

  when(responseValid) {
    when(responsePrefill) {
      for (lane <- 0 until lanes) {
        outputSegments(lane)(responseSegment) := Mux(
          lookupPoison || !activeLookup.laneMask(lane),
          B(0, 128 bits), memoryReadData(lane))
      }
    } otherwise {
      outputSegments(responseLane)(responseSegment) := Mux(
        lookupPoison || !activeLookup.laneMask(responseLane),
        B(0, 128 bits), memoryReadData(0))
    }
    when(responseLast) {
      readActive := False
      outputValid := True
    }
  }

  io.block.valid := outputValid
  io.block.requestId := activeLookup.requestId
  io.block.layer := activeLookup.layer
  io.block.laneRole := activeLookup.laneRole
  io.block.passIndex := activeLookup.passIndex
  io.block.kvHead := activeLookup.kvHead
  io.block.laneMask := activeLookup.laneMask
  io.block.tokenId := activeLookup.tokenId
  io.block.queryHeadId := activeLookup.queryHeadId
  io.block.queryPosition := activeLookup.queryPosition
  io.block.keyCount := activeLookup.keyCount
  io.block.qFactorQ16 := outputFactors
  for (lane <- 0 until lanes; element <- 0 until model.headSize) {
    val segment = element / cfg.bytesPerSegment
    val byte = element % cfg.bytesPerSegment
    io.block.q(lane)(element) :=
      outputSegments(lane)(segment)(byte * 8 + 7 downto byte * 8).asSInt
  }
  when(io.block.fire) {
    outputValid := False
    completedLookupCount := completedLookupCount + 1
  }

  io.busy := loadActive || completionValid || readActive || outputValid
}

object GenerateBitNetQResidentBuffer extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-q-resident-buffer")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetQResidentBuffer())
}
