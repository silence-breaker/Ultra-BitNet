package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

case class BitNetUnifiedActivationFabricAdapterConfig(
    model: BitNetConfig = BitNetConfig.Production,
    streaming: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production),
    unified: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()) {
  require(streaming.cfg == model,
    "streaming activation layout/model mismatch")
  require(model.tokenParallelism == 2 || model.tokenParallelism == 4,
    "the activation interface must be TP2 or TP4")
  require(model.spuCount == 4,
    "one legacy activation group must contain four adjacent features")
  require(unified.layout.tokenLanes >= model.tokenParallelism)
  require(unified.layout.bankCount == 32)
  require(unified.layout.activationFeatureCapacity(
    BitNetUnifiedResidentTensorRegion.Hidden) >=
      streaming.features(BitNetPhysicalActivationBuffer.HiddenResident))
  require(unified.layout.activationFeatureCapacity(
    BitNetUnifiedResidentTensorRegion.Narrow) >=
      streaming.features(BitNetPhysicalActivationBuffer.NarrowWork))
  require(unified.layout.activationFeatureCapacity(
    BitNetUnifiedResidentTensorRegion.Wide) >=
      streaming.features(BitNetPhysicalActivationBuffer.WideWork))
}

/**
  * Memory-free compatibility adapter from the legacy TP4 activation fabric
  * ports to the S32 ports of the unified resident tensor semantic frontend.
  *
  * Four legacy token-lane requests naming the same physical buffer and group
  * are coalesced into one four-word production transaction.  A legacy group
  * has four adjacent features, while one unified bank word contains one
  * feature across all four tokens:
  *
  *   localRow = group / 8
  *   bank     = (group % 8) * 4 + shard
  *   element  = token lane
  *
  * Write membership and all 16 possible S32 elements are captured before the
  * semantic Stream is asserted.  Its payload consequently remains stable for
  * the entire deferred-ready interval used by the semantic frontend: legacy
  * producers are acknowledged only after that frontend has authenticated the
  * physical write commit.  The capture registers are transaction state, not
  * a resident tensor store; this component instantiates no Mem, semantic
  * frontend, related-clock bridge, or physical backend.
  */
class BitNetUnifiedActivationFabricAdapter(
    cfg: BitNetUnifiedActivationFabricAdapterConfig =
      BitNetUnifiedActivationFabricAdapterConfig()) extends Component {
  private val model = cfg.model
  private val streaming = cfg.streaming
  private val unified = cfg.unified
  private val unifiedLayout = unified.layout
  private val laneCount = model.tokenParallelism
  private val shardCount = model.spuCount
  private val bufferCount = BitNetPhysicalActivationBuffer.elements.size
  private val maxFeatures = streaming.featureCapacity.values.max
  private val featureWidth = scala.math.max(1, log2Up(maxFeatures + shardCount))
  private val bankGroupWidth = log2Up(unifiedLayout.bankCount / shardCount)

  val io = new Bundle {
    val legacyReadCommand = Vec(
      slave(Stream(BitNetPhysicalActivationReadCommand(model, streaming))),
      laneCount)
    val legacyReadData = Vec(
      master(Stream(BitNetPhysicalActivationReadBeat(model, streaming))),
      laneCount)
    val legacyWriteData = Vec(
      slave(Stream(BitNetPhysicalActivationWriteBeat(model, streaming))),
      laneCount)

    val unifiedActivationRead = master(Stream(
      BitNetUnifiedResidentActivationGroupRead(unified)))
    val unifiedActivationReadData = slave(Stream(
      BitNetUnifiedResidentActivationGroupReadData(unified)))
    val unifiedActivationWrite = master(Stream(
      BitNetUnifiedResidentActivationGroupWrite(unified)))
    val unifiedActivationWriteFault = slave(Stream(
      BitNetUnifiedResidentActivationGroupWriteFault(unified)))

    val readFaultSticky = out Bits(laneCount bits)
    val writeFaultSticky = out Bits(laneCount bits)
    /** Combinational fail-stop request for the cycle that first observes a
      * local reject or unauthenticated semantic completion. */
    val failStopNow = out Bool()
    val protocolFaultSticky = out Bool()
    val readOutstanding = out Bool()
    val writeOutstanding = out Bool()
    val idle = out Bool()
  }

  // Compile-time buffer properties become small muxes.  Encoding three is
  // deliberately treated as illegal rather than indexing a three-entry Vec.
  val bufferGroups = Vec(UInt((streaming.groupIndexWidth + 1) bits), bufferCount)
  val bufferFeatures = Vec(UInt(featureWidth bits), bufferCount)
  val bufferRegion = Vec(BitNetUnifiedResidentTensorRegion(), bufferCount)
  BitNetPhysicalActivationBuffer.elements.zipWithIndex.foreach {
    case (buffer, index) =>
      bufferGroups(index) := U(
        streaming.groups(buffer), bufferGroups(index).getWidth bits)
      bufferFeatures(index) := U(
        streaming.features(buffer), featureWidth bits)
  }
  bufferRegion(BitNetPhysicalActivationBuffer.HiddenResident.position) :=
    BitNetUnifiedResidentTensorRegion.Hidden
  bufferRegion(BitNetPhysicalActivationBuffer.NarrowWork.position) :=
    BitNetUnifiedResidentTensorRegion.Narrow
  bufferRegion(BitNetPhysicalActivationBuffer.WideWork.position) :=
    BitNetUnifiedResidentTensorRegion.Wide

  private def safeBufferIndex(
      buffer: BitNetPhysicalActivationBuffer.C): UInt = {
    val raw = buffer.asBits.asUInt
    Mux(raw < U(bufferCount, raw.getWidth bits), raw,
      U(0, raw.getWidth bits))
  }

  private def bufferEncodingLegal(
      buffer: BitNetPhysicalActivationBuffer.C): Bool =
    buffer.asBits.asUInt < U(bufferCount, buffer.asBits.getWidth bits)

  private def commandLegal(
      buffer: BitNetPhysicalActivationBuffer.C,
      group: UInt): Bool = {
    val index = safeBufferIndex(buffer)
    bufferEncodingLegal(buffer) &&
      group.resize(streaming.groupIndexWidth + 1) < bufferGroups(index)
  }

  private def regionOf(
      buffer: BitNetPhysicalActivationBuffer.C):
      BitNetUnifiedResidentTensorRegion.C =
    bufferRegion(safeBufferIndex(buffer))

  private def localRowOf(group: UInt): UInt =
    (group >> 3).resize(unifiedLayout.localRowWidth)

  private def bankGroupOf(group: UInt): UInt =
    group.resize(bankGroupWidth)

  private def featureOf(group: UInt, shard: Int): UInt = {
    val base = (group.resize(featureWidth) << 2).resize(featureWidth)
    (base + U(shard, featureWidth bits)).resize(featureWidth)
  }

  private def featureValid(
      buffer: BitNetPhysicalActivationBuffer.C,
      group: UInt,
      shard: Int): Bool = {
    val index = safeBufferIndex(buffer)
    commandLegal(buffer, group) &&
      featureOf(group, shard) < bufferFeatures(index)
  }

  // -------------------------------------------------------------------
  // Read command capture/coalescing.
  // -------------------------------------------------------------------
  val readCommandActive = RegInit(False)
  val readResponseActive = RegInit(False)
  val readLocalFault = RegInit(False)
  val readBuffer = Reg(BitNetPhysicalActivationBuffer()) init
    BitNetPhysicalActivationBuffer.HiddenResident
  val readGroup = Reg(UInt(streaming.groupIndexWidth bits)) init 0
  val readSelected = Reg(Bits(laneCount bits)) init 0
  val readPending = Reg(Bits(laneCount bits)) init 0

  val readAny = io.legacyReadCommand.map(_.valid).reduce(_ || _)
  val readLeader = UInt(log2Up(laneCount) bits)
  readLeader := 0
  for (lane <- (0 until laneCount).reverse) {
    when(io.legacyReadCommand(lane).valid) {
      readLeader := lane
    }
  }
  val readLeaderBuffer = io.legacyReadCommand(readLeader).payload.buffer
  val readLeaderGroup = io.legacyReadCommand(readLeader).payload.group
  val readMatching = Bits(laneCount bits)
  for (lane <- 0 until laneCount) {
    readMatching(lane) := io.legacyReadCommand(lane).valid &&
      io.legacyReadCommand(lane).payload.buffer === readLeaderBuffer &&
      io.legacyReadCommand(lane).payload.group === readLeaderGroup
  }

  val readFault = Reg(Bits(laneCount bits)) init 0
  val writeFault = Reg(Bits(laneCount bits)) init 0
  val protocolFault = RegInit(False)

  when(!readCommandActive && !readResponseActive && readAny &&
      !protocolFault) {
    readCommandActive := True
    readBuffer := readLeaderBuffer
    readGroup := readLeaderGroup
    readSelected := readMatching
    readLocalFault := !commandLegal(readLeaderBuffer, readLeaderGroup)
  }

  val selectedReadSourcesValid = Bool()
  selectedReadSourcesValid := True
  for (lane <- 0 until laneCount) {
    when(readSelected(lane) && !io.legacyReadCommand(lane).valid) {
      selectedReadSourcesValid := False
    }
  }

  val readFeatureValid = Bits(shardCount bits)
  for (shard <- 0 until shardCount) {
    readFeatureValid(shard) := featureValid(readBuffer, readGroup, shard)
  }

  io.unifiedActivationRead.valid := readCommandActive &&
    !readLocalFault && selectedReadSourcesValid && !protocolFault
  io.unifiedActivationRead.region := regionOf(readBuffer)
  io.unifiedActivationRead.localRow := localRowOf(readGroup)
  io.unifiedActivationRead.bankGroup := bankGroupOf(readGroup)

  for (lane <- 0 until laneCount) {
    io.legacyReadCommand(lane).ready := readCommandActive &&
      readSelected(lane) && selectedReadSourcesValid &&
      Mux(readLocalFault, True, io.unifiedActivationRead.ready)
  }

  val legalReadAccepted = io.unifiedActivationRead.fire
  val localReadAccepted = readCommandActive && readLocalFault &&
    selectedReadSourcesValid
  when(legalReadAccepted || localReadAccepted) {
    readCommandActive := False
    readResponseActive := True
    readPending := readSelected
    when(localReadAccepted) {
      readFault := readFault | readSelected
    }
  }

  // -------------------------------------------------------------------
  // Split one unified response into independently backpressured lane beats.
  // A consumed lane is masked while the semantic response remains retained
  // for slower consumers, preventing duplicate Stream transfers.
  // -------------------------------------------------------------------
  val expectedReadIdentity =
    io.unifiedActivationReadData.payload.region === regionOf(readBuffer) &&
    io.unifiedActivationReadData.payload.localRow === localRowOf(readGroup) &&
    io.unifiedActivationReadData.payload.bankGroup === bankGroupOf(readGroup)
  val semanticReadMismatch = io.unifiedActivationReadData.valid &&
    (!readResponseActive || readLocalFault || !expectedReadIdentity)
  val semanticReadFault = io.unifiedActivationReadData.payload.fault ||
    !expectedReadIdentity
  val readResponseValid = readResponseActive &&
    (readLocalFault || io.unifiedActivationReadData.valid)
  val pendingReadersReady = Bool()
  pendingReadersReady := True
  for (lane <- 0 until laneCount) {
    when(readPending(lane) && !io.legacyReadData(lane).ready) {
      pendingReadersReady := False
    }
  }

  for (lane <- 0 until laneCount) {
    io.legacyReadData(lane).valid := readResponseValid && readPending(lane)
    io.legacyReadData(lane).payload.buffer := readBuffer
    io.legacyReadData(lane).payload.group := readGroup
    io.legacyReadData(lane).payload.featureValid := Mux(
      readLocalFault || semanticReadFault,
      B(0, shardCount bits), readFeatureValid)
    io.legacyReadData(lane).payload.fault :=
      readLocalFault || semanticReadFault
    for (shard <- 0 until shardCount) {
      val selectedValue =
        io.unifiedActivationReadData.payload.words(shard).token(lane)
      io.legacyReadData(lane).payload.values(shard) := Mux(
        readLocalFault || semanticReadFault || !readFeatureValid(shard),
        S(0, 32 bits), selectedValue)
    }
  }

  io.unifiedActivationReadData.ready := !readResponseActive ||
    readLocalFault || pendingReadersReady

  when(readResponseActive && readResponseValid) {
    val consumed = Bits(laneCount bits)
    for (lane <- 0 until laneCount) {
      consumed(lane) := readPending(lane) &&
        io.legacyReadData(lane).ready
    }
    when(consumed.orR) {
      readPending := readPending & ~consumed
    }
    when((readPending & ~consumed) === 0) {
      readResponseActive := False
      readLocalFault := False
      readPending := 0
    }
  }

  when(io.unifiedActivationReadData.valid && readResponseActive &&
      (io.unifiedActivationReadData.payload.fault ||
        !expectedReadIdentity)) {
    readFault := readFault | readPending
  }

  // -------------------------------------------------------------------
  // Write capture/coalescing.  Each of the four group words carries an
  // independently masked token-lane vector for partial TP4 beats.
  // -------------------------------------------------------------------
  val writeActive = RegInit(False)
  val writeLocalReject = RegInit(False)
  val writeBuffer = Reg(BitNetPhysicalActivationBuffer()) init
    BitNetPhysicalActivationBuffer.HiddenResident
  val writeGroup = Reg(UInt(streaming.groupIndexWidth bits)) init 0
  val writeSelected = Reg(Bits(laneCount bits)) init 0
  val writeFeatureMasks = Vec.fill(laneCount)(
    Reg(Bits(shardCount bits)) init 0)
  val writeValues = Vec.fill(laneCount)(Vec.fill(shardCount)(
    Reg(SInt(32 bits)) init 0))

  val writeAny = io.legacyWriteData.map(_.valid).reduce(_ || _)
  val writeLeader = UInt(log2Up(laneCount) bits)
  writeLeader := 0
  for (lane <- (0 until laneCount).reverse) {
    when(io.legacyWriteData(lane).valid) {
      writeLeader := lane
    }
  }
  val writeLeaderBuffer = io.legacyWriteData(writeLeader).payload.buffer
  val writeLeaderGroup = io.legacyWriteData(writeLeader).payload.group
  val writeMatching = Bits(laneCount bits)
  for (lane <- 0 until laneCount) {
    writeMatching(lane) := io.legacyWriteData(lane).valid &&
      io.legacyWriteData(lane).payload.buffer === writeLeaderBuffer &&
      io.legacyWriteData(lane).payload.group === writeLeaderGroup
  }

  when(!writeActive && writeAny && !protocolFault) {
    writeActive := True
    writeLocalReject := !commandLegal(writeLeaderBuffer, writeLeaderGroup)
    writeBuffer := writeLeaderBuffer
    writeGroup := writeLeaderGroup
    writeSelected := writeMatching
    for (lane <- 0 until laneCount; shard <- 0 until shardCount) {
      when(writeMatching(lane)) {
        writeFeatureMasks(lane)(shard) :=
          io.legacyWriteData(lane).payload.featureValid(shard)
        writeValues(lane)(shard) :=
          io.legacyWriteData(lane).payload.values(shard)
      } otherwise {
        writeFeatureMasks(lane)(shard) := False
        writeValues(lane)(shard) := 0
      }
    }
  }

  val selectedWriteSourcesValid = Bool()
  selectedWriteSourcesValid := True
  for (lane <- 0 until laneCount) {
    when(writeSelected(lane) && !io.legacyWriteData(lane).valid) {
      selectedWriteSourcesValid := False
    }
  }

  io.unifiedActivationWrite.valid := writeActive && !writeLocalReject &&
    selectedWriteSourcesValid && !protocolFault
  io.unifiedActivationWrite.region := regionOf(writeBuffer)
  io.unifiedActivationWrite.localRow := localRowOf(writeGroup)
  io.unifiedActivationWrite.bankGroup := bankGroupOf(writeGroup)
  for (shard <- 0 until shardCount) {
    io.unifiedActivationWrite.elementMask(shard) := 0
    io.unifiedActivationWrite.words(shard).assignFromBits(
      B(0, unifiedLayout.wordWidth bits))
    when(featureValid(writeBuffer, writeGroup, shard)) {
      for (lane <- 0 until laneCount) {
        io.unifiedActivationWrite.elementMask(shard)(lane) :=
          writeSelected(lane) && writeFeatureMasks(lane)(shard)
        io.unifiedActivationWrite.words(shard).token(lane) :=
          writeValues(lane)(shard)
      }
    }
  }

  val expectedWriteWordMask = Bits(shardCount bits)
  for (shard <- 0 until shardCount) {
    expectedWriteWordMask(shard) :=
      io.unifiedActivationWrite.elementMask(shard).orR
  }

  for (lane <- 0 until laneCount) {
    io.legacyWriteData(lane).ready := writeActive &&
      writeSelected(lane) && selectedWriteSourcesValid &&
      Mux(writeLocalReject, True, io.unifiedActivationWrite.ready)
  }

  val localWriteAccepted = writeActive && writeLocalReject &&
    selectedWriteSourcesValid
  when(io.unifiedActivationWrite.fire || localWriteAccepted) {
    writeActive := False
    writeSelected := 0
    when(localWriteAccepted) {
      writeFault := writeFault | writeSelected
    }
  }

  // A semantic write fault means the authenticated-commit contract failed or
  // a malformed command reached the frontend.  It is consumed to unblock its
  // retained fault slot, but the adapter itself remains fail-stopped.
  val expectedWriteIdentity =
    io.unifiedActivationWriteFault.payload.region === regionOf(writeBuffer) &&
    io.unifiedActivationWriteFault.payload.localRow === localRowOf(writeGroup) &&
    io.unifiedActivationWriteFault.payload.bankGroup ===
      bankGroupOf(writeGroup) &&
    io.unifiedActivationWriteFault.payload.wordMask === expectedWriteWordMask
  io.unifiedActivationWriteFault.ready := True
  when(io.unifiedActivationWriteFault.valid) {
    protocolFault := True
    when(writeActive && expectedWriteIdentity) {
      writeFault := writeFault | writeSelected
    }
  }
  when(semanticReadMismatch) {
    protocolFault := True
  }

  io.readFaultSticky := readFault
  io.writeFaultSticky := writeFault
  io.failStopNow := localReadAccepted || localWriteAccepted ||
    semanticReadMismatch ||
    (io.unifiedActivationReadData.valid && readResponseActive &&
      semanticReadFault) ||
    io.unifiedActivationWriteFault.valid
  io.protocolFaultSticky := protocolFault || semanticReadMismatch
  io.readOutstanding := readCommandActive || readResponseActive
  io.writeOutstanding := writeActive
  io.idle := !readCommandActive && !readResponseActive && !writeActive
}

object GenerateBitNetUnifiedActivationFabricAdapter extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-unified-activation-fabric-adapter")
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetUnifiedActivationFabricAdapter())
}
