package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * The three physical activation stores in the production datapath.
  *
  * This is a liveness-based layout, not one RAM allocation per tensor name.
  * Transformer intermediates whose lifetimes do not overlap deliberately
  * reuse NarrowWork/WideWork.  Gate/Up values, LM logits and the final token
  * are streamed through their consumers and therefore never acquire a large
  * activation-buffer slot.
  */
object BitNetPhysicalActivationBuffer extends SpinalEnum {
  val HiddenResident, NarrowWork, WideWork = newElement()
}

case class BitNetStreamingActivationLayout(cfg: BitNetConfig) {
  import BitNetPhysicalActivationBuffer._

  val buffers: Vector[BitNetPhysicalActivationBuffer.E] =
    Vector(HiddenResident, NarrowWork, WideWork)

  val featureCapacity: Map[BitNetPhysicalActivationBuffer.E, Int] = Map(
    HiddenResident -> cfg.hiddenSize,
    // input/post/final norm output, O projection and Down projection alias.
    NarrowWork -> cfg.hiddenSize,
    // Q-only projection/context and fused Gate-Up -> ReLU2 output alias.
    WideWork -> scala.math.max(cfg.hiddenSize, cfg.ffnSize)
  )

  private def groupsFor(features: Int): Int =
    (features + cfg.spuCount - 1) / cfg.spuCount

  val groupCapacity: Map[BitNetPhysicalActivationBuffer.E, Int] =
    featureCapacity.map { case (buffer, features) => buffer -> groupsFor(features) }

  val groupBase: Map[BitNetPhysicalActivationBuffer.E, Int] = {
    var next = 0
    buffers.map { buffer =>
      val base = next
      next += groupCapacity(buffer)
      buffer -> base
    }.toMap
  }

  val groupsPerTokenLane: Int = buffers.map(groupCapacity).sum
  val groupAddressWidth: Int = scala.math.max(1, log2Up(groupsPerTokenLane))
  val groupIndexWidth: Int = scala.math.max(1, log2Up(groupCapacity.values.max))

  def features(buffer: BitNetPhysicalActivationBuffer.E): Int =
    featureCapacity(buffer)
  def groups(buffer: BitNetPhysicalActivationBuffer.E): Int =
    groupCapacity(buffer)
  def base(buffer: BitNetPhysicalActivationBuffer.E): Int = groupBase(buffer)

  val totalBytes: BigInt = BigInt(cfg.tokenParallelism) *
    groupsPerTokenLane * cfg.spuCount * 4

  /**
    * Static semantic-to-physical contract used by the fixed stage wiring.
    * `None` means the value is consumed as a stream and must not hit BRAM.
    */
  def physicalSlot(slot: BitNetTensorSlot.E):
      Option[BitNetPhysicalActivationBuffer.E] = slot match {
    case BitNetTensorSlot.Hidden => Some(HiddenResident)
    case BitNetTensorSlot.Normalized => Some(NarrowWork)
    case BitNetTensorSlot.Qkv => Some(WideWork) // Q only; K/V go to RoPE/KV.
    case BitNetTensorSlot.AttentionContext => Some(WideWork)
    case BitNetTensorSlot.Projection => Some(NarrowWork)
    case BitNetTensorSlot.FfnActivated => Some(WideWork)
    case BitNetTensorSlot.Down => Some(NarrowWork)
    case BitNetTensorSlot.GateUp => None // fused projection -> ReLU2 stream.
    case BitNetTensorSlot.LogitPartial => None // streaming local argmax.
    case BitNetTensorSlot.TokenResult => None // scalar result register.
  }

  /** Number of semantically valid features when a logical slot owns a pool. */
  def semanticFeatures(slot: BitNetTensorSlot.E): Int = slot match {
    case BitNetTensorSlot.Hidden => cfg.hiddenSize
    case BitNetTensorSlot.Normalized => cfg.hiddenSize
    case BitNetTensorSlot.Qkv => cfg.hiddenSize
    case BitNetTensorSlot.AttentionContext => cfg.hiddenSize
    case BitNetTensorSlot.Projection => cfg.hiddenSize
    case BitNetTensorSlot.FfnActivated => cfg.ffnSize
    case BitNetTensorSlot.Down => cfg.hiddenSize
    case _ => 0
  }
}

case class BitNetPhysicalActivationReadCommand(
    cfg: BitNetConfig,
    layout: BitNetStreamingActivationLayout) extends Bundle {
  val buffer = BitNetPhysicalActivationBuffer()
  val group = UInt(layout.groupIndexWidth bits)
}

case class BitNetPhysicalActivationReadBeat(
    cfg: BitNetConfig,
    layout: BitNetStreamingActivationLayout) extends Bundle {
  val buffer = BitNetPhysicalActivationBuffer()
  val group = UInt(layout.groupIndexWidth bits)
  val values = Vec(SInt(32 bits), cfg.spuCount)
  val featureValid = Bits(cfg.spuCount bits)
  val fault = Bool()
}

case class BitNetPhysicalActivationWriteBeat(
    cfg: BitNetConfig,
    layout: BitNetStreamingActivationLayout) extends Bundle {
  val buffer = BitNetPhysicalActivationBuffer()
  val group = UInt(layout.groupIndexWidth bits)
  val values = Vec(SInt(32 bits), cfg.spuCount)
  val featureValid = Bits(cfg.spuCount bits)
}

/**
  * Production PL-resident activation fabric.
  *
  * Each token lane owns `spuCount` simple-dual-port BRAM stripes.  TP4 prefill
  * therefore performs four independent vector accesses per core cycle;
  * decode uses the same stripes as four feature shards.  Only three live
  * physical buffers are allocated.  The fixed stage coordinator is
  * responsible for changing their semantic owner at the barriers documented
  * by [[BitNetStreamingActivationLayout.physicalSlot]].
  */
class BitNetStreamingActivationFabric(
    cfg: BitNetConfig = BitNetConfig.Production,
    layout: BitNetStreamingActivationLayout =
      BitNetStreamingActivationLayout(BitNetConfig.Production)
) extends Component {
  require(layout.cfg == cfg, "streaming activation layout/config mismatch")

  val io = new Bundle {
    val readCommand = Vec(
      slave(Stream(BitNetPhysicalActivationReadCommand(cfg, layout))),
      cfg.tokenParallelism)
    val readData = Vec(
      master(Stream(BitNetPhysicalActivationReadBeat(cfg, layout))),
      cfg.tokenParallelism)
    val writeData = Vec(
      slave(Stream(BitNetPhysicalActivationWriteBeat(cfg, layout))),
      cfg.tokenParallelism)
    val readFaultSticky = out Bits(cfg.tokenParallelism bits)
    val writeFaultSticky = out Bits(cfg.tokenParallelism bits)
    /** Same-address read/write is illegal; in-place operators pipeline them. */
    val collisionFaultSticky = out Bits(cfg.tokenParallelism bits)
  }

  private val bufferCount = BitNetPhysicalActivationBuffer.elements.size
  private val maxFeatures = layout.featureCapacity.values.max
  private val bufferBase = Vec(UInt(layout.groupAddressWidth bits), bufferCount)
  private val bufferGroups =
    Vec(UInt((layout.groupIndexWidth + 1) bits), bufferCount)
  private val bufferFeatures =
    Vec(UInt(log2Up(maxFeatures + 1) bits), bufferCount)

  BitNetPhysicalActivationBuffer.elements.zipWithIndex.foreach {
    case (buffer, encoding) =>
      bufferBase(encoding) := U(layout.base(buffer), layout.groupAddressWidth bits)
      bufferGroups(encoding) :=
        U(layout.groups(buffer), (layout.groupIndexWidth + 1) bits)
      bufferFeatures(encoding) :=
        U(layout.features(buffer), bufferFeatures(encoding).getWidth bits)
  }

  val readFault = Reg(Bits(cfg.tokenParallelism bits)) init 0
  val writeFault = Reg(Bits(cfg.tokenParallelism bits)) init 0
  val collisionFault = Reg(Bits(cfg.tokenParallelism bits)) init 0
  io.readFaultSticky := readFault
  io.writeFaultSticky := writeFault
  io.collisionFaultSticky := collisionFault

  for (token <- 0 until cfg.tokenParallelism) {
    val memories = Array.fill(cfg.spuCount)(
      Mem(SInt(32 bits), layout.groupsPerTokenLane)
        .addAttribute("ram_style", "block")
    )

    val readIndex = io.readCommand(token).payload.buffer.asBits.asUInt
    val readBase = bufferBase(readIndex)
    val readGroups = bufferGroups(readIndex)
    val readFeatures = bufferFeatures(readIndex)
    val readInRange =
      io.readCommand(token).payload.group.resize(readGroups.getWidth) < readGroups
    val readAddress = (readBase.resize(layout.groupAddressWidth + 1) +
      io.readCommand(token).payload.group.resize(layout.groupAddressWidth + 1))
      .resize(layout.groupAddressWidth)

    val responseValid = Reg(Bool()) init False
    val responseBuffer = Reg(BitNetPhysicalActivationBuffer())
    val responseGroup = Reg(UInt(layout.groupIndexWidth bits)) init 0
    val responseFault = Reg(Bool()) init False
    val responseFeatureValid = Reg(Bits(cfg.spuCount bits)) init 0

    val canIssueRead = !responseValid || io.readData(token).ready
    io.readCommand(token).ready := canIssueRead
    val readFire = io.readCommand(token).valid && canIssueRead
    val readValues = memories.map(_.readSync(
      address = readAddress,
      enable = readFire && readInRange
    ))

    when(canIssueRead) {
      responseValid := io.readCommand(token).valid
      when(io.readCommand(token).valid) {
        responseBuffer := io.readCommand(token).payload.buffer
        responseGroup := io.readCommand(token).payload.group
        responseFault := !readInRange
        for (shard <- 0 until cfg.spuCount) {
          val feature =
            io.readCommand(token).payload.group.resize(readFeatures.getWidth) *
              cfg.spuCount + shard
          responseFeatureValid(shard) := readInRange && feature < readFeatures
        }
        when(!readInRange) { readFault(token) := True }
      }
    }

    io.readData(token).valid := responseValid
    io.readData(token).payload.buffer := responseBuffer
    io.readData(token).payload.group := responseGroup
    io.readData(token).payload.featureValid := responseFeatureValid
    io.readData(token).payload.fault := responseFault
    for (shard <- 0 until cfg.spuCount) {
      io.readData(token).payload.values(shard) := Mux(
        responseFault || !responseFeatureValid(shard),
        S(0, 32 bits),
        readValues(shard)
      )
    }

    val writeIndex = io.writeData(token).payload.buffer.asBits.asUInt
    val writeBase = bufferBase(writeIndex)
    val writeGroups = bufferGroups(writeIndex)
    val writeInRange =
      io.writeData(token).payload.group.resize(writeGroups.getWidth) < writeGroups
    val writeAddress = (writeBase.resize(layout.groupAddressWidth + 1) +
      io.writeData(token).payload.group.resize(layout.groupAddressWidth + 1))
      .resize(layout.groupAddressWidth)
    io.writeData(token).ready := True

    when(io.writeData(token).fire) {
      when(writeInRange) {
        for (shard <- 0 until cfg.spuCount) {
          when(io.writeData(token).payload.featureValid(shard)) {
            memories(shard).write(
              address = writeAddress,
              data = io.writeData(token).payload.values(shard)
            )
          }
        }
      } otherwise {
        writeFault(token) := True
      }
    }

    when(readFire && readInRange && io.writeData(token).fire && writeInRange &&
      readAddress === writeAddress) {
      collisionFault(token) := True
    }
  }
}

object GenerateBitNetStreamingActivationFabric extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-streaming-activation")
  val cfg = BitNetConfig.Production
  SpinalConfig(
    mode = Verilog,
    targetDirectory = target,
    oneFilePerComponent = false
  ).generate(new BitNetStreamingActivationFabric(
    cfg,
    BitNetStreamingActivationLayout(cfg)
  ))
}
