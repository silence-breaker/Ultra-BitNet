package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Reference layout that assigns every semantic tensor a distinct region.
  *
  * Values are Q16.16 i32 until a matrix-input quantizer explicitly converts
  * them to i8.  The memories are striped by feature across the physical SPUs
  * and replicated by token lane.  Consequently prefill can access four tokens
  * concurrently, while decode can access four feature shards of one token in
  * the same cycle.  This layout is useful for isolation tests, but its
  * production shape consumes 160 RAMB36 + 16 RAMB18 on XCZU3EG.  The native
  * top therefore uses [[BitNetStreamingActivationFabric]], which overlays
  * non-overlapping lifetimes and streams Gate/Up and logits directly.
  */
case class BitNetActivationLayout(cfg: BitNetConfig) {
  val slots: Vector[BitNetTensorSlot.E] = Vector(
    BitNetTensorSlot.Hidden,
    BitNetTensorSlot.Normalized,
    BitNetTensorSlot.Qkv,
    BitNetTensorSlot.AttentionContext,
    BitNetTensorSlot.Projection,
    BitNetTensorSlot.GateUp,
    BitNetTensorSlot.FfnActivated,
    BitNetTensorSlot.Down,
    BitNetTensorSlot.LogitPartial,
    BitNetTensorSlot.TokenResult
  )

  private val qkvFeatures = cfg.hiddenSize + 2 * cfg.kvHeadCount * cfg.headSize
  private val slotLengths: Map[BitNetTensorSlot.E, Int] = Map(
    BitNetTensorSlot.Hidden -> cfg.hiddenSize,
    BitNetTensorSlot.Normalized -> cfg.hiddenSize,
    BitNetTensorSlot.Qkv -> qkvFeatures,
    BitNetTensorSlot.AttentionContext -> cfg.hiddenSize,
    BitNetTensorSlot.Projection -> cfg.hiddenSize,
    BitNetTensorSlot.GateUp -> (2 * cfg.ffnSize),
    BitNetTensorSlot.FfnActivated -> cfg.ffnSize,
    BitNetTensorSlot.Down -> cfg.hiddenSize,
    // Only one local candidate per matrix engine is retained.  The full
    // vocabulary logits are reduced as they stream out of the LM engine.
    BitNetTensorSlot.LogitPartial -> cfg.engineCount,
    BitNetTensorSlot.TokenResult -> 1
  )

  require(slots.distinct.size == slots.size)
  require(slots.forall(slotLengths.contains))

  private def groupsFor(length: Int): Int =
    (length + cfg.spuCount - 1) / cfg.spuCount

  val groupBases: Map[BitNetTensorSlot.E, Int] = {
    var next = 0
    slots.map { slot =>
      val base = next
      next += groupsFor(slotLengths(slot))
      slot -> base
    }.toMap
  }

  val groupsPerTokenLane: Int = slots.map(slot => groupsFor(slotLengths(slot))).sum
  val groupAddressWidth: Int = scala.math.max(1, log2Up(groupsPerTokenLane))
  val groupIndexWidth: Int = scala.math.max(
    1,
    log2Up(slots.map(slot => groupsFor(slotLengths(slot))).max)
  )

  def length(slot: BitNetTensorSlot.E): Int = slotLengths(slot)
  def groups(slot: BitNetTensorSlot.E): Int = groupsFor(length(slot))
  def base(slot: BitNetTensorSlot.E): Int = groupBases(slot)

  def totalBytes: BigInt =
    BigInt(cfg.tokenParallelism) * groupsPerTokenLane * cfg.spuCount * 4
}

case class BitNetActivationReadCommand(
    cfg: BitNetConfig,
    layout: BitNetActivationLayout) extends Bundle {
  val slot = BitNetTensorSlot()
  /** Index of a physical-SPU-wide feature group within the selected slot. */
  val group = UInt(layout.groupIndexWidth bits)
}

case class BitNetActivationReadBeat(
    cfg: BitNetConfig,
    layout: BitNetActivationLayout) extends Bundle {
  val slot = BitNetTensorSlot()
  val group = UInt(layout.groupIndexWidth bits)
  val values = Vec(SInt(32 bits), cfg.spuCount)
  /** Tail groups can contain fewer than spuCount semantic features. */
  val featureValid = Bits(cfg.spuCount bits)
  val fault = Bool()
}

case class BitNetActivationWriteBeat(
    cfg: BitNetConfig,
    layout: BitNetActivationLayout) extends Bundle {
  val slot = BitNetTensorSlot()
  val group = UInt(layout.groupIndexWidth bits)
  val values = Vec(SInt(32 bits), cfg.spuCount)
  val featureValid = Bits(cfg.spuCount bits)
}

/**
  * Full-slot activation scratchpad kept as a numerical/integration reference.
  *
  * There are tokenParallelism × spuCount independent simple-dual-port RAMs.
  * Each token lane owns one read command and one write stream, so TP4 prefill
  * can sustain 16 Q16.16 features per core clock.  During decode the four
  * values from lane zero naturally feed the four feature-sharded SPUs and are
  * broadcast once to both output-sharded matrix engines.  Do not instantiate
  * this class in a production top; use BitNetStreamingActivationFabric.
  */
class BitNetActivationFabric(
    cfg: BitNetConfig = BitNetConfig.Production,
    layout: BitNetActivationLayout = BitNetActivationLayout(BitNetConfig.Production)
) extends Component {
  require(layout.cfg == cfg, "activation layout/configuration mismatch")

  val io = new Bundle {
    val readCommand = Vec(
      slave(Stream(BitNetActivationReadCommand(cfg, layout))),
      cfg.tokenParallelism
    )
    val readData = Vec(
      master(Stream(BitNetActivationReadBeat(cfg, layout))),
      cfg.tokenParallelism
    )
    val writeData = Vec(
      slave(Stream(BitNetActivationWriteBeat(cfg, layout))),
      cfg.tokenParallelism
    )
    val readFaultSticky = out Bits(cfg.tokenParallelism bits)
    val writeFaultSticky = out Bits(cfg.tokenParallelism bits)
  }

  private val slotCount = BitNetTensorSlot.elements.size
  private val slotBase = Vec(UInt(layout.groupAddressWidth bits), slotCount)
  private val slotGroups = Vec(UInt((layout.groupIndexWidth + 1) bits), slotCount)
  private val slotLength = Vec(UInt(log2Up(2 * cfg.ffnSize + 1) bits), slotCount)
  BitNetTensorSlot.elements.zipWithIndex.foreach { case (slot, encoding) =>
    require(layout.slots.contains(slot), s"missing activation slot $slot")
    slotBase(encoding) := U(layout.base(slot), layout.groupAddressWidth bits)
    slotGroups(encoding) := U(layout.groups(slot), (layout.groupIndexWidth + 1) bits)
    slotLength(encoding) := U(
      layout.length(slot),
      slotLength(encoding).getWidth bits
    )
  }

  val readFault = Reg(Bits(cfg.tokenParallelism bits)) init 0
  val writeFault = Reg(Bits(cfg.tokenParallelism bits)) init 0
  io.readFaultSticky := readFault
  io.writeFaultSticky := writeFault

  for (token <- 0 until cfg.tokenParallelism) {
    val memories = Array.fill(cfg.spuCount)(
      Mem(SInt(32 bits), layout.groupsPerTokenLane)
        .addAttribute("ram_style", "block")
    )

    val readSlotIndex = io.readCommand(token).payload.slot.asBits.asUInt
    val readBase = slotBase(readSlotIndex)
    val readGroups = slotGroups(readSlotIndex)
    val readInRange = io.readCommand(token).payload.group.resize(readGroups.getWidth) < readGroups
    val readAddress = (readBase.resize(layout.groupAddressWidth + 1) +
      io.readCommand(token).payload.group.resize(layout.groupAddressWidth + 1))
      .resize(layout.groupAddressWidth)

    // One-entry elastic response register.  It lets synchronous BRAM issue a
    // new read on the same cycle in which the previous response is consumed.
    val responseValid = Reg(Bool()) init False
    val responseSlot = Reg(BitNetTensorSlot())
    val responseGroup = Reg(UInt(layout.groupIndexWidth bits)) init 0
    val responseFault = Reg(Bool()) init False
    val responseFeatureValid = Reg(Bits(cfg.spuCount bits)) init 0

    val canIssueRead = !responseValid || io.readData(token).ready
    io.readCommand(token).ready := canIssueRead
    val readFire = io.readCommand(token).valid && canIssueRead

    val readValues = memories.map(_.readSync(readAddress, readFire && readInRange))
    when(canIssueRead) {
      responseValid := io.readCommand(token).valid
      when(io.readCommand(token).valid) {
        responseSlot := io.readCommand(token).payload.slot
        responseGroup := io.readCommand(token).payload.group
        responseFault := !readInRange
        for (shard <- 0 until cfg.spuCount) {
          val feature = io.readCommand(token).payload.group.resize(slotLength(readSlotIndex).getWidth) *
            cfg.spuCount + shard
          responseFeatureValid(shard) := readInRange && feature < slotLength(readSlotIndex)
        }
        when(!readInRange) {
          readFault(token) := True
        }
      }
    }

    io.readData(token).valid := responseValid
    io.readData(token).payload.slot := responseSlot
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

    val writeSlotIndex = io.writeData(token).payload.slot.asBits.asUInt
    val writeBase = slotBase(writeSlotIndex)
    val writeGroups = slotGroups(writeSlotIndex)
    val writeInRange = io.writeData(token).payload.group.resize(writeGroups.getWidth) < writeGroups
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
  }
}
