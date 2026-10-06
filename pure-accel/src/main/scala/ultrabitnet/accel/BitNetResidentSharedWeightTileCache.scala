package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Physical geometry and lease identity of the shared resident weight cache. */
case class BitNetResidentSharedWeightTileCacheConfig(
    dataWidth: Int = 128,
    poolCount: Int = BitNetConfig.Production.engineCount,
    memoryCount: Int = 8,
    depth: Int = 108,
    sequenceIdWidth: Int = 12,
    sourceEpochWidth: Int = 8,
    generationWidth: Int = 16
) {
  require(dataWidth == 128)
  require(poolCount == 1 || poolCount == 2)
  require(memoryCount == 4 || memoryCount == 8)
  require(depth == 108 || depth == 216)
  require(sequenceIdWidth >= 8)
  require(sourceEpochWidth >= 4)
  require(generationWidth >= 8)

  val addressWidth: Int = log2Up(depth)
}

case class BitNetResidentWeightTileCacheAcquire(
    cfg: BitNetResidentSharedWeightTileCacheConfig) extends Bundle {
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val sourceEpoch = UInt(cfg.sourceEpochWidth bits)
}

case class BitNetResidentWeightTileCacheGrant(
    cfg: BitNetResidentSharedWeightTileCacheConfig) extends Bundle {
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val sourceEpoch = UInt(cfg.sourceEpochWidth bits)
  val generation = UInt(cfg.generationWidth bits)
}

case class BitNetResidentWeightTileCacheRelease(
    cfg: BitNetResidentSharedWeightTileCacheConfig) extends Bundle {
  val sequenceId = UInt(cfg.sequenceIdWidth bits)
  val sourceEpoch = UInt(cfg.sourceEpochWidth bits)
  val generation = UInt(cfg.generationWidth bits)
}

case class BitNetResidentWeightTileCacheWrite(
    cfg: BitNetResidentSharedWeightTileCacheConfig) extends Bundle {
  val generation = UInt(cfg.generationWidth bits)
  val address = UInt(cfg.addressWidth bits)
  val data = Bits(cfg.dataWidth bits)
}

case class BitNetResidentWeightTileCacheRead(
    cfg: BitNetResidentSharedWeightTileCacheConfig) extends Bundle {
  val generation = UInt(cfg.generationWidth bits)
  val address = UInt(cfg.addressWidth bits)
  val poolMask = Bits(cfg.poolCount bits)
}

object BitNetResidentWeightTileCacheOwner
    extends SpinalEnum(binarySequential) {
  val Matrix, LmHead = newElement()
}

object BitNetResidentWeightTileCacheState
    extends SpinalEnum(binarySequential) {
  val Idle, GrantMatrix, GrantLm, Owned, Quarantined = newElement()
}

/**
  * One physical two-pool/eight-memory resident weight-tile cache.
  *
  * Matrix and LM clients acquire a latched lease.  Every cache operation then
  * carries the private generation returned by the grant.  A dual acquire,
  * non-owner access, stale generation, invalid address or mismatched release
  * permanently quarantines the cache and suppresses all subsequent writes.
  * Front-end AXI tag/epoch checks remain in the two fetchers; the generation
  * closes the internal stale-operation hole after an epoch wraps or is reused.
  *
  * Address maps:
  *   - Matrix: `slot * 54 + kTile`, range 0..107.
  *   - LM:     `slot * 20 + kTile`, range 0..39, pool == engine.
  */
class BitNetResidentSharedWeightTileCache(
    cfg: BitNetResidentSharedWeightTileCacheConfig =
      BitNetResidentSharedWeightTileCacheConfig()) extends Component {
  val io = new Bundle {
    val matrixAcquire = slave(Stream(
      BitNetResidentWeightTileCacheAcquire(cfg)))
    val matrixGrant = master(Stream(
      BitNetResidentWeightTileCacheGrant(cfg)))
    val matrixRelease = slave(Stream(
      BitNetResidentWeightTileCacheRelease(cfg)))

    val lmAcquire = slave(Stream(BitNetResidentWeightTileCacheAcquire(cfg)))
    val lmGrant = master(Stream(BitNetResidentWeightTileCacheGrant(cfg)))
    val lmRelease = slave(Stream(BitNetResidentWeightTileCacheRelease(cfg)))

    val matrixWrite = Vec.fill(cfg.poolCount)(Vec.fill(cfg.memoryCount)(
      slave(Stream(BitNetResidentWeightTileCacheWrite(cfg)))))
    val matrixRead = slave(Stream(BitNetResidentWeightTileCacheRead(cfg)))
    val matrixReadData = Vec.fill(cfg.poolCount)(Vec.fill(cfg.memoryCount)(
      out Bits(cfg.dataWidth bits)))

    val lmWrite = Vec.fill(cfg.poolCount)(Vec.fill(cfg.memoryCount)(
      slave(Stream(BitNetResidentWeightTileCacheWrite(cfg)))))
    val lmRead = slave(Stream(BitNetResidentWeightTileCacheRead(cfg)))
    val lmReadData = Vec.fill(cfg.poolCount)(Vec.fill(cfg.memoryCount)(
      out Bits(cfg.dataWidth bits)))

    val ownerValid = out Bool()
    val owner = out(BitNetResidentWeightTileCacheOwner())
    val activeSequenceId = out UInt(cfg.sequenceIdWidth bits)
    val activeSourceEpoch = out UInt(cfg.sourceEpochWidth bits)
    val activeGeneration = out UInt(cfg.generationWidth bits)
    val readPipelineValid = out Bool()
    val protocolFaultSticky = out Bool()
    val quarantined = out Bool()
    val acceptedWrites = out UInt(32 bits)
    val acceptedReads = out UInt(32 bits)
    val grantedLeases = out UInt(32 bits)
    val releasedLeases = out UInt(32 bits)
  }

  val memories = Array.tabulate(cfg.poolCount, cfg.memoryCount) {
    (pool, memory) =>
      val mem = Mem(Bits(cfg.dataWidth bits), cfg.depth)
      mem.setName(s"residentWeightTilePool${pool}Memory${memory}")
      mem.addAttribute("ram_style", "block")
      mem
  }

  val state = Reg(BitNetResidentWeightTileCacheState()) init
    BitNetResidentWeightTileCacheState.Idle
  val ownerReg = Reg(BitNetResidentWeightTileCacheOwner()) init
    BitNetResidentWeightTileCacheOwner.Matrix
  val ownerValidReg = RegInit(False)
  val sequenceReg = Reg(UInt(cfg.sequenceIdWidth bits)) init 0
  val sourceEpochReg = Reg(UInt(cfg.sourceEpochWidth bits)) init 0
  val generationReg = Reg(UInt(cfg.generationWidth bits)) init 0
  val nextGeneration = Reg(UInt(cfg.generationWidth bits)) init 1
  val readPipelineReg = RegInit(False)
  val stickyFault = RegInit(False)
  val quarantineReg = RegInit(False)

  val acceptedWriteCount = Reg(UInt(32 bits)) init 0
  val acceptedReadCount = Reg(UInt(32 bits)) init 0
  val grantedLeaseCount = Reg(UInt(32 bits)) init 0
  val releasedLeaseCount = Reg(UInt(32 bits)) init 0

  io.ownerValid := ownerValidReg
  io.owner := ownerReg
  io.activeSequenceId := sequenceReg
  io.activeSourceEpoch := sourceEpochReg
  io.activeGeneration := generationReg
  io.readPipelineValid := readPipelineReg
  io.protocolFaultSticky := stickyFault
  io.quarantined := quarantineReg
  io.acceptedWrites := acceptedWriteCount
  io.acceptedReads := acceptedReadCount
  io.grantedLeases := grantedLeaseCount
  io.releasedLeases := releasedLeaseCount

  val idle = state === BitNetResidentWeightTileCacheState.Idle
  // Acquires are ordinary backpressured Streams.  Matrix has fixed priority
  // only for the cycle in which both clients first request an idle cache; the
  // LM request remains asserted and is granted after Matrix releases.  A
  // waiting request is not a protocol violation.
  io.matrixAcquire.ready := idle
  io.lmAcquire.ready := idle && !io.matrixAcquire.valid

  io.matrixGrant.valid :=
    state === BitNetResidentWeightTileCacheState.GrantMatrix
  io.lmGrant.valid := state === BitNetResidentWeightTileCacheState.GrantLm
  io.matrixGrant.sequenceId := sequenceReg
  io.matrixGrant.sourceEpoch := sourceEpochReg
  io.matrixGrant.generation := generationReg
  io.lmGrant.payload := io.matrixGrant.payload

  val violation = Bool()
  violation := False

  when(io.matrixAcquire.fire || io.lmAcquire.fire) {
    sequenceReg := Mux(io.matrixAcquire.fire,
      io.matrixAcquire.sequenceId, io.lmAcquire.sequenceId)
    sourceEpochReg := Mux(io.matrixAcquire.fire,
      io.matrixAcquire.sourceEpoch, io.lmAcquire.sourceEpoch)
    generationReg := nextGeneration
    when(nextGeneration.andR) {
      nextGeneration := 1
    } otherwise {
      nextGeneration := nextGeneration + 1
    }
    when(io.matrixAcquire.fire) {
      ownerReg := BitNetResidentWeightTileCacheOwner.Matrix
      state := BitNetResidentWeightTileCacheState.GrantMatrix
    } otherwise {
      ownerReg := BitNetResidentWeightTileCacheOwner.LmHead
      state := BitNetResidentWeightTileCacheState.GrantLm
    }
  }

  when(io.matrixGrant.fire || io.lmGrant.fire) {
    ownerValidReg := True
    state := BitNetResidentWeightTileCacheState.Owned
    grantedLeaseCount := grantedLeaseCount + 1
  }

  val matrixOwned = state === BitNetResidentWeightTileCacheState.Owned &&
    ownerValidReg && ownerReg === BitNetResidentWeightTileCacheOwner.Matrix
  val lmOwned = state === BitNetResidentWeightTileCacheState.Owned &&
    ownerValidReg && ownerReg === BitNetResidentWeightTileCacheOwner.LmHead

  val matrixWriteValidBits = Bits(cfg.poolCount * cfg.memoryCount bits)
  val lmWriteValidBits = Bits(cfg.poolCount * cfg.memoryCount bits)
  val acceptedWriteBits = Bits(cfg.poolCount * cfg.memoryCount bits)

  for (pool <- 0 until cfg.poolCount; memory <- 0 until cfg.memoryCount) {
    val ordinal = pool * cfg.memoryCount + memory
    matrixWriteValidBits(ordinal) := io.matrixWrite(pool)(memory).valid
    lmWriteValidBits(ordinal) := io.lmWrite(pool)(memory).valid

    val matrixLegal = matrixOwned &&
      io.matrixWrite(pool)(memory).generation === generationReg &&
      io.matrixWrite(pool)(memory).address < cfg.depth
    val lmLegal = lmOwned &&
      io.lmWrite(pool)(memory).generation === generationReg &&
      io.lmWrite(pool)(memory).address < cfg.depth
    io.matrixWrite(pool)(memory).ready := matrixLegal && !quarantineReg
    io.lmWrite(pool)(memory).ready := lmLegal && !quarantineReg

    when(io.matrixWrite(pool)(memory).valid && !matrixLegal) {
      violation := True
    }
    when(io.lmWrite(pool)(memory).valid && !lmLegal) {
      violation := True
    }

    val selectedAddress = UInt(cfg.addressWidth bits)
    val selectedData = Bits(cfg.dataWidth bits)
    val selectedEnable = Bool()
    selectedAddress := io.matrixWrite(pool)(memory).address
    selectedData := io.matrixWrite(pool)(memory).data
    selectedEnable := io.matrixWrite(pool)(memory).fire
    when(lmOwned) {
      selectedAddress := io.lmWrite(pool)(memory).address
      selectedData := io.lmWrite(pool)(memory).data
      selectedEnable := io.lmWrite(pool)(memory).fire
    }
    memories(pool)(memory).write(
      address = selectedAddress,
      data = selectedData,
      enable = selectedEnable && !quarantineReg)
    acceptedWriteBits(ordinal) := selectedEnable && !quarantineReg
  }

  when(acceptedWriteBits.orR) {
    acceptedWriteCount := acceptedWriteCount +
      CountOne(acceptedWriteBits).resize(32)
  }

  val matrixReadLegal = matrixOwned &&
    io.matrixRead.generation === generationReg &&
    io.matrixRead.address < cfg.depth && io.matrixRead.poolMask.orR
  val lmReadLegal = lmOwned &&
    io.lmRead.generation === generationReg &&
    io.lmRead.address < cfg.depth && io.lmRead.poolMask.orR
  io.matrixRead.ready := matrixReadLegal && !quarantineReg
  io.lmRead.ready := lmReadLegal && !quarantineReg
  when(io.matrixRead.valid && !matrixReadLegal) { violation := True }
  when(io.lmRead.valid && !lmReadLegal) { violation := True }

  val selectedReadAddress = UInt(cfg.addressWidth bits)
  val selectedReadMask = Bits(cfg.poolCount bits)
  val selectedReadFire = Bool()
  selectedReadAddress := io.matrixRead.address
  selectedReadMask := io.matrixRead.poolMask
  selectedReadFire := io.matrixRead.fire
  when(lmOwned) {
    selectedReadAddress := io.lmRead.address
    selectedReadMask := io.lmRead.poolMask
    selectedReadFire := io.lmRead.fire
  }
  readPipelineReg := selectedReadFire
  when(selectedReadFire) {
    acceptedReadCount := acceptedReadCount + 1
  }

  for (pool <- 0 until cfg.poolCount; memory <- 0 until cfg.memoryCount) {
    val data = memories(pool)(memory).readSync(
      address = selectedReadAddress,
      enable = selectedReadFire && selectedReadMask(pool) && !quarantineReg)
    io.matrixReadData(pool)(memory) := data
    io.lmReadData(pool)(memory) := data
  }

  val anyMatrixAccess = io.matrixRead.valid || matrixWriteValidBits.orR
  val anyLmAccess = io.lmRead.valid || lmWriteValidBits.orR
  val anyAccess = anyMatrixAccess || anyLmAccess

  val matrixReleaseIdentity = matrixOwned &&
    io.matrixRelease.sequenceId === sequenceReg &&
    io.matrixRelease.sourceEpoch === sourceEpochReg &&
    io.matrixRelease.generation === generationReg
  val lmReleaseIdentity = lmOwned &&
    io.lmRelease.sequenceId === sequenceReg &&
    io.lmRelease.sourceEpoch === sourceEpochReg &&
    io.lmRelease.generation === generationReg
  io.matrixRelease.ready := matrixReleaseIdentity &&
    !readPipelineReg && !anyAccess && !quarantineReg
  io.lmRelease.ready := lmReleaseIdentity &&
    !readPipelineReg && !anyAccess && !quarantineReg

  // A correctly identified release may be held valid while the synchronous
  // read stage or the owner's last access drains.  Only a non-owner/stale
  // identity is fatal; temporary unavailability is normal backpressure.
  when(io.matrixRelease.valid && !matrixReleaseIdentity) {
    violation := True
  }
  when(io.lmRelease.valid && !lmReleaseIdentity) {
    violation := True
  }
  when(io.matrixRelease.fire || io.lmRelease.fire) {
    ownerValidReg := False
    state := BitNetResidentWeightTileCacheState.Idle
    releasedLeaseCount := releasedLeaseCount + 1
  }

  when(violation) {
    stickyFault := True
    quarantineReg := True
    ownerValidReg := False
    readPipelineReg := False
    state := BitNetResidentWeightTileCacheState.Quarantined
  }
}
