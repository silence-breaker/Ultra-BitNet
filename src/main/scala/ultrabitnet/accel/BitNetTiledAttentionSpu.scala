package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/**
  * Physical contract for the BitNet tiled online-softmax SPU.
  *
  * The production shape is one KV head shared by four query heads, 16 keys per
  * tile, head dimension 128 and at most 4096 cached keys.  QK scores enter as
  * signed Q4.12; alpha and probability weights leave as unsigned Q1.15.  The
  * value vector is intentionally absent from this interface: the bank-local KV
  * coordinator feeds each emitted probability to the unified matrix array's
  * SV/AXPY mode.
  */
case class BitNetTiledAttentionConfig(
    queryHeadsPerKv: Int = 4,
    headDim: Int = 128,
    maxSeq: Int = 4096,
    tileKeys: Int = 16,
    scoreWidth: Int = 16,
    scoreFractionalBits: Int = 12,
    expWidth: Int = 16,
    expFractionalBits: Int = 15,
    sumWidth: Int = 40,
    probabilityFifoDepth: Int = 8
) {
  require(queryHeadsPerKv == 4,
    "BitNet GQA has exactly four query heads per KV head")
  require(headDim == 128,
    "the production BitNet head dimension is 128")
  require(maxSeq >= tileKeys && isPow2(maxSeq))
  require(tileKeys == 16,
    "the BitNet physical softmax tile is fixed at 16 keys")
  require(scoreWidth == 16 && scoreFractionalBits == 12)
  require(expWidth == 16 && expFractionalBits == 15)
  require(sumWidth >= log2Up(maxSeq) + expFractionalBits + 1)
  require(probabilityFifoDepth >= 4)

  val positionWidth: Int = log2Up(maxSeq)
  val keyCountWidth: Int = log2Up(maxSeq + 1)
  val tileOffsetWidth: Int = log2Up(tileKeys)
  val tileKeyCountWidth: Int = log2Up(tileKeys + 1)
  val tileCount: Int = (maxSeq + tileKeys - 1) / tileKeys
  val tileIndexWidth: Int = Math.max(1, log2Up(tileCount))
  val expOne: Int = 1 << expFractionalBits
}

case class BitNetAttentionCommand(cfg: BitNetTiledAttentionConfig)
    extends Bundle {
  val queryPosition = UInt(cfg.positionWidth bits)
  val queryMask = Bits(cfg.queryHeadsPerKv bits)
  /** Number of score beats, from one through maxSeq. */
  val keyCount = UInt(cfg.keyCountWidth bits)
  val causal = Bool()
}

case class BitNetAttentionScoreBeat(cfg: BitNetTiledAttentionConfig)
    extends Bundle {
  val score = Vec(SInt(cfg.scoreWidth bits), cfg.queryHeadsPerKv)
  val keyPosition = UInt(cfg.positionWidth bits)
  /** Duplicated command fields make a misrouted bank-local tile detectable. */
  val queryPosition = UInt(cfg.positionWidth bits)
  val queryMask = Bits(cfg.queryHeadsPerKv bits)
  val positionValid = Bool()
  val first = Bool()
  val last = Bool()
}

/** Must be consumed before the first probability of the same tile. */
case class BitNetAttentionTileUpdate(cfg: BitNetTiledAttentionConfig)
    extends Bundle {
  val alpha = Vec(UInt(cfg.expWidth bits), cfg.queryHeadsPerKv)
  val newMax = Vec(SInt(cfg.scoreWidth bits), cfg.queryHeadsPerKv)
  val queryMask = Bits(cfg.queryHeadsPerKv bits)
  val allMasked = Bits(cfg.queryHeadsPerKv bits)
  val tileIndex = UInt(cfg.tileIndexWidth bits)
  val keyCount = UInt(cfg.tileKeyCountWidth bits)
  val firstTile = Bool()
  val lastTile = Bool()
}

/** One four-head Q1.15 weight vector for the unified array's SV/AXPY mode. */
case class BitNetAttentionProbability(cfg: BitNetTiledAttentionConfig)
    extends Bundle {
  val weight = Vec(UInt(cfg.expWidth bits), cfg.queryHeadsPerKv)
  val keyPosition = UInt(cfg.positionWidth bits)
  val keyIndex = UInt(cfg.positionWidth bits)
  val queryMask = Bits(cfg.queryHeadsPerKv bits)
  val tileIndex = UInt(cfg.tileIndexWidth bits)
  val firstInTile = Bool()
  val lastInTile = Bool()
  val lastInSequence = Bool()
}

/**
  * Final normalization metadata.  Context division is deliberately outside
  * the SPU: the unified array/scaler consumes these denominators after all SV
  * AXPY beats.  This avoids a divider per context lane and any context-RAM RMW
  * combinational chain.
  */
case class BitNetAttentionNormalization(cfg: BitNetTiledAttentionConfig)
    extends Bundle {
  val denominator = Vec(UInt(cfg.sumWidth bits), cfg.queryHeadsPerKv)
  val finalMax = Vec(SInt(cfg.scoreWidth bits), cfg.queryHeadsPerKv)
  val queryMask = Bits(cfg.queryHeadsPerKv bits)
  val allMasked = Bits(cfg.queryHeadsPerKv bits)
}

/**
  * Deployable tile=16 online-softmax SPU.
  *
  * Two register tiles are ping-ponged: QK can fill the next tile while the
  * current tile is replayed.  Scores live only in these 2 x 16 local slots.
  * Four logical exp ports are supplied by two initialized true-dual-port block
  * ROMs.  Each port is synchronous, so no 129-way combinational LUT mux exists.
  * A response FIFO accounts for the ROM read in flight and preserves arbitrary
  * downstream backpressure.
  */
class BitNetTiledAttentionSpu(
    cfg: BitNetTiledAttentionConfig = BitNetTiledAttentionConfig()
) extends Component {
  val io = new Bundle {
    val command = slave(Stream(BitNetAttentionCommand(cfg)))
    val score = slave(Stream(BitNetAttentionScoreBeat(cfg)))
    val tileUpdate = master(Stream(BitNetAttentionTileUpdate(cfg)))
    val probability = master(Stream(BitNetAttentionProbability(cfg)))
    val normalization = master(Stream(BitNetAttentionNormalization(cfg)))

    val busy = out Bool()
    val protocolFault = out Bool()
    val acceptedScores = out UInt(32 bits)
    val emittedProbabilities = out UInt(32 bits)
    val completedCommands = out UInt(32 bits)
  }

  object State extends SpinalEnum {
    val WaitCommand, WaitTile, AlphaIssue, AlphaWait, EmitUpdate, Replay,
        ScaleOldSum, WaitDrain, EmitNormalization = newElement()
  }
  val state = Reg(State()) init State.WaitCommand

  val commandActive = RegInit(False)
  val activeQueryPosition = Reg(UInt(cfg.positionWidth bits)) init 0
  val activeQueryMask = Reg(Bits(cfg.queryHeadsPerKv bits)) init 0
  val activeKeyCount = Reg(UInt(cfg.keyCountWidth bits)) init 0
  val activeCausal = RegInit(True)

  val fault = RegInit(False)
  val acceptedCounter = Reg(UInt(32 bits)) init 0
  val probabilityCounter = Reg(UInt(32 bits)) init 0
  val completedCounter = Reg(UInt(32 bits)) init 0
  io.protocolFault := fault
  io.acceptedScores := acceptedCounter
  io.emittedProbabilities := probabilityCounter
  io.completedCommands := completedCounter
  io.busy := state =/= State.WaitCommand || commandActive

  /* Two physical tile buffers.  Registers are intentional at depth 16. */
  val scoreStorage = Array.tabulate(2, cfg.queryHeadsPerKv) { (_, _) =>
    Vec((0 until cfg.tileKeys).map(_ =>
      Reg(SInt(cfg.scoreWidth bits)) init 0))
  }
  val acceptedStorage = Array.tabulate(2) { _ =>
    Vec((0 until cfg.tileKeys).map(_ =>
      Reg(Bits(cfg.queryHeadsPerKv bits)) init 0))
  }
  val keyPositionStorage = Array.tabulate(2) { _ =>
    Vec((0 until cfg.tileKeys).map(_ =>
      Reg(UInt(cfg.positionWidth bits)) init 0))
  }
  val keyIndexStorage = Array.tabulate(2) { _ =>
    Vec((0 until cfg.tileKeys).map(_ =>
      Reg(UInt(cfg.positionWidth bits)) init 0))
  }

  val bankReady = Vec(Reg(Bool()) init False, 2)
  val bankProcessing = Vec(Reg(Bool()) init False, 2)
  val bankTileMax = Array.tabulate(cfg.queryHeadsPerKv) { _ =>
    Vec(Reg(SInt(cfg.scoreWidth bits)) init 0, 2)
  }
  val bankTileHas = Vec(Reg(Bits(cfg.queryHeadsPerKv bits)) init 0, 2)
  val bankKeyCount = Vec(
    Reg(UInt(cfg.tileKeyCountWidth bits)) init 0, 2)
  val bankTileIndex = Vec(Reg(UInt(cfg.tileIndexWidth bits)) init 0, 2)
  val bankLastTile = Vec(Reg(Bool()) init False, 2)

  /* Input-side ping-pong collector. */
  val collectBank = Reg(Bool()) init False
  val collectOffset = Reg(UInt(cfg.tileOffsetWidth bits)) init 0
  val collectedKeyIndex = Reg(UInt(cfg.positionWidth bits)) init 0
  val collectTileIndex = Reg(UInt(cfg.tileIndexWidth bits)) init 0
  val inputComplete = RegInit(False)
  val collectTileHas = Reg(Bits(cfg.queryHeadsPerKv bits)) init 0
  val collectTileMax = Vec(
    Reg(SInt(cfg.scoreWidth bits)) init 0, cfg.queryHeadsPerKv)

  io.command.ready := state === State.WaitCommand
  when(io.command.fire) {
    fault := False
    acceptedCounter := 0
    probabilityCounter := 0
    when(io.command.keyCount === 0 ||
        io.command.keyCount > U(cfg.maxSeq, cfg.keyCountWidth bits)) {
      fault := True
      commandActive := False
    } otherwise {
      commandActive := True
      activeQueryPosition := io.command.queryPosition
      activeQueryMask := io.command.queryMask
      activeKeyCount := io.command.keyCount
      activeCausal := io.command.causal
      collectBank := False
      collectOffset := 0
      collectedKeyIndex := 0
      collectTileIndex := 0
      inputComplete := False
      collectTileHas := 0
      bankReady(0) := False
      bankReady(1) := False
      bankProcessing(0) := False
      bankProcessing(1) := False
      state := State.WaitTile
    }
  }

  val collectorUnavailable = bankReady(collectBank.asUInt) ||
    bankProcessing(collectBank.asUInt)
  io.score.ready := commandActive && !inputComplete && !collectorUnavailable

  val acceptedNow = Bits(cfg.queryHeadsPerKv bits)
  val collectMaxNext = Vec(SInt(cfg.scoreWidth bits), cfg.queryHeadsPerKv)
  val collectHasNext = Bits(cfg.queryHeadsPerKv bits)
  for (query <- 0 until cfg.queryHeadsPerKv) {
    acceptedNow(query) := activeQueryMask(query) && io.score.positionValid &&
      (!activeCausal || io.score.keyPosition <= activeQueryPosition)
    collectHasNext(query) := collectTileHas(query) || acceptedNow(query)
    collectMaxNext(query) := collectTileMax(query)
    when(acceptedNow(query) &&
        (!collectTileHas(query) || io.score.score(query) > collectTileMax(query))) {
      collectMaxNext(query) := io.score.score(query)
    }
  }

  val expectedLastKey = collectedKeyIndex ===
    (activeKeyCount - 1).resize(cfg.positionWidth)
  val expectedTileEnd = collectOffset ===
    U(cfg.tileKeys - 1, cfg.tileOffsetWidth bits) || expectedLastKey

  when(io.score.fire) {
    acceptedCounter := acceptedCounter + 1

    when(io.score.queryPosition =/= activeQueryPosition ||
        io.score.queryMask =/= activeQueryMask ||
        io.score.first =/= (collectedKeyIndex === 0) ||
        io.score.last =/= expectedLastKey) {
      fault := True
    }

    when(collectBank) {
      for (query <- 0 until cfg.queryHeadsPerKv) {
        scoreStorage(1)(query)(collectOffset) := io.score.score(query)
      }
      acceptedStorage(1)(collectOffset) := acceptedNow
      keyPositionStorage(1)(collectOffset) := io.score.keyPosition
      keyIndexStorage(1)(collectOffset) := collectedKeyIndex
    } otherwise {
      for (query <- 0 until cfg.queryHeadsPerKv) {
        scoreStorage(0)(query)(collectOffset) := io.score.score(query)
      }
      acceptedStorage(0)(collectOffset) := acceptedNow
      keyPositionStorage(0)(collectOffset) := io.score.keyPosition
      keyIndexStorage(0)(collectOffset) := collectedKeyIndex
    }

    when(expectedTileEnd) {
      bankReady(collectBank.asUInt) := True
      bankTileHas(collectBank.asUInt) := collectHasNext
      bankKeyCount(collectBank.asUInt) :=
        (collectOffset.resize(cfg.tileKeyCountWidth) + 1).resized
      bankTileIndex(collectBank.asUInt) := collectTileIndex
      bankLastTile(collectBank.asUInt) := expectedLastKey
      for (query <- 0 until cfg.queryHeadsPerKv) {
        bankTileMax(query)(collectBank.asUInt) := collectMaxNext(query)
      }

      collectBank := !collectBank
      collectOffset := 0
      collectTileHas := 0
      when(expectedLastKey) {
        inputComplete := True
      } otherwise {
        collectedKeyIndex := collectedKeyIndex + 1
        collectTileIndex := collectTileIndex + 1
      }
    } otherwise {
      collectOffset := collectOffset + 1
      collectedKeyIndex := collectedKeyIndex + 1
      collectTileHas := collectHasNext
      for (query <- 0 until cfg.queryHeadsPerKv) {
        collectTileMax(query) := collectMaxNext(query)
      }
    }
  }

  /* Processor-side online state, independent for each of four query heads. */
  val processBank = Reg(Bool()) init False
  val globalHas = Reg(Bits(cfg.queryHeadsPerKv bits)) init 0
  val globalMax = Vec(
    Reg(SInt(cfg.scoreWidth bits)) init 0, cfg.queryHeadsPerKv)
  val globalSum = Vec(Reg(UInt(cfg.sumWidth bits)) init 0,
    cfg.queryHeadsPerKv)

  /* A command is a complete query/KV-head transaction; no state leaks. */
  when(io.command.fire && io.command.keyCount =/= 0 &&
      io.command.keyCount <= U(cfg.maxSeq, cfg.keyCountWidth bits)) {
    processBank := False
    globalHas := 0
    for (query <- 0 until cfg.queryHeadsPerKv) {
      globalMax(query) := 0
      globalSum(query) := 0
    }
  }

  val pendingNewHas = Reg(Bits(cfg.queryHeadsPerKv bits)) init 0
  val pendingNewMax = Vec(
    Reg(SInt(cfg.scoreWidth bits)) init 0, cfg.queryHeadsPerKv)
  val pendingAlphaZero = Reg(Bits(cfg.queryHeadsPerKv bits)) init 0
  val pendingAlphaMagnitude = Vec(
    Reg(UInt((cfg.scoreWidth + 1) bits)) init 0, cfg.queryHeadsPerKv)
  val pendingAlpha = Vec(
    Reg(UInt(cfg.expWidth bits)) init 0, cfg.queryHeadsPerKv)
  val pendingKeyCount = Reg(UInt(cfg.tileKeyCountWidth bits)) init 0
  val pendingTileIndex = Reg(UInt(cfg.tileIndexWidth bits)) init 0
  val pendingLastTile = RegInit(False)

  /*
   * Two true-dual-port block ROMs provide four independent synchronous exp
   * lookups.  The address is round(delta / (1/16)); index 0 is exactly one and
   * magnitudes below -8 are explicitly zeroed after the registered lookup.
   */
  private val expConstants = (0 to 128).map { index =>
    B(Math.round(Math.exp(-index.toDouble / 16.0) * cfg.expOne),
      cfg.expWidth bits)
  }
  val expRoms = Array.fill(2)(
    Mem(Bits(cfg.expWidth bits), initialContent = expConstants))
  expRoms.foreach(_.addAttribute("ram_style", "block"))

  def expAddress(magnitudeQ12: UInt): (UInt, Bool) = {
    val extended = magnitudeQ12.resize(magnitudeQ12.getWidth + 1)
    val step = ((extended + U(128, extended.getWidth bits)) |>> 8)
      .resize(9)
    val address = UInt(8 bits)
    address := step.resize(8)
    when(step > 128) { address := 0 }
    (address, step > 128)
  }

  def selectedScore(query: Int, index: UInt): SInt = {
    val value = SInt(cfg.scoreWidth bits)
    value := scoreStorage(0)(query)(index)
    when(processBank) { value := scoreStorage(1)(query)(index) }
    value
  }

  def selectedAccepted(index: UInt): Bits = {
    val value = Bits(cfg.queryHeadsPerKv bits)
    value := acceptedStorage(0)(index)
    when(processBank) { value := acceptedStorage(1)(index) }
    value
  }

  def selectedKeyPosition(index: UInt): UInt = {
    val value = UInt(cfg.positionWidth bits)
    value := keyPositionStorage(0)(index)
    when(processBank) { value := keyPositionStorage(1)(index) }
    value
  }

  def selectedKeyIndex(index: UInt): UInt = {
    val value = UInt(cfg.positionWidth bits)
    value := keyIndexStorage(0)(index)
    when(processBank) { value := keyIndexStorage(1)(index) }
    value
  }

  val replayIndex = Reg(UInt(cfg.tileOffsetWidth bits)) init 0
  val allReplayIssued = RegInit(False)
  val probabilityFifo = new StreamFifo(
    BitNetAttentionProbability(cfg), cfg.probabilityFifoDepth)

  val betaIssue = state === State.Replay && !allReplayIssued &&
    probabilityFifo.io.availability > 1
  val alphaIssue = state === State.AlphaIssue
  val expReadEnable = alphaIssue || betaIssue

  val replayAccepted = selectedAccepted(replayIndex)
  val replayScore = Vec(SInt(cfg.scoreWidth bits), cfg.queryHeadsPerKv)
  val expReadAddress = Vec(UInt(8 bits), cfg.queryHeadsPerKv)
  val expReadTail = Bits(cfg.queryHeadsPerKv bits)
  val expReadZero = Bits(cfg.queryHeadsPerKv bits)
  for (query <- 0 until cfg.queryHeadsPerKv) {
    replayScore(query) := selectedScore(query, replayIndex)
    val betaDifference = (
      pendingNewMax(query).resize(cfg.scoreWidth + 1) -
        replayScore(query).resize(cfg.scoreWidth + 1)
    ).asUInt
    val betaAddressAndTail = expAddress(betaDifference)
    val alphaAddressAndTail = expAddress(pendingAlphaMagnitude(query))

    expReadAddress(query) := betaAddressAndTail._1
    expReadTail(query) := betaAddressAndTail._2
    expReadZero(query) := !replayAccepted(query)
    when(alphaIssue) {
      expReadAddress(query) := alphaAddressAndTail._1
      expReadTail(query) := alphaAddressAndTail._2
      expReadZero(query) := pendingAlphaZero(query)
    }
  }

  val expReadData = Vec(Bits(cfg.expWidth bits), cfg.queryHeadsPerKv)
  expReadData(0) := expRoms(0).readSync(expReadAddress(0), expReadEnable)
  expReadData(1) := expRoms(0).readSync(expReadAddress(1), expReadEnable)
  expReadData(2) := expRoms(1).readSync(expReadAddress(2), expReadEnable)
  expReadData(3) := expRoms(1).readSync(expReadAddress(3), expReadEnable)

  val expResponseTail = RegNextWhen(expReadTail, expReadEnable) init 0
  val expResponseZero = RegNextWhen(expReadZero, expReadEnable) init 0
  val alphaResponseValid = RegNext(alphaIssue) init False
  val betaResponseValid = RegNext(betaIssue) init False

  val betaResponseKeyPosition =
    RegNextWhen(selectedKeyPosition(replayIndex), betaIssue) init 0
  val betaResponseKeyIndex =
    RegNextWhen(selectedKeyIndex(replayIndex), betaIssue) init 0
  val betaResponseFirst = RegNextWhen(replayIndex === 0, betaIssue) init False
  val betaResponseLast = RegNextWhen(
    replayIndex.resize(cfg.tileKeyCountWidth) === pendingKeyCount - 1,
    betaIssue) init False
  val betaResponseTileIndex =
    RegNextWhen(pendingTileIndex, betaIssue) init 0
  val betaResponseLastSequence = RegNextWhen(
    pendingLastTile &&
      replayIndex.resize(cfg.tileKeyCountWidth) === pendingKeyCount - 1,
    betaIssue) init False

  val betaResponseWeight = Vec(UInt(cfg.expWidth bits),
    cfg.queryHeadsPerKv)
  for (query <- 0 until cfg.queryHeadsPerKv) {
    betaResponseWeight(query) := expReadData(query).asUInt
    when(expResponseZero(query) || expResponseTail(query)) {
      betaResponseWeight(query) := 0
    }
  }

  probabilityFifo.io.push.valid := betaResponseValid
  probabilityFifo.io.push.keyPosition := betaResponseKeyPosition
  probabilityFifo.io.push.keyIndex := betaResponseKeyIndex
  probabilityFifo.io.push.queryMask := activeQueryMask
  probabilityFifo.io.push.tileIndex := betaResponseTileIndex
  probabilityFifo.io.push.firstInTile := betaResponseFirst
  probabilityFifo.io.push.lastInTile := betaResponseLast
  probabilityFifo.io.push.lastInSequence := betaResponseLastSequence
  for (query <- 0 until cfg.queryHeadsPerKv) {
    probabilityFifo.io.push.weight(query) := betaResponseWeight(query)
  }
  io.probability << probabilityFifo.io.pop

  when(betaIssue) {
    when(replayIndex.resize(cfg.tileKeyCountWidth) === pendingKeyCount - 1) {
      allReplayIssued := True
    } otherwise {
      replayIndex := replayIndex + 1
    }
  }

  val tileSum = Vec(Reg(UInt(cfg.sumWidth bits)) init 0,
    cfg.queryHeadsPerKv)
  val completedTileSum = Vec(Reg(UInt(cfg.sumWidth bits)) init 0,
    cfg.queryHeadsPerKv)
  val scaledOldProduct = Vec(
    Reg(UInt((cfg.sumWidth + cfg.expWidth) bits)) init 0,
    cfg.queryHeadsPerKv
  )
  when(betaResponseValid) {
    for (query <- 0 until cfg.queryHeadsPerKv) {
      val added = (tileSum(query).resize(cfg.sumWidth + 1) +
        betaResponseWeight(query).resize(cfg.sumWidth + 1))
      tileSum(query) := added.resize(cfg.sumWidth)
      when(betaResponseLast) {
        completedTileSum(query) := added.resize(cfg.sumWidth)
      }
    }
    when(betaResponseLast) { state := State.WaitDrain }
  }

  io.tileUpdate.valid := state === State.EmitUpdate
  io.tileUpdate.queryMask := activeQueryMask
  io.tileUpdate.allMasked := ~pendingNewHas
  io.tileUpdate.tileIndex := pendingTileIndex
  io.tileUpdate.keyCount := pendingKeyCount
  io.tileUpdate.firstTile := pendingTileIndex === 0
  io.tileUpdate.lastTile := pendingLastTile
  for (query <- 0 until cfg.queryHeadsPerKv) {
    io.tileUpdate.alpha(query) := pendingAlpha(query)
    io.tileUpdate.newMax(query) := pendingNewMax(query)
  }

  io.normalization.valid := state === State.EmitNormalization
  io.normalization.queryMask := activeQueryMask
  io.normalization.allMasked := ~globalHas
  for (query <- 0 until cfg.queryHeadsPerKv) {
    io.normalization.denominator(query) := globalSum(query)
    io.normalization.finalMax(query) := globalMax(query)
  }

  switch(state) {
    is(State.WaitCommand) {}

    is(State.WaitTile) {
      when(bankReady(processBank.asUInt)) {
        bankReady(processBank.asUInt) := False
        bankProcessing(processBank.asUInt) := True
        pendingKeyCount := bankKeyCount(processBank.asUInt)
        pendingTileIndex := bankTileIndex(processBank.asUInt)
        pendingLastTile := bankLastTile(processBank.asUInt)

        for (query <- 0 until cfg.queryHeadsPerKv) {
          val tileHas = bankTileHas(processBank.asUInt)(query)
          val tileMaximum = bankTileMax(query)(processBank.asUInt)
          pendingNewHas(query) := globalHas(query) || tileHas
          pendingNewMax(query) := globalMax(query)
          pendingAlphaZero(query) := !globalHas(query)
          pendingAlphaMagnitude(query) := 0

          when(!globalHas(query) && tileHas) {
            pendingNewMax(query) := tileMaximum
          } elsewhen (globalHas(query) && tileHas &&
              tileMaximum > globalMax(query)) {
            pendingNewMax(query) := tileMaximum
            pendingAlphaMagnitude(query) := (
              tileMaximum.resize(cfg.scoreWidth + 1) -
                globalMax(query).resize(cfg.scoreWidth + 1)
            ).asUInt
          }
        }
        state := State.AlphaIssue
      }
    }

    is(State.AlphaIssue) {
      state := State.AlphaWait
    }

    is(State.AlphaWait) {
      when(alphaResponseValid) {
        for (query <- 0 until cfg.queryHeadsPerKv) {
          pendingAlpha(query) := expReadData(query).asUInt
          when(expResponseZero(query) || expResponseTail(query)) {
            pendingAlpha(query) := 0
          }
        }
        state := State.EmitUpdate
      }
    }

    is(State.EmitUpdate) {
      when(io.tileUpdate.fire) {
        for (query <- 0 until cfg.queryHeadsPerKv) {
          globalHas(query) := pendingNewHas(query)
          globalMax(query) := pendingNewMax(query)
          val scaledProduct = globalSum(query) * pendingAlpha(query)
          scaledProduct.addAttribute("use_dsp", "yes")
          scaledOldProduct(query) := scaledProduct
        }
        state := State.ScaleOldSum
      }
    }

    is(State.ScaleOldSum) {
      for (query <- 0 until cfg.queryHeadsPerKv) {
        val rounded = (
          scaledOldProduct(query).resize(scaledOldProduct(query).getWidth + 1) +
            U(BigInt(1) << (cfg.expFractionalBits - 1),
              (scaledOldProduct(query).getWidth + 1) bits)
        ) |>> cfg.expFractionalBits
        tileSum(query) := rounded.resize(cfg.sumWidth)
      }
        replayIndex := 0
        allReplayIssued := False
        state := State.Replay
    }

    is(State.Replay) {
      // Synchronous exp reads and FIFO pushes are handled above.
    }

    is(State.WaitDrain) {
      when(io.probability.fire && io.probability.lastInTile) {
        for (query <- 0 until cfg.queryHeadsPerKv) {
          globalSum(query) := completedTileSum(query)
        }
        probabilityCounter := probabilityCounter + pendingKeyCount
        bankProcessing(processBank.asUInt) := False
        processBank := !processBank
        when(pendingLastTile) {
          state := State.EmitNormalization
        } otherwise {
          state := State.WaitTile
        }
      }
    }

    is(State.EmitNormalization) {
      when(io.normalization.fire) {
        completedCounter := completedCounter + 1
        commandActive := False
        state := State.WaitCommand
      }
    }
  }
}

/** Bit-accurate software oracle for the tiled softmax stream contract. */
object BitNetTiledAttentionMath {
  case class Position(
      scoresQ12: Seq[Int],
      keyPosition: Int,
      valid: Boolean = true
  )

  case class TileUpdate(
      alphaQ15: Seq[Int],
      newMaxQ12: Seq[Int],
      allMasked: Seq[Boolean],
      firstTile: Boolean,
      lastTile: Boolean
  )

  case class Probability(
      weightQ15: Seq[Int],
      keyPosition: Int,
      keyIndex: Int,
      tileIndex: Int,
      firstInTile: Boolean,
      lastInTile: Boolean,
      lastInSequence: Boolean
  )

  case class Result(
      updates: Seq[TileUpdate],
      probabilities: Seq[Probability],
      denominatorQ15: Seq[BigInt],
      finalMaxQ12: Seq[Int],
      allMasked: Seq[Boolean]
  )

  private val ExpOne = 1 << 15
  private val ExpLut = (0 to 128).map { index =>
    Math.round(Math.exp(-index.toDouble / 16.0) * ExpOne).toInt
  }

  def expNegative(magnitudeQ12: Int): Int = {
    require(magnitudeQ12 >= 0)
    val index = (magnitudeQ12 + 128) >> 8
    if (index <= 128) ExpLut(index) else 0
  }

  def compute(
      queryPosition: Int,
      queryMask: Seq[Boolean],
      positions: Seq[Position],
      tileKeys: Int = 16,
      causal: Boolean = true
  ): Result = {
    require(queryMask.length == 4)
    require(positions.nonEmpty && positions.length <= 4096)
    require(positions.forall(_.scoresQ12.length == 4))
    require(tileKeys == 16)

    val has = Array.fill(4)(false)
    val maximum = Array.fill(4)(0)
    val denominator = Array.fill(4)(BigInt(0))
    val updates = scala.collection.mutable.ArrayBuffer.empty[TileUpdate]
    val probabilities =
      scala.collection.mutable.ArrayBuffer.empty[Probability]

    positions.grouped(tileKeys).zipWithIndex.foreach {
      case (tile, tileIndex) =>
        val tileHas = Array.fill(4)(false)
        val tileMaximum = Array.fill(4)(0)
        for (position <- tile; query <- 0 until 4) {
          val accepted = queryMask(query) && position.valid &&
            (!causal || position.keyPosition <= queryPosition)
          if (accepted && (!tileHas(query) ||
              position.scoresQ12(query) > tileMaximum(query))) {
            tileHas(query) = true
            tileMaximum(query) = position.scoresQ12(query)
          }
        }

        val alpha = Array.fill(4)(0)
        for (query <- 0 until 4) {
          if (has(query)) {
            if (tileHas(query) && tileMaximum(query) > maximum(query)) {
              alpha(query) = expNegative(tileMaximum(query) - maximum(query))
              maximum(query) = tileMaximum(query)
            } else {
              alpha(query) = ExpOne
            }
          } else if (tileHas(query)) {
            maximum(query) = tileMaximum(query)
          }
          denominator(query) =
            (denominator(query) * alpha(query) + (1 << 14)) >> 15
          has(query) ||= tileHas(query)
        }

        updates += TileUpdate(
          alpha.toVector,
          maximum.toVector,
          has.map(!_).toSeq,
          firstTile = tileIndex == 0,
          lastTile = tileIndex == (positions.length - 1) / tileKeys
        )

        tile.zipWithIndex.foreach { case (position, inTileIndex) =>
          val weights = (0 until 4).map { query =>
            val accepted = queryMask(query) && position.valid &&
              (!causal || position.keyPosition <= queryPosition)
            if (accepted) expNegative(maximum(query) -
              position.scoresQ12(query)) else 0
          }
          for (query <- 0 until 4) denominator(query) += weights(query)
          val keyIndex = tileIndex * tileKeys + inTileIndex
          probabilities += Probability(
            weights,
            position.keyPosition,
            keyIndex,
            tileIndex,
            firstInTile = inTileIndex == 0,
            lastInTile = inTileIndex == tile.length - 1,
            lastInSequence = keyIndex == positions.length - 1
          )
        }
    }

    Result(
      updates.toSeq,
      probabilities.toSeq,
      denominator.toVector,
      maximum.toVector,
      has.map(!_).toSeq
    )
  }
}
