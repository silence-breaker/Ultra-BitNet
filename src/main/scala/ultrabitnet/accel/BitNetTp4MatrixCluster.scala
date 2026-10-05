package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/** Geometry of the production related-clock TP4 matrix cluster. */
case class BitNetTp4MatrixClusterConfig(
    matrix: BitNetContinuousMatrixConfig = BitNetContinuousMatrixConfig(),
    svGroupLanes: Int = 8,
    logicalEngineCount: Int = 2,
    physicalEngineCount: Int = 2,
    productionPruneGenericSv: Boolean = true,
    productionPruneLegacyBeat: Boolean = true
) {
  require(matrix.tokenLanesPerEngine == 2)
  require(logicalEngineCount == 1 || logicalEngineCount == 2,
    "the production ABI supports one TP2 or two TP2 logical streams")
  require(physicalEngineCount > 0 && physicalEngineCount <= logicalEngineCount,
    "physical engines must be a non-empty subset of logical engines")
  require(svGroupLanes > 0 && isPow2(svGroupLanes))
  require(matrix.base.siteCount % svGroupLanes == 0)
  require(!productionPruneLegacyBeat || productionPruneGenericSv,
    "generic SV research mode requires the legacy wide beat mailbox")
  val engineCount: Int = physicalEngineCount
  val tokenLanes: Int = logicalEngineCount * matrix.tokenLanesPerEngine
  val svGroupCount: Int = matrix.base.siteCount / svGroupLanes
  val svGroupIndexWidth: Int = log2Up(svGroupCount)
}

/**
  * Two continuous BitNet arrays behind a related 1x/2x clock boundary.
  *
  * This is intentionally not an asynchronous-FIFO CDC.  `slowClk` and
  * `fastClk` must be timing-related, with `fastClk` exactly 2x `slowClk` and a
  * board constraint defining their phase relationship.  One slow-domain
  * payload register is held until the fast core accepts it; a request/ack
  * toggle crosses directly under that related-clock constraint.  A new beat
  * can therefore be captured on every slow edge without materialising a
  * multi-kilobit FIFO in LUTRAM.
  *
  * Fast dot returns are bandwidth matched as TP2 pairs: prefill lane 0/lane 1
  * are collected on consecutive fast clocks and exposed once per slow clock;
  * decode uses validMask=01.  A held 128xS48 SV result is never copied into a
  * 6144-bit mailbox.  The slow side reads eight stable sites at a time and
  * acknowledges the core only after the final group is consumed.
  */
class BitNetTp4MatrixCluster(
    cfg: BitNetTp4MatrixClusterConfig = BitNetTp4MatrixClusterConfig(),
    useVendorPrimitive: Boolean = true
) extends Component {
  private val m = cfg.matrix
  private val b = m.base
  private val feederCfg = BitNetContinuousMatrixFeederConfig(matrix = m)

  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val sequence = Vec(
      slave(Stream(BitNetContinuousSequence(m))), cfg.engineCount)
    val map0Beat = Vec(
      slave(Stream(BitNetMap0PairMatrixBeat(feederCfg))), cfg.engineCount)
    val i8Beat = Vec(
      slave(Stream(BitNetI8MatrixBeat(feederCfg))), cfg.engineCount)
    /** Legacy attention/SV beat exists only in explicit research builds. */
    val beat = if (!cfg.productionPruneLegacyBeat)
      Vec(slave(Stream(BitNetContinuousMatrixBeat(
        m, includeGenericSv = !cfg.productionPruneGenericSv))),
        cfg.engineCount) else null
    val dotResult = Vec(
      master(Stream(BitNetProductionCompactTp2DotResultBeat(m))),
      cfg.engineCount)
    val svResult = if (!cfg.productionPruneGenericSv) Vec(
      master(Stream(BitNetContinuousSvResultGroup(
        m, cfg.svGroupLanes))), cfg.engineCount) else null
    val residentSequence = Vec(
      slave(Stream(BitNetResidentSvSequence(m))), cfg.engineCount)
    val residentBeat = Vec(
      slave(Stream(BitNetResidentSvBeat(m))), cfg.engineCount)
    val residentResult = Vec(
      master(Stream(BitNetResidentSvResultGroup(
        m, cfg.svGroupLanes))), cfg.engineCount)

    val sequenceActive = out Bits(cfg.engineCount bits)
    val draining = out Bits(cfg.engineCount bits)
    val protocolFaultSticky = out Bits(cfg.engineCount bits)
  }

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))
  val fastCd = ClockDomain(
    clock = io.fastClk,
    reset = io.fastResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  val fastCoreArea = new ClockingArea(fastCd) {
    val cores = Array.fill(cfg.engineCount)(
      new BitNetContinuousMatrixArrayCore(
        cfg = m,
        useVendorPrimitive = useVendorPrimitive,
        productionPruneGenericSv = cfg.productionPruneGenericSv,
        productionCompactDotResult = true))
  }

  for (engine <- 0 until cfg.engineCount) {
    // Cross-domain nets are deliberately direct.  Their source payload stays
    // stable for the complete request/ack ownership interval, and the clocks
    // are constrained as related 1x/2x clocks at the production top.
    val sequenceRequestX = Bool()
    val sequenceAckX = Bool()
    val sequencePayloadX = BitNetContinuousSequence(m)
    val typedBeatRequestX = Bool()
    val typedBeatAckX = Bool()
    val typedBeatRetiringX = Bool()
    val typedBeatSourceI8X = Bool()
    val typedBeatSequenceIdX = UInt(m.sequenceIdWidth bits)
    val typedBeatOrdinalX = UInt(m.beatCountWidth bits)
    val typedBeatVariantX = Bool()
    val typedBeatMetaX = BitNetTypedMatrixBeatMeta(feederCfg)
    val typedRawWeightsX = Vec(Bits(8 bits), b.siteCount)
    val sequenceAllowsMap0X = Bool()
    val sequenceAllowsI8X = Bool()
    val typedInputFaultX = Bool()
    val beatRequestX = if (!cfg.productionPruneLegacyBeat) Bool() else null
    val beatAckX = if (!cfg.productionPruneLegacyBeat) Bool() else null
    val beatRetiringX = if (!cfg.productionPruneLegacyBeat) Bool() else null
    val beatPayloadX = if (!cfg.productionPruneLegacyBeat)
      BitNetContinuousMatrixBeat(
        m, includeGenericSv = !cfg.productionPruneGenericSv) else null
    val dotRequestX = Bool()
    val dotAckX = Bool()
    val dotPayloadX = BitNetProductionCompactTp2DotResultBeat(m)
    val svRequestX = if (!cfg.productionPruneGenericSv) Bool() else null
    val svAckX = if (!cfg.productionPruneGenericSv) Bool() else null
    val svPayloadX = if (!cfg.productionPruneGenericSv)
      BitNetContinuousSvRawResult(m) else null
    val residentSequenceRequestX = Bool()
    val residentSequenceAckX = Bool()
    val residentSequencePayloadX = BitNetResidentSvSequence(m)
    val residentBeatRequestX = Bool()
    val residentBeatAckX = Bool()
    val residentBeatPayloadX = BitNetResidentSvBeat(m)
    val residentResultRequestX = Bool()
    val residentResultAckX = Bool()
    val residentResultPayloadX = BitNetResidentSvRawResult(m)
    val sequenceActiveX = Bool()
    val drainingX = Bool()
    val protocolFaultX = Bool()

    Seq(
      sequenceRequestX,
      sequenceAckX,
      sequencePayloadX,
      typedBeatRequestX,
      typedBeatAckX,
      typedBeatRetiringX,
      typedBeatSourceI8X,
      typedBeatSequenceIdX,
      typedBeatOrdinalX,
      typedBeatVariantX,
      typedBeatMetaX,
      typedRawWeightsX,
      sequenceAllowsMap0X,
      sequenceAllowsI8X,
      typedInputFaultX,
      dotRequestX,
      dotAckX,
      dotPayloadX,
      residentSequenceRequestX,
      residentSequenceAckX,
      residentSequencePayloadX,
      residentBeatRequestX,
      residentBeatAckX,
      residentBeatPayloadX,
      residentResultRequestX,
      residentResultAckX,
      residentResultPayloadX,
      sequenceActiveX,
      drainingX,
      protocolFaultX
    ).foreach(_.addTag(crossClockDomain))
    if (!cfg.productionPruneLegacyBeat) {
      Seq(beatRequestX, beatAckX, beatRetiringX, beatPayloadX)
        .foreach(_.addTag(crossClockDomain))
    }
    if (!cfg.productionPruneGenericSv) {
      Seq(svRequestX, svAckX, svPayloadX)
        .foreach(_.addTag(crossClockDomain))
    }

    // ---------------------------------------------------------------------
    // Slow -> fast sequence and wide beat mailboxes.
    // ---------------------------------------------------------------------
    val slowInputArea = new ClockingArea(slowCd) {
      // The core validates a sequence through a short registered checker.
      // Do not let the slow mailbox acknowledge the first data beat before
      // that sequence actually owns the array: doing so strands beat zero in
      // the mailbox and creates a one-off 1x-cycle bubble before beat one.
      val sequenceStarted = BufferCC(
        sequenceActiveX,
        init = False,
        bufferDepth = Some(2),
        inputAttributes = Seq(crossClockDomain))

      val sequenceRequest = RegInit(False)
      val sequencePayload = Reg(BitNetContinuousSequence(m)) init (
        BitNetContinuousSequence(m).getZero)
      val sequenceAckSeen = RegNext(sequenceAckX) init False
      sequenceAckSeen.setName(s"slowSequenceAckSeen_$engine")
      val sequenceEmpty = sequenceRequest === sequenceAckSeen
      io.sequence(engine).ready := sequenceEmpty
      when(io.sequence(engine).fire) {
        sequencePayload := io.sequence(engine).payload
        sequenceRequest := !sequenceRequest
      }
      sequenceRequestX := sequenceRequest
      sequencePayloadX := sequencePayload

      // A sequence fixes the typed physical format.  Exporting the format that
      // the fast core actually accepted avoids accidentally following a
      // subsequently queued sequence payload in the slow domain.
      val sequenceAllowsMap0 = BufferCC(
        sequenceAllowsMap0X,
        init = False,
        bufferDepth = Some(2),
        inputAttributes = Seq(crossClockDomain))
      val sequenceAllowsI8 = BufferCC(
        sequenceAllowsI8X,
        init = False,
        bufferDepth = Some(2),
        inputAttributes = Seq(crossClockDomain))

      // A sequence permits exactly one physical format, so Map0 and I8 share
      // one 8-bit/site store as well as one activation/meta/header mailbox.
      // Map0 occupies only the low nibble and clears the high nibble.
      val typedBeatRequest = RegInit(False)
      val typedBeatSourceI8 = RegInit(False)
      val typedBeatSequenceId = Reg(UInt(m.sequenceIdWidth bits)) init (0)
      val typedBeatOrdinal = Reg(UInt(m.beatCountWidth bits)) init (0)
      val typedBeatVariant = RegInit(False)
      val typedBeatMeta = Reg(BitNetTypedMatrixBeatMeta(feederCfg)) init (
        BitNetTypedMatrixBeatMeta(feederCfg).getZero)
      val typedRawWeights = Vec(
        Reg(Bits(8 bits)) init (0), b.siteCount)
      typedBeatMeta.setName(s"typedMetaMailbox_$engine")
      typedRawWeights.setName(s"typedRawMailbox_$engine")

      // Terminate both fast->slow control paths at slow-domain registers.
      // The direct related-clock wires remain fully timed, but now end at a
      // single FF instead of traversing the feeder/cache ready network inside
      // the 3.760 ns half-cycle.  The request payload is held until ACK, and
      // the retire indication is only an optional early replacement credit,
      // so the extra slow observation cycle preserves mailbox ownership.
      val typedBeatAckSeen = RegNext(typedBeatAckX) init False
      val typedBeatRetireSeen = RegNext(typedBeatRetiringX) init False
      typedBeatAckSeen.setName(s"slowTypedBeatAckSeen_$engine")
      typedBeatRetireSeen.setName(s"slowTypedBeatRetireSeen_$engine")
      val typedBeatEmpty = typedBeatRequest === typedBeatAckSeen
      val typedBeatAvailable = typedBeatEmpty || typedBeatRetireSeen
      // Malformed lane masks never acquire mailbox ownership.  The fast core
      // remains the architectural legality checker, but this slow-domain
      // fail-closed guard makes the registered replacement promise safe
      // without placing any payload decode in its 2x-domain D cone.
      val map0LaneMaskLegal =
        io.map0Beat(engine).payload.meta.laneMask === B"2'b01" ||
          io.map0Beat(engine).payload.meta.laneMask === B"2'b11"
      val i8LaneMaskLegal =
        io.i8Beat(engine).payload.meta.laneMask === B"2'b01" ||
          io.i8Beat(engine).payload.meta.laneMask === B"2'b11"
      io.map0Beat(engine).ready := typedBeatAvailable && sequenceStarted &&
        sequenceAllowsMap0 && map0LaneMaskLegal
      io.i8Beat(engine).ready := typedBeatAvailable && sequenceStarted &&
        sequenceAllowsI8 && i8LaneMaskLegal

      when(io.map0Beat(engine).fire) {
        typedBeatSourceI8 := False
        typedBeatSequenceId := io.map0Beat(engine).payload.sequenceId
        typedBeatOrdinal := io.map0Beat(engine).payload.beatOrdinal
        typedBeatVariant := io.map0Beat(engine).payload.pairSelect ===
          BitNetMap0PairSelect.UpperRows23
        typedBeatMeta := io.map0Beat(engine).payload.meta
        for (site <- 0 until b.siteCount) {
          typedRawWeights(site) :=
            B(0, 4 bits) ## io.map0Beat(engine).payload.rawWeights(site)
        }
        typedBeatRequest := !typedBeatRequest
      }
      when(io.i8Beat(engine).fire) {
        typedBeatSourceI8 := True
        typedBeatSequenceId := io.i8Beat(engine).payload.sequenceId
        typedBeatOrdinal := io.i8Beat(engine).payload.beatOrdinal
        typedBeatVariant := io.i8Beat(engine).payload.use ===
          BitNetI8WeightUse.LmHead
        typedBeatMeta := io.i8Beat(engine).payload.meta
        typedRawWeights := io.i8Beat(engine).payload.rawWeights
        typedBeatRequest := !typedBeatRequest
      }

      // Wrong-format traffic cannot enter the shared mailbox and is sticky.
      // Simultaneous Map0/I8 valids are also malformed even though only the
      // sequence-selected input can receive ready.
      val typedInputFault = RegInit(False)
      when(sequenceStarted &&
        ((io.map0Beat(engine).valid && !sequenceAllowsMap0) ||
          (io.i8Beat(engine).valid && !sequenceAllowsI8) ||
          (io.map0Beat(engine).valid && !map0LaneMaskLegal) ||
          (io.i8Beat(engine).valid && !i8LaneMaskLegal) ||
          (io.map0Beat(engine).valid && io.i8Beat(engine).valid))) {
        typedInputFault := True
      }

      typedBeatRequestX := typedBeatRequest
      typedBeatSourceI8X := typedBeatSourceI8
      typedBeatSequenceIdX := typedBeatSequenceId
      typedBeatOrdinalX := typedBeatOrdinal
      typedBeatVariantX := typedBeatVariant
      typedBeatMetaX := typedBeatMeta
      typedRawWeightsX := typedRawWeights
      typedInputFaultX := typedInputFault

      if (!cfg.productionPruneLegacyBeat) {
        val beatRequest = RegInit(False)
        val beatPayload = Reg(BitNetContinuousMatrixBeat(
          m, includeGenericSv = !cfg.productionPruneGenericSv)) init (
          BitNetContinuousMatrixBeat(
            m, includeGenericSv = !cfg.productionPruneGenericSv).getZero)
        val beatEmpty = beatRequest === beatAckX
        io.beat(engine).ready :=
          (beatEmpty || beatRetiringX) && sequenceStarted
        when(io.beat(engine).fire) {
          beatPayload := io.beat(engine).payload
          beatRequest := !beatRequest
        }
        beatRequestX := beatRequest
        beatPayloadX := beatPayload
      }

      val residentSequenceRequest = RegInit(False)
      val residentSequencePayload = Reg(BitNetResidentSvSequence(m)) init (
        BitNetResidentSvSequence(m).getZero)
      val residentSequenceAckSeen = RegNext(residentSequenceAckX) init False
      residentSequenceAckSeen.setName(
        s"slowResidentSequenceAckSeen_$engine")
      val residentSequenceEmpty =
        residentSequenceRequest === residentSequenceAckSeen
      io.residentSequence(engine).ready := residentSequenceEmpty
      when(io.residentSequence(engine).fire) {
        residentSequencePayload := io.residentSequence(engine).payload
        residentSequenceRequest := !residentSequenceRequest
      }
      residentSequenceRequestX := residentSequenceRequest
      residentSequencePayloadX := residentSequencePayload

      val residentBeatRequest = RegInit(False)
      val residentBeatPayload = Reg(BitNetResidentSvBeat(m)) init (
        BitNetResidentSvBeat(m).getZero)
      val residentBeatAckSeen = RegNext(residentBeatAckX) init False
      residentBeatAckSeen.setName(s"slowResidentBeatAckSeen_$engine")
      val residentBeatEmpty = residentBeatRequest === residentBeatAckSeen
      io.residentBeat(engine).ready := residentBeatEmpty
      when(io.residentBeat(engine).fire) {
        residentBeatPayload := io.residentBeat(engine).payload
        residentBeatRequest := !residentBeatRequest
      }
      residentBeatRequestX := residentBeatRequest
      residentBeatPayloadX := residentBeatPayload
    }

    // ---------------------------------------------------------------------
    // Fast-side consumers and result aggregation.
    // ---------------------------------------------------------------------
    val fastBridgeArea = new ClockingArea(fastCd) {
      val core = fastCoreArea.cores(engine)

      val sequenceAck = RegInit(False)
      val sequenceAllowsMap0 = RegInit(False)
      val sequenceAllowsI8 = RegInit(False)
      core.io.sequence.valid := sequenceRequestX =/= sequenceAck
      core.io.sequence.payload := sequencePayloadX
      when(core.io.sequence.fire) {
        sequenceAck := sequenceRequestX
        sequenceAllowsMap0 := sequencePayloadX.format ===
          BitNetUnifiedOperandFormat.Map0Packed
        sequenceAllowsI8 := sequencePayloadX.format ===
          BitNetUnifiedOperandFormat.I8Pair
      }
      sequenceAckX := sequenceAck
      sequenceAllowsMap0X := sequenceAllowsMap0
      sequenceAllowsI8X := sequenceAllowsI8

      // The only Map0 4->8 extension point is here, in the fast domain and
      // immediately before the current byte-wide core port.  No register is
      // present in the adapter, so the related-clock mailbox remains 4b/site.
      val typedAdapter = new BitNetTypedWeightToContinuousBeatAdapter(
        feederCfg, includeGenericSv = !cfg.productionPruneGenericSv)
      val typedBeatAck = RegInit(False)
      val beatAck = if (!cfg.productionPruneLegacyBeat)
        RegInit(False) else null

      val legacyRequestPending = if (!cfg.productionPruneLegacyBeat)
        beatRequestX =/= beatAck else False
      val typedRequestPending = typedBeatRequestX =/= typedBeatAck

      // Register only ownership in the fast domain.  The source mailbox keeps
      // its payload stable until the matching acknowledgement, so copying the
      // roughly 3-kbit Continuous beat into another register bank would waste
      // thousands of FFs per engine.  The local valid/kind/toggle stage still
      // gives every request a complete capture cycle before it may reach the
      // core and removes the request -> core.ready -> acknowledgement loop.
      val beatStageValid = RegInit(False)
      val beatStageSource = Reg(UInt(2 bits)) init (0)
      val beatStageRequestToggle = RegInit(False)
      val beatStageRetireCredit = RegInit(False)
      val typedRetirePromise = RegInit(False)
      beatStageValid.setName(s"fastBeatStageValid_$engine")
      beatStageSource.setName(s"fastBeatStageSource_$engine")
      beatStageRequestToggle.setName(s"fastBeatStageToggle_$engine")
      beatStageRetireCredit.setName(s"fastBeatStageCredit_$engine")
      typedRetirePromise.setName(s"fastTypedRetirePromise_$engine")

      val sourceLegacy = U(0, 2 bits)
      val sourceMap0 = U(1, 2 bits)
      val sourceI8 = U(2, 2 bits)

      // Reserve the next shared-edge retirement from occupancy alone.  The
      // core has one registered reservation stage, so leave two additional
      // result slots beyond its ordinary `depth - 3` admission threshold.  If
      // outstanding is N at this edge, at most one pending TP2 reservation can
      // raise it to N+2 before a newly staged beat fires on the next edge.
      // Updating this register unconditionally is important: its D cone is
      // only the registered occupancy threshold, never source arbitration,
      // sequence format, lane mask, context metadata or live core.ready.
      // Register the core-mode admission predicate before it reaches the
      // mailbox source/valid clock-enables.  `residentActive` is decoded from
      // the resident FSM and otherwise creates a six-LUT, high-fanout path at
      // 532 MHz.  The typed producer is already authenticated to the exact
      // sequence beat count, so the one-fast-cycle opening/closing latency
      // cannot admit an architectural extra beat; it only moves the first
      // capture one half slow-clock later.
      val matrixIngressOpen = RegNext(
        core.io.sequenceActive && !core.io.residentActive &&
          !core.io.draining && core.io.beatsRemaining =/= 0) init False
      matrixIngressOpen.setName(s"fastMatrixIngressOpen_$engine")
      val conservativeDotCredit = core.io.reservedResults <=
        (m.resultFifoDepth - 5)
      beatStageRetireCredit := conservativeDotCredit

      // A stage is never captured and consumed on the same edge, nor refilled
      // on its retire edge; this prevents a held request from being captured
      // twice.  Only research builds retain the former legacy-first branch.
      if (!cfg.productionPruneLegacyBeat) {
        when(!beatStageValid && matrixIngressOpen) {
          when(legacyRequestPending) {
            beatStageValid := True
            beatStageSource := sourceLegacy
            beatStageRequestToggle := beatRequestX
          } elsewhen (typedRequestPending) {
            beatStageValid := True
            beatStageSource := Mux(
              typedBeatSourceI8X, sourceI8, sourceMap0)
            beatStageRequestToggle := typedBeatRequestX
          }
        }
      } else {
        when(!beatStageValid && matrixIngressOpen && typedRequestPending) {
          beatStageValid := True
          beatStageSource := Mux(
            typedBeatSourceI8X, sourceI8, sourceMap0)
          beatStageRequestToggle := typedBeatRequestX
        }
      }

      val stagedMap0 = beatStageValid && beatStageSource === sourceMap0
      val stagedI8 = beatStageValid && beatStageSource === sourceI8
      val stagedLegacy = if (!cfg.productionPruneLegacyBeat)
        beatStageValid && beatStageSource === sourceLegacy else False
      typedAdapter.io.map0.valid := stagedMap0
      typedAdapter.io.i8.valid := stagedI8
      typedAdapter.io.map0.payload.sequenceId := typedBeatSequenceIdX
      typedAdapter.io.i8.payload.sequenceId := typedBeatSequenceIdX
      typedAdapter.io.map0.payload.beatOrdinal := typedBeatOrdinalX
      typedAdapter.io.i8.payload.beatOrdinal := typedBeatOrdinalX
      typedAdapter.io.map0.payload.pairSelect :=
        BitNetMap0PairSelect.LowerRows01
      when(typedBeatVariantX) {
        typedAdapter.io.map0.payload.pairSelect :=
          BitNetMap0PairSelect.UpperRows23
      }
      typedAdapter.io.i8.payload.use := BitNetI8WeightUse.Qk
      when(typedBeatVariantX) {
        typedAdapter.io.i8.payload.use := BitNetI8WeightUse.LmHead
      }
      typedAdapter.io.map0.payload.meta := typedBeatMetaX
      typedAdapter.io.i8.payload.meta := typedBeatMetaX
      for (site <- 0 until b.siteCount) {
        typedAdapter.io.map0.payload.rawWeights(site) :=
          typedRawWeightsX(site)(3 downto 0)
      }
      typedAdapter.io.i8.payload.rawWeights := typedRawWeightsX

      core.io.beat.valid := beatStageValid
      core.io.beat.payload := typedAdapter.io.beat.payload
      if (!cfg.productionPruneLegacyBeat) {
        when(stagedLegacy) { core.io.beat.payload := beatPayloadX }
      }
      typedAdapter.io.beat.ready := core.io.beat.ready &&
        (stagedMap0 || stagedI8)
      // The generic legacy port includes SV context hazards which cannot be
      // predicted from occupancy.  It therefore receives no early retire
      // promise.  Typed Map0/I8 is dot-only and was fail-closed at ingress, so
      // occupancy plus the accepted sequence state is a complete guarantee.
      if (!cfg.productionPruneLegacyBeat) beatRetiringX := False
      // Register the complete promise before it crosses back to the 1x
      // domain.  Previously the source-kind decode was still combinational,
      // allowing core stage state to traverse the feeder ready network and
      // reach slow cache/lease CEs inside the 3.760 ns fast->slow half-cycle.
      // The acknowledgement toggle still proves actual retirement; delaying
      // this optional replacement credit by one fast cycle cannot retire or
      // duplicate mailbox ownership.
      typedRetirePromise := beatStageRetireCredit &&
        (stagedMap0 || stagedI8)
      typedBeatRetiringX := typedRetirePromise

      // ACK is deliberately delayed until the staged beat is accepted by the
      // core.  The captured toggle, rather than the live request wire, proves
      // exactly which stable slow-domain payload is being retired.
      when(core.io.beat.fire) {
        beatStageValid := False
        switch(beatStageSource) {
          if (!cfg.productionPruneLegacyBeat)
            is(sourceLegacy) { beatAck := beatStageRequestToggle }
          is(sourceMap0) { typedBeatAck := beatStageRequestToggle }
          is(sourceI8) { typedBeatAck := beatStageRequestToggle }
        }
      }
      typedBeatAckX := typedBeatAck
      if (!cfg.productionPruneLegacyBeat) beatAckX := beatAck
      val typedBeatFault = RegInit(False)
      when(typedInputFaultX ||
        (if (!cfg.productionPruneLegacyBeat)
          legacyRequestPending && typedRequestPending else False)) {
        typedBeatFault := True
      }

      val residentSequenceAck = RegInit(False)
      core.io.residentSequence.valid :=
        residentSequenceRequestX =/= residentSequenceAck
      core.io.residentSequence.payload := residentSequencePayloadX
      when(core.io.residentSequence.fire) {
        residentSequenceAck := residentSequenceRequestX
      }
      residentSequenceAckX := residentSequenceAck

      val residentBeatAck = RegInit(False)
      core.io.residentBeat.valid := residentBeatRequestX =/= residentBeatAck
      core.io.residentBeat.payload := residentBeatPayloadX
      when(core.io.residentBeat.fire) {
        residentBeatAck := residentBeatRequestX
      }
      residentBeatAckX := residentBeatAck

      // Pair scalar fast results before crossing to the 1x domain.  A lane-0
      // holding register is enough to absorb the next TP2 pair's first phase
      // while the previous completed pair is consumed on the slow edge.
      val dotRequest = RegInit(False)
      val dotPair = Reg(BitNetProductionCompactTp2DotResultBeat(m)) init (
        BitNetProductionCompactTp2DotResultBeat(m).getZero)
      val lane0Pending = RegInit(False)
      val lane0Result = Reg(BitNetProductionCompactDotResult(m)) init (
        BitNetProductionCompactDotResult(m).getZero)
      val bridgeFault = RegInit(False)

      val dotMailboxEmpty = dotRequest === dotAckX

      // Break the core result RAM -> payload decode -> ready -> core closing
      // loop with a small fast-domain elasticity queue.  Its push.ready is an
      // occupancy-only registered-control path, so the 532 MHz core never sees
      // the TP2 pairing or slow-mailbox state combinationally.  Depth four can
      // retain a complete current pair plus the next pair while the 1x side
      // acknowledges the previous result.
      // Compact at the arithmetic source: the core's 128-deep result FIFO no
      // longer stores research metadata which every production endpoint can
      // reconstruct from its retained launch descriptor.
      val scalarElastic = StreamFifo(
        BitNetProductionCompactDotResult(m), depth = 4)
      scalarElastic.io.push << core.io.compactDotResult
      val scalar = scalarElastic.io.pop
      val scalarIsSingle = scalar.payload.pairPhase ===
        BitNetProductionCompactDotPairPhase.SingleOrDecode
      val scalarIsPairLane0 = scalar.payload.pairPhase ===
        BitNetProductionCompactDotPairPhase.pairedLane(0, m)
      val scalarIsPairLane1 = scalar.payload.pairPhase ===
        BitNetProductionCompactDotPairPhase.pairedLane(1, m)
      scalar.ready := False
      when(scalarIsPairLane0 && !lane0Pending) {
        // Lane 0 may be retained while the previous completed pair is owned
        // by the slow side.
        scalar.ready := True
      } elsewhen(scalarIsPairLane1 && lane0Pending) {
        scalar.ready := dotMailboxEmpty
      } elsewhen(scalarIsSingle && !lane0Pending) {
        scalar.ready := dotMailboxEmpty && !lane0Pending
      } otherwise {
        // Every malformed phase is consumed once the mailbox is free and
        // converted into an observable fail-stop beat instead of deadlocking.
        scalar.ready := dotMailboxEmpty
      }

      val lane1Tag = (lane0Result.accumulatorTag + 1).resize(
        m.accumulatorTagWidth)
      val pairIdentityLegal =
        lane0Result.pairPhase ===
          BitNetProductionCompactDotPairPhase.pairedLane(0, m) &&
          scalar.payload.pairPhase ===
            BitNetProductionCompactDotPairPhase.pairedLane(1, m) &&
          scalar.payload.sequenceId === lane0Result.sequenceId &&
          scalar.payload.accumulatorTag === lane1Tag &&
          scalar.payload.firstK === lane0Result.firstK &&
          scalar.payload.lastK === lane0Result.lastK &&
          scalar.payload.formatClass === lane0Result.formatClass

      when(scalar.fire) {
        when(scalarIsPairLane0 && !lane0Pending) {
          lane0Result := scalar.payload
          lane0Pending := True
        } elsewhen(scalarIsPairLane1 && lane0Pending) {
          dotPair.sequenceId := lane0Result.sequenceId
          dotPair.baseAccumulatorTag := lane0Result.accumulatorTag
          dotPair.firstK := lane0Result.firstK
          dotPair.lastK := lane0Result.lastK
          dotPair.formatClass := lane0Result.formatClass
          for (row <- 0 until b.outputLanes) {
            when(pairIdentityLegal) {
              dotPair.rawRows(0)(row) := lane0Result.values(row)
              dotPair.rawRows(1)(row) := scalar.payload.values(row)
            } otherwise {
              dotPair.rawRows(0)(row) := 0
              dotPair.rawRows(1)(row) := 0
            }
          }
          when(pairIdentityLegal) {
            dotPair.validMask := B"2'b11"
          } otherwise {
            dotPair.validMask := B"2'b00"
            bridgeFault := True
          }
          lane0Pending := False
          dotRequest := !dotRequest
        } elsewhen(scalarIsSingle && !lane0Pending) {
          dotPair.validMask := B"2'b01"
          dotPair.sequenceId := scalar.payload.sequenceId
          dotPair.baseAccumulatorTag := scalar.payload.accumulatorTag
          dotPair.firstK := scalar.payload.firstK
          dotPair.lastK := scalar.payload.lastK
          dotPair.formatClass := scalar.payload.formatClass
          for (row <- 0 until b.outputLanes) {
            dotPair.rawRows(0)(row) := scalar.payload.values(row)
            dotPair.rawRows(1)(row) := 0
          }
          dotRequest := !dotRequest
        } otherwise {
          dotPair.validMask := B"2'b00"
          dotPair.sequenceId := scalar.payload.sequenceId
          dotPair.baseAccumulatorTag := scalar.payload.accumulatorTag
          dotPair.firstK := scalar.payload.firstK
          dotPair.lastK := scalar.payload.lastK
          dotPair.formatClass := scalar.payload.formatClass
          for (local <- 0 until m.tokenLanesPerEngine;
               row <- 0 until b.outputLanes) {
            dotPair.rawRows(local)(row) := 0
          }
          lane0Pending := False
          bridgeFault := True
          dotRequest := !dotRequest
        }
      }
      dotRequestX := dotRequest
      dotPayloadX := dotPair

      if (!cfg.productionPruneGenericSv) {
        // Research compatibility only: publish a stable raw generic-SV result
        // without copying its 6144 data bits.  Production never elaborates
        // this ownership mailbox or its grouped readout below.
        val svRequest = RegInit(False)
        svRequest.setName(s"legacyGenericSvRequest_$engine")
        val svPublished = RegInit(False)
        val svAcknowledged = svPublished && svAckX === svRequest
        val svConsume = RegInit(False)
        svConsume := False
        when(!svConsume && svAcknowledged) { svConsume := True }
        core.io.svResult.ready := svConsume
        when(!svPublished && core.io.svResult.valid) {
          svRequest := !svRequest
          svPublished := True
        }
        when(core.io.svResult.fire) { svPublished := False }
        svRequestX := svRequest
        svPayloadX := core.io.svResult.payload
      }

      // Identical zero-copy ownership for a completed resident context.  P/C
      // ring state remains frozen until the slow side consumes all Vec8 groups.
      val residentResultRequest = RegInit(False)
      val residentResultPublished = RegInit(False)
      val residentResultAcknowledged = residentResultPublished &&
        residentResultAckX === residentResultRequest
      val residentResultConsume = RegInit(False)
      residentResultConsume := False
      when(!residentResultConsume && residentResultAcknowledged) {
        residentResultConsume := True
      }
      core.io.residentRawResult.ready := residentResultConsume
      when(!residentResultPublished && core.io.residentRawResult.valid) {
        residentResultRequest := !residentResultRequest
        residentResultPublished := True
      }
      when(core.io.residentRawResult.fire) {
        residentResultPublished := False
      }
      residentResultRequestX := residentResultRequest
      residentResultPayloadX := core.io.residentRawResult.payload

      sequenceActiveX := core.io.sequenceActive
      drainingX := core.io.draining
      protocolFaultX := core.io.protocolFault || bridgeFault || typedBeatFault
    }

    // ---------------------------------------------------------------------
    // Fast -> slow TP2 result mailbox and grouped SV readout.
    // ---------------------------------------------------------------------
    val slowOutputArea = new ClockingArea(slowCd) {
      val dotAck = RegInit(False)
      io.dotResult(engine).valid := dotRequestX =/= dotAck
      io.dotResult(engine).payload := dotPayloadX
      when(io.dotResult(engine).fire) { dotAck := dotRequestX }
      dotAckX := dotAck

      if (!cfg.productionPruneGenericSv) {
        val svAck = RegInit(False)
        val svGroupIndex = Reg(UInt(cfg.svGroupIndexWidth bits)) init (0)
        svGroupIndex.setName(s"legacyGenericSvGroupIndex_$engine")
        val svAvailable = svRequestX =/= svAck
        val grouped = io.svResult(engine)
        grouped.valid := svAvailable
        grouped.payload.mode := svPayloadX.mode
        grouped.payload.format := svPayloadX.format
        grouped.payload.sequenceId := svPayloadX.sequenceId
        grouped.payload.taskId := svPayloadX.taskId
        grouped.payload.tokenId := svPayloadX.tokenId
        grouped.payload.contextBank := svPayloadX.contextBank
        grouped.payload.contextSlot := svPayloadX.contextSlot
        grouped.payload.headLane := svPayloadX.headLane
        grouped.payload.svLimbShift := svPayloadX.svLimbShift
        grouped.payload.svFactorSigned := svPayloadX.svFactorSigned
        grouped.payload.svFactorFractionBits :=
          svPayloadX.svFactorFractionBits
        grouped.payload.groupIndex := svGroupIndex
        grouped.payload.first := svGroupIndex === 0
        grouped.payload.last := svGroupIndex === cfg.svGroupCount - 1
        for (lane <- 0 until cfg.svGroupLanes) {
          val site = UInt(log2Up(b.siteCount) bits)
          site := ((svGroupIndex.resize(log2Up(b.siteCount)) <<
            log2Up(cfg.svGroupLanes)) + lane).resize(log2Up(b.siteCount))
          grouped.payload.rawP(lane) := svPayloadX.rawP(site)
        }

        when(grouped.fire) {
          when(grouped.payload.last) {
            svGroupIndex := 0
            svAck := svRequestX
          } otherwise {
            svGroupIndex := svGroupIndex + 1
          }
        }
        svAckX := svAck
      }

      val residentResultAck = RegInit(False)
      val residentGroupIndex =
        Reg(UInt(cfg.svGroupIndexWidth bits)) init (0)
      val residentAvailable =
        residentResultRequestX =/= residentResultAck
      val residentGrouped = io.residentResult(engine)
      residentGrouped.valid := residentAvailable
      residentGrouped.payload.schedule := residentResultPayloadX.schedule
      residentGrouped.payload.sequenceId := residentResultPayloadX.sequenceId
      residentGrouped.payload.tokenTag := residentResultPayloadX.tokenTag
      residentGrouped.payload.tokenId := residentResultPayloadX.tokenId
      residentGrouped.payload.kvHead := residentResultPayloadX.kvHead
      residentGrouped.payload.queryHeadId :=
        residentResultPayloadX.queryHeadId
      residentGrouped.payload.contextSlot :=
        residentResultPayloadX.contextSlot
      residentGrouped.payload.residentLane :=
        residentResultPayloadX.residentLane
      residentGrouped.payload.featureHalf :=
        residentResultPayloadX.featureHalf
      residentGrouped.payload.groupIndex := residentGroupIndex
      residentGrouped.payload.first := residentGroupIndex === 0
      residentGrouped.payload.last :=
        residentGroupIndex === cfg.svGroupCount - 1
      for (lane <- 0 until cfg.svGroupLanes) {
        val site = UInt(log2Up(b.siteCount) bits)
        site := ((residentGroupIndex.resize(log2Up(b.siteCount)) <<
          log2Up(cfg.svGroupLanes)) + lane).resize(log2Up(b.siteCount))
        residentGrouped.payload.rawContext(lane) :=
          residentResultPayloadX.rawContext(site)
      }
      when(residentGrouped.fire) {
        when(residentGrouped.payload.last) {
          residentGroupIndex := 0
          residentResultAck := residentResultRequestX
        } otherwise {
          residentGroupIndex := residentGroupIndex + 1
        }
      }
      residentResultAckX := residentResultAck

      io.sequenceActive(engine) := BufferCC(
        sequenceActiveX,
        init = False,
        bufferDepth = Some(2),
        inputAttributes = Seq(crossClockDomain))
      io.draining(engine) := BufferCC(
        drainingX,
        init = False,
        bufferDepth = Some(2),
        inputAttributes = Seq(crossClockDomain))
      io.protocolFaultSticky(engine) := BufferCC(
        protocolFaultX,
        init = False,
        bufferDepth = Some(2),
        inputAttributes = Seq(crossClockDomain))
    }
  }
}

object GenerateBitNetTp4MatrixCluster extends App {
  val target = args.headOption.getOrElse(
    "target/generated-bitnet-tp4-matrix-cluster")
  SpinalConfig(
    targetDirectory = target,
    oneFilePerComponent = false
  ).generateVerilog(new BitNetTp4MatrixCluster())
}
