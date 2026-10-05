package ultrabitnet.accel

import spinal.core._

import scala.language.postfixOps

/**
  * Canonical resident-linear program retained between scheduling and launch.
  *
  * All matrix geometry is a property of the deployed model plus `kind` and
  * `mode`.  Keeping it out of the stored descriptor removes the legacy output
  * ranges, beat counts and duplicated task metadata without weakening the
  * launch boundary: [[BitNetResidentLinearProgramCodec.launchEncodable]]
  * accepts a legacy launch only when a complete encode/expand round trip is
  * bit exact.
  *
  * The two weight scales deliberately remain independent.  Decode assigns
  * different physical output shards to the two engines, and a future stage
  * must not silently merge their scalar identities.
  */
case class BitNetResidentLinearProgram(
    cfg: BitNetResidentLinearEndpointConfig =
      BitNetResidentLinearEndpointConfig()) extends Bundle {
  BitNetResidentLinearProgram.requireCanonical(cfg)

  val requestId = UInt(32 bits)
  val layer = UInt(5 bits)
  val kind = BitNetResidentLinearKind()
  val mode = BitNetMode()
  val tokenMask = Bits(cfg.model.tokenParallelism bits)
  val sequenceId = UInt(12 bits)
  val taskIdBase = UInt(20 bits)
  val tokenBase = UInt(13 bits)
  val activationFactorQ16 = Vec(UInt(32 bits), cfg.model.tokenParallelism)
  val weightScaleQ16 = Vec(UInt(32 bits), cfg.model.engineCount)
}

object BitNetResidentLinearProgram {
  val RequestIdWidth = 32
  val LayerWidth = 5
  val KindWidth = 3
  val ModeWidth = 1
  val TokenMaskWidth = BitNetConfig.Production.tokenParallelism
  val SequenceIdWidth = 12
  val TaskIdWidth = 20
  val TokenBaseWidth = 13
  val ActivationFactorWidth = TokenMaskWidth * 32
  val WeightScaleWidth = BitNetConfig.Production.engineCount * 32

  val PackedWidth: Int = RequestIdWidth + LayerWidth + KindWidth + ModeWidth +
    TokenMaskWidth + SequenceIdWidth + TaskIdWidth + TokenBaseWidth +
    ActivationFactorWidth + WeightScaleWidth

  require(PackedWidth > 0)

  /** Stage one is intentionally the fixed production ABI, not a generic ISA. */
  def requireCanonical(cfg: BitNetResidentLinearEndpointConfig): Unit = {
    val model = cfg.model
    val matrix = cfg.matrix
    require(model.requestIdWidth == RequestIdWidth,
      "resident linear program requires a 32-bit request identity")
    require(model.layerWidth == LayerWidth,
      "resident linear program requires the production 5-bit layer index")
    require(model.sequenceWidth == TokenBaseWidth,
      "resident linear program requires the production 13-bit token base")
    require(model.tokenParallelism == TokenMaskWidth,
      "resident linear program/model token geometry mismatch")
    require(model.engineCount == BitNetConfig.Production.engineCount,
      "resident linear program/model engine geometry mismatch")
    require(matrix.sequenceIdWidth == SequenceIdWidth,
      "resident linear program requires a 12-bit sequence identity")
    require(matrix.taskIdWidth == TaskIdWidth,
      "resident linear program requires a 20-bit task identity")
    require(matrix.base.siteCount == 64 || matrix.base.siteCount == 128)
    require(model.hiddenSize % matrix.base.siteCount == 0)
    require(model.ffnSize % matrix.base.siteCount == 0)
    require(model.hiddenSize % 8 == 0)
    require(model.ffnSize % 8 == 0)
    require((model.kvHeadCount * model.headSize) % 8 == 0)

    // Compact legality does not compare regenerated geometry.  Prove once at
    // elaboration that every constant-table entry is representable and obeys
    // the group-of-four production schedule.
    val outputLimit = BigInt(1) << matrix.base.outputIndexWidth
    val kTileLimit = BigInt(1) << cfg.feeder.kTileCountWidth
    val beatLimit = BigInt(1) << matrix.beatCountWidth
    val hiddenKTiles = BigInt(model.hiddenSize / matrix.base.siteCount)
    val downKTiles = BigInt(model.ffnSize / matrix.base.siteCount)
    val kvFeatures = BigInt(model.kvHeadCount) * model.headSize
    val gateUpFeatures = BigInt(model.ffnSize) * 2
    val projectionShapes = Seq(
      BigInt(model.hiddenSize) -> hiddenKTiles,
      gateUpFeatures -> hiddenKTiles,
      BigInt(model.hiddenSize) -> downKTiles,
      kvFeatures -> hiddenKTiles)
    projectionShapes.foreach { case (outputFeatures, kTiles) =>
      require(outputFeatures > 0 && outputFeatures < outputLimit)
      require(outputFeatures % 8 == 0)
      require(kTiles > 0 && kTiles < kTileLimit)
      val prefillExpected = (outputFeatures / 4) * kTiles * 2
      val decodeExpected = (outputFeatures /
        (4 * model.engineCount)) * kTiles * 2
      require(prefillExpected >= 2 && prefillExpected < beatLimit)
      require(decodeExpected >= 2 && decodeExpected < beatLimit)
    }

    require(model.vocabularySize < Int.MaxValue)
    val physicalLmRows = (BigInt(model.vocabularySize) +
      model.engineCount - 1) / model.engineCount
    val lmExpected = physicalLmRows * hiddenKTiles
    require(physicalLmRows > 0 && physicalLmRows < outputLimit)
    require(hiddenKTiles > 0 && hiddenKTiles < kTileLimit)
    require(lmExpected > 0 && lmExpected < beatLimit)
  }
}

/** Synthesizable, combinational conversion between compact and legacy forms. */
object BitNetResidentLinearProgramCodec {
  private def kindSupported(kind: BitNetResidentLinearKind.C): Bool =
    kind === BitNetResidentLinearKind.AttentionOutput ||
      kind === BitNetResidentLinearKind.GateUp ||
      kind === BitNetResidentLinearKind.Down ||
      kind === BitNetResidentLinearKind.LmHead ||
      kind === BitNetResidentLinearKind.QkvQuery ||
      kind === BitNetResidentLinearKind.QkvKey ||
      kind === BitNetResidentLinearKind.QkvValue

  /** Lossy wiring operation; use `launchEncodable` before accepting it. */
  def encode(
      launch: BitNetResidentLinearLaunch,
      cfg: BitNetResidentLinearEndpointConfig):
      BitNetResidentLinearProgram = {
    BitNetResidentLinearProgram.requireCanonical(cfg)
    val program = BitNetResidentLinearProgram(cfg)
    program.requestId := launch.requestId
    program.layer := launch.layer
    program.kind := launch.kind
    program.mode := launch.mode
    program.tokenMask := launch.tokenMask
    program.sequenceId := launch.sequenceId
    program.taskIdBase := launch.matrixTask.taskIdBase
    program.tokenBase := launch.matrixTask.tokenId(0).resized
    for (lane <- 0 until cfg.model.tokenParallelism) {
      program.activationFactorQ16(lane) :=
        launch.matrixTask.activationFactorQ16(lane)
    }
    for (engine <- 0 until cfg.model.engineCount) {
      program.weightScaleQ16(engine) :=
        launch.matrixTask.weightScaleQ16(engine)
    }
    program
  }

  /**
    * Rebuilds every omitted field from immutable production geometry.
    *
    * LM-head is always lowered to the physical Decode schedule.  Its four
    * token-id entries broadcast `tokenBase`; projection entries are
    * `tokenBase + lane`.
    */
  def expand(
      program: BitNetResidentLinearProgram,
      cfg: BitNetResidentLinearEndpointConfig):
      BitNetResidentLinearLaunch = {
    BitNetResidentLinearProgram.requireCanonical(cfg)
    val model = cfg.model
    val base = cfg.matrix.base
    val launch = BitNetResidentLinearLaunch(cfg)
    val task = launch.matrixTask

    val isLm = program.kind === BitNetResidentLinearKind.LmHead
    val physicalMode = BitNetMode()
    physicalMode := program.mode
    when(isLm) { physicalMode := BitNetMode.Decode }

    launch.requestId := program.requestId
    launch.layer := program.layer
    launch.kind := program.kind
    launch.mode := physicalMode
    launch.tokenMask := program.tokenMask
    launch.sequenceId := program.sequenceId

    task.scheduleMode := physicalMode
    task.operation := BitNetContinuousDotKind.Projection
    when(isLm) { task.operation := BitNetContinuousDotKind.LmHead }
    task.sequenceId := program.sequenceId
    task.taskIdBase := program.taskIdBase
    task.activeTokenMask := program.tokenMask
    task.routeFinalToDynamicQuant := False
    task.residualEnable := False

    val widenedTokenBase = program.tokenBase.resize(base.tokenIdWidth)
    for (lane <- 0 until model.tokenParallelism) {
      task.tokenId(lane) :=
        (widenedTokenBase + U(lane, base.tokenIdWidth bits)).resized
      when(isLm) { task.tokenId(lane) := widenedTokenBase }
      task.activationFactorQ16(lane) := program.activationFactorQ16(lane)
    }

    for (engine <- 0 until model.engineCount) {
      task.weightScaleQ16(engine) := program.weightScaleQ16(engine)
    }

    // These equations execute only while Scala elaborates the fixed model.
    // The emitted RTL is a kind/mode constant table: it contains no geometry
    // divider, multiplier, variable shift or wide expected-beat comparator.
    val hiddenKTiles = model.hiddenSize / base.siteCount
    val downKTiles = model.ffnSize / base.siteCount
    val kvFeatures = model.kvHeadCount * model.headSize
    val lmRowsPerEngine = (model.vocabularySize + model.engineCount - 1) /
      model.engineCount

    val staticKTiles = UInt(cfg.feeder.kTileCountWidth bits)
    val staticPrefillEnd = UInt(base.outputIndexWidth bits)
    val staticPrefillExpected = UInt(cfg.matrix.beatCountWidth bits)
    val staticDecodeEnd0 = UInt(base.outputIndexWidth bits)
    val staticDecodeBase1 = UInt(base.outputIndexWidth bits)
    val staticDecodeEnd1 = UInt(base.outputIndexWidth bits)
    val staticDecodeExpected = UInt(cfg.matrix.beatCountWidth bits)

    // Qkv/AttentionOutput/QkvQuery share the historical hidden-to-hidden
    // default.  The switch below replaces it only for distinct geometries.
    staticKTiles := hiddenKTiles
    staticPrefillEnd := model.hiddenSize
    staticPrefillExpected :=
      BigInt(model.hiddenSize / 4) * hiddenKTiles * 2
    staticDecodeEnd0 := model.hiddenSize / 2
    staticDecodeBase1 := model.hiddenSize / 2
    staticDecodeEnd1 := model.hiddenSize
    staticDecodeExpected := BigInt(model.hiddenSize /
      (4 * model.engineCount)) * hiddenKTiles * 2
    switch(program.kind) {
      is(BitNetResidentLinearKind.GateUp) {
        staticKTiles := hiddenKTiles
        staticPrefillEnd := 2 * model.ffnSize
        staticPrefillExpected := BigInt(model.ffnSize / 2) *
          hiddenKTiles * 2
        staticDecodeEnd0 := model.ffnSize
        staticDecodeBase1 := model.ffnSize
        staticDecodeEnd1 := 2 * model.ffnSize
        staticDecodeExpected := BigInt(model.ffnSize /
          (4 * model.engineCount)) * hiddenKTiles * 2
      }
      is(BitNetResidentLinearKind.Down) {
        staticKTiles := downKTiles
        staticPrefillEnd := model.hiddenSize
        staticPrefillExpected := BigInt(model.hiddenSize / 4) *
          downKTiles * 2
        staticDecodeEnd0 := model.hiddenSize / 2
        staticDecodeBase1 := model.hiddenSize / 2
        staticDecodeEnd1 := model.hiddenSize
        staticDecodeExpected := BigInt(model.hiddenSize /
          (4 * model.engineCount)) * downKTiles * 2
      }
      is(BitNetResidentLinearKind.QkvKey,
          BitNetResidentLinearKind.QkvValue) {
        staticKTiles := hiddenKTiles
        staticPrefillEnd := kvFeatures
        staticPrefillExpected := BigInt(kvFeatures / 4) *
          hiddenKTiles * 2
        staticDecodeEnd0 := kvFeatures / 2
        staticDecodeBase1 := kvFeatures / 2
        staticDecodeEnd1 := kvFeatures
        staticDecodeExpected := BigInt(kvFeatures /
          (4 * model.engineCount)) * hiddenKTiles * 2
      }
      is(BitNetResidentLinearKind.LmHead) {
        staticKTiles := hiddenKTiles
        staticPrefillEnd := lmRowsPerEngine
        staticPrefillExpected := BigInt(lmRowsPerEngine) * hiddenKTiles
        staticDecodeEnd0 := lmRowsPerEngine
        staticDecodeBase1 := 0
        staticDecodeEnd1 := lmRowsPerEngine
        staticDecodeExpected := BigInt(lmRowsPerEngine) * hiddenKTiles
      }
    }

    task.kTilesPerOutput := staticKTiles
    switch(program.mode) {
      is(BitNetMode.Prefill) {
        for (engine <- 0 until model.engineCount) {
          task.outputBase(engine) := 0
          task.outputEndExclusive(engine) := staticPrefillEnd
        }
        task.expectedBeats := staticPrefillExpected
      }
      is(BitNetMode.Decode) {
        task.outputBase(0) := 0
        task.outputEndExclusive(0) := (if (model.engineCount == 1)
          staticPrefillEnd else staticDecodeEnd0)
        if (model.engineCount > 1) {
          task.outputBase(1) := staticDecodeBase1
          task.outputEndExclusive(1) := staticDecodeEnd1
        }
        task.expectedBeats := staticDecodeExpected
      }
    }

    launch
  }

  /**
    * Compact canonical checks only.  Geometry is trusted because `kind`
    * selects an immutable constant-table entry; full legacy authentication is
    * intentionally confined to `launchEncodable` at the conversion boundary.
    */
  def programLegal(
      program: BitNetResidentLinearProgram,
      cfg: BitNetResidentLinearEndpointConfig): Bool = {
    val prefill = program.mode === BitNetMode.Prefill
    val decode = program.mode === BitNetMode.Decode
    val prefixMask = (1 to cfg.model.tokenParallelism).map { count =>
      program.tokenMask === B((BigInt(1) << count) - 1,
        cfg.model.tokenParallelism bits)
    }.reduce(_ || _)
    val modeMaskLegal = (prefill && prefixMask) ||
      (decode && program.tokenMask ===
        B(1, cfg.model.tokenParallelism bits))
    val lmPhysicalLegal =
      program.kind =/= BitNetResidentLinearKind.LmHead || decode
    val prefillScaleLegal = !prefill ||
      (1 until cfg.model.engineCount).map { engine =>
        program.weightScaleQ16(engine) === program.weightScaleQ16(0)
      }.foldLeft(True)(_ && _)

    kindSupported(program.kind) &&
      program.layer < cfg.model.layerCount && modeMaskLegal &&
      lmPhysicalLegal && prefillScaleLegal
  }

  /**
    * Rejects every non-canonical legacy descriptor, even if the legacy
    * protocol alone could execute it safely.  Therefore accepted compression
    * can never change task metadata or geometry.
    */
  def launchEncodable(
      launch: BitNetResidentLinearLaunch,
      cfg: BitNetResidentLinearEndpointConfig): Bool = {
    val encoded = encode(launch, cfg)
    val roundTrip = expand(encoded, cfg)
    kindSupported(launch.kind) &&
      BitNetResidentLinearProtocol.launchLegal(launch, cfg) &&
      roundTrip.asBits === launch.asBits
  }
}

/** Pure-Scala legality and round-trip oracle for generators and tests. */
object BitNetResidentLinearProgramOracle {
  case class Program(
      requestId: BigInt,
      layer: Int,
      kind: BitNetResidentLinearKind.E,
      mode: BitNetMode.E,
      tokenMask: Int,
      sequenceId: Int,
      taskIdBase: Int,
      tokenBase: Int,
      activationFactorQ16: Vector[BigInt],
      weightScaleQ16: Vector[BigInt])

  case class LegacyTask(
      scheduleMode: BitNetMode.E,
      operation: BitNetContinuousDotKind.E,
      sequenceId: Int,
      taskIdBase: Int,
      expectedBeats: Int,
      kTilesPerOutput: Int,
      activeTokenMask: Int,
      tokenId: Vector[Int],
      activationFactorQ16: Vector[BigInt],
      weightScaleQ16: Vector[BigInt],
      outputBase: Vector[Int],
      outputEndExclusive: Vector[Int],
      routeFinalToDynamicQuant: Boolean,
      residualEnable: Boolean)

  case class LegacyLaunch(
      requestId: BigInt,
      layer: Int,
      kind: BitNetResidentLinearKind.E,
      mode: BitNetMode.E,
      tokenMask: Int,
      sequenceId: Int,
      matrixTask: LegacyTask)

  case class Geometry(
      inputFeatures: Int,
      outputFeatures: Int,
      projection: Boolean)

  private def fitsUnsigned(value: BigInt, width: Int): Boolean =
    value >= 0 && value < (BigInt(1) << width)

  private def fitsUnsigned(value: Int, width: Int): Boolean =
    value >= 0 && BigInt(value) < (BigInt(1) << width)

  def geometry(
      kind: BitNetResidentLinearKind.E,
      cfg: BitNetResidentLinearEndpointConfig =
        BitNetResidentLinearEndpointConfig()): Option[Geometry] = {
    val model = cfg.model
    kind match {
      case BitNetResidentLinearKind.AttentionOutput =>
        Some(Geometry(model.hiddenSize, model.hiddenSize, projection = true))
      case BitNetResidentLinearKind.GateUp =>
        Some(Geometry(model.hiddenSize, 2 * model.ffnSize, projection = true))
      case BitNetResidentLinearKind.Down =>
        Some(Geometry(model.ffnSize, model.hiddenSize, projection = true))
      case BitNetResidentLinearKind.QkvQuery =>
        Some(Geometry(model.hiddenSize, model.hiddenSize, projection = true))
      case BitNetResidentLinearKind.QkvKey |
           BitNetResidentLinearKind.QkvValue =>
        Some(Geometry(model.hiddenSize,
          model.kvHeadCount * model.headSize, projection = true))
      case BitNetResidentLinearKind.LmHead =>
        Some(Geometry(model.hiddenSize,
          (model.vocabularySize + 1) / 2, projection = false))
      case BitNetResidentLinearKind.Qkv => None
    }
  }

  def expand(
      program: Program,
      cfg: BitNetResidentLinearEndpointConfig =
        BitNetResidentLinearEndpointConfig()): LegacyLaunch = {
    BitNetResidentLinearProgram.requireCanonical(cfg)
    val model = cfg.model
    val matrix = cfg.matrix
    val isLm = program.kind == BitNetResidentLinearKind.LmHead
    val physicalMode = if (isLm) BitNetMode.Decode else program.mode
    // Atomic Qkv remains illegal/absent from `geometry`, but its diagnostic
    // expansion mirrors the RTL's historical hidden-to-hidden default.
    val shape = geometry(program.kind, cfg).getOrElse(
      Geometry(model.hiddenSize, model.hiddenSize, projection = true))
    val kTiles = shape.inputFeatures / matrix.base.siteCount

    val outputBase = if (shape.projection && physicalMode == BitNetMode.Decode)
      Vector(0, shape.outputFeatures / 2)
    else Vector(0, 0)
    val outputEnd = if (shape.projection && physicalMode == BitNetMode.Decode)
      Vector(shape.outputFeatures / 2, shape.outputFeatures)
    else Vector(shape.outputFeatures, shape.outputFeatures)

    val expected = if (isLm) shape.outputFeatures * kTiles
    else {
      val groups = if (physicalMode == BitNetMode.Prefill)
        shape.outputFeatures / 4 else shape.outputFeatures / 8
      groups * kTiles * 2
    }
    val tokenIds = Vector.tabulate(model.tokenParallelism) { lane =>
      if (isLm) program.tokenBase else program.tokenBase + lane
    }
    val operation = if (isLm) BitNetContinuousDotKind.LmHead
      else BitNetContinuousDotKind.Projection

    LegacyLaunch(
      requestId = program.requestId,
      layer = program.layer,
      kind = program.kind,
      mode = physicalMode,
      tokenMask = program.tokenMask,
      sequenceId = program.sequenceId,
      matrixTask = LegacyTask(
        scheduleMode = physicalMode,
        operation = operation,
        sequenceId = program.sequenceId,
        taskIdBase = program.taskIdBase,
        expectedBeats = expected,
        kTilesPerOutput = kTiles,
        activeTokenMask = program.tokenMask,
        tokenId = tokenIds,
        activationFactorQ16 = program.activationFactorQ16,
        weightScaleQ16 = program.weightScaleQ16,
        outputBase = outputBase,
        outputEndExclusive = outputEnd,
        routeFinalToDynamicQuant = false,
        residualEnable = false))
  }

  /** Mechanical extraction; callers must check `launchEncodable`. */
  def encode(launch: LegacyLaunch): Program = {
    require(launch.matrixTask.tokenId.nonEmpty)
    Program(
      requestId = launch.requestId,
      layer = launch.layer,
      kind = launch.kind,
      mode = launch.mode,
      tokenMask = launch.tokenMask,
      sequenceId = launch.sequenceId,
      taskIdBase = launch.matrixTask.taskIdBase,
      tokenBase = launch.matrixTask.tokenId.head,
      activationFactorQ16 = launch.matrixTask.activationFactorQ16,
      weightScaleQ16 = launch.matrixTask.weightScaleQ16)
  }

  private def programFieldsFit(
      program: Program,
      cfg: BitNetResidentLinearEndpointConfig): Boolean = {
    fitsUnsigned(program.requestId, 32) &&
      fitsUnsigned(program.layer, 5) &&
      fitsUnsigned(program.tokenMask, 4) &&
      fitsUnsigned(program.sequenceId, 12) &&
      fitsUnsigned(program.taskIdBase, 20) &&
      fitsUnsigned(program.tokenBase, 13) &&
      program.activationFactorQ16.size == 4 &&
      program.activationFactorQ16.forall(fitsUnsigned(_, 32)) &&
      program.weightScaleQ16.size == 2 &&
      program.weightScaleQ16.forall(fitsUnsigned(_, 32)) &&
      geometry(program.kind, cfg).nonEmpty
  }

  private def legacyFieldsFit(
      launch: LegacyLaunch,
      cfg: BitNetResidentLinearEndpointConfig): Boolean = {
    val task = launch.matrixTask
    val matrix = cfg.matrix
    val base = matrix.base
    fitsUnsigned(launch.requestId, cfg.model.requestIdWidth) &&
      fitsUnsigned(launch.layer, cfg.model.layerWidth) &&
      fitsUnsigned(launch.tokenMask, cfg.model.tokenParallelism) &&
      fitsUnsigned(launch.sequenceId, matrix.sequenceIdWidth) &&
      fitsUnsigned(task.sequenceId, matrix.sequenceIdWidth) &&
      fitsUnsigned(task.taskIdBase, matrix.taskIdWidth) &&
      fitsUnsigned(task.expectedBeats, matrix.beatCountWidth) &&
      fitsUnsigned(task.kTilesPerOutput, cfg.feeder.kTileCountWidth) &&
      fitsUnsigned(task.activeTokenMask, cfg.feeder.tokenLanes) &&
      task.tokenId.size == cfg.feeder.tokenLanes &&
      task.tokenId.forall(fitsUnsigned(_, base.tokenIdWidth)) &&
      task.activationFactorQ16.size == cfg.feeder.tokenLanes &&
      task.activationFactorQ16.forall(fitsUnsigned(_, 32)) &&
      task.weightScaleQ16.size == cfg.feeder.engineCount &&
      task.weightScaleQ16.forall(fitsUnsigned(_, 32)) &&
      task.outputBase.size == cfg.feeder.engineCount &&
      task.outputBase.forall(fitsUnsigned(_, base.outputIndexWidth)) &&
      task.outputEndExclusive.size == cfg.feeder.engineCount &&
      task.outputEndExclusive.forall(
        fitsUnsigned(_, base.outputIndexWidth))
  }

  /** Pure mirror of BitNetResidentLinearProtocol.launchLegal. */
  def legacyLegal(
      launch: LegacyLaunch,
      cfg: BitNetResidentLinearEndpointConfig =
        BitNetResidentLinearEndpointConfig()): Boolean = {
    BitNetResidentLinearProgram.requireCanonical(cfg)
    if (!legacyFieldsFit(launch, cfg)) return false

    val task = launch.matrixTask
    val model = cfg.model
    val base = cfg.matrix.base
    val prefill = launch.mode == BitNetMode.Prefill
    val decode = launch.mode == BitNetMode.Decode
    val prefixMask = Set(1, 3, 7, 15).contains(launch.tokenMask)
    val modeMaskLegal = (prefill && prefixMask) ||
      (decode && launch.tokenMask == 1)
    val shape = geometry(launch.kind, cfg)
    val projection = shape.exists(_.projection)
    val segmented = Set(
      BitNetResidentLinearKind.QkvQuery,
      BitNetResidentLinearKind.QkvKey,
      BitNetResidentLinearKind.QkvValue).contains(launch.kind)
    val operationLegal =
      (projection && task.operation == BitNetContinuousDotKind.Projection) ||
        (launch.kind == BitNetResidentLinearKind.LmHead &&
          task.operation == BitNetContinuousDotKind.LmHead)
    val lmModeLegal = launch.kind != BitNetResidentLinearKind.LmHead || decode
    val envelopeLegal = launch.kind != BitNetResidentLinearKind.Qkv &&
      launch.layer < model.layerCount &&
      launch.mode == task.scheduleMode &&
      launch.tokenMask == task.activeTokenMask &&
      launch.sequenceId == task.sequenceId &&
      !task.routeFinalToDynamicQuant && !task.residualEnable

    val spans = task.outputBase.indices.map { engine =>
      BigInt(task.outputEndExclusive(engine)) -
        BigInt(task.outputBase(engine))
    }.toVector
    val boundsLegal = spans.forall(_ > 0)
    val groups = spans.map(span => if (projection) (span + 1) / 2 else span)
    val expectedMatches = groups.forall(_ * BigInt(task.kTilesPerOutput) ==
      BigInt(task.expectedBeats))
    val prefillShared = task.outputBase(0) == task.outputBase(1) &&
      task.outputEndExclusive(0) == task.outputEndExclusive(1) &&
      task.weightScaleQ16(0) == task.weightScaleQ16(1)
    val decodeBalanced = groups(0) == groups(1)
    val projectionAlignment = !projection ||
      (task.outputBase.forall(_ % 4 == 0) &&
        task.outputEndExclusive.forall(_ % 4 == 0) &&
        task.expectedBeats % 2 == 0 && task.expectedBeats >= 2)
    val gateUpPrefillShape =
      launch.kind != BitNetResidentLinearKind.GateUp || !prefill ||
        (task.outputBase(0) % 4 == 0 && spans(0) % 4 == 0)

    val segmentedShape = if (!segmented) true else {
      val outputFeatures = shape.get.outputFeatures
      val prefillShape = task.outputBase == Vector(0, 0) &&
        task.outputEndExclusive == Vector(outputFeatures, outputFeatures)
      val decodeShape = task.outputBase == Vector(0, outputFeatures / 2) &&
        task.outputEndExclusive ==
          Vector(outputFeatures / 2, outputFeatures)
      task.kTilesPerOutput == model.hiddenSize / base.siteCount &&
        ((prefill && prefillShape) || (decode && decodeShape))
    }

    envelopeLegal && modeMaskLegal && operationLegal && lmModeLegal &&
      task.expectedBeats != 0 && task.kTilesPerOutput != 0 && boundsLegal &&
      expectedMatches && ((prefill && prefillShared) ||
        (decode && decodeBalanced)) && projectionAlignment &&
      gateUpPrefillShape && segmentedShape
  }

  def programLegal(
      program: Program,
      cfg: BitNetResidentLinearEndpointConfig =
        BitNetResidentLinearEndpointConfig()): Boolean =
    programFieldsFit(program, cfg) && {
      val expanded = expand(program, cfg)
      legacyLegal(expanded, cfg) && encode(expanded) == program
    }

  def launchEncodable(
      launch: LegacyLaunch,
      cfg: BitNetResidentLinearEndpointConfig =
        BitNetResidentLinearEndpointConfig()): Boolean =
    legacyLegal(launch, cfg) && geometry(launch.kind, cfg).nonEmpty && {
      val encoded = encode(launch)
      programLegal(encoded, cfg) && expand(encoded, cfg) == launch
    }
}
