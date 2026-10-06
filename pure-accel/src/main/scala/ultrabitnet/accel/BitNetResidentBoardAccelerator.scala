package ultrabitnet.accel

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axilite._
import spinal.lib.bus.amba4.axis._

import scala.language.postfixOps

/** Board-facing ABI for the retained TP2 resident inference core. */
case class BitNetResidentBoardAcceleratorConfig(
    core: BitNetResidentSessionInferenceTopConfig =
      BitNetResidentSessionInferenceTopConfig(),
    controlAddressWidth: Int = 8) {
  val model: BitNetConfig = core.shell.model
  val preload: BitNetResidentHiddenPreloadEndpointConfig = core.session.preload
  val crossbar: BitNetBankLocalCrossbarConfig = core.shell.crossbar
  val readCfg: BitNetAxiBankReadAdapterConfig = core.shell.memory.memory.read(0)
  val writeCfg: BitNetAxiBankWriteAdapterConfig = core.shell.memory.memory.write(0)

  require(model.tokenParallelism == 2,
    "the AXU3EG R10 board image is fixed to TP2")
  require(model.spuCount == 4 && preload.groupCount == 640)
  require(crossbar.bankCount == 5 && crossbar.dataWidth == 128)
  require(readCfg.axiAddressWidth == 40 && writeCfg.axiAddressWidth == 40)
  require(readCfg.axiIdWidth == 4 && writeCfg.axiIdWidth == 4)
  require(controlAddressWidth >= 8)

  val controlAxi: AxiLite4Config = AxiLite4Config(
    addressWidth = controlAddressWidth, dataWidth = 32)
  // Axi4StreamConfig expresses TDATA width in bytes.
  val hiddenAxis: Axi4StreamConfig = Axi4StreamConfig(
    dataWidth = model.tokenParallelism * model.spuCount * 4,
    useLast = true)

  private def bankAxi(addressWidth: Int, idWidth: Int): Axi4Config = Axi4Config(
    addressWidth = addressWidth,
    dataWidth = crossbar.dataWidth,
    idWidth = idWidth,
    useId = true,
    useRegion = false,
    useBurst = true,
    useLock = true,
    useCache = true,
    useSize = true,
    useQos = true,
    useLen = true,
    useLast = true,
    useResp = true,
    useProt = true,
    useStrb = true,
    useAllStrb = false)

  // All five resident banks use hard PS DDR ports in the first routable board
  // image.  Bank 0 is translated into the unused high-DDR range at 0x8_5000_0000;
  // this removes the PL DDR4 MIG and clock converter from the nearly-full die.
  val psBankAxi: Axi4Config = bankAxi(addressWidth = 49, idWidth = 6)
  val plBankAxi: Axi4Config = psBankAxi
  val bank0PhysicalBase: BigInt = BigInt("850000000", 16)
}

object BitNetResidentBoardRegister {
  val Identity = 0x00
  val Submit = 0x04
  val ResultAck = 0x08
  val IrqEnable = 0x0c
  val SessionId = 0x10
  val ImageEpoch = 0x14
  val RoundFlags = 0x18
  val PositionBase = 0x1c
  val TokenCount = 0x20
  val PsSourceBaseLow = 0x24
  val PsSourceBaseHigh = 0x28
  val PsSourceBytesLow = 0x2c
  val PsSourceBytesHigh = 0x30
  val Status = 0x34
  val HiddenGroup = 0x38
  val HiddenRequestId = 0x3c
  val ResultStatus = 0x40
  val ResultSessionId = 0x44
  val ResultRequestId = 0x48
  val ResultTokenId = 0x4c
  val ResultFault = 0x50
  val ResultPosition = 0x54
  val ShellFaultLow = 0x58
  val ShellFaultHigh = 0x5c
  val HiddenData0 = 0x80
  val HiddenLast = 0xa0
  val HiddenPush = 0xa4
  val HiddenPioStatus = 0xa8

  val IdentityValue = BigInt("48425432", 16) // Fixed board ABI identity; retain compatibility with the released image.
}

/** Thin combinational mapping from the checked native bank ABI to AXI4. */
object BitNetResidentBoardAxiBoundary {
  private def connectBank(
      cfg: BitNetResidentBoardAcceleratorConfig,
      native: BitNetResidentSessionInferenceTop,
      bank: Int,
      axi: Axi4,
      externalIdWidth: Int,
      externalAddressBase: BigInt = 0): Unit = {
    axi.ar.valid := native.io.axiAr(bank).valid
    native.io.axiAr(bank).ready := axi.ar.ready
    axi.ar.addr := native.io.axiAr(bank).address.resize(axi.config.addressWidth) +
      U(externalAddressBase, axi.config.addressWidth bits)
    axi.ar.id := native.io.axiAr(bank).id.resized
    axi.ar.len := native.io.axiAr(bank).len
    axi.ar.size := U(4, 3 bits)
    axi.ar.burst := B"01"
    axi.ar.lock := 0
    axi.ar.cache := 0
    axi.ar.prot := 0
    axi.ar.qos := 0

    native.io.axiR(bank).valid := axi.r.valid
    axi.r.ready := native.io.axiR(bank).ready
    if (externalIdWidth == cfg.readCfg.axiIdWidth) {
      native.io.axiR(bank).id := axi.r.id
    } else {
      native.io.axiR(bank).id := Mux(
        axi.r.id(externalIdWidth - 1 downto cfg.readCfg.axiIdWidth).orR,
        U((1 << cfg.readCfg.axiIdWidth) - 1, cfg.readCfg.axiIdWidth bits),
        axi.r.id(cfg.readCfg.axiIdWidth - 1 downto 0))
    }
    native.io.axiR(bank).data := axi.r.data
    native.io.axiR(bank).resp := axi.r.resp
    native.io.axiR(bank).last := axi.r.last

    axi.aw.valid := native.io.axiAw(bank).valid
    native.io.axiAw(bank).ready := axi.aw.ready
    axi.aw.addr := native.io.axiAw(bank).address.resize(axi.config.addressWidth) +
      U(externalAddressBase, axi.config.addressWidth bits)
    axi.aw.id := native.io.axiAw(bank).id.resized
    axi.aw.len := native.io.axiAw(bank).len
    axi.aw.size := U(4, 3 bits)
    axi.aw.burst := B"01"
    axi.aw.lock := 0
    axi.aw.cache := 0
    axi.aw.prot := 0
    axi.aw.qos := 0

    axi.w.valid := native.io.axiW(bank).valid
    native.io.axiW(bank).ready := axi.w.ready
    axi.w.data := native.io.axiW(bank).data
    axi.w.strb := native.io.axiW(bank).strobe
    axi.w.last := native.io.axiW(bank).last

    native.io.axiB(bank).valid := axi.b.valid
    axi.b.ready := native.io.axiB(bank).ready
    if (externalIdWidth == cfg.writeCfg.axiIdWidth) {
      native.io.axiB(bank).id := axi.b.id
    } else {
      native.io.axiB(bank).id := Mux(
        axi.b.id(externalIdWidth - 1 downto cfg.writeCfg.axiIdWidth).orR,
        U((1 << cfg.writeCfg.axiIdWidth) - 1, cfg.writeCfg.axiIdWidth bits),
        axi.b.id(cfg.writeCfg.axiIdWidth - 1 downto 0))
    }
    native.io.axiB(bank).resp := axi.b.resp
  }

  def connect(
      cfg: BitNetResidentBoardAcceleratorConfig,
      native: BitNetResidentSessionInferenceTop,
      pl: Axi4,
      ps: Vec[Axi4]): Unit = {
    require(ps.length == 4)
    connectBank(cfg, native, 0, pl, cfg.plBankAxi.idWidth,
      cfg.bank0PhysicalBase)
    for (bank <- 1 until cfg.crossbar.bankCount) {
      connectBank(cfg, native, bank, ps(bank - 1), cfg.psBankAxi.idWidth)
    }
  }
}

/** Vivado module-reference metadata for clocks, control, stream and DDR AXI. */
object BitNetResidentBoardVivadoMetadata {
  private def mark(signal: BaseType, interfaceName: String,
                   channel: String, kind: String): Unit =
    signal.addAttribute("X_INTERFACE_INFO",
      s"xilinx.com:interface:$kind:1.0 $interfaceName $channel")

  private def markSignal(signal: BaseType, interfaceName: String,
                         channel: String, kind: String): Unit =
    signal.addAttribute("X_INTERFACE_INFO",
      s"xilinx.com:signal:$kind:1.0 $interfaceName $channel")

  private def annotateAxi4(axi: Axi4, name: String,
                           addressWidth: Int, idWidth: Int): Unit = {
    Seq[(BaseType, String)](
      axi.aw.addr -> "AWADDR", axi.aw.id -> "AWID",
      axi.aw.len -> "AWLEN", axi.aw.size -> "AWSIZE",
      axi.aw.burst -> "AWBURST", axi.aw.lock -> "AWLOCK",
      axi.aw.cache -> "AWCACHE", axi.aw.prot -> "AWPROT",
      axi.aw.qos -> "AWQOS", axi.aw.valid -> "AWVALID",
      axi.aw.ready -> "AWREADY", axi.w.data -> "WDATA",
      axi.w.strb -> "WSTRB", axi.w.last -> "WLAST",
      axi.w.valid -> "WVALID", axi.w.ready -> "WREADY",
      axi.b.id -> "BID", axi.b.resp -> "BRESP",
      axi.b.valid -> "BVALID", axi.b.ready -> "BREADY",
      axi.ar.addr -> "ARADDR", axi.ar.id -> "ARID",
      axi.ar.len -> "ARLEN", axi.ar.size -> "ARSIZE",
      axi.ar.burst -> "ARBURST", axi.ar.lock -> "ARLOCK",
      axi.ar.cache -> "ARCACHE", axi.ar.prot -> "ARPROT",
      axi.ar.qos -> "ARQOS", axi.ar.valid -> "ARVALID",
      axi.ar.ready -> "ARREADY", axi.r.data -> "RDATA",
      axi.r.id -> "RID", axi.r.resp -> "RRESP",
      axi.r.last -> "RLAST", axi.r.valid -> "RVALID",
      axi.r.ready -> "RREADY").foreach { case (signal, channel) =>
      mark(signal, name, channel, "aximm")
    }
    axi.aw.addr.addAttribute("X_INTERFACE_PARAMETER",
      s"XIL_INTERFACENAME $name, PROTOCOL AXI4, " +
        s"ADDR_WIDTH $addressWidth, DATA_WIDTH 128, ID_WIDTH $idWidth, " +
        "READ_WRITE_MODE READ_WRITE, HAS_BURST 1, HAS_LOCK 1, " +
        "HAS_CACHE 1, HAS_PROT 1, HAS_QOS 1, MAX_BURST_LENGTH 256, " +
        "NUM_READ_OUTSTANDING 4, NUM_WRITE_OUTSTANDING 4, " +
        "SUPPORTS_NARROW_BURST 0")
  }

  private def annotateAxiLite(axi: AxiLite4, name: String,
                              addressWidth: Int): Unit = {
    Seq[(BaseType, String)](
      axi.aw.addr -> "AWADDR", axi.aw.prot -> "AWPROT",
      axi.aw.valid -> "AWVALID", axi.aw.ready -> "AWREADY",
      axi.w.data -> "WDATA", axi.w.strb -> "WSTRB",
      axi.w.valid -> "WVALID", axi.w.ready -> "WREADY",
      axi.b.resp -> "BRESP", axi.b.valid -> "BVALID",
      axi.b.ready -> "BREADY", axi.ar.addr -> "ARADDR",
      axi.ar.prot -> "ARPROT", axi.ar.valid -> "ARVALID",
      axi.ar.ready -> "ARREADY", axi.r.data -> "RDATA",
      axi.r.resp -> "RRESP", axi.r.valid -> "RVALID",
      axi.r.ready -> "RREADY").foreach { case (signal, channel) =>
      mark(signal, name, channel, "aximm")
    }
    axi.aw.addr.addAttribute("X_INTERFACE_PARAMETER",
      s"XIL_INTERFACENAME $name, PROTOCOL AXI4LITE, " +
        s"ADDR_WIDTH $addressWidth, DATA_WIDTH 32, READ_WRITE_MODE READ_WRITE")
  }

  private def annotateAxis(
      axis: Stream[Axi4Stream.Axi4StreamBundle], name: String): Unit = {
    mark(axis.data, name, "TDATA", "axis")
    mark(axis.last, name, "TLAST", "axis")
    mark(axis.valid, name, "TVALID", "axis")
    mark(axis.ready, name, "TREADY", "axis")
    axis.data.addAttribute("X_INTERFACE_PARAMETER",
      s"XIL_INTERFACENAME $name, TDATA_NUM_BYTES 32, HAS_TLAST 1, " +
        "HAS_TKEEP 0, HAS_TSTRB 0, HAS_TREADY 1")
  }

  def annotate(cfg: BitNetResidentBoardAcceleratorConfig,
               slowClk: Bool, slowResetn: Bool,
               fastClk: Bool, fastResetn: Bool,
               control: AxiLite4,
               hidden: Stream[Axi4Stream.Axi4StreamBundle],
               pl: Axi4, ps: Vec[Axi4], interrupt: Bool): Unit = {
    annotateAxiLite(control, "S_AXI_CONTROL", cfg.controlAddressWidth)
    annotateAxis(hidden, "S_AXIS_HIDDEN")
    annotateAxi4(pl, "M_AXI_PL", cfg.plBankAxi.addressWidth,
      cfg.plBankAxi.idWidth)
    for (index <- 0 until ps.length) {
      annotateAxi4(ps(index), s"M_AXI_PS$index", cfg.psBankAxi.addressWidth,
        cfg.psBankAxi.idWidth)
    }

    markSignal(slowClk, "SLOW_CLK", "CLK", "clock")
    slowClk.addAttribute("X_INTERFACE_PARAMETER",
      // The PS PLL/divider realizes the nominal 50 MHz PL0 clock as
      // 49.999500 MHz.  Vivado BD validation compares this integer exactly.
      "XIL_INTERFACENAME SLOW_CLK, FREQ_HZ 49999500, " +
        "ASSOCIATED_BUSIF S_AXI_CONTROL:S_AXIS_HIDDEN:M_AXI_PL:" +
        "M_AXI_PS0:M_AXI_PS1:M_AXI_PS2:M_AXI_PS3, " +
        "ASSOCIATED_RESET io_slowResetn")
    markSignal(slowResetn, "SLOW_RESETN", "RST", "reset")
    slowResetn.addAttribute("X_INTERFACE_PARAMETER",
      "XIL_INTERFACENAME SLOW_RESETN, POLARITY ACTIVE_LOW")
    markSignal(fastClk, "FAST_CLK", "CLK", "clock")
    fastClk.addAttribute("X_INTERFACE_PARAMETER",
      "XIL_INTERFACENAME FAST_CLK, FREQ_HZ 99999000, " +
        "ASSOCIATED_RESET io_fastResetn")
    markSignal(fastResetn, "FAST_RESETN", "RST", "reset")
    fastResetn.addAttribute("X_INTERFACE_PARAMETER",
      "XIL_INTERFACENAME FAST_RESETN, POLARITY ACTIVE_LOW")
    markSignal(interrupt, "INTERRUPT", "INTERRUPT", "interrupt")
    interrupt.addAttribute("X_INTERFACE_PARAMETER",
      "XIL_INTERFACENAME INTERRUPT, SENSITIVITY LEVEL_HIGH")
  }
}

/**
  * AXU3EGB module-reference top.  M_AXI_PL is retained as the stable bank-0
  * interface name, but in the resource-minimal board image it connects to PS
  * S_AXI_HPC0_FPD and emits addresses at 0x8_5000_0000.
  *
  * PS software owns UART, tokenization and embedding lookup.  One HiddenResident
  * round is 640 256-bit AXI-Stream beats: lane 0 shards 0..3 followed by lane 1
  * shards 0..3, all signed Q16.16.  TLAST must coincide with group 639.
  */
class BitNetResidentBoardAccelerator(
    cfg: BitNetResidentBoardAcceleratorConfig =
      BitNetResidentBoardAcceleratorConfig(),
    useVendorPrimitive: Boolean = true) extends Component {
  val io = new Bundle {
    val slowClk = in Bool()
    val slowResetn = in Bool()
    val fastClk = in Bool()
    val fastResetn = in Bool()

    val sAxiControl = slave(AxiLite4(cfg.controlAxi))
    val sAxisHidden = slave(Axi4Stream(cfg.hiddenAxis))
    val mAxiPl = master(Axi4(cfg.plBankAxi))
    val mAxiPs = Vec(master(Axi4(cfg.psBankAxi)), 4)
    val interrupt = out Bool()
  }

  val slowCd = ClockDomain(
    clock = io.slowClk,
    reset = io.slowResetn,
    config = ClockDomainConfig(
      clockEdge = RISING,
      resetKind = ASYNC,
      resetActiveLevel = LOW))

  val native = new BitNetResidentSessionInferenceTop(
    cfg.core, useVendorPrimitive = useVendorPrimitive)
  native.io.slowClk := io.slowClk
  native.io.slowResetn := io.slowResetn
  native.io.fastClk := io.fastClk
  native.io.fastResetn := io.fastResetn

  BitNetResidentBoardAxiBoundary.connect(
    cfg, native, io.mAxiPl, io.mAxiPs)
  BitNetResidentBoardVivadoMetadata.annotate(
    cfg, io.slowClk, io.slowResetn, io.fastClk, io.fastResetn,
    io.sAxiControl, io.sAxisHidden, io.mAxiPl, io.mAxiPs, io.interrupt)

  val slowArea = new ClockingArea(slowCd) {
    val bus = AxiLite4SlaveFactory(io.sAxiControl)

    val irqEnable = RegInit(False)
    val sessionId = Reg(UInt(cfg.core.sessionIdWidth bits)) init 0
    val imageEpoch = Reg(UInt(cfg.core.shell.staging.epochWidth bits)) init 0
    val roundMode = Reg(BitNetMode()) init BitNetMode.Prefill
    val lastPromptRound = RegInit(False)
    val newSession = RegInit(False)
    val positionBase = Reg(UInt(cfg.model.sequenceWidth bits)) init 0
    val tokenCount = Reg(UInt(cfg.model.roundTokenCountWidth bits)) init 0
    val psSourceBase = Reg(UInt(cfg.model.addressWidth bits)) init 0
    val psSourceBytes = Reg(UInt((cfg.model.addressWidth + 1) bits)) init 0

    bus.driveAndRead(irqEnable, BitNetResidentBoardRegister.IrqEnable)
    bus.driveAndRead(sessionId, BitNetResidentBoardRegister.SessionId)
    bus.driveAndRead(imageEpoch, BitNetResidentBoardRegister.ImageEpoch)
    bus.driveAndRead(roundMode,
      BitNetResidentBoardRegister.RoundFlags, 0)
    bus.driveAndRead(lastPromptRound,
      BitNetResidentBoardRegister.RoundFlags, 1)
    bus.driveAndRead(newSession,
      BitNetResidentBoardRegister.RoundFlags, 2)
    bus.driveAndRead(positionBase,
      BitNetResidentBoardRegister.PositionBase)
    bus.driveAndRead(tokenCount,
      BitNetResidentBoardRegister.TokenCount)
    bus.driveAndRead(psSourceBase(31 downto 0),
      BitNetResidentBoardRegister.PsSourceBaseLow)
    bus.driveAndRead(psSourceBase(cfg.model.addressWidth - 1 downto 32),
      BitNetResidentBoardRegister.PsSourceBaseHigh)
    bus.driveAndRead(psSourceBytes(31 downto 0),
      BitNetResidentBoardRegister.PsSourceBytesLow)
    bus.driveAndRead(psSourceBytes(cfg.model.addressWidth downto 32),
      BitNetResidentBoardRegister.PsSourceBytesHigh)

    val submitPending = RegInit(False)
    bus.onWrite(BitNetResidentBoardRegister.Submit) {
      when(!submitPending && !native.io.fatalFailStop) {
        submitPending := True
      }
    }
    native.io.roundSubmit.valid := submitPending
    native.io.roundSubmit.sessionId := sessionId
    native.io.roundSubmit.imageEpoch := imageEpoch
    native.io.roundSubmit.mode := roundMode
    native.io.roundSubmit.positionBase := positionBase
    native.io.roundSubmit.tokenCount := tokenCount
    native.io.roundSubmit.lastPromptRound := lastPromptRound
    native.io.roundSubmit.newSession := newSession
    native.io.roundSubmit.psSourceBase := psSourceBase
    native.io.roundSubmit.psSourceBytes := psSourceBytes
    when(native.io.roundSubmit.fire) {
      submitPending := False
    }

    // Low-resource PIO ingress for the first board image.  Software writes
    // eight words at HiddenData0..+0x1c, writes HiddenLast, then any value to
    // HiddenPush.  A held 256-bit beat remains stable under PL backpressure.
    val hiddenPioStaging = Vec.fill(8)(Reg(Bits(32 bits)) init 0)
    for (word <- 0 until 8) {
      bus.driveAndRead(hiddenPioStaging(word),
        BitNetResidentBoardRegister.HiddenData0 + word * 4)
    }
    val hiddenPioLastDriver = RegInit(False)
    bus.driveAndRead(hiddenPioLastDriver,
      BitNetResidentBoardRegister.HiddenLast)
    val hiddenPioPayload = Reg(Bits(256 bits)) init 0
    val hiddenPioLast = RegInit(False)
    val hiddenPioPending = RegInit(False)
    val hiddenPioOverrun = RegInit(False)
    bus.onWrite(BitNetResidentBoardRegister.HiddenPush) {
      when(!hiddenPioPending) {
        for (word <- 0 until 8) {
          hiddenPioPayload(word * 32 + 31 downto word * 32) :=
            hiddenPioStaging(word)
        }
        hiddenPioLast := hiddenPioLastDriver
        hiddenPioPending := True
      } otherwise {
        hiddenPioOverrun := True
      }
    }

    val hiddenGroup = Reg(UInt(cfg.preload.groupCountWidth bits)) init 0
    val hiddenSourceValid = hiddenPioPending || io.sAxisHidden.valid
    val hiddenSourceData = Bits(256 bits)
    val hiddenSourceLast = Bool()
    hiddenSourceData := Mux(hiddenPioPending,
      hiddenPioPayload, io.sAxisHidden.data)
    hiddenSourceLast := Mux(hiddenPioPending,
      hiddenPioLast, io.sAxisHidden.last)
    val hiddenFire = native.io.preloadInput.fire
    native.io.preloadInput.valid := hiddenSourceValid && native.io.hiddenReady
    io.sAxisHidden.ready := native.io.preloadInput.ready &&
      native.io.hiddenReady && !hiddenPioPending
    native.io.preloadInput.requestId := native.io.hiddenRequestId
    native.io.preloadInput.tokenMask := native.io.hiddenTokenMask
    native.io.preloadInput.group := hiddenGroup.resized
    native.io.preloadInput.first := hiddenGroup === 0
    native.io.preloadInput.last := hiddenSourceLast
    for (lane <- 0 until cfg.model.tokenParallelism;
         shard <- 0 until cfg.model.spuCount) {
      val word = lane * cfg.model.spuCount + shard
      native.io.preloadInput.values(lane)(shard) :=
        hiddenSourceData(word * 32 + 31 downto word * 32).asSInt
    }
    when(native.io.roundSubmit.fire) {
      hiddenGroup := 0
    }
    when(hiddenFire) {
      when(hiddenPioPending) {
        hiddenPioPending := False
      }
      when(hiddenSourceLast) {
        hiddenGroup := 0
      } otherwise {
        hiddenGroup := hiddenGroup + 1
      }
    }

    val resultValid = RegInit(False)
    val resultSessionId = Reg(UInt(cfg.core.sessionIdWidth bits)) init 0
    val resultRequestId = Reg(UInt(cfg.model.requestIdWidth bits)) init 0
    val resultRunEpoch = Reg(UInt(cfg.core.shell.event.epochWidth bits)) init 0
    val resultSuccess = RegInit(False)
    val resultFault = Reg(Bits(16 bits)) init 0
    val resultTokenValid = RegInit(False)
    val resultTokenId = Reg(UInt(
      cfg.core.shell.lm.resident.streaming.tokenIdWidth bits)) init 0
    val resultNextPosition = Reg(UInt(cfg.model.sequenceWidth bits)) init 0

    native.io.roundResult.ready := !resultValid
    bus.onWrite(BitNetResidentBoardRegister.ResultAck) {
      resultValid := False
      hiddenPioOverrun := False
    }
    when(native.io.roundResult.fire) {
      resultValid := True
      resultSessionId := native.io.roundResult.sessionId
      resultRequestId := native.io.roundResult.roundRequestId
      resultRunEpoch := native.io.roundResult.runEpoch
      resultSuccess := native.io.roundResult.success
      resultFault := native.io.roundResult.faultCode
      resultTokenValid := native.io.roundResult.tokenValid
      resultTokenId := native.io.roundResult.tokenId
      resultNextPosition := native.io.roundResult.nextPosition
    }

    val status = Bits(32 bits)
    status := 0
    status(0) := native.io.busy
    status(1) := native.io.hiddenReady
    status(2) := submitPending
    status(3) := resultValid
    status(4) := native.io.sessionValid
    status(5) := native.io.sessionQuarantined
    status(6) := native.io.fatalFailStop
    status(7) := native.io.protocolFaultSticky
    status(8) := native.io.shellBusy
    status(9) := native.io.roundControllerBusy

    val resultStatus = Bits(32 bits)
    resultStatus := 0
    resultStatus(0) := resultValid
    resultStatus(1) := resultSuccess
    resultStatus(2) := resultTokenValid
    resultStatus(15 downto 8) := resultRunEpoch.asBits

    val resultPosition = Bits(32 bits)
    resultPosition := 0
    resultPosition(cfg.model.sequenceWidth - 1 downto 0) :=
      resultNextPosition.asBits

    bus.read(B(BitNetResidentBoardRegister.IdentityValue, 32 bits),
      BitNetResidentBoardRegister.Identity)
    bus.read(status, BitNetResidentBoardRegister.Status)
    bus.read(hiddenGroup, BitNetResidentBoardRegister.HiddenGroup)
    bus.read(native.io.hiddenRequestId,
      BitNetResidentBoardRegister.HiddenRequestId)
    bus.read(resultStatus, BitNetResidentBoardRegister.ResultStatus)
    bus.read(resultSessionId,
      BitNetResidentBoardRegister.ResultSessionId)
    bus.read(resultRequestId,
      BitNetResidentBoardRegister.ResultRequestId)
    bus.read(resultTokenId,
      BitNetResidentBoardRegister.ResultTokenId)
    bus.read(resultFault,
      BitNetResidentBoardRegister.ResultFault)
    bus.read(resultPosition,
      BitNetResidentBoardRegister.ResultPosition)
    bus.read(native.io.shellFaultBits(31 downto 0),
      BitNetResidentBoardRegister.ShellFaultLow)
    bus.read(native.io.shellFaultBits(63 downto 32),
      BitNetResidentBoardRegister.ShellFaultHigh)

    val hiddenPioStatus = Bits(32 bits)
    hiddenPioStatus := 0
    hiddenPioStatus(0) := hiddenPioPending
    hiddenPioStatus(1) := hiddenPioOverrun
    hiddenPioStatus(2) := native.io.preloadInput.ready && native.io.hiddenReady
    bus.read(hiddenPioStatus,
      BitNetResidentBoardRegister.HiddenPioStatus)

    io.interrupt := resultValid && irqEnable
  }
}

object GenerateBitNetResidentBoardAccelerator extends App {
  val target = args.headOption.getOrElse(
    "../build/rtl")
  SpinalConfig(targetDirectory = target, oneFilePerComponent = false)
    .generateVerilog(new BitNetResidentBoardAccelerator())
}
