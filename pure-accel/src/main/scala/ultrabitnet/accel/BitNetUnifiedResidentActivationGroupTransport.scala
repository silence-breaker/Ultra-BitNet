package ultrabitnet.accel

import spinal.core._
import spinal.lib._

import scala.language.postfixOps

/** Production activation transport constants.
  *
  * A legacy activation group always names four adjacent features.  In the
  * unified tensor image those features occupy one aligned group of four
  * banks, while every bank word contains all four TP4 token lanes.  Carrying
  * only those four words keeps the slow-domain transport at 512 data bits
  * instead of exposing the complete 32 x 128-bit physical row.
  */
object BitNetUnifiedResidentActivationGroupTransport {
  val BankWords = 4
}

case class BitNetUnifiedResidentActivationGroupRead(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  private val groupCount =
    cfg.layout.bankCount /
      BitNetUnifiedResidentActivationGroupTransport.BankWords

  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankGroup = UInt(log2Up(groupCount) bits)
}

case class BitNetUnifiedResidentActivationGroupReadData(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  private val groupCount =
    cfg.layout.bankCount /
      BitNetUnifiedResidentActivationGroupTransport.BankWords

  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankGroup = UInt(log2Up(groupCount) bits)
  val words = Vec(
    BitNetUnifiedResidentActivationWord(cfg),
    BitNetUnifiedResidentActivationGroupTransport.BankWords)
  val fault = Bool()
  val faultCode = Bits(4 bits)
}

case class BitNetUnifiedResidentActivationGroupWrite(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  private val groupCount =
    cfg.layout.bankCount /
      BitNetUnifiedResidentActivationGroupTransport.BankWords

  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankGroup = UInt(log2Up(groupCount) bits)
  val elementMask = Vec(
    Bits(cfg.layout.tokenLanes bits),
    BitNetUnifiedResidentActivationGroupTransport.BankWords)
  val words = Vec(
    BitNetUnifiedResidentActivationWord(cfg),
    BitNetUnifiedResidentActivationGroupTransport.BankWords)
}

case class BitNetUnifiedResidentActivationGroupWriteFault(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig) extends Bundle {
  private val groupCount =
    cfg.layout.bankCount /
      BitNetUnifiedResidentActivationGroupTransport.BankWords

  val region = BitNetUnifiedResidentTensorRegion()
  val localRow = UInt(cfg.layout.localRowWidth bits)
  val bankGroup = UInt(log2Up(groupCount) bits)
  /** One bit per word/shard, derived from elementMask.orR. */
  val wordMask = Bits(
    BitNetUnifiedResidentActivationGroupTransport.BankWords bits)
  val faultCode = Bits(4 bits)
}

/** Slow-domain adapter between the compact production ABI and the generic
  * semantic frontend's physical-row ABI.
  *
  * No payload state or 4096-bit data snapshot is created: response identity
  * is encoded from the semantic frontend's retained aligned bank mask, reads
  * use four 8:1 word selections at the 266 MHz boundary, and writes broadcast
  * four words while a three-bit decoder generates bank enables.
  */
class BitNetUnifiedResidentActivationGroupSemanticAdapter(
    cfg: BitNetUnifiedResidentTensorScratchpadConfig =
      BitNetUnifiedResidentTensorScratchpadConfig()) extends Component {
  private val layout = cfg.layout
  private val bankWords =
    BitNetUnifiedResidentActivationGroupTransport.BankWords
  private val bankGroupCount = layout.bankCount / bankWords

  require(layout.bankCount % bankWords == 0)
  require(bankWords == 4)
  require(layout.tokenLanes == 2 || layout.tokenLanes == 4)

  private def alignedBankGroup(mask: Bits): UInt = {
    val group = UInt(log2Up(bankGroupCount) bits)
    group := 0
    for (candidate <- 0 until bankGroupCount) {
      when(mask(candidate * bankWords, bankWords bits).orR) {
        group := candidate
      }
    }
    group
  }

  private def alignedWordMask(mask: Bits): Bits = {
    val words = Bits(bankWords bits)
    for (word <- 0 until bankWords) {
      words(word) := (0 until bankGroupCount)
        .map(group => mask(group * bankWords + word))
        .reduce(_ || _)
    }
    words
  }

  val io = new Bundle {
    val groupRead = slave(Stream(
      BitNetUnifiedResidentActivationGroupRead(cfg)))
    val groupReadData = master(Stream(
      BitNetUnifiedResidentActivationGroupReadData(cfg)))
    val groupWrite = slave(Stream(
      BitNetUnifiedResidentActivationGroupWrite(cfg)))
    val groupWriteFault = master(Stream(
      BitNetUnifiedResidentActivationGroupWriteFault(cfg)))

    val wideRead = master(Stream(
      BitNetUnifiedResidentActivationRead(cfg)))
    val wideReadData = slave(Stream(
      BitNetUnifiedResidentActivationReadData(cfg)))
    val wideWrite = master(Stream(
      BitNetUnifiedResidentActivationWrite(cfg)))
    val wideWriteFault = slave(Stream(
      BitNetUnifiedResidentTensorSemanticWriteFault(cfg)))
  }

  io.wideRead.valid := io.groupRead.valid
  io.groupRead.ready := io.wideRead.ready
  io.wideRead.region := io.groupRead.region
  io.wideRead.localRow := io.groupRead.localRow
  for (bank <- 0 until layout.bankCount) {
    io.wideRead.bankMask(bank) :=
      io.groupRead.bankGroup === U(bank / bankWords)
  }

  io.groupReadData.valid := io.wideReadData.valid
  io.wideReadData.ready := io.groupReadData.ready
  io.groupReadData.region := io.wideReadData.region
  io.groupReadData.localRow := io.wideReadData.localRow
  val responseReadBankGroup = alignedBankGroup(io.wideReadData.bankMask)
  io.groupReadData.bankGroup := responseReadBankGroup
  io.groupReadData.fault := io.wideReadData.fault
  io.groupReadData.faultCode := io.wideReadData.faultCode
  for (word <- 0 until bankWords) {
    val selectedBank =
      (responseReadBankGroup.asBits ##
        B(word, log2Up(bankWords) bits)).asUInt
    io.groupReadData.words(word) := io.wideReadData.words(selectedBank)
  }

  io.wideWrite.valid := io.groupWrite.valid
  io.groupWrite.ready := io.wideWrite.ready
  io.wideWrite.region := io.groupWrite.region
  io.wideWrite.localRow := io.groupWrite.localRow
  for (bank <- 0 until layout.bankCount) {
    val word = bank % bankWords
    val groupSelected =
      io.groupWrite.bankGroup === U(bank / bankWords)
    io.wideWrite.bankMask(bank) :=
      groupSelected && io.groupWrite.elementMask(word).orR
    io.wideWrite.elementMask(bank) := Mux(
      groupSelected,
      io.groupWrite.elementMask(word),
      B(0, layout.tokenLanes bits))
    // The element mask is the physical write enable.  Replicating each of the
    // four source words across its eight possible banks is pure wiring and
    // avoids a 32-way zero/scatter network.
    io.wideWrite.words(bank) := io.groupWrite.words(word)
  }

  io.groupWriteFault.valid := io.wideWriteFault.valid
  io.wideWriteFault.ready := io.groupWriteFault.ready
  io.groupWriteFault.region := io.wideWriteFault.region
  io.groupWriteFault.localRow := io.wideWriteFault.localRow
  io.groupWriteFault.bankGroup :=
    alignedBankGroup(io.wideWriteFault.bankMask)
  io.groupWriteFault.wordMask :=
    alignedWordMask(io.wideWriteFault.bankMask)
  io.groupWriteFault.faultCode := io.wideWriteFault.faultCode
}
