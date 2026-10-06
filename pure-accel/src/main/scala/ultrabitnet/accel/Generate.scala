package ultrabitnet.accel

import spinal.core._

/** Generate the honest native control integration boundary (not a fake core). */
object GenerateBitNetControlSubsystem {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse("../generated/bitnet-control")
    SpinalConfig(
      mode = Verilog,
      targetDirectory = target,
      oneFilePerComponent = false,
      defaultConfigForClockDomains = ClockDomainConfig(
        clockEdge = RISING,
        resetKind = ASYNC,
        resetActiveLevel = LOW
      )
    ).generate(new BitNetControlSubsystem(BitNetConfig.Production))
  }
}

/** Generate the tagged five-bank read/write blocks for AXI adapter integration. */
object GenerateBitNetFiveBankFabric {
  def main(args: Array[String]): Unit = {
    val root = args.headOption.getOrElse("../generated/bitnet-memory")
    val cfg = BitNetMemoryConfig()
    SpinalConfig(
      mode = Verilog,
      targetDirectory = s"$root/read",
      oneFilePerComponent = false
    ).generate(new BitNetFiveBankReadFabric(cfg))
    SpinalConfig(
      mode = Verilog,
      targetDirectory = s"$root/write",
      oneFilePerComponent = false
    ).generate(new BitNetFiveBankWriteFabric(cfg))
  }
}

/** Generate the liveness-overlaid production activation scratchpad. */
object GenerateBitNetActivationFabric {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse("../generated/bitnet-activation")
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
}

/**
  * Generate the earlier one-slot-per-tensor scratchpad for comparison only.
  * It is deliberately excluded from the native production top because it
  * stores Gate/Up even though that tensor can flow directly into ReLU2.
  */
object GenerateBitNetFullSlotActivationReference {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse(
      "../generated/bitnet-full-slot-activation-reference")
    val cfg = BitNetConfig.Production
    SpinalConfig(
      mode = Verilog,
      targetDirectory = target,
      oneFilePerComponent = false
    ).generate(new BitNetActivationFabric(cfg, BitNetActivationLayout(cfg)))
  }
}

/** Generate the five-bank local AGU used after logical-task shard expansion. */
object GenerateBitNetBankedAddressGenerator {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse("../generated/bitnet-banked-agu")
    val cfg = BitNetConfig.Production
    SpinalConfig(
      mode = Verilog,
      targetDirectory = target,
      oneFilePerComponent = false
    ).generate(new BitNetBankedAddressGenerator(
      cfg,
      BitNetFiveBankResidentLayout(cfg)
    ))
  }
}

/** Generate the independent two-engine/four-SPU by five-bank crossbar. */
object GenerateBitNetBankLocalCrossbar {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse("../generated/bitnet-bank-crossbar")
    SpinalConfig(
      mode = Verilog,
      targetDirectory = target,
      oneFilePerComponent = false
    ).generate(new BitNetBankLocalCrossbar())
  }
}

/**
  * Generate the numerical online-attention reference only.  This object is
  * intentionally named `Reference`: the production top must use the tiled,
  * banked attention SPU after physical timing closure, never this model.
  */
object GenerateBitNetOnlineAttentionReference {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse(
      "../generated/bitnet-online-attention-reference")
    SpinalConfig(
      mode = Verilog,
      targetDirectory = target,
      oneFilePerComponent = false
    ).generate(new NativeOnlineAttention())
  }
}

/** Generate the deployable tile=16 online-softmax SPU used by KV coordinators. */
object GenerateBitNetTiledAttentionSpu {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse(
      "../generated/bitnet-tiled-attention-spu")
    SpinalConfig(
      mode = Verilog,
      targetDirectory = target,
      oneFilePerComponent = false
    ).generate(new BitNetTiledAttentionSpu())
  }
}

/** Generate the four-token split-half RoPE datapath used for both Q and K. */
object GenerateBitNetRope {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse(
      "../generated/bitnet-rope")
    SpinalConfig(
      mode = Verilog,
      targetDirectory = target,
      oneFilePerComponent = false
    ).generate(new NativeRope(NativeRopeConfig()))
  }
}

/**
  * Generate the head-local Q/K/V quantizer.  A single physical instance is
  * reused for Q(20), K(5) and V(5) heads; vectorLength=128 is therefore the
  * deployable KV geometry, not a reduced numerical test shape.
  */
object GenerateBitNetHeadI8Quantizer {
  def main(args: Array[String]): Unit = {
    val target = args.headOption.getOrElse(
      "../generated/bitnet-head-i8-quantizer")
    val cfg = NativeDynamicI8QuantizerConfig(
      tokenLanes = BitNetConfig.Production.tokenParallelism,
      vectorLength = BitNetConfig.Production.headSize
    )
    SpinalConfig(
      mode = Verilog,
      targetDirectory = target,
      oneFilePerComponent = false
    ).generate(new BitNetHeadI8Quantizer(cfg))
  }
}
