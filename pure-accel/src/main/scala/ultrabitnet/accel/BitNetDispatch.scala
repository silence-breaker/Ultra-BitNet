package ultrabitnet.accel

import spinal.core._
import spinal.lib._

/**
  * Lossless command multicast for the output-sharded matrix engines.
  *
  * A command is latched once and each selected engine independently consumes
  * it under backpressure. Directly wiring one Stream master to several slaves
  * would duplicate or drop commands when ready differs, so the accepted mask
  * is tracked explicitly.
  */
class BitNetEngineCommandMulticast(cfg: BitNetConfig) extends Component {
  val io = new Bundle {
    val input = slave(Stream(BitNetEngineCommand(cfg)))
    val output = Vec(master(Stream(BitNetEngineCommand(cfg))), cfg.engineCount)
    val busy = out Bool()
  }

  val held = Reg(BitNetEngineCommand(cfg))
  val pending = Reg(Bits(cfg.engineCount bits)) init 0
  io.busy := pending.orR
  io.input.ready := !pending.orR

  when(io.input.fire) {
    held := io.input.payload
    pending := io.input.payload.activeEngineMask
  }
  for (engine <- 0 until cfg.engineCount) {
    io.output(engine).valid := pending(engine)
    io.output(engine).payload := held
    when(io.output(engine).fire) {
      pending(engine) := False
    }
  }
}

/** Same lossless multicast for physical SPU shards. */
class BitNetSpuCommandMulticast(cfg: BitNetConfig) extends Component {
  val io = new Bundle {
    val input = slave(Stream(BitNetSpuCommand(cfg)))
    val output = Vec(master(Stream(BitNetSpuCommand(cfg))), cfg.spuCount)
    val busy = out Bool()
  }

  val held = Reg(BitNetSpuCommand(cfg))
  val pending = Reg(Bits(cfg.spuCount bits)) init 0
  io.busy := pending.orR
  io.input.ready := !pending.orR

  when(io.input.fire) {
    held := io.input.payload
    pending := io.input.payload.activeSpuMask
  }
  for (spu <- 0 until cfg.spuCount) {
    io.output(spu).valid := pending(spu)
    io.output(spu).payload := held
    when(io.output(spu).fire) {
      pending(spu) := False
    }
  }
}

/** Round-robin completion merge with a two-entry skid FIFO per engine. */
class BitNetEngineCompletionMerge(cfg: BitNetConfig) extends Component {
  val io = new Bundle {
    val input = Vec(slave(Stream(BitNetEngineCompletion(cfg))), cfg.engineCount)
    val output = master(Stream(BitNetEngineCompletion(cfg)))
  }

  val fifos = Array.fill(cfg.engineCount)(
    StreamFifo(BitNetEngineCompletion(cfg), 2)
  )
  for (engine <- 0 until cfg.engineCount) {
    fifos(engine).io.push << io.input(engine)
  }
  val merged = StreamArbiterFactory.roundRobin.noLock
    .on(Vec(fifos.map(_.io.pop)))
  io.output << merged
}

/** Round-robin completion merge with a two-entry skid FIFO per SPU. */
class BitNetSpuCompletionMerge(cfg: BitNetConfig) extends Component {
  val io = new Bundle {
    val input = Vec(slave(Stream(BitNetSpuCompletion(cfg))), cfg.spuCount)
    val output = master(Stream(BitNetSpuCompletion(cfg)))
  }

  val fifos = Array.fill(cfg.spuCount)(
    StreamFifo(BitNetSpuCompletion(cfg), 2)
  )
  for (spu <- 0 until cfg.spuCount) {
    fifos(spu).io.push << io.input(spu)
  }
  val merged = StreamArbiterFactory.roundRobin.noLock
    .on(Vec(fifos.map(_.io.pop)))
  io.output << merged
}

/**
  * Honest integration boundary while arithmetic shells and AXI adapters are
  * being completed. It contains the fixed model scheduler and all lossless
  * physical-unit dispatch/barrier plumbing, but does not auto-complete work or
  * claim to be an end-to-end inference core.
  */
class BitNetControlSubsystem(
    cfg: BitNetConfig = BitNetConfig.Production) extends Component {
  val io = new Bundle {
    val workload = slave(Stream(BitNetWorkload(cfg)))
    val completion = master(Stream(BitNetWorkloadCompletion(cfg)))
    val engineCommand = Vec(master(Stream(BitNetEngineCommand(cfg))), cfg.engineCount)
    val engineCompletion = Vec(slave(Stream(BitNetEngineCompletion(cfg))), cfg.engineCount)
    val spuCommand = Vec(master(Stream(BitNetSpuCommand(cfg))), cfg.spuCount)
    val spuCompletion = Vec(slave(Stream(BitNetSpuCompletion(cfg))), cfg.spuCount)
    val busy = out Bool()
  }

  val controller = new BitNetWorkloadController(cfg)
  val engineDispatch = new BitNetEngineCommandMulticast(cfg)
  val engineMerge = new BitNetEngineCompletionMerge(cfg)
  val spuDispatch = new BitNetSpuCommandMulticast(cfg)
  val spuMerge = new BitNetSpuCompletionMerge(cfg)

  controller.io.workload << io.workload
  io.completion << controller.io.completion
  controller.io.engineCommand >> engineDispatch.io.input
  controller.io.spuCommand >> spuDispatch.io.input

  for (engine <- 0 until cfg.engineCount) {
    io.engineCommand(engine) << engineDispatch.io.output(engine)
    engineMerge.io.input(engine) << io.engineCompletion(engine)
  }
  controller.io.engineCompletion << engineMerge.io.output

  for (spu <- 0 until cfg.spuCount) {
    io.spuCommand(spu) << spuDispatch.io.output(spu)
    spuMerge.io.input(spu) << io.spuCompletion(spu)
  }
  controller.io.spuCompletion << spuMerge.io.output
  io.busy := controller.io.busy || engineDispatch.io.busy || spuDispatch.io.busy
}
