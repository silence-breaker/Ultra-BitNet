# Ultra-BitNet

A Scala / SpinalHDL BitNet inference accelerator for AXU3EGB (XCZU3EG-1SFVC784I). The released board configuration uses TP2, one physical matrix engine, four SPUs, and five interfaces to shared PS DDR. Control and memory run at 50 MHz; compute runs at 100 MHz.

## Repository

- `doc/`: architecture, build instructions, and architecture diagram assets.
- `src/main/scala/ultrabitnet/accel/`: board-level hardware and its dependencies.
- `src/vivado/`: board integration, implementation, and timing-closure scripts.
- `src/software/`: portable C runtime, Linux MMIO backend and CLI, model-image tools, and tests.
- `build/ultra_bitnet_tp2.bit`: released board bitstream.

## Build

Generate RTL with JDK 17, sbt 1.10.7, Scala 2.12.18, and SpinalHDL 1.11.0:

```sh
cd src
sbt compile generateBoardRtl
```

The entry point is `ultrabitnet.accel.GenerateBitNetResidentBoardAccelerator`; output is `build/rtl/BitNetResidentBoardAccelerator.v`. The first build downloads dependencies.

Build and test the Linux software from the repository root:

```sh
make -C src/software test
```

See [Architecture](doc/ARCHITECTURE.md) and [Build and Deployment](doc/BUILD.md) for hardware integration, reserved-memory requirements, model loading, and prefill/decode commands. Software tests use mocked or file-backed MMIO and do not require a board.

The released implementation reported setup/hold margins of `+1.291 / +0.008 ns` with all routable nets routed. Source and bitstream licensing is described in [LICENSE](LICENSE).
