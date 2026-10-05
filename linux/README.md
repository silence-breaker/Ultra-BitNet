# Linux Deployment

The Linux source tree builds the AArch64/user-space runtime. Board assets are intentionally external release inputs.

## Build on the target or with an AArch64 cross compiler

```bash
make clean all
```

The output is `build/bitnet-linux`. The default Makefile uses the target compiler flags; set `CC`, `CFLAGS`, and `LDFLAGS` for a cross build.

## Install a release package

`install.sh` expects the following files relative to the `linux` directory:

```text
build/bitnet-linux or bin/bitnet-linux
firmware/bitnet_accel.bit.bin
model/BITNET.tar (optional when the model is installed separately)
```

The bitstream and model must match the XSA, PS initialization, memory map, and model manifest. They are not committed to this repository. Download them from a versioned release, then run:

```bash
sudo ./install.sh
```

Before booting Linux, merge `bitnet-reserved-memory.dtsi` into the board device tree. Confirm the reserved ranges in `/proc/iomem` after reboot. Load the FPGA only after checking that no kernel DMA driver owns the accelerator registers.

## Runtime configuration

```bash
export BITNET_PLDDR_ENABLE=1
export BITNET_DUAL_MEMORY_SPLIT=1
sudo -E bitnet-linux --model /opt/bitnet --prompt "Hello" --tokens 8 --temperature 0 --top-k 1
```

Use `BITNET_PLDDR_ENABLE=0` and `BITNET_DUAL_MEMORY_SPLIT=0` only with a legacy single-memory bitstream.
