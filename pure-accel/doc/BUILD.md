# Build and Deployment

## Scala to Board RTL

Use JDK 17, sbt 1.10.7, Scala 2.12.18, and SpinalHDL 1.11.0. Versions are defined in `src/build.sbt` and `src/project/build.properties`. From the repository root:

```powershell
Set-Location src
sbt compile generateBoardRtl
Set-Location ..
```

The board entry point is `ultrabitnet.accel.GenerateBitNetResidentBoardAccelerator`; output is `build/rtl/BitNetResidentBoardAccelerator.v`. Other `Generate...` entry points support local research or module verification and are not the released board top.

## Vivado Board Project

The released image was implemented with Vivado 2025.2 for `xczu3eg-sfvc784-1-i`. `src/vivado/board/design_1_bd.tcl` supplies the AXU3EGB PS configuration. Board scripts connect the 50/100 MHz MMCM, resets, AXI-Lite SmartConnect, five DDR ports, and IRQ.

After generating RTL, create a project from the repository root:

```powershell
vivado -mode batch -source src/vivado/build_bitnet_axu3egb_board.tcl -tclargs project
```

The default directory is `build/vivado-bitnet-r10-tp2-board/`. An existing output directory is never overwritten; supply a new directory as the second Tcl argument. The script also supports `synth`, `impl`, `bitstream`, and `xsa`. Automated implementation is not guaranteed to reproduce the released placement or timing: the release used staged hold closure.

`stitch_bitnet_axu3egb_board_checkpoint.tcl` merges partition DCPs and restores XDC scope. `implement_bitnet_axu3egb_board_checkpoint.tcl` accepts `BITNET_BOARD_IMPL_IN`, `BITNET_BOARD_IMPL_OUT`, and `BITNET_BOARD_IMPL_STAGE` for input checkpoint, output directory, and `place`, `route`, or `bitstream`.

The validated route performed full routing, applied protocol-specific hold treatment with `close_bitnet_axu3egb_board_hold.tcl`, then used `retry_bitnet_axu3egb_board_hold.tcl` in `route_aggressive` mode for `AggressiveExplore` routing and `ExploreWithHoldFix`.

These scripts check input clocks, hierarchy, and paths. The hold flow includes slow/fast direction-level hold exceptions and classified SmartConnect pin-pair exceptions that depend on the current mailbox protocol. Reassess them after any CDC change or new cross-domain path; they are not general-purpose timing fixes. Retain applicable setup, max-delay, and bus-skew checks.

Before generating a new bitstream, require complete routing, nonnegative WNS/WHS, explained CDC/check_timing results, and passing pre-bitstream DRC. The released file is `build/ultra_bitnet_tp2.bit`. Source, tools, or strategy changes require sign-off against new reports.

## Linux Software Build

The software uses a C11 compiler, Make, and Python 3.10 or newer; no third-party Python packages are needed. Native build and tests on Linux, from the repository root:

```sh
make -C src/software test
build/software/bitnetctl --help
```

Outputs are `build/software/bitnetctl` and `build/software/libbitnet.a`. The public protocol API is `src/software/include/bitnet.h`. An application can link the library with its own MMIO callbacks, or compile `bitnet_linux.c` for the supplied backend.

Cross-build for an AArch64 Linux board with its matching SDK/sysroot:

```sh
make -C src/software all CC=aarch64-linux-gnu-gcc AR=aarch64-linux-gnu-ar OUT=../../build/software-aarch64
```

Do not run cross-built tests on an x86 host. Use `make test` natively on the board, or run the host suite separately. Deploy the AArch64 executable and `src/software/tools/` to the board. Example commands below assume a repository checkout and a natively built executable on the board.

## Board Prerequisites

The `.bit` file alone is not a Linux boot image. Boot firmware must initialize the matching PS DDR and clocks; Linux must use a board-specific device tree. Program the PL using the board's supported JTAG or FPGA-manager flow. Linux FPGA-manager deployments may require a converted `.bin`; do not assume it accepts a raw `.bit`.

Merge `src/software/linux/bitnet-uio.dtsi` into the board device tree before boot. It reserves the half-open physical DDR range `[0x800000000, 0x860000000)` (1.5 GiB) with `no-map`, and exposes the `0x80000000` control window through generic UIO. Confirm that the board actually has this physical DDR range and that no firmware, DMA buffer, or Linux allocation uses it. High and low DDR aliases may refer to the same storage on ZynqMP: verify the PS address map and remove any corresponding low alias from OS use too. Do not run the loader until ownership is established.

Enable `CONFIG_UIO` and `CONFIG_UIO_PDRV_GENIRQ`. Where required by the kernel, use `uio_pdrv_genirq.of_id=generic-uio` as a boot/module parameter. The runtime polls, so no interrupt is required in this fragment. Confirm the generated node name, map address, and size under `/sys/class/uio/`; substitute the actual UIO device in every command if it is not `/dev/uio0`.

The loader needs root access to `/dev/mem` and kernel permission to map the reserved region. Its `O_SYNC` mapping must be uncached on the target kernel; validate the memory attributes in the board BSP. If the BSP prohibits this mapping or cannot guarantee coherent visibility to all HP/HPC ports, supply a reserved-memory kernel mapping/loader with explicit synchronization instead. Do not map ordinary Linux-managed RAM or bypass its cache ownership.

```sh
sudo build/software/bitnetctl probe --uio /dev/uio0
sudo build/software/bitnetctl status --uio /dev/uio0
```

Identity must be `0x48425432`. Both commands are read-only. UIO access and image loading share an exclusive device-file lock. All clients must use the same UIO node; do not mix UIO and `/dev/mem` control backends concurrently, because their locks are on different files.

## Model Images and Loading

Use the packed deployment checkpoint from [Microsoft BitNet b1.58 2B 4T on Hugging Face](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T). Download the model files locally and retain the upstream license. The exporter expects packed `U8` linear weights and BF16 embeddings, not the separate BF16 training checkpoint or a GGUF file.

From `pure-accel/`, with Python 3.10 or newer:

```sh
python3 -m pip install numpy huggingface_hub
hf download microsoft/bitnet-b1.58-2B-4T --local-dir /data/bitnet-source --include model.safetensors config.json tokenizer.json tokenizer_config.json special_tokens_map.json LICENSE
python3 src/software/tools/export_model.py --model-dir /data/bitnet-source --output build/model-export
```

An existing local directory containing `model.safetensors`, `config.json`, and `tokenizer.json` can be supplied directly with `--model-dir`; no download is needed. Keep both implementation folders in the public checkout: the wrapper reuses `../mix-accel/src/tools/export_emmc_package.py` for model export, then invokes the pure-accel bank builder. NumPy is required only for model export; the C runtime and bank builder do not need it. The input model is never modified and the output directory must be new. Allow approximately 3 GB of free output space in addition to the source checkpoint.

The output contains:

- `package/BITNET/`: `MANIFEST.TXT`, layer weights `L00/` through `L29/`, BF16 embeddings `EMB/`, LM shards `LMH/`, auxiliary/RoPE data `AUX/`, and tokenizer assets `TOK/`.
- `banks/`: `bank0.bin` through `bank4.bin` and `layout.json`, ready for the pure-accel loader.
- `export.log`: exporter diagnostics. If export fails, its temporary directory and log are retained for inspection.

The shared exporter produces Map1 ternary codes `0=-1`, `1=0`, `2=1`. The bank builder converts only packed Transformer weight bytes to the released compute path's Map0 codes `1=-1`, `2=0`, `3=1`, rejecting reserved input code 3. It does not recode LM int8 weights, floating-point scales, auxiliary data, embeddings, or KV storage. Already-Map0 packages are also accepted. Manifest geometry, headers, sizes, and paths are validated before five-bank striping; `layout.json` records source and native codebooks.

For an existing exported package, run only the bank-building stage:

```sh
python3 src/software/tools/build_image.py /data/BITNET /data/bitnet-images
```

Choose a new output directory; existing images are not overwritten. Expect approximately 1.02 GB of bank files, in addition to the source model package.

Generated checkpoints, packages, and bank images remain local under the ignored `build/` subdirectories. They are not required in Git and must not be added to the public source commit. Transfer `banks/` and `package/BITNET/` to the board as `/data/bitnet-images/` and `/data/BITNET/`, respectively, for the commands below. Retain the upstream model license with any separately distributed model package.

| Image / Region | Physical Base | Bytes Written |
| --- | --- | ---: |
| bank0.bin, HPC0 | `0x850000000` | 204833664 |
| bank1.bin, HP0 | `0x800000000` | 204833664 |
| bank2.bin, HP1 | `0x810000000` | 204833664 |
| bank3.bin, HP2 | `0x820000000` | 204833664 |
| bank4.bin, HP3 | `0x830000000` | 202376064 |
| Source staging, copied from bank0 prefix | `0x840000000` | 104693760 |

After a fresh PL reset, with no active session:

```sh
sudo python3 src/software/tools/load_banks.py /data/bitnet-images --uio /dev/uio0
```

Every image size is checked before any hardware write. The loader refuses an incompatible identity, nonzero status, or pending/overrun PIO state. Do not modify image files during loading. If loading is interrupted, reset and reload the entire image set before inference.

The descriptor default is `--source-offset 0x40000000 --source-bytes 104693760`. The source offset is relative to HP0 base `0x800000000`; the resulting physical address is `0x840000000`. Do not pass that physical address as the descriptor offset.

## Prefill and Decode

The command interface operates on token IDs or embedding vectors, not raw text. Tokenization, special-token policy, output decoding, and generation stopping belong to the application and must match the exported model.

For a two-token prompt (replace the illustrative IDs with your tokenizer output):

```sh
python3 src/software/tools/prepare_hidden.py --package /data/BITNET --tokens 1 2 --output /data/prompt.q16
sudo build/software/bitnetctl round --uio /dev/uio0 --session 1 --epoch 1 --mode prefill --position 0 --tokens 2 --new-session --last-prompt --hidden /data/prompt.q16 --timeout-ms 120000
```

Only the first round uses `--new-session`, and it must be prefill at position 0. For a longer prompt, split into one- or two-token rounds; omit `--last-prompt` until the final prompt round. Continue with the same session and image epoch and the preceding result's `next_position`. The final prompt round returns a token with `token_valid=1`; intermediate prompt rounds do not.

For decode, prepare the returned token's embedding, then submit one token at `next_position`. The example assumes the first result returned token 42 and next position 2; use the actual result values:

```sh
python3 src/software/tools/prepare_hidden.py --package /data/BITNET --tokens 42 --output /data/decode-2.q16
sudo build/software/bitnetctl round --uio /dev/uio0 --session 1 --epoch 1 --mode decode --position 2 --tokens 1 --hidden /data/decode-2.q16 --timeout-ms 120000
```

Repeat with each returned token until the application's stop condition or the 4096-position limit. Decode always uses one token and never `--last-prompt`. Do not change model data under an active session.

Alternatively, `prepare_hidden.py --json vectors.json --output hidden.q16` accepts an array of one or two vectors, each containing 2560 finite numbers. Conversion uses signed Q16.16, round-to-nearest with ties away from zero, and int32 saturation. Files are lane-major little-endian int32: 10240 bytes for one token or 20480 for two. The C runtime interleaves them into 640 PIO groups and zero-fills the inactive lane. Preparation refuses to overwrite an existing output file.

Successful round output has `rc=0`, `result_valid=1`, `success=1`, and validated identity/position fields. Nonzero process exit indicates an error. On timeout, the driver does not ACK a late result; inspect status and recover/reset the PL before retrying. There is no software reset register in this ABI. Restarting the CLI is not a hardware reset, and pending results must not be discarded blindly.

## Verification Scope

`make -C src/software test` compiles with `-Wall -Wextra -Werror`. The C mock exercises TP2 and single-token prefill, decode, all 640 groups, backpressure, invalid descriptors, faults, identity mismatches, and late-result timeout behavior. Python tests cover register agreement with Scala, bank layout and loader sizes, image-copy/build failure handling, embedding conversion, Linux mapping, exclusive access, JSON output, and validation before submission.

The Linux software suite includes the C driver scenarios and 25 Python tests, including exhaustive byte-level Map1-to-Map0 conversion and preservation of non-Transformer data. Tests use mock or file-backed MMIO, synthetic layout metadata, and small image fixtures. The full 30-layer local packed model has also been exported through the public-branch wrapper into five correctly sized bank images.

The repository does not include a bootable PS Linux image, model weights, Vivado checkpoints, or tool caches. The existing bitstream and Scala hardware are unchanged by the software integration.
