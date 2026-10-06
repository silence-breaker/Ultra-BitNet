# Ultra-BitNet

An open-source BitNet inference accelerator for the AXU3EGB board. The
repository combines FPGA RTL/HLS compute kernels, bare-metal and Linux board
runtimes, PC-side JTAG/serial transport, and Git LFS model assets.

## Repository

- `src/docs/`: operation guide and performance notes.
- `src/hardware/accelerator/`: Verilog/SystemVerilog RTL, HLS kernels, and simulation testbenches.
- `src/hardware/board/`: AXU3EGB constraints and board block-design sources.
- `src/hardware/baremetal/`: board-side bare-metal runtime, kernels, protocol, and linker patches.
- `src/hardware/linux/`: board-side Linux runtime, compatibility layer, installer, and tests.
- `src/hardware/scripts/`: Vivado/Vitis build, simulation, model export, eMMC, and board-run scripts.
- `src/hardware/release/`: released board assets, including the bitstream, XSA, ELF, and PS initialization.
- `src/pc/jtag/`: XSDB/JTAG inference scripts.
- `src/pc/relay/`: serial relay, prompt/token utilities, and host test tools.
- `src/pc/socket_json_bridge/`: TCP/serial JSON bridge and message schema.
- `src/models/official_model/`: BitNet model weights and tokenizer tracked with Git LFS.
- `src/tools/`: model conversion, tensor export, tokenizer, and CPU reference tools.

## Build

Run the host regression from the repository root:

```bash
bash src/hardware/scripts/evaluate_host.sh
```

Build the Linux evaluator directly:

```bash
make -C src/hardware/linux clean all
```

Hardware builds require the matching AMD/Xilinx Vivado and Vitis installation:

```tcl
source src/hardware/scripts/build_plddr_dual_project.tcl
```

Generated Vivado projects, XSA files, ELF files, and temporary reports remain
in local build storage unless explicitly exported as release assets.

## Board Deployment

1. Build or obtain the board XSA, bitstream, PS initialization script, and ELF.
2. Use the scripts in `src/hardware/scripts/` for Vivado/Vitis, eMMC, and board operations.
3. Merge `src/hardware/linux/bitnet-reserved-memory.dtsi` into the target device tree for Linux deployment.
4. Use `src/pc/jtag/` or `src/pc/relay/` for JTAG, serial, and host-side inference transport.

Read [`src/docs/OPERATION_GUIDE.md`](src/docs/OPERATION_GUIDE.md) before using a
board. External tool installations, board access, and licensed model assets
are required for hardware execution.

## Model Assets

Model weights and tokenizer files are kept under `src/models/official_model/`
and are managed through Git LFS. A model release should document its name,
version, source, license, layout, required board memory, and conversion command.
The repository does not grant any additional rights to third-party model files.

## Third-Party Software

| Component | Repository use | Release requirement |
|---|---|---|
| Xilinx Vivado/Vitis | FPGA project creation, synthesis, implementation, and Vitis builds | Do not redistribute the tools; follow AMD/Xilinx terms |
| AXU3EGB/Alinx board files | Device, pin, clock, and PS configuration | Confirm the board vendor's redistribution terms |
| Icraft/YOLOv8 components | Vision-side inference and deployment | Follow upstream and SDK terms |
| BitNet model and tokenizer | Weights, vocabulary, RoPE, and AUX assets | Confirm the model license separately |
| Python/C/C++ dependencies | Tools, reference implementation, and protocol tests | Retain the applicable notices |

Do not publish any binary, model, font, image, or SDK whose redistribution
rights have not been confirmed.

## License

Project-owned source code is released under the BSD 3-Clause License in
[`LICENSE`](LICENSE). Third-party tools, board files, dependencies, and model
assets remain subject to their respective licenses.
