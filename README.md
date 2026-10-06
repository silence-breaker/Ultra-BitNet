# Ultra-BitNet
Build the fastest infra for Edged LLM

## Model manifest

Model weights are distributed separately from the source layout. For each model
release, record the following information:

```text
name: <model name>
version: <version>
source: <upstream URL or release ID>
license: <model license>
layout: BITNET/L00..L29 + tokenizer + RoPE + AUX
converter: src/tools/export_layer_bundle.py
```

Distribute models through GitHub Releases or Git LFS. Release notes should
include the board model, required DDR capacity, and conversion command.

## Third-party notices

Before publishing, confirm the version, source, and license for every external
component:

| Component | Repository use | Release requirement |
|---|---|---|
| Xilinx Vivado/Vitis | FPGA project creation, synthesis, implementation, and Vitis builds | Do not redistribute the tools; follow AMD/Xilinx terms |
| AXU3EGB/Alinx board files | Device, pin, clock, and PS configuration | Confirm the board vendor's redistribution terms |
| Icraft/YOLOv8 components | Vision-side inference and deployment | Publish separately under the upstream and SDK terms |
| BitNet model and tokenizer | Weights, vocabulary, RoPE, and AUX assets | Confirm the model license separately from this source license |
| Python/C/C++ dependencies | Tools, reference implementation, and protocol tests | Generate a complete dependency inventory and retain notices |

This notice does not replace the original third-party licenses. Do not publish
any binary, font, image, model, or SDK whose redistribution rights have not
been confirmed.
