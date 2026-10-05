# Fast-BitNet Operation Guide

This guide is the branch-level operating manual for the AXU3EGB/Zynq UltraScale+ BitNet accelerator. It is intentionally separate from the root README so it can be reviewed and merged later.

## 1. Scope and expected result

The project combines a PS-side runtime with PL GEMV and Attention accelerators. A successful bring-up accepts a text prompt, executes the 30-layer BitNet transformer, maintains the KV cache, computes the tied language-model head, and returns generated token text.

The reference implementation has been validated at 125 MHz PL clock, 128-bit DMA/HP0 data width, and 16 Attention lanes. The published performance points are 5.844550 seconds for a 16-token prefill and approximately 0.742594 seconds per token for a 64-token-context decode. Re-measure on the target board before making performance claims.

## 2. Host prerequisites

- Vivado/Vitis 2020.1 with XCZU3EG device support for hardware reconstruction.
- Python 3.10 or newer for conversion and protocol utilities.
- A Linux shell or WSL for `scripts/evaluate_host.sh`.
- An AXU3EGB-compatible board, JTAG connection, UART, and storage for the model assets.
- A licensed BitNet-compatible model and tokenizer. The model is deliberately excluded from this repository.

## 3. Host-only validation

Run these checks before touching the board:

```bash
python3 tools/inspect_safetensors.py --help
bash scripts/evaluate_host.sh
python3 -m py_compile tools/*.py integration/pc_relay/*.py integration/pc_relay/tools/*.py
```

The host checks validate file formats, reference tensor paths, tokenizer conversion entry points, and protocol utilities. They do not prove that a particular bitstream is compatible with the board.

## 4. Model conversion

Keep the source model outside the Git checkout. The conversion flow is:

```bash
python3 tools/inspect_safetensors.py --model /path/to/model.safetensors
python3 tools/export_tokenizer_trie.py \
  --tokenizer-json /path/to/tokenizer.json \
  --trie-out /path/to/BITNET/TOK/TRIE.BIN \
  --decode-out /path/to/BITNET/TOK/DECODE.BIN \
  --pretok-out /path/to/BITNET/TOK/PRET.BIN
python3 tools/export_layer_bundle.py \
  --model /path/to/model.safetensors \
  --output /path/to/BITNET
```

The exact exporter options may vary with the model revision. Record the model version, source URL, and license in a release-specific manifest. Never commit model weights, tokenizer files, generated BIN files, or access tokens.

## 5. Hardware reconstruction

Open a Vivado Tcl shell in the repository root and run the project script:

```tcl
source scripts/build_plddr_dual_project.tcl
```

Use the script's supported action sequence:

```text
project      create the Block Design and project
synth        run synthesis
resume-impl  continue implementation and routing
export-xsa   export the hardware platform for Vitis
```

Confirm the routed timing report has non-negative WNS and zero TNS. Confirm that the generated XSA uses the intended PS DDR/HP configuration and 125 MHz PL clock. Generated project directories belong in local build storage, not in Git.

## 6. Bare-metal build and JTAG bring-up

1. Import the exported XSA into Vitis.
2. Create an AArch64 standalone application from `sw/baremetal`.
3. Configure the linker script and model asset base addresses to match the board design.
4. Build the application and keep the resulting ELF outside the source commit.
5. Load PS initialization, the bitstream, and the ELF with XSDB/JTAG.
6. Open the board UART and confirm the runtime reports its clock, memory map, model assets, and accelerator status.
7. Send a short prompt and verify the response markers and token counters.

The expected functional path is:

```text
prompt -> tokenizer -> embeddings -> 30 transformer layers
       -> GEMV/Attention/RoPE/KV cache -> RMSNorm -> tied LM head -> tokens
```

## 7. Linux deployment

The Linux implementation is useful when the board boots an AArch64 distribution:

```bash
cd linux
make clean all
sudo sh ./install.sh
sudo env BITNET_PLDDR_ENABLE=1 \
  bitnet-linux --model /opt/bitnet --prompt "Hello" \
  --tokens 8 --temperature 0 --top-k 1
```

Before rebooting Linux, merge `linux/bitnet-reserved-memory.dtsi` into the board device tree and reserve the DMA/packet-cache ranges. The boot firmware and XSA must agree on the HP port width, AXI DMA address, and PL clock. Check `/proc/iomem` after reboot and ensure the reserved regions are not assigned to another driver.

## 8. Relay and protocol integration

`integration/pc_relay` contains host-side prompt/token utilities and JTAG/serial relay helpers. `integration/socket_json_bridge` contains the transport schema and protocol implementation. Configure the serial device, board address, and TCP ports through command-line options or a local untracked configuration file.

For continuous operation, use one long-lived process and one transport owner. Do not run a serial getty and the relay on the same UART. Keep captured prompts, reports, logs, and board addresses outside the public repository.

## 9. Troubleshooting checklist

| Symptom | Check |
|---|---|
| FPGA load fails | Bitstream format, FPGA manager state, firmware path, and boot firmware compatibility |
| DMA remains busy | HP width, PL clock, DMA base address, driver conflicts, and reserved memory |
| Model cannot be opened | Model root path and the expected `BITNET` subdirectory layout |
| Incorrect tokens | Tokenizer revision, AUX/scale assets, tensor layout, and reference comparison |
| Out-of-memory | Available DDR, resident weights, KV cache size, and Linux services |
| Relay timeout | UART ownership, JTAG frequency, TCP endpoint, and mailbox schema version |

## 10. Reproducibility record

For each board result, record:

- repository commit and hardware script revision;
- board/device and Vivado/Vitis version;
- PL clock, DMA width, Attention lane count, and DDR configuration;
- prompt token count, generated token count, prefill/decode timing, temperature, and top-k;
- temperature, power, and any reset/recovery events.

These records make it possible to distinguish a source regression from a board, model, or environment difference.
