# Performance Report

## Reference platform

- Board: ALINX AXU3EGB / Xilinx Zynq UltraScale+ MPSoC
- Model: BitNet b1.58 2B 4T
- PL clock: 125 MHz
- DMA and HP0 width: 128 bit
- Attention parallelism: 16 lanes
- Transformer layers: 30
- KV cache: approximately 150 MiB

## Board measurements

| Measurement | Result |
|---|---:|
| Routed WNS | +0.254 ns |
| Routed TNS | 0 ns |
| 16-token prefill | 5.844550 s |
| 64-token prefill | 24.541660 s |
| 64-token-context decode | 0.742594 s/token |
| Compile-time context limit | 4096 tokens |

The 256-token prefill measurement in the original lab record was collected from an earlier 128-bit data-path stage and is not used as a strict scaling point for the final 16-lane bitstream.

## Interpretation

The dominant costs are model-weight movement, serialized layer scheduling, KV-cache traffic, and the tied language-model head. The measurements are board-specific reference points, not guaranteed results for other DDR configurations, temperatures, operating systems, or bitstreams.

## Required follow-up measurements

- Per-operator cycle counts and DMA byte counters
- 256/512/1024/2048/4096-token tests on one final bitstream
- Board power, temperature, and token/J
- Standard language-quality or perplexity evaluation
- Long-duration Linux and bare-metal stability tests
