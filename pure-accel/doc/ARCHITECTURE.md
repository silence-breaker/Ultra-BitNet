# BitNet Accelerator Architecture

## System Partitioning

The processing system (PS) handles tokenization, embedding preparation, model-image loading, and round submission. Programmable logic (PL) handles layer scheduling, matrix operations, attention, activation processing, and final LM/argmax. Identity fields associate inputs, memory transactions, and completion events with a submitted request. Results return through registers, with an optional interrupt.

Model dimensions are fixed at elaboration: 30 layers, hidden size 2560, FFN size 6912, vocabulary 128256, 20 query heads, 5 KV heads, 128 features per head, and maximum sequence length 4096. `BitNetConfig.Production` selects token parallelism 2, one logical and physical matrix engine, and four SPUs. Some internal types retain `Tp4` in their names because they describe generalized interfaces and scheduling; instance counts come from the production configuration.

## Organization

![Ultra-BitNet TP2 architecture: Linux software submits work over AXI-Lite; event-driven PL services share compute and tensor resources; five AXI ports access the same PS DDR.](assets/architecture.png)

[Open the scalable diagram](assets/architecture.svg)

Arrows show major control and data interactions. Local FIFOs, resets, and individual signal connections are omitted. Operator services share physical resources; the five DDR interfaces do not represent five independent memories.

`BitNetResidentBoardAccelerator.scala` is the board-level top. `BitNetResidentSessionInferenceTop.scala` connects the session controller to the complete compute shell. `BitNetResidentPreloadedHiddenFullLayerShell.scala` owns and connects the physical resources. The session controller retains round/session state; the compute shell owns the matrix array, tensor storage, and memory hub.

## Event-Driven Layer Execution

`BitNetResidentLayerEventScheduler` advances when data is committed, not merely when an operator accepts a command. RMS, QKV, attention, output projection, residual, Gate/Up, ReLU-squared, and Down cooperate through identity-bearing commands and completions. Intermediate prefill boundaries are distinguished from final-prompt and decode boundaries requiring final RMS/LM.

Prefill and decode reuse physical resources; mode and configuration determine token/head mapping. Request/session, epoch, and generation fields reject stale completions, out-of-order beats, and out-of-range accesses. Serious protocol errors enter fail-stop/quarantine so invalid results cannot be returned as valid tokens.

## Matrix Compute and Shared Arithmetic

`BitNetContinuousMatrixArrayCore` uses DSP48E2 cascades for continuous matrix computation. Weight conversion precedes the array, keeping the high-frequency path focused on fixed-width multiply-accumulate, accumulation, and output. The production configuration has eight chains; modules such as `BitNetTernaryMap0DspOperandCodec` map ternary weights to DSP operands.

`BitNetTp4MatrixCluster` and resource leases share one physical TP2 core. Linear, QK/SV, and LM use it under ownership and retirement rules rather than duplicating a matrix engine for every operator. FIFOs, skid buffers, and register boundaries absorb backpressure across weight preparation, compute, and return paths.

Non-matrix arithmetic is fixed-point: RMSNorm, dynamic i8 quantization, RoPE, QK scaling, softmax, residual addition, and FFN. FFN activation computes `g > 0 ? g*g*u : 0`; its interpretation depends on the model's quantization, rounding, and saturation rules. `BitNetSharedSpuArithmeticPool` and its adapters organize shared arithmetic with explicit widths and overflow behavior.

## Tensor and Weight Storage

`BitNetUnifiedResidentTensorProductionShell` combines the semantic frontend, related-clock bridge, and physical memory backend. Projection, Query, and LM access it through a packed router. Query/LM aliases are authenticated by owner and generation.

Matrix and LM share a two-slot weight-tile cache, allowing preparation of the next tile while the current tile is consumed. Stage retirement waits for both compute and memory consumption. Packed KV append/read, attention context, and LM candidate FIFOs use dedicated formats and local BRAM to limit wide-data replication in control paths.

## Five-Bank DDR Data Path

`BitNetProductionPlMemoryOwnershipHub` and `BitNetFiveBankAxiMemoryComplex` arbitrate Matrix, LM, KV, auxiliary, and weight-staging traffic. Each bank tracks AXI tags, responses, and transaction ownership. Independent quotas prioritize compute while allowing staging to progress.

All five interfaces in this board image access PS DDR. Bank 0 uses HPC0; banks 1 through 4 use HP0 through HP3. Bank 0 starts at physical address `0x850000000`; banks 1 through 4 start at `0x800000000`, `0x810000000`, `0x820000000`, and `0x830000000`. The source-staging region starts at `0x840000000`. The image does not instantiate a PL DDR MIG.

The round descriptor carries an HP0-relative source offset, not a physical address. The default `source_offset = 0x40000000` is translated by the bank-1 AXI adapter using base `0x800000000`, yielding `0x840000000`. The staging controller requires at least 104693760 source bytes. After first-epoch staging, unchanged model epochs reuse resident Transformer weights.

## Board ABI and Clocks

The AXI-Lite control window starts at `0x80000000`. `BitNetResidentBoardRegister` defines submit, session/image epoch, round parameters, source range, result, fault, IRQ enable, and hidden PIO registers. The hardware identity remains `0x48425432` to match the released bitstream.

Each hidden beat holds 2 token lanes x 4 SPU shards x 32 bits, or 256 bits. Software writes eight registers starting at `0x80`, sets last at `0xa0`, and pushes at `0xa4`; `0xa8` reports PIO status. Every round transfers 640 groups. Words are ordered lane 0 shards 0..3, then lane 1 shards 0..3; feature index is `group * 4 + shard`. Values are signed Q16.16, the inactive second lane is zero, and only group 639 carries last. The Scala top exposes AXI Stream, but the released block design ties that input to zero and uses PIO.

The PS supplies an approximately 50 MHz reference. A common MMCM generates related 50 MHz slow and 100 MHz fast clocks. Memory/control use slow; the array uses fast. Cross-domain mailboxes retain payload until request/ack completes. Protocol behavior and timing constraints must be reviewed together; globally declaring these clocks asynchronous would hide real timing checks.

## Linux Software Integration

`src/software/bitnet.c` implements the round protocol through injectable MMIO/time callbacks. It validates descriptors before submission, honors PIO backpressure, checks input and completion identities, captures results before ACK, and bounds the whole round by one monotonic timeout. A failed round requires recovery before that driver instance can be reused.

`bitnet_linux.c` maps UIO map 0 (or an explicitly selected `/dev/mem` window), performs ordered 32-bit accesses, and takes an exclusive device-file lock. `bitnetctl` exposes `probe`, `status`, and `round`, returning machine-readable JSON. The runtime polls completion; interrupts are not required.

Python tools export the local packed model, validate its package, convert Map1 Transformer weights to Map0 when needed, arrange weights into five bank images, initialize KV storage, load reserved PS DDR, and convert BF16 embeddings or JSON vectors to Q16.16 hidden input. Image layout metadata describes addresses and ranges. Tokenization and the application-level generation loop remain host responsibilities: submit prefill rounds, use the returned token's embedding for decode, and advance using `next_position` while retaining session/image epoch.
