# AXU3EGB BitNet LLM 推理加速器性能技术报告

**报告日期：** 2026-07-18  
**目标平台：** ALINX AXU3EGB / Xilinx Zynq UltraScale+ MPSoC  
**模型：** BitNet b1.58 2B 4T  
**实现工具：** Vivado/Vitis 2020.1  
**报告对象：** 当前最终实现（PL 125 MHz、128-bit DMA、16-lane PL Attention）

## 1. 摘要

本项目在 AXU3EGB 上实现了可执行完整自回归推理的 BitNet b1.58 FPGA 加速器。系统不是单独的矩阵乘法演示：它包含模型权重读取、tokenizer、30 层 Transformer、三值权重 GEMV、RoPE、完整 Attention、KV cache、归一化、残差连接和 tied LM head，并已经在板卡上跑通从文本输入到 token 输出的完整链路。

最终实现的主要结果如下：

| 类别 | 最终结果 |
|---|---:|
| PL 工作频率 | 125 MHz |
| routed WNS / TNS | +0.254 ns / 0 ns |
| DMA MM2S、HP0、GEMV 数据宽度 | 128 bit |
| PL Attention 并行度 | 16 lane |
| LUT | 27,944（39.60%） |
| FF | 28,846（20.44%） |
| BRAM | 45.5（21.06%） |
| DSP | 243（67.50%） |
| 16-token prefill | 5.844550 s，2.738 token/s |
| 64-token prefill | 24.541660 s，2.608 token/s |
| 256-token prefill（128-bit 阶段版本） | 116.707098 s，2.193 token/s |
| 64-token context decode | 0.742594 s/token，1.347 token/s |
| 编译上下文上限 | 4096 token |
| KV cache 占用 | 约 150 MiB |

综合来看，当前设计已经达到“完整模型可运行、PL 时序收敛、数值路径基本一致、长上下文可执行”的阶段。其主要性能限制已从纯计算资源转向模型权重和中间数据的 DDR 搬运、串行层调度以及 LM head 等端到端开销。

## 2. 系统与模型配置

### 2.1 模型配置

当前部署模型为 BitNet b1.58 2B 4T，主要配置为：

| 参数 | 数值 |
|---|---:|
| Transformer 层数 | 30 |
| Query heads | 20 |
| KV heads | 5 |
| Head dimension | 128 |
| GQA 分组 | 4 |
| Vocabulary size | 128,256 |
| 最大编译上下文 | 4096 token |
| 权重量化 | 三值（-1、0、+1） |

模型文件、tokenizer、RoPE 表和辅助缩放参数以 BIN 文件存放在 eMMC/Linux 文件系统中。完整模型归档约 1.414 GiB。

### 2.2 硬件数据通路

最终数据通路由以下模块构成：

1. PS 负责模型调度、文件读取、tokenizer、层级控制和端到端推理流程。
2. AXI DMA 通过 HP0 端口在 DDR 与 PL 之间搬运激活、权重和 Attention 数据。
3. GEMV 核执行三值权重矩阵向量运算。三值权重的数据路径以加法、减法和跳过零权重为核心，避免把每个权重都作为通用乘法处理。
4. Attention 核在 PL 中执行 RoPE、QK 点积、缩放、online softmax 和对 V 的加权累加，并使用 KV cache 支持自回归推理。
5. DMA MM2S、HP0 和 GEMV 主数据路径扩展到 128 bit；Attention 分支通过 AXIS Width Converter 转换为核侧 64-bit 接口。
6. Attention 内部采用 16-lane 并行结构，以提高多 head、head dimension 和历史 token 方向的数据吞吐率。

## 3. 实现结果

### 3.1 时序

最终 routed 结果：

| 指标 | 结果 |
|---|---:|
| PL 目标频率 | 125 MHz |
| 时钟周期 | 8.000 ns |
| WNS | +0.254 ns |
| TNS | 0 ns |
| 时序结论 | 满足约束 |

正 WNS 表明最终布局布线后的最差路径仍有 0.254 ns 裕量。在当前器件和布局下，125 MHz 是经过实现验证的工作频率。报告不依据 WNS 线性外推更高频率；进一步提频需要重新综合、布局布线和板测。

### 3.2 资源占用

| 资源 | 已使用 | 利用率 | 评价 |
|---|---:|---:|---|
| LUT | 27,944 | 39.60% | 仍有逻辑扩展空间 |
| FF | 28,846 | 20.44% | 余量充足 |
| BRAM | 45.5 | 21.06% | 可继续增加片上缓冲 |
| DSP | 243 | 67.50% | 当前最紧张的计算资源 |

资源结构说明：

- LUT、FF 和 BRAM 均未超过 40%，说明增加权重缓存、双缓冲、地址生成和局部流水线仍有实现空间。
- DSP 使用率达到 67.50%，后续不宜简单依靠继续堆叠乘法器提升并行度。
- BitNet 三值权重适合继续将 GEMV 中可替代的乘法映射为加/减/零跳过逻辑，把 DSP 优先留给 Attention 缩放、softmax 近似、量化缩放和其他确实需要乘法的路径。

## 4. 板级功能与数值验证

### 4.1 功能链路

最终 bitstream 已通过板级 smoke test：

```text
BOARD_SMOKE_PASS
```

30 层 greedy 推理已经完成从输入文本到 LM head/tokenizer 输出的完整执行，测试输出 token ID 为 11，对应文本为 `","`。这一结果证明控制、模型文件、DMA、30 层计算、KV cache 和输出链路能够协同运行，但单个 token 输出不能替代语言质量评测。

### 4.2 数值一致性

与离线参考结果比较得到：

| 检查点 | Cosine similarity |
|---|---:|
| Hidden state | 0.998309381 |
| Q projection | 0.999938967 |
| Final norm | 0.977565707 |

Q projection 和 hidden state 的余弦相似度均接近 1，说明 GEMV、激活量化和主要数据搬运路径与参考实现高度一致。Final norm 的相似度相对较低，说明量化误差、定点缩放和跨层误差累积仍会影响最终 logits。

这些结果应解释为“选定中间张量的数值一致性验证”，不能解释为模型准确率或生成质量。此前对 Hugging Face `AutoBitLinear` 的核查表明，参考输出还需要乘以逐层 `weight_scale`；在完成板端读取 AUX 中全部 scale、逐层对齐并重新跑完整回归前，固定 `/256` 等近似缩放不能视为严格数值等价。因此，本报告保守地把 `weight_scale` 端到端等价性列为未闭环项。当前也尚未完成 perplexity、标准任务集准确率或大规模生成质量评测。

## 5. 端到端性能

### 5.1 测试口径

本报告采用现有板级日志。16/64-token prefill 与 64-token context decode 来自最终 16-lane Attention 阶段；256-token prefill 来自此前 128-bit 主数据通路阶段，作为长上下文已运行参考点，不用于严格计算最终 16-lane 版本的长度缩放规律。Prefill 时间覆盖给定上下文 token 的完整 30 层处理；decode 指标包含 LM head 等端到端开销。吞吐率按下式换算：

```text
prefill 吞吐率 = 输入 token 数 / prefill 总时间
decode 吞吐率  = 1 / 平均每 token decode 时间
```

### 5.2 Prefill

| 输入长度 | 总时间 | 平均延迟 | 吞吐率 |
|---:|---:|---:|---:|
| 16 token | 5.844550 s | 365.284 ms/token | 2.738 token/s |
| 64 token | 24.541660 s | 383.463 ms/token | 2.608 token/s |
| 256 token（128-bit 阶段版本） | 116.707098 s | 455.887 ms/token | 2.193 token/s |

观察结果：

- 在同为最终 16-lane 阶段的 16/64-token 两点之间，prefill 吞吐率下降约 4.7%。
- 256-token 点相对 16-token 点低约 19.9%，但两者来自不同优化阶段，该比例只能作为参考，不能归因于上下文增长本身。
- 现有数据呈现平均每 token 时间随上下文增长而增加的现象，但还需要用同一最终 bitstream 补测 256/512 token 才能建立可信的缩放曲线。
- 当前 prefill 仍按 token 逐步执行，没有形成类似 GPU 批量 prefill 的大矩阵并行，因此长 prompt 的吞吐提升有限。

### 5.3 Decode

64-token context 下的端到端 decode 结果：

| 指标 | 结果 |
|---|---:|
| 平均 decode 延迟 | 0.742594 s/token |
| Decode 吞吐率 | 1.347 token/s |
| 8-token 估算生成时间 | 约 5.94 s |
| 32-token 估算生成时间 | 约 23.76 s |

后两项为按照平均 decode 延迟进行的线性估算，不包含 prompt prefill，也不代表不同上下文长度下的实测值。

### 5.4 上下文能力

当前编译上限为 4096 token，KV cache 约占 150 MiB。上下文上限来自软件地址规划、缓存容量和接口字段约束，不代表已经对 4096-token 输入完成端到端稳定性和耗时实测。

已完成的最长公开性能点为 256-token prefill。因此，报告结论应区分：

- **实现支持上限：** 4096 token；
- **当前性能实测上限：** 256-token prefill；
- **当前 decode 性能点：** 64-token context。

## 6. 性能瓶颈分析

### 6.1 权重带宽

约 1.414 GiB 的模型资产无法完整驻留片上存储。每层推理需要从 eMMC/DDR 层次读取或搬运大量权重，三值计算本身虽然减少了乘法复杂度，但不能自动消除权重读取成本。当前性能的首要约束是权重和激活在存储层次间的移动，而不是单个三值乘加的运算数量。

优化优先级：

1. 将权重从 eMMC 预加载或映射到 DDR，避免推理热路径中的文件系统开销。
2. 使用 ping-pong buffer 重叠“下一块权重搬运”和“当前块 GEMV 计算”。
3. 提高 DMA burst 连续性和块尺寸，减少事务启动与尾部 padding 开销。
4. 在 BRAM 余量内缓存高复用的缩放参数、RoPE 表、层元数据和小矩阵。
5. 重新安排三值权重打包，使 128-bit 总线每拍传输的数据都能被并行 lane 有效消费。

### 6.2 层间串行和 DMA 启动开销

30 层 Transformer 仍按层串行执行，每个算子之间存在控制、DMA 启停和缓冲切换成本。对短向量 GEMV，固定启动开销占比较明显。应通过命令队列、批量描述符或数据流融合减少 PS 参与频率。

### 6.3 Attention 随上下文增长

16-lane PL Attention 已降低单次 Attention 延迟，但每个新 token 仍需要读取历史 K/V。随着上下文长度增长，读流量和 QK/V 累加工作量近似线性增加，这与 prefill 单 token 延迟由 365.284 ms 增长到 455.887 ms 的趋势一致。

可进一步考虑：

- K/V 分块与片上窗口缓存；
- RoPE 查表和多 bank ROM；
- QK、online softmax、V accumulation 之间更深的数据流流水；
- 对齐 KV cache 布局，使 128-bit burst 与 16-lane 消费顺序一致；
- 在资源允许时增加 Attention lane，但必须同时检查 DDR 带宽是否已经饱和。

### 6.4 LM head

Vocabulary size 为 128,256，tied LM head 每个 decode token 都需要处理大规模输出投影和 top-k/greedy 选择。decode 指标包含 LM head，因此不能只依据 Transformer 主干核延迟估算端到端速度。可评估分块 LM head、并行 top-k、候选剪枝以及权重流与计算重叠。

## 7. 当前成果的适用边界

当前可以确认：

- 最终 PL 设计在 125 MHz 下通过 routed timing；
- 128-bit DMA 和 16-lane Attention 硬件链路已板测跑通；
- 完整 30 层推理能够生成 token；
- 关键中间张量与参考实现具有较高余弦相似度；
- 16、64、256-token prefill 和 64-token context decode 已取得板级数据；
- 软件编译支持 4096-token 上下文。

当前不能据此确认：

- 4096-token 上下文已完成稳定性和性能验证；
- 模型达到官方浮点/Hugging Face 实现相同的 perplexity 或任务准确率；
- 当前设计具有某个确定的 TOPS/W 或 token/J 指标；
- Linux 用户态兼容层与裸机性能完全相同；
- 单次 `BOARD_SMOKE_PASS` 等价于长时间压力测试通过。

缺失的关键测量包括板级功耗、DDR 实际带宽计数器、各算子周期分解、温度、长时间稳定性、标准语言任务质量以及 Linux 版本的实板性能。

## 8. 下一阶段建议

建议按以下顺序推进，保证每轮都能独立验证收益：

| 优先级 | 工作项 | 目标指标 |
|---:|---|---|
| P0 | 增加逐算子周期与 DMA 字节计数 | 得到 GEMV、Attention、LM head、搬运的精确占比 |
| P0 | 权重 DDR 常驻与双缓冲 | 降低 eMMC/文件读取和 DMA 空泡 |
| P1 | 权重布局与 128-bit burst 对齐 | 提高有效带宽，减少 padding |
| P1 | GEMV/Attention 数据流重叠 | 降低层内串行等待 |
| P1 | 用最终 bitstream 补测 256/512/1024/2048/4096 上下文 | 验证真实最大上下文和性能曲线 |
| P2 | LM head 分块与并行 top-k | 降低 decode 端到端延迟 |
| P2 | 板级功耗测量 | 给出 token/J 和能效结论 |
| P2 | Perplexity/标准数据集评测 | 确认量化与定点缩放后的模型质量 |

性能优化验收应至少同时报告：prefill token/s、decode token/s、上下文长度、功耗、DDR 带宽、时钟频率和生成质量。只报告核级 cycle 或理论峰值不能代表完整 LLM 用户体验。

## 9. 结论

本项目已经在资源受限的 AXU3EGB 上完成 BitNet b1.58 2B 模型的完整 FPGA 推理链路。最终设计以 39.60% LUT、20.44% FF、21.06% BRAM 和 67.50% DSP 实现 125 MHz 时序收敛，并在最终 16-lane 阶段达到 16-token prefill 2.738 token/s、64-token prefill 2.608 token/s，以及 64-token context decode 1.347 token/s。此前 128-bit 阶段还完成了 256-token prefill，结果为 2.193 token/s。

当前实现证明了三值权重 GEMV与完整 PL Attention 在该器件上的可行性。下一阶段最有价值的工作不是单纯增加算术单元，而是测清并解决权重带宽、DMA 空泡、层间串行和 LM head 开销；同时补齐长上下文、功耗和模型质量测试，才能形成完整的工程性能与能效结论。

## 10. 证据文件

本报告主要依据以下工程产物：

```text
build/vivado/axu3egb_bitnet_accel.runs/impl_1/
  design_1_wrapper_timing_summary_routed.rpt
  design_1_wrapper_utilization_placed.rpt

build/
  board_smoke_16lane.log
  infer_hello_30layer_16lane.log
  perf_prefill_16_16lane.log
  perf_prefill_64_16lane.log
  perf_prefill_256_128bit.log
  perf_decode_64ctx_8_16lane.log

sw/baremetal/main.c
hls/bitnet_attention_axis.cpp
READMD.md
```

不同命名日志代表不同优化阶段。本报告的架构结论以最终 125 MHz、128-bit 主数据通路和 16-lane Attention 实现为准；如后续重新生成 bitstream 或修改软件量化路径，应重新运行同一组板测并更新报告。
