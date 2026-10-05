# BitNet 加速器设计架构

## 设计目标与边界

设计面向固定模型的 resident 推理数据流。PS 负责分词、embedding 准备、权重镜像装载和任务提交；PL 负责层内调度、矩阵计算、注意力、激活处理以及最终 LM/argmax。一次提交通过身份标记关联输入、内存事务和完成事件，结果经寄存器和中断返回 PS。

模型维度在 elaboration 时确定：30 层、hidden size 2560、FFN size 6912、词表 128256、20 个 query head、5 个 KV head、每 head 128 维，最大序列长度 4096。当前板级使用 `BitNetConfig.Production`，tokenParallelism 为 2，engineCount 和 physicalMatrixEngineCount 均为 1，spuCount 为 4。部分内部类型名保留 `Tp4`，表示通用接口和调度组织；实际实例数量由生产配置决定，不能由类型名推断。

## 分层组织

![Ultra-BitNet TP2 板级架构：PS 软件经 AXI-Lite 提交任务，PL 内部以事件调度共享计算与存储资源，五路 AXI 接口连接同一 PS DDR。](assets/architecture.png)

[查看可缩放矢量图](assets/architecture.svg)

图中箭头表示主要控制和数据交互，省略局部 FIFO、复位及逐信号连接。计算壳内的算子服务共享物理资源；五路 DDR 接口访问同一 PS DDR 控制器，并非五套独立存储器。

板级顶层在 `BitNetResidentBoardAccelerator.scala`。`BitNetResidentSessionInferenceTop.scala` 连接 session controller 和完整计算壳；`BitNetResidentPreloadedHiddenFullLayerShell.scala` 汇聚各物理资源并建立完整数据通路。session 控制器保存 round/session 状态，计算壳是矩阵阵列、tensor 存储和 memory hub 的唯一所有者。

## 事件驱动的层执行

`BitNetResidentLayerEventScheduler` 以数据真正提交为推进条件。RMS、QKV、注意力、输出投影、残差、Gate/Up、ReLU² 和 Down 之间通过带身份的命令与 completion 协作；前一模块仅发出命令不会被当成计算完成。继续 prefill 的调度边界与需要 final RMS/LM 的末轮或 decode 边界分开处理。

prefill 和 decode 复用物理计算资源，由配置和模式决定 token/head 映射。请求携带 request/session、epoch、generation 等字段，拒绝过期 completion、错序 beat 和越界访问。严重协议错误进入 fail-stop/quarantine，防止错误结果被当作有效 token 返回。

## 矩阵计算与共享算术

`BitNetContinuousMatrixArrayCore` 以 DSP48E2 级联执行连续矩阵计算。权重格式转换在阵列前完成，使高频路径专注于定长乘加、累加和结果输出。生产配置使用八条 chain；三值权重经 `BitNetTernaryMap0DspOperandCodec` 等模块映射到 DSP 操作数。

`BitNetTp4MatrixCluster` 与 lease 控制复用实际单个 TP2 核。Linear、QK/SV、LM 等操作按所有权和退休条件使用计算资源，不为每个算子复制完整矩阵引擎。权重准备、计算和返回路径使用 FIFO、skid buffer 与寄存边界吸收反压。

非矩阵算术采用固定点实现，包括 RMSNorm、动态 i8 量化、RoPE、QK scaling、softmax、残差和 FFN。FFN 激活计算为 `g > 0 ? g*g*u : 0`，需要与模型量化、舍入和饱和规则共同核对。`BitNetSharedSpuArithmeticPool` 及各 adapter 组织共享算术资源；接口中保留明确的位宽和溢出规则。

## Tensor 与权重存储

`BitNetUnifiedResidentTensorProductionShell` 将 semantic frontend、related-clock bridge 和物理存储后端连接为统一 tensor 子系统。Projection、Query、LM 通过 packed router 访问该资源，Query/LM 的 alias 租用通过 owner/generation 认证。

Matrix 和 LM 使用共享 weight tile cache；双槽结构允许准备下一 tile 时继续消费当前 tile。stage retirement 必须等待计算和内存消费结束。packed KV append/read、attention context 和 LM candidate FIFO 使用专用格式与局部 BRAM，减少宽数据在控制路径中的复制。

## 五银行 DDR 数据通路

`BitNetProductionPlMemoryOwnershipHub` 和 `BitNetFiveBankAxiMemoryComplex` 统一仲裁 Matrix、LM、KV、辅助数据和权重搬运流量。每个 bank 维护 AXI tag、读写响应和事务归属；独立配额在保持计算优先的同时允许 staging 前进。

当前 AXU3EGB 镜像的五个接口全部连接 PS DDR：bank 0 经 HPC0，bank 1 至 4 经 HP0 至 HP3。它们是同一 PS DDR 系统的五个访问通道，不是五套独立 DRAM。bank 0 的板级地址偏移为 `0x8_5000_0000`；其他四个逻辑银行基址依次为 `0x8_0000_0000` 至 `0x8_3000_0000`，`0x8_4000_0000` 区域用于源数据 staging。当前镜像不实例化 PL DDR MIG。

## 板级 ABI 与时钟

AXI-Lite 控制窗口基址为 `0x80000000`。寄存器表由 `BitNetResidentBoardRegister` 定义，包含 submit、session/image epoch、round 参数、PS source 地址、result、fault、IRQ enable 和 hidden PIO。

hidden 每拍包含 2 个 token × 4 个 SPU × 32 bit，即 256 bit；PS 通过 `0x80` 起的 8 个数据寄存器、`0xa0` 的 last 和 `0xa4` 的 push 输入数据，`0xa8` 查询状态。Scala 顶层保留 AXI Stream 接口，但当前 BD 将该输入绑零，实际使用 PIO。软件须等待可接收状态，提交后读取带身份的结果并 ACK。硬件 identity 数值固定为 `0x48425432`，用于匹配已发布镜像。

PS 提供约 50 MHz 参考时钟，MMCM 生成相关的 slow 50 MHz 和 fast 100 MHz。内存接口与控制在 slow 域，阵列在 fast 域。跨域 mailbox 保持 payload 直到 request/ack 完成；时序约束和协议行为必须一起检查。构建脚本不应将相关时钟整体声明为异步，以免隐藏真实约束问题。

## 实现边界

布局布线通过说明当前器件、频率和约束下存在可实现的网表。它不替代软件 ABI、DDR 初始化、缓存一致性、模型镜像和端到端数值正确性验证。完整模型推理的性能应以匹配镜像后的实测为准。
