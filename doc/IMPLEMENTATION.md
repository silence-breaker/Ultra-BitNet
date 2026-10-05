# 位流与源码验证记录

## 发布镜像

位流：`build/ultra_bitnet_tp2.bit`。

- 生成日期：2026-08-24。
- 器件：XCZU3EG-1SFVC784I，Vivado 2025.2。
- slow/fast 时钟：50/100 MHz，同一 MMCM 的相关时钟。
- 文件大小：5,568,804 bytes。
- SHA-256：`497e3797a7668be4c0856ffa4dec455588f949b17959151e6818a0df833a1ee8`。
- 输入 routed DCP SHA-256：`45d0b6bd41caa151b71341a2b4e9727be54210ede1f50d9d471910ee398d18b8`。

| 验证项 | 发布结果 |
| --- | --- |
| 完整路由网络 | 135674 / 135674 |
| Routing errors / node overlaps | 0 / 0 |
| Setup WNS / TNS | +1.291 ns / 0 |
| Hold WHS / THS | +0.008 ns / 0 |
| Hold failing endpoints | 0 |
| CDC 报告 | All paths are Safely Timed |
| check_timing | 各项计数为 0 |
| DRC | 0 Error / 0 Critical Warning |
| 普通 DRC Warning / Advisory | 455 / 90 |
| Bitgen 前置 DRC | 0 Error |

普通 DRC 项包括 DSP 输入/输出流水化建议、异步 load 检查、无可布线负载和省电建议。记录上述数量是为了保留验收范围，不表示所有建议已经消除。时序结果是在所采用协议相关约束下得到的。

## 源码迁移验证

验证日期：2026-10-05。板级 Scala 源码与位流记录对应的源码快照之间没有改动。板级依赖闭包包含 145 个 Scala 文件，另含 1 个局部研究生成器的入口检查工具，共 146 个源码文件。

改名后使用 Scala 2.12.18 与 SpinalHDL 1.11.0 编译成功，并成功 elaboration 完整板级顶层。生成 RTL 与当时综合使用的板级 RTL 在统一符号名称并忽略注释和空白后，2,233,477 个 token 完全一致。归一化 token 序列 SHA-256 为 `9d2c6f6c9382cbff5f494f3b2aab5a0fe0ea45124817937600615c53df39b8b2`。

在新仓库内执行 `sbt -batch compile generateBoardRtl`，实际编译全部 146 个 Scala 文件并成功生成板级 Verilog。配套 ROM 初始化文件也与发布设计逐字节一致。elaboration 仍有异步读存储的 write-first 提示，与原设计的结构一致。

本次没有重新进行 Vivado 布局布线，也没有执行板上全模型推理。发布 `.bit` 是原镜像的逐字节复制。源代码的包名和组件名更新不改变固定寄存器地址、identity 数值、计算位宽和数据格式。
