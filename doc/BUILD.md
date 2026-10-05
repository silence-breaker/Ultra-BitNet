# 构建与实现

## Scala 到板级 RTL

工具版本为 JDK 17、sbt 1.10.7、Scala 2.12.18、SpinalHDL 1.11.0，版本定义在 `src/build.sbt` 和 `src/project/build.properties`。

从仓库根目录执行：

```powershell
Set-Location src
sbt compile generateBoardRtl
Set-Location ..
```

板级入口为 `ultrabitnet.accel.GenerateBitNetResidentBoardAccelerator`。生成文件为 `build/rtl/BitNetResidentBoardAccelerator.v`。源码中其他 `Generate...` 入口服务于局部研究或模块验证，不是发布镜像的板级入口。

## Vivado 板级工程

最终镜像由 Vivado 2025.2 在 `xczu3eg-sfvc784-1-i` 上实现。`src/vivado/board/design_1_bd.tcl` 提供 AXU3EGB PS 配置，板级脚本连接 50/100 MHz MMCM、复位、AXI-Lite SmartConnect、五路 DDR 端口和 IRQ。

生成 RTL 后，可从仓库根目录创建新的工程：

```powershell
vivado -mode batch -source src/vivado/build_bitnet_axu3egb_board.tcl -tclargs project
```

默认工程位于 `build/vivado-bitnet-r10-tp2-board/`。输出目录已存在时脚本拒绝覆盖，使用新的目录作为第二个 Tcl 参数。脚本还保留 `synth`、`impl`、`bitstream` 和 `xsa` 阶段。自动实现阶段的策略不保证重现已发布镜像的布局与时序，最终镜像曾经过分阶段 hold 收敛。

## 分阶段收敛

`stitch_bitnet_axu3egb_board_checkpoint.tcl` 合并各分区 DCP 并恢复 XDC scope。`implement_bitnet_axu3egb_board_checkpoint.tcl` 使用 `BITNET_BOARD_IMPL_IN`、`BITNET_BOARD_IMPL_OUT` 和 `BITNET_BOARD_IMPL_STAGE` 指定输入 DCP、输出目录及 `place`、`route` 或 `bitstream` 阶段。

已验证路线先进行完整路由，再由 `close_bitnet_axu3egb_board_hold.tcl` 应用协议相关的 hold 处理，最后通过 `retry_bitnet_axu3egb_board_hold.tcl` 的 `route_aggressive` 模式执行 `AggressiveExplore` 路由和 `ExploreWithHoldFix`。

这些脚本保留输入时钟、层次和路径检查。原 hold 流程包含 slow/fast 时钟方向级 hold exception，以及经过分类检查的 SmartConnect pin-pair hold exception；它们依赖当前 mailbox 协议。修改 CDC 或引入新跨域路径时必须重新审查，不能把 exception 当成通用的时序修复方法。setup、max-delay 和 bus-skew 的适用检查仍须保留。

位流生成前需确认完全路由、WNS/WHS 非负、CDC/check_timing 无未解释问题，并通过 `write_bitstream` 前置 DRC。已发布位流直接存放于 `build/ultra_bitnet_tp2.bit`，无需重新综合即可用于配套板级验证。

## 发布镜像与重建结果

发布位流可直接用于匹配板级配置的验证。源码、工具版本和实现策略的变化可能影响重新生成的位流；重新布局布线后的签核须按新的报告执行。

仓库不包含模型权重、PS 软件运行镜像、Vivado DCP 和工具缓存。
