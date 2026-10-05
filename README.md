# Ultra-BitNet

面向 AXU3EGB（XCZU3EG-1SFVC784I）的 BitNet 推理加速器，使用 Scala / SpinalHDL 描述硬件。当前板级配置为 TP2、单物理矩阵引擎、四路 SPU 与五路 PS DDR 访问接口，控制域和计算域分别运行在 50 MHz 与 100 MHz。

## 仓库结构

- `doc/`：架构、板级接口、构建流程与已验证实现的信息。
- `src/main/scala/ultrabitnet/accel/`：板级顶层及其依赖的 Scala 源码。
- `src/vivado/`：AXU3EGB 工程、分阶段实现和时序收敛脚本。
- `build/ultra_bitnet_tp2.bit`：已完成布局布线和位流生成的板级镜像。

## 生成 RTL

使用 JDK 17、sbt 1.10.7、Scala 2.12.18 和 SpinalHDL 1.11.0：

```powershell
cd src
sbt compile generateBoardRtl
```

生成入口为 `ultrabitnet.accel.GenerateBitNetResidentBoardAccelerator`，输出为 `build/rtl/BitNetResidentBoardAccelerator.v`。首次运行 sbt 需要下载依赖。

## 设计说明

见 [加速器架构](doc/ARCHITECTURE.md)、[构建与实现](doc/BUILD.md) 和 [位流验证记录](doc/IMPLEMENTATION.md)。源码和位流使用 BSD-3-Clause 许可证，详见 [LICENSE](LICENSE)。

现有位流的 setup/hold 最差裕量为 `+1.291 / +0.008 ns`，可布线网络全部完成。完整模型数值验证、配套软件集成和板上推理仍需单独验证，当前没有板测 token/s 数据。
