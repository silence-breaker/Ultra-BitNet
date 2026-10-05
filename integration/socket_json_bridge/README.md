# 悟净 UART - ALINX - PC JSON 回传

## 固定通路

```text
悟净 PS USB-UART (Mini-USB)
  -> Mini-USB/USB-A 数据线
  -> AXU3EG USB3.0 HOST (Type-A)
  -> ALINX /dev/ttyUSB0
  -> serial_to_tcp_bridge.py
  -> AXU3EG PS ETH
  -> PC tcp_json_server.py:9000
```

悟净与 ALINX 分别供电。不得将两块板的 Mini-USB UART 口彼此直连，也不得使用
USB-A 对 USB-A 线。详细消息结构见 `PROTOCOL.md`。

## 1. PC

将 PC 网口配置为固定地址，例如 `192.168.137.1/24`，放行 TCP 9000。启动：

```powershell
cd <deployment_bundle>\socket_json_bridge\pc_receiver
python tcp_json_server.py --host 0.0.0.0 --port 9000 --log-file received_events.jsonl
```

每次启动会创建带时间戳的 JSONL 文件。事故与非事故消息都会落盘。

## 2. ALINX

将 `alinx_bridge` 目录复制到 AXU3EG Linux。连接悟净 PS USB-UART 后检查：

```bash
lsusb
dmesg | grep -i -E 'cp210|ttyUSB|usb'
ls -l /dev/ttyUSB*
```

若设备未出现，执行 `sudo modprobe cp210x`。桥接器只使用 Linux Python 3 标准库，
无需安装第三方依赖。启动：

```bash
cd /root/alinx_bridge
chmod +x start_bridge.sh
./start_bridge.sh 192.168.137.1 /dev/ttyUSB0
```

若当前用户无串口权限，使用 root 运行，或加入串口设备所属用户组。ALINX 只转发以
`ICRAFT_JSON:` 开头且能通过 JSON 校验的行，启动日志和普通调试日志会被丢弃。

确认手动转发正常后，可安装开机服务：

```bash
cd /root/alinx_bridge
chmod +x install_service.sh
./install_service.sh 192.168.137.1 /dev/ttyUSB0
journalctl -u icraft-uart-bridge.service -f
```

安装器把参数写入 `/etc/default/icraft-uart-bridge`。PC 地址或串口设备名变化时，修改该文件并执行
`systemctl restart icraft-uart-bridge.service`。

## 3. 悟净

重新交叉编译当前源码，替换部署包中的 `sdicamera+yolov8n+hdmi`，再把整个部署目录
复制到悟净板，例如 `/root/demo/PLin+SingleNet+HDMI`。

```bash
cd /root/demo/PLin+SingleNet+HDMI
chmod +x ./sdicamera+yolov8n+hdmi
./sdicamera+yolov8n+hdmi ./configs/ZG/sdicamera+yolov8n+hdmi.yaml
```

默认配置启用 UART JSON、关闭板端 TCP。板端每 10 秒输出一行
`ICRAFT_JSON:{"schema_version":"2.0",...}`。

USB-UART 接到 ALINX 后不再适合作为人工登录终端，建议安装仓库中的悟净开机服务：

```bash
chmod +x /root/demo/wujing_systemd/install_service.sh
/root/demo/wujing_systemd/install_service.sh
```

服务模板默认使用 `/dev/ttyPS0`。安装前必须在悟净板确认设备名；如果实际是
`/dev/ttyS0` 等名称，先修改服务文件的 `TTYPath` 和安装脚本中的设备名。安装脚本会
停用 `serial-getty@ttyPS0`，避免登录程序与检测程序争用同一串口；停用后请使用 SSH 或
另一调试口管理悟净。程序已使用 SIGINT/SIGTERM 等待，不再依赖终端回车保持运行。

## 4. 现场标定

YAML 中的 `horizontal_count_line_y` 和 `vertical_count_line_x` 是相对画面宽高的比例。
HDMI 上的黄色十字线就是实际计数线；调整参数并重启程序即可完成现场标定。

## 5. 启动顺序

1. 连接 SDI、HDMI、悟净 Mini-USB 到 AXU3EG USB Host、AXU3EG PS ETH 到 PC。
2. 启动 PC TCP 接收器。
3. 启动 ALINX，确认 `/dev/ttyUSB0` 后运行桥接器。
4. 启动悟净检测程序。
5. PC 应每 10 秒显示一次流量、事故数、灯态与剩余时间。

## 6. 故障定位

- ALINX 没有 `/dev/ttyUSB0`：检查数据线、USB Host 口和 `cp210x` 驱动。
- ALINX 能看到 JSON 但 PC 没数据：检查 PC 地址、防火墙和 9000 端口。
- ALINX 只看到启动日志：确认运行的是重新编译后的板端程序和新 YAML。
- PC 只有事故消息：确认使用本目录更新后的 `tcp_json_server.py`。
- 计数方向相反：摄像机坐标 y 轴向下，需要按现场安装方向调整映射。
