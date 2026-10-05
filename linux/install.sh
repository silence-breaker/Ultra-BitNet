#!/bin/sh
set -eu

PACKAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PREFIX=${PREFIX:-/opt/bitnet}
FIRMWARE_DIR=${FIRMWARE_DIR:-/lib/firmware}

if [ "$(id -u)" -ne 0 ]; then
    echo "请使用 sudo 运行 install.sh" >&2
    exit 1
fi

install -d "$PREFIX" "$FIRMWARE_DIR"
if [ ! -f "$PACKAGE_DIR/bin/bitnet-linux" ]; then
    echo "未发现预编译程序，正在本机编译..."
    make -C "$PACKAGE_DIR" clean all
    install -d "$PACKAGE_DIR/bin"
    cp "$PACKAGE_DIR/build/bitnet-linux" "$PACKAGE_DIR/bin/bitnet-linux"
fi
if [ -f "$PACKAGE_DIR/model/BITNET.tar" ]; then
    tar -xf "$PACKAGE_DIR/model/BITNET.tar" -C "$PREFIX"
else
    cp -a "$PACKAGE_DIR/model/BITNET" "$PREFIX/"
fi
install -m 0644 "$PACKAGE_DIR/firmware/bitnet_accel.bit.bin" \
    "$FIRMWARE_DIR/bitnet_accel.bit.bin"
install -m 0755 "$PACKAGE_DIR/bin/bitnet-linux" /usr/local/bin/bitnet-linux
install -d /etc/profile.d
install -m 0644 "$PACKAGE_DIR/bitnet-split.sh" /etc/profile.d/bitnet-split.sh

echo "安装完成。设备树保留内存和匹配本 XSA 的 FSBL 必须已经在启动时生效。"
echo "加载 PL: echo bitnet_accel.bit.bin > /sys/class/fpga_manager/fpga0/firmware"
echo "推理: sudo bitnet-linux --model $PREFIX --prompt 'Hello' --tokens 8"
echo "PL-DDR 双 lane 和零拷贝 DMA 默认开启。"
echo "仅在旧硬件回退时设置 BITNET_PLDDR_ENABLE=0 BITNET_DIRECT_DMA=0。"
