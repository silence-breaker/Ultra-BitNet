#!/bin/sh
set -eu

PACKAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PREFIX=${PREFIX:-/opt/bitnet}
FIRMWARE_DIR=${FIRMWARE_DIR:-/lib/firmware}

if [ "$(id -u)" -ne 0 ]; then
    echo "Run install.sh with sudo." >&2
    exit 1
fi

install -d "$PREFIX" "$FIRMWARE_DIR"
if [ ! -f "$PACKAGE_DIR/bin/bitnet-linux" ]; then
    echo "Prebuilt binary not found; building locally..."
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

echo "Installation complete. Reserved memory and the matching XSA/FSBL must be active at boot."
echo "Load PL: echo bitnet_accel.bit.bin > /sys/class/fpga_manager/fpga0/firmware"
echo "Inference: sudo bitnet-linux --model $PREFIX --prompt 'Hello' --tokens 8"
echo "PL-DDR dual-lane and direct DMA are enabled by default."
echo "For legacy hardware only: BITNET_PLDDR_ENABLE=0 BITNET_DIRECT_DMA=0."
