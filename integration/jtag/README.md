# JTAG Integration

This directory contains the XSDB smoke-test entry point used by the optional UART/JTAG relay.

The script intentionally does not include a board-specific ELF, bitstream, model, or PS initialization file. Set these variables before running it:

```powershell
$env:BITNET_ELF = "C:\path\to\a.elf"
$env:BITNET_BITSTREAM = "C:\path\to\axu3egb_bitnet_accel.bit"
$env:BITNET_PSU_INIT = "C:\path\to\psu_init.tcl"
$env:BITNET_HW_SERVER = "tcp:127.0.0.1:3121"
$env:BITNET_XSDB = "C:\Xilinx\Vitis\2020.1\bin\xsdb.bat"
```

Start Vivado `hw_server`, connect the AXU3EGB board through JTAG, and then invoke `integration/pc_relay/serial_jtag_cloud_relay.ps1`. The relay also requires `BITNET_MODEL` to point to a locally licensed tokenizer/model directory.

Do not commit the ELF, bitstream, model assets, hardware-server logs, or board-specific paths.
