# AXU3EGB Release Assets

This directory contains the validated bare-metal board assets copied from the deployment bundle:

- `a.elf`: standalone AArch64 application for JTAG/XSDB loading.
- `axu3egb_bitnet_accel.bit`: AXU3EGB accelerator bitstream.
- `axu3egb_bitnet_accel.xsa`: exported hardware platform.
- `psu_init.tcl`: PS initialization script.

The repository currently does not contain generated tokenizer or layer-bundle BIN files. Generate them from `src/models/official_model` with the commands in `docs/OPERATION_GUIDE.md` before running inference. The model and tokenizer are included for a complete reproducible input set; verify their redistribution license in `src/models/official_model/LICENSE`.
