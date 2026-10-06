# Integration Layer

The integration layer is optional. The core repository does not redistribute proprietary board SDKs, model weights, cloud credentials, or generated firmware.

## JTAG/UART relay

Set these environment variables before starting `pc_relay/start_full_chain.ps1`:

```powershell
$env:BITNET_XSDB = "C:\Xilinx\Vitis\2020.1\bin\xsdb.bat"
$env:BITNET_INFER_SCRIPT = "C:\path\to\fast-bitnet\pc\jtag\scripts\run_bitnet_inference_smoke.tcl"
$env:BITNET_MODEL = "C:\licensed\model\directory"
$env:BITNET_SERIAL_PORT = "COM3"
$env:BITNET_CLOUD_DIR = "C:\path\to\local\cloud\server"
```

The model directory and cloud server are external inputs. The relay validates their presence and stops with an English error instead of silently using a machine-specific path.

## Protocol

The JSON/TCP contract is documented in `socket_json_bridge/PROTOCOL.md`. Use `socket_json_bridge/traffic_message.schema.json` to validate messages before sending them to a board or service.
