# Socket JSON Bridge

This directory contains the host-side bridge used to move newline-delimited JSON
messages between a TCP client and a board-side serial or network endpoint.

## Layout

- `alinx_bridge/`: bridge implementation and transport adapters.
- `traffic_message.schema.json`: JSON Schema for the traffic inference message.
- `PROTOCOL.md`: wire format and startup reference.

The bridge does not include model weights, credentials, board firmware, or vendor
tools. Configure the endpoint with command-line options or environment variables
instead of editing source files.

## Quick start

```bash
python -m alinx_bridge --help
python -m alinx_bridge --host 127.0.0.1 --port 9000 --device /dev/ttyUSB0
```

On Windows, use a COM device such as `COM3`. On Linux, use the device path
reported by `udevadm` or `ls /dev/ttyUSB*`. The exact options are listed by the
module's `--help` output.

## Validation

Validate a message against the schema before sending it:

```bash
python -m jsonschema -i message.json traffic_message.schema.json
```

Use `integration/pc_relay/tools/test_client.py` for a minimal end-to-end client.
