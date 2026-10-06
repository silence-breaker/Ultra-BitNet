# JSON Bridge Protocol

## Transport

Messages are UTF-8 JSON objects separated by a single line-feed (`\\n`). A
receiver must process one complete JSON object per line and must not assume a
fixed packet size. TCP framing and serial framing use the same line-delimited
format.

## Message schema

`traffic_message.schema.json` is the authoritative schema. Producers should
validate outgoing messages and consumers should reject malformed or unsupported
messages with an error response. Unknown fields may be ignored for forward
compatibility unless a deployment explicitly enables strict validation.

## Endpoint configuration

Host, port, and serial device are deployment settings. Keep them outside source
control and provide them through command-line options or environment variables.
Do not rely on a fixed private-network address or a fixed device name; these
values vary between boards and host operating systems.

## Example exchange

Request:

```json
{"type":"traffic_frame","timestamp_ms":1710000000000,"objects":[]}
```

Response:

```json
{"type":"inference_result","status":"ok","objects":[]}
```

Each JSON object is followed by `\\n`. Applications should set a read timeout,
log rejected messages, and close the connection after an unrecoverable framing
or schema error.

## Troubleshooting

- Confirm the selected TCP port is reachable and not already in use.
- Confirm the serial device exists and the account has access to it.
- Check that every message is valid UTF-8 and ends with a line-feed.
- Run the bridge with `--help` and the integration test client to inspect the
  active endpoint configuration.
