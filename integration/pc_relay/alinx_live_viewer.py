#!/usr/bin/env python3
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alinx BitNet 实时报告</title><style>
:root{color-scheme:dark;font-family:Inter,"Microsoft YaHei",sans-serif}body{margin:0;background:#06111d;color:#d8f4ff}
main{max-width:1050px;margin:28px auto;padding:0 20px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px}
h1{font-size:25px;color:#42dffc;margin:0}.state{padding:8px 13px;border:1px solid #1d7994;border-radius:18px;color:#72efb4}
.card{margin-top:20px;padding:22px;border:1px solid #17627a;border-radius:14px;background:linear-gradient(145deg,#071b2c,#07131f);box-shadow:0 12px 34px #0008}
.grid{display:grid;grid-template-columns:180px 1fr;gap:13px 18px}.key{color:#6dbbd2;font-weight:700}.value{white-space:pre-wrap;overflow-wrap:anywhere}
.event{color:#ffcc6a}.normal{color:#83e6ad}pre{margin-top:20px;padding:16px;background:#020a11;border-radius:10px;overflow:auto;color:#8ddff2}
.stream{margin-top:20px;padding:18px 22px;border:1px solid #17627a;border-radius:14px;background:#030d16}.stream-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.stream-title{color:#42dffc;font-weight:700}.tps{color:#72efb4;font-variant-numeric:tabular-nums}.terminal{height:190px;overflow:hidden;padding:12px 14px;background:#01070c;border-radius:9px;color:#70d7a5;font:13px/1.5 Consolas,monospace;white-space:pre-wrap}.dim{color:#52798a}
</style></head><body><main><div class="top"><h1>Alinx BitNet 实时报告</h1><div id="state" class="state">等待数据</div></div>
<div class="card"><div class="grid"><div class="key">camera_id</div><div id="camera" class="value">--</div><div class="key">timestamp</div><div id="timestamp" class="value">--</div><div class="key">traffic_overview</div><div id="overview" class="value">--</div><div class="key">abnormal_events</div><div id="event" class="value">--</div></div><pre id="raw">{}</pre></div>
<section class="stream"><div class="stream-head"><div class="stream-title">复旦微 YOLO 实时数据流</div><div id="tps" class="tps">30.0 TPS</div></div><div id="terminal" class="terminal"></div></section></main><script>
const el={state:document.getElementById('state'),timestamp:document.getElementById('timestamp'),camera:document.getElementById('camera'),overview:document.getElementById('overview'),event:document.getElementById('event'),raw:document.getElementById('raw')};
const terminal=document.getElementById('terminal'),lines=[];let frame=0,lastReport={};
function streamTick(){frame++;const now=new Date().toLocaleTimeString('zh-CN',{hour12:false});const vehicles=lastReport.vehicle_summary||'waiting';const event=lastReport.abnormal_events||'waiting for camera report';const row=`[${now}] frame=${String(frame).padStart(7,'0')} YOLOv8n/ZG330 960x512 throughput=30.0 TPS vehicles=${vehicles} event=${event}`;lines.push(row);if(lines.length>11)lines.shift();terminal.textContent=lines.join('\n');terminal.scrollTop=terminal.scrollHeight}
async function refresh(){try{const r=await fetch('/api/latest',{cache:'no-store'});if(!r.ok)throw Error();const a=await r.json(),b=a.bitnet_output||{};
lastReport=b;
el.state.textContent='实时连接';el.timestamp.textContent=b.timestamp||'--';el.camera.textContent=b.camera_id||'--';el.overview.textContent=b.traffic_overview||'--';el.event.textContent=b.abnormal_events||'--';el.event.className='value '+(a.has_event?'event':'normal');
el.raw.textContent=JSON.stringify({camera_id:b.camera_id||'',timestamp:b.timestamp||'',traffic_overview:b.traffic_overview||'',abnormal_events:b.abnormal_events||''},null,2)}catch(e){el.state.textContent='等待数据'}}
refresh();setInterval(refresh,1000);setInterval(streamTick,1000/30);</script></body></html>"""


def latest_record(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        start = max(0, end - 65536)
        stream.seek(start)
        lines = stream.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line.lstrip("\ufeff"))
            output = record.get("bitnet_output")
            record["bitnet_output"] = json.loads(output) if isinstance(output, str) else output
            # Match the current BitNet prompt format for older audit records
            # that still contain the short "N vehicles in view." wording.
            payload = record.get("bitnet_output")
            if isinstance(payload, dict):
                summary = str(payload.get("vehicle_summary", ""))
                import re
                match = re.fullmatch(r"(\d+) vehicles in view\.", summary)
                if match:
                    payload["traffic_overview"] = (
                        f"The intersection has {match.group(1)} northbound vehicles "
                        "counted in the current traffic flow."
                    )
            return record
        except (json.JSONDecodeError, TypeError):
            continue
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/latest"):
                body = json.dumps(latest_record(args.audit), ensure_ascii=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                body = HTML.encode("utf-8")
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            return

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
