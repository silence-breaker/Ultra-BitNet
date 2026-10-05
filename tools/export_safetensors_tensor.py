#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path


def read_header(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def main():
    parser = argparse.ArgumentParser(
        description="Export one tensor from a safetensors file without importing torch."
    )
    parser.add_argument("--model", required=True, help="Path to model.safetensors")
    parser.add_argument("--tensor", required=True, help="Tensor name to export")
    parser.add_argument("--out", required=True, help="Output raw binary path")
    parser.add_argument("--meta", default="", help="Optional output metadata json")
    args = parser.parse_args()

    model_path = Path(args.model)
    out_path = Path(args.out)
    header_len, header = read_header(model_path)

    if args.tensor not in header:
        names = [name for name in header.keys() if name != "__metadata__"]
        raise SystemExit(f"Tensor not found: {args.tensor}\nAvailable tensors: {len(names)}")

    info = header[args.tensor]
    start, end = info["data_offsets"]
    abs_start = 8 + header_len + start
    byte_len = end - start

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("rb") as f:
        f.seek(abs_start)
        data = f.read(byte_len)
    out_path.write_bytes(data)

    meta = {
        "tensor": args.tensor,
        "dtype": info["dtype"],
        "shape": info["shape"],
        "data_offsets": info["data_offsets"],
        "raw_bytes": byte_len,
        "source": str(model_path),
        "output": str(out_path),
    }

    if args.meta:
        meta_path = Path(args.meta)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
