#!/usr/bin/env python3
import argparse
import json
import struct
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect safetensors header.")
    parser.add_argument("model", help="Path to model.safetensors")
    parser.add_argument("--filter", default="", help="Substring filter for tensor names")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    path = Path(args.model)
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

    tensors = [(k, v) for k, v in header.items() if k != "__metadata__"]
    if args.filter:
        tensors = [(k, v) for k, v in tensors if args.filter in k]

    print(f"header_bytes={header_len}")
    print(f"tensor_count={len(tensors)}")
    print(f"metadata={header.get('__metadata__')}")
    print(f"dtype_counts={dict(Counter(v.get('dtype') for _, v in tensors))}")

    for name, info in tensors[: args.limit]:
        print(f"{name}\t{info.get('dtype')}\t{info.get('shape')}\t{info.get('data_offsets')}")


if __name__ == "__main__":
    main()
