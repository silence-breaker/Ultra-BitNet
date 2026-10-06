#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np


CONTROL_MAGIC = 0x424C4331

TENSORS = [
    ("q_proj", "self_attn.q_proj", 0x000000),
    ("k_proj", "self_attn.k_proj", 0x200000),
    ("v_proj", "self_attn.v_proj", 0x300000),
    ("o_proj", "self_attn.o_proj", 0x400000),
    ("gate_proj", "mlp.gate_proj", 0x600000),
    ("up_proj", "mlp.up_proj", 0xB00000),
    ("down_proj", "mlp.down_proj", 0x1000000),
]


def read_header(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def copy_range(src: Path, dst: Path, start: int, size: int, chunk_size: int = 8 * 1024 * 1024):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(start)
        remaining = size
        while remaining:
            data = fin.read(min(chunk_size, remaining))
            if not data:
                raise RuntimeError("Unexpected EOF while exporting tensor")
            fout.write(data)
            remaining -= len(data)


def repack_hf_rows_to_fpga(data: bytes, packed_rows: int, in_features: int) -> bytes:
    packed = np.frombuffer(data, dtype=np.uint8).reshape(packed_rows, in_features)
    output = np.zeros((packed_rows, in_features), dtype=np.uint8)
    group_index = np.arange(packed_rows, dtype=np.int64)

    for lane in range(4):
        output_row = (group_index * 4) + lane
        source_row = output_row % packed_rows
        source_lane = output_row // packed_rows
        codes = (packed[source_row] >> (source_lane[:, None] * 2)) & 0x3
        output |= codes.astype(np.uint8) << (lane * 2)
    return output.tobytes()


def export_tensor(model_path: Path, header_len: int, header: dict, tensor_name: str, dst: Path):
    if tensor_name not in header:
        raise SystemExit(f"Tensor not found: {tensor_name}")

    info = header[tensor_name]
    dtype = info.get("dtype")
    shape = info.get("shape")
    if dtype != "U8":
        raise SystemExit(f"Expected U8 tensor, got {dtype}: {tensor_name}")
    if (not isinstance(shape, list)) or len(shape) != 2:
        raise SystemExit(f"Expected 2D tensor, got shape={shape}: {tensor_name}")

    byte_len = int(info["data_offsets"][1]) - int(info["data_offsets"][0])
    expected = int(shape[0]) * int(shape[1])
    if byte_len != expected:
        raise SystemExit(f"Unexpected byte length for {tensor_name}: got {byte_len}, expected {expected}")

    abs_start = 8 + header_len + int(info["data_offsets"][0])
    with model_path.open("rb") as src:
        src.seek(abs_start)
        packed = src.read(byte_len)
    if len(packed) != byte_len:
        raise RuntimeError(f"Unexpected EOF while exporting {tensor_name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(repack_hf_rows_to_fpga(
        packed, int(shape[0]), int(shape[1])))

    return {
        "tensor": tensor_name,
        "dtype": dtype,
        "shape": shape,
        "out_features": int(shape[0]) * 4,
        "in_features": int(shape[1]),
        "weight_bytes": byte_len,
        "file": str(dst),
    }


def main():
    parser = argparse.ArgumentParser(description="Export BitNet layer U8 weights for AXU3EGB layer-chain test.")
    parser.add_argument("--model", required=True, help="Path to model.safetensors")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--activation-seed", type=lambda x: int(x, 0), default=23)
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=1)
    args = parser.parse_args()

    if args.first_layer < 0 or args.layers <= 0:
        raise SystemExit("--first-layer must be >= 0 and --layers must be > 0")

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    header_len, header = read_header(model_path)

    exported = []
    for layer in range(args.first_layer, args.first_layer + args.layers):
        for short_name, suffix, offset in TENSORS:
            tensor_name = f"model.layers.{layer}.{suffix}.weight"
            dst = out_dir / f"layer{layer}_{short_name}.fpga.bin"
            meta = export_tensor(model_path, header_len, header, tensor_name, dst)
            meta["layer"] = layer
            meta["slot"] = layer - args.first_layer
            meta["short_name"] = short_name
            meta["slot_offset"] = offset
            exported.append(meta)

    control_words = [
        CONTROL_MAGIC,
        args.first_layer,
        args.layers,
        args.activation_seed,
        args.flags,
        0,
        0,
        0,
    ]
    control_bin = out_dir / "layer_chain.ctrl.bin"
    control_bin.parent.mkdir(parents=True, exist_ok=True)
    control_bin.write_bytes(struct.pack("<8I", *control_words))

    meta = {
        "source_model": str(model_path),
        "first_layer": args.first_layer,
        "layers": args.layers,
        "activation_seed": args.activation_seed,
        "flags": args.flags,
        "control_magic": f"0x{CONTROL_MAGIC:08X}",
        "control_bin": str(control_bin),
        "weight_base": "0x56000000",
        "layer_stride": "0x02000000",
        "tensors": exported,
        "fpga_layout": "packed[out_features/4][in_features], 2-bit codes: 0=-1, 1=0, 2=+1",
    }

    meta_json = out_dir / "layer_chain.meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
