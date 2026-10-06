#!/usr/bin/env python3
import argparse
import json
import shutil
import struct
from pathlib import Path

from export_layer_bundle import repack_hf_rows_to_fpga


CONTROL_MAGIC = 0x42574131
DEFAULT_TENSOR = "model.layers.0.self_attn.q_proj.weight"


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


def main():
    parser = argparse.ArgumentParser(
        description="Export a BitNet deploy U8 tensor in FPGA GEMV layout plus a control block."
    )
    parser.add_argument("--model", required=True, help="Path to model.safetensors")
    parser.add_argument("--tensor", default=DEFAULT_TENSOR, help="U8 tensor name")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="layer0_q_proj", help="Output filename prefix")
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=1)
    parser.add_argument("--tensor-kind", type=lambda x: int(x, 0), default=1)
    parser.add_argument("--activation-seed", type=lambda x: int(x, 0), default=13)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    header_len, header = read_header(model_path)

    if args.tensor not in header:
        raise SystemExit(f"Tensor not found: {args.tensor}")

    info = header[args.tensor]
    dtype = info.get("dtype")
    shape = info.get("shape")
    if dtype != "U8":
        raise SystemExit(f"Expected U8 tensor, got {dtype}: {args.tensor}")
    if (not isinstance(shape, list)) or len(shape) != 2:
        raise SystemExit(f"Expected 2D U8 tensor, got shape={shape}: {args.tensor}")

    packed_out, in_features = int(shape[0]), int(shape[1])
    out_features = packed_out * 4
    expected_bytes = packed_out * in_features
    start, end = info["data_offsets"]
    byte_len = int(end) - int(start)
    if byte_len != expected_bytes:
        raise SystemExit(
            f"Unexpected tensor byte length: got {byte_len}, expected {expected_bytes}"
        )

    tensor_bin = out_dir / f"{args.prefix}.fpga.bin"
    control_bin = out_dir / f"{args.prefix}.ctrl.bin"
    meta_json = out_dir / f"{args.prefix}.meta.json"

    abs_start = 8 + header_len + int(start)
    with model_path.open("rb") as src:
        src.seek(abs_start)
        packed = src.read(byte_len)
    if len(packed) != byte_len:
        raise RuntimeError(f"Unexpected EOF while exporting {args.tensor}")
    tensor_bin.parent.mkdir(parents=True, exist_ok=True)
    tensor_bin.write_bytes(repack_hf_rows_to_fpga(
        packed, packed_out, in_features))

    control_words = [
        CONTROL_MAGIC,
        out_features,
        in_features,
        args.flags,
        byte_len,
        args.tensor_kind,
        args.activation_seed,
        0,
    ]
    control_bin.write_bytes(struct.pack("<8I", *control_words))

    meta = {
        "source_model": str(model_path),
        "tensor": args.tensor,
        "dtype": dtype,
        "shape": shape,
        "out_features": out_features,
        "in_features": in_features,
        "weight_bytes": byte_len,
        "flags": args.flags,
        "tensor_kind": args.tensor_kind,
        "activation_seed": args.activation_seed,
        "control_magic": f"0x{CONTROL_MAGIC:08X}",
        "tensor_bin": str(tensor_bin),
        "control_bin": str(control_bin),
        "fpga_layout": "packed[out_features/4][in_features], 2-bit codes: 0=-1, 1=0, 2=+1",
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Keep a stable generic name for board scripts and a descriptive name for humans.
    default_tensor = out_dir / "real_tensor.fpga.bin"
    default_control = out_dir / "real_tensor.ctrl.bin"
    if tensor_bin.resolve() != default_tensor.resolve():
        shutil.copyfile(tensor_bin, default_tensor)
    if control_bin.resolve() != default_control.resolve():
        shutil.copyfile(control_bin, default_control)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
