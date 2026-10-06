#!/usr/bin/env python3
"""Rebuild the Alinx bare-metal BitNet matrix shards from an official checkpoint.

The Hugging Face checkpoint packs four output rows in four large row blocks.
The deployed PL kernel consumes four consecutive output rows per packed byte.
This tool performs that deterministic permutation and verifies every emitted byte.
Non-matrix runtime assets are copied from a known deployment because their binary
formats are firmware-specific (token trie, BF16 embeddings, LM-head shards, AUX).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open


MATRICES = {
    "Q.BIN": "self_attn.q_proj.weight",
    "K.BIN": "self_attn.k_proj.weight",
    "V.BIN": "self_attn.v_proj.weight",
    "O.BIN": "self_attn.o_proj.weight",
    "GATE.BIN": "mlp.gate_proj.weight",
    "UP.BIN": "mlp.up_proj.weight",
    "DOWN.BIN": "mlp.down_proj.weight",
}


def repack_hf_to_pl(packed: np.ndarray) -> np.ndarray:
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError(f"expected a 2-D uint8 packed matrix, got {packed.dtype} {packed.shape}")
    packed_rows, in_features = packed.shape
    out_features = packed_rows * 4
    codes = np.empty((out_features, in_features), dtype=np.uint8)
    for lane in range(4):
        codes[lane * packed_rows : (lane + 1) * packed_rows] = (packed >> (2 * lane)) & 3
    return (
        codes[0::4]
        | (codes[1::4] << 2)
        | (codes[2::4] << 4)
        | (codes[3::4] << 6)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    assets = args.runtime_assets.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # These formats are consumed directly by the existing firmware and are not
    # standard Hugging Face artifacts. Preserve them byte-for-byte.
    for name in ("AUX", "EMB", "LMH", "TOK"):
        src = assets / name
        dst = output / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for name in ("MODEL_AUX.BIN", "MODEL_ROPE.BIN", "MANIFEST.TXT"):
        src = assets / name
        if src.exists():
            shutil.copy2(src, output / name)

    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "layout": "four-consecutive-output-channels-per-byte",
        "matrices": [],
    }
    with safe_open(str(checkpoint), framework="numpy") as tensors:
        for layer in range(30):
            layer_dir = output / f"L{layer:02d}"
            layer_dir.mkdir(exist_ok=True)
            for filename, suffix in MATRICES.items():
                key = f"model.layers.{layer}.{suffix}"
                official = tensors.get_tensor(key)
                rebuilt = repack_hf_to_pl(official)
                target = layer_dir / filename
                target.write_bytes(rebuilt.tobytes(order="C"))
                entry = {
                    "layer": layer,
                    "file": filename,
                    "tensor": key,
                    "shape": list(rebuilt.shape),
                    "bytes": target.stat().st_size,
                }
                if args.compare:
                    previous = args.compare.resolve() / f"L{layer:02d}" / filename
                    entry["matches_previous"] = previous.exists() and target.read_bytes() == previous.read_bytes()
                report["matrices"].append(entry)

    report["all_match_previous"] = bool(args.compare) and all(
        item.get("matches_previous", False) for item in report["matrices"]
    )
    (output / "REBUILD_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "matrix_count": len(report["matrices"]),
        "all_match_previous": report["all_match_previous"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
