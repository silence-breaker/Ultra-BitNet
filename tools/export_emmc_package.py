#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import struct
from pathlib import Path

from export_layer_bundle import TENSORS, copy_range, export_tensor, read_header
from export_tokenizer_trie import (
    DEFAULT_BOS_ID,
    DEFAULT_EOS_ID,
    build_trie,
    flatten,
    write_asset as write_tokenizer_asset,
    write_decode_asset,
    write_pretokenizer_asset,
)


CONTROL_MAGIC = 0x42454D31

EMMC_NAMES = {
    "q_proj": "Q.BIN",
    "k_proj": "K.BIN",
    "v_proj": "V.BIN",
    "o_proj": "O.BIN",
    "gate_proj": "GATE.BIN",
    "up_proj": "UP.BIN",
    "down_proj": "DOWN.BIN",
}

AUX_MAGIC = 0x42584131
AUX_VERSION = 1
AUX_HEADER_WORDS = 16
EMBED_ROWS_PER_SHARD = 8192
HIDDEN_SIZE = 2560
KV_SIZE = 640
FFN_SIZE = 6912
MODEL_LAYERS = 30
VOCAB_SIZE = 128256
ROPE_MAGIC = 0x42505231
ROPE_VERSION = 1
ROPE_HEADER_WORDS = 12
ROPE_Q15_SCALE = 32767
DEFAULT_MAX_POSITION_EMBEDDINGS = 4096
DEFAULT_ROPE_THETA = 500000.0

SCALE_TENSORS = [
    ("q_proj", "self_attn.q_proj"),
    ("k_proj", "self_attn.k_proj"),
    ("v_proj", "self_attn.v_proj"),
    ("o_proj", "self_attn.o_proj"),
    ("gate_proj", "mlp.gate_proj"),
    ("up_proj", "mlp.up_proj"),
    ("down_proj", "mlp.down_proj"),
]


def clamp_i16(value: int) -> int:
    return max(-32768, min(32767, value))


def load_model_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def tensor_range(header_len: int, header: dict, tensor_name: str, dtype: str, shape: list[int]):
    if tensor_name not in header:
        raise SystemExit(f"Tensor not found: {tensor_name}")

    info = header[tensor_name]
    if info.get("dtype") != dtype:
        raise SystemExit(f"Expected {dtype} tensor, got {info.get('dtype')}: {tensor_name}")
    if info.get("shape") != shape:
        raise SystemExit(f"Expected shape={shape}, got {info.get('shape')}: {tensor_name}")

    start, end = [int(x) for x in info["data_offsets"]]
    return 8 + header_len + start, end - start


def read_tensor_bytes(model_path: Path, header_len: int, header: dict,
                      tensor_name: str, dtype: str, shape: list[int]) -> bytes:
    abs_start, size = tensor_range(header_len, header, tensor_name, dtype, shape)
    with model_path.open("rb") as f:
        f.seek(abs_start)
        data = f.read(size)
    if len(data) != size:
        raise RuntimeError(f"Unexpected EOF while reading {tensor_name}")
    return data


def export_aux_assets(model_path: Path, header_len: int, header: dict, bitnet_dir: Path) -> dict:
    aux_dir = bitnet_dir / "AUX"
    aux_path = aux_dir / "AUX.BIN"
    safe_aux_path = bitnet_dir / "MODEL_AUX.BIN"
    aux_dir.mkdir(parents=True, exist_ok=True)

    payload = bytearray(b"\x00" * (AUX_HEADER_WORDS * 4))

    def append_blob(data: bytes) -> int:
        offset = len(payload)
        payload.extend(data)
        if len(payload) & 0x3:
            payload.extend(b"\x00" * (4 - (len(payload) & 0x3)))
        return offset

    final_norm_offset = append_blob(read_tensor_bytes(
        model_path, header_len, header, "model.norm.weight", "BF16", [HIDDEN_SIZE]))

    input_norm_offset = len(payload)
    for layer in range(MODEL_LAYERS):
        payload.extend(read_tensor_bytes(
            model_path, header_len, header,
            f"model.layers.{layer}.input_layernorm.weight", "BF16", [HIDDEN_SIZE]))

    post_norm_offset = len(payload)
    for layer in range(MODEL_LAYERS):
        payload.extend(read_tensor_bytes(
            model_path, header_len, header,
            f"model.layers.{layer}.post_attention_layernorm.weight", "BF16", [HIDDEN_SIZE]))

    attn_sub_norm_offset = len(payload)
    for layer in range(MODEL_LAYERS):
        payload.extend(read_tensor_bytes(
            model_path, header_len, header,
            f"model.layers.{layer}.self_attn.attn_sub_norm.weight", "BF16", [HIDDEN_SIZE]))

    ffn_sub_norm_offset = len(payload)
    for layer in range(MODEL_LAYERS):
        payload.extend(read_tensor_bytes(
            model_path, header_len, header,
            f"model.layers.{layer}.mlp.ffn_sub_norm.weight", "BF16", [FFN_SIZE]))

    scale_offset = len(payload)
    for layer in range(MODEL_LAYERS):
        for _, suffix in SCALE_TENSORS:
            payload.extend(read_tensor_bytes(
                model_path, header_len, header,
                f"model.layers.{layer}.{suffix}.weight_scale", "BF16", [1]))
    if len(payload) & 0x3:
        payload.extend(b"\x00" * (4 - (len(payload) & 0x3)))

    header_words = [
        AUX_MAGIC,
        AUX_VERSION,
        len(payload),
        HIDDEN_SIZE,
        FFN_SIZE,
        MODEL_LAYERS,
        len(SCALE_TENSORS),
        final_norm_offset,
        input_norm_offset,
        post_norm_offset,
        attn_sub_norm_offset,
        ffn_sub_norm_offset,
        scale_offset,
        0,
        0,
        0,
    ]
    payload[:AUX_HEADER_WORDS * 4] = struct.pack("<16I", *header_words)
    aux_path.write_bytes(payload)
    safe_aux_path.write_bytes(payload)

    meta = {
        "path": str(aux_path),
        "safe_host_path": str(safe_aux_path),
        "emmc_path": "0:/BITNET/AUX/AUX.BIN",
        "magic": f"0x{AUX_MAGIC:08X}",
        "version": AUX_VERSION,
        "total_bytes": len(payload),
        "offsets": {
            "final_norm": final_norm_offset,
            "input_layernorm": input_norm_offset,
            "post_attention_layernorm": post_norm_offset,
            "attn_sub_norm": attn_sub_norm_offset,
            "ffn_sub_norm": ffn_sub_norm_offset,
            "weight_scales": scale_offset,
        },
        "scale_order": [name for name, _ in SCALE_TENSORS],
    }
    (aux_dir / "AUX.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def export_embedding_shards(model_path: Path, header_len: int, header: dict, bitnet_dir: Path,
                            rows_per_shard: int = EMBED_ROWS_PER_SHARD) -> dict:
    embed_dir = bitnet_dir / "EMB"
    embed_dir.mkdir(parents=True, exist_ok=True)
    abs_start, total_bytes = tensor_range(
        header_len, header, "model.embed_tokens.weight", "BF16", [VOCAB_SIZE, HIDDEN_SIZE])
    row_bytes = HIDDEN_SIZE * 2
    if total_bytes != VOCAB_SIZE * row_bytes:
        raise SystemExit("Unexpected embedding byte size")

    shards = []
    for shard_index, row_start in enumerate(range(0, VOCAB_SIZE, rows_per_shard)):
        rows = min(rows_per_shard, VOCAB_SIZE - row_start)
        shard_bytes = rows * row_bytes
        shard_name = f"E{shard_index:02d}.BIN"
        dst = embed_dir / shard_name
        copy_range(model_path, dst, abs_start + (row_start * row_bytes), shard_bytes)
        shards.append({
            "file": str(dst),
            "emmc_path": f"0:/BITNET/EMB/{shard_name}",
            "shard": shard_index,
            "row_start": row_start,
            "rows": rows,
            "bytes": shard_bytes,
        })

    meta = {
        "tensor": "model.embed_tokens.weight",
        "dtype": "BF16",
        "shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "tie_word_embeddings": True,
        "row_bytes": row_bytes,
        "rows_per_shard": rows_per_shard,
        "shard_count": len(shards),
        "shards": shards,
    }
    (embed_dir / "EMB.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def bf16_embedding_bytes_to_rowwise_i8(raw: bytes, rows: int) -> tuple[bytes, bytes]:
    import numpy as np

    bf16 = np.frombuffer(raw, dtype="<u2").reshape(rows, HIDDEN_SIZE)
    fp32_bits = bf16.astype(np.uint32) << np.uint32(16)
    fp32 = fp32_bits.view(np.float32)
    row_scales = np.max(np.abs(fp32), axis=1).astype(np.float32) / np.float32(127.0)
    row_scales[row_scales <= np.float32(1.0e-20)] = np.float32(1.0)
    scaled_f = fp32 / row_scales[:, None]
    rounded = np.where(scaled_f >= 0.0,
                       scaled_f + np.float32(0.5),
                       scaled_f - np.float32(0.5)).astype(np.int32)
    clipped = np.clip(rounded, -128, 127).astype(np.int8)
    return clipped.tobytes(), row_scales.astype("<f4", copy=False).tobytes()


def export_lm_head_i8_shards_from_bf16_embedding(bitnet_dir: Path,
                                                 source_embed_dir: Path | None = None,
                                                 rows_per_shard: int = EMBED_ROWS_PER_SHARD) -> dict:
    embed_dir = source_embed_dir if source_embed_dir is not None else (bitnet_dir / "EMB")
    lm_head_dir = bitnet_dir / "LMH"
    lm_head_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    for shard_index, row_start in enumerate(range(0, VOCAB_SIZE, rows_per_shard)):
        rows = min(rows_per_shard, VOCAB_SIZE - row_start)
        src = embed_dir / f"E{shard_index:02d}.BIN"
        dst = lm_head_dir / f"H{shard_index:02d}.BIN"
        expected_src_bytes = rows * HIDDEN_SIZE * 2

        if not src.exists():
            raise SystemExit(f"Missing BF16 embedding shard for lm_head i8 export: {src}")
        raw = src.read_bytes()
        if len(raw) != expected_src_bytes:
            raise SystemExit(
                f"Unexpected BF16 embedding shard size: {src} got={len(raw)} expected={expected_src_bytes}")
        weights_i8, row_scales = bf16_embedding_bytes_to_rowwise_i8(raw, rows)
        dst.write_bytes(weights_i8 + row_scales)
        shards.append({
            "file": str(dst),
            "emmc_path": f"0:/BITNET/LMH/H{shard_index:02d}.BIN",
            "shard": shard_index,
            "row_start": row_start,
            "rows": rows,
            "bytes": rows * (HIDDEN_SIZE + 4),
            "weight_bytes": rows * HIDDEN_SIZE,
            "scale_bytes": rows * 4,
        })

    meta = {
        "tensor": "model.embed_tokens.weight",
        "dtype": "I8_ROW_SCALE_F32",
        "source_dtype": "BF16",
        "layout": "contiguous_i8_weights_then_f32_row_scales",
        "row_scale_dtype": "F32",
        "shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "tie_word_embeddings": True,
        "row_bytes": HIDDEN_SIZE + 4,
        "rows_per_shard": rows_per_shard,
        "shard_count": len(shards),
        "shards": shards,
    }
    (lm_head_dir / "LMH.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def export_tokenizer_assets(tokenizer_json: Path, bitnet_dir: Path,
                            bos_id: int = DEFAULT_BOS_ID,
                            eos_id: int = DEFAULT_EOS_ID) -> dict:
    tok_dir = bitnet_dir / "TOK"
    trie_path = tok_dir / "TRIE.BIN"
    decode_path = tok_dir / "DECODE.BIN"
    pretok_path = tok_dir / "PRET.BIN"

    if not tokenizer_json.exists():
        raise SystemExit(f"Missing tokenizer.json: {tokenizer_json}")

    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise SystemExit(f"Unsupported tokenizer model type: {model.get('type')}")
    if not model.get("ignore_merges", False):
        raise SystemExit("This exporter expects tokenizer.json model.ignore_merges=true")

    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise SystemExit("tokenizer.json does not contain a model.vocab object")

    nodes, max_token_bytes, skipped = build_trie(vocab)
    flat_nodes, flat_edges = flatten(nodes)
    tok_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = write_tokenizer_asset(trie_path, flat_nodes, flat_edges,
                                        max_token_bytes, bos_id, eos_id)
    decode_meta = write_decode_asset(decode_path, vocab, data.get("added_tokens", []), VOCAB_SIZE)
    pretok_meta = write_pretokenizer_asset(pretok_path)

    meta = {
        "source": str(tokenizer_json),
        "path": str(trie_path),
        "emmc_path": "0:/BITNET/TOK/TRIE.BIN",
        "decode_path": str(decode_path),
        "decode_emmc_path": "0:/BITNET/TOK/DECODE.BIN",
        "pretokenizer_path": str(pretok_path),
        "pretokenizer_emmc_path": "0:/BITNET/TOK/PRET.BIN",
        "vocab_tokens": len(vocab),
        "nodes": len(flat_nodes),
        "edges": len(flat_edges),
        "max_token_bytes": max_token_bytes,
        "skipped_tokens": len(skipped),
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "total_bytes": total_bytes,
        "decode": decode_meta,
        "pretokenizer": pretok_meta,
    }
    (tok_dir / "TRIE.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    return meta


def export_rope_assets(bitnet_dir: Path, max_positions: int, head_dim: int,
                       theta: float) -> dict:
    aux_dir = bitnet_dir / "AUX"
    rope_path = aux_dir / "ROPE.BIN"
    safe_rope_path = bitnet_dir / "MODEL_ROPE.BIN"
    pair_count = head_dim // 2
    table_offset = ROPE_HEADER_WORDS * 4
    payload = bytearray(b"\x00" * table_offset)

    if (max_positions <= 0) or (head_dim <= 0) or ((head_dim & 1) != 0):
        raise SystemExit("Invalid RoPE shape")

    aux_dir.mkdir(parents=True, exist_ok=True)
    for pos in range(max_positions):
        for pair in range(pair_count):
            inv_freq = 1.0 / (theta ** ((2.0 * pair) / float(head_dim)))
            angle = float(pos) * inv_freq
            cos_q15 = clamp_i16(int(round(math.cos(angle) * ROPE_Q15_SCALE)))
            sin_q15 = clamp_i16(int(round(math.sin(angle) * ROPE_Q15_SCALE)))
            payload.extend(struct.pack("<hh", cos_q15, sin_q15))

    header_words = [
        ROPE_MAGIC,
        ROPE_VERSION,
        len(payload),
        max_positions,
        head_dim,
        pair_count,
        table_offset,
        ROPE_Q15_SCALE,
        int(round(theta * 1000.0)),
        0,
        0,
        0,
    ]
    payload[:table_offset] = struct.pack("<12I", *header_words)
    rope_path.write_bytes(payload)
    safe_rope_path.write_bytes(payload)

    meta = {
        "path": str(rope_path),
        "safe_host_path": str(safe_rope_path),
        "emmc_path": "0:/BITNET/AUX/ROPE.BIN",
        "magic": f"0x{ROPE_MAGIC:08X}",
        "version": ROPE_VERSION,
        "max_positions": max_positions,
        "head_dim": head_dim,
        "pair_count": pair_count,
        "theta": theta,
        "q15_scale": ROPE_Q15_SCALE,
        "total_bytes": len(payload),
    }
    (aux_dir / "ROPE.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def write_control(out_dir: Path, first_layer: int, layers: int, activation_seed: int, flags: int) -> Path:
    control_words = [
        CONTROL_MAGIC,
        first_layer,
        layers,
        activation_seed,
        flags,
        0,
        0,
        0,
    ]
    control_bin = out_dir / "emmc_layer_chain.ctrl.bin"
    control_bin.parent.mkdir(parents=True, exist_ok=True)
    control_bin.write_bytes(struct.pack("<8I", *control_words))
    return control_bin


def write_manifest(bitnet_dir: Path, meta: dict) -> Path:
    lines = [
        "BITNET_AXU3EGB_EMMC_PACKAGE",
        f"control_magic=0x{CONTROL_MAGIC:08X}",
        f"first_layer={meta['first_layer']}",
        f"layers={meta['layers']}",
        f"hidden_size={meta['hidden_size']}",
        f"kv_size={meta['kv_size']}",
        f"ffn_size={meta['ffn_size']}",
        "layout=packed[out_features/4][in_features]",
        "codes=0:-1,1:0,2:+1",
        f"aux_path={meta['aux']['emmc_path']}",
        f"rope_path={meta['rope']['emmc_path']}",
        f"rope_max_positions={meta['rope']['max_positions']}",
        f"rope_theta={meta['rope']['theta']}",
        f"embedding_shards={meta['embedding']['shard_count']}",
        f"embedding_rows_per_shard={meta['embedding']['rows_per_shard']}",
        f"lm_head_i8_shards={meta['lm_head_i8']['shard_count']}",
        f"lm_head_i8_rows_per_shard={meta['lm_head_i8']['rows_per_shard']}",
        f"tokenizer_path={meta['tokenizer']['emmc_path']}",
        "lm_head=tied_to_embedding_rowwise_i8_f32_scale_resident",
    ]
    manifest = bitnet_dir / "MANIFEST.TXT"
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Export AXU3EGB eMMC package for BitNet layer-chain loading.")
    parser.add_argument("--model", required=True, help="Path to model.safetensors")
    parser.add_argument("--out-dir", required=True, help="Output directory containing BITNET package")
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--activation-seed", type=lambda x: int(x, 0), default=23)
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=1)
    parser.add_argument("--config-json", help="Path to config.json; defaults to the model directory")
    parser.add_argument("--tokenizer-json", help="Path to tokenizer.json; defaults to the model directory")
    args = parser.parse_args()

    if args.first_layer < 0 or args.layers <= 0 or args.first_layer + args.layers > 30:
        raise SystemExit("--first-layer must be >= 0 and --layers must keep the window within 30 layers")

    model_path = Path(args.model)
    config_path = Path(args.config_json) if args.config_json else model_path.parent / "config.json"
    tokenizer_path = Path(args.tokenizer_json) if args.tokenizer_json else model_path.parent / "tokenizer.json"
    out_dir = Path(args.out_dir)
    bitnet_dir = out_dir / "BITNET"
    config = load_model_config(config_path)
    header_len, header = read_header(model_path)

    if bitnet_dir.exists():
        shutil.rmtree(bitnet_dir)

    exported = []
    for layer in range(args.first_layer, args.first_layer + args.layers):
        layer_dir = bitnet_dir / f"L{layer:02d}"
        for short_name, suffix, slot_offset in TENSORS:
            tensor_name = f"model.layers.{layer}.{suffix}.weight"
            dst = layer_dir / EMMC_NAMES[short_name]
            meta = export_tensor(model_path, header_len, header, tensor_name, dst)
            meta["layer"] = layer
            meta["short_name"] = short_name
            meta["emmc_path"] = f"0:/BITNET/L{layer:02d}/{EMMC_NAMES[short_name]}"
            meta["slot_offset"] = slot_offset
            exported.append(meta)

    aux_meta = export_aux_assets(model_path, header_len, header, bitnet_dir)
    rope_meta = export_rope_assets(
        bitnet_dir,
        int(config.get("max_position_embeddings", DEFAULT_MAX_POSITION_EMBEDDINGS)),
        HIDDEN_SIZE // int(config.get("num_attention_heads", 20)),
        float(config.get("rope_theta", DEFAULT_ROPE_THETA)),
    )
    embedding_meta = export_embedding_shards(model_path, header_len, header, bitnet_dir)
    lm_head_i8_meta = export_lm_head_i8_shards_from_bf16_embedding(bitnet_dir)
    tokenizer_meta = export_tokenizer_assets(
        tokenizer_path,
        bitnet_dir,
        int(config.get("bos_token_id", DEFAULT_BOS_ID)),
        int(config.get("eos_token_id", DEFAULT_EOS_ID)),
    )
    control_bin = write_control(out_dir, args.first_layer, args.layers, args.activation_seed, args.flags)
    meta = {
        "source_model": str(model_path),
        "source_config": str(config_path),
        "source_tokenizer": str(tokenizer_path),
        "first_layer": args.first_layer,
        "layers": args.layers,
        "activation_seed": args.activation_seed,
        "flags": args.flags,
        "control_magic": f"0x{CONTROL_MAGIC:08X}",
        "control_bin": str(control_bin),
        "emmc_control_base": "0x4D000000",
        "emmc_package_dir": str(bitnet_dir),
        "hidden_size": HIDDEN_SIZE,
        "kv_size": KV_SIZE,
        "ffn_size": FFN_SIZE,
        "weight_slot_base": "0x56000000",
        "weight_slot_stride": "0x02000000",
        "tensors": exported,
        "aux": aux_meta,
        "rope": rope_meta,
        "embedding": embedding_meta,
        "lm_head_i8": lm_head_i8_meta,
        "tokenizer": tokenizer_meta,
        "fpga_layout": "packed[out_features/4][in_features], 2-bit codes: 0=-1, 1=0, 2=+1",
    }
    manifest = write_manifest(bitnet_dir, meta)
    meta["manifest"] = str(manifest)

    meta_json = out_dir / "emmc_package.meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
