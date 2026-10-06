#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensor_stats(value: torch.Tensor) -> dict:
    data = value.detach().float().cpu().contiguous()
    flat = data.reshape(-1)
    result = {
        "shape": list(data.shape),
        "dtype": str(value.dtype),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "rms": float(torch.sqrt(torch.mean(flat * flat))),
        "absmax": float(flat.abs().max()),
    }
    if data.ndim >= 2:
        last = data.reshape(-1, data.shape[-1])[-1]
        result["last_token"] = {
            "min": float(last.min()),
            "max": float(last.max()),
            "mean": float(last.mean()),
            "rms": float(torch.sqrt(torch.mean(last * last))),
            "absmax": float(last.abs().max()),
        }
    return result


def make_hook(records: dict, captured: dict, name: str):
    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        records[name] = tensor_stats(value)
        if name.startswith("layer_00."):
            captured[name] = value.detach().float().cpu()[0, -1].contiguous()

    return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official Hugging Face BitNet CPU reference path."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    records = {}
    weight_scales = {}
    captured = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        modules = {
            "input_norm": layer.input_layernorm,
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "attn_sub_norm": layer.self_attn.attn_sub_norm,
            "o_proj": layer.self_attn.o_proj,
            "post_norm": layer.post_attention_layernorm,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "ffn_sub_norm": layer.mlp.ffn_sub_norm,
            "down_proj": layer.mlp.down_proj,
            "layer_output": layer,
        }
        for suffix, module in modules.items():
            name = f"layer_{index:02d}.{suffix}"
            handles.append(module.register_forward_hook(make_hook(records, captured, name)))
            if hasattr(module, "weight_scale"):
                weight_scales[name] = float(module.weight_scale.detach().float().cpu())
    def final_norm_hook(_module, _inputs, output):
        records["final_norm"] = tensor_stats(output)
        captured["final_norm"] = output.detach().float().cpu()[0, -1].contiguous()

    handles.append(model.model.norm.register_forward_hook(final_norm_hook))

    inputs = tokenizer(args.prompt, return_tensors="pt")
    forward_started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1].float().cpu()
    forward_seconds = time.perf_counter() - forward_started
    for handle in handles:
        handle.remove()

    top_k = min(max(args.top_k, 1), logits.numel())
    values, indices = torch.topk(logits, top_k)
    top_tokens = [
        {
            "token_id": int(token_id),
            "logit": float(logit),
            "text": tokenizer.decode([int(token_id)]),
        }
        for token_id, logit in zip(indices, values)
    ]
    input_ids = inputs.input_ids[0].tolist()
    result = {
        "model": str(args.model.resolve()),
        "prompt": args.prompt,
        "input_ids": input_ids,
        "input_tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "top_tokens": top_tokens,
        "weight_scales": weight_scales,
        "forward_seconds": forward_seconds,
        "total_seconds": time.perf_counter() - started,
        "tensors": records,
    }

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        final_norm = captured["final_norm"]
        quant_scale = 127.0 / float(final_norm.abs().max())
        final_i8 = torch.round(final_norm * quant_scale).clamp(-128, 127).to(torch.int8)
        (args.dump_dir / "final_norm.f32.bin").write_bytes(final_norm.numpy().tobytes())
        (args.dump_dir / "final_norm.i8.bin").write_bytes(final_i8.numpy().tobytes())
        (args.dump_dir / "logits.f32.bin").write_bytes(logits.numpy().tobytes())
        for name, value in captured.items():
            if name == "final_norm":
                continue
            file_name = name.replace(".", "_") + ".f32.bin"
            (args.dump_dir / file_name).write_bytes(value.numpy().tobytes())
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
