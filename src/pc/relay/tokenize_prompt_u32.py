#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=True
    )
    text = args.input.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    args.output.write_bytes(b"".join(struct.pack("<I", token) for token in token_ids))
    print(f"TOKEN_COUNT={len(token_ids)}")
    print(f"TOKEN_BYTES={len(token_ids) * 4}")


if __name__ == "__main__":
    main()
