#!/usr/bin/env python3
import argparse
import json
import struct
import sys
import unicodedata
from pathlib import Path


MAGIC = 0x314B5442
VERSION = 1
DECODE_MAGIC = 0x31445442
DECODE_VERSION = 1
PRETOK_MAGIC = 0x31505442
PRETOK_VERSION = 1
TOKEN_ID_NONE = 0xFFFFFFFF
DEFAULT_BOS_ID = 128000
DEFAULT_EOS_ID = 128001
DEFAULT_VOCAB_SIZE = 128256

PRETOK_FLAG_LETTER = 1
PRETOK_FLAG_NUMBER = 2
PRETOK_FLAG_SPACE = 4

UNICODE_WHITE_SPACE_RANGES = [
    (0x0009, 0x000D),
    (0x0020, 0x0020),
    (0x0085, 0x0085),
    (0x00A0, 0x00A0),
    (0x1680, 0x1680),
    (0x2000, 0x200A),
    (0x2028, 0x2029),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
]


def bytes_to_unicode():
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = visible[:]
    n = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + n)
            n += 1
    return {byte: chr(char) for byte, char in zip(visible, chars)}


class Node:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children = {}
        self.token_id = TOKEN_ID_NONE


def decode_bytelevel_token(token, byte_decoder):
    out = bytearray()
    for ch in token:
        byte = byte_decoder.get(ch)
        if byte is None:
            return None
        out.append(byte)
    return bytes(out)


def insert_token(nodes, raw_bytes, token_id):
    node_index = 0
    for byte in raw_bytes:
        node = nodes[node_index]
        child = node.children.get(byte)
        if child is None:
            child = len(nodes)
            node.children[byte] = child
            nodes.append(Node())
        node_index = child
    nodes[node_index].token_id = token_id


def build_trie(vocab):
    byte_decoder = {ch: byte for byte, ch in bytes_to_unicode().items()}
    nodes = [Node()]
    skipped = []
    max_token_bytes = 0

    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        raw = decode_bytelevel_token(token, byte_decoder)
        if raw is None or len(raw) == 0:
            skipped.append((token_id, token))
            continue
        insert_token(nodes, raw, token_id)
        max_token_bytes = max(max_token_bytes, len(raw))

    return nodes, max_token_bytes, skipped


def flatten(nodes):
    flat_nodes = []
    flat_edges = []

    for node in nodes:
        child_base = len(flat_edges)
        for byte, child in sorted(node.children.items()):
            flat_edges.append((byte, child))
        flat_nodes.append((child_base, len(node.children), node.token_id))

    return flat_nodes, flat_edges


def write_asset(out_file, nodes, edges, max_token_bytes, bos_id, eos_id):
    header_size = 12 * 4
    node_size = struct.calcsize("<IHHI")
    edge_size = struct.calcsize("<BBHI")
    nodes_offset = header_size
    edges_offset = nodes_offset + (len(nodes) * node_size)
    total_bytes = edges_offset + (len(edges) * edge_size)

    with out_file.open("wb") as f:
        f.write(
            struct.pack(
                "<12I",
                MAGIC,
                VERSION,
                total_bytes,
                len(nodes),
                len(edges),
                max_token_bytes,
                bos_id,
                eos_id,
                nodes_offset,
                edges_offset,
                0,
                0,
            )
        )
        for child_base, child_count, token_id in nodes:
            if child_count > 0xFFFF:
                raise ValueError("A trie node has too many children")
            f.write(struct.pack("<IHHI", child_base, child_count, 0, token_id))
        for byte, child in edges:
            f.write(struct.pack("<BBHI", byte, 0, 0, child))

    return total_bytes


def token_raw_bytes_by_id(vocab):
    byte_decoder = {ch: byte for byte, ch in bytes_to_unicode().items()}
    raw_by_id = {}
    skipped = []

    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        raw = decode_bytelevel_token(token, byte_decoder)
        if raw is None:
            skipped.append((token_id, token))
            continue
        raw_by_id[int(token_id)] = raw

    return raw_by_id, skipped


def write_decode_asset(out_file, vocab, added_tokens, vocab_size=DEFAULT_VOCAB_SIZE):
    raw_by_id, skipped = token_raw_bytes_by_id(vocab)
    special_count = 0
    entries = [(0, 0, 0) for _ in range(vocab_size)]
    payload = bytearray()

    for item in added_tokens:
        token_id = int(item.get("id", -1))
        content = item.get("content")
        if 0 <= token_id < vocab_size and isinstance(content, str):
            raw_by_id[token_id] = content.encode("utf-8")
            special_count += 1

    for token_id, raw in sorted(raw_by_id.items()):
        if not (0 <= token_id < vocab_size):
            continue
        if len(raw) > 0xFFFF:
            raise ValueError(f"Token byte payload too long: id={token_id}")
        offset = len(payload)
        payload.extend(raw)
        entries[token_id] = (offset, len(raw), 1 if token_id >= len(vocab) else 0)

    header_size = 8 * 4
    entry_size = struct.calcsize("<IHH")
    entries_offset = header_size
    payload_offset = entries_offset + (vocab_size * entry_size)
    total_bytes = payload_offset + len(payload)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("wb") as f:
        f.write(struct.pack(
            "<8I",
            DECODE_MAGIC,
            DECODE_VERSION,
            total_bytes,
            vocab_size,
            entries_offset,
            payload_offset,
            special_count,
            0,
        ))
        for offset, length, flags in entries:
            f.write(struct.pack("<IHH", payload_offset + offset if length else 0, length, flags))
        f.write(payload)

    return {
        "path": str(out_file),
        "emmc_path": "0:/BITNET/TOK/DECODE.BIN",
        "magic": f"0x{DECODE_MAGIC:08X}",
        "version": DECODE_VERSION,
        "vocab_size": vocab_size,
        "payload_bytes": len(payload),
        "special_tokens": special_count,
        "skipped_tokens": len(skipped),
        "total_bytes": total_bytes,
    }


def unicode_space_flags(codepoint):
    for start, end in UNICODE_WHITE_SPACE_RANGES:
        if start <= codepoint <= end:
            return PRETOK_FLAG_SPACE
    return 0


def unicode_class_flags(codepoint):
    flags = unicode_space_flags(codepoint)
    category = unicodedata.category(chr(codepoint))

    if category.startswith("L"):
        flags |= PRETOK_FLAG_LETTER
    elif category.startswith("N"):
        flags |= PRETOK_FLAG_NUMBER
    return flags


def build_pretokenizer_ranges():
    ranges = []
    range_start = 0
    last_flags = unicode_class_flags(0)

    for codepoint in range(1, 0x110000):
        flags = unicode_class_flags(codepoint)
        if flags == last_flags:
            continue
        if last_flags != 0:
            ranges.append((range_start, codepoint - 1, last_flags))
        range_start = codepoint
        last_flags = flags

    if last_flags != 0:
        ranges.append((range_start, 0x10FFFF, last_flags))
    return ranges


def write_pretokenizer_asset(out_file):
    ranges = build_pretokenizer_ranges()
    header_size = 8 * 4
    range_size = struct.calcsize("<III")
    ranges_offset = header_size
    total_bytes = ranges_offset + (len(ranges) * range_size)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("wb") as f:
        f.write(struct.pack(
            "<8I",
            PRETOK_MAGIC,
            PRETOK_VERSION,
            total_bytes,
            len(ranges),
            ranges_offset,
            PRETOK_FLAG_LETTER,
            PRETOK_FLAG_NUMBER,
            PRETOK_FLAG_SPACE,
        ))
        for start, end, flags in ranges:
            f.write(struct.pack("<III", start, end, flags))

    return {
        "path": str(out_file),
        "emmc_path": "0:/BITNET/TOK/PRET.BIN",
        "magic": f"0x{PRETOK_MAGIC:08X}",
        "version": PRETOK_VERSION,
        "ranges": len(ranges),
        "total_bytes": total_bytes,
        "unicode_source": f"python-unicodedata-{unicodedata.unidata_version}",
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Export a PS-side ByteLevel vocab trie for AXU3EGB BitNet.")
    parser.add_argument("--tokenizer-json", required=True, help="Path to tokenizer.json")
    parser.add_argument("--out", required=True, help="Output TRIE.BIN path")
    parser.add_argument("--decode-out", help="Optional output DECODE.BIN path")
    parser.add_argument("--pretok-out", help="Optional output PRET.BIN path")
    parser.add_argument("--bos-id", type=int, default=DEFAULT_BOS_ID)
    parser.add_argument("--eos-id", type=int, default=DEFAULT_EOS_ID)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_json)
    out_file = Path(args.out)
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
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
    out_file.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = write_asset(out_file, flat_nodes, flat_edges,
                              max_token_bytes, args.bos_id, args.eos_id)
    decode_meta = None
    pretok_meta = None

    if args.decode_out:
        decode_meta = write_decode_asset(Path(args.decode_out), vocab,
                                         data.get("added_tokens", []),
                                         args.vocab_size)
    if args.pretok_out:
        pretok_meta = write_pretokenizer_asset(Path(args.pretok_out))

    meta = {
        "source": str(tokenizer_path),
        "output": str(out_file),
        "magic": f"0x{MAGIC:08X}",
        "version": VERSION,
        "vocab_tokens": len(vocab),
        "nodes": len(flat_nodes),
        "edges": len(flat_edges),
        "max_token_bytes": max_token_bytes,
        "skipped_tokens": len(skipped),
        "total_bytes": total_bytes,
        "emmc_path": "0:/BITNET/TOK/TRIE.BIN",
        "decode": decode_meta,
        "pretokenizer": pretok_meta,
        "note": "PS-side ByteLevel longest-match trie with optional Unicode regex pre-tokenizer and detokenizer assets.",
    }
    meta_file = out_file.with_suffix(".meta.json")
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
