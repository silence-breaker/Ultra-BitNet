#!/usr/bin/env python3
"""Validate the geometry, headers and payload sizes of a BitNet model package."""

from __future__ import annotations

import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MANIFEST_MAGIC = "BITNET_AXU3EGB_EMMC_PACKAGE"
CONTROL_MAGIC = 0x42454D31
AUX_MAGIC = 0x42584131  # numeric ASCII tag BXA1 used by the package exporter
AUX_VERSION = 1
AUX_HEADER_WORDS = 16
ROPE_MAGIC = 0x42505231
ROPE_VERSION = 1

SUPPORTED_LAYERS = 30
SUPPORTED_HIDDEN_SIZE = 2560
SUPPORTED_KV_SIZE = 640
SUPPORTED_FFN_SIZE = 6912
SUPPORTED_VOCAB_SIZE = 128256
SUPPORTED_SCALE_COUNT = 7
SUPPORTED_Q_HEADS = 20
SUPPORTED_ROPE_POSITIONS = 4096
SUPPORTED_ROPE_THETA = 500000.0
SUPPORTED_TOKENIZER_TRIE_BYTES = 5490460

TOKENIZER_TRIE_HEADER = (
    0x314B5442, 1, 5490460, 274521, 274520, 128,
    128000, 128001, 48, 3294300, 0, 0,
)
TOKENIZER_DECODE_HEADER = (
    0x31445442, 1, 1864889, SUPPORTED_VOCAB_SIZE,
    32, 1026080, 256, 0,
)
# export_tokenizer_trie.py stores the three class flags in separate words;
# PRET.BIN has no additional reserved word in its eight-word v1 header.
TOKENIZER_PRET_HEADER = (
    0x31505442, 1, 9716, 807, 32, 1, 2, 4,
)

PROFILE_LEGACY = "legacy-v0-codes-123"
PROFILE_CURRENT = "exporter-v1-codes-012"
LEGACY_CODES = "1:-1,2:0,3:+1"
CURRENT_CODES = "0:-1,1:0,2:+1"

ALIGNMENT = 32


class PackageError(ValueError):
    """Raised when a package or requested layout violates the ABI."""


REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


def is_link_or_reparse(metadata: os.stat_result) -> bool:
    """Return true for POSIX symlinks and Windows reparse-point aliases."""
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & REPARSE_POINT_FLAG)
    )


def package_regular_files(package_root: Path) -> tuple[Path, ...]:
    """Audit a package tree without following aliases and return its files.

    The caller may name the package root through a symlink; ``locate_package_root``
    resolves that one alias.  Below the resolved root, however, every directory
    component and file must be a real directory entry.  This keeps validation
    and later output-identity checks on exactly the same set of inodes.
    """
    root = package_root.resolve()
    try:
        root_metadata = root.stat()
    except OSError as exc:
        raise PackageError(f"cannot inspect package root {root}: {exc}") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise PackageError(f"package root is not a directory: {root}")

    files: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(str(directory)) as iterator:
                entries = list(iterator)
        except OSError as exc:
            raise PackageError(f"cannot inspect package directory {directory}: {exc}") from exc

        for entry in entries:
            entry_path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PackageError(f"cannot inspect package entry {entry_path}: {exc}") from exc
            try:
                relative = entry_path.relative_to(root).as_posix()
            except ValueError as exc:
                raise PackageError(f"package entry escapes package root: {entry_path}") from exc

            if is_link_or_reparse(metadata):
                raise PackageError(
                    f"package contains a symbolic link or reparse point: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(entry_path)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(entry_path)
            else:
                raise PackageError(f"package contains a non-regular entry: {relative}")

    visit(root)
    return tuple(files)


@dataclass(frozen=True)
class ModelShape:
    first_layer: int
    layers: int
    hidden_size: int
    kv_size: int
    ffn_size: int
    vocab_size: int = SUPPORTED_VOCAB_SIZE


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def parse_int(value: str, field: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise PackageError(f"manifest field {field!r} is not an integer: {value!r}") from exc


def locate_package_root(path: Path) -> Path:
    path = path.resolve()
    candidates = [path, path / "BITNET"] if path.is_dir() else []
    for candidate in candidates:
        if (candidate / "MANIFEST.TXT").is_file():
            # The package root itself may be named through one alias.  Internal
            # aliases are rejected by package_regular_files().
            return candidate.resolve()
    raise PackageError(
        f"cannot find MANIFEST.TXT in {path} or its BITNET subdirectory")


def read_manifest(package_root: Path) -> dict[str, str]:
    manifest_path = package_root / "MANIFEST.TXT"
    try:
        lines = manifest_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PackageError(f"cannot read ASCII manifest: {manifest_path}: {exc}") from exc

    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty or nonempty[0] != MANIFEST_MAGIC:
        got = nonempty[0] if nonempty else "<empty>"
        raise PackageError(f"bad manifest magic: got {got!r}, expected {MANIFEST_MAGIC!r}")

    fields: dict[str, str] = {}
    for line_no, line in enumerate(nonempty[1:], start=2):
        if "=" not in line:
            raise PackageError(f"MANIFEST.TXT:{line_no}: expected key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            raise PackageError(f"MANIFEST.TXT:{line_no}: empty key or value")
        if key in fields:
            raise PackageError(f"MANIFEST.TXT:{line_no}: duplicate field {key!r}")
        fields[key] = value
    return fields


def require_manifest_fields(fields: dict[str, str], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in fields]
    if missing:
        raise PackageError(f"manifest is missing fields: {', '.join(missing)}")


def manifest_asset_path(package_root: Path, emmc_path: str) -> tuple[Path, str]:
    """Resolve ``0:/BITNET/...`` without trusting it to escape the package."""
    normalized = emmc_path.replace("\\", "/")
    if ":" in normalized:
        normalized = normalized.split(":", 1)[1]
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("/", "")]
    if parts and parts[0].upper() == "BITNET":
        parts = parts[1:]
    if not parts or any(part in (".", "..") for part in parts):
        raise PackageError(f"unsafe eMMC asset path: {emmc_path!r}")
    relative = PurePosixPath(*parts).as_posix()
    candidate = package_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(package_root.resolve())
    except ValueError as exc:
        raise PackageError(f"asset escapes package root: {emmc_path!r}") from exc
    return candidate, relative


def expect_file_size(path: Path, relative: str, expected: int) -> None:
    if not path.is_file():
        raise PackageError(f"missing package file: {relative}")
    actual = path.stat().st_size
    if actual != expected:
        raise PackageError(
            f"wrong file size for {relative}: got {actual}, expected {expected}")


def read_words(path: Path, words: int, relative: str) -> tuple[int, ...]:
    byte_count = words * 4
    try:
        with path.open("rb") as stream:
            raw = stream.read(byte_count)
    except OSError as exc:
        raise PackageError(f"cannot read {relative}: {exc}") from exc
    if len(raw) != byte_count:
        raise PackageError(f"short header in {relative}: got {len(raw)}, expected {byte_count}")
    return struct.unpack(f"<{words}I", raw)


def resolve_manifest_profile(fields: dict[str, str]) -> dict[str, Any]:
    """Resolve one of two exact package ABIs, recording every legacy inference."""
    require_manifest_fields(fields, (
        "control_magic", "first_layer", "layers", "hidden_size", "kv_size", "ffn_size",
        "layout", "codes", "aux_path", "embedding_shards",
        "embedding_rows_per_shard", "lm_head",
    ))
    resolved = dict(fields)
    inferred: dict[str, dict[str, Any]] = {}
    control_magic = parse_int(fields["control_magic"], "control_magic")
    if control_magic != CONTROL_MAGIC:
        raise PackageError(
            f"wrong control_magic: got 0x{control_magic:08x}, "
            f"expected 0x{CONTROL_MAGIC:08x}")

    if fields["codes"] == CURRENT_CODES:
        require_manifest_fields(fields, (
            "rope_path", "rope_max_positions", "rope_theta",
            "lm_head_i8_shards", "lm_head_i8_rows_per_shard", "tokenizer_path",
        ))
        if fields["lm_head"] != "tied_to_embedding_rowwise_i8_f32_scale_resident":
            raise PackageError(
                f"current profile has unexpected lm_head: {fields['lm_head']!r}")
        return {
            "name": PROFILE_CURRENT,
            "source": (
                f"manifest codes={CURRENT_CODES} plus explicit RoPE/LMH/tokenizer fields"),
            "resolved_fields": resolved,
            "inferred_manifest_fields": inferred,
            "gemv_ternary_map": 1,
            "codebook": {"0": -1, "1": 0, "2": 1},
        }

    if fields["codes"] == LEGACY_CODES:
        if fields["lm_head"] != "tied_to_embedding":
            raise PackageError(
                f"legacy profile has unexpected lm_head: {fields['lm_head']!r}")
        forbidden_hybrid = [name for name in (
            "rope_path", "rope_max_positions", "lm_head_i8_shards",
            "lm_head_i8_rows_per_shard", "tokenizer_path",
        ) if name in fields]
        if forbidden_hybrid:
            raise PackageError(
                "legacy codebook is mixed with current-profile fields: "
                + ", ".join(forbidden_hybrid))

        legacy_values: dict[str, tuple[str, str]] = {
            "rope_path": (
                "0:/BITNET/AUX/ROPE.BIN",
                "legacy profile fixed path; cross-checked against the RoPE header"),
            "rope_max_positions": (
                str(SUPPORTED_ROPE_POSITIONS),
                "legacy profile model constant; cross-checked against the RoPE header"),
            "lm_head_i8_shards": (
                fields["embedding_shards"],
                "legacy tied-embedding profile inherits embedding_shards"),
            "lm_head_i8_rows_per_shard": (
                fields["embedding_rows_per_shard"],
                "legacy tied-embedding profile inherits embedding_rows_per_shard"),
            "tokenizer_path": (
                "0:/BITNET/TOK/TRIE.BIN",
                "legacy profile fixed path; asset size and siblings are validated"),
        }
        if "rope_theta" not in fields:
            legacy_values["rope_theta"] = (
                str(SUPPORTED_ROPE_THETA),
                "legacy profile model constant; cross-checked against the RoPE header")
        for name, (value, source) in legacy_values.items():
            resolved[name] = value
            inferred[name] = {"source": source, "value": value}
        return {
            "name": PROFILE_LEGACY,
            "source": (
                f"manifest codes={LEGACY_CODES}, lm_head=tied_to_embedding, "
                "and absence of current-profile RoPE/LMH/tokenizer fields"),
            "resolved_fields": resolved,
            "inferred_manifest_fields": inferred,
            "gemv_ternary_map": 0,
            "codebook": {"1": -1, "2": 0, "3": 1},
        }

    raise PackageError(
        f"unsupported ternary codebook/profile: {fields['codes']!r}; "
        f"expected {LEGACY_CODES!r} or {CURRENT_CODES!r}")


def parse_shape(fields: dict[str, str]) -> ModelShape:
    shape = ModelShape(
        first_layer=parse_int(fields["first_layer"], "first_layer"),
        layers=parse_int(fields["layers"], "layers"),
        hidden_size=parse_int(fields["hidden_size"], "hidden_size"),
        kv_size=parse_int(fields["kv_size"], "kv_size"),
        ffn_size=parse_int(fields["ffn_size"], "ffn_size"),
    )
    supported = ModelShape(0, SUPPORTED_LAYERS, SUPPORTED_HIDDEN_SIZE,
                           SUPPORTED_KV_SIZE, SUPPORTED_FFN_SIZE)
    if shape != supported:
        raise PackageError(f"unsupported model shape: got {shape}, expected {supported}")
    if fields["layout"] != "packed[out_features/4][in_features]":
        raise PackageError(f"unsupported packed layout: {fields['layout']!r}")
    for dimension_name, dimension in (
            ("hidden_size", shape.hidden_size),
            ("kv_size", shape.kv_size),
            ("ffn_size", shape.ffn_size)):
        if dimension % 4:
            raise PackageError(f"{dimension_name} must be divisible by four")
    return shape


def expected_weight_sizes(shape: ModelShape) -> tuple[tuple[str, int], ...]:
    h, k, f = shape.hidden_size, shape.kv_size, shape.ffn_size
    return (
        ("Q.BIN", h * h // 4),
        ("K.BIN", k * h // 4),
        ("V.BIN", k * h // 4),
        ("O.BIN", h * h // 4),
        ("GATE.BIN", f * h // 4),
        ("UP.BIN", f * h // 4),
        ("DOWN.BIN", h * f // 4),
    )


def validate_aux(package_root: Path, fields: dict[str, str], shape: ModelShape) -> dict[str, Any]:
    path, relative = manifest_asset_path(package_root, fields["aux_path"])
    if not path.is_file():
        raise PackageError(f"missing package file: {relative}")
    words = read_words(path, AUX_HEADER_WORDS, relative)

    final_norm = AUX_HEADER_WORDS * 4
    input_norm = final_norm + shape.hidden_size * 2
    post_norm = input_norm + shape.layers * shape.hidden_size * 2
    attn_norm = post_norm + shape.layers * shape.hidden_size * 2
    ffn_norm = attn_norm + shape.layers * shape.hidden_size * 2
    scales = ffn_norm + shape.layers * shape.ffn_size * 2
    total = align_up(scales + shape.layers * SUPPORTED_SCALE_COUNT * 2, 4)
    expected_words = (
        AUX_MAGIC, AUX_VERSION, total, shape.hidden_size, shape.ffn_size,
        shape.layers, SUPPORTED_SCALE_COUNT, final_norm, input_norm, post_norm,
        attn_norm, ffn_norm, scales, 0, 0, 0,
    )
    if words != expected_words:
        names = (
            "magic", "version", "total_bytes", "hidden_size", "ffn_size",
            "layers", "scale_count", "final_norm", "input_layernorm",
            "post_attention_layernorm", "attn_sub_norm", "ffn_sub_norm",
            "weight_scales", "reserved13", "reserved14", "reserved15",
        )
        mismatches = [
            f"{name}=0x{actual:08x} expected=0x{expected:08x}"
            for name, actual, expected in zip(names, words, expected_words)
            if actual != expected
        ]
        raise PackageError(f"invalid AUX header in {relative}: " + "; ".join(mismatches))
    expect_file_size(path, relative, total)
    return {
        "relative_path": relative,
        "bytes": total,
        "offsets": {
            "final_norm": final_norm,
            "input_layernorm": input_norm,
            "post_attention_layernorm": post_norm,
            "attn_sub_norm": attn_norm,
            "ffn_sub_norm": ffn_norm,
            "weight_scales": scales,
        },
    }


def validate_rope(package_root: Path, fields: dict[str, str], shape: ModelShape) -> dict[str, Any]:
    path, relative = manifest_asset_path(package_root, fields["rope_path"])
    if not path.is_file():
        raise PackageError(f"missing package file: {relative}")
    words = read_words(path, 12, relative)
    max_positions = parse_int(fields["rope_max_positions"], "rope_max_positions")
    expected_head_dim = shape.hidden_size // SUPPORTED_Q_HEADS
    expected_pairs = expected_head_dim // 2
    expected_bytes = 12 * 4 + max_positions * expected_pairs * 4
    if words[0] != ROPE_MAGIC or words[1] != ROPE_VERSION:
        raise PackageError(f"invalid RoPE magic/version in {relative}")
    checks = {
        "total_bytes": (words[2], expected_bytes),
        "max_positions": (words[3], max_positions),
        "head_dim": (words[4], expected_head_dim),
        "pair_count": (words[5], expected_pairs),
        "table_offset": (words[6], 48),
        "q15_scale": (words[7], 32767),
    }
    try:
        expected_theta_milli = int(round(float(fields["rope_theta"]) * 1000.0))
    except ValueError as exc:
        raise PackageError(
            f"manifest field 'rope_theta' is not numeric: {fields['rope_theta']!r}") from exc
    checks["theta_milli"] = (words[8], expected_theta_milli)
    bad = [f"{name}={got} expected={expected}"
           for name, (got, expected) in checks.items() if got != expected]
    if bad:
        raise PackageError(f"invalid RoPE header in {relative}: " + "; ".join(bad))
    if any(words[index] != 0 for index in (9, 10, 11)):
        raise PackageError(f"nonzero reserved RoPE header words in {relative}")
    expect_file_size(path, relative, expected_bytes)
    return {
        "relative_path": relative,
        "bytes": expected_bytes,
        "max_positions": max_positions,
        "head_dim": expected_head_dim,
    }


def file_record(relative_path: str, size: int, category: str) -> dict[str, Any]:
    return {"bytes": size, "category": category, "relative_path": relative_path}


def validate_exact_header(path: Path, relative: str, expected: tuple[int, ...],
                          field_names: tuple[str, ...]) -> dict[str, Any]:
    if len(expected) != len(field_names):
        raise ValueError("tokenizer header schema mismatch")
    expect_file_size(path, relative, expected[2])
    words = read_words(path, len(expected), relative)
    if words != expected:
        mismatches = [
            f"{name}=0x{actual:08x} expected=0x{wanted:08x}"
            for name, actual, wanted in zip(field_names, words, expected)
            if actual != wanted
        ]
        raise PackageError(
            f"invalid tokenizer header in {relative}: " + "; ".join(mismatches))
    return {
        "relative_path": relative,
        "bytes": expected[2],
        "header": {name: value for name, value in zip(field_names, words)},
    }


def validate_tokenizer(package_root: Path, fields: dict[str, str]) -> dict[str, Any]:
    trie_path, trie_relative = manifest_asset_path(
        package_root, fields["tokenizer_path"])
    if PurePosixPath(trie_relative).name != "TRIE.BIN":
        raise PackageError(
            f"tokenizer_path must name TRIE.BIN, got {trie_relative!r}")
    tokenizer_dir = trie_path.parent
    relative_dir = PurePosixPath(trie_relative).parent

    trie = validate_exact_header(
        trie_path, trie_relative, TOKENIZER_TRIE_HEADER,
        ("magic", "version", "total_bytes", "nodes", "edges",
         "max_token_bytes", "bos_id", "eos_id", "nodes_offset",
         "edges_offset", "reserved10", "reserved11"))
    decode_relative = relative_dir.joinpath("DECODE.BIN").as_posix()
    decode = validate_exact_header(
        tokenizer_dir / "DECODE.BIN", decode_relative, TOKENIZER_DECODE_HEADER,
        ("magic", "version", "total_bytes", "vocab_size", "entries_offset",
         "payload_offset", "special_count", "reserved7"))
    pret_relative = relative_dir.joinpath("PRET.BIN").as_posix()
    pret = validate_exact_header(
        tokenizer_dir / "PRET.BIN", pret_relative, TOKENIZER_PRET_HEADER,
        ("magic", "version", "total_bytes", "range_count", "ranges_offset",
         "letter_flag", "number_flag", "space_flag"))
    return {"trie": trie, "decode": decode, "pretokenizer": pret}


def validate_package(package_root: Path) -> dict[str, Any]:
    package_regular_files(package_root)
    manifest_fields = read_manifest(package_root)
    profile = resolve_manifest_profile(manifest_fields)
    fields = profile["resolved_fields"]
    shape = parse_shape(fields)

    weights: list[dict[str, Any]] = []
    for layer in range(shape.first_layer, shape.first_layer + shape.layers):
        for filename, size in expected_weight_sizes(shape):
            relative = f"L{layer:02d}/{filename}"
            expect_file_size(package_root / PurePosixPath(relative), relative, size)
            weights.append(file_record(relative, size, "transformer_weight"))

    aux = validate_aux(package_root, fields, shape)
    rope = validate_rope(package_root, fields, shape)
    tokenizer = validate_tokenizer(package_root, fields)

    embedding_shards = parse_int(fields["embedding_shards"], "embedding_shards")
    embedding_rows = parse_int(fields["embedding_rows_per_shard"],
                               "embedding_rows_per_shard")
    lm_shards = parse_int(fields["lm_head_i8_shards"], "lm_head_i8_shards")
    lm_rows = parse_int(fields["lm_head_i8_rows_per_shard"],
                        "lm_head_i8_rows_per_shard")
    if embedding_rows <= 0:
        raise PackageError("embedding_rows_per_shard must be positive")
    if lm_rows <= 0:
        raise PackageError("lm_head_i8_rows_per_shard must be positive")
    expected_shards = (shape.vocab_size + embedding_rows - 1) // embedding_rows
    if lm_rows != embedding_rows:
        raise PackageError("embedding and LM-head rows-per-shard must be equal")
    if embedding_shards != expected_shards or lm_shards != expected_shards:
        raise PackageError(
            f"wrong shard count: embedding={embedding_shards}, lm_head={lm_shards}, "
            f"expected={expected_shards}")

    embeddings: list[dict[str, Any]] = []
    lm_head: list[dict[str, Any]] = []
    for shard in range(expected_shards):
        row_start = shard * embedding_rows
        rows = min(embedding_rows, shape.vocab_size - row_start)
        embedding_size = rows * shape.hidden_size * 2
        lm_size = rows * (shape.hidden_size + 4)
        emb_relative = f"EMB/E{shard:02d}.BIN"
        lm_relative = f"LMH/H{shard:02d}.BIN"
        expect_file_size(package_root / PurePosixPath(emb_relative), emb_relative, embedding_size)
        expect_file_size(package_root / PurePosixPath(lm_relative), lm_relative, lm_size)
        embedding_record = file_record(emb_relative, embedding_size, "embedding_bf16")
        embedding_record.update({
            "shard": shard,
            "row_start": row_start,
            "rows": rows,
            "row_bytes": shape.hidden_size * 2,
        })
        embeddings.append(embedding_record)
        lm_record = file_record(lm_relative, lm_size, "lm_head_i8_f32_scale")
        lm_record.update({
            "shard": shard,
            "row_start": row_start,
            "rows": rows,
            "layout": "contiguous_i8_weights_then_f32_row_scales",
            "weight_bytes": rows * shape.hidden_size,
            "weight_offset_bytes": 0,
            "scale_bytes": rows * 4,
            "scale_offset_bytes": rows * shape.hidden_size,
        })
        lm_head.append(lm_record)

    return {
        "shape": shape,
        "manifest_profile": profile["name"],
        "manifest_profile_source": profile["source"],
        "inferred_manifest_fields": profile["inferred_manifest_fields"],
        "gemv_ternary_map": profile["gemv_ternary_map"],
        "ternary_codebook": profile["codebook"],
        "weights": weights,
        "aux": aux,
        "rope": rope,
        "tokenizer": tokenizer,
        "lm_head": lm_head,
        "embeddings": embeddings,
    }
