#!/usr/bin/env python3
"""Plan five K-striped banks for the TP2 board, all backed by PS DDR."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

import model_package as legacy


SCHEMA = "ultra-bitnet-five-bank-k-striped-v3"
BANK_COUNT = 5
ALIGNMENT = 128
PL_PRELOAD_ALIAS = 0x850000000
PS_HIGH_DDR_BASE = 0x800000000
PS_BANK_STRIDE = 0x10000000  # 256 MiB per HP-visible logical shard
PS_BANK_CAPACITY = PS_BANK_STRIDE
PL_BANK_CAPACITY = 1 << 28
KV_HEADS = 5
HEAD_SIZE = 128
KV_SCALE_PACK_NUM = 4
KV_SCALE_CHANNEL_BYTES = KV_SCALE_PACK_NUM * 4
KV_PACKED_BLOCK_BYTES = KV_SCALE_CHANNEL_BYTES + KV_SCALE_PACK_NUM * HEAD_SIZE
LM_TILE_ROWS = 4096
STAGING_PAGE_BYTES = 4096


class NativeLayoutError(RuntimeError):
    pass


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def partition(total: int, count: int, index: int) -> Tuple[int, int]:
    """Return a deterministic, contiguous, nearly equal [start, count]."""
    if total < 0 or count <= 0 or index < 0 or index >= count:
        raise ValueError("invalid partition geometry")
    quotient, remainder = divmod(total, count)
    length = quotient + (1 if index < remainder else 0)
    start = index * quotient + min(index, remainder)
    return start, length


def partition_tiles(total_features: int, count: int,
                    index: int) -> Tuple[int, int]:
    """Partition K only at native 128-feature tile boundaries."""
    if total_features <= 0 or total_features % HEAD_SIZE:
        raise ValueError("K features must contain whole 128-feature tiles")
    tile_start, tile_count = partition(
        total_features // HEAD_SIZE, count, index)
    return tile_start * HEAD_SIZE, tile_count * HEAD_SIZE


def striped_group_fill_cycles(input_features: int,
                              simultaneous_groups: int = 1) -> int:
    """Ideal 128-bit-bank cycles to fill one or more compact 4-row groups."""
    if simultaneous_groups <= 0:
        raise ValueError("simultaneous_groups must be positive")
    stripe_bytes = [partition_tiles(input_features, BANK_COUNT, bank)[1]
                    for bank in range(BANK_COUNT)]
    return max(stripe_bytes) * simultaneous_groups // (128 // 8)


def compact_group_replay_cycles(input_features: int) -> int:
    """Lower and upper row-pair sweeps at one 128-feature tile per cycle."""
    if input_features <= 0 or input_features % HEAD_SIZE:
        raise ValueError("invalid compact group geometry")
    return 2 * input_features // HEAD_SIZE


def tensor_geometry(shape: legacy.ModelShape) -> Dict[str, Tuple[int, int]]:
    return {
        "Q.BIN": (shape.hidden_size, shape.hidden_size),
        "K.BIN": (shape.kv_size, shape.hidden_size),
        "V.BIN": (shape.kv_size, shape.hidden_size),
        "O.BIN": (shape.hidden_size, shape.hidden_size),
        "GATE.BIN": (shape.ffn_size, shape.hidden_size),
        "UP.BIN": (shape.ffn_size, shape.hidden_size),
        "DOWN.BIN": (shape.hidden_size, shape.ffn_size),
    }


# Gate and Up are separate files in the eMMC package, but BitNet's SPU
# consumes them as one dual-output stream.  The resident image therefore uses
# one physical target whose compact groups alternate as
#
#   [Gate rows 4n..4n+3], [Up rows 4n..4n+3].
#
# This keeps the source ternary bytes intact, costs no duplicate payload, and
# lets the PL join retain only one four-feature Gate window while the matching
# Up group is replayed.  It is an image-build transform, never an inference-
# time transpose or a PS callback.
RESIDENT_WEIGHT_ORDER = (
    "Q.BIN", "K.BIN", "V.BIN", "O.BIN", "GATE_UP.BIN", "DOWN.BIN")


def _weight_key(relative_path: str) -> Tuple[int, str]:
    relative = PurePosixPath(relative_path)
    try:
        layer = int(relative.parent.name[1:])
    except (TypeError, ValueError) as exc:
        raise NativeLayoutError(
            "weight path does not contain an Lnn parent: %s" % relative_path
        ) from exc
    return layer, relative.name


def make_banks() -> List[Dict[str, Any]]:
    banks = [{
        "bank": 0,
        "kind": "ps_ddr_hpc",
        "capacity_bytes": PL_BANK_CAPACITY,
        "core_axi_base": 0,
        "preload_axi_base": PL_PRELOAD_ALIAS,
        "hp_port": None,
        "cursor": 0,
        "objects": [],
    }]
    for bank in range(1, BANK_COUNT):
        base = PS_HIGH_DDR_BASE + (bank - 1) * PS_BANK_STRIDE
        banks.append({
            "bank": bank,
            "kind": "ps_ddr_hp",
            "capacity_bytes": PS_BANK_CAPACITY,
            "core_axi_base": base,
            "preload_axi_base": base,
            "hp_port": bank - 1,
            "cursor": 0,
            "objects": [],
        })
    return banks


def allocate(bank: Dict[str, Any], name: str, size: int,
             category: str, alignment: int = ALIGNMENT,
             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if size <= 0:
        raise NativeLayoutError("cannot allocate a non-positive object")
    start = align_up(int(bank["cursor"]), alignment)
    end = start + size
    if end > int(bank["capacity_bytes"]):
        raise NativeLayoutError(
            "bank %d overflow while placing %s: end=%d capacity=%d" %
            (bank["bank"], name, end, bank["capacity_bytes"]))
    record: Dict[str, Any] = {
        "name": name,
        "category": category,
        "bank": bank["bank"],
        "local_offset": start,
        "local_offset_hex": "0x%08x" % start,
        "bytes": size,
        "end_offset_exclusive": end,
        "core_axi_address": int(bank["core_axi_base"]) + start,
        "core_axi_address_hex": "0x%010x" %
                                (int(bank["core_axi_base"]) + start),
        "preload_axi_address": int(bank["preload_axi_base"]) + start,
        "preload_axi_address_hex": "0x%010x" %
                                   (int(bank["preload_axi_base"]) + start),
    }
    if metadata:
        record.update(metadata)
    bank["objects"].append(record)
    bank["cursor"] = end
    return record


def source_piece(relative_path: str, source_offset: int, size: int,
                 destination_offset: int) -> Dict[str, Any]:
    return {
        "relative_path": relative_path,
        "source_offset": source_offset,
        "bytes": size,
        "destination_offset": destination_offset,
    }


def lm_source_pieces(records: Iterable[Dict[str, Any]], row_start: int,
                     rows: int, hidden: int, scales: bool) -> List[Dict[str, Any]]:
    row_end = row_start + rows
    pieces: List[Dict[str, Any]] = []
    covered = 0
    for record in records:
        source_start = int(record["row_start"])
        source_end = source_start + int(record["rows"])
        overlap_start = max(row_start, source_start)
        overlap_end = min(row_end, source_end)
        if overlap_start >= overlap_end:
            continue
        local_row = overlap_start - source_start
        destination_row = overlap_start - row_start
        overlap_rows = overlap_end - overlap_start
        if scales:
            source_offset = int(record["scale_offset_bytes"]) + local_row * 4
            row_bytes = 4
        else:
            source_offset = int(record["weight_offset_bytes"]) + local_row * hidden
            row_bytes = hidden
        pieces.append(source_piece(
            str(record["relative_path"]), source_offset,
            overlap_rows * row_bytes, destination_row * row_bytes))
        covered += overlap_rows
    if covered != rows:
        raise NativeLayoutError(
            "LM source shards cover %d rows, expected %d" % (covered, rows))
    return pieces


def lm_weight_stripes(records: Iterable[Dict[str, Any]], input_start: int,
                      input_features: int, hidden: int) -> List[Dict[str, Any]]:
    """Describe row-strided LM K slices without exploding the manifest."""
    stripes: List[Dict[str, Any]] = []
    covered_rows = 0
    for record in records:
        rows = int(record["rows"])
        row_start = int(record["row_start"])
        stripes.append({
            "relative_path": str(record["relative_path"]),
            "source_offset": int(record["weight_offset_bytes"]) + input_start,
            "rows": rows,
            "source_row_stride": hidden,
            "bytes_per_row": input_features,
            "destination_offset": row_start * input_features,
            "destination_row_stride": input_features,
        })
        covered_rows += rows
    if covered_rows <= 0:
        raise NativeLayoutError("LM source contains no weight rows")
    return stripes


def verify_bank(bank: Dict[str, Any]) -> None:
    previous_end = 0
    for obj in bank["objects"]:
        if int(obj["local_offset"]) < previous_end:
            raise NativeLayoutError("overlap in bank %d" % bank["bank"])
        if int(obj["local_offset"]) % ALIGNMENT:
            raise NativeLayoutError("misaligned object in bank %d" % bank["bank"])
        if int(obj["end_offset_exclusive"]) > int(bank["capacity_bytes"]):
            raise NativeLayoutError("out-of-range object in bank %d" % bank["bank"])
        previous_end = int(obj["end_offset_exclusive"])


def build_from_validated(validated: Dict[str, Any]) -> Dict[str, Any]:
    shape: legacy.ModelShape = validated["shape"]
    if shape.layers != 30 or shape.hidden_size != 2560 or shape.kv_size != 640 \
            or shape.ffn_size != 6912 or shape.vocab_size != 128256:
        raise NativeLayoutError("unsupported native production model shape")
    if shape.kv_size != KV_HEADS * HEAD_SIZE:
        raise NativeLayoutError("KV geometry does not map one head to each bank")

    banks = make_banks()
    geometry = tensor_geometry(shape)
    weight_descriptors: List[Dict[str, Any]] = []

    # Each source byte contains four output weights at one input position.
    # Every bank retains all output groups but owns a whole-tile K slice.  The
    # five slices of one group are fetched concurrently and concatenated in PL.
    # Gate/Up are allocated as one physical target with alternating complete
    # groups.  Keeping the bytes unchanged makes the transform auditable and
    # permits a four-feature streaming join in hardware.
    source_weights: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for source in validated["weights"]:
        key = _weight_key(str(source["relative_path"]))
        if key in source_weights:
            raise NativeLayoutError("duplicate weight source: %s" % (key,))
        source_weights[key] = source
    expected_keys = {
        (layer, filename)
        for layer in range(shape.layers)
        for filename in geometry
    }
    if set(source_weights) != expected_keys:
        missing = sorted(expected_keys - set(source_weights))
        extra = sorted(set(source_weights) - expected_keys)
        raise NativeLayoutError(
            "Transformer weight set mismatch: missing=%s extra=%s" %
            (missing, extra))

    def allocate_standard_weight(layer: int, filename: str) -> None:
        source = source_weights[(layer, filename)]
        relative = str(source["relative_path"])
        out_features, in_features = geometry[filename]
        packed_groups = out_features // 4
        covered_bytes = 0
        for bank_id, bank in enumerate(banks):
            input_start, input_count = partition_tiles(
                in_features, BANK_COUNT, bank_id)
            size = packed_groups * input_count
            record = allocate(
                bank,
                "layer.%02d.%s" % (layer, filename[:-4].lower()),
                size,
                "transformer_weight_shard",
                metadata={
                    "layer": layer,
                    "tensor": filename[:-4].lower(),
                    "input_start": input_start,
                    "input_features": input_count,
                    "total_input_features": in_features,
                    "output_start": 0,
                    "output_features": out_features,
                    "packed_output_group_start": 0,
                    "packed_output_groups": packed_groups,
                    "group_stride_bytes": input_count,
                    "source_strided": {
                        "relative_path": relative,
                        "source_offset": input_start,
                        "records": packed_groups,
                        "source_stride": in_features,
                        "bytes_per_record": input_count,
                        "destination_offset": 0,
                        "destination_stride": input_count,
                    },
                })
            weight_descriptors.append(record)
            covered_bytes += size
        if covered_bytes != int(source["bytes"]):
            raise NativeLayoutError(
                "weight source was not partitioned exactly: " + relative)

    def allocate_gate_up_weight(layer: int) -> None:
        gate = source_weights[(layer, "GATE.BIN")]
        up = source_weights[(layer, "UP.BIN")]
        gate_relative = str(gate["relative_path"])
        up_relative = str(up["relative_path"])
        out_features, in_features = geometry["GATE.BIN"]
        if geometry["UP.BIN"] != (out_features, in_features):
            raise NativeLayoutError("Gate/Up source geometry mismatch")
        source_groups = out_features // 4
        physical_groups = source_groups * 2
        covered_bytes = 0
        for bank_id, bank in enumerate(banks):
            input_start, input_count = partition_tiles(
                in_features, BANK_COUNT, bank_id)
            size = physical_groups * input_count
            record = allocate(
                bank,
                "layer.%02d.gate_up" % layer,
                size,
                "transformer_weight_shard",
                metadata={
                    "layer": layer,
                    "tensor": "gate_up",
                    "input_start": input_start,
                    "input_features": input_count,
                    "total_input_features": in_features,
                    "output_start": 0,
                    # This is the physical row space consumed by the matrix
                    # array.  Four physical rows alternate Gate then Up; both
                    # map to the same four semantic FFN features.
                    "output_features": out_features * 2,
                    "semantic_output_features": out_features,
                    "packed_output_group_start": 0,
                    "packed_output_groups": physical_groups,
                    "group_stride_bytes": input_count,
                    "physical_group_order":
                        "gate[4n:4n+4],up[4n:4n+4]",
                    "source_interleaved": {
                        "gate_relative_path": gate_relative,
                        "up_relative_path": up_relative,
                        "source_offset": input_start,
                        "records_per_source": source_groups,
                        "source_stride": in_features,
                        "bytes_per_record": input_count,
                        "destination_offset": 0,
                        "up_destination_offset": input_count,
                        "destination_stride": 2 * input_count,
                    },
                })
            weight_descriptors.append(record)
            covered_bytes += size
        expected_bytes = int(gate["bytes"]) + int(up["bytes"])
        if covered_bytes != expected_bytes:
            raise NativeLayoutError(
                "Gate/Up sources were not interleaved exactly at layer %d" %
                layer)

    for layer in range(shape.layers):
        for resident_name in RESIDENT_WEIGHT_ORDER:
            if resident_name == "GATE_UP.BIN":
                allocate_gate_up_weight(layer)
            else:
                allocate_standard_weight(layer, resident_name)

    # AUX and RoPE are small and read irregularly.  Replication removes a
    # single-bank serialization point for four SPUs and the shared matrix engine.
    replicated_constants: List[Dict[str, Any]] = []
    for bank in banks:
        for category in ("aux", "rope"):
            source = validated[category]
            record = allocate(
                bank,
                category,
                int(source["bytes"]),
                "replicated_constant",
                metadata={
                    "replica": True,
                    "source": source_piece(
                        str(source["relative_path"]), 0,
                        int(source["bytes"]), 0),
                })
            replicated_constants.append(record)

    # LM weights use the same K striping, so every vocabulary row consumes all
    # five ports.  The much smaller F32 scales remain row-sharded and are read
    # only when a completed row reaches the argmax backend.
    lm_descriptors: List[Dict[str, Any]] = []
    for bank_id, bank in enumerate(banks):
        input_start, input_count = partition_tiles(
            shape.hidden_size, BANK_COUNT, bank_id)
        weight = allocate(
            bank, "lm_head.weights", shape.vocab_size * input_count,
            "lm_head_i8_k_stripe",
            metadata={
                "row_start": 0,
                "rows": shape.vocab_size,
                "input_start": input_start,
                "input_features": input_count,
                "row_stride_bytes": input_count,
                "source_stripes": lm_weight_stripes(
                    validated["lm_head"], input_start, input_count,
                    shape.hidden_size),
            })
        scale_row_start, scale_rows = partition(
            shape.vocab_size, BANK_COUNT, bank_id)
        scale = allocate(
            bank, "lm_head.scales", scale_rows * 4,
            "lm_head_f32_scale_shard",
            metadata={
                "row_start": scale_row_start,
                "rows": scale_rows,
                "source_pieces": lm_source_pieces(
                    validated["lm_head"], scale_row_start, scale_rows,
                    shape.hidden_size, scales=True),
            })
        lm_descriptors.append({
            "bank": bank_id,
            "row_start": 0,
            "rows": shape.vocab_size,
            "input_start": input_start,
            "input_features": input_count,
            "weight_local_offset": weight["local_offset"],
            "weight_row_stride_bytes": input_count,
            "scale_row_start": scale_row_start,
            "scale_rows": scale_rows,
            "scale_local_offset": scale["local_offset"],
        })

    # One compact K/V head is native to each bank.  Match BitNet's cache
    # motion: a full 128-bit Q16.16 scale channel precedes four 128-byte i8
    # rows.  The block is therefore one aligned 528-byte streaming object;
    # readers never issue an unaligned four-byte AXI transaction and see all
    # four factors before their corresponding rows.
    kv_descriptors: List[Dict[str, Any]] = []
    kv_data_per_bank = shape.layers * legacy.SUPPORTED_ROPE_POSITIONS * 2 * HEAD_SIZE
    kv_scale_per_bank = shape.layers * legacy.SUPPORTED_ROPE_POSITIONS * 2 * 4
    kv_blocks_per_kind = legacy.SUPPORTED_ROPE_POSITIONS // KV_SCALE_PACK_NUM
    kv_packed_per_bank = (shape.layers * 2 * kv_blocks_per_kind *
                          KV_PACKED_BLOCK_BYTES)
    for bank_id, bank in enumerate(banks):
        packed = allocate(
            bank, "kv.head%d.packed" % bank_id, kv_packed_per_bank,
            "runtime_kv_i8_q16_packed",
            metadata={
                "zero_initialize": True,
                "head": bank_id,
                "layout": "[layer][kind][block]{scale_channel,4_rows}",
                "shape": [shape.layers, 2, kv_blocks_per_kind],
                "scale_format": "unsigned_q16.16",
                "scale_pack_num": KV_SCALE_PACK_NUM,
                "scale_channel_bytes": KV_SCALE_CHANNEL_BYTES,
                "packed_block_bytes": KV_PACKED_BLOCK_BYTES,
                "strides_bytes": {
                    "layer": 2 * kv_blocks_per_kind * KV_PACKED_BLOCK_BYTES,
                    "kind": kv_blocks_per_kind * KV_PACKED_BLOCK_BYTES,
                    "block": KV_PACKED_BLOCK_BYTES,
                    "scale_word": 4,
                    "row": HEAD_SIZE,
                },
            })
        kv_descriptors.append({
            "bank": bank_id,
            "head": bank_id,
            "packed_local_offset": packed["local_offset"],
            "packed_bytes": packed["bytes"],
            "scale_pack_num": KV_SCALE_PACK_NUM,
            "scale_channel_bytes": KV_SCALE_CHANNEL_BYTES,
            "packed_block_bytes": KV_PACKED_BLOCK_BYTES,
        })

    for bank in banks:
        verify_bank(bank)
        bank["used_bytes"] = align_up(int(bank["cursor"]))
        bank["free_bytes"] = int(bank["capacity_bytes"]) - bank["used_bytes"]
        del bank["cursor"]

    transformer_payload = sum(int(record["bytes"])
                              for record in validated["weights"])
    lm_payload = sum(int(record["bytes"])
                     for record in validated["lm_head"])

    # Only logical bank 0 crosses the physical PS-DDR -> bank-0 boundary.
    # Its Transformer ranges are page-aligned and contiguous, so the PL-owned
    # controller can copy one layer ahead without inventing five physical DDRs
    # or repacking anything at inference time.  Completed destinations are the
    # final resident offsets and therefore survive slot reuse and later decode.
    bank0_transformer = [record for record in weight_descriptors
                         if int(record["bank"]) == 0]
    staging_layers: List[Dict[str, Any]] = []
    for layer in range(shape.layers):
        records = [record for record in bank0_transformer
                   if int(record["layer"]) == layer]
        if len(records) != len(RESIDENT_WEIGHT_ORDER):
            raise NativeLayoutError(
                "bank0 layer %d does not contain every Transformer target" % layer)
        if any(int(left["end_offset_exclusive"]) !=
               int(right["local_offset"])
               for left, right in zip(records, records[1:])):
            raise NativeLayoutError(
                "bank0 layer %d target ranges are not contiguous" % layer)
        start = int(records[0]["local_offset"])
        end = int(records[-1]["end_offset_exclusive"])
        size = end - start
        if start % STAGING_PAGE_BYTES or size % STAGING_PAGE_BYTES:
            raise NativeLayoutError(
                "bank0 layer %d is not 4KiB-page aligned" % layer)
        staging_layers.append({
            "layer": layer,
            # bank0.bin begins with the Transformer prefix, so these are also
            # offsets within the PS-DDR staging allocation.
            "source_image_offset": start,
            "destination_local_offset": start,
            "bytes": size,
            "pages": size // STAGING_PAGE_BYTES,
            "target_ranges": [{
                "tensor": record["tensor"],
                "source_image_offset": int(record["local_offset"]),
                "destination_local_offset": int(record["local_offset"]),
                "bytes": int(record["bytes"]),
                "input_start": int(record["input_start"]),
                "input_features": int(record["input_features"]),
            } for record in records],
        })
    layer_sizes = {int(layer["bytes"]) for layer in staging_layers}
    if len(layer_sizes) != 1:
        raise NativeLayoutError("bank0 layer staging size is not fixed")
    transformer_pl_bytes = sum(int(layer["bytes"])
                               for layer in staging_layers)
    if (staging_layers[0]["source_image_offset"] != 0 or
            staging_layers[-1]["source_image_offset"] +
            staging_layers[-1]["bytes"] != transformer_pl_bytes):
        raise NativeLayoutError("bank0 Transformer staging prefix is not contiguous")
    weight_staging = {
        "controller": "BitNetPsToPlWeightStagingController",
        "copy_engine": "BitNetPsToPlWeightCopyEngine",
        "subsystem": "BitNetPsToPlWeightStagingSubsystem",
        "axi4_master_wrapper": "BitNetPsToPlWeightHubBridge",
        "source_file": "bank0.bin",
        "source_physical_memory": "PS DDR (one shared 4GiB controller)",
        "source_region_offset": 0,
        "source_region_bytes": transformer_pl_bytes,
        "destination_physical_memory": "PS DDR bank0 via HPC0 (256MiB window)",
        "destination_region_offset": 0,
        "destination_region_bytes": transformer_pl_bytes,
        "page_bytes": STAGING_PAGE_BYTES,
        "axi_data_bits": 128,
        "axi_beats_per_page": STAGING_PAGE_BYTES // 16,
        "axi_burst": "INCR, 256 beats, 16 bytes/beat",
        "ps_axi4_default_geometry": {
            "address_bits": 49, "data_bits": 128, "id_bits": 6,
            "mode": "read_only",
        },
        "pl_axi4_default_geometry": {
            "address_bits": 49, "data_bits": 128, "id_bits": 6,
            "mode": "write_only",
        },
        "page_completion_barrier": (
            "matching bank-0 B response accepted with BRESP=OKAY"
        ),
        "slot_count": 2,
        "layer_bytes": next(iter(layer_sizes)),
        "layer_pages": next(iter(layer_sizes)) // STAGING_PAGE_BYTES,
        "completion_tag_fields": ["epoch", "layer", "slot", "page"],
        "persistent_after_first_epoch": True,
        "same_epoch_decode_dma_bytes": 0,
        "copied_categories": ["transformer_weight_shard:bank0"],
        "not_copied_by_controller": [
            "banks1_to_4_transformer_shards (read in place from PS DDR)",
            "AUX/RoPE replicas", "LM head", "runtime KV", "activations"],
        "layers": staging_layers,
    }
    result = {
        "schema": SCHEMA,
        "source_manifest_profile": validated["manifest_profile"],
        "source_ternary_codebook": validated["ternary_codebook"],
        "source_gemv_ternary_map": validated["gemv_ternary_map"],
        "native_ternary_map": 0,
        # K striping is a physical transpose/scatter even when the ternary
        # codebook already uses map0.  It is performed once during image build,
        # never between inference layers or tokens.
        "startup_repack_required": True,
        "startup_repack_reasons": [
            "five_bank_k_striping",
            "gate_up_group_interleave",
        ] + (["ternary_codebook_to_map0"]
             if int(validated["gemv_ternary_map"]) != 0 else []),
        "model": {
            "layers": shape.layers,
            "hidden_size": shape.hidden_size,
            "ffn_size": shape.ffn_size,
            "vocab_size": shape.vocab_size,
            "kv_heads": KV_HEADS,
            "head_size": HEAD_SIZE,
            "max_sequence_length": legacy.SUPPORTED_ROPE_POSITIONS,
        },
        "architecture": {
            "matrix_engines": 1,
            "spus": 4,
            "token_parallelism": 2,
            "activation_storage": "PL BRAM, 2 token lanes x 4 feature shards",
            "score_probability_spill": False,
            "weight_partition": (
                "five logical K stripes: one bank-0 range plus four HP "
                "ranges sharing the single physical PS DDR controller"),
            "gate_up_policy": (
                "one physical target with alternating complete Gate/Up "
                "groups; PL retains one four-feature Gate window"),
            "lm_partition": "all rows K-striped; F32 scales row-sharded",
            "ideal_group_cycles": {
                "hidden_prefill_fill": striped_group_fill_cycles(
                    shape.hidden_size),
                "hidden_replay": compact_group_replay_cycles(
                    shape.hidden_size),
                "hidden_decode_two_group_fill": striped_group_fill_cycles(
                    shape.hidden_size, 2),
                "ffn_prefill_fill": striped_group_fill_cycles(
                    shape.ffn_size),
                "ffn_replay": compact_group_replay_cycles(shape.ffn_size),
                "ffn_decode_two_group_fill": striped_group_fill_cycles(
                    shape.ffn_size, 2),
            },
            "embedding_policy": "PS lookup then one workload input write",
        },
        "banks": banks,
        "weight_descriptors": weight_descriptors,
        "lm_descriptors": lm_descriptors,
        "kv_descriptors": kv_descriptors,
        "replicated_constants": replicated_constants,
        "weight_staging": weight_staging,
        "payload_totals": {
            "transformer_source_bytes": transformer_payload,
            "lm_head_source_bytes": lm_payload,
            "kv_data_physical_bytes": kv_data_per_bank * BANK_COUNT,
            "kv_scale_physical_bytes": kv_scale_per_bank * BANK_COUNT,
            "kv_packed_physical_bytes": kv_packed_per_bank * BANK_COUNT,
            "resident_physical_bytes": sum(int(bank["used_bytes"])
                                           for bank in banks),
        },
        "startup_transfer": {
            "bank0_transformer": (
                "eMMC -> one PS-DDR staging allocation; PL-owned 4KiB DMA "
                "copies one bank0 layer shard ahead into its final bank-0 "
                "resident address"),
            "bank0_other": (
                "AUX/RoPE/LM immutable ranges use the existing startup preload; "
                "runtime KV is initialized in bank-0 DDR and is not weight staging"),
            "banks1_to_4": (
                "eMMC -> reserved logical ranges in the same physical PS DDR; "
                "four HP ports read those K shards in place"),
            "runtime_policy": (
                "first epoch stages each PL shard once; same-epoch decode does "
                "zero weight DMA and no activation crosses PS/PL per layer"),
            "firmware_note": (
                "PS high-DDR ranges require reserved-memory before deployment; "
                "this planner does not change the current SD image"),
        },
    }
    return result


def build_layout(package: Path) -> Dict[str, Any]:
    package_root = legacy.locate_package_root(package)
    return build_from_validated(legacy.validate_package(package_root))


def write_json(layout: Dict[str, Any], output: Path, force: bool) -> None:
    target = Path(os.path.abspath(str(output.expanduser())))
    if target.exists() and not force:
        raise NativeLayoutError("output already exists; pass --force: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(str(temporary), str(target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan the packet-free BitNet five-bank image")
    parser.add_argument("package", type=Path,
                        help="BITNET directory or its parent")
    parser.add_argument("--output", type=Path,
                        default=Path("bitnet-five-bank.json"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        layout = build_layout(args.package)
        write_json(layout, args.output, args.force)
    except (legacy.PackageError, NativeLayoutError, OSError, ValueError) as exc:
        print("BITNET_LAYOUT_ERROR: %s" % exc)
        return 2
    print("BITNET_LAYOUT_OK banks=5 used=%d output=%s" % (
        layout["payload_totals"]["resident_physical_bytes"], args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
