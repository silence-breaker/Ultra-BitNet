#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import plan_five_bank as native  # noqa: E402
import model_package as legacy  # noqa: E402
import load_banks


def synthetic_validated(gemv_map: int = 0):
    shape = legacy.ModelShape(0, 30, 2560, 640, 6912)
    weights = []
    for layer in range(shape.layers):
        for filename, size in legacy.expected_weight_sizes(shape):
            weights.append({
                "relative_path": "L%02d/%s" % (layer, filename),
                "bytes": size,
                "category": "transformer_weight",
            })
    lm = []
    rows_per_shard = 8192
    for shard in range(16):
        row_start = shard * rows_per_shard
        rows = min(rows_per_shard, shape.vocab_size - row_start)
        lm.append({
            "relative_path": "LMH/H%02d.BIN" % shard,
            "row_start": row_start,
            "rows": rows,
            "bytes": rows * (shape.hidden_size + 4),
            "weight_offset_bytes": 0,
            "weight_bytes": rows * shape.hidden_size,
            "scale_offset_bytes": rows * shape.hidden_size,
            "scale_bytes": rows * 4,
        })
    return {
        "shape": shape,
        "manifest_profile": "legacy-v0-codes-123" if gemv_map == 0 else "current-v1",
        "ternary_codebook": {"1": -1, "2": 0, "3": 1},
        "gemv_ternary_map": gemv_map,
        "weights": weights,
        "lm_head": lm,
        "aux": {"relative_path": "AUX/AUX.BIN", "bytes": 881124},
        "rope": {"relative_path": "AUX/ROPE.BIN", "bytes": 1048624},
    }


class BitNetFiveBankPlannerTest(unittest.TestCase):
    def test_loader_matches_production_layout(self):
        layout = native.build_from_validated(synthetic_validated())
        self.assertEqual(tuple(bank['used_bytes'] for bank in layout['banks']), load_banks.SIZES)
        self.assertEqual(tuple(bank['preload_axi_base'] for bank in layout['banks']), load_banks.BASES)
        self.assertEqual(layout['weight_staging']['source_region_bytes'], load_banks.STAGE_BYTES)

    def test_production_payload_is_balanced_and_within_every_bank(self):
        layout = native.build_from_validated(synthetic_validated())
        self.assertEqual(layout["schema"], native.SCHEMA)
        self.assertEqual(len(layout["banks"]), 5)
        self.assertTrue(layout["startup_repack_required"])
        self.assertEqual(layout["startup_repack_reasons"],
                         ["five_bank_k_striping",
                          "gate_up_group_interleave"])

        totals = layout["payload_totals"]
        self.assertEqual(totals["transformer_source_bytes"], 521011200)
        self.assertEqual(totals["lm_head_source_bytes"], 328848384)
        self.assertEqual(totals["kv_data_physical_bytes"], 157286400)
        self.assertEqual(totals["kv_scale_physical_bytes"], 4915200)
        self.assertEqual(totals["kv_packed_physical_bytes"], 162201600)
        self.assertEqual(
            totals["kv_packed_physical_bytes"],
            totals["kv_data_physical_bytes"] +
            totals["kv_scale_physical_bytes"])

        for bank in layout["banks"]:
            self.assertLess(bank["used_bytes"], bank["capacity_bytes"])
            self.assertGreater(bank["free_bytes"], 0)
            previous = 0
            for obj in bank["objects"]:
                self.assertEqual(obj["local_offset"] % native.ALIGNMENT, 0)
                self.assertGreaterEqual(obj["local_offset"], previous)
                previous = obj["end_offset_exclusive"]

        self.assertEqual(layout["banks"][0]["preload_axi_base"], 0x850000000)
        self.assertEqual(layout["banks"][4]["core_axi_base"], 0x830000000)

    def test_transformer_weights_are_k_striped_at_128_feature_boundaries(self):
        layout = native.build_from_validated(synthetic_validated())
        self.assertTrue(all(
            item["packed_block_bytes"] == 528 and
            item["scale_channel_bytes"] == 16 and
            item["scale_pack_num"] == 4
            for item in layout["kv_descriptors"]))
        k = [item for item in layout["weight_descriptors"]
             if item["layer"] == 0 and item["tensor"] == "k"]
        v = [item for item in layout["weight_descriptors"]
             if item["layer"] == 0 and item["tensor"] == "v"]
        self.assertEqual(len(k), 5)
        self.assertEqual(len(v), 5)
        self.assertEqual([item["input_start"] for item in k],
                         [0, 512, 1024, 1536, 2048])
        self.assertTrue(all(item["input_features"] == 512 for item in k + v))
        self.assertTrue(all(item["output_start"] == 0 for item in k + v))
        self.assertTrue(all(item["output_features"] == 640 for item in k + v))
        self.assertTrue(all(item["bytes"] == 81920 for item in k + v))

        down = [item for item in layout["weight_descriptors"]
                if item["layer"] == 0 and item["tensor"] == "down"]
        self.assertEqual([item["input_features"] for item in down],
                         [1408, 1408, 1408, 1408, 1280])
        self.assertEqual(sum(item["bytes"] for item in down),
                         2560 * 6912 // 4)

        gate_up = [item for item in layout["weight_descriptors"]
                   if item["layer"] == 0 and item["tensor"] == "gate_up"]
        self.assertEqual(len(gate_up), 5)
        self.assertTrue(all(item["output_features"] == 2 * 6912 and
                            item["semantic_output_features"] == 6912 and
                            item["packed_output_groups"] == 6912 // 2 and
                            item["physical_group_order"] ==
                            "gate[4n:4n+4],up[4n:4n+4]"
                            for item in gate_up))
        self.assertEqual(sum(item["bytes"] for item in gate_up),
                         2 * 6912 * 2560 // 4)
        for item in gate_up:
            source = item["source_interleaved"]
            self.assertTrue(source["gate_relative_path"].endswith(
                "/GATE.BIN"))
            self.assertTrue(source["up_relative_path"].endswith("/UP.BIN"))
            self.assertEqual(source["destination_stride"],
                             2 * item["input_features"])
            self.assertEqual(source["up_destination_offset"],
                             item["input_features"])

        cycles = layout["architecture"]["ideal_group_cycles"]
        self.assertEqual(cycles, {
            "hidden_prefill_fill": 32,
            "hidden_replay": 40,
            "hidden_decode_two_group_fill": 64,
            "ffn_prefill_fill": 88,
            "ffn_replay": 108,
            "ffn_decode_two_group_fill": 176,
        })
        self.assertLessEqual(cycles["hidden_prefill_fill"],
                             cycles["hidden_replay"])
        self.assertLessEqual(cycles["ffn_prefill_fill"],
                             cycles["ffn_replay"])

    def test_lm_weights_use_all_banks_and_scales_cover_every_row_once(self):
        layout = native.build_from_validated(synthetic_validated())
        descriptors = layout["lm_descriptors"]
        self.assertTrue(all(item["row_start"] == 0 and
                            item["rows"] == 128256 for item in descriptors))
        self.assertEqual([item["input_start"] for item in descriptors],
                         [0, 512, 1024, 1536, 2048])
        self.assertTrue(all(item["input_features"] == 512
                            for item in descriptors))
        self.assertEqual(sum(item["scale_rows"] for item in descriptors),
                         128256)

        for bank in layout["banks"]:
            weights = next(obj for obj in bank["objects"]
                           if obj["category"] == "lm_head_i8_k_stripe")
            scales = next(obj for obj in bank["objects"]
                          if obj["category"] == "lm_head_f32_scale_shard")
            self.assertEqual(sum(piece["rows"] * piece["bytes_per_row"]
                                 for piece in weights["source_stripes"]),
                             weights["bytes"])
            self.assertEqual(sum(piece["bytes"]
                                 for piece in scales["source_pieces"]),
                             scales["bytes"])

    def test_current_codebook_requires_startup_repack(self):
        layout = native.build_from_validated(synthetic_validated(gemv_map=1))
        self.assertTrue(layout["startup_repack_required"])
        self.assertEqual(layout["native_ternary_map"], 0)
        self.assertEqual(layout["startup_repack_reasons"],
                         ["five_bank_k_striping",
                          "gate_up_group_interleave",
                          "ternary_codebook_to_map0"])

    def test_only_the_pl_shard_is_page_staged_and_then_remains_resident(self):
        layout = native.build_from_validated(synthetic_validated())
        staging = layout["weight_staging"]
        self.assertEqual(staging["controller"],
                         "BitNetPsToPlWeightStagingController")
        self.assertEqual(staging["copy_engine"],
                         "BitNetPsToPlWeightCopyEngine")
        self.assertEqual(staging["subsystem"],
                         "BitNetPsToPlWeightStagingSubsystem")
        self.assertEqual(staging["axi4_master_wrapper"],
                         "BitNetPsToPlWeightHubBridge")
        self.assertEqual(staging["source_file"], "bank0.bin")
        self.assertEqual(staging["page_bytes"], 4096)
        self.assertEqual(staging["axi_data_bits"], 128)
        self.assertEqual(staging["axi_beats_per_page"], 256)
        self.assertEqual(staging["ps_axi4_default_geometry"], {
            "address_bits": 49, "data_bits": 128, "id_bits": 6,
            "mode": "read_only",
        })
        self.assertEqual(staging["pl_axi4_default_geometry"], {
            "address_bits": 49, "data_bits": 128, "id_bits": 6,
            "mode": "write_only",
        })
        self.assertIn("BRESP=OKAY", staging["page_completion_barrier"])
        self.assertEqual(staging["slot_count"], 2)
        self.assertEqual(staging["layer_bytes"], 3489792)
        self.assertEqual(staging["layer_pages"], 852)
        self.assertEqual(staging["source_region_bytes"], 104693760)
        self.assertEqual(staging["destination_region_bytes"], 104693760)
        self.assertTrue(staging["persistent_after_first_epoch"])
        self.assertEqual(staging["same_epoch_decode_dma_bytes"], 0)
        self.assertEqual(len(staging["layers"]), 30)

        previous_end = 0
        expected_target_bytes = [327680, 81920, 81920, 327680,
                                 1769472, 901120]
        for layer_index, layer in enumerate(staging["layers"]):
            self.assertEqual(layer["layer"], layer_index)
            self.assertEqual(layer["source_image_offset"], previous_end)
            self.assertEqual(layer["destination_local_offset"], previous_end)
            self.assertEqual(layer["bytes"], staging["layer_bytes"])
            self.assertEqual(layer["pages"], staging["layer_pages"])
            self.assertEqual([target["bytes"]
                              for target in layer["target_ranges"]],
                             expected_target_bytes)
            # Only logical bank0's K slice moves.  Q/K/V/O/GateUp own
            # features [0,512); Down owns [0,1408).
            self.assertEqual([target["input_features"]
                              for target in layer["target_ranges"]],
                             [512, 512, 512, 512, 512, 1408])
            previous_end += layer["bytes"]
        self.assertEqual(previous_end, staging["source_region_bytes"])
        self.assertIn("banks1_to_4_transformer_shards",
                      staging["not_copied_by_controller"][0])

    def test_output_is_deterministic(self):
        first = json.dumps(native.build_from_validated(synthetic_validated()),
                           sort_keys=True)
        second = json.dumps(native.build_from_validated(synthetic_validated()),
                            sort_keys=True)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
