import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / 'tools'))
import build_image
import image_copy
import load_banks
import prepare_hidden


class AbiTests(unittest.TestCase):
    def test_every_register_matches_scala(self):
        scala = (SOFTWARE.parent / 'main/scala/ultrabitnet/accel/BitNetResidentBoardAccelerator.scala').read_text()
        registers = scala.split('object BitNetResidentBoardRegister {')[1].split('val IdentityValue')[0]
        header = (SOFTWARE / 'include/bitnet.h').read_text()
        c = {name: int(value, 16) for name, value in re.findall(r'BN_(\w+)\s*=\s*(0x[0-9a-f]+)', header)}
        expected = {re.sub(r'(?<!^)(?=[A-Z])', '_', name).upper(): int(value, 16)
                    for name, value in re.findall(r'val (\w+) = (0x[0-9a-f]+)', registers)}
        self.assertEqual(c, expected)
        self.assertIn('BigInt("48425432", 16)', scala)


class InputTests(unittest.TestCase):
    def test_lane_major_signed_encoding(self):
        values = [[1.0, -1.0, 0.5, -0.5] * 640, [2.0] * 2560]
        raw = prepare_hidden.encode(values)
        self.assertEqual(len(raw), 20480)
        self.assertEqual(struct.unpack_from('<4i', raw), (65536, -65536, 32768, -32768))
        self.assertEqual(struct.unpack_from('<i', raw, 10240), (131072,))

    def test_rounding_saturation_and_invalid(self):
        self.assertEqual(prepare_hidden.q16(0.5 / 65536), 1)
        self.assertEqual(prepare_hidden.q16(-0.5 / 65536), -1)
        self.assertEqual(prepare_hidden.q16(1e30), 2147483647)
        self.assertEqual(prepare_hidden.q16(-1e30), -2147483648)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError): prepare_hidden.q16(value)
        with self.assertRaises(ValueError): prepare_hidden.encode([[0] * 2559])

    def test_bf16_embedding_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'EMB').mkdir()
            (root / 'EMB/E00.BIN').write_bytes(struct.pack('<H', 0x3f80) * 2560)
            vectors = prepare_hidden.embedding_vectors(root, [0])
            self.assertEqual(vectors, [[1.0] * 2560])
            with self.assertRaises(ValueError): prepare_hidden.embedding_vectors(root, [1])


class ImageTests(unittest.TestCase):
    def test_image_pipeline_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'data.bin').write_bytes(bytes(range(16)))
            bank = {'bank': 0, 'used_bytes': 16, 'capacity_bytes': 32, 'objects': [
                {'name': 'weights', 'local_offset': 0, 'end_offset_exclusive': 16, 'bytes': 16,
                 'source': {'relative_path': 'data.bin', 'source_offset': 0, 'destination_offset': 0, 'bytes': 16}}]}
            layout = {'source_gemv_ternary_map': 0, 'banks': [bank]}
            with mock.patch.object(build_image.model_package, 'locate_package_root', return_value=root), \
                 mock.patch.object(build_image.plan_five_bank, 'build_layout', return_value=layout):
                build_image.build_image(root, root / 'output')
                self.assertEqual((root / 'output/bank0.bin').read_bytes(), bytes(range(16)))
                self.assertEqual(json.loads((root / 'output/layout.json').read_text()), layout)
                with self.assertRaises(image_copy.ImageBuildError):
                    build_image.build_image(root, root / 'output')
                (root / 'data.bin').write_bytes(b'bad')
                with self.assertRaises(image_copy.ImageBuildError):
                    build_image.build_image(root, root / 'failed')
                self.assertFalse((root / 'failed').exists())
                self.assertEqual(list(root.glob('.bitnet-image-*')), [])

    def test_scatter_and_truncated_source(self):
        source = io.BytesIO(bytes(range(64))); output = io.BytesIO(bytes(32))
        image_copy._copy_strided(source, output, 2, 5, 3, 8, 2, 3)
        self.assertEqual(output.getvalue()[5:7], bytes([2, 3]))
        self.assertEqual(output.getvalue()[8:10], bytes([10, 11]))
        with self.assertRaises(image_copy.ImageBuildError):
            image_copy._copy_range(source, output, 63, 0, 2)

    def test_paths_cannot_escape_package(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(image_copy.ImageBuildError):
                image_copy._checked_source(Path(directory), '../outside.bin')

    def test_reject_unknown_map_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(build_image.model_package, 'locate_package_root', return_value=Path(directory)), \
             mock.patch.object(build_image.plan_five_bank, 'build_layout', return_value={'source_gemv_ternary_map': 2}):
            with self.assertRaises(image_copy.ImageBuildError):
                build_image.build_image(directory, Path(directory) / 'output')
            self.assertFalse((Path(directory) / 'output').exists())

    def test_map_conversion_exhaustive(self):
        for value in range(256):
            codes = [(value >> shift) & 3 for shift in (0, 2, 4, 6)]
            if 3 in codes:
                with self.assertRaises(image_copy.ImageBuildError):
                    build_image.map1_to_map0(bytes([value]))
            else:
                converted = build_image.map1_to_map0(bytes([value]))[0]
                self.assertEqual([(converted >> shift) & 3 for shift in (0, 2, 4, 6)], [code + 1 for code in codes])

    def test_conversion_preserves_non_transformer_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bank.bin'
            path.write_bytes(bytes([0, 0x55, 0xaa, 0xff]))
            bank = {'objects': [{'category': 'transformer_weight_shard', 'local_offset': 0, 'bytes': 3},
                                {'category': 'lm_weight', 'local_offset': 3, 'bytes': 1}]}
            build_image.convert_transformer_weights(path, bank)
            self.assertEqual(path.read_bytes(), bytes([0x55, 0xaa, 0xff, 0xff]))

    def test_materialize_small_image_and_zero_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'data.bin').write_bytes(bytes(range(16)))
            bank = {'bank': 0, 'used_bytes': 32, 'capacity_bytes': 64, 'objects': [
                {'name': 'weights', 'local_offset': 0, 'end_offset_exclusive': 16, 'bytes': 16,
                 'source': {'relative_path': 'data.bin', 'source_offset': 0, 'destination_offset': 0, 'bytes': 16}},
                {'name': 'kv', 'local_offset': 16, 'end_offset_exclusive': 32, 'bytes': 16, 'zero_initialize': True}]}
            image_copy.materialize_bank(root, bank, root / 'bank.bin')
            self.assertEqual((root / 'bank.bin').read_bytes(), bytes(range(16)) + bytes(16))

    def test_loader_writes_exact_banks_and_stage_prefix(self):
        buffers = [io.BytesIO(bytes([bank]) * 16) for bank in range(5)]
        writes = []
        with mock.patch.object(load_banks, 'SIZES', (16,) * 5), mock.patch.object(load_banks, 'STAGE_BYTES', 8):
            load_banks.transfer(buffers, lambda address, data: writes.append((address, data)))
        self.assertEqual([a for a, _ in writes], list(load_banks.BASES) + [load_banks.STAGE_BASE])
        self.assertEqual(writes[-1][1], bytes(8))

    def test_loader_rejects_bad_sizes_before_yielding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for bank in range(5): (root / ('bank%d.bin' % bank)).write_bytes(bytes(15 if bank == 4 else 16))
            with mock.patch.object(load_banks, 'SIZES', (16,) * 5), self.assertRaises(ValueError):
                with load_banks.validated_files(root): self.fail('Invalid files were accepted')


@unittest.skipUnless(os.environ.get('BITNETCTL'), 'Linux CLI supplied by make test')
class LinuxCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.registers = self.root / 'registers.bin'
        memory = bytearray(4096); struct.pack_into('<I', memory, 0, 0x48425432)
        self.registers.write_bytes(memory)

    def tearDown(self): self.temp.cleanup()

    def run_cli(self, command, *options):
        return subprocess.run([os.environ['BITNETCTL'], command, '--mem', str(self.registers), '--base', '0', *options],
                              capture_output=True, text=True, timeout=10)

    def test_probe_and_status_do_not_write(self):
        before = self.registers.read_bytes()
        probe = self.run_cli('probe'); self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertTrue(json.loads(probe.stdout)['compatible'])
        status = self.run_cli('status'); self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)['status'], 0)
        self.assertEqual(self.registers.read_bytes(), before)

    def test_wrong_identity_and_truncated_mapping(self):
        self.registers.write_bytes(bytes(4096))
        self.assertNotEqual(self.run_cli('probe').returncode, 0)
        self.registers.write_bytes(bytes(16))
        self.assertNotEqual(self.run_cli('probe').returncode, 0)

    def test_invalid_round_never_submits(self):
        path = self.root / 'hidden.bin'; path.write_bytes(bytes(10240))
        before = self.registers.read_bytes()
        for options in [('--mode', 'bad'), ('--epoch', '256'), ('--source-offset', '0x840000000'), ('--tokens', '3')]:
            result = self.run_cli('round', '--hidden', str(path), '--new-session', *options)
            self.assertEqual(result.returncode, 2, result.stderr)
        path.write_bytes(bytes(4))
        self.assertEqual(self.run_cli('round', '--hidden', str(path)).returncode, 2)
        self.assertEqual(self.registers.read_bytes(), before)

    def test_timeout_is_reported_without_ack(self):
        path = self.root / 'hidden.bin'; path.write_bytes(bytes(10240))
        result = self.run_cli('round', '--hidden', str(path), '--new-session', '--timeout-ms', '1')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['rc'], -4)
        self.assertEqual(struct.unpack_from('<I', self.registers.read_bytes(), 8)[0], 0)

    def test_device_lock_excludes_second_client(self):
        import fcntl
        with self.registers.open('r+b') as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertNotEqual(self.run_cli('probe').returncode, 0)


if __name__ == '__main__': unittest.main()
