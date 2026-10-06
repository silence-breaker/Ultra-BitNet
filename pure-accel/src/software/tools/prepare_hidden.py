"""Encode one or two embedding vectors as lane-major little-endian Q16.16."""
import argparse
import json
import math
from pathlib import Path
import struct


def q16(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Embedding values must be finite')
    # Nearest with ties away from zero, saturating to signed 32-bit Q16.16.
    scaled = min(2147483647.0, max(-2147483648.0, value * 65536.0))
    return math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)


def encode(vectors):
    if len(vectors) not in (1, 2) or any(len(lane) != 2560 for lane in vectors):
        raise ValueError('Expected one or two vectors of 2560 features')
    return b''.join(struct.pack('<i', q16(value)) for lane in vectors for value in lane)


def embedding_vectors(package, tokens):
    if not 1 <= len(tokens) <= 2 or any(not 0 <= token < 128256 for token in tokens):
        raise ValueError('Expected one or two token IDs in [0, 128256)')
    vectors = []
    for token in tokens:
        shard, row = divmod(token, 8192)
        with (Path(package) / 'EMB' / ('E%02d.BIN' % shard)).open('rb') as source:
            source.seek(row * 2560 * 2)
            data = source.read(2560 * 2)
        if len(data) != 5120:
            raise ValueError('Embedding shard ended inside a row')
        vectors.append([struct.unpack('<f', struct.pack('<I', word << 16))[0]
                        for (word,) in struct.iter_unpack('<H', data)])
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--json', type=Path, help='JSON array of 1 or 2 embedding vectors')
    inputs.add_argument('--package', type=Path, help='Exported package containing BF16 EMB shards')
    parser.add_argument('--tokens', nargs='+', type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if (args.package is not None) != (args.tokens is not None):
        parser.error('--tokens is required only with --package')
    try:
        vectors = embedding_vectors(args.package, args.tokens) if args.package else json.loads(args.json.read_text())
        data = encode(vectors)
        with args.output.open('xb') as output:
            output.write(data)
        print(json.dumps({'tokens': len(vectors), 'bytes': len(data), 'output': str(args.output)}))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
