"""Load five prepacked images and the staging prefix into reserved PS DDR."""
import argparse
from contextlib import contextmanager
import mmap
import os
from pathlib import Path
import struct

BASES = (0x850000000, 0x800000000, 0x810000000, 0x820000000, 0x830000000)
SIZES = (204833664, 204833664, 204833664, 204833664, 202376064)
STAGE_BASE, STAGE_BYTES = 0x840000000, 104693760
CHUNK = 4 * 1024 * 1024


@contextmanager
def validated_files(directory):
    files = []
    try:
        for bank, expected in enumerate(SIZES):
            source = (Path(directory) / ('bank%d.bin' % bank)).open('rb')
            files.append(source)
            if os.fstat(source.fileno()).st_size != expected:
                raise ValueError('bank%d.bin must contain %d bytes' % (bank, expected))
        yield files
    finally:
        for source in files:
            source.close()


def transfer(files, write):
    """write(address, data) must complete the physical write before returning."""
    for bank, source in enumerate(files):
        source.seek(0)
        remaining, address = SIZES[bank], BASES[bank]
        while remaining:
            data = source.read(min(CHUNK, remaining))
            if not data:
                raise ValueError('Image truncated during loading')
            write(address, data)
            address += len(data)
            remaining -= len(data)
    files[0].seek(0)
    remaining, address = STAGE_BYTES, STAGE_BASE
    while remaining:
        data = files[0].read(min(CHUNK, remaining))
        if not data:
            raise ValueError('Staging source truncated during loading')
        write(address, data)
        address += len(data)
        remaining -= len(data)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', type=Path)
    parser.add_argument('--uio', default='/dev/uio0', help='Same control UIO device used by bitnetctl')
    parser.add_argument('--memory-device', default='/dev/mem')
    args = parser.parse_args()
    try:
        # Validate every file before mapping memory or writing any image bytes.
        with validated_files(args.images) as files, open(args.uio, 'r+b', buffering=0) as control:
            fcntl.flock(control.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mmap.mmap(control.fileno(), mmap.PAGESIZE) as registers:
                identity, = struct.unpack_from('<I', registers, 0)
                status, = struct.unpack_from('<I', registers, 0x34)
                pio, = struct.unpack_from('<I', registers, 0xa8)
                if identity != 0x48425432 or status != 0 or (pio & 3):
                    raise ValueError('Load images only after PL reset, with a compatible idle device and no session')
                fd = os.open(args.memory_device, os.O_RDWR | os.O_SYNC)
                try:
                    def write(address, data):
                        aligned = address - address % mmap.PAGESIZE
                        delta = address - aligned
                        with mmap.mmap(fd, delta + len(data), offset=aligned) as window:
                            window[delta:delta + len(data)] = data
                    transfer(files, write)
                finally:
                    os.close(fd)
        print('Loaded five banks and the bank-0 Transformer staging prefix.')
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
