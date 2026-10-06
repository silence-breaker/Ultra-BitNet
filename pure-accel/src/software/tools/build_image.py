"""Build Map0 TP2 bank images from an exported Map0 or Map1 model package."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import model_package
import plan_five_bank
from image_copy import ImageBuildError, materialize_bank, COPY_CHUNK


def map1_to_map0(data):
    valid = bytes(n for n in range(256) if all(((n >> shift) & 3) != 3 for shift in (0, 2, 4, 6)))
    if data.translate(None, valid):
        raise ImageBuildError('Reserved Map1 ternary code 3 in Transformer weights')
    table = bytes(sum(((((n >> shift) & 3) + 1) & 3) << shift for shift in (0, 2, 4, 6)) for n in range(256))
    return data.translate(table)


def convert_transformer_weights(path, bank):
    # Only packed ternary Transformer bytes change; scales, LM, KV and padding do not.
    with path.open('r+b') as image:
        for obj in bank['objects']:
            if obj.get('category') != 'transformer_weight_shard':
                continue
            offset, remaining = obj['local_offset'], obj['bytes']
            while remaining:
                image.seek(offset)
                data = image.read(min(remaining, COPY_CHUNK))
                if not data:
                    raise ImageBuildError('Truncated Transformer image')
                image.seek(offset)
                image.write(map1_to_map0(data))
                offset += len(data)
                remaining -= len(data)
        image.flush()
        os.fsync(image.fileno())


def build_image(package, output):
    package = model_package.locate_package_root(Path(package)).resolve()
    layout = plan_five_bank.build_layout(package)
    if layout['source_gemv_ternary_map'] not in (0, 1):
        raise ImageBuildError('Unsupported source ternary map')
    output = Path(output).resolve()
    if output.exists():
        raise ImageBuildError('Output already exists; choose a new directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.bitnet-image-', dir=output.parent))
    try:
        for bank in layout['banks']:
            image = temporary / ('bank%d.bin' % bank['bank'])
            materialize_bank(package, bank, image)
            if layout['source_gemv_ternary_map'] == 1:
                convert_transformer_weights(image, bank)
        (temporary / 'layout.json').write_text(json.dumps(layout, indent=2) + '\n', encoding='utf-8')
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        layout = build_image(args.package, args.output)
        print(json.dumps({'banks': len(layout['banks']), 'output': str(args.output)}))
    except (OSError, ValueError, ImageBuildError, model_package.PackageError, plan_five_bank.NativeLayoutError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
