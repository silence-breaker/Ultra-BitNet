"""Build TP2 bank images from an exported Map0 model package."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import model_package
import plan_five_bank
from image_copy import ImageBuildError, materialize_bank


def build_image(package, output):
    package = model_package.locate_package_root(Path(package)).resolve()
    layout = plan_five_bank.build_layout(package)
    if layout['source_gemv_ternary_map'] != 0:
        raise ImageBuildError('The released image requires Map0 weights; re-export using codes 1=-1, 2=0, 3=1')
    output = Path(output).resolve()
    if output.exists():
        raise ImageBuildError('Output already exists; choose a new directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.bitnet-image-', dir=output.parent))
    try:
        for bank in layout['banks']:
            materialize_bank(package, bank, temporary / ('bank%d.bin' % bank['bank']))
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
