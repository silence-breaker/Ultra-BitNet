"""Export a local packed Hugging Face model and build pure-accel bank images."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from build_image import build_image


def export_model(model, output):
    model, output = Path(model).resolve(), Path(output).resolve()
    for name in ('model.safetensors', 'config.json', 'tokenizer.json'):
        if not (model / name).is_file():
            raise ValueError('Missing model input: ' + str(model / name))
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    exporter = Path(__file__).resolve().parents[4] / 'mix-accel/src/tools/export_emmc_package.py'
    if not exporter.is_file():
        raise ValueError('Keep pure-accel and mix-accel together in the public checkout')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.bitnet-export-', dir=output.parent))
    try:
        with (temporary / 'export.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(exporter), '--model', str(model / 'model.safetensors'),
                            '--out-dir', str(temporary / 'package'), '--layers', '30'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        build_image(temporary / 'package/BITNET', temporary / 'banks')
        os.rename(temporary, output)
    except BaseException:
        # Preserve failed output and its log for diagnosis; never touch the source model.
        print('Incomplete export retained at ' + str(temporary), file=sys.stderr)
        raise
    print('Model package: ' + str(output / 'package/BITNET'))
    print('Map0 bank images: ' + str(output / 'banks'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        export_model(args.model_dir, args.output)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
