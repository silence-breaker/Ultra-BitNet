# Model Manifest

Model weights are not committed to the source repository. Fill in the following fields for each model release:

```text
name: <model name>
version: <version>
source: <upstream URL or release ID>
license: <model license>
sha256: <sha256>
layout: BITNET/L00..L29 + tokenizer + RoPE + AUX
converter: tools/export_layer_bundle.py
```

Distribute models through a GitHub Release or Git LFS. The release notes must include the board model, required DDR capacity, and conversion command.
