# Model Manifest

模型权重不提交到源码仓库。发布模型时请填写以下字段：

```text
name: <model name>
version: <version>
source: <upstream URL or release ID>
license: <model license>
sha256: <sha256>
layout: BITNET/L00..L29 + tokenizer + RoPE + AUX
converter: tools/export_layer_bundle.py
```

推荐通过 GitHub Release 或 Git LFS 分发，并在 Release 说明中提供板卡型号、所需 DDR 容量和转换命令。
