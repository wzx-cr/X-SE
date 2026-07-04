Put the exported GRPO MoE ONNX files here if you want to run without setting MOE_MANIFEST.

Expected layout:

```text
models/
  manifest.json
  router_features.onnx
  experts/
    lisennet_expert.onnx
    fastenhancer_s_expert.onnx
    ulunas_expert.onnx
```

Alternatively, keep the ONNX export directory elsewhere and start with:

```bash
MOE_MANIFEST=/absolute/path/to/manifest.json bash run_server.sh
```

