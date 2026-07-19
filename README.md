# X-SE Speech Enhancement

X-SE is a speech enhancement research codebase centered on `FrozenExpertRouterGRPO`. It trains a lightweight router with GRPO to select or fuse frozen enhancement experts such as LiSenNet, FastEnhancer-S, and UL-UNAS.

The demo website：https://requirements-cattle-ctrl-likelihood.trycloudflare.com/
arXiv preprint are currently under preparation.

## Project Structure

```text
alpha/enh/system/grpo.py                       # core X-SE system
alpha/enh/models.py                            # only SpectralStatsRouter, LiSen, FastEnhancer, UL-UNAS
modules/blocks/                                # only lisen, fastenhancer, ulunas expert implementations
modules/                                       # minimal training framework, datasets, STFT, metrics
examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml
configs/example.yaml                           # public-safe copy of the core config
tools/export_grpo_onnx.py                      # export online MoE runtime assets
tools/export_moe_experts_onnx.py               # export expert ONNX assets
tools/DNSMOS/                                  # DNSMOS instructions and model placeholders
data/, checkpoints/, outputs/                  # placeholder directories, not real assets
```

## Installation

```bash
conda create -n X-SE python=3.10
conda activate X-SE
pip install -r requirements.txt
```

For CUDA training, install a PyTorch build that matches your driver first, then install the remaining packages. For ONNX GPU inference, replace `onnxruntime` with the matching `onnxruntime-gpu` build.

## Data And Model Assets

Place local assets in these paths:

```text
checkpoints/experts/lisennet.ckpt
checkpoints/experts/fastenhancer_s.ckpt
checkpoints/experts/ulunas.ckpt
tools/DNSMOS/DNSMOS/sig_bak_ovr.onnx
outputs/onnx/lisennet_fastenhancerS_ulunas_moe/manifest.json
outputs/onnx/lisennet_fastenhancerS_ulunas_moe/router_features.onnx
outputs/onnx/lisennet_fastenhancerS_ulunas_moe/experts/lisennet_expert.onnx
outputs/onnx/lisennet_fastenhancerS_ulunas_moe/experts/fastenhancer_s_expert.onnx
outputs/onnx/lisennet_fastenhancerS_ulunas_moe/experts/ulunas_expert.onnx
examples/voicebank/data/train/noisy.flac.hdf5
examples/voicebank/data/train/clean.flac.hdf5
examples/voicebank/data/test/test.csv
examples/voicebank/data/noise/noise.wav.hdf5
examples/voicebank/data/noise/noise.csv
examples/voicebank/data/rir/rir.wav.hdf5
examples/voicebank/data/rir/rir.csv
```

There are two different ONNX locations:

- DNSMOS scoring uses `tools/DNSMOS/DNSMOS/sig_bak_ovr.onnx`. This file is required when `valid_MOS`, `test_MOS`, or GRPO DNSMOS reward scoring is enabled.
- Online MoE inference uses the exported runtime directory under `outputs/onnx/lisennet_fastenhancerS_ulunas_moe/`. Keep `manifest.json`, `router_features.onnx`, and the `experts/` ONNX files together because the runtime loads them from the manifest path.

The checkpoint filenames above must match the `init` paths in the config. If your local checkpoint filenames are different, either rename them to the expected names or override the corresponding `router_grpo.experts.*.model.init` path on the command line.

## Training

The checked-in config preserves the original ONNX online branch settings. For a first run before exporting ONNX assets, use the PyTorch inference branch override:

```bash
python -m modules.launch \
  conf=examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml \
  cmd=train \
  router_grpo.inference_branch.runtime=pytorch
```

## Evaluation

```bash
python -m modules.launch \
  conf=examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml \
  cmd=test \
  ckpt=outputs/logs/LiSenNet_FastEnhancerS_ULUNAS_MoE/moe-router-grpo-1a/checkpoints/last.ckpt \
  router_grpo.inference_branch.runtime=pytorch
```

## Denoising

Create a wav list with one wav path per line or a CSV accepted by `modules.denoise`, then run:

```bash
python -m modules.launch \
  conf=examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml \
  cmd=denoise \
  denoise.wavlist=data/noisy.list \
  denoise.out_dir=outputs/enhanced \
  device=cpu \
  router_grpo.inference_branch.runtime=pytorch
```

## ONNX Export And Streaming Demo

If you already have exported ONNX assets, place the full directory at `outputs/onnx/lisennet_fastenhancerS_ulunas_moe/` and skip directly to the test command below. After expert checkpoints are available, you can also export ONNX streaming assets yourself:

```bash
python tools/export_grpo_onnx.py \
  --conf examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml \
  --out outputs/onnx/lisennet_fastenhancerS_ulunas_moe
```

Then test the exported runtime:

```bash
python tools/test_grpo_onnx_stream.py \
  --manifest outputs/onnx/lisennet_fastenhancerS_ulunas_moe/manifest.json \
  --input data/example_noisy.wav \
  --output outputs/example_enhanced.wav
```
