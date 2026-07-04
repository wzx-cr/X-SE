# MoE-GRPO Speech Enhancement

MoE-GRPO is a speech enhancement research codebase centered on `FrozenExpertRouterGRPO`. It trains a lightweight router with GRPO to select or fuse frozen enhancement experts such as LiSenNet, FastEnhancer-S, and UL-UNAS.

This public package keeps only the core training/inference code, GRPO configuration, ONNX export helpers, and placeholder directories. Datasets, checkpoints, generated ONNX files, logs, and experiment outputs are intentionally excluded.

## Project Structure

```text
alpha/enh/system/grpo.py                       # core MoE-GRPO system
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
conda create -n moe-grpo python=3.10
conda activate moe-grpo
pip install -r requirements.txt
```

For CUDA training, install a PyTorch build that matches your driver first, then install the remaining packages. For ONNX GPU inference, replace `onnxruntime` with the matching `onnxruntime-gpu` build.

## Data And Model Assets

Do not commit real data or weights. Place local assets in these paths:

```text
checkpoints/experts/lisennet.ckpt
checkpoints/experts/fastenhancer_s.ckpt
checkpoints/experts/ulunas.ckpt
tools/DNSMOS/DNSMOS/sig_bak_ovr.onnx
examples/voicebank/data/train/noisy.flac.hdf5
examples/voicebank/data/train/clean.flac.hdf5
examples/voicebank/data/test/test.csv
examples/voicebank/data/noise/noise.wav.hdf5
examples/voicebank/data/noise/noise.csv
examples/voicebank/data/rir/rir.wav.hdf5
examples/voicebank/data/rir/rir.csv
```

The DNSMOS ONNX model is required when `valid_MOS`, `test_MOS`, or GRPO DNSMOS reward scoring is enabled. Keep it local under `tools/DNSMOS/DNSMOS/`.

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

After expert checkpoints are available, export ONNX streaming assets:

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

## Notes

- The public repository intentionally excludes datasets, checkpoints, ONNX files, logs, result tables, and generated media.
- `KALDI_ROOT`, proxy settings, and notification tokens must be provided through environment variables if needed.
- No project license has been selected in this package. Choose a license before publishing.
