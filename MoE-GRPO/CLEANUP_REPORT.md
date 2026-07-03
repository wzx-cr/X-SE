# Cleanup Report

## Scope

Core anchors:

- `alpha/enh/system/grpo.py`
- `examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml`

The release package was built as a clean copy under `github_release/MoE-GRPO`. The original project tree was not deleted or moved.

## Must Keep

- `alpha/enh/system/grpo.py`: MoE-GRPO router, online adaptation, streaming runtime.
- `alpha/enh/system/grpo_onnx_iobinding.py`: optional ONNX Runtime I/O binding path.
- `alpha/enh/system/system.py`: minimal `BaseSE` / `UniSE` compatibility layer used by GRPO.
- `alpha/enh/models.py`: only `SpectralStatsRouter`, LiSenNet, FastEnhancer, and UL-UNAS wrappers.
- `alpha/enh/loss_module.py`: placeholder for optional custom losses; the example uses `torch.nn.L1Loss`.
- `modules/`: training framework, dataset readers, STFT, metrics, and schedulers.
- `modules/blocks/`: only `lisen.py`, `fastenhancer.py`, `fastenhancer_core.py`, and `ulunas.py`.
- `examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml`: canonical MoE-GRPO config.
- `tools/export_grpo_onnx.py`, `tools/export_moe_experts_onnx.py`, ONNX check/test helpers.

## Recommended Keep

- `configs/example.yaml`: public-safe copy of the core config.
- `tools/prepare_data.py`, `tools/eval_wavlist.py`: lightweight reproducibility utilities.
- `tools/DNSMOS/README.md` and `.gitkeep` placeholders.
- `data/.gitkeep`, `checkpoints/.gitkeep`, `outputs/.gitkeep`, `exp/.gitkeep`.
- `README.md`, `requirements.txt`, `.gitignore`, `LICENSE_OPTIONS.md`.

## Remove Or Ignore

- `.venv-web/`: local virtual environment.
- `.vscode/`: local IDE settings.
- `.env`: local environment file.
- `examples/SpeechMOS/exp/`: checkpoints, TensorBoard events, logs, result JSON files.
- `examples/*/data/`: datasets and generated HDF5 stores.
- `examples/voicebank/pretrained/*.pt`: pretrained weights.
- `examples/voicebank/local/*.ipynb`, `.ipynb_checkpoints/`: temporary notebook experiments.
- `examples/voicebank/reverbe.csv`: large path table containing local dataset paths.
- `*.ckpt`, `*.pt`, `*.pth`, `*.tar`, `*.onnx`: model weights and exported runtimes.
- `*.wav`, `*.flac`, generated images, result tables, archives.
- `__pycache__/`, `*.pyc`: Python bytecode.
- `fastenhancer-main/`, `ul-unas-main/`: external source snapshots, not needed for the integrated runtime.
- `grpo_moe_web_demo/`, `streaming_se_web_demo/`, `realtime-*`, `stream-adaptive-demo/`: web demos outside the core GRPO training package.
- `grpo_trainer.py`, `_grpo_decompiled.py`, `1.yaml`, `tmp_noisy*`: unrelated or temporary files.
- `test/`: ad hoc local tests with hardcoded paths or GPU assumptions.

## Extra Model-Code Pruning

The release copy was further reduced after review:

- `alpha/enh/models.py` was rewritten to remove unrelated model classes.
- `alpha/enh/system/system.py` was rewritten to remove old enhancement system subclasses.
- `alpha/enh/system/system_GAN.py`, `cleanmel.py`, and `deepfilternet3.py` were removed.
- `modules/blocks/` was reduced to LiSenNet, FastEnhancer, and UL-UNAS only.
- `modules/wamlm/`, `modules/cleanmel/`, and `modules/loss/` were removed.
- `modules/model/arch.py` was reduced to an empty preset placeholder.
- Requirements were reduced by removing an unused vision dependency.

## Sensitive Information Review

Findings in the source tree:

- `.env` only contained a local `PYTHONPATH`; excluded from release.
- `modules/utils/common.py` had a hardcoded private proxy IP; replaced by `HTTP_PROXY` / `HTTPS_PROXY` environment lookup in the release.
- `modules/dataset/kaldi_io_cn.py` had a hardcoded Kaldi install path; replaced by optional `KALDI_ROOT`.
- `modules/blocks/fastenhancer_core.py` had a hardcoded external wav path in its self-test; replaced by `--wav` or synthetic audio.
- `examples/voicebank/run.sh`, local notebooks, and several demo folders contained personal absolute paths; excluded from release.
- `tools/dingtalk.py` used token/secret parameter names and is not core; removed from release.

No real API key, token, password, or private credential is included in the release package after the cleanup pass.

## Recommended GitHub Layout

```text
MoE-GRPO/
  README.md
  requirements.txt
  .gitignore
  LICENSE_OPTIONS.md
  CLEANUP_REPORT.md
  alpha/
  modules/
  configs/example.yaml
  examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml
  tools/
  data/.gitkeep
  checkpoints/.gitkeep
  outputs/.gitkeep
  exp/.gitkeep
```

## Commands To Clean The Original Tree After Review

Do not run these until the release zip has been reviewed:

```powershell
Remove-Item -Recurse -Force .venv-web, .vscode
Remove-Item -Force .env, 1.yaml, _grpo_decompiled.py, tmp_noisy.list
Remove-Item -Recurse -Force tmp_noisy, examples\SpeechMOS\exp
Remove-Item -Recurse -Force examples\voicebank\pretrained, examples\voicebank\local
Remove-Item -Force examples\voicebank\reverbe.csv
Remove-Item -Recurse -Force fastenhancer-main, ul-unas-main
Remove-Item -Recurse -Force grpo_moe_web_demo, streaming_se_web_demo, realtime-finetune-web, realtime-ulunas-web, stream-adaptive-demo
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -File | Where-Object { $_.Extension -in @('.ckpt','.pt','.pth','.tar','.onnx','.wav','.flac','.zip','.log') } | Remove-Item -Force
```
