"""Smoke-test a FrozenExpertRouterGRPO ONNX stream manifest.

Examples:

  # CPU smoke test with generated audio.
  python tools/test_grpo_onnx_stream.py \
    --manifest exp/onnx/lisennet_fastenhancerS_ulunas_moe/manifest.json \
    --providers CPUExecutionProvider \
    --seconds 2

  # Use provider/device assignments from manifest and write enhanced wav.
  python tools/test_grpo_onnx_stream.py \
    --manifest exp/onnx/lisennet_fastenhancerS_ulunas_moe/manifest.json \
    --input noisy.wav \
    --output enhanced_onnx.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path[:] = [
    item for item in sys.path
    if Path(item or ".").resolve() != TOOLS_DIR
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha.enh.system.grpo import OnnxMoEStreamRuntime  # noqa: E402


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if int(sr) != int(sample_rate):
        raise RuntimeError(f"Input sample rate is {sr}, expected {sample_rate}. Resample before testing.")
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def _write_audio(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), int(sample_rate))


def _generated_audio(sample_rate: int, seconds: float) -> np.ndarray:
    length = max(1, int(round(float(sample_rate) * float(seconds))))
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    wav = 0.05 * np.sin(2.0 * np.pi * 440.0 * t)
    wav += 0.005 * np.random.default_rng(0).standard_normal(length).astype(np.float32)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def _read_manifest_sample_rate(manifest_path: Path) -> int:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return int(manifest.get("sample_rate", 16000))


def _override_manifest_providers(manifest_path: Path, providers: list[str] | None) -> Path:
    if not providers:
        return manifest_path
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["providers"] = providers
    router = manifest.get("router")
    if isinstance(router, dict):
        router["providers"] = providers
        router.pop("provider_options", None)
    for expert in manifest.get("experts", []):
        if isinstance(expert, dict):
            expert["providers"] = providers
            expert.pop("provider_options", None)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".json",
        prefix=".manifest_provider_override_",
        dir=str(manifest_path.parent),
        delete=False,
    ) as f:
        json.dump(manifest, f, indent=2)
        return Path(f.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to GRPO ONNX manifest.json.")
    parser.add_argument("--input", default=None, help="Optional input wav. If omitted, generated audio is used.")
    parser.add_argument("--output", default=None, help="Optional enhanced wav output path.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Generated audio length when --input is omitted.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="Override all manifest providers, e.g. CPUExecutionProvider or CUDAExecutionProvider CPUExecutionProvider.",
    )
    parser.add_argument("--workers", type=int, default=3, help="Parallel expert workers.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    sample_rate = _read_manifest_sample_rate(manifest_path)
    wav = _load_audio(Path(args.input), sample_rate) if args.input else _generated_audio(sample_rate, args.seconds)
    target_len = int(wav.shape[-1])

    runtime_manifest = _override_manifest_providers(manifest_path, args.providers)
    try:
        runtime = OnnxMoEStreamRuntime(
            runtime_manifest,
            providers=args.providers,
            sample_rate=sample_rate,
            parallel_experts=True,
            parallel_workers=max(1, int(args.workers)),
        )
        print("manifest:", manifest_path)
        print("sample_rate:", runtime.sample_rate)
        print("hop_samples:", runtime.hop_samples)
        print("expert_modes:", runtime.adapter_modes())

        state = runtime.create_state(batch_size=1)
        outputs = []
        weights = []
        started = time.perf_counter()
        frames = 0
        for offset in range(0, target_len, runtime.hop_samples):
            chunk = wav[offset:offset + runtime.hop_samples]
            valid = int(chunk.shape[-1])
            if valid < runtime.hop_samples:
                chunk = np.pad(chunk, (0, runtime.hop_samples - valid), mode="constant")
            y_hop, w, state, profile = runtime.step(chunk, state=state)
            outputs.append(y_hop[0, :valid].copy())
            weights.append(w.reshape(-1))
            frames += 1
            print(
                f"frame={frames} total_ms={profile['total_step_ms']:.3f} "
                f"expert_ms={profile['expert_stream_step_ms']:.3f} "
                f"router_ms={profile['router_ms']:.3f} "
                f"rtf={(profile['total_step_ms'] / 1000.0) / (runtime.hop_samples / sample_rate):.4f}"
            )
        elapsed = time.perf_counter() - started
        enhanced = np.concatenate(outputs, axis=0)[:target_len] if outputs else np.zeros(0, dtype=np.float32)
        audio_seconds = target_len / max(float(sample_rate), 1.0)
        mean_weights = np.mean(np.stack(weights, axis=0), axis=0) if weights else np.zeros(0, dtype=np.float32)
        print("done")
        print(f"frames: {frames}")
        print(f"audio_seconds: {audio_seconds:.3f}")
        print(f"wall_seconds: {elapsed:.3f}")
        print(f"rtf: {elapsed / max(audio_seconds, 1.0e-8):.4f}")
        print("mean_weights:", np.round(mean_weights, 4).tolist())
        print("output_samples:", int(enhanced.shape[-1]), "input_samples:", target_len)
        if args.output:
            _write_audio(Path(args.output), enhanced, sample_rate)
            print("wrote:", args.output)
        runtime.close()
    finally:
        if runtime_manifest != manifest_path:
            try:
                runtime_manifest.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
