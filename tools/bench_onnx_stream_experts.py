"""Benchmark GRPO MoE streaming expert runtime variants.

Examples:

  python tools/bench_onnx_stream_experts.py \
    --manifest exp/onnx/lisennet_fastenhancerS_ulunas_moe_degru/manifest.json \
    --device cuda:0 --warmup 50 --iters 500

  python tools/bench_onnx_stream_experts.py \
    --manifest exp/onnx/lisennet_fastenhancerS_ulunas_moe_degru/manifest.json \
    --conf examples/DNS2021/conf/MoE.yaml \
    --mode all
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path[:] = [
    item for item in sys.path
    if Path(item or ".").resolve() != TOOLS_DIR
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha.enh.system.grpo import OnnxMoEStreamRuntime  # noqa: E402
from alpha.enh.system.grpo_onnx_iobinding import (  # noqa: E402
    TorchOnnxMoEStreamRuntime,
    check_ort_profile_for_cpu_fallback,
)


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = int(round((len(ordered) - 1) * float(pct) / 100.0))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _summary(name, frame_ms, hop_samples, sample_rate, expert_ms=None, parallel_wall_ms=None):
    expert_ms = expert_ms or {}
    audio_ms = 1000.0 * float(hop_samples) / float(sample_rate)
    return {
        "mode": name,
        "avg_frame_ms": statistics.fmean(frame_ms) if frame_ms else 0.0,
        "p50_frame_ms": _percentile(frame_ms, 50),
        "p90_frame_ms": _percentile(frame_ms, 90),
        "p95_frame_ms": _percentile(frame_ms, 95),
        "p99_frame_ms": _percentile(frame_ms, 99),
        "rtf": (statistics.fmean(frame_ms) / audio_ms) if frame_ms and audio_ms > 0 else 0.0,
        "expert0_ms": statistics.fmean(expert_ms.get(0, [])) if expert_ms.get(0) else 0.0,
        "expert1_ms": statistics.fmean(expert_ms.get(1, [])) if expert_ms.get(1) else 0.0,
        "expert2_ms": statistics.fmean(expert_ms.get(2, [])) if expert_ms.get(2) else 0.0,
        "parallel_wall_ms": statistics.fmean(parallel_wall_ms) if parallel_wall_ms else 0.0,
    }


def _print_summary(summary):
    print(f"\n[{summary['mode']}]")
    for key in (
        "avg_frame_ms",
        "p50_frame_ms",
        "p90_frame_ms",
        "p95_frame_ms",
        "p99_frame_ms",
        "rtf",
        "expert0_ms",
        "expert1_ms",
        "expert2_ms",
        "parallel_wall_ms",
    ):
        print(f"  {key}: {summary[key]:.4f}")


def _device_id(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark is intended for CUDA ONNXRuntime.")
    return device.index if device.index is not None else 0


def bench_onnx_run_serial(args):
    """Baseline: old numpy + sess.run path, with three experts serial."""
    device_id = _device_id(args.device)
    runtime = OnnxMoEStreamRuntime(
        args.manifest,
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": device_id}],
        override_manifest_providers=True,
        parallel_experts=False,
        parallel_workers=1,
    )
    state = runtime.create_state(batch_size=1)
    hop = torch.randn(runtime.hop_samples, device=args.device, dtype=torch.float32) * 0.01
    frame_ms = []
    expert_ms = {0: [], 1: [], 2: []}
    parallel_wall_ms = []
    for idx in range(args.warmup + args.iters):
        torch.cuda.synchronize(device_id)
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        y, w, state, profile = runtime.step(hop.detach().cpu().numpy(), state)
        ended.record()
        ended.synchronize()
        if idx >= args.warmup:
            frame_ms.append(float(started.elapsed_time(ended)))
            parallel_wall_ms.append(float(profile.get("expert_stream_step_ms", 0.0)))
            for expert_idx, value in profile.get("expert_step_ms", {}).items():
                expert_ms.setdefault(int(expert_idx), []).append(float(value))
    runtime.close()
    return _summary("onnx_sess_run_serial", frame_ms, runtime.hop_samples, runtime.sample_rate, expert_ms, parallel_wall_ms)


def bench_iobinding(args, parallel):
    device_id = _device_id(args.device)
    runtime = TorchOnnxMoEStreamRuntime(
        args.manifest,
        device=args.device,
        device_id=device_id,
        parallel_experts=parallel,
        parallel_cuda_streams=bool(args.parallel_cuda_streams),
        enable_profiling=bool(args.ort_profile),
        use_onnx_router=bool(args.use_onnx_router),
    )
    state = runtime.create_state(batch_size=1)
    hop = torch.randn(runtime.hop_samples, device=args.device, dtype=torch.float32) * 0.01
    frame_ms = []
    expert_ms = {0: [], 1: [], 2: []}
    parallel_wall_ms = []
    for idx in range(args.warmup + args.iters):
        if args.use_onnx_router:
            y, w, state, profile = runtime.step(hop, state)
        else:
            _, _, state, profile = runtime.step_experts(hop, state)
        if idx >= args.warmup:
            frame_ms.append(float(profile.get("stream_frame_total_ms", profile.get("total_step_ms", 0.0))))
            parallel_wall_ms.append(float(profile.get("experts_parallel_wall_ms", 0.0)))
            for expert_idx, value in profile.get("expert_step_ms", {}).items():
                expert_ms.setdefault(int(expert_idx), []).append(float(value))
    runtime.close()
    for sess in ([runtime.router] if runtime.router is not None else []) + list(runtime.experts):
        for profile_path in getattr(sess, "profile_paths", []):
            fallback = check_ort_profile_for_cpu_fallback(profile_path)
            print("ORT profile:", profile_path)
            if fallback:
                print(f"WARNING: ONNX Runtime CPU fallback detected. events={len(fallback)}")
    suffix = "onnx_router" if args.use_onnx_router else "experts_only"
    mode = "onnx_iobinding_parallel_streams" if parallel else "onnx_iobinding_serial"
    mode = f"{mode}_{suffix}"
    return _summary(mode, frame_ms, runtime.hop_samples, runtime.sample_rate, expert_ms, parallel_wall_ms)


def bench_pytorch_serial(args):
    if not args.conf:
        print("\n[pytorch_serial] skipped: pass --conf to instantiate the PyTorch model.")
        return None
    from omegaconf import OmegaConf
    from modules.train import get_model

    conf = OmegaConf.load(args.conf)
    conf.ckpt = None
    if conf.get("router_grpo") is not None:
        conf.router_grpo.inference_branch.runtime = "torch"
        conf.router_grpo.inference_branch.parallel_experts = False
    model = get_model(conf).to(args.device).eval()
    if hasattr(model, "stream_parallel_experts"):
        model.stream_parallel_experts = False
    model._reset_stream_infer_state()
    hop = torch.randn(model.stream_hop_samples, device=args.device, dtype=torch.float32) * 0.01
    frame_ms = []
    for idx in range(args.warmup + args.iters):
        torch.cuda.synchronize(_device_id(args.device))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model._stream_moe_step(hop)
        end.record()
        end.synchronize()
        if idx >= args.warmup:
            frame_ms.append(float(start.elapsed_time(end)))
    return _summary("pytorch_serial", frame_ms, model.stream_hop_samples, model.sample_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--conf", default=None, help="Optional YAML config for PyTorch serial benchmark.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument(
        "--mode",
        default="iobind_parallel",
        choices=["all", "pytorch", "onnx_run", "iobind_serial", "iobind_parallel"],
    )
    parser.add_argument("--parallel-cuda-streams", dest="parallel_cuda_streams", action="store_true", default=True)
    parser.add_argument("--no-parallel-cuda-streams", dest="parallel_cuda_streams", action="store_false")
    parser.add_argument("--use-onnx-router", action="store_true")
    parser.add_argument("--ort-profile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    torch.cuda.set_device(_device_id(args.device))

    summaries: list[dict[str, Any]] = []
    if args.mode in ("all", "pytorch"):
        result = bench_pytorch_serial(args)
        if result is not None:
            summaries.append(result)
            _print_summary(result)
    if args.mode in ("all", "onnx_run"):
        result = bench_onnx_run_serial(args)
        summaries.append(result)
        _print_summary(result)
    if args.mode in ("all", "iobind_serial"):
        result = bench_iobinding(args, parallel=False)
        summaries.append(result)
        _print_summary(result)
    if args.mode in ("all", "iobind_parallel"):
        result = bench_iobinding(args, parallel=True)
        summaries.append(result)
        _print_summary(result)

    if len(summaries) >= 2:
        before = summaries[0]
        after = summaries[-1]
        speedup = before["avg_frame_ms"] / max(after["avg_frame_ms"], 1.0e-8)
        print("\n[comparison]")
        print(f"  before: {before['mode']} avg_frame_ms={before['avg_frame_ms']:.4f}, rtf={before['rtf']:.4f}")
        print(f"  after:  {after['mode']} avg_frame_ms={after['avg_frame_ms']:.4f}, rtf={after['rtf']:.4f}")
        print(f"  speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
