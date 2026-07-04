"""Check exported MoE expert ONNX models.

The checker validates:
  - ONNX input/output names, shapes, dtype
  - fixed shapes suitable for I/O Binding
  - CUDAExecutionProvider availability
  - optional PyTorch wrapper vs ONNX Runtime per-frame consistency
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path[:] = [
    item for item in sys.path
    if Path(item or ".").resolve() != TOOLS_DIR
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.train import get_model  # noqa: E402
from tools.export_grpo_onnx import _load_router_checkpoint, _resolve_required_conf  # noqa: E402
from tools.export_moe_experts_onnx import (  # noqa: E402
    _assert_no_onnx_complex_dtype,
    _check_cuda_ep,
    _make_export_item,
    _run_consistency_check,
    _runtime_output_shape_check,
    _spec_frames_from_wav,
    inspect_onnx_io_for_iobinding,
)


def _resolve(path: str | Path, base: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base or Path.cwd()) / path


def _load_model(conf_path: str, ckpt: str | None, device: torch.device):
    resolved = _resolve_required_conf(conf_path)
    conf = OmegaConf.load(resolved)
    conf["ckpt"] = None
    if conf.get("router_grpo") is not None:
        if conf.router_grpo.get("inference_branch") is None:
            conf.router_grpo.inference_branch = {}
        conf.router_grpo.inference_branch.runtime = "torch"
        if conf.router_grpo.get("device_map") is not None:
            conf.router_grpo.device_map.enabled = False
    model = get_model(conf).to(device).eval()
    _load_router_checkpoint(model, ckpt)
    if hasattr(model, "_router_device_override"):
        model._router_device_override = device
    if hasattr(model, "_expert_device_overrides"):
        model._expert_device_overrides = [device for _ in range(len(model.experts))]
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--conf", default=None, help="Config used for export; enables PyTorch vs ONNX consistency checks.")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--wav", default=None, help="Optional noisy wav used for PyTorch-vs-ONNX stream checks.")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--strict", action="store_true", help="Fail if consistency exceeds default thresholds.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")

    item_by_name = {}
    if args.conf:
        model = _load_model(args.conf, args.ckpt, device)
        for idx in range(len(model.experts)):
            item = _make_export_item(model, idx, device)
            item_by_name[item.name] = item

    check_spec_frames = None
    if args.wav:
        check_spec_frames = _spec_frames_from_wav(
            args.wav,
            sample_rate=int(manifest.get("sample_rate", 16000)),
            n_fft=int(manifest.get("n_fft", manifest.get("frame_samples", 512))),
            hop=int(manifest.get("hop_size", manifest.get("hop_samples", 256))),
            device=device,
            max_frames=args.max_frames,
        )

    rows = []
    for expert in manifest.get("experts", []):
        name = expert["name"]
        rel_path = expert.get("onnx_path") or expert.get("path")
        onnx_path = _resolve(rel_path, base_dir)
        _assert_no_onnx_complex_dtype(onnx_path)
        io = inspect_onnx_io_for_iobinding(onnx_path)
        cuda_ok, cuda_msg = _check_cuda_ep(onnx_path, args.device_id)
        runtime_shape = None
        consistency = None
        if name in item_by_name:
            runtime_shape = _runtime_output_shape_check(
                item_by_name[name],
                onnx_path,
                device_id=args.device_id,
                require_cuda_ep=(device.type == "cuda"),
            )
            io["runtime_fixed_shapes"] = bool(runtime_shape["runtime_fixed_shapes"])
            io["runtime_output_shapes"] = runtime_shape["runtime_output_shapes"]
            io["fixed_shapes"] = bool(io["runtime_fixed_shapes"])
            consistency = _run_consistency_check(
                item_by_name[name],
                onnx_path,
                spec_frames=check_spec_frames,
            )
            if args.strict and (
                consistency["max_abs_error"] > 1.0e-3
                or consistency["mean_abs_error"] > 1.0e-4
                or not consistency["shape_match"]
                or not consistency["cache_shape_match"]
            ):
                raise RuntimeError(f"{name} consistency check failed: {consistency}")
        rows.append({
            "expert_name": name,
            "max_abs_error": None if consistency is None else consistency["max_abs_error"],
            "mean_abs_error": None if consistency is None else consistency["mean_abs_error"],
            "snr_diff": None if consistency is None else consistency["snr_diff"],
            "shape_match": None if consistency is None else consistency["shape_match"],
            "cache_shape_match": None if consistency is None else consistency["cache_shape_match"],
            "metadata_fixed_shapes": io["metadata_fixed_shapes"],
            "runtime_fixed_shapes": io.get("runtime_fixed_shapes"),
            "float32": io["float32"],
            "cuda_ep": cuda_ok,
            "cuda_message": cuda_msg,
        })

    print("expert_name\tmax_abs_error\tmean_abs_error\tsnr_diff\tshape_match\tcache_shape_match\tmetadata_fixed_shapes\truntime_fixed_shapes\tfloat32\tcuda_ep")
    for row in rows:
        print(
            f"{row['expert_name']}\t{row['max_abs_error']}\t{row['mean_abs_error']}\t{row['snr_diff']}\t"
            f"{row['shape_match']}\t{row['cache_shape_match']}\t{row['metadata_fixed_shapes']}\t"
            f"{row['runtime_fixed_shapes']}\t{row['float32']}\t{row['cuda_ep']}"
        )
        if not row["cuda_ep"]:
            print(f"  CUDA EP error: {row['cuda_message']}")


if __name__ == "__main__":
    main()
