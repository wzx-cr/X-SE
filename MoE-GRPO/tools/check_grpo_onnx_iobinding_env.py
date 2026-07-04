"""Preflight checks for GRPO ONNX I/O binding runtime.

Run on the server before training:

  python tools/check_grpo_onnx_iobinding_env.py \
    --manifest exp/onnx/lisennet_fastenhancerS_ulunas_moe_degru/manifest.json
"""

from __future__ import annotations

import argparse
import ctypes.util
import os
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path[:] = [
    item for item in sys.path
    if Path(item or ".").resolve() != TOOLS_DIR
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    print("python:", sys.executable)
    print("cwd:", Path.cwd())
    print("repo:", REPO_ROOT)
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))

    try:
        import torch
        print("torch:", torch.__version__)
        print("torch.cuda.is_available:", torch.cuda.is_available())
        print("torch.version.cuda:", torch.version.cuda)
        if torch.cuda.is_available():
            print("cuda device count:", torch.cuda.device_count())
            print("cuda device:", torch.cuda.get_device_name(args.device_id))
    except Exception as exc:
        print("ERROR: torch check failed:", repr(exc))
        raise

    try:
        import onnxruntime as ort
        print("onnxruntime:", ort.__version__)
        print("available providers:", ort.get_available_providers())
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print("ERROR: CUDAExecutionProvider is not available in this onnxruntime build.")
    except Exception as exc:
        print("ERROR: onnxruntime import failed:", repr(exc))
        raise

    libcudnn = ctypes.util.find_library("cudnn")
    print("ctypes cudnn:", libcudnn)
    if not libcudnn:
        print(
            "WARNING: libcudnn was not found by the dynamic linker. "
            "For onnxruntime-gpu 1.23.x you usually need cuDNN 9 on LD_LIBRARY_PATH."
        )

    try:
        from alpha.enh.system import grpo_onnx_iobinding
        cls = getattr(grpo_onnx_iobinding, "TorchOnnxMoEStreamRuntime", None)
        print("grpo_onnx_iobinding path:", Path(grpo_onnx_iobinding.__file__).resolve())
        print("TorchOnnxMoEStreamRuntime exists:", cls is not None)
        if cls is None:
            raise RuntimeError("TorchOnnxMoEStreamRuntime is missing; copy the latest grpo_onnx_iobinding.py.")
    except Exception as exc:
        print("ERROR: grpo_onnx_iobinding check failed:", repr(exc))
        raise

    if args.manifest:
        manifest = Path(args.manifest)
        print("manifest:", manifest.resolve())
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        try:
            runtime = cls(
                manifest,
                device=f"cuda:{args.device_id}",
                device_id=args.device_id,
                parallel_experts=True,
                parallel_cuda_streams=True,
                use_onnx_router=False,
            )
            print("runtime providers:", runtime.providers)
            print("expert modes:", runtime.adapter_modes())
            runtime.close()
        except Exception as exc:
            print("ERROR: failed to construct TorchOnnxMoEStreamRuntime:", repr(exc))
            raise

    print("OK: GRPO ONNX I/O binding environment check passed.")


if __name__ == "__main__":
    main()
