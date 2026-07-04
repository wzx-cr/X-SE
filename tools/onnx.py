"""Compatibility entry point for GRPO ONNX export.

This file is intentionally named onnx.py for the user's old command:
`python tools/onnx.py ...`.  When executed directly, Python puts `tools/` at
the front of sys.path; that can shadow the real `onnx` package during
torch.onnx.export.  Remove that path before loading the actual exporter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent

cleaned = []
for item in sys.path:
    try:
        if Path(item or ".").resolve() == TOOLS_DIR:
            continue
    except Exception:
        pass
    cleaned.append(item)
sys.path[:] = cleaned
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "_grpo_export_onnx",
    TOOLS_DIR / "export_grpo_onnx.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load tools/export_grpo_onnx.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.main()
