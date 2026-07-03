from __future__ import annotations

import json
import math
import os
import site
import sys
import time
import ctypes
import glob
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for ONNXRuntime I/O binding.") from exc

from modules.utils.common import EPS


__all__ = [
    "TorchOnnxMoEStreamRuntime",
    "TorchOrtMoEStreamState",
    "TorchOrtRouterSession",
    "TorchOrtExpertSession",
    "check_ort_profile_for_cpu_fallback",
]


def load_onnxruntime():
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install onnxruntime-gpu before enabling ONNX I/O binding.") from exc
    return ort


def _candidate_nvidia_lib_dirs():
    roots = []
    for item in [*site.getsitepackages(), site.getusersitepackages()]:
        if item and Path(item).exists():
            roots.append(Path(item))
    for prefix in {Path(sys.prefix), Path(sys.exec_prefix)}:
        roots.extend([prefix / "lib", prefix / "Lib" / "site-packages"])
    seen = set()
    result = []
    for root in roots:
        for pattern in (
            "nvidia/*/lib",
            "nvidia/*/bin",
            "nvidia/cudnn/lib",
            "nvidia/cublas/lib",
            "nvidia/cuda_runtime/lib",
        ):
            for path in root.glob(pattern):
                if path.exists():
                    resolved = str(path.resolve())
                    if resolved not in seen:
                        result.append(resolved)
                        seen.add(resolved)
    return result


_PRELOADED_NVIDIA_LIBS = False


def _preload_nvidia_cuda_libs():
    """Best-effort preload for pip nvidia-* packages.

    onnxruntime-gpu loads CUDA EP with dlopen. On cluster environments the
    cuDNN 9 library may exist inside the conda env but not be visible through
    LD_LIBRARY_PATH. Preloading with RTLD_GLOBAL avoids silent CPU fallback.
    """
    global _PRELOADED_NVIDIA_LIBS
    if _PRELOADED_NVIDIA_LIBS:
        return
    _PRELOADED_NVIDIA_LIBS = True
    if os.name == "nt":
        return
    lib_dirs = _candidate_nvidia_lib_dirs()
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
    load_order = [
        "libcudart.so*",
        "libcublasLt.so*",
        "libcublas.so*",
        "libcudnn.so.9*",
        "libcudnn_*.so.9*",
    ]
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for pattern in load_order:
        matches = []
        for lib_dir in lib_dirs:
            matches.extend(glob.glob(str(Path(lib_dir) / pattern)))
        for lib_path in sorted(set(matches)):
            try:
                ctypes.CDLL(lib_path, mode=mode)
            except OSError:
                pass


def check_ort_profile_for_cpu_fallback(profile_path):
    """Return CPUExecutionProvider events from an ONNXRuntime profile json."""
    profile_path = Path(profile_path)
    if not profile_path.exists():
        return []
    try:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    fallback = []
    for event in events:
        args = event.get("args", {}) if isinstance(event, dict) else {}
        provider = str(args.get("provider", args.get("execution_provider", "")))
        if provider == "CPUExecutionProvider":
            fallback.append(event)
    return fallback


def _periodic_hann_torch(length, device, dtype=torch.float32):
    length = int(length)
    if length <= 0:
        return torch.zeros(0, device=device, dtype=dtype)
    idx = torch.arange(length, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * idx / float(length))


def _stream_istft_window_torch(window, n_fft, hop_size):
    window = window.reshape(-1)
    n_fft = int(n_fft)
    hop_size = int(hop_size)
    if window.numel() < n_fft:
        pad = n_fft - window.numel()
        window = F.pad(window, (pad // 2, pad - pad // 2))
    elif window.numel() > n_fft:
        window = window[:n_fft]
    k = int(math.ceil(float(n_fft) / float(hop_size)))
    length = hop_size * (2 * k - 1) + (n_fft - hop_size)
    denom = torch.zeros(length, device=window.device, dtype=window.dtype)
    for idx in range(2 * k - 1):
        start = idx * hop_size
        denom[start:start + n_fft] += window * window
    start = (k - 1) * hop_size
    denom = denom[start:start + n_fft].clamp_min(float(EPS))
    return window / denom


def _plain_shape(shape, batch_size=1):
    result = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            result.append(int(dim))
        elif idx == 0:
            result.append(int(batch_size))
        else:
            result.append(1)
    return tuple(result)


def _bind_input(io, name, tensor, device_id):
    tensor = tensor.contiguous()
    io.bind_input(
        name=name,
        device_type="cuda",
        device_id=int(device_id),
        element_type=np.float32,
        shape=tuple(tensor.shape),
        buffer_ptr=tensor.data_ptr(),
    )


def _bind_output(io, name, tensor, device_id):
    tensor = tensor.contiguous()
    io.bind_output(
        name=name,
        device_type="cuda",
        device_id=int(device_id),
        element_type=np.float32,
        shape=tuple(tensor.shape),
        buffer_ptr=tensor.data_ptr(),
    )


def _cuda_session(path, device_id=0, stream=None, enable_profiling=False, intra_threads=1):
    _preload_nvidia_cuda_libs()
    ort = load_onnxruntime()
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(intra_threads or 1)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_profiling = bool(enable_profiling)
    provider_options = {"device_id": int(device_id)}
    if stream is not None:
        provider_options["user_compute_stream"] = str(stream.cuda_stream)
        provider_options["do_copy_in_default_stream"] = "1"
    session = ort.InferenceSession(
        str(path),
        sess_options=so,
        providers=[("CUDAExecutionProvider", provider_options)],
    )
    actual_providers = session.get_providers()
    if "CUDAExecutionProvider" not in actual_providers:
        raise RuntimeError(
            "ONNXRuntime CUDAExecutionProvider was requested but did not load. "
            f"Actual providers: {actual_providers}. This usually means cuDNN 9 / CUDA 12 "
            "runtime libraries are not visible. Install nvidia-cudnn-cu12 and put its lib "
            "directory on LD_LIBRARY_PATH, or run tools/check_grpo_onnx_iobinding_env.py."
        )
    return session


class CudaStepTimer:
    def __init__(self, enabled, device):
        self.enabled = bool(enabled and torch.cuda.is_available() and torch.device(device).type == "cuda")
        self.device = torch.device(device)
        self.events = {}
        self.cpu_start = {}
        self.cpu_end = {}

    def start(self, name, stream=None):
        if self.enabled:
            event = torch.cuda.Event(enable_timing=True)
            event.record(stream or torch.cuda.current_stream(self.device))
            self.events.setdefault(name, [None, None])[0] = event
        else:
            self.cpu_start[name] = time.perf_counter()

    def end(self, name, stream=None):
        if self.enabled:
            event = torch.cuda.Event(enable_timing=True)
            event.record(stream or torch.cuda.current_stream(self.device))
            self.events.setdefault(name, [None, None])[1] = event
        else:
            self.cpu_end[name] = time.perf_counter()

    def elapsed_ms(self, name):
        if self.enabled:
            start, end = self.events.get(name, (None, None))
            if start is None or end is None:
                return 0.0
            end.synchronize()
            return float(start.elapsed_time(end))
        if name not in self.cpu_start or name not in self.cpu_end:
            return 0.0
        return float((self.cpu_end[name] - self.cpu_start[name]) * 1000.0)


@dataclass
class TorchOrtMoEStreamState:
    input_tail: torch.Tensor
    expert_states: list[dict[str, Any]]
    router_state: dict[str, Any] | None = None
    output_tail: torch.Tensor | None = None
    ola_buffer: torch.Tensor | None = None
    num_steps: int = 0


class TorchOrtRouterSession:
    def __init__(
        self,
        spec,
        base_dir,
        device,
        device_id,
        stream=None,
        sample_rate=16000,
        frame_samples=512,
        hop_samples=256,
        stft_conf=None,
        num_experts=3,
        enable_profiling=False,
    ):
        self.spec = dict(spec or {})
        self.base_dir = Path(base_dir)
        self.device = torch.device(device)
        self.device_id = int(device_id)
        self.stream = stream
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        self.hop_samples = int(hop_samples)
        self.stft_conf = dict(stft_conf or {})
        self.num_experts = int(num_experts)
        self.enable_profiling = bool(enable_profiling)
        self.feature_input = self.spec.get("feature_input", self.spec.get("input", "features"))
        self.features_buf = torch.empty(1, 9, device=self.device, dtype=torch.float32)
        self.weights_buf = torch.empty(1, self.num_experts, device=self.device, dtype=torch.float32)
        self.logits_buf = torch.empty(1, self.num_experts, device=self.device, dtype=torch.float32)
        self.profile_paths = []
        self._load_session()

    def _resolve_path(self, path):
        if not path:
            raise FileNotFoundError(
                "Router ONNX path is missing from manifest. Re-export with tools/onnx.py --export-router "
                "or set router_grpo.inference_branch.onnx.use_onnx_router=false."
            )
        path = Path(path)
        return path if path.is_absolute() else self.base_dir / path

    def _load_session(self):
        path = self._resolve_path(self.spec.get("path", "router_features.onnx"))
        if not path.exists():
            raise FileNotFoundError(f"Router ONNX file not found: {path}")
        self.path = path
        self.session = _cuda_session(
            path,
            device_id=self.device_id,
            stream=self.stream,
            enable_profiling=self.enable_profiling,
            intra_threads=self.spec.get("intra_op_num_threads", 1),
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        if self.feature_input not in self.input_names:
            self.feature_input = self.input_names[0]
        self._bind()

    def _bind(self):
        self.io = self.session.io_binding()
        _bind_input(self.io, self.feature_input, self.features_buf, self.device_id)
        _bind_output(self.io, self.output_names[0], self.weights_buf, self.device_id)
        if len(self.output_names) > 1:
            _bind_output(self.io, self.output_names[1], self.logits_buf, self.device_id)

    def reload(self, path=None):
        if self.enable_profiling:
            try:
                self.profile_paths.append(self.session.end_profiling())
            except Exception:
                pass
        if path is not None:
            self.spec["path"] = str(path)
        self._load_session()

    def close(self):
        if self.enable_profiling:
            try:
                self.profile_paths.append(self.session.end_profiling())
            except Exception:
                pass

    def _stft_mag(self, wav):
        n_fft = int(self.stft_conf.get("n_fft", self.frame_samples))
        win_length = int(self.stft_conf.get("win_length", n_fft))
        hop_length = int(self.stft_conf.get("hop_length", self.hop_samples))
        center = bool(self.stft_conf.get("center", True))
        window = _periodic_hann_torch(win_length, wav.device, wav.dtype)
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window = F.pad(window, (left, n_fft - win_length - left))
        elif win_length > n_fft:
            window = window[:n_fft]
        x = wav
        if center:
            pad = n_fft // 2
            if x.shape[-1] <= 1:
                x = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)
            else:
                x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        if x.shape[-1] < n_fft:
            x = F.pad(x, (0, n_fft - x.shape[-1]))
        frames = x.unfold(-1, n_fft, hop_length)
        if frames.numel() == 0:
            frames = F.pad(x, (0, n_fft - x.shape[-1]))[:, :n_fft].unsqueeze(1)
        frames = frames * window.view(1, 1, -1)
        spec = torch.fft.rfft(frames, n=n_fft, dim=-1)
        return spec.abs().transpose(1, 2).unsqueeze(1).clamp_min(1.0e-8)

    @staticmethod
    def _band_log_energy(mag, start, end):
        if end <= start:
            return torch.zeros(mag.shape[0], device=mag.device, dtype=mag.dtype)
        band = mag[:, :, start:end, :]
        return torch.log(torch.mean(torch.square(band), dim=(1, 2, 3)).clamp_min(1.0e-8))

    def extract_features(self, frame):
        wav = frame
        mag = self._stft_mag(wav)
        rms = torch.log(torch.mean(torch.square(wav), dim=-1).clamp_min(1.0e-8))
        peak = torch.max(torch.abs(wav), dim=-1).values.clamp_max(10.0)
        if wav.shape[-1] > 1:
            zcr = torch.mean((wav[:, 1:] * wav[:, :-1] < 0).to(wav.dtype), dim=-1)
        else:
            zcr = torch.zeros_like(rms)
        log_mag = torch.log(mag)
        log_mag_mean = torch.mean(log_mag, dim=(1, 2, 3))
        log_mag_std = torch.std(log_mag.reshape(log_mag.shape[0], -1), dim=-1, unbiased=False)
        n_freq = int(mag.shape[-2])
        f1 = max(1, n_freq // 3)
        f2 = max(f1 + 1, (2 * n_freq) // 3)
        low = self._band_log_energy(mag, 0, f1)
        mid = self._band_log_energy(mag, f1, min(f2, n_freq))
        high = self._band_log_energy(mag, min(f2, n_freq), n_freq)
        flatness = torch.exp(torch.mean(torch.log(mag), dim=(1, 2, 3))) / torch.mean(mag, dim=(1, 2, 3)).clamp_min(1.0e-8)
        feats = torch.stack([rms, peak, zcr, log_mag_mean, log_mag_std, low, mid, high, flatness], dim=-1)
        return torch.nan_to_num(feats.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def weights(self, frame):
        stream = self.stream or torch.cuda.current_stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            feats = self.extract_features(frame.to(self.device, non_blocking=True))
            self.features_buf.copy_(feats, non_blocking=True)
            self.session.run_with_iobinding(self.io)
        return self.weights_buf, self.logits_buf


class TorchOrtExpertSession:
    def __init__(self, spec, base_dir, device, device_id, stream=None, frame_samples=512, hop_samples=256, sample_rate=16000, enable_profiling=False):
        self.spec = dict(spec or {})
        self.base_dir = Path(base_dir)
        self.device = torch.device(device)
        self.device_id = int(device_id)
        self.stream = stream
        self.frame_samples = int(self.spec.get("frame_samples", frame_samples))
        self.hop_samples = int(self.spec.get("hop_samples", hop_samples))
        self.sample_rate = int(sample_rate)
        self.protocol = str(self.spec.get("protocol", "wave_hop")).lower()
        self.name = str(self.spec.get("name", "expert"))
        self.true_cache = bool(self.spec.get("true_cache", not bool(self.spec.get("fallback", False))))
        self.enable_profiling = bool(enable_profiling)
        self.profile_paths = []
        path = Path(self.spec.get("path", ""))
        if not path:
            raise ValueError("Each ONNX expert manifest entry must define path.")
        self.path = path if path.is_absolute() else self.base_dir / path
        if not self.path.exists():
            raise FileNotFoundError(f"Expert ONNX file not found: {self.path}")
        self.session = _cuda_session(
            self.path,
            device_id=self.device_id,
            stream=self.stream,
            enable_profiling=self.enable_profiling,
            intra_threads=self.spec.get("intra_op_num_threads", 1),
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.cache_inputs = list(self.spec.get("cache_inputs", [])) or self.input_names[1:]
        self.cache_outputs = list(self.spec.get("cache_outputs", [])) or self.output_names[1:]
        self.input_name = self.spec.get("input", self.input_names[0])
        if self.input_name not in self.input_names:
            self.input_name = self.input_names[0]

    def close(self):
        if self.enable_profiling:
            try:
                self.profile_paths.append(self.session.end_profiling())
            except Exception:
                pass

    def _shape_for_input(self, input_name, batch_size):
        explicit = self.spec.get("cache_shapes", {})
        if isinstance(explicit, list) and input_name in self.cache_inputs:
            idx = self.cache_inputs.index(input_name)
            if idx < len(explicit):
                return _plain_shape(explicit[idx], batch_size=batch_size)
        if isinstance(explicit, dict) and input_name in explicit:
            return _plain_shape(explicit[input_name], batch_size=batch_size)
        for item in self.session.get_inputs():
            if item.name == input_name:
                return _plain_shape(item.shape, batch_size=batch_size)
        return (batch_size, 1)

    def _main_input_shape(self, batch_size):
        n_fft = int(self.spec.get("n_fft", self.frame_samples))
        if self.protocol == "spec_frame":
            return (batch_size, n_fft // 2 + 1, 1, 2)
        if self.protocol == "lisen_stft":
            return (batch_size, 3, 1, n_fft // 2 + 1)
        if self.protocol == "fastenhancer_stft":
            freq = n_fft // 2 + 1
            if bool(self.spec.get("discard_last_freq_bin", True)):
                freq -= 1
            return (batch_size, freq, 1, 2)
        if self.protocol == "ulunas_stft":
            return (batch_size, n_fft // 2 + 1, 1, 2)
        return (batch_size, self.hop_samples)

    def _main_output_shape(self, batch_size):
        n_fft = int(self.spec.get("n_fft", self.frame_samples))
        if self.protocol == "spec_frame":
            return (batch_size, n_fft // 2 + 1, 1, 2)
        if self.protocol == "lisen_stft":
            return (batch_size, 2, 1, n_fft // 2 + 1)
        if self.protocol == "fastenhancer_stft":
            return self._main_input_shape(batch_size)
        if self.protocol == "ulunas_stft":
            return (batch_size, n_fft // 2 + 1, 1, 2)
        return (batch_size, self.hop_samples)

    def init_state(self, batch_size=1):
        batch_size = int(batch_size)
        state = {
            "protocol": self.protocol,
            "cache_hit": bool(self.true_cache),
            "num_steps": 0,
            "input_buf": torch.empty(self._main_input_shape(batch_size), device=self.device, dtype=torch.float32),
            "output_buf": torch.empty(self._main_output_shape(batch_size), device=self.device, dtype=torch.float32),
            "caches": {},
            "cache_outputs": {},
        }
        if self.protocol in ("lisen_stft", "fastenhancer_stft", "ulunas_stft", "spec_frame"):
            n_fft = int(self.spec.get("n_fft", self.frame_samples))
            hop = int(self.spec.get("hop_samples", self.hop_samples))
            win_length = int(self.spec.get("win_length", n_fft))
            window = _periodic_hann_torch(win_length, self.device)
            if win_length < n_fft:
                left = (n_fft - win_length) // 2
                window = F.pad(window, (left, n_fft - win_length - left))
            elif win_length > n_fft:
                window = window[:n_fft]
            state.update({
                "n_fft": n_fft,
                "hop_samples": hop,
                "window": window.contiguous(),
                "window_istft": _stream_istft_window_torch(window, n_fft, hop).contiguous(),
                "istft_cache": torch.zeros(batch_size, max(n_fft - hop, 0), device=self.device, dtype=torch.float32),
            })
            if self.protocol == "lisen_stft":
                state["compress_factor"] = float(self.spec.get("compress_factor", 0.3))
                state["prev_phase"] = torch.zeros(batch_size, n_fft // 2 + 1, device=self.device, dtype=torch.float32)
            if self.protocol == "fastenhancer_stft":
                state["compression"] = float(self.spec.get("compression", self.spec.get("input_compression", 0.3)))
                state["eps"] = float(self.spec.get("eps", 1.0e-5))
                state["discard_last_freq_bin"] = bool(self.spec.get("discard_last_freq_bin", True))
                state["stft_cache"] = torch.zeros(batch_size, max(n_fft - hop, 0), device=self.device, dtype=torch.float32)

        for name in self.cache_inputs:
            state["caches"][name] = torch.zeros(self._shape_for_input(name, batch_size), device=self.device, dtype=torch.float32)
        for idx, output_name in enumerate(self.cache_outputs):
            if idx < len(self.cache_inputs):
                shape = tuple(state["caches"][self.cache_inputs[idx]].shape)
            else:
                shape = _plain_shape(self.session.get_outputs()[idx + 1].shape, batch_size=batch_size)
            state["cache_outputs"][output_name] = torch.empty(shape, device=self.device, dtype=torch.float32)
        self._bind_state(state)
        return state

    def _bind_state(self, state):
        io = self.session.io_binding()
        _bind_input(io, self.input_name, state["input_buf"], self.device_id)
        for name in self.cache_inputs:
            if name in self.input_names:
                _bind_input(io, name, state["caches"][name], self.device_id)
        _bind_output(io, self.output_names[0], state["output_buf"], self.device_id)
        for output_name in self.cache_outputs:
            if output_name in self.output_names:
                _bind_output(io, output_name, state["cache_outputs"][output_name], self.device_id)
        state["io_binding"] = io

    def _copy_cache_outputs_to_inputs(self, state):
        for idx, cache_input in enumerate(self.cache_inputs):
            if idx >= len(self.cache_outputs):
                continue
            output_name = self.cache_outputs[idx]
            if output_name in state["cache_outputs"]:
                state["caches"][cache_input].copy_(state["cache_outputs"][output_name], non_blocking=True)

    def _align_hop(self, y):
        hop = int(self.hop_samples)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.shape[-1] < hop:
            y = F.pad(y, (0, hop - y.shape[-1]))
        elif y.shape[-1] > hop:
            y = y[..., :hop]
        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    def _run_ort(self, state):
        self.session.run_with_iobinding(state["io_binding"])
        self._copy_cache_outputs_to_inputs(state)

    def _run_spec_frame(self, frame, state, timer):
        n_fft = int(state["n_fft"])
        hop = int(state["hop_samples"])
        x = frame
        if x.shape[-1] < n_fft:
            x = F.pad(x, (n_fft - x.shape[-1], 0))
        elif x.shape[-1] > n_fft:
            x = x[..., -n_fft:]
        timer.start(f"{self.name}:stft", self.stream)
        spec_complex = torch.fft.rfft(x * state["window"].view(1, -1), n=n_fft, dim=1)
        spec = torch.view_as_real(spec_complex).unsqueeze(2)
        state["input_buf"].copy_(spec, non_blocking=True)
        timer.end(f"{self.name}:stft", self.stream)
        self._run_ort(state)
        timer.start(f"{self.name}:istft", self.stream)
        enhanced = state["output_buf"].reshape(x.shape[0], n_fft // 2 + 1, 1, 2)
        enhanced_complex = torch.complex(enhanced[:, :, 0, 0], enhanced[:, :, 0, 1])
        frame_out = torch.fft.irfft(enhanced_complex, n=n_fft, dim=1)
        frame_out = frame_out * state["window_istft"].view(1, -1)
        istft_cache = state["istft_cache"]
        if istft_cache.numel():
            frame_out[:, :istft_cache.shape[-1]] += istft_cache
        y = frame_out[:, :hop]
        state["istft_cache"].copy_(frame_out[:, hop:], non_blocking=True)
        timer.end(f"{self.name}:istft", self.stream)
        return self._align_hop(y), state

    def _run_ulunas_stft(self, frame, state, timer):
        n_fft = int(state["n_fft"])
        hop = int(state["hop_samples"])
        x = frame
        if x.shape[-1] < n_fft:
            x = F.pad(x, (n_fft - x.shape[-1], 0))
        elif x.shape[-1] > n_fft:
            x = x[..., -n_fft:]
        timer.start(f"{self.name}:stft", self.stream)
        spec_complex = torch.fft.rfft(x * state["window"].view(1, -1), n=n_fft, dim=1)
        spec = torch.view_as_real(spec_complex).unsqueeze(2)
        state["input_buf"].copy_(spec, non_blocking=True)
        timer.end(f"{self.name}:stft", self.stream)
        self._run_ort(state)
        timer.start(f"{self.name}:istft", self.stream)
        enhanced = state["output_buf"].reshape(x.shape[0], n_fft // 2 + 1, 1, 2)
        enhanced_complex = torch.complex(enhanced[:, :, 0, 0], enhanced[:, :, 0, 1])
        frame_out = torch.fft.irfft(enhanced_complex, n=n_fft, dim=1)
        frame_out = frame_out * state["window_istft"].view(1, -1)
        istft_cache = state["istft_cache"]
        if istft_cache.numel():
            frame_out[:, :istft_cache.shape[-1]] += istft_cache
        y = frame_out[:, :hop]
        state["istft_cache"].copy_(frame_out[:, hop:], non_blocking=True)
        timer.end(f"{self.name}:istft", self.stream)
        return self._align_hop(y), state

    def _run_fastenhancer_stft(self, hop_tensor, state, timer):
        n_fft = int(state["n_fft"])
        hop_size = int(state["hop_samples"])
        compression = float(state.get("compression", 0.3))
        eps = float(state.get("eps", 1.0e-5))
        hop = hop_tensor
        if hop.shape[-1] < hop_size:
            hop = F.pad(hop, (0, hop_size - hop.shape[-1]))
        elif hop.shape[-1] > hop_size:
            hop = hop[..., -hop_size:]
        stft_cache = state["stft_cache"]
        frame = torch.cat([stft_cache, hop], dim=1) if stft_cache.numel() else hop
        if frame.shape[-1] < n_fft:
            frame = F.pad(frame, (n_fft - frame.shape[-1], 0))
        elif frame.shape[-1] > n_fft:
            frame = frame[..., -n_fft:]
        cache_len = max(n_fft - hop_size, 0)
        if cache_len > 0:
            state["stft_cache"].copy_(frame[:, -cache_len:], non_blocking=True)
        timer.start(f"{self.name}:stft", self.stream)
        spec_complex = torch.fft.rfft(frame * state["window"].view(1, -1), n=n_fft, dim=1)
        spec = torch.view_as_real(spec_complex).unsqueeze(2)
        if bool(state.get("discard_last_freq_bin", True)):
            spec = spec[:, :-1, :, :]
        mag = torch.linalg.vector_norm(spec, dim=-1, keepdim=True).clamp_min(eps)
        spec_noisy = spec * torch.pow(mag, compression - 1.0)
        state["input_buf"].copy_(spec_noisy, non_blocking=True)
        timer.end(f"{self.name}:stft", self.stream)
        self._run_ort(state)
        timer.start(f"{self.name}:istft", self.stream)
        mask = state["output_buf"]
        spec_c = torch.complex(spec_noisy[..., 0], spec_noisy[..., 1])
        mask_c = torch.complex(mask[..., 0], mask[..., 1])
        spec_hat = spec_c * mask_c
        mag_hat = torch.abs(spec_hat).clamp_min(eps)
        spec_hat = spec_hat * torch.pow(mag_hat, 1.0 / max(compression, 1.0e-8) - 1.0)
        if bool(state.get("discard_last_freq_bin", True)):
            spec_hat = F.pad(spec_hat, (0, 0, 0, 1))
        frame_out = torch.fft.irfft(spec_hat[:, :, 0], n=n_fft, dim=1)
        frame_out = frame_out * state["window_istft"].view(1, -1)
        istft_cache = state["istft_cache"]
        if istft_cache.numel():
            frame_out[:, :istft_cache.shape[-1]] += istft_cache
        y = frame_out[:, :hop_size]
        state["istft_cache"].copy_(frame_out[:, hop_size:], non_blocking=True)
        timer.end(f"{self.name}:istft", self.stream)
        return self._align_hop(y), state

    def _run_lisen_stft(self, frame, state, timer):
        n_fft = int(state["n_fft"])
        hop = int(state["hop_samples"])
        compress_factor = float(state.get("compress_factor", 0.3))
        x = frame
        if x.shape[-1] < n_fft:
            x = F.pad(x, (n_fft - x.shape[-1], 0))
        elif x.shape[-1] > n_fft:
            x = x[..., -n_fft:]
        timer.start(f"{self.name}:stft", self.stream)
        spec_complex = torch.fft.rfft(x * state["window"].view(1, -1), n=n_fft, dim=1)
        spec_mag = torch.abs(spec_complex).clamp_min(1.0e-12)
        src_mag = torch.pow(spec_mag, compress_factor)
        cur_phase = torch.angle(spec_complex)
        gd = torch.diff(
            cur_phase,
            dim=1,
            prepend=torch.zeros(cur_phase.shape[0], 1, device=cur_phase.device, dtype=cur_phase.dtype),
        )
        prev_phase = state["prev_phase"]
        freq_axis = torch.arange(cur_phase.shape[-1], device=cur_phase.device, dtype=cur_phase.dtype).view(1, -1)
        ifd = (cur_phase - prev_phase) - 2.0 * math.pi * (float(hop) / float(n_fft)) * freq_axis
        gd = torch.atan2(gd.sin(), gd.cos())
        ifd = torch.atan2(ifd.sin(), ifd.cos())
        state["prev_phase"].copy_(cur_phase, non_blocking=True)
        features = torch.stack([src_mag, gd / math.pi, ifd / math.pi], dim=1).unsqueeze(2)
        state["input_buf"].copy_(features, non_blocking=True)
        timer.end(f"{self.name}:stft", self.stream)
        self._run_ort(state)
        timer.start(f"{self.name}:istft", self.stream)
        mask = state["output_buf"]
        est_mag = (mask[:, 0, 0, :] + 1.0e-8) * src_mag + (mask[:, 1, 0, :] + 1.0e-8) * src_mag
        est_mag = torch.pow(est_mag.clamp_min(1.0e-12), 1.0 / max(compress_factor, 1.0e-8))
        est_complex = torch.complex(est_mag * cur_phase.cos(), est_mag * cur_phase.sin())
        frame_out = torch.fft.irfft(est_complex, n=n_fft, dim=1)
        frame_out = frame_out * state["window_istft"].view(1, -1)
        istft_cache = state["istft_cache"]
        if istft_cache.numel():
            frame_out[:, :istft_cache.shape[-1]] += istft_cache
        y = frame_out[:, :hop]
        state["istft_cache"].copy_(frame_out[:, hop:], non_blocking=True)
        timer.end(f"{self.name}:istft", self.stream)
        return self._align_hop(y), state

    def stream_step(self, frame, hop, state, timer, ready_event=None):
        stream = self.stream or torch.cuda.current_stream(self.device)
        if ready_event is not None:
            stream.wait_event(ready_event)
        else:
            stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            timer.start(f"{self.name}:total", stream)
            timer.start(f"{self.name}:copy", stream)
            frame = frame.to(self.device, non_blocking=True)
            hop = hop.to(self.device, non_blocking=True)
            timer.end(f"{self.name}:copy", stream)
            if self.protocol == "ulunas_stft":
                y, state = self._run_ulunas_stft(frame, state, timer)
            elif self.protocol == "spec_frame":
                y, state = self._run_spec_frame(frame, state, timer)
            elif self.protocol == "lisen_stft":
                y, state = self._run_lisen_stft(frame, state, timer)
            elif self.protocol == "fastenhancer_stft":
                y, state = self._run_fastenhancer_stft(hop, state, timer)
            else:
                state["input_buf"].copy_(hop, non_blocking=True)
                self._run_ort(state)
                y = self._align_hop(state["output_buf"])
            state["cache_hit"] = bool(self.true_cache)
            state["num_steps"] = int(state.get("num_steps", 0)) + 1
            timer.end(f"{self.name}:total", stream)
            done = torch.cuda.Event(enable_timing=False)
            done.record(stream)
        return y, state, done

    def stream_spec_step(self, spec, state, timer, ready_event=None):
        if self.protocol != "spec_frame":
            raise RuntimeError(f"{self.name} protocol={self.protocol} does not support stream_spec_step.")
        stream = self.stream or torch.cuda.current_stream(self.device)
        if ready_event is not None:
            stream.wait_event(ready_event)
        else:
            stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            timer.start(f"{self.name}:total", stream)
            timer.start(f"{self.name}:copy", stream)
            state["input_buf"].copy_(spec.to(self.device, non_blocking=True), non_blocking=True)
            timer.end(f"{self.name}:copy", stream)
            self._run_ort(state)
            state["cache_hit"] = bool(self.true_cache)
            state["num_steps"] = int(state.get("num_steps", 0)) + 1
            timer.end(f"{self.name}:total", stream)
            done = torch.cuda.Event(enable_timing=False)
            done.record(stream)
        return state["output_buf"], state, done


class TorchOnnxMoEStreamRuntime:
    expects_torch_input = True

    def __init__(
        self,
        manifest_path,
        device="cuda:0",
        device_id=0,
        sample_rate=16000,
        frame_samples=512,
        hop_samples=256,
        stft_conf=None,
        parallel_experts=True,
        parallel_cuda_streams=True,
        enable_profiling=False,
        use_onnx_router=False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("ONNX I/O binding runtime requires CUDA.")
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.base_dir = self.manifest_path.parent
        self.device = torch.device(device)
        self.device_id = int(device_id)
        self.sample_rate = int(self.manifest.get("sample_rate", sample_rate))
        self.frame_samples = int(self.manifest.get("frame_samples", frame_samples))
        self.hop_samples = int(self.manifest.get("hop_samples", hop_samples))
        self.parallel_experts = bool(parallel_experts)
        self.parallel_workers = len(self.manifest.get("experts", []))
        self.parallel_cuda_streams = bool(parallel_cuda_streams)
        self.providers = ["CUDAExecutionProvider"]
        self.override_manifest_providers = True
        self.enable_profiling = bool(enable_profiling)
        self.use_onnx_router = bool(use_onnx_router)
        self._expert_executor = None
        self.streams = []
        num_streams = len(self.manifest.get("experts", [])) + (1 if self.use_onnx_router else 0)
        with torch.cuda.device(self.device_id):
            for _ in range(num_streams if self.parallel_cuda_streams else 0):
                self.streams.append(torch.cuda.Stream(device=self.device_id))
        router_stream = self.streams[0] if self.use_onnx_router and self.streams else None
        if self.use_onnx_router:
            expert_streams = self.streams[1:] if self.streams else [None] * len(self.manifest.get("experts", []))
            self.router = TorchOrtRouterSession(
                self.manifest.get("router", {}),
                self.base_dir,
                self.device,
                self.device_id,
                stream=router_stream,
                sample_rate=self.sample_rate,
                frame_samples=self.frame_samples,
                hop_samples=self.hop_samples,
                stft_conf=self.manifest.get("stft", stft_conf or {}),
                num_experts=len(self.manifest.get("experts", [])),
                enable_profiling=self.enable_profiling,
            )
        else:
            expert_streams = self.streams if self.streams else [None] * len(self.manifest.get("experts", []))
            self.router = None
        self.experts = [
            TorchOrtExpertSession(
                spec,
                self.base_dir,
                self.device,
                self.device_id,
                stream=expert_streams[idx] if idx < len(expert_streams) else None,
                frame_samples=self.frame_samples,
                hop_samples=self.hop_samples,
                sample_rate=self.sample_rate,
                enable_profiling=self.enable_profiling,
            )
            for idx, spec in enumerate(self.manifest.get("experts", []))
        ]
        if not self.experts:
            raise ValueError(f"ONNX MoE manifest has no experts: {self.manifest_path}")
        self.shared_spec_fusion = bool(self.experts and all(expert.protocol == "spec_frame" for expert in self.experts))
        self._init_shared_spec_stft()

    def _init_shared_spec_stft(self):
        self.shared_spec_window = None
        self.shared_spec_istft_window = None
        if not self.shared_spec_fusion:
            return
        n_ffts = {int(expert.spec.get("n_fft", expert.frame_samples)) for expert in self.experts}
        hops = {int(expert.spec.get("hop_samples", expert.hop_samples)) for expert in self.experts}
        if len(n_ffts) != 1 or len(hops) != 1:
            raise RuntimeError(
                "Shared spec-domain MoE requires identical n_fft/hop_samples for all experts; "
                f"got n_fft={sorted(n_ffts)}, hop={sorted(hops)}."
            )
        stft_conf = dict(self.manifest.get("stft", {}) or {})
        self.shared_spec_n_fft = int(next(iter(n_ffts)))
        self.shared_spec_hop = int(next(iter(hops)))
        win_length = int(
            stft_conf.get(
                "win_length",
                self.manifest.get("win_size", self.manifest.get("win_length", self.shared_spec_n_fft)),
            )
        )
        window = _periodic_hann_torch(win_length, self.device)
        if win_length < self.shared_spec_n_fft:
            left = (self.shared_spec_n_fft - win_length) // 2
            window = F.pad(window, (left, self.shared_spec_n_fft - win_length - left))
        elif win_length > self.shared_spec_n_fft:
            window = window[: self.shared_spec_n_fft]
        self.shared_spec_window = window.contiguous()
        self.shared_spec_istft_window = _stream_istft_window_torch(
            window,
            self.shared_spec_n_fft,
            self.shared_spec_hop,
        ).contiguous()

    def outputs_spec_domain(self):
        return bool(self.shared_spec_fusion)

    def reload_router(self, path=None):
        if self.router is None:
            return
        self.router.reload(path)

    def close(self):
        if self._expert_executor is not None:
            self._expert_executor.shutdown(wait=True)
            self._expert_executor = None
        if self.router is not None:
            self.router.close()
        for expert in self.experts:
            expert.close()
        sessions = ([self.router] if self.router is not None else []) + list(self.experts)
        for sess in sessions:
            for profile_path in getattr(sess, "profile_paths", []):
                fallback = check_ort_profile_for_cpu_fallback(profile_path)
                if fallback:
                    print(f"WARNING: ONNX Runtime CPU fallback detected in {profile_path}: {len(fallback)} events")

    def adapter_modes(self):
        return [
            {
                "expert": expert.name,
                "mode": f"onnx_iobinding_{expert.protocol}",
                "true_cache": bool(expert.true_cache),
                "device": ",".join(expert.session.get_providers()),
            }
            for expert in self.experts
        ]

    def fallback_expert_count(self):
        return sum(1 for expert in self.experts if not expert.true_cache)

    def _ensure_expert_executor(self):
        if self._expert_executor is None:
            self._expert_executor = ThreadPoolExecutor(
                max_workers=max(1, len(self.experts)),
                thread_name_prefix="mos_moe_ort_expert",
            )
        return self._expert_executor

    def _expert_stream_task(self, idx, frame, hop, expert_state, timer, ready_event):
        torch.cuda.set_device(self.device_id)
        expert = self.experts[idx]
        y, new_state, event = expert.stream_step(
            frame,
            hop,
            expert_state,
            timer,
            ready_event=ready_event,
        )
        return idx, y, new_state, event

    def _expert_spec_task(self, idx, spec, expert_state, timer, ready_event):
        torch.cuda.set_device(self.device_id)
        expert = self.experts[idx]
        y, new_state, event = expert.stream_spec_step(
            spec,
            expert_state,
            timer,
            ready_event=ready_event,
        )
        return idx, y, new_state, event

    def create_state(self, batch_size=1):
        batch_size = int(batch_size)
        if batch_size != 1:
            raise ValueError("ONNX I/O binding stream runtime currently expects batch_size=1 for fixed-shape buffers.")
        context_len = max(self.frame_samples - self.hop_samples, 0)
        input_tail = torch.zeros(batch_size, context_len, device=self.device, dtype=torch.float32)
        output_tail = torch.zeros(batch_size, context_len, device=self.device, dtype=torch.float32)
        return TorchOrtMoEStreamState(
            input_tail=input_tail,
            expert_states=[expert.init_state(batch_size=batch_size) for expert in self.experts],
            router_state=None,
            output_tail=output_tail,
            ola_buffer=output_tail,
            num_steps=0,
        )

    def _prepare_frame(self, hop, state, timer, current):
        hop = torch.as_tensor(hop, dtype=torch.float32, device=self.device)
        if hop.ndim == 1:
            hop = hop.unsqueeze(0)
        if hop.ndim != 2:
            raise ValueError(f"ONNX MoE step expects [T] or [B,T] hop, got {tuple(hop.shape)}.")
        if hop.shape[0] != 1:
            raise ValueError("ONNX I/O binding stream runtime currently supports batch_size=1.")
        if hop.shape[-1] < self.hop_samples:
            hop = F.pad(hop, (0, self.hop_samples - hop.shape[-1]))
        elif hop.shape[-1] > self.hop_samples:
            hop = hop[..., -self.hop_samples:]
        if state is None:
            state = self.create_state(batch_size=1)

        timer.start("input_cache", current)
        tail = state.input_tail
        frame = torch.cat([tail, hop], dim=-1) if tail.numel() else hop
        if frame.shape[-1] < self.frame_samples:
            frame = F.pad(frame, (self.frame_samples - frame.shape[-1], 0))
        elif frame.shape[-1] > self.frame_samples:
            frame = frame[..., -self.frame_samples:]
        context_len = max(self.frame_samples - self.hop_samples, 0)
        if context_len > 0:
            state.input_tail.copy_(frame[..., -context_len:], non_blocking=True)
        timer.end("input_cache", current)
        return hop, frame, state, context_len

    def _run_experts(self, frame, hop, state, timer, current):
        timer.start("experts_parallel_wall", current)
        ready_event = torch.cuda.Event(enable_timing=False)
        ready_event.record(current)
        expert_outputs = [None] * len(self.experts)
        expert_events = []
        if self.parallel_experts and len(self.experts) > 1:
            executor = self._ensure_expert_executor()
            futures = [
                executor.submit(
                    self._expert_stream_task,
                    idx,
                    frame,
                    hop,
                    state.expert_states[idx],
                    timer,
                    ready_event,
                )
                for idx in range(len(self.experts))
            ]
            for future in futures:
                idx, y, new_state, event = future.result()
                state.expert_states[idx] = new_state
                expert_outputs[idx] = y
                expert_events.append(event)
            for event in expert_events:
                current.wait_event(event)
        else:
            for idx, expert in enumerate(self.experts):
                y, new_state, event = expert.stream_step(
                    frame,
                    hop,
                    state.expert_states[idx],
                    timer,
                    ready_event=ready_event,
                )
                state.expert_states[idx] = new_state
                expert_outputs[idx] = y
                current.wait_event(event)
        timer.end("experts_parallel_wall", current)
        return expert_outputs

    def _shared_spec_from_frame(self, frame, timer, current):
        if self.shared_spec_window is None:
            raise RuntimeError("Shared spec STFT is not initialized.")
        n_fft = int(self.shared_spec_n_fft)
        x = frame
        if x.shape[-1] < n_fft:
            x = F.pad(x, (n_fft - x.shape[-1], 0))
        elif x.shape[-1] > n_fft:
            x = x[..., -n_fft:]
        timer.start("shared_stft", current)
        spec_complex = torch.fft.rfft(x * self.shared_spec_window.view(1, -1), n=n_fft, dim=1)
        spec = torch.view_as_real(spec_complex).unsqueeze(2).contiguous()
        timer.end("shared_stft", current)
        return spec

    def _run_spec_experts(self, spec, state, timer, current):
        timer.start("experts_parallel_wall", current)
        ready_event = torch.cuda.Event(enable_timing=False)
        ready_event.record(current)
        expert_outputs = [None] * len(self.experts)
        expert_events = []
        if self.parallel_experts and len(self.experts) > 1:
            executor = self._ensure_expert_executor()
            futures = [
                executor.submit(
                    self._expert_spec_task,
                    idx,
                    spec,
                    state.expert_states[idx],
                    timer,
                    ready_event,
                )
                for idx in range(len(self.experts))
            ]
            for future in futures:
                idx, y, new_state, event = future.result()
                state.expert_states[idx] = new_state
                expert_outputs[idx] = y
                expert_events.append(event)
            for event in expert_events:
                current.wait_event(event)
        else:
            for idx, expert in enumerate(self.experts):
                y, new_state, event = expert.stream_spec_step(
                    spec,
                    state.expert_states[idx],
                    timer,
                    ready_event=ready_event,
                )
                state.expert_states[idx] = new_state
                expert_outputs[idx] = y
                current.wait_event(event)
        timer.end("experts_parallel_wall", current)
        return expert_outputs

    def _profile_from_timer(self, timer):
        sync_started = time.perf_counter()
        expert_ms = {}
        shared_stft_ms = timer.elapsed_ms("shared_stft")
        shared_istft_ms = timer.elapsed_ms("shared_istft")
        stft_ms = shared_stft_ms
        istft_ms = shared_istft_ms
        copy_ms = 0.0
        for idx, expert in enumerate(self.experts):
            total_key = f"{expert.name}:total"
            expert_ms[idx] = timer.elapsed_ms(total_key)
            stft_ms += timer.elapsed_ms(f"{expert.name}:stft")
            istft_ms += timer.elapsed_ms(f"{expert.name}:istft")
            copy_ms += timer.elapsed_ms(f"{expert.name}:copy")
        total_ms = timer.elapsed_ms("total")
        cpu_gpu_sync_ms = (time.perf_counter() - sync_started) * 1000.0
        profile = {
            "input_cache_ms": timer.elapsed_ms("input_cache"),
            "expert_stream_step_ms": timer.elapsed_ms("experts_parallel_wall"),
            "experts_parallel_wall_ms": timer.elapsed_ms("experts_parallel_wall"),
            "router_ms": timer.elapsed_ms("router"),
            "fusion_ms": timer.elapsed_ms("fusion"),
            "total_step_ms": total_ms,
            "stream_frame_total_ms": total_ms,
            "frame_total_ms": total_ms,
            "stft_ms": stft_ms,
            "istft_ms": istft_ms,
            "shared_stft_ms": shared_stft_ms,
            "shared_istft_ms": shared_istft_ms,
            "io_binding_copy_ms": copy_ms,
            "cpu_gpu_sync_ms": cpu_gpu_sync_ms,
            "cache_hit": 1.0 if self.fallback_expert_count() == 0 else 0.0,
            "fallback_expert_count": float(self.fallback_expert_count()),
            "expert_step_ms": expert_ms,
        }
        for idx, expert in enumerate(self.experts):
            safe = expert.name.replace("-", "_")
            profile[f"{safe}_onnx_ms"] = expert_ms.get(idx, 0.0)
            short = safe
            for suffix in ("_expert", "_onnx"):
                if short.endswith(suffix):
                    short = short[: -len(suffix)]
            profile[f"{short}_onnx_ms"] = expert_ms.get(idx, 0.0)
            profile[f"expert_{short}_ms"] = expert_ms.get(idx, 0.0)
        return profile

    def step_experts(self, hop, state=None):
        current = torch.cuda.current_stream(self.device)
        timer = CudaStepTimer(True, self.device)
        timer.start("total", current)
        hop, frame, state, _ = self._prepare_frame(hop, state, timer, current)
        if self.shared_spec_fusion:
            spec = self._shared_spec_from_frame(frame, timer, current)
            expert_outputs = self._run_spec_experts(spec, state, timer, current)
            expert_wavs = torch.stack([item.to(self.device, non_blocking=True) for item in expert_outputs], dim=1)
            output_domain = "spec"
        else:
            expert_outputs = self._run_experts(frame, hop, state, timer, current)
            expert_wavs = torch.stack([item.to(self.device, non_blocking=True) for item in expert_outputs], dim=1)
            output_domain = "wave"
        timer.end("total", current)
        profile = self._profile_from_timer(timer)
        profile["expert_output_domain"] = output_domain
        profile["shared_spec_fusion"] = 1.0 if self.shared_spec_fusion else 0.0
        return expert_wavs, frame, state, profile

    def fuse_expert_specs(self, expert_specs, weights, state, profile=None):
        if not self.shared_spec_fusion:
            raise RuntimeError("fuse_expert_specs is only valid for spec_frame manifests.")
        current = torch.cuda.current_stream(self.device)
        timer = CudaStepTimer(True, self.device)
        expert_specs = torch.as_tensor(expert_specs, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(weights, dtype=expert_specs.dtype, device=self.device)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        if expert_specs.ndim != 5:
            raise ValueError(f"Expected expert spec outputs [B,E,F,1,2], got {tuple(expert_specs.shape)}.")
        timer.start("fusion", current)
        fused_spec = torch.sum(weights[:, :, None, None, None] * expert_specs, dim=1)
        timer.end("fusion", current)

        timer.start("shared_istft", current)
        n_fft = int(self.shared_spec_n_fft)
        hop = int(self.shared_spec_hop)
        enhanced_complex = torch.complex(fused_spec[:, :, 0, 0], fused_spec[:, :, 0, 1])
        frame_out = torch.fft.irfft(enhanced_complex, n=n_fft, dim=1)
        frame_out = frame_out * self.shared_spec_istft_window.view(1, -1)
        istft_cache = state.output_tail
        if istft_cache is None or istft_cache.shape[-1] != max(n_fft - hop, 0):
            istft_cache = torch.zeros(frame_out.shape[0], max(n_fft - hop, 0), device=self.device, dtype=frame_out.dtype)
            state.output_tail = istft_cache
        if istft_cache.numel():
            frame_out[:, :istft_cache.shape[-1]] += istft_cache
        y = frame_out[:, :hop]
        if istft_cache.numel():
            state.output_tail.copy_(frame_out[:, hop:], non_blocking=True)
            state.ola_buffer = state.output_tail
        state.router_state = {"weights": weights.detach()}
        state.num_steps += 1
        timer.end("shared_istft", current)

        fusion_ms = timer.elapsed_ms("fusion")
        istft_ms = timer.elapsed_ms("shared_istft")
        update = {
            "fusion_ms": fusion_ms + istft_ms,
            "spec_fusion_ms": fusion_ms,
            "shared_istft_ms": istft_ms,
            "istft_ms": float((profile or {}).get("istft_ms", 0.0)) + istft_ms,
        }
        if profile is not None:
            profile.update(update)
        return torch.clamp(torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), state, update

    def step(self, hop, state=None, weights=None):
        if weights is None and self.router is None:
            raise RuntimeError(
                "TorchOnnxMoEStreamRuntime was created with use_onnx_router=False. "
                "Call step_experts(...) and run the PyTorch router/fusion outside ORT, or pass weights=..."
            )

        expert_wavs, frame, state, profile = self.step_experts(hop, state=state)
        current = torch.cuda.current_stream(self.device)
        timer = CudaStepTimer(True, self.device)

        timer.start("router", current)
        if weights is None:
            weights, logits = self.router.weights(frame)
            if self.router.stream is not None:
                current.wait_stream(self.router.stream)
            state.router_state = {"weights": weights, "logits": logits}
        else:
            weights = torch.as_tensor(weights, dtype=expert_wavs.dtype, device=self.device)
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            logits = None
            state.router_state = {"weights": weights}
        timer.end("router", current)

        if self.shared_spec_fusion:
            fused, state, fusion_update = self.fuse_expert_specs(expert_wavs, weights, state, profile=profile)
            fusion_ms = float(fusion_update.get("fusion_ms", 0.0))
        else:
            timer.start("fusion", current)
            fused = torch.sum(weights[:, :, None] * expert_wavs, dim=1)
            fused = torch.clamp(torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            context_len = max(self.frame_samples - self.hop_samples, 0)
            if context_len > 0:
                state.output_tail.copy_(fused[..., -context_len:], non_blocking=True)
                state.ola_buffer = state.output_tail
            state.num_steps += 1
            timer.end("fusion", current)
            fusion_ms = timer.elapsed_ms("fusion")

        router_ms = timer.elapsed_ms("router")
        total_ms = (
            float(profile.get("total_step_ms", 0.0))
            + float(router_ms)
            + float(fusion_ms)
        )
        profile["router_ms"] = router_ms
        profile["fusion_ms"] = fusion_ms
        profile["total_step_ms"] = total_ms
        profile["stream_frame_total_ms"] = total_ms
        profile["frame_total_ms"] = total_ms
        return fused[:, :self.hop_samples], weights, state, profile
