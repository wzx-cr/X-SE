import csv
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Deque, Dict, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is required only for ONNX inference.
    np = None

import modules.model.arch as model_arch
from modules.utils import metrics
from modules.utils.common import (
    EPS,
    NumpyEncoder,
    compact_dict,
    freeze_model,
    merge_dicts,
    refine_state_dict,
    resolve_path,
)
from modules.utils.compute_score import TorchMOS
from modules.utils.logging import logger
from .. import models
from .system import BaseSE, UniSE, _resolve_model_class


def _to_plain_dict(conf):
    if conf is None:
        return {}
    if OmegaConf.is_config(conf):
        return OmegaConf.to_container(conf, resolve=True)
    return dict(conf)


# REFACTOR: Centralize small tensor shape/sanitization helpers so GRPO,
# expert forwarding, and stream inference share one implementation.
class TensorUtils:
    @staticmethod
    def safe_nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0):
        if torch.is_complex(x):
            return torch.complex(
                torch.nan_to_num(x.real, nan, posinf, neginf),
                torch.nan_to_num(x.imag, nan, posinf, neginf),
            )
        return torch.nan_to_num(x, nan, posinf, neginf)

    @staticmethod
    def align_last_dim(a, b):
        """Crop tensors to the minimum last dimension."""
        T = min(a.shape[-1], b.shape[-1])
        return a[..., :T], b[..., :T]

    @staticmethod
    def align_spec(a, b):
        """Crop STFT tensors to common frequency bins and time frames."""
        f, t = min(a.shape[-2], b.shape[-2]), min(a.shape[-1], b.shape[-1])
        return a[..., :f, :t], b[..., :f, :t]

    @staticmethod
    def align_waveform_length(wav, target_len):
        if wav.ndim == 3 and wav.shape[1] == 1:
            wav = wav[:, 0, :]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise RuntimeError(f"Unsupported expert waveform output shape: {wav.shape}")
        if target_len is None:
            return wav
        target_len = int(target_len)
        if wav.shape[-1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[-1]))
        return wav[..., :target_len]

    @staticmethod
    def tensor_tree_to_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {key: TensorUtils.tensor_tree_to_numpy(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(TensorUtils.tensor_tree_to_numpy(item) for item in value)
        return value

    @staticmethod
    def numpy_tree_to_tensor(value, device=None):
        if torch.is_tensor(value):
            return value.to(device=device) if device is not None else value
        if np is not None and isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(value))
            return tensor.to(device=device) if device is not None else tensor
        if isinstance(value, dict):
            return {key: TensorUtils.numpy_tree_to_tensor(item, device=device) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(TensorUtils.numpy_tree_to_tensor(item, device=device) for item in value)
        return value


# REFACTOR: Template method-level latency logging without changing metric keys.
def log_latency(name, log_metrics=True, batch_size=1):
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            start = self._latency_stamp()
            result = fn(self, *args, **kwargs)
            self._record_latency(name, self._latency_elapsed_ms(start), log_metrics=log_metrics, batch_size=batch_size)
            return result
        return wrapper
    return decorator


_ONNX_STREAM_LATENCY_KEYS = ("stream_frame_total_ms", "frame_total_ms", "router_torch_ms", "router_ms", "router_onnx_export_ms", "router_onnx_reload_ms", "expert_lisennet_ms", "expert_fastenhancer_s_ms", "expert_ulunas_ms", "lisennet_onnx_ms", "lisennet_expert_onnx_ms", "fastenhancer_s_onnx_ms", "fastenhancer_s_expert_onnx_ms", "ulunas_onnx_ms", "ulunas_expert_onnx_ms", "experts_parallel_wall_ms", "fusion_ms", "stft_ms", "istft_ms", "shared_stft_ms", "shared_istft_ms", "spec_fusion_ms", "io_binding_copy_ms", "cpu_gpu_sync_ms")


class _FrozenEnhancementExpert(nn.Module):
    """Build, load, and freeze one pretrained enhancement expert."""

    # REFACTOR: Resolve model-specific expert forward strategies once during
    # initialization instead of branching on class-name strings at inference.
    _FORWARD_REGISTRY = {}

    @classmethod
    def register_forward(cls, model_name):
        def decorator(fn):
            cls._FORWARD_REGISTRY[str(model_name)] = fn
            return fn
        return decorator

    def __init__(self, expert_conf, default_model_conf=None, idx=0):
        super().__init__()
        expert_conf = _to_plain_dict(expert_conf)
        default_model_conf = _to_plain_dict(default_model_conf)
        model_conf = _to_plain_dict(expert_conf.get("model", expert_conf))
        model_conf = merge_dicts(default_model_conf, model_conf)
        if "name" not in model_conf:
            raise ValueError(f"expert[{idx}] must define model.name")

        self.name = str(expert_conf.get("name", model_conf.get("name", f"expert{idx}")))
        self.model_conf = model_conf
        self.model_class = _resolve_model_class(model_conf["name"])
        self.spec_data = bool(model_conf.get("spec_data", True))
        self.mag_data = bool(model_conf.get("mag_data", False))
        self.mask = bool(model_conf.get("mask", True))
        self.forward_type = str(model_conf.get("forward_type", model_conf.get("input_type", "auto"))).lower()
        self._forward_uses_default = self.model_class.__name__ not in self._FORWARD_REGISTRY
        self._forward_fn = self._FORWARD_REGISTRY.get(
            self.model_class.__name__,
            self._FORWARD_REGISTRY.get("__default__"),
        )

        self._resolve_param()
        self.model = self._build_model()
        self._load_pretrained(expert_conf)
        freeze_model(self.model)
        self.model.eval()

    def _resolve_param(self):
        param = self.model_conf.get("param")
        if isinstance(param, str):
            param = getattr(model_arch, param, None)
            if param is None:
                raise ValueError(f"Unknown model.param for expert {self.name}: {self.model_conf.get('param')}")
        self.model_conf["param"] = deepcopy(param)

    def _build_model(self):
        in_channels = 0
        if self.spec_data:
            in_channels += 2
        if self.mag_data:
            in_channels += 1

        default_conf = {
            "in_channels": in_channels,
            "mag_decoder": False,
            "spec_decoder": True,
            "debug": False,
            "rnn": {"bidirectional": True},
        }
        model_conf = merge_dicts(default_conf, self.model_conf)
        model_param = model_conf.get("param")
        is_model_base = isinstance(self.model_class, type) and issubclass(self.model_class, models.ModelBase)

        if is_model_base:
            return self.model_class(model_conf)
        if model_param is None:
            raise ValueError(f"expert {self.name}: model.param is required for {self.model_class.__name__}.")
        kwargs = {k: v for k, v in model_conf.items() if k not in ["param", "name", "init", "strict"]}
        return self.model_class(model_param, **kwargs)

    @staticmethod
    def _strip_prefix(state_dict, prefix):
        prefix = str(prefix)
        return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    def _load_pretrained(self, expert_conf):
        ckpt_path = self.model_conf.get("init") or expert_conf.get("ckpt")
        require_init = bool(expert_conf.get("require_init", True))
        if not ckpt_path:
            if require_init:
                raise ValueError(f"expert {self.name} requires a pretrained checkpoint in model.init or ckpt.")
            logger.warning(f"[FrozenExpertRouterGRPO] expert {self.name} has no pretrained checkpoint.")
            return

        ckpt_path = resolve_path(ckpt_path)
        logger.info(f"[FrozenExpertRouterGRPO] load expert {self.name}: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_prefix = expert_conf.get("state_prefix", self.model_conf.get("state_prefix"))
        if state_prefix:
            state_dict = self._strip_prefix(state_dict, state_prefix)
        else:
            refined = refine_state_dict(state_dict)
            state_dict = refined if refined else state_dict
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

        strict = bool(expert_conf.get("strict", self.model_conf.get("strict", False)))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            logger.warning(
                f"[FrozenExpertRouterGRPO] expert {self.name} loaded with "
                f"missing keys: {missing}, unexpected keys: {unexpected}"
            )
        else:
            logger.info(f"[FrozenExpertRouterGRPO] expert {self.name} loaded successfully.")


# REFACTOR: Register model-specific expert forward strategies near the expert
# wrapper. The owner supplies STFT/output adapters, so behavior stays identical.
def register_forward(model_name):
    return _FrozenEnhancementExpert.register_forward(model_name)


@register_forward("__default__")
def _forward_feature_expert(owner, expert, noisy_local, noisy_spec_local, target_len, output_device):
    est_spec = owner._feature_expert_spec(expert, noisy_local, noisy_spec_local).to(output_device, non_blocking=True)
    return owner._spec_output_to_wav(est_spec, target_len)


@dataclass
class MoEStreamState:
    input_tail: torch.Tensor
    expert_states: List[Any]
    router_state: Any = None
    output_tail: Any = None
    ola_buffer: Any = None
    num_steps: int = 0


def _load_onnxruntime():
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "ONNX inference branch requires onnxruntime or onnxruntime-gpu. "
            "Install it in the training environment before enabling "
            "router_grpo.inference_branch.runtime=onnx."
        ) from exc
    if np is None:
        raise RuntimeError("ONNX inference branch requires numpy.")
    return ort


def _periodic_hann_np(length):
    if np is None:
        raise RuntimeError("numpy is required for ONNX stream windows.")
    length = int(length)
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    n = np.arange(length, dtype=np.float32)
    return (0.5 - 0.5 * np.cos((2.0 * np.pi * n) / float(length))).astype(np.float32)


def _stream_istft_window_np(window, n_fft, hop_size):
    window = np.asarray(window, dtype=np.float32).reshape(-1)
    n_fft = int(n_fft)
    hop_size = int(hop_size)
    if window.size < n_fft:
        pad = n_fft - window.size
        window = np.pad(window, (pad // 2, pad - pad // 2))
    elif window.size > n_fft:
        window = window[:n_fft]
    k = int(math.ceil(float(n_fft) / float(hop_size)))
    length = hop_size * (2 * k - 1) + (n_fft - hop_size)
    denom = np.zeros((length,), dtype=np.float32)
    for idx in range(2 * k - 1):
        start = idx * hop_size
        denom[start:start + n_fft] += window * window
    start = (k - 1) * hop_size
    denom = np.maximum(denom[start:start + n_fft], float(EPS))
    return (window / denom).astype(np.float32)


def _onnx_plain_shape(shape, batch_size=1):
    result = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            result.append(int(dim))
        elif idx == 0:
            result.append(int(batch_size))
        else:
            result.append(1)
    return result


def _onnx_session(path, providers=None, provider_options=None, intra_threads=1):
    ort = _load_onnxruntime()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = int(intra_threads or 1)
    sess_options.inter_op_num_threads = 1
    try:
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    providers = providers or ["CPUExecutionProvider"]
    if provider_options:
        return ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
    return ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)


class OnnxRouterSession:
    """Small router session exported as features -> weights/logits."""

    def __init__(self, spec, base_dir, providers, sample_rate, frame_samples, hop_samples, stft_conf=None):
        spec = dict(spec or {})
        raw_path = spec.get("path", "router_features.onnx")
        if not raw_path:
            raise FileNotFoundError(
                "Router ONNX path is missing from manifest. Re-export with tools/onnx.py --export-router "
                "or set router_grpo.inference_branch.onnx.use_onnx_router=false."
            )
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(base_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Router ONNX file not found: {path}")
        self.path = path
        self.spec = spec
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        self.hop_samples = int(hop_samples)
        self.stft_conf = dict(stft_conf or {})
        self.providers = providers or ["CPUExecutionProvider"]
        self.provider_options = spec.get("provider_options")
        self.intra_threads = spec.get("intra_op_num_threads", 1)
        self.session = _onnx_session(
            path,
            providers=spec.get("providers", self.providers),
            provider_options=self.provider_options,
            intra_threads=self.intra_threads,
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.feature_input = spec.get("feature_input", spec.get("input", "features"))
        if self.feature_input not in self.input_names:
            self.feature_input = self.input_names[0]

    def reload(self, path=None):
        if path is not None:
            path = Path(path)
            if not path.is_absolute():
                # `path` may already be relative to the process cwd, e.g.
                # exp/onnx/run/router_features.onnx. Only join with the
                # manifest directory for bare manifest-relative names.
                if path.exists():
                    path = path.resolve()
                else:
                    path = self.path.parent / path
            self.path = path
        self.session = _onnx_session(
            self.path,
            providers=self.spec.get("providers", self.providers),
            provider_options=self.provider_options,
            intra_threads=self.intra_threads,
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def _stft_mag(self, wav):
        n_fft = int(self.stft_conf.get("n_fft", self.frame_samples))
        win_length = int(self.stft_conf.get("win_length", n_fft))
        hop_length = int(self.stft_conf.get("hop_length", self.hop_samples))
        center = bool(self.stft_conf.get("center", True))
        window = _periodic_hann_np(win_length)
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window = np.pad(window, (left, n_fft - win_length - left))
        elif win_length > n_fft:
            window = window[:n_fft]

        x = np.asarray(wav, dtype=np.float32)
        if center:
            pad = n_fft // 2
            if x.shape[-1] <= 1:
                x = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
            else:
                x = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
        if x.shape[-1] < n_fft:
            x = np.pad(x, ((0, 0), (0, n_fft - x.shape[-1])), mode="constant")

        frames = []
        for start in range(0, max(x.shape[-1] - n_fft, 0) + 1, hop_length):
            frames.append(x[:, start:start + n_fft] * window[None, :])
        if not frames:
            frames.append(np.pad(x, ((0, 0), (0, n_fft - x.shape[-1])), mode="constant")[:, :n_fft] * window[None, :])
        stacked = np.stack(frames, axis=-1)
        spec = np.fft.rfft(stacked, n=n_fft, axis=1)
        return np.abs(spec).astype(np.float32)[:, None, :, :]

    @staticmethod
    def _band_log_energy(mag, start, end):
        if end <= start:
            return np.zeros((mag.shape[0],), dtype=np.float32)
        band = mag[:, :, start:end, :]
        return np.log(np.maximum(np.mean(np.square(band), axis=(1, 2, 3)), 1.0e-8)).astype(np.float32)

    def extract_features(self, frame):
        wav = np.asarray(frame, dtype=np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]
        if wav.ndim > 2:
            wav = np.mean(wav, axis=1)
        mag = np.maximum(self._stft_mag(wav), 1.0e-8)
        rms = np.log(np.maximum(np.mean(np.square(wav), axis=-1), 1.0e-8))
        peak = np.minimum(np.max(np.abs(wav), axis=-1), 10.0)
        if wav.shape[-1] > 1:
            zcr = np.mean((wav[:, 1:] * wav[:, :-1]) < 0, axis=-1).astype(np.float32)
        else:
            zcr = np.zeros_like(rms, dtype=np.float32)

        log_mag = np.log(mag)
        log_mag_mean = np.mean(log_mag, axis=(1, 2, 3))
        log_mag_std = np.std(log_mag.reshape(log_mag.shape[0], -1), axis=-1)
        n_freq = mag.shape[-2]
        f1 = max(1, n_freq // 3)
        f2 = max(f1 + 1, (2 * n_freq) // 3)
        low = self._band_log_energy(mag, 0, f1)
        mid = self._band_log_energy(mag, f1, min(f2, n_freq))
        high = self._band_log_energy(mag, min(f2, n_freq), n_freq)
        flatness = np.exp(np.mean(np.log(mag), axis=(1, 2, 3))) / np.maximum(np.mean(mag, axis=(1, 2, 3)), 1.0e-8)
        feats = np.stack([rms, peak, zcr, log_mag_mean, log_mag_std, low, mid, high, flatness], axis=-1)
        return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def weights(self, frame):
        feats = self.extract_features(frame)
        outputs = self.session.run(None, {self.feature_input: feats})
        weights = np.asarray(outputs[0], dtype=np.float32)
        logits = np.asarray(outputs[1], dtype=np.float32) if len(outputs) > 1 else np.log(np.maximum(weights, 1.0e-8))
        return weights, logits


class OnnxExpertStreamSession:
    """One ONNX expert stream_step session, driven by a manifest entry."""

    def __init__(self, spec, base_dir, providers, frame_samples, hop_samples, sample_rate):
        spec = dict(spec or {})
        path = Path(spec.get("path", ""))
        if not path:
            raise ValueError("Each ONNX expert manifest entry must define path.")
        if not path.is_absolute():
            path = Path(base_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Expert ONNX file not found: {path}")
        self.path = path
        self.spec = spec
        self.name = str(spec.get("name", path.stem))
        self.protocol = str(spec.get("protocol", "wave_hop")).lower()
        self.frame_samples = int(spec.get("frame_samples", frame_samples))
        self.hop_samples = int(spec.get("hop_samples", hop_samples))
        self.sample_rate = int(sample_rate)
        self.session = _onnx_session(
            path,
            providers=spec.get("providers", providers),
            provider_options=spec.get("provider_options"),
            intra_threads=spec.get("intra_op_num_threads", 1),
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.cache_inputs = list(spec.get("cache_inputs", []))
        self.cache_outputs = list(spec.get("cache_outputs", []))
        if not self.cache_inputs:
            self.cache_inputs = self.input_names[1:]
        if not self.cache_outputs:
            self.cache_outputs = self.output_names[1:]
        self.true_cache = bool(spec.get("true_cache", not bool(spec.get("fallback", False))))

    def _cache_shape_for_input(self, input_name, batch_size):
        explicit = self.spec.get("cache_shapes", {})
        if isinstance(explicit, list):
            idx = self.cache_inputs.index(input_name)
            if idx < len(explicit):
                return _onnx_plain_shape(explicit[idx], batch_size=batch_size)
        if isinstance(explicit, dict) and input_name in explicit:
            return _onnx_plain_shape(explicit[input_name], batch_size=batch_size)
        for item in self.session.get_inputs():
            if item.name == input_name:
                return _onnx_plain_shape(item.shape, batch_size=batch_size)
        return [batch_size, 1]

    def init_state(self, batch_size=1):
        batch_size = int(batch_size)
        state = {
            "protocol": self.protocol,
            "cache_hit": bool(self.true_cache),
            "num_steps": 0,
        }
        if self.protocol == "ulunas_stft":
            n_fft = int(self.spec.get("n_fft", self.frame_samples))
            hop = int(self.spec.get("hop_samples", self.hop_samples))
            win_length = int(self.spec.get("win_length", n_fft))
            window = _periodic_hann_np(win_length)
            if win_length < n_fft:
                left = (n_fft - win_length) // 2
                window = np.pad(window, (left, n_fft - win_length - left))
            elif win_length > n_fft:
                window = window[:n_fft]
            state.update({
                "n_fft": n_fft,
                "hop_samples": hop,
                "window": window.astype(np.float32),
                "window_istft": _stream_istft_window_np(window, n_fft, hop),
                "istft_cache": np.zeros((batch_size, max(n_fft - hop, 0)), dtype=np.float32),
            })
        elif self.protocol == "lisen_stft":
            n_fft = int(self.spec.get("n_fft", self.frame_samples))
            hop = int(self.spec.get("hop_samples", self.hop_samples))
            win_length = int(self.spec.get("win_length", n_fft))
            window = _periodic_hann_np(win_length)
            if win_length < n_fft:
                left = (n_fft - win_length) // 2
                window = np.pad(window, (left, n_fft - win_length - left))
            elif win_length > n_fft:
                window = window[:n_fft]
            state.update({
                "n_fft": n_fft,
                "hop_samples": hop,
                "compress_factor": float(self.spec.get("compress_factor", 0.3)),
                "window": window.astype(np.float32),
                "window_istft": _stream_istft_window_np(window, n_fft, hop),
                "prev_phase": np.zeros((batch_size, n_fft // 2 + 1), dtype=np.float32),
                "istft_cache": np.zeros((batch_size, max(n_fft - hop, 0)), dtype=np.float32),
            })
        elif self.protocol == "fastenhancer_stft":
            n_fft = int(self.spec.get("n_fft", self.frame_samples))
            hop = int(self.spec.get("hop_samples", self.hop_samples))
            win_length = int(self.spec.get("win_length", n_fft))
            window = _periodic_hann_np(win_length)
            if win_length < n_fft:
                left = (n_fft - win_length) // 2
                window = np.pad(window, (left, n_fft - win_length - left))
            elif win_length > n_fft:
                window = window[:n_fft]
            state.update({
                "n_fft": n_fft,
                "hop_samples": hop,
                "compression": float(self.spec.get("compression", self.spec.get("input_compression", 0.3))),
                "eps": float(self.spec.get("eps", 1.0e-5)),
                "discard_last_freq_bin": bool(self.spec.get("discard_last_freq_bin", True)),
                "window": window.astype(np.float32),
                "window_istft": _stream_istft_window_np(window, n_fft, hop),
                "stft_cache": np.zeros((batch_size, max(n_fft - hop, 0)), dtype=np.float32),
                "istft_cache": np.zeros((batch_size, max(n_fft - hop, 0)), dtype=np.float32),
            })
        state["caches"] = {
            name: np.zeros(self._cache_shape_for_input(name, batch_size), dtype=np.float32)
            for name in self.cache_inputs
        }
        return state

    @staticmethod
    def _align_hop(y, hop_samples):
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[None, :]
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        hop_samples = int(hop_samples)
        if y.shape[-1] < hop_samples:
            y = np.pad(y, ((0, 0), (0, hop_samples - y.shape[-1])), mode="constant")
        elif y.shape[-1] > hop_samples:
            y = y[..., :hop_samples]
        return np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _run_wave_session(self, x, state, input_name=None):
        input_name = input_name or self.spec.get("input", self.input_names[0])
        if input_name not in self.input_names:
            input_name = self.input_names[0]
        feeds = {input_name: np.asarray(x, dtype=np.float32)}
        feeds.update(state.get("caches", {}))
        outputs = self.session.run(None, feeds)
        y = outputs[0]
        caches = dict(state.get("caches", {}))
        for idx, cache_input in enumerate(self.cache_inputs):
            if idx < len(self.cache_outputs) and self.cache_outputs[idx] in self.output_names:
                out_idx = self.output_names.index(self.cache_outputs[idx])
            else:
                out_idx = idx + 1
            if out_idx < len(outputs):
                caches[cache_input] = np.asarray(outputs[out_idx], dtype=np.float32)
        state["caches"] = caches
        return self._align_hop(y, self.hop_samples), state

    def _run_ulunas_stft(self, frame, state):
        n_fft = int(state["n_fft"])
        hop = int(state["hop_samples"])
        x = np.asarray(frame, dtype=np.float32)
        if x.shape[-1] < n_fft:
            x = np.pad(x, ((0, 0), (n_fft - x.shape[-1], 0)), mode="constant")
        elif x.shape[-1] > n_fft:
            x = x[..., -n_fft:]
        spec_complex = np.fft.rfft(x * state["window"][None, :], n=n_fft, axis=1).astype(np.complex64)
        spec = np.empty((x.shape[0], n_fft // 2 + 1, 1, 2), dtype=np.float32)
        spec[..., 0] = spec_complex.real[:, :, None]
        spec[..., 1] = spec_complex.imag[:, :, None]
        mix_name = self.spec.get("input", "mix")
        if mix_name not in self.input_names:
            mix_name = self.input_names[0]
        feeds = {mix_name: spec}
        feeds.update(state.get("caches", {}))
        outputs = self.session.run(None, feeds)
        caches = dict(state.get("caches", {}))
        for idx, cache_input in enumerate(self.cache_inputs):
            if idx < len(self.cache_outputs) and self.cache_outputs[idx] in self.output_names:
                out_idx = self.output_names.index(self.cache_outputs[idx])
            else:
                out_idx = idx + 1
            if out_idx < len(outputs):
                caches[cache_input] = np.asarray(outputs[out_idx], dtype=np.float32)
        state["caches"] = caches

        enhanced = np.asarray(outputs[0], dtype=np.float32).reshape(x.shape[0], n_fft // 2 + 1, 1, 2)
        enhanced_complex = enhanced[:, :, 0, 0] + 1j * enhanced[:, :, 0, 1]
        frame_out = np.fft.irfft(enhanced_complex, n=n_fft, axis=1).astype(np.float32)
        frame_out *= state["window_istft"][None, :]
        istft_cache = np.asarray(state["istft_cache"], dtype=np.float32)
        if istft_cache.size:
            frame_out[:, :istft_cache.shape[-1]] += istft_cache
        y = frame_out[:, :hop]
        state["istft_cache"] = frame_out[:, hop:].astype(np.float32)
        return self._align_hop(y, self.hop_samples), state

    def _run_fastenhancer_stft(self, hop, state):
        n_fft = int(state["n_fft"])
        hop_size = int(state["hop_samples"])
        compression = float(state.get("compression", 0.3))
        eps = float(state.get("eps", 1.0e-5))
        hop = np.asarray(hop, dtype=np.float32)
        if hop.shape[-1] < hop_size:
            hop = np.pad(hop, ((0, 0), (0, hop_size - hop.shape[-1])), mode="constant")
        elif hop.shape[-1] > hop_size:
            hop = hop[..., -hop_size:]

        stft_cache = np.asarray(state["stft_cache"], dtype=np.float32)
        frame = np.concatenate([stft_cache, hop], axis=1) if stft_cache.size else hop
        if frame.shape[-1] < n_fft:
            frame = np.pad(frame, ((0, 0), (n_fft - frame.shape[-1], 0)), mode="constant")
        elif frame.shape[-1] > n_fft:
            frame = frame[..., -n_fft:]
        cache_len = max(n_fft - hop_size, 0)
        state["stft_cache"] = frame[:, -cache_len:].astype(np.float32) if cache_len > 0 else frame[:, :0]

        spec_complex = np.fft.rfft(frame * state["window"][None, :], n=n_fft, axis=1).astype(np.complex64)
        spec = np.empty((frame.shape[0], n_fft // 2 + 1, 1, 2), dtype=np.float32)
        spec[..., 0] = spec_complex.real[:, :, None]
        spec[..., 1] = spec_complex.imag[:, :, None]
        if bool(state.get("discard_last_freq_bin", True)):
            spec = spec[:, :-1, :, :]
        mag = np.maximum(np.linalg.norm(spec, axis=-1, keepdims=True), eps).astype(np.float32)
        spec_noisy = (spec * np.power(mag, compression - 1.0)).astype(np.float32)

        spec_name = self.spec.get("input", "spec_noisy")
        if spec_name not in self.input_names:
            spec_name = self.input_names[0]
        feeds = {spec_name: spec_noisy}
        feeds.update(state.get("caches", {}))
        outputs = self.session.run(None, feeds)
        caches = dict(state.get("caches", {}))
        for idx, cache_input in enumerate(self.cache_inputs):
            if idx < len(self.cache_outputs) and self.cache_outputs[idx] in self.output_names:
                out_idx = self.output_names.index(self.cache_outputs[idx])
            else:
                out_idx = idx + 1
            if out_idx < len(outputs):
                caches[cache_input] = np.asarray(outputs[out_idx], dtype=np.float32)
        state["caches"] = caches

        mask = np.asarray(outputs[0], dtype=np.float32)
        spec_c = spec_noisy[..., 0] + 1j * spec_noisy[..., 1]
        mask_c = mask[..., 0] + 1j * mask[..., 1]
        spec_hat = spec_c * mask_c
        mag_hat = np.maximum(np.abs(spec_hat), eps)
        spec_hat = spec_hat * np.power(mag_hat, 1.0 / max(compression, 1.0e-8) - 1.0)
        if bool(state.get("discard_last_freq_bin", True)):
            spec_hat = np.pad(spec_hat, ((0, 0), (0, 1), (0, 0)), mode="constant")
        frame_out = np.fft.irfft(spec_hat[:, :, 0].astype(np.complex64), n=n_fft, axis=1).astype(np.float32)
        frame_out *= state["window_istft"][None, :]
        istft_cache = np.asarray(state["istft_cache"], dtype=np.float32)
        if istft_cache.size:
            frame_out[:, :istft_cache.shape[-1]] += istft_cache
        y = frame_out[:, :hop_size]
        state["istft_cache"] = frame_out[:, hop_size:].astype(np.float32)
        return self._align_hop(y, self.hop_samples), state

    def _run_lisen_stft(self, frame, state):
        n_fft = int(state["n_fft"])
        hop = int(state["hop_samples"])
        compress_factor = float(state.get("compress_factor", 0.3))
        x = np.asarray(frame, dtype=np.float32)
        if x.shape[-1] < n_fft:
            x = np.pad(x, ((0, 0), (n_fft - x.shape[-1], 0)), mode="constant")
        elif x.shape[-1] > n_fft:
            x = x[..., -n_fft:]

        spec_complex = np.fft.rfft(x * state["window"][None, :], n=n_fft, axis=1).astype(np.complex64)
        spec_mag = np.maximum(np.abs(spec_complex), 1.0e-12).astype(np.float32)
        src_mag = np.power(spec_mag, compress_factor).astype(np.float32)
        cur_phase = np.angle(spec_complex).astype(np.float32)

        gd = np.diff(
            cur_phase,
            axis=1,
            prepend=np.zeros((cur_phase.shape[0], 1), dtype=np.float32),
        )
        prev_phase = np.asarray(state["prev_phase"], dtype=np.float32)
        freq_axis = np.arange(cur_phase.shape[-1], dtype=np.float32)[None, :]
        ifd = (cur_phase - prev_phase) - 2.0 * np.pi * (float(hop) / float(n_fft)) * freq_axis
        gd = np.arctan2(np.sin(gd), np.cos(gd)).astype(np.float32)
        ifd = np.arctan2(np.sin(ifd), np.cos(ifd)).astype(np.float32)
        state["prev_phase"] = cur_phase.astype(np.float32)

        features = np.stack([src_mag, gd / np.pi, ifd / np.pi], axis=1)[:, :, None, :].astype(np.float32)
        feat_name = self.spec.get("input", "features")
        if feat_name not in self.input_names:
            feat_name = self.input_names[0]
        feeds = {feat_name: features}
        feeds.update(state.get("caches", {}))
        outputs = self.session.run(None, feeds)
        caches = dict(state.get("caches", {}))
        for idx, cache_input in enumerate(self.cache_inputs):
            if idx < len(self.cache_outputs) and self.cache_outputs[idx] in self.output_names:
                out_idx = self.output_names.index(self.cache_outputs[idx])
            else:
                out_idx = idx + 1
            if out_idx < len(outputs):
                caches[cache_input] = np.asarray(outputs[out_idx], dtype=np.float32)
        state["caches"] = caches

        mask = np.asarray(outputs[0], dtype=np.float32)
        if mask.ndim != 4:
            mask = mask.reshape(mask.shape[0], 2, 1, -1)
        est_mag = (mask[:, 0, 0, :] + 1.0e-8) * src_mag + (mask[:, 1, 0, :] + 1.0e-8) * src_mag
        est_mag = np.power(np.maximum(est_mag, 1.0e-12), 1.0 / max(compress_factor, 1.0e-8))
        est_complex = est_mag * (np.cos(cur_phase) + 1j * np.sin(cur_phase))
        frame_out = np.fft.irfft(est_complex.astype(np.complex64), n=n_fft, axis=1).astype(np.float32)
        frame_out *= state["window_istft"][None, :]
        istft_cache = np.asarray(state["istft_cache"], dtype=np.float32)
        if istft_cache.size:
            frame_out[:, :istft_cache.shape[-1]] += istft_cache
        y = frame_out[:, :hop]
        state["istft_cache"] = frame_out[:, hop:].astype(np.float32)
        return self._align_hop(y, self.hop_samples), state

    def stream_step(self, frame, hop, state):
        if self.protocol == "ulunas_stft":
            y, state = self._run_ulunas_stft(frame, state)
        elif self.protocol == "lisen_stft":
            y, state = self._run_lisen_stft(frame, state)
        elif self.protocol == "fastenhancer_stft":
            y, state = self._run_fastenhancer_stft(hop, state)
        elif self.protocol == "wave_frame":
            y, state = self._run_wave_session(frame, state)
        else:
            y, state = self._run_wave_session(hop, state)
        state["cache_hit"] = bool(self.true_cache)
        state["num_steps"] = int(state.get("num_steps", 0)) + 1
        return y, state


class OnnxMoEStreamRuntime:
    """Manifest-driven ONNX online inference branch for FrozenExpertRouterGRPO."""

    def __init__(
        self,
        manifest_path,
        providers=None,
        provider_options=None,
        override_manifest_providers=False,
        parallel_cuda_streams=False,
        sample_rate=16000,
        frame_samples=512,
        hop_samples=256,
        stft_conf=None,
        parallel_experts=True,
        parallel_workers=1,
        use_onnx_router=True,
    ):
        if np is None:
            raise RuntimeError("numpy is required for ONNX stream inference.")
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.manifest_path = manifest_path
        self.base_dir = manifest_path.parent
        self.manifest = manifest
        self.sample_rate = int(manifest.get("sample_rate", sample_rate))
        self.frame_samples = int(manifest.get("frame_samples", frame_samples))
        self.hop_samples = int(manifest.get("hop_samples", hop_samples))
        self.providers = providers or manifest.get("providers", ["CPUExecutionProvider"])
        self.provider_options = provider_options
        self.override_manifest_providers = bool(override_manifest_providers or provider_options is not None)
        self.parallel_cuda_streams = bool(parallel_cuda_streams)
        self._cuda_streams = []
        self.parallel_experts = bool(parallel_experts)
        self.parallel_workers = max(1, int(parallel_workers))
        self.use_onnx_router = bool(use_onnx_router)
        self._executor = None
        router_stream = self._new_cuda_stream() if self.parallel_cuda_streams and self.use_onnx_router else None
        if self.use_onnx_router:
            router_spec = self._apply_provider_override(manifest.get("router", {}), cuda_stream=router_stream)
            self.router = OnnxRouterSession(
                router_spec,
                self.base_dir,
                self.providers,
                sample_rate=self.sample_rate,
                frame_samples=self.frame_samples,
                hop_samples=self.hop_samples,
                stft_conf=manifest.get("stft", stft_conf or {}),
            )
        else:
            self.router = None
        experts = manifest.get("experts", [])
        if not experts:
            raise ValueError(f"ONNX MoE manifest has no experts: {manifest_path}")
        self.experts = []
        for spec in experts:
            stream = self._new_cuda_stream() if self.parallel_cuda_streams else None
            self.experts.append(
                OnnxExpertStreamSession(
                    self._apply_provider_override(spec, cuda_stream=stream),
                    self.base_dir,
                    self.providers,
                    frame_samples=self.frame_samples,
                    hop_samples=self.hop_samples,
                    sample_rate=self.sample_rate,
                )
            )

    def _cuda_provider_index(self, providers):
        for idx, provider in enumerate(providers or []):
            if str(provider) == "CUDAExecutionProvider":
                return idx
        return None

    def _provider_device_id(self, providers=None, provider_options=None):
        providers = providers or self.providers
        provider_options = provider_options if provider_options is not None else self.provider_options
        cuda_idx = self._cuda_provider_index(providers)
        if cuda_idx is None:
            return None
        if isinstance(provider_options, list) and cuda_idx < len(provider_options):
            try:
                return int(provider_options[cuda_idx].get("device_id", 0))
            except Exception:
                return 0
        if isinstance(provider_options, dict):
            try:
                return int(provider_options.get("device_id", 0))
            except Exception:
                return 0
        return 0

    def _new_cuda_stream(self):
        if not torch.cuda.is_available():
            return None
        if self._cuda_provider_index(self.providers) is None:
            return None
        device_id = self._provider_device_id()
        if device_id is None:
            return None
        with torch.cuda.device(int(device_id)):
            stream = torch.cuda.Stream(device=int(device_id))
        self._cuda_streams.append(stream)
        return stream

    def _apply_provider_override(self, spec, cuda_stream=None):
        spec = dict(spec or {})
        if self.override_manifest_providers:
            spec["providers"] = self.providers
            if self.provider_options is None:
                spec.pop("provider_options", None)
            else:
                if isinstance(self.provider_options, list):
                    spec["provider_options"] = [dict(item) for item in self.provider_options]
                elif isinstance(self.provider_options, dict):
                    spec["provider_options"] = dict(self.provider_options)
                else:
                    spec["provider_options"] = self.provider_options
        if cuda_stream is not None:
            providers = spec.get("providers", self.providers)
            cuda_idx = self._cuda_provider_index(providers)
            if cuda_idx is not None:
                provider_options = spec.get("provider_options")
                if not isinstance(provider_options, list):
                    provider_options = [{} for _ in providers]
                else:
                    provider_options = [dict(item) for item in provider_options]
                while len(provider_options) < len(providers):
                    provider_options.append({})
                provider_options[cuda_idx]["device_id"] = self._provider_device_id(
                    providers=providers,
                    provider_options=provider_options,
                ) or 0
                provider_options[cuda_idx]["user_compute_stream"] = str(cuda_stream.cuda_stream)
                provider_options[cuda_idx].setdefault("do_copy_in_default_stream", "1")
                spec["provider_options"] = provider_options
        return spec

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def reload_router(self, path=None):
        if self.router is None:
            return
        self.router.reload(path)

    def adapter_modes(self):
        return [
            {
                "expert": expert.name,
                "mode": f"onnx_{expert.protocol}",
                "true_cache": bool(expert.true_cache),
                "device": ",".join(expert.session.get_providers()),
            }
            for expert in self.experts
        ]

    def fallback_expert_count(self):
        return sum(1 for expert in self.experts if not expert.true_cache)

    def create_state(self, batch_size=1):
        batch_size = int(batch_size)
        context_len = max(self.frame_samples - self.hop_samples, 0)
        return MoEStreamState(
            input_tail=torch.zeros(batch_size, context_len, dtype=torch.float32),
            expert_states=[
                TensorUtils.numpy_tree_to_tensor(expert.init_state(batch_size=batch_size))
                for expert in self.experts
            ],
            router_state=None,
            output_tail=torch.zeros(batch_size, context_len, dtype=torch.float32),
            ola_buffer=torch.zeros(batch_size, context_len, dtype=torch.float32),
            num_steps=0,
        )

    def _to_numpy_state(self, state):
        return TensorUtils.tensor_tree_to_numpy(state)

    def _to_torch_state(self, state):
        return TensorUtils.numpy_tree_to_tensor(state)

    def _ensure_executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="mos_moe_onnx_expert",
            )
        return self._executor

    def _expert_task(self, idx, frame, hop, state):
        start = time.perf_counter()
        y, state = self.experts[idx].stream_step(frame, hop, self._to_numpy_state(state))
        return idx, y, self._to_torch_state(state), (time.perf_counter() - start) * 1000.0

    def _prepare_frame(self, hop, state=None):
        hop = np.asarray(hop, dtype=np.float32)
        if hop.ndim == 1:
            hop = hop[None, :]
        if hop.ndim != 2:
            raise ValueError(f"ONNX MoE step expects [T] or [B,T] hop, got {hop.shape}.")
        batch_size = int(hop.shape[0])
        if hop.shape[-1] < self.hop_samples:
            hop = np.pad(hop, ((0, 0), (0, self.hop_samples - hop.shape[-1])), mode="constant")
        elif hop.shape[-1] > self.hop_samples:
            hop = hop[..., -self.hop_samples:]

        if state is None or state.input_tail.shape[0] != batch_size:
            state = self.create_state(batch_size=batch_size)

        input_start = time.perf_counter()
        tail = np.asarray(TensorUtils.tensor_tree_to_numpy(state.input_tail), dtype=np.float32)
        frame = np.concatenate([tail, hop], axis=-1) if tail.size else hop
        if frame.shape[-1] < self.frame_samples:
            frame = np.pad(frame, ((0, 0), (self.frame_samples - frame.shape[-1], 0)), mode="constant")
        elif frame.shape[-1] > self.frame_samples:
            frame = frame[..., -self.frame_samples:]
        context_len = max(self.frame_samples - self.hop_samples, 0)
        input_tail = frame[..., -context_len:].astype(np.float32) if context_len > 0 else frame[:, :0]
        state.input_tail = TensorUtils.numpy_tree_to_tensor(input_tail.copy())
        input_cache_ms = (time.perf_counter() - input_start) * 1000.0
        return hop, frame, state, input_cache_ms, context_len

    def _run_experts(self, frame, hop, state):
        expert_start = time.perf_counter()
        if self.parallel_experts and len(self.experts) > 1 and self.parallel_workers > 1:
            executor = self._ensure_executor()
            futures = [
                executor.submit(self._expert_task, idx, frame, hop, state.expert_states[idx])
                for idx in range(len(self.experts))
            ]
            results = [future.result() for future in futures]
        else:
            results = [
                self._expert_task(idx, frame, hop, state.expert_states[idx])
                for idx in range(len(self.experts))
            ]
        results.sort(key=lambda item: item[0])
        expert_wavs = []
        expert_step_ms = {}
        for idx, y, expert_state, value_ms in results:
            state.expert_states[idx] = expert_state
            expert_wavs.append(y)
            expert_step_ms[idx] = float(value_ms)
        expert_wavs = np.stack(expert_wavs, axis=1).astype(np.float32)
        expert_ms = (time.perf_counter() - expert_start) * 1000.0
        return expert_wavs, expert_ms, expert_step_ms

    def _add_expert_latency_profile(self, profile, expert_step_ms):
        for idx, expert in enumerate(self.experts):
            value_ms = float(expert_step_ms.get(idx, 0.0))
            safe = str(expert.name).replace("-", "_")
            profile[f"{safe}_onnx_ms"] = value_ms
            short = safe[:-7] if safe.endswith("_expert") else safe
            profile[f"{short}_onnx_ms"] = value_ms
            profile[f"{short}_expert_onnx_ms"] = value_ms
            profile[f"expert_{short}_ms"] = value_ms

    def step_experts(self, hop, state=None):
        total_start = time.perf_counter()
        hop, frame, state, input_cache_ms, _ = self._prepare_frame(hop, state=state)
        expert_wavs, expert_ms, expert_step_ms = self._run_experts(frame, hop, state)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        profile = {
            "input_cache_ms": input_cache_ms,
            "expert_stream_step_ms": expert_ms,
            "experts_parallel_wall_ms": expert_ms,
            "router_ms": 0.0,
            "fusion_ms": 0.0,
            "total_step_ms": total_ms,
            "stream_frame_total_ms": total_ms,
            "frame_total_ms": total_ms,
            "cache_hit": 1.0 if self.fallback_expert_count() == 0 else 0.0,
            "fallback_expert_count": float(self.fallback_expert_count()),
            "expert_step_ms": expert_step_ms,
            "expert_output_domain": "wave",
        }
        self._add_expert_latency_profile(profile, expert_step_ms)
        return expert_wavs, frame, state, profile

    def step(self, hop, state=None):
        if self.router is None:
            raise RuntimeError(
                "OnnxMoEStreamRuntime was created with use_onnx_router=False. "
                "Call step_experts(...) and run the PyTorch router/fusion outside ONNXRuntime."
            )
        total_start = time.perf_counter()
        hop, frame, state, input_cache_ms, context_len = self._prepare_frame(hop, state=state)
        expert_wavs, expert_ms, expert_step_ms = self._run_experts(frame, hop, state)

        router_start = time.perf_counter()
        weights, logits = self.router.weights(frame)
        state.router_state = TensorUtils.numpy_tree_to_tensor({"weights": weights.copy(), "logits": logits.copy()})
        router_ms = (time.perf_counter() - router_start) * 1000.0

        fusion_start = time.perf_counter()
        if weights.shape[-1] != expert_wavs.shape[1]:
            raise RuntimeError(
                f"ONNX router returned {weights.shape[-1]} weights for {expert_wavs.shape[1]} experts."
            )
        fused = np.sum(weights[:, :, None] * expert_wavs, axis=1)
        fused = np.clip(np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)
        output_tail = fused[..., -context_len:].copy() if context_len > 0 else fused[:, :0].copy()
        state.output_tail = TensorUtils.numpy_tree_to_tensor(output_tail)
        state.ola_buffer = state.output_tail
        state.num_steps += 1
        fusion_ms = (time.perf_counter() - fusion_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0
        profile = {
            "input_cache_ms": input_cache_ms,
            "expert_stream_step_ms": expert_ms,
            "router_ms": router_ms,
            "fusion_ms": fusion_ms,
            "total_step_ms": total_ms,
            "cache_hit": 1.0 if self.fallback_expert_count() == 0 else 0.0,
            "fallback_expert_count": float(self.fallback_expert_count()),
            "expert_step_ms": expert_step_ms,
        }
        self._add_expert_latency_profile(profile, expert_step_ms)
        return fused[:, :self.hop_samples], weights, state, profile


class StreamingExpertAdapter:
    """Common streaming interface for one frozen expert.

    Fallback mode deliberately calls the original full expert forward on the
    32 ms analysis frame and releases only the newest hop. It preserves the old
    frame-by-frame behavior but does not speed up inference.
    """

    def __init__(self, owner, expert_idx, expert, allow_model_stream=True):
        self.owner = owner
        self.expert_idx = int(expert_idx)
        self.expert = expert
        self.model = getattr(expert, "model", None)
        self.device = owner._expert_device(self.expert_idx)
        self.output_device = owner._router_device()
        self.frame_samples = int(owner.stream_frame_samples)
        self.hop_samples = int(owner.stream_hop_samples)
        self.allow_model_stream = bool(allow_model_stream)
        self._model_init_stream = getattr(self.model, "init_stream_state", None)
        self._model_stream_step = getattr(self.model, "stream_step", None)
        self.uses_true_stream_cache = (
            self.allow_model_stream
            and callable(self._model_init_stream)
            and callable(self._model_stream_step)
        )
        self.kind = "model_stream" if self.uses_true_stream_cache else "fallback_full_forward"

    @property
    def fallback(self):
        return not self.uses_true_stream_cache

    def init_state(self, device, batch_size=1, dtype=torch.float32):
        device = torch.device(device or self.device)
        if self.uses_true_stream_cache:
            model_state = self._call_model_init_state(device, batch_size, dtype)
            return {
                "kind": self.kind,
                "model_state": model_state,
                "cache_hit": True,
                "num_steps": 0,
            }
        return {
            "kind": self.kind,
            "cache_hit": False,
            "num_steps": 0,
        }

    def _call_model_init_state(self, device, batch_size, dtype):
        kwargs = {
            "device": device,
            "batch_size": int(batch_size),
            "dtype": dtype,
            "frame_samples": self.frame_samples,
            "hop_samples": self.hop_samples,
            "sample_rate": int(self.owner.sample_rate),
        }
        try:
            return self._model_init_stream(**kwargs)
        except TypeError:
            try:
                return self._model_init_stream(device=device, batch_size=int(batch_size))
            except TypeError:
                return self._model_init_stream(device, int(batch_size))

    def _call_model_stream_step(self, frame, state):
        model_state = state.get("model_state")
        try:
            out = self._model_stream_step(frame, model_state)
        except TypeError:
            out = self._model_stream_step(frame, state=model_state)

        if isinstance(out, dict):
            y_hop = out.get("y_hop", out.get("est", out.get("wav")))
            new_model_state = out.get("state", out.get("stream_state", model_state))
        elif isinstance(out, tuple) and len(out) >= 2:
            y_hop, new_model_state = out[0], out[1]
        else:
            y_hop, new_model_state = out, model_state
        state["model_state"] = new_model_state
        state["cache_hit"] = True
        state["num_steps"] = int(state.get("num_steps", 0)) + 1
        y_hop = TensorUtils.align_waveform_length(TensorUtils.safe_nan_to_num(y_hop), self.hop_samples)
        return y_hop.detach(), state

    def _fallback_stream_step(self, frame, state):
        with torch.no_grad():
            self.expert.eval()
            est = self.owner._expert_enhanced_wav(
                self.expert,
                frame,
                target_len=frame.shape[-1],
            )
            est = TensorUtils.align_waveform_length(est, frame.shape[-1])
            y_hop = est[..., -self.hop_samples:]
            y_hop = TensorUtils.align_waveform_length(TensorUtils.safe_nan_to_num(y_hop), self.hop_samples)
        state["cache_hit"] = False
        state["num_steps"] = int(state.get("num_steps", 0)) + 1
        return y_hop.detach(), state

    def stream_step(self, frame_or_hop, state):
        self.device = self.owner._expert_device(self.expert_idx)
        frame = torch.as_tensor(frame_or_hop, dtype=torch.float32, device=self.device)
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        if frame.shape[-1] < self.frame_samples:
            frame = F.pad(frame, (self.frame_samples - frame.shape[-1], 0))
        elif frame.shape[-1] > self.frame_samples:
            frame = frame[..., -self.frame_samples:]
        if state.get("kind") == "model_stream":
            return self._call_model_stream_step(frame, state)
        return self._fallback_stream_step(frame, state)


class FastEnhancerStreamingExpertAdapter(StreamingExpertAdapter):
    def __init__(self, owner, expert_idx, expert):
        super().__init__(owner, expert_idx, expert, allow_model_stream=False)
        self.kind = "fastenhancer_native"
        self.uses_true_stream_cache = True

    @property
    def fallback(self):
        return False

    def init_state(self, device, batch_size=1, dtype=torch.float32):
        device = torch.device(device or self.device)
        state = self.owner._new_fastenhancer_stream_state(self.expert, device=device, dtype=dtype)
        state["cache_hit"] = True
        state["num_steps"] = 0
        return state

    def stream_step(self, frame_or_hop, state):
        self.device = self.owner._expert_device(self.expert_idx)
        frame = torch.as_tensor(frame_or_hop, dtype=torch.float32, device=self.device)
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        hop = frame[..., -self.hop_samples:].reshape(-1)
        y_hop, state = self.owner._fastenhancer_stream_step(self.expert, hop, state)
        y_hop = TensorUtils.align_waveform_length(TensorUtils.safe_nan_to_num(y_hop), self.hop_samples)
        state["cache_hit"] = True
        state["num_steps"] = int(state.get("num_steps", 0)) + 1
        return y_hop.detach(), state


class ULUNASStreamingExpertAdapter(StreamingExpertAdapter):
    """UL-UNAS frame cache adapted from ulunas_onnx/stream/ulunas_stream.py."""

    CONV_CACHE_SHAPES = [
        (1, 2, 129),
        (24, 1, 65),
        (24, 1, 33),
        (24, 1, 33),
        (12, 1, 33),
        (12, 2, 65),
    ]
    TFA_CACHE_HIDDEN = [24, 48, 48, 64, 32, 64, 48, 48, 24, 2]
    INTER_CACHE_SHAPES = [(33, 16), (33, 16)]

    def __init__(self, owner, expert_idx, expert):
        super().__init__(owner, expert_idx, expert, allow_model_stream=False)
        self.kind = "ulunas_native"
        self.uses_true_stream_cache = True

    @property
    def fallback(self):
        return False

    def _core(self):
        return getattr(getattr(self.expert, "model", None), "enhancer", None)

    @classmethod
    def _zeros_cache(cls, shapes, batch_size, device, dtype):
        total = sum(math.prod(shape) for shape in shapes)
        return torch.zeros(batch_size, total, device=device, dtype=dtype)

    @classmethod
    def _unpack_cache(cls, cache, shapes):
        bsz = cache.shape[0]
        parts = []
        offset = 0
        for shape in shapes:
            n = math.prod(shape)
            parts.append(cache[:, offset:offset + n].view(bsz, *shape))
            offset += n
        return parts

    @staticmethod
    def _pack_cache(parts):
        return torch.cat([part.reshape(part.shape[0], -1) for part in parts], dim=1)

    @classmethod
    def _unpack_tfa_cache(cls, cache):
        bsz = cache.shape[0]
        parts = []
        offset = 0
        for hidden in cls.TFA_CACHE_HIDDEN:
            parts.append(cache[:, offset:offset + hidden].view(1, bsz, hidden))
            offset += hidden
        return parts

    @staticmethod
    def _pack_tfa_cache(parts):
        return torch.cat([part.reshape(part.shape[1], -1) for part in parts], dim=1)

    @staticmethod
    def _stream_temporal_conv(x, conv, cache):
        inp = torch.cat([cache, x], dim=2)
        if isinstance(conv, nn.ConvTranspose2d):
            kt = conv.kernel_size[0]
            inp_padded = F.pad(inp, (0, 0, kt - 1, 0))
            y = conv(inp_padded)[:, :, -1:, :]
        else:
            y = conv(inp)
        return y, inp[:, :, 1:, :]

    @staticmethod
    def _stream_ctfa(ctfa, x, h_cache):
        zt = torch.mean(x.pow(2), dim=-1)
        at, h_cache = ctfa.ta_gru(zt.transpose(1, 2), h_cache)
        at = torch.sigmoid(ctfa.ta_fc(at).transpose(1, 2))
        af = torch.sigmoid(ctfa.fa(x))
        return at[..., None] * x * af[:, None], h_cache

    def _stream_xconv(self, block, x, conv_cache, tfa_cache):
        x, conv_cache = self._stream_temporal_conv(x, block.ops[1], conv_cache)
        x = block.ops[2](x)
        x = block.ops[3](x)
        x, tfa_cache = self._stream_ctfa(block.ops[4], x, tfa_cache)
        x = block.ops[5](x)
        return x, conv_cache, tfa_cache

    def _stream_xdws(self, block, x, tfa_cache, conv_cache=None):
        h = block.pconv(x)
        if conv_cache is None:
            h = block.dconv[0](h)
            h = block.dconv[1](h)
        else:
            h, conv_cache = self._stream_temporal_conv(h, block.dconv[1], conv_cache)
        h = block.dconv[2](h)
        h = block.dconv[3](h)
        h, tfa_cache = self._stream_ctfa(block.dconv[4], h, tfa_cache)
        return h, conv_cache, tfa_cache

    def _stream_xmb(self, block, x, tfa_cache, conv_cache=None):
        residual = x
        x = block.pconv1(x)
        if conv_cache is None:
            x = block.dconv(x)
        else:
            x, conv_cache = self._stream_temporal_conv(x, block.dconv[1], conv_cache)
            x = block.dconv[2](x)
            x = block.dconv[3](x)
        x = block.pconv2[0](x)
        x = block.pconv2[1](x)
        x, tfa_cache = self._stream_ctfa(block.pconv2[2], x, tfa_cache)
        if x.shape == residual.shape:
            x = x + residual
        return block.shuffle(x), conv_cache, tfa_cache

    @staticmethod
    def _stream_dpgrnn(block, x, inter_cache):
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = block.intra_rnn(intra_x)[0]
        intra_x = block.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, block.width, block.input_size)
        intra_x = block.intra_ln(intra_x)
        intra_out = x + intra_x

        x = intra_out.permute(0, 2, 1, 3)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x, inter_cache = block.inter_rnn(inter_x, inter_cache)
        inter_x = block.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], block.width, -1, block.input_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = block.inter_ln(inter_x)
        return (intra_out + inter_x).permute(0, 3, 1, 2), inter_cache

    def init_state(self, device, batch_size=1, dtype=torch.float32):
        device = torch.device(device or self.device)
        core = self._core()
        n_fft = int(getattr(core, "n_fft", self.frame_samples))
        hop_len = int(getattr(core, "hop_len", getattr(core, "hop_length", self.hop_samples)))
        win_len = int(getattr(core, "win_len", getattr(core, "win_length", n_fft)))
        if hop_len != self.hop_samples:
            logger.warning(
                f"[FrozenExpertRouterGRPO] ULUNAS expert {self.expert.name} hop_len={hop_len} "
                f"differs from inference hop={self.hop_samples}; adapter will align the released hop."
            )
        conv_cache = self._zeros_cache(self.CONV_CACHE_SHAPES, batch_size, device, dtype)
        tfa_cache = torch.zeros(batch_size, sum(self.TFA_CACHE_HIDDEN), device=device, dtype=dtype)
        inter_cache = self._zeros_cache(self.INTER_CACHE_SHAPES, batch_size, device, dtype)
        window = torch.hann_window(win_len, device=device, dtype=dtype)
        if win_len < n_fft:
            left = (n_fft - win_len) // 2
            window = F.pad(window, (left, n_fft - win_len - left))
        elif win_len > n_fft:
            window = window[:n_fft]
        return {
            "kind": self.kind,
            "conv_cache": conv_cache,
            "tfa_cache": tfa_cache,
            "inter_cache": inter_cache,
            "istft_cache": torch.zeros(batch_size, n_fft - hop_len, device=device, dtype=dtype),
            "window": window,
            "window_istft": self.owner._stream_istft_window(window, n_fft, hop_len),
            "n_fft": n_fft,
            "hop_len": hop_len,
            "cache_hit": True,
            "num_steps": 0,
        }

    def _stream_spec_step(self, spec, state):
        core = self._core()
        spec_in = spec.permute(0, 3, 2, 1)
        feat = torch.log10(torch.norm(spec_in, dim=1, keepdim=True).clamp(1e-12))
        feat = core.erb.bm(feat)

        conv_parts = self._unpack_cache(state["conv_cache"], self.CONV_CACHE_SHAPES)
        tfa_parts = self._unpack_tfa_cache(state["tfa_cache"])
        inter_parts = self._unpack_cache(state["inter_cache"], self.INTER_CACHE_SHAPES)
        conv_e0, conv_e1, conv_e2, conv_d2, conv_d3, conv_d4 = conv_parts
        tfa0, tfa1, tfa2, tfa3, tfa4, tfa5, tfa6, tfa7, tfa8, tfa9 = tfa_parts
        inter0, inter1 = inter_parts

        en_outs = []
        feat, conv_e0, tfa0 = self._stream_xconv(core.encoder.en_convs[0], feat, conv_e0, tfa0)
        en_outs.append(feat)
        feat, conv_e1, tfa1 = self._stream_xmb(core.encoder.en_convs[1], feat, tfa1, conv_e1)
        en_outs.append(feat)
        feat, conv_e2, tfa2 = self._stream_xdws(core.encoder.en_convs[2], feat, tfa2, conv_e2)
        en_outs.append(feat)
        feat, _, tfa3 = self._stream_xmb(core.encoder.en_convs[3], feat, tfa3)
        en_outs.append(feat)
        feat, _, tfa4 = self._stream_xdws(core.encoder.en_convs[4], feat, tfa4)
        en_outs.append(feat)

        feat, inter0 = self._stream_dpgrnn(core.dpgrnn[0], feat, inter0)
        feat, inter1 = self._stream_dpgrnn(core.dpgrnn[1], feat, inter1)

        feat, _, tfa5 = self._stream_xdws(core.decoder.de_convs[0], feat + en_outs[4], tfa5)
        feat, _, tfa6 = self._stream_xmb(core.decoder.de_convs[1], feat + en_outs[3], tfa6)
        feat, conv_d2, tfa7 = self._stream_xdws(core.decoder.de_convs[2], feat + en_outs[2], tfa7, conv_d2)
        feat, conv_d3, tfa8 = self._stream_xmb(core.decoder.de_convs[3], feat + en_outs[1], tfa8, conv_d3)
        feat, conv_d4, tfa9 = self._stream_xconv(core.decoder.de_convs[4], feat + en_outs[0], conv_d4, tfa9)

        mask = core.erb.bs(torch.sigmoid(feat))
        spec_out = (spec_in * mask).permute(0, 3, 2, 1)
        state["conv_cache"] = self._pack_cache([conv_e0, conv_e1, conv_e2, conv_d2, conv_d3, conv_d4]).detach()
        state["tfa_cache"] = self._pack_tfa_cache([tfa0, tfa1, tfa2, tfa3, tfa4, tfa5, tfa6, tfa7, tfa8, tfa9]).detach()
        state["inter_cache"] = self._pack_cache([inter0, inter1]).detach()
        return spec_out, state

    def stream_step(self, frame_or_hop, state):
        self.device = self.owner._expert_device(self.expert_idx)
        frame = torch.as_tensor(frame_or_hop, dtype=torch.float32, device=self.device)
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        n_fft = int(state["n_fft"])
        hop_len = int(state["hop_len"])
        if frame.shape[-1] < n_fft:
            frame = F.pad(frame, (n_fft - frame.shape[-1], 0))
        elif frame.shape[-1] > n_fft:
            frame = frame[..., -n_fft:]

        window = state["window"].to(device=frame.device, dtype=frame.dtype)
        spec = torch.fft.rfft(frame * window, n=n_fft, dim=1)
        spec = torch.view_as_real(spec).unsqueeze(2)
        spec_hat, state = self._stream_spec_step(spec, state)
        spec_hat = torch.complex(spec_hat[..., 0], spec_hat[..., 1]).squeeze(2)

        frame_hat = torch.fft.irfft(spec_hat, n=n_fft, dim=1)
        frame_hat = frame_hat * state["window_istft"].to(device=frame.device, dtype=frame.dtype)
        istft_cache = state["istft_cache"].to(device=frame.device, dtype=frame.dtype)
        frame_hat[:, :istft_cache.shape[-1]] += istft_cache
        y_hop = frame_hat[:, :hop_len]
        state["istft_cache"] = frame_hat[:, hop_len:].detach()
        state["cache_hit"] = True
        state["num_steps"] = int(state.get("num_steps", 0)) + 1
        y_hop = TensorUtils.align_waveform_length(TensorUtils.safe_nan_to_num(y_hop), self.hop_samples)
        return y_hop.detach(), state


class LiSenStreamingExpertAdapter(StreamingExpertAdapter):
    """Causal frame cache for the local LiSenNet implementation.

    The LiSenNet model body is streamed with cached causal convolutions and
    inter-frame GRU states. The original offline forward uses whole-utterance
    Griffin-Lim phase refinement; this adapter uses the current noisy phase for
    causal reconstruction because future frames are unavailable online.
    """

    def __init__(self, owner, expert_idx, expert):
        super().__init__(owner, expert_idx, expert, allow_model_stream=False)
        self.kind = "lisen_native"
        self.uses_true_stream_cache = True

    @property
    def fallback(self):
        return False

    def _core(self):
        return getattr(self.expert, "model", None)

    def init_state(self, device, batch_size=1, dtype=torch.float32):
        device = torch.device(device or self.device)
        core = self._core()
        n_fft = int(getattr(core, "n_fft", self.frame_samples))
        hop_len = int(getattr(core, "hop_length", self.hop_samples))
        window = torch.hann_window(n_fft, device=device, dtype=dtype)
        return {
            "kind": self.kind,
            "prev_phase": torch.zeros(batch_size, n_fft // 2 + 1, device=device, dtype=dtype),
            "istft_cache": torch.zeros(batch_size, n_fft - hop_len, device=device, dtype=dtype),
            "window": window,
            "window_istft": self.owner._stream_istft_window(window, n_fft, hop_len),
            "encoder_cache": {},
            "block_states": [self._new_block_state() for _ in range(len(core.blocks))],
            "decoder_cache": {},
            "n_fft": n_fft,
            "hop_len": hop_len,
            "cache_hit": True,
            "num_steps": 0,
        }

    @staticmethod
    def _new_block_state():
        return {
            "inter_hidden": None,
            "conv_glu_cache": None,
        }

    @staticmethod
    def _lazy_cache(state, key, x, frames):
        cache = state.get(key)
        if cache is None or cache.shape[0] != x.shape[0] or cache.shape[1] != x.shape[1] or cache.shape[3] != x.shape[3]:
            cache = x.new_zeros(x.shape[0], x.shape[1], frames, x.shape[3])
        return cache

    def _stream_dsconv(self, module, x, state, key):
        cache = self._lazy_cache(state, key, x, 1)
        y = module(torch.cat([cache, x], dim=2))[..., -1:, :]
        state[key] = x.detach()
        return y, state

    def _stream_conv_glu(self, module, x, state):
        res = x
        x = module.norm(x)
        x, v = module.fc1(x).chunk(2, dim=1)
        cache = state.get("conv_glu_cache")
        if cache is None or cache.shape[0] != x.shape[0] or cache.shape[1] != x.shape[1] or cache.shape[3] != x.shape[3]:
            cache = x.new_zeros(x.shape[0], x.shape[1], 2, x.shape[3])
        y = module.dwconv(torch.cat([cache, x], dim=2))[..., -1:, :]
        state["conv_glu_cache"] = torch.cat([cache, x], dim=2)[..., -2:, :].detach()
        y = module.act(y) * v
        y = module.dropout(y)
        y = module.fc2(y)
        return y + res, state

    def _stream_dual_path_rnn(self, module, x, state):
        bsz, emb, time, freq = x.size()
        if time != 1:
            raise RuntimeError(f"LiSen streaming expects one frame, got T={time}.")
        x = x.permute(0, 2, 3, 1)

        x_res = x
        y = module.intra_norm(x)
        y = y.reshape(bsz * time, freq, emb)
        y = module.intra_rnn_attn(y)
        y = y.reshape(bsz, time, freq, emb)
        y = y + x_res

        x_res = y
        y = module.inter_norm(y)
        y = y.permute(0, 2, 1, 3).reshape(bsz * freq, time, emb)
        hidden = state.get("inter_hidden")
        rnn = module.inter_rnn_attn.rnn
        if hidden is None or hidden.shape[1] != y.shape[0] or hidden.device != y.device or hidden.dtype != y.dtype:
            hidden = y.new_zeros(rnn.num_layers, y.shape[0], rnn.hidden_size)
        y, hidden = rnn(y, hidden)
        y = module.inter_rnn_attn.dense(y)
        state["inter_hidden"] = hidden.detach()
        y = y.reshape(bsz, freq, time, emb).permute(0, 2, 1, 3)
        y = y + x_res
        return y.permute(0, 3, 1, 2), state

    def _stream_dpr(self, module, x, state):
        x, state = self._stream_dual_path_rnn(module.dp_rnn_attn, x, state)
        x, state = self._stream_conv_glu(module.conv_glu, x, state)
        return x, state

    def _stream_encoder(self, encoder, x, state):
        out_list = []
        x = encoder.conv_1(x)
        x, state = self._stream_dsconv(encoder.conv_2, x, state, "enc_conv_2")
        out_list.append(x)
        x, state = self._stream_dsconv(encoder.conv_3, x, state, "enc_conv_3")
        out_list.append(x)
        x, state = self._stream_dsconv(encoder.conv_4, x, state, "enc_conv_4")
        out_list.append(x)
        return out_list, state

    def _stream_mask_conv(self, decoder, x, state):
        cache = self._lazy_cache(state, "mask_conv", x, 1)
        y = decoder.mask_conv(torch.cat([cache, x], dim=2))[..., -1:, :]
        state["mask_conv"] = x.detach()
        return y, state

    def _stream_decoder(self, decoder, x, encoder_out_list, state):
        skips = list(encoder_out_list)
        x = decoder.up1(torch.cat([x, skips.pop()], dim=1))
        x = decoder.up2(torch.cat([x, skips.pop()], dim=1))
        x = decoder.up3(torch.cat([x, skips.pop()], dim=1))
        x, state = self._stream_mask_conv(decoder, x, state)
        x = x.permute(0, 3, 2, 1)
        x = decoder.lsigmoid(x).permute(0, 3, 2, 1)
        return x, state

    def _stft_frame(self, frame, state):
        n_fft = int(state["n_fft"])
        window = state["window"].to(device=frame.device, dtype=frame.dtype)
        spec = torch.fft.rfft(frame * window, n=n_fft, dim=1)
        return spec.unsqueeze(1)

    def stream_step(self, frame_or_hop, state):
        core = self._core()
        self.device = self.owner._expert_device(self.expert_idx)
        frame = torch.as_tensor(frame_or_hop, dtype=torch.float32, device=self.device)
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        n_fft = int(state["n_fft"])
        hop_len = int(state["hop_len"])
        if frame.shape[-1] < n_fft:
            frame = F.pad(frame, (n_fft - frame.shape[-1], 0))
        elif frame.shape[-1] > n_fft:
            frame = frame[..., -n_fft:]

        src_spec = core.power_compress(self._stft_frame(frame, state))
        src_mag = src_spec.abs()
        src_pha = src_spec.angle()
        cur_phase = src_pha[:, 0, :]
        gd = torch.diff(
            cur_phase,
            dim=1,
            prepend=torch.zeros(cur_phase.shape[0], 1, device=cur_phase.device, dtype=cur_phase.dtype),
        )
        prev_phase = state["prev_phase"].to(device=cur_phase.device, dtype=cur_phase.dtype)
        freq_axis = torch.arange(cur_phase.shape[-1], device=cur_phase.device, dtype=cur_phase.dtype)
        ifd = (cur_phase - prev_phase) - 2 * torch.pi * (float(hop_len) / float(n_fft)) * freq_axis[None, :]
        gd = torch.atan2(gd.sin(), gd.cos()).unsqueeze(1)
        ifd = torch.atan2(ifd.sin(), ifd.cos()).unsqueeze(1)
        state["prev_phase"] = cur_phase.detach()

        x = torch.stack([src_mag, gd / torch.pi, ifd / torch.pi], dim=1)
        encoder_out_list, state["encoder_cache"] = self._stream_encoder(core.encoder, x, state["encoder_cache"])
        x = encoder_out_list[-1]
        for idx, block in enumerate(core.blocks):
            x, state["block_states"][idx] = self._stream_dpr(block, x, state["block_states"][idx])
        mask, state["decoder_cache"] = self._stream_decoder(core.decoder, x, encoder_out_list, state["decoder_cache"])

        est_mag = (mask[:, 0] + 1e-8) * src_mag + (mask[:, 1] + 1e-8) * src_mag
        noisy_phase = src_pha
        est_spec = torch.complex(est_mag * noisy_phase.cos(), est_mag * noisy_phase.sin())
        est_spec = core.power_uncompress(est_spec)
        frame_hat = torch.fft.irfft(est_spec.squeeze(1), n=n_fft, dim=1)
        frame_hat = frame_hat * state["window_istft"].to(device=frame.device, dtype=frame.dtype)
        istft_cache = state["istft_cache"].to(device=frame.device, dtype=frame.dtype)
        frame_hat[:, :istft_cache.shape[-1]] += istft_cache
        y_hop = frame_hat[:, :hop_len]
        state["istft_cache"] = frame_hat[:, hop_len:].detach()
        state["cache_hit"] = True
        state["num_steps"] = int(state.get("num_steps", 0)) + 1
        y_hop = TensorUtils.align_waveform_length(TensorUtils.safe_nan_to_num(y_hop), self.hop_samples)
        return y_hop.detach(), state


class SlidingWindowGRPO(UniSE):
    """Speech-enhancement GRPO adaptor (FlowSE-GRPO + RLHF-SE).

    Action: equivalent complex mask = enhanced_spec / noisy_spec, with Gaussian exploration.
    Rewards: DNSMOS-based, optional distortion penalty, relative or absolute.
    """

    def __init__(self, conf):
        super().__init__(conf)
        self._init_grpo_state(conf)

    def _init_grpo_state(self, conf, ref_model=None):
        self.automatic_optimization = False

        # Disable Lightning's gradient clipping
        if conf['trainer'].get('gradient_clip_val', 0):
            logger.warning("manual_optimization: disabling trainer.gradient_clip_val")
            conf['trainer']['gradient_clip_val'] = 0
        if conf['trainer'].get('strategy') is None:
            conf['trainer']['strategy'] = 'ddp_find_unused_parameters_true'
        if "DNSMOS" not in conf:
            raise ValueError("DNSMOS config is required")

        grpo = conf.get("grpo", {})
        # Window & sampling
        self.window_size = int(grpo.get("window_size", 5))
        self.min_window = int(grpo.get("min_window", self.window_size))
        self.num_actions = int(grpo.get("num_actions", 4))
        self.sigma = float(grpo.get("sigma", 0.1))
        # PPO
        self.clip_eps = float(grpo.get("clip_eps", 0.2))
        self.update_epochs = int(grpo.get("update_epochs", 1))
        self.train_minibatch_size = int(grpo.get("train_minibatch_size", 0))
        # KL
        self.beta = float(grpo.get("beta", grpo.get("beta_kl", 0.0)))
        self.ref_update_steps = int(grpo.get("ref_update_steps", 0))
        # Mask stabilization
        self.mask_den_floor = float(grpo.get("mask_den_floor", 1e-6))
        self.mask_abs_clip = grpo.get("mask_abs_clip", 5.0)
        self.mask_abs_clip = None if self.mask_abs_clip is None else float(self.mask_abs_clip)
        # Reward
        self.dns_weight = float(grpo.get("dns_weight", 1.0))
        self.distortion_weight = float(grpo.get("distortion_weight", 0.0))
        self.distortion_to_clean = bool(grpo.get("distortion_to_clean", False))
        self.relative_mos = bool(grpo.get("relative_mos", True))
        self.reward_scale = float(grpo.get("reward_scale", 1.0))
        self.reward_base = str(grpo.get("reward_base", "noisy")).lower()
        # Advantages
        self.adv_mode = str(grpo.get("adv_mode", "zscore")).lower().replace("-", "_")
        self.skip_zero_std_groups = bool(grpo.get("skip_zero_std_groups", True))
        self.min_group_std = float(grpo.get("min_group_std", 1e-6))
        self.adv_std_floor = float(grpo.get("adv_std_floor", 0.05))
        self.adv_tau = float(grpo.get("adv_tau", 0.05))
        self.adv_scale = float(grpo.get("adv_scale", 1.0))
        self.adv_clip_max = float(grpo.get("adv_clip_max", 5.0))
        self.adv_eps = float(grpo.get("adv_eps", 1e-5))
        # MSE anchor
        self.lambda_mse = float(grpo.get("lambda_mse", 0.0))
        self.mse_target = str(grpo.get("mse_target", "clean")).lower()
        # Misc
        self.post_update_ratio_diag = bool(grpo.get("post_update_ratio_diag", True))
        self.force_eval_logprob = bool(grpo.get("force_eval_logprob", False))
        self.logp_reduce = str(grpo.get("logp_reduce", "mean")).lower()
        self.kl_reduce = str(grpo.get("kl_reduce", self.logp_reduce)).lower()
        self.policy_eval_mode = bool(grpo.get("policy_eval_mode", True))
        self.max_grad_norm = grpo.get("max_grad_norm", 1.0)
        self.debug_log_interval = int(grpo.get("debug_log_interval", 10))
        self.debug_log_print = bool(grpo.get("debug_log_print", True))
        self.latency_log = bool(grpo.get("latency_log", True))
        self._last_latency_ms: Dict[str, float] = {}

        # Windows
        self.window: Deque[torch.Tensor] = deque(maxlen=self.window_size)
        self.clean_window: Deque[torch.Tensor] = deque(maxlen=self.window_size)
        self.reset_window_on_utt_change = bool(grpo.get("reset_window_on_utt_change", False))
        self._last_window_utt = None
        self._grpo_update_count = 0
        self._grpo_warmup_skip_count = 0
        self.mos_worker = TorchMOS(conf)
        if ref_model is None:
            ref_model = deepcopy(self.model)
        self.ref_model = freeze_model(ref_model.eval())
        self._policies_synced = False

        logger.info(f"[GRPO] window={self.window_size}, actions={self.num_actions}, "
                     f"sigma={self.sigma}, clip_eps={self.clip_eps}, beta_kl={self.beta}, "
                     f"adv_mode={self.adv_mode}, lambda_mse={self.lambda_mse}, "
                     f"reward_base={self.reward_base}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_wav(self, x):
        """Convert various input formats to a B x T tensor."""
        if x is None or isinstance(x, str):
            return None
        if isinstance(x, (list, tuple)):
            x = [t for t in x if torch.is_tensor(t)]
            if not x:
                return None
            max_len = max(t.shape[-1] for t in x)
            return torch.stack([F.pad(t, (0, max_len - t.shape[-1])) for t in x], dim=0)
        return torch.as_tensor(x)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.window.clear()
        self.clean_window.clear()
        self._last_window_utt = None
        self._grpo_update_count = 0
        self._grpo_warmup_skip_count = 0

    def _latency_stamp(self):
        self._sync_latency_device()
        return time.perf_counter()

    def _latency_elapsed_ms(self, start):
        self._sync_latency_device()
        return (time.perf_counter() - start) * 1000.0

    def _sync_latency_device(self):
        try:
            device = self._grpo_device() if hasattr(self, "_grpo_device") else torch.device(self.device)
        except Exception:
            return
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _record_latency(self, name, value_ms, log_metrics=True, batch_size=1):
        value_ms = float(value_ms)
        self._last_latency_ms[name] = value_ms
        if not (log_metrics and self.latency_log):
            return
        try:
            _ = self.trainer
        except RuntimeError:
            return
        self.log(
            f"latency/{name}_ms",
            torch.as_tensor(
                value_ms,
                dtype=torch.float32,
                device=self._grpo_device() if hasattr(self, "_grpo_device") else self.device,
            ),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=max(int(batch_size), 1),
        )

    def _record_latency_dict(self, latency, log_metrics=True, batch_size=1):
        if not latency:
            return
        for name, value_ms in latency.items():
            self._record_latency(name, value_ms, log_metrics=log_metrics, batch_size=batch_size)

    def speech_distortion(self, est, target):
        T = min(est.shape[-1], target.shape[-1])
        diff = est[..., :T] - target[..., :T]
        return diff.reshape(diff.shape[0], -1).pow(2).mean(dim=-1)

    def _sq_reduce_flat(self, x, reduce="mean"):
        x = x.reshape(x.shape[0], -1)
        sq = x.real ** 2 + x.imag ** 2 if torch.is_complex(x) else x ** 2
        return sq.mean(dim=-1) if reduce == "mean" else sq.sum(dim=-1)

    # ------------------------------------------------------------------
    # Policy sampling, log-prob, reward, KL
    # ------------------------------------------------------------------
    def _log_prob(self, actions, mean, std=None):
        if std is None:
            std = self.sigma
        actions, mean = TensorUtils.align_last_dim(actions, mean)
        std_t = torch.as_tensor(float(std) if not torch.is_tensor(std) else std,
                                device=mean.device, dtype=mean.real.dtype if torch.is_complex(mean) else mean.dtype)
        std_t = std_t.clamp_min(EPS)
        var = std_t ** 2 + EPS
        diff = actions.detach() - mean
        log_sqrt_2pi = math.log(math.sqrt(2.0 * math.pi))
        log_val = -(diff ** 2) / (2 * var) - torch.log(std_t) - log_sqrt_2pi

        if torch.is_complex(diff):
            # Concatenate real and imag log-probs
            lp = torch.cat([log_val.real.flatten(1), log_val.imag.flatten(1)], dim=-1)
        else:
            lp = log_val.flatten(1)
        return lp.mean(-1) if self.logp_reduce == "mean" else lp.sum(-1)

    def _sample_actions_with_logprob(self, mean):
        std = torch.as_tensor(self.sigma, device=mean.device, dtype=mean.real.dtype if torch.is_complex(mean) else mean.dtype)
        noise = torch.randn_like(mean) if not torch.is_complex(mean) else \
                torch.randn_like(mean.real) + 1j * torch.randn_like(mean.real)
        actions = mean + std * noise
        logp = self._log_prob(actions, mean, std)
        return actions, logp.detach(), std.detach(), noise

    def _equivalent_mask_from_spec(self, est_spec, noisy_spec):
        """M = est * conj(noisy) / (|noisy|^2 + den_floor) with magnitude clipping."""
        est_spec, noisy_spec = TensorUtils.align_spec(est_spec, noisy_spec)
        power = noisy_spec.real ** 2 + noisy_spec.imag ** 2
        mask = est_spec * torch.conj(noisy_spec) / power.clamp_min(self.mask_den_floor)
        mask = TensorUtils.safe_nan_to_num(mask)
        if self.mask_abs_clip is not None:
            mag = torch.abs(mask).clamp_min(EPS)
            mask = mask * torch.clamp(self.mask_abs_clip / mag, max=1.0)
        return mask

    def _action_to_wav(self, action_spec, noisy_spec, target_len):
        action_spec, noisy_spec = TensorUtils.align_spec(action_spec, noisy_spec)
        wav = self.stft.apply_istft(noisy_spec * action_spec)
        return torch.clamp(torch.nan_to_num(wav[..., :target_len], 0.0), -1.0, 1.0)

    def compute_reward(self, est_wav, noisy_wav, base_wav=None, distortion_target=None):
        est_wav = torch.clamp(torch.nan_to_num(est_wav, 0.0), -1.0, 1.0)
        noisy_wav = torch.clamp(torch.nan_to_num(noisy_wav, 0.0), -1.0, 1.0)
        if base_wav is not None:
            base_wav = torch.clamp(torch.nan_to_num(base_wav, 0.0), -1.0, 1.0)
        if distortion_target is not None:
            distortion_target = torch.clamp(torch.nan_to_num(distortion_target, 0.0), -1.0, 1.0)

        mos_est = self.mos_worker.batch_scores(est_wav)[..., 2]
        mos_base = self.mos_worker.batch_scores(base_wav if base_wav is not None else noisy_wav)[..., 2]
        mos_term = (mos_est - mos_base) if self.relative_mos else mos_est
        mos_term = mos_term * self.reward_scale

        dist_target = distortion_target if distortion_target is not None else (base_wav if base_wav is not None else noisy_wav)
        distortion = self.speech_distortion(est_wav, dist_target)
        reward = self.dns_weight * mos_term - self.distortion_weight * distortion
        return dict(reward=reward, mos=mos_term, mos_abs=mos_est, mos_base=mos_base, distortion=distortion)

    def _kl_between(self, pred, ref):
        pred, ref = TensorUtils.align_last_dim(pred, ref)
        sq_norm = self._sq_reduce_flat(pred - ref, reduce=self.kl_reduce)
        return sq_norm.mean() / (2 * (self.sigma ** 2) + EPS)

    # ------------------------------------------------------------------
    # Model forward logic
    # ------------------------------------------------------------------
    def _model_enhanced_spec(self, model, noisy, noisy_spec=None):
        """Obtain enhanced complex spectrum from a model."""
        if noisy_spec is None:
            noisy_spec = self.stft.apply_stft(self.padding(noisy))
        try:
            # General path: model accepts complex spectrum directly
            return model(noisy_spec)
        except (TypeError, RuntimeError):
            # Fallback for models expecting features
            return self._enhanced_spec_from_features(model, noisy_spec, noisy)

    def _enhanced_spec_from_features(self, model, noisy_spec, noisy):
        """Fallback: compute magnitude/phase features and decode."""
        noisy_mag = torch.abs(noisy_spec)
        data = []
        if self.spec_data:
            data += [noisy_spec.real, noisy_spec.imag]
        if self.mag_data:
            data.append(noisy_mag)
        if not data:
            return self._legacy_model_forward(model, noisy)
        noisy_input = torch.cat(data, dim=1)
        out = model(noisy_input)
        if isinstance(out, tuple):
            est_spec, est_mag = out
        else:
            est_spec, est_mag = out, None
        # Reconstruct complex spectrum
        if est_spec is not None:
            est_spec = torch.complex(est_spec[:, 0], est_spec[:, 1]).unsqueeze(1)
            if self.mask:
                est_spec = est_spec * noisy_spec
        elif est_mag is not None:
            est_mag = est_mag * noisy_mag if self.mask else est_mag
            est_spec = est_mag * torch.exp(1j * torch.angle(noisy_spec))
        else:
            raise RuntimeError("Model output not recognized")
        return est_spec

    def _legacy_model_forward(self, model, noisy):
        enhanced = model(self.padding(noisy))
        if isinstance(enhanced, dict):
            spec = enhanced.get("est")
            if spec is not None and torch.is_complex(spec):
                return spec.unsqueeze(1) if spec.ndim == 3 else spec
        return self.stft.apply_stft(enhanced)

    def _policy_forward_stable(self, noisy_flat, T_wav, model=None, force_eval=False):
        """Compute deterministic action mean (equivalent mask) with stable settings."""
        model = model or self.model
        was_training = model.training
        if force_eval:
            model.eval()
        else:
            model.train()
        # Temporarily disable batch norm and dropout
        bn_drop = []
        rnn_drop = []
        if self.policy_eval_mode or force_eval:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                  nn.LayerNorm, nn.Dropout)):
                    bn_drop.append((m, m.training))
                    m.eval()
                if isinstance(m, (nn.LSTM, nn.GRU)) and hasattr(m, "dropout"):
                    rnn_drop.append((m, m.dropout))
                    m.dropout = 0.0
        try:
            noisy_spec = self.stft.apply_stft(self.padding(noisy_flat))
            est_spec = self._model_enhanced_spec(model, noisy_flat, noisy_spec)
            return self._equivalent_mask_from_spec(est_spec, noisy_spec)
        finally:
            for m, tr in bn_drop:
                m.train(tr)
            for m, d in rnn_drop:
                m.dropout = d
            model.train(was_training)

    # ------------------------------------------------------------------
    # GRPO sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect_flowgrpo_samples(self, noisy_group, clean_group=None):
        G, K = noisy_group.shape[0], self.num_actions
        T_wav = noisy_group.shape[-1]
        noisy_rep = noisy_group.unsqueeze(1).expand(G, K, -1).reshape(G * K, -1).detach()
        clean_flat = (
            clean_group.unsqueeze(1).expand(G, K, -1).reshape(G * K, -1).detach()
            if clean_group is not None else None
        )

        self.model.eval()
        self.ref_model.eval()
        # Policy mean
        mean = TensorUtils.safe_nan_to_num(self._policy_forward_stable(noisy_rep, T_wav, model=self.model,
                                                                  force_eval=self.force_eval_logprob))
        # Sample actions
        actions, log_probs, action_std, _ = self._sample_actions_with_logprob(mean)
        # Ref/det baseline means
        ref_mean = TensorUtils.safe_nan_to_num(self._policy_forward_stable(noisy_rep, T_wav, model=self.ref_model, force_eval=True))
        det_mean = mean  # deterministic current mean

        noisy_spec_flat = self.stft.apply_stft(self.padding(noisy_rep))
        est_wav = self._action_to_wav(actions, noisy_spec_flat, T_wav)
        base_wav = self._action_to_wav(ref_mean, noisy_spec_flat, T_wav)
        det_wav = self._action_to_wav(det_mean, noisy_spec_flat, T_wav)

        # Reward base selection
        if self.reward_base in ("ref", "sft", "pretrained"):
            base_mos = self.mos_worker.batch_scores(base_wav)[..., 2]
        elif self.reward_base in ("det", "deterministic", "current"):
            base_mos = self.mos_worker.batch_scores(det_wav)[..., 2]
        else:  # noisy
            base_mos = self.mos_worker.batch_scores(noisy_group)[..., 2].repeat_interleave(K, dim=0)

        dist_target = clean_flat[..., :T_wav] if (clean_flat is not None and self.distortion_to_clean) else base_wav

        rew = self.compute_reward(est_wav, noisy_rep[..., :T_wav], base_wav=base_wav, distortion_target=dist_target)
        ref_rew = self.compute_reward(base_wav, noisy_rep[..., :T_wav], base_wav=base_wav, distortion_target=dist_target)
        det_rew = self.compute_reward(det_wav, noisy_rep[..., :T_wav], base_wav=base_wav, distortion_target=dist_target)

        rewards = TensorUtils.safe_nan_to_num(rew["reward"]).detach()
        ref_rewards_flat = TensorUtils.safe_nan_to_num(ref_rew["reward"]).detach()
        det_rewards_flat = TensorUtils.safe_nan_to_num(det_rew["reward"]).detach()

        return {
            "G": G, "K": K, "T_wav": T_wav,
            "noisy_flat": noisy_rep.detach(),
            "noisy_spec_flat": noisy_spec_flat.detach(),
            "actions": actions.detach(),
            "log_probs": TensorUtils.safe_nan_to_num(log_probs).detach(),
            "action_std": action_std.detach() if torch.is_tensor(action_std) else action_std,
            "ref_mean": ref_mean.detach(),
            "baseline_wav": base_wav.detach(),
            "clean_flat": clean_flat[..., :T_wav].detach() if clean_flat is not None else None,
            "rewards": rewards,
            "ref_rewards": ref_rewards_flat.reshape(G, K)[:, 0].detach(),
            "det_rewards": det_rewards_flat.reshape(G, K)[:, 0].detach(),
            "mos": TensorUtils.safe_nan_to_num(rew["mos"]).detach(),
            "mos_abs": TensorUtils.safe_nan_to_num(rew["mos_abs"]).detach(),
            "distortion": TensorUtils.safe_nan_to_num(rew["distortion"]).detach(),
            "ref_mos_abs": TensorUtils.safe_nan_to_num(ref_rew["mos_abs"]).reshape(G, K)[:, 0].detach(),
            "det_mos_abs": TensorUtils.safe_nan_to_num(det_rew["mos_abs"]).reshape(G, K)[:, 0].detach(),
        }

    # ------------------------------------------------------------------
    # Advantages
    # ------------------------------------------------------------------
    def _compute_flowgrpo_advantages(self, samples):
        G, K = samples["G"], samples["K"]
        rewards = samples["rewards"].reshape(G, K)
        group_mean = rewards.mean(dim=1)
        group_best = rewards.max(dim=1).values
        rew_std = rewards.std(dim=1, unbiased=False).clamp_min(self.adv_eps)
        valid = rew_std > self.min_group_std

        ref = samples.get("ref_rewards")
        det = samples.get("det_rewards")
        if ref is not None:
            ref = ref.to(rewards.device)
        if det is not None:
            det = det.to(rewards.device)

        mode = self.adv_mode
        # Choose baseline
        if mode in ("zscore", "flow_zscore", "grpo"):
            advantages = (rewards - group_mean[:, None]) / rew_std[:, None]
            baseline = None
        else:
            if mode.startswith(("ref", "rlhf", "soft_sft", "soft_pretrained")):
                baseline = ref
            elif mode.startswith(("det", "soft_det")):
                baseline = det
            elif mode.startswith(("max_ref_det", "baseline", "soft_max")):
                baseline = torch.maximum(ref, det) if ref is not None and det is not None else (ref or det)
            else:
                raise ValueError(f"Unknown adv_mode: {mode}")
            if baseline is None:
                raise RuntimeError("Baseline required but not available")
            raw_adv = rewards - baseline[:, None]
            if mode.endswith("_signed") or mode.endswith("_soft") or "_signed" in mode or "_soft" in mode:
                # Signed advantages, scaled by adv_tau
                advantages = (raw_adv / max(self.adv_tau, self.adv_eps)) * self.adv_scale
            else:
                # Positive-only or centered
                if "positive" in mode or mode in ("rlhf", "rlhf_positive"):
                    raw_adv = torch.clamp(raw_adv, min=0.0)
                    has_pos = raw_adv.any(dim=1, keepdim=True)
                    raw_adv = torch.where(has_pos, raw_adv, torch.zeros_like(raw_adv))
                adv_std = raw_adv.std(dim=1, keepdim=True, unbiased=False).clamp_min(max(self.adv_eps, self.adv_std_floor))
                advantages = (raw_adv / adv_std) * self.adv_scale

        if self.skip_zero_std_groups:
            advantages = torch.where(valid[:, None], advantages, torch.zeros_like(advantages))

        advantages = TensorUtils.safe_nan_to_num(advantages).clamp(-self.adv_clip_max, self.adv_clip_max)
        samples["advantages"] = advantages.reshape(G * K).detach()

        # Diagnostics
        adv_stats = {"rew_std": rew_std, "valid_group_frac": valid.float().mean(),
                     "group_mean": group_mean, "group_best": group_best,
                     "group_best_minus_mean": group_best - group_mean}
        if ref is not None:
            adv_stats.update(ref_reward=ref, best_minus_ref=group_best - ref,
                             mean_minus_ref=group_mean - ref,
                             best_gt_ref_frac=(group_best > ref).float().mean(),
                             action_gt_ref_frac=(rewards > ref[:, None]).float().mean())
        if det is not None:
            adv_stats.update(det_reward=det, best_minus_det=group_best - det,
                             mean_minus_det=group_mean - det,
                             best_gt_det_frac=(group_best > det).float().mean(),
                             action_gt_det_frac=(rewards > det[:, None]).float().mean())
        return adv_stats

    def _compute_current_logp_and_kl_mse(self, sample):
        noisy = sample["noisy_flat"]
        actions = sample["actions"]
        T = sample["T_wav"]
        preds = TensorUtils.safe_nan_to_num(
            self._policy_forward_stable(noisy, T, model=self.model, force_eval=self.force_eval_logprob))
        log_prob = TensorUtils.safe_nan_to_num(self._log_prob(actions, preds, sample.get("action_std", self.sigma)))
        grpo_device = self._grpo_device() if hasattr(self, "_grpo_device") else torch.device(self.device)
        kl = torch.zeros((), device=grpo_device)
        if self.beta > 0:
            kl = self._kl_between(preds, sample["ref_mean"])
        mse = torch.zeros((), device=grpo_device)
        if self.lambda_mse > 0:
            target_wav = None
            if self.mse_target == "clean":
                target_wav = sample.get("clean_flat")
            elif self.mse_target in ("ref", "pretrained", "sft"):
                target_wav = sample.get("baseline_wav")
            if target_wav is not None:
                noisy_spec = sample.get("noisy_spec_flat")
                det_wav = self._action_to_wav(preds, noisy_spec, T)
                mse = self.speech_distortion(det_wav, target_wav[..., :T]).mean()
        return preds, log_prob, kl, mse

    @torch.no_grad()
    def _log_post_update_ratio_diag(self, sample, adv, old_log_prob, info_dict):
        if not self.post_update_ratio_diag:
            return
        try:
            _, logp_post, _, _ = self._compute_current_logp_and_kl_mse(sample)
            ratio_post = torch.exp(logp_post - old_log_prob).clamp(max=10.0)
            info_dict["ratio_post_mean"].append(ratio_post.mean())
            pos = adv > 0
            neg = adv < 0
            if pos.any():
                info_dict["ratio_post_pos_mean"].append(ratio_post[pos].mean())
            if neg.any():
                info_dict["ratio_post_neg_mean"].append(ratio_post[neg].mean())
        except Exception as e:
            if getattr(self, "global_rank", 0) == 0:
                logger.warning(f"Post-update ratio diagnostic failed: {e}")

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        return self._grpo_update_step(batch, batch_idx, opt)

    @staticmethod
    def _plain_utt_id(utt):
        if utt is None:
            return None
        if torch.is_tensor(utt):
            if utt.numel() == 1:
                return str(utt.detach().cpu().item())
            return tuple(utt.detach().cpu().reshape(-1).tolist())
        return str(utt)

    def _utt_ids_for_batch(self, utt, batch_size):
        if utt is None:
            return [None] * batch_size
        if isinstance(utt, (list, tuple)):
            if len(utt) == batch_size:
                return [self._plain_utt_id(item) for item in utt]
            if len(utt) == 1:
                return [self._plain_utt_id(utt[0])] * batch_size
        if torch.is_tensor(utt) and utt.ndim > 0 and utt.shape[0] == batch_size:
            return [self._plain_utt_id(item) for item in utt]
        return [self._plain_utt_id(utt)] * batch_size

    def _maybe_reset_window_for_utt(self, utt_id):
        if not self.reset_window_on_utt_change or utt_id is None:
            return
        if self._last_window_utt is None:
            self._last_window_utt = utt_id
            return
        if self._last_window_utt == utt_id:
            return
        self.window.clear()
        self.clean_window.clear()
        self._last_window_utt = utt_id

    def _backward_grpo_loss(self, loss):
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None and getattr(trainer, "training", False):
            self.manual_backward(loss)
        else:
            loss.backward()

    def _log_if_trainer_attached(self, name, value, **kwargs):
        try:
            _ = self.trainer
        except RuntimeError:
            return
        self.log(name, value, **kwargs)

    def _grpo_update_step(self, batch, batch_idx, opt, update_reference=True, log_metrics=True):
        if hasattr(self, "_apply_device_map"):
            self._apply_device_map()
        grpo_device = self._grpo_device() if hasattr(self, "_grpo_device") else torch.device(self.device)
        step_start = self._latency_stamp()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        if not self._policies_synced:
            self._sync_policies_from_model()
            logger.info("[GRPO] Synced ref policy from model")

        utt, noisy_wav, clean_wav = self._parse_batch(batch)
        noisy_wav = self._ensure_wav(noisy_wav).to(grpo_device).detach()
        clean_wav = self._ensure_wav(clean_wav)
        if clean_wav is not None:
            clean_wav = clean_wav.to(grpo_device).detach()

        B = noisy_wav.shape[0]
        utt_ids = self._utt_ids_for_batch(utt, B)
        for i in range(B):
            self._maybe_reset_window_for_utt(utt_ids[i])
            self.window.append(noisy_wav[i].detach())
            if clean_wav is not None:
                self.clean_window.append(clean_wav[i].detach())
            else:
                self.clean_window.clear()

        if len(self.window) < self.min_window:
            self._grpo_warmup_skip_count += 1
            warmup_ms = self._latency_elapsed_ms(step_start)
            self._record_latency(
                "grpo_warmup_step",
                warmup_ms,
                log_metrics=log_metrics,
                batch_size=max(B, 1),
            )
            self._log_if_trainer_attached("train/grpo_warmup_skips", torch.as_tensor(
                float(self._grpo_warmup_skip_count), device=grpo_device
            ), on_step=True, on_epoch=True, sync_dist=True)
            self._log_if_trainer_attached("train/grpo_window_fill", torch.as_tensor(
                float(len(self.window)) / max(float(self.min_window), 1.0), device=grpo_device
            ), on_step=True, on_epoch=True, sync_dist=True)
            return torch.tensor(0.0, device=grpo_device, requires_grad=True)

        noisy_group = torch.stack(list(self.window), 0)
        clean_group = torch.stack(list(self.clean_window), 0) if len(self.clean_window) == len(self.window) else None

        collect_start = self._latency_stamp()
        samples = self._collect_flowgrpo_samples(noisy_group, clean_group)
        collect_ms = self._latency_elapsed_ms(collect_start)
        adv_start = self._latency_stamp()
        adv_stats = self._compute_flowgrpo_advantages(samples)
        adv_ms = self._latency_elapsed_ms(adv_start)

        rewards = samples["rewards"]
        advantages = samples["advantages"]
        total = rewards.shape[0]
        mini_size = self.train_minibatch_size if self.train_minibatch_size > 0 else total

        info = defaultdict(list)
        last_loss = torch.zeros((), device=grpo_device)
        optimizer_start = self._latency_stamp()

        for _ in range(max(1, self.update_epochs)):
            perm = torch.randperm(total, device=grpo_device)
            for start in range(0, total, mini_size):
                idx = perm[start:start + mini_size]
                mb = {
                    k: v[idx] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == total else v
                    for k, v in samples.items()
                }
                _, log_prob, kl, mse = self._compute_current_logp_and_kl_mse(mb)
                old_logp = mb["log_probs"]
                adv = mb["advantages"].clamp(-self.adv_clip_max, self.adv_clip_max)

                ratio = torch.exp(log_prob - old_logp).clamp(max=10.0)
                policy_loss = -torch.min(adv * ratio, adv * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)).mean()
                loss = policy_loss + self.beta * kl + self.lambda_mse * mse

                opt.zero_grad()
                self._backward_grpo_loss(loss)
                # Gradient clipping
                if self.max_grad_norm is not None and float(self.max_grad_norm) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.max_grad_norm))
                else:
                    grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in self.model.parameters() if p.grad is not None]))
                opt.step()

                # Collect mini-batch diagnostics
                self._log_post_update_ratio_diag(mb, adv.detach(), old_logp.detach(), info)
                info["approx_kl"].append(0.5 * (log_prob - old_logp).pow(2).mean())
                info["clipfrac"].append(((ratio - 1).abs() > self.clip_eps).float().mean())
                info["policy_loss"].append(policy_loss.detach())
                info["kl_loss"].append(kl.detach())
                info["mse_loss"].append(mse.detach())
                info["loss"].append(loss.detach())
                info["ratio_mean"].append(ratio.mean())
                last_loss = loss.detach()

        optimizer_ms = self._latency_elapsed_ms(optimizer_start)
        if update_reference:
            self._update_reference()
        total_ms = self._latency_elapsed_ms(step_start)
        latency = dict(samples.get("latency_ms", {}))
        latency.update({
            "grpo_collect_total": collect_ms,
            "grpo_advantage": adv_ms,
            "grpo_optimizer": optimizer_ms,
            "grpo_update_total": total_ms,
        })
        self._record_latency_dict(latency, log_metrics=log_metrics, batch_size=max(total, 1))
        if log_metrics:
            self._log_metrics(samples, adv_stats, info, total, rewards, advantages, last_loss, grad_norm)
        self._grpo_update_count += 1
        self._log_if_trainer_attached("train/grpo_updates", torch.as_tensor(
            float(self._grpo_update_count), device=grpo_device
        ), on_step=True, on_epoch=True, sync_dist=True)
        self._log_if_trainer_attached("train/grpo_loss", last_loss, on_step=True, on_epoch=True, sync_dist=True)
        return last_loss

    def _parse_batch(self, batch):
        """Unify batch parsing: return (utt_id_or_None, noisy_wav, clean_wav)."""
        if isinstance(batch, dict):
            def first_value(keys):
                for key in keys:
                    if key in batch and batch[key] is not None:
                        return batch[key]
                return None
            noisy = first_value(["noisy", "noisy_wav", "mix"])
            clean = first_value(["clean", "clean_wav"])
            utt = first_value(["utt_id", "utt", "wav_id"])
            return utt, noisy, clean
        # Assume parent class has unpack_wav_batch
        raw = self.unpack_wav_batch(batch)
        if isinstance(raw, (list, tuple)):
            clean = raw[0] if len(raw) > 0 else None
            noisy = raw[1] if len(raw) > 1 else None
            utt = raw[2] if len(raw) > 2 else None
            return utt, noisy, clean
        return None, raw.get("noisy", raw.get("noisy_wav")), raw.get("clean", raw.get("clean_wav"))

    def _sync_policies_from_model(self):
        if hasattr(self, "_apply_device_map"):
            self._apply_device_map()
        self.ref_model.load_state_dict(self.model.state_dict())
        self._policies_synced = True

    def on_fit_start(self):
        super().on_fit_start()
        if hasattr(self, "_apply_device_map"):
            self._apply_device_map(force=True)
        else:
            self.ref_model.to(self.device)
        if not self._policies_synced:
            self._sync_policies_from_model()
            logger.info("[GRPO] Synced ref policy at fit start")

    def _update_reference(self):
        if self.ref_update_steps and (self.global_step + 1) % self.ref_update_steps == 0:
            if hasattr(self, "_apply_device_map"):
                self._apply_device_map()
            self.ref_model.load_state_dict(self.model.state_dict())
            logger.info(f"[GRPO] Updated reference model at step {self.global_step + 1}")

    def _log_metrics(self, samples, adv_stats, info, total, rewards, advantages, loss, grad_norm):
        """Centralized metric logging."""
        def mean_of(key):
            vals = info.get(key)
            device = self._grpo_device() if hasattr(self, "_grpo_device") else self.device
            return torch.stack(vals).mean() if vals else torch.zeros((), device=device)

        # Reward/advantage summaries
        self.log("grpo/reward_mean", rewards.mean(), sync_dist=True, batch_size=total)
        self.log("grpo/reward_max", rewards.max(), sync_dist=True, batch_size=total)
        self.log("grpo/reward_min", rewards.min(), sync_dist=True, batch_size=total)
        self.log("grpo/group_best", adv_stats["group_best"].mean(), sync_dist=True, batch_size=samples["G"])
        self.log("grpo/group_mean", adv_stats["group_mean"].mean(), sync_dist=True, batch_size=samples["G"])
        self.log("grpo/best_minus_mean", adv_stats["group_best_minus_mean"].mean(), sync_dist=True, batch_size=samples["G"])
        self.log("grpo/adv_pos_frac", (advantages > 0).float().mean(), sync_dist=True, batch_size=total)
        if "ref_reward" in adv_stats:
            self.log("grpo/ref_reward", adv_stats["ref_reward"].mean(), sync_dist=True, batch_size=samples["G"])
            self.log("grpo/best_gt_ref_frac", adv_stats["best_gt_ref_frac"], sync_dist=True)
            self.log("grpo/mean_minus_ref", adv_stats["mean_minus_ref"].mean(), sync_dist=True, batch_size=samples["G"])
        if "det_reward" in adv_stats:
            self.log("grpo/det_reward", adv_stats["det_reward"].mean(), sync_dist=True, batch_size=samples["G"])
            self.log("grpo/best_gt_det_frac", adv_stats["best_gt_det_frac"], sync_dist=True)
            self.log("grpo/mean_minus_det", adv_stats["mean_minus_det"].mean(), sync_dist=True, batch_size=samples["G"])
        # Log ref/det absolute MOS
        if "ref_mos_abs" in samples:
            self.log("grpo/ref_mos_abs", samples["ref_mos_abs"].mean(), sync_dist=True, batch_size=samples["G"])
        if "det_mos_abs" in samples:
            self.log("grpo/det_mos_abs", samples["det_mos_abs"].mean(), sync_dist=True, batch_size=samples["G"])

        # PPO diagnostics
        self.log("grpo/loss", loss, prog_bar=True, sync_dist=True)
        self.log("grpo/policy_loss", mean_of("policy_loss"), sync_dist=True)
        self.log("grpo/kl_loss", mean_of("kl_loss"), sync_dist=True)
        self.log("grpo/mse_loss", mean_of("mse_loss"), sync_dist=True)
        self.log("grpo/approx_kl", mean_of("approx_kl"), sync_dist=True)
        self.log("grpo/clipfrac", mean_of("clipfrac"), sync_dist=True)
        self.log("grpo/ratio_mean", mean_of("ratio_mean"), sync_dist=True)
        self.log("grpo/logp_old", samples["log_probs"].mean(), sync_dist=True, batch_size=total)

        if self.debug_log_interval > 0 and self.global_step % self.debug_log_interval == 0:
            self.log("dbg/grad_norm", grad_norm, sync_dist=True)
            if self.debug_log_print and getattr(self, "global_rank", 0) == 0:
                logger.info(
                    f"[GRPO step={self.global_step}] "
                    f"rew mean={rewards.mean().item():.4g} best-mean={adv_stats['group_best_minus_mean'].mean().item():.4g} "
                    f"loss={loss.item():.4g} ratio={mean_of('ratio_mean').item():.4g} "
                    f"clipfrac={mean_of('clipfrac').item():.3f}"
                    + (f" best>ref={adv_stats['best_gt_ref_frac'].item():.3f}" if "best_gt_ref_frac" in adv_stats else "")
                    + (f" best>det={adv_stats['best_gt_det_frac'].item():.3f}" if "best_gt_det_frac" in adv_stats else "")
                )


class FrozenExpertRouterGRPO(SlidingWindowGRPO):
    """Test-time GRPO adaptation for a router over frozen pretrained experts.

    Experts are loaded from checkpoints and never receive gradients. Each expert
    first produces an enhanced waveform; GRPO explores in router-logit space,
    converts sampled logits to expert weights, and soft-mixes the enhanced
    waveforms while updating only the router parameters.
    """

    def _coerce_device(self, spec, fallback=None):
        if fallback is None:
            try:
                fallback = torch.device(self.device)
            except Exception:
                fallback = torch.device("cpu")
        fallback = torch.device(fallback)
        if spec is None or spec == "":
            return fallback
        if isinstance(spec, torch.device):
            device = spec
        elif isinstance(spec, int):
            device = torch.device(f"cuda:{spec}")
        else:
            text = str(spec).strip().lower()
            if text in ("auto", "default"):
                return fallback
            if text.isdigit():
                device = torch.device(f"cuda:{text}")
            else:
                device = torch.device(text)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning(
                    f"[FrozenExpertRouterGRPO] requested {device}, but CUDA is unavailable; using {fallback}."
                )
                return fallback
            count = torch.cuda.device_count()
            index = 0 if device.index is None else int(device.index)
            if index >= count:
                logger.warning(
                    f"[FrozenExpertRouterGRPO] requested {device}, but only {count} CUDA devices are visible; "
                    f"using {fallback}."
                )
                return fallback
            return torch.device(f"cuda:{index}")
        return device

    def _init_device_map(self, router_grpo):
        device_map = _to_plain_dict(router_grpo.get("device_map", router_grpo.get("multi_gpu", {})))
        self._device_map_conf = device_map
        self._device_map_enabled = bool(device_map.get("enabled", bool(device_map)))
        self._device_map_placed = False

        try:
            default_device = torch.device(self.device)
        except Exception:
            default_device = torch.device("cpu")
        visible_cuda = [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())] \
            if torch.cuda.is_available() else []

        if not self._device_map_enabled:
            self._router_device_override = None
            self._expert_device_overrides = None
            return

        grpo_spec = device_map.get("grpo_device", device_map.get("router_device", None))
        grpo_fallback = visible_cuda[0] if visible_cuda else default_device
        grpo_device = self._coerce_device(grpo_spec, fallback=grpo_fallback)

        raw_expert_devices = device_map.get("expert_devices", None)
        if raw_expert_devices is None:
            raw_expert_devices = device_map.get("experts", None)
        if isinstance(raw_expert_devices, (str, int, torch.device)):
            raw_expert_devices = [raw_expert_devices]
        if not raw_expert_devices:
            raw_expert_devices = [dev for dev in visible_cuda if dev != grpo_device] or [grpo_device]
        expert_pool = [self._coerce_device(item, fallback=grpo_device) for item in raw_expert_devices]
        expert_pool = expert_pool or [grpo_device]

        explicit_map = _to_plain_dict(device_map.get("expert_device_map", {}))
        expert_devices = []
        for idx, expert in enumerate(self.experts):
            mapped = explicit_map.get(str(idx), explicit_map.get(expert.name, None))
            if mapped is None:
                expert_devices.append(expert_pool[idx % len(expert_pool)])
            else:
                expert_devices.append(self._coerce_device(mapped, fallback=expert_pool[idx % len(expert_pool)]))

        self._router_device_override = grpo_device
        self._expert_device_overrides = expert_devices
        logger.info(
            "[FrozenExpertRouterGRPO] multi-GPU device map enabled: "
            f"grpo/router={grpo_device}, experts="
            f"{[(expert.name, str(expert_devices[idx])) for idx, expert in enumerate(self.experts)]}"
        )

    def _router_device(self):
        device = getattr(self, "_router_device_override", None)
        if device is not None:
            return torch.device(device)
        try:
            return torch.device(self.device)
        except Exception:
            return torch.device("cpu")

    def _grpo_device(self):
        return self._router_device()

    def _expert_device(self, expert_or_idx):
        devices = getattr(self, "_expert_device_overrides", None)
        if not devices:
            return self._router_device()
        if isinstance(expert_or_idx, int):
            idx = int(expert_or_idx)
        else:
            idx = int(getattr(expert_or_idx, "_moe_expert_idx", 0))
        if idx < 0 or idx >= len(devices):
            return self._router_device()
        return torch.device(devices[idx])

    def _device_map_status(self):
        return {
            "enabled": bool(getattr(self, "_device_map_enabled", False)),
            "grpo_device": str(self._router_device()),
            "expert_devices": [
                {
                    "expert": expert.name,
                    "device": str(self._expert_device(idx)),
                }
                for idx, expert in enumerate(getattr(self, "experts", []))
            ],
        }

    def _sync_stream_devices(self):
        if not torch.cuda.is_available():
            return
        devices = {self._router_device()}
        devices.update(self._expert_device(idx) for idx in range(len(getattr(self, "experts", []))))
        for device in devices:
            device = torch.device(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    @staticmethod
    def _first_module_device(module):
        for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            return tensor.device
        return None

    def _move_module_to(self, module, device):
        if module is None:
            return False
        device = torch.device(device)
        current = self._first_module_device(module)
        if current == device:
            return False
        module.to(device)
        return True

    def _apply_device_map(self, force=False):
        if not getattr(self, "_device_map_enabled", False):
            return
        if self._device_map_placed and not force:
            return

        router_device = self._router_device()
        moved = False
        moved |= self._move_module_to(self.model, router_device)
        moved |= self._move_module_to(getattr(self, "ref_model", None), router_device)
        moved |= self._move_module_to(getattr(self, "stft", None), router_device)
        moved |= self._move_module_to(getattr(self, "loss", None), router_device)
        moved |= self._move_module_to(getattr(self, "valid_loss", None), router_device)
        infer_model = self._get_stream_infer_model() if hasattr(self, "_get_stream_infer_model") else None
        moved |= self._move_module_to(infer_model, router_device)

        for idx, expert in enumerate(self.experts):
            expert_device = self._expert_device(idx)
            setattr(expert, "_moe_expert_idx", idx)
            setattr(expert, "_moe_device", expert_device)
            moved |= self._move_module_to(expert, expert_device)
        for idx, adapter in enumerate(getattr(self, "_stream_expert_adapters", [])):
            adapter.device = self._expert_device(idx)
            adapter.output_device = router_device

        self._device_map_placed = True
        if moved and self._moe_stream_state is not None:
            self._moe_stream_state = None
            self._stream_expert_states = None
            self._stream_infer_context = None

    def __init__(self, conf):
        BaseSE.__init__(self, conf)

        router_grpo = _to_plain_dict(conf.get("router_grpo", conf.get("expert_router", {})))
        experts_conf = router_grpo.get("experts", conf.get("experts", None))
        if OmegaConf.is_config(experts_conf):
            experts_conf = OmegaConf.to_container(experts_conf, resolve=True)
        if not experts_conf:
            raise ValueError("FrozenExpertRouterGRPO requires router_grpo.experts.")

        expert_defaults = router_grpo.get("expert_defaults", router_grpo.get("expert_model_defaults", {}))
        self.experts = nn.ModuleList([
            _FrozenEnhancementExpert(expert_conf, expert_defaults, idx)
            for idx, expert_conf in enumerate(experts_conf)
        ])
        for idx, expert in enumerate(self.experts):
            setattr(expert, "_moe_expert_idx", idx)
        if len(self.experts) < 2:
            logger.warning("[FrozenExpertRouterGRPO] only one expert configured; router learning is degenerate.")

        router_conf = merge_dicts(_to_plain_dict(conf.get("model", {})), _to_plain_dict(router_grpo.get("router", {})))
        router_conf["name"] = "SpectralStatsRouter"
        router_conf["num_experts"] = len(self.experts)
        self.model_conf = router_conf
        self.model = models.SpectralStatsRouter(router_conf)

        self.adapt_in_test = bool(router_grpo.get("adapt_in_test", True))
        self.test_update_before_eval = bool(router_grpo.get("test_update_before_eval", True))
        self._router_test_optimizer = None
        self.router_grpo_conf = router_grpo

        infer_branch = _to_plain_dict(router_grpo.get("inference_branch", {}))
        train_branch = _to_plain_dict(router_grpo.get("training_branch", {}))
        train_stream_sim = _to_plain_dict(router_grpo.get("train_stream_sim", {}))
        self.stream_online_infer = bool(infer_branch.get("enabled", router_grpo.get("online_infer", True)))
        self.stream_inference_runtime = str(infer_branch.get("runtime", infer_branch.get("backend", "torch"))).lower()
        self.stream_onnx_conf = _to_plain_dict(infer_branch.get("onnx", {}))
        self.stream_onnx_strict = bool(self.stream_onnx_conf.get("strict", True))
        self.stream_onnx_use_iobinding = bool(
            self.stream_onnx_conf.get("use_iobinding", self.stream_onnx_conf.get("iobinding", True))
        )
        self.stream_onnx_cuda_only = bool(
            self.stream_onnx_conf.get("cuda_only", self.stream_onnx_use_iobinding)
        )
        self.stream_onnx_enable_profiling = bool(
            self.stream_onnx_conf.get("enable_profiling", self.stream_onnx_conf.get("profiling", False))
        )
        self.stream_onnx_use_onnx_router = bool(self.stream_onnx_conf.get("use_onnx_router", False))
        self.stream_onnx_router_sync_interval_steps = max(
            1,
            int(self.stream_onnx_conf.get("router_sync_interval_steps", 100)),
        )
        self._stream_onnx_router_sync_count = 0
        self.stream_onnx_manifest = self.stream_onnx_conf.get("manifest", self.stream_onnx_conf.get("manifest_path"))
        self.stream_onnx_export_dir = self.stream_onnx_conf.get("export_dir", "exp/onnx/moe_stream")
        self.stream_onnx_router_live_export = bool(
            self.stream_onnx_conf.get("router_live_export", self.stream_onnx_use_onnx_router)
        )
        self.stream_onnx_router_path = self.stream_onnx_conf.get(
            "router_path",
            str(Path(self.stream_onnx_export_dir) / "router_features.onnx"),
        )
        self.stream_onnx_providers = self.stream_onnx_conf.get("providers", None)
        self.stream_onnx_provider_options = self.stream_onnx_conf.get("provider_options", None)
        self.stream_onnx_force_provider_device_id = self.stream_onnx_conf.get(
            "force_provider_device_id",
            self.stream_onnx_conf.get("force_device_id", self.stream_onnx_conf.get("device_id", None)),
        )
        self.stream_onnx_override_manifest_providers = bool(
            self.stream_onnx_conf.get(
                "override_manifest_providers",
                self.stream_onnx_force_provider_device_id is not None,
            )
        )
        if self.stream_onnx_force_provider_device_id is not None:
            provider_device_id = int(self.stream_onnx_force_provider_device_id)
            if self.stream_onnx_providers is None:
                self.stream_onnx_providers = (
                    ["CUDAExecutionProvider"]
                    if self.stream_onnx_cuda_only
                    else ["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            provider_options = []
            for provider in self.stream_onnx_providers:
                if str(provider) == "CUDAExecutionProvider":
                    provider_options.append({"device_id": provider_device_id})
                else:
                    provider_options.append({})
            self.stream_onnx_provider_options = provider_options
        self.stream_frame_ms = float(infer_branch.get("frame_ms", router_grpo.get("infer_frame_ms", 32.0)))
        self.stream_hop_ms = float(infer_branch.get("hop_ms", router_grpo.get("infer_hop_ms", 16.0)))
        self.stream_frame_samples = max(1, int(round(float(self.sample_rate) * self.stream_frame_ms / 1000.0)))
        self.stream_hop_samples = max(1, int(round(float(self.sample_rate) * self.stream_hop_ms / 1000.0)))
        if self.stream_frame_samples < self.stream_hop_samples:
            raise ValueError(
                "FrozenExpertRouterGRPO inference_branch.frame_ms must be >= inference_branch.hop_ms."
            )
        self.stream_parallel_experts = bool(infer_branch.get("parallel_experts", True))
        self.stream_parallel_expert_workers = int(
            infer_branch.get("parallel_expert_workers", min(3, len(self.experts)))
        )
        self.stream_parallel_expert_workers = max(1, min(self.stream_parallel_expert_workers, len(self.experts)))
        self.stream_onnx_parallel_cuda_streams = bool(
            self.stream_onnx_conf.get(
                "parallel_cuda_streams",
                self.stream_onnx_force_provider_device_id is not None and self.stream_parallel_experts,
            )
        )
        self.stream_stateful_experts = bool(infer_branch.get("stateful_experts", True))
        self.stream_strict_expert_state = bool(infer_branch.get("strict_expert_state", False))
        self.stream_fallback_history_ms = float(infer_branch.get("fallback_history_ms", 512.0))
        self.stream_fallback_history_samples = max(
            self.stream_frame_samples,
            int(round(float(self.sample_rate) * self.stream_fallback_history_ms / 1000.0)),
        )
        self.stream_report_frame_rtf = bool(infer_branch.get("report_frame_rtf", False))
        self.train_stream_sim_enabled = bool(train_stream_sim.get("enabled", False))
        self.train_stream_infer_before_update = bool(train_stream_sim.get("infer_before_update", True))
        self.train_stream_use_online_inference = bool(train_stream_sim.get("use_online_inference_branch", True))
        self.train_stream_sync_infer_after_update = bool(train_stream_sim.get("sync_infer_model_after_update", True))
        self.train_stream_log_inference_audio_metric = bool(train_stream_sim.get("log_inference_audio_metric", False))
        self.train_stream_log_latency = bool(train_stream_sim.get("log_latency", True))
        self.train_stream_reset_infer_on_epoch_start = bool(
            train_stream_sim.get("reset_infer_state_on_epoch_start", True)
        )
        self._last_train_stream_infer_utt = None
        self.stream_async_train = bool(train_branch.get("async", router_grpo.get("async_train", True)))
        self.stream_train_queue_limit = int(train_branch.get("max_queue", router_grpo.get("max_train_queue", 0)))
        self.__dict__["_stream_infer_model"] = None
        self._stream_infer_lock = None
        self._stream_train_lock = None
        self._stream_train_executor = None
        self._stream_expert_executor = None
        self._stream_train_futures: Deque = deque()
        self._stream_train_error = None
        self._stream_infer_context = None
        self._moe_stream_state = None
        self._onnx_stream_runtime = None
        self._onnx_stream_state = None
        self._stream_expert_states = None
        self._stream_fallback_warned = set()
        self._reset_stream_epoch_profile()
        self._init_device_map(router_grpo)
        self._stream_expert_adapters = self._build_streaming_expert_adapters()

        self._init_grpo_state(conf, ref_model=deepcopy(self.model))
        self._apply_device_map(force=True)
        self.window_reward = bool(router_grpo.get(
            "window_reward", router_grpo.get("streaming_window_reward", False)
        ))
        self.stream_adapt_in_denoise = bool(router_grpo.get("adapt_in_denoise", False))
        self.reset_window_on_utt_change = bool(router_grpo.get("reset_window_on_utt_change", False))
        logger.info(
            "[FrozenExpertRouterGRPO] experts={}, trainable_router_params={}".format(
                [expert.name for expert in self.experts],
                sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            )
        )
        if self.window_reward:
            logger.info(
                "[FrozenExpertRouterGRPO] rolling-window DNSMOS reward is enabled: "
                f"window={self.window_size}, min_window={self.min_window}, actions={self.num_actions}"
            )
        if self.stream_online_infer:
            logger.info(
                "[FrozenExpertRouterGRPO] online inference branch: "
                f"frame={self.stream_frame_samples} samples ({self.stream_frame_ms:g} ms), "
                f"hop={self.stream_hop_samples} samples ({self.stream_hop_ms:g} ms), "
                f"async_train={self.stream_async_train}, "
                f"parallel_experts={self.stream_parallel_experts}, "
                f"expert_workers={self.stream_parallel_expert_workers}, "
                f"stateful_experts={self.stream_stateful_experts}, "
                f"report_frame_rtf={self.stream_report_frame_rtf}, "
                f"runtime={self.stream_inference_runtime}, "
                f"onnx_iobinding={self.stream_onnx_use_iobinding}, "
                f"onnx_cuda_only={self.stream_onnx_cuda_only}, "
                f"onnx_use_onnx_router={self.stream_onnx_use_onnx_router}, "
                f"onnx_router_sync_interval_steps={self.stream_onnx_router_sync_interval_steps}, "
                f"onnx_parallel_cuda_streams={self.stream_onnx_parallel_cuda_streams}, "
                f"expert_stream_modes={self._stream_expert_adapter_modes()}"
            )
        if self.train_stream_sim_enabled:
            logger.info(
                "[FrozenExpertRouterGRPO] train stream simulation is enabled: "
                f"infer_before_update={self.train_stream_infer_before_update}, "
                f"online_infer={self.train_stream_use_online_inference}, "
                f"sync_after_update={self.train_stream_sync_infer_after_update}"
            )

    def init_model(self, model_conf):
        # Optional router initialization. Experts are loaded from router_grpo.experts.
        BaseSE.init_model(self, model_conf)

    def configure_optimizers(self):
        self._apply_device_map()
        opt_conf = OmegaConf.to_container(self.conf["optimizer"], resolve=True)
        return self.get_optimizer(opt_conf, self._router_parameters())

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._apply_device_map()
        self._reset_stream_epoch_profile()
        self._last_train_stream_infer_utt = None
        if self.train_stream_sim_enabled and self.train_stream_reset_infer_on_epoch_start:
            self._reset_stream_infer_state()
            self._sync_stream_infer_model()

    def on_train_epoch_end(self):
        summary = self._stream_epoch_profile_summary()
        if summary is not None:
            device = self._router_device()
            for name, value in summary.items():
                self._log_if_trainer_attached(
                    f"stream_epoch/{name}",
                    torch.as_tensor(float(value), dtype=torch.float32, device=device),
                    sync_dist=True,
                )
            if getattr(self.trainer, "is_global_zero", True):
                logger.info(
                    "[FrozenExpertRouterGRPO] stream epoch "
                    f"{int(getattr(self.trainer, 'current_epoch', 0))}: "
                    f"frames={int(summary['frames'])}, "
                    f"rtf_mean={summary['frame_rtf_mean']:.4f}, "
                    f"rtf_max={summary['frame_rtf_max']:.4f}, "
                    f"total_mean={summary['total_step_ms_mean']:.3f} ms, "
                    f"experts_mean={summary['expert_stream_step_ms_mean']:.3f} ms, "
                    f"router_mean={summary['router_ms_mean']:.3f} ms, "
                    f"fusion_mean={summary['fusion_ms_mean']:.3f} ms, "
                    f"fallback_experts_mean={summary['fallback_expert_count_mean']:.3f}"
                )
        super().on_train_epoch_end()

    def training_step(self, batch, batch_idx):
        self._apply_device_map()
        opt = self.optimizers()
        if not self.train_stream_sim_enabled:
            return self._grpo_update_step(batch, batch_idx, opt)

        step_start = self._latency_stamp()
        infer_info = {}
        if self.train_stream_infer_before_update:
            _, infer_info = self._training_online_infer_step(batch, batch_idx, log_metrics=True)

        loss = self._grpo_update_step(batch, batch_idx, opt)
        if self.train_stream_sync_infer_after_update:
            self._sync_stream_infer_model()

        step_ms = self._latency_elapsed_ms(step_start)
        batch_size = int(infer_info.get("batch_size", 1)) if isinstance(infer_info, dict) else 1
        self._record_latency(
            "train_stream_step_total",
            step_ms,
            log_metrics=self.train_stream_log_latency,
            batch_size=max(batch_size, 1),
        )
        return loss

    def _maybe_reset_train_stream_infer_for_utt(self, utt_id):
        if utt_id is None:
            return
        if self._last_train_stream_infer_utt is None:
            self._last_train_stream_infer_utt = utt_id
            return
        if self._last_train_stream_infer_utt == utt_id:
            return
        self._reset_stream_infer_state()
        self._last_train_stream_infer_utt = utt_id

    def _training_online_infer_step(self, batch, batch_idx, log_metrics=True):
        self._apply_device_map()
        grpo_device = self._grpo_device()
        utt, noisy_wav, clean_wav = self._parse_batch(batch)
        noisy_wav = self._ensure_wav(noisy_wav).to(grpo_device).detach()
        clean_wav = self._ensure_wav(clean_wav)
        if clean_wav is not None:
            clean_wav = clean_wav.to(grpo_device).detach()

        batch_size = noisy_wav.shape[0]
        utt_ids = self._utt_ids_for_batch(utt, batch_size)
        self._ensure_stream_locks()
        if self._use_onnx_stream_inference():
            if self.__dict__.get("_onnx_stream_runtime", None) is None:
                self._sync_stream_infer_model()
            if self._onnx_stream_state is None:
                self._reset_stream_infer_state()
        else:
            if self._get_stream_infer_model() is None:
                self._sync_stream_infer_model()
            if self._moe_stream_state is None:
                self._reset_stream_infer_state()

        infer_start = self._latency_stamp()
        enhanced = []
        weight_rows = []
        frame_counts = []
        with torch.no_grad():
            for idx in range(batch_size):
                y = noisy_wav[idx].detach()
                if batch_size > 1:
                    self._reset_stream_infer_state()
                else:
                    self._maybe_reset_train_stream_infer_for_utt(utt_ids[idx])

                if self.train_stream_use_online_inference:
                    with self._stream_infer_lock:
                        est_wav, weights, infer_frames = self._denoise_online_inference_branch(y)
                else:
                    self.model.eval()
                    padded = self.padding(y.unsqueeze(0))
                    out = self.forward(padded, train=False, ret_weights=True)
                    if isinstance(out, tuple):
                        est_wav, weights = out
                    else:
                        est_wav, weights = out, None
                    est_wav = est_wav.squeeze(0)[:y.shape[-1]]
                    infer_frames = 1

                enhanced.append(est_wav.detach())
                frame_counts.append(float(infer_frames))
                if weights is not None:
                    weights = weights.detach()
                    weight_rows.append(weights.reshape(-1, weights.shape[-1]).mean(dim=0))

            if batch_size > 1:
                self._reset_stream_infer_state()

        infer_ms = self._latency_elapsed_ms(infer_start)
        max_len = max((item.shape[-1] for item in enhanced), default=0)
        if max_len > 0:
            enhanced_wav = torch.stack([
                F.pad(item, (0, max_len - item.shape[-1])) if item.shape[-1] < max_len else item
                for item in enhanced
            ], dim=0)
        else:
            enhanced_wav = noisy_wav.new_zeros((batch_size, 0))

        mean_frames = sum(frame_counts) / max(len(frame_counts), 1)
        audio_seconds = noisy_wav.shape[-1] * batch_size / max(float(self.sample_rate), 1.0)
        rtf = (infer_ms / 1000.0) / max(audio_seconds, EPS)
        mean_weights = torch.stack(weight_rows, dim=0).mean(dim=0) if weight_rows else None

        if log_metrics:
            if self.train_stream_log_latency:
                self._record_latency(
                    "train_stream_infer_total",
                    infer_ms,
                    log_metrics=True,
                    batch_size=max(batch_size, 1),
                )
                self._log_if_trainer_attached(
                    "latency/train_stream_infer_per_sec_rtf",
                    torch.as_tensor(rtf, dtype=torch.float32, device=grpo_device),
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=max(batch_size, 1),
                )
            self._log_if_trainer_attached(
                "train_stream/inference_frames",
                torch.as_tensor(mean_frames, dtype=torch.float32, device=grpo_device),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=max(batch_size, 1),
            )
            self._log_stream_adapter_mode_metrics(batch_size=max(batch_size, 1))
            if mean_weights is not None:
                for idx, expert in enumerate(self.experts):
                    if idx >= mean_weights.numel():
                        break
                    safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in expert.name)
                    self._log_if_trainer_attached(
                        f"train_stream/weight_{idx}_{safe_name}",
                        mean_weights[idx],
                        on_step=True,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=max(batch_size, 1),
                    )
            if self.train_stream_log_inference_audio_metric and clean_wav is not None and max_len > 0:
                metric_len = min(clean_wav.shape[-1], enhanced_wav.shape[-1])
                l1 = (enhanced_wav[..., :metric_len] - clean_wav[..., :metric_len]).abs().mean()
                self._log_if_trainer_attached(
                    "train_stream/l1",
                    l1,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=max(batch_size, 1),
                )

        info = {
            "batch_size": batch_size,
            "inference_frames": mean_frames,
            "inference_total_frames": sum(frame_counts),
            "infer_ms": infer_ms,
            "rtf": rtf,
            "weights": mean_weights.detach() if mean_weights is not None else None,
            "batch_idx": batch_idx,
        }
        return enhanced_wav, info

    def _router_parameters(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("FrozenExpertRouterGRPO has no trainable router parameters.")
        return params

    def _new_router_optimizer(self):
        self._apply_device_map()
        opt_conf = OmegaConf.to_container(self.conf["optimizer"], resolve=True)
        configured = self.get_optimizer(opt_conf, self._router_parameters())
        if isinstance(configured, dict):
            return configured["optimizer"]
        return configured

    def _get_test_optimizer(self):
        if self._router_test_optimizer is None:
            self._router_test_optimizer = self._new_router_optimizer()
        return self._router_test_optimizer

    def _extract_dict_tensor(self, out, keys):
        if not isinstance(out, dict):
            return out
        for key in keys:
            value = out.get(key)
            if value is not None:
                return value
        return None

    def _spec_output_to_wav(self, est_spec, target_len):
        if torch.is_complex(est_spec):
            if est_spec.ndim == 2:
                est_spec = est_spec.unsqueeze(0).unsqueeze(1)
            elif est_spec.ndim == 3:
                est_spec = est_spec.unsqueeze(1)
            elif est_spec.ndim != 4:
                raise RuntimeError(f"Unsupported expert spectrum output shape: {est_spec.shape}")
        elif est_spec.ndim == 4 and est_spec.shape[1] == 2:
            est_spec = torch.complex(est_spec[:, 0], est_spec[:, 1]).unsqueeze(1)
        elif est_spec.ndim >= 4 and est_spec.shape[-1] == 2:
            est_spec = torch.complex(est_spec[..., 0], est_spec[..., 1])
            if est_spec.ndim == 3:
                est_spec = est_spec.unsqueeze(1)
        else:
            raise RuntimeError(f"Unsupported expert spectrum output shape: {est_spec.shape}")
        wav = self.stft.apply_istft(est_spec)
        return TensorUtils.align_waveform_length(wav, target_len)

    def _waveform_output_to_wav(self, out, target_len):
        out = self._extract_dict_tensor(
            out, ["est", "enhanced", "enhanced_wav", "wav", "audio", "spec", "est_spec"]
        )
        if isinstance(out, (list, tuple)):
            out = out[0] if out else None
        if out is None:
            raise RuntimeError("Expert waveform output is empty.")
        if torch.is_complex(out):
            return self._spec_output_to_wav(out, target_len)
        if out.ndim == 4 and out.shape[1] == 2:
            out = torch.complex(out[:, 0], out[:, 1]).unsqueeze(1)
            return self._spec_output_to_wav(out, target_len)
        if out.ndim >= 4 and out.shape[-1] == 2:
            out = torch.complex(out[..., 0], out[..., 1])
            if out.ndim == 3:
                out = out.unsqueeze(1)
            return self._spec_output_to_wav(out, target_len)
        return TensorUtils.align_waveform_length(out, target_len)

    def _legacy_expert_forward(self, expert, noisy, target_len=None):
        out = expert.model(self.padding(noisy))
        return self._waveform_output_to_wav(out, noisy.shape[-1] if target_len is None else target_len)

    def _legacy_expert_spec(self, expert, noisy):
        wav = self._legacy_expert_forward(expert, noisy, noisy.shape[-1])
        return self.stft.apply_stft(self.padding(wav))

    def _feature_expert_spec(self, expert, noisy, noisy_spec):
        noisy_mag = torch.abs(noisy_spec)
        data = []
        if expert.spec_data:
            data.extend([noisy_spec.real, noisy_spec.imag])
        if expert.mag_data:
            data.append(noisy_mag)
        if not data:
            return self._legacy_expert_spec(expert, noisy)

        noisy_input = torch.cat(data, dim=1)
        out = expert.model(noisy_input)
        if isinstance(out, dict):
            out = self._extract_dict_tensor(out, ["est", "enhanced", "spec", "est_spec"])
            if out is None:
                return self._legacy_expert_spec(expert, noisy)

        if isinstance(out, tuple):
            est_spec = out[0] if len(out) > 0 else None
            est_mag = out[1] if len(out) > 1 else None
        else:
            est_spec, est_mag = out, None

        if est_spec is not None:
            if torch.is_complex(est_spec):
                est_spec = est_spec.unsqueeze(1) if est_spec.ndim == 3 else est_spec
            else:
                est_spec = torch.complex(est_spec[:, 0], est_spec[:, 1]).unsqueeze(1)
            return est_spec * noisy_spec if expert.mask else est_spec

        if est_mag is not None:
            est_mag = est_mag * noisy_mag if expert.mask else est_mag
            return est_mag * torch.exp(1j * torch.angle(noisy_spec))

        raise RuntimeError(f"Expert {expert.name} returned neither spectrum nor magnitude.")

    def _expert_enhanced_wav(self, expert, noisy, noisy_spec=None, target_len=None):
        output_device = noisy.device
        expert_device = self._expert_device(expert)
        noisy_local = noisy.to(expert_device, non_blocking=True)
        noisy_spec_local = noisy_spec.to(expert_device, non_blocking=True) if noisy_spec is not None else None
        target_len = noisy.shape[-1] if target_len is None else target_len
        if expert.forward_type in ("wav", "wave", "waveform", "time"):
            return self._legacy_expert_forward(expert, noisy_local, target_len).to(output_device, non_blocking=True)
        if noisy_spec_local is None:
            noisy_spec_local = self.stft.apply_stft(self.padding(noisy)).to(expert_device, non_blocking=True)
        try:
            return expert._forward_fn(self, expert, noisy_local, noisy_spec_local, target_len, output_device)
        except (TypeError, RuntimeError):
            if expert.forward_type != "auto" or not getattr(expert, "_forward_uses_default", False):
                raise
            return self._legacy_expert_forward(expert, noisy_local, target_len).to(output_device, non_blocking=True)

    def _stack_expert_wavs(self, noisy, noisy_spec=None, target_len=None):
        total_start = self._latency_stamp()
        target_len = noisy.shape[-1] if target_len is None else int(target_len)
        if (
            getattr(self, "stream_parallel_experts", False)
            and len(self.experts) > 1
            and getattr(self, "stream_parallel_expert_workers", 1) > 1
        ):
            executor = self._ensure_stream_expert_executor()
            futures = [
                executor.submit(self._stream_expert_wav_task, expert_idx, noisy, noisy_spec, target_len)
                for expert_idx in range(len(self.experts))
            ]
            results = [future.result() for future in futures]
            results.sort(key=lambda item: item[0])
            wavs = [item[1] for item in results]
            total_ms = self._latency_elapsed_ms(total_start)
            self._record_latency("experts_total", total_ms, log_metrics=False, batch_size=noisy.shape[0])
            for _, _, latency_name, value_ms in results:
                self._record_latency(latency_name, value_ms, log_metrics=False, batch_size=noisy.shape[0])
            return torch.stack(wavs, dim=1)

        wavs = []
        expert_latency = {}
        with torch.no_grad():
            for expert_idx, expert in enumerate(self.experts):
                expert_start = self._latency_stamp()
                expert.eval()
                est_wav = self._expert_enhanced_wav(expert, noisy, noisy_spec, target_len)
                est_wav = TensorUtils.align_waveform_length(est_wav, target_len)
                wavs.append(TensorUtils.safe_nan_to_num(est_wav).detach())
                safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in expert.name)
                expert_latency[f"expert_{expert_idx}_{safe_name}"] = self._latency_elapsed_ms(expert_start)
        total_ms = self._latency_elapsed_ms(total_start)
        self._record_latency("experts_total", total_ms, log_metrics=False, batch_size=noisy.shape[0])
        for name, value_ms in expert_latency.items():
            self._record_latency(name, value_ms, log_metrics=False, batch_size=noisy.shape[0])
        return torch.stack(wavs, dim=1)

    def _ensure_stream_expert_executor(self):
        if self._stream_expert_executor is None:
            self._stream_expert_executor = ThreadPoolExecutor(
                max_workers=self.stream_parallel_expert_workers,
                thread_name_prefix="mos_moe_stream_expert",
            )
        return self._stream_expert_executor

    def _stream_expert_wav_task(self, expert_idx, noisy, noisy_spec, target_len):
        expert = self.experts[expert_idx]
        start = time.perf_counter()
        with torch.no_grad():
            expert.eval()
            est_wav = self._expert_enhanced_wav(expert, noisy, noisy_spec, target_len)
            est_wav = TensorUtils.align_waveform_length(est_wav, target_len)
            est_wav = TensorUtils.safe_nan_to_num(est_wav).detach()
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in expert.name)
        return expert_idx, est_wav, f"expert_{expert_idx}_{safe_name}", (time.perf_counter() - start) * 1000.0

    @staticmethod
    def _stream_istft_window(window, n_fft, hop_size):
        window = window.reshape(-1)
        if window.numel() < n_fft:
            pad = n_fft - window.numel()
            window = F.pad(window, (pad // 2, pad - pad // 2))
        elif window.numel() > n_fft:
            window = window[:n_fft]
        k = int(math.ceil(float(n_fft) / float(hop_size)))
        length = hop_size * (2 * k - 1) + (n_fft - hop_size)
        win_sq = window.square().view(1, -1, 1).expand(1, -1, 2 * k - 1)
        denom = F.fold(
            win_sq,
            output_size=(1, length),
            kernel_size=(1, n_fft),
            stride=(1, hop_size),
        ).view(-1)
        start = (k - 1) * hop_size
        denom = denom[start:start + n_fft].clamp_min(EPS)
        return window / denom

    def _fastenhancer_core(self, expert):
        model = getattr(expert, "model", None)
        core = getattr(model, "enhancer", None)
        if core is None or not hasattr(core, "model_forward") or not hasattr(core, "stft"):
            return None
        stft = core.stft
        required = ("n_fft", "hop_size", "window")
        if not all(hasattr(stft, name) for name in required):
            return None
        return core

    @staticmethod
    def _ulunas_core(expert):
        core = getattr(getattr(expert, "model", None), "enhancer", None)
        required = ("erb", "encoder", "dpgrnn", "decoder")
        if core is None or not all(hasattr(core, name) for name in required):
            return None
        if not hasattr(core.encoder, "en_convs") or not hasattr(core.decoder, "de_convs"):
            return None
        return core

    @staticmethod
    def _lisen_core(expert):
        core = getattr(expert, "model", None)
        required = ("encoder", "blocks", "decoder", "n_fft", "hop_length", "power_compress", "power_uncompress")
        if core is None or not all(hasattr(core, name) for name in required):
            return None
        return core

    def _build_streaming_expert_adapters(self):
        adapters = []
        for idx, expert in enumerate(self.experts):
            model = getattr(expert, "model", None)
            has_model_stream = callable(getattr(model, "init_stream_state", None)) and callable(
                getattr(model, "stream_step", None)
            )
            if self.stream_stateful_experts and has_model_stream:
                adapter = StreamingExpertAdapter(self, idx, expert, allow_model_stream=True)
            elif self.stream_stateful_experts and self._fastenhancer_core(expert) is not None:
                adapter = FastEnhancerStreamingExpertAdapter(self, idx, expert)
            elif self.stream_stateful_experts and self._ulunas_core(expert) is not None:
                adapter = ULUNASStreamingExpertAdapter(self, idx, expert)
            elif self.stream_stateful_experts and self._lisen_core(expert) is not None:
                adapter = LiSenStreamingExpertAdapter(self, idx, expert)
            else:
                adapter = StreamingExpertAdapter(self, idx, expert, allow_model_stream=False)

            if adapter.fallback:
                if self.stream_strict_expert_state:
                    raise RuntimeError(
                        f"Expert {expert.name} does not expose a native streaming cache API. "
                        "Disable inference_branch.strict_expert_state or add init_stream_state/stream_step."
                    )
                logger.warning(
                    f"[FrozenExpertRouterGRPO] expert {expert.name} uses fallback_full_forward stream adapter; "
                    "this preserves old frame-by-frame output but does not speed up per-hop inference."
                )
            else:
                logger.info(
                    f"[FrozenExpertRouterGRPO] expert {expert.name} uses true streaming adapter: {adapter.kind}."
                )
            adapters.append(adapter)
        return adapters

    def _stream_expert_adapter_modes(self):
        runtime = self.__dict__.get("_onnx_stream_runtime", None)
        if self._use_onnx_stream_inference() and runtime is not None:
            return runtime.adapter_modes()
        return [
            {
                "expert": adapter.expert.name,
                "mode": adapter.kind,
                "true_cache": bool(not adapter.fallback),
                "device": str(self._expert_device(adapter.expert_idx)),
            }
            for adapter in getattr(self, "_stream_expert_adapters", [])
        ]

    def _stream_fallback_expert_count(self):
        runtime = self.__dict__.get("_onnx_stream_runtime", None)
        if self._use_onnx_stream_inference() and runtime is not None:
            return runtime.fallback_expert_count()
        return sum(1 for adapter in getattr(self, "_stream_expert_adapters", []) if adapter.fallback)

    def _stream_cache_hit(self):
        adapters = getattr(self, "_stream_expert_adapters", [])
        return bool(adapters) and self._stream_fallback_expert_count() == 0

    def _log_stream_adapter_mode_metrics(self, batch_size=1):
        for idx, adapter in enumerate(getattr(self, "_stream_expert_adapters", [])):
            safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in adapter.expert.name)
            self._log_if_trainer_attached(
                f"stream/expert_{idx}_{safe_name}_true_cache",
                torch.as_tensor(
                    0.0 if adapter.fallback else 1.0,
                    dtype=torch.float32,
                    device=self._grpo_device(),
                ),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=max(int(batch_size), 1),
            )

    def _new_fastenhancer_stream_state(self, expert, device=None, dtype=torch.float32):
        device = device or self.device
        core = self._fastenhancer_core(expert)
        if core is None:
            raise RuntimeError(f"Expert {expert.name} has no FastEnhancer streaming core.")
        stft = core.stft
        n_fft = int(stft.n_fft)
        hop_size = int(stft.hop_size)
        return {
            "kind": "fastenhancer_native",
            "stft_cache": torch.zeros(1, n_fft - hop_size, device=device, dtype=dtype),
            "istft_cache": torch.zeros(1, n_fft - hop_size, device=device, dtype=dtype),
            "model_cache": [
                item.to(device=device, dtype=dtype)
                for item in core.initialize_cache(torch.zeros(1, hop_size, device=device, dtype=dtype))
            ],
            "window_istft": self._stream_istft_window(stft.window.to(device=device, dtype=dtype), n_fft, hop_size),
        }

    def _new_stream_expert_state(self, expert_idx, device=None, dtype=torch.float32):
        expert = self.experts[expert_idx]
        device = device or self.device
        core = self._fastenhancer_core(expert)
        if core is not None:
            return self._new_fastenhancer_stream_state(expert, device=device, dtype=dtype)

        if self.stream_strict_expert_state:
            raise RuntimeError(
                f"Expert {expert.name} does not expose a native streaming cache API. "
                "Disable inference_branch.strict_expert_state or add stream_step/init_stream_state to the model."
            )
        if expert_idx not in self._stream_fallback_warned:
            logger.warning(
                f"[FrozenExpertRouterGRPO] expert {expert.name} has no native stream cache; "
                f"using bounded waveform-history state ({self.stream_fallback_history_ms:g} ms)."
            )
            self._stream_fallback_warned.add(expert_idx)
        return {
            "kind": "history",
            "history": torch.zeros(0, device=device, dtype=dtype),
        }

    def _reset_stream_expert_states(self):
        self._stream_expert_states = [
            self._new_stream_expert_state(idx, device=self.device)
            for idx in range(len(self.experts))
        ]

    def _ensure_stream_expert_states(self, device=None, dtype=torch.float32):
        if (
            self._stream_expert_states is None
            or len(self._stream_expert_states) != len(self.experts)
        ):
            self._reset_stream_expert_states()
        device = device or self.device
        for idx, state in enumerate(self._stream_expert_states):
            if state is None:
                self._stream_expert_states[idx] = self._new_stream_expert_state(idx, device=device, dtype=dtype)
                continue
            probe = state.get("stft_cache", state.get("history"))
            if probe is not None and (probe.device != device or probe.dtype != dtype):
                self._stream_expert_states[idx] = self._new_stream_expert_state(idx, device=device, dtype=dtype)

    def _stream_expert_state_kinds(self):
        if self._use_onnx_stream_inference() and self._onnx_stream_state is not None:
            return [str(state.get("protocol", "onnx")) for state in self._onnx_stream_state.expert_states]
        if not self._stream_expert_states:
            return []
        return [str(state.get("kind", "unknown")) for state in self._stream_expert_states]

    def _fastenhancer_stream_step(self, expert, hop, state):
        core = self._fastenhancer_core(expert)
        if core is None:
            raise RuntimeError(f"Expert {expert.name} has no FastEnhancer streaming core.")
        stft = core.stft
        hop = hop.reshape(1, -1)
        hop_size = int(stft.hop_size)
        if hop.shape[-1] < hop_size:
            hop = F.pad(hop, (0, hop_size - hop.shape[-1]))
        elif hop.shape[-1] > hop_size:
            hop = hop[..., -hop_size:]

        window = stft.window.to(device=hop.device, dtype=hop.dtype)
        x = torch.cat([state["stft_cache"].to(device=hop.device, dtype=hop.dtype), hop], dim=1)
        state["stft_cache"] = x[:, -state["stft_cache"].shape[-1]:].detach()
        spec = torch.fft.rfft(x * window, n=int(stft.n_fft), dim=1)
        spec = torch.view_as_real(spec).unsqueeze(2)
        if getattr(stft, "discard_last_freq_bin", False):
            spec = spec[:, :-1, :, :]
        compression = float(getattr(stft, "compression", 1.0))
        eps = float(getattr(stft, "eps", 1.0e-5))
        mag = torch.linalg.norm(spec, dim=-1, keepdim=True).clamp(min=eps)
        spec_noisy = spec * mag.pow(compression - 1.0)

        mask, cache_out = core.model_forward(spec_noisy, *state["model_cache"])
        state["model_cache"] = [item.detach() for item in cache_out]
        spec_hat = torch.view_as_complex(spec_noisy.contiguous()) * torch.view_as_complex(mask.contiguous())
        mag_hat = spec_hat.abs().clamp_min(eps)
        spec_hat = spec_hat * mag_hat.pow(1.0 / compression - 1.0)
        if getattr(stft, "discard_last_freq_bin", False):
            spec_hat = F.pad(spec_hat, (0, 0, 0, 1))

        frame = torch.fft.irfft(spec_hat.squeeze(-1), n=int(stft.n_fft), dim=1)
        frame = frame * state["window_istft"].to(device=hop.device, dtype=hop.dtype)
        istft_cache = state["istft_cache"].to(device=hop.device, dtype=hop.dtype)
        frame[:, :istft_cache.shape[-1]] += istft_cache
        out = frame[:, :hop_size]
        state["istft_cache"] = frame[:, hop_size:].detach()
        return out, state

    def _history_stream_step(self, expert, hop, state):
        history = state["history"].to(device=hop.device, dtype=hop.dtype)
        history = torch.cat([history, hop.reshape(-1)], dim=0)
        if history.shape[-1] > self.stream_fallback_history_samples:
            history = history[-self.stream_fallback_history_samples:]
        with torch.no_grad():
            out = self._expert_enhanced_wav(expert, history.unsqueeze(0), target_len=history.shape[-1])
        out = out[..., -hop.shape[-1]:]
        state["history"] = history.detach()
        return out, state

    def _stream_expert_step_task(self, expert_idx, hop, state):
        expert = self.experts[expert_idx]
        start = time.perf_counter()
        with torch.no_grad():
            expert.eval()
            if str(state.get("kind", "")).startswith("fastenhancer"):
                est_wav, state = self._fastenhancer_stream_step(expert, hop, state)
            else:
                est_wav, state = self._history_stream_step(expert, hop, state)
            est_wav = TensorUtils.align_waveform_length(TensorUtils.safe_nan_to_num(est_wav), hop.shape[-1]).detach()
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in expert.name)
        return expert_idx, est_wav, state, f"expert_{expert_idx}_{safe_name}", (time.perf_counter() - start) * 1000.0

    def _stream_expert_hops(self, hop):
        self._ensure_stream_expert_states(device=hop.device, dtype=hop.dtype)
        total_start = self._latency_stamp()
        if (
            not self.stream_parallel_experts
            or len(self.experts) <= 1
            or self.stream_parallel_expert_workers <= 1
        ):
            results = [
                self._stream_expert_step_task(idx, hop, self._stream_expert_states[idx])
                for idx in range(len(self.experts))
            ]
        else:
            executor = self._ensure_stream_expert_executor()
            futures = [
                executor.submit(self._stream_expert_step_task, idx, hop, self._stream_expert_states[idx])
                for idx in range(len(self.experts))
            ]
            results = [future.result() for future in futures]

        results.sort(key=lambda item: item[0])
        wavs = []
        for idx, wav, state, latency_name, value_ms in results:
            self._stream_expert_states[idx] = state
            wavs.append(wav)
            self._record_latency(latency_name, value_ms, log_metrics=False, batch_size=1)
        self._record_latency(
            "experts_total",
            self._latency_elapsed_ms(total_start),
            log_metrics=False,
            batch_size=1,
        )
        return torch.stack(wavs, dim=1)

    def _stack_expert_wavs_parallel_frame(self, noisy, noisy_spec=None, target_len=None):
        if noisy_spec is None:
            noisy_spec = self.stft.apply_stft(self.padding(noisy))
        target_len = noisy.shape[-1] if target_len is None else int(target_len)
        total_start = self._latency_stamp()
        if len(self.experts) <= 1 or self.stream_parallel_expert_workers <= 1:
            return self._stack_expert_wavs(noisy, noisy_spec, target_len=target_len)

        executor = self._ensure_stream_expert_executor()
        futures = [
            executor.submit(self._stream_expert_wav_task, expert_idx, noisy, noisy_spec, target_len)
            for expert_idx in range(len(self.experts))
        ]
        results = [future.result() for future in futures]
        results.sort(key=lambda item: item[0])
        wavs = [item[1] for item in results]

        total_ms = self._latency_elapsed_ms(total_start)
        self._record_latency("experts_total", total_ms, log_metrics=False, batch_size=noisy.shape[0])
        for _, _, latency_name, value_ms in results:
            self._record_latency(latency_name, value_ms, log_metrics=False, batch_size=noisy.shape[0])
        return torch.stack(wavs, dim=1)

    @staticmethod
    def _router_temperature(router):
        return max(float(getattr(router, "temperature", 1.0)), 1e-6)

    def _router_outputs(self, noisy, noisy_spec, router):
        out = router(noisy, noisy_spec, return_logits=True)
        if isinstance(out, tuple) and len(out) == 2:
            weights, logits = out
        else:
            weights = out
            logits = torch.log(weights.clamp_min(EPS))
        return weights, logits

    def _weights_from_logits(self, logits, router=None):
        router = router or self.model
        return torch.softmax(logits / self._router_temperature(router), dim=-1)

    def _combine_expert_waveforms(self, weights, expert_wavs):
        if weights.shape[-1] != expert_wavs.shape[1]:
            raise RuntimeError(
                f"Router returned {weights.shape[-1]} weights for {expert_wavs.shape[1]} experts."
            )
        mean = (weights[:, :, None] * expert_wavs).sum(dim=1)
        return torch.clamp(TensorUtils.safe_nan_to_num(mean), -1.0, 1.0)

    def _router_policy_mean(
        self, noisy, noisy_spec, router, return_weights=False, target_len=None, parallel_experts=False
    ):
        if parallel_experts:
            expert_wavs = self._stack_expert_wavs_parallel_frame(noisy, noisy_spec, target_len=target_len)
        else:
            expert_wavs = self._stack_expert_wavs(noisy, noisy_spec, target_len=target_len)
        weights, _ = self._router_outputs(noisy, noisy_spec, router)
        mean = self._combine_expert_waveforms(weights, expert_wavs)
        if return_weights:
            return mean, weights
        return mean

    def _sample_moe_params_with_logprob(self, mean_logits, router=None):
        dtype = mean_logits.dtype
        std = torch.as_tensor(self.sigma, device=mean_logits.device, dtype=dtype)
        noise = torch.randn_like(mean_logits)
        action_logits = mean_logits + std * noise
        action_weights = self._weights_from_logits(action_logits, router)
        logp = self._log_prob(action_logits, mean_logits, std)
        return action_logits, action_weights, logp.detach(), std.detach(), noise

    def _policy_forward_stable(self, noisy_flat, T_wav, model=None, force_eval=False):
        router = model or self.model
        was_training = router.training
        if force_eval:
            router.eval()
        else:
            router.train()

        module_states = []
        if self.policy_eval_mode or force_eval:
            for module in router.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                      nn.LayerNorm, nn.Dropout)):
                    module_states.append((module, module.training))
                    module.eval()
        try:
            noisy_spec = self.stft.apply_stft(self.padding(noisy_flat))
            return self._router_policy_mean(noisy_flat, noisy_spec, router, target_len=T_wav)
        finally:
            for module, training in module_states:
                module.train(training)
            router.train(was_training)

    @staticmethod
    def _chunks_to_trajectory_windows(chunk_wav, n_chunks, n_actions):
        return chunk_wav.reshape(n_chunks, n_actions, -1).permute(1, 0, 2).reshape(n_actions, -1)

    # REFACTOR: Merge chunk-level and rolling-window GRPO sampling. The expensive
    # STFT/expert/router/action path is shared; use_window_reward only changes
    # the final reward view and return reshaping.
    @torch.no_grad()
    def _collect_grpo_samples(self, noisy_group, clean_group=None, use_window_reward=False):
        collect_start = self._latency_stamp()
        G, K = noisy_group.shape[0], self.num_actions
        T_wav = noisy_group.shape[-1]
        noisy_rep = noisy_group.unsqueeze(1).expand(G, K, -1).reshape(G * K, -1).detach()
        clean_flat = (
            clean_group.unsqueeze(1).expand(G, K, -1).reshape(G * K, -1).detach()
            if clean_group is not None else None
        )

        self.model.eval()
        self.ref_model.eval()

        stft_start = self._latency_stamp()
        noisy_spec_flat = self.stft.apply_stft(self.padding(noisy_rep))
        stft_ms = self._latency_elapsed_ms(stft_start)
        expert_start = self._latency_stamp()
        expert_wavs = self._stack_expert_wavs(noisy_rep, noisy_spec_flat, target_len=T_wav)
        expert_ms = self._latency_elapsed_ms(expert_start)

        router_start = self._latency_stamp()
        weights, logits = self._router_outputs(noisy_rep, noisy_spec_flat, self.model)
        mean_wav = self._combine_expert_waveforms(weights, expert_wavs)
        action_logits, action_weights, log_probs, action_std, moe_noise = self._sample_moe_params_with_logprob(
            logits, self.model
        )
        action_wav = self._combine_expert_waveforms(action_weights, expert_wavs)

        ref_weights, ref_logits = self._router_outputs(noisy_rep, noisy_spec_flat, self.ref_model)
        ref_wav = self._combine_expert_waveforms(ref_weights, expert_wavs)
        det_wav = mean_wav

        router_ms = self._latency_elapsed_ms(router_start)

        if use_window_reward:
            est_reward_wav = self._chunks_to_trajectory_windows(action_wav, G, K)
            base_reward_wav = self._chunks_to_trajectory_windows(ref_wav, G, K)
            det_reward_wav = self._chunks_to_trajectory_windows(det_wav, G, K)
            noisy_reward_wav = noisy_group.reshape(1, -1).expand(K, -1).contiguous()
            if clean_group is not None and self.distortion_to_clean:
                dist_target = clean_group.reshape(1, -1).expand(K, -1).contiguous()
            else:
                dist_target = base_reward_wav
        else:
            est_reward_wav = action_wav
            base_reward_wav = ref_wav
            det_reward_wav = det_wav
            noisy_reward_wav = noisy_rep[..., :T_wav]
            dist_target = clean_flat[..., :T_wav] if (clean_flat is not None and self.distortion_to_clean) else ref_wav

        reward_start = self._latency_stamp()
        rew = self.compute_reward(
            est_reward_wav,
            noisy_reward_wav,
            base_wav=base_reward_wav,
            distortion_target=dist_target,
        )
        ref_rew = self.compute_reward(
            base_reward_wav,
            noisy_reward_wav,
            base_wav=base_reward_wav,
            distortion_target=dist_target,
        )
        det_rew = self.compute_reward(
            det_reward_wav,
            noisy_reward_wav,
            base_wav=base_reward_wav,
            distortion_target=dist_target,
        )
        reward_ms = self._latency_elapsed_ms(reward_start)

        latency_ms = {
            "grpo_sample_total": self._latency_elapsed_ms(collect_start),
            "grpo_sample_stft": stft_ms,
            "grpo_sample_experts": expert_ms,
            "grpo_sample_router": router_ms,
            "grpo_sample_reward": reward_ms,
        }
        latency_ms.update({
            name: value_ms
            for name, value_ms in self._last_latency_ms.items()
            if name == "experts_total" or name.startswith("expert_")
        })

        sample = {
            "G": G, "K": K, "T_wav": T_wav,
            "noisy_flat": noisy_rep.detach(),
            "noisy_spec_flat": noisy_spec_flat.detach(),
            "expert_wavs": expert_wavs.detach(),
            "actions": action_wav.detach(),
            "moe_actions": action_logits.detach(),
            "action_weights": action_weights.detach(),
            "log_probs": TensorUtils.safe_nan_to_num(log_probs).detach(),
            "action_std": action_std.detach() if torch.is_tensor(action_std) else action_std,
            "ref_mean": ref_wav.detach(),
            "ref_moe_mean": ref_logits.detach(),
            "baseline_wav": ref_wav.detach(),
            "clean_flat": clean_flat[..., :T_wav].detach() if clean_flat is not None else None,
            "router_weights": weights.detach(),
            "action_router_weights": action_weights.detach(),
            "ref_router_weights": ref_weights.detach(),
            "moe_param_noise": moe_noise.detach(),
            "latency_ms": latency_ms,
        }

        if use_window_reward:
            window_rewards = TensorUtils.safe_nan_to_num(rew["reward"]).detach()
            mos = TensorUtils.safe_nan_to_num(rew["mos"]).detach()
            mos_abs = TensorUtils.safe_nan_to_num(rew["mos_abs"]).detach()
            distortion = TensorUtils.safe_nan_to_num(rew["distortion"]).detach()
            ref_rewards = TensorUtils.safe_nan_to_num(ref_rew["reward"]).detach()
            det_rewards = TensorUtils.safe_nan_to_num(det_rew["reward"]).detach()
            ref_mos_abs = TensorUtils.safe_nan_to_num(ref_rew["mos_abs"]).detach()
            det_mos_abs = TensorUtils.safe_nan_to_num(det_rew["mos_abs"]).detach()
            sample.update({
                "rewards": window_rewards.unsqueeze(0).expand(G, K).reshape(G * K).detach(),
                "ref_rewards": ref_rewards[:1].expand(G).detach(),
                "det_rewards": det_rewards[:1].expand(G).detach(),
                "mos": mos.unsqueeze(0).expand(G, K).reshape(G * K).detach(),
                "mos_abs": mos_abs.unsqueeze(0).expand(G, K).reshape(G * K).detach(),
                "distortion": distortion.unsqueeze(0).expand(G, K).reshape(G * K).detach(),
                "ref_mos_abs": ref_mos_abs[:1].expand(G).detach(),
                "det_mos_abs": det_mos_abs[:1].expand(G).detach(),
                "window_rewards": window_rewards,
                "window_mos_abs": mos_abs,
                "window_chunks": G,
                "window_len_samples": noisy_reward_wav.shape[-1],
            })
        else:
            rewards = TensorUtils.safe_nan_to_num(rew["reward"]).detach()
            ref_rewards_flat = TensorUtils.safe_nan_to_num(ref_rew["reward"]).detach()
            det_rewards_flat = TensorUtils.safe_nan_to_num(det_rew["reward"]).detach()
            sample.update({
                "rewards": rewards,
                "ref_rewards": ref_rewards_flat.reshape(G, K)[:, 0].detach(),
                "det_rewards": det_rewards_flat.reshape(G, K)[:, 0].detach(),
                "mos": TensorUtils.safe_nan_to_num(rew["mos"]).detach(),
                "mos_abs": TensorUtils.safe_nan_to_num(rew["mos_abs"]).detach(),
                "distortion": TensorUtils.safe_nan_to_num(rew["distortion"]).detach(),
                "ref_mos_abs": TensorUtils.safe_nan_to_num(ref_rew["mos_abs"]).reshape(G, K)[:, 0].detach(),
                "det_mos_abs": TensorUtils.safe_nan_to_num(det_rew["mos_abs"]).reshape(G, K)[:, 0].detach(),
            })
        return sample

    def _collect_flowgrpo_samples(self, noisy_group, clean_group=None):
        return self._collect_grpo_samples(noisy_group, clean_group, use_window_reward=bool(self.window_reward))

    def _compute_current_logp_and_kl_mse(self, sample):
        noisy = sample["noisy_flat"]
        T = sample["T_wav"]
        noisy_spec = sample.get("noisy_spec_flat")
        if noisy_spec is None:
            noisy_spec = self.stft.apply_stft(self.padding(noisy))

        expert_wavs = sample.get("expert_wavs")
        if expert_wavs is None:
            expert_wavs = self._stack_expert_wavs(noisy, noisy_spec, target_len=T)

        weights, logits = self._router_outputs(noisy, noisy_spec, self.model)
        preds = self._combine_expert_waveforms(weights, expert_wavs)

        action_logits = sample["moe_actions"]
        log_prob = TensorUtils.safe_nan_to_num(self._log_prob(action_logits, logits, sample.get("action_std", self.sigma)))

        grpo_device = self._grpo_device()
        kl = torch.zeros((), device=grpo_device)
        if self.beta > 0:
            kl = self._kl_between(logits, sample["ref_moe_mean"])

        mse = torch.zeros((), device=grpo_device)
        if self.lambda_mse > 0:
            target_wav = None
            if self.mse_target == "clean":
                target_wav = sample.get("clean_flat")
            elif self.mse_target in ("ref", "pretrained", "sft"):
                target_wav = sample.get("baseline_wav")
            if target_wav is not None:
                mse = self.speech_distortion(preds, target_wav[..., :T]).mean()
        return preds, log_prob, kl, mse

    def forward(self, noisy_wav, clean_wav=None, train=True, ret_weights=False):
        self._apply_device_map()
        router_device = self._router_device()
        noisy_wav = noisy_wav.to(router_device, non_blocking=True)
        if clean_wav is not None:
            clean_wav = clean_wav.to(router_device, non_blocking=True)
        forward_start = self._latency_stamp()
        stft_start = self._latency_stamp()
        noisy_spec = self.stft.apply_stft(noisy_wav)
        stft_ms = self._latency_elapsed_ms(stft_start)
        policy_start = self._latency_stamp()
        est_wav, weights = self._router_policy_mean(
            noisy_wav, noisy_spec, self.model, return_weights=True,
            target_len=noisy_wav.shape[-1]
        )
        policy_ms = self._latency_elapsed_ms(policy_start)
        total_ms = self._latency_elapsed_ms(forward_start)
        batch_size = noisy_wav.shape[0] if noisy_wav.ndim > 1 else 1
        self._record_latency("infer_forward_total", total_ms, log_metrics=not train, batch_size=batch_size)
        self._record_latency("infer_stft", stft_ms, log_metrics=not train, batch_size=batch_size)
        self._record_latency("infer_policy", policy_ms, log_metrics=not train, batch_size=batch_size)
        self._record_latency("infer_experts", self._last_latency_ms.get("experts_total", 0.0),
                             log_metrics=not train, batch_size=batch_size)
        if (not train) or clean_wav is None:
            return (est_wav, weights) if ret_weights else est_wav

        min_len = min(clean_wav.shape[-1], est_wav.shape[-1], noisy_wav.shape[-1])
        clean = clean_wav[..., :min_len]
        est = est_wav[..., :min_len]
        noisy = noisy_wav[..., :min_len]
        if self.loss is None:
            loss = torch.zeros((), device=est.device, dtype=est.dtype)
        elif self.loss_name == "wSDRLoss":
            loss = self.loss(clean, est, noisy)
        elif self.loss_name in ["CMSELoss"]:
            loss = self.loss(self.stft.apply_stft(clean), self.stft.apply_stft(est))
        else:
            loss = self.loss(clean, est)
        return loss, est_wav

    @staticmethod
    def _pad_stream_chunk(chunk, target_len):
        pad = int(target_len) - int(chunk.shape[-1])
        if pad <= 0:
            return chunk
        if chunk.shape[-1] == 0:
            return F.pad(chunk, (0, pad))
        return torch.cat([chunk, chunk[..., -1:].expand(*chunk.shape[:-1], pad)], dim=-1)

    def _use_onnx_stream_inference(self):
        return self.stream_online_infer and self.stream_inference_runtime in ("onnx", "ort", "onnxruntime")

    def _resolve_stream_onnx_path(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        try:
            return Path(resolve_path(str(path)))
        except Exception:
            return Path.cwd() / path

    def _stream_onnx_manifest_path(self):
        manifest = self.stream_onnx_manifest
        if not manifest:
            manifest = Path(self.stream_onnx_export_dir) / "manifest.json"
        return self._resolve_stream_onnx_path(manifest)

    def _stream_onnx_router_snapshot_path(self):
        return self._resolve_stream_onnx_path(self.stream_onnx_router_path)

    # REFACTOR: Decorator replaces repeated perf_counter/_record_latency boilerplate
    # for infrequent ONNX export I/O while keeping the metric key unchanged.
    @log_latency(name="router_onnx_export", log_metrics=True, batch_size=1)
    def _export_router_onnx_snapshot(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        class _RouterFeatureOnnxWrapper(nn.Module):
            def __init__(self, router):
                super().__init__()
                self.router = router

            def forward(self, features):
                router = self.router
                if getattr(router, "fixed_logits", None) is not None:
                    logits = router.fixed_logits.to(device=features.device, dtype=features.dtype)
                    logits = logits.unsqueeze(0).expand(features.shape[0], -1)
                else:
                    logits = router.net(features) + router.logit_bias.to(device=features.device, dtype=features.dtype)
                weights = torch.softmax(logits / max(float(getattr(router, "temperature", 1.0)), 1.0e-6), dim=-1)
                return weights, logits

        router_device = self._router_device()
        was_training = self.model.training
        wrapper = _RouterFeatureOnnxWrapper(deepcopy(self.model).to(router_device).eval())
        dummy = torch.zeros(1, int(getattr(self.model, "input_dim", 9)), device=router_device, dtype=torch.float32)
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                str(path),
                input_names=["features"],
                output_names=["weights", "logits"],
                dynamic_axes={
                    "features": {0: "batch"},
                    "weights": {0: "batch"},
                    "logits": {0: "batch"},
                },
                opset_version=int(self.stream_onnx_conf.get("opset", 17)),
                do_constant_folding=True,
            )
        self.model.train(was_training)
        logger.info(f"[FrozenExpertRouterGRPO] exported router ONNX snapshot: {path}")
        return path

    @log_latency(name="router_onnx_reload", log_metrics=True, batch_size=1)
    def _reload_stream_onnx_router_snapshot(self, runtime, router_path):
        runtime.reload_router(router_path.resolve() if router_path.exists() else router_path)

    def _sync_stream_onnx_router_snapshot(self, force=False):
        if (
            not self._use_onnx_stream_inference()
            or not self.stream_onnx_use_onnx_router
            or not self.stream_onnx_router_live_export
        ):
            return
        if not force:
            self._stream_onnx_router_sync_count = int(getattr(self, "_stream_onnx_router_sync_count", 0)) + 1
            if self._stream_onnx_router_sync_count % int(self.stream_onnx_router_sync_interval_steps) != 0:
                return
        router_path = self._stream_onnx_router_snapshot_path()
        self._export_router_onnx_snapshot(router_path)
        runtime = self.__dict__.get("_onnx_stream_runtime", None)
        if runtime is not None:
            self._reload_stream_onnx_router_snapshot(runtime, router_path)
            logger.info("[FrozenExpertRouterGRPO] reloaded ONNX router snapshot; expert cache states were kept.")

    def _assert_onnx_cuda_only_runtime(self, runtime):
        if not self.stream_onnx_cuda_only or runtime is None:
            return
        checks = []
        router = getattr(runtime, "router", None)
        router_session = getattr(router, "session", None)
        if router_session is not None:
            checks.append(("router", router_session.get_providers()))
        for idx, expert in enumerate(getattr(runtime, "experts", [])):
            session = getattr(expert, "session", None)
            if session is not None:
                name = getattr(expert, "name", f"expert_{idx}")
                checks.append((name, session.get_providers()))
        missing = [(name, providers) for name, providers in checks if "CUDAExecutionProvider" not in providers]
        if missing:
            details = "; ".join(f"{name}: {providers}" for name, providers in missing)
            raise RuntimeError(
                "router_grpo.inference_branch.onnx.cuda_only=true, but some ONNX Runtime sessions "
                f"did not activate CUDAExecutionProvider: {details}"
            )

    def _ensure_onnx_stream_runtime(self):
        if not self._use_onnx_stream_inference():
            return None
        runtime = self.__dict__.get("_onnx_stream_runtime", None)
        if runtime is not None:
            return runtime

        manifest_path = self._stream_onnx_manifest_path()
        if not manifest_path.exists():
            message = (
                f"ONNX stream manifest not found: {manifest_path}. "
                "Run tools/export_grpo_onnx.py for this config first, or set "
                "router_grpo.inference_branch.runtime=torch."
            )
            if self.stream_onnx_strict:
                raise FileNotFoundError(message)
            logger.warning(f"[FrozenExpertRouterGRPO] {message} Falling back to PyTorch stream inference.")
            self.stream_inference_runtime = "torch"
            return None

        if self.stream_onnx_use_onnx_router and self.stream_onnx_router_live_export:
            self._sync_stream_onnx_router_snapshot(force=True)
        if self.stream_onnx_use_iobinding:
            try:
                from . import grpo_onnx_iobinding as _grpo_onnx_iobinding
            except ImportError as exc:
                raise ImportError(
                    "Failed to import alpha.enh.system.grpo_onnx_iobinding. "
                    "Make sure grpo_onnx_iobinding.py was copied to the server together with grpo.py."
                ) from exc
            TorchOnnxMoEStreamRuntime = getattr(
                _grpo_onnx_iobinding,
                "TorchOnnxMoEStreamRuntime",
                None,
            )
            if TorchOnnxMoEStreamRuntime is None:
                raise ImportError(
                    "alpha.enh.system.grpo_onnx_iobinding does not define "
                    "TorchOnnxMoEStreamRuntime. The server is running an incomplete or stale "
                    "grpo_onnx_iobinding.py; copy the latest file from this workspace."
                )

            device_id = (
                int(self.stream_onnx_force_provider_device_id)
                if self.stream_onnx_force_provider_device_id is not None
                else (self._router_device().index if self._router_device().index is not None else 0)
            )
            runtime = TorchOnnxMoEStreamRuntime(
                manifest_path,
                device=f"cuda:{device_id}",
                device_id=device_id,
                sample_rate=int(self.sample_rate),
                frame_samples=int(self.stream_frame_samples),
                hop_samples=int(self.stream_hop_samples),
                stft_conf=_to_plain_dict(self.conf.get("stft", {})),
                parallel_experts=self.stream_parallel_experts,
                parallel_cuda_streams=self.stream_onnx_parallel_cuda_streams,
                enable_profiling=self.stream_onnx_enable_profiling,
                use_onnx_router=self.stream_onnx_use_onnx_router,
            )
        else:
            runtime = OnnxMoEStreamRuntime(
                manifest_path,
                providers=self.stream_onnx_providers,
                provider_options=self.stream_onnx_provider_options,
                override_manifest_providers=self.stream_onnx_override_manifest_providers,
                sample_rate=int(self.sample_rate),
                frame_samples=int(self.stream_frame_samples),
                hop_samples=int(self.stream_hop_samples),
                stft_conf=_to_plain_dict(self.conf.get("stft", {})),
                parallel_experts=self.stream_parallel_experts,
                parallel_workers=self.stream_parallel_expert_workers,
                parallel_cuda_streams=self.stream_onnx_parallel_cuda_streams,
                use_onnx_router=self.stream_onnx_use_onnx_router,
            )
        self._assert_onnx_cuda_only_runtime(runtime)
        self.__dict__["_onnx_stream_runtime"] = runtime
        if self.stream_onnx_use_iobinding and not self.stream_onnx_use_onnx_router:
            logger.info("[MoE ONNX Runtime] experts=ONNX, router=PyTorch, fusion=PyTorch")
        elif self.stream_onnx_use_iobinding:
            logger.info("[MoE ONNX Runtime] experts=ONNX, router=ONNX, fusion=PyTorch")
        elif not self.stream_onnx_use_onnx_router:
            logger.info("[MoE ONNX Runtime] experts=ONNX/CPU, router=PyTorch, fusion=PyTorch")
        logger.info(
            "[FrozenExpertRouterGRPO] ONNX online inference branch ready: "
            f"manifest={manifest_path}, providers={runtime.providers}, "
            f"override_manifest_providers={runtime.override_manifest_providers}, "
            f"iobinding={self.stream_onnx_use_iobinding}, "
            f"cuda_only={self.stream_onnx_cuda_only}, "
            f"use_onnx_router={self.stream_onnx_use_onnx_router}, "
            f"parallel_experts={runtime.parallel_experts}, "
            f"parallel_workers={runtime.parallel_workers}, "
            f"parallel_cuda_streams={runtime.parallel_cuda_streams}, "
            f"expert_stream_modes={runtime.adapter_modes()}"
        )
        return runtime

    def _reset_onnx_stream_state(self):
        runtime = self._ensure_onnx_stream_runtime()
        if runtime is None:
            self._onnx_stream_state = None
            return
        self._onnx_stream_state = self._coerce_moe_stream_state(
            runtime.create_state(batch_size=1),
            device=self._onnx_runtime_state_device(runtime),
        )

    def _new_moe_stream_state(self, device=None, dtype=torch.float32, batch_size=1):
        self._apply_device_map()
        device = torch.device(device or self._router_device())
        batch_size = int(batch_size)
        context_len = max(int(self.stream_frame_samples) - int(self.stream_hop_samples), 0)
        input_tail = torch.zeros(batch_size, context_len, dtype=dtype, device=device)
        output_tail = torch.zeros(batch_size, 0, dtype=dtype, device=device)
        ola_buffer = torch.zeros(batch_size, max(context_len, 0), dtype=dtype, device=device)
        expert_states = []
        for idx, adapter in enumerate(self._stream_expert_adapters):
            try:
                expert_states.append(
                    adapter.init_state(device=self._expert_device(idx), batch_size=batch_size, dtype=dtype)
                )
            except Exception as exc:
                if self.stream_strict_expert_state:
                    raise
                fallback = StreamingExpertAdapter(self, idx, self.experts[idx], allow_model_stream=False)
                self._stream_expert_adapters[idx] = fallback
                logger.warning(
                    f"[FrozenExpertRouterGRPO] failed to initialize true stream state for "
                    f"expert {self.experts[idx].name} ({exc}); falling back to full-frame forward."
                )
                expert_states.append(
                    fallback.init_state(device=self._expert_device(idx), batch_size=batch_size, dtype=dtype)
                )
        state = MoEStreamState(
            input_tail=input_tail,
            expert_states=expert_states,
            router_state=None,
            output_tail=output_tail,
            ola_buffer=ola_buffer,
            num_steps=0,
        )
        self._stream_expert_states = state.expert_states
        self._stream_infer_context = state.input_tail.reshape(-1).detach() if batch_size == 1 else state.input_tail
        return state

    def _ensure_moe_stream_state(self, device=None, dtype=torch.float32, batch_size=1):
        self._apply_device_map()
        device = torch.device(device or self._router_device())
        context_len = max(int(self.stream_frame_samples) - int(self.stream_hop_samples), 0)
        state = self._moe_stream_state
        needs_reset = (
            state is None
            or state.input_tail.shape[0] != int(batch_size)
            or state.input_tail.shape[-1] != context_len
            or state.input_tail.device != device
            or state.input_tail.dtype != dtype
            or len(state.expert_states) != len(self._stream_expert_adapters)
        )
        if needs_reset:
            state = self._new_moe_stream_state(device=device, dtype=dtype, batch_size=batch_size)
            self._moe_stream_state = state
        self._stream_expert_states = state.expert_states
        self._stream_infer_context = state.input_tail.reshape(-1).detach() if int(batch_size) == 1 else state.input_tail
        return state

    def _reset_stream_epoch_profile(self):
        self._stream_epoch_profile = {
            "frames": 0.0,
            "input_cache_ms_sum": 0.0,
            "expert_stream_step_ms_sum": 0.0,
            "router_ms_sum": 0.0,
            "fusion_ms_sum": 0.0,
            "total_step_ms_sum": 0.0,
            "frame_rtf_sum": 0.0,
            "frame_rtf_max": 0.0,
            "cache_hit_sum": 0.0,
            "fallback_expert_count_sum": 0.0,
        }

    def _accumulate_stream_epoch_profile(
        self,
        *,
        input_cache_ms,
        expert_ms,
        router_ms,
        fusion_ms,
        total_ms,
        frame_rtf,
        cache_hit,
        fallback_count,
        batch_size=1,
    ):
        if not hasattr(self, "_stream_epoch_profile") or self._stream_epoch_profile is None:
            self._reset_stream_epoch_profile()
        weight = float(max(int(batch_size), 1))
        profile = self._stream_epoch_profile
        profile["frames"] += weight
        profile["input_cache_ms_sum"] += float(input_cache_ms) * weight
        profile["expert_stream_step_ms_sum"] += float(expert_ms) * weight
        profile["router_ms_sum"] += float(router_ms) * weight
        profile["fusion_ms_sum"] += float(fusion_ms) * weight
        profile["total_step_ms_sum"] += float(total_ms) * weight
        profile["frame_rtf_sum"] += float(frame_rtf) * weight
        profile["frame_rtf_max"] = max(float(profile["frame_rtf_max"]), float(frame_rtf))
        profile["cache_hit_sum"] += float(cache_hit) * weight
        profile["fallback_expert_count_sum"] += float(fallback_count) * weight

    def _stream_epoch_profile_summary(self):
        profile = getattr(self, "_stream_epoch_profile", None)
        if not profile or float(profile.get("frames", 0.0)) <= 0.0:
            return None
        frames = float(profile["frames"])
        return {
            "frames": frames,
            "input_cache_ms_mean": float(profile["input_cache_ms_sum"]) / frames,
            "expert_stream_step_ms_mean": float(profile["expert_stream_step_ms_sum"]) / frames,
            "router_ms_mean": float(profile["router_ms_sum"]) / frames,
            "fusion_ms_mean": float(profile["fusion_ms_sum"]) / frames,
            "total_step_ms_mean": float(profile["total_step_ms_sum"]) / frames,
            "frame_rtf_mean": float(profile["frame_rtf_sum"]) / frames,
            "frame_rtf_max": float(profile["frame_rtf_max"]),
            "cache_hit_mean": float(profile["cache_hit_sum"]) / frames,
            "fallback_expert_count_mean": float(profile["fallback_expert_count_sum"]) / frames,
        }

    def _record_stream_profile_ms(self, name, value_ms, batch_size=1):
        key = f"stream/{name}"
        value_ms = float(value_ms)
        self._last_latency_ms[key] = value_ms
        self._log_if_trainer_attached(
            f"{key}_ms",
            torch.as_tensor(value_ms, dtype=torch.float32, device=self._router_device()),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=max(int(batch_size), 1),
        )

    def _record_stream_profile_scalar(self, name, value, batch_size=1):
        key = f"stream/{name}"
        value = float(value)
        self._last_latency_ms[key] = value
        self._log_if_trainer_attached(
            key,
            torch.as_tensor(value, dtype=torch.float32, device=self._router_device()),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=max(int(batch_size), 1),
        )

    def _reset_stream_infer_state(self):
        if self._use_onnx_stream_inference():
            self._reset_onnx_stream_state()
            return
        self._moe_stream_state = self._new_moe_stream_state(
            device=self._router_device(),
            dtype=torch.float32,
            batch_size=1,
        )

    def _new_stream_infer_model(self):
        snapshot = deepcopy(self.model).to(self._router_device()).eval()
        freeze_model(snapshot)
        return snapshot

    def _ensure_stream_locks(self):
        if self._stream_infer_lock is None:
            self._stream_infer_lock = threading.RLock()
        if self._stream_train_lock is None:
            self._stream_train_lock = threading.RLock()

    def _get_stream_infer_model(self):
        return self.__dict__.get("_stream_infer_model", None)

    def _set_stream_infer_model(self, model):
        self.__dict__["_stream_infer_model"] = model

    def _sync_stream_infer_model(self):
        self._ensure_stream_locks()
        with self._stream_infer_lock:
            if self._use_onnx_stream_inference():
                runtime_exists = self.__dict__.get("_onnx_stream_runtime", None) is not None
                self._ensure_onnx_stream_runtime()
                if runtime_exists and self.stream_onnx_use_onnx_router:
                    self._sync_stream_onnx_router_snapshot()
                if not self.stream_onnx_use_onnx_router:
                    self._apply_device_map()
                    router_device = self._router_device()
                    infer_model = self._get_stream_infer_model()
                    if infer_model is None:
                        self._set_stream_infer_model(self._new_stream_infer_model())
                    else:
                        infer_model.to(router_device)
                        infer_model.load_state_dict(self.model.state_dict())
                        infer_model.eval()
                return
            self._apply_device_map()
            router_device = self._router_device()
            infer_model = self._get_stream_infer_model()
            if infer_model is None:
                self._set_stream_infer_model(self._new_stream_infer_model())
            else:
                infer_model.to(router_device)
                infer_model.load_state_dict(self.model.state_dict())
                infer_model.eval()

    def _ensure_stream_train_executor(self):
        if self._stream_train_executor is None:
            self._stream_train_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mos_moe_stream_train",
            )
        return self._stream_train_executor

    def _prune_stream_train_futures(self):
        while self._stream_train_futures and self._stream_train_futures[0].done():
            self._stream_train_futures.popleft()

    def _wait_stream_train_futures(self):
        futures = list(self._stream_train_futures)
        self._stream_train_futures.clear()
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                self._stream_train_error = exc
                logger.warning(f"[FrozenExpertRouterGRPO] async stream training failed: {exc}")

    def _on_stream_train_done(self, future):
        try:
            future.result()
        except Exception as exc:
            self._stream_train_error = exc
            logger.warning(f"[FrozenExpertRouterGRPO] async stream training failed: {exc}")
            return
        self._sync_stream_infer_model()

    def _shutdown_stream_executors(self):
        if self._stream_train_executor is not None:
            self._wait_stream_train_futures()
            self._stream_train_executor.shutdown(wait=True)
            self._stream_train_executor = None
        if self._stream_expert_executor is not None:
            self._stream_expert_executor.shutdown(wait=True)
            self._stream_expert_executor = None
        runtime = self.__dict__.get("_onnx_stream_runtime", None)
        if runtime is not None:
            runtime.close()

    def _run_stream_train_step(self, adapt_batch, batch_idx, update_reference=True):
        self._ensure_stream_locks()
        with self._stream_train_lock:
            opt = self._get_test_optimizer()
            was_training = self.model.training
            try:
                with torch.enable_grad():
                    self.model.train()
                    loss = self._grpo_update_step(
                        adapt_batch,
                        batch_idx,
                        opt,
                        update_reference=update_reference,
                        log_metrics=False,
                    )
            finally:
                self.model.train(was_training)
            if torch.is_tensor(loss):
                return float(loss.detach().cpu())
            return float(loss)

    def _schedule_stream_train(self, y_chunk, chunk_samples, stream_id):
        self._prune_stream_train_futures()
        if self.stream_train_queue_limit > 0 and len(self._stream_train_futures) >= self.stream_train_queue_limit:
            return False, "busy"

        batch_idx = int(getattr(self, "_stream_batch_idx", 0))
        self._stream_batch_idx = batch_idx + 1
        adapt_chunk = self._pad_stream_chunk(y_chunk.detach().clone(), chunk_samples).unsqueeze(0)
        adapt_batch = {"noisy_wav": adapt_chunk, "wav_id": [str(stream_id)]}

        if not self.stream_async_train:
            self._run_stream_train_step(adapt_batch, batch_idx, update_reference=True)
            self._sync_stream_infer_model()
            return True, "updated"

        executor = self._ensure_stream_train_executor()
        future = executor.submit(self._run_stream_train_step, adapt_batch, batch_idx, True)
        future.add_done_callback(self._on_stream_train_done)
        self._stream_train_futures.append(future)
        return True, "queued"

    def _stream_adapter_step_task(self, expert_idx, frame, expert_state):
        adapter = self._stream_expert_adapters[expert_idx]
        start = time.perf_counter()
        with torch.no_grad():
            adapter.expert.eval()
            y_hop, new_state = adapter.stream_step(frame, expert_state)
            y_hop = TensorUtils.align_waveform_length(TensorUtils.safe_nan_to_num(y_hop), self.stream_hop_samples)
            y_hop = y_hop.to(self._router_device(), non_blocking=True)
        value_ms = (time.perf_counter() - start) * 1000.0
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in adapter.expert.name)
        return expert_idx, y_hop.detach(), new_state, f"stream/expert_{expert_idx}_{safe_name}", value_ms

    def _stream_moe_step(self, x_hop, state=None):
        self._apply_device_map()
        self._ensure_stream_locks()
        if self.stream_report_frame_rtf:
            self._sync_stream_devices()
        total_start = self._latency_stamp()
        router_device = self._router_device()
        hop = torch.as_tensor(x_hop, dtype=torch.float32, device=router_device)
        if hop.ndim == 1:
            hop = hop.unsqueeze(0)
        if hop.ndim != 2:
            raise ValueError(f"_stream_moe_step expects [T] or [B,T] hop, got shape {tuple(hop.shape)}")
        batch_size = int(hop.shape[0])
        hop_len = int(self.stream_hop_samples)
        if hop.shape[-1] < hop_len:
            hop = F.pad(hop, (0, hop_len - hop.shape[-1]))
        elif hop.shape[-1] > hop_len:
            hop = hop[..., -hop_len:]

        with self._stream_infer_lock:
            state = state or self._ensure_moe_stream_state(
                device=hop.device,
                dtype=hop.dtype,
                batch_size=batch_size,
            )
            input_cache_start = self._latency_stamp()
            tail = state.input_tail.to(device=hop.device, dtype=hop.dtype)
            frame = torch.cat([tail, hop], dim=-1) if tail.numel() else hop
            frame_len = int(self.stream_frame_samples)
            if frame.shape[-1] < frame_len:
                frame = F.pad(frame, (frame_len - frame.shape[-1], 0))
            elif frame.shape[-1] > frame_len:
                frame = frame[..., -frame_len:]
            context_len = max(frame_len - hop_len, 0)
            state.input_tail = frame[..., -context_len:].detach() if context_len > 0 else frame.new_zeros(batch_size, 0)
            input_cache_ms = self._latency_elapsed_ms(input_cache_start)

            if self._get_stream_infer_model() is None:
                self._sync_stream_infer_model()
            router = self._get_stream_infer_model()
            router.eval()

            expert_start = self._latency_stamp()
            if (
                self.stream_parallel_experts
                and len(self._stream_expert_adapters) > 1
                and self.stream_parallel_expert_workers > 1
            ):
                executor = self._ensure_stream_expert_executor()
                futures = [
                    executor.submit(self._stream_adapter_step_task, idx, frame, state.expert_states[idx])
                    for idx in range(len(self._stream_expert_adapters))
                ]
                results = [future.result() for future in futures]
            else:
                results = [
                    self._stream_adapter_step_task(idx, frame, state.expert_states[idx])
                    for idx in range(len(self._stream_expert_adapters))
                ]
            results.sort(key=lambda item: item[0])
            expert_wavs = []
            for idx, wav, new_state, latency_name, value_ms in results:
                state.expert_states[idx] = new_state
                expert_wavs.append(wav)
                self._last_latency_ms[latency_name] = float(value_ms)
            expert_wavs = torch.stack(expert_wavs, dim=1)
            expert_ms = self._latency_elapsed_ms(expert_start)

            router_start = self._latency_stamp()
            noisy_spec = self.stft.apply_stft(frame)
            weights, _ = self._router_outputs(frame, noisy_spec, router)
            state.router_state = {"weights": weights.detach()}
            router_ms = self._latency_elapsed_ms(router_start)

            fusion_start = self._latency_stamp()
            est_wav = self._combine_expert_waveforms(weights, expert_wavs)
            est_wav = TensorUtils.align_waveform_length(est_wav, hop_len)
            if context_len > 0:
                prev_out_tail = state.output_tail.to(device=est_wav.device, dtype=est_wav.dtype)
                out_history = torch.cat([prev_out_tail, est_wav], dim=-1) if prev_out_tail.numel() else est_wav
                state.output_tail = out_history[..., -context_len:].detach()
                state.ola_buffer = state.output_tail
            else:
                state.output_tail = est_wav.new_zeros(batch_size, 0)
                state.ola_buffer = state.output_tail
            state.num_steps += 1
            self._moe_stream_state = state
            self._stream_expert_states = state.expert_states
            self._stream_infer_context = state.input_tail.reshape(-1).detach() if batch_size == 1 else state.input_tail
            fusion_ms = self._latency_elapsed_ms(fusion_start)

            if self.stream_report_frame_rtf:
                self._sync_stream_devices()
            total_ms = self._latency_elapsed_ms(total_start)
            frame_seconds = float(hop_len) / max(float(self.sample_rate), 1.0)
            frame_rtf = (total_ms / 1000.0) / max(frame_seconds, EPS)
            fallback_count = self._stream_fallback_expert_count()
            cache_hit = 1.0 if fallback_count == 0 else 0.0
            self._record_stream_profile_ms("input_cache", input_cache_ms, batch_size=batch_size)
            self._record_stream_profile_ms("expert_stream_step", expert_ms, batch_size=batch_size)
            self._record_stream_profile_ms("router", router_ms, batch_size=batch_size)
            self._record_stream_profile_ms("fusion", fusion_ms, batch_size=batch_size)
            self._record_stream_profile_ms("total_step", total_ms, batch_size=batch_size)
            self._record_stream_profile_scalar("frame_rtf", frame_rtf, batch_size=batch_size)
            self._record_stream_profile_scalar("cache_hit", cache_hit, batch_size=batch_size)
            self._record_stream_profile_scalar("fallback_expert_count", fallback_count, batch_size=batch_size)
            self._accumulate_stream_epoch_profile(
                input_cache_ms=input_cache_ms,
                expert_ms=expert_ms,
                router_ms=router_ms,
                fusion_ms=fusion_ms,
                total_ms=total_ms,
                frame_rtf=frame_rtf,
                cache_hit=cache_hit,
                fallback_count=fallback_count,
                batch_size=batch_size,
            )
        return est_wav.squeeze(0)[:hop_len], weights.detach(), state

    def _stream_infer_frame(self, frame, hop):
        est_hop, weights, _ = self._stream_moe_step(hop)
        return est_hop, weights

    # REFACTOR: Normalize ONNX stream state to MoEStreamState and keep persistent
    # state tensors; numpy exists only as a temporary ONNX Runtime feed format.
    def _onnx_runtime_state_device(self, runtime):
        return self._router_device() if getattr(runtime, "expects_torch_input", False) else torch.device("cpu")

    def _coerce_moe_stream_state(self, state, device=None):
        if state is None:
            return None
        return MoEStreamState(
            input_tail=TensorUtils.numpy_tree_to_tensor(getattr(state, "input_tail"), device=device),
            expert_states=TensorUtils.numpy_tree_to_tensor(getattr(state, "expert_states", []), device=device),
            router_state=TensorUtils.numpy_tree_to_tensor(getattr(state, "router_state", None), device=device),
            output_tail=TensorUtils.numpy_tree_to_tensor(getattr(state, "output_tail", None), device=device),
            ola_buffer=TensorUtils.numpy_tree_to_tensor(getattr(state, "ola_buffer", None), device=device),
            num_steps=int(getattr(state, "num_steps", 0)),
        )

    def _store_onnx_tail_state(self, state, y_hop_t, frame_weights, context_len, device):
        if context_len > 0:
            output_tail = y_hop_t[..., -context_len:].detach().to(device=device, dtype=torch.float32)
            if state.output_tail is not None and tuple(state.output_tail.shape) == tuple(output_tail.shape):
                state.output_tail.copy_(output_tail, non_blocking=True)
            else:
                state.output_tail = output_tail
            state.ola_buffer = state.output_tail
        else:
            state.output_tail = y_hop_t.new_zeros(y_hop_t.shape[0], 0).to(device=device, dtype=torch.float32)
            state.ola_buffer = state.output_tail
        state.router_state = {"weights": frame_weights.detach().to(device=device, dtype=torch.float32)}
        state.num_steps += 1

    def _record_onnx_stream_profile(self, profile, hop_len):
        for name, key in (
            ("input_cache", "input_cache_ms"),
            ("expert_stream_step", "expert_stream_step_ms"),
            ("router", "router_ms"),
            ("fusion", "fusion_ms"),
            ("total_step", "total_step_ms"),
        ):
            self._record_stream_profile_ms(name, profile.get(key, 0.0), batch_size=1)
        for latency_key in _ONNX_STREAM_LATENCY_KEYS:
            if latency_key in profile:
                metric_name = latency_key[:-3] if latency_key.endswith("_ms") else latency_key
                self._record_latency(metric_name, profile.get(latency_key, 0.0), log_metrics=True, batch_size=1)
        frame_seconds = float(hop_len) / max(float(self.sample_rate), 1.0)
        frame_rtf = (float(profile.get("total_step_ms", 0.0)) / 1000.0) / max(frame_seconds, EPS)
        self._record_stream_profile_scalar("frame_rtf", frame_rtf, batch_size=1)
        self._record_stream_profile_scalar("cache_hit", profile.get("cache_hit", 0.0), batch_size=1)
        self._record_stream_profile_scalar(
            "fallback_expert_count",
            profile.get("fallback_expert_count", 0.0),
            batch_size=1,
        )
        self._accumulate_stream_epoch_profile(
            input_cache_ms=profile.get("input_cache_ms", 0.0),
            expert_ms=profile.get("expert_stream_step_ms", 0.0),
            router_ms=profile.get("router_ms", 0.0),
            fusion_ms=profile.get("fusion_ms", 0.0),
            total_ms=profile.get("total_step_ms", 0.0),
            frame_rtf=frame_rtf,
            cache_hit=profile.get("cache_hit", 0.0),
            fallback_count=profile.get("fallback_expert_count", 0.0),
            batch_size=1,
        )

    def _torch_stream_step(self, hop, state):
        est_hop, frame_weights, state = self._stream_moe_step(hop, state=state)
        return est_hop, frame_weights, state

    def _router_weights_for_onnx_frame(self, frame_t):
        router_start = time.perf_counter()
        router = self._get_stream_infer_model()
        if router is None:
            self._sync_stream_infer_model()
            router = self._get_stream_infer_model() or self.model
        was_training = router.training
        router.eval()
        try:
            with torch.no_grad():
                noisy_spec_t = self.stft.apply_stft(frame_t)
                frame_weights, _ = self._router_outputs(frame_t, noisy_spec_t, router)
        finally:
            router.train(was_training)
        return frame_weights, (time.perf_counter() - router_start) * 1000.0

    def _merge_onnx_torch_profile(self, profile, router_torch_ms, fusion_ms):
        base_step_ms = float(
            profile.get(
                "total_step_ms",
                float(profile.get("input_cache_ms", 0.0))
                + float(profile.get("experts_parallel_wall_ms", 0.0)),
            )
        )
        total_ms = base_step_ms + float(router_torch_ms) + float(fusion_ms)
        profile["router_torch_ms"] = router_torch_ms
        profile["router_ms"] = router_torch_ms
        profile["fusion_ms"] = fusion_ms
        profile["total_step_ms"] = total_ms
        profile["stream_frame_total_ms"] = total_ms
        profile["frame_total_ms"] = total_ms

    @staticmethod
    def _stream_tensor(value, device, dtype):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype, non_blocking=True)
        return torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=dtype)

    def _onnx_stream_step(self, hop, state):
        runtime = self._ensure_onnx_stream_runtime()
        if runtime is None:
            return self._torch_stream_step(hop, self._moe_stream_state)

        hop_len = int(self.stream_hop_samples)
        state_device = self._onnx_runtime_state_device(runtime)
        state = self._coerce_moe_stream_state(state or self._onnx_stream_state, device=state_device)
        torch_input = bool(getattr(runtime, "expects_torch_input", False))
        use_torch_router = not self.stream_onnx_use_onnx_router and hasattr(runtime, "step_experts")
        hop_input = hop if torch_input else hop.detach().cpu().numpy().astype("float32", copy=False)

        if use_torch_router:
            expert_wavs, frame, state, profile = runtime.step_experts(hop_input, state=state)
            state = self._coerce_moe_stream_state(state, device=state_device)
            expert_wavs_t = self._stream_tensor(expert_wavs, hop.device, hop.dtype)
            frame_t = self._stream_tensor(frame, hop.device, hop.dtype)
            frame_weights, router_torch_ms = self._router_weights_for_onnx_frame(frame_t)
            fusion_start = time.perf_counter()
            spec_domain = torch_input and (
                bool(getattr(runtime, "outputs_spec_domain", lambda: False)())
                or str(profile.get("expert_output_domain", "")).lower() == "spec"
            )
            if spec_domain:
                y_hop_t, state, fusion_profile = runtime.fuse_expert_specs(
                    expert_wavs_t, frame_weights, state, profile=profile
                )
                state = self._coerce_moe_stream_state(state, device=state_device)
                fusion_ms = float(fusion_profile.get("fusion_ms", (time.perf_counter() - fusion_start) * 1000.0))
            else:
                y_hop_t = self._combine_expert_waveforms(frame_weights, expert_wavs_t)
                fusion_ms = (time.perf_counter() - fusion_start) * 1000.0
                context_len = max(int(self.stream_frame_samples) - int(self.stream_hop_samples), 0)
                self._store_onnx_tail_state(state, y_hop_t, frame_weights, context_len, state_device)
            y_hop = TensorUtils.align_waveform_length(y_hop_t, hop_len).reshape(-1)
            self._merge_onnx_torch_profile(profile, router_torch_ms, fusion_ms)
        else:
            y_hop_raw, frame_weights_raw, state, profile = runtime.step(hop_input, state=state)
            state = self._coerce_moe_stream_state(state, device=state_device)
            y_hop = self._stream_tensor(y_hop_raw, hop.device, hop.dtype).reshape(-1)
            frame_weights = self._stream_tensor(frame_weights_raw, hop.device, hop.dtype)

        self._record_onnx_stream_profile(profile, hop_len)
        self._onnx_stream_state = state
        return y_hop, frame_weights, state

    # REFACTOR: Share the online hop loop for PyTorch and ONNX stream steps.
    def _run_stream_loop(self, hop, state, step_fn):
        hop_len = int(self.stream_hop_samples)
        y_chunk = torch.as_tensor(hop, dtype=torch.float32, device=self._router_device()).reshape(-1)
        outputs = []
        weights = []
        offset = 0
        while offset < y_chunk.shape[-1]:
            release_len = min(hop_len, y_chunk.shape[-1] - offset)
            step_hop = y_chunk[offset:offset + release_len]
            if release_len < hop_len:
                step_hop = F.pad(step_hop, (0, hop_len - release_len))

            est_hop, frame_weights, state = step_fn(step_hop, state)
            outputs.append(est_hop.reshape(-1)[:release_len])
            if frame_weights is not None:
                weights.append(frame_weights)
            offset += release_len

        est_chunk = torch.cat(outputs, dim=-1) if outputs else y_chunk.new_zeros(0)
        mean_weights = torch.stack(weights, dim=0).mean(dim=0) if weights else None
        return est_chunk, mean_weights, len(outputs), state

    def _denoise_online_onnx_inference_branch(self, y_chunk):
        runtime = self._ensure_onnx_stream_runtime()
        if runtime is None:
            return self._denoise_online_inference_branch(y_chunk)
        if self._onnx_stream_state is None:
            self._reset_onnx_stream_state()
        est_chunk, mean_weights, infer_frames, state = self._run_stream_loop(
            y_chunk,
            self._onnx_stream_state,
            self._onnx_stream_step,
        )
        self._onnx_stream_state = state
        return est_chunk, mean_weights, infer_frames

    def _denoise_online_inference_branch(self, y_chunk):
        if self._use_onnx_stream_inference():
            return self._denoise_online_onnx_inference_branch(y_chunk)

        y_chunk = torch.as_tensor(y_chunk, dtype=torch.float32, device=self._router_device()).reshape(-1)
        state = self._ensure_moe_stream_state(device=y_chunk.device, dtype=torch.float32, batch_size=1)
        est_chunk, mean_weights, infer_frames, state = self._run_stream_loop(
            y_chunk,
            state,
            self._torch_stream_step,
        )
        self._moe_stream_state = state
        return est_chunk, mean_weights, infer_frames

    def start_stream_adaptation(self, stream_id="denoise_stream", reset=True):
        """Initialize persistent online-router adaptation for chunk APIs."""
        self._apply_device_map()
        self._ensure_stream_locks()
        if reset:
            self._shutdown_stream_executors()
        self.ref_model.to(self._router_device())
        if not self._policies_synced:
            self._sync_policies_from_model()
        self._router_test_optimizer = self._new_router_optimizer()
        self._stream_batch_idx = 0
        self._stream_id = str(stream_id)
        self._stream_was_training = self.model.training
        self._stream_train_error = None
        self._sync_stream_infer_model()
        if reset:
            self.window.clear()
            self.clean_window.clear()
            self._last_window_utt = None
            self._reset_stream_infer_state()
        self.model.eval()

    def stop_stream_adaptation(self):
        """Restore router training state after persistent chunk adaptation."""
        self._shutdown_stream_executors()
        was_training = getattr(self, "_stream_was_training", False)
        self.model.train(was_training)

    def denoise_stream_chunk(
        self,
        y_chunk,
        chunk=None,
        stream_id=None,
        adapt=True,
        return_info=False,
    ):
        """Enhance one streaming chunk and optionally update the router.

        Unlike `denoise`, this method does not clear the GRPO rolling window on
        every call. In streaming mode inference is released in STFT-sized hops,
        while router GRPO updates are queued on the training branch.
        """
        self._apply_device_map()
        y_chunk = torch.as_tensor(y_chunk, dtype=torch.float32, device=self._router_device())
        if y_chunk.ndim == 2 and y_chunk.shape[0] == 1:
            y_chunk = y_chunk[0]
        if y_chunk.ndim != 1:
            raise ValueError(
                f"FrozenExpertRouterGRPO.denoise_stream_chunk expects mono audio, got shape {tuple(y_chunk.shape)}"
            )
        if y_chunk.numel() == 0:
            empty = y_chunk.detach().clone()
            return (empty, {"updated": False, "weights": []}) if return_info else empty

        if not hasattr(self, "_router_test_optimizer") or self._router_test_optimizer is None:
            self.start_stream_adaptation(stream_id=stream_id or "denoise_stream", reset=False)

        stream_id = str(stream_id or getattr(self, "_stream_id", "denoise_stream"))
        if chunk is None or float(chunk) <= 0:
            chunk_samples = int(y_chunk.shape[-1])
        else:
            chunk_samples = int(round(self.sample_rate * float(chunk)))
        chunk_samples = max(chunk_samples, int(y_chunk.shape[-1]))

        total_start = self._latency_stamp()
        updated = False
        train_status = "disabled"
        adapt_ms = 0.0
        if adapt and self.stream_adapt_in_denoise:
            adapt_start = self._latency_stamp()
            updated, train_status = self._schedule_stream_train(y_chunk, chunk_samples, stream_id)
            adapt_ms = self._latency_elapsed_ms(adapt_start)

        infer_start = self._latency_stamp()
        with torch.no_grad():
            if self.stream_online_infer:
                self._ensure_stream_locks()
                with self._stream_infer_lock:
                    est_chunk, weights, infer_frames = self._denoise_online_inference_branch(y_chunk)
            else:
                self.model.eval()
                chunk_in = self.padding(y_chunk.unsqueeze(0))
                est_chunk = self.forward(chunk_in, train=False, ret_weights=True)
                if isinstance(est_chunk, tuple):
                    est_chunk, weights = est_chunk
                else:
                    weights = None
                est_chunk = est_chunk.squeeze(0)[:y_chunk.shape[-1]]
                infer_frames = 1
        infer_ms = self._latency_elapsed_ms(infer_start)

        self._prune_stream_train_futures()
        total_ms = self._latency_elapsed_ms(total_start)
        self._record_latency("stream_chunk_infer", infer_ms, log_metrics=False, batch_size=1)
        self._record_latency("stream_chunk_adapt", adapt_ms, log_metrics=False, batch_size=1)
        self._record_latency("stream_chunk_total", total_ms, log_metrics=False, batch_size=1)
        if not return_info:
            return est_chunk

        weight_list = []
        if weights is not None:
            weight_list = weights.detach().cpu().reshape(-1).tolist()
        return est_chunk, {
            "updated": updated,
            "train_status": train_status,
            "train_queue": len(self._stream_train_futures),
            "train_error": str(self._stream_train_error) if self._stream_train_error is not None else None,
            "async_train": self.stream_async_train,
            "inference_runtime": self.stream_inference_runtime,
            "onnx_manifest": str(self._stream_onnx_manifest_path()) if self._use_onnx_stream_inference() else None,
            "device_map": self._device_map_status(),
            "parallel_experts": self.stream_parallel_experts,
            "parallel_expert_workers": self.stream_parallel_expert_workers,
            "stateful_experts": self.stream_stateful_experts,
            "expert_state_kinds": self._stream_expert_state_kinds(),
            "expert_stream_modes": self._stream_expert_adapter_modes(),
            "cache_hit": self._stream_cache_hit(),
            "fallback_expert_count": self._stream_fallback_expert_count(),
            "moe_stream_steps": (
                int(self._onnx_stream_state.num_steps)
                if self._use_onnx_stream_inference() and self._onnx_stream_state is not None
                else int(self._moe_stream_state.num_steps) if self._moe_stream_state is not None else 0
            ),
            "inference_frames": infer_frames,
            "inference_frame_ms": self.stream_frame_ms if self.stream_online_infer else None,
            "inference_hop_ms": self.stream_hop_ms if self.stream_online_infer else None,
            "weights": weight_list,
            "batch_idx": int(getattr(self, "_stream_batch_idx", 0)),
            "window": len(getattr(self, "window", [])),
            "latency_ms": {
                "stream_chunk_infer": round(infer_ms, 3),
                "stream_chunk_adapt": round(adapt_ms, 3),
                "stream_chunk_total": round(total_ms, 3),
                "infer_forward_total": round(self._last_latency_ms.get("infer_forward_total", 0.0), 3),
                "infer_experts": round(self._last_latency_ms.get("infer_experts", 0.0), 3),
                "grpo_update_total": round(self._last_latency_ms.get("grpo_update_total", 0.0), 3),
                "grpo_sample_reward": round(self._last_latency_ms.get("grpo_sample_reward", 0.0), 3),
                "stream/input_cache_ms": round(self._last_latency_ms.get("stream/input_cache", 0.0), 3),
                "stream/expert_stream_step_ms": round(
                    self._last_latency_ms.get("stream/expert_stream_step", 0.0), 3
                ),
                "stream/router_ms": round(self._last_latency_ms.get("stream/router", 0.0), 3),
                "stream/fusion_ms": round(self._last_latency_ms.get("stream/fusion", 0.0), 3),
                "stream/total_step_ms": round(self._last_latency_ms.get("stream/total_step", 0.0), 3),
                "stream/frame_rtf": round(self._last_latency_ms.get("stream/frame_rtf", 0.0), 4),
                "stream/cache_hit": round(self._last_latency_ms.get("stream/cache_hit", 0.0), 3),
                "stream/fallback_expert_count": round(
                    self._last_latency_ms.get("stream/fallback_expert_count", 0.0), 3
                ),
            },
        }

    def denoise(self, y, chunk=-1):
        if chunk <= 0 or not self.stream_adapt_in_denoise:
            return BaseSE.denoise(self, y, chunk)

        self._apply_device_map()
        y = torch.as_tensor(y, dtype=torch.float32, device=self._router_device())
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
        if y.ndim != 1:
            raise ValueError(f"FrozenExpertRouterGRPO.denoise expects mono audio, got shape {tuple(y.shape)}")

        chunk_samples = int(round(self.sample_rate * float(chunk)))
        if chunk_samples <= 0:
            raise ValueError(f"Invalid denoise.chunk: {chunk}")

        est_wav = torch.zeros_like(y)
        n_chunk = int(math.ceil(y.shape[-1] / chunk_samples))
        stream_id = "denoise_stream"
        self.start_stream_adaptation(stream_id=stream_id, reset=True)
        try:
            for idx in range(n_chunk):
                start = idx * chunk_samples
                end = min((idx + 1) * chunk_samples, y.shape[-1])
                y_chunk = y[start:end]
                if y_chunk.numel() == 0:
                    continue

                est_chunk = self.denoise_stream_chunk(
                    y_chunk,
                    chunk=chunk,
                    stream_id=stream_id,
                    adapt=True,
                    return_info=False,
                )
                if est_chunk.shape[-1] < y_chunk.shape[-1]:
                    est_chunk = F.pad(est_chunk, (0, y_chunk.shape[-1] - est_chunk.shape[-1]))
                est_wav[start:end] = est_chunk[:y_chunk.shape[-1]]
        finally:
            self.stop_stream_adaptation()
        return est_wav

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._apply_device_map(force=True)
        self.window.clear()
        self.clean_window.clear()
        self._last_window_utt = None
        self._router_test_optimizer = None
        self.ref_model.to(self._router_device())
        if not self._policies_synced:
            self._sync_policies_from_model()
            logger.info("[FrozenExpertRouterGRPO] Synced ref router at test start")
        if self.adapt_in_test:
            logger.info("[FrozenExpertRouterGRPO] test-time GRPO updates are enabled for router only.")

    def _adapt_router_on_test_batch(self, batch, batch_idx):
        if not self.adapt_in_test:
            return None
        self._apply_device_map()
        opt = self._get_test_optimizer()
        self.model.train()
        loss = self._grpo_update_step(batch, batch_idx, opt, update_reference=True, log_metrics=True)
        self.model.eval()
        return loss

    def test_step(self, batch, batch_idx):
        if self.adapt_in_test and self.test_update_before_eval:
            with torch.enable_grad():
                self._adapt_router_on_test_batch(batch, batch_idx)
            return BaseSE.test_step(self, batch, batch_idx)

        loss = BaseSE.test_step(self, batch, batch_idx)
        if self.adapt_in_test:
            with torch.enable_grad():
                self._adapt_router_on_test_batch(batch, batch_idx)
        return loss

    def _log_metrics(self, samples, adv_stats, info, total, rewards, advantages, loss, grad_norm):
        super()._log_metrics(samples, adv_stats, info, total, rewards, advantages, loss, grad_norm)
        window_rewards = samples.get("window_rewards")
        if window_rewards is not None:
            self.log("stream/window_reward_mean", window_rewards.mean(),
                     sync_dist=True, batch_size=window_rewards.shape[0])
            self.log("stream/window_reward_best", window_rewards.max(),
                     sync_dist=True, batch_size=window_rewards.shape[0])
            self.log("stream/window_chunks",
                     torch.as_tensor(float(samples.get("window_chunks", 0)), device=self._router_device()),
                     sync_dist=True)
            self.log("stream/window_seconds",
                     torch.as_tensor(
                         float(samples.get("window_len_samples", 0)) / max(float(self.sample_rate), 1.0),
                         device=self._router_device()),
                     sync_dist=True)
        weights = samples.get("router_weights")
        if weights is None:
            return
        entropy = -(weights.clamp_min(EPS) * weights.clamp_min(EPS).log()).sum(dim=-1)
        self.log("router/entropy", entropy.mean(), sync_dist=True, batch_size=weights.shape[0])
        action_weights = samples.get("action_router_weights")
        if action_weights is not None:
            action_entropy = -(
                action_weights.clamp_min(EPS) * action_weights.clamp_min(EPS).log()
            ).sum(dim=-1)
            self.log("router/action_entropy", action_entropy.mean(),
                     sync_dist=True, batch_size=action_weights.shape[0])
        moe_noise = samples.get("moe_param_noise")
        if moe_noise is not None:
            self.log("router/moe_param_noise_rms", moe_noise.pow(2).mean().sqrt(),
                     sync_dist=True, batch_size=moe_noise.shape[0])
        for idx, expert in enumerate(self.experts):
            safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in expert.name)
            self.log(f"router/weight_{idx}_{safe_name}", weights[:, idx].mean(),
                     sync_dist=True, batch_size=weights.shape[0])
            if action_weights is not None:
                self.log(f"router/action_weight_{idx}_{safe_name}", action_weights[:, idx].mean(),
                         sync_dist=True, batch_size=action_weights.shape[0])


class FrozenExpertOracleBaseline(FrozenExpertRouterGRPO):
    """Oracle baseline over frozen experts.

    Each expert enhances every utterance. For each utterance and metric, the
    reported baseline score is the best score among experts.
    """

    REF_METRIC_KEYS = ("PESQ", "STOI", "eSTOI", "SI_SNR")
    MOS_METRIC_KEYS = ("SIG", "BAK", "OVL")
    METRIC_KEYS = REF_METRIC_KEYS + MOS_METRIC_KEYS
    METRIC_ALIASES = {
        "PESQ": "PESQ",
        "STOI": "STOI",
        "ESTOI": "eSTOI",
        "E_STOI": "eSTOI",
        "SI_SNR": "SI_SNR",
        "SISNR": "SI_SNR",
        "SI-SNR": "SI_SNR",
        "SIG": "SIG",
        "MOS_SIG": "SIG",
        "BAK": "BAK",
        "MOS_BAK": "BAK",
        "OVL": "OVL",
        "OVR": "OVL",
        "OVRL": "OVL",
        "MOS_OVL": "OVL",
        "MOS_OVR": "OVL",
        "MOS_OVRL": "OVL",
    }

    def __init__(self, conf):
        super().__init__(conf)
        oracle_conf = _to_plain_dict(
            conf.get("oracle_baseline", self.router_grpo_conf.get("oracle_baseline", {}))
        )
        self.oracle_select_metric = self._canonical_metric_key(
            oracle_conf.get("select_metric", oracle_conf.get("selection_metric", "OVL"))
        )
        if self.oracle_select_metric is None:
            self.oracle_select_metric = "OVL"
        self.oracle_save_per_utt = bool(oracle_conf.get("save_per_utt", True))
        self.oracle_per_utt_file = str(oracle_conf.get("per_utt_file", "oracle_baseline_per_utt.csv"))
        self.oracle_json_file = str(oracle_conf.get("json_file", "oracle_baseline_results.json"))
        self.oracle_include_selected = bool(oracle_conf.get("include_selected_summary", True))
        self.oracle_mos_worker = getattr(self, "MOS_worker", None)
        self.oracle_mos_worker_name = "ComputeMOS"
        if self.oracle_mos_worker is None:
            self.oracle_mos_worker = self.mos_worker
            self.oracle_mos_worker_name = type(self.mos_worker).__name__
        self._oracle_missing_select_warned = False
        self._oracle_save_warned = False

        self.adapt_in_test = False
        self.test_update_before_eval = False
        self.stream_adapt_in_denoise = False
        logger.info(
            "[FrozenExpertOracleBaseline] experts={}, select_metric={}, mos_worker={}, "
            "per_metric_oracle=True".format(
                [expert.name for expert in self.experts],
                self.oracle_select_metric,
                self.oracle_mos_worker_name,
            )
        )

    def configure_optimizers(self):
        return None

    @classmethod
    def _canonical_metric_key(cls, key):
        if key is None:
            return None
        text = str(key).strip()
        if not text:
            return None
        norm = text.upper().replace("-", "_")
        return cls.METRIC_ALIASES.get(norm)

    @staticmethod
    def _as_float(value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.item())
        return float(value)

    @staticmethod
    def _normalize_wav_ids(wav_ids):
        if isinstance(wav_ids, str):
            return [wav_ids]
        if torch.is_tensor(wav_ids):
            wav_ids = wav_ids.detach().cpu().tolist()
        return [str(item) for item in wav_ids]

    def _expert_wavs_for_noisy_batch(self, noisy_wav):
        padded_noisy = self.padding(noisy_wav)
        noisy_spec = self.stft.apply_stft(padded_noisy)
        expert_wavs = self._stack_expert_wavs(
            padded_noisy,
            noisy_spec,
            target_len=padded_noisy.shape[-1],
        )
        return expert_wavs[..., :noisy_wav.shape[-1]]

    def _score_expert_wavs_mos(self, expert_wavs):
        batch, n_experts, n_samples = expert_wavs.shape
        flat_wavs = expert_wavs.reshape(batch * n_experts, n_samples)
        with torch.no_grad():
            scores = self.oracle_mos_worker.batch_scores(flat_wavs)
        if not torch.is_tensor(scores):
            scores = torch.as_tensor(scores, dtype=expert_wavs.dtype, device=expert_wavs.device)
        else:
            scores = scores.to(device=expert_wavs.device, dtype=expert_wavs.dtype)
        return scores.reshape(batch, n_experts, len(self.MOS_METRIC_KEYS))

    @staticmethod
    def _gather_expert_wavs(expert_wavs, expert_indices):
        batch_indices = torch.arange(expert_wavs.shape[0], device=expert_wavs.device)
        return expert_wavs[batch_indices, expert_indices]

    def forward(self, noisy_wav, clean_wav=None, train=True, ret_weights=False):
        expert_wavs = self._expert_wavs_for_noisy_batch(noisy_wav)
        if self.oracle_select_metric in self.MOS_METRIC_KEYS:
            mos_scores = self._score_expert_wavs_mos(expert_wavs)
            score_idx = self.MOS_METRIC_KEYS.index(self.oracle_select_metric)
        else:
            mos_scores = self._score_expert_wavs_mos(expert_wavs)
            score_idx = self.MOS_METRIC_KEYS.index("OVL")
        expert_indices = mos_scores[..., score_idx].argmax(dim=1)
        est_wav = self._gather_expert_wavs(expert_wavs, expert_indices)
        if ret_weights:
            weights = F.one_hot(expert_indices, num_classes=len(self.experts)).type_as(est_wav)
            return est_wav, weights
        if (not train) or clean_wav is None:
            return est_wav

        min_len = min(clean_wav.shape[-1], est_wav.shape[-1], noisy_wav.shape[-1])
        clean = clean_wav[..., :min_len]
        est = est_wav[..., :min_len]
        noisy = noisy_wav[..., :min_len]
        if self.loss is None:
            loss = torch.zeros((), device=est.device, dtype=est.dtype)
        elif self.loss_name == "wSDRLoss":
            loss = self.loss(clean, est, noisy)
        elif self.loss_name in ["CMSELoss"]:
            loss = self.loss(self.stft.apply_stft(clean), self.stft.apply_stft(est))
        else:
            loss = self.loss(clean, est)
        return loss, est_wav

    def prepare_valid_test(self):
        super().prepare_valid_test()
        self.oracle_records = []

    def on_test_epoch_start(self):
        BaseSE.on_test_epoch_start(self)

    def _need_mos_scores(self, MOS):
        return bool(MOS) or self.oracle_select_metric in self.MOS_METRIC_KEYS

    def _need_ref_scores(self, metric, clean_wav):
        return clean_wav is not None and (
            bool(metric) or self.oracle_select_metric in self.REF_METRIC_KEYS
        )

    def _batch_selected_indices_from_records(self, records, batch_size, n_experts, metric_key):
        if metric_key is None:
            return None
        values = torch.full((batch_size, n_experts), -float("inf"))
        for record in records:
            value = record.get(metric_key)
            if value is None:
                continue
            values[int(record["batch_index"]), int(record["expert_idx"])] = float(value)
        if not torch.isfinite(values).any(dim=1).all():
            return None
        return values.argmax(dim=1).to(dtype=torch.long)

    def process_valid_test(self, batch, metric=False, MOS=False):
        clean_wav, noisy_wav, wav_id_list = self.unpack_wav_batch(batch)
        wav_id_list = self._normalize_wav_ids(wav_id_list)
        expert_wavs = self._expert_wavs_for_noisy_batch(noisy_wav)
        batch_size, n_experts, n_samples = expert_wavs.shape
        expert_names = [str(getattr(expert, "name", f"expert{idx}")) for idx, expert in enumerate(self.experts)]

        records = []
        for batch_idx in range(batch_size):
            wav_id = wav_id_list[batch_idx] if batch_idx < len(wav_id_list) else str(batch_idx)
            for expert_idx, expert_name in enumerate(expert_names):
                records.append({
                    "wav_id": wav_id,
                    "batch_index": batch_idx,
                    "expert_idx": expert_idx,
                    "expert": expert_name,
                })

        if self._need_mos_scores(MOS):
            mos_scores = self._score_expert_wavs_mos(expert_wavs).detach().cpu()
            for record in records:
                scores = mos_scores[int(record["batch_index"]), int(record["expert_idx"])]
                for key_idx, key in enumerate(self.MOS_METRIC_KEYS):
                    record[key] = self._as_float(scores[key_idx])

        if self._need_ref_scores(metric, clean_wav):
            clean_trim = clean_wav[..., :n_samples]
            for record in records:
                b_idx = int(record["batch_index"])
                e_idx = int(record["expert_idx"])
                record["_metric_future"] = self.pool.submit(
                    metrics.eval,
                    clean_trim[b_idx, :].detach().cpu(),
                    expert_wavs[b_idx, e_idx, :].detach().cpu(),
                )

        loss = torch.zeros((), device=noisy_wav.device)
        selected_indices = self._batch_selected_indices_from_records(
            records,
            batch_size,
            n_experts,
            self.oracle_select_metric,
        )
        if selected_indices is not None:
            selected_wavs = self._gather_expert_wavs(expert_wavs, selected_indices.to(expert_wavs.device))
            if self.save_wav_conf:
                self.save_wav_file(selected_wavs, clean_wav, noisy_wav, wav_id_list)
            if clean_wav is not None and self.loss is not None:
                min_len = min(clean_wav.shape[-1], selected_wavs.shape[-1], noisy_wav.shape[-1])
                if self.loss_name == "wSDRLoss":
                    loss = self.loss(
                        clean_wav[..., :min_len],
                        selected_wavs[..., :min_len],
                        noisy_wav[..., :min_len],
                    )
                elif self.loss_name in ["CMSELoss"]:
                    loss = self.loss(
                        self.stft.apply_stft(clean_wav[..., :min_len]),
                        self.stft.apply_stft(selected_wavs[..., :min_len]),
                    )
                else:
                    loss = self.loss(clean_wav[..., :min_len], selected_wavs[..., :min_len])
        elif self.save_wav_conf and not self._oracle_save_warned:
            logger.warning(
                "[FrozenExpertOracleBaseline] save_wav is skipped because select_metric "
                f"{self.oracle_select_metric} is not available before async metric collection."
            )
            self._oracle_save_warned = True

        self.oracle_records.extend(records)
        return loss

    def _resolve_oracle_records(self):
        for record in self.oracle_records:
            future = record.pop("_metric_future", None)
            if future is None:
                continue
            scores = future.result()
            for key, value in scores.items():
                metric_key = self._canonical_metric_key(key)
                if metric_key is not None:
                    record[metric_key] = self._as_float(value)
        return self.oracle_records

    @staticmethod
    def _finite_metric_values(records, metric_key):
        values = []
        for record in records:
            if metric_key not in record:
                continue
            value = float(record[metric_key])
            if math.isfinite(value):
                values.append((value, record))
        return values

    def _select_metric_for_group(self, records):
        if self._finite_metric_values(records, self.oracle_select_metric):
            return self.oracle_select_metric
        for metric_key in self.METRIC_KEYS:
            if self._finite_metric_values(records, metric_key):
                if not self._oracle_missing_select_warned:
                    logger.warning(
                        "[FrozenExpertOracleBaseline] select_metric "
                        f"{self.oracle_select_metric} is unavailable; using {metric_key} for selected-expert stats."
                    )
                    self._oracle_missing_select_warned = True
                return metric_key
        return None

    def _build_local_oracle_summary(self, records):
        groups = {}
        for record in records:
            groups.setdefault(record["wav_id"], []).append(record)

        metric_keys = [
            metric_key for metric_key in self.METRIC_KEYS
            if any(metric_key in record for record in records)
        ]
        best_sums = {metric_key: 0.0 for metric_key in metric_keys}
        best_counts = {metric_key: 0.0 for metric_key in metric_keys}
        selected_sums = {metric_key: 0.0 for metric_key in metric_keys}
        selected_counts = {metric_key: 0.0 for metric_key in metric_keys}
        expert_counts = [0.0 for _ in self.experts]
        expert_metric_sums = {
            str(getattr(expert, "name", f"expert{idx}")): {metric_key: 0.0 for metric_key in metric_keys}
            for idx, expert in enumerate(self.experts)
        }
        expert_metric_counts = {
            str(getattr(expert, "name", f"expert{idx}")): {metric_key: 0.0 for metric_key in metric_keys}
            for idx, expert in enumerate(self.experts)
        }
        per_utt_rows = []

        for wav_id, group in groups.items():
            select_metric = self._select_metric_for_group(group)
            selected_record = group[0]
            selected_score = None
            if select_metric is not None:
                selected_score, selected_record = max(
                    self._finite_metric_values(group, select_metric),
                    key=lambda item: item[0],
                )
            expert_idx = int(selected_record.get("expert_idx", 0))
            if 0 <= expert_idx < len(expert_counts):
                expert_counts[expert_idx] += 1.0

            row = {
                "id": wav_id,
                "selected_metric": select_metric or "",
                "selected_expert": selected_record.get("expert", ""),
                "selected_score": selected_score,
            }
            for metric_key in metric_keys:
                values = self._finite_metric_values(group, metric_key)
                if values:
                    best_value, best_record = max(values, key=lambda item: item[0])
                    best_sums[metric_key] += best_value
                    best_counts[metric_key] += 1.0
                    row[f"best_{metric_key}"] = best_value
                    row[f"best_{metric_key}_expert"] = best_record.get("expert", "")
                if metric_key in selected_record:
                    selected_value = float(selected_record[metric_key])
                    if math.isfinite(selected_value):
                        selected_sums[metric_key] += selected_value
                        selected_counts[metric_key] += 1.0
                        row[f"selected_{metric_key}"] = selected_value
                for record in group:
                    expert_name = str(record.get("expert", ""))
                    if expert_name not in expert_metric_sums or metric_key not in record:
                        continue
                    expert_value = float(record[metric_key])
                    if math.isfinite(expert_value):
                        expert_metric_sums[expert_name][metric_key] += expert_value
                        expert_metric_counts[expert_name][metric_key] += 1.0
            per_utt_rows.append(row)

        return {
            "metric_keys": metric_keys,
            "best_sums": best_sums,
            "best_counts": best_counts,
            "selected_sums": selected_sums,
            "selected_counts": selected_counts,
            "expert_counts": expert_counts,
            "expert_metric_sums": expert_metric_sums,
            "expert_metric_counts": expert_metric_counts,
            "num_utts": len(groups),
            "per_utt_rows": per_utt_rows,
        }

    def _distributed_metric_average(self, sums, counts, metric_keys):
        if not metric_keys:
            return {}
        data = torch.tensor(
            [[float(sums.get(key, 0.0)), float(counts.get(key, 0.0))] for key in metric_keys],
            dtype=torch.float64,
            device=self.device,
        )
        if getattr(getattr(self, "trainer", None), "world_size", 1) > 1:
            data = self.all_gather(data).sum(dim=0)
        result = {}
        for idx, metric_key in enumerate(metric_keys):
            denom = float(data[idx, 1].item())
            if denom > 0:
                result[metric_key] = float(data[idx, 0].item() / denom)
        return result

    def _distributed_expert_counts(self, expert_counts):
        counts = torch.tensor(expert_counts, dtype=torch.float64, device=self.device)
        if getattr(getattr(self, "trainer", None), "world_size", 1) > 1:
            counts = self.all_gather(counts).sum(dim=0)
        return {
            str(getattr(expert, "name", f"expert{idx}")): int(counts[idx].item())
            for idx, expert in enumerate(self.experts)
        }

    def _distributed_num_utts(self, num_utts):
        count = torch.tensor(float(num_utts), dtype=torch.float64, device=self.device)
        if getattr(getattr(self, "trainer", None), "world_size", 1) > 1:
            count = self.all_gather(count).sum()
        return int(count.item())

    def _distributed_expert_metric_averages(self, sums, counts, metric_keys):
        expert_names = [str(getattr(expert, "name", f"expert{idx}")) for idx, expert in enumerate(self.experts)]
        if not expert_names or not metric_keys:
            return {}
        data = torch.tensor(
            [
                [
                    [
                        float(sums.get(expert_name, {}).get(metric_key, 0.0)),
                        float(counts.get(expert_name, {}).get(metric_key, 0.0)),
                    ]
                    for metric_key in metric_keys
                ]
                for expert_name in expert_names
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if getattr(getattr(self, "trainer", None), "world_size", 1) > 1:
            data = self.all_gather(data).sum(dim=0)
        result = {}
        for expert_idx, expert_name in enumerate(expert_names):
            expert_result = {}
            for metric_idx, metric_key in enumerate(metric_keys):
                denom = float(data[expert_idx, metric_idx, 1].item())
                if denom > 0:
                    expert_result[metric_key] = float(data[expert_idx, metric_idx, 0].item() / denom)
            result[expert_name] = expert_result
        return result

    def _oracle_epoch_summary(self):
        records = self._resolve_oracle_records()
        local = self._build_local_oracle_summary(records)
        metric_keys = local["metric_keys"]
        best = self._distributed_metric_average(local["best_sums"], local["best_counts"], metric_keys)
        selected = self._distributed_metric_average(
            local["selected_sums"],
            local["selected_counts"],
            metric_keys,
        )
        return {
            "select_metric": self.oracle_select_metric,
            "num_utts": self._distributed_num_utts(local["num_utts"]),
            "best": best,
            "selected": selected,
            "expert_counts": self._distributed_expert_counts(local["expert_counts"]),
            "expert_means": self._distributed_expert_metric_averages(
                local["expert_metric_sums"],
                local["expert_metric_counts"],
                metric_keys,
            ),
            "per_utt_rows": local["per_utt_rows"],
        }

    def on_validation_epoch_end(self):
        summary = self._oracle_epoch_summary()
        for metric_key, value in summary["best"].items():
            self.log(
                f"valid/oracle_best_{metric_key}",
                torch.tensor(value, dtype=torch.float32, device=self.device),
                sync_dist=False,
            )
        if self.oracle_include_selected:
            for metric_key, value in summary["selected"].items():
                self.log(
                    f"valid/oracle_selected_{metric_key}",
                    torch.tensor(value, dtype=torch.float32, device=self.device),
                    sync_dist=False,
                )
        if getattr(self.trainer, "is_global_zero", True):
            logger.info(f"[FrozenExpertOracleBaseline valid best] {compact_dict(summary['best'])}")

    @staticmethod
    def _write_rows_csv(path, rows):
        if not rows:
            return
        fieldnames = []
        preferred = ["id", "selected_metric", "selected_expert", "selected_score"]
        for key in preferred:
            if any(key in row for row in rows):
                fieldnames.append(key)
        extra_keys = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
        fieldnames.extend(extra_keys)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def on_test_epoch_end(self):
        summary = self._oracle_epoch_summary()
        if not getattr(self.trainer, "is_global_zero", True):
            return

        root_path = Path(self.conf.get("root_dir", self.trainer.default_root_dir))
        root_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[FrozenExpertOracleBaseline test best] {compact_dict(summary['best'])}")
        logger.info(f"[FrozenExpertOracleBaseline expert counts] {summary['expert_counts']}")

        serializable_summary = dict(summary)
        serializable_summary.pop("per_utt_rows", None)
        with open(root_path.joinpath(self.oracle_json_file), "w") as f:
            json.dump(serializable_summary, f, indent=2, cls=NumpyEncoder)

        ref_results = {
            key: summary["best"][key]
            for key in self.REF_METRIC_KEYS
            if key in summary["best"]
        }
        if ref_results:
            with open(root_path.joinpath("test_results.json"), "w") as f:
                json.dump(ref_results, f, cls=NumpyEncoder)

        mos_results = {
            key: summary["best"][key]
            for key in self.MOS_METRIC_KEYS
            if key in summary["best"]
        }
        if mos_results:
            with open(root_path.joinpath("DNSMOS_results.json"), "w") as f:
                json.dump(mos_results, f, cls=NumpyEncoder)

        if self.oracle_save_per_utt:
            if getattr(getattr(self, "trainer", None), "world_size", 1) > 1:
                logger.warning(
                    "[FrozenExpertOracleBaseline] per-utterance CSV is written only for rank 0 in distributed test."
                )
            self._write_rows_csv(root_path.joinpath(self.oracle_per_utt_file), summary["per_utt_rows"])
