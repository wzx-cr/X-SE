"""Export MoE expert stream-step ONNX models and a unified manifest.

This exporter is intentionally expert-only. The GRPO router stays in PyTorch by
default. Every exported expert consumes one STFT frame:

  spec_in: [B, F, T=1, 2]
  cache_in_*

and returns:

  spec_out: [B, F, T=1, 2]
  cache_out_*
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from modules.utils.common import NumpyEncoder, resolve_path  # noqa: E402
from alpha.enh.system.grpo import (  # noqa: E402
    LiSenStreamingExpertAdapter,
)
from tools.export_grpo_onnx import (  # noqa: E402
    _load_router_checkpoint,
    _prepare_export_module,
    _resolve_required_conf,
    _simplify_onnx,
)


EPS = 1.0e-8


@dataclass
class ExportItem:
    name: str
    source: str
    wrapper: nn.Module
    inputs: tuple[torch.Tensor, ...]
    input_names: list[str]
    output_names: list[str]
    cache_shapes: list[list[int]]
    checkpoint: str | None
    wrapper_name: str
    notes: str
    decompose_gru: bool = True
    gru_export: str = "decomposed"


def _plain(conf):
    if conf is None:
        return {}
    return OmegaConf.to_container(conf, resolve=True) if OmegaConf.is_config(conf) else dict(conf)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    try:
        return Path(resolve_path(str(path)))
    except Exception:
        return (Path.cwd() / path).resolve()


def _cuda_index(device_spec: Any) -> int:
    text = str(device_spec).strip().lower()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda"):
        return int(text.split(":", 1)[1]) if ":" in text else 0
    return 0


def _expert_checkpoint(expert) -> str | None:
    conf = getattr(expert, "conf", None)
    conf = _plain(conf) if conf is not None else {}
    model_conf = _plain(conf.get("model", {}))
    init = model_conf.get("init")
    return str(init) if init is not None else None


def _complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
        ],
        dim=-1,
    )


def _compress_spec(spec: torch.Tensor, compression: float, eps: float = 1.0e-5) -> torch.Tensor:
    mag = torch.sqrt(spec[..., 0:1] * spec[..., 0:1] + spec[..., 1:2] * spec[..., 1:2] + eps)
    return spec * mag.pow(float(compression) - 1.0)


def _uncompress_spec(spec: torch.Tensor, compression: float, eps: float = 1.0e-5) -> torch.Tensor:
    mag = torch.sqrt(spec[..., 0:1] * spec[..., 0:1] + spec[..., 1:2] * spec[..., 1:2] + eps)
    return spec * mag.pow(1.0 / max(float(compression), EPS) - 1.0)


def _wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def diff_onnx_safe(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim = x.dim() + dim
    length = int(x.shape[dim])
    left = x.narrow(dim, 1, length - 1)
    right = x.narrow(dim, 0, length - 1)
    return left - right


def diff_onnx_safe_zero_prepend(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim = x.dim() + dim
    zeros_shape = list(x.shape)
    zeros_shape[dim] = 1
    prepended = torch.cat([x.new_zeros(tuple(zeros_shape)), x], dim=dim)
    return diff_onnx_safe(prepended, dim=dim)


class FastEnhancerSpecONNXWrapper(nn.Module):
    """FastEnhancer spec2spec wrapper following scripts/export_onnx_spec.py."""

    def __init__(self, core):
        super().__init__()
        self.core = core
        if not callable(getattr(core, "model_forward", None)) or not callable(getattr(core, "initialize_cache", None)):
            raise RuntimeError(
                "FastEnhancer-S current core is non-streaming Model.forward(x); "
                "need to use official FastEnhancer ONNXModel/spec wrapper."
            )
        self.input_compression = float(
            getattr(core, "input_compression", getattr(getattr(core, "stft", None), "compression", 0.3))
        )
        self.eps = float(getattr(getattr(core, "stft", None), "eps", 1.0e-5))

    def forward(self, spec_in, *cache_in):
        # Official FastEnhancer spec export consumes [B, F+1, T, 2], drops the
        # Nyquist bin for the model, then pads it back after reconstruction.
        spec_noisy = spec_in[:, :-1, :, :]
        spec_noisy = _compress_spec(spec_noisy, self.input_compression, eps=self.eps)
        mask, cache_out = self.core.model_forward(spec_noisy, *cache_in)
        spec_hat = _complex_mul(spec_noisy, mask)
        spec_hat = _uncompress_spec(spec_hat, self.input_compression, eps=self.eps)
        spec_hat = F.pad(spec_hat, (0, 0, 0, 0, 0, 1))
        return (spec_hat, *cache_out)


class ULUNASSpecONNXWrapper(nn.Module):
    """Official-style UL-UNAS stream wrapper for spec-frame ONNX export.

    This mirrors Xiaobin-Rong/ul-unas `ulunas_onnx/stream/ulunas_stream.py`:
    input is one real/imag STFT frame [B,F,1,2], state is packed as
    conv_cache/tfa_cache/inter_cache, and output is one enhanced STFT frame.
    """

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

    def __init__(self, core):
        super().__init__()
        if core is None:
            raise RuntimeError("UL-UNAS streaming core is missing; cannot export spec-frame ONNX.")
        for name in ("erb", "encoder", "dpgrnn", "decoder"):
            if not hasattr(core, name):
                raise RuntimeError(f"UL-UNAS core is missing `{name}`; cannot export official stream wrapper.")
        self.erb = core.erb
        self.encoder = core.encoder
        self.dpgrnn = core.dpgrnn
        self.decoder = core.decoder
        self.n_fft = int(getattr(core, "n_fft", 512))
        self.hop_len = int(getattr(core, "hop_len", getattr(core, "hop_length", 256)))

    @classmethod
    def init_caches(cls, batch_size: int = 1, device=None, dtype=torch.float32):
        conv_size = sum(math.prod(shape) for shape in cls.CONV_CACHE_SHAPES)
        tfa_size = sum(cls.TFA_CACHE_HIDDEN)
        inter_size = sum(math.prod(shape) for shape in cls.INTER_CACHE_SHAPES)
        return (
            torch.zeros(batch_size, conv_size, device=device, dtype=dtype),
            torch.zeros(batch_size, tfa_size, device=device, dtype=dtype),
            torch.zeros(batch_size, inter_size, device=device, dtype=dtype),
        )

    @classmethod
    def _unpack_conv_cache(cls, conv_cache):
        bsz = conv_cache.shape[0]
        caches = []
        offset = 0
        for shape in cls.CONV_CACHE_SHAPES:
            n = math.prod(shape)
            caches.append(conv_cache[:, offset:offset + n].view(bsz, *shape))
            offset += n
        return caches

    @staticmethod
    def _pack_conv_cache(caches):
        return torch.cat([cache.reshape(cache.shape[0], -1) for cache in caches], dim=1)

    @classmethod
    def _unpack_tfa_cache(cls, tfa_cache):
        bsz = tfa_cache.shape[0]
        caches = []
        offset = 0
        for hidden in cls.TFA_CACHE_HIDDEN:
            caches.append(tfa_cache[:, offset:offset + hidden].view(1, bsz, hidden))
            offset += hidden
        return caches

    @staticmethod
    def _pack_tfa_cache(caches):
        return torch.cat([cache.reshape(cache.shape[1], -1) for cache in caches], dim=1)

    @classmethod
    def _unpack_inter_cache(cls, inter_cache):
        bsz = inter_cache.shape[0]
        caches = []
        offset = 0
        for shape in cls.INTER_CACHE_SHAPES:
            n = math.prod(shape)
            caches.append(inter_cache[:, offset:offset + n].view(bsz, *shape))
            offset += n
        return caches

    @staticmethod
    def _pack_inter_cache(caches):
        return torch.cat([cache.reshape(cache.shape[0], -1) for cache in caches], dim=1)

    @staticmethod
    def _stream_temporal_conv(x, conv, cache):
        inp = torch.cat([cache, x], dim=2)
        if isinstance(conv, nn.ConvTranspose2d):
            kt = conv.kernel_size[0]
            inp_padded = F.pad(inp, (0, 0, kt - 1, 0))
            y = conv(inp_padded)[:, :, -1:, :]
        else:
            y = conv(inp)
        new_cache = inp[:, :, 1:, :]
        return y, new_cache

    @staticmethod
    def _stream_fa(fa, x):
        bsz, _channels, time_steps, freq = x.shape
        x = torch.mean(x.pow(2), dim=1)
        pad_len = int(getattr(fa, "pad_len", 0))
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        group = int(fa.r)
        groups = int(fa.H)
        padded_freq = int(fa.F_pad)
        x = x.view(bsz, time_steps, groups, group)
        x = x.reshape(bsz * time_steps, groups, group)

        num_directions = 2 if bool(fa.gru.bidirectional) else 1
        h0_batch = int(x.shape[0])
        h0 = x.new_zeros(int(fa.gru.num_layers) * num_directions, h0_batch, int(fa.gru.hidden_size))
        x, _ = fa.gru(x, h0)
        x = fa.fc(x)
        x = x.reshape(bsz, time_steps, groups, group)
        x = x.reshape(bsz, time_steps, padded_freq)
        if pad_len > 0:
            x = x[..., :freq]
        return x

    def _stream_grnn(self, grnn, x, h=None):
        if h is None:
            num_directions = 2 if bool(grnn.bidirectional) else 1
            batch = int(x.shape[0])
            h = x.new_zeros(int(grnn.num_layers) * num_directions, batch, int(grnn.hidden_size))
        return grnn(x, h)

    def _stream_ctfa(self, ctfa, x, h_cache):
        zt = torch.mean(x.pow(2), dim=-1)
        at, h_cache = ctfa.ta_gru(zt.transpose(1, 2), h_cache)
        at = torch.sigmoid(ctfa.ta_fc(at).transpose(1, 2))
        af = torch.sigmoid(self._stream_fa(ctfa.fa, x))
        y = at[..., None] * x * af[:, None]
        return y, h_cache

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
        x = block.shuffle(x)
        return x, conv_cache, tfa_cache

    def _stream_dpgrnn(self, block, x, inter_cache):
        x = x.permute(0, 2, 3, 1)

        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self._stream_grnn(block.intra_rnn, intra_x)[0]
        intra_x = block.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, block.width, block.input_size)
        intra_x = block.intra_ln(intra_x)
        intra_out = x + intra_x

        x = intra_out.permute(0, 2, 1, 3)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x, inter_cache = self._stream_grnn(block.inter_rnn, inter_x, inter_cache)
        inter_x = block.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], block.width, -1, block.input_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = block.inter_ln(inter_x)
        inter_out = intra_out + inter_x

        return inter_out.permute(0, 3, 1, 2), inter_cache

    def forward(self, mix, conv_cache, tfa_cache, inter_cache):
        spec = mix.permute(0, 3, 2, 1)
        feat = torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1.0e-12))
        feat = self.erb.bm(feat)

        (
            conv_cache_e0,
            conv_cache_e1,
            conv_cache_e2,
            conv_cache_d2,
            conv_cache_d3,
            conv_cache_d4,
        ) = self._unpack_conv_cache(conv_cache)
        (
            tfa_cache_0,
            tfa_cache_1,
            tfa_cache_2,
            tfa_cache_3,
            tfa_cache_4,
            tfa_cache_5,
            tfa_cache_6,
            tfa_cache_7,
            tfa_cache_8,
            tfa_cache_9,
        ) = self._unpack_tfa_cache(tfa_cache)
        inter_cache_0, inter_cache_1 = self._unpack_inter_cache(inter_cache)

        en_outs = []
        feat, conv_cache_e0, tfa_cache_0 = self._stream_xconv(
            self.encoder.en_convs[0], feat, conv_cache_e0, tfa_cache_0
        )
        en_outs.append(feat)
        feat, conv_cache_e1, tfa_cache_1 = self._stream_xmb(
            self.encoder.en_convs[1], feat, tfa_cache_1, conv_cache_e1
        )
        en_outs.append(feat)
        feat, conv_cache_e2, tfa_cache_2 = self._stream_xdws(
            self.encoder.en_convs[2], feat, tfa_cache_2, conv_cache_e2
        )
        en_outs.append(feat)
        feat, _, tfa_cache_3 = self._stream_xmb(self.encoder.en_convs[3], feat, tfa_cache_3)
        en_outs.append(feat)
        feat, _, tfa_cache_4 = self._stream_xdws(self.encoder.en_convs[4], feat, tfa_cache_4)
        en_outs.append(feat)

        feat, inter_cache_0 = self._stream_dpgrnn(self.dpgrnn[0], feat, inter_cache_0)
        feat, inter_cache_1 = self._stream_dpgrnn(self.dpgrnn[1], feat, inter_cache_1)

        feat, _, tfa_cache_5 = self._stream_xdws(
            self.decoder.de_convs[0], feat + en_outs[4], tfa_cache_5
        )
        feat, _, tfa_cache_6 = self._stream_xmb(
            self.decoder.de_convs[1], feat + en_outs[3], tfa_cache_6
        )
        feat, conv_cache_d2, tfa_cache_7 = self._stream_xdws(
            self.decoder.de_convs[2], feat + en_outs[2], tfa_cache_7, conv_cache_d2
        )
        feat, conv_cache_d3, tfa_cache_8 = self._stream_xmb(
            self.decoder.de_convs[3], feat + en_outs[1], tfa_cache_8, conv_cache_d3
        )
        feat, conv_cache_d4, tfa_cache_9 = self._stream_xconv(
            self.decoder.de_convs[4], feat + en_outs[0], conv_cache_d4, tfa_cache_9
        )

        mask = self.erb.bs(torch.sigmoid(feat))
        enh = (spec * mask).permute(0, 3, 2, 1)
        conv_cache = self._pack_conv_cache(
            [conv_cache_e0, conv_cache_e1, conv_cache_e2, conv_cache_d2, conv_cache_d3, conv_cache_d4]
        )
        tfa_cache = self._pack_tfa_cache(
            [
                tfa_cache_0,
                tfa_cache_1,
                tfa_cache_2,
                tfa_cache_3,
                tfa_cache_4,
                tfa_cache_5,
                tfa_cache_6,
                tfa_cache_7,
                tfa_cache_8,
                tfa_cache_9,
            ]
        )
        inter_cache = self._pack_inter_cache([inter_cache_0, inter_cache_1])
        return enh, conv_cache, tfa_cache, inter_cache


def _clone_gru_time_major(src: nn.GRU) -> nn.GRU:
    if not isinstance(src, nn.GRU):
        raise TypeError(f"Expected nn.GRU, got {type(src)!r}")
    clone = nn.GRU(
        input_size=int(src.input_size),
        hidden_size=int(src.hidden_size),
        num_layers=int(src.num_layers),
        bias=bool(src.bias),
        batch_first=False,
        dropout=float(src.dropout),
        bidirectional=bool(src.bidirectional),
    )
    clone.load_state_dict(src.state_dict())
    return clone


class _LiSenTimeMajorGRUs(nn.Module):
    def __init__(self, dual_path_rnn):
        super().__init__()
        self.intra = _clone_gru_time_major(dual_path_rnn.intra_rnn_attn.rnn)
        self.inter = _clone_gru_time_major(dual_path_rnn.inter_rnn_attn.rnn)


class LiSenNetStreamONNXWrapper(nn.Module):
    """Local LiSenNet spec2spec stream wrapper.

    This is not hyyan2k's original offline LiSenNet export. The local MOS-SE
    model uses Griffin-Lim offline, which is not streamable. This wrapper uses
    the current noisy phase and cached causal network states.
    """

    def __init__(self, adapter, cache_names: list[str]):
        super().__init__()
        self.adapter = adapter
        self.cache_names = list(cache_names)
        core = self.adapter._core()
        self.time_major_grus = nn.ModuleList([
            _LiSenTimeMajorGRUs(block.dp_rnn_attn)
            for block in core.blocks
        ])
        self.debug_gru_shapes = True

    @staticmethod
    def _shape(tensor):
        return tuple(int(dim) for dim in tensor.shape)

    def _print_gru_debug(self, block_idx: int, name: str, gru: nn.GRU, x, h0=None, y=None, hn=None):
        if not self.debug_gru_shapes:
            return
        parts = [
            f"[LiSenNet GRU debug] block={block_idx}",
            f"gru={name}",
            f"batch_first={bool(gru.batch_first)}",
            f"input={self._shape(x)}",
        ]
        if h0 is not None:
            parts.append(f"h0={self._shape(h0)}")
        if y is not None:
            parts.append(f"output={self._shape(y)}")
        if hn is not None:
            parts.append(f"hn={self._shape(hn)}")
        print(" ".join(parts), flush=True)

    def _stream_dual_path_rnn(self, block_idx: int, module, x, state):
        bsz, emb, time, freq = x.size()
        if time != 1:
            raise RuntimeError(f"LiSen ONNX streaming expects one frame, got T={time}.")
        x = x.permute(0, 2, 3, 1)

        x_res = x
        y = module.intra_norm(x)
        y = y.reshape(bsz * time, freq, emb)
        y_tm = y.transpose(0, 1)
        intra_gru = self.time_major_grus[block_idx].intra
        intra_dirs = 2 if intra_gru.bidirectional else 1
        intra_h0 = y_tm.new_zeros(intra_gru.num_layers * intra_dirs, y_tm.shape[1], intra_gru.hidden_size)
        self._print_gru_debug(block_idx, "intra", intra_gru, y_tm, intra_h0)
        y_tm, intra_hn = intra_gru(y_tm, intra_h0)
        self._print_gru_debug(block_idx, "intra", intra_gru, y_tm, intra_h0, y_tm, intra_hn)
        y = y_tm.transpose(0, 1)
        y = module.intra_rnn_attn.dense(y)
        y = y.reshape(bsz, time, freq, emb)
        y = y + x_res

        x_res = y
        y = module.inter_norm(y)
        y = y.permute(0, 2, 1, 3).reshape(bsz * freq, time, emb)
        y_tm = y.transpose(0, 1)
        hidden = state["inter_hidden"]
        if int(hidden.shape[1]) != int(y_tm.shape[1]):
            raise RuntimeError(
                f"LiSen inter GRU hidden batch mismatch: input_batch={int(y_tm.shape[1])}, "
                f"hidden_batch={int(hidden.shape[1])}, hidden_shape={tuple(hidden.shape)}"
            )
        inter_gru = self.time_major_grus[block_idx].inter
        self._print_gru_debug(block_idx, "inter", inter_gru, y_tm, hidden)
        y_tm, hidden = inter_gru(y_tm, hidden)
        self._print_gru_debug(block_idx, "inter", inter_gru, y_tm, state["inter_hidden"], y_tm, hidden)
        y = y_tm.transpose(0, 1)
        y = module.inter_rnn_attn.dense(y)
        state["inter_hidden"] = hidden
        y = y.reshape(bsz, freq, time, emb).permute(0, 2, 1, 3)
        y = y + x_res
        return y.permute(0, 3, 1, 2), state

    def _stream_dpr(self, block_idx: int, module, x, state):
        x, state = self._stream_dual_path_rnn(block_idx, module.dp_rnn_attn, x, state)
        x, state = self.adapter._stream_conv_glu(module.conv_glu, x, state)
        return x, state

    def forward(self, spec_in, *cache_in):
        core = self.adapter._core()
        cache_map = {name: tensor for name, tensor in zip(self.cache_names, cache_in)}
        prev_phase = cache_map["prev_phase"]
        encoder_state = {
            "enc_conv_2": cache_map["enc_conv_2"],
            "enc_conv_3": cache_map["enc_conv_3"],
            "enc_conv_4": cache_map["enc_conv_4"],
        }
        block_states = []
        block_idx = 0
        while f"block_{block_idx}_inter_hidden" in cache_map:
            block_states.append({
                "inter_hidden": cache_map[f"block_{block_idx}_inter_hidden"],
                "conv_glu_cache": cache_map[f"block_{block_idx}_conv_glu_cache"],
            })
            block_idx += 1
        decoder_state = {"mask_conv": cache_map["mask_conv"]}

        compression = float(getattr(core, "compress_factor", 0.3))
        src_spec = _compress_spec(spec_in, compression)
        src_mag = torch.sqrt(
            src_spec[..., 0] * src_spec[..., 0]
            + src_spec[..., 1] * src_spec[..., 1]
            + 1.0e-8
        ).permute(0, 2, 1)
        src_pha = torch.atan2(src_spec[..., 1], src_spec[..., 0]).permute(0, 2, 1)
        cur_phase = src_pha[:, 0, :]
        gd = diff_onnx_safe_zero_prepend(cur_phase, dim=1)
        freq_axis = torch.arange(cur_phase.shape[-1], device=cur_phase.device, dtype=cur_phase.dtype)
        hop_len = float(getattr(core, "hop_length", 256))
        n_fft = float(getattr(core, "n_fft", 512))
        ifd = (cur_phase - prev_phase) - 2.0 * math.pi * (hop_len / n_fft) * freq_axis[None, :]
        gd = _wrap_phase(gd).unsqueeze(1)
        ifd = _wrap_phase(ifd).unsqueeze(1)

        x = torch.stack([src_mag, gd / math.pi, ifd / math.pi], dim=1)
        encoder_out, encoder_state = self.adapter._stream_encoder(core.encoder, x, encoder_state)
        x = encoder_out[-1]
        for idx, block in enumerate(core.blocks):
            x, block_states[idx] = self._stream_dpr(idx, block, x, block_states[idx])
        mask, decoder_state = self.adapter._stream_decoder(core.decoder, x, encoder_out, decoder_state)

        est_mag = (mask[:, 0] + 1.0e-8) * src_mag + (mask[:, 1] + 1.0e-8) * src_mag
        est_spec = torch.stack(
            [est_mag * torch.cos(src_pha), est_mag * torch.sin(src_pha)],
            dim=-1,
        ).permute(0, 2, 1, 3)
        spec_out = _uncompress_spec(est_spec, compression)

        outputs = [cur_phase]
        outputs.extend([encoder_state["enc_conv_2"], encoder_state["enc_conv_3"], encoder_state["enc_conv_4"]])
        for state in block_states:
            outputs.append(state["inter_hidden"])
            outputs.append(state["conv_glu_cache"])
        outputs.append(decoder_state["mask_conv"])
        return (spec_out, *outputs)


def _flatten_lisen_cache(state: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    items = [
        ("prev_phase", state["prev_phase"]),
        ("enc_conv_2", state["encoder_cache"]["enc_conv_2"]),
        ("enc_conv_3", state["encoder_cache"]["enc_conv_3"]),
        ("enc_conv_4", state["encoder_cache"]["enc_conv_4"]),
    ]
    for idx, block_state in enumerate(state["block_states"]):
        items.append((f"block_{idx}_inter_hidden", block_state["inter_hidden"]))
        items.append((f"block_{idx}_conv_glu_cache", block_state["conv_glu_cache"]))
    items.append(("mask_conv", state["decoder_cache"]["mask_conv"]))
    return items


def _make_fastenhancer_item(model, idx: int, device: torch.device) -> ExportItem:
    expert = model.experts[idx]
    core = model._fastenhancer_core(expert)
    print("[FastEnhancer ONNX debug] core type:", type(core), flush=True)
    if core is not None and hasattr(core, "forward"):
        try:
            signature = inspect.signature(core.forward)
        except Exception as exc:
            signature = f"<unavailable: {exc}>"
        print("[FastEnhancer ONNX debug] forward signature:", signature, flush=True)
    if core is None or not callable(getattr(core, "initialize_cache", None)) or not callable(getattr(core, "model_forward", None)):
        raise RuntimeError(
            "FastEnhancer-S current core is non-streaming Model.forward(x); "
            "need to use official FastEnhancer ONNXModel/spec wrapper."
        )
    if hasattr(core, "remove_weight_reparameterizations"):
        core.remove_weight_reparameterizations()
    stft = core.stft
    n_fft = int(stft.n_fft)
    hop = int(stft.hop_size)
    if hop != 256:
        raise RuntimeError(f"{expert.name} hop_size={hop}, expected 256.")
    cache = [
        item.to(device=device, dtype=torch.float32)
        for item in core.initialize_cache(torch.zeros(1, hop, device=device, dtype=torch.float32))
    ]
    spec_in = torch.zeros(1, n_fft // 2 + 1, 1, 2, device=device, dtype=torch.float32)
    wrapper = FastEnhancerSpecONNXWrapper(core).to(device).eval()
    input_names = ["spec_in"] + [f"cache_in_{i}" for i in range(len(cache))]
    output_names = ["spec_out"] + [f"cache_out_{i}" for i in range(len(cache))]
    return ExportItem(
        name=expert.name,
        source="aask1357/fastenhancer",
        wrapper=wrapper,
        inputs=(spec_in, *cache),
        input_names=input_names,
        output_names=output_names,
        cache_shapes=[list(t.shape) for t in cache],
        checkpoint=_expert_checkpoint(expert),
        wrapper_name="FastEnhancerSpecONNXWrapper",
        notes=(
            "Spec2spec export follows FastEnhancer scripts/export_onnx_spec.py; "
            "wrapper calls core.model_forward(spec_noisy, *cache) rather than waveform core.forward(x)."
        ),
    )


def _make_ulunas_item(model, idx: int, device: torch.device, native_gru: bool = False) -> ExportItem:
    expert = model.experts[idx]
    core = getattr(getattr(expert, "model", None), "enhancer", None)
    if core is None:
        raise RuntimeError(f"{expert.name} has no `model.enhancer`; cannot export UL-UNAS stream ONNX.")
    wrapper = ULUNASSpecONNXWrapper(core).to(device).eval()
    n_fft = int(wrapper.n_fft)
    hop = int(wrapper.hop_len)
    if hop != 256:
        raise RuntimeError(f"{expert.name} hop_size={hop}, expected 256.")
    cache = [
        item.to(device=device, dtype=torch.float32)
        for item in wrapper.init_caches(batch_size=1, device=device, dtype=torch.float32)
    ]
    spec_in = torch.zeros(1, n_fft // 2 + 1, 1, 2, device=device, dtype=torch.float32)
    return ExportItem(
        name=expert.name,
        source="Xiaobin-Rong/ul-unas",
        wrapper=wrapper,
        inputs=(spec_in, *cache),
        input_names=["mix", "conv_cache", "tfa_cache", "inter_cache"],
        output_names=["enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"],
        cache_shapes=[list(t.shape) for t in cache],
        checkpoint=_expert_checkpoint(expert),
        wrapper_name="ULUNASSpecONNXWrapper",
        notes=(
            "Spec2spec export mirrors Xiaobin-Rong/ul-unas ulunas_onnx/stream/ulunas_stream.py; "
            "inputs are mix/conv_cache/tfa_cache/inter_cache and outputs are enh/*_out. "
            + (
                "UL-UNAS keeps native ONNX GRU nodes for performance testing."
                if native_gru else
                "UL-UNAS GRU modules are decomposed to primitive ops for CUDA EP compatibility."
            )
        ),
        decompose_gru=not bool(native_gru),
        gru_export="onnx_gru_native" if native_gru else "decomposed",
    )


def _make_lisen_item(model, idx: int, device: torch.device) -> ExportItem:
    expert = model.experts[idx]
    adapter = LiSenStreamingExpertAdapter(model, idx, expert)
    state = adapter.init_state(device=device, batch_size=1, dtype=torch.float32)
    dummy_frame = torch.zeros(1, int(state["n_fft"]), device=device, dtype=torch.float32)
    with torch.no_grad():
        _, state = adapter.stream_step(dummy_frame, state)
    cache_items = _flatten_lisen_cache(state)
    cache_names = [name for name, _ in cache_items]
    cache = [tensor.to(device=device, dtype=torch.float32) for _, tensor in cache_items]
    for name, tensor in cache_items:
        if "inter_hidden" in name:
            print(
                f"[LiSenNet GRU debug] cache name={name} shape={tuple(int(dim) for dim in tensor.shape)}",
                flush=True,
            )
    n_fft = int(state["n_fft"])
    hop = int(state["hop_len"])
    if hop != 256:
        raise RuntimeError(f"{expert.name} hop_size={hop}, expected 256.")
    spec_in = torch.zeros(1, n_fft // 2 + 1, 1, 2, device=device, dtype=torch.float32)
    wrapper = LiSenNetStreamONNXWrapper(adapter, cache_names).to(device).eval()
    return ExportItem(
        name=expert.name,
        source="local_modified",
        wrapper=wrapper,
        inputs=(spec_in, *cache),
        input_names=["spec_in"] + [f"cache_in_{i}" for i in range(len(cache))],
        output_names=["spec_out"] + [f"cache_out_{i}" for i in range(len(cache))],
        cache_shapes=[list(t.shape) for t in cache],
        checkpoint=_expert_checkpoint(expert),
        wrapper_name="LiSenNetStreamONNXWrapper",
        notes=(
            "Local MOS-SE LiSen implementation. It is not hyyan2k official ONNX and not "
            "FastEnhancer's models/lisennet class. The wrapper uses noisy phase instead of "
            "offline Griffin-Lim for streamability."
        ),
    )


def _make_export_item(model, idx: int, device: torch.device, ulunas_native_gru: bool = False) -> ExportItem:
    expert = model.experts[idx]
    class_name = expert.model_class.__name__
    if class_name == "FastEnhancer":
        return _make_fastenhancer_item(model, idx, device)
    if class_name == "ULUNAS":
        return _make_ulunas_item(model, idx, device, native_gru=ulunas_native_gru)
    if class_name == "LiSen":
        return _make_lisen_item(model, idx, device)
    raise RuntimeError(f"Unsupported expert {expert.name}: {class_name}")


def inspect_onnx_io_for_iobinding(onnx_path: str | Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph

    def shape_of(value):
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(str(dim.dim_param))
            else:
                dims.append(None)
        return dims

    def dtype_of(value):
        elem = value.type.tensor_type.elem_type
        return onnx.TensorProto.DataType.Name(elem)

    inputs = [
        {"name": item.name, "shape": shape_of(item), "dtype": dtype_of(item)}
        for item in graph.input
    ]
    outputs = [
        {"name": item.name, "shape": shape_of(item), "dtype": dtype_of(item)}
        for item in graph.output
    ]
    fixed = all(all(isinstance(dim, int) and dim > 0 for dim in io["shape"]) for io in [*inputs, *outputs])
    float32 = all(io["dtype"] == "FLOAT" for io in [*inputs, *outputs])
    return {
        "inputs": inputs,
        "outputs": outputs,
        "metadata_fixed_shapes": fixed,
        "runtime_fixed_shapes": None,
        "fixed_shapes": fixed,
        "float32": float32,
    }


def _assert_no_onnx_complex_dtype(onnx_path: str | Path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    complex_types = {
        onnx.TensorProto.COMPLEX64,
        onnx.TensorProto.COMPLEX128,
    }
    offenders = []
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type in complex_types:
            offenders.append(value.name)
    for init in model.graph.initializer:
        if init.data_type in complex_types:
            offenders.append(init.name)
    if offenders:
        raise RuntimeError(
            f"ONNX graph contains complex dtype tensors: {offenders}. "
            "Spec experts must use real/imag float representation [B,F,1,2]."
        )


def _onnx_value_shape(value) -> list[Any]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def debug_onnx_gru_nodes(onnx_path: str | Path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shape_map: dict[str, list[Any]] = {}
    for value in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]:
        if value.type.HasField("tensor_type"):
            shape_map[value.name] = _onnx_value_shape(value)
    for init in inferred.graph.initializer:
        shape_map[init.name] = [int(dim) for dim in init.dims]

    found = False
    for node in inferred.graph.node:
        if node.op_type != "GRU":
            continue
        found = True
        print(f"[ONNX GRU debug] node={node.name or '<unnamed>'}", flush=True)
        print(f"[ONNX GRU debug]   inputs={list(node.input)}", flush=True)
        for name in node.input:
            if name:
                print(f"[ONNX GRU debug]     input {name}: shape={shape_map.get(name)}", flush=True)
        print(f"[ONNX GRU debug]   outputs={list(node.output)}", flush=True)
        for name in node.output:
            if name:
                print(f"[ONNX GRU debug]     output {name}: shape={shape_map.get(name)}", flush=True)
    if not found:
        print(f"[ONNX GRU debug] no GRU nodes found in {onnx_path}", flush=True)


def _load_wav_mono(path: str | Path, sample_rate: int) -> torch.Tensor:
    path = str(path)
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        else:
            wav = wav.reshape(-1)
        if int(sr) != int(sample_rate):
            wav = torchaudio.functional.resample(wav.unsqueeze(0), int(sr), int(sample_rate)).squeeze(0)
        return wav.to(torch.float32)
    except Exception:
        try:
            import soundfile as sf
        except Exception as exc:
            raise RuntimeError(
                "Reading --check-wav requires torchaudio or soundfile. "
                "Install one of them, or omit --check-wav."
            ) from exc
        data, sr = sf.read(path, always_2d=False, dtype="float32")
        if int(sr) != int(sample_rate):
            raise RuntimeError(
                f"--check-wav sample_rate={sr}, expected {sample_rate}. "
                "Install torchaudio to enable resampling, or use a 16 kHz wav."
            )
        tensor = torch.from_numpy(data)
        if tensor.ndim == 2:
            tensor = tensor.mean(dim=1)
        return tensor.reshape(-1).to(torch.float32)


def _spec_frames_from_wav(
    wav_path: str | Path,
    sample_rate: int,
    n_fft: int,
    hop: int,
    device: torch.device,
    max_frames: int | None = None,
) -> torch.Tensor:
    wav = _load_wav_mono(wav_path, sample_rate).to(device=device, dtype=torch.float32)
    if wav.numel() == 0:
        wav = torch.zeros(hop, device=device, dtype=torch.float32)
    context_len = max(int(n_fft) - int(hop), 0)
    tail = torch.zeros(context_len, device=device, dtype=torch.float32)
    window = torch.hann_window(int(n_fft), periodic=True, device=device, dtype=torch.float32)
    frames = []
    offset = 0
    while offset < wav.shape[-1]:
        release_len = min(int(hop), wav.shape[-1] - offset)
        hop_tensor = wav[offset: offset + release_len]
        if release_len < hop:
            hop_tensor = F.pad(hop_tensor, (0, int(hop) - release_len))
        frame = torch.cat([tail, hop_tensor], dim=0) if context_len else hop_tensor
        if frame.shape[-1] < n_fft:
            frame = F.pad(frame, (int(n_fft) - frame.shape[-1], 0))
        elif frame.shape[-1] > n_fft:
            frame = frame[-int(n_fft):]
        if context_len:
            tail = frame[-context_len:].detach()
        spec = torch.fft.rfft(frame.unsqueeze(0) * window.view(1, -1), n=int(n_fft), dim=1)
        frames.append(torch.view_as_real(spec).unsqueeze(2))
        offset += int(hop)
        if max_frames is not None and len(frames) >= int(max_frames):
            break
    if not frames:
        spec = torch.zeros(1, int(n_fft) // 2 + 1, 1, 2, device=device, dtype=torch.float32)
        frames.append(spec)
    return torch.stack(frames, dim=0)


def _check_cuda_ep(path: Path, device_id: int) -> tuple[bool, str]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return False, f"onnxruntime import failed: {exc}"
    try:
        sess = ort.InferenceSession(
            str(path),
            providers=[("CUDAExecutionProvider", {"device_id": int(device_id)})],
        )
        providers = sess.get_providers()
        ok = "CUDAExecutionProvider" in providers
        return ok, ",".join(providers)
    except Exception as exc:
        return False, (
            f"{exc}. Check libcudnn.so.9, LD_LIBRARY_PATH, onnxruntime-gpu, CUDA 12 / cuDNN 9."
        )


def _expected_output_shapes(item: ExportItem) -> list[list[int]]:
    return [list(item.inputs[0].shape)] + [list(tensor.shape) for tensor in item.inputs[1:]]


def _runtime_output_shape_check(
    item: ExportItem,
    path: Path,
    device_id: int,
    require_cuda_ep: bool,
    runs: int = 2,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    expected_shapes = _expected_output_shapes(item)
    feeds0 = {
        name: tensor.detach().cpu().numpy().astype("float32", copy=True)
        for name, tensor in zip(item.input_names, item.inputs)
    }

    def _print_run_context(label: str, sess, feeds):
        print(f"[ONNX runtime debug] expert={item.name} provider={label}", flush=True)
        print(f"[ONNX runtime debug] onnx_inputs={[inp.name for inp in sess.get_inputs()]}", flush=True)
        print(f"[ONNX runtime debug] output_names={item.output_names}", flush=True)
        for name, value in feeds.items():
            print(
                f"[ONNX runtime debug] feed name={name} shape={list(value.shape)} dtype={value.dtype}",
                flush=True,
            )

    def _run_with_providers(label: str, providers):
        try:
            sess = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise RuntimeError(
                f"{item.name} failed to create ONNXRuntime session with {label}: {exc}"
            ) from exc

        actual_providers = sess.get_providers()
        if label == "CUDAExecutionProvider" and "CUDAExecutionProvider" not in actual_providers:
            raise RuntimeError(
                f"{item.name} CUDAExecutionProvider was requested but did not load. "
                f"Actual providers: {actual_providers}."
            )

        session_input_names = [inp.name for inp in sess.get_inputs()]
        session_output_names = [out.name for out in sess.get_outputs()]
        if session_input_names != item.input_names:
            raise RuntimeError(
                f"{item.name} ONNX input names mismatch. expected={item.input_names}, actual={session_input_names}"
            )
        if session_output_names != item.output_names:
            raise RuntimeError(
                f"{item.name} ONNX output names mismatch. expected={item.output_names}, actual={session_output_names}"
            )

        io_types = [inp.type for inp in sess.get_inputs()] + [out.type for out in sess.get_outputs()]
        if any(dtype != "tensor(float)" for dtype in io_types):
            raise RuntimeError(f"{item.name} ONNX I/O dtype must be float32, got {io_types}")

        feeds = {name: value.copy() for name, value in feeds0.items()}
        shape_runs = []
        dtype_runs = []
        for run_idx in range(max(1, int(runs))):
            try:
                outputs = sess.run(item.output_names, feeds)
            except Exception as exc:
                _print_run_context(label, sess, feeds)
                debug_onnx_gru_nodes(path)
                raise RuntimeError(
                    f"{item.name} sess.run failed on {label} at dummy_run={run_idx}: {exc}"
                ) from exc
            if len(outputs) != len(item.output_names):
                raise RuntimeError(
                    f"{item.name} ONNX output count mismatch. expected={len(item.output_names)}, actual={len(outputs)}"
                )
            output_shapes = [list(value.shape) for value in outputs]
            output_dtypes = [str(value.dtype) for value in outputs]
            if any(value.dtype != np.float32 for value in outputs):
                raise RuntimeError(f"{item.name} runtime output dtype must be float32, got {output_dtypes}")
            if output_shapes != expected_shapes:
                raise RuntimeError(
                    f"{item.name} runtime output shape mismatch on {label}. "
                    f"expected={expected_shapes}, actual={output_shapes}"
                )
            shape_runs.append(output_shapes)
            dtype_runs.append(output_dtypes)
            for cache_input_name, value in zip(item.input_names[1:], outputs[1:]):
                feeds[cache_input_name] = np.asarray(value, dtype=np.float32)

        first_shapes = shape_runs[0]
        for idx, shapes in enumerate(shape_runs[1:], start=1):
            if shapes != first_shapes:
                raise RuntimeError(
                    f"{item.name} runtime output shapes changed between dummy runs on {label}: "
                    f"run0={first_shapes}, run{idx}={shapes}"
                )
        return {
            "providers": actual_providers,
            "runtime_output_shapes": first_shapes,
            "runtime_output_dtypes": dtype_runs[0],
        }

    cpu_result = _run_with_providers("CPUExecutionProvider", ["CPUExecutionProvider"])
    if require_cuda_ep:
        cuda_result = _run_with_providers(
            "CUDAExecutionProvider",
            [("CUDAExecutionProvider", {"device_id": int(device_id)})],
        )
        if cuda_result["runtime_output_shapes"] != cpu_result["runtime_output_shapes"]:
            raise RuntimeError(
                f"{item.name} CPU/CUDA runtime output shape mismatch: "
                f"cpu={cpu_result['runtime_output_shapes']}, cuda={cuda_result['runtime_output_shapes']}"
            )
        selected = cuda_result
        providers = cuda_result["providers"]
    else:
        selected = cpu_result
        providers = cpu_result["providers"]

    return {
        "providers": providers,
        "cpu_providers": cpu_result["providers"],
        "runtime_fixed_shapes": True,
        "runtime_output_shapes": selected["runtime_output_shapes"],
        "runtime_output_dtypes": selected["runtime_output_dtypes"],
        "expected_output_shapes": expected_shapes,
    }


def _run_consistency_check(
    item: ExportItem,
    path: Path,
    frames: int = 3,
    spec_frames: torch.Tensor | None = None,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    device = item.inputs[0].device
    freq = int(item.inputs[0].shape[1])
    if spec_frames is None:
        spec_frames = torch.randn(frames, 1, freq, 1, 2, device=device, dtype=torch.float32) * 0.05
    else:
        spec_frames = spec_frames.to(device=device, dtype=torch.float32)
        if spec_frames.ndim != 5 or tuple(spec_frames.shape[2:]) != (freq, 1, 2):
            raise ValueError(f"Expected spec_frames [N,1,{freq},1,2], got {tuple(spec_frames.shape)}.")
        frames = int(spec_frames.shape[0])

    with torch.no_grad():
        pt_cache = [tensor.clone() for tensor in item.inputs[1:]]
        pt_out = []
        for idx in range(frames):
            result = item.wrapper(spec_frames[idx], *pt_cache)
            pt_out.append(result[0].detach().cpu())
            pt_cache = [tensor.detach() for tensor in result[1:]]
        pt_out = torch.cat(pt_out, dim=2).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_cache = [tensor.detach().cpu().numpy().astype("float32", copy=True) for tensor in item.inputs[1:]]
    ort_out = []
    for idx in range(frames):
        feeds = {item.input_names[0]: spec_frames[idx].detach().cpu().numpy().astype("float32", copy=False)}
        feeds.update({name: value for name, value in zip(item.input_names[1:], ort_cache)})
        outputs = sess.run(item.output_names, feeds)
        ort_out.append(outputs[0])
        ort_cache = [np.asarray(value, dtype=np.float32) for value in outputs[1:]]
    ort_out = np.concatenate(ort_out, axis=2)

    diff = np.abs(pt_out - ort_out)
    signal = np.mean(pt_out ** 2) + 1.0e-12
    noise = np.mean((pt_out - ort_out) ** 2) + 1.0e-12
    cache_shape_match = all(tuple(a.shape) == tuple(b.shape) for a, b in zip(pt_cache, ort_cache))
    return {
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "snr_diff": float(10.0 * math.log10(signal / noise)),
        "shape_match": tuple(pt_out.shape) == tuple(ort_out.shape),
        "cache_shape_match": bool(cache_shape_match),
    }


def _export_item(
    item: ExportItem,
    path: Path,
    opset: int,
    check: bool,
    device_id: int,
    simplify: bool,
    require_cuda_ep: bool,
    check_spec_frames: torch.Tensor | None = None,
) -> dict[str, Any]:
    decompose_gru = bool(item.decompose_gru)
    _prepare_export_module(item.wrapper, decompose_gru=decompose_gru, label=item.name)
    kwargs = {
        "input_names": item.input_names,
        "output_names": item.output_names,
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] exporting expert={item.name}, path={path}", flush=True)
    print(f"[export] expert={item.name}, gru_export={item.gru_export}, decompose_gru={item.decompose_gru}", flush=True)
    torch.onnx.export(item.wrapper, item.inputs, str(path), **kwargs)
    if simplify:
        _simplify_onnx(path)

    import onnx
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    _assert_no_onnx_complex_dtype(path)
    if item.wrapper_name in {
        "LiSenNetStreamONNXWrapper",
        "FastEnhancerSpecONNXWrapper",
        "ULUNASSpecONNXWrapper",
    }:
        debug_onnx_gru_nodes(path)

    io = inspect_onnx_io_for_iobinding(path)
    if not io["float32"]:
        raise RuntimeError(f"{item.name} ONNX I/O dtype must be float32: {io}")
    runtime_shapes = _runtime_output_shape_check(
        item,
        path,
        device_id=device_id,
        require_cuda_ep=require_cuda_ep,
        runs=2,
    )
    io["runtime_fixed_shapes"] = bool(runtime_shapes["runtime_fixed_shapes"])
    io["runtime_outputs"] = [
        {"name": name, "shape": shape, "dtype": dtype}
        for name, shape, dtype in zip(
            item.output_names,
            runtime_shapes["runtime_output_shapes"],
            runtime_shapes["runtime_output_dtypes"],
        )
    ]
    io["runtime_output_shapes"] = runtime_shapes["runtime_output_shapes"]
    io["expected_output_shapes"] = runtime_shapes["expected_output_shapes"]
    io["fixed_shapes"] = bool(io["runtime_fixed_shapes"])
    cuda_ok = bool((not require_cuda_ep) or "CUDAExecutionProvider" in runtime_shapes["providers"])
    cuda_msg = ",".join(runtime_shapes["providers"])
    if not io["metadata_fixed_shapes"] and io["runtime_fixed_shapes"]:
        print(
            f"[export] WARNING: expert={item.name} ONNX metadata has symbolic output dims, but runtime output shapes "
            "are fixed under the exported streaming configuration.",
            flush=True,
        )
    consistency = None
    if check:
        consistency = _run_consistency_check(item, path, spec_frames=check_spec_frames)
        if consistency["max_abs_error"] > 1.0e-3 or consistency["mean_abs_error"] > 1.0e-4:
            raise RuntimeError(
                f"{item.name} ONNX consistency check failed: "
                f"max_abs_error={consistency['max_abs_error']:.6g}, "
                f"mean_abs_error={consistency['mean_abs_error']:.6g}"
            )
    if not io["runtime_fixed_shapes"]:
        raise RuntimeError(f"{item.name} ONNX runtime shapes are not suitable for I/O Binding: {io}")
    return {
        "io": io,
        "runtime_shape_check": runtime_shapes,
        "cuda_ep": {"ok": bool(cuda_ok), "message": cuda_msg},
        "consistency": consistency,
    }


def _scoreq_dnsmos_audit(root: Path) -> list[str]:
    patterns = ["*scoreq*onnx*", "*DNSMOS*onnx*", "*dnsmos*onnx*", "sig_bak_ovr.onnx"]
    found = []
    for pattern in patterns:
        found.extend(str(path.relative_to(root)) for path in root.rglob(pattern) if path.is_file())
    return sorted(set(found))


def _write_report(path: Path, manifest: dict[str, Any], audit: list[str]) -> None:
    lines = [
        "# MoE Expert ONNX Export Report",
        "",
        f"- sample_rate: {manifest['sample_rate']}",
        f"- n_fft: {manifest['n_fft']}",
        f"- hop_size: {manifest['hop_size']}",
        f"- format: {manifest['format']}",
        "",
        "## Experts",
        "",
    ]
    for expert in manifest["experts"]:
        lines.extend([
            f"### {expert['name']}",
            "",
            f"- source: {expert['source']}",
            f"- checkpoint: {expert.get('checkpoint')}",
            f"- wrapper: {expert.get('wrapper')}",
            f"- gru_export: {expert.get('gru_export')}",
            f"- decompose_gru: `{expert.get('decompose_gru')}`",
            f"- onnx_path: {expert['onnx_path']}",
            f"- input_names: `{expert['input_names']}`",
            f"- output_names: `{expert['output_names']}`",
            f"- cache_shapes: `{expert['cache_shapes']}`",
            f"- runtime_output_shapes: `{expert.get('runtime_output_shapes')}`",
            f"- cuda_ep: `{expert.get('cuda_ep')}`",
            f"- iobinding_ready: `{expert.get('iobinding_ready')}`",
            f"- consistency: `{expert.get('consistency')}`",
            f"- notes: {expert.get('notes', '')}",
            "",
        ])
    lines.extend([
        "## Evaluation ONNX Audit",
        "",
        "The following files look like SCOREQ/DNSMOS/evaluation ONNX assets and were not added as experts:",
        "",
    ])
    if audit:
        lines.extend([f"- {item}" for item in audit])
    else:
        lines.append("- none found")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True)
    parser.add_argument("--out-dir", "--out", dest="out_dir", required=True)
    parser.add_argument("--format", default="spec", choices=["spec"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-wav", default=None, help="Optional noisy wav used for PyTorch-vs-ONNX stream checks.")
    parser.add_argument("--check-max-frames", type=int, default=50)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument(
        "--ulunas-native-gru",
        action="store_true",
        help=(
            "Export UL-UNAS with official-style stream wrapper but keep native ONNX GRU nodes. "
            "This may be faster than decomposed GRU if CUDAExecutionProvider can run it."
        ),
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    conf_path = _resolve_required_conf(args.conf)
    conf = OmegaConf.load(conf_path)
    if args.overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.overrides))
    conf["ckpt"] = None
    if conf.get("router_grpo") is not None:
        if conf.router_grpo.get("inference_branch") is None:
            conf.router_grpo.inference_branch = {}
        conf.router_grpo.inference_branch.runtime = "torch"
        if conf.router_grpo.get("device_map") is not None:
            conf.router_grpo.device_map.enabled = False

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")
    device_id = _cuda_index(device)

    model = get_model(conf).to(device).eval()
    _load_router_checkpoint(model, args.ckpt)
    if hasattr(model, "_router_device_override"):
        model._router_device_override = device
    if hasattr(model, "_expert_device_overrides"):
        model._expert_device_overrides = [device for _ in range(len(model.experts))]

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = int(getattr(model, "sample_rate", 16000))
    n_fft = int(getattr(model, "stream_frame_samples", 512))
    hop = int(getattr(model, "stream_hop_samples", 256))
    if sample_rate != 16000 or hop != 256:
        raise RuntimeError(f"Expected sample_rate=16000 and hop=256, got sample_rate={sample_rate}, hop={hop}.")

    check_spec_frames = None
    if args.check and args.check_wav:
        check_spec_frames = _spec_frames_from_wav(
            args.check_wav,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop=hop,
            device=device,
            max_frames=args.check_max_frames,
        )

    expert_entries = []
    for idx in range(len(model.experts)):
        item = _make_export_item(model, idx, device, ulunas_native_gru=bool(args.ulunas_native_gru))
        onnx_name = f"{item.name}.onnx"
        onnx_path = out_dir / onnx_name
        started = time.perf_counter()
        result = _export_item(
            item,
            onnx_path,
            opset=int(args.opset),
            check=bool(args.check),
            device_id=device_id,
            simplify=bool(args.simplify),
            require_cuda_ep=(device.type == "cuda"),
            check_spec_frames=check_spec_frames,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        entry = {
            "name": item.name,
            "onnx_path": onnx_name,
            "path": onnx_name,
            "source": item.source,
            "checkpoint": item.checkpoint,
            "wrapper": item.wrapper_name,
            "protocol": "spec_frame",
            "true_cache": True,
            "input": item.input_names[0],
            "n_fft": n_fft,
            "win_length": n_fft,
            "frame_samples": n_fft,
            "hop_samples": hop,
            "input_names": item.input_names,
            "output_names": item.output_names,
            "cache_inputs": item.input_names[1:],
            "cache_outputs": item.output_names[1:],
            "cache_shapes": item.cache_shapes,
            "runtime_output_shapes": result["runtime_shape_check"]["runtime_output_shapes"],
            "expected_output_shapes": result["runtime_shape_check"]["expected_output_shapes"],
            "decompose_gru": bool(item.decompose_gru),
            "gru_export": item.gru_export,
            "io": result["io"],
            "cuda_ep": result["cuda_ep"],
            "iobinding_ready": bool(result["io"]["runtime_fixed_shapes"] and result["io"]["float32"]),
            "consistency": result["consistency"],
            "export_ms": round(elapsed_ms, 3),
            "notes": item.notes,
        }
        expert_entries.append(entry)
        print(f"[export_moe_experts_onnx] exported {item.name}: {onnx_path}")

    manifest = {
        "version": 2,
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_size": hop,
        "hop_samples": hop,
        "win_size": n_fft,
        "frame_samples": n_fft,
        "frame_ms": 1000.0 * n_fft / sample_rate,
        "hop_ms": 1000.0 * hop / sample_rate,
        "format": "spec",
        "stft": {"name": "torch", "n_fft": n_fft, "win_length": n_fft, "hop_length": hop},
        "router": {"protocol": "pytorch", "exported": False, "path": None},
        "experts": expert_entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, cls=NumpyEncoder), encoding="utf-8")
    audit = _scoreq_dnsmos_audit(REPO_ROOT)
    _write_report(out_dir / "export_report.md", manifest, audit)
    print(f"[export_moe_experts_onnx] wrote {manifest_path}")
    print(f"[export_moe_experts_onnx] wrote {out_dir / 'export_report.md'}")


if __name__ == "__main__":
    main()
