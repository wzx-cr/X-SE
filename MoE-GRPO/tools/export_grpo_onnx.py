"""Export FrozenExpertRouterGRPO online-inference ONNX assets.

This script keeps training unchanged. It exports only the online inference
branch assets used by router_grpo.inference_branch.runtime=onnx:

  - one ONNX stream-step file per expert
  - router_features.onnx: optional router feature vector -> weights/logits,
    exported only with --export-router
  - manifest.json: runtime protocol, paths, cache input/output names

Run from the repository root, for example:

  python tools/export_grpo_onnx.py \
    --conf examples/voicebank/conf/lisennet_fastenhancerS_ulunas_moe.yaml \
    --out exp/onnx/lisennet_fastenhancerS_ulunas_moe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
cleaned_sys_path = []
for item in sys.path:
    try:
        if Path(item or ".").resolve() == TOOLS_DIR:
            continue
    except Exception:
        pass
    cleaned_sys_path.append(item)
sys.path[:] = cleaned_sys_path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.train import get_model  # noqa: E402
from modules.utils.common import NumpyEncoder, resolve_path  # noqa: E402
from alpha.enh.system.grpo import (  # noqa: E402
    LiSenStreamingExpertAdapter,
)


def _plain_conf(conf):
    return OmegaConf.to_container(conf, resolve=True) if OmegaConf.is_config(conf) else dict(conf)


def _cuda_index(device_spec: Any):
    if device_spec is None:
        return None
    text = str(device_spec).strip().lower()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda"):
        if ":" not in text:
            return 0
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _onnx_provider_fields(device_spec: Any):
    cuda_idx = _cuda_index(device_spec)
    if cuda_idx is None:
        return {}
    return {
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "provider_options": [{"device_id": int(cuda_idx)}, {}],
    }


def _device_map_for_export(conf):
    router_grpo = _plain_conf(conf.get("router_grpo", {}))
    device_map = router_grpo.get("device_map", router_grpo.get("multi_gpu", {})) or {}
    if not device_map or not bool(device_map.get("enabled", bool(device_map))):
        return None, []
    grpo_device = device_map.get("grpo_device", device_map.get("router_device", "cuda:0"))
    expert_devices = device_map.get("expert_devices", device_map.get("experts", [])) or []
    if isinstance(expert_devices, (str, int)):
        expert_devices = [expert_devices]
    return grpo_device, list(expert_devices)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    try:
        return Path(resolve_path(str(path)))
    except Exception:
        return REPO_ROOT / path


def _resolve_required_conf(path: str | Path) -> Path:
    conf_path = _resolve(path)
    if conf_path.exists():
        return conf_path

    requested_name = Path(path).name
    candidates = []
    search_roots = [REPO_ROOT / "examples", REPO_ROOT]
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        patterns = [requested_name, "*moe*.yaml", "*GRPO*.yaml", "*grpo*.yaml"]
        for pattern in patterns:
            for item in root.rglob(pattern):
                if item.is_file() and item not in seen:
                    candidates.append(item)
                    seen.add(item)
                if len(candidates) >= 12:
                    break
            if len(candidates) >= 12:
                break
        if len(candidates) >= 12:
            break

    lines = [
        f"Config file not found: {conf_path}",
        "Check that the YAML has been copied to this machine, or pass the real path with --conf.",
    ]
    if candidates:
        lines.append("Available candidate configs:")
        lines.extend(f"  - {item.relative_to(REPO_ROOT)}" for item in candidates)
    raise FileNotFoundError("\n".join(lines))


def _load_router_checkpoint(model, ckpt_path: str | None) -> None:
    if not ckpt_path:
        return
    ckpt_path = _resolve(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[export_grpo_onnx] loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"[export_grpo_onnx] missing keys: {len(missing)}")
    if unexpected:
        print(f"[export_grpo_onnx] unexpected keys: {len(unexpected)}")


class ExportableGRU(nn.Module):
    """GRU implemented with primitive ops so ONNX export avoids the GRU op.

    ONNX Runtime CUDA can fail on the fused GRU kernel when several expert
    sessions run concurrently on one GPU. This module preserves nn.GRU
    weights but expands the recurrence into MatMul/Add/Sigmoid/Tanh nodes.
    """

    def __init__(self, gru: nn.GRU):
        super().__init__()
        if not isinstance(gru, nn.GRU):
            raise TypeError(f"ExportableGRU expects nn.GRU, got {type(gru)!r}")
        self.gru = gru
        self.input_size = int(gru.input_size)
        self.hidden_size = int(gru.hidden_size)
        self.num_layers = int(gru.num_layers)
        self.bias = bool(gru.bias)
        self.batch_first = bool(gru.batch_first)
        self.dropout = float(gru.dropout)
        self.bidirectional = bool(gru.bidirectional)

    def _suffix(self, layer: int, reverse: bool) -> str:
        suffix = f"_l{layer}"
        if reverse:
            suffix += "_reverse"
        return suffix

    def _weights(self, layer: int, reverse: bool):
        suffix = self._suffix(layer, reverse)
        w_ih = getattr(self.gru, f"weight_ih{suffix}")
        w_hh = getattr(self.gru, f"weight_hh{suffix}")
        b_ih = getattr(self.gru, f"bias_ih{suffix}", None) if self.bias else None
        b_hh = getattr(self.gru, f"bias_hh{suffix}", None) if self.bias else None
        return w_ih, w_hh, b_ih, b_hh

    def flatten_parameters(self):
        return None

    def _cell(self, x_t, h_t, layer: int, reverse: bool):
        w_ih, w_hh, b_ih, b_hh = self._weights(layer, reverse)
        gate_x = F.linear(x_t, w_ih, b_ih)
        gate_h = F.linear(h_t, w_hh, b_hh)
        x_r, x_z, x_n = gate_x.chunk(3, dim=-1)
        h_r, h_z, h_n = gate_h.chunk(3, dim=-1)
        reset = torch.sigmoid(x_r + h_r)
        update = torch.sigmoid(x_z + h_z)
        new = torch.tanh(x_n + reset * h_n)
        return new + update * (h_t - new)

    def _direction(self, x, h0, layer: int, reverse: bool):
        seq_len = int(x.shape[0])
        h_t = h0
        outputs = []
        if reverse:
            step_range = range(seq_len - 1, -1, -1)
        else:
            step_range = range(seq_len)
        for t in step_range:
            h_t = self._cell(x[t], h_t, layer, reverse)
            outputs.append(h_t)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=0), h_t

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        batch = int(x.shape[1])
        num_directions = 2 if self.bidirectional else 1
        if hx is None:
            hx = x.new_zeros(self.num_layers * num_directions, batch, self.hidden_size)
        layer_input = x
        h_out = []
        for layer in range(self.num_layers):
            fw_out, fw_h = self._direction(layer_input, hx[layer * num_directions], layer, False)
            if self.bidirectional:
                bw_out, bw_h = self._direction(layer_input, hx[layer * num_directions + 1], layer, True)
                layer_output = torch.cat([fw_out, bw_out], dim=-1)
                h_out.extend([fw_h, bw_h])
            else:
                layer_output = fw_out
                h_out.append(fw_h)
            layer_input = layer_output
        if self.batch_first:
            layer_input = layer_input.transpose(0, 1)
        return layer_input, torch.stack(h_out, dim=0)


def _replace_gru_for_export(module: nn.Module) -> int:
    if isinstance(module, ExportableGRU):
        return 0
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GRU):
            setattr(module, name, ExportableGRU(child))
            replaced += 1
        else:
            replaced += _replace_gru_for_export(child)
    return replaced


def _count_onnx_ops(path: Path, op_type: str):
    try:
        import onnx
    except Exception:
        return None
    model = onnx.load(str(path))
    return sum(1 for node in model.graph.node if node.op_type == op_type)


def _simplify_onnx(path: Path) -> bool:
    try:
        import onnx
        from onnxsim import simplify
    except Exception as exc:
        print(f"[export_grpo_onnx] warning: onnxsim not available; skip simplify for {path}: {exc}")
        return False
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    simplified, ok = simplify(onnx_model)
    if not ok:
        raise RuntimeError(f"onnxsim failed to validate simplified model: {path}")
    onnx.save(simplified, str(path))
    print(f"[export_grpo_onnx] simplified ONNX model: {path}")
    return True


def _prepare_export_module(module: nn.Module, decompose_gru: bool, label: str) -> int:
    if not decompose_gru:
        return 0
    replaced = _replace_gru_for_export(module)
    print(f"[export_grpo_onnx] {label}: decomposed {replaced} nn.GRU module(s) for ONNX export")
    return replaced


def _export_onnx(
    module: nn.Module,
    args,
    path: Path,
    input_names,
    output_names,
    opset: int,
    dynamo: bool,
    verify_no_gru: bool = False,
    do_constant_folding: bool = True,
    simplify: bool = False,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": int(opset),
        "do_constant_folding": bool(do_constant_folding),
    }
    if dynamo:
        kwargs["dynamo"] = True
        kwargs["external_data"] = False
    try:
        torch.onnx.export(module, args, str(path), **kwargs)
    except TypeError:
        kwargs.pop("dynamo", None)
        kwargs.pop("external_data", None)
        torch.onnx.export(module, args, str(path), **kwargs)
    simplified = _simplify_onnx(path) if simplify else False
    if verify_no_gru:
        gru_nodes = _count_onnx_ops(path, "GRU")
        if gru_nodes is None:
            print("[export_grpo_onnx] warning: onnx package not available; cannot verify GRU node count")
        elif gru_nodes:
            raise RuntimeError(
                f"{path} still contains {gru_nodes} ONNX GRU node(s). "
                "Re-export without --dynamo or update the wrapper decomposition."
            )
        else:
            print(f"[export_grpo_onnx] verified no ONNX GRU nodes: {path}")
    return bool(simplified)


class FastEnhancerSpecFrameOnnxWrapper(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, spec_noisy, *model_cache):
        mask, cache_out = self.core.model_forward(spec_noisy, *model_cache)
        return (mask, *cache_out)


class ULUNASSpecFrameOnnxWrapper(nn.Module):
    """Official-style UL-UNAS stream wrapper for spec-frame ONNX export."""

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
    def init_caches(cls, batch_size=1, device=None, dtype=torch.float32):
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
        return y, inp[:, :, 1:, :]

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
        return (intra_out + inter_x).permute(0, 3, 1, 2), inter_cache

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


def _flatten_lisen_state(state: dict[str, Any]):
    items = [
        ("prev_phase", state["prev_phase"]),
        ("istft_cache", state["istft_cache"]),
        ("enc_conv_2", state["encoder_cache"]["enc_conv_2"]),
        ("enc_conv_3", state["encoder_cache"]["enc_conv_3"]),
        ("enc_conv_4", state["encoder_cache"]["enc_conv_4"]),
    ]
    for idx, block_state in enumerate(state["block_states"]):
        items.append((f"block_{idx}_inter_hidden", block_state["inter_hidden"]))
        items.append((f"block_{idx}_conv_glu_cache", block_state["conv_glu_cache"]))
    items.append(("mask_conv", state["decoder_cache"]["mask_conv"]))
    return items


def _flatten_lisen_neural_state(state: dict[str, Any]):
    items = [
        ("enc_conv_2", state["encoder_cache"]["enc_conv_2"]),
        ("enc_conv_3", state["encoder_cache"]["enc_conv_3"]),
        ("enc_conv_4", state["encoder_cache"]["enc_conv_4"]),
    ]
    for idx, block_state in enumerate(state["block_states"]):
        items.append((f"block_{idx}_inter_hidden", block_state["inter_hidden"]))
        items.append((f"block_{idx}_conv_glu_cache", block_state["conv_glu_cache"]))
    items.append(("mask_conv", state["decoder_cache"]["mask_conv"]))
    return items


class LiSenMaskFrameOnnxWrapper(nn.Module):
    def __init__(self, adapter, cache_names):
        super().__init__()
        self.adapter = adapter
        self.expert = adapter.expert
        self.cache_names = list(cache_names)

    def forward(self, features, *caches):
        cache_map = {name: tensor for name, tensor in zip(self.cache_names, caches)}
        block_states = []
        block_idx = 0
        while f"block_{block_idx}_inter_hidden" in cache_map:
            block_states.append({
                "inter_hidden": cache_map[f"block_{block_idx}_inter_hidden"],
                "conv_glu_cache": cache_map[f"block_{block_idx}_conv_glu_cache"],
            })
            block_idx += 1
        state = {
            "encoder_cache": {
                "enc_conv_2": cache_map["enc_conv_2"],
                "enc_conv_3": cache_map["enc_conv_3"],
                "enc_conv_4": cache_map["enc_conv_4"],
            },
            "block_states": block_states,
            "decoder_cache": {"mask_conv": cache_map["mask_conv"]},
        }
        core = self.adapter._core()
        encoder_out_list, state["encoder_cache"] = self.adapter._stream_encoder(core.encoder, features, state["encoder_cache"])
        x = encoder_out_list[-1]
        for idx, block in enumerate(core.blocks):
            x, state["block_states"][idx] = self.adapter._stream_dpr(block, x, state["block_states"][idx])
        mask, state["decoder_cache"] = self.adapter._stream_decoder(core.decoder, x, encoder_out_list, state["decoder_cache"])
        flat = _flatten_lisen_neural_state(state)
        return (mask, *[tensor for _, tensor in flat])


def _export_fastenhancer(
    model,
    idx: int,
    out_dir: Path,
    opset: int,
    dynamo: bool,
    device: torch.device,
    decompose_gru: bool,
):
    expert = model.experts[idx]
    core = model._fastenhancer_core(expert)
    if core is None:
        raise RuntimeError(f"Expert {expert.name} has no FastEnhancer core.")
    stft = core.stft
    n_fft = int(stft.n_fft)
    hop_size = int(stft.hop_size)
    freq_bins = n_fft // 2 + 1
    if bool(getattr(stft, "discard_last_freq_bin", False)):
        freq_bins -= 1
    model_cache = [
        item.to(device=device, dtype=torch.float32)
        for item in core.initialize_cache(torch.zeros(1, hop_size, device=device, dtype=torch.float32))
    ]
    wrapper = FastEnhancerSpecFrameOnnxWrapper(core).to(device).eval()
    _prepare_export_module(wrapper, decompose_gru, expert.name)
    spec_noisy = torch.zeros(1, freq_bins, 1, 2, device=device)
    inputs = (spec_noisy, *model_cache)
    cache_names = [f"model_cache_{i}" for i in range(len(model_cache))]
    input_names = ["spec_noisy"] + cache_names
    output_names = ["mask"] + [f"{name}_out" for name in cache_names]
    path = out_dir / f"{expert.name}.onnx"
    simplified = _export_onnx(
        wrapper,
        inputs,
        path,
        input_names,
        output_names,
        opset=opset,
        dynamo=dynamo,
        verify_no_gru=decompose_gru,
    )
    return {
        "name": expert.name,
        "path": str(path.name),
        "protocol": "fastenhancer_stft",
        "true_cache": True,
        "input": "spec_noisy",
        "cache_inputs": cache_names,
        "cache_outputs": [f"{name}_out" for name in cache_names],
        "n_fft": n_fft,
        "win_length": int(getattr(stft, "win_size", n_fft)),
        "compression": float(getattr(stft, "compression", getattr(core, "input_compression", 0.3))),
        "eps": float(getattr(stft, "eps", 1.0e-5)),
        "discard_last_freq_bin": bool(getattr(stft, "discard_last_freq_bin", True)),
        "frame_samples": int(model.stream_frame_samples),
        "hop_samples": hop_size,
        "gru_export": "decomposed" if decompose_gru else "onnx_gru",
    }


def _export_ulunas(
    model,
    idx: int,
    out_dir: Path,
    opset: int,
    dynamo: bool,
    device: torch.device,
    decompose_gru: bool,
    official_export: bool = True,
    simplify: bool = True,
):
    expert = model.experts[idx]
    core = getattr(getattr(expert, "model", None), "enhancer", None)
    if core is None:
        raise RuntimeError(f"Expert {expert.name} has no `model.enhancer`; cannot export UL-UNAS stream ONNX.")
    wrapper = ULUNASSpecFrameOnnxWrapper(core).to(device).eval()
    ulunas_decompose_gru = bool(decompose_gru)
    _prepare_export_module(wrapper, ulunas_decompose_gru, expert.name)
    n_fft = int(wrapper.n_fft)
    hop_size = int(wrapper.hop_len)
    conv_cache, tfa_cache, inter_cache = wrapper.init_caches(batch_size=1, device=device, dtype=torch.float32)
    inputs = (
        torch.zeros(1, n_fft // 2 + 1, 1, 2, device=device),
        conv_cache,
        tfa_cache,
        inter_cache,
    )
    input_names = ["mix", "conv_cache", "tfa_cache", "inter_cache"]
    output_names = ["enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"]
    path = out_dir / f"{expert.name}.onnx"
    export_opset = 11 if official_export else opset
    export_dynamo = False if official_export else dynamo
    do_constant_folding = False if official_export else True
    if official_export:
        print(
            f"[export_grpo_onnx] exporting {expert.name} with official UL-UNAS ONNX style: "
            "opset=11, legacy exporter, do_constant_folding=False, onnxsim=simplify"
        )
    simplified = _export_onnx(
        wrapper,
        inputs,
        path,
        input_names,
        output_names,
        opset=export_opset,
        dynamo=export_dynamo,
        verify_no_gru=ulunas_decompose_gru,
        do_constant_folding=do_constant_folding,
        simplify=bool(simplify and official_export),
    )
    return {
        "name": expert.name,
        "path": str(path.name),
        "protocol": "ulunas_stft",
        "true_cache": True,
        "input": "mix",
        "cache_inputs": input_names[1:],
        "cache_outputs": output_names[1:],
        "n_fft": n_fft,
        "win_length": n_fft,
        "frame_samples": int(model.stream_frame_samples),
        "hop_samples": hop_size,
        "export_style": "official_ulunas_onnx" if official_export else "mos_se",
        "opset": int(export_opset),
        "constant_folding": bool(do_constant_folding),
        "onnxsim_simplified": bool(simplified),
        "gru_export": "decomposed" if ulunas_decompose_gru else "onnx_gru",
    }


def _export_lisen(
    model,
    idx: int,
    out_dir: Path,
    opset: int,
    dynamo: bool,
    device: torch.device,
    decompose_gru: bool,
):
    expert = model.experts[idx]
    adapter = LiSenStreamingExpertAdapter(model, idx, expert)
    state = adapter.init_state(device=device, batch_size=1, dtype=torch.float32)
    dummy_frame = torch.zeros(1, int(state["n_fft"]), device=device)
    with torch.no_grad():
        _, state = adapter.stream_step(dummy_frame, state)
    flat = _flatten_lisen_neural_state(state)
    cache_names = [name for name, _ in flat]
    wrapper = LiSenMaskFrameOnnxWrapper(
        adapter,
        cache_names=cache_names,
    ).to(device).eval()
    _prepare_export_module(wrapper, decompose_gru, expert.name)
    features = torch.zeros(1, 3, 1, int(state["n_fft"]) // 2 + 1, device=device)
    inputs = (features, *[tensor for _, tensor in flat])
    input_names = ["features"] + cache_names
    output_names = ["mask"] + [f"{name}_out" for name in cache_names]
    path = out_dir / f"{expert.name}.onnx"
    _export_onnx(
        wrapper,
        inputs,
        path,
        input_names,
        output_names,
        opset=opset,
        dynamo=dynamo,
        verify_no_gru=decompose_gru,
    )
    return {
        "name": expert.name,
        "path": str(path.name),
        "protocol": "lisen_stft",
        "true_cache": True,
        "input": "features",
        "cache_inputs": cache_names,
        "cache_outputs": [f"{name}_out" for name in cache_names],
        "n_fft": int(state["n_fft"]),
        "win_length": int(state["n_fft"]),
        "compress_factor": float(getattr(expert.model, "compress_factor", 0.3)),
        "frame_samples": int(model.stream_frame_samples),
        "hop_samples": int(model.stream_hop_samples),
        "gru_export": "decomposed" if decompose_gru else "onnx_gru",
    }


def _export_expert(
    model,
    idx: int,
    experts_dir: Path,
    opset: int,
    dynamo: bool,
    device: torch.device,
    decompose_gru: bool,
    ulunas_official_export: bool = True,
    ulunas_simplify: bool = True,
):
    expert = model.experts[idx]
    class_name = expert.model_class.__name__
    print(f"[export_grpo_onnx] exporting expert[{idx}] {expert.name}: {class_name}")
    if class_name == "FastEnhancer":
        return _export_fastenhancer(model, idx, experts_dir, opset, dynamo, device, decompose_gru)
    if class_name == "ULUNAS":
        return _export_ulunas(
            model,
            idx,
            experts_dir,
            opset,
            dynamo,
            device,
            decompose_gru,
            official_export=ulunas_official_export,
            simplify=ulunas_simplify,
        )
    if class_name == "LiSen":
        return _export_lisen(model, idx, experts_dir, opset, dynamo, device, decompose_gru)
    raise NotImplementedError(
        f"No native ONNX stream exporter for expert {expert.name} ({class_name}). "
        "Add a model.init_stream_state/model.stream_step exporter or provide a manifest entry manually."
    )


def _apply_expert_provider(spec: dict[str, Any], idx: int, expert_devices: list[Any]) -> dict[str, Any]:
    if not expert_devices:
        return spec
    provider_fields = _onnx_provider_fields(expert_devices[idx % len(expert_devices)])
    if provider_fields:
        spec = dict(spec)
        spec.update(provider_fields)
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="FrozenExpertRouterGRPO YAML config.")
    parser.add_argument("--out", required=True, help="Directory to write ONNX files and manifest.json.")
    parser.add_argument("--ckpt", default=None, help="Optional Lightning checkpoint with trained router weights.")
    parser.add_argument("--device", default="cpu", help="Export device, e.g. cpu or cuda:0.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamo", action="store_true", help="Use torch.onnx dynamo exporter when available.")
    parser.add_argument(
        "--decompose-gru",
        action="store_true",
        help="Export nn.GRU as primitive ops instead of ONNX GRU nodes for parallel CUDA execution.",
    )
    parser.add_argument(
        "--ulunas-native-gru",
        action="store_true",
        help=(
            "Keep native ONNX GRU nodes for UL-UNAS. This is the default unless --decompose-gru is set; "
            "the flag is provided for command clarity."
        ),
    )
    parser.add_argument(
        "--ulunas-official-export",
        dest="ulunas_official_export",
        action="store_true",
        default=True,
        help="Use the official UL-UNAS stream ONNX export style: opset 11, legacy exporter, no constant folding, onnxsim.",
    )
    parser.add_argument(
        "--no-ulunas-official-export",
        dest="ulunas_official_export",
        action="store_false",
        help="Export UL-UNAS with the generic MOS-SE ONNX exporter settings.",
    )
    parser.add_argument(
        "--no-ulunas-simplify",
        dest="ulunas_simplify",
        action="store_false",
        default=True,
        help="Disable onnxsim simplification for the official UL-UNAS export path.",
    )
    parser.add_argument(
        "--export-router",
        action="store_true",
        help="Also export router_features.onnx for the legacy ONNX-router path. Default is PyTorch router.",
    )
    parser.add_argument("--keep-going", action="store_true", help="Write successful experts even if one export fails.")
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dotlist overrides.")
    args = parser.parse_args()
    if args.ulunas_native_gru and args.decompose_gru:
        raise RuntimeError("--ulunas-native-gru conflicts with --decompose-gru.")

    conf_path = _resolve_required_conf(args.conf)
    conf = OmegaConf.load(conf_path)
    if args.overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.overrides))
    router_device, expert_devices = _device_map_for_export(conf)
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

    model = get_model(conf)
    _load_router_checkpoint(model, args.ckpt)
    model.to(device)
    if hasattr(model, "_router_device_override"):
        model._router_device_override = device
    if hasattr(model, "_expert_device_overrides"):
        model._expert_device_overrides = [device for _ in range(len(getattr(model, "experts", [])))]
    model.eval()

    out_dir = _resolve(args.out)
    experts_dir = out_dir / "experts"
    out_dir.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)

    router_manifest = {
        "path": None,
        "protocol": "pytorch",
        "exported": False,
    }
    if args.export_router:
        router_path = out_dir / "router_features.onnx"
        model.stream_onnx_router_path = str(router_path)
        model._export_router_onnx_snapshot(router_path)
        router_manifest = {
            "path": router_path.name,
            "protocol": "features",
            "feature_input": "features",
            "exported": True,
            **_onnx_provider_fields(router_device),
        }
    else:
        print("[export_grpo_onnx] skipping router ONNX export; runtime will use PyTorch router by default")

    expert_specs = []
    errors = []
    for idx in range(len(model.experts)):
        try:
            spec = _export_expert(
                model,
                idx,
                experts_dir,
                args.opset,
                args.dynamo,
                device,
                args.decompose_gru,
                ulunas_official_export=bool(args.ulunas_official_export),
                ulunas_simplify=bool(args.ulunas_simplify),
            )
            expert_specs.append(_apply_expert_provider(spec, idx, expert_devices))
        except Exception as exc:
            errors.append((idx, model.experts[idx].name, repr(exc)))
            print(f"[export_grpo_onnx] failed expert[{idx}] {model.experts[idx].name}: {exc}")
            if not args.keep_going:
                raise

    manifest = {
        "version": 1,
        "sample_rate": int(model.sample_rate),
        "frame_samples": int(model.stream_frame_samples),
        "hop_samples": int(model.stream_hop_samples),
        "gru_export": "decomposed" if args.decompose_gru else "onnx_gru",
        "stft": _plain_conf(conf.get("stft", {})),
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"],
        "router": router_manifest,
        "experts": [
            {**spec, "path": str(Path("experts") / spec["path"])}
            for spec in expert_specs
        ],
        "export_errors": errors,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    print(f"[export_grpo_onnx] wrote manifest: {manifest_path}")
    print("[export_grpo_onnx] enable with:")
    print(f"  router_grpo.inference_branch.runtime=onnx")
    print(f"  router_grpo.inference_branch.onnx.manifest={manifest_path}")
    print(f"  router_grpo.inference_branch.onnx.use_onnx_router={str(bool(args.export_router)).lower()}")


if __name__ == "__main__":
    main()
