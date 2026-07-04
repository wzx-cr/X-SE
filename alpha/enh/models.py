import torch
from torch import nn
from torchaudio.functional import melscale_fbanks

from modules.blocks.fastenhancer import create_fastenhancer_model
from modules.blocks.lisen import (
    DPR as LiSenDPR,
    Encoder as LiSenEncoder,
    MaskDecoder as LiSenMaskDecoder,
)
from modules.blocks.ulunas import ULUNAS as ULUNASCore
from modules.utils.common import merge_dicts


class ModelBase(torch.nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        self.model_conf = model_conf

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class SpectralStatsRouter(ModelBase):
    """
    Lightweight expert router used by FrozenExpertRouterGRPO.

    It maps noisy-speech spectral statistics to a probability over frozen
    enhancement experts. The router is intentionally small so GRPO can adapt it
    online without touching expert parameters.
    """

    def __init__(self, model_conf):
        super().__init__(model_conf)
        default_conf = {
            "num_experts": 2,
            "input_dim": 9,
            "hidden_dims": [32, 32],
            "dropout": 0.0,
            "temperature": 1.0,
            "init_logits": None,
            "fixed_logits": None,
        }
        self.model_conf = merge_dicts(default_conf, model_conf)
        self.num_experts = int(self.model_conf["num_experts"])
        self.input_dim = int(self.model_conf["input_dim"])
        self.temperature = float(self.model_conf.get("temperature", 1.0))

        layers = [nn.LayerNorm(self.input_dim)]
        in_dim = self.input_dim
        for hidden_dim in self.model_conf.get("hidden_dims", []):
            hidden_dim = int(hidden_dim)
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            dropout = float(self.model_conf.get("dropout", 0.0))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.num_experts))
        self.net = nn.Sequential(*layers)

        init_logits = self.model_conf.get("init_logits")
        if init_logits is None:
            init_logits = torch.zeros(self.num_experts)
        else:
            init_logits = torch.as_tensor(init_logits, dtype=torch.float32)
            if init_logits.numel() != self.num_experts:
                raise ValueError(
                    f"init_logits has {init_logits.numel()} values, expected {self.num_experts}."
                )
        self.logit_bias = nn.Parameter(init_logits.float().clone())

        fixed_logits = self.model_conf.get("fixed_logits")
        if fixed_logits is None:
            self.fixed_logits = None
        else:
            fixed_logits = torch.as_tensor(fixed_logits, dtype=torch.float32)
            if fixed_logits.numel() != self.num_experts:
                raise ValueError(
                    f"fixed_logits has {fixed_logits.numel()} values, expected {self.num_experts}."
                )
            self.register_buffer("fixed_logits", fixed_logits.float().clone())

    @staticmethod
    def _band_log_energy(mag, start, end):
        if end <= start:
            return torch.zeros(mag.shape[0], device=mag.device, dtype=mag.dtype)
        band = mag[:, :, start:end, :]
        return torch.log(band.pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-8))

    def extract_features(self, noisy_wav, noisy_spec):
        if noisy_wav.ndim == 3 and noisy_wav.shape[1] == 1:
            noisy_wav = noisy_wav[:, 0, :]
        elif noisy_wav.ndim == 1:
            noisy_wav = noisy_wav.unsqueeze(0)
        elif noisy_wav.ndim > 2:
            noisy_wav = noisy_wav.mean(dim=1)

        mag = torch.abs(noisy_spec).float().clamp_min(1e-8)
        wav = noisy_wav.float()
        rms = torch.log(wav.pow(2).mean(dim=-1).clamp_min(1e-8))
        peak = wav.abs().amax(dim=-1).clamp_max(10.0)
        if wav.shape[-1] > 1:
            zcr = ((wav[:, 1:] * wav[:, :-1]) < 0).float().mean(dim=-1)
        else:
            zcr = torch.zeros_like(rms)

        log_mag = torch.log(mag)
        log_mag_mean = log_mag.mean(dim=(1, 2, 3))
        log_mag_std = log_mag.flatten(1).std(dim=-1, unbiased=False)

        n_freq = mag.shape[-2]
        f1 = max(1, n_freq // 3)
        f2 = max(f1 + 1, (2 * n_freq) // 3)
        low = self._band_log_energy(mag, 0, f1)
        mid = self._band_log_energy(mag, f1, min(f2, n_freq))
        high = self._band_log_energy(mag, min(f2, n_freq), n_freq)

        flatness = (
            torch.exp(torch.log(mag).mean(dim=(1, 2, 3)))
            / mag.mean(dim=(1, 2, 3)).clamp_min(1e-8)
        )
        feats = torch.stack(
            [rms, peak, zcr, log_mag_mean, log_mag_std, low, mid, high, flatness],
            dim=-1,
        )
        return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, noisy_wav, noisy_spec, return_logits=False):
        if self.fixed_logits is not None:
            batch_size = noisy_wav.shape[0] if noisy_wav.ndim > 1 else 1
            logits = self.fixed_logits.to(device=noisy_wav.device, dtype=noisy_wav.dtype)
            logits = logits.unsqueeze(0).expand(batch_size, -1)
        else:
            feats = self.extract_features(noisy_wav, noisy_spec)
            logits = self.net(feats) + self.logit_bias
        weights = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        if return_logits:
            return weights, logits
        return weights


class ULUNAS(ModelBase):
    """UL-UNAS waveform enhancer wrapper."""

    def __init__(self, model_conf):
        super().__init__(model_conf)
        default_conf = {
            "sample_rate": 16000,
            "n_fft": 512,
            "hop_length": 256,
            "win_length": 512,
            "high_lim": None,
            "erb_low": 65,
            "erb_high": 64,
            "types": [0, 2, 1, 2, 1],
            "strides": [2, 2, 1, 1, 1],
            "groups": [1, 2, 2, 2, 2],
            "channels": [12, 24, 24, 32, 16],
            "kernels": [(3, 3), (2, 3), (2, 3), (1, 5), (1, 5)],
            "widths": [65, 33, 33, 33, 33],
            "mono_mode": "mean",
            "return_dict": False,
        }
        self.model_conf = merge_dicts(default_conf, model_conf)
        hop_length = self.model_conf.get("hop_len", self.model_conf["hop_length"])
        win_length = self.model_conf.get("win_len", self.model_conf["win_length"])
        self.mono_mode = self.model_conf["mono_mode"]
        self.return_dict = bool(self.model_conf.get("return_dict", False))
        self.enhancer = ULUNASCore(
            n_fft=self.model_conf["n_fft"],
            hop_len=hop_length,
            win_len=win_length,
            sample_rate=self.model_conf["sample_rate"],
            high_lim=self.model_conf["high_lim"],
            erb_low=self.model_conf["erb_low"],
            erb_high=self.model_conf["erb_high"],
            types=self.model_conf["types"],
            strides=self.model_conf["strides"],
            groups=self.model_conf["groups"],
            channels=self.model_conf["channels"],
            kernels=self.model_conf["kernels"],
            widths=self.model_conf["widths"],
        )

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        if not any(k.startswith("enhancer.") for k in state_dict.keys()):
            core_keys = set(self.enhancer.state_dict().keys())
            overlap = sum(1 for key in state_dict.keys() if key in core_keys)
            if overlap > 0:
                state_dict = {f"enhancer.{key}": value for key, value in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    def _to_mono(self, noisy_wav):
        if noisy_wav.ndim == 1:
            return noisy_wav.unsqueeze(0)
        if noisy_wav.ndim == 2:
            return noisy_wav
        if noisy_wav.ndim == 3:
            if noisy_wav.shape[1] == 1:
                return noisy_wav[:, 0, :]
            if self.mono_mode == "first":
                return noisy_wav[:, 0, :]
            if self.mono_mode != "mean":
                raise ValueError(f"Unsupported mono_mode: {self.mono_mode}")
            return noisy_wav.mean(dim=1)
        raise ValueError(f"ULUNAS expects [B,T] or [B,C,T] waveform input, got {noisy_wav.shape}")

    def forward(self, noisy_wav):
        est = self.enhancer(self._to_mono(noisy_wav).contiguous())
        if self.return_dict:
            return {"est": est}
        return est


class FastEnhancer(ModelBase):
    """FastEnhancer waveform enhancer wrapper."""

    def __init__(self, model_conf):
        super().__init__(model_conf)
        default_conf = {
            "size": "b",
            "config_group": "fastenhancer",
            "mono_mode": "mean",
            "return_dict": False,
            "return_spec": False,
            "flatten_parameters": True,
        }
        self.model_conf = merge_dicts(default_conf, model_conf)
        self.mono_mode = self.model_conf["mono_mode"]
        self.return_dict = bool(self.model_conf.get("return_dict", False))
        self.return_spec = bool(self.model_conf.get("return_spec", False))
        self.remove_weight_reparameterizations = bool(
            self.model_conf.get("remove_weight_reparameterizations", False)
        )
        self.enhancer = create_fastenhancer_model(self.model_conf)

    @staticmethod
    def _strip_prefix(state_dict, prefix):
        if any(key.startswith(prefix) for key in state_dict.keys()):
            return {
                (key[len(prefix):] if key.startswith(prefix) else key): value
                for key, value in state_dict.items()
            }
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        for prefix in ("module.", "_module.", "model."):
            state_dict = self._strip_prefix(state_dict, prefix)

        if not any(key.startswith("enhancer.") for key in state_dict.keys()):
            core_keys = set(self.enhancer.state_dict().keys())
            overlap = sum(1 for key in state_dict.keys() if key in core_keys)
            if overlap > 0:
                state_dict = {f"enhancer.{key}": value for key, value in state_dict.items()}
        result = super().load_state_dict(state_dict, strict=strict)
        if self.remove_weight_reparameterizations:
            if not hasattr(self.enhancer, "remove_weight_reparameterizations"):
                raise AttributeError("FastEnhancer core does not support remove_weight_reparameterizations().")
            self.enhancer.remove_weight_reparameterizations()
        return result

    def _to_mono(self, noisy_wav):
        if noisy_wav.ndim == 1:
            return noisy_wav.unsqueeze(0)
        if noisy_wav.ndim == 2:
            return noisy_wav
        if noisy_wav.ndim == 3:
            if noisy_wav.shape[1] == 1:
                return noisy_wav[:, 0, :]
            if self.mono_mode == "first":
                return noisy_wav[:, 0, :]
            if self.mono_mode != "mean":
                raise ValueError(f"Unsupported mono_mode: {self.mono_mode}")
            return noisy_wav.mean(dim=1)
        raise ValueError(f"FastEnhancer expects [B,T] or [B,C,T] waveform input, got {noisy_wav.shape}")

    def forward(self, noisy_wav):
        wav_hat, spec_hat = self.enhancer(self._to_mono(noisy_wav).contiguous())
        if self.return_dict or self.return_spec:
            results = {"est": wav_hat}
            if self.return_spec:
                results["est_spec"] = spec_hat
            return results
        return wav_hat


class LiSen(ModelBase):
    """LiSenNet waveform enhancer wrapper."""

    def __init__(self, model_conf):
        super().__init__(model_conf)
        default_conf = {
            "num_channels": 16,
            "n_blocks": 2,
            "n_fft": 512,
            "hop_length": 256,
            "compress_factor": 0.3,
            "in_channels": 3,
            "decoder_out_channel": 2,
            "decoder_beta": 1.0,
            "dropout_p": 0.1,
        }
        self.model_conf = merge_dicts(default_conf, model_conf)
        self.n_fft = self.model_conf["n_fft"]
        self.n_freqs = self.n_fft // 2 + 1
        self.hop_length = self.model_conf["hop_length"]
        self.compress_factor = self.model_conf["compress_factor"]
        self.encoder = LiSenEncoder(
            in_channels=self.model_conf["in_channels"],
            num_channels=self.model_conf["num_channels"],
        )
        hidden_dim = (self.model_conf["num_channels"] // 2) * 3
        self.blocks = nn.Sequential(
            *[
                LiSenDPR(
                    emb_dim=self.model_conf["num_channels"],
                    hidden_dim=hidden_dim,
                    n_freqs=self.n_freqs // (2 ** 3),
                    dropout_p=self.model_conf["dropout_p"],
                )
                for _ in range(self.model_conf["n_blocks"])
            ]
        )
        self.decoder = LiSenMaskDecoder(
            self.n_freqs,
            num_channels=self.model_conf["num_channels"],
            out_channel=self.model_conf["decoder_out_channel"],
            beta=self.model_conf["decoder_beta"],
        )

    def apply_stft(self, x, return_complex=True):
        assert x.ndim == 2
        window = torch.hann_window(self.n_fft, device=x.device)
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            onesided=True,
            return_complex=return_complex,
        ).transpose(1, 2)

    def apply_istft(self, x, length=None):
        assert x.ndim == 3
        window = torch.hann_window(self.n_fft, device=x.device)
        return torch.istft(
            x.transpose(1, 2),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            onesided=True,
            length=length,
            return_complex=False,
        )

    def power_compress(self, x):
        mag = torch.abs(x) ** self.compress_factor
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def power_uncompress(self, x):
        mag = torch.abs(x) ** (1.0 / self.compress_factor)
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def mel_scale(self, mag, sr=16000, f_min=0.0, f_max=8000.0, n_mels=64):
        if not hasattr(self, "fb"):
            fb = melscale_fbanks(
                n_freqs=self.n_freqs,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                sample_rate=sr,
            )
            setattr(self, "fb", fb.to(mag.device))
        mag = mag ** (1 / self.compress_factor)
        mel = mag @ self.fb
        return mel ** self.compress_factor

    @staticmethod
    def cal_gd(x):
        batch, time, freq = x.size()
        x_gd = torch.diff(x, dim=2, prepend=torch.zeros(batch, time, 1, device=x.device))
        return torch.atan2(x_gd.sin(), x_gd.cos())

    def cal_ifd(self, x):
        batch, time, freq = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(batch, 1, freq, device=x.device))
        freq_axis = torch.arange(freq, device=x.device)
        x_ifd = x_if - 2 * torch.pi * (self.hop_length / self.n_fft) * freq_axis[None, None, :]
        return torch.atan2(x_ifd.sin(), x_ifd.cos())

    def griffinlim(self, mag, pha=None, length=None, n_iter=2, momentum=0.99):
        mag = mag.detach()
        mag = mag ** (1.0 / self.compress_factor)
        assert 0 <= momentum < 1
        momentum = momentum / (1 + momentum)
        if pha is None:
            pha = torch.rand(mag.size(), dtype=mag.dtype, device=mag.device)
        tprev = torch.tensor(0.0, dtype=mag.dtype, device=mag.device)
        for _ in range(n_iter):
            inverse = self.apply_istft(
                torch.complex(mag * pha.cos(), mag * pha.sin()),
                length=length,
            )
            rebuilt = self.apply_stft(inverse)
            pha = rebuilt - tprev.mul_(momentum)
            pha = pha.angle()
            tprev = rebuilt
        return pha

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src
        src_spec = self.power_compress(self.apply_stft(src))
        src_mag = src_spec.abs()
        src_pha = src_spec.angle()
        src_gd = self.cal_gd(src_pha)
        src_ifd = self.cal_ifd(src_pha)

        tgt_spec = self.power_compress(self.apply_stft(tgt))
        tgt_mag = tgt_spec.abs()

        x = torch.stack([src_mag, src_gd / torch.pi, src_ifd / torch.pi], dim=1)
        encoder_out_list = self.encoder(x)
        x = self.blocks(encoder_out_list[-1])
        x = self.decoder(x, encoder_out_list)

        est_mag = (x[:, 0] + 1e-8) * src_mag + (x[:, 1] + 1e-8) * src_mag
        est_pha = self.griffinlim(est_mag.detach(), src_pha, tgt.size(-1))
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
        est = self.apply_istft(self.power_uncompress(est_spec), length=tgt.size(-1))

        return {
            "tgt": tgt,
            "tgt_spec": tgt_spec,
            "tgt_mag": tgt_mag,
            "est": est,
            "est_spec": est_spec,
            "est_mag": est_mag,
        }
