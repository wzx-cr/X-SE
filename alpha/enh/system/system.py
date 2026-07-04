import random

import numpy as np
import torch
from omegaconf import OmegaConf

import modules.model.arch as model_arch
from modules.stft import STFT
from modules.system import BaseModule
from modules.utils import metrics
from modules.utils.common import (
    freeze_model,
    merge_dicts,
    refine_state_dict,
    resolve_path,
    torch_float32,
)
from modules.utils.logging import logger
from .. import loss_module, models


def _resolve_model_class(model_name):
    if hasattr(models, model_name):
        return getattr(models, model_name)
    raise AttributeError(f"module alpha.enh.models has no attribute `{model_name}`")


class BaseSE(BaseModule):
    """Minimal speech-enhancement base used by MoE-GRPO."""

    def __init__(self, conf):
        super().__init__(conf)
        self.stft = STFT(self.stft_conf)
        self.model_class = _resolve_model_class(conf["model"]["name"])
        self.resolve_model_param()
        self.model_param = self.conf["model"].get("param", None)
        self.mix_param = conf["data"].get("mix_param", None)
        self.loss_name = conf["loss"].get("name")
        self.loss = self.get_loss(conf["loss"])
        self.valid_loss = self.init_valid_loss(conf.get("valid_loss", None))
        self.feature = None
        self.deepfilter = self.init_deepfilter(conf)
        self.model_conf = OmegaConf.to_container(self.conf["model"], resolve=True)
        self.check_nan = conf["system"].get("check_nan", False)
        self.rms_norm = conf["system"].get("rms_norm", False)

    def resolve_model_param(self):
        model_param = self.conf["model"].get("param", None)
        if model_param is not None:
            self.conf["model"]["param"] = getattr(model_arch, model_param, None)

    def get_loss(self, loss_conf):
        name = loss_conf.get("name")
        if not name:
            return None
        if hasattr(loss_module, name):
            return getattr(loss_module, name)(loss_conf)
        if hasattr(torch.nn, name):
            return getattr(torch.nn, name)()
        raise ValueError(f"Loss name error: {name}")

    def init_valid_loss(self, loss_conf):
        return self.get_loss(loss_conf) if loss_conf else None

    def init_model(self, model_conf):
        model_path = model_conf.get("init")
        if not model_path:
            return
        model_path = resolve_path(model_path)
        logger.info(f"Model initialize with: {model_path}")
        model_dict = torch.load(model_path, map_location="cpu")
        if "state_dict" in model_dict:
            refined_state_dict = refine_state_dict(model_dict["state_dict"])
            model_dict = refined_state_dict if refined_state_dict else model_dict["state_dict"]
        missing, unexpected = self.model.load_state_dict(model_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                f"Loaded pretrained weights with missing keys: {missing}, unexpected keys: {unexpected}"
            )
        else:
            logger.info("Pretrained weights loaded successfully with strict match.")

    def init_deepfilter(self, conf):
        deepfilter = conf.get("deepfilter", None)
        if deepfilter:
            assert self.mask is True
            deepfilter.update(
                {
                    "n_freq": conf["feature"]["n_feature"]
                    if self.feature
                    else conf["stft"]["n_fft"] // 2 + 1
                }
            )
        return deepfilter

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def denoise(self, y, chunk=-1):
        y = torch_float32(y, self.device)
        if chunk > 0:
            counter = 0
            chunk_samples = int(round(self.sample_rate * float(chunk)))
            n_chunk = int(np.ceil(len(y) / chunk_samples))
            est_wav = torch.zeros_like(y)
            for idx in range(n_chunk):
                start = idx * chunk_samples
                end = None if idx == n_chunk - 1 else (idx + 1) * chunk_samples
                y_chunk = y[start:end]
                with torch.no_grad():
                    est_wav_chunk = self.forward(
                        self.padding(y_chunk.unsqueeze(0)), train=False
                    ).squeeze()
                est_wav_chunk = est_wav_chunk[0 : len(y_chunk)]
                est_wav[start:end] = est_wav_chunk
                counter += len(y_chunk)
            assert len(y) == counter
        else:
            with torch.no_grad():
                est_wav = self.forward(self.padding(y.unsqueeze(0)), train=False).squeeze()
            est_wav = est_wav[0 : len(y)]
        return est_wav

    @property
    def snr(self):
        return random.randint(self.mix_param.snr_lower, self.mix_param.snr_upper)

    def _calc_stft(self, x):
        x_stft = self.stft.apply_stft(x)
        x_mag = torch.abs(x_stft)
        x_phase = torch.angle(x_stft)
        return x_mag, x_phase

    def _calc_istft(self, x_mag, x_phase):
        x_stft = x_mag * torch.exp(1.0j * x_phase)
        return self.stft.apply_istft(x_stft)

    def test_stft(self, x):
        x_mag, x_phase = self._calc_stft(x)
        x_r = self._calc_istft(x_mag, x_phase)
        score = metrics.eval(x, x_r, self.sample_rate)
        print(score)


class UniSE(BaseSE):
    """Compatibility parent for SlidingWindowGRPO."""

    pass
