import warnings
from typing import Optional, Tuple, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class STFT(nn.Module):
    '''Short Time Fourier Transform
    forward(x):
        x: [B, T_wav] or [B, 1, T_wav]
        output: [B, n_fft//2+1, T_spec, 2]   (magnitude = False)
        output: [B, n_fft//2+1, T_spec]      (magnitude = True)
    inverse(x):
        x: [B,  n_fft//2+1, T_spec, 2]
        output: [B, T_wav]
    '''

    __constants__ = ["normalize", "center", "magnitude", "n_fft",
                     "hop_size", "win_size", "padding", "clip", "pad_mode"]
    __annotations__ = {'window': Optional[Tensor]}

    def __init__(
        self, n_fft: int, hop_size: int, win_size: Optional[int] = None,
        center: bool = True, magnitude: bool = False,
        win_type: Optional[str] = "hann",
        window: Optional[Tensor] = None, normalized: bool = False,
        pad_mode: str = "reflect",
        device=None, dtype=None
    ):
        super().__init__()
        self.normalized = normalized
        self.center = center
        self.magnitude = magnitude
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.padding = 0 if center else (n_fft + 1 - hop_size) // 2
        self.clip = (hop_size % 2 == 1)
        self.pad_mode = pad_mode
        if win_size is None:
            win_size = n_fft
        
        if window is not None:
            win_size = window.size(-1)
        elif win_type is None:
            window = torch.ones(win_size, device=device, dtype=dtype)
        elif win_type == "povey":
            window = torch.hann_window(
                win_size,
                periodic=False,
                device=device,
                dtype=dtype
            ).pow(0.85)
        elif win_type == "hann-sqrt":
            window = torch.hann_window(
                win_size,
                periodic=False,
                device=device,
                dtype=dtype
            ).pow(0.5)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size,
                device=device, dtype=dtype)
        self.register_buffer("window", window, persistent=False)
        self.window: Tensor
        self.win_size = win_size
        assert n_fft >= win_size, f"n_fft({n_fft}) must be bigger than win_size({win_size})"

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T_wav] or [B, 1, T_wav]
        # output: [B, n_fft//2+1, T_spec(, 2)]
        if x.dim() == 3:  # [B, 1, T] -> [B, T]
            x = x.squeeze(1)
        if self.padding > 0:
            x = F.pad(x.unsqueeze(0), (self.padding, self.padding), mode=self.pad_mode).squeeze(0)

        spec = torch.stft(x, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
            window=self.window, center=self.center, pad_mode=self.pad_mode,
            normalized=self.normalized, onesided=True, return_complex=True)

        if self.magnitude:
            spec = spec.abs()
        else:
            spec = torch.view_as_real(spec)
        
        if self.clip:
            spec = spec[:, :, :-1]

        return spec

    def inverse(self, spec: Tensor) -> Tensor:
        # x: [B, n_fft//2+1, T_spec, 2]
        # output: [B, T_wav]
        if not self.center:
            raise NotImplementedError("center=False is currently not implemented. "
                "Please set center=True")

        spec = torch.view_as_complex(spec.contiguous())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav = torch.istft(spec, self.n_fft, hop_length=self.hop_size,
                win_length=self.win_size, center=self.center, normalized=self.normalized,
                window=self.window, onesided=True, return_complex=False)

        return wav

    def inverse_complex(self, spec: Tensor) -> Tensor:
        # x: [B, n_fft//2+1, T_spec] (complex)
        # output: [B, T_wav]
        if not self.center:
            raise NotImplementedError("center=False is currently not implemented. "
                "Please set center=True")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav = torch.istft(spec, self.n_fft, hop_length=self.hop_size,
                win_length=self.win_size, center=self.center, normalized=self.normalized,
                window=self.window, onesided=True, return_complex=False)

        return wav


class CompressedSTFT(STFT):
    def __init__(
        self,
        n_fft: int,
        hop_size: int,
        win_size: int,
        win_type: str = "hann",
        normalized: bool = False,
        compression: float = 1.0,
        discard_last_freq_bin: bool = False,
        eps: float = 1.0e-5,
    ) -> None:
        assert compression <= 1.0, compression
        super().__init__(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=win_type, normalized=normalized, magnitude=False
        )
        self.compression = compression
        self.eps = eps
        self.discard_last_freq_bin = discard_last_freq_bin
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, T_wav] or [B, T_wav]
        # output: [B, n_fft//2, T, 2] (real) if discard_last_freq_bin=True
        # output: [B, n_fft//2+1, T, 2] (real) if discard_last_freq_bin=False
        x = super().forward(x)
        if self.discard_last_freq_bin:
            x = x[:, :-1, :, :]
        mag = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x * mag.pow(self.compression - 1.0)
        return x

    def inverse(self, x: Tensor) -> Tensor:
        # x: [B, n_fft//2, T] (complex) if discard_last_freq_bin=True
        # x: [B, n_fft//2+1, T] (complex) if discard_last_freq_bin=False
        # output: [B, T_wav]
        mag_compressed = x.abs()
        x = x * mag_compressed.pow(1.0 / self.compression - 1.0)
        if self.discard_last_freq_bin:
            x = F.pad(x, (0, 0, 0, 1))  # [B, n_fft//2F+1, T]
        return super().inverse_complex(x)


class ONNXSTFT(nn.Module):
    '''Short-Time Fourier Transform
    STFT: Implemented using torch.stft, which can be converted into onnx.stft.
    ISTFT: Implemented using torch.fft.irfft, because onnx.istft is currently not implemented.
    forward(x):
        x: [B, hop_size*L] or [B, hop_size*L]
        output: [B, N//2+1, L, 2]
    inverse(x):
        x: [B, N//2+1, L, 2]
        output: [B, hop_size*L]
    '''

    __constants__ = ["n_fft", "hop_size", "cache_len", "normalized",
                     "window", "weight"]

    def __init__(
        self,
        n_fft: int,
        hop_size: int,
        win_size: Optional[int] = None,
        win_type: Optional[str] = "hann",
        normalized: bool = False,
        device=None,
        dtype=None
    ):
        assert n_fft % 2 == 0, f"`n_fft` must be an even number, but given {n_fft}."
        assert normalized == False
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.cache_len = n_fft - hop_size
        self.normalized = normalized

        if dtype is None:
            dtype = torch.float32
        factory_kwargs = {'device': device, 'dtype': dtype}

        if win_size is None:
            win_size = n_fft
        assert n_fft >= win_size, \
            f"n_fft({n_fft}) must be bigger than win_size({win_size})"

        # Get window
        if win_type is None:
            window = torch.ones(n_fft, **factory_kwargs)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(
                win_size, **factory_kwargs)
            if win_size < n_fft:
                padding = n_fft - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        self.register_buffer("window", window, persistent=False)
        self.window: Tensor

        # Get iSTFT weight
        K = (n_fft + hop_size - 1) // hop_size  # <=> math.ceil(n_fft / hop_size)
        L = hop_size * (2*K-1) + (n_fft - hop_size)
        win_sq = window.square().view(1, -1, 1)     # [1, n_fft, 1]
        win_sq = win_sq.expand(1, -1, 2*K-1)        # [1, n_fft, 2*K-1]
        win_sq_sum = F.fold(
            win_sq,
            output_size = (1, L),
            kernel_size = (1, n_fft),
            stride = (1, hop_size),
            padding = (0, 0)
        ).view(-1)  # [n_fft-hop_size + hop_size*(2*K-1)]
        win_sq_sum = win_sq_sum[(K-1)*hop_size:(K-1)*hop_size + n_fft]  # [n_fft]
        window_istft = window / win_sq_sum
        self.register_buffer("window_istft", window_istft, persistent=False)
        self.window_istft: Tensor

    def initialize_cache(self, x: Tensor) -> List[Tensor]:
        cache_stft = torch.zeros(x.size(0), self.cache_len, dtype=x.dtype, device=x.device)
        cache_istft = torch.zeros(x.size(0), self.cache_len, dtype=x.dtype, device=x.device)
        return [cache_stft, cache_istft]

    def forward(self, x: Tensor, cache: Tensor) -> Tuple[Tensor, Tensor]:
        '''x: [B=1, hop_size]
        cache: [B=1, n_fft-hop_size]
        output: [B, n_fft//2, T=1, 2]
        '''
        x = torch.cat([cache, x], dim=1)  # [B, n_fft]
        cache = x[:, -self.cache_len:]    # [B, n_fft-hop_size]
        x = x * self.window
        x = torch.fft.rfft(x, dim=1)            # [1, self.n_fft//2+1] (complex)
        x = torch.view_as_real(x).unsqueeze(2)  # [1, self.n_fft//2+1, 1, 2] (real)
        # x = x.stft(n_fft=self.n_fft, hop_length=self.hop_size,
        #            window=self.window, normalized=self.normalized,
        #            center=False, onesided=True, return_complex=True)
        # x = torch.view_as_real(x)
        return x, cache

    def inverse(self, x: Tensor, cache: Tensor) -> Tuple[Tensor, Tensor]:
        '''input:
            x: [B, N//2+1, T=1, 2]
            cache: [B, N-H]
        output:
            x: [B, H*T=H]
            cache: [B, N-H]
        '''
        # Below is an original irFFT code.
        # x = torch.view_as_complex(x.view(self.n_fft//2+1, 2))
        # x = torch.fft.irfft(x).view(1, self.n_fft)
        # ONNX doesn't support irFFT with an input of [n_fft//2+1, 2].

        # Method 1) X: [B, n_fft//2+1] -> X_full: [B, n_fft] -> iFFT -> real part
        # x_full = nn.functional.pad(
        #     x.squeeze(2),
        #     (0, 0, 0, self.n_fft//2-1),
        #     mode='reflect'
        # )                                       # [B, n_fft, 2]
        # x_full[:, self.n_fft//2+1:, 1] *= -1    # complex conjugate
        # x_full = torch.view_as_complex(x_full)  # [B, n_fft] (complex)
        # x = torch.fft.ifft(x_full, dim=1).real  # [B, n_fft] (real)

        # Method 2)
        # x[n] = 1/N sigma_{k=0}^{N-1}{e^{j 2 \pi k / N * n} X[k]}
        #      = 2/N Re{ sigma_{k=0}^{N/2}{e^{j 2 \pi k / N * n} X[k]} } - 1/N*(X[0]+(-1)^n*X[N/2])
        x_0 = x[:, 0:1, 0, 0]
        x_last = x[:, -1:, 0, 0]
        x = nn.functional.pad(
            x.squeeze(2),
            (0, 0, 0, self.n_fft//2-1)
        )   # [B, n_fft, 2]
        x = torch.fft.ifft(torch.view_as_complex(x), dim=1).real        # [B, n_fft]
        # x = torch.fft.irfft(torch.view_as_complex(x), dim=1, n=self.n_fft)  # [B, n_fft]
        x = x.reshape(-1, self.n_fft//2, 2)                             # [B, n_fft//2, 2]
        correction = torch.stack([x_0 + x_last, x_0 - x_last], dim=2)   # [B, 1, 2]
        x = 2 * x - correction / self.n_fft
        x =  x.view(-1, self.n_fft)
        # irFFT end

        x = x * self.window_istft
        x[:, :cache.size(1)] += cache
        out = x[:, :-(self.n_fft - self.hop_size)]      # [B, H*T]
        cache = x[:, -(self.n_fft - self.hop_size):]    # [B, N-H]
        return out, cache


if __name__=="__main__":
    """Export STFT, iSTFT to ONNXRuntime"""
    import argparse
    import onnx
    import onnxruntime
    import librosa
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np

    parser = argparse.ArgumentParser(description="Export STFT, iSTFT to ONNXRuntime.")
    parser.add_argument('--test-streaming', action='store_true')
    parser.add_argument('--wav', default=None, help='Optional 48 kHz wav used for the export self-test.')
    args = parser.parse_args()

    class ONNXiSTFT(ONNXSTFT):
        def forward(self, x: Tensor, cache: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
            return super().inverse(x, cache)

    N, H, W = 1024, 512, 1024
    SR = 48_000
    win_type = "hann"
    stft = ONNXSTFT(N, H, win_type=win_type)
    istft = ONNXiSTFT(N, H, win_type=win_type)
    if args.wav:
        x_np = librosa.load(args.wav, sr=SR)[0]
    else:
        x_np = np.random.randn(SR).astype("float32") * 0.01
    x = torch.from_numpy(x_np).view(1, -1)
    T = x.size(-1) // H * H
    x = x[:, :T]
    x = torch.nn.functional.pad(x, (N-H, 0))
    
    if args.test_streaming:
        window = stft.window
        cache = istft.initialize_cache(x)
        x_hat = []
        for i in tqdm(range(0, T, H)):
            x_in = x[:, i : i + H]
            spec = stft(x_in)
            x_out, cache = istft(spec, cache)
            x_hat.append(x_out)
        x_hat = torch.cat(x_hat, dim=1).squeeze(0)
        x = x[0, :-(N-H)]
        plt.plot(x)
        plt.plot(x_hat - x)
        plt.savefig("onnx/delete_it.png")
        exit()

    # Prepare inputs
    x_in = x[:, :H]
    cache_stft, cache_istft = stft.initialize_cache(x)
    spec, _ = stft(x_in, cache_stft)

    # Export STFT to ONNX
    torch.onnx.export(
        stft,
        args=(x_in, cache_stft),
        f="onnx/delete_it.onnx",
        input_names = ['wav_in', 'cache_in'],
        output_names = ['spec_out', 'cache_out'],
        dynamo=True
    )
    onnx_stft = onnx.load("onnx/delete_it.onnx")
    onnx.checker.check_model(onnx_stft)
    # onnx_stft, check = simplify(onnx_stft)

    # Export iSTFT to ONNX
    torch.onnx.export(
        istft,
        args=(spec, cache_istft),
        f="onnx/delete_it.onnx",
        input_names = ['spec_in', 'cache_in'],
        output_names = ['wav_out', 'cache_out'],
        dynamo=True
    )
    onnx_istft = onnx.load("onnx/delete_it.onnx")
    onnx.checker.check_model(onnx_istft)
    # onnx_stft, check = simplify(onnx_istft)

    # Merge STFT, iSTFT
    merged_model = onnx.compose.merge_models(
        onnx_stft, onnx_istft,
        io_map=[('spec_out', 'spec_in')],
        prefix1='stft_', prefix2='istft_'
    )
    onnx.checker.check_model(merged_model)
    onnx.save(merged_model, "onnx/delete_it.onnx")
    sess = onnxruntime.InferenceSession(
        "onnx/delete_it.onnx",
        providers=['CPUExecutionProvider']
    )
    print([x.name for x in sess.get_inputs()])
    print([x.name for x in sess.get_outputs()])
    onnx_input = {
        "stft_cache_in": cache_stft.numpy(),
        "istft_cache_in": cache_istft.numpy()
    }
    x = x.numpy()
    y_hat = []
    for i in tqdm(range(0, T, H)):
        onnx_input["stft_wav_in"] = x[:, i : i + H]
        out = sess.run(None, onnx_input)
        y_hat.append(out[0][0])
        onnx_input["stft_cache_in"] = out[1]
        onnx_input["istft_cache_in"] = out[2]
    y_hat = np.concatenate(y_hat, axis=0)
    y = x[0, :-(N-H)]
    import matplotlib.pyplot as plt
    plt.plot(y)
    plt.plot(y_hat - y)
    plt.savefig("onnx/delete_it.png")

import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm as weight_norm_fn
from torch.nn.utils.parametrize import remove_parametrizations
from torch import Tensor



class StridedConv1d(nn.Conv1d):
    """Same as nn.Conv1d with stride > 1.
    We just want to show that MAC of StridedConv is not Cin x Cout x K x T (ptflops calculation),
    but Cin x Cout x K x (T / S) where K: kernel_size, T: time, S: stride."""
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        stride: int = 1, padding: int = 0, dilation: int = 1,
        groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
        device=None, dtype=None
    ):
        assert kernel_size % stride == 0, (
            f'kernel_size k and stride s must satisfy k=(2n+1)s, but '
            f'got k={kernel_size}, s={stride}. Use naive Conv1d instead.'
        )
        assert groups == 1, (
            f'groups must be 1, but got {groups}. '
            f'Use naive Conv1d instead.'
        )
        assert dilation == 1, (
            f'dilation must be 1, but got {dilation}. '
            f'Use naive Conv1d instead.'
        )
        assert padding_mode == 'zeros', (
            f'Only `zeros` padding mode is supported for '
            f'StridedConv1d, but got {padding_mode}. '
            f'Use naive Conv1d instead.'
        )
        self.original_stride = stride
        self.original_padding = padding
        super().__init__(
            in_channels*stride, out_channels, kernel_size//stride,
            stride=1, padding=0, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
            device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, Ci, Ti] -> conv1d -> [B, Co, Ti // s]
        <=> x: [B, Ci, Ti] -> reshape to [B, Ci*s, Ti//s] -> conv1d -> [B, Co, Ti//S]"""
        stride = self.original_stride
        padding = self.original_padding
        x = F.pad(x, (padding, padding))
        B, C, T = x.shape
        x = x.view(B, C, T//stride, stride).permute(0, 3, 1, 2).reshape(B, C*stride, T//stride)
        return super().forward(x)


class ScaledConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        *args,
        normalize: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.scale = nn.Parameter(torch.ones(1))
        self.weight_norm = True

    def remove_weight_reparameterizations(self):
        if self.normalize:
            weight = F.normalize(self.weight, dim=(0, 1, 2)).mul_(self.scale)
        else:
            weight = self.weight * self.scale
        self.weight.data.copy_(weight)
        self.weight_norm = False
        self.scale = None

    def forward(self, x: Tensor) -> Tensor:
        if self.weight_norm:
            if self.normalize:
                weight = F.normalize(self.weight, dim=(0, 1, 2)).mul_(self.scale)
            else:
                weight = self.weight * self.scale
        else:
            weight = self.weight
        return F.conv_transpose1d(
            x, weight, self.bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, dilation=self.dilation,
        )


def calculate_positional_embedding(channels: int, freq: int) -> Tensor:
    # f0: [1/F, 2/F, ..., 1] * pi
    # c: [1, ..., F-1] -> log-spaced, numel = C//2
    f = torch.arange(1, freq+1, dtype=torch.float32) * (math.pi / freq)
    c = torch.linspace(
        start=math.log(1),
        end=math.log(freq-1),
        steps=channels//2,
        dtype=torch.float32
    ).exp()
    grid = f.view(-1, 1) * c.view(1, -1)            # [F, C//2]
    pe = torch.cat((grid.sin(), grid.cos()), dim=1) # [F, C]
    return pe


class ChannelsLastBatchNorm(nn.BatchNorm1d):
    def forward(self, x: Tensor) -> Tensor:
        """input/output: [T, B, F, C]"""
        T, B, F, C = x.shape
        x = x.view(T*B*F, C, 1)
        return super().forward(x).view(T, B, F, C)


class ChannelsLastSyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, x: Tensor) -> Tensor:
        """input/output: [T, B, F, C]"""
        T, B, F, C = x.shape
        x = x.view(T*B*F, C, 1)
        return super().forward(x).view(T, B, F, C)


class Attention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        attn_bias: bool,
    ):
        super().__init__()
        self.channels = channels // num_heads
        self.num_heads = num_heads
        self.scale: float = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels*3, bias=attn_bias)

    def forward(self, x: Tensor) -> Tensor:
        '''input / output: [T*B, F, C]'''
        TB, Freq, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(TB, Freq, self.num_heads, -1).transpose(1, 2)     # [TB, NH, F, C']
        q = qkv[:, :, :, :self.channels]
        k = qkv[:, :, :, self.channels:self.channels*2]
        v = qkv[:, :, :, self.channels*2:]
        out = F.scaled_dot_product_attention(q, k, v, scale=None)     # [TB, NH, F, C'']
        out = out.transpose(1, 2).reshape(TB, Freq, -1)     # [TB, F, Cout]
        return out


class RNNFormerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        freq: int,
        num_heads: int,
        weight_norm: bool,
        activation: str,
        activation_kwargs: tp.Dict[str, tp.Any],
        positional_embedding: tp.Optional[str],
        attn_bias: bool = False,
        eps: float = 1e-8,
        post_act: bool = False,
        pre_norm: bool = False,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.freq = freq
        self.pre_norm = pre_norm

        def Act(**kwargs):
            if post_act:
                return getattr(nn, activation)(**kwargs)
            return nn.Identity(**kwargs)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = ChannelsLastSyncBatchNorm
        else:
            BatchNorm = ChannelsLastBatchNorm

        self.rnn_pre_norm = BatchNorm(channels, eps, affine=False) if pre_norm else nn.Identity()
        self.rnn = nn.GRU(channels, channels, batch_first=False)
        self.rnn_fc = nn.Linear(channels, channels, bias=False)
        self.rnn_post_norm = BatchNorm(channels, eps)
        self.rnn_act = Act(**activation_kwargs)

        self.attn_pre_norm = BatchNorm(channels, eps, affine=False) if pre_norm else nn.Identity()
        self.attn = Attention(channels, num_heads, attn_bias)
        self.attn_fc = nn.Linear(channels, channels, bias=False)
        self.attn_post_norm = BatchNorm(channels, eps)
        self.attn_act = Act(**activation_kwargs)

        self.dropout = nn.Identity() if p_dropout == 0.0 else nn.Dropout(p=p_dropout, inplace=True)

        self.pe = None
        if positional_embedding is not None:
            pe = calculate_positional_embedding(channels, freq) # [F, C]
            if positional_embedding == "fixed":
                self.register_buffer("pe", pe)
                self.pe: Tensor
            elif positional_embedding == "train":
                self.pe = nn.Parameter(pe)

        self.weight_norm = weight_norm
        if weight_norm:
            self.rnn = weight_norm_fn(self.rnn, name="weight_ih_l0")
            self.rnn = weight_norm_fn(self.rnn, name="weight_hh_l0")
            self.attn.qkv = weight_norm_fn(self.attn.qkv)

    def remove_weight_reparameterizations(self):
        if self.weight_norm:
            remove_parametrizations(self.rnn, "weight_ih_l0")
            remove_parametrizations(self.rnn, "weight_hh_l0")
            remove_parametrizations(self.attn.qkv, "weight")
            self.flatten_parameters()
            self.weight_norm = False

        for fc, norm in (
            (self.rnn_fc, self.rnn_post_norm),
            (self.attn_fc, self.attn_post_norm)
        ):
            std = norm.running_var.add(norm.eps).sqrt()
            fc.weight.data *= norm.weight.view(-1, 1) / std.view(-1, 1)
            fc.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
        self.rnn_post_norm = nn.Identity()
        self.attn_post_norm = nn.Identity()

        if self.pre_norm:
            # w @ (x - mean) / std + bias
            # = (w \cdot gamma) @ x + (bias + w @ beta)
            # where gamma = 1/std, beta = -mean/std
            # 1. Attn
            norm = self.attn_pre_norm
            std = norm.running_var.add(norm.eps).sqrt()
            beta = -norm.running_mean / std
            w_matmul_beta = (self.attn.qkv.weight.data @ beta.view(-1, 1)).squeeze(1)

            self.attn.qkv.weight.data /= std
            attn_bias = torch.zeros(self.attn.qkv.weight.size(0))
            if self.attn.qkv.bias is not None:
                attn_bias = self.attn.qkv.bias.data
            self.attn.qkv.bias = nn.Parameter(attn_bias + w_matmul_beta)
            self.attn_pre_norm = nn.Identity()

            # 2. GRU
            norm = self.rnn_pre_norm
            std = norm.running_var.add(norm.eps).sqrt()
            beta = -norm.running_mean / std
            w_matmul_beta = (self.rnn.weight_ih_l0.data @ beta.view(-1, 1)).squeeze(1)

            self.rnn.weight_ih_l0.data /= std
            self.rnn.bias_ih_l0.data.add_(w_matmul_beta)
            self.rnn_pre_norm = nn.Identity()

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> Tensor:
        return x.new_zeros(1, self.freq, self.rnn.hidden_size)

    def forward(self, x: Tensor, h: tp.Optional[Tensor]) -> tp.Tuple[Tensor, Tensor]:
        TIME, BATCH, FREQ, CH = x.shape     # [T, B, F, C]
        x_in = x
        x = self.rnn_pre_norm(x)            # [T, B, F, C]
        x = x.view(TIME, FREQ*BATCH, CH)    # [T, B*F, C]
        x, h = self.rnn(x, h)               # [T, B*F, C]
        x = x.view(TIME, BATCH, FREQ, CH)   # [T, B, F, C]
        x = self.rnn_fc(x)                  # [T, B, F, C]
        x = self.dropout(x)
        x = self.rnn_post_norm(x)           # [T, B, F, C]
        x = self.rnn_act(x)                 # [T, B, F, C]
        x = x.add_(x_in)                    # [T, B, F, C]

        if self.pe is not None:
            x = x.add_(self.pe)
        x_in = x
        x = self.attn_pre_norm(x)           # [T, B, F, C]
        x = x.view(TIME*BATCH, FREQ, CH)    # [T*B, F, C]
        x = self.attn(x)                    # [T*B, F, C]
        x = x.view(TIME, BATCH, FREQ, CH)   # [T, B, F, C]
        x = self.attn_fc(x)                 # [T, B, F, C]
        x = self.dropout(x)
        x = self.attn_post_norm(x)          # [T, B, F, C]
        x = self.attn_act(x)                # [T, B, F, C]
        x = x.add_(x_in)                    # [T, B, F, C
        return x, h


@dataclass
class RNNFormerConfig:
    num_blocks: int = 3
    channels: int = 32
    freq: int = 32
    num_heads: int = 4
    eps: float = 1e-8
    positional_embedding: tp.Optional[str] = "train"    # None | "fixed" | "train"
    attn_bias: bool = False
    post_act: bool = False
    pre_norm: bool = False
    p_dropout: float = 0.0


def rf_pre_post_lin(
    n_freq: int,
    n_filter: int,
    init: tp.Optional[str],
    bias: bool,
    sr: int = 16_000
) -> tp.Tuple[nn.Module, nn.Module]:
    assert init in [None, "linear", "linear_fixed", "mel", "mel_fixed"]
    pre = nn.Linear(n_freq, n_filter, bias=bias)
    post = nn.Linear(n_filter, n_freq, bias=bias)

    if init is None:
        return pre, post

    if init.startswith("linear"):
        delta = (n_freq - 1) / (n_filter - 1)
        f_filter = torch.linspace(0, n_freq-1, n_filter)
        f_freqs = torch.linspace(0, n_freq-1, n_freq)
    elif init.startswith("mel"):
        def freq_idx_to_mel(f: float) -> float:
            hz = f / n_freq * sr / 2
            return 2595.0 * math.log10(1 + hz / 700)

        max_hz = sr / 2 * (n_freq - 1) / n_freq
        delta_hz = max_hz / (n_freq - 1)
        max_mel = freq_idx_to_mel(n_freq - 1)

        def mel_idx_to_freq_idx(n: float) -> float:
            mel = n / (n_filter - 1) * max_mel
            return 700.0 * (10 ** (mel / 2595) - 1) / delta_hz

        # We want to ensure that each mel filter covers at least 1 frequency bin.
        # We use linear filters for the low frequencies where mel filters are too narow,
        # and use mel filters for the high frequencies.
        # The transition point is determined by the condition that
        # the next mel filter is at least 1 frequency bin away from the current mel filter.
        f_filter = []
        f_cur = mel_idx_to_freq_idx(0)
        for n_start in range(0, n_filter-1):
            f_next = mel_idx_to_freq_idx(n_start+1)
            if f_next - f_cur >= 1 and n_start <= f_cur:
                print(n_start)
                break
            f_filter.append(n_start)
            f_cur = f_next
        f_filter.extend([mel_idx_to_freq_idx(n) for n in range(n_start, n_filter)])
        f_filter = torch.tensor(f_filter, dtype=torch.float32)  # [n_filter]
        f_freqs = torch.arange(n_freq, dtype=torch.float32)     # [n_freq]
        delta = (f_filter[1:] - f_filter[:-1]).unsqueeze(1)     # [n_filter-1, 1]
    else:
        raise RuntimeError(f"init={init} is not supported.")

    down = f_filter[1:, None] - f_freqs[None, :]    # [n_filter - 1, freq]
    up   = f_freqs[None, :] - f_filter[:-1, None]   # [n_filter - 1, freq]
    down = down / delta
    up   = up / delta
    down = F.pad(down, (0, 0, 0, 1), value=1.0)     # [n_filter, n_freq]
    up   = F.pad(up, (0, 0, 1, 0), value=1.0)       # [n_filter, n_freq]
    pre_weight = torch.max(up.new_zeros(1), torch.min(down, up))
    pre_weight = pre_weight / pre_weight.sum(dim=1, keepdim=True)
    post_weight = pre_weight.transpose(0, 1)
    post_weight = post_weight / post_weight.sum(dim=1, keepdim=True)

    if init.endswith("_fixed"):
        delattr(pre, "weight")
        delattr(post, "weight")
        pre.register_buffer("weight", pre_weight.contiguous().clone())
        post.register_buffer("weight", post_weight.contiguous().clone())
    else:
        pre.weight.data.copy_(pre_weight)
        post.weight.data.copy_(post_weight)

    return pre, post


class ONNXModel(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        kernel_size: tp.List[int] = [8, 3, 3],
        stride: int = 4,
        rnnformer_kwargs: tp.Dict[str, tp.Any] = dict(),
        activation: str = "ReLU",
        activation_kwargs: tp.Dict[str, tp.Any] = dict(inplace=True),
        n_fft: int = 512,
        hop_size: int = 256,
        win_size: int = 512,
        window: tp.Optional[str] = "hann",
        stft_normalized: bool = False,
        mask: tp.Optional[str] = None,
        input_compression: float = 0.3,
        weight_norm: bool = False,
        normalize_final_conv: bool = False,
        pre_post_init: tp.Optional[str] = None,
        resnet: bool = False,
    ):
        super().__init__()
        self.input_compression = input_compression
        self.stft = self.get_stft(n_fft, hop_size, win_size, window, stft_normalized)
        rnnformer_config = RNNFormerConfig(**rnnformer_kwargs)
        self.rf_ch = rnnformer_config.channels
        self.rf_freq = rnnformer_config.freq
        if mask is None:
            self.mask = nn.Identity()
        elif mask == "sigmoid":
            self.mask = nn.Sigmoid()
        elif mask == "tanh":
            self.mask = nn.Tanh()
        else:
            raise RuntimeError(f"model_kwargs.mask={mask} is not supported.")
        self.weight_norm = weight_norm
        self.resnet = resnet

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm1d

        def norm(module):
            if self.weight_norm:
                return weight_norm_fn(module)
            return module

        Act = getattr(nn, activation)

        # Encoder PreNet
        assert kernel_size[0] % stride == 0
        assert (kernel_size[0] - stride) % 2 == 0
        self.enc_pre = nn.Sequential(
            StridedConv1d(  # in_channels = 2 = [real, imag]
                2, channels, kernel_size[0], stride=stride,
                padding=(kernel_size[0] - stride) // 2, bias=False
            ),
            BatchNorm(channels),
            Act(**activation_kwargs),
        )

        # Encoder
        self.encoder = nn.ModuleList()
        for idx in range(1, len(kernel_size)):
            module = nn.Sequential(
                nn.Conv1d(
                    channels, channels, kernel_size[idx], 
                    padding=(kernel_size[idx] - 1) // 2, bias=False
                ),
                BatchNorm(channels),
                Act(**activation_kwargs),
            )
            self.encoder.append(module)

        # RNNFormer PreNet
        freq = n_fft // 2 // stride
        rf_pre, rf_post = rf_pre_post_lin(freq, self.rf_freq, pre_post_init, bias=False)
        self.rf_pre = nn.Sequential(
            rf_pre,
            nn.Conv1d(channels, self.rf_ch, 1, bias=False),
            BatchNorm(self.rf_ch),
        )

        # RNNFormer Blocks
        rf_list = []
        for _ in range(rnnformer_config.num_blocks):
            block = RNNFormerBlock(
                self.rf_ch, self.rf_freq,
                rnnformer_config.num_heads,
                eps=rnnformer_config.eps, weight_norm=weight_norm,
                activation=activation, activation_kwargs=activation_kwargs,
                positional_embedding=rnnformer_config.positional_embedding,
                attn_bias=rnnformer_config.attn_bias,
                post_act=rnnformer_config.post_act,
                pre_norm=rnnformer_config.pre_norm,
                p_dropout=rnnformer_config.p_dropout,
            )
            rf_list.append(block)
            rnnformer_config.positional_embedding = None
        self.rf_block = nn.ModuleList(rf_list)

        # RNNFormer PostNet
        self.rf_post = nn.Sequential(
            rf_post,
            nn.Conv1d(self.rf_ch, channels, 1, bias=False),
            BatchNorm(channels),
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for idx in range(len(kernel_size)-1, 0, -1):
            module = nn.Sequential(
                nn.Conv1d(channels*2, channels, 1, bias=False),
                BatchNorm(channels),
                Act(**activation_kwargs),
                nn.Conv1d(
                    channels, channels, kernel_size[idx],
                    padding=(kernel_size[idx] - 1) // 2, bias=False
                ),
                BatchNorm(channels),
                Act(**activation_kwargs),
            )
            self.decoder.append(module)

        # Decoder PostNet
        # out_channels = 2 = [real, imag] of the mask
        upsample = ScaledConvTranspose1d(
            channels, 2, kernel_size[0], stride=stride,
            bias=True,
            padding=(kernel_size[0] - stride) // 2,
            normalize=normalize_final_conv,
        )
        self.dec_post = nn.Sequential(
            nn.Conv1d(channels*2, channels, 1, bias=False),
            BatchNorm(channels),
            Act(**activation_kwargs),
            upsample
        )

    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool
    ) -> nn.Module:
        return ONNXSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized
        )

    @torch.no_grad()
    def remove_weight_reparameterizations(self):
        """ 1. Remove weight_norm """
        if self.weight_norm:
            with torch.enable_grad():
                # RNNFormer
                for block in self.rf_block:
                    block.remove_weight_reparameterizations()

                # Decoder
                self.dec_post[3].remove_weight_reparameterizations()
            self.weight_norm = False

        """ 2. Merge BatchNorm into Conv
        y = (conv(x) - mean) / std * gamma + beta \
          = conv(x) * (gamma / std) + (beta - mean * gamma / std)
        <=> y = conv'(x) where
          W'[c, :, :] = W[c, :, :] * (gamma / std)
          b' = (beta - mean * gamma / std)
        """
        def merge_conv_bn(conv: nn.Module, norm: nn.Module, error_message: str = "") -> nn.Module:
            assert conv.bias is None, error_message
            std = norm.running_var.add(norm.eps).sqrt()
            conv.weight.data *= norm.weight.view(-1, 1, 1) / std.view(-1, 1, 1)
            conv.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
            return conv

        # Encoder PreNet
        conv = merge_conv_bn(self.enc_pre[0], self.enc_pre[1], "enc_pre")
        self.enc_pre = nn.Sequential(conv, self.enc_pre[2])

        # Encoder
        new_encoder = nn.ModuleList()
        for idx, module in enumerate(self.encoder):
            conv = merge_conv_bn(module[0], module[1], f"enc.{idx}")
            new_module = nn.Sequential(
                conv,       # Conv-BN Merged
                module[2],  # Activation
            )
            new_encoder.append(new_module)
        self.encoder = new_encoder

        # RNNFormer PreNet
        conv = merge_conv_bn(self.rf_pre[1], self.rf_pre[2], "rf_pre")
        self.rf_pre = nn.Sequential(
            self.rf_pre[0],     # Linear
            conv,               # Conv-BN Merged
        )

        # RNNFormer PostNet
        conv = merge_conv_bn(self.rf_post[1], self.rf_post[2], "rf_post")
        self.rf_post = nn.Sequential(
            self.rf_post[0],    # Linear
            conv,               # Conv-BN Merged
        )

        # Decoder
        new_decoder = nn.ModuleList()
        for idx, module in enumerate(self.decoder):
            conv1 = merge_conv_bn(module[0], module[1], f"dec.{idx}.0")
            conv2 = merge_conv_bn(module[3], module[4], f"dec.{idx}.1")
            new_module = nn.Sequential(
                conv1,      # Conv-BN Merged
                module[2],  # Activation
                conv2,      # Conv-BN Merged
                module[5],  # Activation
            )
            new_decoder.append(new_module)
        self.decoder = new_decoder

        # Decoder PostNet
        conv = merge_conv_bn(self.dec_post[0], self.dec_post[1], "dec_post")
        self.dec_post = nn.Sequential(
            conv,               # Conv-BN Merged
            self.dec_post[2],   # Activation
            self.dec_post[3]    # Transposed Convolution
        )

    def flatten_parameters(self):
        for rf in self.rf_block:
            rf.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_list = []
        for block in self.rf_block:
            cache_list.append(block.initialize_cache(x))
        return cache_list

    def model_forward(self, spec_noisy: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # spec_noisy: [B, F, T, 2]
        cache_in_list = [*args]
        cache_out_list = []
        if len(cache_in_list) == 0:
            cache_in_list = [None for _ in range(len(self.rf_block))]
        x = spec_noisy

        B, FREQ, T, _ = x.shape
        x = x.permute(0, 2, 3, 1)       # [B, T, 2, F]
        x = x.reshape(B*T, 2, FREQ)     # [B*T, 2, F]

        # Encoder PreNet
        x = self.enc_pre(x)
        encoder_outs = [x]

        # Encoder
        for module in self.encoder:
            x_in = x
            x = module(x)
            encoder_outs.append(x)      # [B*T, C, F']
            if self.resnet:
                x = x.add_(x_in)

        # RNNFormer
        x_in = x
        x = self.rf_pre(x)              # [B*T, C, F']
        _, C, _FREQ = x.shape
        x = x.view(B, T, C, _FREQ)      # [B, T, C, F']
        x = x.permute(1, 0, 3, 2)       # [T, B, F', C]
        x = x.contiguous()
        for block, cache_in in zip(self.rf_block, cache_in_list):
            x, cache_out = block(x, cache_in)   # [T, B, F', C]
            cache_out_list.append(cache_out)
        x = x.permute(1, 0, 3, 2)       # [B, T, C, F']
        x = x.reshape(B*T, C, _FREQ)    # [B*T, C, F']
        x = self.rf_post(x)             # [B*T, C, F']
        if self.resnet:
            x = x.add_(x_in)

        # Decoder
        for module in self.decoder:
            x_in = x
            x = torch.cat([x, encoder_outs.pop(-1)], dim=1)     # [B*T, 2*C, F]
            x = module(x)                                       # [B*T, C, F] or [B*T, 2, F]
            if self.resnet:
                x = x.add_(x_in)

        # Decoder PostNet
        x = torch.cat([x, encoder_outs.pop(-1)], dim=1)     # [B*T, 2*C, F]
        x = self.dec_post(x)                                # [B*T, 2, F]
        x = x.reshape(B, T, 2, FREQ).permute(0, 3, 1, 2)    # [B, F, T, 2]

        # Mask
        mask = self.mask(x).contiguous()
        return mask, cache_out_list

    def forward(
        self,
        spec_noisy: Tensor,
        *args
    ) -> tp.Tuple[Tensor, ...]:
        """ input/output: [B, n_fft//2+1, T_spec, 2]"""
        # Compress
        spec_noisy = spec_noisy[:, :-1, :, :]   # [B, n_fft//2, T_spec, 2]
        mag = torch.linalg.norm(
            spec_noisy,
            dim=-1,
            keepdim=True
        ).clamp(min=1.0e-5)
        spec_noisy = spec_noisy * mag.pow(self.input_compression - 1.0)

        # Model forward
        mask, cache_out_list = self.model_forward(spec_noisy, *args)
        spec_hat = torch.stack(
            [
                spec_noisy[..., 0] * mask[..., 0] - spec_noisy[..., 1] * mask[..., 1],
                spec_noisy[..., 0] * mask[..., 1] + spec_noisy[..., 1] * mask[..., 0],
            ],
            dim=3
        )   # [B, F, T, 2]

        # Uncompress
        mag_compressed = torch.linalg.norm(
            spec_hat,
            dim=3,
            keepdim=True
        )
        spec_hat = spec_hat * mag_compressed.pow(1.0 / self.input_compression - 1.0)
        spec_hat = F.pad(spec_hat, (0, 0, 0, 0, 0, 1))    # [B, F+1, T, 2]
        return spec_hat, *cache_out_list


class Model(ONNXModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool
    ) -> nn.Module:
        return CompressedSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized,
            compression=self.input_compression,
            discard_last_freq_bin=True,
        )

    def forward(self, noisy: Tensor) -> tp.Tuple[Tensor, Tensor]:
        """ input/output: [B, T_wav]"""
        spec_noisy = self.stft(noisy)                   # [B, F, T, 2]
        mask, _ = self.model_forward(spec_noisy)        # [B, F, T, 2]
        spec_hat = torch.view_as_complex(spec_noisy) \
            * torch.view_as_complex(mask)       # [B, F, T]
        wav_hat = self.stft.inverse(spec_hat)   # [B, T_wav]
        return wav_hat, torch.view_as_real(spec_hat)


def test():
    x = torch.randn(3, 16_000)
    from utils import get_hparams
    hparams = get_hparams("configs/fastenhancer/t.yaml")
    model = Model(**hparams["model_kwargs"])
    wav_out, spec_out = model(x)
    (wav_out.sum() + spec_out.sum()).backward()
    print(spec_out.shape)

    model.remove_weight_reparameterizations()
    model.flatten_parameters()
    total_params = sum(p.numel() for n, p in model.named_parameters())
    print(f"Number of total parameters: {total_params}")
    # for n, p in model.named_parameters():
    #     print(n, p.shape)


if __name__ == "__main__":
    test()
