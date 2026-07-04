import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

__all__ = [
    "CustomLayerNorm",
    "RNN",
    "DualPathRNN",
    "ConvolutionalGLU",
    "DPR",
    "LearnableSigmoid2d",
    "DSConv",
    "USConv",
    "SPConvTranspose2d",
    "NoiseDetector",
    "Encoder",
    "MaskDecoder",
]


class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, (
            f"Expect x to have {self.num_dims} dimensions, but got {x.ndim}"
        )
        mu_ = x.mean(dim=self.stat_dims, keepdim=True)
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class RNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout_p=0.1, bidirectional=False):
        super().__init__()
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dense = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim,
            emb_dim,
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dense(x)
        return x


class DualPathRNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_freqs=32, dropout_p=0.1):
        super().__init__()
        self.intra_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.intra_rnn_attn = RNN(
            emb_dim,
            hidden_dim // 2,
            dropout_p,
            bidirectional=True,
        )
        self.inter_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.inter_rnn_attn = RNN(
            emb_dim,
            hidden_dim,
            dropout_p,
            bidirectional=False,
        )

    def forward(self, x):
        bsz, emb, time, freq = x.size()
        x = x.permute(0, 2, 3, 1)

        x_res = x
        x = self.intra_norm(x)
        x = x.reshape(bsz * time, freq, emb)
        x = self.intra_rnn_attn(x)
        x = x.reshape(bsz, time, freq, emb)
        x = x + x_res

        x_res = x
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz * freq, time, emb)
        x = self.inter_rnn_attn(x)
        x = x.reshape(bsz, freq, time, emb).permute(0, 2, 1, 3)
        x = x + x_res

        x = x.permute(0, 3, 1, 2)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x


class DPR(nn.Module):
    def __init__(self, emb_dim=16, hidden_dim=24, n_freqs=32, dropout_p=0.1):
        super().__init__()
        self.dp_rnn_attn = DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p)
        self.conv_glu = ConvolutionalGLU(
            emb_dim,
            n_freqs=n_freqs,
            expansion_factor=2,
            dropout_p=dropout_p,
        )

    def forward(self, x):
        x = self.dp_rnn_attn(x)
        x = self.conv_glu(x)
        return x


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 4
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3)),
        )
        self.high_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(2, 5),
                stride=(1, 3),
            ),
        )
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x_low = x[..., : self.low_freqs]
        x_high = x[..., self.low_freqs :]
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        x = self.norm(x)
        x = self.act(x)
        return x


class USConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3)),
        )
        self.high_conv = SPConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            r=3,
        )

    def forward(self, x):
        x_low = x[..., : self.low_freqs]
        x_high = x[..., self.low_freqs :]
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad = nn.ConstantPad2d(
            (
                kernel_size[1] // 2,
                kernel_size[1] // 2,
                kernel_size[0] - 1,
                0,
            ),
            value=0.0,
        )
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * r,
            kernel_size=kernel_size,
            stride=(1, 1),
        )
        self.r = r

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, height, width = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, height, width))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, height, -1))
        return out


class NoiseDetector(nn.Module):
    def __init__(
        self,
        in_channels=1,
        emb_dim=16,
        hidden_dim=32,
        n_freqs=64,
        dropout_p=0.1,
    ):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim // 4, (1, 1), (1, 1)),
            CustomLayerNorm((1, n_freqs), stat_dims=(1, 3)),
            nn.PReLU(emb_dim // 4),
        )
        self.conv_2 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(emb_dim // 4, emb_dim // 2, (2, 3), (1, 2)),
            CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3)),
            nn.PReLU(emb_dim // 2),
        )
        self.conv_3 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(emb_dim // 2, emb_dim, (2, 3), (1, 2)),
            CustomLayerNorm((1, n_freqs // 4), stat_dims=(1, 3)),
            nn.PReLU(emb_dim),
        )
        self.dpr = DPR(emb_dim, hidden_dim, n_freqs=n_freqs // 4)
        self.down = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(emb_dim, emb_dim * 2, (2, 3), (1, 2)),
            CustomLayerNorm((1, n_freqs // 8), stat_dims=(1, 3)),
            nn.PReLU(emb_dim * 2),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear_block = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.PReLU(emb_dim * 2),
            nn.Linear(emb_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.dpr(x)
        x = self.down(x)
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3).reshape(batch * time, channels, freq)
        x = self.pool(x).squeeze(-1).reshape(batch, time, channels)
        x = self.linear_block(x).squeeze(-1)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=16):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels // 4, (1, 1), (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )
        self.conv_2 = DSConv(num_channels // 4, num_channels // 2, n_freqs=257)
        self.conv_3 = DSConv(num_channels // 2, (num_channels // 4) * 3, n_freqs=128)
        self.conv_4 = DSConv((num_channels // 4) * 3, num_channels, n_freqs=64)

    def forward(self, x):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x)
        out_list.append(x)
        x = self.conv_3(x)
        out_list.append(x)
        x = self.conv_4(x)
        out_list.append(x)
        return out_list


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channels=64, out_channel=2, beta=1):
        super().__init__()
        self.up1 = USConv(num_channels * 2, (num_channels // 4) * 3, n_freqs=32)
        self.up2 = USConv((num_channels // 4) * 3 * 2, num_channels // 2, n_freqs=64)
        self.up3 = USConv((num_channels // 2) * 2, num_channels // 4, n_freqs=128)
        self.mask_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(num_channels // 4, out_channel, (2, 2)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def forward(self, x, encoder_out_list):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1)
        x = self.lsigmoid(x).permute(0, 3, 2, 1)
        return x
