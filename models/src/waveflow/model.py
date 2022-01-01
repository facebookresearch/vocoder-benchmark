import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Tuple

from models.src.waveflow.utils import add_weight_norms, fused_gate
from models.src.waveflow.base import FlowBase


class NonCausalLayer2D(nn.Module):
    def __init__(self,
                 h_dilation,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        self.h_pad_size = h_dilation * (radix - 1)
        self.pad_size = dilation * (radix - 1) // 2

        self.W = nn.Conv2d(residual_channels, dilation_channels * 2,
                           kernel_size=radix,
                           dilation=(h_dilation, dilation), bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv2d(
                dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv2d(
                dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

    def forward(self, x, y):
        tmp = F.pad(x, [self.pad_size] * 2 + [self.h_pad_size, 0])
        xy = self.W(tmp) + y
        zw, zf = xy.chunk(2, 1)
        z = fused_gate(zw, zf)
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output + x[:, :, -output.size(2):], skip
        else:
            return None, skip

    def reverse_mode_forward(self, x, y, buffer=None):
        if buffer is None:
            buffer = F.pad(x, [0, 0, self.h_pad_size, 0])
        else:
            buffer = torch.cat((buffer[:, :, 1:], x), 2)
        tmp = F.pad(buffer, [self.pad_size] * 2)
        xy = self.W(tmp) + y
        zw, zf = xy.chunk(2, 1)
        z = fused_gate(zw, zf)
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        if len(z):
            output = z[0]
            return output + x, skip, buffer
        else:
            return None, skip, buffer


class WN2D(nn.Module):
    def __init__(self,
                 n_group,
                 aux_channels,
                 dilation_channels=256,
                 residual_channels=256,
                 skip_channels=256,
                 bias=False,
                 zero_init=True):
        super().__init__()

        dilation_dict = {
            8: [1] * 8,
            16: [1] * 8,
            32: [1, 2, 4] * 2 + [1, 2],
            64: [1, 2, 4, 8, 16, 1, 2, 4],
            128: [1, 2, 4, 8, 16, 32, 64, 1],
        }

        self.h_dilations = dilation_dict[n_group]
        dilations = 2 ** torch.arange(8)
        self.dilations = dilations.tolist()
        self.n_group = n_group
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.aux_chs = aux_channels
        self.r_field = sum(self.dilations) * 2 + 1
        self.h_r_field = sum(self.h_dilations) * 2 + 1

        self.V = nn.Conv1d(aux_channels, dilation_channels *
                           2 * 8, 1, bias=bias)
        self.V.apply(add_weight_norms)

        self.start = nn.Conv2d(1, residual_channels, 1, bias=bias)
        self.start.apply(add_weight_norms)

        self.layers = nn.ModuleList(NonCausalLayer2D(hd, d,
                                                     dilation_channels,
                                                     residual_channels,
                                                     skip_channels,
                                                     3,
                                                     bias) for hd, d in zip(self.h_dilations[:-1], self.dilations[:-1]))
        self.layers.append(NonCausalLayer2D(self.h_dilations[-1], self.dilations[-1],
                                            dilation_channels,
                                            residual_channels,
                                            skip_channels,
                                            3,
                                            bias,
                                            last_layer=True))
        self.layers.apply(add_weight_norms)

        self.end = nn.Conv2d(skip_channels, 2, 1, bias=bias)
        if zero_init:
            self.end.weight.data.zero_()
            if bias:
                self.end.bias.data.zero_()

    def forward(self, x, y):
        x = self.start(x)
        y = self.V(y).unsqueeze(2)
        cum_skip = 0
        for layer, v in zip(self.layers, y.chunk(len(self.layers), 1)):
            x, skip = layer(x, v)
            cum_skip = cum_skip + skip
        return self.end(cum_skip).chunk(2, 1)

    def reverse_mode_forward(self, x, y=None, cond=None, buffer_list=None):
        x = self.start(x)
        new_buffer_list = []
        if buffer_list is None:
            buffer_list = [None] * len(self.layers)
        if cond is None:
            cond = self.V(y).unsqueeze(2).chunk(len(self.layers), 1)

        cum_skip = 0
        for layer, buf, v in zip(self.layers, buffer_list, cond):
            x, skip, buf = layer.reverse_mode_forward(x, v, buf)
            new_buffer_list.append(buf)
            cum_skip = cum_skip + skip

        return self.end(cum_skip).chunk(2, 1) + (cond, new_buffer_list,)


class WaveFlow(FlowBase):
    def __init__(self,
                 flows,
                 n_group,
                 n_mels,
                 hop_length,
                 reverse_mode=False,
                 **kwargs):
        super().__init__(hop_length, reverse_mode)
        self.flows = flows
        self.n_group = n_group
        self.n_mels = n_mels

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(1, 1, (3, hop_length * 2 + 1),
                               stride=(1, hop_length), padding=(1, hop_length)),
            nn.LeakyReLU(0.4, True)
        )
        self.upsampler.apply(add_weight_norms)

        self.WNs = nn.ModuleList()

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        for k in range(flows):
            self.WNs.append(WN2D(n_group, n_mels * n_group, **kwargs))

    def forward_computation(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.upsampler(h.unsqueeze(1)).squeeze(1)
        y = y[..., :x.size(-1)]

        batch_dim = x.size(0)
        x = x.view(batch_dim, 1, -1, self.n_group).transpose(2, 3).contiguous()
        y = y.view(batch_dim, y.size(1), -1,
                   self.n_group).transpose(2, 3).reshape(batch_dim, y.size(1) * self.n_group, -1)

        logdet: Tensor = 0
        for WN in self.WNs:
            x0 = x[:, :, :1]
            log_s, t = WN(x[:, :, :-1], y)
            xout = x[:, :, 1:] * log_s.exp() + t
            logdet += log_s.sum((1, 2, 3))
            x = torch.cat((xout.flip(2), x0), 2)

        return x.squeeze(1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet

    def reverse_computation(self, z: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.upsampler(h.unsqueeze(1)).squeeze(1)
        y = y[..., :z.size(-1)]

        batch_dim = z.size(0)
        z = z.view(batch_dim, 1, -1, self.n_group).transpose(2, 3).contiguous()
        y = y.view(batch_dim, y.size(1), -1, self.n_group).transpose(2,
                                                                     3).reshape(batch_dim, y.size(1) * self.n_group, -1)

        logdet: Tensor = None
        for WN in self.WNs[::-1]:
            z = z.flip(2)
            xnew = z[:, :, :1]
            x = [xnew]

            buffer_list = None
            cond = None
            for i in range(1, self.n_group):
                log_s, t, cond, buffer_list = WN.reverse_mode_forward(
                    xnew, y if cond is None else None, cond, buffer_list)
                xnew = (z[:, :, i:i+1] - t) / log_s.exp()
                x.append(xnew)

                if logdet is None:
                    logdet = -log_s.sum((1, 2, 3))
                else:
                    logdet -= log_s.sum((1, 2, 3))
            z = torch.cat(x, 2)

        z = z.squeeze(1).transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet
