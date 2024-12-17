# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors

# coding: utf-8

from __future__ import absolute_import, print_function, with_statement

import math

import torch

from models.src.wavenet_vocoder import conv # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavenet_vocoder import conv 
from torch import nn
from torch._tensor import Tensor
from torch.nn import functional as F
from torch.nn.modules.sparse import Embedding


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def Conv1d(in_channels, out_channels, kernel_size, dropout: int = 0, **kwargs):
    m = conv.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def Embedding(
    # pyre-fixme[2]: Parameter must be annotated.
    num_embeddings,
    # pyre-fixme[2]: Parameter must be annotated.
    embedding_dim,
    # pyre-fixme[2]: Parameter must be annotated.
    padding_idx,
    std: float = 0.01,
    # pyre-fixme[11]: Annotation `Embedding` is not defined as a type.
) -> Embedding:
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs):
    freq_axis_kernel_size = kernel_size[0]
    m = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
    m.weight.data.fill_(1.0 / freq_axis_kernel_size)
    # pyre-fixme[16]: Optional type has no attribute `data`.
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def Conv1d1x1(in_channels, out_channels, bias: bool = True):
    """1-by-1 convolution layer"""
    return Conv1d(
        in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
    )


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 forward"""
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit

    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        residual_channels,
        # pyre-fixme[2]: Parameter must be annotated.
        gate_channels,
        # pyre-fixme[2]: Parameter must be annotated.
        kernel_size,
        # pyre-fixme[2]: Parameter must be annotated.
        skip_out_channels=None,
        cin_channels: int = -1,
        gin_channels: int = -1,
        dropout: float = 1 - 0.95,
        # pyre-fixme[2]: Parameter must be annotated.
        padding=None,
        dilation: int = 1,
        causal: bool = True,
        bias: bool = True,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        # pyre-fixme[4]: Attribute must be annotated.
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
            *args,
            **kwargs,
        )

        # local conditioning
        if cin_channels > 0:
            # pyre-fixme[4]: Attribute must be annotated.
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=False)
        else:
            self.conv1x1c = None

        # global conditioning
        if gin_channels > 0:
            # pyre-fixme[4]: Attribute must be annotated.
            self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, bias=False)
        else:
            self.conv1x1g = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        # pyre-fixme[4]: Attribute must be annotated.
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        # pyre-fixme[4]: Attribute must be annotated.
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x: Tensor, c=None, g=None):
        return self._forward(x, c, g, False)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def incremental_forward(self, x: Tensor, c=None, g=None):
        return self._forward(x, c, g, True)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _forward(self, x: Tensor, c, g, is_incremental):
        """Forward

        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not

        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, : residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb

        # global conditioning
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb

        x = torch.tanh(a) * torch.sigmoid(b)

        # For skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

        # For residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self) -> None:
        for c in [
            self.conv,
            self.conv1x1_out,
            self.conv1x1_skip,
            self.conv1x1c,
            self.conv1x1g,
        ]:
            if c is not None:
                c.clear_buffer()
