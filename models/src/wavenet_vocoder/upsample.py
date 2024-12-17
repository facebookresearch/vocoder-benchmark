# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors

# coding: utf-8
from __future__ import absolute_import, print_function, with_statement

import numpy as np
from torch import nn
from torch.nn import functional as F


class Stretch2d(nn.Module):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, x_scale, y_scale, mode: str = "nearest") -> None:
        super(Stretch2d, self).__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.x_scale = x_scale
        # pyre-fixme[4]: Attribute must be annotated.
        self.y_scale = y_scale
        self.mode = mode

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode
        )


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _get_activation(upsample_activation):
    nonlinear = getattr(nn, upsample_activation)
    return nonlinear


class UpsampleNetwork(nn.Module):
    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        upsample_scales,
        upsample_activation: str = "none",
        # pyre-fixme[2]: Parameter must be annotated.
        upsample_activation_params={},
        mode: str = "nearest",
        freq_axis_kernel_size: int = 1,
        cin_pad: int = 0,
        cin_channels: int = 80,
    ) -> None:
        super(UpsampleNetwork, self).__init__()
        self.up_layers = nn.ModuleList()
        total_scale = np.prod(upsample_scales)
        # pyre-fixme[4]: Attribute must be annotated.
        self.indent = cin_pad * total_scale
        for scale in upsample_scales:
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            k_size = (freq_axis_kernel_size, scale * 2 + 1)
            padding = (freq_axis_padding, scale)
            stretch = Stretch2d(scale, 1, mode)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / np.prod(k_size))
            conv = nn.utils.weight_norm(conv)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
            if upsample_activation != "none":
                nonlinear = _get_activation(upsample_activation)
                self.up_layers.append(nonlinear(**upsample_activation_params))

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, c):
        """
        Args:
            c : B x C x T
        """

        # B x 1 x C x T
        c = c.unsqueeze(1)
        for f in self.up_layers:
            c = f(c)
        # B x C x T
        c = c.squeeze(1)

        if self.indent > 0:
            c = c[:, :, self.indent : -self.indent]
        return c


class ConvInUpsampleNetwork(nn.Module):
    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        upsample_scales,
        upsample_activation: str = "none",
        # pyre-fixme[2]: Parameter must be annotated.
        upsample_activation_params={},
        mode: str = "nearest",
        freq_axis_kernel_size: int = 1,
        cin_pad: int = 0,
        cin_channels: int = 80,
    ) -> None:
        super(ConvInUpsampleNetwork, self).__init__()
        # To capture wide-context information in conditional features
        # meaningless if cin_pad == 0
        ks = 2 * cin_pad + 1
        self.conv_in = nn.Conv1d(cin_channels, cin_channels, kernel_size=ks, bias=False)
        self.upsample = UpsampleNetwork(
            upsample_scales,
            upsample_activation,
            upsample_activation_params,
            mode,
            freq_axis_kernel_size,
            cin_pad=0,
            cin_channels=cin_channels,
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, c):
        c_up = self.upsample(self.conv_in(c))
        return c_up
