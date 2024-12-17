# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Causal convolusion layer modules."""

from typing import Dict

import torch


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module with customized initialization."""

    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        in_channels,
        # pyre-fixme[2]: Parameter must be annotated.
        out_channels,
        # pyre-fixme[2]: Parameter must be annotated.
        kernel_size,
        dilation: int = 1,
        bias: bool = True,
        pad: str = "ConstantPad1d",
        pad_params: Dict[str, float] = {"value": 0.0},
    ) -> None:
        """Initialize CausalConv1d module."""
        super(CausalConv1d, self).__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        return self.conv(self.pad(x))[:, :, : x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(
        # pyre-fixme[2]: Parameter must be annotated.
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        in_channels,
        # pyre-fixme[2]: Parameter must be annotated.
        out_channels,
        # pyre-fixme[2]: Parameter must be annotated.
        kernel_size,
        # pyre-fixme[2]: Parameter must be annotated.
        stride,
        bias: bool = True,
    ) -> None:
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, bias=bias
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.stride = stride

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).

        Returns:
            Tensor: Output tensor (B, out_channels, T_out).

        """
        return self.deconv(x)[:, :, : -self.stride]
