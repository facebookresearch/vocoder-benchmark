# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Residual stack module in MelGAN."""

from typing import Dict

import torch

from models.src.parallel_wavegan.layers.causal_conv import ( # @oss-only
# @fb-only[end= ]: from langtech.tts.vocoders.models.src.parallel_wavegan.layers.causal_conv import (
    CausalConv1d,
)


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(
        self,
        kernel_size: int = 3,
        channels: int = 32,
        dilation: int = 1,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, float] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        # pyre-fixme[2]: Parameter must be annotated.
        pad_params={},
        use_causal_conv: bool = False,
    ) -> None:
        """Initialize ResidualStack module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(ResidualStack, self).__init__()

        # defile residual stack part
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            # pyre-fixme[4]: Attribute must be annotated.
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params),
                torch.nn.Conv1d(
                    channels, channels, kernel_size, dilation=dilation, bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )
        else:
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )

        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, chennels, T).

        """
        return self.stack(c) + self.skip_layer(c)
