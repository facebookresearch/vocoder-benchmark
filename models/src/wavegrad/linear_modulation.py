# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


import torch

from models.src.wavegrad.base import BaseModule # @oss-only
# @fb-only[end= ]: from langtech.tts.vocoders.models.src.wavegrad.base import BaseModule

from models.src.wavegrad.layers import ( # @oss-only
# @fb-only[end= ]: from langtech.tts.vocoders.models.src.wavegrad.layers import (
    Conv1dWithInitialization,
)
from torch._tensor import Tensor


LINEAR_SCALE = 5000


class PositionalEncoding(BaseModule):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, n_channels) -> None:
        super(PositionalEncoding, self).__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.n_channels = n_channels

    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, noise_level) -> Tensor:
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(
            half_dim
        )
        exponents = 1e-4**exponents
        exponents = LINEAR_SCALE * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        # pyre-fixme[16]: `int` has no attribute `sin`.
        # pyre-fixme[16]: `int` has no attribute `cos`.
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class FeatureWiseLinearModulation(BaseModule):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, in_channels, out_channels, input_dscaled_by) -> None:
        super(FeatureWiseLinearModulation, self).__init__()
        self.signal_conv = torch.nn.Sequential(
            *[
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2),
            ]
        )
        self.positional_encoding = PositionalEncoding(in_channels)
        self.scale_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.shift_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x, noise_level):
        outputs = self.signal_conv(x)
        outputs = outputs + self.positional_encoding(noise_level).unsqueeze(-1)
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        return scale, shift


class FeatureWiseAffine(BaseModule):
    def __init__(self) -> None:
        super(FeatureWiseAffine, self).__init__()

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x, scale, shift):
        outputs = scale * x + shift
        return outputs
