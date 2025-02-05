# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


import torch

from models.src.wavegrad.base import BaseModule # @oss-only
# @fb-only[end= ]: from langtech.tts.vocoders.models.src.wavegrad.base import BaseModule

from models.src.wavegrad.interpolation import ( # @oss-only
# @fb-only[end= ]: from langtech.tts.vocoders.models.src.wavegrad.interpolation import (
    InterpolationBlock,
)

from models.src.wavegrad.layers import ( # @oss-only
# @fb-only[end= ]: from langtech.tts.vocoders.models.src.wavegrad.layers import (
    Conv1dWithInitialization,
)


class ConvolutionBlock(BaseModule):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, in_channels, out_channels, dilation) -> None:
        super(ConvolutionBlock, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        outputs = self.leaky_relu(x)
        outputs = self.convolution(outputs)
        return outputs


class DownsamplingBlock(BaseModule):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, in_channels, out_channels, factor, dilations) -> None:
        super(DownsamplingBlock, self).__init__()
        in_sizes = [in_channels] + [out_channels for _ in range(len(dilations) - 1)]
        out_sizes = [out_channels for _ in range(len(in_sizes))]
        self.main_branch = torch.nn.Sequential(
            *(
                [
                    InterpolationBlock(
                        scale_factor=factor,
                        mode="linear",
                        align_corners=False,
                        downsample=True,
                    )
                ]
                + [
                    ConvolutionBlock(in_size, out_size, dilation)
                    for in_size, out_size, dilation in zip(
                        in_sizes, out_sizes, dilations
                    )
                ]
            )
        )
        self.residual_branch = torch.nn.Sequential(
            *[
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                ),
                InterpolationBlock(
                    scale_factor=factor,
                    mode="linear",
                    align_corners=False,
                    downsample=True,
                ),
            ]
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        outputs = self.main_branch(x)
        outputs = outputs + self.residual_branch(x)
        return outputs
