# pyre-ignore-all-errors
# pylint: skip-file

import torch

from models.src.wavegrad.base import BaseModule # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavegrad.base import BaseModule 

from models.src.wavegrad.interpolation import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavegrad.interpolation import ( 
    InterpolationBlock,
)

from models.src.wavegrad.layers import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavegrad.layers import ( 
    Conv1dWithInitialization,
)

from models.src.wavegrad.linear_modulation import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavegrad.linear_modulation import ( 
    FeatureWiseAffine,
)


class BasicModulationBlock(BaseModule):
    """
    Linear modulation part of UBlock, represented by sequence of the following layers:
        - Feature-wise Affine
        - LReLU
        - 3x1 Conv
    """

    def __init__(self, n_channels, dilation) -> None:
        super(BasicModulationBlock, self).__init__()
        self.featurewise_affine = FeatureWiseAffine()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x, scale, shift):
        outputs = self.featurewise_affine(x, scale, shift)
        outputs = self.leaky_relu(outputs)
        outputs = self.convolution(outputs)
        return outputs


class UpsamplingBlock(BaseModule):
    def __init__(self, in_channels, out_channels, factor, dilations) -> None:
        super(UpsamplingBlock, self).__init__()
        self.first_block_main_branch = torch.nn.ModuleDict(
            {
                "upsampling": torch.nn.Sequential(
                    *[
                        torch.nn.LeakyReLU(0.2),
                        InterpolationBlock(
                            scale_factor=factor, mode="linear", align_corners=False
                        ),
                        Conv1dWithInitialization(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=dilations[0],
                            dilation=dilations[0],
                        ),
                    ]
                ),
                "modulation": BasicModulationBlock(out_channels, dilation=dilations[1]),
            }
        )
        self.first_block_residual_branch = torch.nn.Sequential(
            *[
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                ),
                InterpolationBlock(
                    scale_factor=factor, mode="linear", align_corners=False
                ),
            ]
        )
        self.second_block_main_branch = torch.nn.ModuleDict(
            {
                f"modulation_{idx}": BasicModulationBlock(
                    out_channels, dilation=dilations[2 + idx]
                )
                for idx in range(2)
            }
        )

    def forward(self, x, scale, shift):
        # First upsampling residual block
        outputs = self.first_block_main_branch["upsampling"](x)
        outputs = self.first_block_main_branch["modulation"](outputs, scale, shift)
        outputs = outputs + self.first_block_residual_branch(x)

        # Second residual block
        residual = self.second_block_main_branch["modulation_0"](outputs, scale, shift)
        outputs = outputs + self.second_block_main_branch["modulation_1"](
            residual, scale, shift
        )
        return outputs
