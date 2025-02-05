# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


import torch

from models.src.wavegrad.base import BaseModule # @oss-only
# @fb-only[end= ]: from langtech.tts.vocoders.models.src.wavegrad.base import BaseModule


class InterpolationBlock(BaseModule):
    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        scale_factor,
        mode: str = "linear",
        align_corners: bool = False,
        downsample: bool = False,
    ) -> None:
        super(InterpolationBlock, self).__init__()
        self.downsample = downsample
        # pyre-fixme[4]: Attribute must be annotated.
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        outputs = torch.nn.functional.interpolate(
            x,
            size=(
                x.shape[-1] * self.scale_factor
                if not self.downsample
                else x.shape[-1] // self.scale_factor
            ),
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False,
        )
        return outputs
