# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


import torch

from models.src.wavegrad.base import BaseModule # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavegrad.base import BaseModule 


class Conv1dWithInitialization(BaseModule):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, **kwargs) -> None:
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        return self.conv1d(x)
