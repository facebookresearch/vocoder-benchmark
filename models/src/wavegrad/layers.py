# pyre-ignore-all-errors
# pylint: skip-file

import torch
from langtech.tts.vocoders.models.src.wavegrad.base import BaseModule


class Conv1dWithInitialization(BaseModule):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)
