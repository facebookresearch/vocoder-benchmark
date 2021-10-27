# pyre-ignore-all-errors
# pylint: skip-file
# license information for WaveGrad can be found at https://github.com/ivanvovk/WaveGrad/blob/master/LICENSE
# Copyright (c) Facebook, Inc. All Rights Reserved

import torch


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
