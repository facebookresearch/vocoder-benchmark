# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors

# license information for WaveGrad can be found at https://github.com/ivanvovk/WaveGrad/blob/master/LICENSE

import torch


class BaseModule(torch.nn.Module):
    def __init__(self) -> None:
        super(BaseModule, self).__init__()

    @property
    def nparams(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
