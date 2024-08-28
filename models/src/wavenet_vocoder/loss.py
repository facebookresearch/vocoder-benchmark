# pyre-ignore-all-errors


import torch

from models.src.wavenet_vocoder.mixture import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavenet_vocoder.mixture import ( 
    discretized_mix_logistic_loss,
    mix_gaussian_loss,
)
from torch import nn


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class DiscretizedMixturelogisticLoss(nn.Module):
    def __init__(self, config) -> None:
        self.config = config
        super(DiscretizedMixturelogisticLoss, self).__init__()

    def forward(self, input, target, log_scale_min: float = -7.0):
        losses = discretized_mix_logistic_loss(
            input,
            target,
            num_classes=self.config.model.quantize_channels,
            log_scale_min=log_scale_min,
            reduce=False,
        )
        assert losses.size() == target.size()
        return losses.mean()


class MixtureGaussianLoss(nn.Module):
    def __init__(self, config) -> None:
        self.config = config
        super(MixtureGaussianLoss, self).__init__()

    def forward(self, input, target, log_scale_min: float = -7.0):
        losses = mix_gaussian_loss(
            input, target, log_scale_min=log_scale_min, reduce=False
        )
        assert losses.size() == target.size()
        return losses.mean()
