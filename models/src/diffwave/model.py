# pyre-strict
# pyre-fixme[51]: Mode `pyre-ignore-all-errors` is unused. This conflicts with
#  `pyre-strict` mode set on line 1.
# pyre-ignore-all-errors


# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import MEL_NUM_BANDS # @oss-only
# @fb-only: from langtech.tts.vocoders.datasets import MEL_NUM_BANDS 
from torch._tensor import Tensor
from torch.nn.modules.conv import Conv1d

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


# pyre-fixme[2]: Parameter must be annotated.
# pyre-fixme[11]: Annotation `Conv1d` is not defined as a type.
def Conv1d(*args, **kwargs) -> Conv1d:
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, max_steps) -> None:
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    # pyre-fixme[3]: Return type must be annotated.
    def forward(self, diffusion_step: Tensor):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t: Tensor) -> Tensor:
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    # pyre-fixme[2]: Parameter must be annotated.
    def _build_embedding(self, max_steps) -> Tensor:
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, n_mels) -> None:
        super().__init__()
        # pyre-fixme[6]: For 3rd argument expected `Union[Tuple[int, int], int]` but
        #  got `List[int]`.
        # pyre-fixme[6]: For 4th argument expected `Union[Tuple[int, int], int]` but
        #  got `List[int]`.
        # pyre-fixme[6]: For 5th argument expected `Union[Tuple[int, int], int]` but
        #  got `List[int]`.
        self.conv1 = ConvTranspose2d(1, 1, [3, 20], stride=[1, 10], padding=[1, 5])
        # pyre-fixme[6]: For 3rd argument expected `Union[Tuple[int, int], int]` but
        #  got `List[int]`.
        # pyre-fixme[6]: For 4th argument expected `Union[Tuple[int, int], int]` but
        #  got `List[int]`.
        # pyre-fixme[6]: For 5th argument expected `Union[Tuple[int, int], int]` but
        #  got `List[int]`.
        self.conv2 = ConvTranspose2d(1, 1, [3, 60], stride=[1, 30], padding=[1, 15])

    def forward(self, x: Tensor) -> Tensor:
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, n_mels, residual_channels: int, dilation) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(512, residual_channels)
        # pyre-fixme[4]: Attribute must be annotated.
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        # pyre-fixme[4]: Attribute must be annotated.
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, params) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.params = params
        # pyre-fixme[4]: Attribute must be annotated.
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.spectrogram_upsampler = SpectrogramUpsampler(MEL_NUM_BANDS)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    MEL_NUM_BANDS,
                    params.residual_channels,
                    2 ** (i % params.dilation_cycle_length),
                )
                for i in range(params.residual_layers)
            ]
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.skip_projection = Conv1d(
            params.residual_channels, params.residual_channels, 1
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, audio, spectrogram, diffusion_step):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, spectrogram, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
