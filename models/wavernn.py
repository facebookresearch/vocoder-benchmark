# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
WaveRNN Neural Vocoder.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.models

from datasets import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.datasets import ( 
    AUDIO_SAMPLE_RATE,
    DatasetConfig,
    MEL_HOP_SAMPLES,
    MEL_NUM_BANDS,
)

from models.framework import Vocoder, ConfigProtocol # @oss-only
# @fb-only: from langtech.tts.vocoders.models.framework import ConfigProtocol, Vocoder 

from models.src.ptflops.flops_counter import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.ptflops.flops_counter import ( 
    get_model_complexity_info,
)

from models.src.wavenet_vocoder import lrschedule # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavenet_vocoder import lrschedule 

from utils import remove_none_values_from_dict # @oss-only
# @fb-only: from langtech.tts.vocoders.utils import remove_none_values_from_dict 
from omegaconf import MISSING, OmegaConf
from torch import Tensor
from torch._tensor import Tensor
from torch.distributions.categorical import Categorical
from tqdm import tqdm


@dataclass
class SchedulerConfig:
    """
    Configuration for the WaveRNN scheduler.
    """

    anneal_rate: Optional[float] = None
    anneal_interval: Optional[int] = None
    warmup_steps: Optional[int] = None
    T: Optional[int] = None
    M: Optional[int] = None


@dataclass
class ModelConfig:
    """
    Configuration for the WaveRNN model.
    """

    upsample_scales: List[int] = MISSING
    n_classes: int = MISSING
    n_res_block: int = MISSING
    n_rnn: int = MISSING
    n_fc: int = MISSING
    kernel_size: int = MISSING
    n_hidden: int = MISSING
    n_output: int = MISSING
    n_iterations: int = MISSING
    learning_rate: float = MISSING
    lr_schedule: Optional[str] = None
    lr_schedule_kwargs: Optional[SchedulerConfig] = None


@dataclass
class Config:
    """
    Configuration for the WaveRNN model and dataset.
    """

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING


class WaveRNN(Vocoder):
    """
    WaveRNN model.
    """

    command: str = "wavernn"

    def __init__(self, config: Config) -> None:
        """
        Create a new WaveRNN.
        """
        super().__init__(config)

        self.config = config
        # pyre-fixme[4]: Attribute must be annotated.
        self.model = torch.nn.DataParallel(
            torchaudio.models.WaveRNN(
                n_res_block=config.model.n_res_block,
                n_rnn=config.model.n_rnn,
                n_fc=config.model.n_fc,
                kernel_size=config.model.kernel_size,
                n_hidden=config.model.n_hidden,
                n_output=config.model.n_output,
                upsample_scales=config.model.upsample_scales,
                n_classes=config.model.n_classes,
                hop_length=MEL_HOP_SAMPLES,
                n_freq=MEL_NUM_BANDS,
            )
        )
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.model.learning_rate
        )
        self.criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
        self.compand: torch.nn.Module = torchaudio.transforms.MuLawEncoding()
        self.expand: torch.nn.Module = torchaudio.transforms.MuLawDecoding()

    @staticmethod
    def default_config() -> ConfigProtocol:
        """
        Returns the OmegaConf config for this model.
        """
        return Config()

    def get_optimizers(
        self,
    ) -> List[
        Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]
    ]:
        """
        Get optimizers and schedulers used in this model. This is a method
        instead of just a pair of attributes because GANs tend to have
        multiple optimizers.

        Returns:
          A list of tuples. Each tuple contains an optimizer used by the
          model and an optional scheduler used for that optimizer. These are
          saved and restored from the checkpoints.
        """
        return [(self.optimizer, None)]

    def is_done(self) -> bool:
        """
        Checks if a model is done training.

        Returns:
          Whether the model is done training.
        """
        return self.global_step >= self.config.model.n_iterations

    def initialize(self) -> None:
        """
        Called after model creation.
        """

    def loss(self, spectrograms: Tensor, waveforms: Tensor) -> Tensor:
        """
        Compute loss on a batch.

        Returns:
          The negative log likelihood loss.
        """
        target = self.compand(waveforms[:, 1:]).long()

        # Forward pass.

        waveforms = self.compand(waveforms)
        # pyre-fixme[9]: waveforms has type `Tensor`; used as `float`.
        waveforms = self.label_2_float(waveforms, self.model.module.n_classes)
        output = self.model(waveforms.unsqueeze(1), spectrograms.unsqueeze(1))
        output = output.squeeze(1)[:, :-1, :]
        loss = self.criterion(output.transpose(1, 2), target)

        return loss

    # pyre-fixme[14]: `train_step` overrides method defined in `Vocoder` inconsistently.
    def train_step(
        self, spectrograms: Tensor, waveforms: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Runs a single train step of the model.

        Returns:
          A tuple containing overall model loss and a list of losses to log
          to Tensorboard. The first loss is printed to the console and logged
          to Tensorboard.
        """
        # Forward pass.
        loss = self.loss(spectrograms, waveforms)

        # Learning rate schedule
        if self.config.model.lr_schedule:
            current_lr = self.config.model.learning_rate
            lr_schedule_fn = getattr(lrschedule, self.config.model.lr_schedule)
            lr_schedule_kwargs = remove_none_values_from_dict(
                OmegaConf.to_container(self.config.model.lr_schedule_kwargs)
            )
            current_lr = lr_schedule_fn(
                current_lr, self.global_step, **lr_schedule_kwargs
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        # Backward pass if nan loss is not detected.
        if not torch.isnan(loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss, {}

        print("Nan loss found. Back propagation step is skipped for this iteration.")
        return loss.new_zeros([1]), {}

    # pyre-fixme[14]: `validation_losses` overrides method defined in `Vocoder`
    #  inconsistently.
    def validation_losses(
        self, spectrograms: Tensor, waveforms: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute validation losses.

        Returns:
          A dictionary mapping loss name (e.g. 'nll_loss') to the validation value.
        """
        return {
            "nll_loss": self.loss(spectrograms, waveforms),
        }

    # pyre-fixme[14]: `generate` overrides method defined in `Vocoder` inconsistently.
    def generate(self, spectrograms: Tensor, training: bool = False) -> Tensor:
        """
        Generate a sample from this model.

        Returns:
          A 1D float tensor containing the output waveform.
        """
        self.model.eval()
        if training:
            spectrograms = spectrograms[:, :, :200]
        output = []

        rnn1 = self.get_gru_cell(self.model.module.rnn1)
        rnn2 = self.get_gru_cell(self.model.module.rnn2)

        with torch.no_grad():
            spectrograms, aux = self.model.module.upsample(spectrograms)
            spectrograms = spectrograms.transpose(1, 2)
            aux = aux.transpose(1, 2)
            batch_size, seq_len, _ = spectrograms.size()
            h1 = spectrograms.new_zeros(batch_size, self.model.module.n_rnn)
            h2 = spectrograms.new_zeros(batch_size, self.model.module.n_rnn)
            x = spectrograms.new_zeros(batch_size, 1)

            d = self.model.module.n_aux
            aux_split = [aux[:, :, d * i : d * (i + 1)] for i in range(4)]

            for i in tqdm(range(seq_len)):
                m_t = spectrograms[:, i, :]

                a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.model.module.fc(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.model.module.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.model.module.fc2(x))

                logits = self.model.module.fc3(x)

                # pyre-fixme[6]: For 1st argument expected `Categorical` but got `str`.
                x, output = self.get_x_from_dist("random", logits, output)

        output = torch.stack(output).transpose(0, 1)
        output = self.expand(output.flatten())

        self.model.train()

        return output

    def get_gru_cell(self, gru: torch.nn.Module) -> torch.nn.Module:
        """
        Create a GRU cell.

        Args:
          gru: The GRU Cell.

        Returns:
          The modified GRU cell.
        """
        gru_cell = torch.nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def get_x_from_dist(self, distrib: Categorical, logits: Tensor, history=None):
        """
        Sampling from a given distribution

        Returns:
            a tuple of current sample x and history of an array of previous samples
        """
        if history is None:
            history = []
        if distrib == "argmax":
            x = torch.argmax(logits, dim=1)
            history.append(x)
            # pyre-fixme[16]: `float` has no attribute `unsqueeze`.
            # pyre-fixme[6]: For 1st argument expected `int` but got `Tensor`.
            x = self.label_2_float(x, self.model.module.n_classes).unsqueeze(-1)
        elif distrib == "random":
            posterior = F.softmax(logits, dim=1)
            distrib = torch.distributions.Categorical(posterior)
            x = distrib.sample().float()
            history.append(x)
            x = self.label_2_float(x, self.model.module.n_classes).unsqueeze(-1)
        else:
            raise RuntimeError("Unknown sampling mode - ", distrib)
        return x, history

    # pyre-fixme[2]: Parameter must be annotated.
    def label_2_float(self, x: int, n_classes) -> float:
        return 2 * x / (n_classes - 1.0) - 1.0

    def get_complexity(
        self,
    ) -> List[float]:
        """
        Return A list with the number of FLOPS and parametrs used in this model.
        """

        # Prepare the input format.
        model = self.model.module
        waveforms = torch.rand(1, AUDIO_SAMPLE_RATE)
        spectrograms = torch.rand(
            1,
            MEL_NUM_BANDS,
            int(AUDIO_SAMPLE_RATE / MEL_HOP_SAMPLES)
            + 2 * self.config.dataset.padding_frames,
        )
        batch_size = waveforms.size(0)
        h1 = torch.zeros(
            1,
            batch_size,
            model.n_rnn,
            dtype=waveforms.dtype,
            device=waveforms.device,
        )
        h2 = torch.zeros(
            1,
            batch_size,
            model.n_rnn,
            dtype=waveforms.dtype,
            device=waveforms.device,
        )

        stats = np.array([0.0, 0.0])

        if torch.cuda.is_available():
            waveforms = waveforms.cuda()
            spectrograms = spectrograms.cuda()
            h1 = h1.cuda()
            h2 = h2.cuda()

        # Feed data to network and compute the model complexity.
        stats += np.array(
            get_model_complexity_info(
                model.upsample,
                ([spectrograms]),
            )
        )
        spectrograms, aux = model.upsample(spectrograms)
        spectrograms = spectrograms.transpose(1, 2)
        aux = aux.transpose(1, 2)

        aux_idx = [model.n_aux * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0] : aux_idx[1]]
        a2 = aux[:, :, aux_idx[1] : aux_idx[2]]
        a3 = aux[:, :, aux_idx[2] : aux_idx[3]]
        a4 = aux[:, :, aux_idx[3] : aux_idx[4]]

        x = torch.cat([waveforms.unsqueeze(-1), spectrograms, a1], dim=-1)
        stats += np.array(
            get_model_complexity_info(
                model.fc,
                ([x]),
            )
        )
        x = model.fc(x)

        res = x
        stats += np.array(
            get_model_complexity_info(
                model.rnn1,
                ([x, h1]),
            )
        )
        x, _ = model.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=-1)

        stats += np.array(
            get_model_complexity_info(
                model.rnn2,
                ([x, h2]),
            )
        )
        x, _ = model.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=-1)
        stats += np.array(
            get_model_complexity_info(
                model.fc1,
                ([x]),
            )
        )
        x = model.fc1(x)

        stats += np.array(
            get_model_complexity_info(
                model.relu1,
                ([x]),
            )
        )
        x = model.relu1(x)

        x = torch.cat([x, a4], dim=-1)
        stats += np.array(
            get_model_complexity_info(
                model.fc2,
                ([x]),
            )
        )
        x = model.fc2(x)

        stats += np.array(
            get_model_complexity_info(
                model.relu2,
                ([x]),
            )
        )
        x = model.relu2(x)

        stats += np.array(
            get_model_complexity_info(
                model.fc3,
                ([x]),
            )
        )
        x = model.fc3(x)

        # pyre-fixme[7]: Expected `List[float]` but got `ndarray[typing.Any,
        #  dtype[typing.Any]]`.
        return stats
