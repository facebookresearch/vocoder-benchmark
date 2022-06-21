# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
WaveGrad Neural Vocoder.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio

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

from models.src.wavegrad import diffusion_process # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavegrad import diffusion_process 
from omegaconf import MISSING
from torch import Tensor


@dataclass
class NoiseSchedulerConfig:
    """
    Configuration for the WaveGrad noise scheduler.
    """

    n_iter: int = MISSING
    betas_range: List[float] = MISSING


@dataclass
class ModelConfig:
    """
    Configuration for the WaveGrad model.
    """

    factors: List[int] = MISSING
    upsampling_preconv_out_channels: int = MISSING
    upsampling_out_channels: List[int] = MISSING
    upsampling_dilations: List[Any] = MISSING
    downsampling_preconv_out_channels: int = MISSING
    downsampling_out_channels: List[int] = MISSING
    downsampling_dilations: List[Any] = MISSING
    n_iterations: int = MISSING
    grad_clip_threshold: int = MISSING
    scheduler_step_size: int = MISSING
    scheduler_gamma: float = MISSING
    learning_rate: float = MISSING
    noise_schedule_interval: int = MISSING
    training_noise_schedule: NoiseSchedulerConfig = MISSING
    test_noise_schedule: NoiseSchedulerConfig = MISSING


@dataclass
class Config:
    """
    Configuration for the WaveGrad model and dataset.
    """

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING


class WaveGrad(Vocoder):
    """
    WaveGrad model.
    """

    command: str = "wavegrad"

    def __init__(self, config: Config) -> None:
        """
        Create a new WaveGrad.
        """
        super().__init__(config)

        self.config = config
        self.model = torch.nn.DataParallel(diffusion_process.WaveGrad(config))
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.model.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.model.scheduler_step_size,
            gamma=config.model.scheduler_gamma,
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
        return [(self.optimizer, self.scheduler)]

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
        # pyre-fixme[29]: `Union[torch._tensor.Tensor,
        #  torch.nn.modules.module.Module]` is not a function.
        return self.model.module.compute_loss(spectrograms, waveforms)

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
        if not self.global_step % self.config.model.noise_schedule_interval:
            # pyre-fixme[29]: `Union[torch._tensor.Tensor,
            #  torch.nn.modules.module.Module]` is not a function.
            self.model.module.set_new_noise_schedule(
                init=torch.linspace,
                init_kwargs={
                    "steps": self.config.model.training_noise_schedule.n_iter,
                    "start": self.config.model.training_noise_schedule.betas_range[0],
                    "end": self.config.model.training_noise_schedule.betas_range[1],
                },
            )
        self.model.zero_grad()

        # Forward pass.
        loss = self.loss(spectrograms, waveforms)

        # Backward pass.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Logs.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=self.parameters(),
            max_norm=self.config.model.grad_clip_threshold,
        )
        loss_stats = {"total_loss": loss, "grad_norm": grad_norm}

        return loss, loss_stats

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

        # pyre-fixme[29]: `Union[torch._tensor.Tensor,
        #  torch.nn.modules.module.Module]` is not a function.
        self.model.module.set_new_noise_schedule(
            init=torch.linspace,
            init_kwargs={
                "steps": self.config.model.test_noise_schedule.n_iter,
                "start": self.config.model.test_noise_schedule.betas_range[0],
                "end": self.config.model.test_noise_schedule.betas_range[1],
            },
        )

        return {
            "nll_loss": self.loss(spectrograms, waveforms),
        }

    # pyre-fixme[14]: `generate` overrides method defined in `Vocoder` inconsistently.
    def generate(self, spectrograms: Tensor, training: bool = False) -> Tensor:
        self.model.eval()

        if training:
            spectrograms = spectrograms[:, :, :200]

        with torch.no_grad():
            # pyre-fixme[29]: `Union[torch._tensor.Tensor,
            #  torch.nn.modules.module.Module]` is not a function.
            self.model.module.set_new_noise_schedule(
                init=torch.linspace,
                init_kwargs={
                    "steps": self.config.model.test_noise_schedule.n_iter,
                    "start": self.config.model.test_noise_schedule.betas_range[0],
                    "end": self.config.model.test_noise_schedule.betas_range[1],
                },
            )
            output = self.model.module.forward(
                spectrograms, store_intermediate_states=False
            )

        self.model.train()
        return output.flatten()

    def get_complexity(
        self,
    ) -> List[float]:
        """
        Returns A list with the number of FLOPS and parameters used in this model.
        """

        # Prepare the input format.
        spectrograms = torch.rand(
            1,
            MEL_NUM_BANDS,
            int(AUDIO_SAMPLE_RATE / MEL_HOP_SAMPLES)
            + 2 * self.config.dataset.padding_frames,
        )

        if torch.cuda.is_available():
            spectrograms = spectrograms.cuda()

        # Feed data to network and compute the model complexity.
        with torch.no_grad():
            # pyre-fixme[29]: `Union[torch._tensor.Tensor,
            #  torch.nn.modules.module.Module]` is not a function.
            self.model.module.set_new_noise_schedule(
                init=torch.linspace,
                init_kwargs={
                    "steps": self.config.model.test_noise_schedule.n_iter,
                    "start": self.config.model.test_noise_schedule.betas_range[0],
                    "end": self.config.model.test_noise_schedule.betas_range[1],
                },
            )
            flops, n_params = get_model_complexity_info(
                self.model.module, ([spectrograms])
            )
            n_iter = self.config.model.test_noise_schedule.n_iter

            return [flops / n_iter, n_params]
