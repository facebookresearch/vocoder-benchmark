# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
WaveNet Neural Vocoder.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import models.src.wavenet_vocoder.wavenet as wavenet # @oss-only
# @fb-only: import langtech.tts.vocoders.models.src.wavenet_vocoder.wavenet as wavenet 
import numpy as np
import torch
import torchaudio
import torchaudio.models

from datasets import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.datasets import ( 
    DatasetConfig,
    MEL_NUM_BANDS,
    MEL_HOP_SAMPLES,
    AUDIO_SAMPLE_RATE,
)

from models.framework import Vocoder, ConfigProtocol # @oss-only
# @fb-only: from langtech.tts.vocoders.models.framework import Vocoder, ConfigProtocol 

from models.src.ptflops.flops_counter import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.ptflops.flops_counter import ( 
    get_model_complexity_info,
    conv_flops_counter_hook,
)

from models.src.wavenet_vocoder import lrschedule # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavenet_vocoder import lrschedule 

from models.src.wavenet_vocoder.conv import Conv1d # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavenet_vocoder.conv import Conv1d 

from models.src.wavenet_vocoder.loss import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavenet_vocoder.loss import ( 
    DiscretizedMixturelogisticLoss,
    MixtureGaussianLoss,
)

from models.src.wavenet_vocoder.mixture import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavenet_vocoder.mixture import ( 
    sample_from_discretized_mix_logistic,
    sample_from_mix_gaussian,
)

from models.src.wavenet_vocoder.util import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.wavenet_vocoder.util import ( 
    is_mulaw_quantize,
    is_scalar_input,
)

from utils import remove_none_values_from_dict # @oss-only
# @fb-only: from langtech.tts.vocoders.utils import remove_none_values_from_dict 
from omegaconf import MISSING, OmegaConf
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm


@dataclass
class SchedulerConfig:
    """
    Configuration for the WaveNet scheduler.
    """

    anneal_rate: Optional[float] = None
    anneal_interval: Optional[int] = None
    warmup_steps: Optional[int] = None
    T: Optional[int] = None
    M: Optional[int] = None


@dataclass
class UpsampleConfig:
    """
    Configuration for the WaveNet model.
    """

    upsample_scales: List[int] = MISSING
    cin_channels: int = MISSING
    cin_pad: int = MISSING


@dataclass
class ModelConfig:
    """
    Configuration for the WaveNet model.
    """

    quantize_channels: int = MISSING
    out_channels: int = MISSING
    layers: int = MISSING
    stacks: int = MISSING
    residual_channels: int = MISSING
    gate_channels: int = MISSING
    skip_out_channels: int = MISSING
    cin_channels: int = MISSING
    gin_channels: int = MISSING
    n_speakers: int = MISSING
    dropout: float = MISSING
    kernel_size: int = MISSING
    cin_pad: int = MISSING
    upsample_conditional_features: bool = MISSING
    input_type: str = MISSING
    output_distribution: str = MISSING
    n_iterations: int = MISSING
    learning_rate: float = MISSING
    upsample_params: UpsampleConfig = MISSING
    lr_schedule: Optional[str] = None
    lr_schedule_kwargs: Optional[SchedulerConfig] = None


@dataclass
class Config:
    """
    Configuration for the WaveNet model and dataset.
    """

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING


class WaveNet(Vocoder):
    """
    WaveNet model.
    """

    command: str = "wavenet"

    def __init__(self, config: Config) -> None:
        """
        Create a new WaveNet.
        """
        super().__init__(config)

        self.config = config
        self.scalar_input: bool = is_scalar_input(config.model.input_type)

        self.model = torch.nn.DataParallel(
            wavenet.WaveNet(
                out_channels=config.model.out_channels,
                layers=config.model.layers,
                stacks=config.model.stacks,
                residual_channels=config.model.residual_channels,
                gate_channels=config.model.gate_channels,
                skip_out_channels=config.model.skip_out_channels,
                cin_channels=config.model.cin_channels,
                gin_channels=config.model.gin_channels,
                n_speakers=config.model.n_speakers,
                dropout=config.model.dropout,
                kernel_size=config.model.kernel_size,
                cin_pad=config.model.cin_pad,
                upsample_conditional_features=config.model.upsample_conditional_features,
                upsample_params=config.model.upsample_params,
                scalar_input=self.scalar_input,
                output_distribution=config.model.output_distribution,
            )
        )

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.model.learning_rate
        )
        self.compand: torch.nn.Module = torchaudio.transforms.MuLawEncoding(
            config.model.quantize_channels
        )
        self.expand: torch.nn.Module = torchaudio.transforms.MuLawDecoding(
            config.model.quantize_channels
        )

        if is_mulaw_quantize(config.model.input_type):
            self.criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
        else:
            if config.model.output_distribution == "Logistic":
                self.criterion: torch.nn.Module = DiscretizedMixturelogisticLoss(config)
            elif config.model.output_distribution == "Normal":
                self.criterion: torch.nn.Module = MixtureGaussianLoss(config)
            else:
                raise RuntimeError(
                    "Not supported output distribution type: {}".format(
                        config.model.output_distribution
                    )
                )

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

        # Forward pass.

        target = waveforms[:, 1:]  # [batch_size, n_samples-1]

        if self.config.model.input_type in ["mulaw", "mulaw-quantize"]:
            waveforms = self.compand(waveforms)  # [batch_size, n_samples]
            target = self.compand(target)

            if self.config.model.input_type == "mulaw":
                waveforms = self.label_2_float(
                    waveforms, self.config.model.quantize_channels
                )
                waveforms = waveforms.unsqueeze(1)

                target = self.label_2_float(target, self.config.model.quantize_channels)
            else:
                waveforms = F.one_hot(waveforms, self.config.model.quantize_channels)
                waveforms = waveforms.transpose(1, 2).float()

        elif self.config.model.input_type == "raw":
            waveforms = waveforms.unsqueeze(1)
        else:
            raise RuntimeError(
                "Not supported input type: {}".format(self.config.model.input_type)
            )

        output = self.model(waveforms, c=spectrograms)
        if self.config.model.input_type in ["mulaw", "raw"]:
            target = target.unsqueeze(2)  # [batch_size, n_samples-1, 1]
        loss = self.criterion(output[:, :, :-1], target)

        return loss

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

        # Backward pass.
        self.optimizer.zero_grad()
        loss.backward()
        # pyre-fixme[20]: Argument `closure` expected.
        self.optimizer.step()
        return loss, {}

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

    def generate(self, spectrograms: Tensor, training: bool = False) -> Tensor:
        """
        Generate a sample from this model.

        Returns:
          A 1D float tensor containing the output waveform.
        """
        self.model.eval()
        self.model.module.clear_buffer()

        with torch.no_grad():
            spectrograms = self.model.module.upsample_net(spectrograms)
            seq_len = (
                22050 if training else spectrograms.size(-1)
            )  # synthesize the first second only during training
            batch_size = spectrograms.shape[0]
            spectrograms = spectrograms.transpose(1, 2).contiguous()

            if self.scalar_input:
                x = spectrograms.new_zeros(batch_size, 1, 1)
            else:
                x = spectrograms.new_zeros(
                    batch_size, 1, self.config.model.out_channels
                )

            output = []
            for t in tqdm(range(seq_len)):
                # Conditioning features for single time step
                ct = spectrograms[:, t, :].unsqueeze(1)
                x = self.model.module.first_conv.incremental_forward(x)
                skips = 0
                for f in self.model.module.conv_layers:
                    x, h = f.incremental_forward(x, ct, None)
                    skips += h
                skips *= math.sqrt(1.0 / len(self.model.module.conv_layers))
                x = skips
                for f in self.model.module.last_conv_layers:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError:
                        x = f(x)
                x, output = self.get_x_from_dist(
                    self.config.model.output_distribution,
                    x,
                    history=output,
                    B=batch_size,
                )
        output = torch.stack(output).transpose(0, 1)

        if self.config.model.input_type in ["mulaw", "mulaw-quantize"]:
            if self.config.model.input_type == "mulaw-quantize":
                output = torch.argmax(output, dim=2)
            else:
                output = self.float_2_label(output, self.config.model.quantize_channels)
            output = self.expand(output.long())
        elif self.config.model.input_type != "raw":
            raise RuntimeError(
                "Not supported input type: {}".format(self.config.model.input_type)
            )
        self.model.module.clear_buffer()
        self.model.train()

        return output.flatten()

    def get_x_from_dist(
        self,
        distrib: str,
        logits: Tensor,
        history: List[Tensor],
        B: int,
        softmax: bool = True,
        quantize: bool = True,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Sampling from a given distribution

        Returns:
            a tuple of current sample x and history of an array of previous samples
        """
        # Generate next input by sampling
        if self.scalar_input:
            if distrib == "Logistic":
                x = sample_from_discretized_mix_logistic(logits.view(B, -1, 1))
            elif distrib == "Normal":
                x = sample_from_mix_gaussian(logits.view(B, -1, 1))
            else:
                raise AssertionError()
        else:
            x = F.softmax(logits.view(B, -1), dim=1) if softmax else logits.view(B, -1)
            if quantize:
                dist = torch.distributions.OneHotCategorical(x)
                x = dist.sample()

        history.append(x.data)
        return history[-1], history

    def label_2_float(self, x, n_classes):
        return 2 * x / (n_classes - 1.0) - 1.0

    def float_2_label(self, x, n_classes):
        return (x + 1.0) * (n_classes - 1) / 2

    def get_complexity(
        self,
    ) -> List[float]:
        """
        Returns A list with the number of FLOPS and parameters used in this model.
        """

        # Prepare the input format.
        waveforms = torch.rand(1, AUDIO_SAMPLE_RATE)
        spectrograms = torch.rand(
            1,
            MEL_NUM_BANDS,
            int(AUDIO_SAMPLE_RATE / MEL_HOP_SAMPLES)
            + 2 * self.config.dataset.padding_frames,
        )

        if self.config.model.input_type in ["mulaw", "mulaw-quantize"]:
            waveforms = self.compand(waveforms)  # [batch_size, n_samples]

            if self.config.model.input_type == "mulaw":
                waveforms = self.label_2_float(
                    waveforms, self.config.model.quantize_channels
                )
                waveforms = waveforms.unsqueeze(1)
            else:
                waveforms = F.one_hot(waveforms, self.config.model.quantize_channels)
                waveforms = waveforms.transpose(1, 2).float()

        elif self.config.model.input_type == "raw":
            waveforms = waveforms.unsqueeze(1)
        else:
            raise RuntimeError(
                "Not supported input type: {}".format(self.config.model.input_type)
            )

        model = self.model.module
        custom_modules_hooks = {Conv1d: conv_flops_counter_hook}
        stats = np.array([0.0, 0.0])

        if torch.cuda.is_available():
            waveforms = waveforms.cuda()
            spectrograms = spectrograms.cuda()

        # Feed data to network and compute the model complexity.
        with torch.no_grad():
            stats += np.array(
                get_model_complexity_info(
                    model.upsample_net,
                    ([spectrograms]),
                    custom_modules_hooks=custom_modules_hooks,
                )
            )

            spectrograms = model.upsample_net(spectrograms)
            assert spectrograms.size(-1) == waveforms.size(-1)

            stats += np.array(
                get_model_complexity_info(
                    model.first_conv,
                    ([waveforms]),
                    custom_modules_hooks=custom_modules_hooks,
                )
            )

            waveforms = model.first_conv(waveforms)
            skips = 0

            for f in model.conv_layers:
                stats += np.array(
                    get_model_complexity_info(
                        f,
                        ([waveforms, spectrograms, None]),
                        custom_modules_hooks=custom_modules_hooks,
                    )
                )
                waveforms, h = f(waveforms, spectrograms, None)
                skips += h

            skips *= math.sqrt(1.0 / len(model.conv_layers))

            waveforms = skips
            for f in model.last_conv_layers:
                stats += np.array(
                    get_model_complexity_info(
                        f, ([waveforms]), custom_modules_hooks=custom_modules_hooks
                    )
                )
                waveforms = f(waveforms)

        return stats
