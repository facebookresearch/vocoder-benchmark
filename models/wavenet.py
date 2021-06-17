"""
WaveNet Neural Vocoder.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import langtech.tts.vocoders.models.wavenet_vocoder.util as util
import langtech.tts.vocoders.models.wavenet_vocoder.wavenet as wavenet
import torch
import torchaudio
import torchaudio.models
from langtech.tts.vocoders.datasets import DatasetConfig
from langtech.tts.vocoders.models.framework import Vocoder, ConfigProtocol
from omegaconf import MISSING
from torch import Tensor


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

        # config.model.upsample_params.cin_channels = config.model.cin_channels
        # config.model.upsample_params.cin_pad = config.model.cin_pad

        self.config = config

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
                scalar_input=util.is_scalar_input(config.model.input_type),
                output_distribution=config.model.output_distribution,
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
        waveforms = self.label_2_float(  # pyre-ignore
            waveforms, self.config.model.out_channels
        )
        output = self.model(waveforms.unsqueeze(1), c=spectrograms)  # pyre-ignore
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

        # Backward pass.
        self.optimizer.zero_grad()
        loss.backward()
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

    def generate(self, _spectrograms: Tensor, _training: bool = False) -> Tensor:
        """
        Generate a sample from this model.

        Returns:
          A 1D float tensor containing the output waveform.
        """
        raise NotImplementedError("WaveNet.generate() not yet implemented")
