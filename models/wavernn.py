"""
WaveRNN Neural Vocoder.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torchaudio
import torchaudio.models
from langtech.tts.vocoders.datasets import DatasetConfig, MEL_HOP_SAMPLES, MEL_NUM_BANDS
from langtech.tts.vocoders.models.framework import Vocoder, ConfigProtocol
from omegaconf import MISSING
from torch import Tensor


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
        output = self.model(  # pyre-ignore
            waveforms.unsqueeze(1), spectrograms.unsqueeze(1)
        )
        output = output.squeeze(1)[:, :-1, :]
        loss = self.criterion(output.transpose(1, 2), target)

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
        raise NotImplementedError("WaveRNN.generate() not yet implemented")
