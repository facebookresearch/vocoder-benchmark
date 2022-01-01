"""
WaveFlow Neural Vocoder.
"""
import torch
from torch import Tensor
from models.src import waveflow
from datasets import (  # @oss-only
    # @fb-only: from langtech.tts.vocoders.datasets import (
    DatasetConfig,
    MEL_NUM_BANDS,
    MEL_HOP_SAMPLES,
    AUDIO_SAMPLE_RATE,
)
from models.framework import Vocoder, ConfigProtocol
from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf
from typing import Any, Dict, Optional, Tuple, Union, List
from models.src.ptflops.flops_counter import (  # @oss-only
    # @fb-only: from langtech.tts.vocoders.models.src.ptflops.flops_counter import (
    get_model_complexity_info,
    conv_flops_counter_hook,
)


@dataclass
class ModelConfig:
    """
    Configuration for the WaveFlow model.
    """
    flows: int = MISSING
    n_group: int = MISSING
    dilation_channels: int = MISSING
    residual_channels: int = MISSING
    skip_channels: int = MISSING
    learning_rate: float = MISSING
    n_iterations: int = MISSING
    training_sigma: float = MISSING
    evaluate_sigma: float = MISSING


@dataclass
class Config:
    """
    Configuration for the WaveFlow model and dataset.
    """

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING


class WaveFlow(Vocoder):
    """
    WaveFlow model.
    """

    command: str = "waveflow"

    def __init__(self, config: Config) -> None:
        """
        Create a new WaveFlow.
        """
        super().__init__(config)

        self.config = config

        self.model = torch.nn.DataParallel(
            waveflow.WaveFlow(
                n_flows=self.config.model.flows,
                n_group=self.config.model.n_group,
                n_mels=MEL_NUM_BANDS,
                hop_length=MEL_HOP_SAMPLES,
                dilation_channels=self.config.model.dilation_channels,
                residual_channels=self.config.model.residual_channels,
                skip_channels=self.config.model.skip_channels
            )
        )

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.model.learning_rate
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
        Tuple[torch.optim.Optimizer,
              Optional[torch.optim.lr_scheduler._LRScheduler]]
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

        sigma2 = self.config.model.training_sigma ** 2
        z, logdet = self.model(waveforms, spectrograms)
        z = z.view(-1)
        loss = 0.5 * z @ z / sigma2 - logdet.sum()
        loss = loss / z.numel()
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
        self.optimizer.zero_grad()

        # Forward pass.
        loss = self.loss(spectrograms, waveforms)
        # Backward pass.
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

    def generate(self, spectrograms: Tensor, training: bool = False) -> Tensor:
        self.model.eval()
        model: waveflow.WaveFlow = self.model.module
        if spectrograms.ndim == 2:
            spectrograms = spectrograms.unsqueeze(0)

        with torch.no_grad():
            x = model.infer(spectrograms, self.config.model.evaluate_sigma)
        self.model.train()
        return x.flatten()

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

        waveforms = torch.randn(
            spectrograms.shape[0],
            MEL_HOP_SAMPLES * spectrograms.shape[-1],
        )

        if torch.cuda.is_available():
            waveforms = waveforms.cuda()
            spectrograms = spectrograms.cuda()

        # Feed data to network and compute the model complexity.
        with torch.no_grad():
            return get_model_complexity_info(
                self.model,
                (
                    [
                        waveforms,
                        spectrograms
                    ]
                ),
            )
