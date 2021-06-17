"""
WaveNet Neural Vocoder.
"""
import math
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
from torch.nn import functional as F
from tqdm import tqdm


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
        waveforms = self.label_2_float(waveforms, self.config.model.out_channels)
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

    def generate(self, spectrograms: Tensor, training: bool = False) -> Tensor:
        """
        Generate a sample from this model.

        Returns:
          A 1D float tensor containing the output waveform.
        """
        self.model.eval()  # pyre-ignore
        self.model.module.clear_buffer()  # pyre-ignore

        with torch.no_grad():
            spectrograms = self.model.module.upsample_net(spectrograms)
            seq_len = (
                22050 if training else spectrograms.size(-1)
            )  # synthesize the first second only during training
            batch_size = spectrograms.shape[0]
            spectrograms = spectrograms.transpose(1, 2).contiguous()
            x = spectrograms.new_zeros(batch_size, 1, 1)
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
                x, output = self.get_x_from_dist("random", x[:, 0], output)
                x = x.unsqueeze(1)

        output = torch.stack(output).transpose(0, 1)
        output = self.expand(output.flatten())

        self.model.module.clear_buffer()
        self.model.train()  # pyre-ignore

        return output

    def get_x_from_dist(self, distrib, logits, history=None):  # pyre-ignore
        """
        Sampling from a given distribution

        Returns:
            a tuple of current sample x and history of an array of previous samples
        """
        # set_trace()
        if history is None:
            history = []
        if distrib == "argmax":
            x = torch.argmax(logits, dim=1)
            history.append(x)
            x = self.label_2_float(x, self.config.model.out_channels).unsqueeze(-1)
        elif distrib == "random":
            posterior = F.softmax(logits, dim=1)
            distrib = torch.distributions.Categorical(posterior)
            x = distrib.sample().float()
            history.append(x)
            x = self.label_2_float(x, self.config.model.out_channels).unsqueeze(-1)
        else:
            raise RuntimeError("Unknown sampling mode - ", distrib)
        return x, history

    def label_2_float(self, x, n_classes):  # pyre-ignore
        return 2 * x / (n_classes - 1.0) - 1.0
