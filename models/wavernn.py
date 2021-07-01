"""
WaveRNN Neural Vocoder.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.models
from langtech.tts.vocoders.datasets import DatasetConfig, MEL_HOP_SAMPLES, MEL_NUM_BANDS
from langtech.tts.vocoders.models.framework import Vocoder, ConfigProtocol
from langtech.tts.vocoders.models.src.wavenet_vocoder import lrschedule
from langtech.tts.vocoders.utils import remove_none_values_from_dict
from omegaconf import MISSING, OmegaConf
from torch import Tensor
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
        waveforms = self.label_2_float(
            waveforms, self.model.module.n_classes  # pyre-ignore
        )
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

        # Learning rate schedule
        if self.config.model.lr_schedule:
            current_lr = self.config.model.learning_rate
            lr_schedule_fn = getattr(
                lrschedule, self.config.model.lr_schedule  # pyre-ignore
            )
            lr_schedule_kwargs = remove_none_values_from_dict(
                OmegaConf.to_container(self.config.model.lr_schedule_kwargs)
            )
            current_lr = lr_schedule_fn(
                current_lr, self.global_step, **lr_schedule_kwargs
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr  # pyre-ignore

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
        if training:
            spectrograms = spectrograms[:, :, :200]
        output = []

        rnn1 = self.get_gru_cell(self.model.module.rnn1)  # pyre-ignore
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

                x, output = self.get_x_from_dist("random", logits, output)

        output = torch.stack(output).transpose(0, 1)
        output = self.expand(output.flatten())

        self.model.train()  # pyre-ignore

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

    def get_x_from_dist(self, distrib, logits, history=None):  # pyre-ignore
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
            x = self.label_2_float(
                x, self.model.module.n_classes  # pyre-ignore
            ).unsqueeze(-1)
        elif distrib == "random":
            posterior = F.softmax(logits, dim=1)
            distrib = torch.distributions.Categorical(posterior)
            x = distrib.sample().float()
            history.append(x)
            x = self.label_2_float(x, self.model.module.n_classes).unsqueeze(-1)
        else:
            raise RuntimeError("Unknown sampling mode - ", distrib)
        return x, history

    def label_2_float(self, x, n_classes):  # pyre-ignore
        return 2 * x / (n_classes - 1.0) - 1.0
