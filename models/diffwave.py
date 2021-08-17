"""
DiffWave Neural Vocoder.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch

from datasets import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.datasets import ( 
    DatasetConfig,
    MEL_NUM_BANDS,
    MEL_HOP_SAMPLES,
    AUDIO_SAMPLE_RATE,
)

from models.framework import Vocoder, ConfigProtocol # @oss-only
# @fb-only: from langtech.tts.vocoders.models.framework import Vocoder, ConfigProtocol 

from models.src.diffwave import model # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.diffwave import model 

from models.src.ptflops.flops_counter import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.ptflops.flops_counter import ( 
    get_model_complexity_info,
)
from omegaconf import MISSING
from torch import Tensor


@dataclass
class NoiseSchedulerConfig:
    """
    Configuration for the DiffWave noise scheduler.
    """

    n_iter: int = MISSING
    betas_range: List[float] = MISSING


@dataclass
class ModelConfig:
    """
    Configuration for the DiffWave model.
    """

    residual_layers: int = MISSING
    residual_channels: int = MISSING
    dilation_cycle_length: int = MISSING
    training_noise_schedule: NoiseSchedulerConfig = MISSING
    inference_noise_schedule: List[float] = MISSING
    noise_schedule: Optional[List[float]] = None
    learning_rate: float = MISSING
    n_iterations: int = MISSING
    fp16: Optional[bool] = False
    max_grad_norm: Optional[int] = None


@dataclass
class Config:
    """
    Configuration for the DiffWave model and dataset.
    """

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING


class DiffWave(Vocoder):
    """
    DiffWave model.
    """

    command: str = "diffwave"

    def __init__(self, config: Config) -> None:
        """
        Create a new DiffWave.
        """
        super().__init__(config)

        self.config = config
        self.config.model.noise_schedule = np.linspace(
            self.config.model.training_noise_schedule.betas_range[0],
            self.config.model.training_noise_schedule.betas_range[1],
            self.config.model.training_noise_schedule.n_iter,
        ).tolist()
        self.model = torch.nn.DataParallel(model.DiffWave(self.config.model))
        self.autocast = torch.cuda.amp.autocast(enabled=config.model.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.model.fp16)

        # Train noise config
        beta = np.array(self.config.model.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level: torch.Tensor = torch.tensor(noise_level.astype(np.float32))

        # Inference noise config
        beta = np.array(self.config.model.inference_noise_schedule)
        inference_noise_level = np.cumprod(1 - beta)
        self.inference_noise_level: torch.Tensor = torch.tensor(
            inference_noise_level.astype(np.float32)
        )

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.model.learning_rate
        )

        self.criterion: torch.nn.Module = torch.nn.L1Loss()

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

    def loss(self, spectrograms: Tensor, waveforms: Tensor, mode: str) -> Tensor:
        """
        Compute loss on a batch.

        Returns:
          The negative log likelihood loss.
        """
        N, T = waveforms.shape
        device = waveforms.device
        if mode == "train":
            noise_level = self.noise_level.to(device)
            noise_schedule = self.config.model.noise_schedule
        elif mode == "val":
            noise_level = self.inference_noise_level.to(device)
            noise_schedule = self.config.model.inference_noise_schedule

        with self.autocast:
            t = torch.randint(
                0, len(noise_schedule), [N], device=waveforms.device  # pyre-ignore
            )
            noise_scale = noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(waveforms)
            noisy_waveforms = (
                noise_scale_sqrt * waveforms + (1.0 - noise_scale) ** 0.5 * noise
            )

            predicted = self.model(noisy_waveforms, spectrograms, t)  # pyre-ignore
            loss = self.criterion(noise, predicted.squeeze(1))

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

        for param in self.model.parameters():  # pyre-ignore
            param.grad = None

        # Forward pass.
        loss = self.loss(spectrograms, waveforms, "train")

        # Backward pass.
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = torch.nn.utils.clip_grad_norm_(  # pyre-ignore
            self.model.parameters(), self.config.model.max_grad_norm or 1e9
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_stats = {"total_loss": loss, "grad_norm": self.grad_norm}

        return loss, loss_stats  # pyre-ignore

    def validation_losses(
        self, spectrograms: Tensor, waveforms: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute validation losses.

        Returns:
          A dictionary mapping loss name (e.g. 'nll_loss') to the validation value.
        """
        return {
            "nll_loss": self.loss(spectrograms, waveforms, "val"),
        }

    def generate(self, spectrograms: Tensor, training: bool = False) -> Tensor:
        self.model.eval()  # pyre-ignore

        device = spectrograms.device

        if training:
            spectrograms = spectrograms[:, :, :200]

        with torch.no_grad():
            training_noise_schedule = np.array(self.config.model.noise_schedule)
            inference_noise_schedule = np.array(
                self.config.model.inference_noise_schedule
            )

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                            talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5
                        )
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            # Expand rank 2 tensors by adding a batch dimension.
            if len(spectrograms.shape) == 2:
                spectrograms = spectrograms.unsqueeze(0)
            spectrograms = spectrograms.to(device)

            audio = torch.randn(
                spectrograms.shape[0],
                MEL_HOP_SAMPLES * spectrograms.shape[-1],
                device=device,
            )

            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n] ** 0.5
                c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
                audio = c1 * (
                    audio
                    - c2
                    * self.model(  # pyre-ignore
                        audio, spectrograms, torch.tensor([T[n]], device=audio.device)
                    ).squeeze(1)
                )
                if n > 0:
                    noise = torch.randn_like(audio)
                    sigma = (
                        (1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]
                    ) ** 0.5
                    audio += sigma * noise
                audio = torch.clamp(audio, -1.0, 1.0)

        self.model.train()  # pyre-ignore

        return audio.flatten()

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

        device = spectrograms.device
        waveforms = torch.randn(
            spectrograms.shape[0],
            MEL_HOP_SAMPLES * spectrograms.shape[-1],
            device=device,
        )
        Tn = torch.tensor([0.1], device=waveforms.device)

        # Feed data to network and compute the model complexity.
        with torch.no_grad():
            return get_model_complexity_info(
                self.model,
                (
                    [
                        waveforms,
                        spectrograms,
                        Tn,
                    ]
                ),
            )
