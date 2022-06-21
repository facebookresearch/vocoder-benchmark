# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
WaveGAN Neural Vocoder.
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import models.src.parallel_wavegan.models as models # @oss-only
# @fb-only: import langtech.tts.vocoders.models.src.parallel_wavegan.models as models 

import models.src.parallel_wavegan.optimizers as optimizers # @oss-only
# @fb-only: import langtech.tts.vocoders.models.src.parallel_wavegan.optimizers as optimizers 
import numpy as np
import torch
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

from models.src.parallel_wavegan.layers.pqmf import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.parallel_wavegan.layers.pqmf import ( 
    PQMF,
)

from models.src.parallel_wavegan.layers.residual_block import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.parallel_wavegan.layers.residual_block import ( 
    Conv1d,
    Conv1d1x1,
)

from models.src.parallel_wavegan.losses.stft_loss import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.parallel_wavegan.losses.stft_loss import ( 
    MultiResolutionSTFTLoss,
)

from models.src.ptflops.flops_counter import ( # @oss-only
# @fb-only: from langtech.tts.vocoders.models.src.ptflops.flops_counter import ( 
    conv_flops_counter_hook,
    get_model_complexity_info,
)

from utils import remove_none_values_from_dict # @oss-only
# @fb-only: from langtech.tts.vocoders.utils import remove_none_values_from_dict 
from omegaconf import MISSING, OmegaConf
from torch import Tensor


@dataclass
class LossConfig:
    """
    Configuration for the ParallelWaveGAN loss.
    """

    fft_sizes: List[int] = MISSING
    hop_sizes: List[int] = MISSING
    win_lengths: List[int] = MISSING
    window: str = MISSING


@dataclass
class OptimizerConfig:
    """
    Configuration for the ParallelWaveGAN optimizer.
    """

    lr: float = MISSING
    eps: float = MISSING
    weight_decay: float = MISSING
    amsgrad: Optional[bool] = None


@dataclass
class SchedulerConfig:
    """
    Configuration for the ParallelWaveGAN optimizer scheduler.
    """

    gamma: float = MISSING
    step_size: Optional[int] = None
    milestones: Optional[List[int]] = None


@dataclass
class UpsampleConfig:
    """
    Configuration for Upsampling network parameters.
    """

    upsample_scales: List[int] = MISSING


@dataclass
class GeneratorConfig:
    """
    Configuration for the ParallelWaveGAN generator model.
    """

    in_channels: int = MISSING
    out_channels: int = MISSING
    kernel_size: int = MISSING
    channels: Optional[int] = None
    upsample_scales: Optional[List[int]] = None
    stack_kernel_size: Optional[int] = None
    stacks: int = MISSING
    use_weight_norm: bool = MISSING
    use_causal_conv: Optional[bool] = None
    layers: Optional[int] = None
    residual_channels: Optional[int] = None
    gate_channels: Optional[int] = None
    skip_channels: Optional[int] = None
    aux_channels: Optional[int] = None
    aux_context_window: Optional[int] = None
    dropout: Optional[float] = None
    upsample_net: Optional[str] = None
    upsample_params: Optional[UpsampleConfig] = None


@dataclass
class DownsamplePoolingConfig:
    """
    Configuration for the downsample pooling.
    """

    kernel_size: int = MISSING
    stride: int = MISSING
    padding: int = MISSING
    count_include_pad: bool = MISSING


@dataclass
class NonlinearActivationConfig:
    negative_slope: float = MISSING


@dataclass
class DiscriminatorConfig:
    """
    Configuration for the ParallelWaveGAN discriminator model.
    """

    in_channels: int = MISSING
    out_channels: int = MISSING
    scales: Optional[int] = None
    layers: Optional[int] = None
    downsample_pooling: Optional[str] = None
    downsample_pooling_params: Optional[DownsamplePoolingConfig] = None
    kernel_sizes: Optional[List[int]] = None
    kernel_size: Optional[int] = None
    channels: Optional[int] = None
    max_downsample_channels: Optional[int] = None
    downsample_scales: Optional[List[int]] = None
    nonlinear_activation: str = MISSING
    nonlinear_activation_params: NonlinearActivationConfig = MISSING
    use_weight_norm: bool = MISSING
    bias: Optional[bool] = None
    layers: Optional[int] = None
    conv_channels: Optional[int] = None


@dataclass
class ModelConfig:
    """
    Configuration for the ParallelWaveGAN model.
    """

    # train config
    discriminator_train_start_steps: int = MISSING
    n_iterations: int = MISSING

    # generator config
    generator_type: str = "ParallelWaveGANGenerator"
    generator_params: GeneratorConfig = MISSING
    generator_optimizer_type: str = "RAdam"
    generator_optimizer: OptimizerConfig = MISSING
    generator_grad_norm: int = MISSING
    generator_scheduler_type: str = "StepLR"
    generator_scheduler_params: SchedulerConfig = MISSING

    # discriminator config
    discriminator_type: str = "ParallelWaveGANDiscriminator"
    discriminator_params: DiscriminatorConfig = MISSING
    discriminator_optimizer_type: str = "RAdam"
    discriminator_optimizer: OptimizerConfig = MISSING
    discriminator_grad_norm: int = MISSING
    discriminator_scheduler_type: str = "StepLR"
    discriminator_scheduler_params: SchedulerConfig = MISSING

    # loss config
    use_feat_match_loss: bool = False
    stft_loss_params: LossConfig = MISSING
    use_subband_stft_loss: bool = False
    subband_stft_loss_params: LossConfig = MISSING
    lambda_feat_match: Optional[float] = None
    lambda_adv: float = MISSING


@dataclass
class Config:
    """
    Configuration for the ParallelWaveGAN model and dataset.
    """

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING


class ParallelWaveGAN(Vocoder):
    """
    ParallelWaveGAN model.
    """

    command: str = "parallel_wavegan"

    def __init__(self, config: Config) -> None:
        """
        Create a new ParallelWaveGAN.
        """
        super().__init__(config)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # fix-me
        self.config: Config = remove_none_values_from_dict(
            OmegaConf.to_container(config)
        )

        # Model
        generator_class = getattr(
            models,
            self.config.model.generator_type,
        )
        discriminator_class = getattr(
            models,
            self.config.model.discriminator_type,
        )

        self.model = torch.nn.ModuleDict(
            {
                "generator": generator_class(**self.config.model.generator_params),
                "discriminator": discriminator_class(
                    **self.config.model.discriminator_params
                ),
            }
        )
        self.model["generator"] = torch.nn.DataParallel(self.model["generator"]).to(
            self.device
        )
        self.model["discriminator"] = torch.nn.DataParallel(
            self.model["discriminator"]
        ).to(self.device)

        # Optimizer
        generator_optimizer_class = getattr(
            optimizers,
            self.config.model.generator_optimizer_type,
        )
        discriminator_optimizer_class = getattr(
            optimizers,
            self.config.model.discriminator_optimizer_type,
        )

        self.optimizer: Dict[str, torch.optim.Optimizer] = {
            "generator": generator_optimizer_class(
                self.model["generator"].module.parameters(),
                **self.config.model.generator_optimizer,
            ),
            "discriminator": discriminator_optimizer_class(
                self.model["discriminator"].module.parameters(),
                **self.config.model.discriminator_optimizer,
            ),
        }

        # Scheduler
        generator_scheduler_class = getattr(
            torch.optim.lr_scheduler,
            self.config.model.generator_scheduler_type,
        )
        discriminator_scheduler_class = getattr(
            torch.optim.lr_scheduler,
            self.config.model.discriminator_scheduler_type,
        )

        self.scheduler: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {
            "generator": generator_scheduler_class(
                optimizer=self.optimizer["generator"],
                **self.config.model.generator_scheduler_params,
            ),
            "discriminator": discriminator_scheduler_class(
                optimizer=self.optimizer["discriminator"],
                **self.config.model.discriminator_scheduler_params,
            ),
        }

        # Loss
        self.criterion: Dict[str, torch.nn.Module] = {
            "stft": MultiResolutionSTFTLoss(
                **self.config.model.stft_loss_params  # pyre-ignore
            ).to(self.device),
            "mse": torch.nn.MSELoss().to(self.device),
        }
        if self.config.model.use_feat_match_loss:
            self.criterion["l1"] = torch.nn.L1Loss().to(self.device)

        if self.config.model.generator_params.out_channels > 1:
            self.criterion["pqmf"] = PQMF(
                subbands=config.model.generator_params.out_channels
            ).to(self.device)
        if self.config.model.use_subband_stft_loss:
            assert self.config.model.generator_params.out_channels > 1
            self.criterion["sub_stft"] = MultiResolutionSTFTLoss(
                **self.config.model.subband_stft_loss_params  # pyre-ignore
            ).to(self.device)
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
        Tuple[
            torch.optim.Optimizer,
            Optional[torch.optim.lr_scheduler._LRScheduler],
        ]
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
        return [
            (self.optimizer["generator"], self.scheduler["generator"]),
            (self.optimizer["discriminator"], self.scheduler["discriminator"]),
        ]

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

    # Linter complains that this function is too complex, but to avoid any possible
    # errors, the code is pulled from https://fburl.com/yxdei7mq with minimal
    # refactor, hence the 'noqa' below.
    # pyre-fixme[14]: `train_step` overrides method defined in `Vocoder` inconsistently.
    def train_step(  # noqa
        self, spectrograms: Tensor, waveforms: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Runs a single train step of the model.

        Returns:
          A tuple containing overall model loss and a list of losses to log
          to Tensorboard. The first loss is printed to the console and logged
          to Tensorboard.
        """
        # reset logger
        total_train_loss = defaultdict(lambda: torch.Tensor([0]))

        # parse batch
        y = waveforms.unsqueeze(1).to(self.device)

        x = [spectrograms]
        if self.config.model.generator_type == "ParallelWaveGANGenerator":
            z = torch.randn(y.shape)  # (B, 1, T)
            x = [z, x[0]]
        x = tuple([x_.to(self.device) for x_ in x])

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)

        # reconstruct the signal from multi-band signal
        if self.config.model.generator_params.out_channels > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)  # pyre-ignore

        # multi-resolution sfft loss
        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        total_train_loss["spectral_convergence_loss"] += sc_loss.item()
        total_train_loss["log_stft_magnitude_loss"] += mag_loss.item()
        gen_loss = sc_loss + mag_loss

        # subband multi-resolution stft loss
        if self.config.model.use_subband_stft_loss:
            gen_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)  # pyre-ignore
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            total_train_loss["sub_spectral_convergence_loss"] += sub_sc_loss.item()
            total_train_loss["sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        # adversarial loss
        if self.global_step > self.config.model.discriminator_train_start_steps:
            p_ = self.model["discriminator"](y_)
            if not isinstance(p_, list):
                # for standard discriminator
                adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
                total_train_loss["adversarial_loss"] += adv_loss.item()
            else:
                # for multi-scale discriminator
                adv_loss = 0.0
                for i in range(len(p_)):
                    adv_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size())
                    )
                adv_loss /= i + 1
                total_train_loss["adversarial_loss"] += adv_loss.item()  # pyre-ignore

                # feature matching loss
                if self.config.model.use_feat_match_loss:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](y)
                    fm_loss = 0.0
                    for i in range(len(p_)):
                        for j in range(len(p_[i]) - 1):
                            fm_loss += self.criterion["l1"](p_[i][j], p[i][j].detach())
                    fm_loss /= (i + 1) * (j + 1)
                    total_train_loss["feature_matching_loss"] += fm_loss.item()
                    adv_loss += (
                        self.config.model.lambda_feat_match * fm_loss  # pyre-ignore
                    )

            # add adversarial loss to generator loss
            gen_loss += self.config.model.lambda_adv * adv_loss

        total_train_loss["generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config.model.generator_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config.model.generator_grad_norm,
            )
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.global_step > self.config.model.discriminator_train_start_steps:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model["generator"](*x)
            if self.config.model.generator_params.out_channels > 1:
                y_ = self.criterion["pqmf"].synthesis(y_)  # pyre-ignore

            # discriminator loss
            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_.detach())
            if not isinstance(p, list):
                # for standard discriminator
                real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
                fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
                dis_loss = real_loss + fake_loss
            else:
                # for multi-scale discriminator
                real_loss = 0.0
                fake_loss = 0.0
                for i in range(len(p)):
                    real_loss += self.criterion["mse"](
                        p[i][-1], p[i][-1].new_ones(p[i][-1].size())
                    )
                    fake_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size())
                    )
                real_loss /= i + 1
                fake_loss /= i + 1
                dis_loss = real_loss + fake_loss

            total_train_loss["real_loss"] += real_loss.item()
            total_train_loss["fake_loss"] += fake_loss.item()
            total_train_loss["discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config.model.discriminator_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config.model.discriminator_grad_norm,
                )
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        return gen_loss, total_train_loss

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
        # reset logger
        total_eval_loss = defaultdict(lambda: torch.Tensor([0]))

        y = waveforms.unsqueeze(1).to(self.device)

        x = [spectrograms]
        if self.config.model.generator_type == "ParallelWaveGANGenerator":
            z = torch.randn(y.shape)  # (B, 1, T)
            x = [z, x[0]]
        x = tuple([x_.to(self.device) for x_ in x])

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)
        if self.config.model.generator_params.out_channels > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)  # pyre-ignore

        # multi-resolution stft loss
        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        aux_loss = sc_loss + mag_loss

        # subband multi-resolution stft loss
        if self.config.model.use_subband_stft_loss:
            aux_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)  # pyre-ignore
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            total_eval_loss["sub_spectral_convergence_loss"] += sub_sc_loss.item()
            total_eval_loss["sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        # adversarial loss
        p_ = self.model["discriminator"](y_)
        if not isinstance(p_, list):
            # for standard discriminator
            adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            gen_loss = aux_loss + self.config.model.lambda_adv * adv_loss
        else:
            # for multi-scale discriminator
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size())
                )
            adv_loss /= i + 1
            gen_loss = aux_loss + self.config.model.lambda_adv * adv_loss

            # feature matching loss
            if self.config.model.use_feat_match_loss:
                p = self.model["discriminator"](y)
                fm_loss = 0.0
                for i in range(len(p_)):
                    for j in range(len(p_[i]) - 1):
                        fm_loss += self.criterion["l1"](p_[i][j], p[i][j])
                fm_loss /= (i + 1) * (j + 1)
                total_eval_loss[
                    "feature_matching_loss"
                ] += fm_loss.item()  # pyre-ignore
                gen_loss += (
                    self.config.model.lambda_adv
                    * self.config.model.lambda_feat_match  # pyre-ignore
                    * fm_loss
                )

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y)
        p_ = self.model["discriminator"](y_)

        # discriminator loss
        if not isinstance(p_, list):
            # for standard discriminator
            real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
            fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            dis_loss = real_loss + fake_loss
        else:
            # for multi-scale discriminator
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](
                    p[i][-1], p[i][-1].new_ones(p[i][-1].size())
                )
                fake_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size())
                )
            real_loss /= i + 1
            fake_loss /= i + 1
            dis_loss = real_loss + fake_loss

        # add to total eval loss
        total_eval_loss["adversarial_loss"] += adv_loss.item()
        total_eval_loss["spectral_convergence_loss"] += sc_loss.item()
        total_eval_loss["log_stft_magnitude_loss"] += mag_loss.item()
        total_eval_loss["generator_loss"] += gen_loss.item()
        total_eval_loss["real_loss"] += real_loss.item()
        total_eval_loss["fake_loss"] += fake_loss.item()
        total_eval_loss["discriminator_loss"] += dis_loss.item()

        return total_eval_loss

    # pyre-fixme[14]: `generate` overrides method defined in `Vocoder` inconsistently.
    def generate(
        self, spectrograms: Tensor, training: bool = False
    ) -> Tensor:  # fix-me
        """
        Generate a sample from this model.

        Returns:
          A 1D float tensor containing the output waveform.
        """
        self.model["generator"].eval()

        if training:
            spectrograms = spectrograms[:, :, :200]

        with torch.no_grad():
            spectrograms = spectrograms[0].transpose(0, 1)  # (T', C)
            output = self.model["generator"].module.inference(spectrograms).flatten()
        self.model["generator"].train()
        return output

    def get_complexity(
        self,
    ) -> List[float]:
        """
        Returns A list with the number of FLOPS and parameters used in this model.
        """

        # Prepare the input format.
        waveforms = torch.rand(1, 1, AUDIO_SAMPLE_RATE)
        spectrograms = torch.rand(
            1,
            MEL_NUM_BANDS,
            int(AUDIO_SAMPLE_RATE / MEL_HOP_SAMPLES)
            + 2 * self.config.dataset.padding_frames,
        )
        model = self.model["generator"].module
        stats = np.array([0.0, 0.0])
        custom_modules_hooks = {
            Conv1d1x1: conv_flops_counter_hook,
            Conv1d: conv_flops_counter_hook,
        }

        if torch.cuda.is_available():
            waveforms = waveforms.cuda()
            spectrograms = spectrograms.cuda()

        with torch.no_grad():

            # MelGAN and MB-MelGAN
            if self.config.model.generator_type == "MelGANGenerator":
                return get_model_complexity_info(
                    model,
                    ([spectrograms]),
                    custom_modules_hooks=custom_modules_hooks,
                )

            # ParallelWaveGAN
            stats += np.array(
                get_model_complexity_info(
                    model.upsample_net,
                    ([spectrograms]),
                    custom_modules_hooks=custom_modules_hooks,
                )
            )
            spectrograms = model.upsample_net(spectrograms)
            assert spectrograms.size(-1) == waveforms.size(-1)

            # encode to hidden representation
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
                        ([waveforms, spectrograms]),
                        custom_modules_hooks=custom_modules_hooks,
                    )
                )
                waveforms, h = f(waveforms, spectrograms)
                skips += h

            skips *= math.sqrt(1.0 / len(model.conv_layers))

            # apply final layers
            waveforms = skips
            for f in model.last_conv_layers:
                stats += np.array(
                    get_model_complexity_info(
                        f,
                        ([waveforms]),
                        custom_modules_hooks=custom_modules_hooks,
                    )
                )
                waveforms = f(waveforms)
        return stats
