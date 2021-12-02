# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Dataset commands.
"""
import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import Tuple, Set, List, Union, Iterator

import click
import torch
import torchaudio

from utils import die_if # @oss-only
# @fb-only: from langtech.tts.vocoders.utils import die_if 
from omegaconf import MISSING

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Librosa can have a lot of stdout warnings.
    import librosa

# Filename to put the training split into.
SPLIT_JSON: str = "data_split.json"

# Datasets we can load and use.
KNOWN_DATASETS: Set[str] = {"ljspeech", "libritts", "vctk"}

# How to split LJ dataset. LJ is split by taking the first K1 samples as test,
# K2 samples as validation, and the rest as training.
NUM_LJSPEECH_TEST_SAMPLES: int = 20
NUM_LJSPEECH_VALIDATION_SAMPLES: int = 10

# LibriTTS data splits
LIBRITTS_TRAIN_SPLITS: List[str] = [
    "train-clean-100",
    "train-clean-360",
]
LIBRITTS_VALIDATION_SPLITS: List[str] = ["dev-clean"]
LIBRITTS_TEST_SPLITS: List[str] = ["test-clean"]
LIBRITTS_ALL_SPLITS: List[str] = (
    LIBRITTS_TRAIN_SPLITS + LIBRITTS_VALIDATION_SPLITS + LIBRITTS_TEST_SPLITS
)

# Percentage of each of the different splits for VCTK dataset
VCTK_TRAIN_SPLIT_PRC: float = 0.85
VCTK_VALIDATION_SPLIT_PRC: float = 0.1
VCTK_TEST_SPLIT_PRC: float = 0.05

# Audio sampling frequency for models.
AUDIO_SAMPLE_RATE: int = 24000

# Mel spectrogram parameters.
MEL_NUM_BANDS: int = 80
MEL_F_MIN: int = 0
MEL_F_MAX: int = 12000
MEL_N_FFT: int = 1024
MEL_HOP_SAMPLES: int = 300
MEL_WIN_SAMPLES: int = 960
MIN_LEVEL_DB: int = -100

# How many clips to shuffle between during data loading.
SHUFFLE_BUFFER_SIZE: int = 250


class Audio2Mel(torch.nn.Module):
    """Converts from waveforms to log Mel spectrograms."""

    def __init__(self) -> None:
        """
        Initialize this module.

        Args:
          config: Parameters to use for waveform to mel conversion.
        """
        super().__init__()

        window = torch.hann_window(MEL_WIN_SAMPLES).float()
        mel_basis = librosa.filters.mel(
            AUDIO_SAMPLE_RATE, MEL_N_FFT, MEL_NUM_BANDS, MEL_F_MIN, MEL_F_MAX
        )
        mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

        self.n_fft: int = MEL_N_FFT
        self.hop_length: int = MEL_HOP_SAMPLES
        self.win_length: int = MEL_WIN_SAMPLES
        self.padding: int = (MEL_N_FFT - MEL_HOP_SAMPLES) // 2
        self.sample_rate: int = AUDIO_SAMPLE_RATE

    def num_frames(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute number of spectrogram frames when given samples.

        Args:
          lengths: Lengths of input tensors in samples.

        Returns:
          Lengths of output tensors in frames.
        """
        return torch.div(lengths, self.hop_length, rounding_mode="floor")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mels.

        Args:
          audio: Input audio of shape [batch_size, timesteps].

        Returns:
          Mel spectrograms of shape [batch_size, n_mels, timesteps / hop_length].
        """
        assert audio.ndim == 2, "Expecting input of shape [batch_size, num_samples]"
        audio = torch.nn.functional.pad(
            audio.unsqueeze(1), (self.padding, self.padding), "reflect"
        ).squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = 20 * torch.log10(torch.clamp(mel_output, min=1e-5))
        return self.normalize(log_mel_spec)

    def normalize(self, S: torch.Tensor) -> torch.Tensor:
        """
        Normalize mels input.
        """
        return torch.clip((S - MIN_LEVEL_DB) / -MIN_LEVEL_DB, 0, 1)


@dataclass
class DatasetConfig:
    """
    How data should be loaded.
    """

    # How many clips to include in each batch.
    batch_size: int = MISSING

    # How many samples to include each clip.
    frames_per_clip: int = MISSING

    # How many clips to generate from each utterance.
    clips_per_utterance: int = MISSING

    # How many frames of padding to include on the spectrogram.
    padding_frames: int = MISSING

    # How many worker processes to use for data loading.
    dataloader_num_workers: int = MISSING

    # How far to prefetch for dataloading.
    dataloader_prefetch_factor: int = MISSING


class VocoderDataset(torch.utils.data.IterableDataset):
    """
    A dataset loading vocoder samples.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        indices: List[Union[str, int]],
        config: DatasetConfig,
        validation: bool,
        generate: bool,
    ) -> None:
        """
        Create the dataset.

        Args:
          dataset: The dataset to index.
          indices: The list of indices.
          config: Configuration for this dataset from the model.
          validation: Whether loading for validation or training.
          generate: Whether loading for audio generation.
        """
        self.dataset = dataset
        self.indices = indices
        self.config = config
        self.mel = Audio2Mel()
        self.validation = validation
        self.generate = generate

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over this dataset.

        Yields:
          Samples from the dataset. The first element is the spectrograms and
          the second is the waveforms.
        """
        shuffle_buffer = []
        for clip_spec, clip_wave in self.raw_iter():
            if self.generate:
                yield clip_spec, clip_wave
                continue

            clips = self.extract_clips(clip_spec, clip_wave)
            if self.validation:
                # When validation, any order is fine.
                yield from clips
            else:
                # For training, we want some randomness, so keep a shuffle
                # buffer. When the shuffle buffer is full, shuffle it and yield
                # all samples.
                shuffle_buffer.extend(clips)
                if len(shuffle_buffer) > SHUFFLE_BUFFER_SIZE:
                    random.shuffle(shuffle_buffer)
                    yield from shuffle_buffer
                    shuffle_buffer = []

        if shuffle_buffer:
            random.shuffle(shuffle_buffer)
            yield from shuffle_buffer

    def extract_clips(
        self, spectrogram: torch.Tensor, waveform: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract clips from a spectrogram and waveform.
        """
        clips = []

        # For validation, extract all samples in the utterance.
        total_clip_frames = 2 * self.config.padding_frames + self.config.frames_per_clip
        if self.validation:
            start_frame = 0
            while start_frame + total_clip_frames <= spectrogram.shape[-1]:
                end_frame = start_frame + total_clip_frames
                start_sample = (
                    start_frame + self.config.padding_frames
                ) * MEL_HOP_SAMPLES
                end_sample = (end_frame - self.config.padding_frames) * MEL_HOP_SAMPLES
                clips.append(
                    (
                        spectrogram[:, start_frame:end_frame],
                        waveform[start_sample:end_sample],
                    )
                )
                start_frame += self.config.frames_per_clip
        else:
            # For training, take random clips from the utterance, aligned to
            # the spectrogram boundary.
            for _ in range(self.config.clips_per_utterance):
                max_start = spectrogram.shape[-1] - total_clip_frames
                if max_start < 0:
                    break

                start_frame = random.randint(0, max_start)
                end_frame = start_frame + total_clip_frames
                start_sample = (
                    start_frame + self.config.padding_frames
                ) * MEL_HOP_SAMPLES
                end_sample = (end_frame - self.config.padding_frames) * MEL_HOP_SAMPLES
                clips.append(
                    (
                        spectrogram[:, start_frame:end_frame],
                        waveform[start_sample:end_sample],
                    )
                )

        return clips

    def raw_iter(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over the dataset without extracting clips or batching.

        Yields:
          Samples from the dataset. The first element is the spectrograms and
          the second is the waveforms.

          Spectrogram shape: [num_bands, num_frames]
          Waveform shape: [num_frames * hop_samples]
        """
        # Split data among workers.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            indices = [
                idx
                for i, idx in enumerate(self.indices)
                if i % worker_info.num_workers  # pylint: disable=no-member
                == worker_info.id  # pylint: disable=no-member
            ]
        else:
            indices = self.indices

        for key in indices:
            waveform, sr = self.dataset[key][:2]

            # Resample the waveform to a fixed sample rate
            waveform = librosa.resample(waveform[0].numpy(), sr, AUDIO_SAMPLE_RATE)
            waveform = torch.clip(torch.tensor([waveform]), -1, 1)

            # Pad to make sure waveform is a multiple of hop length.
            padding = (
                MEL_HOP_SAMPLES - waveform.numel() % MEL_HOP_SAMPLES
            ) % MEL_HOP_SAMPLES
            waveform = torch.nn.functional.pad(waveform, (0, padding))

            # Compute spectrogram of waveform.
            # Length of spectrogram is exactly waveform length over hop length.
            spectrogram = self.mel(waveform)

            yield spectrogram.squeeze(0), waveform.squeeze(0)


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    indices: List[Union[str, int]],
    config: DatasetConfig,
    validation: bool,
    generate: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create a data loader given a dataset and a list of samples.

    Args:
      dataset: The dataset to index.
      indices: The list of indices.
      config: Configuration for the dataset from the model.
      validation: Whether loading for validation or training.
      generate: Whether to use for audio generation.
        If so, includes the entire sample with batch size one.
    """
    dataset = VocoderDataset(dataset, indices, config, validation, generate)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1 if generate else config.batch_size,
        num_workers=config.dataloader_num_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
    )


def load_dataset(
    path: str, config: DatasetConfig, num_generate_samples: int
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Load a dataset. If it's not available, crash.

    Args:
      path: Path to the dataset.
      config: Configuration for the dataset from the model.
      num_generate_samples: How many samples from validation set to include
        in generation set.

    Returns:
      A tuple containing a training data loader, a validation dataloader, and
      an audio generation data loader.
    """
    with open(os.path.join(path, SPLIT_JSON), "r") as handle:
        split = json.load(handle)

    name = split["dataset"]
    die_if(
        name not in KNOWN_DATASETS,
        f"dataset name '{name}' must be one of {KNOWN_DATASETS}",
    )

    if name == "ljspeech":
        dset = torchaudio.datasets.LJSPEECH(root=path, download=False)

        return (
            create_dataloader(dset, split["train"], config, validation=False),
            create_dataloader(dset, split["validation"], config, validation=True),
            create_dataloader(
                dset,
                split["test"][:num_generate_samples],
                config,
                validation=False,
                generate=True,
            ),
        )
    elif name == "libritts":
        dset_train = torch.utils.data.ConcatDataset(
            [
                torchaudio.datasets.LIBRITTS(root=path, url=split, download=False)
                for split in LIBRITTS_TRAIN_SPLITS
            ]
        )
        dset_validation = torch.utils.data.ConcatDataset(
            [
                torchaudio.datasets.LIBRITTS(root=path, url=split, download=False)
                for split in LIBRITTS_VALIDATION_SPLITS
            ]
        )

        return (
            create_dataloader(dset_train, split["train"], config, validation=False),
            create_dataloader(
                dset_validation, split["validation"], config, validation=True
            ),
            create_dataloader(
                dset_validation,
                split["test"][:num_generate_samples],
                config,
                validation=False,
                generate=True,
            ),
        )
    elif name == "vctk":
        dset = torchaudio.datasets.VCTK_092(root=path, download=False)

        return (
            create_dataloader(dset, split["train"], config, validation=False),
            create_dataloader(dset, split["validation"], config, validation=True),
            create_dataloader(
                dset,
                split["test"][:num_generate_samples],
                config,
                validation=False,
                generate=True,
            ),
        )

    raise ValueError(f"Unknown dataset {name}")


@click.command("download")
@click.option(
    "--dataset",
    required=True,
    help="Which dataset to download",
    type=click.Choice(KNOWN_DATASETS),
)
@click.option(
    "--path",
    required=True,
    help="Where to place the dataset",
)
def download_command(dataset: str, path: str) -> None:
    """
    Download a dataset.
    """
    if dataset == "ljspeech":
        torchaudio.datasets.LJSPEECH(root=path, download=True)
    elif dataset == "libritts":
        for split in LIBRITTS_ALL_SPLITS:
            torchaudio.datasets.LIBRITTS(root=path, url=split, download=True)
    elif dataset == "vctk":
        torchaudio.datasets.VCTK_092(root=path, download=True)


@click.command("split")
@click.option(
    "--dataset",
    required=True,
    help="Which dataset to download",
    type=click.Choice(KNOWN_DATASETS),
)
@click.option(
    "--path",
    required=True,
    help="Where the dataset is",
)
def split_command(dataset: str, path: str) -> None:
    """
    Split into train / validation / test.
    """
    if dataset == "ljspeech":
        dset = torchaudio.datasets.LJSPEECH(root=path, download=False)
        num_samples = len(dset)
        split = {
            "dataset": "ljspeech",
            "test": list(range(0, NUM_LJSPEECH_TEST_SAMPLES)),
            "validation": list(
                range(
                    NUM_LJSPEECH_TEST_SAMPLES,
                    NUM_LJSPEECH_TEST_SAMPLES + NUM_LJSPEECH_VALIDATION_SAMPLES,
                )
            ),
            "train": list(
                range(
                    NUM_LJSPEECH_TEST_SAMPLES + NUM_LJSPEECH_VALIDATION_SAMPLES,
                    num_samples,
                )
            ),
        }
    elif dataset == "libritts":
        dset_train = torch.utils.data.ConcatDataset(
            [
                torchaudio.datasets.LIBRITTS(root=path, url=split, download=False)
                for split in LIBRITTS_TRAIN_SPLITS
            ]
        )
        dset_validation = torch.utils.data.ConcatDataset(
            [
                torchaudio.datasets.LIBRITTS(root=path, url=split, download=False)
                for split in LIBRITTS_VALIDATION_SPLITS
            ]
        )
        dset_test = torch.utils.data.ConcatDataset(
            [
                torchaudio.datasets.LIBRITTS(root=path, url=split, download=False)
                for split in LIBRITTS_TEST_SPLITS
            ]
        )
        num_train_samples = len(dset_train)
        num_validation_samples = len(dset_validation)
        num_test_samples = len(dset_test)
        split = {
            "dataset": "libritts",
            "train": list(range(num_train_samples)),
            "validation": list(range(num_validation_samples)),
            "test": list(range(num_test_samples)),
        }
    elif dataset == "vctk":
        dset = torchaudio.datasets.VCTK_092(root=path, download=False)
        num_samples = len(dset)
        num_train_samples = int(num_samples * VCTK_TRAIN_SPLIT_PRC)
        num_validation_samples = int(num_samples * VCTK_VALIDATION_SPLIT_PRC)

        indices = list(range(num_samples))
        random.Random(42).shuffle(indices)
        split = {
            "dataset": "vctk",
            "train": indices[0:num_train_samples],
            "validation": indices[
                num_train_samples : num_train_samples + num_validation_samples
            ],
            "test": indices[num_train_samples + num_validation_samples :],
        }

    with open(os.path.join(path, SPLIT_JSON), "w") as handle:
        json.dump(split, handle)  # pyre-ignore
