"""
Main command-line entry point for 'vocoder' command.
"""
from typing import List, Type

import click
import langtech.tts.vocoders.datasets as datasets
import langtech.tts.vocoders.models.diffwave as diffwave
import langtech.tts.vocoders.models.framework as framework
import langtech.tts.vocoders.models.parallel_wavegan as parallel_wavegan
import langtech.tts.vocoders.models.wavegrad as wavegrad
import langtech.tts.vocoders.models.wavenet as wavenet
import langtech.tts.vocoders.models.wavernn as wavernn


# List the models available in this repository.
MODELS: List[Type[framework.Vocoder]] = [
    wavegrad.WaveGrad,
    wavernn.WaveRNN,
    wavenet.WaveNet,
    diffwave.DiffWave,
    parallel_wavegan.ParallelWaveGAN,
]


# Create all the commands available for the models.
MODEL_COMMANDS: List[click.Group] = [
    framework.create_model_commands(model) for model in MODELS
]

DATASET_COMMANDS = [
    datasets.download_command,
    datasets.split_command,
]


@click.group()
def main() -> None:
    """
    Vocoder benchmarking CLI.
    """


@main.group("dataset")
def dataset() -> None:
    """
    Dataset processing.
    """


for command in DATASET_COMMANDS:
    dataset.add_command(command)
for command in MODEL_COMMANDS:
    main.add_command(command)
