# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Main command-line entry point for 'vocoder' command.
"""
from typing import List, Type

import click

import datasets as datasets # @oss-only
# @fb-only: import langtech.tts.vocoders.datasets as datasets 

import models.diffwave as diffwave # @oss-only
# @fb-only: import langtech.tts.vocoders.models.diffwave as diffwave 

import models.framework as framework # @oss-only
# @fb-only: import langtech.tts.vocoders.models.framework as framework 

import models.parallel_wavegan as parallel_wavegan # @oss-only
# @fb-only: import langtech.tts.vocoders.models.parallel_wavegan as parallel_wavegan 

import models.wavegrad as wavegrad # @oss-only
# @fb-only: import langtech.tts.vocoders.models.wavegrad as wavegrad 

import models.wavenet as wavenet # @oss-only
# @fb-only: import langtech.tts.vocoders.models.wavenet as wavenet 

import models.wavernn as wavernn # @oss-only
# @fb-only: import langtech.tts.vocoders.models.wavernn as wavernn 


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

if __name__ == "__main__":
    main()
