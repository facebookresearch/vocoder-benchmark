"""
Vocoder utilities.
"""
import os
import sys
from typing import Tuple
from typing import Union

import librosa
import numpy as np
import soundfile
import torch
from omegaconf import OmegaConf


def die_if(condition: bool, message: str) -> None:
    """
    Die if the condition is met.

    Args:
      condition: The condition to check.
      message: Message to print upon dying.
    """
    if condition:
        print(f"ERROR: {message}", file=sys.stderr, flush=True)
        hard_exit()


def hard_exit(code: int = 1) -> None:
    """
    Exit un-gracefully but immediately. Used when you need to exit with
    background processes still running.

    Args:
      code: The exit code. Defaults to failure.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # pylint: disable=protected-access


def write_audio(
    filename: str, data: Union["torch.Tensor", np.ndarray], sample_rate: int
) -> None:
    """
    Write audio to disk.

    Args:
      filename: The filename. Must end in .wav or .flac.
      data: A 1D data array (or multiple dimensions where all but one are one).
      sample_rate: The sample rate of the audio.

    Raises:
      ValueError: If filename has incorrect extension or if data is incorrect shape.
    """

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if not filename.endswith(".wav") and not filename.endswith(".flac"):
        raise ValueError(f"Filename {filename} must end with .wav or .flac")

    data_shape = data.shape
    data = np.squeeze(data)
    if data.ndim > 1:
        raise ValueError(
            f"Sample data of shape {data_shape} has too many non-1 dimensions"
        )

    soundfile.write(filename, data, sample_rate)


def remove_none_values_from_dict(config_dict):
    """
    Iterate over input configuration and remove None-value params

    Returns:
        dictionary of valid configuration params
    """
    if isinstance(config_dict, dict):
        config = {}
        for k in config_dict:
            if config_dict[k] is not None:
                config[k] = remove_none_values_from_dict(config_dict[k])
        return OmegaConf.create(config)

    return config_dict


def read_audio(filename: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    """
    Read audio from disk.

    Args:
      filename: The filename. Must end in .wav or .flac.
      sample_rate: Target sample rate.

    Returns:
        waveform: A 1D data array (or multiple dimensions where all but one are one).

    Raises:
      ValueError: If filename has incorrect extension.
    """

    if not filename.endswith(".wav") and not filename.endswith(".flac"):
        raise ValueError(f"Filename {filename} must end with .wav or .flac")

    waveform, sr = soundfile.read(filename)
    return librosa.resample(waveform, sr, sample_rate)
