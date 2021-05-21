"""
Vocoder utilities.
"""
import os
import sys
from typing import Union

import numpy as np
import soundfile
import torch


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
