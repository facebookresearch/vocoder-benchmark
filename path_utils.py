"""
Utilities around path handling. This file is replaced with a different file
externally; this version is only for FB-internal use.
"""
import os

import libfb.py.parutil


def get_default_config_path(config_file: str) -> str:
    """
    Get the path to the default config for a model.

    Args:
      config_file: Configuration file name.

    Returns:
      A path to the default config YAML file.
    """
    return libfb.py.parutil.get_file_path(
        os.path.join("langtech", "tts", "vocoders", "config", config_file)
    )
