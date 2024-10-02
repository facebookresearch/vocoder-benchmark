# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities around path handling. This file is replaced with a different file
externally; this version is only for FB-internal use.
"""

import os

# @fb-only: import libfb.py.parutil 


def get_default_config_path(config_file: str) -> str:
    """
    Get the path to the default config for a model.

    Args:
      config_file: Configuration file name.

    Returns:
      A path to the default config YAML file.
    """
    # @fb-only: return libfb.py.parutil.get_file_path( 
        # @fb-only: os.path.join("langtech", "tts", "vocoders", "config", config_file) 
    # @fb-only: ) 
    return os.path.join("config", config_file) # @oss-only
