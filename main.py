# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-unsafe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Main entrypoint for vocoders.

This file is FB-internal only.
"""

import libfb.py.ctypesmonkeypatch

libfb.py.ctypesmonkeypatch.install()

from cli import main # @oss-only
# @fb-only: from langtech.tts.vocoders.cli import main 

if __name__ == "__main__":
    main()
