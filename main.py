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
