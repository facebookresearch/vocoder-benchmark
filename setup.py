"""
setup.py for vocoder benchmark
"""

from setuptools import setup, find_packages

setup(
    name="vocoder-benchmark",
    version="0.1",
    description="A repository for benchmarking neural vocoders by their quality and speed.",
    packages=find_packages(),
    install_requires=[
        "click",
        "numpy",
        "six",
        "tqdm",
        "torch == 1.9",
        "tensorboard",
        "omegaconf",
        "librosa",
        "pylint",
        "scipy",
        "soundfile",
        "torchaudio",
        "pyre-check",
        "pytorch_msssim",
        "pthflops",
    ],
    extras_require={},
)
