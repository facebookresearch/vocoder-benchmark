<p align="center">
    <a href="./LICENSE"><img alt="CC BY-NC License" src="https://img.shields.io/badge/License-CC%20BY--NC-blue" /></a>
    <a href="./CONTRIBUTING.md"><img alt="Codecov" src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"></a>
    <a href="https://github.com/facebookresearch/vocoder-benchmark/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/facebookresearch/vocoder-benchmark"></a>
</p>

# VocBench: A Neural Vocoder Benchmark for Speech Synthesis

PyTorch implementation for VocBench framework.

[[arXiv](https://arxiv.org/abs/2112.03099)]

## Installation

1. **Python** >= 3.6
2. Get VocBench code

```sh
$ git clone https://github.com/facebookresearch/vocoder-benchmark.git
$ cd vocoder-benchmark
```
3. Install dependencies

```sh
$ python3 -m venv vocbench
# activate the virtualenv
$ source vocbench/bin/activate
# Upgrade pip
$ python -m pip install --upgrade pip
# Install dependences
$ pip install -e .
```

4. To use VocBench cli, make sure to set paths in your `.bashrc` or `.bash_profile`
appropriately.

```sh
VOCODER_BENCHMARK=/path/to/vocoder-benchmark
export PATH=$VOCODER_BENCHMARK/bin:$PATH
```

5. Change the binary file permission and test your installation

```sh
$ chomd +x $VOCODER_BENCHMARK/bin/vocoder
$ vocoder --help
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

  Vocoder benchmarking CLI.

Options:
  --help  Show this message and exit.

Commands:
  dataset           Dataset processing.
  diffwave          Create, train, or use diffwave models.
  parallel_wavegan  Create, train, or use parallel_wavegan models.
  wavegrad          Create, train, or use wavegrad models.
  wavenet           Create, train, or use wavenet models.
  wavernn           Create, train, or use wavernn models.
```

## Usage

### Download dataset

```sh
$ vocoder dataset --help # For more information on how to download/split dataset

# e.g. download and split LJ Speech
$ vocoder dataset download --dataset ljspeech --path ~/local/datasets/lj # Download and unzip dataset files
$ vocoder dataset split --dataset ljspeech --path ~/local/datasets/lj  # Create train / validation / test splits
```

### Training

```sh
$ vocoder [model-cmd] train --help

# e.g. train wavenet on LJ Speech dataset
$ vocoder wavenet train --path ~/local/models/wavenet --dataset ~/local/datasets/lj --config $VOCODER_BENCHMARK/config/wavenet_mulaw_normal.yaml
```

*For MelGAN and Parallel WaveGAN, they both use the same model cmd. You will need to choose the right configuration for each of them
```sh
# MelGAN
$ vocoder parallel_wavegan train --path ~/local/models/melgan --dataset ~/local/datasets/lj --config $VOCODER_BENCHMARK/config/melgan.v1.yaml

# Parallel WaveGAN
$ vocoder parallel_wavegan train --path ~/local/models/parallel_wavegan --dataset ~/local/datasets/lj --config $VOCODER_BENCHMARK/config/parallel_wavegan.yaml
```

Example of configuration files for each model is provided under `config` directory.

### Synthesize

```sh
$ vocoder [model-cmd] synthesize --help
Usage: cli.py [model-cmd] synthesize [OPTIONS] INPUT_FILE OUTPUT_FILE

  Synthesize with the model.

Options:
  --path TEXT     Directory for the model  [required]
  --length TEXT   The length of the output sample in seconds
  --offset FLOAT  Offset in seconds of the sample
  --help          Show this message and exit.
```

### Evaluate

```sh
$ vocoder [model-cmd] evaluate --help
Usage: cli.py [model-cmd] evaluate [OPTIONS]

  Evaluate a given vocoder.

Options:
  --path TEXT        Directory for the model  [required]
  --dataset TEXT     Name of the dataset to use  [required]
  --checkpoint TEXT  Checkpoint path (default: load latest checkpoint)
  --help             Show this message and exit.
```

*Frechet Audio Distance is currently not implemented. We use Google Research opensource [repository](https://github.com/google-research/google-research/tree/master/frechet_audio_distance) to get FAD results.


## Reference Repositories

* [Pytorch](https://github.com/pytorch/pytorch), Pytorch.
* [Audio](https://github.com/pytorch/audio), Pytorch.
* [FAD](https://github.com/google-research/google-research/tree/master/frechet_audio_distance), Google Research.
* [WaveNet](https://github.com/r9y9/wavenet_vocoder), Ryuichi Yamamoto.
* [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), Tomoki Hayashi.
* [WaveGrad](https://github.com/ivanvovk/WaveGrad), Ivan Vovk.
* [DiffWave](https://github.com/lmnt-com/diffwave), LMNT.
* [Flops counter](https://github.com/sovrasov/flops-counter.pytorch), Vladislav Sovrasov.


## License
The majority of VocBench is licensed under CC-BY-NC, however portions of the project are available under separate license terms: Wavenet, ParallelWaveGAN, and flops counter are licensed under the MIT license; diffwave is licensed under the Apache 2.0 license; WaveGrad is licensed under the BSD-3 license.

## Used by
<details><summary>List of papers that used our work (Feel free to add your own paper by making a pull request)</summary><p>
</p></details>
