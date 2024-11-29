# Decoding

## Installation

This project has been tested on Ubuntu 22.04.

1. Get the [UV Python package manager](https://github.com/astral-sh/uv)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create virtual environment and install dependencies
   ```bash
   uv sync
   ```

3. Activate environment
   ```bash
   source .venv/bin/activate
   ```

4. Install optional dependencies

   For jupyter notebooks
   ```bash
   uv sync --extra notebook
   ```
   For fetching datasets
   ```bash
   uv sync --extra aws
   ```
   Or both
   ```bash
   uv sync --extra notebook --extra aws
   ```

## Data
To fetch datasets
```bash
#Lebel2023
aws s3 sync --no-sign-request s3://openneuro.org/ds003020 data/lebel2023/
#Li2022
aws s3 sync --no-sign-request s3://openneuro.org/ds003643 data/li2022/
```
For Li2022, brain mask available [here](https://nist.mni.mcgill.ca/colin-27-average-brain/).

To use AudioCLIP
```bash
git clone https://github.com/LouisJalouzot/AudioCLIP
wget -O AudioCLIP/assets/AudioCLIP-Full-Training.pt https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
wget -O AudioCLIP/assets/bpe_simple_vocab_16e6.txt.gz https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz
```

To build SRM datasets
```bash
pip install git+https://github.com/hugorichard/FastSRM
```