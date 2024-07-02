# Decoding
To fetch datasets
```bash
#Lebel2023
aws s3 sync --no-sign-request s3://openneuro.org/ds003020 data/lebel2023/
#Li2022
aws s3 sync --no-sign-request s3://openneuro.org/ds003643 data/li2022/
```
Requires
```bash
sudo apt install awscli
```
or
```bash
pip install awscli
```

To use AudioCLIP
```bash
git clone https://github.com/LouisJalouzot/AudioCLIP
wget -O AudioCLIP/assets/AudioCLIP-Full-Training.pt https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
wget -O AudioCLIP/assets/bpe_simple_vocab_16e6.txt.gz https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz
```