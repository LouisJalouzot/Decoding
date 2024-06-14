#!.env/bin/python
from src.train import train

train(
    subject="UTS01",
    model="sentence-transformers/all-mpnet-base-v2",
    decoder="brain_decoder",
    context_length=6,
    lag=3,
    smooth=2,
    # subsample_voxels=10_000,
    weight_decay=1e-6,
    batch_size=1024,
    temperature=0.05,
)
