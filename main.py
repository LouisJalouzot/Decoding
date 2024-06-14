#!.env/bin/python
from src.train import train

train(
    subject="UTS01",
    model="sentence-transformers/all-mpnet-base-v2",
    decoder="brain_decoder",
    context_length=4,
    lag=2,
    halflife=1,
    subsample_voxels=None,
    weight_decay=1e-4,
    batch_size=128,
    temperature=0.01,
)
