#!.env/bin/python
from src.train import train

train(
    subject="UTS03",
    model="sentence-transformers/all-mpnet-base-v2",
    # model="mel",
    decoder="brain_decoder",
    context_length=4,
    lag=2,
    smooth=0,
    subsample_voxels=10_000,
    hidden_size_backbone=512,
    hidden_size_projector=512,
    dropout=0.7,
    n_res_blocks=2,
    n_proj_blocks=1,
    lr=1e-4,
    weight_decay=1e-6,
    batch_size=1024,
    temperature=0.01,
)
