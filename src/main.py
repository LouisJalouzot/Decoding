from hashlib import sha1

import wandb
from src.train import train
from src.utils import ignore


def main(
    subject: str = "UTS00",
    decoder: str = "ridge",
    model: str = "clip",
    context_length: int = 2,
    tr: int = 2,
    lag: int = 2,
    smooth: int = 1,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 0,
    subsample_voxels: int = None,
    latents_batch_size: int = 64,
    **decoder_params,
):
    config = {
        key: value
        for key, value in locals().items()
        if key not in ignore and key != "decoder_params"
    }
    config.update(decoder_params)
    wandb.init(
        config=config,
        id=sha1(repr(sorted(config.items())).encode()).hexdigest(),
        save_code=True,
    )
    output = train(**config)
    wandb.finish()
    return output
