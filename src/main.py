from hashlib import sha1

import wandb
from src.train import train
from src.utils import ignore


def main(
    subjects: str = "UTS03",
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 2,
    smooth: int = 0,
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
