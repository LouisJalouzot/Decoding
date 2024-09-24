import json
from hashlib import sha1
from typing import Dict, List, Union

import wandb
from src.main import main
from src.utils import memory

cache_ignore = ["return_data", "log_nlp_distances"]
config_ignore = ["cache", "wandb_mode", "tags", "kwargs"]
wandb_ignore = config_ignore + ["return_data", "n_jobs", "verbose"]


def wandb_wrapper(
    datasets: Union[str, List[str]] = ["lebel2023"],
    subjects: Dict[str, List[str]] = None,
    runs: Dict[str, Dict[str, List[str]]] = None,
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 0,
    smooth: int = 0,
    multi_subject_mode: str = "individual",
    return_data: bool = False,
    cache: bool = False,
    wandb_mode: str = "online",
    tags: List[str] = None,
    **kwargs,
):
    config = {
        key: value
        for key, value in locals().items()
        if key not in config_ignore and value is not None
    }
    config.update(kwargs)
    config_wandb = {
        key: value for key, value in config.items() if key not in wandb_ignore
    }
    for key, value in config.items():
        if isinstance(value, dict):
            config_wandb[key + "_id"] = json.dumps(value, sort_keys=True)
    wandb.init(
        config=config_wandb,
        id=sha1(repr(sorted(config.items())).encode()).hexdigest(),
        project="fMRI-Decoding-v6",
        save_code=True,
        mode=wandb_mode,
        tags=tags,
    )
    if cache:
        output = memory.cache(main, ignore=cache_ignore)(**config)
    else:
        output = main(**config)
    wandb.finish()

    return output
