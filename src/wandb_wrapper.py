import json
from hashlib import sha1
from typing import Dict, List, Union

import wandb
from src.main import main
from src.utils import ignore, memory, wandb_ignore


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
    cache_model: bool = True,
    wandb_mode: str = "online",
    **kwargs,
):
    config = {
        key: value
        for key, value in locals().items()
        if key not in ["cache_model", "wandb_mode", "kwargs"]
        and value is not None
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
        # project="fMRI-Decoding-v5",
        project="Test",
        save_code=True,
        mode=wandb_mode,
    )
    if cache_model:
        output = memory.cache(main, ignore=ignore)(**config)
    else:
        output = main(**config)
    wandb.finish()

    return output
