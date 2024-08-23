import json
import os
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Union

import wandb
from src.train import train
from src.utils import console, device, ignore, memory

wandb.require("core")


def main(
    datasets: Union[str, List[str]] = ["lebel2023"],
    subjects: Dict[str, List[str]] = None,
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 0,
    smooth: int = 0,
    multi_subject_mode: str = "individual",
    return_data: bool = False,
    caching: bool = True,
    **kwargs,
):
    console.log("Running on device", device)
    if subjects is None:
        subjects = {
            dataset: sorted(os.listdir(f"datasets/{dataset}")) for dataset in datasets
        }
    runs = {
        dataset: {
            subject: sorted(
                [Path(f).stem for f in os.listdir(f"datasets/{dataset}/{subject}")]
            )
            for subject in subjects[dataset]
        }
        for dataset in subjects
    }

    config = {key: value for key, value in locals().items() if key != "kwargs"}
    config.update(kwargs)
    config_wandb = {key: value for key, value in config.items() if key not in ignore}
    for key, value in config.items():
        if isinstance(value, dict):
            config_wandb[key + "_id"] = json.dumps(value, sort_keys=True)
    wandb.init(
        config=config_wandb,
        id=sha1(repr(sorted(config.items())).encode()).hexdigest(),
        project="fMRI-Decoding-v4",
        save_code=True,
        mode=("disabled" if return_data else "online"),
    )
    if caching:
        output = memory.cache(train)(**config)
    else:
        output = train(**config)
    wandb.finish()
    return output
