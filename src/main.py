import os
from hashlib import sha1
from pathlib import Path
from typing import List, Union

import wandb
from src.train import train
from src.utils import ignore, memory

wandb.require("core")


@memory.cache
def main(
    datasets: Union[str, List[str]] = "lebel2023",
    subjects: List[str] = None,
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 0,
    smooth: int = 0,
    multi_subject_mode: str = "individual",
    **kwargs,
):
    if subjects is None:
        if isinstance(datasets, str):
            subjects = {datasets: f for f in os.listdir(f"datasets/{datasets}")}
        elif isinstance(datasets, list):
            subjects = {}
            for dataset in datasets:
                subjects[dataset] = os.listdir(f"datasets/{dataset}")
    runs = {
        dataset: {
            subject: [Path(f).stem for f in os.listdir(f"datasets/{dataset}/{subject}")]
            for subject in subjects[dataset]
        }
        for dataset in subjects
    }

    config = {
        key: value
        for key, value in locals().items()
        if key not in ignore and key != "kwargs"
    }
    config.update(kwargs)
    wandb.init(
        config=config,
        id=sha1(repr(sorted(config.items())).encode()).hexdigest(),
        project="fMRI-Decoding-v3",
        save_code=True,
    )
    output = train(**config)
    wandb.finish()
    return output
