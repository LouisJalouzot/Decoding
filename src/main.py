import os
from hashlib import sha1
from typing import List, Union

import wandb
from src.train import train
from src.utils import ignore, memory


# @memory.cache
def main(
    datasets: Union[str, List[str]] = "lebel2023/all_subjects",
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 0,
    smooth: int = 0,
    multi_subject_mode: str = "individual",
    **kwargs,
):
    if isinstance(datasets, str):
        subjects = [f"{datasets}/{f}" for f in os.listdir(f"data/{datasets}")]
    elif isinstance(datasets, list):
        subjects = []
        for dataset in datasets:
            subjects.extend([f"{datasets}/{f}" for f in os.listdir(f"data/{dataset}")])
    config = {
        key: value
        for key, value in locals().items()
        if key not in ignore and key != "kwargs"
    }
    config.update(kwargs)
    wandb.init(
        config=config,
        id=sha1(repr(sorted(config.items())).encode()).hexdigest(),
        project="fMRI-Decoding",
        save_code=True,
    )
    output = train(**config)
    wandb.finish()
    return output
