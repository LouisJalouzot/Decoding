import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from joblib import memory
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

memory = memory.Memory(location=".cache", compress=9, verbose=0)

progress = Progress(
    SpinnerColumn(),
    TaskProgressColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
)

ignore = ["verbose", "n_jobs", "latents_batch_size"]


def create_symlink(input: Path, target: Path):
    input.symlink_to(target.relative_to(input.parent, walk_up=True))


def _get_free_gpu():
    rows = (
        subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
        )
        .decode("utf-8")
        .split("\n")
    )
    free_rams = tuple(map(lambda x: float(x.rstrip(" [MiB]")), rows[1:-1]))
    max_free = max(free_rams)
    max_free_idxs = tuple(
        i for i in range(len(free_rams)) if abs(max_free - free_rams[i]) <= 200
    )
    return np.random.choice(max_free_idxs)


if torch.cuda.is_available():
    device = torch.device(f"cuda:{_get_free_gpu()}")
else:
    device = torch.device("cpu")

console.log("Running on device", device)


def ewma(data, halflife):
    # TODO make it robust to division by 0
    if halflife == 0:
        return data
    alpha_rev = np.exp(-np.log(2) / halflife)
    n = data.shape[0]
    pows = alpha_rev ** (np.arange(n + 1))
    scale_arr = 1 / pows[:-1]
    offsets = data[0] * pows[1:, None]
    pw0 = (1 - alpha_rev) * alpha_rev ** (n - 1)
    mult = data * pw0 * scale_arr[:, None]
    cumsums = mult.cumsum(0)
    out = offsets + cumsums * scale_arr[::-1, None]
    return out


class CustomDataloader:
    def __init__(
        self,
        X_ds: xr.DataArray,
        Y_ds: xr.DataArray,
        batch_size: int,
        shuffle: bool = False,
        per_subject: bool = False,
    ):
        self.X_ds = X_ds
        self.Y_ds = Y_ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.per_subject = per_subject
        self.run_ids = X_ds.run_id.values
        self.n_run_ids = len(self.run_ids)

    def __iter__(self):
        if self.per_subject:
            for subject, X_ds_subject in self.X_ds.groupby("subject"):
                run_ids = X_ds_subject.run_id.values
                for i in range(0, len(run_ids), self.batch_size):
                    batch_run_ids = run_ids[i : i + self.batch_size]
                    X_ds_batch = X_ds_subject.sel(run_id=batch_run_ids)
                    Y_ds_batch = self.Y_ds.sel(run=X_ds_batch.run)
                    yield subject, X_ds_batch, Y_ds_batch
        else:
            if self.shuffle:
                np.random.shuffle(self.run_ids)
            for i in range(0, self.n_run_ids, self.batch_size):
                batch_run_ids = self.run_ids[i : i + self.batch_size]
                X_ds_batch = self.X_ds.sel(run_id=batch_run_ids)
                Y_ds_batch = self.Y_ds.sel(run=X_ds_batch.run)
                yield X_ds_batch, Y_ds_batch
