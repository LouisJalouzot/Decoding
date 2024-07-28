import subprocess
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import xbatcher
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


def standard_scale(ds: xr.DataArray, along=["run_id", "tr"]):
    ds_mean = ds.mean(dim=along, skipna=True)
    ds_scale = ds.fillna(0).std(dim=along, skipna=True)
    ds_scale = xr.where(ds_scale < 1e-6, 1, ds_scale)
    return (ds - ds_mean) / ds_scale
