import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

ignore = [
    "latents_batch_size",
    "return_data",
    "log_run_metrics",
    "extra_metrics",
    "extra_metrics_loop",
]
wandb_ignore = ignore + ["cache", "wandb_mode", "n_jobs", "verbose"]


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


def compute_gradient_norm(model, norm_type=2):
    """
    Computes the norm of gradients of all parameters in the model.

    Args:
        model (torch.nn.Module): The neural network model.
        norm_type (float): The type of norm to compute (default is 2, which is the L2 norm).

    Returns:
        float: The computed norm of the gradients.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)

    return total_norm
