import itertools
import os
import subprocess
from typing import Union

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

console = Console(width=None if os.isatty(1) else 500)

memory = memory.Memory(location=".cache", verbose=0)

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


def merge_drop(df, merge_to_drop):
    indicator = df.merge(merge_to_drop, how="left", indicator=True)._merge
    indicator = (indicator != "both").values

    return df[indicator]


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


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def corr(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    axis: int = -1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate the correlation between two arrays or tensors along a given axis.

    Args:
        X: Input array or tensor.
        Y: Input array or tensor.
        axis: Axis along which to calculate the correlation.

    Returns:
        Correlation array or tensor.

    """
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        mX = X - np.mean(X, axis=axis, keepdims=True)
        mY = Y - np.mean(Y, axis=axis, keepdims=True)
        norm_mX = np.sqrt(np.sum(mX**2, axis=axis, keepdims=True))
        norm_mX[norm_mX == 0] = 1.0
        norm_mY = np.sqrt(np.sum(mY**2, axis=axis, keepdims=True))
        norm_mY[norm_mY == 0] = 1.0
        return np.sum(mX / norm_mX * mY / norm_mY, axis=axis)
    elif isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor):
        mX = X - torch.mean(X, dim=axis, keepdim=True)
        mY = Y - torch.mean(Y, dim=axis, keepdim=True)
        norm_mX = torch.sqrt(torch.sum(mX**2, dim=axis, keepdim=True))
        norm_mX[norm_mX == 0] = 1.0
        norm_mY = torch.sqrt(torch.sum(mY**2, dim=axis, keepdim=True))
        norm_mY[norm_mY == 0] = 1.0
        return torch.sum(mX / norm_mX * mY / norm_mY, dim=axis)
    else:
        raise ValueError(
            "Input types not supported. Supported types are np.ndarray and torch.Tensor."
        )


class BatchIncrementalMean:
    def __init__(self):
        self.n = 0

    def extend(self, new_values):
        batch_size = len(new_values)
        batch_sum = sum(new_values)
        new_n = self.n + batch_size
        if self.n == 0:
            self.mean = batch_sum / batch_size
        else:
            self.mean = (self.n * self.mean + batch_sum) / new_n
        self.n = new_n

        return self.mean


def batch_combinations(iterable, r, batch_size):
    # Create an iterator for the combinations
    combinations = itertools.combinations_with_replacement(iterable, r)
    batch = []

    # Yield combinations in batches
    for comb in combinations:
        batch.append(comb)
        if len(batch) == batch_size:
            yield np.array(batch).swapaxes(0, 1)
            batch = []

    # Yield any remaining combinations if they don't fill the last batch
    if batch:
        yield np.array(batch).swapaxes(0, 1)


def negatives_from_df(df):
    negatives = df.drop_duplicates(["dataset", "run"]).Y

    return torch.cat(tuple(negatives)).to(device)
