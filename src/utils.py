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


class MultiSubjectBatchloader:
    def __init__(self, X: np.ndarray, Y: np.ndarray, subjects: np.ndarray):
        self.X = X
        self.Y = Y
        self.subjects = subjects
        self.unique_subjects = np.unique(subjects)

    def __iter__(self):
        for subject in self.unique_subjects:
            indices = self.subjects == subject
            yield subject, tuple(self.X[indices]), tuple(self.Y[indices])


class SingleSubjectDataloader:
    def __init__(self, X: pd.Series, Y: pd.Series, batch_size: int):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.X), self.batch_size):
            X_batch = tuple(self.X.values[i : i + self.batch_size])
            Y_batch = tuple(self.Y.values[i : i + self.batch_size])
            yield X_batch, Y_batch


class MultiSubjectDataloader:
    def __init__(
        self, Xs: pd.DataFrame, Ys: pd.DataFrame, batch_size: int, shuffle: bool = False
    ):
        self.Xs = Xs
        self.Ys = Ys
        self.batch_size = batch_size
        self.shuffle = shuffle
        indices = np.where(Xs.notna().values)
        self.runs_indices = indices[0]
        self.subjects_indices = indices[1]
        self.n_runs = len(self.runs_indices)
        self.indices = np.arange(self.n_runs)

    def __iter__(self, per_subject=False):
        if per_subject:
            for subject_id, subject in enumerate(self.Xs.columns):
                X = self.Xs[subject].dropna()
                Y = self.Ys[subject].dropna()
                subject_dl = SingleSubjectDataloader(X, Y, self.batch_size)
                yield subject_id, subject, subject_dl
        else:
            if self.shuffle:
                np.random.shuffle(self.indices)
            for i in range(0, self.n_runs, self.batch_size):
                batch_indices = self.indices[i : i + self.batch_size]
                runs_indices = self.runs_indices[batch_indices]
                subjects_indices = self.subjects_indices[batch_indices]
                X = self.Xs.values[runs_indices, subjects_indices]
                Y = self.Ys.values[runs_indices, subjects_indices]
                subjects = self.Xs.columns.values[subjects_indices]
                yield MultiSubjectBatchloader(X, Y, subjects)
