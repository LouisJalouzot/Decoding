import subprocess
from typing import List, Union

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
from torch.nn.utils.rnn import pack_sequence

from src.textgrids import TextGrid

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

DEFAULT_BAD_WORDS = frozenset(
    ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", ""]
)


def get_textgrid(textgrid_path):
    textgrid = TextGrid(textgrid_path)
    grtranscript = textgrid.tiers[1].make_simple_transcript()
    grtranscript = [
        (float(start), float(stop), text)
        for start, stop, text in grtranscript
        if text.lower().strip("{}").strip() not in DEFAULT_BAD_WORDS
    ]
    return grtranscript


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

    def __iter__(self):
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
