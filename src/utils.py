import subprocess
from typing import List, Union

import numpy as np
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
    ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", ""]
)


def get_textgrid(textgrid_path):
    textgrid = TextGrid(textgrid_path)
    grtranscript = textgrid.tiers[1].make_simple_transcript()
    grtranscript = [
        (float(start), float(stop), "," if text == "sp" else text)
        for start, stop, text in grtranscript
        if text.lower().strip("{}").strip() not in DEFAULT_BAD_WORDS
    ]
    if grtranscript[0][2] == ",":
        grtranscript = grtranscript[1:]
    if grtranscript[-1][2] == ",":
        grtranscript = grtranscript[:-1]
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


class DataloaderTimeSeries:
    def __init__(
        self,
        X: Union[List[torch.Tensor], List[np.array]],
        Y: Union[List[torch.Tensor], List[np.array]],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.X = [torch.Tensor(x) for x in X]
        self.Y = [torch.Tensor(y) for y in Y]
        self.lengths = np.array([x.shape[0] for x in X])
        self.n_stories = len(self.lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert len(X) == len(Y)
        assert all(x.shape[0] == y.shape[0] for x, y in zip(X, Y))

        if np.all(self.lengths > batch_size):
            raise ValueError(
                f"All stories lengths are greater than the batch_size ({batch_size})."
            )
        elif np.any(self.lengths > batch_size):
            console.log(
                f"[red]{(self.lengths > batch_size).sum()} stories have a length greater than the batch_size ({batch_size}) so they won't be used."
            )
            self.X = [x for x in X if x.shape[0] <= batch_size]
            self.Y = [y for y in Y if y.shape[0] <= batch_size]
            self.lengths = [l for l in self.lengths if l <= batch_size]
            self.n_stories = len(self.lengths)

    def __iter__(self):
        indices = np.arange(self.n_stories)
        if self.shuffle:
            np.random.shuffle(indices)
        self.batches = []
        sequence_index = 0
        while sequence_index < self.n_stories:
            batch_length = 0
            batch_indices = []
            while sequence_index < self.n_stories:
                i = indices[sequence_index]
                length = self.lengths[i]
                if batch_length + length > self.batch_size:
                    break
                batch_indices.append(i)
                sequence_index += 1
                batch_length += length
            self.batches.append(batch_indices)
        return self

    def __next__(self):
        if self.batches == []:
            raise StopIteration
        else:
            batch_indices = self.batches.pop(0)
            X = pack_sequence([self.X[i] for i in batch_indices], enforce_sorted=False)
            Y = torch.cat([self.Y[i] for i in batch_indices])
            return X, Y
