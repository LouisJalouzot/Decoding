import subprocess

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

from src.textgrids import TextGrid

console = Console()

memory = memory.Memory(location=".cache", compress=9, verbose=0)


DEFAULT_BAD_WORDS = frozenset(
    ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", ""]
)

ignore = ["verbose", "n_jobs"]


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


def _get_progress(**kwargs):
    return Progress(
        SpinnerColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        **kwargs,
    )


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
