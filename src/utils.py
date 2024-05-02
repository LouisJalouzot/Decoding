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

memory = memory.Memory(location=".cache", compress=9)


DEFAULT_BAD_WORDS = frozenset(
    ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"]
)


def get_textgrid(textgrid_path):
    textgrid = TextGrid(textgrid_path)
    grtranscript = textgrid.tiers[1].make_simple_transcript()
    return [
        x
        for x in grtranscript
        if x[2].lower().strip("{}").strip() not in DEFAULT_BAD_WORDS
    ]


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
