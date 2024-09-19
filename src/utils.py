import string
import subprocess
import zipfile
from pathlib import Path
from typing import Union

import nltk
import numpy as np
import requests
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
    "return_data",
    "log_extra_metrics",
]
wandb_ignore = ignore + ["cache_model", "wandb_mode", "n_jobs", "verbose"]
nlp_cols = [
    "glove_bow",
    "POS",
    "POS_restricted",
    "glove_bow_POS_restricted",
]


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


def load_glove_embeddings():
    glove_file = Path("data/glove.6B.50d.txt")
    if not glove_file.exists():
        url = (
            "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        )
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 131072
        zip_path = Path("data/glove.6B.zip")

        with open(zip_path, "wb") as file, progress:
            task = progress.add_task(
                "Downloading GloVe embeddings", total=total_size // chunk_size
            )
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                progress.update(task, advance=1)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("glove.6B.50d.txt", "data/")
        zip_path.unlink()
        console.log(f"Saved GloVe embeddings to {glove_file}")

    embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def get_glove_bows(chunks):
    glove_embeddings = load_glove_embeddings()
    # Remove punctuation
    chunks = [
        chunk.translate(str.maketrans("", "", string.punctuation))
        for chunk in chunks
    ]
    chunks = [chunk.lower().split() for chunk in chunks]
    glove_bows = []
    for chunk in chunks:
        glove_bow = np.zeros(50)
        for word in chunk:
            glove_bow += glove_embeddings.get(word, np.zeros(50))
        glove_bows.append(glove_bow / max(1, len(chunk)))

    return glove_bows


def nltk_pos_tag(chunks):
    tokenized_chunks = [nltk.tokenize.word_tokenize(chunk) for chunk in chunks]
    pos_tagged_chunks = [
        [e[1] for e in nltk.tag.pos_tag(tokenized_chunk, tagset="universal")]
        for tokenized_chunk in tokenized_chunks
    ]
    return tokenized_chunks, pos_tagged_chunks
