import os
from pathlib import Path
from typing import Dict, Tuple

import h5py
import nibabel as nib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.prepare_latents import prepare_latents
from src.utils import console, progress


def fetch_data(
    subject_path: str,
    model: str,
    tr: int,
    context_length: int,
    smooth: int,
    lag: int,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray]]:
    working_dir = Path(os.getcwd())
    # run names should not contain dots
    runs = [Path(subject_path) / f.split(".")[0] for f in os.listdir(subject_path)]
    runs = sorted(list(set(runs)))  # drop duplicates
    Xs, Ys = {}, {}
    task = progress.add_task(f"Retrieving runs for {subject_path}", total=len(runs))
    for i, run in enumerate(runs):
        if run.with_suffix(".npy"):
            X = np.load(run.with_suffix(".npy"))
        elif run.with_suffix(".hf5"):
            with h5py.File(run.with_suffix(".hf5"), "r") as f:
                X = f["data"][:]
        elif run.with_suffix(".nii.gz"):
            X = nib.load(run.with_suffix(".nii.gz")).get_fdata()
            X = np.moveaxis(X, -1, 0)
            X = X.reshape(X.shape[0], -1)
        elif run.with_suffix(".npz"):
            X = np.load(run.with_suffix(".npz"), allow_pickle=True)["arr_0"]
        else:
            raise FileNotFoundError(f"No brain image found at {runs[0]}")
        X = X.astype(np.float32)
        X = np.nan_to_num(X, nan=0)
        if smooth > 0:
            new_X = X.copy()
            count = np.ones((X.shape[0], 1))
            for i in range(1, smooth + 1):
                new_X[i:] += X[:-i]
                count[i:] += 1
            X = new_X / count
        # Resolve paths for more efficient caching
        # Make paths relative for persistent caching across machines
        textgrid_path = (
            run.with_suffix(".TextGrid")
            .resolve()
            .relative_to(working_dir, walk_up=True)
        )
        audio_path = (
            run.with_suffix(".wav").resolve().relative_to(working_dir, walk_up=True)
        )
        Y = prepare_latents(
            textgrid_path, audio_path, model, tr, context_length, batch_size
        )
        if "lebel2023" in str(subject_path):
            Y = Y[5:-5]  # trim first and last 10 seconds on Lebel
        if lag > 0:
            X = X[lag:]
            Y = Y[:-lag]
        elif lag < 0:
            X = X[:lag]
            Y = Y[-lag:]
        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y)
        if Y.shape[0] > X.shape[0]:
            if Y.shape[0] > X.shape[0] + 1:
                console.log(
                    f"[red]{Y.shape[0] - X.shape[0]} > 1 latents trimmed for subject {run}"
                )
            # More latents than brain scans, drop last seconds of run
            Y = Y[: X.shape[0]]
        assert Y.shape[0] == X.shape[0]
        Xs[run] = X
        Ys[run] = Y
        progress.update(task, advance=1)
    progress.remove_task(task)

    return Xs, Ys
