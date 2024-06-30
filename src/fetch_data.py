import os
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.prepare_latents import prepare_latents
from src.utils import progress


def fetch_data(
    subject: str,
    model: str,
    tr: int,
    context_length: int,
    subsample_voxels: int,
    smooth: int,
    lag: int,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray]]:
    brain_images_path = Path("data/lebel/derivative/preprocessed_data") / subject
    runs = [run.replace(".hf5", "") for run in os.listdir(brain_images_path)]
    n_voxels = h5py.File(
        brain_images_path / f"{runs[0]}.hf5",
        "r",
    )[
        "data"
    ].shape[1]
    if subsample_voxels is not None:
        selected_voxels = np.random.permutation(n_voxels)[:subsample_voxels]
        selected_voxels = np.sort(selected_voxels)
    else:
        selected_voxels = np.arange(n_voxels)

    Xs, Ys = {}, {}
    task = progress.add_task("", total=len(runs))
    for run in runs:
        file_path = brain_images_path / f"{run}.hf5"
        file = h5py.File(file_path, "r")["data"]
        X = file[:, selected_voxels].astype(np.float32)
        X = np.nan_to_num(X, nan=0)
        if smooth > 0:
            new_X = X.copy()
            count = np.ones((X.shape[0], 1))
            for i in range(1, smooth + 1):
                new_X[i:] += X[:-i]
                count[i:] += 1
            X = new_X / count
        X = StandardScaler().fit_transform(X[lag:])
        Y = prepare_latents(run, model, tr, context_length, batch_size)
        Y = StandardScaler().fit_transform(Y[:-lag])
        if Y.shape[0] > X.shape[0]:
            # More latents than brain scans (first brain scans where removed)
            # We drop the corresponding latents
            Y = Y[-X.shape[0] :]
        Xs[run] = X
        Ys[run] = Y
        progress.update(task, description=f"Run: {run}", advance=1)
    progress.remove_task(task)

    return Xs, Ys
