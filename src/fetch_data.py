import os
import re
from pathlib import Path
from typing import Dict, Tuple

import h5py
import nibabel as nib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.prepare_latents import prepare_latents
from src.utils import console, progress


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
    dataset, subject = subject.split("/")
    if dataset.lower() == "lebel2023":
        brain_images_path = (
            Path("data/lebel2023/derivative/preprocessed_data") / subject
        )
    elif dataset.lower() == "li2022":
        brain_images_path = Path("data/li2022/derivatives") / subject / "func"
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    runs = sorted(os.listdir(brain_images_path))
    if runs[0].endswith(".hf5"):
        n_voxels = h5py.File(brain_images_path / runs[0], "r")["data"].shape[1]
    elif runs[0].endswith(".nii.gz"):
        n_voxels = np.prod(nib.load(brain_images_path / runs[0]).shape[:-1])
    elif runs[0].endswith(".npz"):
        n_voxels = np.load(brain_images_path / runs[0], allow_pickle=True)[
            "arr_0"
        ].shape[1]
    else:
        raise ValueError(f"File format not supported for {runs[0]}")
    if subsample_voxels is not None:
        selected_voxels = np.random.permutation(n_voxels)[:subsample_voxels]
        selected_voxels = np.sort(selected_voxels)
    else:
        selected_voxels = np.arange(n_voxels)

    Xs, Ys = {}, {}
    task = progress.add_task("", total=len(runs))
    for i, run in enumerate(runs):
        if run.endswith(".hf5"):
            file = h5py.File(brain_images_path / run, "r")["data"]
        elif run.endswith(".nii.gz"):
            file = nib.load(brain_images_path / run).get_fdata()
            file = np.moveaxis(file, -1, 0)
            file = file.reshape(file.shape[0], -1)
        elif run.endswith(".npz"):
            file = np.load(brain_images_path / run, allow_pickle=True)["arr_0"]
        else:
            raise ValueError(f"File format not supported for {run}")
        if dataset.lower() == "lebel2023":
            run = run.replace(".hf5", "")
        elif dataset.lower() == "li2022":
            run = i + 1
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
        Y = prepare_latents(dataset, run, model, tr, context_length, batch_size)
        Y = StandardScaler().fit_transform(Y)
        if dataset.lower() == "lebel2023":
            Y = Y[5:-5]  # trim first and last 10 seconds on Lebel
        if lag > 0:
            Y = Y[:-lag]
        if Y.shape[0] > X.shape[0]:
            if Y.shape[0] > X.shape[0] + 1:
                console.log(
                    f"[red]{Y.shape[0] - X.shape[0]} > 1 latents trimmed for subject {subject} and run {run}"
                )
            # More latents than brain scans, drop last seconds of run
            Y = Y[: X.shape[0]]
        assert Y.shape[0] == X.shape[0]
        Xs[run] = X
        Ys[run] = Y
        progress.update(task, description=f"Run: {run}", advance=1)
    progress.remove_task(task)

    return Xs, Ys
