import os
from pathlib import Path
from typing import List, Tuple

import h5py
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
    batch_size: int = 64,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Fetches data for training.

    Args:
        subject (str): The subject identifier.
        model (str): The model to use.
        tr (int): The tr value.
        context_length (int): The context length.
        lag (int, optional): The lag value. Defaults to 0.
        verbose (bool, optional): Whether to display progress. Defaults to False.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]: The fetched data.

    """
    brain_images_path = Path("data/lebel/derivative/preprocessed_data") / subject
    stories = [story.replace(".hf5", "") for story in os.listdir(brain_images_path)]
    n_voxels = h5py.File(
        brain_images_path / f"{stories[0]}.hf5",
        "r",
    )[
        "data"
    ].shape[1]
    if subsample_voxels is not None:
        selected_voxels = np.random.permutation(n_voxels)[:subsample_voxels]
        selected_voxels = np.sort(selected_voxels)
    else:
        selected_voxels = np.arange(n_voxels)

    console.log(f"Fetching brain images and latents for {len(stories)} stories")
    output = {}
    with progress:
        task = progress.add_task("", total=len(stories))
        for story in stories:
            file_path = brain_images_path / f"{story}.hf5"
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
            Y = prepare_latents(story, model, tr, context_length, batch_size)
            Y = StandardScaler().fit_transform(Y[:-lag])
            if Y.shape[0] > X.shape[0]:
                # More latents than brain scans (first brain scans where removed)
                # We drop the corresponding latents
                Y = Y[-X.shape[0] :]
            output[story] = X, Y
            progress.update(task, description=f"Story: {story}", advance=1)

    return output
