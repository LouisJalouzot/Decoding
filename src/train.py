import os
from pathlib import Path

import h5py
import numpy as np

from src.prepare_latents import prepare_latents
from src.utils import _get_progress, device, memory


def fetch_data(
    subject, model_class, model_name, tr, context_length, lag=0, verbose=False
):
    data_path = Path("data/lebel/derivative")
    brain_features_path = data_path / "preprocessed_data" / subject
    textgrids_path = data_path / "TextGrids"
    stories = [story.replace(".hf5", "") for story in os.listdir(brain_features_path)]
    Xs, Ys, n_scans = [], [], []
    with _get_progress(transient=not verbose) as progress:
        if verbose:
            task = progress.add_task(
                f"Loading latents and brain images for subject {subject}",
                total=len(stories),
            )

        for story in stories:
            X = h5py.File(brain_features_path / f"{story}.hf5", "r")["data"][:]
            Y = prepare_latents(
                textgrids_path / f"{story}.TextGrid",
                model_class=model_class,
                model_name=model_name,
                tr=tr,
                context_length=context_length,
            )
            n = X.shape[0]
            n_scans.append(n)
            # If Y.shape[0] > X.shape[0] this means that the first brain scans where removed so we drop the corresponding latents
            Y = Y[-n:]
            if lag > 0:
                Y_init = Y.copy()
                count = np.ones(Y.shape[0])
                for i in range(lag):
                    Y[i + 1 :] += Y_init[: -i - 1]
                    count[i + 1 :] += 1
                Y /= count[:, None]
            Xs.append(X)
            Ys.append(Y)

            if verbose:
                progress.update(task, description=f"Story: {story}", advance=1)

    return np.concatenate(Xs), np.concatenate(Ys), n_scans


def train(
    decoder="ridge",
    model="clip",
    model_name="ViT-L/14",
    tr=2,
    subject="UTS01",
    test_ratio=0.2,
    verbose=False,
):
    return
