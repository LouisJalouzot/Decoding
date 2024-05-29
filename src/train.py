import os
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler

from src.fast_ridge import fast_ridge, scores
from src.metrics import latent_rank
from src.prepare_latents import prepare_latents
from src.utils import _get_progress, console, device, memory

"""
train.py

This module contains functions for training and fetching data.

"""


def fetch_data(
    subject: str,
    model_class: str,
    model_name: str,
    tr: int,
    context_length: int,
    lag: int = 0,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Fetches data for training.

    Args:
        subject (str): The subject identifier.
        model_class (str): The model class.
        model_name (str): The model name.
        tr (int): The tr value.
        context_length (int): The context length.
        lag (int, optional): The lag value. Defaults to 0.
        verbose (bool, optional): Whether to display progress. Defaults to False.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]: The fetched data.

    """
    data_path = Path("data/lebel/derivative")
    brain_features_path = data_path / "preprocessed_data" / subject
    textgrids_path = data_path / "TextGrids"
    stories = [story.replace(".hf5", "") for story in os.listdir(brain_features_path)]
    Xs, Ys = [], []
    with _get_progress(transient=not verbose) as progress:

        if verbose:
            task = progress.add_task(
                f"Loading latents and brain images for subject {subject}",
                total=len(stories),
            )
        for story in stories:
            X = h5py.File(brain_features_path / f"{story}.hf5", "r")["data"][:]
            # Some voxels are NaNs, we replace them with zeros
            X = np.nan_to_num(X, nan=0)
            Y = prepare_latents(
                textgrids_path / f"{story}.TextGrid",
                model_class=model_class,
                model_name=model_name,
                tr=tr,
                context_length=context_length,
            )
            # If Y.shape[0] > X.shape[0] this means that the first brain scans where removed so we drop the corresponding latents
            Y = Y[-X.shape[0] :]
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

    return Xs, Ys, np.array(stories)


def train(
    subject: str = "UTS00",
    decoder: str = "ridge",
    model_class: str = "clip",
    model_name: str = "ViT-L/14",
    context_length: int = 2,
    tr: int = 2,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 0,
    alphas: Union[List[float], np.ndarray] = np.logspace(-3, 10, 10),
    verbose: bool = False,
    n_jobs: int = -2,
) -> Union[dict, Tuple[dict, RidgeCV, RobustScaler]]:
    """

    Trains the model.

    Args:
        subject (str, optional): The subject identifier. Defaults to "UTS00".
        decoder (str, optional): The decoder type. Defaults to "ridge".
        model_class (str, optional): The model class. Defaults to "clip".
        model_name (str, optional): The model name. Defaults to "ViT-L/14".
        context_length (int, optional): The context length. Defaults to 2.
        tr (int, optional): The tr value. Defaults to 2.
        valid_ratio (float, optional): The validation ratio. Defaults to 0.2.
        test_ratio (float, optional): The test ratio. Defaults to 0.1.
        seed (int, optional): The random seed. Defaults to 0.
        alphas (Union[List[float], np.ndarray], optional): The alpha values. Defaults to np.logspace(-3, 10, 10).
        verbose (bool, optional): Whether to display progress. Defaults to False.
        n_jobs (int, optional): The number of jobs. Defaults to -2.

    Returns:
        Union[dict, Tuple[dict, RidgeCV, RobustScaler]]: The training results.

    """
    np.random.seed(seed)
    Xs, Ys, stories = fetch_data(
        subject,
        model_class,
        model_name,
        tr,
        context_length,
        verbose=verbose,
    )
    if decoder.lower() == "fast_ridge":
        return fast_ridge(Xs, Ys, alphas=alphas, cv=False, verbose=verbose)
    elif decoder.lower() == "ridge":
        scaler = RobustScaler()
        model = RidgeCV(alphas=alphas, alpha_per_target=True)

        n_stories = len(stories)
        shuffled_indices = np.random.permutation(n_stories)
        Xs = [Xs[i] for i in shuffled_indices]
        Ys = [Ys[i] for i in shuffled_indices]

        stories = stories[shuffled_indices]
        n_test = max(1, int(test_ratio * n_stories))
        if verbose:
            console.log(
                f"{n_test} test stories: {', '.join(stories[:n_test])}",
            )
        if verbose:
            console.log(
                f"{n_stories - n_test} train stories: {', '.join(stories[n_test:])}",
            )
        X_train = np.concatenate(Xs[n_test:])
        X_test = np.concatenate(Xs[:n_test])
        Y_train = scaler.fit_transform(np.concatenate(Ys[n_test:]))
        Y_test = scaler.transform(np.concatenate(Ys[:n_test]))
        if verbose:
            console.log(f"Fitting RidgeCV on X: {X_train.shape} and Y: {Y_train.shape}")
        model = model.fit(X_train, Y_train)
        output = {}
        for t, (X, Y) in [
            ("train", (X_train, Y_train)),
            ("test", (X_test, Y_test)),
        ]:
            Y_pred = model.predict(X)
            for key, value in scores(Y, Y_pred).items():
                output[f"{t}_{key}"] = value
            output[f"{t}_ranks"] = latent_rank(Y, Y_pred)
            output[f"{t}_median_rank"] = np.median(output[f"{t}_ranks"])
        return output, model, scaler
