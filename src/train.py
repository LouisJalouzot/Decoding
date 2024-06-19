import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.brain_decoder import train_brain_decoder
from src.prepare_latents import prepare_latents
from src.ridge import fast_ridge, fast_ridge_cv, ridge
from src.utils import console, ignore, memory, progress

"""
train.py

This module contains functions for training and fetching data.

"""


def fetch_data(
    subject: str,
    model: str,
    tr: int,
    context_length: int,
    smooth: int,
    # halflife: float,
    lag: int,
    verbose: bool = False,
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
    brain_features_path = Path("data/lebel/derivative/preprocessed_data") / subject
    stories = [story.replace(".hf5", "") for story in os.listdir(brain_features_path)]
    Xs, Ys = [], []
    task = progress.add_task(
        f"Loading latents and brain images for subject {subject}",
        total=len(stories),
        visible=verbose,
    )
    for story in stories:
        X = h5py.File(brain_features_path / f"{story}.hf5", "r")["data"][:]
        # Some voxels are NaNs, we replace them with zeros
        X = np.nan_to_num(X, nan=0)
        # Smoothing brain signal using moving average
        new_X = X.copy()
        count = np.ones((X.shape[0], 1))
        for i in range(1, smooth + 1):
            new_X[i:] += X[:-i]
            count[i:] += 1
        X = new_X / count
        # if halflife > 0:
        #     X = pd.DataFrame(X).ewm(halflife=halflife).mean().to_numpy()
        Y = prepare_latents(
            story,
            model=model,
            tr=tr,
            context_length=context_length,
        )
        if lag > 0:
            X = X[lag:]
            Y = Y[:-lag]
        if Y.shape[0] > X.shape[0]:
            # More latents than brain scans (first brain scans where removed)
            # We drop the corresponding latents
            Y = Y[-X.shape[0] :]
        Xs.append(X.astype(np.float32))
        Ys.append(Y.astype(np.float32))

        progress.update(task, description=f"Story: {story}", advance=1)

    return Xs, Ys, np.array(stories)


# @memory.cache(ignore=["verbose"])
def train(
    subject: str = "UTS00",
    decoder: str = "ridge",
    model: str = "clip",
    context_length: int = 2,
    tr: int = 2,
    lag: int = 2,
    # halflife: int = 2,
    smooth: int = 1,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 0,
    subsample_voxels: int = None,
    verbose: bool = True,
    **decoder_params,
) -> dict:
    """

    Trains the model.

    Args:
        subject (str, optional): The subject identifier. Defaults to "UTS00".
        decoder (str, optional): The decoder type. Defaults to "ridge".
        model (str, optional): The model to use. Defaults to "clip".
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
    setup_config = {
        key: value
        for key, value in locals().items()
        if key not in ignore and key != "decoder_params"
    }
    np.random.seed(seed)
    Xs, Ys, stories = fetch_data(
        subject=subject,
        model=model,
        tr=tr,
        context_length=context_length,
        smooth=smooth,
        lag=lag,
        verbose=verbose,
    )
    n_stories = len(stories)
    shuffled_indices = np.random.permutation(n_stories)
    n_voxels = Xs[0].shape[1]
    if subsample_voxels is not None:
        selected_voxels = np.random.permutation(n_voxels)[:subsample_voxels]
    else:
        selected_voxels = np.arange(n_voxels)
    Xs = [Xs[i][:, selected_voxels] for i in shuffled_indices]
    Ys = [Ys[i] for i in shuffled_indices]
    stories = stories[shuffled_indices]
    n_valid = max(1, int(valid_ratio * n_stories))
    n_test = max(1, int(test_ratio * n_stories))
    n_train = n_stories - n_valid - n_test
    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(np.concatenate(Xs[n_test + n_valid :]))
    X_valid = scaler.transform(np.concatenate(Xs[n_test : n_test + n_valid]))
    X_test = scaler.transform(np.concatenate(Xs[:n_test]))
    Y_train = scaler.fit_transform(np.concatenate(Ys[n_test + n_valid :]))
    Y_valid = scaler.transform(np.concatenate(Ys[n_test : n_test + n_valid]))
    Y_test = scaler.transform(np.concatenate(Ys[:n_test]))

    if verbose and not decoder.endswith("_cv"):
        console.log(
            f"X_train: {X_train.shape}, Y_train: {Y_train.shape}\n"
            f"X_valid: {X_valid.shape}, Y_valid: {Y_valid.shape}\n"
            f"X_test: {X_test.shape}, Y_test: {Y_test.shape}\n"
            f"{n_train} train stories: {', '.join(stories[n_test+n_valid:])}\n"
            f"{n_valid} valid stories: {', '.join(stories[n_test:n_test+n_valid])}\n"
            f"{n_test} test stories: {', '.join(stories[:n_test])}"
        )

    if decoder.lower() == "fast_ridge":
        output = fast_ridge(
            X_train,
            X_valid,
            X_test,
            Y_train,
            Y_valid,
            Y_test,
            verbose=verbose,
            **decoder_params,
        )
    elif decoder.lower() == "fast_ridge_cv":
        output = fast_ridge_cv(Xs, Ys, verbose=verbose, **decoder_params)
    elif decoder.lower() == "ridge":
        output = ridge(
            X_train,
            X_valid,
            X_test,
            Y_train,
            Y_valid,
            Y_test,
            verbose=verbose,
            **decoder_params,
        )
    else:
        output = train_brain_decoder(
            X_train,
            Y_train,
            X_valid,
            Y_valid,
            X_test,
            Y_test,
            decoder=decoder,
            seed=seed,
            setup_config=setup_config,
            verbose=verbose,
            **decoder_params,
        )

    torch.cuda.empty_cache()

    console.log(
        f"Train relative median rank: {output['train/relative_median_rank']:.3f} "
        f"(size {int(output['train/size'])})\n"
        f"Test relative median rank: {output['test/relative_median_rank']:.3f} "
        f"(size {int(output['test/size'])})"
    )

    return output
