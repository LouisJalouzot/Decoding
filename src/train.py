import os
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler

from src.brain_decoder import train_brain_decoder
from src.prepare_latents import prepare_latents
from src.ridge import fast_ridge, fast_ridge_cv, ridge
from src.skorch import skorch
from src.utils import _get_progress, console, device, ignore, memory

"""
train.py

This module contains functions for training and fetching data.

"""


def fetch_data(
    subject: str,
    model: str,
    tr: int,
    context_length: int,
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
                model=model,
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
            Xs.append(X.astype(np.float32))
            Ys.append(Y.astype(np.float32))

            if verbose:
                progress.update(task, description=f"Story: {story}", advance=1)

    return Xs, Ys, np.array(stories)


# @memory.cache(ignore=ignore)
def train(
    subject: str = "UTS00",
    decoder: str = "ridge",
    model: str = "clip",
    context_length: int = 2,
    tr: int = 2,
    lag: int = 2,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
    **decoder_params,
) -> Union[dict, Tuple[dict, RidgeCV, RobustScaler]]:
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
    setup_config = {key: value for key, value in locals().items() if key not in ignore and key != "decoder_params"}
    np.random.seed(seed)
    Xs, Ys, stories = fetch_data(
        subject,
        model,
        tr,
        context_length,
        lag,
        verbose=verbose,
    )
    n_stories = len(stories)
    shuffled_indices = np.random.permutation(n_stories)
    Xs = [Xs[i] for i in shuffled_indices]
    Ys = [Ys[i] for i in shuffled_indices]
    stories = stories[shuffled_indices]
    n_valid = max(1, int(valid_ratio * n_stories))
    n_test = max(1, int(test_ratio * n_stories))
    n_train = n_stories - n_valid - n_test
    X_train = np.concatenate(Xs[n_test + n_valid :])
    X_valid = np.concatenate(Xs[n_test : n_test + n_valid])
    X_test = np.concatenate(Xs[:n_test])
    scaler = RobustScaler()
    Y_train = scaler.fit_transform(np.concatenate(Ys[n_test + n_valid :]))
    Y_valid = scaler.transform(np.concatenate(Ys[n_test : n_test + n_valid]))
    Y_test = scaler.transform(np.concatenate(Ys[:n_test]))
    if verbose and not decoder.endswith("_cv"):
        console.log(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        console.log(f"X_valid: {X_valid.shape}, Y_valid: {Y_valid.shape}")
        console.log(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
        console.log(
            f"{n_train} train stories: {', '.join(stories[n_test+n_valid:])}",
        )
        console.log(
            f"{n_valid} valid stories: {', '.join(stories[n_test:n_test+n_valid])}",
        )
        console.log(
            f"{n_test} test stories: {', '.join(stories[:n_test])}",
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
    elif decoder.lower() in ["simple_mlp", "simple_mlp_contrastive", "brain_decoder_contrastive",]:
        output = skorch(
            X_train,
            Y_train,
            X_valid,
            Y_valid,
            X_test,
            Y_test,
            seed=seed,
            decoder=decoder,
            setup_config=setup_config,
            verbose=verbose,
            **decoder_params,
        )
    elif decoder.lower() == "brain_decoder":
        output = train_brain_decoder(
            X_train,
            Y_train,
            X_valid,
            Y_valid,
            X_test,
            Y_test,
            seed=seed,
            setup_config=setup_config,
            verbose=verbose,
            **decoder_params,
        )
        
    torch.cuda.empty_cache()

    console.log(
        f"Train relative median rank: {output["train/relative_median_rank"]:.3f} "
        f"(size {output["train/size"]})"
    )
    console.log(
        f"Test relative median rank: {output["test/relative_median_rank"]:.3f} "
        f"(size {output["test/size"]})"
    )
    
    return output
