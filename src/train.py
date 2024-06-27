import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.brain_decoder import train_brain_decoder
from src.fetch_data import fetch_data
from src.ridge import fast_ridge, fast_ridge_cv, ridge
from src.utils import console, memory

"""
train.py

This module contains functions for training and fetching data.

"""


@memory.cache
def train(
    subject: str = "UTS00",
    decoder: str = "ridge",
    model: str = "clip",
    context_length: int = 2,
    tr: int = 2,
    lag: int = 2,
    smooth: int = 1,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 0,
    subsample_voxels: int = None,
    latents_batch_size: int = 64,
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
    np.random.seed(seed)
    data = fetch_data(
        subject=subject,
        model=model,
        tr=tr,
        context_length=context_length,
        subsample_voxels=subsample_voxels,
        smooth=smooth,
        lag=lag,
        batch_size=latents_batch_size,
    )
    stories = list(data.keys())
    n_stories = len(stories)
    shuffled_indices = np.random.permutation(n_stories)
    Xs = [Xs[i] for i in shuffled_indices]
    Ys = [Ys[i] for i in shuffled_indices]
    stories = stories[shuffled_indices]
    lengths = [x.shape[0] for x in Xs]
    n_valid = max(1, int(valid_ratio * n_stories))
    n_test = max(1, int(test_ratio * n_stories))
    n_train = n_stories - n_valid - n_test
    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(np.concatenate(Xs[n_test + n_valid :]))
    lengths_train = lengths[n_test + n_valid :]
    X_valid = scaler.transform(np.concatenate(Xs[n_test : n_test + n_valid]))
    lengths_valid = lengths[n_test : n_test + n_valid]
    X_test = scaler.transform(np.concatenate(Xs[:n_test]))
    lengths_test = lengths[:n_test]
    Y_train = scaler.fit_transform(np.concatenate(Ys[n_test + n_valid :]))
    Y_valid = scaler.transform(np.concatenate(Ys[n_test : n_test + n_valid]))
    Y_test = scaler.transform(np.concatenate(Ys[:n_test]))

    if not decoder.endswith("_cv"):
        console.log(
            f"X_train: {X_train.shape}, Y_train: {Y_train.shape}\n"
            f"X_valid: {X_valid.shape}, Y_valid: {Y_valid.shape}\n"
            f"X_test: {X_test.shape}, Y_test: {Y_test.shape}\n"
            f"{n_train} train stories, {n_valid} valid stories, {n_test} test stories"
        )

    if decoder.lower() == "fast_ridge":
        output = fast_ridge(
            X_train,
            X_valid,
            X_test,
            Y_train,
            Y_valid,
            Y_test,
            **decoder_params,
        )
    elif decoder.lower() == "fast_ridge_cv":
        output = fast_ridge_cv(Xs, Ys, **decoder_params)
    elif decoder.lower() == "ridge":
        output = ridge(
            X_train,
            X_valid,
            X_test,
            Y_train,
            Y_valid,
            Y_test,
            **decoder_params,
        )
    else:
        if decoder.lower() in ["rnn", "gru", "lstm"]:
            X_train = np.split(X_train, np.cumsum(lengths_train)[:-1])
            X_valid = np.split(X_valid, np.cumsum(lengths_valid)[:-1])
            X_test = np.split(X_test, np.cumsum(lengths_test)[:-1])
            Y_train = np.split(Y_train, np.cumsum(lengths_train)[:-1])
            Y_valid = np.split(Y_valid, np.cumsum(lengths_valid)[:-1])
            Y_test = np.split(Y_test, np.cumsum(lengths_test)[:-1])
        output = train_brain_decoder(
            X_train,
            Y_train,
            X_valid,
            Y_valid,
            X_test,
            Y_test,
            decoder=decoder,
            seed=seed,
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
