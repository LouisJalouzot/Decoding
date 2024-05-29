from typing import Dict

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances, r2_score


def corr(X: np.ndarray, Y: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calculate the correlation between two arrays along a given axis.

    Args:
        X: Input array.
        Y: Input array.
        axis: Axis along which to calculate the correlation.

    Returns:
        Correlation array.

    """
    mX = X - np.mean(X, axis=axis, keepdims=True)
    mY = Y - np.mean(Y, axis=axis, keepdims=True)
    norm_mX = np.sqrt(np.sum(mX**2, axis=axis, keepdims=True))
    norm_mX[norm_mX == 0] = 1.0
    norm_mY = np.sqrt(np.sum(mY**2, axis=axis, keepdims=True))
    norm_mY[norm_mY == 0] = 1.0
    return np.sum(mX / norm_mX * mY / norm_mY, axis=axis)


def latent_rank(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    metric: str = "cosine",
    n_jobs: int = -2,
) -> np.ndarray:
    """
    Calculate the latent rank between true and predicted values.

    Args:
        Y_true: True values.
        Y_pred: Predicted values.
        metric: Distance metric to use for calculating pairwise distances.
        n_jobs: Number of parallel jobs to run for pairwise distance calculation.

    Returns:
        Latent rank array.

    """
    return np.diagonal(
        rankdata(
            pairwise_distances(Y_true, Y_pred, metric=metric, n_jobs=n_jobs),
            axis=1,
        )
    )


def scores(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    metric: str = "cosine",
    n_jobs: int = -2,
) -> Dict[str, np.ndarray]:
    """
    Calculate different scores between true and predicted values.

    Args:
        Y_true: True values.
        Y_pred: Predicted values.
        metric: Distance metric to use for calculating pairwise distances.
        n_jobs: Number of parallel jobs to run for pairwise distance calculation.

    Returns:
        Dictionary of scores.

    """
    ranks = latent_rank(Y_true, Y_pred, metric=metric, n_jobs=n_jobs)
    return {
        "mse": ((Y_true - Y_pred) ** 2).mean(axis=0),
        "r2": r2_score(Y_true, Y_pred, multioutput="raw_values"),
        "r": corr(Y_true, Y_pred, axis=0),
        "ranks": ranks,
        "median_rank": np.median(ranks),
    }
