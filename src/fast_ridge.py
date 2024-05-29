from typing import Dict, List, Union

import numpy as np
from sklearn.preprocessing import RobustScaler

from src.metrics import scores
from src.utils import _get_progress


def fast_ridge(
    Xs: List[np.ndarray],
    Ys: List[np.ndarray],
    alphas: np.ndarray = np.logspace(-3, 10, 10),
    cv: bool = True,
    verbose: bool = False,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Perform fast ridge regression.

    Args:
        Xs: List of input arrays.
        Ys: List of target arrays.
        alphas: Array of alpha values.
        cv: Whether to perform cross-validation.
        verbose: Whether to display progress.

    Returns:
        Dictionary of results.

    """
    n_scans = [X.shape[0] for X in Xs]
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    indices = np.arange(X.shape[0])
    n_voxels = X.shape[1]
    n_features = Y.shape[1]
    start = 0
    output = {
        "coefs": np.zeros((n_features, n_voxels)),
        "alpha": np.zeros(n_features),
        "center": np.zeros(n_features),
        "scale": np.zeros(n_features),
    }
    scaler = RobustScaler()
    for score in ["mse", "r", "r2"]:
        for type in ["train", "test"]:
            output[f"{type}_{score}"] = np.zeros(n_features)
    with _get_progress(transient=not verbose) as progress:
        if cv and verbose:
            task = progress.add_task(f"Nested CV", total=len(n_scans) ** 2)
        for i, n in enumerate(n_scans):
            train_mask = (indices < start) | (indices >= start + n)
            test_mask = (indices >= start) & (indices < start + n)
            start_start = 0
            val_mse = np.zeros((len(alphas), n_features))
            for j, m in enumerate(n_scans):
                if i != j:
                    train_train_mask = train_mask & (
                        (indices < start_start) | (indices >= start_start + m)
                    )
                    val_mask = (indices >= start_start) & (indices < start_start + m)
                    X_train = X[train_train_mask]
                    Y_train = scaler.fit_transform(Y[train_train_mask])
                    coefs = (
                        np.linalg.pinv(
                            (X_train.T @ X_train)[None]
                            + alphas[:, None, None] * np.eye(n_voxels)[None]
                        )
                        @ X_train.T
                        @ Y_train
                    ).swapaxes(1, 2)
                    Y_val = scaler.fit_transform(Y[val_mask])
                    val_mse += ((Y_val.T - coefs @ X[val_mask].T) ** 2).mean(axis=-1)
                    if verbose:
                        progress.update(task, advance=1)
                start_start += m
                if not cv:
                    break
            best_alpha_index = val_mse.argmin(axis=0)
            output["alpha"] += alphas[best_alpha_index]
            X_train = X[train_mask]
            Y_train = scaler.fit_transform(Y[train_mask])
            output["center"] += scaler.center_
            output["scale"] += scaler.scale_
            coefs = (
                np.linalg.pinv(
                    (X_train.T @ X_train)[None]
                    + alphas[:, None, None] * np.eye(n_voxels)[None]
                )
                @ X_train.T
                @ Y_train
            )
            coefs = coefs[best_alpha_index, :, np.arange(n_features)]
            output["coefs"] += coefs
            for t, Y_true, Y_pred in [
                (
                    "train",
                    scaler.transform(Y[train_mask]),
                    X[train_mask] @ coefs.T,
                ),
                (
                    "test",
                    scaler.transform(Y[test_mask]),
                    X[test_mask] @ coefs.T,
                ),
            ]:
                for key, value in scores(Y_true, Y_pred).items():
                    output[f"{t}_{key}"] += value
            start += n
            if not cv:
                return output
            if verbose:
                progress.update(task, advance=1)
    for key in output:
        output[key] /= len(n_scans)
    return output
