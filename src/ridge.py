from typing import Dict, List, Union

import numpy as np
from sklearn.preprocessing import RobustScaler

from src.metrics import scores
from src.utils import _get_progress


def fast_ridge(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_valid: np.ndarray,
    Y_test: np.ndarray,
    alphas: np.ndarray = np.logspace(-3, 10, 10),
    verbose: bool = False,
) -> Dict[str, Union[np.ndarray, float]]:
    n_features = X_train.shape[1]
    n_targets = Y_train.shape[1]
    output = {}
    with _get_progress(transient=not verbose) as progress:
        if verbose:
            task = progress.add_task(
                "Fitting for each alpha",
                total=2 * len(alphas),
            )

        coefs = []
        val_mse = []
        for alpha in alphas:
            c = (
                np.linalg.pinv(X_train.T @ X_train + alpha * np.eye(n_features))
                @ X_train.T
                @ Y_train
            )
            coefs.append(c)
            val_mse.append(((X_valid @ c - Y_valid) ** 2).mean(axis=0))

            if verbose:
                progress.update(task, description="Fine-tuning", advance=1)
        val_mse = np.stack(val_mse)
        best_alpha_index = val_mse.argmin(axis=0)
        output["alpha"] = alphas[best_alpha_index]

        X_train = np.vstack([X_train, X_valid])
        Y_train = np.vstack([Y_train, Y_valid])
        coefs = []
        for alpha in alphas:
            coefs.append(
                np.linalg.pinv(X_train.T @ X_train + alpha * np.eye(n_features))
                @ X_train.T
                @ Y_train
            )

            if verbose:
                progress.update(task, description="Fitting", advance=1)
        coefs = np.stack(coefs)
        coefs = coefs[best_alpha_index, :, np.arange(n_targets)]
        output["coefs"] = coefs

        for t, Y_true, Y_pred in [
            ("train", Y_train, X_train @ coefs.T),
            ("valid", Y_valid, X_valid @ coefs.T),
            ("test", Y_test, X_test @ coefs.T),
        ]:
            for key, value in scores(Y_true, Y_pred).items():
                output[f"{t}_{key}"] = value

    return output


def fast_ridge_cv(
    Xs: List[np.ndarray],
    Ys: List[np.ndarray],
    alphas: np.ndarray = np.logspace(-3, 10, 10),
    verbose: bool = False,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Perform fast ridge regression.

    Args:
        Xs: List of input arrays.
        Ys: List of target arrays.
        alphas: Array of alpha values.
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
                    Y_train,
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
            if verbose:
                progress.update(task, advance=1)
    for key in output:
        output[key] /= len(n_scans)
    return output
