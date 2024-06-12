from typing import Dict, List, Union

import numpy as np
import torch
import torchmetrics.functional as tmf
from sklearn.metrics import pairwise_distances, r2_score

from src.utils import console


def corr(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    axis: int = -1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate the correlation between two arrays or tensors along a given axis.

    Args:
        X: Input array or tensor.
        Y: Input array or tensor.
        axis: Axis along which to calculate the correlation.

    Returns:
        Correlation array or tensor.

    """
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        mX = X - np.mean(X, axis=axis, keepdims=True)
        mY = Y - np.mean(Y, axis=axis, keepdims=True)
        norm_mX = np.sqrt(np.sum(mX**2, axis=axis, keepdims=True))
        norm_mX[norm_mX == 0] = 1.0
        norm_mY = np.sqrt(np.sum(mY**2, axis=axis, keepdims=True))
        norm_mY[norm_mY == 0] = 1.0
        return np.sum(mX / norm_mX * mY / norm_mY, axis=axis)
    elif isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor):
        mX = X - torch.mean(X, dim=axis, keepdim=True)
        mY = Y - torch.mean(Y, dim=axis, keepdim=True)
        norm_mX = torch.sqrt(torch.sum(mX**2, dim=axis, keepdim=True))
        norm_mX[norm_mX == 0] = 1.0
        norm_mY = torch.sqrt(torch.sum(mY**2, dim=axis, keepdim=True))
        norm_mY[norm_mY == 0] = 1.0
        return torch.sum(mX / norm_mX * mY / norm_mY, dim=axis)
    else:
        raise ValueError(
            "Input types not supported. Supported types are np.ndarray and torch.Tensor."
        )


def retrieval_metrics(
    Y_true: Union[np.ndarray, torch.Tensor],
    Y_pred: Union[np.ndarray, torch.Tensor],
    negatives: Union[np.ndarray, torch.Tensor] = None,
    metric: str = "cosine",
    top_k_accuracies: List[int] = [],
    n_jobs: int = -2,
    return_ranks: bool = False,
) -> Dict[str, float]:
    """
    Calculate retrieval rank metrics between true and predicted values.

    Args:
        Y_true: True values.
        Y_pred: Predicted values.
        negatives: Negative values for contrastive loss.
        metric: Distance metric to use for calculating pairwise distances.
        top_k_accuracies: List of top-k values for accuracy calculation.
        n_jobs: Number of parallel jobs to run for pairwise distance calculation.

    Returns:
        Dictionary with batch size, median relative rank, and top-k accuracies.

    """
    if Y_true.shape[0] != Y_pred.shape[0]:
        console.log(
            f"[bold red]Batch size mismatch:[/] {Y_true.shape[0]} {Y_pred.shape[0]}"
        )
    if negatives is None:
        negatives = Y_true
    size = len(negatives)
    output = {"size": size}
    if isinstance(Y_true, np.ndarray) and isinstance(Y_pred, np.ndarray):
        # TODO Adapt the numpy part to mimic the pytorch part with negatives
        pdists = pairwise_distances(
            Y_true,
            Y_pred,
            metric=metric,
            n_jobs=n_jobs,
        )
        ground_truth_dist = pdists.diagonal()[:, None]
        ranks = (pdists < ground_truth_dist).sum(1)
        output["relative_median_rank"] = (np.median(ranks) - 1) / (size - 1)
        for top_k in top_k_accuracies:
            accuracy = (ranks < top_k).mean()
            output[f"top_{top_k}_accuracy"] = accuracy
    elif (
        isinstance(Y_true, torch.Tensor)
        and isinstance(Y_pred, torch.Tensor)
        and isinstance(negatives, torch.Tensor)
    ):
        if metric == "cosine":
            dist_to_negatives = 1 - tmf.pairwise_cosine_similarity(
                Y_pred,
                negatives,
            )
            dist_to_ground_truth = 1 - torch.cosine_similarity(
                Y_true, Y_pred, dim=1
            ).reshape(-1, 1)
        elif metric == "euclidean":
            dist_to_negatives = tmf.pairwise_euclidean_distance(
                Y_pred,
                negatives,
            )
            dist_to_ground_truth = (
                ((Y_true - Y_pred) ** 2).sum(dim=1).sqrt().reshape(-1, 1)
            )
        else:
            raise ValueError(
                "Metric not supported. Supported metrics are cosine and euclidean."
            )
        ranks = (dist_to_ground_truth > dist_to_negatives).sum(1).float()
        output["relative_median_rank"] = torch.quantile(ranks, q=0.5).item() / size
        for top_k in top_k_accuracies:
            accuracy = (ranks < top_k).float().mean().item()
            output[f"top_{top_k}_accuracy"] = accuracy
        ranks = ranks.cpu()
    else:
        raise ValueError(
            "Input types not supported. Supported types are np.ndarray and torch.Tensor."
        )
    if return_ranks:
        output["relative_ranks"] = ranks / size
    return output


def scores(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    metric: str = "cosine",
    top_k_accuracies: List[int] = [1, 5, 10, 50],
    n_jobs: int = -2,
) -> Dict[str, np.ndarray]:
    """
    Calculate different scores between true and predicted values.

    Args:
        Y_true: True values.
        Y_pred: Predicted values.
        metric: Distance metric to use for calculating pairwise distances.
        top_k_accuracies: List of top-k values for accuracy calculation.
        n_jobs: Number of parallel jobs to run for pairwise distance calculation.

    Returns:
        Dictionary of scores.

    """
    output = {
        "mse": ((Y_true - Y_pred) ** 2).mean(axis=0),
        "r2": r2_score(Y_true, Y_pred, multioutput="raw_values"),
        "r": corr(Y_true, Y_pred, axis=0),
    }
    output.update(
        retrieval_metrics(
            Y_true,
            Y_pred,
            None,
            metric,
            top_k_accuracies,
            n_jobs,
        )
    )
    return output
