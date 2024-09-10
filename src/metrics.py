from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torchmetrics as tm
from sklearn.metrics import r2_score

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
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    negatives: torch.Tensor = None,
    metric: str = "cosine",
    top_k_accuracies: List[int] = [],
    return_ranks: bool = False,
):
    """
    Calculate retrieval rank metrics between true and predicted values.

    Args:
        Y_true: True values.
        Y_pred: Predicted values.
        negatives: Negative values for contrastive loss.
        metric: Distance metric to use for calculating pairwise distances.
        top_k_accuracies: List of top-k values for accuracy calculation.

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
    if metric == "cosine":
        dist_to_negatives = 1 - tm.functional.pairwise_cosine_similarity(
            Y_pred,
            negatives,
        )
        dist_to_ground_truth = 1 - torch.cosine_similarity(
            Y_true, Y_pred, dim=1
        ).reshape(-1, 1)
    elif metric == "euclidean":
        dist_to_negatives = tm.functional.pairwise_euclidean_distance(
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
        accuracy = ranks < top_k
        accuracy = accuracy.cpu().numpy().mean()
        output[f"top_{top_k}_accuracy"] = accuracy
    ranks = ranks.cpu()
    if return_ranks:
        output["relative_ranks"] = ranks / size
    return output


def scores(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "median_r2": np.median(
            r2_score(Y_true, Y_pred, multioutput="raw_values")
        ),
        "median_r": np.median(corr(Y_true, Y_pred, axis=0)),
    }


def nlp_metrics(
    Y_pred: torch.Tensor,
    true_chunks: List[str],
    candidates: torch.Tensor,
    chunks_with_context: pd.Series,
    metric: str = "cosine",
    n_nlp_candidates: int = 10,
):
    if metric == "cosine":
        dist_to_candidates = 1 - tm.functional.pairwise_cosine_similarity(
            Y_pred,
            candidates,
        )
    elif metric == "euclidean":
        dist_to_candidates = tm.functional.pairwise_euclidean_distance(
            Y_pred,
            candidates,
        )
    else:
        raise ValueError(
            "Metric not supported. Supported metrics are cosine and euclidean."
        )
    candidates_sorted = dist_to_candidates.argsort(axis=1)[:, :n_nlp_candidates]
    top_candidates = chunks_with_context.values[candidates_sorted.flatten()]
    bertscore = tm.text.bert.BERTScore(
        "microsoft/deberta-xlarge-mnli", idf=True, rescale_with_baseline=True
    )
