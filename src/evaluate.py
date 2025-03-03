import logging
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
import torchmetrics as tm
from sklearn.metrics import r2_score

from src.base_decoders import MeanDecoder
from src.utils import corr, device

logger = logging.getLogger(__name__)


def retrieval_metrics(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    negatives: torch.Tensor = None,
    metric: str = "cosine",
    top_k_accuracies: List[int] = [],
    top_k_percent_accuracies: List[int] = [],
    return_ranks: bool = False,
    return_negatives_dist: bool = False,
):
    if Y_true.shape[0] != Y_pred.shape[0]:
        logger.warning(
            f"Batch size mismatch: {Y_true.shape[0]} {Y_pred.shape[0]}"
        )
    if negatives is None:
        negatives = Y_true
    retrieval_size = len(negatives)
    output = {}
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
    # All aggregations will be applied in aggregate_metrics_df
    ranks = (dist_to_ground_truth > dist_to_negatives).sum(1)
    ranks = ranks.double().cpu().numpy()
    for top_k in top_k_accuracies:
        accuracy = ranks < top_k
        output[f"top_{top_k}_accuracy"] = accuracy
    for top_kp in top_k_percent_accuracies:
        accuracy = ranks < retrieval_size * top_kp / 100
        output[f"top_{top_kp}_percent_accuracy"] = accuracy
    if return_ranks:
        output["rank_median"] = ranks
        output["relative_rank_median"] = ranks / retrieval_size
        output["mean_reciprocal_rank"] = 1 / (ranks + 1)
    if return_negatives_dist:
        output["negatives_dist"] = dist_to_negatives.cpu().numpy()
    return output


def aggregate_metrics_df(df):
    output = {}
    subject_id = df.dataset + "_" + df.subject + "/"
    for key in df.columns:
        if key not in [
            "dataset",
            "subject",
            "run",
            "tr",
            "top",
            "chunk",
            "candidate",
        ]:
            if "median" in key:
                output[key] = df[key].median()
                subjects_df = df[key].groupby(subject_id).median()
            elif key == "n_trs":
                output[key] = df[key].sum()
                subjects_df = df[key].groupby(subject_id).sum()
            else:
                output[key] = df[key].mean()
                subjects_df = df[key].groupby(subject_id).mean()
            for k, v in subjects_df.items():
                output[k + key] = v

    return output


def evaluate(
    df,
    decoder,
    negatives,
    top_k_accuracies=[],
    top_k_percent_accuracies=[],
    nlp_distances=None,
    n_candidates=10,
    return_tables=False,
):
    decoder.eval()
    metrics = defaultdict(list)
    ranks = defaultdict(list)

    get_candidates = (return_tables) or (nlp_distances is not None)
    if get_candidates:
        candidates = defaultdict(list)
        all_chunks = df.drop_duplicates(["dataset", "run"])
        all_chunks = df[["chunks", "chunks_index"]]
        all_chunks = all_chunks.explode(["chunks", "chunks_index"])
        all_chunks = all_chunks.sort_values("chunks_index").chunks.values
        assert len(all_chunks) == len(negatives)
        if n_candidates is None:
            n_candidates = len(negatives)

        if nlp_distances is not None:
            corresp = nlp_distances["corresp"]
            assert len(all_chunks) == len(corresp)

    negatives = negatives.to(device)
    if isinstance(decoder.decoder, MeanDecoder):
        decoder.decoder.set_mean_from_Ys(negatives)

    with torch.no_grad():
        for _, row in df.iterrows():
            run_id = row[["dataset", "subject", "run"]]
            # Compute losses and other run-level metrics
            X = row.X.to(device)
            Y = row.Y.to(device)
            X_proj = decoder.projector[row.dataset][row.subject](X)
            Y_preds = decoder(X_proj)
            mixco = decoder.mixco_loss(X_proj, Y)
            symm_nce = decoder.symm_nce_loss(X_proj, Y, Y_preds)
            mse = decoder.mse_loss(X_proj, Y, Y_preds)
            mean_r = corr(Y, Y_preds).mean().item()
            mean_r2 = r2_score(
                row.Y, Y_preds.cpu(), multioutput="raw_values"
            ).mean()
            # Store metrics
            for k, v in run_id.items():
                metrics[k].append(v)
            metrics["mixco"].append(mixco.item())
            metrics["symm_nce"].append(symm_nce.item())
            metrics["mse"].append(mse.item())
            metrics["mean_r"].append(mean_r)
            metrics["mean_r2"].append(mean_r2)
            n_trs = row.n_trs
            metrics["n_trs"].append(n_trs)

            # Compute retrieval metrics
            r_metrics = retrieval_metrics(
                Y,
                Y_preds,
                negatives,
                top_k_accuracies=top_k_accuracies,
                top_k_percent_accuracies=top_k_percent_accuracies,
                return_ranks=True,
                return_negatives_dist=get_candidates,
            )
            if get_candidates:
                # Retrieve candidates (closest to preds)
                negatives_dist = r_metrics.pop("negatives_dist")
                candidates_idx = negatives_dist.argsort(axis=1)[
                    :, :n_candidates
                ]
                candidates_distances = negatives_dist[
                    np.arange(row.n_trs).reshape(-1, 1), candidates_idx
                ]
                candidates_idx = candidates_idx.reshape(-1)
                chunk_idx = np.repeat(row.chunks_index, n_candidates)
                # Store candidates
                for k, v in run_id.items():
                    candidates[k].extend([v] * n_trs * n_candidates)
                candidates["tr"].extend(np.repeat(range(n_trs), n_candidates))
                candidates["top"].extend(
                    np.tile(np.arange(n_candidates) + 1, n_trs)
                )
                candidates["dist"].extend(candidates_distances.reshape(-1))
                candidates["chunk"].extend(all_chunks[chunk_idx])
                candidates["candidate"].extend(all_chunks[candidates_idx])

                if nlp_distances is not None:
                    # If nlp_distances are provided, store them
                    corresp_idx = corresp[chunk_idx, candidates_idx]
                    for k, v in nlp_distances.items():
                        if k != "corresp":
                            candidates[k].extend(v[corresp_idx])

            # Store retrieval metrics
            for k, v in run_id.items():
                ranks[k].extend([v] * n_trs)
            ranks["tr"].extend(range(n_trs))
            for k, v in r_metrics.items():
                ranks[k].extend(v)

    output = {}
    metrics = pd.DataFrame(metrics)
    output.update(aggregate_metrics_df(metrics))
    metrics["retrieval_size"] = len(negatives)
    output["retrieval_size"] = len(negatives)
    ranks = pd.DataFrame(ranks)
    output.update(aggregate_metrics_df(ranks))
    metrics = metrics.merge(ranks)
    if get_candidates:
        candidates = pd.DataFrame(candidates)
        output.update(aggregate_metrics_df(candidates))
        metrics = metrics.merge(candidates)

    if return_tables:
        output["metrics"] = metrics

    return output
