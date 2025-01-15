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
    output = {"retrieval_size": retrieval_size, "size": len(Y_true)}
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
    ranks = (dist_to_ground_truth > dist_to_negatives).sum(1).double()
    for top_k in top_k_accuracies:
        accuracy = ranks < top_k
        accuracy = accuracy.cpu().numpy().mean()
        output[f"top_{top_k}_accuracy"] = accuracy
    if return_ranks:
        output["relative_rank"] = (ranks / retrieval_size).cpu().numpy()
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
            if key == "relative_rank":
                suffix = "_median"
                output[key + suffix] = df[key].median()
                subjects_df = df[key].groupby(subject_id).median()
            elif key == "size":
                suffix = ""
                output[key + suffix] = df[key].sum()
                subjects_df = df[key].groupby(subject_id).sum()
            else:
                suffix = ""
                output[key + suffix] = df[key].mean()
                subjects_df = df[key].groupby(subject_id).mean()
            for k, v in subjects_df.items():
                output[k + key + suffix] = v

    return output


def evaluate(
    df,
    decoder,
    negatives,
    top_k_accuracies=[],
    nlp_distances=None,
    n_candidates=10,
    return_tables=False,
):
    decoder.eval()

    metrics = defaultdict(list)
    nlp = defaultdict(list)
    relative_ranks = defaultdict(list)

    if nlp_distances is not None:
        all_chunks = df.drop_duplicates(["dataset", "run"])
        all_chunks = all_chunks.chunks_with_context.explode().values
        corresp = nlp_distances["corresp"]
        assert len(all_chunks) == len(corresp)
        assert len(all_chunks) == len(negatives)

    negatives = negatives.to(device)
    if isinstance(decoder.decoder, MeanDecoder):
        decoder.decoder.set_mean_from_Ys(negatives)

    with torch.no_grad():
        for _, row in df.iterrows():
            run_id = row[["dataset", "subject", "run"]]
            for k, v in run_id.items():
                relative_ranks[k].extend([v] * row.n_trs)
                metrics[k].append(v)
                if nlp_distances is not None:
                    nlp[k].extend([v] * row.n_trs * n_candidates)
            if nlp_distances is not None:
                nlp["tr"].extend(np.repeat(range(row.n_trs), n_candidates))
            relative_ranks["tr"].extend(range(row.n_trs))

            X = row.X.to(device)
            Y = row.Y.to(device)
            X_proj = decoder.projector[row.dataset][row.subject](X)
            Y_preds = decoder(X_proj)
            # Evaluate losses
            mixco = decoder.mixco_loss(X_proj, Y)
            symm_nce = decoder.symm_nce_loss(X_proj, Y, Y_preds)
            mse = decoder.mse_loss(X_proj, Y, Y_preds)
            mean_r = corr(Y, Y_preds).mean().item()
            mean_r2 = r2_score(
                row.Y, Y_preds.cpu(), multioutput="raw_values"
            ).mean()
            metrics["mixco"].extend([mixco.item()])
            metrics["symm_nce"].extend([symm_nce.item()])
            metrics["mse"].extend([mse.item()])
            metrics["mean_r"].extend([mean_r])
            metrics["mean_r2"].extend([mean_r2])

            # Evaluate retrieval metrics
            r_metrics = retrieval_metrics(
                Y,
                Y_preds,
                negatives,
                top_k_accuracies=top_k_accuracies,
                return_ranks=True,
                return_negatives_dist=(nlp_distances is not None),
            )
            for key, value in r_metrics.items():
                if key not in ["negatives_dist", "relative_rank"]:
                    metrics[key].extend([value])
            relative_ranks["relative_rank"].extend(r_metrics["relative_rank"])

            if nlp_distances is not None:
                negatives_dist = r_metrics["negatives_dist"]
                candidates_idx = negatives_dist.argsort(axis=1)[
                    :, :n_candidates
                ]
                candidates_distances = negatives_dist[
                    np.arange(row.n_trs).reshape(-1, 1), candidates_idx
                ]
                nlp["dist"].extend(candidates_distances.reshape(-1))
                chunk_idx = np.repeat(row.chunks_index, n_candidates)
                candidates_idx = candidates_idx.reshape(-1)
                corresp_idx = corresp[chunk_idx, candidates_idx]
                for k, v in nlp_distances.items():
                    if k != "corresp":
                        nlp[k].extend(v[corresp_idx])
                if return_tables:
                    nlp["chunk"].extend(
                        np.repeat(row.chunks_with_context, n_candidates)
                    )
                    nlp["top"].extend(
                        np.tile(np.arange(n_candidates) + 1, row.n_trs)
                    )
                    nlp["candidate"].extend(all_chunks[candidates_idx])

    output = {}
    metrics = pd.DataFrame(metrics)
    relative_ranks = pd.DataFrame(relative_ranks)
    dfs = [metrics, relative_ranks]
    if nlp_distances is not None:
        nlp = pd.DataFrame(nlp)
        dfs.append(nlp)
        if return_tables:
            output["nlp_distances"] = nlp
    for df in dfs:
        output.update(aggregate_metrics_df(df))
    if return_tables:
        output["metrics"] = metrics
        output["relative_ranks"] = relative_ranks

    return output
