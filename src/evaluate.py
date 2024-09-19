from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
import torchmetrics as tm
from sklearn.metrics import r2_score

import wandb
from src.utils import BatchIncrementalMean, console, corr, device


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
        console.log(
            f"[bold red]Batch size mismatch:[/] {Y_true.shape[0]} {Y_pred.shape[0]}"
        )
    if negatives is None:
        negatives = Y_true
    size = len(Y_true)
    retrieval_size = len(negatives)
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
    for top_k in top_k_accuracies:
        accuracy = ranks < top_k
        accuracy = accuracy.cpu().numpy().mean()
        output[f"top_{top_k}_accuracy"] = accuracy
    if return_ranks:
        output["relative_rank"] = (ranks / retrieval_size).cpu().numpy()
    if return_negatives_dist:
        output["negatives_dist"] = dist_to_negatives.cpu().numpy()
    return output


def evaluate(
    df,
    decoder,
    negatives,
    watching_subjects=None,
    top_k_accuracies=[],
    nlp_distances=None,
    n_candidates=10,
    return_tables=False,
):
    decoder.eval()

    if return_tables:
        watching_subjects = None
        all_chunks = df.chunks_with_context.explode().values
        corresp = nlp_distances.pop("corresp")
        metrics = defaultdict(list)
        nlp = defaultdict(list)
    else:
        metrics = defaultdict(BatchIncrementalMean)
        nlp = defaultdict(BatchIncrementalMean)
    relative_ranks = defaultdict(list)
    negatives = negatives.to(device)

    with torch.no_grad():
        for _, row in df.iterrows():
            prefix = [""]
            if (
                watching_subjects is not None
                and row.dataset in watching_subjects
                and row.subject in watching_subjects[row.dataset]
            ):
                prefix.append(f"{row.dataset}_{row.subject}/")

            if return_tables:
                run_id = row[["dataset", "subject", "run"]]
                for k, v in run_id.items():
                    metrics[k].append(v)
                    relative_ranks[k].extend([v] * row.n_trs)
                    if nlp_distances is not None:
                        nlp[k].extend([v] * row.n_trs * n_candidates)
                relative_ranks["tr"].extend(range(row.n_trs))

            X = row.X.to(device)
            Y = row.Y.to(device)
            with torch.autocast(device_type=device.type):
                X_proj = decoder.projector[row.dataset][row.subject](X)
                # Evaluate losses
                _, mixco = decoder.mixco_loss(X_proj, Y)[1]
                Y_preds, symm_nce = decoder.symm_nce_loss(X_proj, Y)
                _, mse = decoder.mse_loss(X_proj, Y, Y_preds)
                mean_r = corr(Y, Y_preds).mean().item()
                mean_r2 = r2_score(
                    row.Y, Y_preds.cpu(), multioutput="raw_values"
                ).mean()
                for p in prefix:
                    metrics[p + "mixco"].extend([mixco.item()])
                    metrics[p + "symm_nce"].extend([symm_nce.item()])
                    metrics[p + "mse"].extend([mse.item()])
                    metrics[p + "mean_r"].extend([mean_r])
                    metrics[p + "mean_r2"].extend([mean_r2])

            # Evaluate retrieval metrics
            r_metrics = retrieval_metrics(
                Y,
                Y_preds,
                negatives,
                top_k_accuracies=top_k_accuracies,
                return_ranks=True,
                return_negatives_dist=nlp_distances is not None,
            )
            for key, value in r_metrics.items():
                if key not in ["negatives_dist", "relative_rank"]:
                    metrics[key].extend([value])
            relative_ranks["relative_rank"].extend(r_metrics["relative_rank"])

            if nlp_distances is not None:
                negatives_dist = r_metrics["negatives_dist"]
                candidates_idx = negatives_dist.argsort(axis=1)[
                    :, :n_candidates
                ].reshape(-1)
                chunk_idx = np.tile(row.chunk_index.values, n_candidates)
                corresp_idx = corresp[chunk_idx, candidates_idx]
                for k, v in nlp_distances.items():
                    for p in prefix:
                        nlp[p + k].extcandidates_idxend(v[corresp_idx])
                if return_tables:
                    nlp_distances["chunk"].extend(
                        row.chunks_with_context * n_candidates
                    )
                    nlp_distances["top"].extend(
                        np.tile(np.arange(n_candidates), row.n_trs) + 1
                    )
                    nlp_distances["candidate"].extend(
                        all_chunks[candidates_idx]
                    )

    output = {"retrieval_size": len(negatives)}
    relative_ranks = pd.DataFrame(relative_ranks)
    output["relative_rank_median"] = relative_ranks.relative_rank.median()
    if return_tables:
        metrics = pd.DataFrame(metrics)
        output["metrics"] = metrics
        output["relative_ranks"] = relative_ranks
        dfs = [metrics]
        if nlp_distances is not None:
            nlp = pd.DataFrame(nlp)
            output["nlp_distances"] = nlp
            dfs.append(nlp)
        for df in dfs:
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
                    if key == "size":
                        output[key] = df[key].sum()
                    else:
                        output[key] = df[key].mean()
    else:
        output.update(metrics)
        if nlp_distances is not None:
            output.update(nlp)

    return output
