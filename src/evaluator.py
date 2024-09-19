import string
from collections import defaultdict
from typing import List

import nltk
import numpy as np
import pandas as pd
import torch
import torchmetrics as tm
import wandb
import werpy
from sklearn.metrics import r2_score

from src.utils import console, corr, device, load_glove_embeddings, nltk_pos_tag


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


class Evaluator:
    def __init__(self, log_extra_metrics=False):
        self.log_extra_metrics = log_extra_metrics

        if self.log_extra_metrics:
            import warnings

            warnings.filterwarnings("ignore", module="transformers")
            nltk.download("universal_tagset", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            self.bertscore = tm.text.bert.BERTScore(
                "distilbert-base-uncased-distilled-squad",
                rescale_with_baseline=True,
                device=device,
            )
            self.glove_embeddings = load_glove_embeddings()

    def evaluate(
        self,
        df,
        decoder,
        negatives,
        negative_chunks=None,
        top_k_accuracies=[],
        log_extra_metrics=None,
        n_candidates=10,
        log_tables=False,
    ):
        if log_extra_metrics is None:
            log_extra_metrics = self.log_extra_metrics
        if log_extra_metrics:
            assert (
                negative_chunks is not None
            ), "Negative chunks required for extra metrics"
            assert len(negative_chunks) == len(negatives)
            assert n_candidates is not None
            negative_chunks = np.array(negative_chunks)

        decoder.eval()

        output = {}
        metrics = defaultdict(list)
        relative_ranks = defaultdict(list)
        extra_metrics = defaultdict(list)
        negatives = negatives.to(device)

        with torch.no_grad():
            for _, row in df.iterrows():
                run_id = row[["dataset", "subject", "run"]]
                for k, v in run_id.items():
                    metrics[k].append(v)
                    relative_ranks[k].extend([v] * row.n_trs)
                    if log_extra_metrics:
                        extra_metrics[k].extend([v] * row.n_trs * n_candidates)
                relative_ranks["tr"].extend(range(row.n_trs))
                X = row.X.to(device)
                Y = row.Y.to(device)
                with torch.autocast(device_type=device.type):
                    X_proj = decoder.projector[row.dataset][row.subject](X)
                    # Evaluate losses
                    _, mixco = decoder.mixco_loss(X_proj, Y)
                    metrics["mixco"].append(mixco.item())
                    Y_preds, symm_nce = decoder.symm_nce_loss(X_proj, Y)
                    metrics["symm_nce"].append(symm_nce.item())
                    metrics["mse"].append(
                        decoder.mse_loss(X_proj, Y, Y_preds)[1].item()
                    )
                    metrics["mean_r"].append(corr(Y, Y_preds).mean().item())
                    metrics["mean_r2"].append(
                        r2_score(
                            row.Y, Y_preds.cpu(), multioutput="raw_values"
                        ).mean()
                    )

                # Evaluate retrieval metrics
                r_metrics = retrieval_metrics(
                    Y,
                    Y_preds,
                    negatives,
                    top_k_accuracies=top_k_accuracies,
                    return_ranks=True,
                    return_negatives_dist=log_extra_metrics,
                )
                relative_ranks["relative_rank"].extend(
                    r_metrics["relative_rank"]
                )

                if log_extra_metrics:
                    negatives_dist = r_metrics["negatives_dist"]
                    top_negatives_idx = negatives_dist.argsort(axis=1)[
                        :, :n_candidates
                    ]
                    top_candidates = negative_chunks[top_negatives_idx]
                    extra_metrics["chunk"].extend(
                        row.chunks_with_context * n_candidates
                    )
                    extra_metrics["top"].extend(
                        np.tile(np.arange(n_candidates), row.n_trs) + 1
                    )
                    extra_metrics["candidate"].extend(
                        top_candidates.reshape(-1)
                    )

                for key, value in r_metrics.items():
                    if key not in ["negatives_dist", "relative_rank"]:
                        metrics[key].append(value)

        if log_extra_metrics:
            chunks = extra_metrics["chunk"]
            candidates = extra_metrics["candidate"]

            # Compute GloVe bag of words cosine similarity
            chunks_clean, chunks_glove = zip(
                *[self.get_bow_glove_embedding(chunk) for chunk in chunks]
            )
            candidates_clean, candidates_glove = zip(
                *[self.get_bow_glove_embedding(chunk) for chunk in candidates]
            )
            glove_cosine = torch.cosine_similarity(
                torch.from_numpy(np.array(chunks_glove)),
                torch.from_numpy(np.array(candidates_glove)),
                dim=-1,
            )
            extra_metrics["glove_bow_cosine"] = glove_cosine.tolist()

            # Compute POS
            chunks_pos = nltk_pos_tag(chunks)
            candidates_pos = nltk_pos_tag(candidates)

            # Compute POS GloVe bag of words cosine similarity
            restrict_pos = lambda chunk, pos: " ".join(
                w for w, p in zip(chunk, pos) if p in ["NOUN", "VERB", "ADJ"]
            )
            chunks_reduced_pos = [
                restrict_pos(c, p) for c, p in zip(chunks_clean, chunks_pos)
            ]
            candidates_reduced_pos = [
                restrict_pos(c, p)
                for c, p in zip(candidates_clean, candidates_pos)
            ]
            _, chunks_reduced_pos_glove = zip(
                *[
                    self.get_bow_glove_embedding(chunk)
                    for chunk in chunks_reduced_pos
                ]
            )
            _, candidates_reduced_pos_glove = zip(
                *[
                    self.get_bow_glove_embedding(chunk)
                    for chunk in candidates_reduced_pos
                ]
            )
            pos_glove_cosine = torch.cosine_similarity(
                torch.from_numpy(np.array(chunks_reduced_pos_glove)),
                torch.from_numpy(np.array(candidates_reduced_pos_glove)),
                dim=-1,
            )
            extra_metrics["pos_glove_bow_cosine"] = pos_glove_cosine.tolist()

            # Compute POS WER
            extra_metrics["pos_wer"] = werpy.wers(
                [" ".join(chunk) for chunk in chunks_pos],
                [" ".join(candidate) for candidate in candidates_pos],
            )

            # Compute WER
            extra_metrics["wer"] = werpy.wers(chunks, candidates)

            # Compute F1 BERTScore
            bert_f1 = self.bertscore(candidates, chunks)["f1"]
            extra_metrics["bert_f1"] = bert_f1.tolist()

        df_metrics = pd.DataFrame(metrics)
        df_relative_rank = pd.DataFrame(relative_ranks)
        dfs = [df_metrics, df_relative_rank]
        if log_extra_metrics:
            df_extra_metrics = pd.DataFrame(extra_metrics)
            dfs.append(df_extra_metrics)
        if log_tables:
            output["metrics"] = wandb.Table(dataframe=df_metrics)
            output["relative_rank"] = wandb.Table(dataframe=df_relative_rank)
            if log_extra_metrics:
                output["extra_metrics"] = wandb.Table(
                    dataframe=df_extra_metrics
                )
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
                    if key == "relative_rank":
                        output[key + "_median"] = df[key].median()
                    elif key == "size":
                        output[key] = df[key].sum()
                    else:
                        output[key] = df[key].mean()
        output = {"retrieval_size": len(negatives)}

        return output
