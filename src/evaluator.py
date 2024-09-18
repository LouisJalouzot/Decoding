import os
import string
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import nltk
import numpy as np
import pandas as pd
import requests
import torch
import torchmetrics as tm
import werpy
from sklearn.metrics import r2_score

import wandb
from src.utils import console, device, progress


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
        output["relative_ranks"] = (ranks / retrieval_size).cpu().numpy()
    if return_negatives_dist:
        output["negatives_dist"] = dist_to_negatives.cpu().numpy()
    return output


def load_glove_embeddings():
    glove_file = Path("data/glove.6B.50d.txt")
    if not glove_file.exists():
        url = (
            "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        )
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 131072
        zip_path = Path("data/glove.6B.zip")

        with open(zip_path, "wb") as file, progress:
            task = progress.add_task(
                "Downloading GloVe embeddings", total=total_size // chunk_size
            )
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                progress.update(task, advance=1)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("glove.6B.50d.txt", "data/")
        zip_path.unlink()
        console.log(f"Saved GloVe embeddings to {glove_file}")

    embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def nltk_pos_tag(chunks):
    tokenized_chunks = [nltk.tokenize.word_tokenize(chunk) for chunk in chunks]
    pos_tagged_chunks = [
        " ".join(
            e[1] for e in nltk.tag.pos_tag(tokenized_chunk, tagset="universal")
        )
        for tokenized_chunk in tokenized_chunks
    ]
    return np.array(pos_tagged_chunks)


class Evaluator:
    def __init__(self, extra_metrics=False):
        self.extra_metrics = extra_metrics

        if self.extra_metrics:
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

    def get_glove_embedding(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.lower().split()
        glove_bow = np.zeros(50)
        for word in words:
            glove_bow += self.glove_embeddings.get(word, np.zeros(50))
        return glove_bow / max(1, len(words))

    def evaluate(
        self,
        df,
        decoder,
        negatives,
        negative_chunks=None,
        top_k_accuracies=[],
        extra_metrics=None,
        n_candidates=10,
    ):
        if extra_metrics is None:
            extra_metrics = self.extra_metrics
        if extra_metrics:
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
        negatives = negatives.to(device)

        # extra_metrics_data = []

        with torch.no_grad():
            for _, row in df.iterrows():
                run_id = row[["dataset", "subject", "run"]]
                for k, v in run_id.items():
                    metrics[k].append(v)
                    relative_ranks[k].extend([v] * row.n_trs)
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
                    return_negatives_dist=extra_metrics,
                )
                relative_ranks["relative_ranks"].extend(
                    r_metrics["relative_ranks"]
                )

                if extra_metrics:
                    negatives_dist = r_metrics["negatives_dist"]
                    top_negatives_idx = negatives_dist.argsort(axis=1)[
                        :, :n_candidates
                    ]
                    top_negatives_chunks = negative_chunks[top_negatives_idx]
                    r_metrics["extra_metrics_size"] = top_negatives_chunks.size
                    chunks = np.array(row.chunks_with_context)

                    # all_chunks.extend(chunks)
                    # all_top_negatives_chunks.extend(top_negatives_chunks)

                for key, value in r_metrics.items():
                    if key not in ["negatives_dist", "relative_ranks"]:
                        metrics[key].append(value)

        if extra_metrics:
            all_chunks = np.array(all_chunks)
            all_top_negatives_chunks = np.array(all_top_negatives_chunks)

            # Compute GloVe bag of words cosine similarity
            chunks_glove = np.array(
                [self.get_glove_embedding(chunk) for chunk in all_chunks]
            )[:, None]
            top_negatives_glove = np.array(
                [
                    self.get_glove_embedding(chunk)
                    for chunk in all_top_negatives_chunks.flatten()
                ]
            ).reshape(all_top_negatives_chunks.shape + (-1,))
            glove_cosine = (chunks_glove * top_negatives_glove).sum(axis=-1) / (
                np.linalg.norm(chunks_glove, axis=-1)
                * np.linalg.norm(top_negatives_glove, axis=-1)
            ).clip(1e-9)
            metrics["glove_bow_cosine"] = torch.from_numpy(
                glove_cosine.reshape(-1)
            )

            # Compute POS WER
            chunks_pos = nltk_pos_tag(all_chunks)
            top_negatives_chunks_pos = nltk_pos_tag(
                all_top_negatives_chunks.flatten()
            ).reshape(-1, 10)
            pos_wer = werpy.wers(chunks_pos, top_negatives_chunks_pos.T)
            metrics["pos_wer"] = torch.Tensor(pos_wer).reshape(-1)

            # Compute WER
            wers = werpy.wers(
                all_chunks, all_top_negatives_chunks.reshape(-1, 10).T
            )
            metrics["wer"] = torch.Tensor(wers).reshape(-1)

            # Compute F1 BERTScore
            metrics["bert_f1"] = self.bertscore(
                all_top_negatives_chunks.flatten(), all_chunks.repeat(10)
            )["f1"]

        df_metrics = pd.DataFrame(metrics)
        output["metrics"] = wandb.Table(dataframe=df_metrics)
        df_relative_ranks = pd.DataFrame(relative_ranks)
        output["relative_ranks"] = wandb.Table(dataframe=df_relative_ranks)
        for df in df_metrics, df_relative_ranks:
            for key in df.columns:
                if key not in ["dataset", "subject", "run", "tr"]:
                    if key == "relative_ranks":
                        output[key + "_median"] = df[key].median()
                    else:
                        output[key] = df[key].mean()
        output["retrieval_size"] = len(negatives)

        return output
