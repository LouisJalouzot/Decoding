import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from sklearn.preprocessing import StandardScaler

import wandb
from src.nlp_metrics import compute_nlp_distances, nlp_cols, nlp_dist_cols
from src.prepare_data import (
    compute_chunk_index,
    find_best_encoding_voxels,
    generate_splits,
    read,
)
from src.prepare_latents import prepare_latents
from src.train import train
from src.utils import console, device, progress


def main(
    datasets: Union[str, List[str]] = "lebel2023/all_subjects",
    subjects: Dict[str, List[str]] = None,
    runs: Dict[str, Dict[str, List[str]]] = None,
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 3,
    smooth: int = 0,
    stack: int = 0,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    top_encoding_voxels: int = None,
    token_aggregation: str = "mean",
    return_data: bool = False,
    return_tables: bool = False,
    log_nlp_distances: bool = False,
    n_folds: int = None,
    fold: int = None,
    **decoder_params,
):
    console.log("Running on device", device)
    # Find subject and run names in data if not provided
    if runs is None:
        if subjects is None:
            subjects = {
                dataset: sorted(os.listdir(f"datasets/{dataset}"))
                for dataset in datasets
            }
        runs = {
            dataset: {
                subject: sorted(
                    [
                        Path(f).stem
                        for f in os.listdir(f"datasets/{dataset}/{subject}")
                    ]
                )
                for subject in subjects[dataset]
            }
            for dataset in subjects
        }
    else:
        subjects = {dataset: list(runs[dataset].keys()) for dataset in runs}

    if wandb.run is not None:
        wandb.config["subjects"] = subjects
        wandb.config["runs"] = runs

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # Fetch brain scans
    n_runs = sum([len(r) for s in runs.values() for r in s.values()])
    if n_runs == 0:
        raise ValueError(
            f"No runs found in datasets {datasets}, have you run preprocess.py?"
        )
    with joblib_progress(
        f"Loading brain scans for datasets {datasets}",
        total=n_runs,
        console=console,
    ):
        df = Parallel(n_jobs=-1)(
            delayed(read)(dataset, subject, run, lag, smooth, stack)
            for dataset in runs
            for subject in runs[dataset]
            for run in runs[dataset][subject]
        )
    df = pd.DataFrame(
        sorted(df),
        columns=["dataset", "subject", "run", "n_trs", "n_voxels", "X"],
    )

    # Fetch latents
    runs = df[["dataset", "run", "n_trs"]].drop_duplicates()
    n_runs = len(runs)
    scaler = StandardScaler()
    with progress:
        task = progress.add_task("", total=n_runs)
        progress.update(task, description=f"Fetching latents for {n_runs} runs")
        latents = []
        for _, (dataset, run, n_trs) in runs.iterrows():
            data = prepare_latents(
                dataset=dataset.split("_")[
                    0
                ],  # lebel2023_balanced -> lebel2023
                run=run,
                model=model,
                tr=tr,
                context_length=context_length,
                token_aggregation=token_aggregation,
            )
            scaler.partial_fit(data.Y)
            if stack > 0:
                data = data.apply(lambda x: x[stack:])
            if lag > 0:
                data = data.apply(lambda x: x[:-lag])
            elif lag < 0:
                data = data.apply(lambda x: x[-lag:])
            if data.Y.shape[0] > n_trs + 1:
                console.log(
                    f"[red]{data.Y.shape[0] - n_trs} > 1 latents trimmed for run {run} in dataset {dataset}"
                )
            # If more latents than brain scans, drop last seconds of run
            data = data.apply(lambda x: x[:n_trs])
            data["dataset"] = dataset
            data["run"] = run
            latents.append(data)
            progress.update(task, advance=1)
        progress.update(task, completed=True)
    latents = pd.concat(latents, axis=1).T
    latents["Y"] = latents.Y.apply(
        lambda x: torch.from_numpy(scaler.transform(x).astype(np.float32))
    )
    df = df.merge(latents, on=["dataset", "run"])

    # Compute (CV) splits
    df_splits = generate_splits(
        df=df,
        n_folds=n_folds,
        fold=fold,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if wandb.run is not None:
        wandb.log({"splits": wandb.Table(data=df_splits)})
    df["orig_X"] = (
        df.X
    )  # Save original X for multiple encoding voxels selections
    df = df.merge(df_splits)

    outputs = {}
    for fold, df_fold in df.groupby("fold"):
        if top_encoding_voxels is not None:
            df_fold = find_best_encoding_voxels(df_fold, top_encoding_voxels)

        df_train = df_fold[df_fold.split == "train"]
        df_valid = df_fold[df_fold.split == "valid"]
        df_test = df_fold[df_fold.split == "test"]
        nlp_distances = {}
        if log_nlp_distances:
            chunks_index = compute_chunk_index(df_train)
            df_train = df_train.merge(chunks_index)
            chunks_index = compute_chunk_index(df_valid)
            df_valid = df_valid.merge(chunks_index)
            chunks_index = compute_chunk_index(df_test)
            df_test = df_test.merge(chunks_index)
            for split, df_split in [
                ("test", df_test),
                ("valid", df_valid),
                ("train", df_train),
            ]:
                data = df_split.drop_duplicates(["dataset", "run"])[nlp_cols]
                data = data.to_dict("series")
                for k, v in data.items():
                    data[k] = v.explode().values
                nlp_distances[split] = compute_nlp_distances(**data)

        console.log(
            f"\n[bold]Fold {fold}/{n_folds}[/]\n"
            f"Train split: {df_train.run.nunique()} runs with {len(df_train)} occurrences and {df_train.n_trs.sum()} scans\n"
            f"Valid split: {df_valid.run.nunique()} runs with {len(df_valid)} occurrences and {df_valid.n_trs.sum()} scans\n"
            f"Test split: {df_test.run.nunique()} runs with {len(df_test)} occurrences and {df_test.n_trs.sum()} scans"
        )

        if return_data:
            outputs[fold] = df_train, df_valid, df_test, nlp_distances
        else:
            outputs[fold] = train(
                df_train,
                df_valid,
                df_test,
                decoder=decoder,
                return_tables=return_tables,
                nlp_distances=nlp_distances,
                **decoder_params,
            )

    return outputs
