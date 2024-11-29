import os
from collections import defaultdict
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import wandb
from src.nlp_metrics import compute_nlp_distances, nlp_cols
from src.prepare_data import (
    compute_chunk_index,
    find_best_encoding_voxels,
    read_brain_volume,
    split_dataframe,
)
from src.prepare_latents import prepare_latents
from src.train import train
from src.utils import console, progress, set_seeds


def align_latents(data, lag, smooth, stack):
    if stack > 0:
        data = data.apply(lambda x: x[stack:])
    if lag > 0:
        data = data.apply(lambda x: x[:-lag])
    elif lag < 0:
        data = data.apply(lambda x: x[-lag:])

    return data


def decoding(
    meta: dict,
    seed: int,
    return_tables: bool,
    log_nlp_distances: bool,
    datasets: list[str],
    subjects: dict[str, list[str]] | None,
    runs: dict[str, dict[str, list[str]]] | None,
    tr: int,
    splitting: dict,
    fine_tuning_cfg: dict,
    alignment_cfg: dict,
    top_encoding_voxels: int | None,
    latents_cfg: dict,
    decoder_cfg: dict,
    train_cfg: dict,
):
    set_seeds(seed)

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

    # Fetch brain scans
    n_runs = sum([len(r) for s in runs.values() for r in s.values()])
    if n_runs == 0:
        raise ValueError(
            f"No runs found in datasets {datasets}, have you run preprocess.py?"
        )
    with progress:
        task = progress.add_task(
            f"Loading brain scans for datasets {datasets}",
            total=n_runs,
        )
        df = []
        for dataset in runs:
            for subject in runs[dataset]:
                for run in runs[dataset][subject]:
                    df.append(
                        read_brain_volume(
                            dataset, subject, run, **alignment_cfg
                        )
                    )
                    progress.update(task, advance=1)
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
                # lebel2023_balanced -> lebel2023
                dataset=dataset.split("_")[0],
                run=run,
                tr=tr,
                **latents_cfg,
            )
            scaler.partial_fit(data.Y)
            data = align_latents(data, **alignment_cfg)
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
    df = split_dataframe(df, seed, **splitting, **fine_tuning_cfg)

    n_folds = splitting["n_folds"]
    outputs = defaultdict(list)
    for fold, df_fold in df.groupby("fold"):
        if top_encoding_voxels is not None:
            df_fold = find_best_encoding_voxels(df_fold, top_encoding_voxels)

        in_dims = df_fold[["dataset", "subject", "n_voxels"]].drop_duplicates()
        in_dims = in_dims.set_index(["dataset", "subject"]).n_voxels
        in_dims = {
            level: in_dims.xs(level).to_dict()
            for level in in_dims.index.levels[0]
        }

        df_train = df_fold[df_fold.split == "train"]
        df_valid = df_fold[df_fold.split == "valid"]
        df_test = df_fold[df_fold.split == "test"]
        df_ft_train = df_fold[df_fold.split == "ft_train"]
        df_ft_valid = df_fold[df_fold.split == "ft_valid"]
        nlp_distances = {}
        if log_nlp_distances:
            chunks_index = compute_chunk_index(df_train)
            df_train = df_train.merge(chunks_index)
            chunks_index = compute_chunk_index(df_valid)
            df_valid = df_valid.merge(chunks_index)
            chunks_index = compute_chunk_index(df_test)
            df_test = df_test.merge(chunks_index)
            chunks_index = compute_chunk_index(df_ft_train)
            df_ft_train = df_ft_train.merge(chunks_index)
            chunks_index = compute_chunk_index(df_ft_valid)
            df_ft_valid = df_ft_valid.merge(chunks_index)
            for split, df_split in [
                ("test", df_test),
                ("valid", df_valid),
                ("train", df_train),
                ("ft_train", df_ft_train),
                ("ft_valid", df_ft_valid),
            ]:
                data = df_split.drop_duplicates(["dataset", "run"])[nlp_cols]
                data = data.to_dict("series")
                for k, v in data.items():
                    data[k] = v.explode().values
                nlp_distances[split] = compute_nlp_distances(**data)

        console.log(
            f"\n[bold]Fold {fold}/{n_folds}[/]\n"
            if n_folds is not None
            else ""
            + f"Train split: {df_train.run.nunique()} runs with {len(df_train)} occurrences and {df_train.n_trs.sum()} scans\n"
            + f"Valid split: {df_valid.run.nunique()} runs with {len(df_valid)} occurrences and {df_valid.n_trs.sum()} scans\n"
            + f"Test split: {df_test.run.nunique()} runs with {len(df_test)} occurrences and {df_test.n_trs.sum()} scans"
        )
        if not df_ft_train.empty:
            console.log(
                f"Fine-tune train split: {df_ft_train.run.nunique()} runs with {len(df_ft_train)} occurrences and {df_ft_train.n_trs.sum()} scans\n"
                f"Fine-tune valid split: {df_ft_valid.run.nunique()} runs with {len(df_ft_valid)} occurrences and {df_ft_valid.n_trs.sum()} scans"
            )

        if meta["return_data"]:
            outputs[fold] = df_train, df_valid, df_test, nlp_distances
        else:
            prefix = f"fold_{fold}/" if n_folds is not None else ""
            metrics, _decoder = train(
                df_train,
                df_valid,
                df_test,
                df_ft_train,
                df_ft_valid,
                in_dims,
                decoder_cfg=decoder_cfg,
                return_tables=return_tables,
                nlp_distances=nlp_distances,
                metrics_prefix=prefix,
                ft_params=fine_tuning_cfg["train_cfg"],
                metrics_prefix=prefix,
                **train_cfg,
            )
            outputs.update({prefix + k: v for k, v in metrics.items()})
            # outputs[prefix + "decoder"] = _decoder # Not saving decoder for now
            if n_folds is not None:
                for k, v in metrics.items():
                    if isinstance(v, Number):
                        outputs[k].append(v)

    # If CV has been used, average metrics over folds
    if not n_folds is None:
        for k, v in outputs.items():
            if not "fold" in k and isinstance(v[0], Number):
                mean_v = np.mean(v)
                outputs[k] = mean_v
                if wandb.run is not None:
                    wandb.summary[k] = mean_v

    return outputs
