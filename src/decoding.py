import logging
import os
from collections import defaultdict
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

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
from src.utils import device, progress, set_seeds

logger = logging.getLogger(__name__)


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
    log_nlp_distances: bool | str | list[str],
    n_candidates: int,
    datasets: str | list[str],
    subjects: dict[str, list[str]] | None,
    runs: dict[str, dict[str, list[str]]] | None,
    tr: int,
    splitting: dict,
    fine_tune_cfg: dict,
    alignment_cfg: dict,
    top_encoding_voxels: int | None,
    latents_cfg: dict,
    wrapper_cfg: dict,
    decoder_cfg: dict,
    train_cfg: dict,
    performance_ceiling: bool,
):
    num_cpus = psutil.cpu_count()
    ram = psutil.virtual_memory().total / (1024**3)
    logger.info(f"Number of available CPUs: {num_cpus}")
    logger.info(f"Available RAM: {ram:.3g} GB")
    logger.info(f"Using device {device}")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(device.index)
        vram = gpu.total_memory / (1024**3)
        logger.info(f"Available VRAM: {vram:.3g} GB")

    set_seeds(seed)

    if isinstance(datasets, str):
        datasets = [datasets]

    # Find subject and run names in data if not provided
    if runs is None:
        if subjects is None:
            subjects = {
                dataset: sorted(os.listdir(f"datasets/{dataset}"))
                for dataset in datasets
            }
        else:
            for dataset, s in subjects.items():
                if not isinstance(s, list):
                    subjects[dataset] = [str(s)]
                else:
                    subjects[dataset] = [str(sub) for sub in s]
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
        wandb.summary.update({"subjects": subjects, "runs": runs})

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
                    progress.update(task, advance=1, refresh=True)
    df = pd.DataFrame(
        sorted(df),
        columns=["dataset", "subject", "run", "n_trs", "n_voxels", "X"],
    )

    # Fetch latents
    runs = df[["dataset", "run", "n_trs"]].drop_duplicates()
    n_runs = len(runs)
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
            data = align_latents(data, **alignment_cfg)
            data["dataset"] = dataset
            data["run"] = run
            # Truncate latents to match the number of TRs in the brain scans
            for k, v in data.items():
                if isinstance(v, np.ndarray | torch.Tensor):
                    data[k] = v[:n_trs]
            latents.append(data)
            progress.update(task, advance=1, refresh=True)
    latents = pd.concat(latents, axis=1).T
    df = df.merge(latents, on=["dataset", "run"])
    # Truncate brain scans to match the number of TRs in the latents
    df["X"] = df.apply(lambda r: r.X[: r.Y.shape[0]], axis=1)
    df["n_trs"] = df.apply(lambda r: r.Y.shape[0], axis=1)
    if performance_ceiling:
        # Setup to train the model to predict Y from Y to get measure ceilings
        n_features = df.Y.iloc[0].shape[1]
        for i, row in df.iterrows():
            df.at[i, "X"] = row.Y
            df.at[i, "n_voxels"] = n_features

    # Compute (CV) train/valid/test splits
    df = split_dataframe(df, seed, **splitting, **fine_tune_cfg)
    if meta["return_data"]:
        return df

    n_folds = splitting["n_folds"]
    outputs = defaultdict(list)
    for fold, df_fold in df.groupby("fold"):
        if top_encoding_voxels is not None:
            values_for_hash = (
                df_fold[["dataset", "subject", "run", "split", "fold"]].values,
                alignment_cfg,
            )
            voxels_to_keep = find_best_encoding_voxels(
                df_fold, top_encoding_voxels, values_for_hash
            )
            df_fold = df_fold.drop(columns=["n_voxels"]).merge(voxels_to_keep)
            for i, r in df_fold.iterrows():
                df_fold.at[i, "X"] = r.orig_X[:, r.voxels]
            progress.update(task, advance=1, refresh=True)

        in_dims = df_fold[["dataset", "subject", "n_voxels"]].drop_duplicates()
        in_dims = in_dims.set_index(["dataset", "subject"]).n_voxels
        in_dims = {
            level: in_dims.xs(level).to_dict()
            for level in in_dims.index.levels[0]
        }

        df_train = df_fold[df_fold.split == "train"]
        df_train = df_train.merge(compute_chunk_index(df_train))
        df_valid = df_fold[df_fold.split == "valid"]
        df_valid = df_valid.merge(compute_chunk_index(df_valid))
        df_test = df_fold[df_fold.split == "test"]
        df_test = df_test.merge(compute_chunk_index(df_test))
        df_ft_train = df_fold[df_fold.split == "ft_train"]
        df_ft_valid = df_fold[df_fold.split == "ft_valid"]
        if not df_ft_train.empty:
            df_ft_train = df_ft_train.merge(compute_chunk_index(df_ft_train))
            df_ft_valid = df_ft_valid.merge(compute_chunk_index(df_ft_valid))

        # Determine which splits to compute NLP distances for
        splits_to_compute = []
        if log_nlp_distances:
            if log_nlp_distances in [True, "all"]:
                # Compute for all splits (original behavior)
                splits_to_compute = [
                    "test",
                    "valid",
                    "train",
                    "ft_train",
                    "ft_valid",
                ]
            elif isinstance(log_nlp_distances, str):
                # Compute for a specific split
                splits_to_compute = [log_nlp_distances]
            elif isinstance(log_nlp_distances, list):
                # Compute for multiple specific splits
                splits_to_compute = log_nlp_distances
            else:
                logger.warning(
                    f"Invalid value for log_nlp_distances: {log_nlp_distances}. "
                    "Expected True, 'all', a split name, or a list of split names."
                )

        nlp_distances = {}
        for split, df_split in [
            ("test", df_test),
            ("valid", df_valid),
            ("train", df_train),
            ("ft_train", df_ft_train),
            ("ft_valid", df_ft_valid),
        ]:
            if df_split.empty or split not in splits_to_compute:
                continue
            data = df_split.drop_duplicates(["dataset", "run"])[nlp_cols]
            data = data.to_dict("series")
            for k, v in data.items():
                data[k] = v.explode().values
            nlp_distances[split] = compute_nlp_distances(**data)

        if n_folds is not None:
            logger.info(f"Fold {fold}/{n_folds}")
        logger.info(
            f"Train split: {df_train.run.nunique()} runs with {len(df_train)} occurrences and {df_train.n_trs.sum()} scans"
        )
        logger.info(
            f"Valid split: {df_valid.run.nunique()} runs with {len(df_valid)} occurrences and {df_valid.n_trs.sum()} scans"
        )
        logger.info(
            f"Test split: {df_test.run.nunique()} runs with {len(df_test)} occurrences and {df_test.n_trs.sum()} scans"
        )
        if not df_ft_train.empty:
            logger.info(
                f"Fine-tune train split: {df_ft_train.run.nunique()} runs with {len(df_ft_train)} occurrences and {df_ft_train.n_trs.sum()} scans"
            )
            logger.info(
                f"Fine-tune valid split: {df_ft_valid.run.nunique()} runs with {len(df_ft_valid)} occurrences and {df_ft_valid.n_trs.sum()} scans"
            )

        prefix = f"fold_{fold}/" if n_folds is not None else ""
        metrics, _decoder = train(
            df_train,
            df_valid,
            df_test,
            df_ft_train,
            df_ft_valid,
            in_dims,
            decoder_cfg=decoder_cfg,
            wrapper_cfg=wrapper_cfg,
            train_cfg=train_cfg,
            return_tables=return_tables,
            nlp_distances=nlp_distances,
            n_candidates=n_candidates,
            metrics_prefix=prefix,
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
