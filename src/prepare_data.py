import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

import wandb
from src.utils import memory, merge_drop, progress

logger = logging.getLogger(__name__)


def read_brain_volume(dataset, subject, run, lag, smooth, stack):
    path = Path("datasets") / dataset / subject / run
    path = path.with_suffix(".npy")
    X = torch.from_numpy(np.load(path))
    if smooth > 0:
        new_X = X.clone()
        count = np.ones((X.shape[0], 1))
        for i in range(1, smooth + 1):
            new_X[i:] += X[:-i]
            count[i:] += 1
        X = (new_X / count).to(torch.float32)
    if stack > 0:
        X = X.unfold(0, stack + 1, 1).flatten(-2)
    if lag > 0:
        X = X[lag:]
    elif lag < 0:
        X = X[:lag]
    return dataset, subject, run, X.shape[0], X.shape[1], X


def compute_chunk_index(df):
    chunks_index = df[["dataset", "run", "n_trs"]].drop_duplicates()
    chunks_index["chunks_index"] = (
        chunks_index.n_trs.shift(1).cumsum().fillna(0).astype(int)
    )
    chunks_index["chunks_index"] = chunks_index.apply(
        lambda row: np.arange(row.n_trs) + row.chunks_index, axis=1
    )

    return chunks_index


def split_dataframe(
    df,
    seed,
    n_folds,
    fold,
    train_ratio,
    valid_ratio,
    test_ratio,
    fine_tune_subjects: dict | None,
    fine_tune_disjoint: bool | None,
    overlap_ratio: float | None,
    **kwargs,
):
    subjects_runs = df[["dataset", "subject", "run"]].drop_duplicates()
    n_subjects = (
        subjects_runs.groupby("dataset")
        .subject.apply("nunique")
        .reset_index(name="n_subjects")
    )
    n_subjects_total = n_subjects.n_subjects.sum()

    occurrences = (
        subjects_runs.groupby(["dataset", "run"])
        .subject.apply(list)
        .apply(len)
        .reset_index(name="occurrences")
    )
    occurrences = occurrences.merge(n_subjects)
    # Main runs are runs that are present for all subjects
    main_runs = occurrences[occurrences.occurrences == occurrences.n_subjects]
    main_runs = main_runs[["dataset", "run"]].drop_duplicates()
    main_runs = main_runs.reset_index(drop=True)
    # Runs which are not main go into the train split
    extra_runs = subjects_runs[["dataset", "run"]].drop_duplicates()
    extra_runs = merge_drop(extra_runs, main_runs)
    extra_runs["split"] = "train"

    # Select test runs
    if n_folds is not None:
        # Use K-fold cross validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        train_test_indices = kf.split(main_runs.index)
    else:
        # Use a single split
        train_indices, test_indices = train_test_split(
            main_runs.index,
            test_size=test_ratio,
            random_state=seed,
        )
        if test_ratio == 0:
            test_indices = []
        train_test_indices = [(train_indices, test_indices)]

    splits = []
    for i, (train_indices, test_indices) in enumerate(train_test_indices, 1):
        if fold is not None and i != fold:
            continue

        test_ratio_split = len(test_indices) / len(main_runs)
        if n_folds is None:
            # If using K-fold, train and valid ratio are relative to the data not used for testing
            valid_ratio_split = valid_ratio / (1 - test_ratio_split)
            if train_ratio is not None:
                train_ratio_split = train_ratio / (1 - test_ratio_split)
            else:
                train_ratio_split = None
            if overlap_ratio is not None:
                overlap_ratio_split = overlap_ratio / (1 - test_ratio_split)
            else:
                overlap_ratio_split = None
        else:
            # Otherwise, all ratios are relative to the whole dataset
            train_ratio_split = train_ratio
            valid_ratio_split = valid_ratio
            overlap_ratio_split = overlap_ratio

        if overlap_ratio_split is not None:
            if overlap_ratio_split == 0:
                overlap_indices = []

            overlap_indices, train_indices = train_test_split(
                train_indices,
                train_size=overlap_ratio_split,
                random_state=seed,
            )
            if train_ratio_split is not None:
                train_ratio_split -= overlap_ratio_split
                if train_ratio_split * n_subjects_total > 1:
                    logger.warning(
                        (
                            f"Fold [{i}/{n_folds}]: "
                            if n_folds is not None
                            else ""
                        )
                        + "Unbalanced numbers of not overlapping runs between subjects (try decreasing train_ratio or overlap_ratio)"
                    )
            else:
                train_ratio_split = 1 / n_subjects_total
            if valid_ratio > 0:
                valid_ratio_split /= 1 - overlap_ratio_split
                train_indices, valid_indices = train_test_split(
                    train_indices,
                    test_size=valid_ratio_split,
                    random_state=seed,
                )
                train_ratio_split /= 1 - valid_ratio_split
            else:
                valid_indices = []

            train_runs = main_runs.iloc[overlap_indices].copy()
            train_runs["split"] = "train"
            train_runs = [train_runs]
            for subject in subjects_runs.subject.unique():
                if len(train_indices) == 0:
                    raise ValueError(
                        "Not enough runs, try decreasing overlap_ratio or train_ratio"
                    )
                assert (
                    train_ratio_split > 0
                ), "Increase train_ratio or decrease overlap_ratio"
                if train_ratio_split >= 1:
                    subject_train_indices = train_indices
                else:
                    subject_train_indices, train_indices = train_test_split(
                        train_indices,
                        train_size=train_ratio_split,
                        random_state=seed,
                    )
                subject_train_runs = main_runs.iloc[
                    subject_train_indices
                ].copy()
                subject_train_runs["split"] = "train"
                subject_train_runs["subject"] = subject
                train_runs.append(subject_train_runs)
                train_ratio_split /= 1 - train_ratio_split
            train_runs = pd.concat(train_runs)
        else:
            if valid_ratio > 0:
                if (
                    train_ratio_split is not None
                    and train_ratio_split + valid_ratio_split >= 1
                ):
                    train_ratio_split = None
                train_indices, valid_indices = train_test_split(
                    train_indices,
                    train_size=train_ratio_split,
                    test_size=valid_ratio_split,
                    random_state=seed,
                )
            else:
                valid_indices = []
            train_runs = main_runs.iloc[train_indices].copy()
            train_runs["split"] = "train"

        valid_runs = main_runs.iloc[valid_indices].copy()
        valid_runs["split"] = "valid"
        test_runs = main_runs.iloc[test_indices].copy()
        test_runs["split"] = "test"

        df_split = pd.concat([train_runs, valid_runs, test_runs, extra_runs])
        if fine_tune_subjects is not None:
            # Make a dataframe out of fine_tune for merging
            fine_tune_df = [
                (k, v) for k, vals in fine_tune_subjects.items() for v in vals
            ]
            fine_tune_df = pd.DataFrame(
                fine_tune_df, columns=["dataset", "subject"]
            )
            subjects_runs_ft_valid = valid_runs.merge(fine_tune_df)
            subjects_runs_ft_valid["split"] = "ft_valid"
            # Get the runs for the fine-tuning subjects and remove the valid and test ones
            subjects_runs_ft = subjects_runs.merge(fine_tune_df)
            subjects_runs_ft = merge_drop(subjects_runs_ft, test_runs)
            subjects_runs_ft = merge_drop(subjects_runs_ft, valid_runs)

            if fine_tune_disjoint:
                # Remove the fine-tuning runs from the main splits
                # If the fine-tuning subject has extra runs, fine-tune on them
                subjects_runs_ft_extra = subjects_runs_ft.merge(extra_runs)
                if len(subjects_runs_ft_extra) > 1:
                    subjects_runs_ft = subjects_runs_ft_extra
                runs_ft = subjects_runs_ft[["dataset", "run"]].drop_duplicates()
                df_split = merge_drop(df_split, runs_ft)

            subjects_runs_ft["split"] = "ft_train"
            df_split = pd.concat(
                [df_split, subjects_runs_ft, subjects_runs_ft_valid]
            )

        df_split["fold"] = i
        splits.append(df_split)
    splits = pd.concat(splits)

    # Treat NaNs and non NaNs for subjects separately
    if "subject" in splits:
        sub = splits[~splits.subject.isna()]
        no_sub = splits[splits.subject.isna()].drop("subject", axis=1)
        df = pd.concat([df.merge(sub), df.merge(no_sub)])
    else:
        df = df.merge(splits)

    if wandb.run is not None:
        splits_to_log = df[
            ["fold", "dataset", "subject", "run", "split"]
        ].drop_duplicates()
        fig = px.scatter(
            splits_to_log,
            x="run",
            y="subject",
            color="split",
            facet_row="fold",
            facet_col="dataset",
            height=200 * splits_to_log["fold"].nunique(),
            width=400 * splits_to_log["dataset"].nunique(),
        )
        wandb.log(
            {
                "splits": wandb.Table(data=splits_to_log),
                "splits_viz": wandb.Plotly(fig),
            }
        )

    # Save original X for multiple encoding voxels selections
    if "X" in df:
        df["orig_X"] = df.X

    return df


# Ignore `df` for hashing efficiency, identification is provided in `values_for_hash`
@memory.cache(ignore=["df"])
def find_best_encoding_voxels(
    df, top_encoding_voxels, values_for_hash
) -> pd.DataFrame:
    dataset_subject = df[["dataset", "subject"]].drop_duplicates()
    voxels_to_keep = []
    with progress:
        task = progress.add_task(
            f"Fitting a Ridge encoder for each subject and keeping the best {top_encoding_voxels} voxels.",
            total=len(dataset_subject),
        )
        for _, (dataset, subject) in dataset_subject.iterrows():
            subject_sel = (df.dataset == dataset) & (df.subject == subject)
            df_train_sel = df[subject_sel & (df.split.str.endswith("train"))]
            if isinstance(top_encoding_voxels, dict):
                n_voxels = top_encoding_voxels[dataset]
            else:
                n_voxels = top_encoding_voxels
            if df_train_sel.n_voxels.iloc[0] <= n_voxels:
                progress.update(task, advance=1, refresh=True)
                continue
            X = np.concatenate(tuple(df_train_sel.orig_X))
            Y = np.concatenate(tuple(df_train_sel.Y))
            ridge = Ridge().fit(Y, X)
            df_valid_sel = df[subject_sel & (df.split.str.endswith("valid"))]
            X = np.concatenate(tuple(df_valid_sel.orig_X))
            Y = np.concatenate(tuple(df_valid_sel.Y))
            X_preds = ridge.predict(Y)
            r2 = r2_score(X, X_preds, multioutput="raw_values")
            voxels = r2.argsort()[-n_voxels:]  # type: ignore
            voxels_to_keep.append([dataset, subject, voxels, len(voxels), r2])
            progress.update(task, advance=1, refresh=True)

    return pd.DataFrame(
        voxels_to_keep,
        columns=["dataset", "subject", "voxels", "n_voxels", "r2"],
    )
