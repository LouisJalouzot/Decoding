from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

import wandb
from src.utils import merge_drop, progress


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
    n_folds,
    fold,
    train_ratio,
    valid_ratio,
    test_ratio,
    seed,
    fine_tune: dict[str, list[str]] = None,
    fine_tune_disjoint: bool = True,
):
    # TODO implement leave out
    if train_ratio is None:
        train_ratio = 1 - valid_ratio - test_ratio

    df = df[["dataset", "subject", "run"]].drop_duplicates()
    n_subjects = (
        df.groupby("dataset")
        .subject.apply("nunique")
        .reset_index(name="n_subjects")
    )
    occurrences = (
        df.groupby(["dataset", "run"])
        .subject.apply(list)
        .apply(len)
        .reset_index(name="occurrences")
    )
    occurrences = occurrences.merge(n_subjects)
    # Main runs are runs that are present for all subjects
    main_runs = occurrences[occurrences.occurrences == occurrences.n_subjects]
    main_runs = main_runs[["dataset", "run"]].drop_duplicates()
    main_runs = main_runs.reset_index(drop=True)
    subjects_runs = df[["dataset", "subject", "run"]].drop_duplicates()
    # Runs which are not main go into the train split
    extra_runs = df[["dataset", "run"]].drop_duplicates()
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
            test_size=max(test_ratio, 1 - train_ratio - valid_ratio),
            random_state=seed,
        )
        if test_ratio == 0:
            test_indices = []
        train_test_indices = [(train_indices, test_indices)]

    splits = []
    for i, (train_indices, test_indices) in enumerate(train_test_indices, 1):
        if fold is not None and fold != i:
            continue

        if valid_ratio > 0:
            train_indices, valid_indices = train_test_split(
                train_indices,
                train_size=train_ratio / (1 - test_ratio),
                test_size=valid_ratio / (1 - test_ratio),
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
        if fine_tune is not None:
            # Make fine_tune a dataframe for merging
            fine_tune = [(k, v) for k, vals in fine_tune.items() for v in vals]
            fine_tune = pd.DataFrame(fine_tune, columns=["dataset", "subject"])
            subjects_runs_ft = subjects_runs.merge(fine_tune)

            if fine_tune_disjoint:
                # Remove the fine-tuning runs from the main splits
                # If the fine-tuning subject has extra runs, fine-tune on them
                subjects_runs_ft_extra = subjects_runs_ft.merge(extra_runs)
                if len(subjects_runs_ft_extra) > 1:
                    subjects_runs_ft = subjects_runs_ft_extra

                runs_ft = subjects_runs_ft[["dataset", "run"]].drop_duplicates()
                df_split = merge_drop(df_split, runs_ft)

            subjects_runs_ft["split"] = "ft_train"
            df_split = pd.concat([df_split, subjects_runs_ft])

        df_split["fold"] = i
        splits.append(df_split)
        print(splits)

    splits = pd.concat(splits)

    if wandb.run is not None:
        wandb.log({"splits": wandb.Table(data=splits)})

    # Fix for NaN subjects
    df = df.merge(splits)

    # Save original X for multiple encoding voxels selections
    if "X" in df:
        df["orig_X"] = df.X

    return df


def find_best_encoding_voxels(df, top_encoding_voxels):
    dataset_subject = df[["dataset", "subject"]].drop_duplicates()
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
                progress.update(task, advance=1)
                continue
            X = np.concatenate(tuple(df_train_sel.orig_X))
            Y = np.concatenate(tuple(df_train_sel.Y))
            ridge = Ridge().fit(Y, X)
            df_valid_sel = df[subject_sel & (df.split.str.endswith("valid"))]
            X = np.concatenate(tuple(df_valid_sel.orig_X))
            Y = np.concatenate(tuple(df_valid_sel.Y))
            X_preds = ridge.predict(Y)
            r2 = r2_score(X, X_preds, multioutput="raw_values")
            voxels_to_keep = r2.argsort()[-n_voxels:]  # type: ignore
            df.loc[subject_sel, "X"] = df[subject_sel].orig_X.apply(
                lambda X: X[:, voxels_to_keep]
            )
            df.loc[subject_sel, "n_voxels"] = n_voxels
            progress.update(task, advance=1)

    return df
