from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

import wandb
from src.utils import progress


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


def generate_splits(
    df,
    n_folds=None,
    fold=None,
    train_ratio=None,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=0,
):
    """Generate train/valid/test splits for each dataset"""
    if train_ratio is None:
        train_ratio = 1 - valid_ratio - test_ratio
    df = df[["dataset", "run", "subject"]].copy()
    df["n_subjects"] = df.groupby("dataset").subject.transform("nunique")
    df = (
        df.groupby(["dataset", "run", "n_subjects"])
        .subject.apply(list)
        .reset_index(name="subjects")
    )
    df["occurrences"] = df.subjects.apply(len)
    df["fold"] = 1
    df["split"] = "train"

    df_splits = []
    for _, df_dataset in df.groupby("dataset"):
        # Main runs are runs that have all subjects
        main_runs_indices = df_dataset.query("occurrences == n_subjects").index

        if n_folds is not None:
            # Use K-fold cross validation
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            kf = kf.split(main_runs_indices)
            for i, (train_indices, test_indices) in enumerate(kf, 1):
                if fold is not None and i != fold:
                    continue

                _, valid_indices = train_test_split(
                    train_indices,
                    train_size=train_ratio / (1 - test_ratio),
                    test_size=valid_ratio / (1 - test_ratio),
                    random_state=seed,
                )

                # Going back to the df_dataset indices
                train_indices = main_runs_indices[train_indices]
                valid_indices = main_runs_indices[valid_indices]
                test_indices = main_runs_indices[test_indices]

                df_fold = df_dataset.copy()
                df_fold["fold"] = i
                df_fold.loc[test_indices, "split"] = "test"
                df_fold.loc[valid_indices, "split"] = "valid"
                df_splits.append(df_fold)
        else:
            # Use a single split
            train_indices, test_indices = train_test_split(
                main_runs_indices,
                test_size=test_ratio,
                random_state=seed,
            )  # Unlike KFold, train_test_split returns the elements of the original array, not the indices
            _, valid_indices = train_test_split(
                train_indices,
                train_size=train_ratio / (1 - test_ratio),
                test_size=valid_ratio / (1 - test_ratio),
                random_state=seed,
            )
            df_dataset.loc[test_indices, "split"] = "test"
            df_dataset.loc[valid_indices, "split"] = "valid"
            df_splits.append(df_dataset)

    return pd.concat(df_splits).drop(
        columns=["n_subjects", "subjects", "occurrences"]
    )


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
    subjects_runs = df[["dataset", "subject", "run"]].drop_duplicates()

    if fine_tune is not None:
        # Make fine_tune a dataframe for merging
        fine_tune = [(k, v) for k, vals in fine_tune.items() for v in vals]
        fine_tune = pd.DataFrame(fine_tune, columns=["dataset", "subject"])
        df_fine_tune = subjects_runs.merge(fine_tune)
        df_fine_tune = generate_splits(
            df=df_fine_tune,
            n_folds=n_folds,
            fold=fold,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            seed=seed,
        ).merge(fine_tune)
        df_fine_tune = df_fine_tune.replace(
            {"split": {"train": "ft_train", "valid": "ft_valid"}}
        )
        if fine_tune_disjoint:
            tmp = df_fine_tune[df_fine_tune.split != "test"]
            # Remove fine tuning subjects and fine-tuning train and valid runs from the main splits
            for to_drop in ["subject", "run"]:
                subjects_runs = subjects_runs.merge(
                    tmp[["dataset", to_drop]],
                    how="left",
                    indicator=True,
                )
                subjects_runs = subjects_runs[subjects_runs._merge != "both"]
                subjects_runs = subjects_runs.drop(columns="_merge")

    df_splits = generate_splits(
        df=subjects_runs,
        n_folds=n_folds,
        fold=fold,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    if fine_tune is not None:
        if fine_tune_disjoint:
            df_splits = df_splits.merge(subjects_runs)
        df_splits = pd.concat([df_splits, df_fine_tune]).drop_duplicates()

    df = df.merge(df_splits)

    if wandb.run is not None:
        wandb.log({"splits": wandb.Table(data=df_splits)})

    # Save original X for multiple encoding voxels selections
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
            df_train_sel = df[subject_sel & (df.split == "train")]
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
            df_valid_sel = df[subject_sel & (df.split == "valid")]
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
