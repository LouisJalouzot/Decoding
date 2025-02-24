import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

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
        X = new_X / count
    if stack > 0:
        X = X.unfold(0, stack + 1, 1).flatten(-2)
    if lag > 0:
        X = X[lag:]
    elif lag < 0:
        X = X[:lag]
    return dataset, subject, run, X.shape[0], X.shape[1], X.float()


def compute_chunk_index(df):
    if df.empty:
        return df

    chunks_index = df[["dataset", "run", "n_trs"]].drop_duplicates()
    chunks_index["chunks_index"] = (
        chunks_index.n_trs.shift(1).cumsum().fillna(0).astype(int)
    )
    chunks_index["chunks_index"] = chunks_index.apply(
        lambda row: np.arange(row.n_trs) + row.chunks_index, axis=1
    )

    return chunks_index


def shuffle_split_runs(runs, ratio=None):
    if ratio is None:
        return runs, []

    runs = np.random.permutation(runs)
    n = int(ratio * len(runs))
    if ratio > 0 and n == 0:
        n = 1

    return runs[:n], runs[n:]


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
    overlap: float,
    **kwargs,
):
    splits = []
    subjects_runs = df[["dataset", "subject", "run"]].drop_duplicates()
    # For each dataset
    for dataset, subject_runs in subjects_runs.groupby("dataset"):
        subjects = subject_runs.subject.unique()
        n_subjects = len(subjects)
        occurrences = subjects_runs.groupby("run").subject.nunique()
        # Main runs are runs that are present for all subjects
        main_runs = occurrences[occurrences == n_subjects].index
        n_main_runs = len(main_runs)
        extra_runs = occurrences[occurrences < n_subjects].index

        # Select test runs
        if n_folds is not None:
            # Use K-fold cross validation
            dataset_splits = KFold(
                n_splits=n_folds, shuffle=True, random_state=seed
            ).split(main_runs)
            # KFold returns indices, get back actual runs
            dataset_splits = [
                (main_runs[test], main_runs[train])
                for train, test in dataset_splits
            ]
        else:
            # Use a single split
            dataset_splits = [shuffle_split_runs(main_runs, test_ratio)]

        for i, (test_runs, train_valid_runs) in enumerate(
            dataset_splits, start=1
        ):
            if fold is not None and i != fold:
                # Select only the requested fold
                continue

            # Update ratio to account for the allocated test runs
            test_ratio_split = len(test_runs) / n_main_runs
            if n_folds is not None:
                # When using cross validation, train and valid ratio are relative to the amount of data remaining after test split
                valid_ratio_split = valid_ratio
                train_ratio_split = train_ratio
            else:
                # When using a single split, train and valid ratio are relative to the amount of data before test split
                valid_ratio_split = valid_ratio / (1 - test_ratio_split)
                if train_ratio is not None:
                    train_ratio_split = train_ratio / (1 - test_ratio_split)
                else:
                    train_ratio_split = None

            # then get valid runs
            valid_runs, train_runs = shuffle_split_runs(
                train_valid_runs, valid_ratio_split
            )
            # Actual valid_ratio of the split
            valid_ratio_split = len(valid_runs) / len(train_valid_runs)
            if train_ratio_split is not None:
                train_ratio_split /= 1 - valid_ratio_split

            # Then we need to get runs overlapping between subjects
            if train_ratio_split is not None:
                train_overlap_ratio_split = train_ratio_split * overlap
            else:
                train_overlap_ratio_split = None
            n_train_runs = len(train_runs)
            overlap_runs, train_runs = shuffle_split_runs(
                train_runs, train_overlap_ratio_split
            )
            if train_ratio_split is not None and train_overlap_ratio_split < 1:
                # Actual train_overlap_ratio of the split
                train_overlap_ratio_split = len(overlap_runs) / n_train_runs
                train_ratio_split /= 1 - train_overlap_ratio_split
            # Remaining total ratio of runs to allocate without overlap
            if train_ratio_split is not None:
                total_train_ratio_split = (
                    train_ratio_split * n_subjects * (1 - overlap)
                )
                if total_train_ratio_split > 1:
                    raise ValueError(
                        f"train_ratio * n_subjects * (1 - overlap) / (1 - valid_ratio) > 1, try increasing overlap or decreasing train_ratio"
                    )
            else:
                total_train_ratio_split = None
            train_runs, unused_runs = shuffle_split_runs(
                train_runs, total_train_ratio_split
            )
            if overlap < 1 and len(train_runs) < n_subjects:
                raise ValueError(
                    f"Not enough remaining runs ({len(train_runs)}) to allocate to the {n_subjects} subjects without overlapping, try increasing overlap or decreasing train_ratio"
                )
            else:
                train_runs = np.random.permutation(train_runs)
                train_runs = np.array_split(train_runs, n_subjects)
                train_runs = zip(subjects, train_runs)

            df_splits = subject_runs.copy()
            df_splits["split"] = pd.NA
            df_splits["split"] = df_splits["split"].astype("string")
            df_splits.loc[df_splits.run.isin(test_runs), "split"] = "test"
            df_splits.loc[df_splits.run.isin(valid_runs), "split"] = "valid"
            df_splits.loc[df_splits.run.isin(overlap_runs), "split"] = "train"
            df_splits.loc[df_splits.run.isin(extra_runs), "split"] = "train"
            for e in train_runs:
                subject, runs = e
                df_splits.loc[
                    (df_splits.subject == subject) & (df_splits.run.isin(runs)),
                    "split",
                ] = "train"
            df_splits["fold"] = i
            splits.append(df_splits)

            # TODO: fix
            # if fine_tune_subjects is not None:
            #     # Make a dataframe out of fine_tune for merging
            #     fine_tune_df = [
            #         (k, v)
            #         for k, vals in fine_tune_subjects.items()
            #         for v in vals
            #     ]
            #     fine_tune_df = pd.DataFrame(
            #         fine_tune_df, columns=["dataset", "subject"]
            #     )
            #     subjects_runs_ft_valid = valid_runs.merge(fine_tune_df)
            #     subjects_runs_ft_valid["split"] = "ft_valid"
            #     # Get the runs for the fine-tuning subjects and remove the valid and test ones
            #     subjects_runs_ft = subjects_runs.merge(fine_tune_df)
            #     subjects_runs_ft = merge_drop(subjects_runs_ft, test_runs)
            #     subjects_runs_ft = merge_drop(subjects_runs_ft, valid_runs)

            #     if fine_tune_disjoint:
            #         # Remove the fine-tuning runs from the main splits
            #         # If the fine-tuning subject has extra runs, fine-tune on them
            #         subjects_runs_ft_extra = subjects_runs_ft.merge(extra_runs)
            #         if len(subjects_runs_ft_extra) > 1:
            #             subjects_runs_ft = subjects_runs_ft_extra
            #         runs_ft = subjects_runs_ft[
            #             ["dataset", "run"]
            #         ].drop_duplicates()
            #         df_split = merge_drop(df_split, runs_ft)

            #     subjects_runs_ft["split"] = "ft_train"
            #     df_split = pd.concat(
            #         [df_split, subjects_runs_ft, subjects_runs_ft_valid]
            #     )

    splits = pd.concat(splits)

    if wandb.run is not None:
        fig = px.scatter(
            splits,
            x="run",
            y="subject",
            color="split",
            facet_row="fold",
            facet_col="dataset",
            height=(50 * splits["subject"].nunique()) * splits["fold"].nunique()
            + 300,
            width=(20 * splits["run"].nunique()) * splits["dataset"].nunique()
            + 200,
            category_orders={"split": ["train", "valid", "test"]},
        )
        wandb.log({"splits": wandb.Plotly(fig)})

    df = df.merge(splits)
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
