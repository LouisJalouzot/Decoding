from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from sklearn.preprocessing import StandardScaler

import wandb
from src.brain_decoder import train_brain_decoder
from src.prepare_latents import prepare_latents
from src.utils import console, progress


def read(dataset, subject, run, lag, smooth, stack):
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
    lag -= stack
    if lag > 0:
        X = X[lag:]
    elif lag < 0:
        X = X[:lag]
    return dataset, subject, run, X.shape[0], X.shape[1], X


def train(
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
    latents_batch_size: int = 64,
    return_data: bool = False,
    top_encoding_voxels: int = None,
    token_aggregation: str = "mean",  # Choose from ["first", "last", "max", "mean"]
    **decoder_params,
):
    assert (
        lag >= stack
    ), "Stacking induces lag so we should have lag >= stack for clarity"
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_runs = sum([len(r) for s in runs.values() for r in s.values()])
    if n_runs == 0:
        raise ValueError(
            f"No runs found in datasets {datasets}, have you run preprocess.py?"
        )
    with joblib_progress(f"Loading brain scans for datasets {datasets}", total=n_runs):
        df = Parallel(n_jobs=-1)(
            delayed(read)(dataset, subject, run, lag, smooth, stack)
            for dataset in runs
            for subject in runs[dataset]
            for run in runs[dataset][subject]
        )
    df = pd.DataFrame(
        sorted(df), columns=["dataset", "subject", "run", "n_trs", "n_voxels", "X"]
    )
    df["subject_id"] = df.apply(lambda x: x.dataset + "/" + x.subject, axis=1)
    assert (
        df.groupby(["dataset", "run"]).n_trs.nunique().eq(1).all()
    ), "Runs should have the same number of TRs for all subjects"
    n_trs_by_run = df.groupby(["dataset", "run"]).n_trs.first().reset_index()
    # runs sorted by decreasing number of occurrences
    runs = (
        df.groupby(["dataset", "run"])
        .subject.count()
        .sort_values(ascending=False)
        .reset_index()[["dataset", "run"]]
    )
    n_runs = len(runs)
    scaler = StandardScaler()
    with progress:
        task = progress.add_task("", total=n_runs)
        progress.update(task, description=f"Fetching latents for {n_runs} runs")
        latents = []
        for _, (dataset, run, n_trs) in n_trs_by_run.iterrows():
            Y, chunks = prepare_latents(
                dataset=dataset,
                run=run,
                model=model,
                tr=tr,
                context_length=context_length,
                token_aggregation=token_aggregation,
                batch_size=latents_batch_size,
            )
            scaler.partial_fit(Y)
            if lag > 0:
                Y = Y[:-lag]
                chunks = chunks.iloc[:-lag]
            elif lag < 0:
                Y = Y[-lag:]
                chunks = chunks.iloc[-lag:]
            if Y.shape[0] > n_trs + 1:
                console.log(
                    f"[red]{Y.shape[0] - n_trs} > 1 latents trimmed for run {run} in dataset {dataset}"
                )
            # If more latents than brain scans, drop last seconds of run
            Y = Y[:n_trs]
            chunks = chunks.iloc[:n_trs]
            latents.append(
                [
                    dataset,
                    run,
                    Y.shape[1],
                    Y,
                    list(chunks.chunk),
                    list(chunks.chunk_with_context),
                ]
            )
            progress.update(task, advance=1)
        progress.update(task, completed=True)
    latents = pd.DataFrame(
        latents,
        columns=["dataset", "run", "hidden_dim", "Y", "chunks", "chunks_with_context"],
    )
    latents["Y"] = latents.Y.apply(
        lambda x: torch.from_numpy(scaler.transform(x).astype(np.float32))
    )
    df = df.merge(latents, on=["dataset", "run"])

    run_counts = (
        df.groupby(["dataset", "run"])
        .subject.count()
        .to_frame(name="occurrences")
        .reset_index()
    )
    n_subjects = (
        df.groupby("dataset")
        .subject.nunique()
        .to_frame(name="n_subjects")
        .reset_index()
    )
    run_counts = run_counts.merge(n_subjects)
    # By default put all runs in the train split
    run_counts["split"] = "train"

    # Main runs are runs that have all subjects
    main_runs = run_counts.occurrences == run_counts.n_subjects

    # Distribute those runs in train, valid and test splits
    def return_split_sizes(x):
        n_runs = len(x)
        n_valid = max(1, int(valid_ratio * n_runs))
        n_test = max(1, int(test_ratio * n_runs))
        n_train = n_runs - n_valid - n_test
        return ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test

    run_counts.loc[main_runs, "split"] = (
        run_counts[main_runs].groupby("dataset").split.transform(return_split_sizes)
    )
    df = df.merge(run_counts[["dataset", "run", "split"]])
    assert np.isin(
        df.subject_id.unique(), df[df.split == "train"].subject_id.unique()
    ).all(), "All subjects should have at least one run in the train split"

    if top_encoding_voxels is not None:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        with progress:
            task = progress.add_task(
                f"Fitting a Ridge encoder for each subject and keeping the best {top_encoding_voxels} voxels.",
                total=df.subject_id.nunique(),
            )
            for _, (dataset, subject_id) in (
                df[["dataset", "subject_id"]].drop_duplicates().iterrows()
            ):
                subject_sel = df.subject_id == subject_id
                df_train_sel = df[subject_sel & (df.split == "train")]
                if isinstance(top_encoding_voxels, dict):
                    n_voxels = top_encoding_voxels[dataset]
                else:
                    n_voxels = top_encoding_voxels
                if df_train_sel.n_voxels.iloc[0] <= n_voxels:
                    progress.update(task, advance=1)
                    continue
                X = np.concatenate(tuple(df_train_sel.X))
                Y = np.concatenate(tuple(df_train_sel.Y))
                ridge = Ridge().fit(Y, X)
                df_valid_sel = df[subject_sel & (df.split == "valid")]
                X = np.concatenate(tuple(df_valid_sel.X))
                Y = np.concatenate(tuple(df_valid_sel.Y))
                X_preds = ridge.predict(Y)
                r2 = r2_score(X, X_preds, multioutput="raw_values")
                voxels_to_keep = r2.argsort()[-n_voxels:]  # type: ignore
                df.loc[subject_sel, "X"] = df[subject_sel].X.apply(
                    lambda X: X[:, voxels_to_keep]
                )
                df.loc[subject_sel, "n_voxels"] = n_voxels
                progress.update(task, advance=1)

    wandb.log(
        {
            "df": wandb.Table(
                dataframe=df.drop(columns=["X", "Y", "chunks", "chunks_with_context"])
            )
        },
        step=0,
    )

    df_train = df[df.split == "train"]
    df_valid = df[df.split == "valid"]
    df_test = df[df.split == "test"]

    run_counts = run_counts.split.value_counts()
    console.log(
        f"Train split: {run_counts.train} runs with {len(df_train)} occurrences and {df_train.n_trs.sum()} scans.\n"
        f"Valid split: {run_counts.valid} runs with {len(df_valid)} occurrences and {df_valid.n_trs.sum()} scans.\n"
        f"Test split: {run_counts.test} runs with {len(df_test)} occurrences and {df_test.n_trs.sum()} scans."
    )

    if return_data:
        return df_train, df_valid, df_test

    output = train_brain_decoder(
        df_train, df_valid, df_test, decoder=decoder, **decoder_params
    )

    torch.cuda.empty_cache()

    return output
