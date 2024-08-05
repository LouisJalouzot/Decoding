import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from sklearn.preprocessing import StandardScaler

from src.brain_decoder import train_brain_decoder
from src.prepare_latents import prepare_latents
from src.utils import console, progress


def read(dataset, subject, run, lag, smooth, stack):
    path = Path("datasets") / dataset / subject / run
    path = path.with_suffix(".npy")
    X = torch.from_numpy(np.load(path))
    if smooth > 0:
        new_X = X.copy()
        count = np.ones((X.shape[0], 1))
        for i in range(1, smooth + 1):
            new_X[i:] += X[:-i]
            count[i:] += 1
        X = (new_X / count).astype(np.float32)
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
    **decoder_params,
) -> dict:
    assert (
        lag >= stack
    ), "Stacking induces lag so we should have lag >= stack for clarity"
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_runs = sum([len(r) for s in runs.values() for r in s.values()])
    with joblib_progress(f"Loading brain scans for datasets {datasets}", total=n_runs):
        df = Parallel(n_jobs=-1, backend="threading")(
            delayed(read)(dataset, subject, run, lag, smooth, stack)
            for dataset in runs
            for subject in runs[dataset]
            for run in runs[dataset][subject]
        )
    df = pd.DataFrame(
        df, columns=["dataset", "subject", "run", "n_trs", "n_voxels", "X"]
    )
    assert (
        df.groupby(["dataset", "run"]).n_trs.nunique().eq(1).all()
    ), "Runs should have the same number of TRs for all subjects"
    n_trs_by_run = df.groupby(["dataset", "run"]).n_trs.first().reset_index()
    # runs sorted by decreasing number of occurrences
    runs = (
        df.groupby(["dataset", "run"])
        .X.count()
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
            Y = prepare_latents(
                dataset=dataset,
                run=run,
                model=model,
                tr=tr,
                context_length=context_length,
                batch_size=latents_batch_size,
            )
            scaler.partial_fit(Y)
            if lag > 0:
                Y = Y[:-lag]
            elif lag < 0:
                Y = Y[-lag:]
            if Y.shape[0] > n_trs + 1:
                console.log(
                    f"[red]{Y.shape[0] - n_trs} > 1 latents trimmed for run {run} in dataset {dataset}"
                )
            # If more latents than brain scans, drop last seconds of run
            Y = Y[:n_trs]
            latents.append([dataset, run, Y.shape[1], Y])
            progress.update(task, advance=1)
    latents = pd.DataFrame(latents, columns=["dataset", "run", "hidden_dim", "Y"])
    latents["Y"] = latents.Y.apply(
        lambda x: torch.from_numpy(scaler.transform(x).astype(np.float32))
    )
    df = df.merge(latents, on=["dataset", "run"])
    n_valid = max(1, int(valid_ratio * n_runs))
    n_test = max(1, int(test_ratio * n_runs))
    test_runs = runs.iloc[:n_test]
    df_test = df.merge(test_runs)
    valid_runs = runs.iloc[n_test : n_test + n_valid]
    df_valid = df.merge(valid_runs)
    train_runs = runs.iloc[n_test + n_valid :]
    df_train = df.merge(train_runs)
    console.log(
        f"Train split: {n_runs - n_valid - n_test} runs with {len(df_train)} occurrences and {df_train.n_trs.sum()} scans.\n"
        f"Valid split: {n_valid} runs with {len(df_valid)} occurrences and {df_valid.n_trs.sum()} scans.\n"
        f"Test split: {n_test} runs with {len(df_test)} occurrences and {df_test.n_trs.sum()} scans."
    )

    output = train_brain_decoder(
        df_train, df_valid, df_test, decoder=decoder, **decoder_params
    )

    torch.cuda.empty_cache()

    return output
