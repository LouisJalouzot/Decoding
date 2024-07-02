from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.brain_decoder import train_brain_decoder
from src.fetch_data import fetch_data
from src.utils import console, progress


def scale_in_df(df, col):
    scaler = StandardScaler(copy=False)
    mask = df[col].notna()
    scaler.fit(np.concatenate(df.loc[mask, col].values))
    df.loc[mask, col] = df.loc[mask, col].apply(scaler.transform)


def train(
    subjects: Union[str, List[str]] = "UTS03",
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 3,
    smooth: int = 0,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    subsample_voxels: int = None,
    latents_batch_size: int = 64,
    scale_across_runs: bool = True,
    scale_across_subjects: bool = False,
    **decoder_params,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    with progress:
        if isinstance(subjects, str):
            subjects = [subjects]
        subjects = sorted(subjects)
        Xs, Ys = {}, {}
        task = progress.add_task("", total=len(subjects))
        for subject in subjects:
            progress.update(task, description=f"Fetching data for subject {subject}")
            subj_Xs, subj_Ys = fetch_data(
                subject=subject,
                model=model,
                tr=tr,
                context_length=context_length,
                subsample_voxels=subsample_voxels,
                smooth=smooth,
                lag=lag,
                batch_size=latents_batch_size,
            )
            Xs[subject] = subj_Xs
            Ys[subject] = subj_Ys
            progress.update(task, advance=1)
        Xs = pd.DataFrame(Xs)
        Ys = pd.DataFrame(Ys)
        # runs sorted by decreasing number of subjects
        runs = Xs.isna().sum(axis=1).sort_values().index
        Xs = Xs.loc[runs]
        Ys = Ys.loc[runs]
        runs = list(Xs.index)
        if scale_across_runs:
            task = progress.add_task("Scaling across runs", total=2 * len(subjects))
            for df in [Xs, Ys]:
                for subject in subjects:
                    scale_in_df(df, subject)
                    progress.update(task, advance=1)
        if scale_across_subjects:
            task = progress.add_task("Scaling across subjects", total=2 * len(runs))
            for df in [Xs, Ys]:
                for run in runs:
                    scale_in_df(df.T, run)
                    progress.update(task, advance=1)
    n_runs = len(runs)
    n_valid = max(1, int(valid_ratio * n_runs))
    n_test = max(1, int(test_ratio * n_runs))
    test_runs = runs[:n_test]
    valid_runs = runs[n_test : n_test + n_valid]
    train_runs = runs[n_test + n_valid :]
    for split, selected_runs in [
        ("Train", train_runs),
        ("Valid", valid_runs),
        ("Test", test_runs),
    ]:
        n_runs_split = Xs.loc[selected_runs].notna().values.sum()
        n_scans = (
            Xs.loc[selected_runs]
            .map(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 0)
            .values.sum()
        )
        console.log(f"{split} split: {n_runs_split} runs and {n_scans} scans.")

    Xs = Xs.map(lambda x: torch.Tensor(x) if isinstance(x, np.ndarray) else x)
    Ys = Ys.map(lambda y: torch.Tensor(y) if isinstance(y, np.ndarray) else y)
    output = train_brain_decoder(
        Xs, Ys, train_runs, valid_runs, test_runs, decoder=decoder, **decoder_params
    )

    torch.cuda.empty_cache()

    return output
