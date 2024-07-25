from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.brain_decoder import train_brain_decoder
from src.prepare_latents import prepare_latents
from src.utils import console, progress


def train(
    datasets: Union[str, List[str]] = "lebel2023/all_subjects",
    subjects: List[str] = None,
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 3,
    smooth: int = 0,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    latents_batch_size: int = 64,
    **decoder_params,
) -> dict:
    if isinstance(datasets, str):
        datasets = [datasets]
    X_ds = []
    for dataset in datasets:
        dataset = Path(f"data/{dataset}").with_suffix(".zarr")
        X_ds.append(xr.open_zarr(dataset).data)
    X_ds = xr.concat(X_ds, dim="run_id").squeeze()
    X_ds = X_ds.set_xindex(coord_names=["subject", "run"])
    if subjects is not None:
        ds = ds.sel(run_id=ds.subject.isin(subjects))
    if smooth > 0:
        X_ds = X_ds.rolling(tr=smooth, min_periods=1).mean()
    if lag > 0:
        X_ds = X_ds.sel(tr=slice(lag, None))
    X_ds["n_trs"] = X_ds.n_trs - np.abs(lag)

    np.random.seed(seed)
    torch.manual_seed(seed)

    runs = np.unique(X_ds.run.values)
    trs_by_run = X_ds.n_trs.groupby("run").last()
    max_n_trs = trs_by_run.max().values.item()

    Y_ds = []
    with progress:
        task = progress.add_task("Fetching latents for each run", total=len(runs))
        for run_array in trs_by_run:
            run = run_array.run.item()
            Y_run = prepare_latents(
                run=run,
                model=model,
                tr=tr,
                context_length=context_length,
                batch_size=latents_batch_size,
            )
            Y_run = xr.DataArray(Y_run, dims=["tr", "hidden_dim"], coords={"run": run})
            n_trs = run_array.values.item()
            if lag < 0:
                Y_run = Y_run.sel(tr=slice(lag, None))
            else:
                Y_run = Y_run.sel(tr=slice(-lag))
            assert (
                Y_run.tr.size >= n_trs
            ), f"There should not be more latents than brain scans for {run}."
            if Y_run.tr.size > n_trs + 1:
                console.log(
                    f"[red]{Y_run.tr.size - n_trs} > 1 latents trimmed for run {run}"
                )
            Y_run = Y_run.sel(tr=slice(n_trs))
            Y_run = Y_run.assign_coords(n_trs=n_trs)
            Y_run = Y_run.pad({"tr": (0, max_n_trs - n_trs)}, constant_values=np.nan)
            Y_ds.append(Y_run)
            progress.update(task, advance=1)
    Y_ds = xr.concat(Y_ds, dim="run")
    # TODO Continue
    # n_runs = len(runs)
    # n_valid = max(1, int(valid_ratio * n_runs))
    # n_test = max(1, int(test_ratio * n_runs))
    # test_runs = runs[:n_test]
    # valid_runs = runs[n_test : n_test + n_valid]
    # train_runs = runs[n_test + n_valid :]
    # for split, selected_runs in [
    #     ("Train", train_runs),
    #     ("Valid", valid_runs),
    #     ("Test", test_runs),
    # ]:
    #     n_runs_split = Xs.loc[selected_runs].notna().values.sum()
    #     n_scans = (
    #         Xs.loc[selected_runs]
    #         .map(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 0)
    #         .values.sum()
    #     )
    #     console.log(f"{split} split: {n_runs_split} runs and {n_scans} scans.")

    # output = train_brain_decoder(
    #     Xs, Ys, train_runs, valid_runs, test_runs, decoder=decoder, **decoder_params
    # )

    # torch.cuda.empty_cache()

    return X_ds, Y_ds
