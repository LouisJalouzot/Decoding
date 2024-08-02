from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import xarray as xr
from dask.diagnostics import ProgressBar

from src.brain_decoder import train_brain_decoder
from src.prepare_latents import prepare_latents
from src.utils import console, progress, standard_scale


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
        dataset = Path(f"data/{dataset}")
        if dataset.with_suffix(".nc").exists():
            X_ds.append(xr.open_dataarray(dataset.with_suffix(".nc")))
        elif dataset.with_suffix(".zarr").exists():
            X_ds.append(xr.open_zarr(dataset.with_suffix(".zarr")).data)
        else:
            raise FileNotFoundError(f"Dataset {dataset} not found")
    X_ds = xr.concat(X_ds, dim="run_id").squeeze()
    X_ds = X_ds.set_xindex(coord_names=["subject", "run"])
    if subjects is not None:
        X_ds = X_ds[X_ds.subject.isin(subjects)]
    if smooth > 0:
        X_ds = X_ds.rolling(tr=smooth, min_periods=1).mean()
    if lag > 0:
        X_ds = X_ds.sel(tr=slice(lag, None))
        X_ds = X_ds.assign_coords(tr=np.arange(X_ds.tr.size))
    X_ds["n_trs"] = X_ds.n_trs - np.abs(lag)
    console.log("Loading brain scans")
    with ProgressBar():
        X_ds = X_ds.compute()

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Unique runs sorted by decreasing number occurrences to select the ones with more data for valid and test
    runs = X_ds.subject.groupby("run").count().sortby("run").run
    n_runs = runs.size
    trs_by_run = X_ds.n_trs.groupby("run").last()
    max_n_trs = trs_by_run.max().item()

    Y_ds = []
    with progress:
        task = progress.add_task("Fetching latents for each run", total=n_runs)
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
            Y_run = standard_scale(Y_run, along="tr")
            n_trs = run_array.item()
            if lag < 0:
                Y_run = Y_run.sel(tr=slice(lag, None))
            elif lag > 0:
                Y_run = Y_run.sel(tr=slice(-lag))
            assert (
                Y_run.tr.size >= n_trs
            ), f"There should not be more brain scans than latents for {run}, {Y_run.tr.size} < {n_trs}"
            if Y_run.tr.size > n_trs + 1:
                console.log(
                    f"[red]{Y_run.tr.size - n_trs} > 1 latents trimmed for run {run}"
                )
            Y_run = Y_run.sel(tr=slice(n_trs))
            Y_run = Y_run.assign_coords(n_trs=n_trs)
            Y_run = Y_run.pad({"tr": (0, max_n_trs - n_trs)}, constant_values=np.nan)
            Y_ds.append(Y_run)
            progress.update(task, advance=1)
    Y_ds = standard_scale(xr.concat(Y_ds, dim="run"), along=["run", "tr"])
    Y_ds = Y_ds.assign_coords(tr=np.arange(Y_ds.tr.size))
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
        X_ds_sel = X_ds[X_ds.run.isin(selected_runs)]
        occurrences = X_ds_sel.subject.count().item()
        n_scans = X_ds_sel.n_trs.sum().item()
        console.log(
            f"{split} split: {len(selected_runs)} runs, {occurrences} occurrences and {n_scans} scans."
        )

    output = train_brain_decoder(
        X_ds, Y_ds, train_runs, valid_runs, test_runs, decoder=decoder, **decoder_params
    )

    torch.cuda.empty_cache()

    return output
