import os
import shutil
from collections import defaultdict
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from joblib_progress import joblib_progress


def read(subject, run):
    with h5py.File(run, "r") as f:
        a = da.from_array(f["data"][...]).astype(np.float32)
    a = xr.DataArray(a, dims=["trs", "voxels"])
    a = a.expand_dims(dim="runs", axis=0)
    a["runs"] = [run.stem]
    a = a.expand_dims(dim="subjects", axis=0)
    a["subjects"] = [f"lebel2023/{subject}"]
    return a


def create_zarr_dataset(subjects=["UTS01", "UTS02", "UTS03"], name="3_subjects"):
    dataset_path = Path(f"data/lebel2023/{name}.zarr")
    if dataset_path.exists():
        if input(
            f"{dataset_path} already exists, do you want to overwrite it? (Y/n)"
        ) not in ["y", ""]:
            return
        else:
            shutil.rmtree(dataset_path)
    path = Path("data/lebel2023/derivative/preprocessed_data")
    runs = {subject: list((path / subject).iterdir()) for subject in subjects}
    with joblib_progress(
        f"Loading data for subjects " + ", ".join(subjects),
        total=sum([len(runs[subject]) for subject in subjects]),
    ):
        out = Parallel(n_jobs=-1)(
            delayed(read)(subject, run) for subject in subjects for run in runs[subject]
        )
    n_voxels = max([a.voxels.size for a in out])
    n_trs = max([a.trs.size for a in out])
    ds = defaultdict(list)
    for a in out:
        a = a.pad(
            {"voxels": (0, n_voxels - a.voxels.size), "trs": (0, n_trs - a.trs.size)},
            constant_values=np.nan,
        )
        ds[a.subjects.item()].append(a)
    with ProgressBar():
        for subject in ds:
            ds[subject] = xr.concat(ds[subject], dim="runs")
        ds = xr.concat(list(ds.values()), dim="subjects")
        ds -= ds.mean(dim="trs", skipna=True)
        ds /= ds.std(dim="trs", skipna=True)
        ds = ds.chunk("auto")
        ds.to_zarr(dataset_path)
