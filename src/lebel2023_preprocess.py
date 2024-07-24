import shutil
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from joblib_progress import joblib_progress


def read(subject, run_name, run, n_trs, n_voxels):
    with h5py.File(run, "r") as f:
        a = xr.DataArray(f["data"][...], dims=["trs", "voxels"]).astype(np.float32)
    a = a.fillna(0)
    a_mean = a.mean(dim="trs")
    a_std = a.std(dim="trs")
    a = (a - a_mean) / xr.where(a_std < 1e-6, 1, a_std)
    a = a.pad(
        {"voxels": (0, n_voxels - a.voxels.size), "trs": (0, n_trs - a.trs.size)},
        constant_values=np.nan,
    )
    a = a.assign_coords(
        dataset="lebel2023",
        subject=subject,
        run=run_name,
        n_voxels=a.voxels.size,
        n_trs=a.trs.size,
    )
    a = a.expand_dims(dim="run_id", axis=0)
    a["run_id"] = [f"lebel2023/{subject}/{run_name}"]
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
    n_voxels, n_trs = 0, 0
    for subject in subjects:
        for run in runs[subject]:
            with h5py.File(run, "r") as f:
                trs, voxels = f["data"].shape
            n_trs = max(n_trs, trs)
            n_voxels = max(n_voxels, voxels)
    with joblib_progress(
        f"Loading data for subjects " + ", ".join(subjects),
        total=sum([len(runs[subject]) for subject in subjects]),
    ):
        ds = Parallel(n_jobs=-1)(
            delayed(read)(subject, run.stem, run, n_trs, n_voxels)
            for subject in subjects
            for run in runs[subject]
        )
    with ProgressBar():
        ds = xr.concat(ds, dim="run_id").chunk(
            {"run_id": 1, "voxels": n_voxels, "trs": n_trs}
        )
        grouped = ds.groupby("subject")
        ds_mean = grouped.mean(dim=["run_id", "trs"], skipna=True)
        ds_std = grouped.fillna(0).std(dim=["run_id", "trs"], skipna=True)
        ds_scaled = (ds - ds_mean) / xr.where(ds_std < 1e-6, 1, ds_std)
        ds_scaled.to_dataset(name="data").to_zarr(dataset_path)
