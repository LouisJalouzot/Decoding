import shutil
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

print(Client())


def read(subject, run):
    with h5py.File(run, "r") as f:
        a = da.from_array(f["data"][...]).astype(np.float32)
    a = xr.DataArray(a, dims=["trs", "voxels"])
    a = a.assign_coords(
        dataset="lebel2023",
        subject=subject,
        run=run.stem,
        n_voxels=a.voxels.size,
        n_trs=a.trs.size,
    )
    a = a.expand_dims(dim="run_id", axis=0)
    a["run_id"] = [f"lebel2023/{subject}/{run.stem}"]
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
        ds = Parallel(n_jobs=-1)(
            delayed(read)(subject, run) for subject in subjects for run in runs[subject]
        )
    n_voxels = max([a.voxels.size for a in ds])
    n_trs = max([a.trs.size for a in ds])
    for i, a in enumerate(ds):
        a = a.pad(
            {"voxels": (0, n_voxels - a.voxels.size), "trs": (0, n_trs - a.trs.size)},
            constant_values=np.nan,
        )
        ds[i] = a
    with ProgressBar():
        ds = xr.concat(ds, dim="run_id")
        ds -= ds.groupby("subject").mean(dim="trs", skipna=True)
        ds_std = ds.groupby("subject").std(dim="trs", skipna=True)
        ds /= xr.where(ds_std < 1e-6, 1, ds_std)
        ds = ds.chunk("auto")
        ds.to_zarr(dataset_path)
