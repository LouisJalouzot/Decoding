import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from joblib_progress import joblib_progress


def standard_scale(ds: xr.DataArray, along=["run_id", "tr"]):
    ds_mean = ds.mean(dim=along, skipna=True)
    ds_scale = ds.fillna(0).std(dim=along, skipna=True)
    ds_scale = xr.where(ds_scale < 1e-6, 1, ds_scale)
    return (ds - ds_mean) / ds_scale


def read(subject, run_name, run):
    a = xr.open_dataarray(run).astype(np.float32)
    a = a.rename({"phony_dim_0": "tr", "phony_dim_1": "voxel"})
    a = a.expand_dims(dim="run_id", axis=0)
    a["run_id"] = [f"lebel2023/{subject}/{run_name}"]
    a = a.assign_coords(
        dataset=("run_id", ["lebel2023"]),
        subject=("run_id", [f"lebel2023/{subject}"]),
        run=("run_id", [f"lebel2023/{run_name}"]),
        n_voxels=("run_id", [a.voxel.size]),
        n_trs=("run_id", [a.tr.size]),
        tr=np.arange(a.tr.size),
        voxel=np.arange(a.voxel.size),
    )
    return standard_scale(a.fillna(0), along="tr")


def create_zarr_dataset(
    subjects=["UTS01", "UTS02", "UTS03"], name="3_subjects", format="nc"
):
    dataset_path = Path(f"data/lebel2023/{name}.{format}")
    if dataset_path.exists():
        if input(
            f"{dataset_path} already exists, do you want to overwrite it? (Y/n)"
        ) not in ["y", ""]:
            return
        else:
            if dataset_path.is_file():
                dataset_path.unlink()
            else:
                shutil.rmdir(dataset_path)
    path = Path("data/lebel2023/derivative/preprocessed_data")
    runs = {subject: list((path / subject).iterdir()) for subject in subjects}
    with joblib_progress(
        f"Loading data for subjects " + ", ".join(subjects),
        total=sum([len(runs[subject]) for subject in subjects]),
    ):
        ds = Parallel(n_jobs=-1)(
            delayed(read)(subject, run.stem, run)
            for subject in subjects
            for run in runs[subject]
        )
    with ProgressBar():
        ds = xr.concat(ds, dim="run_id")
        ds = ds.chunk({"run_id": 1, "voxel": ds.voxel.size, "tr": ds.tr.size})
        ds_scaled = ds.groupby("subject").map(standard_scale)
        if format == "zarr":
            ds_scaled.to_dataset(name="data").to_zarr(dataset_path)
        else:
            ds_scaled.to_netcdf(dataset_path, engine="netcdf4")
