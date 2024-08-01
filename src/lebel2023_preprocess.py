import shutil
from collections import defaultdict
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
    a = xr.open_dataset(run).data.astype(np.float32)
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
    return subject, standard_scale(a.fillna(0), along="tr").chunk({"run_id": 1})


def create_zarr_dataset(
    subjects=["UTS01", "UTS02", "UTS03"], name="3_subjects", format="nc"
):
    dataset_path = Path(f"data/lebel2023/{name}.{format}")
    if dataset_path.exists():
        if input(
            f"{dataset_path} already exists, do you want to overwrite it? (Y/n)"
        ) not in ["y", "Y", ""]:
            return
        else:
            if dataset_path.is_file():
                dataset_path.unlink()
            else:
                shutil.rmtree(dataset_path)
    path = Path("data/lebel2023/derivative/preprocessed_data")
    runs = {subject: list((path / subject).iterdir()) for subject in subjects}
    with joblib_progress(
        f"Loading data for subjects " + ", ".join(subjects),
        total=sum([len(runs[subject]) for subject in subjects]),
    ):
        res = Parallel(n_jobs=-1)(
            delayed(read)(subject, run.stem, run)
            for subject in subjects
            for run in runs[subject]
        )
    ds = defaultdict(list)
    for subject, data in res:
        ds[subject].append(data)
    with ProgressBar():
        for subject in ds:
            print("Scaling", subject)
            ds[subject] = standard_scale(
                xr.concat(ds[subject], dim="run_id"),
                along=["run_id", "tr"],
            )
        ds = xr.concat(list(ds.values()), dim="run_id")
        if format == "zarr":
            ds.to_dataset(name="data").to_zarr(dataset_path)
        else:
            ds.to_netcdf(dataset_path, engine="netcdf4")
