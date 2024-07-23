import os
import shutil
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from src.utils import create_symlink, progress


def create_symlinks_lebel2023():
    path = Path("data/lebel2023")
    assert (
        path.exists()
    ), f"{path} does not exist, either the working directory {os.getcwd()} is not the root of the repo or the data has not been downloaded."
    data_dir = path / "derivative" / "preprocessed_data"
    subjects = sorted(os.listdir(data_dir))
    ### Building "all_subjects" directory
    target_dir = path / "all_subjects"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    for subject in subjects:
        (target_dir / subject).mkdir(parents=True, exist_ok=True)
        for run in os.listdir(data_dir / subject):
            create_symlink(target_dir / subject / run, data_dir / subject / run)
            run = run.replace(".hf5", ".TextGrid")
            create_symlink(
                target_dir / subject / run, data_dir.parent / "TextGrids" / run
            )
            run = run.replace(".TextGrid", ".wav")
            create_symlink(
                target_dir / subject / run, data_dir.parent.parent / "stimuli" / run
            )
    ### Building "3_subjects" directory
    target_dir_3 = path / "3_subjects"
    if target_dir_3.exists():
        shutil.rmtree(target_dir_3)
    target_dir_3.mkdir(parents=True, exist_ok=True)
    for subject in subjects[:3]:
        create_symlink(target_dir_3 / subject, target_dir / subject)


def create_3_subjects_dataset():
    with progress:
        path = Path("data/lebel2023/derivative/preprocessed_data")
        subjects = ["UTS01", "UTS02", "UTS03"]
        ds = []
        n_voxels = 0
        subject_task = progress.add_task("", total=len(subjects))
        for subject in subjects:
            progress.update(
                subject_task, description="Loading data for subject " + subject
            )
            n_trs = 0
            subject_data = []
            runs = list((path / subject).iterdir())
            for run in runs:
                with h5py.File(run, "r") as f:
                    a = da.from_array(f["data"][...]).astype(np.float32)
                a = xr.DataArray(a, dims=["trs", "voxels"])
                a = a.expand_dims(dim="runs", axis=0)
                a["runs"] = [run.stem]
                a = a.expand_dims(dim="subjects", axis=0)
                a["subjects"] = [subject]
                n_voxels = max(n_voxels, a.voxels.size)
                n_trs = max(n_trs, a.trs.size)
                subject_data.append(a)
            ds.append(subject_data)
            progress.update(subject_task, update=1)
    with ProgressBar():
        for i in range(len(ds)):
            for j in range(len(ds[i])):
                ds[i][j] = ds[i][j].pad(
                    {
                        "voxels": (0, n_voxels - ds[i][j].voxels.size),
                        "trs": (0, n_trs - ds[i][j].trs.size),
                    },
                    constant_values=np.nan,
                )
        ds = xr.combine_nested(ds, concat_dim=["subjects", "runs"])
        ds -= ds.mean(dim="trs", skipna=True)
        ds /= ds.std(dim="trs", skipna=True)
        ds.to_zarr("data/lebel2023/3_subjects.zarr")
