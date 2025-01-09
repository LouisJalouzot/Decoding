import logging
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from bids import BIDSLayout
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from sklearn.preprocessing import StandardScaler

from src.utils import console, progress


def read(f):
    scaler = StandardScaler()
    with h5py.File(f, "r") as hf5:
        X = hf5["data"][...]
    X = np.nan_to_num(X, nan=0)
    X = scaler.fit_transform(X)
    return f.stem, X


def create_lebel2023_dataset():
    source_path = Path("data/lebel2023")
    target_path = Path("datasets/lebel2023")
    assert (
        source_path.exists()
    ), f"{source_path} does not exist, either the working directory {os.getcwd()} is not the root of the repo or the data has not been downloaded."
    source_path_subjects = source_path / "derivatives" / "preprocessed_data"
    subjects = sorted(os.listdir(source_path_subjects))
    console.log(f"Found {len(subjects)} subjects.")
    for subject in subjects:
        target_path_subject = target_path / subject
        if target_path_subject.exists():
            if input(
                f"{target_path_subject} already exists. Do you want to recreate it (Y/n)? "
            ) not in ["", "y", "Y"]:
                continue
            else:
                shutil.rmtree(target_path_subject)
        run_files = list((source_path_subjects / subject).iterdir())
        run_files = [f for f in run_files if not "wheretheressmoke" in f.name]
        with joblib_progress(
            f"Reading brain scans for subject {subject}",
            total=len(run_files),
            console=console,
        ):
            runs = Parallel(n_jobs=-1)(delayed(read)(f) for f in run_files)
        scaler = StandardScaler()
        with progress:
            task = progress.add_task(
                f"Scaling and saving brain scans for subject {subject}",
                total=2 * len(runs),
            )
            for _, X in runs:
                scaler.partial_fit(X)
                progress.update(task, advance=1)
            target_path_subject.mkdir(parents=True, exist_ok=True)
            for run, X in runs:
                X = scaler.transform(X)
                np.save(
                    target_path_subject / f"{run}.npy", X.astype(np.float32)
                )
                progress.update(task, advance=1)
            progress.update(task, completed=True)


def process_bold_file(bold_file, subject, story, mask, dataset_path):
    path = dataset_path / subject / (story + ".npy")
    if path.exists():
        return

    scaler = StandardScaler()
    # Mask and drop first and last 10 volumes like original dataset
    a = bold_file.get_fdata()[mask, 10:-10]
    a = scaler.fit_transform(a.T)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, a)


def create_lebel2023_fmriprep_dataset():
    dataset_path = Path("datasets/lebel2023_fmriprep")
    if dataset_path.exists():
        if input(
            f"{dataset_path} already exists. Do you want to recreate it (Y/n)? "
        ) in ["", "y", "Y"]:
            shutil.rmtree(dataset_path)
        else:
            if input("Do you to continue its creation (Y/n)? ") not in [
                "",
                "y",
                "Y",
            ]:
                return

    console.log("Creating Lebel2023 fMRIprep dataset")
    console.log("Indexing BIDS dataset")
    layout = BIDSLayout("data/lebel2023/derivatives/fmriprep", validate=False)
    mask_files = layout.get(
        suffix="mask", extension="nii.gz", space="MNI152NLin2009cAsym"
    )
    mask_files = [f for f in mask_files if "task" in f.filename]
    mask_files = [f for f in mask_files if not "Localizer" in f.filename]
    mask_files = [f for f in mask_files if not "wheretheressmoke" in f.filename]
    mask = None
    with progress:
        task = progress.add_task("Combining masks", total=len(mask_files))
        for f in mask_files:
            run_mask = f.get_image().get_fdata()
            if mask is None:
                mask = run_mask == 1
            else:
                mask &= run_mask == 1
            progress.update(task, advance=1, refresh=True)
    console.log(f"{mask.sum()} voxels in the combined mask")
    bold_files = layout.get(suffix="bold", extension="nii.gz")
    bold_files = [f for f in bold_files if not "Localizer" in f.filename]
    bold_files = [f for f in bold_files if not "wheretheressmoke" in f.filename]
    with joblib_progress(
        "Processing BOLD files", total=len(bold_files), console=console
    ):
        Parallel(n_jobs=-1)(
            delayed(process_bold_file)(
                f.get_image(),
                f.entities["subject"],
                f.entities["task"],
                mask,
                dataset_path,
            )
            for f in bold_files
        )
