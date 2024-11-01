import os
import shutil
from pathlib import Path

import h5py
import numpy as np
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
    source_path_subjects = source_path / "derivative" / "preprocessed_data"
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
        with joblib_progress(
            f"Reading brain scans for subject {subject}", total=len(run_files)
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


def create_lebel2023_balanced_dataset():
    source_path = Path("datasets/lebel2023")
    target_path = Path("datasets/lebel2023_balanced")

    # Get all subjects
    subjects = sorted([d for d in source_path.iterdir() if d.is_dir()])

    # Get runs for each subject
    subject_runs = {}
    for subject in subjects:
        subject_runs[subject.name] = {f.stem for f in subject.glob("*.npy")}

    # Find common runs across all subjects
    common_runs = set.intersection(*subject_runs.values())
    console.log(f"Found {len(common_runs)} runs common to all subjects")

    # Create symbolic links for common runs
    for subject in subjects:
        subject_target = target_path / subject.name
        subject_target.mkdir(parents=True, exist_ok=True)
        for run in common_runs:
            source_file = source_path / subject.name / f"{run}.npy"
            target_file = subject_target / f"{run}.npy"
            relative_source = os.path.relpath(source_file, target_file.parent)
            target_file.unlink(missing_ok=True)
            target_file.symlink_to(relative_source)
    console.log(f"Created balanced dataset at {target_path}")
