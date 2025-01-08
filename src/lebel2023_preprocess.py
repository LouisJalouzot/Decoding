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


def create_lebel2023_balanced_dataset(dataset="lebel2023"):
    source_path = Path("datasets") / dataset
    target_path = Path("datasets") / (dataset + "_balanced")

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


def create_lebel2023_mean_subject(
    dataset="lebel2023", name="lebel2023_mean", subjects=None
):
    """Create a mean subject by averaging brain activity across selected subjects."""
    source_path = Path("datasets") / dataset
    mean_subject_path = Path("datasets") / name / "mean"

    # Check if target directory exists
    if mean_subject_path.exists():
        if input(
            f"{mean_subject_path} already exists. Do you want to recreate it (y/N)? "
        ) in ["y", "Y"]:
            shutil.rmtree(mean_subject_path)
        else:
            if input("Do you want to continue its creation (Y/n)? ") not in [
                "",
                "y",
                "Y",
            ]:
                return

    all_subjects = sorted([d for d in source_path.iterdir() if d.is_dir()])

    # Filter subjects if specified
    if subjects is not None:
        selected_subjects = [d for d in all_subjects if d.name in subjects]
        if not selected_subjects:
            raise ValueError("No matching subjects found")
        all_subjects = selected_subjects

    # Create mean subject directory
    mean_subject_path.mkdir(parents=True, exist_ok=True)

    # Get runs common to all selected subjects
    subject_runs = {}
    for subject in all_subjects:
        subject_runs[subject.name] = {f.stem for f in subject.glob("*.npy")}
    common_runs = set.intersection(*subject_runs.values())

    console.log(
        f"Processing {len(common_runs)} runs common to {len(all_subjects)} subjects in {source_path}"
    )

    # Process runs sequentially
    with progress:
        total_steps = len(common_runs) * len(all_subjects)
        task = progress.add_task("Creating mean subject", total=total_steps)
        for run in sorted(common_runs):
            output_path = Path("datasets") / name / "mean" / f"{run}.npy"
            if not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                mean_data = None
                for subject in all_subjects:
                    run_file = subject / f"{run}.npy"
                    if mean_data is None:
                        mean_data = np.load(run_file)
                    else:
                        mean_data += np.load(run_file)
                    progress.update(task, advance=1, refresh=True)
                mean_data /= len(all_subjects)
                scaler = StandardScaler()
                mean_data = scaler.fit_transform(mean_data)
                np.save(output_path, mean_data.astype(np.float32))
            else:
                # Skip all subjects for this run if file exists
                progress.update(task, advance=len(all_subjects), refresh=True)

    console.log(f"Created mean subject at {mean_subject_path}")
