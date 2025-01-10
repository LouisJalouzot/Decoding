import os
import shutil
from pathlib import Path

import numpy as np
from fastsrm.identifiable_srm import IdentifiableFastSRM
from sklearn.preprocessing import StandardScaler

from src.utils import console, progress


def create_balanced_dataset(dataset="lebel2023"):
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


def create_mean_subject(dataset="lebel2023", subjects=None):
    """Create a mean subject by averaging brain activity across selected subjects."""
    source_path = Path("datasets") / dataset
    mean_subject_path = Path("datasets") / (dataset + "_mean") / "mean"

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
            output_path = mean_subject_path / f"{run}.npy"
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
