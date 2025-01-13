import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm.auto import tqdm

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
    print(f"Found {len(common_runs)} runs common to all subjects")

    # Create symbolic links for common runs
    total_operations = len(subjects) * len(common_runs)
    with tqdm(total=total_operations, desc="Creating balanced dataset") as pbar:
        for subject in subjects:
            subject_target = target_path / subject.name
            subject_target.mkdir(parents=True, exist_ok=True)
            for run in common_runs:
                source_file = source_path / subject.name / f"{run}.npy"
                target_file = subject_target / f"{run}.npy"
                relative_source = os.path.relpath(
                    source_file, target_file.parent
                )
                target_file.unlink(missing_ok=True)
                target_file.symlink_to(relative_source)
                pbar.update(1)
    print(f"Created balanced dataset at {target_path}")


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

    print(
        f"Processing {len(common_runs)} runs common to {len(all_subjects)} subjects in {source_path}"
    )

    # Process runs sequentially
    total_operations = len(common_runs)
    with tqdm(total=total_operations, desc="Creating mean subject") as pbar:
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
                mean_data /= len(all_subjects)
                scaler = StandardScaler()
                mean_data = scaler.fit_transform(mean_data)
                np.save(output_path, mean_data.astype(np.float32))
            pbar.update(1)

    print(f"Created mean subject at {mean_subject_path}")


def create_pca_dataset(
    dataset="lebel2023_fmriprep",
    n_components=768,
    per_subject=False,
):
    """Create a PCA-reduced version of the dataset."""
    source_path = Path("datasets") / dataset
    dataset = f"{dataset}_pca{n_components}"
    dataset_path = Path("datasets") / dataset
    dataset_path.mkdir(parents=True, exist_ok=True)

    subjects = sorted([d for d in source_path.iterdir() if d.is_dir()])

    def fit_pca(subject, ipca):
        """Fit IPCA on a subject's runs."""
        for run_file in tqdm(
            sorted(subject.glob("*.npy")),
            desc=f"Fitting IPCA on {subject.name}",
            leave=False,
        ):
            ipca.partial_fit(np.load(run_file))

    def transform_and_save(subject, ipca):
        """Transform and save a subject's runs."""
        scaler = StandardScaler()
        for run_file in tqdm(
            sorted(subject.glob("*.npy")),
            desc=f"Transforming {subject.name}",
            leave=False,
        ):
            data = ipca.transform(np.load(run_file))
            data = scaler.fit_transform(data)
            output_path = dataset_path / subject.name / run_file.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, data.astype(np.float32))

    if per_subject:
        # Process each subject independently
        for subject in tqdm(subjects, desc="Processing subjects"):
            ipca = IncrementalPCA(n_components=n_components, copy=False)
            fit_pca(subject, ipca)
            transform_and_save(subject, ipca)
    else:
        # First fit IPCA on all subjects
        ipca = IncrementalPCA(n_components=n_components, copy=False)
        for subject in tqdm(subjects, desc="Fitting IPCA on all subjects"):
            fit_pca(subject, ipca)

        # Then transform all subjects
        for subject in tqdm(subjects, desc="Transforming all subjects"):
            transform_and_save(subject, ipca)

    print(f"Created PCA dataset at {dataset_path}")
