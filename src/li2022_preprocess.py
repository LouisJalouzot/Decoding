import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from nilearn import image
from sklearn.preprocessing import StandardScaler

from src.utils import progress


def create_li2022_datasets(lang="EN"):
    path = Path("data/li2022")
    assert (
        path.exists()
    ), f"{path} does not exist, either the working directory {Path(os.getcwd())} is not the root of the repo or the data has not been downloaded."
    mask_path = path / "colin27_t1_tal_lin_mask.nii"
    assert mask_path.exists(), "You need to download the mask first (c.f. README.md)"
    input_path = path / "derivatives"
    target_path = Path("datasets") / f"li2022_{lang}"
    target_path_SS = Path("datasets") / f"li2022_{lang}_SS"
    for path in [target_path, target_path_SS]:
        if path.exists():
            if input(
                f"{path} already exists. Do you want to recreate it (Y/n)? "
            ) not in ["", "y", "Y"]:
                return
            else:
                shutil.rmtree(path)
    subjects = sorted([f.stem for f in input_path.glob(f"sub-{lang}*")])
    example_lpp_img = nib.load(
        next(iter((input_path / subjects[0] / "func").iterdir()))
    )
    acquisition_affine = example_lpp_img.affine.copy()
    acquisition_affine[np.arange(3), np.arange(3)] = 3.75
    mask = nib.load(mask_path)
    mask_SS = image.resample_to_img(
        mask,
        image.resample_img(example_lpp_img, target_affine=acquisition_affine),
        interpolation="nearest",
    )
    mask_SS = np.where(mask_SS.get_fdata().astype(bool))
    mask = image.resample_to_img(
        mask,
        example_lpp_img,
        interpolation="nearest",
    )
    mask = np.where(mask.get_fdata().astype(bool))

    with progress:
        task = progress.add_task(
            f"Loading and slicing brain scans for {len(subjects)} subjects",
            total=len(subjects),
        )
        for subject in subjects:
            input_subject_path = input_path / subject / "func"
            target_subject_path = target_path / subject.replace("sub-", "")
            target_subject_path_SS = target_path_SS / subject.replace("sub-", "")
            runs = []
            runs_SS = []
            scaler = StandardScaler()
            scaler_SS = StandardScaler()
            paths = sorted(input_subject_path.glob("*.nii.gz"))
            assert (
                len(paths) == 9
            ), f"Expected 9 runs at {input_subject_path}, got {len(path)}."
            subject_task = progress.add_task(f"Loading {subject}", total=len(paths))
            for path in paths:
                img = nib.load(path)
                img_SS = image.resample_img(img, target_affine=acquisition_affine)
                img = img.get_fdata()[mask].T
                img_SS = img_SS.get_fdata()[mask_SS].T
                img = np.nan_to_num(img.astype(np.float32), nan=0)
                img_SS = np.nan_to_num(img_SS.astype(np.float32), nan=0)
                img = StandardScaler().fit_transform(img)
                img_SS = StandardScaler().fit_transform(img_SS)
                scaler.partial_fit(img)
                scaler_SS.partial_fit(img_SS)
                runs.append(img)
                runs_SS.append(img_SS)
                progress.update(subject_task, advance=1)
            progress.remove_task(subject_task)
            subject_task = progress.add_task(f"Saving {subject}", total=len(paths))
            target_subject_path.mkdir(parents=True, exist_ok=True)
            target_subject_path_SS.mkdir(parents=True, exist_ok=True)
            for i, (run, run_SS) in enumerate(zip(runs, runs_SS)):
                np.save(target_subject_path / f"{i+1}.npy", scaler.transform(run)[5:-5])
                np.save(
                    target_subject_path_SS / f"{i+1}.npy",
                    scaler_SS.transform(run_SS)[5:-5],
                )
                progress.update(subject_task, advance=1)
            progress.remove_task(subject_task)
            progress.update(task, advance=1)


def build_mean_subject(lang="EN", SS=True):
    if SS:
        input_path = Path(f"datasets/li2022_{lang}_SS")
        target_path = Path(f"datasets/li2022_{lang}_SS_mean")
    else:
        input_path = Path(f"datasets/li2022_{lang}")
        target_path = Path(f"datasets/li2022_{lang}_mean")

    if target_path.exists():
        if input(
            f"{target_path} already exists. Do you want to recreate it (Y/n)? "
        ) not in ["", "y", "Y"]:
            return
        else:
            shutil.rmtree(target_path)
    target_path = target_path / f"mean_{lang}"
    target_path.mkdir(parents=True, exist_ok=True)
    subjects = os.listdir(input_path)
    for run in range(1, 10):
        with joblib_progress(
            f"Building mean subject for run {run}/9", total=len(subjects)
        ):
            mean = sum(
                Parallel(n_jobs=-1, return_as="generator_unordered")(
                    delayed(np.load)(input_path / subject / f"{run}.npy")
                    for subject in subjects
                )
            ) / len(subjects)
        mean = StandardScaler().fit_transform(mean)
        np.save(target_path / f"{run}.npy", mean)
