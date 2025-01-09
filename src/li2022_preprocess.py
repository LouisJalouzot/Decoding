import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from nilearn import image
from sklearn.preprocessing import StandardScaler

from src.utils import console

SUBJECTS_TO_TRIM = [
    "EN061",
    "EN064",
    "EN065",
    "EN068",
    "EN070",
    "EN074",
    "EN075",
    "EN078",
    "EN083",
    "EN089",
    "EN091",
    "EN092",
    "EN094",
    "EN095",
    "EN096",
    "EN097",
    "EN099",
    "EN103",
    "EN104",
    "EN105",
    "EN108",
    "EN115",
]


def slice_subject(input_path, target_path, mask, acquisition_affine=None):
    if target_path.exists() and len(os.listdir(target_path)) == 9:
        return
    runs = []
    scaler = StandardScaler()
    paths = sorted(input_path.glob("*.nii.gz"))
    assert len(paths) == 9, f"Expected 9 runs at {input_path}, got {len(path)}."
    for path in paths:
        img = nib.load(path)
        if acquisition_affine is not None:
            img = image.resample_img(img, target_affine=acquisition_affine)
        img = img.get_fdata()[mask].T
        img = np.nan_to_num(img.astype(np.float32), nan=0)
        img = StandardScaler().fit_transform(img)
        scaler.partial_fit(img)
        runs.append(img)
    target_path.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(runs):
        np.save(target_path / f"{i+1}.npy", scaler.transform(run)[5:-5])


def create_li2022_datasets(lang="EN"):
    path = Path("data/li2022")
    assert (
        path.exists()
    ), f"{path} does not exist, either the working directory {Path(os.getcwd())} is not the root of the repo or the data has not been downloaded."
    mask_path = path / "colin27_t1_tal_lin_mask.nii"
    assert (
        mask_path.exists()
    ), "You need to download the mask first (c.f. README.md)"
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

    with joblib_progress(
        f"Loading and slicing brain scans for {len(subjects)} subjects",
        total=2 * len(subjects),
        console=console,
    ):
        Parallel(n_jobs=-1)(
            delayed(slice_subject)(
                input_path / subject / "func",
                t_path / subject.replace("sub-", ""),
                m,
                affine,
            )
            for subject in subjects
            for t_path, m, affine in [
                (target_path, mask, None),
                (target_path_SS, mask_SS, acquisition_affine),
            ]
        )
