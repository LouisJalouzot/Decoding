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


def build_mean_subject(lang="EN", SS=True, trimmed=False):
    input_path = f"datasets/li2022_{lang}"
    if SS:
        input_path += "_SS"
    target_path = input_path
    if trimmed:
        target_path += "_trimmed"
    input_path = Path(input_path)
    target_path = Path(target_path + "_mean")

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
    if trimmed:
        subjects = [s for s in subjects if s not in SUBJECTS_TO_TRIM]
    for run in range(1, 10):
        with joblib_progress(
            f"Building mean subject for run {run}/9",
            total=len(subjects),
            console=console,
        ):
            mean = sum(
                Parallel(n_jobs=-1, return_as="generator_unordered")(
                    delayed(np.load)(input_path / subject / f"{run}.npy")
                    for subject in subjects
                )
            ) / len(subjects)
        mean = StandardScaler().fit_transform(mean)
        np.save(target_path / f"{run}.npy", mean)


def build_SRM_dataset(
    input_path="datasets/li2022_EN_SS",
    n_components=2000,
    valid_ratio=0.1,
    test_ratio=0.1,
):
    from fastsrm.identifiable_srm import IdentifiableFastSRM

    input_path = Path("datasets/li2022_EN_SS")
    if n_components >= 1000:
        ext = (
            f"_SRM_{n_components//1000}k_valid_{valid_ratio}_test_{test_ratio}"
        )
    else:
        ext = f"_SRM_{n_components}_valid_{valid_ratio}_test_{test_ratio}"
    target_path = input_path.with_name(input_path.name + ext)
    if target_path.exists():
        if input(
            f"{target_path} already exists. Do you want to recreate it (Y/n)? "
        ) not in ["", "y", "Y"]:
            return
        else:
            shutil.rmtree(target_path)
    subjects = sorted(os.listdir(input_path))
    n_subjects = len(subjects)
    runs = sorted([p.stem for p in (input_path / subjects[0]).iterdir()])
    n_runs = len(runs)
    X = np.full((n_subjects, n_runs), np.nan, dtype=object)

    def read(i, j, input_path):
        return i, j, np.load(input_path / subjects[i] / f"{runs[j]}.npy").T

    with joblib_progress(
        f"Loading brain scans for {n_subjects} subjects with {n_runs} runs each",
        total=n_subjects * n_runs,
        console=console,
    ):
        out = Parallel(n_jobs=-1)(
            delayed(read)(i, j, input_path)
            for i in range(n_subjects)
            for j in range(n_runs)
        )

    for i, j, x in out:
        X[i, j] = x

    n_valid = max(1, int(valid_ratio * n_runs))
    n_test = max(1, int(test_ratio * n_runs))

    srm = IdentifiableFastSRM(
        n_components=n_components, n_jobs=-1, verbose=True
    )
    srm.fit(X[:, n_test + n_valid :])

    def write(target_path, subject, run, X, W):
        (target_path / subject).mkdir(parents=True, exist_ok=True)
        np.save(target_path / subject / f"{run}.npy", X.T @ W)

    with joblib_progress(
        "Projecting and saving brain scans",
        total=n_subjects * n_runs,
        console=console,
    ):
        Parallel(n_jobs=-1)(
            delayed(write)(
                target_path
                / (
                    "test"
                    if j < n_test
                    else ("valid" if j < n_test + n_valid else "train")
                ),
                subjects[i],
                runs[j],
                X[i, j],
                srm.basis_list[i],
            )
            for i in range(n_subjects)
            for j in range(n_runs)
        )
