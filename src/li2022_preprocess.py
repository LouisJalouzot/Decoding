import os
import shutil
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from joblib import Parallel, cpu_count, delayed
from joblib_progress import joblib_progress
from nilearn import image
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.utils import console

path = Path("data/li2022")
assert (
    path.exists()
), f"{path} does not exist, either the working directory {Path(os.getcwd())} is not the root of the repo or the data has not been downloaded."


def slice_subject(input_path, target_path, mask):
    runs = []
    scaler = StandardScaler()
    for path in sorted(input_path.iterdir()):
        img = nib.load(path).get_fdata()[mask].T
        img = np.nan_to_num(img, nan=0)
        img = StandardScaler().fit_transform(img)
        scaler.partial_fit(img)
        runs.append(img)
    assert len(runs) == 9, f"Expected 9 runs at {target_path}, got {len(runs)}."
    target_path.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(runs):
        np.save(target_path / f"{i+1}.npy", scaler.transform(run)[5:-5])


def create_li2022_dataset(lang="EN"):
    mask_path = path / "colin27_t1_tal_lin_mask.nii"
    assert mask_path.exists(), "You need to download the mask first (c.f. README.md)"
    input_path = path / "derivatives"
    target_path = Path("datasets") / f"li2022_{lang}"
    if target_path.exists():
        if input(
            f"{target_path} already exists. Do you want to recreate it (Y/n)? "
        ) not in ["", "y", "Y"]:
            return
        else:
            shutil.rmtree(target_path)
    subjects = sorted([f.stem for f in input_path.glob(f"sub-{lang}*")])
    example_lpp_img = nib.load(
        next(iter((input_path / subjects[0] / "func").iterdir()))
    )
    mask = nib.load(mask_path)
    mask = image.resample_to_img(
        mask,
        example_lpp_img,
        interpolation="nearest",
    )
    mask = np.where(mask.get_fdata().astype(bool))

    with joblib_progress(
        f"Reading and slicing brain scans for {len(subjects)} subjects",
        total=len(subjects),
    ):
        Parallel(n_jobs=cpu_count() // 2, prefer="processes")(
            delayed(slice_subject)(
                input_path / subject / "func",
                target_path / subject.replace("sub-", ""),
                mask,
            )
            for subject in subjects
        )


def build_mean_subject(lang="EN", input_path=None, target_path=None, n_jobs=-1):
    if input_path is None:
        input_path = path / f"all_{lang}"
    else:
        input_path = Path(input_path)
    if target_path is None:
        target_path = path / f"mean_{lang}" / f"mean_{lang}"
    else:
        target_path = Path(target_path)
    subjects = os.listdir(input_path)
    n_subjects = len(subjects)
    console.log(f"Building mean subject for {n_subjects} subjects.")
    input("Subjects found are: " + ", ".join(subjects) + ". Continue?")
    target_path.mkdir(parents=True, exist_ok=True)

    def aux(run):
        target_text = target_path / f"{run}.TextGrid"
        if not target_text.exists():
            create_symlink(target_text, text_file(lang, run))
        target_audio = target_path / f"{run}.wav"
        if not target_audio.exists():
            create_symlink(target_audio, audio_file(lang, run))
        if (target_path / f"{run}.npy").exists():
            return
        data = None
        for subject in tqdm(subjects):
            file = input_path / subject / str(run)
            if file.with_suffix(".npy").exists():
                subject_data = np.load(file.with_suffix(".npy"))
            elif file.with_suffix(".nii.gz").exists():
                subject_data = nib.load(file.with_suffix(".nii.gz")).get_fdata()
            elif file.with_suffix(".hf5").exists():
                with h5py.File(file.with_suffix(".hf5"), "r") as f:
                    subject_data = f["data"][:]
            else:
                raise FileNotFoundError(
                    f"Subject {subject} does not have a brain image for run {run} (supposed to be at {file})."
                )
            if data is None:
                data = subject_data
            else:
                data += subject_data
        data /= n_subjects
        np.save(target_path / f"{run}.npy", data)

    Parallel(n_jobs=n_jobs)(delayed(aux)(run) for run in range(n_runs))
