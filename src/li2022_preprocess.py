import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray
from joblib import Parallel, delayed
from nilearn import image
from tqdm.auto import tqdm

from src.utils import console, create_symlink

# TODO remove first and last 10s of each run
n_runs = 9
path = Path("data/li2022")
assert (
    path.exists()
), f"{path} does not exist, either the working directory {Path(os.getcwd())} is not the root of the repo or the data has not been downloaded."


def text_file(lang, run):
    return path / "annotation" / lang / f"lpp{lang}_section{run+1}.TextGrid"


def audio_file(lang, run):
    return path / "stimuli" / f"task-lpp{lang}_section-{run+1}.wav"


def create_symlinks_li2022(lang="EN"):
    data_dir = path / "derivatives"
    subjects = [f for f in sorted(os.listdir(data_dir)) if lang in f]
    ### Building "all_lang" directory
    target_dir = path / f"all_{lang}_raw"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    for subject in subjects:
        sub_target_dir = target_dir / subject.replace("sub-", "")
        sub_target_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(os.listdir(data_dir / subject / "func"))
        for run, file in enumerate(files):
            run_file = f"{run}.nii.gz"
            create_symlink(
                sub_target_dir / run_file, data_dir / subject / "func" / file
            )
            run_file = run_file.replace(".nii.gz", ".TextGrid")
            create_symlink(sub_target_dir / run_file, text_file(lang, run))
            run_file = run_file.replace(".TextGrid", ".wav")
            create_symlink(sub_target_dir / run_file, audio_file(lang, run))


def resample_and_slice_brain(lang="EN", n_jobs=-1):
    mask_path = path / "colin27_t1_tal_lin_mask.nii"
    assert mask_path.exists(), "Need to download the mask first."
    input_path = path / f"all_{lang}_raw"
    target_path = path / f"all_{lang}"
    subjects = sorted(os.listdir(input_path))
    n_subjects = len(subjects)
    console.log(f"Resampling and slicing {n_subjects} subjects for {lang}.")
    input("Subjects found are: " + ", ".join(subjects) + ". Continue?")

    example_lpp_img = nib.load(input_path / subjects[0] / "0.nii.gz")
    acquisition_affine = example_lpp_img.affine.copy()
    acquisition_affine[np.arange(3), np.arange(3)] = 3.75
    example_lpp_img = image.resample_img(
        example_lpp_img, target_affine=acquisition_affine
    )
    mask = nib.load(mask_path)
    mask = image.resample_to_img(
        mask,
        example_lpp_img,
        interpolation="nearest",
    )
    mask = np.where(mask.get_fdata().astype(bool))

    def aux(subject, run):
        target_file = target_path / subject / f"{run}.npy"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_text = target_path / subject / f"{run}.TextGrid"
        if not target_text.exists():
            create_symlink(target_text, text_file(lang, run))
        target_audio = target_path / subject / f"{run}.wav"
        if not target_audio.exists():
            create_symlink(target_audio, audio_file(lang, run))
        input_file = input_path / subject / f"{run}.nii.gz"
        if not input_file.exists():
            raise FileNotFoundError(
                f"File {input_file} does not exist for subject {subject} and run {run}."
            )
        if target_file.exists():
            return
        img = image.resample_img(nib.load(input_file), target_affine=acquisition_affine)
        np.save(target_file, img.get_fdata()[mask].T)

    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(aux)(subject, run) for subject in subjects for run in range(n_runs)
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
