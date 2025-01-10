import os
import shutil
from pathlib import Path
import h5py
import numpy as np
from bids import BIDSLayout
from joblib import Parallel, delayed
from nilearn.decomposition import CanICA
from sklearn.preprocessing import StandardScaler
from nilearn.plotting import plot_prob_atlas
from nilearn.masking import intersect_masks
from nilearn import image
from tqdm.auto import tqdm


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
    print(f"Found {len(subjects)} subjects.")
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
        runs = Parallel(n_jobs=-1)(
            delayed(read)(f)
            for f in tqdm(
                run_files,
                desc=f"Reading brain scans for subject {subject}",
                leave=False,
            )
        )
        scaler = StandardScaler()
        for _, X in tqdm(
            runs, desc=f"Scaling brain scans for subject {subject}", leave=False
        ):
            scaler.partial_fit(X)
        target_path_subject.mkdir(parents=True, exist_ok=True)
        for run, X in tqdm(
            runs, desc=f"Saving brain scans for subject {subject}", leave=False
        ):
            X = scaler.transform(X)
            np.save(target_path_subject / f"{run}.npy", X.astype(np.float32))


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

    print("Creating Lebel2023 fMRIprep dataset")
    print("Indexing BIDS dataset")
    layout = BIDSLayout("data/lebel2023/derivatives/fmriprep", validate=False)
    mask_files = layout.get(
        suffix="mask", extension="nii.gz", space="MNI152NLin2009cAsym"
    )
    mask_files = [f for f in mask_files if "task" in f.filename]
    mask_files = [f for f in mask_files if not "Localizer" in f.filename]
    mask_files = [f for f in mask_files if not "wheretheressmoke" in f.filename]
    mask = None
    for f in tqdm(mask_files, desc="Combining masks"):
        run_mask = f.get_image().get_fdata()
        if mask is None:
            mask = run_mask == 1
        else:
            mask &= run_mask == 1
    print(f"{mask.sum()} voxels in the combined mask")
    bold_files = layout.get(suffix="bold", extension="nii.gz")
    bold_files = [f for f in bold_files if not "Localizer" in f.filename]
    bold_files = [f for f in bold_files if not "wheretheressmoke" in f.filename]

    Parallel(n_jobs=-1)(
        delayed(process_bold_file)(
            f.get_image(),
            f.entities["subject"],
            f.entities["task"],
            mask,
            dataset_path,
        )
        for f in tqdm(bold_files, desc="Processing BOLD files")
    )


def create_lebel2023_fmripep_canica_dataset(
    n_components: int = 768,
    per_subject: bool = False,
    input_path: str = "data/lebel2023/derivatives/fmriprep",
):
    input_path = Path(input_path)

    dataset_path = f"datasets/lebel2023_fmriprep_balanced_canica_{n_components}"
    if per_subject:
        dataset_path += "_indiv"
    dataset_path = Path(dataset_path)
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

    print("Creating Lebel2023 fMRIprep CanICA dataset")
    print("Indexing BIDS dataset")
    layout = BIDSLayout(input_path, validate=False)

    mask_file = input_path / "combined_mask.nii.gz"
    if mask_file.exists():
        mask = image.load_img(mask_file)
    else:
        mask_files = layout.get(
            suffix="mask", extension="nii.gz", space="MNI152NLin2009cAsym"
        )
        mask_files = [f for f in mask_files if "task" in f.filename]
        mask_files = [f for f in mask_files if not "Localizer" in f.filename]
        mask_files = [
            f for f in mask_files if not "wheretheressmoke" in f.filename
        ]
        mask = intersect_masks(mask_files, threshold=1, connected=False)
        mask.to_filename(mask_file)

    bold_files = layout.get(
        suffix="bold", extension="nii.gz", space="MNI152NLin2009cAsym"
    )
    bold_files = [f for f in bold_files if not "Localizer" in f.filename]
    bold_files = [f for f in bold_files if not "wheretheressmoke" in f.filename]
    bold_files = np.random.choice(bold_files, 5)  # TODO: remove

    subjects = sorted(layout.get_subjects())
    tasks = sorted(layout.get_tasks())
    tasks = [
        t for t in tasks if "wheretheressmoke" not in t and "Localizer" not in t
    ]
    print(f"Found {len(subjects)} subjects and {len(tasks)} tasks.")

    canica = CanICA(
        n_components=n_components,
        t_r=2,
        n_jobs=-1,
        verbose=1,
        mask=mask,
        random_state=0,
        memory=".cache",
        memory_level=2,
    )

    if per_subject:
        data = [
            (
                "_" + subject,
                [f for f in bold_files if f.entities["subject"] == subject],
            )
            for subject in subjects
        ]
    else:
        data = [("", bold_files)]

    for name, files in tqdm(data, desc="Computing CanICA"):
        if len(files) == 0:
            continue
        projected_imgs = canica.fit_transform([f.get_image() for f in files])

        components_img = canica.components_img_
        dataset_path.mkdir(parents=True, exist_ok=True)
        components_img.to_filename(
            dataset_path / f"canica_components{name}.nii.gz"
        )
        plot_prob_atlas(components_img, title="All ICA components").savefig(
            dataset_path / f"canica_components{name}.png"
        )

        scaler = StandardScaler(copy=False)
        description = "Scaling and saving images"
        if name:
            description += f" for {name}"
        for img, f in (
            pbar := tqdm(list(zip(projected_imgs, files)), desc=description)
        ):
            # Drop first and last 10 volumes like original dataset
            img = scaler.fit_transform(img[10:-10])
            target_path = dataset_path / f.entities["subject"]
            target_path.mkdir(parents=True, exist_ok=True)
            target_file = target_path / f.entities["task"]
            np.save(target_file, img)
            pbar.set_postfix(
                {
                    "Subject": f.entities["subject"],
                    "Task": f.entities["task"],
                    "Shape": img.shape,
                }
            )
