import shutil
from pathlib import Path
from bids import BIDSLayout
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img
from tqdm.auto import tqdm
from scipy.io import loadmat
import numpy as np
import pandas as pd

TR = 0.71


def process_bold_file(bold_file, subject, run, mask, dataset_path):
    path = dataset_path / subject / (str(run) + ".npy")
    if path.exists():
        return

    events = f"data/SMN4Lang/sub-{subject}/func/sub-{subject}_task-RDR_run-{run}_events.tsv"
    events = pd.read_csv(events, sep="\t")
    events = events[events.stim_file.str.startswith("audio")]
    assert len(events) == 1
    onset, duration, _ = events.iloc[0]
    start = int(onset / TR)
    end = int((onset + duration) / TR)

    scaler = StandardScaler(copy=False)
    a = bold_file.get_fdata()[mask, start:end]
    a = scaler.fit_transform(a.T)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, a)


def create_smn4lang_dataset():
    dataset_path = Path("datasets/smn4lang")
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

    print("Creating SMN4Lang dataset")
    print("Indexing BIDS dataset")
    layout = BIDSLayout(
        "data/SMN4Lang/derivatives/preprocessed_data", validate=False
    )
    mask = load_mni152_brain_mask(resolution=2, threshold=0.75)
    bold_files = layout.get(suffix="bold", extension="nii.gz", task="RDR")
    mask = resample_to_img(
        mask,
        bold_files[0].get_image(),
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    mask = mask.get_fdata() == 1

    Parallel(n_jobs=12)(
        delayed(process_bold_file)(
            f.get_image(),
            f.entities["subject"],
            f.entities["run"],
            mask,
            dataset_path,
        )
        for f in tqdm(bold_files, desc="Processing BOLD files")
    )


def create_smn4lang_textgrids():
    annotation_files = list(
        Path(
            "data/SMN4Lang/derivatives/annotations/time_align/word-level"
        ).glob("*.mat")
    )
    for file in tqdm(annotation_files):
        # Load .mat file
        data = loadmat(file)
        starts = data["start"].flatten() - 10.65
        ends = data["end"].flatten() - 10.65
        words = data["word"].flatten()

        # Calculate total duration
        total_duration = ends[-1]

        # Create TextGrid header
        textgrid = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            f"xmin = 0",
            f"xmax = {total_duration}",
            "tiers? <exists>",
            "size = 1",
            "item []:",
            "    item [1]:",
            '        class = "IntervalTier"',
            '        name = "text words"',
            f"        xmin = 0",
            f"        xmax = {total_duration}",
            f"        intervals: size = {len(words)}",
        ]

        # Add intervals
        for i in range(len(words)):
            interval = [
                f"        intervals [{i+1}]:",
                f"            xmin = {starts[i]}",
                f"            xmax = {ends[i]}",
                f'            text = "{words[i].strip()}"',
            ]
            textgrid.extend(interval)

        # # Write to file
        with open(file.with_suffix(".TexGrid"), "w", encoding="utf-8") as f:
            f.write("\n".join(textgrid))
