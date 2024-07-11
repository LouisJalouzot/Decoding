import os
import shutil
from pathlib import Path

path = Path("data/lebel2023")
assert (
    path.exists()
), f"{path} does not exist, either the working directory {os.getcwd()} is not the root of the repo or the data has not been downloaded."


def create_symlink(input, target):
    input.symlink_to(target.relative_to(input, walk_up=True))


if __name__ == "__main__":
    data_dir = path / "derivative" / "preprocessed_data"
    subjects = sorted(os.listdir(data_dir))
    ### Building "all_subjects" directory
    target_dir = path / "all_subjects"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    for subject in subjects:
        (target_dir / subject).mkdir(parents=True, exist_ok=True)
        for run in os.listdir(data_dir / subject):
            create_symlink(target_dir / subject / run, data_dir / subject / run)
            run = run.replace(".hf5", ".TextGrid")
            create_symlink(
                target_dir / subject / run, data_dir.parent / "TextGrids" / run
            )
            run = run.replace(".TextGrid", ".wav")
            create_symlink(
                target_dir / subject / run, data_dir.parent.parent / "stimuli" / run
            )
    ### Building "3_subjects" directory
    target_dir_3 = path / "3_subjects"
    if target_dir_3.exists():
        shutil.rmtree(target_dir_3)
    target_dir_3.mkdir(parents=True, exist_ok=True)
    for subject in subjects[:3]:
        create_symlink(target_dir_3 / subject, target_dir / subject)
