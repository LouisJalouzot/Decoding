from itertools import combinations
from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

n_configs = 0
default_config = (
    f"--wandb_project 'fMRI-Decoding-wrap-up' "
    f"--datasets lebel2023_balanced "
    f"--top_encoding_voxels 3000 "
)
with config_path.open("w") as config_file:
    subjects = [i for i in range(1, 9) if i != 3]
    for r in range(1, len(subjects) + 1):
        for setup_subjects in combinations(subjects, r):
            setup_subjects = sorted([*setup_subjects, 3])
            subjects_str = ", ".join(f'"UTS0{j}"' for j in setup_subjects)
            setup = (
                f"--tags Combinations Balanced "
                f"--subjects '{{\"lebel2023_balanced\": [{subjects_str}]}}'"
            )
            config_file.write(default_config + setup + "\n")
            n_configs += 1

print("Number of configurations:", n_configs)
