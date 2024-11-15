from pathlib import Path

import numpy as np

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

range = np.logspace(np.log10(500), np.log10(80000), num=15, dtype=int)
range = np.unique(
    [round(voxel, -int(np.floor(np.log10(voxel)))) for voxel in range]
)

n_configs = 0
with config_path.open("w") as config_file:
    for top_encoding_voxels in range:
        config_file.write(
            f"--wandb_project 'fMRI-Decoding-wrap-up' "
            f'--subjects \'{{"lebel2023": ["UTS01", "UTS02", "UTS03"]}}\' '
            f"--valid_ratio 0.04 --test_ratio 0.09 --return_tables --cache --tags 'Number of voxels' UTS0123 "
            f"--top_encoding_voxels {top_encoding_voxels} --lag 2 --context_length 3 --smooth 0 --stack 0\n"
        )
        n_configs += 1

print("Number of configurations:", n_configs)
