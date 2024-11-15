from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

n_configs = 0
default_config = (
    f"--wandb_project 'fMRI-Decoding-wrap-up' "
    f'--subjects \'{{"lebel2023": ["UTS01", "UTS02", "UTS03"]}}\' '
    f"--valid_ratio 0.04 --test_ratio 0.09 --return_tables --cache "
    f"--top_encoding_voxels 3000 "
    f"--tags 'Grid search' UTS0123 "
)
param_names = ["context_length", "lag", "smooth", "stack"]
params = [
    ("context_length", range(1, 7)),
    ("lag", range(1, 7)),
    ("smooth", range(1, 7)),
    ("stack", range(1, 5)),
]
with config_path.open("w") as config_file:
    config_file.write(
        default_config
        + " ".join(param_names)
        + "".join(f" --{p} 0 " for p in param_names)
        + "\n"
    )
    n_configs += 1

    for param, values in params:
        for value in values:
            config_file.write(
                default_config
                + param
                + "".join(f" --{p} 0 " for p in param_names if p != param)
                + f"--{param} {value} \n"
            )
            n_configs += 1

print("Number of configurations:", n_configs)
