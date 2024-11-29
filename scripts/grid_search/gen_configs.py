from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)


def get_tags(top_encoding_voxels=None, **kwargs):
    tags = ["--tags"]
    if top_encoding_voxels is not None:
        tags.append(f"'n_voxels={top_encoding_voxels // 1000}k'")
    # Iterate over kwargs
    for k, v in kwargs.items():
        if v is not None:
            tags.append(f"'{k}={v}'")

    return " ".join(tags)


def write_config(*args, **kwargs):
    args = list(args)
    for k, v in kwargs.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k}")
            else:
                args.append(f"--{k} {v}")

    return " ".join(args) + "\n"


n_configs = 0
default_config = (
    "--project_name fMRI-Decoding-grid-search "
    '--subjects \'{{"lebel2023": ["UTS01", "UTS02", "UTS03"]}}\' '
    "--top_encoding_voxels 3000 "
)
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
