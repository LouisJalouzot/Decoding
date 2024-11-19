from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

n_configs = 0
default_config = "--wandb_project fMRI-Decoding-wrap-up --datasets lebel2023_balanced --return_tables --cache "
with config_path.open("w") as config_file:
    for mode in ["shared", "individual"]:
        for top_encoding_voxels in [
            " --top_encoding_voxels 80000",
            " 'Reliable voxels' --top_encoding_voxels 3000",
        ]:
            config_file.write(
                default_config
                + f"--multi_subject_mode {mode} --tags 'Multi subjects' Balanced {mode}{top_encoding_voxels}"
                + "\n"
            )
            n_configs += 1

print("Number of configurations:", n_configs)
