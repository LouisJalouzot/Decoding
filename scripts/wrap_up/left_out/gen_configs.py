from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

n_configs = 0
default_config = (
    "--wandb_project 'fMRI-Decoding-wrap-up' "
    "--return_tables "
    "--cache "
    "--datasets lebel2023_balanced "
    "--tags 'Left out'"
)
with config_path.open("w") as config_file:
    for i in range(1, 9):
        leave_out = f' --leave_out \'{{"lebel2023_balanced": ["UTS0{i}"]}}\' '
        for top_encoding_voxels in [
            " '80k voxels' --top_encoding_voxels 80000",
            " '3k voxels' --top_encoding_voxels 3000",
        ]:
            for multi_subject_mode in ["individual", "shared"]:
                setup = (
                    f" {multi_subject_mode} S{i}"
                    + top_encoding_voxels
                    + leave_out
                    + f"--multi_subject_mode {multi_subject_mode}"
                )
                config_file.write(default_config + setup + "\n")
                n_configs += 1

print("Number of configurations:", n_configs)
