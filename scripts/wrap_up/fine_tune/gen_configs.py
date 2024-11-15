from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

n_configs = 0
default_config = (
    "--wandb_project 'fMRI-Decoding-wrap-up' "
    "--return_tables "
    "--cache "
    "--tags 'Fine tune'"
)
subjects = ' --subjects \'{"lebel2023": ["UTS01", "UTS02", "UTS03", "'
with config_path.open("w") as config_file:
    for i in range(4, 9):
        for fine_tune_disjoint in [""]:
            for top_encoding_voxels in [
                " '3k voxels' --top_encoding_voxels 3000",
            ]:
                for multi_subject_mode in ["individual"]:
                    setup = (
                        f"{' Disjoint' if fine_tune_disjoint else ''} {multi_subject_mode} S{i}"
                        + top_encoding_voxels
                        + subjects
                        + f"UTS0{i}\"]}}'"
                        + f' --fine_tune \'{{"lebel2023": ["UTS0{i}"]}}\''
                        + f" --multi_subject_mode {multi_subject_mode}"
                    )
                    config_file.write(default_config + setup + "\n")
                    n_configs += 1

print("Number of configurations:", n_configs)
