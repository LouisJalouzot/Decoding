from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

n_configs = 0
default_config = (
    f"--wandb_project 'fMRI-Decoding-wrap-up' "
    f"--return_tables "
    f"--cache "
    f"--datasets lebel2023_balanced "
    f"--top_encoding_voxels 3000 "
)
with config_path.open("w") as config_file:
    for i in range(1, 9):
        for top_encoding_voxels in [
            " --top_encoding_voxels 80000",
            " 'Reliable voxels' --top_encoding_voxels 3000",
        ]:
            subject = "UTS0" + str(i)
            setup = (
                f'--subjects \'{{"lebel2023_balanced": ["{subject}"]}}\' '
                f"--tags 'Single subject' {subject} Balanced"
            )
            if i == 3:
                setup += " 'Multi subject' Combinations"
            setup += top_encoding_voxels
            config_file.write(default_config + setup + "\n")
            n_configs += 1

print("Number of configurations:", n_configs)
