from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

n_configs = 0
default_config = (
    f"--wandb_project 'fMRI-Decoding-wrap-up' "
    f'--subjects \'{{"lebel2023": ["UTS01", "UTS02", "UTS03"]}}\' '
    f"--valid_ratio 0.04 "
    f"--test_ratio 0.09 "
    f"--return_tables "
    f"--cache "
    f"--lag 2 "
    f"--context_length 3 "
    f"--smooth 0 "
    f"--stack 0 "
    f"--tags UTS0123 Setup "
)
setups = [
    "'Mean baseline' --max_epochs 0 --decoder mean_decoder",
    "'Random baseline' --max_epochs 0 --decoder random_decoder",
]
with config_path.open("w") as config_file:
    for setup in setups:
        config_file.write(default_config + setup + "\n")
        n_configs += 1
    for decoder in ["simple_mlp", "brain_decoder"]:
        for loss in ["mse", "symm_nce", "mixco"]:
            for top_encoding_voxels in [" --top_encoding_voxels 3000", ""]:
                tag = "Brain Decoder" if decoder == "brain_decoder" else "MLP"
                tag += (
                    ""
                    if loss == "mse"
                    else " NCE" if loss == "symm_nce" else " MixCo"
                )
                tag += " reliable voxels" if top_encoding_voxels else ""
                setup = f"'{tag}' --decoder {decoder} --loss {loss} --monitor valid/{loss}{top_encoding_voxels}"
                config_file.write(default_config + setup + "\n")
                n_configs += 1

print("Number of configurations:", n_configs)
