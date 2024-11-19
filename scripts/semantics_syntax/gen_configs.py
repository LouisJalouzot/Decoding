from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)

n_configs = 0
default_config = (
    "--wandb_project 'fMRI-Decoding-wrap-up' "
    '--subjects \'{"lebel2023": ["UTS01", "UTS02", "UTS03"]}\' '
    "--valid_ratio 0.04 "
    "--test_ratio 0.09 "
    "--log_nlp_distances "
    "--return_tables "
    "--cache "
    "--n_candidates 100 "
    "--context_length 4 "
    "--tags UTS0123 'Context length 4' 'Semantics Syntax' "
)
setups = [
    "",
    "'Random baseline' --decoder random_decoder --max_epochs 0",
    "'Mean baseline' --decoder mean_decoder --max_epochs 0",
]
with config_path.open("w") as config_file:
    for setup in setups:
        config_file.write(default_config + setup + "\n")
        n_configs += 1

print("Number of configurations:", n_configs)
