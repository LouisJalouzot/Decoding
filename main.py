#!.env/bin/python
import argparse
import json

from src.wandb_wrapper import wandb_wrapper

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="*", default=["lebel2023"])
parser.add_argument("--subjects", type=json.loads)
parser.add_argument(
    "--multi_subject_mode",
    type=str,
    choices=["individual", "shared", "dataset"],
)
parser.add_argument("--model", type=str, default="bert-base-uncased")
parser.add_argument("--decoder", type=str, default="brain_decoder")
parser.add_argument("--loss", type=str, default="mixco")
parser.add_argument("--valid_ratio", type=float, default=0.06)
parser.add_argument("--test_ratio", type=float, default=0.06)
parser.add_argument("--context_length", type=int, default=3)
parser.add_argument("--lag", type=int, default=2)
parser.add_argument("--smooth", type=int, default=0)
parser.add_argument("--stack", type=int, default=0)
parser.add_argument("--dropout", type=float)
parser.add_argument("--patience", type=int)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--latents_batch_size", type=int)
parser.add_argument("--num_layers", type=int)
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--hidden_size_projector", type=int)
parser.add_argument("--n_res_blocks", type=int)
parser.add_argument("--n_proj_blocks", type=int)
parser.add_argument("--monitor", type=str)
parser.add_argument(
    "--top_encoding_voxels",
    type=lambda x: int(x) if x.isdigit() else json.loads(x),
)
parser.add_argument(
    "--token_aggregation",
    type=str,
    choices=[
        "first",
        "last",
        "max",
        "mean",
        "chunk_mean",
        "chunk_max",
    ],
)
parser.add_argument("--wandb_mode", type=str)
parser.add_argument("--tags", nargs="*", type=str)
parser.add_argument("--cache", action="store_true")
parser.add_argument("--log_nlp_distances", action="store_true")
parser.add_argument("--return_tables", action="store_true")
parser.add_argument("--n_candidates", type=int)

args = parser.parse_args()
args = {key: value for key, value in vars(args).items() if value is not None}
wandb_wrapper(**args)
