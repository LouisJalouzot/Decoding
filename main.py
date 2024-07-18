#!.env/bin/python
import argparse

from src.main import main

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", default="lebel2023/all_subjects")
parser.add_argument("--subjects", nargs="*")
parser.add_argument("--multi_subject_mode", type=str, choices=["individual", "shared"])
parser.add_argument("--model", type=str, default="bert-base-uncased")
parser.add_argument("--decoder", type=str, default="brain_decoder")
parser.add_argument("--loss", type=str, default="mixco")
parser.add_argument("--context_length", type=int, default=6)
parser.add_argument("--lag", type=int, default=0)
parser.add_argument("--smooth", type=int, default=0)
parser.add_argument("--subsample_voxels", type=int)
parser.add_argument("--dropout", type=float, default=0.7)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--temperature", type=float, default=0.05)
parser.add_argument("--latents_batch_size", type=int)
parser.add_argument("--num_layers", type=int)
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--hidden_size_projector", type=int)
parser.add_argument("--n_res_blocks", type=int)
parser.add_argument("--n_proj_blocks", type=int)
parser.add_argument("--scale_across_runs", type=bool)
parser.add_argument("--scale_across_subjects", type=bool)

args = parser.parse_args()
args = {key: value for key, value in vars(args).items() if value is not None}
main(**args)
