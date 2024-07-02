#!.env/bin/python
import argparse

from src.main import main

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("--subjects", nargs="*", default="UTS03", help="Subject name")
parser.add_argument("--model", type=str, default="bert-base-uncased")
parser.add_argument("--decoder", type=str, default="brain_decoder", help="Decoder name")
parser.add_argument("--loss", type=str, default="mixco")
parser.add_argument("--context_length", type=int, default=6, help="Context length")
parser.add_argument("--lag", type=int, default=0, help="Lag")
parser.add_argument("--smooth", type=int, default=0, help="Smooth")
parser.add_argument("--subsample_voxels", type=int)
parser.add_argument("--dropout", type=float, default=0.7, help="Dropout rate")
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--temperature", type=float, default=0.05, help="Temperature")
parser.add_argument("--latents_batch_size", type=int)
parser.add_argument("--num_layers", type=int)
parser.add_argument("--hidden_size_backbone", type=int)
parser.add_argument("--hidden_size_projector", type=int)
parser.add_argument("--n_res_blocks", type=int)
parser.add_argument("--n_proj_blocks", type=int)

args = parser.parse_args()
args = {key: value for key, value in vars(args).items() if value is not None}
main(**args)
