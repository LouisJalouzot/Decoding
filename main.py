#!.env/bin/python
import argparse

from src.train import train

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("--subject", type=str, default="UTS03", help="Subject name")
parser.add_argument(
    "--model",
    type=str,
    default="sentence-transformers/all-mpnet-base-v2",
    help="Model name",
)
parser.add_argument("--decoder", type=str, default="brain_decoder", help="Decoder name")
parser.add_argument("--loss", type=str, default="mixco")
parser.add_argument("--context_length", type=int, default=5, help="Context length")
parser.add_argument("--lag", type=int, default=3, help="Lag")
parser.add_argument("--smooth", type=int, default=0, help="Smooth")
parser.add_argument(
    "--subsample_voxels", type=int, default=None, help="Number of voxels to subsample"
)
parser.add_argument(
    "--hidden_size_backbone", type=int, default=512, help="Hidden size for backbone"
)
parser.add_argument(
    "--hidden_size_projector", type=int, default=512, help="Hidden size for projector"
)
parser.add_argument("--dropout", type=float, default=0.7, help="Dropout rate")
parser.add_argument(
    "--n_res_blocks", type=int, default=2, help="Number of residual blocks"
)
parser.add_argument(
    "--n_proj_blocks", type=int, default=1, help="Number of projection blocks"
)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--temperature", type=float, default=0.05, help="Temperature")
parser.add_argument("--verbose", type=bool, default=True)

args = parser.parse_args()

train(**vars(args))
