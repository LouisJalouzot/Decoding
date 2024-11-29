import argparse
from hashlib import sha1

import hydra
import psutil
from omegaconf import DictConfig, OmegaConf

import wandb
from src.decoding import decoding
from src.utils import console, device, memory

wandb.Table.MAX_ARTIFACT_ROWS = 1000000000


def wandb_wrapper(cfg: dict, id_config: dict):

    if cfg["meta"]["log_wandb"]:
        wandb.init(
            config=id_config,
            id=sha1(repr(sorted(id_config.items())).encode()).hexdigest(),
            project=cfg["meta"]["wandb_project"],
            save_code=True,
            tags=cfg["meta"]["tags"],
            mode=cfg["meta"]["wandb_mode"],
        )

    num_cpus = psutil.cpu_count()
    ram = psutil.virtual_memory().total / (1024**3)
    console.log(
        f"Number of available CPUs: [green]{num_cpus}[/]\n"
        f"Available RAM: [green]{ram:.3g} GB[/]\n"
        f"Using device [green]{device}[/]"
    )

    output = decoding(**cfg)

    if wandb.run is not None:
        wandb.finish()

    return output


@hydra.main(version_base=None, config_path="src/config", config_name="config")
def main(
    cfg: DictConfig,
) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print(cfg)
    return
    # meta parameters have no impact on the results so they should be ignored for caching and for the wandb id
    id_config = {k: v for k, v in cfg.items() if k != "meta"}
    if cfg["meta"]["cache"]:
        return memory.cache(wandb_wrapper, ignore=["cfg"])(cfg, id_config)
    else:
        return wandb_wrapper(cfg, id_config)


if __name__ == "__main__":
    main()


# #!.env/bin/python
# import argparse
# import json

# from src.utils import memory
# from src.wandb_wrapper import wandb_wrapper

# parser = argparse.ArgumentParser()
# parser.add_argument("--datasets", nargs="*", default=["lebel2023"])
# parser.add_argument("--subjects", type=json.loads)
# parser.add_argument("--leave_out", type=json.loads)
# parser.add_argument("--fine_tune", type=json.loads)
# parser.add_argument("--fine_tune_disjoint", action="store_true")
# parser.add_argument(
#     "--multi_subject_mode",
#     type=str,
#     choices=["individual", "shared", "dataset"],
# )
# parser.add_argument("--model", type=str, default="bert-base-uncased")
# parser.add_argument("--decoder", type=str, default="brain_decoder")
# parser.add_argument("--loss", type=str, default="mixco")
# parser.add_argument("--train_ratio", type=float)
# parser.add_argument("--valid_ratio", type=float, default=0.05)
# parser.add_argument("--test_ratio", type=float, default=0.1)
# parser.add_argument("--n_folds", type=int)
# parser.add_argument("--fold", type=int)
# parser.add_argument("--context_length", type=int, default=3)
# parser.add_argument("--lag", type=int, default=2)
# parser.add_argument("--smooth", type=int, default=0)
# parser.add_argument("--stack", type=int, default=0)
# parser.add_argument("--dropout", type=float)
# parser.add_argument("--patience", type=int)
# parser.add_argument("--max_epochs", type=int)
# parser.add_argument("--lr", type=float)
# parser.add_argument("--lr_ft", type=float)
# parser.add_argument("--lr_ft_decoder", type=float)
# parser.add_argument("--weight_decay", type=float)
# parser.add_argument("--weight_decay_ft", type=float)
# parser.add_argument("--weight_decay_ft_decoder", type=float)
# parser.add_argument("--temperature_ft", type=float)
# parser.add_argument("--return_data", action="store_true")
# parser.add_argument("--batch_size", type=int)
# parser.add_argument("--latents_batch_size", type=int)
# parser.add_argument("--num_layers", type=int)
# parser.add_argument("--hidden_size", type=int)
# parser.add_argument("--hidden_size_projector", type=int)
# parser.add_argument("--n_res_blocks", type=int)
# parser.add_argument("--n_proj_blocks", type=int)
# parser.add_argument("--monitor", type=str)
# parser.add_argument(
#     "--top_encoding_voxels",
#     type=lambda x: int(x) if x.isdigit() else json.loads(x),
# )
# parser.add_argument(
#     "--token_aggregation",
#     type=str,
#     choices=[
#         "first",
#         "last",
#         "max",
#         "mean",
#         "chunk_mean",
#         "chunk_max",
#     ],
# )
# parser.add_argument("--no_wandb", action="store_true")
# parser.add_argument("--wandb_project", type=str)
# parser.add_argument("--tags", nargs="*", type=str)
# parser.add_argument("--cache", action="store_true")
# parser.add_argument("--force_rerun", action="store_true")
# parser.add_argument("--log_nlp_distances", action="store_true")
# parser.add_argument("--return_tables", action="store_true")
# parser.add_argument("--n_candidates", type=int)

# args = parser.parse_args()
# args = {key: value for key, value in vars(args).items() if value is not None}

# force_rerun = args.pop("force_rerun", False)
# if args.pop("cache"):
#     output = memory.cache(
#         wandb_wrapper,
#         ignore=["return_data", "log_nlp_distances"],
#         cache_validation_callback=lambda *args: not force_rerun,
#     )(**args)
# else:
#     wandb_wrapper(**args)
