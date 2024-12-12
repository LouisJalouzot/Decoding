from hashlib import sha1

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from src.decoding import decoding
from src.utils import memory

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

    output = decoding(**cfg)

    if wandb.run is not None:
        wandb.finish()

    return output


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(
    cfg: DictConfig,
) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # meta parameters have no impact on the results so they should be ignored for caching and for the wandb id
    id_config = {k: v for k, v in cfg.items() if k != "meta"}
    if cfg["meta"]["cache"]:
        return memory.cache(wandb_wrapper, ignore=["cfg"])(cfg, id_config)
    else:
        return wandb_wrapper(cfg, id_config)


if __name__ == "__main__":
    main()
