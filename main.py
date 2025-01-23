import sys
import traceback
import hydra
from omegaconf import DictConfig


def wandb_wrapper(cfg: dict, id_config: dict):
    # Do the imports here instead of globally for a quicker startup time of main on frontal nodes
    from src.decoding import decoding
    import wandb
    from hashlib import sha1

    wandb.Table.MAX_ARTIFACT_ROWS = 1000000000

    if cfg["meta"]["log_wandb"]:
        wandb.init(
            config=id_config,
            id=sha1(repr(sorted(id_config.items())).encode()).hexdigest(),
            project=cfg["meta"]["wandb_project"],
            save_code=True,
            tags=cfg["meta"]["tags"],
            mode=cfg["meta"]["wandb_mode"],
        )

    outputs = decoding(**cfg)

    if wandb.run is not None:
        wandb.finish()

    return outputs


def cache_wrapper(cfg: DictConfig):
    from omegaconf import OmegaConf
    from src.utils import memory

    cfg = OmegaConf.to_container(cfg, resolve=True)
    # meta parameters have no impact on the results so they should be ignored for caching and for the wandb id
    id_config = {k: v for k, v in cfg.items() if k != "meta"}

    if cfg["meta"]["cache"]:
        return memory.cache(wandb_wrapper, ignore=["cfg"])(cfg, id_config)
    else:
        return wandb_wrapper(cfg, id_config)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # To circumvent a bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664
    try:
        return cache_wrapper(cfg)

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
