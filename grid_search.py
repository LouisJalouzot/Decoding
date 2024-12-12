import hydra
from omegaconf import DictConfig, OmegaConf

from src.decoding import decoding


@hydra.main(version_base=None, config_path="configs", config_name="grid_search")
def main(cfg: DictConfig) -> None:
    from src.utils import console

    console.quiet = True
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg["latents_cfg"]["model"] == "llm2vec":
        if not cfg["latents_cfg"]["token_aggregation"] == "mean":
            return 1

    output = decoding(**cfg)

    return output["test/relative_rank_median"]


if __name__ == "__main__":
    main()
