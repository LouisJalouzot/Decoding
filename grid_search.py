import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.decoding import decoding


@hydra.main(version_base=None, config_path="configs", config_name="grid_search")
def main(cfg: DictConfig) -> None:
    # from src.utils import console

    # console.quiet = True
    cfg = OmegaConf.to_container(cfg, resolve=True)

    output = decoding(**cfg)

    torch.cuda.empty_cache()

    return output["test/relative_rank_median"]


if __name__ == "__main__":
    main()
