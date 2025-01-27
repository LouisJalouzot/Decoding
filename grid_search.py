import sys
import traceback
import hydra
from omegaconf import DictConfig, OmegaConf


def grid_search(cfg):
    from src.decoding import decoding

    cfg = OmegaConf.to_container(cfg, resolve=True)

    return decoding(**cfg)


@hydra.main(version_base=None, config_path="configs", config_name="grid_search")
def main(cfg: DictConfig) -> None:
    # To circumvent a bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664
    try:
        output = grid_search(cfg)
        return output["test/mean_reciprocal_rank"]

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 0


if __name__ == "__main__":
    main()
