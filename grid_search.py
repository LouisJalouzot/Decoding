import hydra
import numpy as np
import psutil
from omegaconf import DictConfig, OmegaConf

from src.decoding import decoding
from src.utils import console, device


@hydra.main(version_base=None, config_path="configs", config_name="grid_search")
def main(
    cfg: DictConfig,
) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg["latents_cfg"]["model"] == "llm2vec":
        if not cfg["latents_cfg"]["token_aggregation"] == "mean":
            console.log(
                "llm2vec requires token_aggregation to be 'mean'. Skipping this configuration."
            )
            return np.inf

    num_cpus = psutil.cpu_count()
    ram = psutil.virtual_memory().total / (1024**3)
    console.log(
        f"Number of available CPUs: [green]{num_cpus}[/]\n"
        f"Available RAM: [green]{ram:.3g} GB[/]\n"
        f"Using device [green]{device}[/]"
    )

    output = decoding(**cfg)

    return output["test/relative_rank_median"]


if __name__ == "__main__":
    main()
