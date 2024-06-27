from collections import defaultdict
from typing import List

import numpy as np
import torch

from src.brain_decoder_multi_subject import train_brain_decoder_multi_subject
from src.fetch_data import fetch_data
from src.utils import console, memory


@memory.cache
def train_multi_subject(
    subjects: List[str] = ["UTS00"],
    decoder: str = "brain_decoder",
    model: str = "bert-base-uncased",
    context_length: int = 6,
    tr: int = 2,
    lag: int = 3,
    smooth: int = 0,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 0,
    subsample_voxels: int = None,
    latents_batch_size: int = 64,
    **decoder_params,
) -> dict:
    """

    Trains the model.

    Args:
        subject (str, optional): The subject identifier. Defaults to "UTS00".
        decoder (str, optional): The decoder type. Defaults to "ridge".
        model (str, optional): The model to use. Defaults to "clip".
        context_length (int, optional): The context length. Defaults to 2.
        tr (int, optional): The tr value. Defaults to 2.
        valid_ratio (float, optional): The validation ratio. Defaults to 0.2.
        test_ratio (float, optional): The test ratio. Defaults to 0.1.
        seed (int, optional): The random seed. Defaults to 0.
        alphas (Union[List[float], np.ndarray], optional): The alpha values. Defaults to np.logspace(-3, 10, 10).
        verbose (bool, optional): Whether to display progress. Defaults to False.
        n_jobs (int, optional): The number of jobs. Defaults to -2.

    Returns:
        Union[dict, Tuple[dict, RidgeCV, RobustScaler]]: The training results.

    """
    np.random.seed(seed)
    data = defaultdict([])
    for subject in subjects:
        console.log("Fetching data for", subject)
        d = fetch_data(
            subject=subject,
            model=model,
            tr=tr,
            context_length=context_length,
            subsample_voxels=subsample_voxels,
            smooth=smooth,
            lag=lag,
            batch_size=latents_batch_size,
        )
        for story, (X, Y) in d.items():
            data[story].append((X, Y))
    stories = sorted(data.keys(), key=lambda x: len(data[x]))
    n_stories = len(stories)
    n_valid = max(1, int(valid_ratio * n_stories))
    n_test = max(1, int(test_ratio * n_stories))
    stories_train = stories[n_test + n_valid :]
    stories_valid = stories[n_test : n_valid + n_test]
    stories_test = stories[:n_test]

    for split, stories_list in [
        ("train", stories_train),
        ("valid", stories_valid),
        ("test", stories_test),
    ]:
        console.log(
            f"{len(stories_list)} {split} stories corresponding to {sum(len(data[story]) for story in stories_list)} runs",
        )

        output = train_brain_decoder_multi_subject(
            data,
            stories_train,
            stories_valid,
            stories_test,
            decoder=decoder,
            seed=seed,
            **decoder_params,
        )

    torch.cuda.empty_cache()

    console.log(
        f"Train relative median rank: {output['train/relative_median_rank']:.3f} "
        f"(size {int(output['train/size'])})\n"
        f"Test relative median rank: {output['test/relative_median_rank']:.3f} "
        f"(size {int(output['test/size'])})"
    )

    return output
