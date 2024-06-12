from typing import Callable, List, Tuple

import clip
import numpy as np

from src.utils import device, get_textgrid, memory


def get_model(
    model: str,
) -> Callable[[str], np.ndarray]:
    """
    Get the model for encoding text.

    Args:
        model (str): Which model to use. Either "clip" or a model name from Hugging Face.

    Returns:
        Callable[[str], np.ndarray]: The model function that takes a text input and returns the encoded features.
    """
    import torch

    if model.lower() == "clip":
        clip_model, _ = clip.load("ViT-L/14", device=device)

        def model(text: str) -> np.ndarray:
            with torch.no_grad():
                return clip_model.encode_text(
                    clip.tokenize(text, truncate=True).to(device),
                )

        return model
    else:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        auto_model = AutoModel.from_pretrained(model).to(device)
        auto_model.eval()

        def model(text: str) -> np.ndarray:
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            with torch.no_grad():
                return auto_model(**inputs).last_hidden_state.mean(dim=1)

        return model


def compute_chunks(textgrid_path: str, tr: int, context_length: int) -> List[str]:
    """
    Compute the chunks of text from the given textgrid file.

    Args:
        textgrid_path (str): The path to the textgrid file.
        tr (int): The time resolution.
        context_length (int): The number of previous chunks to include for context.

    Returns:
        List[str]: The list of computed chunks of text.
    """
    goodtranscript = get_textgrid(textgrid_path)

    offsets = np.array([x[1] for x in goodtranscript])
    words = [x[2].strip("{}").strip() for x in goodtranscript]
    words = np.array([(x if x == "I" else x.lower()) for x in words])
    group_indices = np.array(offsets // tr, dtype=int)
    unique_indices = np.arange(group_indices.max())

    chunks = [" ".join(list(words[group_indices == idx])) for idx in unique_indices]
    return [
        " ".join([chunks[k] for k in range(max(0, i - context_length), i + 1)]).strip()
        for i in range(len(chunks))
    ]


@memory.cache
def prepare_latents(
    textgrid_path: str,
    model: str = "clip",
    tr: int = 2,
    context_length: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare the latents for the given textgrid file.

    Args:
        textgrid_path (str): The path to the textgrid file.
        model_class (str, optional): The class of the model. Defaults to "clip".
        tr (int, optional): The time resolution. Defaults to 2.
        context_length (int, optional): The number of previous chunks to include for context. Defaults to 0.
        verbose (bool, optional): Whether to display progress. Defaults to False.

    Returns:
        Tuple[np.ndarray, List[str]]: The encoded features and the aggregated chunks of text.
    """
    chunks = compute_chunks(textgrid_path, tr, context_length)
    model = get_model(model)
    latents = model(chunks).cpu().numpy()
    return latents
