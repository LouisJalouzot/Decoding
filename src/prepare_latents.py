from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from src.utils import device, get_textgrid, memory


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return (token_embeddings * input_mask_expanded).sum(
        dim=1
    ) / input_mask_expanded.sum(1).clamp(min=1e-9)


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
    story: str,
    model: str = "clip",
    tr: int = 2,
    context_length: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare the latents for the given story.

    Args:
        story (str): Story name.
        model_class (str, optional): The class of the model. Defaults to "clip".
        tr (int, optional): The time resolution. Defaults to 2.
        context_length (int, optional): The number of previous chunks to include for context. Defaults to 0.
        verbose (bool, optional): Whether to display progress. Defaults to False.

    Returns:
        np.ndarray: Latents.
    """
    path = Path("data/lebel")
    if model.lower() == "mel":
        import torchaudio

        path = path / "stimuli" / (story + ".wav")
        wav, sample_rate = torchaudio.load(str(path))
        n_channels = wav.shape[0] if len(wav.shape) > 1 else 1
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=sample_rate * tr * context_length,
            hop_length=sample_rate * tr,
            n_mels=768 // n_channels,
            normalized=True,
        ).to(device)
        latents = mel(wav.to(device)).reshape(768, -1).T.cpu().numpy()
        return latents
    else:
        import torch

        path = path / "derivative" / "TextGrids" / (story + ".TextGrid")
        chunks = compute_chunks(path, tr, context_length)

        if model.lower() == "clip":
            import clip

            clip_model, _ = clip.load("ViT-L/14", device=device)
            clip_model.eval()
            with torch.no_grad():
                return (
                    clip_model.encode_text(
                        clip.tokenize(chunks, truncate=True).to(device),
                    )
                    .cpu()
                    .numpy()
                )
        else:
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model)
            auto_model = AutoModel.from_pretrained(model).to(device)
            auto_model.eval()

            inputs = tokenizer(
                chunks, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            with torch.no_grad():
                model_output = auto_model(**inputs)
                latents = mean_pooling(
                    model_output,
                    inputs.attention_mask,
                )
                return F.normalize(latents, p=2, dim=1).cpu().numpy()
