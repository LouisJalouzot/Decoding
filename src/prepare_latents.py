from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.utils import device, get_textgrid, memory, progress


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
    group_indices = offsets // tr
    unique_indices = np.arange(group_indices.max() + 1)

    chunks = [" ".join(list(words[group_indices == idx])) for idx in unique_indices]
    return [
        " ".join([chunks[k] for k in range(max(0, i - context_length), i + 1)]).strip()
        for i in range(len(chunks))
    ]


def prepare_mel(audio_path: Path, tr: int, context_length: int):
    import torchaudio

    wav, sample_rate = torchaudio.load(str(audio_path))
    n_channels = wav.shape[0] if len(wav.shape) > 1 else 1
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=sample_rate * tr * context_length,
        hop_length=sample_rate * tr,
        n_mels=768 // n_channels,
        normalized=True,
    ).to(device)
    return mel(wav.to(device)).reshape(768, -1).T.cpu().numpy()


def prepare_audioclip(
    story, textgrid_path, audio_path, tr, context_length, model, verbose
):
    import librosa
    import torch

    text_chunks = compute_chunks(textgrid_path, tr, context_length)
    target_sample_rate = 44_100
    wav, _ = librosa.load(audio_path, sr=target_sample_rate, mono=False)
    wav = torch.from_numpy(wav)
    audio_chunk_size = target_sample_rate * tr
    chunked_audio = wav.split(audio_chunk_size, dim=-1)
    size = len(text_chunks)
    chunked_audio = chunked_audio[:size]  # truncate audio to match text length

    latents = []
    import sys

    sys.path.append("AudioCLIP")
    from AudioCLIP.model import AudioCLIP

    model = AudioCLIP(
        pretrained="AudioCLIP/assets/AudioCLIP-Full-Training.pt",
    ).to(device)
    model.eval()
    with torch.no_grad():
        task = progress.add_task(
            f"Computing AudioCLIP latents for story {story}",
            total=size,
            visible=verbose,
        )
        for i in range(size):
            audio_chunk = torch.concat(
                chunked_audio[max(0, i - context_length) : i + 1], dim=-1
            ).to(device)
            if len(audio_chunk.shape) == 1:
                audio_chunk = audio_chunk.reshape(1, 1, -1)
            else:
                audio_chunk = audio_chunk[None]
            ((audio_latent, _, text_latent), _), _ = model(
                audio=audio_chunk, text=[[text_chunks[i]]]
            )
            latent = torch.concat(
                (text_latent.squeeze(), audio_latent.squeeze()),
            )
            latents.append(latent.cpu().numpy())
            if verbose:
                progress.update(task, advance=1)
        progress.update(task, visible=False)
    return np.vstack(latents)


def prepare_clap(
    story, textgrid_path, audio_path, tr, context_length, batch_size, verbose
):
    import librosa
    import torch
    from transformers import AutoProcessor, ClapModel

    text_chunks = compute_chunks(textgrid_path, tr, context_length)
    target_sample_rate = 48_000
    wav, _ = librosa.load(audio_path, sr=target_sample_rate, mono=True)
    size = len(text_chunks)
    audio_chunk_size = target_sample_rate * tr
    chunked_audio = [
        wav[
            max(0, (i - context_length) * audio_chunk_size) : (i + 1) * audio_chunk_size
        ]
        for i in range(size)
    ]

    model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    processor = AutoProcessor.from_pretrained("laion/larger_clap_general")
    latents = []
    model.eval()
    with torch.no_grad():
        task = progress.add_task(
            f"Computing CLAP latents for story {story}",
            total=-(-size // batch_size),
            visible=verbose,
        )
        for i in range(0, size, batch_size):
            inputs = processor(
                text=text_chunks[i : i + batch_size],
                audios=chunked_audio[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                sampling_rate=target_sample_rate,
            ).to(device)
            outputs = model(**inputs)
            batch_latents = torch.cat(
                (outputs.text_embeds, outputs.audio_embeds), dim=1
            )
            latents.append(batch_latents.cpu())
            progress.update(task, advance=1)
        progress.update(task, visible=False)
    return np.vstack(latents)


@memory.cache(ignore=["batch_size", "verbose"])
def prepare_latents(
    story: str,
    model: str,
    tr: int,
    context_length: int,
    batch_size: int = 64,
    verbose: bool = True,
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
    import torch

    path = Path("data/lebel")
    audio_path = path / "stimuli" / (story + ".wav")
    textgrid_path = path / "derivative" / "TextGrids" / (story + ".TextGrid")

    if model.lower() == "mel":
        latents = prepare_mel(audio_path, tr, context_length)
    elif model.lower() == "audioclip":
        latents = prepare_audioclip(
            story, textgrid_path, audio_path, tr, context_length, model, verbose
        )
    elif model.lower() == "clap":
        latents = prepare_clap(
            story, textgrid_path, audio_path, tr, context_length, batch_size, verbose
        )
    else:
        chunks = compute_chunks(textgrid_path, tr, context_length)
        from sentence_transformers import SentenceTransformer

        # If model is not a sentence transformer, mean pooling will be applied
        model = SentenceTransformer(
            model,
            device=device,
            truncate_dim=None,
            trust_remote_code=True,
        )
        latents = model.encode(chunks)
    latents /= np.linalg.norm(latents, ord=2, axis=1, keepdims=True)
    torch.cuda.empty_cache()
    return latents.astype(np.float32)
