from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.textgrids import TextGrid
from src.utils import device, memory, progress

DEFAULT_BAD_WORDS = frozenset(
    ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", ""]
)


def compute_chunks(textgrid_path: str, tr: int, context_length: int) -> pd.DataFrame:
    """
    Compute the chunks of text from the given textgrid file.

    Args:
        textgrid_path (str): The path to the textgrid file.
        tr (int): The time resolution.
        context_length (int): The number of previous chunks to include for context.

    Returns:
        List[str]: The list of computed chunks of text.
    """
    transcript = TextGrid(textgrid_path).tiers[-1].simple_transcript
    transcript["chunk_id"] = pd.Categorical(
        transcript.xmax // tr,
        categories=range(int(np.ceil(transcript.xmax.max() / tr))),
    )
    transcript["text"] = (
        transcript["text"].str.lower().str.strip("{} ").replace("i", "I")
    )
    transcript = transcript[~transcript.text.isin(DEFAULT_BAD_WORDS)]
    transcript = (
        transcript.groupby("chunk_id", observed=False)
        .text.apply(lambda x: " ".join(x))
        .to_frame(name="chunk")
    )
    transcript["chunk_with_context"] = [
        w.str.cat(sep=" ") for w in transcript.chunk.rolling(context_length)
    ]
    return transcript.apply(lambda s: s.str.strip("# ").str.replace(" #", ","))


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


def prepare_audioclip(textgrid_path, audio_path, tr, context_length, model, verbose):
    # TODO fix for new chunks format
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
            f"Computing AudioCLIP latents for run {textgrid_path.stem}",
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
                audio=audio_chunk,
                text=[[text_chunks.chunk_with_context.iloc[i]]],
            )
            latent = torch.concat(
                (text_latent.squeeze(), audio_latent.squeeze()),
            )
            latents.append(latent.cpu().numpy())
            if verbose:
                progress.update(task, advance=1)
        progress.update(task, completed=True, visible=False)
    return np.vstack(latents)


def prepare_clap(textgrid_path, audio_path, tr, context_length, batch_size, verbose):
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
            f"Computing CLAP latents for run {textgrid_path.stem}",
            total=-(-size // batch_size),
            visible=verbose,
        )
        for i in range(0, size, batch_size):
            inputs = processor(
                text=text_chunks.chunk_with_context.iloc[i : i + batch_size],
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
        progress.update(task, completed=True, visible=False)
    return np.vstack(latents)


@memory.cache(ignore=["batch_size", "verbose"])
def prepare_latents(
    dataset: str,
    run: str,
    model: str,
    tr: int,
    context_length: int,
    token_aggregation: str = "mean",
    batch_size: int = 64,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:

    if "lebel2023" in dataset:
        textgrid_path = f"data/lebel2023/derivative/TextGrids/{run}.TextGrid"
        audio_path = f"data/lebel2023/stimuli/{run}.wav"
    elif "li2022" in dataset:
        audio_path = f"data/li2022/stimuli/task-lppEN_section_{run}.wav"
        textgrid_path = f"data/li2022/annotation/EN/lppEN_section{run}.TextGrid"
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    if model.lower() == "mel":
        latents = prepare_mel(audio_path, tr, context_length)
    elif model.lower() == "audioclip":
        latents = prepare_audioclip(
            textgrid_path, audio_path, tr, context_length, model, verbose
        )
    elif model.lower() == "clap":
        latents = prepare_clap(
            textgrid_path, audio_path, tr, context_length, batch_size, verbose
        )
    else:
        import warnings

        from sentence_transformers import SentenceTransformer

        warnings.filterwarnings("ignore", module="transformers")
        chunks = compute_chunks(textgrid_path, tr, context_length)

        model = SentenceTransformer(
            model,
            device=device,
            truncate_dim=None,
            trust_remote_code=True,
        )
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        latents = model.encode(
            chunks.chunk_with_context,
            batch_size=batch_size,
            output_value="token_embeddings",
        )
        if token_aggregation == "first":
            latents = [l[0] for l in latents]
        elif token_aggregation == "last":
            latents = [l[1] for l in latents]
        elif token_aggregation == "mean":
            latents = [l.mean(axis=0) for l in latents]
        elif token_aggregation == "max":
            latents = [l.max(axis=0)[0] for l in latents]
        else:
            n_tokens = model.tokenize(chunks.chunk)["attention_mask"].sum(axis=1)
            if token_aggregation == "chunk_mean":
                latents = [
                    l[-length:].mean(axis=0) for l, length in zip(latents, n_tokens)
                ]
            elif token_aggregation == "chunk_max":
                latents = [
                    l[-length:].max(axis=0)[0] for l, length in zip(latents, n_tokens)
                ]
            else:
                raise ValueError(
                    f"Unsupported token aggregation method {token_aggregation}"
                )

    latents = np.array([l.cpu() for l in latents])
    latents /= np.linalg.norm(latents, ord=2, axis=1, keepdims=True)
    latents = StandardScaler().fit_transform(latents)
    chunks = compute_chunks(textgrid_path, tr, context_length)
    if "lebel2023" in dataset or "li2022" in dataset:
        latents = latents[5:-5]
        chunks = chunks.iloc[5:-5]
    return latents.astype(np.float32), chunks
