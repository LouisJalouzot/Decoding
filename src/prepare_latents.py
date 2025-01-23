from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.nlp_metrics import get_glove_bows, nltk_pos_tag
from src.textgrids import TextGrid
from src.utils import device, memory, progress

DEFAULT_BAD_WORDS = frozenset(
    ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", ""]
)

tag_map = {
    "NOUN": "N",
    "VERB": "V",
    "ADJ": "J",
    "ADV": "R",
    "PRON": "P",
    "DET": "D",
    "ADP": "A",
    "CONJ": "C",
    "PRT": "T",
    "NUM": "M",
    "X": "X",
}


def compute_chunks(
    textgrid_path: str,
    tr: int,
    context_length: int,
) -> pd.DataFrame:
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
    if "lppCN" in textgrid_path or "SMN4Lang" in textgrid_path:
        combiner = ""
    else:
        combiner = " "
    transcript = (
        transcript.groupby("chunk_id", observed=False)
        .text.apply(lambda x: combiner.join(x))
        .to_frame(name="chunks")
    )
    transcript["chunks_with_context"] = [
        w.str.cat(sep=" ")
        for w in transcript.chunks.rolling(context_length + 1)
    ]
    transcript = transcript.apply(
        lambda s: s.str.strip("# ").str.replace(" #", ",")
    )
    chunks_with_context = transcript.chunks_with_context.tolist()
    transcript["glove_bow"] = get_glove_bows(chunks_with_context)
    tokenized_chunks, pos_tagged_chunks = nltk_pos_tag(chunks_with_context)
    encoded_pos = []
    for tags in pos_tagged_chunks:
        encoded_pos_chunk = [tag_map.get(tag, "?") for tag in tags]
        encoded_pos.append(" ".join(encoded_pos_chunk))
    transcript["pos"] = encoded_pos
    transcript["pos_restricted"] = [
        " ".join(
            tok
            for tok, tag in zip(toks, tags)
            if tag in ["NOUN", "VERB", "ADJ"]
        )
        for toks, tags in zip(tokenized_chunks, pos_tagged_chunks)
    ]
    transcript["glove_bow_pos_restricted"] = get_glove_bows(
        transcript.pos_restricted
    )

    return transcript


def prepare_wav2vec(audio_path, tr, context_length):
    import librosa
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    model_name = "facebook/wav2vec2-base-960h"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    sampling_rate = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(audio_path, sr=sampling_rate)

    chunk_length = tr * sampling_rate
    audios = [
        audio[max(0, i - chunk_length * context_length) : i + chunk_length]
        for i in range(0, len(audio), chunk_length)
    ]

    latents = []

    for audio_chunk in audios:
        inputs = processor(
            audio_chunk, return_tensors="pt", sampling_rate=sampling_rate
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        latent = outputs.last_hidden_state[0].mean(0).cpu().numpy()
        latents.append(latent)

    return np.vstack(latents)


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
    text_chunks, audio_path, tr, context_length, model, verbose
):
    import librosa
    import torch

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
            f"Computing AudioCLIP latents for run {audio_path.stem}",
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
                text=[[text_chunks.chunks_with_context.iloc[i]]],
            )
            latent = torch.concat(
                (text_latent.squeeze(), audio_latent.squeeze()),
            )
            latents.append(latent.cpu().numpy())
            if verbose:
                progress.update(task, advance=1)
        progress.update(task, completed=True, visible=False)
    return np.vstack(latents)


def prepare_clap(
    text_chunks, audio_path, tr, context_length, verbose, batch_size=32
):
    import librosa
    import torch
    from transformers import AutoProcessor, ClapModel

    target_sample_rate = 48_000
    wav, _ = librosa.load(audio_path, sr=target_sample_rate, mono=True)
    size = len(text_chunks)
    audio_chunk_size = target_sample_rate * tr
    chunked_audio = [
        wav[
            max(0, (i - context_length) * audio_chunk_size) : (i + 1)
            * audio_chunk_size
        ]
        for i in range(size)
    ]

    model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    processor = AutoProcessor.from_pretrained("laion/larger_clap_general")
    latents = []
    model.eval()
    with torch.no_grad():
        task = progress.add_task(
            f"Computing CLAP latents for run {audio_path.stem}",
            total=-(-size // batch_size),
            visible=verbose,
        )
        for i in range(0, size, batch_size):
            inputs = processor(
                text=text_chunks.chunks_with_context.iloc[i : i + batch_size],
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


@memory.cache(ignore=["verbose"])
def prepare_latents(
    dataset: str,
    run: str,
    model: str,
    tr: int,
    context_length: int,
    token_aggregation: str = "mean",
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    import warnings

    warnings.filterwarnings("ignore", module="transformers")

    if dataset == "lebel2023":
        textgrid_path = f"data/lebel2023/derivatives/TextGrids/{run}.TextGrid"
        audio_path = f"data/lebel2023/stimuli/{run}.wav"
    elif dataset == "li2022":
        audio_path = f"data/li2022/stimuli/task-lppEN_section-{run}.wav"
        textgrid_path = f"data/li2022/annotation/EN/lppEN_section{run}.TextGrid"
    elif dataset == "smn4lang":
        textgrid_path = f"data/SMN4Lang/derivatives/annotations/time_align/word-level/story_{run}_word_time.TexGrid"
    else:
        raise ValueError(f"Unsupported dataset {dataset}")
    chunks = compute_chunks(textgrid_path, tr, context_length)

    if model.lower() == "wav2vec":
        latents = prepare_wav2vec(audio_path, tr, context_length)
    elif model.lower() == "mel":
        latents = prepare_mel(audio_path, tr, context_length)
    elif model.lower() == "audioclip":
        latents = prepare_audioclip(
            chunks, audio_path, tr, context_length, model, verbose
        )
    elif model.lower() == "clap":
        latents = prepare_clap(chunks, audio_path, tr, context_length, verbose)
    elif model.lower() == "llm2vec":
        # TIP: Because of annoying multiprocess with CUDA, use on CPU or with only 1 viisble GPU
        from llm2vec import LLM2Vec

        assert token_aggregation == "mean", "Only mean aggregation is supported"
        model = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
        peft_model = "unsup-simcse"
        l2v = LLM2Vec.from_pretrained(
            model,
            peft_model_name_or_path=model + "-" + peft_model,
            merge_peft=True,
            device_map=device.type,
            max_length=None,
            torch_dtype=(
                torch.bfloat16 if device.type == "cuda" else torch.float32
            ),
        )
        latents = l2v.encode(
            chunks.chunks_with_context.tolist(), show_progress_bar=False
        )
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model,
            device=device,
            truncate_dim=None,
            trust_remote_code=True,
        )
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        latents = model.encode(
            chunks.chunks_with_context,
            output_value="token_embeddings",
            show_progress_bar=False,
        )

        if token_aggregation == "first":
            latents = [l[0] for l in latents]
        elif token_aggregation == "last":
            latents = [l[-1] for l in latents]
        elif token_aggregation == "mean":
            latents = [l.mean(axis=0) for l in latents]
        elif token_aggregation == "max":
            latents = [l.max(axis=0)[0] for l in latents]
        else:
            n_tokens = model.tokenize(chunks.chunks)["attention_mask"].sum(
                axis=1
            )
            if token_aggregation == "chunk_mean":
                latents = [
                    l[-length:].mean(axis=0)
                    for l, length in zip(latents, n_tokens)
                ]
            elif token_aggregation == "chunk_max":
                latents = [
                    l[-length:].max(axis=0)[0]
                    for l, length in zip(latents, n_tokens)
                ]
            else:
                raise ValueError(
                    f"Unsupported token aggregation method {token_aggregation}"
                )

        latents = np.vstack([l.cpu().numpy() for l in latents])

    latents /= np.linalg.norm(latents, ord=2, axis=1, keepdims=True)
    latents = StandardScaler().fit_transform(latents)
    if "lebel2023" in dataset or "li2022" in dataset:
        latents = latents[5:-5]
        chunks = chunks.iloc[5:-5]
    data = {"Y": latents.astype(np.float32)}
    for col in chunks:
        data[col] = chunks[col].values
    return pd.Series(data)
