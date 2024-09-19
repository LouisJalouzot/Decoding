import string
import zipfile
from itertools import combinations_with_replacement
from pathlib import Path

import nltk
import numpy as np
import requests
import torch
import werpy
from evaluate import load
from joblib import Parallel, cpu_count, delayed
from joblib_progress import joblib_progress

from src.utils import console, device, memory, progress

nlp_dist_cols = [
    "glove_bow",
    "POS",
    "POS_restricted",
    "glove_bow_POS_restricted",
]
nlp_cols = [
    "n_trs",
    "chunks_index",
    "chunks_with_context",
] + nlp_dist_cols


def load_glove_embeddings():
    glove_file = Path("data/glove.6B.50d.txt")
    if not glove_file.exists():
        url = (
            "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        )
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 131072
        zip_path = Path("data/glove.6B.zip")

        with open(zip_path, "wb") as file, progress:
            task = progress.add_task(
                "Downloading GloVe embeddings", total=total_size // chunk_size
            )
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                progress.update(task, advance=1)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("glove.6B.50d.txt", "data/")
        zip_path.unlink()
        console.log(f"Saved GloVe embeddings to {glove_file}")

    embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def get_glove_bows(chunks):
    glove_embeddings = load_glove_embeddings()
    # Remove punctuation
    chunks = [
        chunk.translate(str.maketrans("", "", string.punctuation))
        for chunk in chunks
    ]
    chunks = [chunk.lower().split() for chunk in chunks]
    glove_bows = []
    for chunk in chunks:
        glove_bow = np.zeros(50)
        for word in chunk:
            glove_bow += glove_embeddings.get(word, np.zeros(50))
        glove_bows.append(glove_bow / max(1, len(chunk)))

    return glove_bows


def nltk_pos_tag(chunks):
    tokenized_chunks = [nltk.tokenize.word_tokenize(chunk) for chunk in chunks]
    pos_tagged_chunks = [
        [e[1] for e in nltk.tag.pos_tag(tokenized_chunk, tagset="universal")]
        for tokenized_chunk in tokenized_chunks
    ]
    return tokenized_chunks, pos_tagged_chunks


def get_glove_bows(chunks):
    glove_embeddings = load_glove_embeddings()
    # Remove punctuation
    chunks = [
        chunk.translate(str.maketrans("", "", string.punctuation))
        for chunk in chunks
    ]
    chunks = [chunk.lower().split() for chunk in chunks]
    glove_bows = []
    for chunk in chunks:
        glove_bow = np.zeros(50)
        for word in chunk:
            glove_bow += glove_embeddings.get(word, np.zeros(50))
        glove_bows.append(glove_bow / max(1, len(chunk)))

    return glove_bows


def nltk_pos_tag(chunks):
    tokenized_chunks = [nltk.tokenize.word_tokenize(chunk) for chunk in chunks]
    pos_tagged_chunks = [
        [e[1] for e in nltk.tag.pos_tag(tokenized_chunk, tagset="universal")]
        for tokenized_chunk in tokenized_chunks
    ]
    return tokenized_chunks, pos_tagged_chunks


def return_all_pairs(l):
    pairs = combinations_with_replacement(l, 2)
    return np.array(list(pairs)).swapaxes(0, 1)


@memory.cache(ignore=["batch_size"])
def compute_nlp_distances(df, batch_size=4096):
    import warnings

    warnings.filterwarnings("ignore", module="transformers")
    distances = {}
    n_chunks = df.n_trs.sum()
    console.log(f"Computing NLP pairwise distances for {n_chunks} chunks")

    # Building correspondance matrix (chunk index i and j -> index k in the flattened array of distances)
    n_pairs = n_chunks * (n_chunks - 1) // 2 + n_chunks
    n_batches = -(-n_pairs // batch_size)
    corresp = np.zeros((n_chunks, n_chunks), dtype=int)
    indices = return_all_pairs(df.chunks_index.explode())
    corresp[*indices] = np.arange(n_pairs)
    corresp_null_diag = corresp.copy()
    np.fill_diagonal(corresp_null_diag, 0)
    corresp += corresp_null_diag.T
    distances["corresp"] = corresp

    console.log("Computing GloVe BOW distances")
    for glove in ["glove_bow", "glove_bow_POS_restricted"]:
        glove_bow = df[glove].explode()
        glove_pairs = return_all_pairs(glove_bow)
        glove_bow_cosine = torch.cosine_similarity(
            torch.from_numpy(glove_pairs[0]),
            torch.from_numpy(glove_pairs[1]),
            dim=-1,
        )
        distances[glove + "_cosine"] = glove_bow_cosine.numpy()

    pos_pairs = return_all_pairs(df.POS.explode())
    chunks_pairs = return_all_pairs(df.chunks_with_context.explode())
    for name, pairs in [("pos_wer", pos_pairs), ("wer", chunks_pairs)]:
        with joblib_progress(f"Computing {name}", total=n_batches):
            wers = sum(
                Parallel(n_jobs=-1)(
                    delayed(werpy.wers)(
                        pairs[0][i : i + batch_size],
                        pairs[1][i : i + batch_size],
                    )
                    for i in range(0, n_pairs, batch_size)
                ),
                [],
            )
            distances[name] = np.array(wers)

    console.log("Computing BERT F1 Scores")
    bertscore = load("bertscore")
    scores = bertscore.compute(
        predictions=chunks_pairs[0],
        references=chunks_pairs[1],
        model_type="distilbert-base-uncased-distilled-squad",
        device=device,
        use_fast_tokenizer=True,
        verbose=True,
        nthreads=-1,
    )["f1"]
    distances["bert_f1"] = np.array(scores)

    return distances
