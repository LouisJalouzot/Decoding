import logging
import string
import zipfile
from itertools import combinations_with_replacement
from pathlib import Path

import nltk
import numpy as np
import requests
from fuzzywuzzy import fuzz
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from evaluate import load
from src.utils import batch_combinations, console, device, memory, progress

logger = logging.getLogger(__name__)

nlp_dist_cols = [
    "glove_bow",
    "pos",
    "pos_restricted",
    "glove_bow_pos_restricted",
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
        logger.info(f"Saved GloVe embeddings to {glove_file}")

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
    nltk.download("universal_tagset", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    tokenized_chunks = [nltk.tokenize.word_tokenize(chunk) for chunk in chunks]
    pos_tagged_chunks = [
        [e[1] for e in nltk.tag.pos_tag(tokenized_chunk, tagset="universal")]
        for tokenized_chunk in tokenized_chunks
    ]
    return tokenized_chunks, pos_tagged_chunks


def return_all_pairs(l):
    pairs = combinations_with_replacement(l, 2)
    return np.array(list(pairs)).swapaxes(0, 1)


def cosine_similarity_batch(a, b, axis=-1):
    ab = np.sum(a * b, axis=axis)
    norm_a = np.linalg.norm(a, axis=axis)
    norm_b = np.linalg.norm(b, axis=axis)
    return ab / (norm_a * norm_b).clip(1e-9)


def batch_ratio(a, b):
    return np.array([fuzz.ratio(a_, b_) / 100 for a_, b_ in zip(a, b)])


@memory.cache(ignore=["batch_size"])
def compute_nlp_distances(
    n_trs,
    chunks_index,
    chunks_with_context,
    glove_bow,
    pos,
    pos_restricted,
    glove_bow_pos_restricted,
    batch_size=4096,
):
    # import warnings

    # warnings.filterwarnings("ignore", module="transformers")
    n_chunks = sum(n_trs)

    # Building correspondance matrix (chunk index i and j -> index k in the flattened array of distances)
    n_pairs = n_chunks * (n_chunks - 1) // 2 + n_chunks
    n_batches = -(-n_pairs // batch_size)
    corresp = np.zeros((n_chunks, n_chunks), dtype=int)
    indices = return_all_pairs(chunks_index)
    corresp[*indices] = np.arange(n_pairs)
    corresp_null_diag = corresp.copy()
    np.fill_diagonal(corresp_null_diag, 0)
    corresp += corresp_null_diag.T
    distances = {"corresp": corresp}

    generators = [
        (cosine_similarity_batch, batch_combinations(glove_bow, 2, batch_size)),
        (
            cosine_similarity_batch,
            batch_combinations(glove_bow_pos_restricted, 2, batch_size),
        ),
        (batch_ratio, batch_combinations(pos, 2, batch_size)),
        (batch_ratio, batch_combinations(chunks_with_context, 2, batch_size)),
    ]

    with joblib_progress(
        f"Computing NLP distances for {n_chunks} chunks",
        total=len(generators) * n_batches,
        console=console,
    ):
        results = Parallel(n_jobs=-1, max_nbytes=None)(
            delayed(lambda a, b, f: f(a, b))(a, b, f)
            for f, generator in generators
            for a, b in generator
        )

    for i, name in enumerate(
        [
            "glove_bow_cosine",
            "glove_bow_pos_restricted_cosine",
            "pos_ratio",
            "ratio",
        ]
    ):
        distances[name] = np.concatenate(
            results[i * n_batches : (i + 1) * n_batches]
        )

    # chunks_pairs = return_all_pairs(df.chunks_with_context.explode())
    # logger.info("Computing BERT F1 Scores")
    # bertscore = load("bertscore")
    # scores = bertscore.compute(
    #     predictions=chunks_pairs[0],
    #     references=chunks_pairs[1],
    #     model_type="distilbert-base-uncased-distilled-squad",
    #     device=device,
    #     use_fast_tokenizer=True,
    #     verbose=True,
    #     nthreads=-1,
    # )["f1"]
    # distances["bert_f1"] = np.array(scores)

    return distances
