#!.env/bin/python

import json
import re
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as tmf
from rich.live import Live
from rich.table import Table
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import pipeline

from src.main import main
from src.utils import console, device

datasets = ["li2022_EN_SS_trimmed_mean"]
subjects = None

datasets = ["lebel2023"]
subjects = {"lebel2023": ["UTS03"]}

config = {
    "datasets": datasets,
    "subjects": subjects,
    "model": "google/flan-t5-small",
    "decoder": "brain_decoder",
    "loss": "mixco",
    "valid_ratio": 0.1,
    "test_ratio": 0.1,
    "context_length": 6,
    "lag": 3,
    "smooth": 6,
    "stack": 0,
    "dropout": 0.7,
    "patience": 20,
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "batch_size": 1,
    "temperature": 0.05,
    # "top_encoding_voxels": 5000,
}

# gpt2 = pipeline("text-generation", model="gpt2", device=device)

# # Fetch data and decoder

df_train, df_valid, df_test = main(
    return_data=True, cache=False, wandb_mode="disabled", **config
)
_, decoder = main(wandb_mode="disabled", **config)
decoder = decoder.to(device)

# Decode T5

row = df_train.sample(1).iloc[0]
with torch.no_grad():
    predicted_latents = decoder(
        decoder.projector[row.dataset + "/" + row.subject](row.X.to(device))
    )

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-small", device_map=device
)

decoded_chunks = []
table = Table(
    "Chunk",
    "Duration",
    "Correct",
    "Predicted",
    title=f"Decoding {row.run}",
)
n_tokens = tokenizer(row.chunks, return_tensors="pt", padding=True, truncation=True)[
    "attention_mask"
].sum(axis=1)
n_chunks = row.X.shape[0]
with Live(table, console=console, vertical_overflow="visible"):
    for i, (chunk, max_new_tokens) in enumerate(zip(row.chunks, n_tokens)):
        start = time()

        latent_norm = torch.norm(predicted_latents[i], p=2)

        def hook_fn(module, input, output):
            output_norm = torch.norm(output, p=2)
            return (
                output + output_norm * predicted_latents[None, [i]] / latent_norm
            ) / 2

        hook_handle = model.encoder.block[-1].layer[-1].register_forward_hook(hook_fn)

        input_text = "Continue: " + " ".join(
            decoded_chunks[max(0, -config["context_length"]) :]
        )
        input_ids = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)

        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        decoded_chunks.append(generated_text)
        color = "[green]" if generated_text == chunk else ""
        table.add_row(
            f"{i+1} / {n_chunks}",
            f"{time() - start:.3g}s",
            color + chunk,
            color + generated_text,
        )

        hook_handle.remove()

# Decode simple

# model = SentenceTransformer(config["model"], device=device)

# chunks = set(df_train.drop_duplicates(["dataset", "run"]).text.sum())
# chunks |= set(df_valid.drop_duplicates(["dataset", "run"]).text.sum())
# chunks = pd.Series(list(chunks))
# n_possible_chunks = len(chunks)

# row = df_train.sample(1).iloc[0]
# with torch.no_grad():
#     predicted_latents = decoder(
#         decoder.projector[row.dataset + "/" + row.subject](row.X.to(device))
#     )

# decoded_chunks = []
# table = Table(
#     "Chunk",
#     "Duration",
#     "Correct",
#     "Predicted",
#     "Rank",
#     title=f"Decoding {row.run}",
# )
# n_chunks = row.X.shape[0]
# with Live(table, console=console, vertical_overflow="visible"):
#     for i in range(n_chunks):
#         start = time()
#         context_sentence = " ".join(decoded_chunks[-config["context_length"] :])
#         continuations = context_sentence + " " + chunks
#         continuations_latents = model.encode(
#             continuations,
#             batch_size=len(continuations) // 4,
#             convert_to_numpy=False,
#             convert_to_tensor=True,
#         )
#         scores = (
#             tmf.pairwise_cosine_similarity(
#                 predicted_latents[[i]], continuations_latents
#             )[0]
#             .cpu()
#             .numpy()
#         )
#         ranks = scores.argsort()[::-1].argsort()
#         rank = ranks[chunks == row.text[i]].item()
#         best_continuation = chunks[scores.argmax()]
#         decoded_chunks.append(best_continuation)

#         color = "[green]" if best_continuation == row.text[i] else ""
#         table.add_row(
#             f"{i+1} / {n_chunks}",
#             f"{time() - start:.3g}s",
#             color + row.text[i],
#             color + best_continuation,
#             f"{rank} / {n_possible_chunks}",
#         )

# Decode simple 2

# chunks = list(chunks)
# latents = model.encode(chunks, convert_to_tensor=True)
# scores = tmf.pairwise_cosine_similarity(predicted_latents, latents)

# decoded_chunks = []
# table = Table(
#     "Chunk",
#     # "Duration",
#     "Correct",
#     "Predicted",
#     # "Rank",
#     title=f"Decoding {row.run}",
# )
# n_chunks = row.X.shape[0]
# with Live(table, console=console, vertical_overflow="visible"):
#     for i in range(n_chunks):
#         # start = time()
#         decoded_chunk = chunks[scores[i].argmax().item()]

#         color = "[green]" if decoded_chunk == row.text[i] else ""
#         table.add_row(
#             f"{i+1} / {n_chunks}",
#             # f"{time() - start:.3g}s",
#             color + row.text[i],
#             color + decoded_chunk,
#             # f"{rank} / {n_possible_chunks}",
#         )

# # Decode Tang

# from semantic_decoding.decoding.Decoder import Decoder, Hypothesis

# from semantic_decoding.decoding.GPT import GPT
# from semantic_decoding.decoding.LanguageModel import LanguageModel

# data_lm = Path("data/data_lm")
# with open(data_lm / "perceived" / "vocab.json", "r") as f:
#     gpt_vocab = json.load(f)
# with open(data_lm / "decoder_vocab.json", "r") as f:
#     decoder_vocab = json.load(f)
# gpt = GPT(path=data_lm / "perceived" / "model", vocab=gpt_vocab, device=device)
# lm = LanguageModel(gpt, decoder_vocab, nuc_mass=0.9, nuc_ratio=0.1)

# gpt_decoder = Decoder(word_times=range(sum(row.num_words)), beam_width=50)

# model = SentenceTransformer(config["model"], device=device)
# clear_output()

# with tqdm(total=sum(row.num_words)) as pbar:
#     for i, num_words in enumerate(row.num_words):
#         # if i > 0:
#         #     print("\033[F\033[K\033[F\033[K", end="")
#         pbar.set_description(f"Chunk {i+1} / {len(row.num_words)}")
#         context_window = sum(row.num_words[max(0, i - config["context_length"]) : i])
#         for _ in range(num_words):
#             beam_nucs = lm.beam_propose(gpt_decoder.beam, context_window)
#             for c, (hyp, nextensions) in enumerate(gpt_decoder.get_hypotheses()):
#                 nuc, logprobs = beam_nucs[c]
#                 if len(nuc) < 1:
#                     continue
#                 extend_words = [
#                     " ".join(hyp.words[-context_window:] + [x]) for x in nuc
#                 ]
#                 embs = model.encode(
#                     extend_words, convert_to_numpy=False, convert_to_tensor=True
#                 )
#                 scores = tmf.pairwise_cosine_similarity(predicted_latents[[i]], embs)[
#                     0
#                 ].cpu()
#                 embs = [None] * len(embs)
#                 local_extensions = [
#                     Hypothesis(parent=hyp, extension=x) for x in zip(nuc, scores, embs)
#                 ]
#                 gpt_decoder.add_extensions(local_extensions, scores, nextensions)
#             gpt_decoder.extend(verbose=False)
#             context_window += 1
#             pbar.update(1)
#         best_hyp = np.argmax([sum(hyp.logprobs) for hyp in gpt_decoder.beam])
#         best_hyp = gpt_decoder.beam[best_hyp].words
#         predicted_chunks = []
#         current_index = len(best_hyp)
#         for num_words in row.num_words[
#             max(0, i + 1 - config["context_length"]) : i + 1
#         ]:
#             if current_index - num_words < 0:
#                 break
#             predicted_chunks.append(
#                 " ".join(best_hyp[current_index - num_words : current_index])
#             )
#             current_index -= num_words
#         print(
#             "Correct:        ",
#             " | ".join(row.text[max(0, i + 1 - config["context_length"]) : i + 1]),
#         )
#         print("Best hypothesis:", " | ".join(predicted_chunks[::-1]))
#         print()
