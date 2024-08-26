# %%
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as tmf
from IPython.display import clear_output
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import pipeline

from src.main import main
from src.prepare_latents import compute_chunks
from src.utils import device, progress

# %%
datasets = ["li2022_EN_SS_trimmed_mean"]
subjects = None

# %%
# datasets = ["lebel2023"]
# subjects = {"lebel2023": ["UTS03"]}

# %%
config = {
    "datasets": datasets,
    "subjects": subjects,
    "model": "bert-base-uncased",
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
}

# %%
gpt2 = pipeline("text-generation", model="gpt2", device=device)

# %% [markdown]
# # Fetch data and decoder

# %%
df_train, df_valid, df_test = main(
    return_data=True, caching=False, wandb_mode="disabled", **config
)
_, decoder = main(wandb_mode="disabled", **config)
decoder = decoder.to(device)
clear_output()

# %%
df_chunks = []
df = pd.concat(
    [
        df_train[["dataset", "run"]],
        df_valid[["dataset", "run"]],
        df_test[["dataset", "run"]],
    ]
).drop_duplicates()
for _, row in tqdm(df.iterrows(), total=len(df)):
    if row.dataset == "lebel2023":
        textgrid_path = f"data/lebel2023/derivative/TextGrids/{row.run}.TextGrid"
    else:
        textgrid_path = f"data/li2022/annotation/EN/lppEN_section{row.run}.TextGrid"
    chunks = compute_chunks(textgrid_path, 2, 0)
    num_words = [len(chunk.split(" ")) for chunk in chunks]
    df_chunks.append([row.dataset, row.run, chunks, num_words])
df_chunks = pd.DataFrame(df_chunks, columns=["dataset", "run", "text", "num_words"])

# %%
df_train = df_train.drop(columns=["text"]).merge(df_chunks)
df_valid = df_valid.drop(columns=["text"]).merge(df_chunks)
df_test = df_test.drop(columns=["text"]).merge(df_chunks)

# %%
if "wheretheressmoke" in df_train.run.values:
    row = df_train[df_train.run == "wheretheressmoke"].iloc[0]
else:
    _, row = next(iter(df_train.iterrows()))
with torch.no_grad():
    predicted_latents = decoder(
        decoder.projector[row.dataset + "/" + row.subject](row.X.to(device))
    )

# %% [markdown]
# # Decode Tang

from semantic_decoding.decoding.Decoder import Decoder, Hypothesis

# %%
from semantic_decoding.decoding.GPT import GPT
from semantic_decoding.decoding.LanguageModel import LanguageModel

data_lm = Path("data/data_lm")
with open(data_lm / "perceived" / "vocab.json", "r") as f:
    gpt_vocab = json.load(f)
with open(data_lm / "decoder_vocab.json", "r") as f:
    decoder_vocab = json.load(f)
gpt = GPT(path=data_lm / "perceived" / "model", vocab=gpt_vocab, device=device)
lm = LanguageModel(gpt, decoder_vocab, nuc_mass=0.9, nuc_ratio=0.1)

# %%
gpt_decoder = Decoder(word_times=range(sum(row.num_words)), beam_width=50)

# %%
model = SentenceTransformer(config["model"], device=device)
clear_output()

# %%
with tqdm(total=sum(row.num_words)) as pbar:
    for i, num_words in enumerate(row.num_words):
        # if i > 0:
        #     print("\033[F\033[K\033[F\033[K", end="")
        pbar.set_description(f"Chunk {i+1} / {len(row.num_words)}")
        context_window = sum(row.num_words[max(0, i - config["context_length"]) : i])
        for _ in range(num_words):
            beam_nucs = lm.beam_propose(gpt_decoder.beam, context_window)
            for c, (hyp, nextensions) in enumerate(gpt_decoder.get_hypotheses()):
                nuc, logprobs = beam_nucs[c]
                if len(nuc) < 1:
                    continue
                extend_words = [
                    " ".join(hyp.words[-context_window:] + [x]) for x in nuc
                ]
                embs = model.encode(
                    extend_words, convert_to_numpy=False, convert_to_tensor=True
                )
                scores = tmf.pairwise_cosine_similarity(predicted_latents[[i]], embs)[
                    0
                ].cpu()
                embs = [None] * len(embs)
                local_extensions = [
                    Hypothesis(parent=hyp, extension=x) for x in zip(nuc, scores, embs)
                ]
                gpt_decoder.add_extensions(local_extensions, scores, nextensions)
            gpt_decoder.extend(verbose=False)
            context_window += 1
            pbar.update(1)
        best_hyp = np.argmax([sum(hyp.logprobs) for hyp in gpt_decoder.beam])
        best_hyp = gpt_decoder.beam[best_hyp].words
        predicted_chunks = []
        current_index = len(best_hyp)
        for num_words in row.num_words[
            max(0, i + 1 - config["context_length"]) : i + 1
        ]:
            if current_index - num_words < 0:
                break
            predicted_chunks.append(
                " ".join(best_hyp[current_index - num_words : current_index])
            )
            current_index -= num_words
        print(
            "Correct:        ",
            " | ".join(row.text[max(0, i + 1 - config["context_length"]) : i + 1]),
        )
        print("Best hypothesis:", " | ".join(predicted_chunks[::-1]))
        print()
