import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils import STORIES_BY_SESSION, compute_retrieval_metrics, replace_with_average

# ENCODER_MODEL = 'CLIP'
ENCODER_MODEL = "SBERT"

all_sessions = list(range(len(STORIES_BY_SESSION)))
train_sessions = all_sessions[:-1]
test_sessions = all_sessions[-1:]
train_stories = [
    story for session in train_sessions for story in STORIES_BY_SESSION[session]
]
test_stories = [
    story for session in test_sessions for story in STORIES_BY_SESSION[session]
]

# 1. Load brain features

# Subsample vertices for faster computation
all_brain_features_path = Path(
    "/storage/store2/work/athual/data/lebel-2022/ds003020/derivative/preprocessed_data/UTS03"
)

story = train_stories[0]
brain_features_path = all_brain_features_path / f"{story}.hf5"
run = h5py.File(brain_features_path, "r")
n_vertices = run["data"].shape[1]

selected_vertices = np.random.permutation(n_vertices)[:10000]


def get_brain_features(stories):
    all_features = []
    for story in stories:
        brain_features_path = all_brain_features_path / f"{story}.hf5"
        run = h5py.File(brain_features_path, "r")

        brain_features = run["data"][:]
        all_features.append(
            StandardScaler().fit_transform(brain_features[:, selected_vertices])
        )

    return all_features


train_brain_features = get_brain_features(train_stories)
test_brain_features = get_brain_features(test_stories)

train_brain_features = [replace_with_average(run) for run in train_brain_features]
test_brain_features = [replace_with_average(run) for run in test_brain_features]

print(len(train_stories))
print(len(test_stories))

# 2. Load text features

# Get session length per story
train_story_lengths = [
    train_brain_features[i].shape[0] for i in range(len(train_brain_features))
]
test_story_lengths = [
    test_brain_features[i].shape[0] for i in range(len(test_brain_features))
]


if ENCODER_MODEL == "CLIP":
    all_text_features_path = Path(
        "/storage/store2/work/athual/outputs/tang-2023-reproduction/clip-features"
    )
elif ENCODER_MODEL == "BERT":
    all_text_features_path = Path(
        "/storage/store2/work/athual/outputs/tang-2023-reproduction/bert-features"
    )
elif ENCODER_MODEL == "SBERT":
    all_text_features_path = Path(
        "/storage/store2/work/athual/outputs/tang-2023-reproduction/sbert-features"
    )
else:
    raise NotImplementedError


def get_text_features(stories, story_lenghts, n_context=0):
    all_features = []
    for i, story in enumerate(stories):
        story_features_path = (
            all_text_features_path / f"{story}_ncontext-{n_context}.npy"
        )
        all_features.append(
            StandardScaler().fit_transform(np.load(story_features_path))[
                -story_lenghts[i] :, :
            ]
        )

    return all_features


n_context = 4
train_text_features = get_text_features(
    train_stories, train_story_lengths, n_context=n_context
)
test_text_features = get_text_features(
    test_stories, test_story_lengths, n_context=n_context
)

lag = 2

brain_decoder = Ridge(alpha=5e4)

brain_decoder.fit(
    np.concatenate([run[lag:] for run in train_brain_features], axis=0),
    np.concatenate([run[:-lag] for run in train_text_features], axis=0),
)

predictions = brain_decoder.predict(
    np.concatenate([run[lag:] for run in test_brain_features], axis=0)
)

ground_truth = np.concatenate([run[:-lag] for run in test_text_features], axis=0)
negatives = ground_truth

metrics = compute_retrieval_metrics(
    torch.tensor(predictions),
    ground_truth=torch.tensor(ground_truth),
    negatives=torch.tensor(negatives),
    should_plot=True,
    return_scores=True,
    top_k=[1, 5, 10],
)

print(metrics)

scores = metrics["scores"]

ranks_all = np.argsort(scores, axis=1).numpy()[:, ::-1]
ranks_ground_truth = (scores > scores[:, [0]]).sum(dim=1).numpy()

all_chunks = []
for i, story in enumerate(test_stories):
    chunks = np.load(
        all_text_features_path / f"{story}_ncontext-{n_context}_chunks.npy"
    )
    print(chunks.shape)
    all_chunks.append(chunks[-test_story_lengths[i] :])

ground_truth_chunks = np.concatenate([run[:-lag] for run in all_chunks], axis=0)
ground_truth_chunks.shape

ground_truth_chunks[6]
ranks_all[6]

from sentence_transformers import util as sbertutils

for i in range(0, len(predictions), 100):
    prediction = predictions[i].astype("float32")
    query = ground_truth_chunks[i]
    cos_scores = sbertutils.cos_sim(prediction, negatives)
    top_k = min(5, len(negatives))
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(ground_truth_chunks[idx])  # , "(Score: {:.4f})".format(score))

for i in range(0, len(predictions), 100):
    print(ranks_ground_truth[i])
    print(ground_truth_chunks[i])
    print(ground_truth_chunks[ranks_all[i, 0] - 1])
    print()

output_path = Path("/storage/store2/work/athual/outputs/tang-2023-reproduction")

with open(output_path / f"brain_decoder_ridge_{ENCODER_MODEL}.pkl", "wb") as f:
    pickle.dump(brain_decoder, f)

with open(output_path / f"predictions_{ENCODER_MODEL}.npy", "wb") as f:
    np.save(f, predictions)
