import zipfile
from pathlib import Path

import nltk
import requests
from huggingface_hub import snapshot_download

from src.lebel2023_preprocess import (
    create_lebel2023_dataset,
    create_lebel2023_fmripep_canica_dataset,
    create_lebel2023_fmriprep_dataset,
)
from src.li2022_preprocess import create_li2022_datasets
from src.smn4lang_preprocess import (
    create_smn4lang_dataset,
    create_smn4lang_textgrids,
)
from src.preprocess import (
    create_balanced_dataset,
    create_mean_subject,
    create_pca_dataset,
)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
glove_txt = data_dir / "glove.6B.50d.txt"
glove_zip = data_dir / "glove.6B.zip"

if not glove_txt.exists():
    print("Downloading GloVe embeddings...")
    response = requests.get(
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
    )
    with open(glove_zip, "wb") as f:
        f.write(response.content)

    # Extract the zip file
    with zipfile.ZipFile(glove_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    # Clean up the zip file
    glove_zip.unlink()
    print("GloVe embeddings downloaded and extracted.")

# Download required HF models
print("Checking for required models")
models = [
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse",
    "bert-base-uncased",
    "meta-llama/Llama-3.1-8B-Instruct",
    "TencentBAC/Conan-embedding-v1",
]

for model_name in models:
    try:
        snapshot_download(repo_id=model_name, local_files_only=True)
        print(f"Model {model_name} found in cache")
    except Exception:
        print(f"Downloading {model_name}")
        snapshot_download(repo_id=model_name)
        print(f"Model {model_name} downloaded successfully")

# Download necessary data before computing on nodes without internet access
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("universal_tagset")

# create_lebel2023_dataset()
# create_lebel2023_fmriprep_dataset()
# create_balanced_dataset("lebel2023")
# create_balanced_dataset("lebel2023_fmriprep")
# create_mean_subject("lebel2023_fmriprep", "lebel2023_fmriprep_mean")
# create_mean_subject(
#     "lebel2023_fmriprep",
#     "lebel2023_fmriprep_123_mean",
#     ["UTS01", "UTS02", "UTS03"],
# )
# create_li2022_datasets()
# create_li2022_datasets("FR")
# create_li2022_datasets("CN")
# create_mean_subject("li2022")
create_smn4lang_textgrids()
create_smn4lang_dataset()
# create_lebel2023_fmripep_canica_dataset(per_subject=True)
# create_lebel2023_fmripep_canica_dataset(per_subject=True, n_components=768)
# create_lebel2023_fmripep_canica_dataset(per_subject=False, n_components=768)
# create_pca_dataset("lebel2023_fmriprep", 128, per_subject=True)
# create_pca_dataset("lebel2023_fmriprep", 128, per_subject=False)
