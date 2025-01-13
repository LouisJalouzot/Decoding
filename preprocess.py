from src.lebel2023_preprocess import (
    create_lebel2023_dataset,
    create_lebel2023_fmriprep_dataset,
    create_lebel2023_fmripep_canica_dataset,
)
from src.li2022_preprocess import create_li2022_datasets
from src.preprocess import (
    create_balanced_dataset,
    create_mean_subject,
    create_pca_dataset,
)

create_lebel2023_dataset()
create_lebel2023_fmriprep_dataset()
create_balanced_dataset("lebel2023")
create_balanced_dataset("lebel2023_fmriprep")
create_mean_subject("lebel2023_fmriprep", "lebel2023_fmriprep_mean")
create_mean_subject(
    "lebel2023_fmriprep",
    "lebel2023_fmriprep_123_mean",
    ["UTS01", "UTS02", "UTS03"],
)
create_li2022_datasets()
create_li2022_datasets("FR")
create_li2022_datasets("CN")
create_mean_subject("li2022")
# create_lebel2023_fmripep_canica_dataset(per_subject=True)
# create_lebel2023_fmripep_canica_dataset(per_subject=True, n_components=768)
# create_lebel2023_fmripep_canica_dataset(per_subject=False, n_components=768)
# create_pca_dataset("lebel2023_fmriprep", 128, per_subject=True)
# create_pca_dataset("lebel2023_fmriprep", 128, per_subject=False)
