from src.lebel2023_preprocess import (
    create_lebel2023_balanced_dataset,
    create_lebel2023_dataset,
    create_lebel2023_fmriprep_dataset,
    create_lebel2023_mean_subject,
)
from src.li2022_preprocess import build_mean_subject, create_li2022_datasets

create_lebel2023_dataset()
create_lebel2023_fmriprep_dataset()
create_lebel2023_balanced_dataset("lebel2023")
create_lebel2023_balanced_dataset("lebel2023_fmriprep")
create_lebel2023_mean_subject("lebel2023_fmriprep", "lebel2023_fmriprep_mean")
create_lebel2023_mean_subject(
    "lebel2023_fmriprep",
    "lebel2023_fmriprep_123_mean",
    ["UTS01", "UTS02", "UTS03"],
)
create_li2022_datasets()
create_li2022_datasets("FR")
create_li2022_datasets("CN")
build_mean_subject(SS=True)
build_mean_subject(SS=True, trimmed=True)
