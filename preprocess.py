from src.lebel2023_preprocess import (
    create_lebel2023_balanced_dataset,
    create_lebel2023_dataset,
)
from src.li2022_preprocess import build_mean_subject, create_li2022_datasets

create_lebel2023_dataset()
create_lebel2023_balanced_dataset()
create_li2022_datasets()
create_li2022_datasets("FR")
create_li2022_datasets("CN")
build_mean_subject(SS=True)
build_mean_subject(SS=True, trimmed=True)
