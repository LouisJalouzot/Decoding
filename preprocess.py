from src.lebel2023_preprocess import create_lebel2023_dataset
from src.li2022_preprocess import (
    build_mean_subject,
    build_SRM_dataset,
    create_li2022_datasets,
)

create_lebel2023_dataset()
create_li2022_datasets()
build_mean_subject(SS=True)
build_mean_subject(SS=True, trimmed=True)
build_mean_subject(SS=False)
build_mean_subject(SS=False, trimmed=True)
build_SRM_dataset(input_path="datasets/li2022_EN_SS", n_components=500)
build_SRM_dataset(input_path="datasets/li2022_EN_SS", n_components=2000)
build_SRM_dataset(input_path="datasets/li2022_EN", n_components=500)
build_SRM_dataset(input_path="datasets/li2022_EN", n_components=2000)
