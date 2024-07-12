from src.lebel2023_preprocess import create_symlinks_lebel2023
from src.li2022_preprocess import (
    build_mean_subject,
    create_symlinks_li2022,
    resample_and_slice_brain,
)
from src.utils import console

console.log("Creating symlinks for Lebel2023")
create_symlinks_lebel2023()
console.log("Done, see the datasets at lebel2023/all_subjects and lebel2023/3_subjects")

console.log("Creating symlinks for Li2022 for English")
create_symlinks_li2022()
console.log("Done, see the dataset li2022/all_EN_raw")

console.log("Resampling and slicing brains for Li2022 for English")
resample_and_slice_brain()
console.log("Done, see the dataset at li2022/all_EN")

console.log("Building mean subject for Li2022")
build_mean_subject()
console.log("Done, see the dataset at li2022/mean_EN")
