#!.env/bin/python
import os
import re
from pathlib import Path

import h5py
import numpy as np
from bids import BIDSLayout
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

os.makedirs("figs", exist_ok=True)
lang = "EN"
n_runs = 3
layout = BIDSLayout("data/li2022/derivatives", validate=False, is_derivative=True)
subjects = layout.get_subjects()
subjects = [s for s in subjects if re.match(lang + r"\d+brain$", s)][:2]
n_subjects = len(subjects)


def read(subject, run, path):
    img = h5py.File(path, "r")["data"][...]
    img = img.reshape(-1, img.shape[-1])
    img = StandardScaler(copy=False).fit_transform(img.T).T
    return subject, run, path, img.shape[1], img


res = Parallel(n_jobs=5, verbose=10)(
    delayed(read)(subject, run, path)
    for subject in subjects
    for run, path in enumerate(layout.get(subject=subject)[:3])
)
res = {(subject, run): (img, path, n_scans) for subject, run, path, n_scans, img in res}
imgs = [[res[(subject, run)][0].T for run in range(n_runs)] for subject in subjects]
corresp = {}
for subject in subjects:
    start = 0
    for run in range(n_runs):
        _, path, n_scans = res[(subject, run)]
        corresp[path] = (subject, run, slice(start, start + n_scans))
        start += n_scans

from fastsrm.identifiable_srm import IdentifiableFastSRM

for n_components in tqdm([100]): #, 1000, 10000, 100000]):
    srm = IdentifiableFastSRM(n_components=n_components, n_jobs=5, verbose=True)
    S = srm.fit_transform(imgs)
    np.savez_compressed(
        f"data/li2022/SRM_{lang}_{n_components // 1000}k.npz",
        **{subject+"_S": S[i] for i, subject in enumerate(subjects)},
        **{subject+"_W": srm.basis_list[i] for i, subject in enumerate(subjects)},
    )
    break
    for subject in subjects:
        for run, img in enumerate(layout.get(subject=subject)):
            new_path = img.path.replace(f"sub-{subject}", f"sub-{subject}SRM")
            new_path = new_path.replace("run-{i}", f"run-{run}")
