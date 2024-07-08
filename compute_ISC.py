import os
from pathlib import Path

import matplotlib as mpl
import nibabel as nib
import numpy as np
import pandas as pd
from bids import BIDSLayout
from nilearn import plotting
from nilearn.glm import fdr_threshold
from scipy.stats import norm, ttest_1samp
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from src.metrics import corr

lang = "EN"
n_runs = 9


def average(run):
    layout = BIDSLayout("data/li2022/derivatives", validate=False, is_derivative=True)
    subjects = [s for s in layout.get_subjects() if s.startswith(lang)]
    a = None
    affine = None
    img = layout.get(subject=subjects[0])[0]
    new_path = img.path.replace("sub-" + subjects[0], "sub-mean" + lang)
    new_path = Path(new_path.replace(f"run-{img.entities['run']}", f"run-{run+1:02d}"))
    if new_path.exists():
        return
    for subject in tqdm(subjects, desc=f"Run {run}", leave=False):
        imgs = sorted([f.path for f in layout.get(subject=subject)])
        assert len(imgs) == n_runs
        img = nib.load(imgs[run])
        if a is None:
            a = img.get_fdata()
            affine = img.affine
        else:
            a += img.get_fdata()
    new_path.parent.mkdir(parents=True, exist_ok=True)
    a /= len(subjects)
    nib.Nifti1Image(a, affine).to_filename(new_path)
    print(new_path)


for run in tqdm(range(n_runs)):
    average(run)

# # ISC

lang = "EN"

layout = BIDSLayout("data/li2022/derivatives", validate=False, is_derivative=True)
subjects = layout.get_subjects()
subjects = [s for s in subjects if s.startswith(lang)]
n_subjects = len(subjects)


def read(path):
    return path, nib.load(path).get_fdata()


mean_imgs = sorted([f.path for f in layout.get(subject="meanEN")])
mean_imgs = sorted(process_map(read, mean_imgs))
mean_imgs = np.concatenate([m[1] for m in mean_imgs], axis=-1) * n_subjects

corrs = []
for subject in tqdm(subjects):
    subject_imgs = sorted([f.path for f in layout.get(subject=subject)])
    subject_imgs = sorted(process_map(read, subject_imgs, leave=False))
    subject_imgs = np.concatenate([m[1] for m in subject_imgs], axis=-1)
    c = corr(mean_imgs - subject_imgs, subject_imgs, axis=-1)
    corrs.append(c)
corrs = np.stack(corrs)

pvalues = np.nan_to_num(ttest_1samp(np.arctanh(corrs), popmean=0, axis=0).pvalue, nan=1)
zscores = norm.ppf(1 - pvalues)
thresh = fdr_threshold(zscores.reshape(-1), 5e-2)
signif = np.where(zscores > thresh)

df = pd.DataFrame(signif, index=["x", "y", "z"]).T
df["zscore"] = zscores[signif]
df.to_csv(f"data/li2022/ISC_voxels_{lang}.csv", index=False)
np.save(
    f"data/li2022/ISC_voxels_{lang}.npy",
    {"pval": pvalues, "zscore": zscores, "thresh": thresh, "signif": signif},
)

for subject in tqdm(subjects):
    imgs = sorted([f.path for f in layout.get(subject=subject)])
    assert len(imgs) == n_runs
    for img_file in imgs:
        img = nib.load(img_file)
        img = nib.Nifti1Image(img.get_fdata()[signif], img.affine)
        new_img_file = img_file.replace("sub-" + subject, "sub-" + subject + "ISC")
        Path(new_img_file).parent.mkdir(parents=True, exist_ok=True)
        img.to_filename(new_img_file)

os.makedirs("figs", exist_ok=True)

cmap = mpl.cm.jet
max_zscore = df.zscore.abs().max()
norm = mpl.colors.Normalize(vmin=-max_zscore, vmax=max_zscore)
cbar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
colors = cmap(norm(df.zscore))

plotting.view_connectome(
    adjacency_matrix=np.diag(df.zscore),
    node_coords=df[["x", "y", "z"]],
    node_color=colors,
    edge_cmap=cmap,
).save_as_html("figs/li2022_ics.html")
