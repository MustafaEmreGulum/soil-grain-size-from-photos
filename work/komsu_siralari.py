import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

book = json.loads(Path("outputs/toprak_doku_dersi.ipynb").read_text(encoding="utf-8"))
env = {}
for cell in [c for c in book["cells"] if c["cell_type"] == "code"][:3]:
    exec("".join(cell["source"]), env)
features, labels = env["features"], env["labels"]
records = []
for sample in labels.index:
    train = features.drop(index=sample)
    scaler = StandardScaler().fit(train)
    delta = scaler.transform(train) - scaler.transform(features.loc[[sample]])
    photo_dist = np.linalg.norm(delta, axis=1)
    lab_dist = np.abs((labels.loc[train.index] - labels.loc[sample]).to_numpy()[:, :-1]) @ env["widths"]
    photo = train.index[np.argsort(photo_dist, kind="stable")].tolist()
    lab = train.index[np.argsort(lab_dist, kind="stable")].tolist()
    assert photo[:3] == env["neighbors"][sample]
    assert len(set(lab_dist)) == len(lab_dist)
    overlap = len(set(photo[:3]) & set(lab[:3]))
    records.append((sample, overlap, spearmanr(photo_dist, lab_dist).statistic))
    if sample == "H038":
        print("H038", [(n, lab.index(n)+1) for n in photo[:3]], [(n,photo.index(n)+1) for n in lab[:3]])
result = pd.DataFrame(records, columns=["sample", "overlap", "rho"])
print(result.to_string(index=False))
print("totals", result.overlap.sum(), result.overlap.value_counts().to_dict(), "mean rho", result.rho.mean())
