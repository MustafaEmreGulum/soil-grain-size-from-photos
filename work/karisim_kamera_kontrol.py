import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

book = json.loads(Path("outputs/toprak_ridge_deneyi.ipynb").read_text(encoding="utf-8"))
env = {}
with contextlib.redirect_stdout(io.StringIO()):
    for cell in [c for c in book["cells"] if c["cell_type"] == "code"][:4]:
        exec("".join(cell["source"]), env)
labels = env["labels"]
knn = pd.DataFrame.from_dict(env["predictions"], orient="index", columns=labels.columns).loc[labels.index]
pooled = (knn + env["ridge"]) / 2
assert np.all(np.diff(pooled, axis=1) >= -1e-9) and pooled["200"].eq(100).all()
scores = env["scores"].copy()
scores["50/50"] = np.abs((pooled-labels).to_numpy()[:, :-1]) @ env["widths"]
print(scores.mean().round(3))
print("wins", (scores["50/50"] < scores.iloc[:,1]).sum(), "H038", scores.loc["H038"].to_list())

# Compare the same soil observed by two phones, with exactly the same training set.
df = pd.DataFrame(env["rows"], columns=["sample", *env["features"].columns])
phones = [path.stem.rsplit("_", 2)[0] for path in env["photos"]]
df["phone"] = phones
print(df["phone"].value_counts())
groups = df.groupby(["sample", "phone"]).median()
gaps = []
for sample in labels.index:
    shots = groups.loc[sample]
    if not {"Motorola_Edge", "Samsung_A52"}.issubset(shots.index):
        continue
    train = env["features"].drop(index=sample)
    scaler = StandardScaler().fit(train)
    camera_estimates = []
    for phone in ["Motorola_Edge", "Samsung_A52"]:
        distance = np.linalg.norm(scaler.transform(train) - scaler.transform(shots.loc[[phone]]), axis=1)
        near = train.index[np.argsort(distance, kind="stable")[:3]]
        camera_estimates.append(labels.loc[near, "2"].mean())
    gaps.append(abs(camera_estimates[1] - camera_estimates[0]))
print("paired count",len(gaps),"median 2mm gap",np.median(gaps),"range",min(gaps),max(gaps))
