import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

out = Path("outputs")
book = json.loads((out / "toprak_doku_dersi.ipynb").read_text(encoding="utf-8"))
env = {}
for cell in [c for c in book["cells"] if c["cell_type"] == "code"][:3]:
    exec("".join(cell["source"]), env)
assert round(env["scores"]["Üç komşu"].mean(), 2) == 44.74
train = env["features"]
labels = env["labels"]
template = pd.read_csv(env["data"] / "sample_submission.csv").set_index("sample_id")
assert template.index.is_unique and template.columns.equals(labels.columns)
camera_map = {c.phone.replace(" ", "").lower(): c for c in env["cameras"].itertuples()}
rows = []
for path in sorted(env["data"].joinpath("Test_All_Photos").rglob("*")):
    if path.suffix.lower() != ".jpg":
        continue
    phone, label = path.stem.split("_HPC_", 1)
    sample = "HPC_" + re.sub(r" \(\d+\)$", "", label)
    sample = sample.replace("Münster_BS6_9,0-10m", "Muenster_BS6_9_0-10m")
    assert sample in template.index, path
    camera = camera_map[phone.replace("_", "").lower()]
    patch = env["physical_patch"](path, camera)
    rows.append([sample, *env["texture"](patch)])
assert len(rows) == 35
test = pd.DataFrame(rows, columns=["sample_id", *train.columns]).groupby("sample_id").median()
assert set(test.index) == set(template.index) and len(test) == 10
scaler = StandardScaler().fit(train)
reference = scaler.transform(train)
for sample, vector in zip(template.index, scaler.transform(test.loc[template.index])):
    distance = np.linalg.norm(reference - vector, axis=1)
    nearest = train.index[np.argsort(distance, kind="stable")[:3]]
    template.loc[sample] = labels.loc[nearest].mean()
dest = out / "submission_3nn_doku.csv"
template.to_csv(dest, float_format="%.8f")
saved = pd.read_csv(dest).set_index("sample_id")
pd.testing.assert_frame_equal(saved, template, atol=1e-8, rtol=0)
assert np.isfinite(saved).all().all() and np.all(np.diff(saved, axis=1) >= 0)
assert saved.ge(0).all().all() and saved.le(100).all().all() and saved["200"].eq(100).all()
print(dest, saved.shape, "OK")
