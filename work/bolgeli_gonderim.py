import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

out = Path("outputs")
notebook = json.loads((out / "toprak_bolge_deneyi.ipynb").read_text(encoding="utf-8"))
env = {}
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
for cell in code_cells[:3]:
    exec("".join(cell["source"]), env)
assert round(env["scores"]["Bölgeli"].mean(), 3) == 44.184

train = env["features"]
labels = env["labels"]
data = env["data"]
template = pd.read_csv(data / "sample_submission.csv").set_index("sample_id")
assert template.shape == (10, 11) and template.index.is_unique
assert template.columns.equals(labels.columns)
cameras = {row.phone.replace(" ", "").lower(): row for row in env["cameras"].itertuples()}
rows = []
for path in sorted((data / "Test_All_Photos").rglob("*")):
    if path.suffix.lower() != ".jpg":
        continue
    phone, label = path.stem.split("_HPC_", 1)
    sample = "HPC_" + re.sub(r" \(\d+\)$", "", label)
    sample = sample.replace("Münster_BS6_9,0-10m", "Muenster_BS6_9_0-10m")
    assert sample in template.index
    patch = env["physical_patch"](path, cameras[phone.replace("_", "").lower()])
    gray = env["normalized_gray"](patch)
    texture = env["texture_at_six_scales"]
    regions = [gray[y:y+100, x:x+100] for y in range(0, 400, 100) for x in range(0, 400, 100)]
    spread = np.std([texture(region) for region in regions], axis=0)
    rows.append([sample, *texture(gray), *spread])
test = pd.DataFrame(rows, columns=["sample_id", *train.columns]).groupby("sample_id").median()
assert len(rows) == 35 and set(test.index) == set(template.index)
test = test.loc[template.index]
assert np.isfinite(test).all().all()

def predict(columns):
    scaler = StandardScaler().fit(train[columns])
    fitted = scaler.transform(train[columns])
    curves = []
    for vector in scaler.transform(test[columns]):
        distances = np.linalg.norm(fitted - vector, axis=1)
        nearest = train.index[np.argsort(distances, kind="stable")[:3]]
        curves.append(labels.loc[nearest].mean().to_numpy())
    return pd.DataFrame(curves, index=template.index, columns=template.columns)

first = pd.read_csv(out / "submission_3nn_doku.csv").set_index("sample_id")
pd.testing.assert_frame_equal(predict(env["scales"]), first, atol=1e-8, rtol=0)
predicted = predict(train.columns)
dest = out / "submission_3nn_16_bolge.csv"
predicted.to_csv(dest, float_format="%.8f")
saved = pd.read_csv(dest).set_index("sample_id")
pd.testing.assert_frame_equal(saved, predicted, atol=1e-8, rtol=0)
assert saved.index.equals(template.index) and saved.columns.equals(template.columns)
assert np.isfinite(saved).all().all() and np.all(np.diff(saved, axis=1) >= 0)
assert saved.ge(0).all().all() and saved.le(100).all().all() and saved["200"].eq(100).all()
print("CSV kontrolü tamamlandı:", dest, saved.shape)
print("İlk gönderime göre değişen satırlar:", (saved != first).any(axis=1).sum())
