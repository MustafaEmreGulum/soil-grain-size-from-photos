import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

out = Path("outputs")
book = json.loads((out / "toprak_doku_dersi.ipynb").read_text(encoding="utf-8"))
env = {}
for cell in [c for c in book["cells"] if c["cell_type"] == "code"][:3]:
    exec("".join(cell["source"]), env)

target = "H038"
train = env["features"].drop(index=target)
labels = env["labels"]
assert len(train) == 23 and target not in train.index
scaler = StandardScaler().fit(train)
scaled = scaler.transform(train)
photo_features, records, patches = [], [], []
for path in env["photos"]:
    camera, sample, photo_id = path.stem.rsplit("_", 2)
    if sample != target:
        continue
    patch = env["physical_patch"](path, env["camera_by_name"][camera.lower()])
    features = env["texture"](patch)
    photo_features.append(features)
    vector = scaler.transform(pd.DataFrame([features], columns=train.columns))
    distance = np.linalg.norm(scaled - vector, axis=1)
    nearest = train.index[np.argsort(distance, kind="stable")[:3]]
    prediction = labels.loc[nearest].mean()
    error = env["emd"](labels.loc[target].to_numpy(), prediction.to_numpy())
    records.append(dict(phone=camera.replace("_"," "), photo=photo_id, neighbors=" / ".join(nearest),
                        fine=prediction["0.63"], two=prediction["2"], emd=error))
    patches.append(patch)
results = pd.DataFrame(records)
assert len(results) == 8 and np.allclose(np.median(photo_features, axis=0), env["features"].loc[target])
print(results.round(2).to_string(index=False))
print("Actual and minimum 3-neighbor prediction at 2mm:",
      labels.loc[target, "2"], labels.loc[train.index, "2"].nsmallest(3).to_dict(),
      labels.loc[train.index, "2"].nsmallest(3).mean())
print("Reference at 2 mm and EMD:", env["predictions"][target][list(labels.columns).index("2")],
      env["scores"].loc[target].tolist())

fig, axes = plt.subplots(2, 4, figsize=(10, 5.7), layout="constrained")
real = labels.loc[target,"2"]
fig.suptitle(f"H038 • 2 mm’den ince tanelerin kütle payı\nLaboratuvar ölçümü %{real:.1f}".replace(".", ","), fontsize=16)
for ax, patch, row in zip(axes.flat, patches, results.itertuples()):
    ax.imshow(patch)
    ax.set_title(f"{row.phone} · {row.photo}", fontsize=10)
    pct = f"%{row.two:.1f}".replace(".", ",")
    ax.set_xlabel(f"Model tahmini {pct}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.savefig(out / "h038_sekiz_cekim.png", dpi=125)
