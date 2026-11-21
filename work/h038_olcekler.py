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

features = env["features"]
labels = env["labels"]
cols = features.columns.tolist()
samples = ["H038", "H371", "H617"]
phones = ["Motorola_Edge", "Samsung_A52"]
train = features.drop(index="H038")
scaler = StandardScaler().fit(train)
scales = scaler.scale_
df = pd.DataFrame(env["rows"], columns=["sample", *cols])
df["phone"] = [p.stem.rsplit("_", 2)[0] for p in env["photos"]]
df["photo"] = env["photos"]
summary = {}

def summarize(name, table):
    squares = ((table.loc[samples[1:]] - table.loc["H038"]) / scales) ** 2
    delta = squares.loc["H371"] - squares.loc["H617"]
    distance = np.sqrt(squares.sum(axis=1))
    assert np.isclose(delta.sum(), distance["H371"]**2 - distance["H617"]**2)
    summary[name] = delta
    print(name, "distances", distance.round(3).to_dict(), "delta", delta.round(3).tolist())
    print("drop-one gaps", (delta.sum() - delta).round(3).tolist())

summarize("Birleşik", features)
fig = plt.figure(figsize=(8.5, 7.9), layout="constrained")
grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95])
for row, phone in enumerate(phones):
    shots = df[df.phone.eq(phone)]
    grouped = shots.groupby("sample")[cols].median()
    summarize(phone, grouped)
    for col, sample in enumerate(samples):
        subset = shots[shots["sample"].eq(sample)]
        deviation = ((subset[cols] - grouped.loc[sample]) / scales).pow(2).sum(axis=1)
        path = subset.loc[deviation.idxmin(), "photo"]
        patch = env["physical_patch"](path, env["camera_by_name"][phone.lower()])
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(patch)
        ax.set_title(f"{sample} · {phone.replace('_', ' ')}", fontsize=10)
        ax.set_axis_off()
        print(path.name, len(subset))
ax = fig.add_subplot(grid[2, :])
x = np.arange(len(cols))
for offset, (phone, color) in enumerate(zip(phones, ["#1e7165", "#ac5b20"])):
    ax.bar(x + (offset - 0.5) * 0.33, summary[phone], width=0.33,
           label=phone.replace("_", " "), color=color)
ax.set_xticks(x, [s.replace(".", ",") for s in cols])
ax.set_xlabel("Parlaklıkları karşılaştırılan iki noktanın uzaklığı (mm)")
ax.set_ylabel("Kare uzaklık farkı")
ax.set_title("Pozitif değer, o ölçekte H617’nin H038’e daha yakın göründüğünü gösterir", fontsize=10)
ax.axhline(0, color="#444444", linewidth=0.8)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
fig.savefig(out / "h038_alti_olcek.png", dpi=130)
for sample in samples:
    print(sample, "2 mm", labels.loc[sample, "2"], "EMD", env["emd"](labels.loc["H038"].to_numpy(), labels.loc[sample].to_numpy()))
