"""Aynı işleme ve filtrelerle merkez kesit boyutunu eşitleme kontrolü."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

work = Path(__file__).resolve().parent.parent / "work"
repo = work / "vitalclick"
sys.path.insert(0, str(repo / "src"))
from features import CANON_PPM, load_canonical
from gsd import COLS, emd
from run import META, feature_sets

keys = ["log_22.0", "log_16.0", "open_9.0"]
cache = pd.read_csv(repo / "cache/features.csv")
phones = ["Motorola_Edge", "Samsung_A52"]
original = cache[(cache.split == "train") & (cache.aug == 0) & cache.phone.isin(phones)]
paired = original.groupby("sample_id").phone.nunique()
original = original[original.sample_id.isin(paired[paired == 2].index)]
paths = {p.name: p for p in (work / "soil-data").rglob("*.jpg")}
cv2.setNumThreads(2)

def measure(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = gray - cv2.GaussianBlur(gray, (0, 0), 40 * CANON_PPM)
    gn = g / (float(g.std()) + 1e-6)
    values = []
    for scale in [22.0, 16.0]:
        sigma = scale * CANON_PPM
        lap = cv2.Laplacian(cv2.GaussianBlur(gn, (0, 0), sigma), cv2.CV_32F) * sigma**2
        values.append(np.log(float(np.mean(np.abs(lap))) + 1e-6))
    gray8 = np.clip(gn * 40 + 128, 0, 255).astype(np.uint8)
    small = cv2.resize(gray8, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    radius = round(9 * CANON_PPM / 2)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    opened = cv2.morphologyEx(small, cv2.MORPH_OPEN, disk)
    values.append(float((small.astype(np.float32) - opened).mean() / 40))
    return values

rows = []
short_sides = {p: [] for p in phones}
side = round(100 * CANON_PPM)
for row in original.itertuples():
    rgb, ppm = load_canonical(str(paths[row.file]), row.phone)
    short_sides[row.phone].append(min(rgb.shape[:2]) / CANON_PPM)
    with Image.open(paths[row.file]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        source_side = round(100 * ppm)
        w, h = image.size
        assert min(w, h) >= source_side
        x, y = (w-source_side)//2, (h-source_side)//2
        square = image.crop((x, y, x+source_side, y+source_side))
        resampling = Image.Resampling.LANCZOS if CANON_PPM < ppm else Image.Resampling.BICUBIC
        fixed = np.array(square.resize((side, side), resampling))
    for mode, patch in [("Özgün", rgb), ("100 mm", fixed)]:
        rows.append([row.file, row.sample_id, row.phone, mode, *measure(patch)])
frame = pd.DataFrame(rows, columns=["file", "sample_id", "phone", "Kesit", *keys])
ref = original.set_index("file")[keys]
rerun = frame[frame.Kesit == "Özgün"].set_index("file").loc[ref.index, keys]
print("Çekim sayısı", len(ref), "yeniden hesap / önbellek en büyük mutlak farkı", (rerun-ref).abs().max().to_dict())
print("Özgün kısa kenar aralığı (mm)", {p:(min(v),max(v)) for p,v in short_sides.items()})
for mode in ["önbellek", "Özgün", "100 mm"]:
    dataset = original if mode == "önbellek" else frame[frame.Kesit == mode]
    table = dataset.groupby(["phone","sample_id"])[keys].median()
    m, s = [table.loc[p] for p in phones]
    assert m.index.equals(s.index) and len(m)==21
    scale = np.sqrt((m.var()+s.var())/2)
    if mode == "Özgün":
        initial_scale = scale
    absolute = (s-m).abs().median()
    print(mode, "standart fark", (absolute/scale).round(3).to_dict(),
          "ham mutlak fark", absolute.round(5).to_dict(), "Samsung yüksek", (s>m).sum().to_dict())
    if mode == "100 mm":
        print("Özgün ölçekle 100 mm farkı", (absolute/initial_scale).round(3).to_dict())

# Her eğitim turunda da ilk üç sıradaki aynı üç özelliğin görüntü kesitini değiştir.
sets = feature_sets([c for c in cache if c not in META])
all_keys = list(dict.fromkeys(sets["R1"] + sets["R2"]))
labels = pd.read_csv(work / "soil-data/Training_labels_updated.csv", index_col=0)[COLS]
ids = labels.index[labels.index.isin(original.sample_id)]
assert len(ids) == 21 and len(all_keys) == 30
records = []
for mode in ["önbellek", "Özgün", "100 mm"]:
    data = original.set_index("file").copy()
    if mode != "önbellek":
        data[keys] = frame[frame.Kesit == mode].set_index("file").loc[data.index, keys]
    table = data.groupby(["phone", "sample_id"])[all_keys].median()
    for source, other in [phones, phones[::-1]]:
        for sample in ids:
            pool = ids.drop(sample)
            train = table.loc[source].loc[pool]
            scaler = StandardScaler().fit(train)
            x = scaler.transform(train)
            q = scaler.transform(table.loc[other].loc[[sample]])
            y = labels.loc[pool]
            axes = []
            for frac in [y["0.063"], 100-y["2"]]:
                est = PLSRegression(n_components=1, scale=False).fit(x, frac)
                axes.append(est.predict(np.vstack([x, q])).ravel().clip(0, 100))
            z = np.column_stack(axes)
            for kind, dist in [("Öklid", np.linalg.norm(x-q, axis=1)),
                               ("İnce/iri", np.linalg.norm(z[:-1]-z[-1], axis=1))]:
                near = pool[np.argsort(dist, kind="stable")[:3]]
                curve = labels.loc[near].mean().values
                records.append([mode, source, sample, kind, emd(labels.loc[sample], curve)[0]])
scores = pd.DataFrame(records, columns=["Kesit","Eğitim","Numune","Model","Hata"])
assert len(scores) == 3*2*21*2 and np.isfinite(scores.Hata).all()
overview = scores.pivot_table(index=["Eğitim","Model"], columns="Kesit", values="Hata")
print(overview.round(3).to_string())
print(overview.groupby("Model").mean().round(3).to_string())
assert round(overview.groupby("Model").mean().at["İnce/iri","önbellek"],3) == 42.813
print("Eşleşmiş numunelerde 100 mm kazancı (iki yön ortalaması)")
for name, group in scores.groupby("Model"):
    p = group.pivot_table(index="Numune", columns="Kesit",values="Hata")
    delta = p["Özgün"] - p["100 mm"]
    print(name, (delta>1e-8).sum(), (delta.abs()<=1e-8).sum(), (delta < -1e-8).sum())
