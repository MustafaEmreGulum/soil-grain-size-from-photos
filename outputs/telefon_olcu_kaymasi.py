"""Motorola/Samsung eşlerinde ölçü kayması ve beş ölçü çıkarma kontrolü."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

work = Path(__file__).resolve().parent.parent / "work"
repo = work / "vitalclick"
sys.path.insert(0, str(repo / "src"))
from gsd import COLS, emd
from run import META, feature_sets

labels = pd.read_csv(work / "soil-data/Training_labels_updated.csv", index_col=0)[COLS]
cache = pd.read_csv(repo / "cache/features.csv")
photos = cache[(cache.split == "train") & (cache.aug == 0)]
sets = feature_sets([c for c in cache if c not in META])
keys = list(dict.fromkeys(sets["R1"] + sets["R2"]))
full = photos.groupby("sample_id")[keys].median().loc[labels.index]
def phone_frame(phone):
    return photos.loc[photos.phone == phone].groupby("sample_id")[keys].median()

motorola = phone_frame("Motorola_Edge")
samsung = phone_frame("Samsung_A52")
paired = labels.index[labels.index.isin(motorola.index) & labels.index.isin(samsung.index)]
assert len(paired) == 21 and len(full) == 24 and len(keys) == 30

def paired_shifts(ids):
    m, s = motorola.loc[ids], samsung.loc[ids]
    # Telefonlar arası ortalama fark, paydadaki topraklar arası yayılımı şişirmesin.
    scale = np.sqrt((m.var() + s.var()) / 2)
    assert scale.gt(0).all()
    changes = (s - m) / scale
    return pd.DataFrame({"Mutlak fark": changes.abs().median(),
                         "Yönlü fark": changes.median(),
                         "Samsung daha yüksek": changes.gt(0).sum()}).sort_values(
                             "Mutlak fark", ascending=False, kind="stable")

ranking = paired_shifts(paired)
print("21 eşte standartlaştırılmış ortanca fark (Samsung eksi Motorola)")
print(ranking.round(3).to_string())
records, removed = [], []
for setting, src, dst, ids in [("24 toprak", full, full, labels.index),
                               ("Motorola → Samsung", motorola, samsung, paired),
                               ("Samsung → Motorola", samsung, motorola, paired)]:
    for sample in ids:
        others = ids.drop(sample)
        calibration = paired[paired.isin(others)]
        assert sample not in calibration and len(calibration) >= 20
        drop = paired_shifts(calibration).index[:5]
        if setting == "Motorola → Samsung":
            removed.extend(drop)
        variants = [("Tam", keys), ("25 ölçü", [k for k in keys if k not in drop])]
        if setting != "24 toprak":
            variants.append(("Ortanca düzeltme", keys))
        for name, cols in variants:
            train = src.loc[others, cols]
            scaler = StandardScaler().fit(train)
            x = scaler.transform(train)
            query = dst.loc[[sample], cols]
            if name == "Ortanca düzeltme":
                offset = (dst.loc[others, cols] - src.loc[others, cols]).median()
                query = query - offset
            q = scaler.transform(query)
            targets = labels.loc[others]
            points = []
            for y in [targets["0.063"], 100 - targets["2"]]:
                est = PLSRegression(n_components=1, scale=False).fit(x, y)
                points.append(est.predict(np.vstack([x, q])).ravel().clip(0, 100))
            axes = np.column_stack(points)
            for method, distance in [("Öklid", np.linalg.norm(x-q, axis=1)),
                                     ("İnce/iri", np.linalg.norm(axes[:-1]-axes[-1], axis=1))]:
                chosen = others[np.argsort(distance, kind="stable")[:3]]
                curve = labels.loc[chosen].mean().to_numpy()
                records.append([setting, sample, name, method, emd(labels.loc[sample].values, curve)[0], tuple(sorted(chosen))])
df = pd.DataFrame(records, columns=["Sınav", "Numune", "Ölçüler", "Model", "Hata", "Komşular"])
assert len(df) == 4*24+6*21*2 and not df.duplicated(["Sınav","Numune","Ölçüler","Model"]).any()
scores = df.pivot(index=["Sınav","Numune","Model"], columns="Ölçüler", values="Hata")
scores["Kazanç"] = scores["Tam"] - scores["25 ölçü"]
print("Tura göre ilk beşe girme sayısı:", pd.Series(removed).value_counts().to_dict())
means = scores.groupby(["Sınav","Model"]).mean()
assert round(means.loc[("24 toprak","Öklid"), "Tam"], 3) == 39.743
assert round(means.loc[("24 toprak","İnce/iri"), "Tam"], 3) == 40.144
cross = scores.drop("24 toprak", level="Sınav").groupby(["Numune","Model"]).mean()
assert round(cross.xs("İnce/iri",level="Model")["Tam"].mean(), 3) == 42.813
print(means.round(3))
print("İki çapraz yönün ortalaması")
print(cross.groupby("Model").mean().round(3))
for name, group in cross.groupby("Model"):
    for variant in ["25 ölçü","Ortanca düzeltme"]:
        delta = group["Tam"] - group[variant]
        print(name, variant, "iyileşen/eşit/kötüleşen", (delta>1e-8).sum(), (delta.abs()<=1e-8).sum(), (delta< -1e-8).sum())
