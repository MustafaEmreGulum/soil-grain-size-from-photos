"""Aynı 30 görüntü ölçüsüyle Öklid, tek PLS ekseni ve ince/iri pay tahminlerini karşılaştır."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

work = Path(__file__).resolve().parent.parent / "work"
repo = work / "vitalclick"
sys.path.insert(0, str(repo / "src"))
from run import META, feature_sets
from gsd import emd

cache = pd.read_csv(repo / "cache/features.csv")
photos = cache.loc[(cache.split == "train") & (cache.aug == 0)]
labels = pd.read_csv(work / "soil-data/Training_labels_updated.csv", index_col=0)
sets = feature_sets([c for c in cache if c not in META])
cols = list(dict.fromkeys(sets["R1"] + sets["R2"]))
assert len(cols) == 30 and len(photos) == 127
all_soils = photos.groupby("sample_id")[cols].median().loc[labels.index]
phones = ["Motorola_Edge", "Samsung_A52"]
per_phone = [photos.loc[photos.phone == p].groupby("sample_id")[cols].median() for p in phones]
paired = labels.index[labels.index.isin(per_phone[0].index) & labels.index.isin(per_phone[1].index)]
assert len(paired) == 21 and len(all_soils) == 24

def evaluate(train_frame, view_frame, ids, condition):
    records = []
    for sample in ids:
        others = ids.drop(sample)
        scaler = StandardScaler().fit(train_frame.loc[others])
        train = scaler.transform(train_frame.loc[others])
        query = scaler.transform(view_frame.loc[[sample]])
        # Yalnızca eğitim numunelerinin değişken 10 laboratuvar yüzdesini kullan.
        targets = labels.loc[others].to_numpy()
        pls = PLSRegression(n_components=1, scale=False).fit(train, targets[:, :-1])
        fractions = labels.loc[others, ["0.063", "2"]].copy()
        fractions["2"] = 100 - fractions["2"]
        points = []
        for fraction in fractions.columns:
            reg = PLSRegression(n_components=1, scale=False).fit(train, fractions[fraction])
            points.append(reg.predict(np.vstack([train, query])).ravel().clip(0, 100))
        points = np.column_stack(points)
        distances = {
            "Öklid": np.linalg.norm(train - query, axis=1),
            "PLS": np.abs(pls.transform(train) - pls.transform(query)).ravel(),
            "İnce ve iri": np.linalg.norm(points[:-1] - points[-1], axis=1),
        }
        for method, distance in distances.items():
            nearest = np.argsort(distance, kind="stable")[:3]
            curve = targets[nearest].mean(axis=0)
            truth = labels.loc[sample].to_numpy()
            lab_distance = emd(targets, truth)
            lab_best = np.argsort(lab_distance, kind="stable")[:3]
            records.append([condition, method, sample, emd(truth, curve)[0],
                            lab_distance[nearest].mean(), len(set(nearest) & set(lab_best)),
                            ", ".join(others[nearest]), *(points[-1] if method == "İnce ve iri" else [np.nan, np.nan])])
    return records

records = evaluate(all_soils, all_soils, labels.index, "24 toprak")
for a, b in [(0, 1), (1, 0)]:
    records.extend(evaluate(per_phone[a], per_phone[b], paired, phones[a] + " → " + phones[b]))
results = pd.DataFrame(records, columns=["Sınav","Mesafe","Numune","Hata","Komşu uzaklığı","İsabet","Komşular","İnce ekseni","İri ekseni"])
assert len(results) == (24 + 2 * 21) * 3 and not results.duplicated(["Sınav","Mesafe","Numune"]).any()
assert np.isfinite(results[["Hata","Komşu uzaklığı"]]).all().all()
summary = results.groupby(["Sınav","Mesafe"]).agg({"Hata":"mean","Komşu uzaklığı":"mean","İsabet":"sum"})
cross = results[results["Sınav"] != "24 toprak"].groupby("Mesafe").Hata.mean()
assert round(summary.at[("24 toprak","Öklid"), "Hata"], 3) == 39.743
assert round(cross["Öklid"], 3) == 49.594
assert round(summary.at[("24 toprak","PLS"), "Hata"], 3) == 39.722
assert round(cross["PLS"], 3) == 43.383
print(summary.round(3))
print("Çapraz ortalama", cross.round(3).to_dict())
for group, frame in results.groupby(results["Sınav"] == "24 toprak"):
    errors = frame.groupby(["Numune","Mesafe"]).Hata.mean().unstack()
    for old, new in [("Öklid","PLS"), ("PLS","İnce ve iri")]:
        gain = errors[old] - errors[new]
        print("24 toprak" if group else "21 toprak çapraz", old, "→", new, "iyileşen / eşit / kötüleşen",
              (gain > 1e-8).sum(), (gain.abs() <= 1e-8).sum(), (gain < -1e-8).sum())
print(results[(results["Sınav"] == "24 toprak") & results.Numune.isin(["H038","H405"])].round(2).to_string(index=False))
local = results.loc[results["Sınav"] == "24 toprak"]
groups = local.assign(üçlü=local["Komşular"].map(lambda text: tuple(sorted(text.split(", "))))).pivot(
    index="Numune", columns="Mesafe", values="üçlü")
print("Aynı üç komşu:", sum(a == b for a, b in zip(groups["PLS"], groups["İnce ve iri"])), "/ 24")

# Ara tahminlerin kendi hedeflerine göre hatası, yüzde puan cinsinden.
fractions = pd.DataFrame({"İnce": labels["0.063"], "İri": 100 - labels["2"]})
direct = results.loc[results.Mesafe == "İnce ve iri"]
for condition, subset in direct.groupby("Sınav"):
    ids = labels.index if condition == "24 toprak" else paired
    predicted = subset.set_index("Numune").loc[ids, ["İnce ekseni", "İri ekseni"]].copy()
    predicted.columns = fractions.columns
    measured = fractions.loc[ids]
    baseline = pd.DataFrame([measured.drop(index=s).mean() for s in ids], index=ids)
    assert predicted.shape == measured.shape and np.isfinite(predicted).all().all()
    model_error = (predicted - measured).abs()
    mean_error = (baseline - measured).abs()
    gain = mean_error - model_error
    print(condition, pd.DataFrame({"Ortalama tahmin MAE": mean_error.mean(),
                                   "Fotoğraf MAE": model_error.mean(),
                                   "Fotoğraf kazancı (%)": 100 * gain.mean() / mean_error.mean(),
                                   "Fotoğraf daha yakın": (gain > 1e-8).sum(),
                                   "Eşit": (gain.abs() <= 1e-8).sum()}).round(2).to_string())

    # Aynı saklı numune için ilk iki tahmini, seçilen üçlünün nihai eğrisiyle eşleştir.
    selected = subset.set_index("Numune").loc[ids]
    curves = []
    for sample, neighbor_names in selected["Komşular"].items():
        neighbors = neighbor_names.split(", ")
        assert len(set(neighbors)) == 3 and sample not in neighbors
        curves.append(labels.loc[neighbors].mean().to_numpy())
    final = pd.DataFrame(curves, index=ids, columns=labels.columns)
    final_fractions = pd.DataFrame({"İnce": final["0.063"], "İri": 100 - final["2"]})
    final_error = (final_fractions - measured).abs()
    delta = final_error - model_error
    print("İki pay için eşleşmiş kontrol, yüzde puan")
    print(pd.DataFrame({"Doğrudan MAE": model_error.mean(), "Komşu MAE": final_error.mean(),
                        "Doğrudan kazandı": (delta > 1e-8).sum(), "Eşit": (delta.abs() <= 1e-8).sum(),
                        "Komşu kazandı": (delta < -1e-8).sum()}).round(3).to_string())
    component = (final - labels.loc[ids]).abs().iloc[:, :-1].mul(np.diff(np.log10(labels.columns.astype(float))), axis=1)
    np.testing.assert_allclose(component.sum(axis=1), selected["Hata"], atol=1e-9, rtol=0)
    if condition == "24 toprak":
        print(pd.concat({"Ölçüm": measured, "Doğrudan": predicted, "Komşu": final_fractions}, axis=1).round(2).to_string())
        contributions = component.mean()
        print(pd.DataFrame({"EMD katkısı": contributions, "Hatadaki pay (%)": 100 * contributions / contributions.sum()})
              .sort_values("EMD katkısı", ascending=False).round(3).to_string())
