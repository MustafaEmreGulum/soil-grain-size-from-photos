"""Elek payları ve tüm eğri. Kökteki work/ klasöründe veri ve vitalclick gerekir."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import isotonic_regression
from sklearn.preprocessing import StandardScaler

work = Path(__file__).resolve().parent.parent / "work"
repo = work / "vitalclick"
sys.path.insert(0, str(repo / "src"))
import run as published
from gsd import COLS, DLOG, curve_to_logq, emd
from run import META, feature_sets

labels = pd.read_csv(work / "soil-data/Training_labels_updated.csv", index_col=0)[COLS]
cache = pd.read_csv(repo / "cache/features.csv")
original = cache[(cache.split == "train") & (cache.aug == 0)]
all_columns = [c for c in cache.columns if c not in META]
sets = feature_sets(all_columns)
keys = list(dict.fromkeys(sets["R1"] + sets["R2"]))
full = original.groupby("sample_id")[keys].median().loc[labels.index]
phones = ["Motorola_Edge", "Samsung_A52"]
motorola, samsung = [original[original.phone == p].groupby("sample_id")[keys].median() for p in phones]
paired = labels.index[labels.index.isin(motorola.index) & labels.index.isin(samsung.index)]
assert len(keys) == 30 and len(original) == 127 and len(full) == 24 and len(paired) == 21

def evaluate(train_frame, query_frame, ids):
    raw, direct, knn, mean = [], [], [], []
    for sample in ids:
        pool = ids.drop(sample)
        scaler = StandardScaler().fit(train_frame.loc[pool])
        train = scaler.transform(train_frame.loc[pool])
        query = scaler.transform(query_frame.loc[[sample]])
        targets = labels.loc[pool]
        # 200 mm'de yüzde daima 100, diğer on elek için bağımsız tek bileşenli PLS.
        estimates = []
        for sieve in COLS[:-1]:
            model = PLSRegression(n_components=1, scale=False).fit(train, targets[sieve])
            estimates.append(model.predict(np.vstack([train, query])).ravel().clip(0, 100))
        estimates = np.column_stack(estimates)
        # 2 mm altı yerine 100 eksi bu payı kullanmak Öklid uzaklığını değiştirmez.
        axes = estimates[:, [COLS.index("0.063"), COLS.index("2")]]
        distance = np.linalg.norm(axes[:-1] - axes[-1], axis=1)
        nearest = np.argsort(distance, kind="stable")[:3]
        knn.append(targets.iloc[nearest].mean().to_numpy())
        mean.append(targets.mean().to_numpy())
        raw.append(np.append(estimates[-1], 100))
        direct.append(np.append(isotonic_regression(estimates[-1]), 100))
    truth = labels.loc[ids].to_numpy()
    direct, knn, raw = map(np.array, [direct, knn, raw])
    assert np.isfinite(direct).all() and (np.diff(direct) >= -1e-9).all()
    assert direct.min() >= 0 and direct.max() <= 100 and (direct[:, -1] == 100).all()
    print("Elek başına mutlak hata ve ham doğrudan tahminin kazandığı numune sayısı")
    table = pd.DataFrame(index=COLS[:-1])
    for name, curves in [("Ortalama", mean), ("Komşu", knn), ("Ham", raw), ("Düzeltilmiş", direct)]:
        table[name] = np.abs(truth - curves).mean(axis=0)[:-1]
    gain = np.abs(truth - knn) - np.abs(truth - raw)
    table["Ham kazandı"] = (gain[:, :-1] > 1e-8).sum(axis=0)
    print(table.round(3).to_string())
    combined = (knn + direct) / 2
    assert (np.diff(combined) >= -1e-9).all() and (combined[:, -1] == 100).all()
    errors = pd.DataFrame({"Komşu": emd(truth, knn), "Ham": emd(truth, raw), "Doğrudan": emd(truth, direct),
                           "Eşit karışım": emd(truth, combined)}, index=ids)
    assert (errors["Eşit karışım"] <= (errors["Komşu"] + errors["Doğrudan"]) / 2 + 1e-9).all()
    np.testing.assert_allclose(table["Düzeltilmiş"] @ DLOG, errors["Doğrudan"].mean())
    print("Monotonluk düzeltmesi gereken satır:", int((np.diff(raw) < -1e-8).any(axis=1).sum()))
    return errors

def v3_errors(ids, source=None, view=None, with_augmentation=True):
    published.LINEAR_ONLY = True
    training = cache[cache.split == "train"] if with_augmentation else original
    held = original
    if source is not None:
        training = training[training.phone == source]
        held = held[held.phone == view]
    errors = {}
    for sample in ids:
        pool = ids.drop(sample)
        refs = training[training.sample_id.isin(pool)]
        query = held[held.sample_id == sample]
        assert set(refs.sample_id) == set(pool) and len(query) > 0
        quantiles = pd.DataFrame(curve_to_logq(labels.loc[pool].to_numpy()), index=pool)
        returned, _, curve = published.fit_predict(refs, query, quantiles, all_columns)
        assert list(returned) == [sample]
        errors[sample] = emd(labels.loc[sample].to_numpy(), curve)[0]
    return pd.Series(errors)

blocks = {}
for name, train, query, ids in [("24 toprak", full, full, labels.index),
                                ("Motorola → Samsung", motorola, samsung, paired),
                                ("Samsung → Motorola", samsung, motorola, paired)]:
    print(name)
    blocks[name] = evaluate(train, query, ids)
blocks["24 toprak"]["v3"] = v3_errors(labels.index)
blocks["Motorola → Samsung"]["v3"] = v3_errors(paired, *phones)
blocks["Samsung → Motorola"]["v3"] = v3_errors(paired, *phones[::-1])
blocks["24 toprak"]["v3 özgün"] = v3_errors(labels.index, with_augmentation=False)
blocks["Motorola → Samsung"]["v3 özgün"] = v3_errors(paired, *phones, with_augmentation=False)
blocks["Samsung → Motorola"]["v3 özgün"] = v3_errors(paired, *phones[::-1], with_augmentation=False)
assert round(blocks["24 toprak"]["v3"].mean(), 3) == 38.758
blocks["Çapraz ortalama"] = (blocks["Motorola → Samsung"] + blocks["Samsung → Motorola"]) / 2
for name, errors in blocks.items():
    change = errors["Komşu"] - errors["Doğrudan"]
    print(name, errors.mean().round(3).to_dict(), "iyileşen/eşit/kötüleşen",
          (change > 1e-8).sum(), (change.abs() <= 1e-8).sum(), (change < -1e-8).sum(),
          "ortanca kazanç", round(change.median(), 3))
    for reference in ["Komşu", "Doğrudan", "v3"]:
        gain = errors[reference] - errors["Eşit karışım"]
        print("Karışım /", reference, "iyileşen/eşit/kötüleşen",
              (gain > 1e-8).sum(), (gain.abs() <= 1e-8).sum(), (gain < -1e-8).sum())
    gain = errors["v3 özgün"] - errors["v3"]
    print("Yapay çekim eklenince iyileşen/eşit/kötüleşen",
          (gain > 1e-8).sum(), (gain.abs() <= 1e-8).sum(), (gain < -1e-8).sum())
    if name == "24 toprak":
        print(errors.assign(Kazanç=change).round(2).sort_values("Kazanç").to_string())
assert round(blocks["24 toprak"]["Komşu"].mean(), 3) == 40.144
assert round(blocks["Çapraz ortalama"]["Komşu"].mean(), 3) == 42.813
