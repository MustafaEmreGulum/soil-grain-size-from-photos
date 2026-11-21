"""V3 ve ince/iri KNN eğrilerinin önceden belirlenmiş eşit ağırlıklı ortalaması."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

out = Path(__file__).resolve().parent
work = out.parent / "work"
repo = work / "vitalclick"
sys.path.insert(0, str(repo / "src"))
import run as published
from gsd import COLS, curve_to_logq, emd

cache = pd.read_csv(repo / "cache/features.csv")
labels = pd.read_csv(work / "soil-data/Training_labels_updated.csv", index_col="sample_id")[COLS]
training = cache[cache.split == "train"]
original = training[training.aug == 0]
columns = [c for c in cache if c not in published.META]
sets = published.feature_sets(columns)
keys = list(dict.fromkeys(sets["R1"] + sets["R2"]))
published.LINEAR_ONLY = True

def predict(refs, query):
    train = refs[refs.aug == 0].groupby("sample_id")[keys].median()
    ids = labels.index[labels.index.isin(train.index)]
    train = train.loc[ids]
    q = query.groupby("sample_id")[keys].median()
    assert not ids.intersection(q.index).size and (query.aug == 0).all()
    y = labels.loc[ids]
    # V3 özgün ve türetilmiş çekimlerle eğitilir. Testte yalnızca özgün çekimler kullanılır.
    quantiles = pd.DataFrame(curve_to_logq(y.values), index=ids)
    order, _, curves = published.fit_predict(refs, query, quantiles, columns)
    v3 = pd.DataFrame(curves, index=order, columns=COLS).loc[q.index]
    scaler = StandardScaler().fit(train)
    x = scaler.transform(train)
    both = np.vstack([x, scaler.transform(q)])
    coords = []
    for target in [y["0.063"], 100-y["2"]]:
        model = PLSRegression(n_components=1, scale=False).fit(x, target)
        coords.append(model.predict(both).ravel().clip(0, 100))
    points = np.column_stack(coords)
    neighbors = []
    for point in points[len(ids):]:
        d = np.linalg.norm(points[:len(ids)] - point, axis=1)
        neighbors.append(y.iloc[np.argsort(d, kind="stable")[:3]].mean())
    knn = pd.DataFrame(neighbors, index=q.index, columns=COLS)
    blend = (v3 + knn) / 2
    assert np.isfinite(blend).all().all() and blend.ge(0).all().all() and blend.le(100).all().all()
    assert (blend["200"] == 100).all() and np.all(np.diff(blend, axis=1) >= -1e-9)
    return {"v3": v3, "KNN": knn, "Karışım": blend}


if __name__ == "__main__":
    phones = ["Motorola_Edge", "Samsung_A52"]
    paired = labels.index
    for phone in phones:
        paired = paired[paired.isin(original.loc[original.phone == phone,"sample_id"])]
    assert len(paired) == 21 and len(labels) == 24 and len(keys) == 30
    blocks = {}
    for name, ids, source, view in [("24 toprak", labels.index, None, None),
                                    ("Motorola → Samsung", paired, *phones),
                                    ("Samsung → Motorola", paired, *phones[::-1])]:
        tr = training if source is None else training[training.phone == source]
        te = original if view is None else original[original.phone == view]
        rows = []
        for sample in ids:
            pool = ids.drop(sample)
            refs = tr[tr.sample_id.isin(pool)]
            assert set(refs.sample_id) == set(pool)
            predictions = predict(refs, te[te.sample_id == sample])
            rows.append({k: emd(labels.loc[sample], c.loc[sample])[0] for k,c in predictions.items()})
        blocks[name] = pd.DataFrame(rows, index=ids)
    blocks["Çapraz ortalama"] = (blocks["Motorola → Samsung"] + blocks["Samsung → Motorola"]) / 2
    for name, block in blocks.items():
        gain = block["v3"] - block["Karışım"]
        assert (block["Karışım"] <= (block["v3"] + block["KNN"])/2 + 1e-8).all()
        print(name, block.mean().round(4).to_dict(), "v3'e göre kazanç", round(gain.mean(),4),
              "iyileşen/eşit/kötüleşen", (gain>1e-8).sum(), (gain.abs()<=1e-8).sum(), (gain < -1e-8).sum())
        if name in ["24 toprak", "Çapraz ortalama"]:
            print("Ortanca kazanç", round(gain.median(),3), "en büyük kazanç/kayıp",
                  gain.idxmax(), round(gain.max(),3), gain.idxmin(), round(gain.min(),3))
    np.testing.assert_allclose(blocks["24 toprak"][["v3","KNN"]].mean(), [38.758,40.144], atol=.00051, rtol=0)
    np.testing.assert_allclose(blocks["Çapraz ortalama"][["v3","KNN"]].mean(), [42.409,42.813], atol=.00051, rtol=0)
    if all(blocks[n]["Karışım"].mean() < blocks[n]["v3"].mean() for n in ["24 toprak","Çapraz ortalama"]):
        test = cache[(cache.split == "test") & (cache.aug == 0)]
        template = pd.read_csv(work / "soil-data/sample_submission.csv", index_col="sample_id")
        pred = predict(training, test)["Karışım"]
        assert len(test) == 35 and pred.shape == template.shape == (10,11)
        assert set(pred.index) == set(template.index) and pred.columns.equals(template.columns)
        pred = pred.loc[template.index]
        dest = out / "submission_v3_knn_50_50.csv"
        pred.to_csv(dest, float_format="%.8f")
        saved = pd.read_csv(dest, index_col="sample_id")
        pd.testing.assert_frame_equal(saved, pred, rtol=0, atol=1e-8)
        assert np.all(np.diff(saved, axis=1) >= 0) and saved["200"].eq(100).all()
        print("Doğrulanmış yerel aday:", dest)
