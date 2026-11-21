"""Test fotoğraflarının eğitimdeki 3-komşu mesafesine göre kapsamı ve yakın referans kontrolü."""
from pathlib import Path
from itertools import combinations
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

work = Path(__file__).resolve().parent.parent / "work"
repo = work / "vitalclick"
sys.path.insert(0, str(repo / "src"))
import run as published
from gsd import COLS, curve_to_logq, emd

data = pd.read_csv(repo / "cache/features.csv")
cols = [k for k in data if k not in published.META]
sets = published.feature_sets(cols)
keys = list(dict.fromkeys(sets["R1"] + sets["R2"]))
train_rows = data[data.split == "train"]
original = train_rows[train_rows.aug == 0]
test_rows = data[(data.split == "test") & (data.aug == 0)]
labels = pd.read_csv(work / "soil-data/Training_labels_updated.csv", index_col=0)[COLS]
train = original.groupby("sample_id")[keys].median().loc[labels.index]
test = test_rows.groupby("sample_id")[keys].median()
assert len(train) == 24 and len(test) == 10 and len(keys) == 30
assert set(test.index) == set(pd.read_csv(work / "soil-data/sample_submission.csv").sample_id)

def nearby(ref, query):
    sc = StandardScaler().fit(ref)
    d = np.linalg.norm(sc.transform(ref) - sc.transform(query), axis=1)
    return pd.Series(d, index=ref.index).sort_values(kind="stable")

def coverage(ref, query):
    rank = nearby(ref, query)
    outside = ((query.iloc[0] < ref.min()) | (query.iloc[0] > ref.max())).sum()
    return [rank.iloc[:3].mean(), outside, ", ".join(rank.index[:3])]

for label, queries in [("Eğitim", train), ("Test", test)]:
    rows = []
    for s in queries.index:
        ref = train.drop(index=s) if label == "Eğitim" else train
        rows.append(coverage(ref, queries.loc[[s]]))
    tab = pd.DataFrame(rows,index=queries.index,columns=["Üç komşuya uzaklık","Dışarıda","Komşular"])
    print(label, tab.iloc[:,:2].agg(["min","median","max"]).round(3).to_string())
    if label == "Eğitim":
        max_train = tab.iloc[:,0].max()
    else:
        print(tab.round(3).sort_values("Üç komşuya uzaklık").to_string())
        print("Eğitim en büyüğünü geçen test toprağı:", (tab.iloc[:,0] > max_train).sum())

# Tek çekimin uzaklığını yine numune ortancalarına göre ölç. Eğitimde aynı toprağın tamamı dışarıda.
def photo_distances(raw, training):
    rows = []
    for sample, shots in raw.groupby("sample_id"):
        ref = train.drop(index=sample) if training else train
        sc = StandardScaler().fit(ref)
        dist = np.linalg.norm(sc.transform(shots[keys])[:, None, :] - sc.transform(ref)[None, :, :], axis=2)
        rows.extend((sample, value) for value in np.sort(dist, axis=1)[:, :3].mean(axis=1))
    return pd.DataFrame(rows, columns=["Numune", "Uzaklık"])

photo_ref = photo_distances(original, True)
photo_test = photo_distances(test_rows, False)
assert len(photo_ref) == 127 and len(photo_test) == 35
threshold = photo_ref["Uzaklık"].max()
print("Özgün eğitim çekimlerinde en yüksek uzaklık:", round(threshold, 3))
print(photo_test.assign(Aşım=photo_test["Uzaklık"] > threshold).groupby("Numune")
      .agg(Çekim=("Uzaklık","size"), En_az=("Uzaklık","min"), En_çok=("Uzaklık","max"),
           Aşım=("Aşım","sum")).round(3).to_string())

# Aynı saklı toprağı bu kez en yakın iki referansını da dışarıda tutarak tahmin et.
published.LINEAR_ONLY = True
aug_means = train_rows.groupby("sample_id")[keys].mean()
orig_means = original.groupby("sample_id")[keys].mean()
logq = pd.DataFrame(curve_to_logq(labels.to_numpy()), index=labels.index)

def held_errors(held):
    ids = train.index.drop(held)
    sc = StandardScaler().fit(train.loc[ids])
    x, q = sc.transform(train.loc[ids]), sc.transform(train.loc[held])
    y = labels.loc[ids]
    coords = []
    for frac in [y["0.063"], 100-y["2"]]:
        reg = PLSRegression(n_components=1, scale=False).fit(x, frac)
        coords.append(reg.predict(np.vstack([x,q])).ravel().clip(0,100))
    z = np.column_stack(coords)
    idx, _, v3 = published.fit_predict(aug_means.loc[ids].reset_index(),
                                       orig_means.loc[held].reset_index(), logq.loc[ids], keys)
    v3 = pd.DataFrame(v3, index=idx, columns=COLS)
    result = {}
    for sample, query in zip(held, z[len(ids):]):
        n = np.argsort(np.linalg.norm(z[:len(ids)]-query,axis=1), kind="stable")[:3]
        knn = y.iloc[n].mean().to_numpy()
        result[sample] = [emd(labels.loc[sample],knn)[0], emd(labels.loc[sample],v3.loc[sample])[0]]
    return result

methods = ["İnce/iri 3 komşu", "v3"]
records = []
closest = {}
for s in train.index:
    others = train.index.drop(s)
    closest[s] = set(nearby(train.loc[others], train.loc[[s]]).index[:2])
    records.append(["Normal", s, *held_errors([s])[s]])
for held in combinations(train.index, 3):
    for sample, errors in held_errors(list(held)).items():
        records.append(["Herhangi iki", sample, *errors])
        if set(held) - {sample} == closest[sample]:
            records.append(["İki yakın", sample, *errors])

scores = pd.DataFrame(records, columns=["Sınav","Numune",*methods])
counts = scores.groupby(["Sınav","Numune"]).size().unstack()
assert (counts.loc["Herhangi iki"] == 253).all() and (counts.loc[["Normal","İki yakın"]] == 1).all().all()
groups = scores.groupby(["Sınav","Numune"])[methods].mean()
means = groups.groupby(level="Sınav").mean()
np.testing.assert_allclose(means.loc["Normal"], [40.144,38.758], atol=.00051, rtol=0)
np.testing.assert_allclose(means.loc["İki yakın"], [41.387,40.788], atol=.00051, rtol=0)
print(means.round(3))
extra = groups.loc["İki yakın"] - groups.loc["Herhangi iki"]
print("Yakın çifti çıkarmanın ek hatası: ortalama, ortanca ve pozitif numune sayısı")
print(extra.mean().round(3), extra.median().round(3), (extra>1e-8).sum(), sep="\n")
print("Yakın çift ile rastgele çiftin beklenen hatası arasındaki fark, her numune")
print(extra.round(2).to_string())
