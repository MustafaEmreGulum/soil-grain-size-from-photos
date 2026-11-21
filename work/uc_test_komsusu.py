import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

out = Path("outputs")
book = json.loads((out / "toprak_goreli_doku.ipynb").read_text(encoding="utf8"))
env = {}
for i in [1, 3]:
    exec("".join(book["cells"][i]["source"]), env)
test_code = "".join(book["cells"][-1]["source"]).split("    test_rows.append")[0]
exec(test_code + "    test_rows.append([sample, *values])", env)
keys = env["features"].columns
train_rows = pd.DataFrame(env["rows"], columns=["sample_id", *keys]).set_index("sample_id")
test_rows = pd.DataFrame(env["test_rows"], columns=["sample_id", *keys]).set_index("sample_id")
labels = env["labels"]
assert len(test_rows) == 35 and len(train_rows) == 127
neighbors, files = {}, {}
for model in ["doku", "goreli_doku"]:
    saved = pd.read_csv(out / f"submission_3nn_{model}.csv", index_col=0)
    files[model] = saved
    frames = []
    for rows in [train_rows, test_rows]:
        if model == "goreli_doku":
            total = rows.sum(axis=1)
            rows = rows.div(total.where(total > 0, 1), axis=0)
        frames.append(rows.groupby(level=0).median())
    train, test = frames
    train = train.loc[labels.index]
    scaler = StandardScaler().fit(train)
    ref = scaler.transform(train)
    for sample, query in zip(saved.index, scaler.transform(test.loc[saved.index])):
        nearest = train.index[np.argsort(np.linalg.norm(ref - query, axis=1), kind="stable")[:3]]
        np.testing.assert_allclose(labels.loc[nearest].mean(), saved.loc[sample], atol=1e-8, rtol=0)
        neighbors[model, sample] = nearest.tolist()
old, new = files.values()
distances = (new-old).abs().iloc[:, :-1] @ np.diff(np.log10(labels.columns.astype(float)))
chosen = set()
for sample in distances.nlargest(3).index:
    print(sample)
    for model in files:
        ids = neighbors[model, sample]
        chosen.update(ids)
        print(model, ids, files[model].loc[sample, ["0.063","2"]].to_dict())
print(labels.loc[sorted(chosen)].round(2).to_string())
