"""Kütle payı raporundaki karşılaştırmayı v3'ün yerel görüntü ölçülerinden yeniden üretir."""
from pathlib import Path

import numpy as np
import pandas as pd

from v3_knn_esit_karisim import labels, original, published, sets, training
import gsd


def project_percentages(raw):
    """En yakın, negatif olmayan ve toplamı 100 olan pay vektörü."""
    ordered = np.sort(raw)[::-1]
    shifts = (ordered.cumsum() - 100) / np.arange(1, len(raw) + 1)
    threshold = shifts[np.flatnonzero(ordered > shifts)[-1]]
    return np.maximum(raw - threshold, 0.0)


def evaluate(name, ids, refs, queries):
    cols = list(dict.fromkeys(sets["R1"] + sets["R2"]))
    x = refs.groupby("sample_id")[cols].mean().loc[ids]
    test = queries.groupby("sample_id")[cols].mean().loc[ids]
    assert queries.aug.eq(0).all()
    rows = []
    for soil in ids:
        pool = ids.drop(soil)
        lab = labels.loc[pool].to_numpy()
        targets = {"v3": gsd.curve_to_logq(lab), "pay": np.diff(lab, axis=1, prepend=0)}
        for method, target in targets.items():
            parts = []
            for features in sets.values():
                model = published.Component(features, "pls").fit(x.loc[pool, features].to_numpy(), target)
                parts.append(model.predict(test.loc[[soil], features].to_numpy())[0])
            raw = np.mean(parts, axis=0)
            curve = gsd.logq_to_curve(raw)[0] if method == "v3" else project_percentages(raw).cumsum()
            assert np.isfinite(curve).all() and (np.diff(curve) >= 0).all() and curve[0] >= 0
            np.testing.assert_allclose(curve[-1], 100, rtol=0, atol=1e-9)
            row = dict(exam=name, sample_id=soil, model=method, emd=gsd.emd(labels.loc[soil], curve)[0])
            row.update(zip(gsd.COLS, curve))
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    moto, sam = "Motorola_Edge", "Samsung_A52"
    common = labels.index.intersection(original[original.phone == moto].sample_id).intersection(
        original[original.phone == sam].sample_id)
    assert len(labels) == 24 and len(common) == 21
    blocks = [evaluate("24", labels.index, training, original)]
    for source, other in [(moto, sam), (sam, moto)]:
        blocks.append(evaluate(source + " -> " + other, common, training[training.phone == source],
                               original[original.phone == other]))
    result = pd.concat(blocks, ignore_index=True)
    keys = ["exam", "sample_id"]
    archived = pd.read_csv(out / "ikinci_iri_aday_sonuclari.csv").set_index(keys)
    reproduced = result.query("model == 'v3'").set_index(keys)
    assert len(reproduced) == len(archived) == 66 and reproduced.index.is_unique
    np.testing.assert_allclose(reproduced.loc[archived.index, "emd"], archived.v3_emd, atol=1e-7, rtol=0)
    result.to_csv(out / "kutle_payi_kontrol_sonuclari.csv", index=False, float_format="%.9f")
    print(result.groupby(["exam", "model"]).emd.mean().round(6))
