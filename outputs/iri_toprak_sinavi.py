"""Beş iri toprağı, eğitimde dört iri referans varken ve yokken karşılaştır."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from v3_knn_esit_karisim import labels, original, predict, training
from gsd import curve_to_logq, emd


def d50(curves):
    return 10 ** curve_to_logq(curves, [50])[:, 0]


def measure(pool, targets):
    assert len(pool) == 19 and pool.intersection(targets).empty
    refs = training[training.sample_id.isin(pool)]
    query = original[original.sample_id.isin(targets)]
    assert set(refs.sample_id) == set(pool) and set(query.sample_id) == set(targets)
    results = predict(refs, query)
    truth = labels.loc[targets]
    rows = []
    for model in ["v3", "KNN"]:
        curves = results[model].loc[targets]
        frame = pd.DataFrame({
            "model": model, "lab_d50_mm": d50(truth), "d50_mm": d50(curves),
            "emd": emd(truth, curves),
        }, index=targets)
        frame["d50_ratio"] = frame.d50_mm / frame.lab_d50_mm
        rows.append(frame.reset_index())
    return pd.concat(rows, ignore_index=True)


def main():
    by_size = pd.Series(d50(labels), index=labels.index).sort_values(ascending=False)
    coarse = by_size.index[:5]
    fine = labels.index.drop(coarse).sort_values()
    assert list(coarse) == ["H368", "H374", "H615", "H367", "H366"] and len(fine) == 19
    holdout = measure(fine, coarse)
    assert holdout.loc[holdout.model == "KNN", "d50_mm"].max() <= by_size.loc[fine].max() + 1e-8

    groups = list(combinations(fine, 4))
    draws = np.random.default_rng(42).choice(len(groups), size=100, replace=False)
    controls = []
    for draw in draws:
        rest = fine.drop(list(groups[draw]))
        for sample in coarse:
            targets = coarse[coarse == sample]
            pool = rest.append(coarse.drop(sample))
            controls.append(measure(pool, targets))

    raw = pd.concat(controls, ignore_index=True)
    keys = ["sample_id", "model"]
    assert raw.groupby(keys).size().eq(100).all() and len(raw) == 1000
    control = raw.groupby(keys)[["d50_mm", "d50_ratio", "emd"]].mean().add_prefix("control_")
    result = holdout.join(control, on=keys, validate="one_to_one")
    result["emd_difference"] = result.emd - result.control_emd
    assert np.isfinite(result.select_dtypes("number")).all().all()
    out = Path(__file__).parent / "iri_toprak_sonuclari.csv"
    result.to_csv(out, index=False, float_format="%.8f")
    print(result.round(3).to_string(index=False))
    for model, group in result.groupby("model"):
        dif = group.emd_difference
        leave_one_out = (dif.sum() - dif) / (len(dif) - 1)
        print(model, "stress/control/delta EMD", group.emd.mean(), group.control_emd.mean(),
              dif.mean(), "median D50 ratios", group.d50_ratio.median(), group.control_d50_ratio.median(),
              "delta omitting one soil", leave_one_out.min(), leave_one_out.max())


if __name__ == "__main__":
    main()
