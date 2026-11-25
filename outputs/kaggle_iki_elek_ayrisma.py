"""Nested leave-one-soil-out test of a common shift predicted from 20 mm disagreement."""
from itertools import combinations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ince_iki_elek_kalibrasyon import AREAS, AREA_COLS
from tekrar_cekimi_genel_emd import corrected, factor_for
from v3_knn_esit_karisim import COLS, keys, labels, original, out, published, training
from gsd import DLOG, curve_to_logq, emd

PAIR = ["6.3", "20"]
POS = [COLS.index(s) for s in PAIR]
AREA = AREAS.groupby("sample_id")[AREA_COLS].mean()
TARGET = pd.DataFrame(curve_to_logq(labels[COLS].to_numpy()), index=labels.index)


def predict(eligible, held):
    ids = sorted(set(eligible) - set(held))
    assert set(ids).isdisjoint(held)
    fit = training[training.sample_id.isin(ids)]
    test = original[original.sample_id.isin(held)]
    order, _, raw = published.fit_predict(fit, test, TARGET.loc[ids], keys)
    assert set(order) == set(held)
    ref = AREA.loc[ids]
    factor = factor_for(ref)
    records = {}
    for soil, values in zip(order, raw):
        base = corrected(pd.Series(values, index=COLS), AREA.loc[soil], ref, factor, 100.)
        disagreement = values[POS[1]] - base[POS[1]]
        records[soil] = (base, disagreement)
    return records


def evaluate(eligible, tag, archive):
    # Each pair fit omits both the outer test soil and one inner validation soil.
    calibration = {soil: [] for soil in eligible}
    for pair in combinations(eligible, 2):
        inner = predict(eligible, pair)
        for outer, held in (pair, pair[::-1]):
            base, gap = inner[held]
            target = (labels.loc[held, PAIR].to_numpy() - base[POS]).mean()
            calibration[outer].append((gap, target))
    rows = []
    for soil in eligible:
        x, y = np.asarray(calibration[soil]).T
        # No intercept: agreement of the two models implies no correction.
        slope = (x @ y) / (x @ x) if x @ x else 0.
        base, gap = predict(eligible, [soil])[soil]
        truth = labels.loc[soil, COLS].to_numpy()
        if tag == "24":
            np.testing.assert_allclose(base, archive.loc[soil, COLS].to_numpy(float), rtol=0, atol=1e-7)
        shift = np.clip(slope * gap, base[COLS.index("2")]-base[POS[0]],
                        base[COLS.index("63")]-base[POS[1]])
        candidate = base.copy()
        candidate[POS] += shift
        assert (np.diff(candidate) >= -1e-8).all()
        np.testing.assert_allclose(np.diff(candidate[POS]), np.diff(base[POS]), atol=1e-8)
        old_score, score = emd(truth, [base, candidate])
        old_err, new_err = abs(base[POS]-truth[POS]), abs(candidate[POS]-truth[POS])
        np.testing.assert_allclose(score-old_score, (new_err-old_err) @ DLOG[POS], atol=1e-8)
        rows.append(dict(group=tag, sample_id=soil, n_inner=len(x), active_inner=(abs(x)>1e-7).sum(),
                         v3_p20=base[POS[1]]+gap, gap=gap, slope=slope, shift=shift,
                         base_p63=base[POS[0]], base_p20=base[POS[1]], lab_p63=truth[POS[0]],
                         lab_p20=truth[POS[1]], new_p63=candidate[POS[0]], new_p20=candidate[POS[1]],
                         old_emd=old_score, new_emd=score, gain=old_score-score))
    return pd.DataFrame(rows)


def main():
    archive = pd.read_csv(out/"tekrar_cekimi_genel_emd.csv").query(
        "exam == 'toprak' and model == 'mevcut'").set_index("sample_id")
    all_ids = sorted(labels.index)
    full = evaluate(all_ids, "24", archive)
    pure = pd.read_csv(out/"v3_tam_yuzey_20mm_sonuclar.csv").query("exam=='24'").set_index("sample_id")
    np.testing.assert_allclose(100-full.v3_p20, pure.loc[full.sample_id, "v3_iri"], atol=1e-7, rtol=0)
    removed = evaluate([s for s in all_ids if s != "H637"], "23_h637_disarida", archive)
    results = pd.concat([full, removed], ignore_index=True)
    results.to_csv(out/"kaggle_iki_elek_ayrisma.csv", index=False, float_format="%.9f")
    print(results.groupby("group")[["old_emd","new_emd","gain"]].mean().round(5))
    print(full.loc[full.gap.abs()>1e-7,["sample_id","gap","shift","gain"]].round(3).to_string(index=False))
    print("without H637 score only", full.loc[full.sample_id.ne("H637"),"gain"].mean())


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
