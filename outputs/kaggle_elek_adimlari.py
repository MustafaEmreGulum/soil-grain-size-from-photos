"""Evaluate cumulative coarse-to-fine corrections with nested soil/site holdouts."""
from functools import lru_cache

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ince_iki_elek_kalibrasyon import AREAS, AREA_COLS
from kaggle_iki_elek_ayrisma import TARGET, POS
from saha_ve_egri_deneyi import SITE, folds
from tekrar_cekimi_genel_emd import corrected, factor_for
from v3_knn_esit_karisim import COLS, keys, labels, original, out, published, training
from gsd import DLOG, emd

START = COLS.index("20")


@lru_cache(maxsize=None)
def curves(train_ids, query_ids, src, dst):
    assert set(train_ids).isdisjoint(query_ids)
    fit = training[training.sample_id.isin(train_ids)]
    obs = original[original.sample_id.isin(query_ids)]
    ref_areas, test_areas = AREAS, AREAS
    if src:
        fit, obs = fit[fit.phone.eq(src)], obs[obs.phone.eq(dst)]
        ref_areas, test_areas = AREAS[AREAS.phone.eq(src)], AREAS[AREAS.phone.eq(dst)]
    ref = ref_areas.groupby("sample_id")[AREA_COLS].mean().loc[list(train_ids)]
    test = test_areas.groupby("sample_id")[AREA_COLS].mean()
    factor = factor_for(ref)
    order, _, values = published.fit_predict(fit, obs, TARGET.loc[list(train_ids)], keys)
    assert set(order) == set(query_ids)
    result = {}
    for soil, raw in zip(order, values):
        base = corrected(pd.Series(raw, index=COLS), test.loc[soil], ref, factor, 100.)
        result[soil] = base, raw-base
    return result


def slopes(ids, src, by_site):
    groups = {}
    for soil in ids:
        groups.setdefault(SITE[soil] if by_site else soil, []).append(soil)
    xs, ys = [], []
    for held in groups.values():
        train_ids = tuple(sorted(set(ids)-set(held)))
        for soil, (base, gap) in curves(train_ids, tuple(sorted(held)), src, src).items():
            xs.append(gap)
            ys.append(labels.loc[soil, COLS].to_numpy()-base)
    x, y = np.asarray(xs), np.asarray(ys)
    denom = (x*x).sum(axis=0)
    separate = np.divide((x*y).sum(axis=0), denom, out=np.zeros_like(denom), where=denom>1e-12)
    common20 = x[:, START] @ y[:, POS].mean(axis=1) / denom[START] if denom[START]>1e-12 else 0.
    return separate, common20


def main():
    archive = pd.read_csv(out/"tekrar_cekimi_genel_emd.csv").query("model=='mevcut'").set_index(["exam","sample_id"])
    previous = pd.read_csv(out/"kaggle_iki_elek_ayrisma.csv").query("group=='24'").set_index("sample_id")
    rows = []
    for exam, hold, ids, src, dst in folds():
        ids, hold = tuple(sorted(ids)), tuple(sorted(hold))
        coeff, common20 = slopes(ids, src, exam == "saha")
        for soil, (base, gap) in curves(ids, hold, src, dst).items():
            saved = archive.loc[(exam, soil)]
            np.testing.assert_allclose(base, saved[COLS].to_numpy(float), atol=1e-7, rtol=0)
            truth = labels.loc[soil, COLS].to_numpy()
            current = base.copy()
            last = emd(truth, base)[0]
            np.testing.assert_allclose(last, saved.emd, atol=1e-7, rtol=0)
            for k in range(START, -1, -1):
                if k == START:
                    # Isolate the previous joint experiment's 20 mm proposal, including its original bounds.
                    shift = np.clip(common20*gap[k], base[k-2]-base[k-1], base[k+1]-base[k])
                    proposed = base[k]+shift
                    if exam == "toprak":
                        np.testing.assert_allclose(proposed, previous.at[soil,"new_p20"], atol=1e-7, rtol=0)
                else:
                    proposed = base[k]+coeff[k]*gap[k]
                before = current.copy()
                current[k] = np.clip(proposed, base[k-1] if k else 0., current[k+1])
                assert np.all(np.diff(current)>=-1e-8)
                np.testing.assert_array_equal(before[k+1:], current[k+1:])
                score = emd(truth,current)[0]
                olderr, newerr = abs(base[k]-truth[k]), abs(current[k]-truth[k])
                np.testing.assert_allclose(last-score, (olderr-newerr)*DLOG[k], atol=1e-8)
                rows.append(dict(exam=exam,sample_id=soil,sieve=COLS[k],truth=truth[k],
                                 old=base[k],new=current[k],old_mae=olderr,new_mae=newerr,
                                 baseline_emd=saved.emd,stage_emd=score,gain=last-score))
                last = score
    result = pd.DataFrame(rows)
    assert len(result)==95*(START+1) and not result.duplicated(["exam","sample_id","sieve"]).any()
    result.to_csv(out/"kaggle_elek_adimlari.csv",index=False,float_format="%.9f")
    summary = result.groupby(["exam","sieve"],sort=False)[["old_mae","new_mae","stage_emd","gain"]].mean()
    print(summary.round(4).to_string())


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
