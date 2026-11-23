"""Üç ince elek geçişini iki dış elek arasında birlikte tahmin et."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from coklu_elek_deneyi import COLS, OUT, TEXTURE, bounds, emd, labels, photo_features, scenarios
from gsd import DLOG

TARGETS = ["0.02", "0.063", "0.2"]
POS = [COLS.index(s) for s in TARGETS]
LEFT, RIGHT = "0.0063", "0.63"


def candidates(ref, obs, base):
    truth = labels.loc[ref.index]
    span = truth[RIGHT] - truth[LEFT]
    valid = span > 0
    shares = truth.loc[valid, TARGETS].sub(truth.loc[valid, LEFT], axis=0).div(span[valid], axis=0)
    assert len(shares) and np.isfinite(shares).all().all() and shares.min().min() >= 0 and shares.max().max() <= 1
    weights = span[valid] / span[valid].mean()
    x = ref.loc[valid, TEXTURE]
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=len(x)).fit(scaler.transform(x), shares, sample_weight=weights)
    estimates = {"ortak": np.average(shares, axis=0, weights=weights),
                 "doku": model.predict(scaler.transform(obs[TEXTURE].to_frame().T))[0]}
    left, right = base[COLS.index(LEFT)], base[COLS.index(RIGHT)]
    curves = {"referans": base}
    for name, ratios in estimates.items():
        values = left + (right-left)*ratios
        curves[name] = bounds(base, dict(zip(TARGETS, values)))
    return curves


def main():
    prior = pd.read_csv(OUT / "iki_model_elek_toprak.csv").query("model == 'aralik'")
    curves = prior.pivot(index=["exam","sample_id"], columns="elek_mm", values="tahmin_pct")
    curves.columns = [f"{s:g}" for s in curves.columns]
    single = pd.read_csv(OUT / "ince_0063_oran_sonuclar.csv").set_index(["exam","sample_id","model"])
    records = []
    for exam, sample, ref, obs, _ in scenarios(photo_features()):
        assert sample not in ref.index
        base = curves.loc[(exam,sample), COLS].to_numpy()
        actual = labels.loc[sample,COLS].to_numpy()
        for name, pred in candidates(ref, obs, base).items():
            assert np.isfinite(pred).all() and np.all(np.diff(pred) >= -1e-8)
            np.testing.assert_array_equal(np.delete(pred, POS), np.delete(base, POS))
            score = emd(actual,pred)[0]
            baseline = single.loc[(exam,sample,"referans"),"emd"]
            change = (np.abs(actual[POS]-base[POS])-np.abs(actual[POS]-pred[POS])) @ DLOG[POS]
            np.testing.assert_allclose(baseline-score, change, rtol=0, atol=1e-7)
            records.append(dict(exam=exam, sample_id=sample, model=name, emd=score,
                                kazanc=baseline-score, tek_elek_emd=single.loc[(exam,sample,name),"emd"],
                                sinir_disi=not pred[POS[0]]-1e-8 <= actual[POS[1]] <= pred[POS[2]]+1e-8,
                                **{s:pred[k] for s,k in zip(TARGETS,POS)},
                                **{"hata_"+s:pred[k]-actual[k] for s,k in zip(TARGETS,POS)}))
    detail = pd.DataFrame(records)
    assert len(detail)==213 and not detail.duplicated(["exam","sample_id","model"]).any()
    detail.to_csv(OUT/"ince_uc_elek_sonuclar.csv", index=False, float_format="%.9f")
    main_rows = detail[detail.exam=="24"]
    blocks = pd.concat([detail, main_rows[~main_rows.sample_id.isin(["H037","H038"])].assign(exam="22"),
                        main_rows[~main_rows.sample_id.isin(["H616","H405"])].assign(exam="22_H616_H405_haric")])
    rows = []
    for (exam, name), group in blocks.groupby(["exam","model"], sort=False):
        gain = group.kazanc
        rows.append(dict(exam=exam, model=name, n=len(gain), emd=group.emd.mean(),
                         **{"mae_"+s:group["hata_"+s].abs().mean() for s in TARGETS},
                         kazanc=gain.mean(), ortanca=gain.median(), iyilesen=(gain>1e-7).sum(),
                         kotulesen=(gain< -1e-7).sum(), tek_haric_min=((gain.sum()-gain)/(len(gain)-1)).min(),
                         sinir_disi=group.sinir_disi.sum()))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/"ince_uc_elek_ozet.csv", index=False, float_format="%.9f")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
