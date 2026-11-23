"""0,02 ve 0,2 mm geçişlerini sabitleyerek aralarındaki 0,063 mm geçişini tahmin et."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from coklu_elek_deneyi import COLS, OUT, TEXTURE, emd, labels, photo_features, scenarios
from gsd import DLOG

LOW, MID, HIGH = "0.02", "0.063", "0.2"
K = COLS.index(MID)


def ratios(ref, query):
    y = labels.loc[ref.index]
    span = y[HIGH] - y[LOW]
    used = span > 0
    x = ref.loc[used, TEXTURE]
    share = (y.loc[used,MID] - y.loc[used,LOW]) / span[used]
    assert len(x) and share.between(0,1).all()
    # Aynı ortak oranı ve Ridge sabit terimini karşılaştırabilmek için aynı ağırlıklar.
    w = span[used] / span[used].mean()
    common = np.average(share, weights=w)
    scaler = StandardScaler().fit(x, sample_weight=w)
    ridge = Ridge(alpha=len(x)).fit(scaler.transform(x), share, sample_weight=w)
    by_image = np.clip(ridge.predict(scaler.transform(query[TEXTURE])), 0, 1)
    return {"ortak":np.full(len(query), common), "doku":by_image}


def evaluate():
    detail = pd.read_csv(OUT / "iki_model_elek_toprak.csv").query("model == 'aralik'")
    curves = detail.pivot(index=["exam","sample_id"], columns="elek_mm", values="tahmin_pct")
    curves.columns = [f"{s:g}" for s in curves.columns]
    archived = pd.read_csv(OUT / "orta_elek_kismi_sonuclar.csv").query("model == 'aralik'").set_index(["exam","sample_id"])
    records = []
    for exam, sample, ref, obs, _ in scenarios(photo_features()):
        assert sample not in ref.index
        b = curves.loc[(exam,sample), COLS].to_numpy()
        truth = labels.loc[sample, COLS].to_numpy()
        baseline = emd(truth, b)[0]
        np.testing.assert_allclose(baseline, archived.loc[(exam,sample),"emd"], atol=1e-7, rtol=0)
        estimates = ratios(ref, obs.to_frame().T)
        predictions = {"referans":b[K], **{key:b[K-1]+r[0]*(b[K+1]-b[K-1]) for key,r in estimates.items()}}
        floor = abs(truth[K]-np.clip(truth[K], b[K-1], b[K+1]))
        for name, value in predictions.items():
            curve = b.copy()
            curve[K] = value
            assert np.all(np.diff(curve)>=-1e-8) and np.isfinite(curve).all()
            error = emd(truth, curve)[0]
            gain = baseline - error
            np.testing.assert_allclose(gain, (abs(truth[K]-b[K])-abs(truth[K]-value))*DLOG[K], atol=1e-8)
            records.append(dict(exam=exam, sample_id=sample, model=name, gercek=truth[K], tahmin=value,
                                alt=b[K-1], ust=b[K+1], mutlak_hata=abs(truth[K]-value),
                                sinirlar_ile_min_hata=floor, emd=error, kazanc=gain))
    rows = pd.DataFrame(records)
    assert len(rows)==213 and not rows.duplicated(["exam","sample_id","model"]).any()
    rows.to_csv(OUT/"ince_0063_oran_sonuclar.csv", index=False, float_format="%.9f")
    groups = pd.concat([rows, rows[(rows.exam=="24") & ~rows.sample_id.isin(["H037","H038"])].assign(exam="22")])
    summary = []
    for (exam, model), group in groups.groupby(["exam","model"], sort=False):
        g = group.kazanc
        summary.append(dict(exam=exam, model=model, n=len(group), emd=group.emd.mean(),
                            mae=group.mutlak_hata.mean(), kazanc=g.mean(), ortanca=g.median(),
                            iyilesen=(g>1e-7).sum(), kotulesen=(g< -1e-7).sum(),
                            tek_haric_min=((g.sum()-g)/(len(g)-1)).min(),
                            aralik_disi=(group.sinirlar_ile_min_hata>1e-7).sum(),
                            en_iyi_mumkun_mae=group.sinirlar_ile_min_hata.mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT/"ince_0063_oran_ozet.csv", index=False, float_format="%.9f")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    evaluate()
