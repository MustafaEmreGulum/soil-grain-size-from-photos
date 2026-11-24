"""Aynı altı doku girdisiyle 200 mm'den 0,002 mm'ye koşullu ve doğrudan tahmin."""
import numpy as np
import pandas as pd
from sklearn.isotonic import isotonic_regression

from coklu_elek_deneyi import OUT, TEXTURE
from ince_0063_oran import fit_share
from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels, repo
from gsd import emd


def main():
    photos = pd.read_csv(repo / "cache/granulometry_train.csv")
    assert len(photos) == 127 and photos.file.is_unique and labels["200"].eq(100).all()
    archive = pd.read_csv(OUT / "ince_paya_2mm_sonuclar.csv").set_index(["exam", "sample_id"])
    rows = []
    for exam, hold, ids, src, dst in folds():
        source = photos if src is None else photos[photos.phone == src]
        target = photos if dst is None else photos[photos.phone == dst]
        train = source.groupby("sample_id")[TEXTURE].median().loc[ids]
        query = target.groupby("sample_id")[TEXTURE].median().loc[hold]
        assert set(train.index).isdisjoint(query.index)
        truth = labels.loc[ids]
        chain = np.full((len(hold), len(COLS)), 100.)
        direct = chain.copy()
        for k in reversed(range(len(COLS)-1)):
            fraction = fit_share(train, query, truth[COLS[k]], truth[COLS[k+1]])["doku"]
            chain[:,k] = chain[:,k+1] * fraction
            direct[:,k] = 100 * fit_share(train, query, truth[COLS[k]], truth["200"])["doku"]
        direct = np.stack([isotonic_regression(curve) for curve in direct])
        saved = archive.loc[[(exam,s) for s in hold]]
        for name, values in [("mevcut",saved[COLS].to_numpy()),("dogrudan",direct),("zincir",chain)]:
            assert np.isfinite(values).all() and (values >= 0).all() and (values[:,-1] == 100).all()
            assert (np.diff(values,axis=1) >= -1e-8).all()
            scores = emd(labels.loc[hold,COLS], values)
            if name == "mevcut":
                np.testing.assert_allclose(scores, saved.emd, rtol=0, atol=1e-7)
            for sample, score, curve in zip(hold, scores, values, strict=True):
                rows.append(dict(exam=exam,sample_id=sample,model=name,emd=score,**dict(zip(COLS,curve))))
    result = pd.DataFrame(rows)
    assert len(result)==285 and not result.duplicated(["exam","sample_id","model"]).any()
    result.to_csv(OUT / "tam_elek_zinciri_sonuclar.csv", index=False, float_format="%.9f")
    errors = result.copy()
    errors[COLS] = np.abs(result[COLS].to_numpy() - labels.loc[result.sample_id,COLS].to_numpy())
    summary = errors.groupby(["exam","model"])[["emd",*reversed(COLS)]].mean()
    summary.to_csv(OUT / "tam_elek_zinciri_ozet.csv",float_format="%.9f")
    print(summary.loc["toprak"].round(3).to_string())
    print(summary.emd.unstack().round(3).to_string())


if __name__ == "__main__":
    main()
