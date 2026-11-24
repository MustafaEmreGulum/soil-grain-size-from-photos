"""Saklı 0,63 mm tahminine bağlanan 0,2 mm oranını sınar."""
import numpy as np
import pandas as pd

from coklu_elek_deneyi import OUT, TEXTURE, bounds
from ince_0063_oran import fit_share
from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels, original, repo
from gsd import DLOG, emd

STEM = "sabit_063_gecis"
UPPER, TARGET, LOWER = "0.63", "0.2", "0.063"
K = COLS.index(TARGET)
DETAIL = ["exam", "sample_id"]


def run():
    photos = pd.read_csv(repo / "cache/granulometry_train.csv")
    baseline = pd.read_csv(OUT / "ince_paya_2mm_sonuclar.csv").set_index(DETAIL)
    assert photos.file.is_unique and set(photos.file) == set(original.file)
    rows = []
    for exam, hold, ids, src, dst in folds():
        assert set(ids).isdisjoint(hold)
        source = photos if src is None else photos[photos.phone == src]
        target = photos if dst is None else photos[photos.phone == dst]
        train = source.groupby("sample_id")[TEXTURE].median().loc[ids]
        query = target.groupby("sample_id")[TEXTURE].median().loc[hold]
        shares = fit_share(train, query, labels.loc[ids, TARGET], labels.loc[ids, UPPER])
        for i, sample in enumerate(hold):
            saved = baseline.loc[(exam, sample)]
            base = saved[COLS].to_numpy(float)
            actual = labels.loc[sample, COLS].to_numpy(float)
            np.testing.assert_allclose(emd(actual, base)[0], saved.emd, atol=1e-7, rtol=0)
            estimates = {"mevcut": base[K], **{name: base[K+1]*p[i] for name,p in shares.items()}}
            for name, raw in estimates.items():
                curve = bounds(base, {TARGET:raw})
                np.testing.assert_array_equal(np.delete(curve,K), np.delete(base,K))
                assert np.isfinite(curve).all() and (np.diff(curve)>=0).all()
                error = abs(actual[K]-curve[K])
                score = emd(actual,curve)[0]
                np.testing.assert_allclose(saved.emd-score, DLOG[K]*(abs(actual[K]-base[K])-error),
                                           atol=1e-7, rtol=0)
                rows.append(dict(exam=exam, sample_id=sample, model=name, sabit_063=base[K+1],
                                 sabit_0063=base[K-1], gercek_02=actual[K], ham_02=raw,
                                 tahmin_02=curve[K], hata_02=error, emd=score))
    result = pd.DataFrame(rows)
    assert len(result) == 285 and not result.duplicated(DETAIL + ["model"]).any()
    result.to_csv(OUT / f"{STEM}_sonuclar.csv", index=False, float_format="%.9f")
    print(result.groupby(["exam","model"])[["hata_02","emd"]].mean().round(4).to_string())


if __name__ == "__main__":
    run()
