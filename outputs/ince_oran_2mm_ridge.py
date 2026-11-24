"""2 mm geçişini sabit tutarak 0,063 / 2 mm ince oranını doku ile tahmin et."""
import numpy as np
import pandas as pd

from coklu_elek_deneyi import OUT, TEXTURE
from ince_0063_oran import fit_share
from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels, original, repo
from gsd import emd

FINE = COLS.index("0.063")


def evaluate_photos(photos):
    archive = pd.read_csv(OUT / "ince_paya_2mm_sonuclar.csv").set_index(["exam", "sample_id"])
    assert photos.file.is_unique and set(photos.file) == set(original.file)
    records = []
    for exam, hold, ids, src, dst in folds():
        assert set(ids).isdisjoint(hold)
        source = photos if src is None else photos.loc[photos.phone == src]
        target = photos if dst is None else photos.loc[photos.phone == dst]
        train = source.groupby("sample_id")[TEXTURE].median().loc[ids]
        query = target.groupby("sample_id")[TEXTURE].median().loc[hold]
        estimates = fit_share(train, query, labels.loc[ids, "0.063"], labels.loc[ids, "2"])
        for i, sample in enumerate(hold):
            saved = archive.loc[(exam, sample)]
            base = saved[COLS].to_numpy(float)
            total = saved["2"]
            assert total > 0
            actual = labels.loc[sample, COLS].to_numpy(float)
            np.testing.assert_allclose(emd(actual, base)[0], saved.emd, atol=1e-7, rtol=0)
            shares = {"referans": base[FINE] / total, **{name: a[i] for name, a in estimates.items()}}
            for name, ratio in shares.items():
                raw_fine = total * ratio
                curve = base.copy()
                curve[FINE] = np.clip(raw_fine, base[FINE-1], base[FINE+1])
                assert np.all(np.diff(curve) >= 0) and curve[0] >= 0 and curve[-1] == 100
                records.append(dict(exam=exam, sample_id=sample, model=name,
                                    oran_gercek=labels.loc[sample,"0.063"] / labels.loc[sample,"2"],
                                    oran_ham=ratio, tahmin_2mm=total, ham_0063=raw_fine,
                                    tahmin_0063=curve[FINE], gercek_0063=actual[FINE],
                                    sinirlandi=abs(curve[FINE]-raw_fine)>1e-7,
                                    mutlak_hata=abs(curve[FINE]-actual[FINE]), emd=emd(actual,curve)[0]))
    results = pd.DataFrame(records)
    assert len(results) == 285 and not results.duplicated(["exam", "sample_id", "model"]).any()
    return results


def evaluate():
    results = evaluate_photos(pd.read_csv(repo / "cache/granulometry_train.csv"))
    results.to_csv(OUT / "ince_oran_2mm_ridge_sonuclar.csv", index=False, float_format="%.9f")
    print(results.groupby(["exam","model"])[["mutlak_hata","emd"]].mean().round(3))


if __name__ == "__main__":
    evaluate()
