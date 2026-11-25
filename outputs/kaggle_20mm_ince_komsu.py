"""Compare monotone one-sieve and two-sieve oracle bounds on the same held-out soils."""
import numpy as np
import pandas as pd

from kaggle_elek_hata_izi import OUT, gsd


def main():
    archived = pd.read_csv(OUT / "tekrar_cekimi_genel_emd.csv").query(
        "exam == 'toprak' and model == 'mevcut'").set_index("sample_id").sort_index()
    pred = archived[gsd.COLS]
    truth = pd.read_csv(OUT.parent / "work/soil-data/Training_labels_updated.csv",
                        index_col="sample_id").loc[pred.index, gsd.COLS]
    assert len(pred) == 24 and pred.index.is_unique
    base = gsd.emd(truth, pred)
    np.testing.assert_allclose(base, archived.emd, atol=1e-7)
    rows = pd.DataFrame(index=pred.index)
    rows["engel"] = np.select([truth["0.063"] < pred["0.02"], truth["0.063"] > pred["0.2"]],
                              ["0.02_fazla", "0.2_eksik"], default="yok")
    blocked = rows.engel.ne("yok")
    assert blocked.sum() == 7
    fine = pred.copy()
    fine["0.063"] = truth["0.063"].clip(pred["0.02"], pred["0.2"])
    pair = pred.copy()
    # A monotone laboratory pair stays monotone when clipped to common outer limits.
    for group, cols, left, right in (("0.02_fazla", ["0.02","0.063"], "0.0063", "0.2"),
                                     ("0.2_eksik", ["0.063","0.2"], "0.02", "0.63")):
        ids = rows.index[rows.engel.eq(group)]
        for c in cols:
            pair.loc[ids,c] = truth.loc[ids,c].clip(pred.loc[ids,left], pred.loc[ids,right])
    only20 = pred.copy()
    only20["20"] = truth["20"].clip(pred["6.3"], pred["63"])
    all_pass = pred.copy()
    assert pred["63"].eq(100).all()
    all_pass["20"] = 100.
    for name, curve in (("mevcut",pred), ("20_tam_gecis",all_pass), ("20_ideal",only20),
                        ("ince_tek_yedi",pred.mask(blocked, fine, axis=0)), ("ince_cift_yedi",pair)):
        assert np.all(np.diff(curve, axis=1)>=-1e-8)
        rows[name] = gsd.emd(truth,curve)
    rows["63_ustu_gercek"] = 100-truth["63"]
    rows["20_ustu_gercek"] = 100-truth["20"]
    rows["20_ustu_tahmin"] = 100-pred["20"]
    rows["20_kalan_hata"] = (only20["20"]-truth["20"]).abs()
    rows["ince_cift_kalan_hata_0063"] = (pair["0.063"]-truth["0.063"]).abs()
    rows.to_csv(OUT/"kaggle_20mm_ince_komsu.csv", float_format="%.9f")
    print(rows.iloc[:,1:6].mean(), rows.engel.value_counts())
    print(rows.loc[blocked,["engel","ince_cift_kalan_hata_0063"]])
    print("63mm",rows[rows["63_ustu_gercek"]>0]["63_ustu_gercek"])
    print("20 collisions", rows[rows["20_kalan_hata"]>1e-7].index.tolist(),
          "20 MAE", (pred["20"]-truth["20"]).abs().mean(), "flat20 MAE",rows["20_ustu_gercek"].mean())


if __name__ == "__main__":
    main()
