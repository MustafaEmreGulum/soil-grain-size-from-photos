"""Quantify two label-informed counterfactuals on the 24 held-out Kaggle curves."""
import numpy as np
import pandas as pd

from kaggle_elek_hata_izi import OUT, gsd


def main():
    trace = pd.read_csv(OUT / "kaggle_elek_hata_izi.csv", dtype={"elek_mm": str})
    pred = trace.pivot(index="sample_id", columns="elek_mm", values="tahmin")[gsd.COLS]
    truth = pd.read_csv(OUT.parent / "work/soil-data/Training_labels_updated.csv",
                        index_col="sample_id").loc[pred.index, gsd.COLS]
    recorded_truth = trace.pivot(index="sample_id", columns="elek_mm", values="gercek")[gsd.COLS]
    np.testing.assert_allclose(recorded_truth, truth)
    archive = pd.read_csv(OUT / "tekrar_cekimi_genel_emd.csv").query(
        "exam == 'toprak' and model == 'mevcut'").set_index("sample_id").loc[pred.index]
    np.testing.assert_allclose(archive[gsd.COLS], pred, atol=1e-7, rtol=0)
    assert len(pred) == 24 and pred["2"].gt(0).all() and truth["2"].gt(0).all()
    fine = gsd.COLS[:gsd.COLS.index("2")]
    total = pred.copy()
    total[fine] = pred[fine].div(pred["2"], axis=0).mul(truth["2"], axis=0)
    total["2"] = truth["2"]
    # Retain the true anchor; lift a coarser sieve only where monotonicity requires it.
    total = total.cummax(axis=1)
    shape = pred.copy()
    shape[fine] = truth[fine].div(truth["2"], axis=0).mul(pred["2"], axis=0)
    both = total.copy()
    both[fine] = truth[fine]
    prior = pd.read_csv(OUT / "ince_pay_hata_ayrimi.csv").query("exam == 'toprak'").set_index("sample_id").loc[pred.index]
    for curve, column in [(total, "hata_2mm_bilinse"), (shape, "hata_oran_bilinse")]:
        np.testing.assert_allclose(curve["0.063"] - truth["0.063"], prior[column], atol=1e-7, rtol=0)
    scores = pd.DataFrame(index=pred.index)
    curves = {"mevcut":pred, "toplam":total, "dagilim":shape, "ikisi":both}
    for name, curve in curves.items():
        assert np.isfinite(curve).all().all() and curve.ge(0).all().all() and curve.le(100+1e-8).all().all()
        assert (np.diff(curve, axis=1) >= -1e-8).all() and curve["200"].eq(100).all()
        scores[name] = gsd.emd(truth, curve)
    np.testing.assert_allclose(scores.mevcut, archive.emd, atol=1e-7, rtol=0)
    coarse = gsd.COLS[gsd.COLS.index("2")+1:]
    scores["ust_elekte_duzeltme"] = total[coarse].gt(pred[coarse]+1e-8).any(axis=1)
    weights = pd.Series(np.r_[gsd.DLOG, 0.], index=gsd.COLS)
    scores["ust_elek_emd_kazanci"] = (
        (pred[coarse]-truth[coarse]).abs() - (total[coarse]-truth[coarse]).abs()
    ).mul(weights[coarse], axis=1).sum(axis=1)
    for name in ["toplam","dagilim","ikisi"]:
        scores["kazanc_"+name] = scores.mevcut - scores[name]
    scores.to_csv(OUT / "kaggle_2mm_emd_ayrimi.csv", float_format="%.9f")
    print(scores.round(3).to_string())
    print("Means",scores.mean().round(4).to_dict())
    gain = scores.toplam - scores.dagilim
    print("shape wins", (gain>1e-8).sum(), "ties", (gain.abs()<=1e-8).sum(),
          "min leave-one-out advantage", ((gain.sum()-gain)/23).min(),
          "without H037 H038", gain.drop(["H037","H038"]).mean())
    print("shape gain by sieve", ((pred-truth).abs()-(shape-truth).abs()).mul(weights, axis=1).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
