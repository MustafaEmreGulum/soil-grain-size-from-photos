"""Compare archived fine-sieve predictions on the controlled 63 mm baseline."""
import numpy as np
import pandas as pd

from kaggle_elek_hata_izi import OUT, gsd


def main():
    saved = pd.read_csv(OUT / "ince_oran_2mm_ridge_sonuclar.csv").query("exam == 'toprak'")
    reference = saved.query("model == 'referans'").set_index("sample_id").sort_index()
    ridge = saved.query("model == 'doku'").set_index("sample_id").sort_index()
    archive = pd.read_csv(OUT / "tekrar_cekimi_genel_emd.csv").query(
        "exam == 'toprak' and model == 'mevcut'").set_index("sample_id").sort_index()
    assert archive.index.is_unique and len(archive) == 24
    assert archive.index.equals(reference.index) and archive.index.equals(ridge.index)
    truth = pd.read_csv(OUT.parent / "work/soil-data/Training_labels_updated.csv",
                        index_col="sample_id").loc[archive.index, gsd.COLS]
    for key, series in [("gercek_0063", truth["0.063"]), ("tahmin_2mm", archive["2"])]:
        for model in (reference, ridge):
            np.testing.assert_allclose(model[key], series, atol=1e-7, rtol=0)
    np.testing.assert_allclose(reference.tahmin_0063, archive["0.063"], atol=1e-7, rtol=0)
    results = pd.DataFrame({"gercek": truth["0.063"], "mevcut": archive["0.063"],
                            "ridge": ridge.tahmin_0063, "ridge_sinirlandi": ridge.sinirlandi})
    results["yari_yariya"] = (results.mevcut + results.ridge) / 2
    for model in ("mevcut", "ridge", "yari_yariya"):
        curve = archive[gsd.COLS].copy()
        curve["0.063"] = results[model]
        assert (np.diff(curve, axis=1) >= -1e-8).all()
        results["hata_" + model] = results[model] - results.gercek
        results["emd_" + model] = gsd.emd(truth, curve)
    np.testing.assert_allclose(results.emd_mevcut, archive.emd, atol=1e-7, rtol=0)
    np.testing.assert_allclose(results.emd_ridge-results.emd_mevcut, ridge.emd-reference.emd, atol=1e-7, rtol=0)
    results["ridge_kazanci"] = results.hata_mevcut.abs() - results.hata_ridge.abs()
    direction = lambda error: np.select([error > 1e-7, error < -1e-7], ["fazla", "eksik"], default="esit")
    results["yon_mevcut"] = direction(results.hata_mevcut)
    results["yon_ridge"] = direction(results.hata_ridge)
    results["alt_002"] = archive["0.02"]
    results["ust_02"] = archive["0.2"]
    closest = results.gercek.clip(results.alt_002, results.ust_02)
    results["komsular_sabitken_min_hata"] = (results.gercek - closest).abs()
    results.to_csv(OUT / "kaggle_0063_ridge_hata_yonu.csv", float_format="%.9f")
    print(results.groupby(["yon_mevcut","yon_ridge"]).size())
    print(results[["gercek","mevcut","ridge","ridge_kazanci"]].sort_values("ridge_kazanci").round(2).to_string())
    print(results.filter(like="hata_").abs().mean(), results.filter(like="emd_").mean())
    print("signed-error correlation", results.hata_mevcut.corr(results.hata_ridge))
    print("retrospective minimum MAE", results[["hata_mevcut","hata_ridge"]].abs().min(axis=1).mean())


if __name__ == "__main__":
    main()
