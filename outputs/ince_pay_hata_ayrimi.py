"""Saklı tahminlerde 0,063 mm hatasını toplam geçiş ve koşullu pay üzerinden incele."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def diagnose():
    predictions = pd.read_csv(OUT / "ince_paya_2mm_sonuclar.csv")
    labels = pd.read_csv(OUT.parent / "work/soil-data/Training_labels_updated.csv")
    cols = ["sample_id", "2", "0.063"]
    rows = predictions[["exam", "site", *cols]].merge(
        labels[cols], on="sample_id", suffixes=("_tahmin", "_gercek"), validate="many_to_one"
    )
    assert len(rows) == len(predictions) == 95 and not rows.duplicated(["exam", "sample_id"]).any()
    for suffix in ("tahmin", "gercek"):
        total, fine = rows["2_" + suffix], rows["0.063_" + suffix]
        assert total.between(0, 100).all() and total.gt(0).all() and fine.between(0, total).all()
        rows["oran_" + suffix] = fine / total
    # Her karşı-olguda yalnız bir bileşeni laboratuvar değeriyle değiştir.
    fine = rows["0.063_gercek"]
    rows["hata"] = rows["0.063_tahmin"] - fine
    rows["hata_2mm_bilinse"] = rows["2_gercek"] * rows.oran_tahmin - fine
    rows["hata_oran_bilinse"] = rows["2_tahmin"] * rows.oran_gercek - fine
    # İki olası değiştirme sırasına eşit ağırlık vererek işaretli hatayı paylaştır.
    rows["katki_2mm"] = (rows["2_tahmin"] - rows["2_gercek"]) * (rows.oran_tahmin + rows.oran_gercek) / 2
    rows["katki_oran"] = (rows.oran_tahmin - rows.oran_gercek) * (rows["2_tahmin"] + rows["2_gercek"]) / 2
    np.testing.assert_allclose(rows.katki_2mm + rows.katki_oran, rows.hata, rtol=0, atol=1e-9)
    return rows


if __name__ == "__main__":
    rows = diagnose()
    rows.to_csv(OUT / "ince_pay_hata_ayrimi.csv", index=False, float_format="%.9f")
    metrics = ["hata", "hata_2mm_bilinse", "hata_oran_bilinse"]
    print(rows.groupby("exam")[metrics].agg(lambda s: s.abs().mean()).round(3).to_string())
    soils = rows.query("exam == 'toprak'").set_index("sample_id")
    print(soils.loc[["H037", "H038"]].round(3).to_string())
    print("Oran katkisi daha buyuk:", (soils.katki_oran.abs() > soils.katki_2mm.abs()).sum())
    print("H037 ve H038 haric:", soils.drop(["H037", "H038"])[metrics].abs().mean().round(3).to_dict())
