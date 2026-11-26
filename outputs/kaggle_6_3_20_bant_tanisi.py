"""Compare held-out band mass, its two endpoints and the visible area for the 24 soils."""
import numpy as np
import pandas as pd

from kaggle_elek_hata_izi import OUT


def main():
    source = pd.read_csv(OUT / "ortak_63_20_sonuclar.csv").query("exam == 'toprak'").set_index("sample_id")
    photos = pd.read_csv(OUT / "maske_elek_alanlari.csv").query("split == 'train'")
    labels = pd.read_csv(OUT.parent / "work/soil-data/Training_labels_updated.csv", index_col="sample_id")
    baseline = pd.read_csv(OUT / "tekrar_cekimi_genel_emd.csv").query(
        "exam == 'toprak' and model == 'mevcut'").set_index("sample_id")
    assert len(source) == 24 and source.index.is_unique and set(source.index) == set(baseline.index)
    assert len(photos) == 127 and photos.file.is_unique and set(photos.sample_id) == set(source.index)
    out = source[["gercek6_3","mevcut6_3","gercek20","mevcut20","sinira_takildi"]].copy()
    for diameter, key in (("6.3", "6_3"), ("20", "20")):
        np.testing.assert_allclose(out["gercek" + key], labels.loc[out.index, diameter], atol=1e-7, rtol=0)
        np.testing.assert_allclose(out["mevcut" + key], baseline.loc[out.index, diameter], atol=1e-7, rtol=0)
        out["gecis_hatasi_" + key] = out["mevcut" + key] - out["gercek" + key]
    assert (photos["a_6.3"] >= photos.a_20 - 1e-9).all()
    areas = photos.assign(bant_alan=photos["a_6.3"]-photos.a_20).groupby("sample_id").bant_alan
    out["gorunur_bant_alani"] = areas.mean()
    out["gercek_bant"] = out.gercek20 - out.gercek6_3
    out["mevcut_bant"] = out.mevcut20 - out.mevcut6_3
    out["bant_hatasi"] = out.mevcut_bant - out.gercek_bant
    np.testing.assert_allclose(out.bant_hatasi, out.gecis_hatasi_20 - out.gecis_hatasi_6_3, atol=1e-8)
    out["ortak_bant"] = source.ortak20 - source.ortak6_3
    expected_trigger = source.alan_gecis20.lt(source.saf_v3_gecis6_3 - 1e-9)
    assert (out.sinira_takildi == expected_trigger).all()
    out.to_csv(OUT/"kaggle_6_3_20_bant_tanisi.csv", float_format="%.9f")
    print(out.sort_values("bant_hatasi", key=abs, ascending=False).round(2).to_string())
    print("band MAE", out.bant_hatasi.abs().mean())
    for col in ("gercek_bant","bant_hatasi"):
        print("area Spearman", col, out.gorunur_bant_alani.corr(out[col],method="spearman"))
    print("under / over / equal", (out.bant_hatasi < -1e-7).sum(), (out.bant_hatasi > 1e-7).sum(),
          (out.bant_hatasi.abs() <= 1e-7).sum())


if __name__ == "__main__":
    main()
