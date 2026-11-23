"""Yarı düzeltme ve orta aralığın görünür tane alanlarına göre paylaşımı."""
import numpy as np
import pandas as pd

from coklu_elek_deneyi import AREA, BASE, COLS, OUT, area_updates, bounds, emd, labels, photo_features, scenarios

BANDS = ["0.63", "2", "6.3", "20"]
IDX = [COLS.index(s) for s in BANDS]
INNER = IDX[1:3]


def alternatives(base, full, ref, obs):
    # Durum 1: fiziksel olarak geçerli iki tahminin tam ortası.
    half = base.copy()
    half[INNER] = (base[INNER] + full[INNER]) / 2

    # Durum 2: üç ayrı boyut sınıfının alanını kütleyle kalibre et.
    acols = ["a_" + s for s in BANDS]
    x = -np.diff(ref[acols].to_numpy(), axis=1)
    a = -np.diff(obs[acols].to_numpy(dtype=float))
    y = np.diff(labels.loc[ref.index, BANDS].to_numpy(), axis=1)
    assert min(x.min(), a.min(), y.min()) >= -1e-8
    denom = (x*x).sum(axis=0)
    factors = np.divide((x*y).sum(axis=0), denom, out=np.zeros(3), where=denom > 0)
    mass = a * factors
    band = base.copy()
    if mass.sum() > 0 and not np.any((a > 0) & (denom == 0)):
        # 0,63 ve 20 mm çapa ait geçişler v3 + 20 mm referansından gelir.
        span = base[IDX[-1]] - base[IDX[0]]
        band[INNER] = base[IDX[0]] + span * np.cumsum(mass / mass.sum())[:2]
    return {BASE:base, "tam":full, "yari":half, "aralik":band}


def check(curve, base):
    np.testing.assert_array_equal(np.delete(curve, INNER), np.delete(base, INNER))
    assert np.isfinite(curve).all() and (np.diff(curve) >= -1e-8).all()
    assert curve[0] >= 0 and curve[-1] == 100


def evaluate():
    archived = pd.read_csv(OUT / "coklu_elek_sonuclar.csv").set_index(["exam", "sample_id", "model"])
    records = []
    for exam, sample, ref, obs, raw in scenarios(photo_features()):
        assert sample not in ref.index
        updates = area_updates(ref, obs, labels.loc[ref.index])
        base = bounds(raw, {s:updates[s] for s in ["20"] if s in updates})
        full = bounds(base, {s:updates[s] for s in BANDS[1:3] if s in updates})
        for name, curve in alternatives(base, full, ref, obs).items():
            check(curve, base)
            actual = labels.loc[sample, COLS].values
            error = emd(actual, curve)[0]
            if name in (BASE, "tam"):
                old = archived.loc[(exam, sample, BASE if name == BASE else "alan_2+6.3+63")]
                np.testing.assert_allclose(curve, old[COLS].to_numpy(float), rtol=0, atol=1e-7)
                np.testing.assert_allclose(error, old.emd, atol=1e-7, rtol=0)
            records.append(dict(exam=exam, sample_id=sample, model=name, emd=error,
                                kazanc=emd(actual,base)[0]-error,
                                **{s:curve[k] for s,k in zip(BANDS, IDX)},
                                orta_pay=curve[IDX[2]]-curve[IDX[0]],
                                pay_2_6=curve[IDX[2]]-curve[IDX[1]],
                                gercek_orta_pay=actual[IDX[2]]-actual[IDX[0]],
                                gercek_pay_2_6=actual[IDX[2]]-actual[IDX[1]]))
    rows = pd.DataFrame(records)
    assert len(rows) == 71*4 and not rows.duplicated(["exam","sample_id","model"]).any()
    rows.to_csv(OUT / "orta_elek_kismi_sonuclar.csv", index=False, float_format="%.9f")
    summary = []
    for (exam, name), group in rows.groupby(["exam","model"], sort=False):
        g = group.kazanc
        summary.append(dict(exam=exam, model=name, n=len(g), emd=group.emd.mean(), kazanc=g.mean(),
                            ortanca=g.median(), iyilesen=(g>1e-7).sum(), kotulesen=(g< -1e-7).sum(),
                            tek_haric_min=((g.sum()-g)/(len(g)-1)).min(),
                            sifir_orta=(group.orta_pay<=1e-7).sum(), sifir_2_6=(group.pay_2_6<=1e-7).sum(),
                            mae_orta=(group.orta_pay-group.gercek_orta_pay).abs().mean(),
                            mae_2_6=(group.pay_2_6-group.gercek_pay_2_6).abs().mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "orta_elek_kismi_ozet.csv", index=False, float_format="%.9f")
    print(summary.round(3).to_string(index=False))


def test_curves():
    photos = pd.read_csv(OUT / "maske_elek_alanlari.csv")
    grouped = photos.groupby(["split","sample_id"])[["a_"+s for s in AREA]].mean()
    ref, obs = grouped.loc["train"], grouped.loc["test"]
    base = pd.read_csv(OUT / "submission_v3_tam_yuzey_20mm.csv", index_col="sample_id")
    full = pd.read_csv(OUT / "submission_v3_tam_yuzey_2_6_3_20mm.csv", index_col="sample_id")
    template = pd.read_csv(OUT.parent / "work/soil-data/sample_submission.csv", index_col="sample_id")
    assert base.index.equals(full.index) and base.index.equals(template.index)
    assert base.columns.tolist() == full.columns.tolist() == template.columns.tolist() == COLS
    assert set(obs.index) == set(base.index) and set(ref.index) == set(labels.index)
    results = []
    candidates = {name:base.copy() for name in ("yari", "aralik")}
    for sample in base.index:
        b = base.loc[sample].values
        curves = alternatives(b, full.loc[sample].values, ref, obs.loc[sample])
        for name, curve in curves.items():
            check(curve, b)
            results.append(dict(sample_id=sample, model=name, **dict(zip(BANDS,curve[IDX])),
                                orta_pay=curve[IDX[2]]-curve[IDX[0]], pay_2_6=curve[IDX[2]]-curve[IDX[1]]))
        for name, candidate in candidates.items():
            candidate.loc[sample] = curves[name]
    pd.DataFrame(results).to_csv(OUT / "orta_elek_kismi_test.csv", index=False, float_format="%.8f")
    for name, candidate in candidates.items():
        dest = OUT / f"submission_v3_orta_elek_{name}.csv"
        candidate.to_csv(dest, float_format="%.6f")
        saved = pd.read_csv(dest, index_col="sample_id")
        pd.testing.assert_frame_equal(saved, candidate, atol=.00000051, rtol=0)
        for sample, row in saved.iterrows():
            check(row.values, base.loc[sample].values)
        if name == "yari":
            pd.testing.assert_frame_equal(saved, (base+full)/2, atol=1e-7, rtol=0)


if __name__ == "__main__":
    evaluate()
    test_curves()
