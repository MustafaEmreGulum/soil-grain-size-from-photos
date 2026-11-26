"""Sabit v3 + 20 mm referansına karşı elek düzeltmeleri ve koşullu çap."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iri_toprak_sinavi import d50, labels, original, predict, training
from gsd import COLS, LOGX, emd

OUT = Path(__file__).resolve().parent
FINE = ["0.063", "0.2", "0.63"]
TEXTURE = [f"{kind}_{r}" for r in ("0.2528", "0.5684", "1.0438") for kind in ("go", "gc")]
AREA = ["0.63", "2", "6.3", "20", "63"]
BASE = "v3+20"


def photo_features():
    gran = pd.read_csv(OUT.parent / "work/vitalclick/cache/granulometry_train.csv")
    radii = [c[3:] for c in gran if c.startswith("go_")]
    log_diam = np.log10(2 * np.array(radii, dtype=float))
    cumulative = (gran[["go_" + r for r in radii]].values + gran[["gc_" + r for r in radii]].values) / 2
    assert (np.diff(cumulative, axis=1) >= -1e-9).all()
    gran["log_texture_d50"] = [np.interp(.5, *first_crossings(c, log_diam)) for c in cumulative]
    areas = pd.read_csv(OUT / "maske_elek_alanlari.csv").query("split == 'train'")
    keys = ["file", "sample_id", "phone"]
    frame = areas.merge(gran[keys + TEXTURE + ["log_texture_d50"]], on=keys, validate="one_to_one")
    assert len(frame) == len(areas) == len(gran) == 127
    return frame


def first_crossings(cumulative, log_diam):
    values, first = np.unique(cumulative, return_index=True)
    return values, log_diam[first]


def soil_features(frame, phone=None):
    group = (frame if phone is None else frame[frame.phone == phone]).groupby("sample_id")
    return group[["a_" + s for s in AREA]].mean().join(group[TEXTURE + ["log_texture_d50"]].median())


def bounds(curve, updates):
    """Sabit elekleri koru, ardışık değişen eleklerde eşit ağırlıklı monoton projeksiyon yap."""
    out = curve.copy()
    indexes = sorted(COLS.index(s) for s in updates)
    for block in np.split(indexes, np.where(np.diff(indexes) != 1)[0] + 1):
        if len(block):
            lo, hi = block[0], block[-1]
            values = [updates[COLS[k]] for k in block]
            out[block] = np.clip(isotonic_regression(values), out[lo-1], out[hi+1])
    return out


def area_updates(ref, obs, truth):
    updates = {}
    for sieve in AREA:
        x = ref["a_" + sieve].to_numpy()
        a = obs["a_" + sieve]
        if a > 0 and x @ x > 0:
            retained = 100 - truth[sieve].to_numpy()
            updates[sieve] = 100 - a * max(0., (x @ retained) / (x @ x))
    return updates


def variants(ref, obs, raw):
    truth = labels.loc[ref.index]
    updates = area_updates(ref, obs, truth)
    at20 = {s:v for s,v in updates.items() if s == "20"}
    base = bounds(raw, at20)
    curves = {"v3": raw, BASE: base}
    for sieves in [["0.63"], ["2"], ["6.3"], ["63"], ["2", "6.3", "63"]]:
        curves["alan_" + "+".join(sieves)] = bounds(base, {s:updates[s] for s in sieves if s in updates})

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=len(ref))).fit(ref[TEXTURE], truth[FINE])
    fine = ridge.predict(obs[TEXTURE].to_frame().T)[0]
    mix = {s: (val + base[COLS.index(s)]) / 2 for s,val in zip(FINE, fine)}
    curves["ince_doku"] = bounds(base, mix)

    # Fotoğraf ölçeği referans aralığını geçerse laboratuvar/görüntü çap oranını aktar.
    triggered = obs.log_texture_d50 > ref.log_texture_d50.max()
    shifted = raw.copy()
    if triggered:
        offset = np.median(np.log10(d50(truth)) - ref.log_texture_d50)
        delta = obs.log_texture_d50 + offset - np.log10(d50(raw)[0])
        shifted = np.interp(LOGX - delta, LOGX, raw, left=0., right=100.)
        shifted[-1] = 100.
    curves["kosullu_cap"] = bounds(shifted, at20)
    return curves, triggered


def scenarios(frame):
    archived = pd.read_csv(OUT / "kutle_payi_kontrol_sonuclari.csv").query("model == 'v3'")
    for exam, rows in archived.groupby("exam"):
        src, dst = (None, None) if exam == "24" else exam.split(" -> ")
        ref, obs = soil_features(frame, src), soil_features(frame, dst)
        ids = pd.Index(rows.sample_id)
        assert len(ids) == (24 if exam == "24" else 21)
        for sample, row in rows.set_index("sample_id").iterrows():
            yield exam, sample, ref.loc[ids.drop(sample)], obs.loc[sample], row[COLS].to_numpy(float)
    coarse = pd.Series(d50(labels), index=labels.index).nlargest(5).index
    ids = labels.index.drop(coarse)
    raw = predict(training[training.sample_id.isin(ids)], original[original.sample_id.isin(coarse)])["v3"]
    previous = pd.read_csv(OUT / "iri_toprak_sonuclari.csv").query("model == 'v3'").set_index("sample_id")
    np.testing.assert_allclose(emd(labels.loc[coarse], raw.loc[coarse]), previous.loc[coarse,"emd"], atol=1e-6)
    feats = soil_features(frame)
    for sample in coarse:
        yield "iri_5", sample, feats.loc[ids], feats.loc[sample], raw.loc[sample].values


def test_candidate():
    photos = pd.read_csv(OUT / "maske_elek_alanlari.csv")
    grouped = photos.groupby(["split", "sample_id"])[["a_" + s for s in AREA]].mean()
    ref, obs = grouped.loc["train"], grouped.loc["test"]
    base = pd.read_csv(OUT / "submission_v3_tam_yuzey_20mm.csv", index_col="sample_id")
    template = pd.read_csv(OUT.parent / "work/soil-data/sample_submission.csv", index_col="sample_id")
    assert base.index.equals(template.index) and base.columns.tolist() == template.columns.tolist() == COLS
    assert set(obs.index) == set(base.index) and set(ref.index) == set(labels.index)
    candidate = base.copy()
    changes = []
    for sample, row in base.iterrows():
        a = obs.loc[sample]
        updates = area_updates(ref, a, labels.loc[ref.index])
        # 63 mm had zero area in all 162 images. Export only the two supported changes.
        curve = bounds(row.values, {s:updates[s] for s in ("2", "6.3") if s in updates})
        for s in ("2", "6.3"):
            candidate.loc[sample,s] = round(curve[COLS.index(s)], 4)
            col = "a_" + s
            changes.append(dict(sample_id=sample, elek_mm=s, once=row[s], sonra=candidate.loc[sample,s],
                                alan=a[col], egitim_min=ref[col].min(), egitim_max=ref[col].max(),
                                aralik_disi=not ref[col].min() <= a[col] <= ref[col].max()))
    pd.testing.assert_frame_equal(candidate.drop(columns=["2","6.3"]), base.drop(columns=["2","6.3"]))
    assert np.isfinite(candidate).all().all() and (np.diff(candidate.values) >= 0).all()
    assert candidate.min().min() >= 0 and candidate["200"].eq(100).all()
    dest = OUT / "submission_v3_tam_yuzey_2_6_3_20mm.csv"
    candidate.to_csv(dest)
    pd.testing.assert_frame_equal(pd.read_csv(dest, index_col="sample_id"), candidate)
    pd.DataFrame(changes).to_csv(OUT / "coklu_elek_test_kontrol.csv", index=False, float_format="%.8f")


def main():
    records = []
    for exam, sample, ref, obs, raw in scenarios(photo_features()):
        assert sample not in ref.index
        curves, trigger = variants(ref, obs, raw)
        actual = labels.loc[sample, COLS].values
        for name, curve in curves.items():
            assert np.isfinite(curve).all() and (np.diff(curve) >= -1e-8).all()
            assert curve.min() >= 0 and curve[-1] == 100
            records.append(dict(exam=exam, sample_id=sample, model=name, n_ref=len(ref),
                                cap_kosulu=bool(trigger), emd=emd(actual,curve)[0], d50_mm=d50(curve)[0],
                                d50_hata_mm=abs(d50(curve)[0] - d50(actual)[0]), **dict(zip(COLS, curve))))
    result = pd.DataFrame(records)
    keys = ["exam", "sample_id"]
    ref = result[result.model == BASE].set_index(keys).emd
    previous = pd.read_csv(OUT / "v3_tam_yuzey_20mm_sonuclar.csv").set_index(keys)
    np.testing.assert_allclose(ref.loc[previous.index], previous.duzeltilmis_emd, atol=1e-7, rtol=0)
    result["kazanc"] = ref.reindex(pd.MultiIndex.from_frame(result[keys])).values - result.emd
    assert not result.duplicated(keys + ["model"]).any()
    result.to_csv(OUT / "coklu_elek_sonuclar.csv", index=False, float_format="%.9f")
    summary = []
    for (exam, model), rows in result.groupby(["exam", "model"], sort=False):
        g = rows.kazanc
        summary.append(dict(exam=exam, model=model, n=len(g), emd=rows.emd.mean(), kazanc=g.mean(),
                            ortanca=g.median(), iyilesen=(g>1e-7).sum(), kotulesen=(g< -1e-7).sum(),
                            tek_haric_min=((g.sum()-g)/(len(g)-1)).min(), d50_mae_mm=rows.d50_hata_mm.mean(),
                            **{"mae_"+s:(rows.set_index("sample_id")[s] - labels.loc[rows.sample_id,s]).abs().mean()
                               for s in COLS}))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "coklu_elek_ozet.csv", index=False, float_format="%.9f")
    print(summary.pivot(index="model", columns="exam", values="emd").round(3))
    test_candidate()


if __name__ == "__main__":
    main()
