"""6,3 mm sınırına takılan 20 mm alan tahminini iki elekte birlikte sına."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coklu_elek_deneyi import area_updates, bounds, photo_features, soil_features
from ince_paya_2mm_aktarimi import transfer
from orta_elek_kismi_duzeltme import band_curve
from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels, original, out, predict, training
from gsd import emd


def experiment():
    photos = photo_features()
    features = {phone: soil_features(photos, phone) for phone in [None, *photos.phone.unique()]}
    archive = pd.read_csv(out / "ustten_20_63_sonuclar.csv").query("model == 'mevcut'").set_index(["exam", "sample_id"])
    # No positive 63 mm candidate was found in any training photo at the coarse scale.
    coarse = pd.read_csv(out / "ikinci_olcek_egitim_fotograflar.csv")
    assert set(coarse.file) == set(photos.file) and coarse.n63.eq(0).all()
    k6, k20 = COLS.index("6.3"), COLS.index("20")
    rows = []
    for exam, hold, ids, src, dst in folds():
        assert set(hold).isdisjoint(ids)
        refs, query = training[training.sample_id.isin(ids)], original[original.sample_id.isin(hold)]
        if src:
            refs, query = refs[refs.phone == src], query[query.phone == dst]
        raw = predict(refs, query)["v3"]
        ref = features[src].loc[ids]
        for sample in hold:
            obs = features[dst].loc[sample]
            v3 = raw.loc[sample, COLS].to_numpy()
            updates = area_updates(ref, obs, labels.loc[ids])
            at20 = {"20": updates["20"]} if "20" in updates else {}
            old20 = bounds(v3, at20)
            base = transfer(band_curve(old20, ref, obs), v3)
            base[COLS.index("63")] = 100.
            saved = archive.loc[(exam, sample)]
            np.testing.assert_allclose(base, saved[COLS].to_numpy(float), atol=1e-7, rtol=0)

            target = at20.get("20", np.nan)
            conflict = bool(target < v3[k6] - 1e-9)
            candidate = bounds(base, {"6.3": base[k6], "20": target}) if conflict else base.copy()
            untouched = [j for j in range(len(COLS)) if j not in (k6, k20)]
            np.testing.assert_array_equal(candidate[untouched], base[untouched])
            assert np.isfinite(candidate).all() and np.all(np.diff(candidate) >= -1e-9)
            truth = labels.loc[sample, COLS].to_numpy()
            score, new_score = emd(truth, [base, candidate])
            np.testing.assert_allclose(score, saved.emd, atol=1e-7, rtol=0)
            rows.append(dict(exam=exam, sample_id=sample, site=saved.site, n_train=len(ids),
                             alan20_pct=obs.a_20, egitim_max_alan20=ref.a_20.max(),
                             egitim_pozitif_alan_sayisi=int(ref.a_20.gt(0).sum()),
                             alan_gecis20=target, saf_v3_gecis6_3=v3[k6],
                             sinira_takildi=conflict, sabit_gecis2=base[k6-1], sabit_gecis63=base[k20+1],
                             gercek6_3=truth[k6], mevcut6_3=base[k6], ortak6_3=candidate[k6],
                             gercek20=truth[k20], mevcut20=base[k20], ortak20=candidate[k20],
                             mevcut_emd=score, ortak_emd=new_score, emd_kazanc=score-new_score))
    results = pd.DataFrame(rows)
    assert len(results) == len(archive) == 95 and not results.duplicated(["exam", "sample_id"]).any()
    for sieve in ("6_3", "20"):
        for model in ("mevcut", "ortak"):
            results[f"{model}_hata{sieve}"] = (results[model+sieve] - results["gercek"+sieve]).abs()
    results.to_csv(out / "ortak_63_20_sonuclar.csv", index=False, float_format="%.9f")
    print(results.groupby("exam")[["mevcut_emd", "ortak_emd", "mevcut_hata6_3", "ortak_hata6_3",
                                   "mevcut_hata20", "ortak_hata20"]].mean().round(4).to_string())
    print(results.query("exam == 'toprak' and sinira_takildi")[["sample_id", "emd_kazanc"]].to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(1):
        experiment()
