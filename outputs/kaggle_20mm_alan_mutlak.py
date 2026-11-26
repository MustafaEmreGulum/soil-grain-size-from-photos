"""Test an area-to-mass slope fitted by absolute error on each training fold."""
import numpy as np
import pandas as pd

from kaggle_elek_hata_izi import OUT, gsd


def main():
    photos = pd.read_csv(OUT/"v3_tam_yuzey_20mm_fotograflar.csv")
    truth = pd.read_csv(OUT.parent/"work/soil-data/Training_labels_updated.csv", index_col="sample_id")
    raw = pd.read_csv(OUT/"kutle_payi_kontrol_sonuclari.csv").query("model == 'v3'").set_index(["exam","sample_id"])
    fitted = pd.read_csv(OUT/"v3_tam_yuzey_20mm_sonuclar.csv").set_index(["exam","sample_id"])
    reference = pd.read_csv(OUT/"tekrar_cekimi_genel_emd.csv").query("model == 'mevcut'").set_index(["exam","sample_id"])
    results = []
    for exam, rows in fitted.groupby(level="exam"):
        ids = rows.index.get_level_values("sample_id")
        src, dst = (photos, photos) if exam == "24" else [
            photos[photos.phone.eq(phone)] for phone in exam.split(" -> ")]
        x = src.groupby("sample_id").alan_pct.mean().loc[ids]
        area = dst.groupby("sample_id").alan_pct.mean().loc[ids]
        for soil in ids:
            old = fitted.loc[(exam, soil)]
            curve = reference.loc[("toprak" if exam=="24" else exam, soil)]
            v3 = raw.loc[(exam,soil)]
            train = x.drop(soil)
            train = train[train.gt(0)]
            mass = 100-truth.loc[train.index,"20"]
            ratios = (mass/train).sort_values()
            weights = train.loc[ratios.index].cumsum()
            slope = ratios.iloc[weights.searchsorted(weights.iloc[-1]/2)]
            old_slope = (train @ mass)/(train @ train)
            a = area.at[soil]
            np.testing.assert_allclose([old_slope,a], [old.carpan,old.alan_pct], rtol=0,atol=1e-7)
            def passing(factor):
                if a == 0:
                    return v3["20"]
                return np.clip(100-factor*a,v3["6.3"],v3["63"])
            np.testing.assert_allclose(passing(old_slope),100-old.duzeltilmis_iri,atol=1e-7,rtol=0)
            # Apply only the change in the area estimate to the final baseline series.
            p = curve[gsd.COLS].to_numpy(float)
            proposal = np.clip(curve["20"] + passing(slope)-passing(old_slope),curve["6.3"],curve["63"])
            y = truth.loc[soil,gsd.COLS].to_numpy()
            np.testing.assert_allclose(gsd.emd(y,p)[0],curve.emd,atol=1e-7)
            p[gsd.COLS.index("20")] = proposal
            assert np.all(np.diff(p)>=-1e-7)
            results.append(dict(exam=exam,sample_id=soil,n_positive=len(train),slope_old=old_slope,
                                slope_new=slope,old20=curve["20"],new20=proposal,true20=truth.at[soil,"20"],
                                old_mae=curve.mae20,new_mae=abs(proposal-truth.at[soil,"20"]),
                                old_emd=curve.emd,new_emd=gsd.emd(y,p)[0]))
    result = pd.DataFrame(results)
    assert len(result)==66 and not result.duplicated(["exam","sample_id"]).any()
    result["gain"] = result.old_emd-result.new_emd
    np.testing.assert_allclose(result.gain,(result.old_mae-result.new_mae)*gsd.DLOG[gsd.COLS.index("20")],atol=1e-7)
    result.to_csv(OUT/"kaggle_20mm_alan_mutlak.csv",index=False,float_format="%.9f")
    print(result.groupby("exam")[["old_mae","new_mae","old_emd","new_emd","gain"]].mean().round(4))
    r=result[result.exam.eq("24")]
    print(r.loc[r.gain.abs().gt(1e-7),["sample_id","slope_new","gain"]].to_string(index=False))


if __name__ == "__main__":
    main()
