"""63 mm sabitken, 20 mm geçişini diğer eğitim topraklarının eğrilerinden tahmin et."""
import numpy as np
import pandas as pd

from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels, out
from gsd import emd


def interval_fraction(ids):
    truth = labels.loc[ids]
    span = truth["63"]-truth["6.3"]
    span = span[span>0]
    ratios = ((truth.loc[span.index,"63"]-truth.loc[span.index,"20"])/span).sort_values()
    # Kütle payı mutlak hatasını en aza indiren ortak oran, açıklık ağırlıklı ortancadır.
    weights = span.loc[ratios.index].cumsum()
    return ratios.iloc[weights.searchsorted(weights.iloc[-1]/2)]


def main():
    archive = pd.read_csv(out/"ince_paya_2mm_sonuclar.csv").set_index(["exam","sample_id"])
    old = pd.read_csv(out/"ikinci_olcek_egitim_sonuclar.csv").query("model=='tam_gecis'").set_index(["exam","sample_id"])
    photos = pd.read_csv(out/"maske_elek_alanlari.csv").query("split=='train'")
    assert pd.read_csv(out/"ikinci_olcek_egitim_fotograflar.csv").n63.eq(0).all()
    rows = []
    for exam,hold,ids,_,phone in folds():
        assert set(hold).isdisjoint(ids)
        ratio = interval_fraction(ids)
        seen = (photos if phone is None else photos[photos.phone==phone]).groupby("sample_id").n_20.max()
        for s in hold:
            key = (exam,s)
            baseline = archive.loc[key,COLS].copy()
            baseline["63"] = 100.
            y = labels.loc[s,COLS]
            ref_score = emd(y,baseline)[0]
            assert abs(ref_score-old.loc[key,"emd"]) < 1e-7
            estimate = baseline["63"]-ratio*(baseline["63"]-baseline["6.3"])
            for method,p20 in [("mevcut",baseline["20"]),("egri",estimate),
                               ("aday_yoksa_egri",estimate if seen[s]==0 else baseline["20"])]:
                pred = baseline.copy()
                pred["20"] = p20
                assert np.diff(pred).min()>=-1e-7
                rows.append(dict(exam=exam,sample_id=s,model=method,aday=seen[s]>0,
                                 oran=ratio,gecen_6_3=pred["6.3"],iri_tahmin=100-p20,iri_gercek=100-y["20"],
                                 hata20=abs(p20-y["20"]),emd=emd(y,pred)[0]))
    result=pd.DataFrame(rows)
    assert len(result)==285 and not result.duplicated(["exam","sample_id","model"]).any()
    result.to_csv(out/"iri_pay_20_63_egri.csv",index=False,float_format="%.8f")
    print(result.groupby(["exam","model"])[["hata20","emd"]].mean().round(4))
    primary=result.query("exam=='toprak'")
    print(primary[primary.sample_id.isin(["H037","H038"])].round(3).to_string(index=False))
    print(primary.groupby(["aday","model"]).hata20.mean())
    d=primary.pivot(index="sample_id",columns="model",values="hata20")
    print((d.sub(d.mevcut,axis=0)<-1e-7).sum(),(d.sub(d.mevcut,axis=0)>1e-7).sum())


if __name__=="__main__":
    main()
