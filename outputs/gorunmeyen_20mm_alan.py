"""20 mm üstü görünmeyen topraklarda ara boyut alanının ayırt etme gücü."""
import numpy as np
import pandas as pd

from v3_knn_esit_karisim import COLS, labels, out
from gsd import DLOG


def compare(group):
    pos = group.loc[group.lab_iri_pct>0,"ara_alan_pct"].to_numpy()
    neg = group.loc[group.lab_iri_pct==0,"ara_alan_pct"].to_numpy()
    diff = pos[:,None] - neg[None,:]
    return dict(iri_var=len(pos),iri_yok=len(neg),ortanca_var=np.median(pos),ortanca_yok=np.median(neg),
                dogru_cift=int((diff>0).sum()),esit_cift=int((diff==0).sum()),tum_cift=diff.size,
                auc=((diff>0).sum()+.5*(diff==0).sum())/diff.size)


def main():
    photos = pd.read_csv(out/"maske_elek_alanlari.csv").query("split=='train'").copy()
    assert len(photos)==127 and photos.file.is_unique
    photos["ara_alan_pct"] = photos["a_6.3"] - photos["a_20"]
    assert photos.ara_alan_pct.between(0,100).all()
    soils = photos.groupby("sample_id").agg(ara_alan_pct=("ara_alan_pct","mean"),
                                             cekim=("file","size"),aday=("n_20","max"))
    coarse = pd.read_csv(out/"ikinci_olcek_egitim_topraklar.csv",index_col="sample_id").loc[soils.index]
    assert np.array_equal(soils.aday.eq(0),coarse.en_buyuk_kisa_mm.le(20))
    cohort = soils.loc[soils.aday==0,["ara_alan_pct","cekim"]].copy()
    cohort["lab_iri_pct"] = 100-labels.loc[cohort.index,"20"]
    assert len(cohort)==14 and (cohort.lab_iri_pct>0).sum()==6
    cohort.sort_values("ara_alan_pct").to_csv(out/"gorunmeyen_20mm_alan.csv",float_format="%.7f")
    for tag,group in [("14",cohort),("12",cohort.drop(index=["H037","H038"]))]:
        print(tag,compare(group))

    ref = pd.read_csv(out/"ince_paya_2mm_sonuclar.csv").query("exam=='toprak'").set_index("sample_id")
    truth = labels.loc[ref.index,COLS]
    residual = (ref[COLS]-truth).abs().iloc[:,:-1].mul(DLOG,axis=1)
    np.testing.assert_allclose(residual.sum(axis=1),ref.emd,rtol=0,atol=1e-7)
    # Tek bir EMD terimini geriye dönük sıfırlamak yalnızca bir üst sınır verir.
    print("20 mm cohort EMD ceiling",residual.loc[cohort.index,"20"].sum()/len(ref))
    for group in (residual,residual.drop(index=["H037","H038"])):
        print("Top EMD terms",group.mean().nlargest(3).to_dict())


if __name__=="__main__":
    main()
