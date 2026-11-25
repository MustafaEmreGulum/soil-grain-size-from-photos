"""63 mm kararı sabitken 20–63 mm kütle payını tahmin et, kalan eğriyi orantılı taşı."""
import numpy as np
import pandas as pd

from coklu_elek_deneyi import TEXTURE, photo_features, soil_features
from ince_0063_oran import fit_share
from saha_ve_egri_deneyi import SITE, folds
from v3_knn_esit_karisim import COLS, labels, out
from gsd import emd

K = COLS.index("20")


def redistribute(base, share, default_shape):
    curve = base.copy()
    curve[K] = base[K+1] * (1-share)
    # Alt dağılımın oranları saklı numunenin mevcut tahmininden gelir.
    shape = base[:K]/base[K] if base[K]>0 else default_shape
    curve[:K] = curve[K]*shape
    return curve


def main():
    photos = photo_features()
    by_phone = {phone:soil_features(photos,phone) for phone in [None,*photos.phone.unique()]}
    for frame in by_phone.values():
        frame["alan20_63"] = frame.a_20-frame.a_63
        assert frame.alan20_63.between(0,100).all()
    archive = pd.read_csv(out/"ince_paya_2mm_sonuclar.csv").set_index(["exam","sample_id"])
    scores = pd.read_csv(out/"ikinci_olcek_egitim_sonuclar.csv").query("model=='tam_gecis'").set_index(["exam","sample_id"])
    chain = pd.read_csv(out/"tam_elek_zinciri_sonuclar.csv").query("model=='zincir'").set_index(["exam","sample_id"])
    assert pd.read_csv(out/"ikinci_olcek_egitim_fotograflar.csv").n63.eq(0).all()
    rows=[]
    for exam,hold,ids,src,dst in folds():
        assert set(hold).isdisjoint(ids)
        train, query = by_phone[src].loc[ids], by_phone[dst].loc[hold]
        y = labels.loc[ids]
        target = y["63"]-y["20"]
        shares = {name:fit_share(train,query,target,y["63"],columns)["doku"]
                  for name,columns in [("doku",TEXTURE),("doku_alan",[*TEXTURE,"alan20_63"])]}
        default_shape = y.loc[y["20"]>0,COLS[:K]].div(y.loc[y["20"]>0,"20"],axis=0).median().to_numpy()
        for i,s in enumerate(hold):
            key=(exam,s)
            base=archive.loc[key,COLS].to_numpy(float)
            base[K+1]=100.
            truth=labels.loc[s,COLS].to_numpy(float)
            previous=chain.loc[key]
            np.testing.assert_allclose(1-shares["doku"][i],previous["20"]/previous["63"],atol=1e-8,rtol=0)
            curves={"mevcut":base}
            for name,share in shares.items():
                curves[name]=redistribute(base,share[i],default_shape)
            for name,curve in curves.items():
                assert np.isfinite(curve).all() and (np.diff(curve)>=-1e-8).all() and curve.min()>=0
                np.testing.assert_array_equal(curve[K+1:],base[K+1:])
                score = emd(truth,curve)[0]
                if name=="mevcut":
                    np.testing.assert_allclose(score,scores.loc[key,"emd"],atol=1e-7,rtol=0)
                band=curve[K+1]-curve[K]
                band_truth=truth[K+1]-truth[K]
                rows.append(dict(exam=exam,sample_id=s,site=SITE[s],model=name,alan=query.loc[s,"alan20_63"],
                                 iri_gercek=100-truth[K],iri_tahmin=100-curve[K],aralik_gercek=band_truth,
                                 aralik_tahmin=band,hata_aralik=abs(band-band_truth),hata20=abs(curve[K]-truth[K]),
                                 emd=score,**dict(zip(COLS,curve))))
    result=pd.DataFrame(rows)
    assert len(result)==285 and not result.duplicated(["exam","sample_id","model"]).any()
    result.to_csv(out/"ustten_20_63_sonuclar.csv",index=False,float_format="%.9f")
    print(result.groupby(["exam","model"])[["hata20","hata_aralik","emd"]].mean().round(3))
    print(result.query("exam=='toprak' and sample_id in ['H037','H038']")[["sample_id","model","iri_gercek","iri_tahmin"]].round(3))
    print("Esit saha agirligi:",result.query("exam=='saha'").groupby(["site","model"])[["emd","hata20"]].mean().groupby("model").mean().round(3))


if __name__=="__main__":
    main()
