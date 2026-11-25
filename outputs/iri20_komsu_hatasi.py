"""20 mm üstü payın eksik tahminini iç sınama hatalarıyla düzelt."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from coklu_elek_deneyi import TEXTURE, area_updates, bounds, photo_features, soil_features
from saha_ve_egri_deneyi import SITE, folds
from ustten_20_63 import K, redistribute
from v3_knn_esit_karisim import COLS, labels, original, out, predict, training
from gsd import emd


def baseline20(ids, hold, source, destination, features):
    train_rows = training[training.sample_id.isin(ids)]
    query_rows = original[original.sample_id.isin(hold)]
    if source:
        train_rows = train_rows[train_rows.phone==source]
    if destination:
        query_rows = query_rows[query_rows.phone==destination]
    raw = predict(train_rows,query_rows)["v3"]
    train = features[source].loc[ids]
    result = {}
    for s in hold:
        updates = area_updates(train,features[destination].loc[s],labels.loc[ids])
        curve = bounds(raw.loc[s].to_numpy(),{k:v for k,v in updates.items() if k=="20"})
        result[s] = 100-curve[K]
    return pd.Series(result)


def inputs(soil, retained):
    frame = soil.loc[retained.index,TEXTURE].copy()
    frame["alan_20_63"] = soil.loc[retained.index,"a_20"]-soil.loc[retained.index,"a_63"]
    frame["mevcut_iri"] = retained
    return frame


def main():
    photos=photo_features()
    features={p:soil_features(photos,p) for p in [None,*photos.phone.unique()]}
    ref=pd.read_csv(out/"ustten_20_63_sonuclar.csv").query("model=='mevcut'").set_index(["exam","sample_id"])
    rows=[]
    for exam,hold,ids,src,dst in folds():
        ids=pd.Index(sorted(ids))
        assert set(ids).isdisjoint(hold)
        groups=np.array([SITE[s] if exam=="saha" else s for s in ids])
        if exam=="saha":
            assert set(groups).isdisjoint(SITE[s] for s in hold)
        inner=[]
        for a,b in GroupKFold(n_splits=4).split(ids,groups=groups):
            assert set(groups[a]).isdisjoint(groups[b])
            inner.append(baseline20(ids[a],ids[b],src,src,features))
        retained=pd.concat(inner).loc[ids]
        assert retained.index.is_unique and np.isfinite(retained).all()
        residual=100-labels.loc[ids,"20"]-retained
        base_hold=baseline20(ids,hold,src,dst,features)
        X=inputs(features[src],retained)
        scaler=StandardScaler().fit(X)
        ref_points=scaler.transform(X)
        query=scaler.transform(inputs(features[dst],base_hold))
        default_shape=labels.loc[ids,COLS[:K]].div(labels.loc[ids,"20"].replace(0,np.nan),axis=0).median().to_numpy()
        for i,s in enumerate(base_hold.index):
            key=(exam,s)
            base=ref.loc[key,COLS].to_numpy(dtype=float)
            np.testing.assert_allclose(100-base[K],base_hold[s],atol=1e-7,rtol=0)
            nearest=ids[np.argsort(np.linalg.norm(ref_points-query[i],axis=1),kind="stable")[:3]]
            median=float(residual.loc[nearest].median())
            delta=min(base[K],max(0.,median))
            curve=base if delta<1e-7 else redistribute(base,(base[K+1]-base[K]+delta)/base[K+1],default_shape)
            np.testing.assert_array_equal(curve[K+1:],base[K+1:])
            assert np.isfinite(curve).all() and (np.diff(curve)>=-1e-8).all()
            truth=labels.loc[s,COLS]
            baseline_emd=emd(truth,base)[0]
            np.testing.assert_allclose(baseline_emd,ref.loc[key,"emd"],rtol=0,atol=1e-7)
            rows.append(dict(exam=exam,sample_id=s,site=SITE[s],komsular=" ".join(nearest),
                             komsu_hatalari=" ".join(f"{v:.6f}" for v in residual.loc[nearest]),
                             iri_gercek=100-truth["20"],iri_mevcut=100-base[K],iri_yeni=100-curve[K],
                             hata_mevcut=abs(base[K]-truth["20"]),hata_yeni=abs(curve[K]-truth["20"]),
                             emd_mevcut=baseline_emd,emd_yeni=emd(truth,curve)[0],**dict(zip(COLS,curve))))
    data=pd.DataFrame(rows)
    assert len(data)==95 and not data.duplicated(["exam","sample_id"]).any()
    data.to_csv(out/"iri20_komsu_hatasi_sonuclar.csv",index=False,float_format="%.9f")
    metrics=["hata_mevcut","hata_yeni","emd_mevcut","emd_yeni"]
    print(data.groupby("exam")[metrics].mean().round(4))
    print(data.query("sample_id in ['H037','H038'] and exam in ['toprak','saha']")[["exam","sample_id","iri_gercek","iri_mevcut","iri_yeni","komsular"]])
    local=data.query("exam=='toprak'")
    print(local.groupby(local.iri_gercek.eq(0))[metrics].mean())
    delta=local.hata_yeni-local.hata_mevcut
    print("iyilesen/kotulesen/esit",sum(delta< -1e-7),sum(delta>1e-7),sum(abs(delta)<=1e-7))


if __name__=="__main__":
    main()
