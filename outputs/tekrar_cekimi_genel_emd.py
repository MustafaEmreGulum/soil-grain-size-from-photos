"""Tekrarlı çekimlerden elde edilen iki otomatik değişikliği tüm eleklerde karşılaştır."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coklu_elek_deneyi import bounds
from ince_paya_2mm_aktarimi import transfer
from orta_elek_kismi_duzeltme import BANDS, band_curve
from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, cache, keys, labels, original, out, published, training, work
import gsd


def raw_curves(refs, photos):
    ids = refs.sample_id.unique()
    targets = pd.DataFrame(gsd.curve_to_logq(labels.loc[ids, COLS].to_numpy()), index=ids)
    assert photos.file.is_unique and set(photos.file).isdisjoint(photos.sample_id)
    both = pd.concat([photos, photos.assign(sample_id=photos.file)], ignore_index=True)
    idx, _, values = published.fit_predict(refs, both, targets, keys)
    return pd.DataFrame(values, index=idx, columns=COLS)


def corrected(raw, obs, ref, factor, pass63):
    updates = {"20": 100-factor*obs.a_20} if factor > 0 and obs.a_20 > 0 else {}
    base = raw.to_numpy()
    curve = transfer(band_curve(bounds(base, updates), ref, obs), base)
    curve[COLS.index("63")] = pass63
    assert np.all(np.diff(curve) >= -1e-8)
    return curve


def compare(sample, subset, raw, ref, factor, pass63):
    obs = subset.mean()
    base = corrected(raw.loc[sample], obs, ref, factor, pass63)
    single = np.stack([corrected(raw.loc[f], row, ref, factor, pass63) for f,row in subset.iterrows()])
    area_mass = factor*subset.a_20
    votes = (subset.a_20 > 0) & (area_mass > 100-raw.loc[subset.index, "6.3"]+1e-7)
    candidate = base.copy()
    j,k,l = (COLS.index(s) for s in ("2","6.3","20"))
    quorum = votes.sum() >= 2 and votes.sum() > len(subset)/2
    if quorum and area_mass.mean() > 100-base[l]:
        # Keep the proportion of the two original sub-bands between 2 and 20 mm.
        # A collapsed span stays collapsed when its coarse endpoint is lowered.
        span = base[l]-base[j]
        pass20 = np.clip(100-area_mass.mean(), base[j], base[l+1])
        if span > 0:
            candidate[k] = base[j]+(pass20-base[j])*(base[k]-base[j])/span
        candidate[l] = pass20
    averaged = single.mean(axis=0)
    # Follow-up, chosen after the whole-curve average was scored.
    coarse_average = bounds(base, {"6.3": averaged[k], "20": averaged[l]})
    curves = {"mevcut":base, "egri_ortalamasi":averaged, "cogunluk":candidate, "iki_elek_ortalamasi":coarse_average}
    for curve in curves.values():
        assert np.isfinite(curve).all() and (np.diff(curve)>=-1e-8).all() and curve[0]>=0 and curve[-1]==100
    return curves, single, votes


def factor_for(ref):
    x = ref.a_20
    return max(0., (x @ (100-labels.loc[x.index,"20"]))/(x @ x)) if x @ x > 0 else 0.


def kaggle():
    photos = pd.read_csv(out/"maske_elek_alanlari.csv").query("split=='train'").set_index("file")
    coarse = pd.read_csv(out/"ikinci_olcek_egitim_fotograflar.csv").set_index("file")
    assert photos.index.is_unique and set(photos.index)==set(coarse.index) and coarse.n63.eq(0).all()
    archive = pd.read_csv(out/"ustten_20_63_sonuclar.csv").query("model=='mevcut'").set_index(["exam","sample_id"])
    acols = ["a_"+s for s in BANDS]
    rows = []
    for exam, hold, ids, src, dst in folds():
        train = training[training.sample_id.isin(ids)]
        query = original[original.sample_id.isin(hold)]
        src_area = photos[photos.sample_id.isin(ids)]
        if src:
            train, query = train[train.phone==src], query[query.phone==dst]
            src_area = src_area[src_area.phone==src]
        assert set(ids).isdisjoint(hold) and set(query.sample_id)==set(hold)
        ref = src_area.groupby("sample_id")[acols].mean().loc[ids]
        raw = raw_curves(train, query)
        factor = factor_for(ref)
        for sample in hold:
            subset = photos.loc[query.loc[query.sample_id==sample,"file"], acols]
            curves, _, votes = compare(sample, subset, raw, ref, factor, 100.)
            truth = labels.loc[sample, COLS].to_numpy()
            saved = archive.loc[(exam,sample)]
            np.testing.assert_allclose(curves["mevcut"], saved[COLS].to_numpy(float), atol=1e-7, rtol=0)
            for model, curve in curves.items():
                score = gsd.emd(truth,curve)[0]
                if model=="mevcut":
                    np.testing.assert_allclose(score,saved.emd,rtol=0,atol=1e-7)
                rows.append(dict(exam=exam,sample_id=sample,model=model,n=len(subset),votes=int(votes.sum()),
                                 emd=score, delta_emd=score-saved.emd,
                                 mae20=abs(curve[COLS.index("20")]-truth[COLS.index("20")]),
                                 mae6_3=abs(curve[COLS.index("6.3")]-truth[COLS.index("6.3")]),
                                 **dict(zip(COLS,curve))))
    df = pd.DataFrame(rows)
    assert len(df)==380 and not df.duplicated(["exam","sample_id","model"]).any()
    df.to_csv(out/"tekrar_cekimi_genel_emd.csv",index=False,float_format="%.9f")
    print(df.groupby(["exam","model"])[["emd","mae20","mae6_3"]].mean().round(4).to_string())


def submission():
    base = pd.read_csv(out/"submission_v3_63mm_kontrollu.csv",index_col="sample_id")
    template = pd.read_csv(work/"soil-data/sample_submission.csv",index_col="sample_id")
    assert base.index.equals(template.index) and list(base)==list(template)==COLS
    acols = ["a_"+s for s in BANDS]
    areas = pd.read_csv(out/"maske_elek_alanlari.csv")
    ref = areas.query("split=='train'").groupby("sample_id")[acols].mean()
    test = areas.query("split=='test'").set_index("file")
    photos = cache.query("split=='test' and aug==0")
    assert set(photos.file)==set(test.index) and len(photos)==len(test)==35
    assert photos.set_index("file").sample_id.sort_index().equals(test.sample_id.sort_index())
    raw = raw_curves(training, photos)
    factor = factor_for(ref)
    result = base.copy()
    for sample, saved in base.iterrows():
        sub = test.loc[test.sample_id==sample,acols]
        pooled = corrected(raw.loc[sample],sub.mean(),ref,factor,100.)
        # The production file rounds intermediate predictions to four decimals.
        np.testing.assert_allclose(pooled[6:9],saved.to_numpy()[6:9],atol=.00025,rtol=0)
        single = [corrected(raw.loc[f],row,ref,factor,100.) for f,row in sub.iterrows()]
        avg = np.mean(single,axis=0)
        # Only these two points are taken from the photo curves. Use the deployed
        # 2 and 63 mm values as bounds, including the final coarse-scale check.
        result.loc[sample,COLS] = bounds(saved.to_numpy(),{"6.3":avg[COLS.index("6.3")],"20":avg[COLS.index("20")]})
    path = out/"submission_v3_iki_elek_cekim_ortalamasi.csv"
    result.to_csv(path,float_format="%.6f")
    saved = pd.read_csv(path,index_col="sample_id")
    assert saved.index.equals(template.index) and list(saved)==COLS and len(saved)==10
    assert np.isfinite(saved).all().all() and saved.ge(0).all().all() and saved.le(100).all().all()
    assert (np.diff(saved.to_numpy(),axis=1)>=0).all() and saved["200"].eq(100).all()
    pd.testing.assert_frame_equal(saved.drop(columns=["6.3","20"]),base.drop(columns=["6.3","20"]))
    print((saved[["6.3","20"]]-base[["6.3","20"]]).round(4))


def montreal():
    import cv2
    from iri_alan_iki_aralik import measure
    from montreal_20mm_sinama import NATIVE_PPM, PHOTOS, PHYSICAL_CAMERA, features
    meta = pd.read_csv(PHOTOS.parent/"psd_mass_path_80_to_0_08mm_no_ballast.csv",dtype={"sample_id":str})
    meta = meta.loc[meta.sample_id.isin(["0196","0201"]) & meta.dry_or_moist.eq("dry")].copy()
    meta["file"] = meta.image_path.map(lambda s:Path(s).name)
    meta = meta.set_index("file",drop=False).sort_index()
    assert meta.index.is_unique and meta.groupby("sample_id").size().to_dict()=={"0196":6,"0201":7}
    assert meta.groupby("sample_id").file.apply(lambda x:x.str.rsplit("_",n=1).str[0].nunique()).eq(1).all()
    areas = measure(meta)
    cv2.setNumThreads(1)
    inputs = []
    with patch.dict(gsd.PHONES,{PHYSICAL_CAMERA:(8192,NATIVE_PPM)}):
        for row in meta.itertuples():
            values = features.image_features(PHOTOS/row.file, PHYSICAL_CAMERA)
            inputs.append(dict(sample_id=row.sample_id,file=row.file,**{k:values[k] for k in keys}))
    raw = raw_curves(training,pd.DataFrame(inputs)).round(4)
    ref = pd.read_csv(out/"maske_elek_alanlari.csv").query("split=='train'").groupby("sample_id")[list(areas)].mean()
    factor = factor_for(ref)
    old = pd.read_csv(out/"iri_alan_iki_aralik_sonuclar.csv",dtype={"sample_id":str}).set_index("sample_id")
    photo_rows, soil_rows = [], []
    idx = COLS.index("20")
    for sample, group in meta.groupby("sample_id"):
        assert group["%_20mm"].nunique()==1
        lab = 100-group["%_20mm"].iloc[0]
        subset = areas.loc[group.index]
        # The final 63 mm control touches only that column. Only 20 mm is reported externally.
        curves, single, votes = compare(sample,subset,raw,ref,factor,100.)
        # The two-sieve result must also fit the tighter raw 63 mm bound.
        # Thus the final optional 63 mm check cannot change this reported 20 mm value.
        assert curves["iki_elek_ortalamasi"][idx] <= raw.at[sample,"63"] + 1e-8
        first = group.index[0]
        np.testing.assert_allclose([subset.at[first,"a_20"],100-single[0,idx]],
                                   old.loc[sample,["a_20","mass20_model"]].to_numpy(float),atol=.00011,rtol=0)
        for f,curve in zip(group.index,single):
            photo_rows.append(dict(sample_id=sample,file=f,area20=subset.at[f,"a_20"],
                                   from_area=factor*subset.at[f,"a_20"],cap6_3=100-raw.at[f,"6.3"],
                                   predicted20=100-curve[idx],lab20=lab,vote=votes.at[f]))
        for model,curve in curves.items():
            soil_rows.append(dict(sample_id=sample,model=model,n=len(group),votes=int(votes.sum()),
                                  predicted20=100-curve[idx],lab20=lab))
    for filename,records in [("cekimler",photo_rows),("topraklar",soil_rows)]:
        frame = pd.DataFrame(records)
        frame.to_csv(out/f"tekrar_cekimi_montreal_{filename}.csv",index=False,float_format="%.6f")
        print(frame.round(3).to_string(index=False))


def chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = pd.read_csv(out/"tekrar_cekimi_genel_emd.csv")
    local = scores.query("exam=='toprak' and model=='iki_elek_ortalamasi'").sort_values("delta_emd")
    photos = pd.read_csv(out/"tekrar_cekimi_montreal_cekimler.csv",dtype={"sample_id":str})
    fig, axes = plt.subplots(1,3,figsize=(12,4.3),gridspec_kw={"width_ratios":[1.25,1,1]},layout="constrained")
    ax = axes[0]
    ranks = np.arange(len(local))
    ax.bar(ranks,local.delta_emd,color=np.where(local.delta_emd<0,"#14785a","#c8643b"))
    ax.axhline(0,color="#38424b",linewidth=.8)
    ax.set_xticks([0,1,len(local)-1],local.sample_id.iloc[[0,1,-1]].tolist(),rotation=40)
    ax.set_xlabel("Hata farkına göre sıralanan 24 toprak")
    ax.set_ylabel("Yeni EMD − mevcut EMD")
    ax.set_title("Genel skora katkı",loc="left")
    for ax,(sample,group) in zip(axes[1:],photos.groupby("sample_id")):
        n = np.arange(1,len(group)+1)
        ax.plot(n,group.from_area,"^--",color="#bb621f",label="Alandan hesap")
        ax.plot(n,group.predicted20,"o-",color="#28539b",label="Son tahmin")
        ax.axhline(group.lab20.iloc[0],color="#14785a",label="Laboratuvar")
        ax.set_xticks(n)
        ax.set_ylim(0,50)
        ax.set_xlabel("Aynı tepsinin kuru çekim numarası")
        ax.set_title(f"Montréal {sample}",loc="left")
    axes[1].set_ylabel("20 mm üstü kütle (%)")
    axes[-1].legend(fontsize=8,loc="upper right",frameon=False)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
    fig.savefig(out/"tekrar_cekimi_karsilastirma.png",dpi=150,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    tasks = {"kaggle":kaggle,"montreal":montreal,"gonderim":submission,"grafik":chart}
    parser.add_argument("veri",choices=tasks)
    with threadpool_limits(1):
        tasks[parser.parse_args().veri]()
