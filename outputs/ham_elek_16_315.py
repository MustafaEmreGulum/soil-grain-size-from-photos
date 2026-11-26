"""Özgün 16 ve 31,5 mm etiketlerini görüntüden tahmin edip 20 mm'ye taşı."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from skimage.measure import regionprops

from coklu_elek_deneyi import TEXTURE, photo_features, soil_features
from ince_0063_oran import fit_share
from iri_63_gorsel import PPM
from saha_ve_egri_deneyi import SITE, folds
from ustten_20_63 import K, redistribute
from v3_knn_esit_karisim import COLS, labels, out, work
from gsd import emd

WEIGHT = np.log(20/16)/np.log(31.5/16)


def lab_points():
    rows = []
    for sample in labels.index:
        table = pd.read_csv(work/"soil-data"/"zenodo_14725633"/f"{sample}.csv").dropna(how="all")
        assert table.notna().all().all()
        raw = table.set_index("diam").finer.sort_index()
        assert raw.index.is_unique and raw.between(0,100).all() and raw.iloc[-1]==100
        points = {}
        for d in (16.,31.5):
            if d in raw.index:
                points[d] = raw.loc[d]
            else:
                assert raw.index.max()<d
                points[d] = 100.
        mid = (1-WEIGHT)*points[16.]+WEIGHT*points[31.5]
        rows.append(dict(sample_id=sample,p16=points[16.],p315=points[31.5],p20_ham=mid,
                         p20_kaggle=labels.loc[sample,"20"],iki_olcum=16. in raw.index and 31.5 in raw.index))
    result = pd.DataFrame(rows).set_index("sample_id")
    np.testing.assert_allclose(result.p20_ham,result.p20_kaggle,atol=.000051,rtol=0)
    result.to_csv(out/"ham_elek_16_315_etiket.csv",float_format="%.8f")
    return result


def areas(photos, archive):
    records = []
    with ZipFile(archive) as z:
        for row in photos.itertuples():
            with np.load(BytesIO(z.read(row.file+".npz"))) as data:
                mask = data["mask"]
            counts = np.zeros(3)
            for label,box in enumerate(find_objects(mask),1):
                if box is None or 2*min(s.stop-s.start for s in box)<=16*PPM:
                    continue
                grain, = regionprops((mask[box]==label).astype(np.uint8))
                diameter = grain.axis_minor_length/PPM
                counts += grain.area*(diameter>np.array([16.,20.,31.5]))
            a16,a20,a315 = 100*counts/mask.size
            np.testing.assert_allclose(a20,row.a_20,atol=1e-7,rtol=0)
            records.append(dict(file=row.file,a16=a16,a315=a315))
    return photos.merge(pd.DataFrame(records),on="file",validate="one_to_one")


def main(archive):
    lab = lab_points()
    photos = areas(photo_features(),archive)
    features = {}
    for phone in [None,*photos.phone.unique()]:
        frame = photos if phone is None else photos[photos.phone==phone]
        soils = soil_features(frame).join(frame.groupby("sample_id")[["a16","a315"]].mean())
        for col in ("a16","a315","a_20"):
            soils[col] -= soils.a_63
        features[phone] = soils
    old = pd.read_csv(out/"ustten_20_63_sonuclar.csv").set_index(["exam","sample_id","model"])
    results = []
    for exam,hold,ids,src,dst in folds():
        assert set(ids).isdisjoint(hold)
        ref,query = features[src].loc[ids],features[dst].loc[hold]
        # Yüzdeler 63 mm altında kalan kütle içinde öğrenilir. Test girdisi yalnız görüntüdür.
        cap = labels.loc[ids,"63"]
        def passing(target, area):
            retained = (cap-target).clip(lower=0)
            return 1-fit_share(ref,query,retained,cap,[*TEXTURE,area])["doku"]
        direct = passing(labels.loc[ids,"20"],"a_20")
        small,large = passing(lab.loc[ids,"p16"],"a16"),passing(lab.loc[ids,"p315"],"a315")
        reversed_pair = small>large
        center=(small+large)/2
        small=np.where(reversed_pair,center,small)
        large=np.where(reversed_pair,center,large)
        mid=(1-WEIGHT)*small+WEIGHT*large
        ref_shape=labels.loc[ids,COLS[:K]].div(labels.loc[ids,"20"].replace(0,np.nan),axis=0).median().to_numpy()
        for i,s in enumerate(hold):
            key=(exam,s)
            base=old.loc[(*key,"mevcut"),COLS].to_numpy(float)
            truth=labels.loc[s,COLS].to_numpy(float)
            ceiling=base[K+1]
            for method,curve in [("mevcut",base),("dogrudan",redistribute(base,1-direct[i],ref_shape)),
                                 ("iki_elek",redistribute(base,1-mid[i],ref_shape))]:
                assert np.isfinite(curve).all() and curve.min()>=0 and np.all(np.diff(curve)>=-1e-8)
                np.testing.assert_array_equal(curve[K+1:],base[K+1:])
                score=emd(truth,curve)[0]
                if method!="iki_elek":
                    saved=old.loc[(*key,"mevcut" if method=="mevcut" else "doku_alan")]
                    np.testing.assert_allclose(score,saved.emd,atol=1e-7,rtol=0)
                results.append(dict(exam=exam,sample_id=s,site=SITE[s],model=method,iki_olcum=lab.loc[s,"iki_olcum"],
                                    p16_tahmin=ceiling*small[i] if method=="iki_elek" else np.nan,
                                    p315_tahmin=ceiling*large[i] if method=="iki_elek" else np.nan,
                                    sira_duzeltildi=bool(method=="iki_elek" and reversed_pair[i]),
                                    iri_gercek=100-truth[K],iri_tahmin=100-curve[K],hata20=abs(curve[K]-truth[K]),emd=score))
    df=pd.DataFrame(results)
    assert len(df)==285 and not df.duplicated(["exam","sample_id","model"]).any()
    df.to_csv(out/"ham_elek_16_315_sonuclar.csv",index=False,float_format="%.9f")
    print(df.groupby(["exam","model"])[["hata20","emd"]].mean().round(4))
    primary=df.query("exam=='toprak'")
    print(primary.groupby(["iki_olcum","model"])[["hata20","emd"]].mean().round(4))
    print(primary[primary.sample_id.isin(["H037","H038"])][["sample_id","model","iri_tahmin"]].to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    main(parser.parse_args().archive)
