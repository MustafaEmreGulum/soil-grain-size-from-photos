"""Predict laboratory passing at 4 and 8 mm from their mask areas; change only 6.3 mm."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from skimage.measure import regionprops

from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels, out, work
from gsd import DLOG, emd

SIZES = np.array([4., 6.3, 8.])
AREA = ["alan4", "alan63", "alan8"]
WEIGHT = np.log(6.3/4) / np.log(8/4)
K = COLS.index("6.3")


def lab_targets():
    rows = []
    for soil in labels.index:
        table = pd.read_csv(work/"soil-data"/"zenodo_14725633"/f"{soil}.csv").dropna(how="all")
        raw = table.set_index("diam").finer.sort_index()
        assert raw.index.is_unique and raw.iloc[-1] == 100
        values = []
        for size in (4., 8.):
            if size <= raw.index.max():
                values.append(raw.loc[size])
            else:
                values.append(100.)
        rows.append(values)
    result = pd.DataFrame(rows, columns=["p4", "p8"], index=labels.index)
    reconstructed = (1-WEIGHT)*result.p4 + WEIGHT*result.p8
    np.testing.assert_allclose(reconstructed, labels["6.3"], atol=.000051, rtol=0)
    result["p63"] = labels["6.3"]
    return result


def measure(archive):
    photos = pd.read_csv(out/"maske_elek_alanlari.csv").query("split == 'train'").set_index("file")
    assert len(photos) == 127 and photos.index.is_unique
    rows = []
    with ZipFile(archive) as z:
        for name, photo in photos.iterrows():
            with np.load(BytesIO(z.read(name+".npz"))) as data:
                mask = data["mask"]
            area = np.zeros(3)
            count63 = 0
            for label, box in enumerate(find_objects(mask), 1):
                if box is None or 2*min(s.stop-s.start for s in box) <= 4*4.5:
                    continue
                grain, = regionprops((mask[box] == label).astype("uint8"))
                diameter = grain.axis_minor_length/4.5
                area += grain.area * (diameter > SIZES)
                count63 += int(diameter > 6.3)
            percent = 100*area/mask.size
            assert count63 == photo["n_6.3"] and np.all(np.diff(percent) <= 0)
            np.testing.assert_allclose(percent[1], photo["a_6.3"], atol=1e-8, rtol=0)
            rows.append(dict(file=name, sample_id=photo.sample_id, phone=photo.phone, **dict(zip(AREA,percent))))
    return pd.DataFrame(rows)


def evaluate(photos, lab):
    archive = pd.read_csv(out/"tekrar_cekimi_genel_emd.csv").query("model=='mevcut'").set_index(["exam","sample_id"])
    means = {None:photos.groupby("sample_id")[AREA].mean()}
    means.update({p:b.groupby("sample_id")[AREA].mean() for p,b in photos.groupby("phone")})
    rows = []
    for exam, hold, ids, src, dst in folds():
        assert set(hold).isdisjoint(ids)
        x = means[src].loc[ids].to_numpy()
        y = 100-lab.loc[ids,["p4","p63","p8"]].to_numpy()
        denom = (x*x).sum(axis=0)
        assert np.all(denom > 0)
        factor = (x*y).sum(axis=0)/denom
        for soil in hold:
            p4, direct, p8 = np.clip(100-means[dst].loc[soil].to_numpy()*factor, 0, 100)
            reversed_pair = p4 > p8
            if reversed_pair:
                p4 = p8 = (p4+p8)/2
            mix = (1-WEIGHT)*p4 + WEIGHT*p8
            saved = archive.loc[(exam,soil)]
            base = saved[COLS].to_numpy(float)
            truth = labels.loc[soil,COLS].to_numpy()
            np.testing.assert_allclose(emd(truth,base)[0], saved.emd, atol=1e-7, rtol=0)
            for model, proposal in (("mevcut",base[K]),("tek_6_3",direct),("4_8",mix)):
                curve = base.copy()
                curve[K] = np.clip(proposal,base[K-1],base[K+1])
                error = emd(truth,curve)[0]
                mae = abs(curve[K]-truth[K])
                np.testing.assert_allclose(error-saved.emd,DLOG[K]*(mae-saved.mae6_3),atol=1e-7,rtol=0)
                assert np.all(np.diff(curve)>=-1e-8)
                rows.append(dict(exam=exam,sample_id=soil,model=model,n_train=len(ids),
                                 p4_pred=p4,p4_lab=lab.at[soil,"p4"],p8_pred=p8,p8_lab=lab.at[soil,"p8"],
                                 p63_raw=proposal,p63_pred=curve[K],p63_lab=truth[K],
                                 clipped=abs(curve[K]-proposal)>1e-7,reversed_pair=reversed_pair,
                                 mae=mae,emd=error,gain=saved.emd-error))
    result = pd.DataFrame(rows)
    assert len(result)==285 and not result.duplicated(["exam","sample_id","model"]).any()
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_archive",type=Path)
    args=parser.parse_args()
    lab=lab_targets()
    features=measure(args.mask_archive)
    features.to_csv(out/"kaggle_4_8_elek_alanlari.csv",index=False,float_format="%.10f")
    result=evaluate(features,lab)
    result.to_csv(out/"kaggle_4_8_elek_sonuclar.csv",index=False,float_format="%.9f")
    print(result.groupby(["exam","model"])[["mae","emd","gain"]].mean().round(4))
