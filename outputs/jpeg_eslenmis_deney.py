"""Aynı 4 px/mm görüntüyü, eğitim JPEG tablosuyla ve ek sıkıştırmasız karşılaştır."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin

from saha_ve_egri_deneyi import AREA, SITE, folds, area_updates, bounds, band_curve
from v3_knn_esit_karisim import COLS, cache, keys, labels, out, predict, training, work
import features
import gsd

MEASUREMENTS = out / "jpeg_eslenmis_olculer.csv"
METHODS = ("kontrol", "jpeg")


def header(path):
    with Image.open(path) as im:
        return im.quantization, JpegImagePlugin.get_sampling(im)


def extract(photos):
    donors = photos[(photos.split == "train") & photos.phone.isin(["Motorola_Edge", "Samsung_A52"])]
    profile = header(donors.path.iloc[0])
    assert len(donors) == 124 and all(header(p) == profile for p in donors.path)
    tables, subsampling = profile
    canonical = features.load_canonical

    def measure(row):
        rgb, ppm = canonical(row.path, row.phone)
        encoded = BytesIO()
        Image.fromarray(rgb).save(encoded, format="JPEG", qtables=tables, subsampling=subsampling)
        encoded.seek(0)
        with Image.open(encoded) as im:
            assert im.quantization == tables and JpegImagePlugin.get_sampling(im) == subsampling
            jpeg = np.asarray(im.convert("RGB"))
        assert jpeg.shape == rgb.shape
        pixel_change = np.abs(rgb.astype(np.int16) - jpeg).mean()
        records = []
        for method, image in zip(METHODS, (rgb, jpeg)):
            values = features.image_features(image, ppm)
            records.append(dict(file=row.file, sample_id=row.sample_id, split=row.split, phone=row.phone,
                                aug=0, eff_ppm=ppm, method=method, pixel_mae=pixel_change,
                                **{k:values[k] for k in keys}))
        return records

    cv2.setNumThreads(1)
    records = []
    # The original loader above supplies both arms; reuse the unmodified v3 extractor on their RGB arrays.
    with patch.object(features, "load_canonical", lambda image, ppm: (image, ppm)):
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, rows in enumerate(pool.map(measure, photos.itertuples()), 1):
                records.extend(rows)
                if i % 25 == 0 or i == len(photos):
                    print(f"Ölçülen çekim {i}/{len(photos)}", flush=True)
    pd.DataFrame(records).to_csv(MEASUREMENTS, index=False, float_format="%.12g")


def get_curves(refs, queries, train_areas, test_areas):
    raw = predict(refs, queries)["v3"]
    result = raw.copy()
    for sample, row in raw.iterrows():
        obs = test_areas.loc[sample]
        updates = area_updates(train_areas, obs, labels.loc[train_areas.index])
        base = bounds(row.values, {s:v for s,v in updates.items() if s=="20"})
        result.loc[sample] = band_curve(base, train_areas, obs)
    return result


def summarize(scores):
    groups = []
    for exam, samples in scores[scores.exam!="test"].groupby("exam"):
        errors = samples.pivot(index="sample_id", columns="method", values="emd")
        gain = errors.kontrol-errors.jpeg
        sites = samples.drop_duplicates("sample_id").set_index("sample_id").site
        groups.append(dict(exam=exam, n=len(gain), kontrol=errors.kontrol.mean(), jpeg=errors.jpeg.mean(),
                           kazanc=gain.mean(), ortanca_kazanc=gain.median(), iyilesen=(gain>1e-7).sum(),
                           kotulesen=(gain < -1e-7).sum(), tek_haric_min=((gain.sum()-gain)/(len(gain)-1)).min(),
                           saha_esit_kazanc=gain.groupby(sites).mean().mean()))
    summary = pd.DataFrame(groups)
    summary.to_csv(out/"jpeg_eslenmis_ozet.csv",index=False,float_format="%.9f")
    print(summary.round(4).to_string(index=False))


def evaluate(frame):
    assert len(frame)==324 and not frame.duplicated(["file","method"]).any()
    control = frame[frame.method=="kontrol"].set_index("file").sort_index()
    encoded = frame[frame.method=="jpeg"].set_index("file").sort_index()
    archived = cache[cache.aug==0].set_index("file").sort_index()
    pd.testing.assert_index_equal(control.index, archived.index)
    pd.testing.assert_index_equal(control.index, encoded.index)
    assert np.isfinite(frame[keys]).all().all()
    areas = pd.read_csv(out/"maske_elek_alanlari.csv")
    previous = pd.read_csv(out/"saha_ve_egri_sonuclar.csv").query("model=='referans'").set_index(["exam","sample_id"])
    submission = pd.read_csv(out/"submission_v3_orta_elek_aralik.csv",index_col="sample_id")
    table = []
    columns = ["a_"+s for s in AREA]
    jobs = [*folds(), ("test", submission.index, labels.index, None, None)]
    for exam, hold, ids, src, dst in jobs:
        refs = training[training.sample_id.isin(ids)]
        source = areas[(areas.split=="train") & areas.sample_id.isin(ids)]
        subset = frame[frame.sample_id.isin(hold)]
        dest = areas[areas.sample_id.isin(hold)]
        if src:
            refs, source = refs[refs.phone==src], source[source.phone==src]
            subset, dest = subset[subset.phone==dst], dest[dest.phone==dst]
        assert set(refs.sample_id)==set(ids) and set(hold).isdisjoint(ids)
        train_areas = source.groupby("sample_id")[columns].mean()
        test_areas = dest.groupby("sample_id")[columns].mean()
        preds = {m:get_curves(refs, subset[subset.method==m], train_areas, test_areas) for m in METHODS}
        # A fresh local extraction can differ slightly from the published OpenCV cache.
        stored = archived[archived.sample_id.isin(hold)]
        if dst:
            stored = stored[stored.phone==dst]
        original_curve = get_curves(refs, stored, train_areas, test_areas)
        for sample in hold:
            old = original_curve.loc[sample].values
            if exam=="test":
                assert np.max(np.abs(old-submission.loc[sample].values)) < .001
            else:
                np.testing.assert_allclose(gsd.emd(labels.loc[sample],old)[0],previous.loc[(exam,sample),"emd"],atol=1e-7,rtol=0)
            diff = preds["jpeg"].loc[sample]-preds["kontrol"].loc[sample]
            for m, curves in preds.items():
                curve = curves.loc[sample].values
                assert np.isfinite(curve).all() and (np.diff(curve)>=-1e-8).all() and curve.min()>=0 and curve[-1]==100
                err = gsd.emd(labels.loc[sample],curve)[0] if exam!="test" else np.nan
                table.append(dict(exam=exam,sample_id=sample,site=SITE.get(sample),method=m,emd=err,
                                  cift_egri_farki=gsd.emd(preds["kontrol"].loc[sample],preds["jpeg"].loc[sample])[0],
                                  fark_0063=diff["0.063"],arsivden_uzaklik=gsd.emd(old,curve)[0],**dict(zip(COLS,curve))))
    scores = pd.DataFrame(table)
    assert len(scores)==210 and not scores.duplicated(["exam","sample_id","method"]).any()
    scores.to_csv(out/"jpeg_eslenmis_sonuclar.csv",index=False,float_format="%.9f")
    scale = control[control.split=="train"].groupby("sample_id")[keys].mean().std()
    assert (scale>0).all()
    shift = ((encoded[keys]-control[keys])/scale).abs()
    by_phone = shift.groupby(control.phone.replace({"iPhone_16":"iPhone16"})).median()
    by_phone.to_csv(out/"jpeg_eslenmis_doku.csv",float_format="%.8f")
    summarize(scores)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild",action="store_true")
    args = parser.parse_args()
    if args.rebuild or not MEASUREMENTS.is_file():
        extract(gsd.list_images(work/"soil-data"))
    evaluate(pd.read_csv(MEASUREMENTS))
