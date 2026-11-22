"""Dondurulmuş Kaggle modellerini sekiz bağımsız Montréal toprağında sına (FRDR v1.0)."""
import hashlib
import io
from pathlib import PurePosixPath
from unittest.mock import patch
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from threadpoolctl import threadpool_limits
from urllib3.util.retry import Retry

from sirali_elek_modeli import coordinates, fit_predict, select_penalties, soil_features
from v3_knn_esit_karisim import keys, labels, original, out, published, training
import features
import gsd


ROOT = "https://www.frdr-dfdr.ca/repo/files/1/published/publication_1311/submitted_data/"
META_PATH = "metadata/psd_mass_path_80_to_0_0018mm.csv"
FEATURE_FILE = out / "montreal_pilot_olculeri.csv"
PHYSICAL_CAMERA = "Montreal_Canon_R5"


def eligible_images(metadata):
    columns = sorted((c for c in metadata if c.startswith("%_") and c.endswith("mm")),
                     key=lambda c: float(c[2:-2]))
    sizes = np.array([float(c[2:-2]) for c in columns])
    assert sizes[0] <= min(gsd.SIZES) and sizes[-1] == 80
    natural_dry = metadata.dry_or_moist.eq("dry") & metadata.image_path.str.contains(r"_(?:bulk|split[^_]*)_",
                                                                                      case=False, regex=True)
    dry = metadata[natural_dry].sort_values("image_path")
    assert (dry.groupby("sample_id")[columns].nunique(dropna=False) <= 1).all().all()
    first = dry.drop_duplicates("sample_id").set_index("sample_id").sort_index()
    curves = first[columns].to_numpy()
    ok = (np.isfinite(curves).all(axis=1) & (curves[:, 0] >= 0) & (curves[:, -1] == 100)
          & (np.diff(curves, axis=1) >= 0).all(axis=1))
    return first[ok], sizes, columns


def image_targets(picked, sizes, columns):
    targets = np.array([np.r_[np.interp(gsd.LOGX[:-1], np.log10(sizes), row), 100]
                        for row in picked[columns].to_numpy()])
    return picked.image_path.map(lambda s: PurePosixPath(s).name), pd.DataFrame(targets, index=picked.index, columns=gsd.COLS)


def select_images(metadata):
    candidates, sizes, columns = eligible_images(metadata)
    ids = sorted(np.random.default_rng(42).choice(candidates.index.to_numpy(), size=8, replace=False))
    print("Metadata soil IDs:", metadata.sample_id.nunique(), "eligible:", len(candidates), "selected:", ids, flush=True)
    return image_targets(candidates.loc[ids], sizes, columns)


def read_photo(session, name, max_bytes):
    if name != PurePosixPath(name).name or not name.lower().endswith(".jpg"):
        raise ValueError(f"Unexpected photo filename: {name}")
    url = ROOT + "images_soil/" + quote(name)
    with session.get(url, stream=True, timeout=(15, 90)) as res:
        res.raise_for_status()
        image = io.BytesIO()
        for chunk in res.iter_content(512 * 1024):
            if image.tell() + len(chunk) > max_bytes:
                raise ValueError(f"Photo exceeds its transfer limit: {name}")
            image.write(chunk)
    image.seek(0)
    return image


def image_features(session, names):
    if FEATURE_FILE.exists():
        saved = pd.read_csv(FEATURE_FILE, dtype={"sample_id": str}).set_index("sample_id")
        pd.testing.assert_series_equal(saved.filename, names, check_names=False)
        assert saved.sha256.str.fullmatch(r"[a-f0-9]{64}").all() and np.isfinite(saved[keys]).all().all()
        return saved
    rows = []
    total = 0
    cv2.setNumThreads(1)
    with patch.dict(gsd.PHONES, {PHYSICAL_CAMERA: (8192, 1000 / 39.4)}):
        for sample, name in names.items():
            assert name.lower().endswith(".jpg") and name.startswith(sample + "_")
            image = read_photo(session, name, min(60 * 1024**2, 384 * 1024**2 - total))
            total += len(image.getbuffer())
            digest = hashlib.sha256(image.getbuffer()).hexdigest()
            values = features.image_features(image, PHYSICAL_CAMERA)
            rows.append(dict(sample_id=sample, filename=name, sha256=digest, **{k: values[k] for k in keys}))
            print("Extracted", sample, "total MiB", round(total / 1024**2, 1), flush=True)
    data = pd.DataFrame(rows).set_index("sample_id")
    assert np.isfinite(data[keys]).all().all()
    temp = FEATURE_FILE.with_suffix(".tmp")
    data.to_csv(temp)
    temp.replace(FEATURE_FILE)
    return data


def local_single_view():
    """Estimate the expected local error for one random original photo per held-out soil."""
    reference = pd.read_csv(out / "v3_kadraj_sonuclari.csv").query("exam == '24' and model == 'v3'").set_index("sample_id")
    assert len(reference) == 24 and reference.index.is_unique
    rows = []
    for soil, photos in original.groupby("sample_id"):
        pool = labels.drop(index=soil)
        refs = training[training.sample_id != soil]
        targets = pd.DataFrame(gsd.curve_to_logq(pool.to_numpy()), index=pool.index)
        # In the published predictor, the grouping column determines which photos are averaged.
        # Give each original file its own group to predict it separately with the same fit.
        assert photos.file.is_unique and set(refs.sample_id) == set(pool.index)
        per_file = photos.assign(sample_id=photos.file)
        order, _, curves = published.fit_predict(refs, per_file, targets, keys)
        _, _, pooled = published.fit_predict(refs, photos, targets, keys)
        np.testing.assert_allclose(pooled[0], reference.loc[soil, gsd.COLS].astype(float), atol=1e-8, rtol=0)
        truth = labels.loc[soil].to_numpy()
        single_errors = gsd.emd(truth, curves)
        mean_error = gsd.emd(truth, pool.mean().to_numpy())[0]
        for name, curve, error in zip(order, curves, single_errors):
            rows.append(dict(sample_id=soil, file=name, single_emd=error, all_views_emd=reference.at[soil, "emd"],
                             mean_curve_emd=mean_error, **dict(zip(gsd.COLS, curve))))
    per_file = pd.DataFrame(rows)
    assert len(per_file) == len(original) and not per_file.duplicated(["sample_id", "file"]).any()
    per_file.to_csv(out / "montreal_tek_cekim_yerel_kontrol.csv", index=False, float_format="%.9f")
    summary = per_file.groupby("sample_id")[["single_emd", "all_views_emd", "mean_curve_emd"]].mean()
    print("24 equally weighted Kaggle soils:", summary.mean().to_dict(), flush=True)


def main():
    with requests.Session() as session:
        retries = Retry(total=2, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(ROOT + META_PATH, timeout=30)
        response.raise_for_status()
        meta = pd.read_csv(io.BytesIO(response.content), dtype={"sample_id": str})
        names, truth = select_images(meta)
        test = image_features(session, names)

    assert set(truth.index).isdisjoint(labels.index)
    query = test[keys].reset_index()
    q = pd.DataFrame(gsd.curve_to_logq(labels.to_numpy()), index=labels.index)
    order, _, v3 = published.fit_predict(training, query, q, keys)
    pd.testing.assert_index_equal(order, truth.index)
    ids = labels.index.sort_values()
    train, source = (soil_features(rows).loc[ids].to_numpy() for rows in (training, original))
    y = labels.loc[ids].to_numpy()
    p = select_penalties(train, source, y, "grup_pls", ("konum_yayilim",))["konum_yayilim"]
    x, xt = coordinates(train, test[keys].to_numpy(), y, "grup_pls")
    ordinal = fit_predict(x, y, xt, "konum_yayilim", p)
    rows = []
    truth.to_csv(out / "montreal_pilot_referans.csv", float_format="%.9f")
    for model, pred in [("v3", v3), ("sirali_pls", ordinal), ("egitim_ortalamasi", np.tile(y.mean(axis=0), (len(test), 1)))]:
        scores = gsd.emd(truth, pred)
        for sample, curve, score in zip(truth.index, pred, scores):
            rows.append(dict(sample_id=sample, model=model, emd=score, **dict(zip(gsd.COLS, curve))))
        print(model, scores.mean(), flush=True)
    pd.DataFrame(rows).to_csv(out / "montreal_pilot_sonuclari.csv", index=False, float_format="%.9f")
    local_single_view()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
