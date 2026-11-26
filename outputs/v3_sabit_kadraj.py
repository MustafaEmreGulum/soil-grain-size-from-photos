"""Tüm v3 özellikleriyle %60 kadrajı ve 100 mm'lik merkez kesitini karşılaştır."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits

from v3_knn_esit_karisim import keys, labels, original, out, published, training, work
import features
import gsd


PHONES = ("Motorola_Edge", "Samsung_A52")
FEATURE_CSV = out / "v3_100mm_fotograf_olculeri.csv"
META = ["file", "split", "sample_id", "phone", "aug"]
TARGETS = pd.DataFrame(gsd.curve_to_logq(labels.to_numpy()), index=labels.index)


@lru_cache(maxsize=12)
def fixed_crop(path, phone, dx_mm=0, dy_mm=0):
    ref_side, ppm = gsd.PHONES[phone]
    with Image.open(path) as f:
        image = ImageOps.exif_transpose(f).convert("RGB")
        w, h = image.size
        eff_ppm = ppm * max(w, h) / ref_side
        side = round(100 * eff_ppm)
        assert side <= min(w, h), path
        left = (w - side) // 2 + round(dx_mm * eff_ppm)
        top = (h - side) // 2 + round(dy_mm * eff_ppm)
        assert 0 <= left <= w - side and 0 <= top <= h - side, path
        square = image.crop((left, top, left + side, top + side))
        resampling = Image.Resampling.LANCZOS if features.CANON_PPM < eff_ppm else Image.Resampling.BICUBIC
        pixels = round(100 * features.CANON_PPM)
        return np.asarray(square.resize((pixels, pixels), resampling)), eff_ppm


def fixed_features(rebuild=False):
    if rebuild or not FEATURE_CSV.exists():
        photos = gsd.list_images(str(work / "soil-data")).query("split == 'train'")
        assert photos.file.tolist() == original.file.tolist()
        def measure_photo(row):
            records = []
            for aug in range(9):
                seed = aug * 1009 + row.Index if aug else None
                data = features.image_features(row.path, row.phone, seed)
                records.append(dict(file=row.file, split="train", sample_id=row.sample_id, phone=row.phone,
                                    aug=aug, **{key: data[key] for key in keys}))
            return records

        cv2.setNumThreads(1)
        records = []
        with patch.object(features, "load_canonical", fixed_crop), ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(measure_photo, row) for row in photos.itertuples()]
            for count, future in enumerate(as_completed(futures), 1):
                records.extend(future.result())
                if count % 10 == 0 or count == len(photos):
                    print(f"100 mm: {count}/{len(photos)} photos", flush=True)
        fixed_crop.cache_clear()
        table = pd.DataFrame(records).sort_values(["aug", "file"])
        temporary = FEATURE_CSV.with_suffix(".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(FEATURE_CSV)
    fixed = pd.read_csv(FEATURE_CSV)
    sort_keys = ["aug", "file"]
    pd.testing.assert_frame_equal(fixed[META].sort_values(sort_keys).reset_index(drop=True),
                                  training[META].sort_values(sort_keys).reset_index(drop=True))
    assert len(keys) == 30 and len(fixed) == 127 * 9 and np.isfinite(fixed[keys]).all().all()
    return fixed


def photo_median(data):
    # Keep all eight training variants attached to the original photo.
    photo = data.groupby(["sample_id", "file"])[keys].mean()
    return photo.groupby("sample_id")[keys].median().reset_index()


def evaluate(data, method, median=False):
    shared = labels.index
    for phone in PHONES:
        shared = shared[shared.isin(original.loc[original.phone == phone, "sample_id"])]
    assert len(shared) == 21
    rows = []
    for ids, source, target in [(labels.index, None, None), (shared, *PHONES), (shared, *PHONES[::-1])]:
        exam = "24" if source is None else f"{source} -> {target}"
        train = data if source is None else data[data.phone == source]
        test = data[data.aug == 0] if target is None else data[(data.aug == 0) & (data.phone == target)]
        train = train[train.sample_id.isin(ids)]
        if median:
            train, test = photo_median(train), photo_median(test)
        for soil in ids:
            refs = train[train.sample_id != soil]
            assert set(refs.sample_id) == set(ids.drop(soil))
            order, _, curves = published.fit_predict(refs, test[test.sample_id == soil], TARGETS.loc[ids.drop(soil)], keys)
            assert list(order) == [soil]
            curve = curves[0]
            assert np.isfinite(curve).all() and 0 <= curve[0] and np.all(np.diff(curve) >= 0) and curve[-1] == 100
            rows.append(dict(exam=exam, sample_id=soil, model=method, emd=gsd.emd(labels.loc[soil], curve)[0],
                             **dict(zip(gsd.COLS, curve))))
    return pd.DataFrame(rows)


def gain_summary(base, candidate):
    by = ["exam", "sample_id"]
    gain = base.set_index(by).emd - candidate.set_index(by).emd
    assert len(gain) == 66 and gain.notna().all()
    report = []
    for exam, g in gain.groupby(level="exam"):
        leave_one = (g.sum() - g) / (len(g) - 1)
        report.append(dict(exam=exam, gain=g.mean(), median=g.median(), wins=(g > 0).sum(),
                           loo_min=leave_one.min(), loo_max=leave_one.max()))
    return pd.DataFrame(report).set_index("exam")


def passes(report):
    return report.loc["24", "loo_min"] > 0 and (report.drop("24").gain >= 0).all()


def sensitivity(data):
    originals = data[data.aug == 0]
    rows = []
    for soil in labels.index:
        photos = originals[originals.sample_id == soil]
        # Fit the same fold once. Each query group is the soil with one photo omitted.
        queries = [photos.assign(sample_id="all")]
        for file in photos.file:
            queries.append(photos[photos.file != file].assign(sample_id=file))
        refs = data[data.sample_id != soil]
        order, _, curves = published.fit_predict(refs, pd.concat(queries), TARGETS.drop(soil), keys)
        curves = pd.DataFrame(curves, index=order, columns=gsd.COLS)
        full = curves.loc["all"]
        for file, curve in curves.drop(index="all").iterrows():
            rows.append(dict(sample_id=soil, omitted_file=file, photo_count=len(photos),
                             curve_change_emd=gsd.emd(full, curve)[0], full_emd=gsd.emd(labels.loc[soil], full)[0],
                             omitted_emd=gsd.emd(labels.loc[soil], curve)[0]))
    return pd.DataFrame(rows)


def main(rebuild=False):
    base = evaluate(training, "v3")
    idx = ["exam", "sample_id"]
    saved = pd.read_csv(out / "dalgacik_sonuclari.csv").query("model == 'v3'").set_index(idx).sort_index()
    current = base.set_index(idx).sort_index()
    pd.testing.assert_index_equal(current.index, saved.index)
    np.testing.assert_allclose(current.emd, saved.emd, atol=1e-7, rtol=0)
    np.testing.assert_allclose(current[gsd.COLS], saved[["pred_" + c for c in gsd.COLS]], atol=1e-7, rtol=0)
    print("Confirmed 66 baseline curves and scores", flush=True)
    fixed = fixed_features(rebuild)
    candidate = evaluate(fixed, "100mm")
    result = [base, candidate]
    report = gain_summary(base, candidate)
    print(report.round(5).to_string(), flush=True)
    chosen, comparison = (fixed, candidate) if passes(report) else (training, base)
    influence = sensitivity(chosen)
    assert len(influence) == len(original) and influence.omitted_file.is_unique
    full = comparison[comparison.exam == "24"].set_index("sample_id").emd
    np.testing.assert_allclose(influence.full_emd, influence.sample_id.map(full), rtol=0, atol=1e-7)
    influence.insert(0, "model", comparison.model.iloc[0])
    influence.to_csv(out / "v3_cekim_duyarliligi.csv", index=False, float_format="%.9f")
    large = influence.groupby("sample_id").curve_change_emd.max().ge(10)
    print("Soils with at least one change of 10 EMD:", int(large.sum()), flush=True)
    if large.sum() >= 3:
        robust = evaluate(chosen, comparison.model.iloc[0] + "_ortanca", median=True)
        result.append(robust)
        print(gain_summary(comparison, robust).round(5).to_string(), flush=True)
    scores = pd.concat(result, ignore_index=True)
    scores.to_csv(out / "v3_kadraj_sonuclari.csv", index=False, float_format="%.9f")
    print(scores.pivot_table(index="exam", columns="model", values="emd").round(4).to_string())
    columns = ["0.063", "0.2", "0.63", "20"]
    absolute = scores[columns].sub(labels.loc[scores.sample_id, columns].to_numpy()).abs()
    print(absolute.groupby([scores.exam, scores.model]).mean().round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-features", action="store_true", help="Fotoğraf ölçülerini özgün JPEG'lerden tekrar üret.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.rebuild_features)
