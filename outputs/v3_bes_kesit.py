"""Ortak fiziksel ölçekte beş kesitin ortalama v3 özelliklerini sınar."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from v3_knn_esit_karisim import keys, original, out, work
from v3_sabit_kadraj import META, evaluate, fixed_crop, fixed_features, gain_summary, passes
import features
import gsd


OFFSETS = [(-20, -20), (20, -20), (-20, 20), (20, 20)]
CACHE = out / "v3_bes_kesit_fotograf_olculeri.csv"


def loaded_crop(pixels_and_scale, phone):
    # Reuse decoded pixels for the original and its eight training augmentations.
    return pixels_and_scale


def multi_features(center, photos, n_aug):
    """Add four peripheral feature vectors to the exact center used by the single-crop model."""
    index = ["file", "aug"]
    centers = center.set_index(index).sort_index()
    assert centers.index.is_unique

    def measure(row):
        original_vectors = centers.loc[row.file, keys].sort_index()
        assert list(original_vectors.index) == list(range(n_aug + 1))
        total = original_vectors.to_numpy(copy=True)
        for dx, dy in OFFSETS:
            view = fixed_crop(row.path, row.phone, dx, dy)
            for aug in range(n_aug + 1):
                seed = aug * 1009 + row.Index if aug else None
                values = features.image_features(view, row.phone, seed)
                total[aug] += [values[k] for k in keys]
        frame = pd.DataFrame(total / (1 + len(OFFSETS)), columns=keys)
        return frame.assign(file=row.file, aug=original_vectors.index.to_numpy())

    blocks = []
    cv2.setNumThreads(1)
    with patch.object(features, "load_canonical", loaded_crop), ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(measure, row) for row in photos.itertuples()]
        for count, future in enumerate(as_completed(futures), 1):
            blocks.append(future.result())
            if count % 10 == 0 or count == len(photos):
                print(f"Five crops: {count}/{len(photos)} photos", flush=True)
    fixed_crop.cache_clear()
    values = pd.concat(blocks).set_index(index).sort_index()
    pd.testing.assert_index_equal(values.index, centers.index)
    assert np.isfinite(values.to_numpy()).all()
    return center[META].merge(values, on=index, validate="one_to_one")


def main():
    center = fixed_features()
    if not CACHE.exists():
        photos = gsd.list_images(str(work / "soil-data")).query("split == 'train'")
        assert photos.file.tolist() == original.file.tolist()
        candidate = multi_features(center, photos, 8)
        temp = CACHE.with_suffix(".tmp")
        candidate.to_csv(temp, index=False)
        temp.replace(CACHE)
    candidate = pd.read_csv(CACHE)
    pd.testing.assert_frame_equal(candidate[META], center[META])
    assert len(candidate) == 1143 and np.isfinite(candidate[keys].to_numpy()).all()
    scores = evaluate(candidate, "bes_100mm")
    scores.to_csv(out / "v3_bes_kesit_sonuclari.csv", index=False, float_format="%.9f")
    archive = pd.read_csv(out / "v3_kadraj_sonuclari.csv")
    print("EMD", scores.groupby("exam").emd.mean().to_dict())
    reports = {}
    for method in ["100mm", "v3"]:
        report = reports[method] = gain_summary(archive[archive.model == method], scores)
        print(method, report.round(5).to_string(), sep="\n", flush=True)
    print("Kaggle candidate against pure v3:", passes(reports["v3"]), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
