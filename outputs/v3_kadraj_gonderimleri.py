"""Sabit kadraj ve özgün kadraj + ortanca modellerinin Kaggle test tahminleri."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from v3_knn_esit_karisim import cache, keys, labels, out, published, training, work
from v3_sabit_kadraj import TARGETS, fixed_crop, fixed_features, photo_median
import features
import gsd


def fixed_test_features(photos):
    def measure(row):
        values = features.image_features(row.path, row.phone)
        return dict(file=row.file, sample_id=row.sample_id, **{k: values[k] for k in keys})

    cv2.setNumThreads(1)
    with patch.object(features, "load_canonical", fixed_crop), ThreadPoolExecutor(max_workers=4) as executor:
        test = pd.DataFrame(executor.map(measure, photos.itertuples()))
    fixed_crop.cache_clear()
    assert np.isfinite(test[keys]).all().all()
    return test


def write_predictions(train, test, template, filename):
    assert set(train.sample_id) == set(labels.index) and set(test.sample_id) == set(template.index)
    assert set(train.sample_id).isdisjoint(test.sample_id)
    ids, _, curves = published.fit_predict(train, test, TARGETS, keys)
    result = pd.DataFrame(curves, index=ids, columns=gsd.COLS).loc[template.index]
    assert result.shape == template.shape == (10, 11) and result.columns.equals(template.columns)
    assert np.isfinite(result).all().all() and result.ge(0).all().all() and result.le(100).all().all()
    assert (np.diff(result, axis=1) >= 0).all() and result["200"].eq(100).all()
    path = out / filename
    tmp = path.with_suffix(".tmp")
    result.to_csv(tmp, index_label="sample_id", float_format="%.8f")
    pd.testing.assert_frame_equal(pd.read_csv(tmp, index_col="sample_id"), result, atol=1e-8, rtol=0)
    tmp.replace(path)
    print(filename, "10 x 11 OK", flush=True)


def main():
    template = pd.read_csv(work / "soil-data/sample_submission.csv", index_col="sample_id")
    assert template.index.is_unique and template.index.name == "sample_id"
    photos = gsd.list_images(str(work / "soil-data")).query("split == 'test'").reset_index(drop=True)
    standard = cache.query("split == 'test' and aug == 0").reset_index(drop=True)
    pd.testing.assert_frame_equal(photos[["file", "phone", "sample_id"]], standard[["file", "phone", "sample_id"]])
    assert len(photos) == 35 and photos.file.is_unique
    write_predictions(photo_median(training), photo_median(standard), template, "submission_v3_ortanca.csv")
    write_predictions(fixed_features(), fixed_test_features(photos), template, "submission_v3_100mm.csv")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
