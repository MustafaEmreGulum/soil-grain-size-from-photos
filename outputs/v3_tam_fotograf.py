"""Özgün çekimlerde v3 merkez kadrajını tam fotoğrafla aynı PLS sınavında karşılaştır."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import v3_sabit_kadraj as v3


def run():
    photos = v3.gsd.list_images(str(v3.work / "soil-data")).query("split == 'train'").set_index("file")
    center = v3.original.set_index("file").loc[photos.index].reset_index()
    assert len(photos) == 127 and photos.index.is_unique and photos.sample_id.nunique() == 24
    assert center.aug.eq(0).all() and center.file.tolist() == photos.index.tolist()
    assert np.array_equal(center[["sample_id", "phone"]].values, photos[["sample_id", "phone"]].values)
    cv2.setNumThreads(1)
    # Check that the implementation on this machine reproduces the archived centre features.
    for _, r in photos.groupby("phone").head(1).iterrows():
        values = v3.features.image_features(r.path, r.phone)
        np.testing.assert_allclose([values[k] for k in v3.keys],
                                   center.loc[center.file == r.name, v3.keys].iloc[0].to_numpy(float),
                                   rtol=1e-4, atol=1e-5)

    def measure(row):
        values = v3.features.image_features(row.path, row.phone)
        return [values[k] for k in v3.keys]

    with patch.object(v3.features, "CROP", 1.0), ThreadPoolExecutor(max_workers=4) as pool:
        vectors = []
        for count, vector in enumerate(pool.map(measure, photos.itertuples()), 1):
            vectors.append(vector)
            if count % 20 == 0 or count == len(photos):
                print(f"{count}/{len(photos)}", flush=True)
    full = center.copy()
    full[v3.keys] = vectors
    assert np.isfinite(full[v3.keys]).all().all()
    with threadpool_limits(limits=1):
        baseline = v3.evaluate(center, "merkez")
        candidate = v3.evaluate(full, "tam")
    results = pd.concat([baseline, candidate], ignore_index=True)
    results.to_csv(v3.out / "v3_tam_fotograf_sonuclari.csv", index=False, float_format="%.9f")
    print(results.groupby(["exam", "model"]).emd.mean().round(4).to_string())
    print(v3.gain_summary(baseline, candidate).round(4).to_string())


if __name__ == "__main__":
    run()
