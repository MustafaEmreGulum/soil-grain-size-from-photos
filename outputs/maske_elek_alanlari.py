"""Kaggle'dan alınan 4,5 piksel/mm maskelerde kısa eksene göre yüzey alanı."""
from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

OUT = Path(__file__).resolve().parent
SIEVES = ["0.63", "2", "6.3", "20", "63"]


def extract(archive):
    cache = OUT.parent / "work/vitalclick/cache"
    photos = pd.concat([pd.read_csv(cache / f"granulometry_{split}.csv")
                        for split in ("train", "test")]).set_index("file")
    assert photos.index.is_unique and len(photos) == 162
    records = []
    with ZipFile(archive) as z:
        assert set(z.namelist()) == {name + ".npz" for name in photos.index}
        for name, meta in photos.iterrows():
            with np.load(BytesIO(z.read(name + ".npz"))) as data:
                mask = data["mask"]
            grains = regionprops_table(mask, properties=("area", "axis_minor_length"))
            row = dict(file=name, split=meta.split, sample_id=meta.sample_id, phone=meta.phone)
            for sieve in SIEVES:
                selected = grains["axis_minor_length"] > 4.5 * float(sieve)
                row["n_" + sieve] = int(selected.sum())
                row["a_" + sieve] = 100 * grains["area"][selected].sum() / mask.size
            records.append(row)
    result = pd.DataFrame(records).set_index("file")
    archived = pd.concat([pd.read_csv(OUT / f) for f in
                          ("v3_tam_yuzey_20mm_fotograflar.csv", "test20_fotograflar.csv")]).set_index("file").loc[result.index]
    np.testing.assert_allclose(result["a_20"], archived.alan_pct, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(result["n_20"], archived.aday_sayisi)
    result.to_csv(OUT / "maske_elek_alanlari.csv", float_format="%.12f")


if __name__ == "__main__":
    extract(Path(sys.argv[1]))
