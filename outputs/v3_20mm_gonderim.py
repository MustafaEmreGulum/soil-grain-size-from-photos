"""Kaggle tam yüzey ölçümlerinden saf v3'ün 20 mm gönderim adayını üret."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WORK = OUT.parent / "work"
DATA = WORK / "soil-data"
sys.path.insert(0, str(WORK / "vitalclick" / "src"))
import gsd


def main():
    baseline_path = WORK / "v3_puan_kontrol.csv"
    content = baseline_path.read_bytes().replace(b"\r\n", b"\n")
    git_header = f"blob {len(content)}\0".encode()
    assert hashlib.sha1(git_header + content).hexdigest() == "fa5b1134728960728c87a3114d5f49392578b194"
    baseline = pd.read_csv(baseline_path, index_col="sample_id")
    template = pd.read_csv(DATA / "sample_submission.csv", index_col="sample_id")
    assert baseline.index.equals(template.index) and baseline.columns.equals(template.columns)
    assert list(baseline.columns) == gsd.COLS

    labels = pd.read_csv(DATA / "Training_labels_updated.csv", index_col="sample_id")
    train = pd.read_csv(OUT / "v3_tam_yuzey_20mm_fotograflar.csv")
    assert train.file.nunique() == len(train) == 127 and train.alan_pct.between(0, 100).all()
    train_area = train.groupby("sample_id").alan_pct.mean()
    assert set(train_area.index) == set(labels.index) and len(train_area) == 24
    coarse = 100 - labels.loc[train_area.index, "20"]
    assert train_area @ train_area > 0
    factor = max(0.0, (train_area @ coarse) / (train_area @ train_area))

    photos = gsd.list_images(DATA).query("split == 'test'").set_index("file")
    test = pd.read_csv(OUT / "test20_fotograflar.csv").set_index("file")
    assert test.index.is_unique and len(test) == 35 and set(test.index) == set(photos.index)
    assert test.phone.equals(photos.loc[test.index, "phone"])
    assert test.alan_pct.between(0, 100).all() and test.aday_sayisi.ge(0).all()
    assert (test.alan_pct.gt(0) == test.aday_sayisi.gt(0)).all()
    grouped = test.join(photos.sample_id).groupby("sample_id")
    comparison = grouped.agg(cekim=("alan_pct", "size"), ortalama_alan_pct=("alan_pct", "mean"),
                             pozitif_cekim=("alan_pct", lambda s: s.gt(0).sum()))
    assert len(comparison) == 10 and set(comparison.index) == set(template.index)
    comparison = comparison.loc[template.index]
    area = comparison.ortalama_alan_pct
    corrected = baseline.copy()
    estimated = (100 - factor * area).clip(lower=baseline["6.3"], upper=baseline["63"])
    corrected.loc[area.gt(0), "20"] = estimated.loc[area.gt(0)].round(4)
    comparison["alan_kutle_carpani"] = factor
    comparison["saf_v3_20"] = baseline["20"]
    comparison["duzeltilmis_20"] = corrected["20"]
    comparison["fark_yuzde_puan"] = (corrected["20"] - baseline["20"]).round(4)

    values = corrected.to_numpy()
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 100
    assert (np.diff(values, axis=1) >= 0).all() and corrected["200"].eq(100).all()
    pd.testing.assert_frame_equal(corrected.drop(columns="20"), baseline.drop(columns="20"))
    pd.testing.assert_frame_equal(corrected.loc[area.eq(0)], baseline.loc[area.eq(0)])
    sub_file = OUT / "submission_v3_tam_yuzey_20mm.csv"
    corrected.to_csv(sub_file)
    pd.testing.assert_frame_equal(pd.read_csv(sub_file, index_col="sample_id"), corrected)
    comparison.to_csv(OUT / "v3_20mm_test_karsilastirma.csv", float_format="%.9f")
    print(comparison.round(4).to_string())


if __name__ == "__main__":
    main()
