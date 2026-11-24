"""Aynı maskelerde alan ve alan × kısa çap ile kütle tahminini karşılaştır."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coklu_elek_deneyi import AREA, OUT, area_updates, bounds
from ince_paya_2mm_aktarimi import transfer
from maske_elek_alanlari import measure
from orta_elek_kismi_duzeltme import band_curve
from saha_ve_egri_deneyi import SITE, folds
from v3_knn_esit_karisim import COLS, labels, original, predict, training
from gsd import emd


def soil_measures(photos, prefix):
    columns = [prefix + s for s in AREA]
    # Ortak kalibrasyon kodu her iki ölçüyü de aynı sütun adlarıyla alır.
    means = photos.groupby("sample_id")[columns].mean()
    return means.set_axis(["a_" + s for s in AREA], axis=1)


def adjusted_curve(raw, train, obs):
    updates = area_updates(train, obs, labels.loc[train.index])
    base = bounds(raw, {s: val for s, val in updates.items() if s == "20"})
    curve = transfer(band_curve(base, train, obs), raw)
    assert curve.min() >= 0 and curve.max() <= 100 and curve[-1] == 100
    return curve


def evaluate(photos):
    archived = pd.read_csv(OUT / "ince_paya_2mm_sonuclar.csv").set_index(["exam", "sample_id"])
    records = []
    train_photos = photos.query("split == 'train'")
    for exam, hold, ids, src, dst in folds():
        refs, query = training[training.sample_id.isin(ids)], original[original.sample_id.isin(hold)]
        source, target = train_photos, train_photos
        if src:
            refs, query = refs[refs.phone == src], query[query.phone == dst]
            source, target = source[source.phone == src], target[target.phone == dst]
        assert set(ids).isdisjoint(hold)
        raw_curves = predict(refs, query)["v3"]
        for prefix, model in [("a_", "alan"), ("v_", "alan_kisa_cap")]:
            train = soil_measures(source, prefix).loc[ids]
            test = soil_measures(target, prefix).loc[hold]
            for sample in hold:
                raw = raw_curves.loc[sample, COLS].to_numpy()
                curve = adjusted_curve(raw, train, test.loc[sample])
                error = emd(labels.loc[sample, COLS], curve)[0]
                prev = archived.loc[(exam, sample)]
                if model == "alan":
                    np.testing.assert_allclose(curve, prev[COLS].to_numpy(float), atol=1e-7, rtol=0)
                    np.testing.assert_allclose(error, prev.emd, atol=1e-7, rtol=0)
                records.append(dict(exam=exam, sample_id=sample, site=SITE[sample], model=model,
                                    emd=error, kazanc=prev.emd-error, **dict(zip(COLS, curve))))
    result = pd.DataFrame(records)
    assert len(result) == 2 * len(archived) == 190
    assert not result.duplicated(["exam", "sample_id", "model"]).any()
    result.to_csv(OUT / "maske_kisa_cap_sonuclar.csv", index=False, float_format="%.9f")
    print(result.groupby(["exam", "model"]).emd.mean().unstack().round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Ölçümleri yeniden çıkarmak için maske ZIP dosyası")
    args = parser.parse_args()
    path = OUT / "maske_kisa_cap_olculeri.csv"
    if args.archive:
        measure(args.archive, include_volume=True).to_csv(path, float_format="%.12f")
    photos = pd.read_csv(path, index_col="file")
    baseline = pd.read_csv(OUT / "maske_elek_alanlari.csv", index_col="file")
    pd.testing.assert_frame_equal(photos[baseline.columns], baseline, atol=1e-10, rtol=0)
    evaluate(photos)
