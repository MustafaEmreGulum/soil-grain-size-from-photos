"""2 mm sabitken ara elek oranlarını ve taneler arası düz yüzey alanını sına."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd

from coklu_elek_deneyi import OUT, TEXTURE, bounds
from h615_otomatik_tas import surface
from ince_0063_oran import fit_share
from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels, repo, work
from gsd import emd, list_images

STEM = "ara_elek_zinciri"
STEPS = ["0.63", "0.2", "0.063"]
SMOOTH = "duz_alan_pct"


def extract(archive):
    photos = list_images(work / "soil-data").query("split == 'train'").set_index("file")
    boxes = pd.read_csv(OUT / "dinov2_hiyerarsik_kadrajlar.csv", index_col="file")
    rows = []
    window = np.ones((5, 5), np.uint8)
    with ZipFile(archive) as z:
        for name, row in photos.iterrows():
            rgb, _, _ = surface(row, boxes.loc[name])
            with np.load(BytesIO(z.read(name + ".npz"))) as data:
                mask = data["mask"]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            assert gray.shape == mask.shape
            valid = cv2.erode((mask == 0).astype(np.uint8), window,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            mean = cv2.blur(gray, (5, 5))
            variance = np.maximum(0, cv2.blur(gray*gray, (5, 5)) - mean*mean)
            lo, hi = np.percentile(gray, [10, 90])
            assert hi > lo
            smooth = valid & (variance <= (.10 * (hi-lo))**2)
            rows.append(dict(file=name, sample_id=row.sample_id, phone=row.phone,
                             uygun_alan_pct=100*valid.mean(), duz_alan_pct=100*smooth.mean()))
    result = pd.DataFrame(rows)
    assert len(result) == 127 and result.file.is_unique
    return result


def experiment(photos):
    reference = pd.read_csv(OUT / "ince_paya_2mm_sonuclar.csv").set_index(["exam","sample_id"])
    records = []
    for exam, hold, ids, src, dst in folds():
        assert set(ids).isdisjoint(hold)
        source = photos if src is None else photos[photos.phone == src]
        target = photos if dst is None else photos[photos.phone == dst]
        curves = reference.loc[[(exam,s) for s in hold], COLS].to_numpy()
        models = {"mevcut":curves}
        for method, cols in [("doku",TEXTURE), ("duz_doku",[*TEXTURE,SMOOTH])]:
            train = source.groupby("sample_id")[cols].median().loc[ids]
            query = target.groupby("sample_id")[cols].median().loc[hold]
            total = curves[:, COLS.index("2")]
            shares = {}
            for upper, lower in [("2","0.63"),("0.63","0.2"),("0.2","0.063"),("2","0.2"),("2","0.063")]:
                shares[upper, lower] = fit_share(train, query, labels.loc[ids,lower],
                                                 labels.loc[ids,upper], cols)["doku"]
            chain, direct = {}, {}
            previous = "2"
            amount = total
            for sieve in STEPS:
                amount = amount * shares[previous,sieve]
                chain[sieve] = amount
                direct[sieve] = total * shares["2",sieve]
                previous = sieve
            for mode, updates in [("tek", {"0.063":direct["0.063"]}), ("dogrudan_uc",direct),
                                  ("sadece_ara",{s:chain[s] for s in STEPS[:2]}), ("zincir",chain)]:
                models[f"{method}_{mode}"] = np.array([
                    bounds(base, {s:v[i] for s,v in updates.items()}) for i,base in enumerate(curves)])
        fixed = [i for i,s in enumerate(COLS) if s not in STEPS]
        for model, predicted in models.items():
            np.testing.assert_array_equal(curves[:,fixed],predicted[:,fixed])
            assert np.isfinite(predicted).all() and np.all(np.diff(predicted,axis=1)>=-1e-8)
            truth = labels.loc[hold,COLS]
            scores = emd(truth, predicted)
            if model == "mevcut":
                np.testing.assert_allclose(scores,reference.loc[[(exam,s) for s in hold],"emd"],atol=1e-7,rtol=0)
            for i,s in enumerate(hold):
                records.append(dict(exam=exam,sample_id=s,model=model,emd=scores[i],**dict(zip(COLS,predicted[i]))))
    results = pd.DataFrame(records)
    assert len(results) == 95*9 and not results.duplicated(["exam","sample_id","model"]).any()
    old = pd.read_csv(OUT / "ince_oran_2mm_ridge_sonuclar.csv").query("model == 'doku'").set_index(["exam","sample_id"])
    single = results.query("model == 'doku_tek'").set_index(["exam","sample_id"])
    np.testing.assert_allclose(single.emd, old.loc[single.index,"emd"], atol=1e-7, rtol=0)
    results.to_csv(OUT / f"{STEM}_sonuclar.csv", index=False, float_format="%.9f")
    errors = (results[COLS].to_numpy() - labels.loc[results.sample_id,COLS].to_numpy())
    summary = results[["exam","model","emd"]].join(pd.DataFrame(np.abs(errors),columns=COLS))
    summary = summary.groupby(["exam","model"]).mean()
    summary.to_csv(OUT / f"{STEM}_ozet.csv",float_format="%.9f")
    print(summary.xs("toprak").round(3).to_string())
    print(summary.emd.unstack(0).round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    path = OUT / f"{STEM}_alan.csv"
    if args.archive:
        extract(args.archive).to_csv(path, index=False, float_format="%.9f")
    source = pd.read_csv(repo / "cache/granulometry_train.csv")
    photos = source.merge(pd.read_csv(path), on=["file","phone","sample_id"], validate="one_to_one")
    assert len(photos) == len(source) == 127
    experiment(photos)
