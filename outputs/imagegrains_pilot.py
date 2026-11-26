"""ImageGrains 2.0 pilot: paired, calibrated views; outlined references in millimetres."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

from v3_knn_esit_karisim import labels, out, work
from v3_sabit_kadraj import PHONES
import gsd


PPM = 4.5
SIDE_MM = 52
SIDE = int(SIDE_MM * PPM)
CORE = (np.arange(SIDE) + 0.5) / PPM
CORE = ((CORE >= 13) & (CORE < 39))
CORE = CORE[:, None] & CORE[None, :]
PILOT = out / "imagegrains_pilot_secimi.csv"
REF = out / "imagegrains_pilot_referans.json"
MASKS = out / "imagegrains_pilot_maskeleri.npz"
CHECKSUM = "5a5b74bc70a63b6605afd50465240969"


def select():
    rng = np.random.default_rng(20261121)
    photos = gsd.list_images(str(work / "soil-data")).query("split == 'train'")
    paired = sorted(set.intersection(*(set(photos.loc[photos.phone == p, "sample_id"]) for p in PHONES)))
    assert len(paired) == 21
    coarse = labels.loc[paired, "0.63"] < 50
    ids = sorted(s for group in (True, False) for s in rng.choice(coarse.index[coarse == group], 3, replace=False))
    picked = photos[photos.sample_id.isin(ids) & photos.phone.isin(PHONES)].sort_values("file")
    picked = picked.drop_duplicates(["sample_id", "phone"]).sort_values(["sample_id", "phone"])
    picked = picked.iloc[rng.permutation(len(picked))].copy()
    picked.index = pd.Index([f"P{i:02d}" for i in range(1, len(picked) + 1)], name="panel")
    assert len(picked) == 12 and (picked.groupby("sample_id").phone.nunique() == 2).all()
    return picked


def crop(row, offset=(0, 0)):
    with Image.open(row.path) as handle:
        rgb = ImageOps.exif_transpose(handle).convert("RGB")
    ref_side, ref_ppm = gsd.PHONES[row.phone]
    ppm = ref_ppm * max(rgb.size) / ref_side
    side = round(SIDE_MM * ppm)
    w, h = rgb.size
    assert min(w, h) >= side
    x = (w - side) // 2 + round(offset[0] * ppm)
    y = (h - side) // 2 + round(offset[1] * ppm)
    assert 0 <= x <= w - side and 0 <= y <= h - side
    return np.asarray(rgb.crop((x, y, x+side, y+side)).resize((SIDE, SIDE), Image.Resampling.LANCZOS))


def coarse_labels(mask):
    keep = [p.label for p in regionprops(mask) if p.axis_minor_length >= 6.3 * PPM]
    return np.where(np.isin(mask, keep), mask, 0)


def blind_panels(picked, cache_dir, offsets=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(picked), 4):
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        for ax, (panel, row) in zip(axes.flat, picked.iloc[i:i+4].iterrows()):
            offset = (0, 0) if offsets is None else offsets[panel]
            ax.imshow(crop(row, offset), extent=(0, SIDE_MM, SIDE_MM, 0), interpolation="nearest")
            ax.add_patch(Rectangle((13, 13), 26, 26, fill=False, edgecolor="#04c5df", linewidth=0.75))
            ax.set(xticks=np.arange(0, 53, 4), yticks=np.arange(0, 53, 4), title=panel)
            ax.tick_params(labelsize=7)
            ax.grid(linewidth=0.25, alpha=0.32)
        fig.tight_layout()
        fig.savefig(cache_dir / f"blind_{i//4+1}.png", dpi=125)
        plt.close(fig)


def load_model(weights):
    import hashlib
    import os
    os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(weights.parent.resolve()))
    from cellpose import models
    import torch
    with weights.open("rb") as f:
        assert hashlib.file_digest(f, "md5").hexdigest() == CHECKSUM
    torch.set_num_threads(4)
    model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=str(weights.resolve()),
                                 use_bfloat16=False)
    assert Path(model.pretrained_model).resolve() == weights.resolve()
    return model


def segment(model, rgb):
    pred, *_ = model.eval(rgb, batch_size=1, bsize=256, flow_threshold=0.4, cellprob_threshold=0.0)
    assert pred.shape == (SIDE, SIDE) and pred.min() >= 0 and pred.max() <= 65535
    return pred.astype(np.uint16)


def infer(picked, weights):
    model = load_model(weights)
    masks = {}
    for panel, row in picked.iterrows():
        pred = segment(model, crop(row))
        masks[panel] = pred
        print("Segmented", panel, "instances", pred.max(), flush=True)
    np.savez_compressed(MASKS, **masks)


def ref_mask(polygons):
    mask = np.zeros((SIDE, SIDE), dtype=np.uint16)
    for number, boundary in enumerate(polygons, start=1):
        coords = np.rint(np.asarray(boundary) * PPM - 0.5).astype(np.int32)
        cv2.fillPoly(mask, [coords], number)
    return mask


def evaluated(mask):
    large = coarse_labels(mask)
    ids, counts = np.unique(large[CORE], return_counts=True)
    return np.where(np.isin(large, ids[(ids != 0) & (counts >= PPM**2)]), large, 0)


def matches(reference, prediction):
    rids = np.unique(reference[reference != 0])
    pids = np.unique(prediction[prediction != 0])
    ious = np.zeros((len(rids), len(pids)))
    for i, rid in enumerate(rids):
        a = reference == rid
        for j, pid in enumerate(pids):
            b = prediction == pid
            ious[i, j] = np.count_nonzero(a & b) / np.count_nonzero(a | b)
    # Maximise the number of valid pairs first, then their IoU when counts tie.
    quality = (ious >= 0.5) * (1 + min(ious.shape)) + ious
    rows, cols = linear_sum_assignment(quality, maximize=True)
    return len(rids), len(pids), int((ious[rows, cols] >= 0.5).sum())


def audit_masks(picked, annotations, get_mask, figure_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.segmentation import find_boundaries
    rows = []
    assert set(annotations) == set(picked.index)
    height = (len(picked) + 3) // 4
    fig, axes = plt.subplots(height, 4, figsize=(12, 3 * height + .8))
    for axis, (panel, row) in zip(axes.flat, picked.iterrows()):
        ref = evaluated(ref_mask(annotations[panel]))
        estimated = evaluated(get_mask(panel, row))
        nref, nmodel, tp = matches(ref, estimated)
        actual = (ref[CORE] > 0).mean() * 100
        predicted = (estimated[CORE] > 0).mean() * 100
        rows.append(dict(panel=panel, sample_id=row.sample_id, phone=row.phone, reference=nref,
                         predicted=nmodel, matched=tp, reference_area_pct=actual,
                         model_area_pct=predicted, abs_difference_pp=abs(predicted-actual)))
        rgb = crop(row, row.get("offset", (0, 0))).copy()
        rgb[find_boundaries(ref, mode="inner")] = [255, 214, 0]
        rgb[find_boundaries(estimated, mode="inner")] = [222, 40, 200]
        axis.imshow(rgb, extent=(0, SIDE_MM, SIDE_MM, 0))
        axis.add_patch(Rectangle((13, 13), 26, 26, fill=False, edgecolor="#00dae0", linewidth=0.9))
        axis.set_title(f"{panel} · {row.sample_id} · {row.phone.split('_')[0]}", fontsize=10)
        axis.set_axis_off()
    fig.suptitle("Kısa ekseni en az 6,3 mm olan taşlar", fontsize=15, y=0.993)
    fig.text(0.5, 0.953, "Sarı çizgi gözle çizilen sınırı, mor çizgi ImageGrains sınırını, turkuaz kare ölçüm alanını gösterir.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.89, bottom=0.02, wspace=0.08, hspace=0.19)
    fig.savefig(figure_path, dpi=125)
    plt.close(fig)
    return pd.DataFrame(rows)


def audit_by_phone(scores):
    stats = scores.groupby("phone").agg(reference=("reference", "sum"), predicted=("predicted", "sum"),
                                        matched=("matched", "sum"), area_mae=("abs_difference_pp", "mean"))
    stats["precision"] = stats.matched / stats.predicted
    stats["recall"] = stats.matched / stats.reference
    stats["passed"] = (stats.reference >= 3) & (stats.precision >= .7) & (stats.recall >= .7) & (stats.area_mae <= 15)
    return stats


def score(picked):
    annotations = json.loads(REF.read_text(encoding="utf-8"))
    with np.load(MASKS) as masks:
        assert set(masks.files) == set(picked.index)
        scores = audit_masks(picked, annotations, lambda panel, row: masks[panel],
                             out / "imagegrains_pilot_karsilastirma.png")
    scores.to_csv(out / "imagegrains_pilot_sonuclari.csv", index=False, float_format="%.6f")
    print(audit_by_phone(scores).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["blind", "infer", "score"])
    parser.add_argument("--weights", type=Path, default=work.parent / ".venv/IG2_full_set_cp_SAM")
    args = parser.parse_args()
    picked = select()
    if args.step == "blind":
        picked[["sample_id", "phone", "file"]].to_csv(PILOT)
        blind_panels(picked, work.parent / ".venv/ig_pilot")
    elif args.step == "infer":
        infer(picked, args.weights)
    else:
        score(picked)


if __name__ == "__main__":
    main()
