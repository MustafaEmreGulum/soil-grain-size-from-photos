"""H037 ilk Motorola çekiminde tam toprak kadrajı ile eski beş karenin kapsamı."""
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.measure import regionprops

import iri_tas_kalan_doku as prev

ig = prev.ig
STEM = "h037_tam_yuzey"
ARCHIVE = ig.out / f"{STEM}_maske.npz"


def surface():
    row = prev.photos().loc[prev.H037]
    box = pd.read_csv(ig.out / "dinov2_hiyerarsik_kadrajlar.csv").set_index("file").loc[prev.H037]
    with Image.open(row.path) as image:
        photo = ImageOps.exif_transpose(image).convert("RGB")
    w, h = photo.size
    assert w < h and (h, w) == (box.width, box.height)
    native_box = (w - box.bottom, box.left, w - box.top, box.right)
    ref, base = ig.gsd.PHONES[row.phone]
    ppm = base * max(w, h) / ref
    roi = photo.crop(native_box)
    shape = tuple(round(size * ig.PPM / ppm) for size in roi.size)
    rgb = np.asarray(roi.resize(shape, Image.Resampling.LANCZOS))
    return row, photo, native_box, ppm, rgb


def infer():
    row, _, box, _, rgb = surface()
    model = ig.load_model(prev.near.WEIGHTS)
    print("Full surface (native orientation)", rgb.shape, flush=True)
    mask, *_ = model.eval(rgb, batch_size=1, bsize=256, flow_threshold=.4, cellprob_threshold=0.)
    assert mask.shape == rgb.shape[:2] and mask.min() >= 0 and mask.max() <= 65535
    tmp = ARCHIVE.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, photo_name=row.file, box=box, mask=mask.astype(np.uint16))
    tmp.replace(ARCHIVE)
    print("Full surface saved; instances", mask.max(), flush=True)


def previous_areas(row, photo, box, ppm, shape):
    """Project the archived 234 px patches onto the full surface using pixel centers."""
    w, h = photo.size
    left, top, right, bottom = box
    height, width = shape
    xx = left + (np.arange(width) + .5) * (right-left) / width
    yy = top + (np.arange(height) + .5) * (bottom-top) / height
    patch_side = round(ig.SIDE_MM * ppm)
    squares = np.zeros(shape, dtype=np.uint8)
    old = np.zeros(shape, dtype=np.uint16)
    with np.load(prev.ARCHIVE) as archive:
        patches = archive[row.file]
    next_id = 1
    for idx, ((_, (dx, dy)), patch) in enumerate(zip(prev.fields(row), patches), 1):
        x0 = (w - patch_side) // 2 + round(dx * ppm)
        y0 = (h - patch_side) // 2 + round(dy * ppm)
        px = np.floor((xx - x0) * ig.SIDE / patch_side).astype(int)
        py = np.floor((yy - y0) * ig.SIDE / patch_side).astype(int)
        valid = ((0 <= py) & (py < ig.SIDE))[:, None] & ((0 <= px) & (px < ig.SIDE))[None, :]
        lookup = np.ix_(np.clip(py, 0, ig.SIDE-1), np.clip(px, 0, ig.SIDE-1))
        core = valid & ig.CORE[lookup]
        assert core.any() and not squares[core].any()
        squares[core] = idx
        labels = ig.evaluated(patch)
        projected = labels[lookup]
        for label in np.unique(labels[labels > 0]):
            footprint = valid & (projected == label)
            assert footprint.any() and not old[footprint].any()
            old[footprint] = next_id
            next_id += 1
    return squares, old


def draw(rgb, squares, grains, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    from skimage.segmentation import find_boundaries

    photo = np.rot90(rgb)
    squares = np.rot90(squares)
    grains = np.rot90(grains)
    overlay = photo.copy()
    inside = set(counts.loc[counts.bes_karede, "etiket"])
    for stone in regionprops(grains):
        rr, cc = stone.coords.T
        tone = np.array([248, 207, 46] if stone.label in inside else [235, 56, 124])
        overlay[rr, cc] = (.66 * overlay[rr, cc] + .34 * tone).astype(np.uint8)
    for eligible, color in [(True, [255, 213, 31]), (False, [239, 43, 116])]:
        ids = counts.loc[counts.bes_karede == eligible, "etiket"].to_numpy(dtype=int)
        boundary = find_boundaries(np.where(np.isin(grains, ids), grains, 0), mode="thick")
        overlay[boundary] = color
    fig, axes = plt.subplots(2, 1, figsize=(12, 12.1))
    for ax, frame, title in zip(axes, (photo, overlay),
                                ("H037 · Motorola · taranan toprak yüzeyi", "Aynı yüzeyde taşlar ve önceki beş kare")):
        ax.imshow(frame, interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=15, weight="bold")
        ax.set_axis_off()
    for square in regionprops(squares):
        y0, x0, y1, x1 = square.bbox
        axes[1].add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, edgecolor="#16eaef", facecolor="none", lw=1.5))
    styles = [Line2D([], [], color="#09abad", lw=2, label="Önceki 26 mm’lik kareler"),
              Patch(color="#f2c928", label="Kareye giren iri taş"),
              Patch(color="#ec3679", label="Beş karenin dışında kalan iri taş")]
    fig.legend(handles=styles, loc="lower center", ncols=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.023, right=.977, top=.964, bottom=.052, hspace=.10)
    fig.savefig(ig.out / f"{STEM}.png", dpi=140, facecolor="white")
    plt.close(fig)


def score():
    row, photo, box, ppm, rgb = surface()
    with np.load(ARCHIVE) as z:
        assert z["photo_name"].item() == row.file and tuple(z["box"]) == box
        raw = z["mask"]
    assert raw.shape == rgb.shape[:2]
    squares, old = previous_areas(row, photo, box, ppm, raw.shape)
    coarse = ig.coarse_labels(raw)
    records = []
    for grain in regionprops(coarse):
        rr, cc = grain.coords.T
        overlap = np.bincount(squares[rr, cc], minlength=6)[1:]
        cy, cx = np.floor(np.asarray(grain.centroid) + .5).astype(int)
        y0, x0, y1, x1 = grain.bbox
        records.append(dict(etiket=grain.label, kisa_eksen_mm=grain.axis_minor_length / ig.PPM,
                            alan_mm2=grain.area / ig.PPM**2, karede_mm2=overlap.sum() / ig.PPM**2,
                            bes_karede=bool(overlap.max() >= ig.PPM**2), merkez_karede=bool(squares[cy, cx]),
                            kadraj_sinirinda=bool(y0 == 0 or x0 == 0 or y1 == raw.shape[0] or x1 == raw.shape[1])))
    counts = pd.DataFrame(records, columns=["etiket", "kisa_eksen_mm", "alan_mm2", "karede_mm2",
                                            "bes_karede", "merkez_karede", "kadraj_sinirinda"])
    counts.to_csv(ig.out / f"{STEM}_taslar.csv", index=False, float_format="%.6f")
    print(counts.to_string(index=False))
    print("ROIs area pct", 100*np.mean(squares > 0), "approx cm2", squares.size / (100*ig.PPM**2))
    print("Within and total coarse area mm2", counts.karede_mm2.sum(), counts.alan_mm2.sum())
    print("Object counts within, total, center within, partial overlap below 1mm2, edge",
          counts.bes_karede.sum(), len(counts), counts.merkez_karede.sum(),
          ((counts.karede_mm2 > 0) & ~counts.bes_karede).sum(), counts.kadraj_sinirinda.sum())
    for oid in np.unique(old[old > 0]):
        old_obj = old == oid
        covered, n = np.unique(raw[old_obj], return_counts=True)
        matches = [(k, n0 / (old_obj.sum() + np.count_nonzero(raw == k) - n0))
                   for k, n0 in zip(covered, n) if k > 0]
        best, iou = max(matches, key=lambda pair: pair[1], default=(0, 0.))
        print("Previous object", oid, "best full raw ID / IoU", best, iou,
              "coarse", bool(best and (coarse == best).any()))
    draw(rgb, squares, coarse, counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["infer", "score"])
    args = parser.parse_args()
    if args.step == "infer":
        infer()
    else:
        score()
