"""H037'nin aynı Motorola çekiminde 4 × 4 bölge ve tam yüzey tespitlerini karşılaştır."""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage.measure import regionprops

import h037_tam_yuzey as full

ig = full.ig
STEM = "h037_onalti_bolge"
ARCHIVE = ig.out / f"{STEM}_maskeler.npz"
MARGIN = int(np.ceil(13 * ig.PPM))
THRESHOLD = 6.3
MIN_IOU = .5
BOUNDARY = ig.out / f"{STEM}_sinir_kontrolu.npz"


def with_margin(core, shape):
    y0, x0, y1, x1 = core
    h, w = shape
    return max(0, y0-MARGIN), max(0, x0-MARGIN), min(h, y1+MARGIN), min(w, x1+MARGIN)


def layout(shape):
    h, w = shape
    ys = np.rint(np.linspace(0, h, 5)).astype(int)
    xs = np.rint(np.linspace(0, w, 5)).astype(int)
    for i, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
        for j, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
            # Label rows and columns as seen in the landscape display (90° counterclockwise).
            name = f"r{4-j}c{i+1}"
            core = (y0, x0, y1, x1)
            yield name, core, with_margin(core, shape)


def boundary_layout(ref):
    # The one coarse stone previously found inside the five squares is the preselected test case.
    coarse = pd.read_csv(ig.out / "h037_tam_yuzey_taslar.csv")
    label = coarse.loc[coarse.bes_karede, "etiket"].item()
    yy, xx = np.nonzero(ref == label)
    sy, sx = np.rint([yy.mean()+.5, xx.mean()+.5]).astype(int)
    tiles = [core for _, core, _ in layout(ref.shape) if core[0] <= sy < core[2]]
    shared = min((core[3] for core in tiles[:-1]), key=lambda x: abs(x - sx))
    left = next(core for core in tiles if core[3] == shared)
    right = next(core for core in tiles if core[1] == shared)
    assert left[1] < sx < right[3] and xx.min() < sx <= xx.max()
    cores = [(left[0], left[1], left[2], sx), (right[0], sx, right[2], right[3])]
    return [(name, core, with_margin(core, ref.shape)) for name, core in zip(["side0", "side1"], cores)]


def infer(stress=False):
    from cellpose import models, transforms

    row, _, box, _, rgb = full.surface()
    # Use the exact full-frame input normalization from Cellpose before slicing.
    image = transforms.convert_image(rgb)
    image = transforms.normalize_img(image[None], **dict(models.normalize_default, normalize=True, invert=False))[0]
    assert image.shape == rgb.shape
    cache = ig.work.parent / ".venv" / STEM
    cache.mkdir(exist_ok=True)
    model = None
    tiles = list(layout(image.shape[:2]))
    output = ARCHIVE
    if stress:
        with np.load(full.ARCHIVE) as ref:
            tiles = boundary_layout(ref["mask"])
        output = BOUNDARY
    masks = {}
    for number, (name, _, outer) in enumerate(tiles, 1):
        y0, x0, y1, x1 = outer
        path = cache / f"{name}.npy"
        if path.exists():
            mask = np.load(path)
        else:
            if model is None:
                model = ig.load_model(full.prev.near.WEIGHTS)
            patch = np.ascontiguousarray(image[y0:y1, x0:x1])
            mask, *_ = model.eval(patch, normalize=False, batch_size=1, bsize=256,
                                  flow_threshold=.4, cellprob_threshold=0.)
            assert mask.min() >= 0 and mask.max() <= 65535
            mask = mask.astype(np.uint16)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, mask)
            tmp.replace(path)
        assert mask.shape == (y1-y0, x1-x0) and mask.dtype == np.uint16
        masks[name] = mask
        print(f"{number}/{len(tiles)} {name}", flush=True)
    tmp = output.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **masks, photo_name=row.file, box=box, margin=MARGIN, shape=image.shape[:2])
    tmp.replace(output)


@dataclass
class Grain:
    tile: str
    label: int
    pixels: np.ndarray
    minor: float
    owned: bool
    clearance: float
    bbox: tuple


def read_grains(archive, tiles, shape):
    grains = []
    for name, (y0, x0, y1, x1), (oy, ox, ey, ex) in tiles:
        mask = archive[name]
        assert mask.shape == (ey-oy, ex-ox)
        for obj in regionprops(mask):
            cy, cx = np.asarray(obj.centroid) + [oy+.5, ox+.5]
            owned = y0 <= cy < y1 and x0 <= cx < x1
            rr, cc = obj.coords.T
            pixels = np.ravel_multi_index((rr+oy, cc+ox), shape)
            a, b, c, d = obj.bbox
            bbox = (a+oy, b+ox, c+oy, d+ox)
            # Prefer the mask with most surrounding context when duplicate owners occur.
            clearance = min(cy-oy, ey-cy, cx-ox, ex-cx)
            grains.append(Grain(name, obj.label, pixels, obj.axis_minor_length / ig.PPM, owned, clearance, bbox))
    return grains


def deduplicate(grains):
    kept = []
    buckets = {}
    for grain in sorted((g for g in grains if g.owned), key=lambda g: (-g.clearance, g.tile, g.label)):
        y0, x0, y1, x1 = grain.bbox
        bins = [(i, j) for i in range(y0//64, (y1-1)//64+1) for j in range(x0//64, (x1-1)//64+1)]
        nearby = {i for cell in bins for i in buckets.get(cell, [])}
        duplicate = False
        for i in nearby:
            other = kept[i]
            if other.tile == grain.tile:
                continue
            n = np.intersect1d(other.pixels, grain.pixels, assume_unique=True).size
            if n >= MIN_IOU * (len(other.pixels) + len(grain.pixels) - n):
                duplicate = True
                break
        if not duplicate:
            for cell in bins:
                buckets.setdefault(cell, []).append(len(kept))
            kept.append(grain)
    return kept


def overlaps(grain, ref, areas):
    ids, counts = np.unique(ref.ravel()[grain.pixels], return_counts=True)
    return [(int(k), n / (len(grain.pixels) + areas[k] - n)) for k, n in zip(ids, counts) if k]


def match(grains, ref):
    from scipy.optimize import linear_sum_assignment
    areas = np.bincount(ref.ravel())
    weights = np.zeros((len(grains), len(areas)))
    for i, grain in enumerate(grains):
        for label, iou in overlaps(grain, ref, areas):
            if iou >= MIN_IOU:
                weights[i, label] = 1 + min(weights.shape) + iou
    a, b = linear_sum_assignment(weights, maximize=True)
    return {int(label): (grains[i], weights[i,label] - (1 + min(weights.shape)))
            for i, label in zip(a, b) if weights[i,label] > 0}


def draw(rgb, field, table, kept, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from skimage.segmentation import find_boundaries

    colors = {"ortak": "#facc15", "yalniz_tam": "#ed4685", "yalniz_16": "#58f3b8"}
    names = {"ortak": "İki taramada da iri", "yalniz_tam": "Yalnız tam yüzeyde iri",
             "yalniz_16": "Yalnız 16 bölgede iri"}
    lookup = {(g.tile, g.label): g for g in kept}
    def outline(ax, binary, color, **style):
        ax.contour(binary.astype(float), levels=[.5], colors=[color], **style)
    fig, ax = plt.subplots(figsize=(12, 6.9))
    ax.imshow(np.rot90(rgb), interpolation="nearest")
    grid = np.rot90(field)
    seam = find_boundaries(grid, mode="thick")
    ax.imshow(np.ma.masked_where(~seam, seam), cmap=matplotlib.colors.ListedColormap(["#6ad6ee"]),
              interpolation="nearest", alpha=.70)
    for n, rec in enumerate(table.itertuples(), 1):
        binary = np.zeros(ref.shape, dtype=bool)
        if rec.durum == "yalniz_tam":
            binary = ref == rec.tam_id
        else:
            binary.ravel()[lookup[rec.bolge, int(rec.bolge_id)].pixels] = True
        shown = np.rot90(binary)
        outline(ax, shown, colors[rec.durum], linewidths=1.8)
        yy, xx = np.nonzero(shown)
        ax.annotate(str(n), (xx.mean(), yy.min()), xytext=(0, 7), textcoords="offset points", ha="center",
                    color="white", weight="bold", fontsize=8,
                    bbox=dict(boxstyle="round,pad=.23", facecolor="#222831", edgecolor="none", alpha=.88))
    ax.set_axis_off()
    fig.suptitle("H037 · aynı toprak yüzeyinde 4 × 4 tarama", x=.04, ha="left", y=.985,
                 fontsize=16, fontweight="bold")
    handles = [Line2D([], [], color="#37a2bc", lw=2, label="Bölge sınırı")]
    handles += [Line2D([], [], color=c, lw=3, label=names[key]) for key, c in colors.items() if key in set(table.durum)]
    fig.legend(handles=handles, loc="lower center", ncols=len(handles), frameon=False)
    fig.subplots_adjust(top=.93, bottom=.075, left=.025, right=.975)
    fig.savefig(ig.out / f"{STEM}.png", dpi=145)
    plt.close(fig)


def draw_boundary(rgb, ref, target, tiles, views, stress):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yy, _ = np.nonzero(ref == target)
    cy = int(round(yy.mean()))
    split = tiles[0][1][3]
    half = round(11*ig.PPM)
    crop = np.s_[cy-half:cy+half, split-half:split+half]
    frame = np.rot90(rgb[crop])
    lookup = {(g.tile,g.label): g for g in views}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.9))
    for ax, (name, core, _) in zip(axes, tiles):
        ax.imshow(frame, interpolation="nearest")
        ax.axhline(half-.5, color="#59dbf1", lw=1.3, ls="--")
        rows = stress[stress.gorunum == name]
        detail = "Eşleşen maske yok"
        if not rows.empty:
            rec = rows.sort_values("iou").iloc[-1]
            g = lookup[name, rec.yerel_id]
            mask = np.zeros(ref.shape, bool)
            mask.ravel()[g.pixels] = True
            ax.contour(np.rot90(mask[crop]).astype(float), levels=[.5], colors=["#facc15"], linewidths=1.8)
            ry, rx = np.unravel_index(g.pixels, ref.shape)
            ax.plot(ry.mean()-(cy-half), (split+half-1)-rx.mean(), "w+", ms=9, mew=1.5)
            state = "birleşimde korundu" if rec.birlesimde else "birleşime alınmadı"
            detail = f"{g.minor:.2f} mm · {state}".replace(".", ",")
        ax.set_title("Sayım alanı çizginin " + ("altında" if core[3] == split else "üstünde"), fontsize=11)
        ax.set_xlabel(detail, fontsize=9)
        ax.set(xticks=[], yticks=[])
    fig.suptitle("H037 · bölge çizgisi taşın içinden geçtiğinde", fontsize=14, weight="bold")
    fig.text(.5, .025, "Sarı çizgi taş maskesi, beyaz artı maskenin merkezi.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.035, right=.965, top=.82, bottom=.14, wspace=.12)
    fig.savefig(ig.out / f"{STEM}_sinir.png", dpi=150)
    plt.close(fig)


def score():
    row, _, box, _, rgb = full.surface()
    with np.load(full.ARCHIVE) as z:
        ref = z["mask"]
    shape = ref.shape
    tiles = list(layout(shape))
    field = np.zeros(shape, np.uint8)
    for i, (_, (y0, x0, y1, x1), _) in enumerate(tiles, 1):
        assert not field[y0:y1, x0:x1].any()
        field[y0:y1, x0:x1] = i
    assert np.all(field > 0)
    with np.load(ARCHIVE) as z:
        assert z["photo_name"].item() == row.file and tuple(z["box"]) == box
        assert int(z["margin"]) == MARGIN and tuple(z["shape"]) == shape
        grains = read_grains(z, tiles, shape)
    kept = deduplicate(grains)
    linked = match(kept, ref)
    areas = np.bincount(ref.ravel())
    reference = {p.label: p for p in regionprops(ref)}
    ref_large = {k for k,p in reference.items() if p.axis_minor_length >= THRESHOLD*ig.PPM}
    records = []
    taken = set()
    for k, p in reference.items():
        grain, iou = linked.get(k, (None, 0.))
        if grain is not None:
            taken.add((grain.tile, grain.label))
        is_large = grain is not None and grain.minor >= THRESHOLD
        if k not in ref_large and not is_large:
            continue
        records.append(dict(tam_id=k, tam_mm=p.axis_minor_length/ig.PPM, bolge=grain.tile if grain else "",
                            bolge_id=grain.label if grain else "", bolge_mm=grain.minor if grain else np.nan,
                            iou=iou, durum="ortak" if is_large and k in ref_large else
                            ("yalniz_tam" if k in ref_large else "yalniz_16")))
    for g in kept:
        if g.minor >= THRESHOLD and (g.tile, g.label) not in taken:
            records.append(dict(tam_id=np.nan, tam_mm=np.nan, bolge=g.tile, bolge_id=g.label,
                                bolge_mm=g.minor, iou=0., durum="yalniz_16"))
    table = pd.DataFrame(records)
    table.to_csv(ig.out / f"{STEM}_eslesmeler.csv", index=False, float_format="%.6f")
    print(table.to_string(index=False))
    crossing = {k for k,p in reference.items() if np.unique(field[tuple(p.coords.T)]).size > 1}
    print("Natural grid crossing: raw ref, matched, coarse ref", len(crossing), len(crossing & linked.keys()),
          len(crossing & ref_large))
    print("All raw, owner, deduplicated", len(grains), sum(g.owned for g in grains), len(kept))
    print("Coarse raw, owner, deduplicated", sum(g.minor >= THRESHOLD for g in grains),
          sum(g.owned and g.minor >= THRESHOLD for g in grains), sum(g.minor >= THRESHOLD for g in kept))
    stress_tiles = boundary_layout(ref)
    with np.load(BOUNDARY) as z:
        assert z["photo_name"].item() == row.file and tuple(z["box"]) == box and int(z["margin"]) == MARGIN
        views = read_grains(z, stress_tiles, shape)
    target = pd.read_csv(ig.out / "h037_tam_yuzey_taslar.csv").query("bes_karede").etiket.item()
    outcomes = []
    stress_kept = {(g.tile, g.label) for g in deduplicate(views)}
    for g in views:
        for k, v in overlaps(g, ref, areas):
            if k == target and v >= MIN_IOU:
                outcomes.append(dict(tam_id=k, gorunum=g.tile, yerel_id=g.label, bolge_mm=g.minor, iou=v, merkez_iceride=g.owned,
                                     birlesimde=(g.tile,g.label) in stress_kept, iri=g.minor >= THRESHOLD))
    stress = pd.DataFrame(outcomes, columns=["tam_id", "gorunum", "yerel_id", "bolge_mm", "iou", "merkez_iceride", "birlesimde", "iri"])
    stress.to_csv(ig.out / f"{STEM}_sinir.csv", index=False, float_format="%.6f")
    print("Deliberate boundary test\n", stress.to_string(index=False))
    draw(rgb, field, table, kept, ref)
    draw_boundary(rgb, ref, target, stress_tiles, views, stress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["infer", "boundary", "score"])
    args = parser.parse_args()
    if args.step == "score":
        score()
    else:
        infer(stress=args.step == "boundary")
