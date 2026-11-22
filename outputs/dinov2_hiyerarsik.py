"""Tepsinin içinde genel görünüm ile genel görünüm + 50 mm yerel doku karşılaştırması."""
import argparse
import math

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy.ndimage import median_filter
from threadpoolctl import threadpool_limits

import dinov2_kesit_ridge as dino


STEM = "dinov2_hiyerarsik"
ROI = dino.OUT / f"{STEM}_kadrajlar.csv"
EDGE_MM = 50


def landscape(path):
    with Image.open(path) as f:
        rgb = ImageOps.exif_transpose(f).convert("RGB")
    return rgb if rgb.width >= rgb.height else rgb.transpose(Image.Transpose.ROTATE_90)


def edge_inset(channel):
    depth, width = channel.shape
    # A tray rail is cool/neutral compared with the soil. Look for a continuous
    # rail in several strips, even when it tilts slightly relative to the frame.
    border = channel[:round(depth * .23), round(.18 * width):round(.82 * width)]
    strips = np.array_split(border, 5, axis=1)
    limits = []
    for strip in strips:
        lines = median_filter(np.median(strip, axis=1), size=3, mode="nearest")
        rail = np.flatnonzero(lines < 1)
        limits.append(int(rail[-1] + 1) if rail.size else 0)
    median = np.median(limits)
    inner_edge = min(np.quantile(limits, .85), median + .018 * depth)
    # Extra clearance when a rail is visible also covers its tilt at the corners.
    margin = .032 if median > 0 else .014
    return math.ceil(inner_edge + depth * margin)


def soil_box(rgb):
    size = (420, round(420 * rgb.height / rgb.width))
    small = np.asarray(rgb.resize(size, Image.Resampling.BOX), dtype=np.int16)
    color = small[:, :, 0] - small[:, :, 2]
    h, w = color.shape
    return (math.ceil(edge_inset(color.T) * rgb.width / w), math.ceil(edge_inset(color) * rgb.height / h),
            rgb.width - math.ceil(edge_inset(color.T[::-1]) * rgb.width / w),
            rgb.height - math.ceil(edge_inset(color[::-1]) * rgb.height / h))


def grid_axis(size, side):
    assert size >= side
    starts = np.rint(np.linspace(0, size - side, math.ceil(size / side))).astype(int)
    centers = starts + side / 2
    boundaries = np.r_[0, (centers[1:] + centers[:-1]) / 2, size]
    assert starts[0] == 0 and starts[-1] + side == size and np.all(np.diff(starts) <= side)
    return starts, np.diff(boundaries)


def layout(meta):
    xs, widths = grid_axis(meta.right - meta.left, meta.side_px)
    ys, heights = grid_axis(meta.bottom - meta.top, meta.side_px)
    return [(meta.left + x, meta.top + y, w * h) for y, h in zip(ys, heights) for x, w in zip(xs, widths)]


def prepare(photos):
    rows = []
    for file, row in photos.iterrows():
        rgb = landscape(row.path)
        left, top, right, bottom = soil_box(rgb)
        original_width, ppm = dino.gsd.PHONES[row.phone]
        side = round(EDGE_MM * ppm * rgb.width / original_width)
        assert 0 <= left < right <= rgb.width and 0 <= top < bottom <= rgb.height
        nx, _ = grid_axis(right - left, side)
        ny, _ = grid_axis(bottom - top, side)
        rows.append(dict(file=file, sample_id=row.sample_id, phone=row.phone, width=rgb.width, height=rgb.height,
                         left=left, top=top, right=right, bottom=bottom, side_px=side, tiles=len(nx) * len(ny),
                         retained_fraction=(right - left) * (bottom - top) / (rgb.width * rgb.height)))
    meta = pd.DataFrame(rows).set_index("file")
    meta.to_csv(ROI, float_format="%.9f")
    preview(photos, meta)


def preview(photos, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    selected = ["Motorola_Edge_H037_01.jpg", "Samsung_A52_H037_01.jpg", "Motorola_Edge_60_fusion_H374_01.jpg"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, file in zip(axes, selected):
        m = meta.loc[file]
        rgb = landscape(photos.loc[file].path)
        ax.imshow(rgb, extent=[0, rgb.width, rgb.height, 0])
        for x, y, _ in layout(m):
            ax.add_patch(Rectangle((x, y), m.side_px, m.side_px, fill=False, lw=.55, ec="#fff3be", alpha=.75))
        ax.add_patch(Rectangle((m.left, m.top), m.right - m.left, m.bottom - m.top, fill=False, lw=2.4, ec="#00ebdb"))
        ax.set_title(f"{m.phone.replace('_', ' ')} · {m.sample_id}", fontsize=10)
        ax.set_axis_off()
    fig.text(.02, .94, "Hangi yüzeyi ölçtük?", size=15, weight="bold")
    fig.text(.02, .055, "Turkuaz çizgi kullanılan toprak alanını, açık çizgiler 50 × 50 mm bölgeleri gösteriyor.", size=10)
    fig.subplots_adjust(left=.01, right=.99, top=.86, bottom=.14, wspace=.06)
    fig.savefig(dino.OUT / f"{STEM}_alanlar.png", dpi=150)
    plt.close(fig)


def extract(photos, meta):
    model, mean, std = dino.load_encoder()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    def encode(images):
        batch = torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0, 3, 1, 2).float() / 255
        return model((batch - mean) / std).cpu().numpy().astype(np.float64)

    vectors = {"genel": [], "genel_yerel": []}
    with torch.inference_mode():
        for i, (file, row) in enumerate(photos.iterrows(), 1):
            rgb = landscape(row.path)
            m = meta.loc[file]
            crop = rgb.crop((m.left, m.top, m.right, m.bottom))
            # Keep aspect ratio to within a single ViT patch, without padding the soil with background.
            dims = tuple(max(14, 14 * round(16 * k / max(crop.size))) for k in crop.size)
            global_vec = encode([crop.resize(dims, Image.Resampling.LANCZOS)])[0]
            tiles = layout(m)
            patches = [rgb.crop((x, y, x + m.side_px, y + m.side_px)).resize((224, 224), Image.Resampling.LANCZOS)
                       for x, y, _ in tiles]
            local = np.concatenate([encode(patches[j:j + 8]) for j in range(0, len(patches), 8)])
            weights = np.array([area for _, _, area in tiles])
            assert len(tiles) == m.tiles and np.isclose(weights.sum(), crop.width * crop.height)
            avg = np.average(local, axis=0, weights=weights)
            spread = np.sqrt(np.average((local - avg) ** 2, axis=0, weights=weights))
            vectors["genel"].append(global_vec)
            vectors["genel_yerel"].append(np.concatenate([global_vec, avg, spread]))
            if i % 15 == 0 or i == len(photos):
                print(f"Encoded {i}/{len(photos)}", flush=True)
    tables = {k: pd.DataFrame(data, index=photos.index) for k, data in vectors.items()}
    np.testing.assert_array_equal(tables["genel"], tables["genel_yerel"].iloc[:, :384])
    assert all(np.isfinite(t).all().all() for t in tables.values())
    return tables


def experiment(photos):
    meta = pd.read_csv(ROI, index_col="file").loc[photos.index]
    pd.testing.assert_frame_equal(meta[["sample_id", "phone"]], photos[["sample_id", "phone"]])
    vectors = extract(photos, meta)
    np.savez_compressed(dino.OUT / f"{STEM}_temsiller.npz", files=photos.index.to_numpy(dtype=str),
                        genel_yerel=vectors["genel_yerel"].to_numpy())
    ref = pd.read_csv(dino.OUT / "ikinci_iri_aday_sonuclari.csv").set_index(["exam", "sample_id"]).v3_emd
    moto, sam = "Motorola_Edge", "Samsung_A52"
    paired = dino.labels.index.intersection(photos[photos.phone == moto].sample_id)
    paired = paired.intersection(photos[photos.phone == sam].sample_id)
    assert len(paired) == 21
    exams = [(dino.labels.index, None, None), (paired, moto, sam), (paired, sam, moto)]
    with threadpool_limits(limits=1):
        rows = [dino.evaluate(photos, vectors, ids, src, dst, ref) for ids, src, dst in exams]
    scores = pd.concat(rows, ignore_index=True)
    assert len(scores) == 132 and not scores.duplicated(["exam", "sample_id", "model"]).any()
    scores.to_csv(dino.OUT / f"{STEM}_sonuclar.csv", index=False, float_format="%.9f")
    print(scores.pivot_table(index="exam", columns="model", values="emd").to_string(), flush=True)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = pd.read_csv(dino.OUT / f"{STEM}_sonuclar.csv")
    refs = pd.read_csv(dino.OUT / "v3_kadraj_sonuclari.csv").query("model == 'v3'")
    inputs = pd.concat([scores, refs], ignore_index=True)
    for exam, block in inputs.groupby("exam", sort=False):
        pivot = block.pivot(index="sample_id", columns="model", values="emd")
        print(exam, "mean EMD", pivot.mean().round(5).to_dict())
        for base in ["genel", "v3"]:
            gain = pivot[base] - pivot["genel_yerel"]
            omit_one = (gain.sum() - gain) / (len(gain) - 1)
            print(f"{base} minus hierarchy", "mean", gain.mean(), "median", gain.median(),
                  "wins", int((gain > 0).sum()), "omit one", omit_one.min(), omit_one.max())
    main = inputs[inputs.exam == "24"]
    fine = main.assign(fine_error=(main["0.2"] - main.sample_id.map(dino.labels["0.2"])).abs())
    print("0.2 mm average absolute error", fine.groupby("model").fine_error.mean().round(3).to_dict())
    sample = main[main.sample_id == "H037"].set_index("model")
    print("H037", sample[["0.2", "emd"]].to_string())

    fig, (bars, curve) = plt.subplots(1, 2, figsize=(11, 4.4), width_ratios=[.85, 1.2])
    methods = ["v3", "genel", "genel_yerel"]
    names = ["Saf v3", "Genel", "Genel + yerel"]
    colors = ["#657684", "#35658e", "#b45920"]
    means = main.groupby("model").emd.mean().loc[methods]
    bars.barh(names, means, color=colors)
    bars.invert_yaxis()
    bars.set_xlim(0, means.max() * 1.18)
    for y, value in enumerate(means):
        bars.text(value + .7, y, f"{value:.2f}".replace(".", ","), va="center")
    bars.set_xlabel("Ortalama EMD · düşük değer daha iyi")
    bars.set_title("24 toprakta tüm eğrinin hatası", loc="left")

    x = dino.gsd.SIZES
    curve.plot(x, dino.labels.loc["H037", dino.gsd.COLS], color="#17695c", marker="o", lw=2, label="Laboratuvar")
    for method, name, color, style in zip(methods, names, colors, [":", "--", "-."]):
        curve.plot(x, sample.loc[method, dino.gsd.COLS].to_numpy(float), color=color,
                   ls=style, lw=1.9, label=name)
    curve.set(xscale="log", ylim=(-2, 105), xlabel="Tane çapı (mm)", ylabel="Bu çaptan ince malzeme (% kütle)")
    curve.set_xticks([.002, .02, .2, 2, 20, 200], ["0,002", "0,02", "0,2", "2", "20", "200"])
    curve.set_title("H037 · laboratuvar ve üç tahmin", loc="left")
    curve.grid(axis="y", color="#ddd", lw=.6)
    curve.legend(loc="lower right", frameon=False)
    for ax in (bars, curve):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.5, w_pad=2.2)
    fig.savefig(dino.OUT / f"{STEM}_karsilastirma.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "run", "report"])
    args = parser.parse_args()
    photos = dino.gsd.list_images(str(dino.work / "soil-data")).query("split == 'train'").set_index("file").sort_index()
    assert len(photos) == 127 and photos.index.is_unique and set(photos.sample_id) == set(dino.labels.index)
    if args.step == "prepare":
        prepare(photos)
    elif args.step == "run":
        experiment(photos)
    else:
        report()
