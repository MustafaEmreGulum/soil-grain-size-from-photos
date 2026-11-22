"""Görünür iri taşları ayırmanın numune düzeyindeki tane boyutu tahminine etkisi."""
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.measure import regionprops
from threadpoolctl import threadpool_limits

import imagegrains_pilot as ig
import imagegrains_bes_bolge as near
import imagegrains_uzak_alan as far


STEM = "iri_tas_kalan_doku"
ARCHIVE = ig.out / f"{STEM}_maskeler.npz"
CACHE = ig.work.parent / ".venv" / STEM
H037 = "Motorola_Edge_H037_01.jpg"
DISTANCES = (0.25, 0.5, 1, 2, 4, 8)
STONE = ["iri_alan_pct", "iri_adet_cm2", "iri_kisa_eksen_mm"]
MODES = {
    "butun": [f"butun_{mm}" for mm in DISTANCES],
    "butun_ve_tas": [f"butun_{mm}" for mm in DISTANCES] + STONE,
    "kalan_ve_tas": [f"kalan_{mm}" for mm in DISTANCES] + STONE,
}


def photos():
    rows = ig.gsd.list_images(str(ig.work / "soil-data")).query("split == 'train'")
    rows = rows.set_index("file", drop=False).sort_index()
    assert len(rows) == 127 and rows.index.is_unique and set(rows.sample_id) == set(ig.labels.index)
    return rows


def fields(row):
    long_names, distant = far.layout(row)
    inner = [(name, offset) for name, offset in near.OFFSETS.items() if name not in long_names]
    return inner + list(zip(("uzak_eksi", "uzak_arti"), distant))


def check(stack):
    assert stack.shape == (5, ig.SIDE, ig.SIDE) and stack.dtype == np.uint16


def example(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.segmentation import find_boundaries

    row = rows.loc[H037]
    _, offset = fields(row)[-1]
    rgb = ig.crop(row, offset)
    with np.load(far.ARCHIVE) as source:
        large = ig.evaluated(source[H037][1])
    foreground = large > 0
    kept = rgb.copy()
    kept[foreground] = (71, 77, 86)
    outline = rgb.copy()
    edge = find_boundaries(large, mode="outer").astype(np.uint8)
    outline[cv2.dilate(edge, np.ones((2, 2), np.uint8)) > 0] = (199, 44, 90)
    pct = 100 * foreground[ig.CORE].mean()
    with Image.open(row.path) as src:
        photo = ImageOps.exif_transpose(src).convert("RGB")
    native_w, native_h = photo.size
    ref, ppm = ig.gsd.PHONES[row.phone]
    ppm *= max(photo.size) / ref
    portrait = photo.height > photo.width
    if portrait:
        photo = photo.transpose(Image.Transpose.ROTATE_90)
    fig = plt.figure(figsize=(10.2, 7.1))
    grid = fig.add_gridspec(2, 3, height_ratios=[.82, 1], hspace=.27, wspace=.11)
    top = fig.add_subplot(grid[0, :])
    top.imshow(photo, extent=(0, photo.width, photo.height, 0))
    for name, (dx, dy) in fields(row):
        cx, cy = native_w / 2 + dx * ppm, native_h / 2 + dy * ppm
        if portrait:
            cx, cy = cy, native_w - cx
        color = "#ffcb52" if name == "uzak_arti" else "#15d9da"
        top.add_patch(Rectangle((cx-13*ppm, cy-13*ppm), 26*ppm, 26*ppm, ec=color, lw=1.8, fill=False))
    top.set_title("H037 · Motorola Edge · beş ölçüm alanı", loc="left", fontsize=13)
    top.set_axis_off()
    for axis, title, patch in zip([fig.add_subplot(grid[1, i]) for i in range(3)],
                                  ["Özgün görüntü", "İri taşın sınırı", "Geriye kalan yüzey"],
                                  [rgb, outline, kept]):
        view = np.rot90(patch) if portrait else patch
        axis.imshow(view, extent=(0, 52, 52, 0), interpolation="nearest")
        axis.add_patch(Rectangle((13, 13), 26, 26, ec="#ffcb52", lw=1.2, fill=False))
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_axis_off()
    shown_pct = f"{pct:.2f}".replace(".", ",")
    fig.text(.03, .045, f"Sarı kare 26 × 26 mm. İri taş bu karenin %{shown_pct} kadarını kaplıyor.", fontsize=10)
    fig.text(.03, .014, "Pembe çizgi modelin taş sınırı. Gri pikseller kalan dokunun ölçümünden çıkarılıyor.", fontsize=10)
    fig.subplots_adjust(left=.025, right=.98, top=.94, bottom=.096)
    fig.savefig(ig.out / f"{STEM}_H037.png", dpi=140)
    plt.close(fig)


def h037_fields(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    selected = rows[rows.sample_id == "H037"]
    assert len(selected) == 6
    positions = [(0, 0), (0, -30), (0, 30), (-100, 0), (100, 0)]
    headings = ["Merkez", "30 mm üst", "30 mm alt", "100 mm sol", "100 mm sağ"]
    core_rows, core_cols = np.nonzero(ig.CORE)
    core = np.s_[core_rows.min():core_rows.max()+1, core_cols.min():core_cols.max()+1]
    fig, axes = plt.subplots(len(selected), 5, figsize=(7.8, 8.4))
    detected = 0
    for i, (file, row) in enumerate(selected.iterrows()):
        archive_ready = ARCHIVE.exists()
        with np.load(ARCHIVE if archive_ready else CACHE / f"{file}.npz") as z:
            masks = z[file if archive_ready else "masks"]
        offsets = fields(row)
        portrait = offsets[-1][1][1] != 0
        for mask, (_, (dx, dy)) in zip(masks, offsets):
            pos = (dy, -dx) if portrait else (dx, dy)
            ax = axes[i, positions.index(pos)]
            rgb = ig.crop(row, (dx, dy)).copy()
            valid = ig.evaluated(mask)
            found = valid[core] > 0
            if found.any():
                detected += 1
                rgb[valid > 0] = (.77 * rgb[valid > 0] + .23 * np.array([236, 71, 143])).astype(np.uint8)
                rgb[find_boundaries(valid, mode="outer")] = [211, 28, 95]
                ax.text(.97, .03, f"%{100*found.mean():.1f}".replace(".", ","), transform=ax.transAxes,
                        color="white", ha="right", va="bottom", fontsize=8,
                        bbox=dict(facecolor="#30212c", alpha=.8, edgecolor="none", pad=1.5))
            ax.imshow(np.rot90(rgb[core]) if portrait else rgb[core], interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            ax.spines[:].set_visible(False)
        axes[i, 0].set_ylabel(row.phone.split("_")[0] + " " + file[-6:-4].lstrip("0"), fontsize=10)
    for ax, title in zip(axes[0], headings):
        ax.set_title(title, fontsize=11)
    fig.suptitle("H037 · altı çekimde aynı beş ölçüm konumu", fontsize=15, fontweight="bold", y=.987)
    fig.text(.06, .024, f"Pembe, modelin işaretlediği iri taş alanı. 30 karenin {detected} tanesinde işaret var. Her kare 26 × 26 mm.", fontsize=9)
    fig.subplots_adjust(left=.085, right=.99, bottom=.057, top=.92, hspace=.08, wspace=.02)
    fig.savefig(ig.out / f"{STEM}_H037_tum_alanlar.png", dpi=130)
    plt.close(fig)
    print("H037 fields with coarse detections:", detected, "/ 30")


def segment_all(rows):
    if ARCHIVE.exists():
        with np.load(ARCHIVE) as saved:
            assert set(saved.files) == set(rows.index)
            for name in saved.files:
                check(saved[name])
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_csv(ig.PILOT).set_index("file")
    near_names = list(near.OFFSETS)
    model = None
    completed = {}
    with np.load(near.ARCHIVE) as n, np.load(far.ARCHIVE) as f, np.load(ig.MASKS) as p:
        assert set(n.files) == set(f.files)
        for i, (file, row) in enumerate(rows.iterrows(), 1):
            path = CACHE / f"{file}.npz"
            if path.exists():
                with np.load(path) as z:
                    stack = z["masks"]
            else:
                selected = fields(row)
                if file in n.files:
                    cached = n[file]
                    stack = np.stack([cached[near_names.index(name)] for name, _ in selected[:3]] + list(f[file]))
                else:
                    masks = []
                    for name, offset in selected:
                        if name == "merkez" and file in pilot.index:
                            mask = p[pilot.loc[file, "panel"]]
                        else:
                            if model is None:
                                model = ig.load_model(near.WEIGHTS)
                            mask = ig.segment(model, ig.crop(row, offset))
                        masks.append(mask)
                    stack = np.stack(masks)
                check(stack)
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, masks=stack)
                temp.replace(path)
            check(stack)
            completed[file] = stack
            print(f"{i}/{len(rows)}  {file}", flush=True)
    temp = ARCHIVE.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **completed)
    temp.replace(ARCHIVE)


def contrast_pairs(gray, removed, lag):
    """Sum and count intensity differences whose entire horizontal/vertical span is eligible."""
    all_sum = 0.
    kept_sum = 0.
    all_count = 0
    kept_count = 0
    for values, blocked in [(gray, removed), (gray.T, removed.T)]:
        delta = np.abs(values[:, lag:] - values[:, :-lag])
        prefix = np.pad(blocked.cumsum(axis=1, dtype=np.int32), ((0, 0), (1, 0)))
        valid = (prefix[:, lag + 1:] - prefix[:, :-(lag + 1)]) == 0
        all_sum += delta.sum(dtype=np.float64)
        kept_sum += delta[valid].sum(dtype=np.float64)
        all_count += delta.size
        kept_count += valid.sum()
    return np.array([all_sum, all_count, kept_sum, kept_count])


def measurements(rows):
    records = []
    pixels = np.flatnonzero(ig.CORE.any(axis=0))
    core = slice(pixels[0], pixels[-1] + 1)
    with np.load(ARCHIVE) as archive:
        assert set(archive.files) == set(rows.index)
        for file, row in rows.iterrows():
            masks = archive[file]
            check(masks)
            totals = np.zeros((len(DISTANCES), 4))
            area_px = centers = positive_fields = 0
            size_area = 0.
            for raw, (_, offset) in zip(masks, fields(row)):
                large = ig.evaluated(raw)
                labels_core = large[core, core]
                positive_fields += int(labels_core.any())
                for grain in regionprops(large):
                    area = np.count_nonzero(labels_core == grain.label)
                    area_px += area
                    size_area += area * grain.axis_minor_length / ig.PPM
                    cy, cx = (np.asarray(grain.centroid) + 0.5) / ig.PPM
                    centers += int(13 <= cx < 39 and 13 <= cy < 39)
                rgb = ig.crop(row, offset)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[core, core].astype(np.float64) / 255
                for i, mm in enumerate(DISTANCES):
                    lag = int(np.floor(mm * ig.PPM + 0.5))
                    totals[i] += contrast_pairs(gray, labels_core > 0, lag)
            measured_px = len(masks) * int(ig.CORE.sum())
            record = dict(file=file, sample_id=row.sample_id, phone=row.phone,
                          iri_tasli_kare=positive_fields,
                          iri_alan_pct=100 * area_px / measured_px,
                          iri_adet_cm2=centers / (measured_px / (100 * ig.PPM**2)),
                          iri_kisa_eksen_mm=size_area / area_px if area_px else 0.)
            for mm, (total, n, kept, k) in zip(DISTANCES, totals):
                assert n > 0 and 0 <= k <= n
                record[f"butun_{mm}"] = total / n
                record[f"kalan_{mm}"] = kept / k if k else np.nan
                record[f"kalan_cift_{mm}"] = int(k)
                if not area_px:
                    assert k == n and abs(total - kept) < 1e-8
            records.append(record)
    result = pd.DataFrame(records).set_index("file")
    result.to_csv(ig.out / f"{STEM}_fotograflar.csv", float_format="%.12f")
    return result


def test_models(meta, modes=MODES, stem=STEM):
    with np.load(ig.out / "dinov2_hiyerarsik_temsiller.npz") as z:
        names = pd.Index(z["files"], name="file")
        assert names.is_unique and set(names) == set(meta.index)
        picture = pd.DataFrame(z["genel_yerel"][:, :384], index=names, dtype=np.float64).loc[meta.index]
    np.testing.assert_array_equal(picture.index, meta.index)
    cols = list(dict.fromkeys(col for columns in modes.values() for col in columns))
    paired = set.intersection(*(set(meta.loc[meta.phone == p, "sample_id"]) for p in ig.PHONES))
    assert len(paired) == 21
    targets = pd.DataFrame(ig.gsd.curve_to_logq(ig.labels), index=ig.labels.index)
    experiments = [(None, None, ig.labels.index), (*ig.PHONES, sorted(paired)), (*ig.PHONES[::-1], sorted(paired))]
    records = []
    for source, dest, ids in experiments:
        train = meta if source is None else meta[meta.phone == source]
        query = meta if dest is None else meta[meta.phone == dest]
        src_global = picture.loc[train.index].groupby(train.sample_id).mean()
        dst_global = picture.loc[query.index].groupby(query.sample_id).mean()
        src_local = train.groupby("sample_id")[cols].mean()
        dst_local = query.groupby("sample_id")[cols].mean()
        exam = "24" if source is None else f"{source} -> {dest}"
        for soil in ids:
            pool = pd.Index(ids).drop(soil)
            assert soil not in pool and len(pool) in (20, 23)
            reducer = make_pipeline(StandardScaler(), PCA(n_components=8, whiten=True, svd_solver="full"))
            global_x = reducer.fit_transform(src_global.loc[pool])
            global_q = reducer.transform(dst_global.loc[[soil]])
            for model, columns in modes.items():
                scaler = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
                x = np.column_stack([global_x, scaler.fit_transform(src_local.loc[pool, columns])])
                q = np.column_stack([global_q, scaler.transform(dst_local.loc[[soil], columns])])
                ridge = Ridge(alpha=10, solver="svd").fit(x, targets.loc[pool])
                curve = ig.gsd.logq_to_curve(ridge.predict(q))[0]
                assert np.isfinite(curve).all() and np.all(np.diff(curve) >= 0) and 0 <= curve[0] and curve[-1] == 100
                emd = ig.gsd.emd(ig.labels.loc[soil], curve)[0]
                records.append(dict(exam=exam, sample_id=soil, model=model, emd=emd, **dict(zip(ig.gsd.COLS, curve))))
    result = pd.DataFrame(records)
    assert len(result) == 66 * len(modes) and not result.duplicated(["exam", "sample_id", "model"]).any()
    result.to_csv(ig.out / f"{stem}_sonuclar.csv", index=False, float_format="%.9f")
    return result


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    local = pd.read_csv(ig.out / f"{STEM}_fotograflar.csv", index_col="file")
    previous = pd.read_csv(ig.out / "imagegrains_uzak_alan_fotograflar.csv", index_col="file")
    assert len(local) == 127 and len(previous) == 58 and local.index.is_unique
    np.testing.assert_allclose(local.loc[previous.index, "iri_alan_pct"], previous.bes_uzak, rtol=0, atol=1e-8)
    results = pd.read_csv(ig.out / f"{STEM}_sonuclar.csv")
    reference = pd.read_csv(ig.out / "v3_kadraj_sonuclari.csv").query("model == 'v3'")
    combined = pd.concat([results, reference], ignore_index=True)
    paired = combined.pivot(index=["exam", "sample_id"], columns="model", values="emd")
    assert paired.shape == (66, 4) and not paired.isna().any().any()
    actual = ig.labels.loc[combined.sample_id, ig.gsd.COLS].to_numpy()
    curves = combined[ig.gsd.COLS].to_numpy()
    assert (np.diff(curves, axis=1) >= 0).all() and (curves[:, -1] == 100).all()
    weights = np.diff(np.log10(np.asarray(ig.gsd.COLS, dtype=float)))
    np.testing.assert_allclose(np.abs(curves - actual)[:, :-1] @ weights, combined.emd, rtol=0, atol=1e-8)
    print("Mean EMD\n", paired.groupby(level="exam").mean().round(4))
    for earlier, later in [("butun", "kalan_ve_tas"), ("butun_ve_tas", "kalan_ve_tas"),
                           ("butun", "butun_ve_tas"), ("v3", "kalan_ve_tas")]:
        gain = paired[earlier] - paired[later]
        for exam, values in gain.groupby(level="exam"):
            omitted = (values.sum() - values) / (len(values) - 1)
            print(earlier, "->", later, exam, "mean/median/wins/omit-range",
                  round(values.mean(), 4), round(values.median(), 4), int((values > 0).sum()),
                  (round(omitted.min(), 4), round(omitted.max(), 4)))
    for col in ("0.2", "6.3"):
        obs = combined.query("exam == '24'").copy()
        obs["abs_error"] = abs(obs[col].to_numpy() - ig.labels.loc[obs.sample_id, col].to_numpy())
        print(col, "mean abs error", obs.groupby("model").abs_error.mean().round(4).to_dict())
    h037 = combined.query("exam == '24' and sample_id == 'H037'").set_index("model")
    print("H037\n", h037[["0.2", "6.3", "emd"]].round(4))
    print("Lab H037", ig.labels.loc["H037", ["0.2", "6.3"]].to_dict())
    print("Photo count with detected coarse area", int((local.iri_alan_pct > 0).sum()))
    print("Detected coarse fields", int(local.iri_tasli_kare.sum()), "/", 5 * len(local))
    h037_local = local.loc[local.sample_id == "H037"]
    print("H037 fields by photo\n", h037_local[["iri_tasli_kare", "iri_alan_pct"]].round(4))

    names = {"butun": "Bütün doku", "butun_ve_tas": "Bütün doku + taş ölçüleri",
             "kalan_ve_tas": "Kalan doku + taş ölçüleri", "v3": "Saf v3"}
    colors = {"butun": "#64748b", "butun_ve_tas": "#95644b", "kalan_ve_tas": "#087f78", "v3": "#384a7f"}
    mean = paired.xs("24").mean()
    fig, (bars, curve) = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for y, (model, name) in enumerate(names.items()):
        bars.barh(y, mean[model], color=colors[model], height=.48)
        bars.text(mean[model] + .65, y, f"{mean[model]:.2f}".replace(".", ","), va="center")
    bars.set_yticks(range(len(names)), names.values())
    bars.invert_yaxis()
    bars.set(xlim=(0, mean.max() * 1.18), xlabel="Ortalama EMD · düşük değer daha iyi")
    bars.set_title("24 toprağın bütün elek eğrisi", loc="left", fontweight="bold")
    sizes = ig.gsd.SIZES
    curve.plot(sizes, ig.labels.loc["H037", ig.gsd.COLS], color="#131a23", marker="o", markersize=3,
               linewidth=2.2, label="Laboratuvar")
    for model in ("butun", "kalan_ve_tas", "v3"):
        curve.plot(sizes, h037.loc[model, ig.gsd.COLS].to_numpy(float), color=colors[model],
                   linestyle="--" if model == "butun" else "-", linewidth=1.7, label=names[model])
    curve.axvline(.2, color="#a0a6b0", lw=.8, zorder=0)
    curve.set(xscale="log", ylim=(-3, 105), xlabel="Elek açıklığı (mm)", ylabel="Elekten geçen kütle (%)")
    curve.set_xticks([.002, .02, .2, 2, 20, 200], ["0,002", "0,02", "0,2", "2", "20", "200"])
    curve.set_title("H037 tahminleri ve laboratuvar ölçümü", loc="left", fontweight="bold")
    curve.legend(fontsize=8, frameon=False, loc="lower right")
    for axis in (bars, curve):
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(ig.out / f"{STEM}_karsilastirma.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["masks", "example", "h037", "score", "report"])
    args = parser.parse_args()
    if args.step == "example":
        example(photos())
    elif args.step == "h037":
        h037_fields(photos())
    elif args.step == "masks":
        segment_all(photos())
    elif args.step == "report":
        report()
    else:
        with threadpool_limits(limits=1):
            scores = test_models(measurements(photos()))
        print(scores.pivot_table(index="exam", columns="model", values="emd").to_string())
