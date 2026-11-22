"""Eşlenmiş beş alan ve bütün yüzeyin 4 × 4 bölgesinden numune eğrisi tahmini."""
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits

import iri_tas_kalan_doku as prev


ig = prev.ig
STEM = "bes_ve_onalti_doku"
MODES = {key: [f"{key}_{mm}" for mm in prev.DISTANCES] for key in ("eski_bes", "bes", "onalti")}
LAGS = np.floor(np.asarray(prev.DISTANCES) * ig.PPM + .5).astype(int)
CORE = np.flatnonzero(ig.CORE.any(axis=0))
RESULT = ig.out / f"{STEM}_sonuclar.csv"
PHOTO = ig.out / f"{STEM}_fotograflar.csv"


def tile_grid(shape):
    ys, xs = [np.rint(np.linspace(0, size, 5)).astype(int) for size in shape]
    return [(y, x, y1, x1) for y, y1 in zip(ys[:-1], ys[1:]) for x, x1 in zip(xs[:-1], xs[1:])]


def pair_totals(gray, boxes, shared_edges=False):
    """Assign each horizontal/right or vertical/down pair to its first pixel's box."""
    totals = np.zeros((len(LAGS), 2), dtype=np.int64)
    for y, x, y1, x1 in boxes:
        bottom, right = gray.shape if shared_edges else (y1, x1)
        for i, d in enumerate(LAGS):
            end_x, end_y = min(x1, right - d), min(y1, bottom - d)
            assert end_x > x and end_y > y
            dx = cv2.absdiff(gray[y:y1, x:end_x], gray[y:y1, x+d:end_x+d])
            dy = cv2.absdiff(gray[y:end_y, x:x1], gray[y+d:end_y+d, x:x1])
            totals[i] += (dx.sum(dtype=np.int64) + dy.sum(dtype=np.int64), dx.size + dy.size)
    return totals


def contrasts(totals):
    return totals[:, 0] / totals[:, 1] / 255.


def calibrated_surface(row, meta):
    with Image.open(row.path) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
    w, h = photo.size
    assert (max(w,h), min(w,h)) == (meta.width, meta.height)
    left, top, right, bottom = meta.left, meta.top, meta.right, meta.bottom
    if h > w:
        left, top, right, bottom = w-bottom, left, w-top, right
    ref, ppm = ig.gsd.PHONES[row.phone]
    ppm *= max(w,h) / ref
    roi = photo.crop((left, top, right, bottom))
    shape = tuple(round(size * ig.PPM / ppm) for size in roi.size)
    rgb = np.asarray(roi.resize(shape, Image.Resampling.LANCZOS))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return photo, (left, top, right, bottom), ppm, gray


def measure_photo(row, meta):
    photo, (left, top, right, bottom), ppm, gray = calibrated_surface(row, meta)
    w, h = photo.size
    shape = (gray.shape[1], gray.shape[0])
    xs = left + (np.arange(shape[0])+.5) * (right-left) / shape[0]
    ys = top + (np.arange(shape[1])+.5) * (bottom-top) / shape[1]
    source_side = round(ig.SIDE_MM * ppm)
    core_edges = np.asarray([CORE[0], CORE[-1]+1]) * source_side / ig.SIDE
    legacy = np.zeros((len(LAGS), 2), dtype=np.int64)
    boxes = []
    for _, (dx, dy) in prev.fields(row):
        x0 = (w-source_side)//2 + round(dx*ppm)
        y0 = (h-source_side)//2 + round(dy*ppm)
        crop = photo.crop((x0,y0,x0+source_side,y0+source_side)).resize((ig.SIDE,ig.SIDE), Image.Resampling.LANCZOS)
        old_gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        legacy += pair_totals(old_gray, [(CORE[0],CORE[0],CORE[-1]+1,CORE[-1]+1)])
        assert left <= x0+core_edges[0] < x0+core_edges[1] <= right
        assert top <= y0+core_edges[0] < y0+core_edges[1] <= bottom
        x, x1 = np.searchsorted(xs, x0+core_edges)
        y, y1 = np.searchsorted(ys, y0+core_edges)
        boxes.append((y,x,y1,x1))
    covered = np.zeros(gray.shape, dtype=bool)
    for y,x,y1,x1 in boxes:
        assert not covered[y:y1,x:x1].any()
        covered[y:y1,x:x1] = True
    five = pair_totals(gray, boxes)
    grid = tile_grid(gray.shape)
    sixteen = pair_totals(gray, grid, shared_edges=True)
    # Independent full-frame integer calculation verifies that every seam pair was counted once.
    full = gray.astype(np.int16)
    for (total, count), lag in zip(sixteen, LAGS):
        horizontal = np.abs(full[:, lag:] - full[:, :-lag])
        vertical = np.abs(full[lag:, :] - full[:-lag, :])
        assert total == horizontal.sum() + vertical.sum() and count == horizontal.size + vertical.size
    record = dict(file=row.file, sample_id=row.sample_id, phone=row.phone,
                  bes_piksel=int(covered.sum()), tam_piksel=gray.size)
    for columns, sums in zip(MODES.values(), (legacy, five, sixteen)):
        record.update(zip(columns, contrasts(sums)))
    return record


def measure():
    photos = prev.photos()
    roi = pd.read_csv(ig.out / "dinov2_hiyerarsik_kadrajlar.csv", index_col="file").loc[photos.index]
    assert roi.index.is_unique and np.array_equal(roi.sample_id, photos.sample_id)
    records = []
    for i, (file, row) in enumerate(photos.iterrows(), 1):
        records.append(measure_photo(row, roi.loc[file]))
        if i % 25 == 0 or i == len(photos):
            print("Photos", i, "/", len(photos), flush=True)
    local = pd.DataFrame(records).set_index("file")
    old = pd.read_csv(ig.out / "iri_tas_kalan_doku_fotograflar.csv", index_col="file").loc[local.index]
    np.testing.assert_allclose(local[MODES["eski_bes"]], old[prev.MODES["butun"]], atol=5.1e-13, rtol=0)
    local.to_csv(PHOTO, float_format="%.15f")
    return local


def run():
    cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        local = measure()
        result = prev.test_models(local, modes=MODES, stem=STEM)
    old = pd.read_csv(ig.out / "iri_tas_kalan_doku_sonuclar.csv").query("model == 'butun'")
    cols = ["emd", *ig.gsd.COLS]
    def indexed(frame):
        return frame.set_index(["exam","sample_id"]).sort_index()[cols]
    historic, replay = indexed(old), indexed(result.query("model == 'eski_bes'"))
    pd.testing.assert_index_equal(historic.index, replay.index)
    np.testing.assert_allclose(historic, replay, atol=1e-7, rtol=0)
    print(result.groupby(["exam", "model"]).emd.mean().unstack().round(5).to_string())


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tested = pd.read_csv(RESULT)
    v3 = pd.read_csv(ig.out / "v3_kadraj_sonuclari.csv").query("model == 'v3'")
    all_rows = pd.concat([tested, v3], ignore_index=True)
    wide = all_rows.pivot(index=["exam","sample_id"], columns="model", values="emd")
    assert wide.shape == (66,4) and wide.notna().all().all()
    curves = all_rows[ig.gsd.COLS].to_numpy()
    lab = ig.labels.loc[all_rows.sample_id, ig.gsd.COLS].to_numpy()
    assert np.all(np.diff(curves) >= 0) and np.all(curves[:, -1] == 100)
    np.testing.assert_allclose(np.abs(curves - lab)[:,:-1] @ np.diff(np.log10(ig.gsd.SIZES)),
                               all_rows.emd, rtol=0, atol=1e-8)
    wide["kazanc"] = wide.bes - wide.onalti
    print(wide.groupby(level="exam").mean().round(5).to_string())
    for exam, group in wide.groupby(level="exam"):
        g = group.kazanc
        omit = (g.sum()-g)/(len(g)-1)
        print(exam, "wins, ties, median, omission min/max", (g>0).sum(), (g==0).sum(), g.median(), omit.min(), omit.max())
    main = wide.xs("24")
    gain = main.kazanc.sort_values()
    print("extremes", gain.head(3).round(3).to_dict(), gain.tail(3).round(3).to_dict())
    print("without H037, H038", gain.drop(["H037","H038"]).mean())
    for col in ["0.2", "6.3"]:
        by_model = all_rows.query("exam == '24'").pivot(index="sample_id",columns="model",values=col)
        print(col, "mean abs errors", by_model.sub(ig.labels.loc[by_model.index,col],axis=0).abs().mean().round(5).to_dict())
    h037 = all_rows.query("exam == '24' and sample_id == 'H037'").set_index("model")
    print("H037\n", h037[["emd","0.2"]].round(5))
    coverage = pd.read_csv(PHOTO).eval("100 * bes_piksel / tam_piksel")
    print("5-window area, min median max", coverage.min(), coverage.median(), coverage.max())
    print("16 better than v3", (main.onalti < main.v3).sum())

    fig, (bars, curve) = plt.subplots(1,2,figsize=(10.6,6.5),layout="constrained")
    colors = np.where(gain >= 0, "#087f78", "#bc5a36")
    bars.barh(np.arange(len(gain)), gain, color=colors, height=.72)
    bars.set_yticks(np.arange(len(gain)), gain.index, fontsize=8)
    bars.axvline(0, color="#344054", lw=.8)
    bars.set_xlabel("EMD farkı (5 alan eksi 16 bölge)\nPozitif değer 16 bölgede iyileşme")
    bars.set_title("Her toprağın hata farkı", loc="left", fontweight="bold")
    sizes = ig.gsd.SIZES
    curve.plot(sizes, ig.labels.loc["H037",ig.gsd.COLS], color="#17212b", linewidth=2.2,
               marker="o", markersize=3.4, label="Laboratuvar")
    for key, label, color, linestyle in (("bes","Beş alan","#a35638","--"),("onalti","16 bölge","#087f78","-"),
                                         ("v3","Saf v3","#68768e",":")):
        curve.plot(sizes, h037.loc[key,ig.gsd.COLS].to_numpy(float), color=color, linestyle=linestyle, label=label, lw=2)
    curve.set(xscale="log", ylim=(-3,104), xlabel="Elek açıklığı (mm)", ylabel="Elekten geçen kütle (%)")
    curve.set_xticks([.002,.02,.2,2,20,200],["0,002","0,02","0,2","2","20","200"])
    curve.set_title("H037’nin elek eğrisi", loc="left", fontweight="bold")
    curve.legend(loc="lower right", frameon=False)
    for ax in (bars,curve):
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(ig.out / f"{STEM}.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", nargs="?", default="run", choices=["run", "report"])
    args = parser.parse_args()
    if args.step == "run":
        run()
    else:
        report()
