"""Aynı merkez kadrajda 4 piksel/mm ve doğal JPEG çözünürlüğünü karşılaştır."""
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ince_oran_2mm_ridge import OUT, TEXTURE, evaluate_photos, repo
from v3_knn_esit_karisim import work
import granulometry as gran
import gsd

STEM = "ince_doku_cozunurluk"


def paired_images(row):
    gray, ppm = gran.load_native(row.path, row.phone)
    size = tuple(round(n * 4 / ppm) for n in gray.shape[::-1])
    small = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize(size, Image.Resampling.LANCZOS),
                       dtype=np.float32)
    return [("ozgun", gray, ppm), ("4ppm", small, 4.)]


def measure(row):
    records = []
    for arm, image, resolution in paired_images(row):
        bright, dark = gran.cumulative(*gran.spectrum(gran.prepare(image, resolution), resolution))
        values = {f"{prefix}_{r}": v for prefix, array in [("go", bright), ("gc", dark)]
                  for r, v in zip(gran.RADII_MM, array)}
        records.append(dict(file=row.file, phone=row.phone, sample_id=row.sample_id, kol=arm,
                            ppm=resolution, **{c:values[c] for c in TEXTURE}))
    return records


def extract():
    images = gsd.list_images(work / "soil-data").query("split == 'train'")
    assert len(images) == 127 and images.file.is_unique
    records = []
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, pair in enumerate(pool.map(measure, images.itertuples()), 1):
            records.extend(pair)
            if i % 30 == 0:
                print(f"{i}/127", flush=True)
    return pd.DataFrame(records)


def plot_detail():
    photos = gsd.list_images(work / "soil-data").set_index("file")
    fig, axes = plt.subplots(2, 2, figsize=(6, 6.8))
    for i, sample in enumerate(["H616", "H405"]):
        photo = photos.loc[f"Motorola_Edge_{sample}_01.jpg"]
        for ax, (arm, gray, ppm) in zip(axes[i], reversed(paired_images(photo))):
            side = round(20 * ppm)
            y, x = [(n-side)//2 for n in gray.shape]
            ax.imshow(gray[y:y+side,x:x+side], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{sample}   ·   {side} × {side} piksel")
            ax.set_axis_off()
    fig.suptitle("Aynı merkez bölgesi  ·  yaklaşık 20 × 20 mm", fontsize=14)
    fig.text(.27,.92,"4 piksel/mm",ha="center")
    fig.text(.75,.92,"Özgün JPEG",ha="center")
    fig.subplots_adjust(top=.84, bottom=.03, wspace=.06, hspace=.20, left=.035, right=.965)
    fig.savefig(OUT / f"{STEM}_detay.png", dpi=150)
    plt.close(fig)


def evaluate(frame):
    assert len(frame) == 254 and not frame.duplicated(["file","kol"]).any()
    assert np.isfinite(frame[TEXTURE]).all().all()
    cached = pd.read_csv(repo / "cache/granulometry_train.csv").set_index("file").sort_index()
    native = frame.query("kol=='ozgun'").set_index("file").sort_index()
    pd.testing.assert_index_equal(native.index, cached.index)
    np.testing.assert_allclose(native.ppm, cached.eff_ppm, atol=1e-8, rtol=0)
    print("Max native feature difference:", np.max(np.abs(native[TEXTURE]-cached[TEXTURE]).values))
    old = pd.read_csv(OUT / "ince_oran_2mm_ridge_sonuclar.csv")
    key = ["exam", "sample_id"]
    base = old.query("model == 'referans'").set_index(key)
    result = base[["gercek_0063","tahmin_0063","mutlak_hata","emd"]].add_suffix("_mevcut")
    for arm, photos in frame.groupby("kol"):
        scores = evaluate_photos(photos).query("model == 'doku'").set_index(key)
        assert scores.index.equals(result.index)
        if arm == "ozgun":
            reference = old.query("model == 'doku'").set_index(key)
            diff = (scores.tahmin_0063-reference.tahmin_0063).abs().max()
            print("Max native prediction difference (pp):", diff)
            assert diff < .02
        result = result.join(scores[["oran_ham","ham_0063","tahmin_0063","mutlak_hata","emd"]].add_suffix("_"+arm))
    assert len(result) == 95 and result.notna().all().all()
    result.to_csv(OUT / f"{STEM}_sonuclar.csv", float_format="%.9f")
    print(result.filter(regex="^(emd_|mutlak_hata_)").groupby("exam").mean().round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    path = OUT / f"{STEM}_olculer.csv"
    if args.rebuild or not path.is_file():
        extract().to_csv(path, index=False, float_format="%.12g")
    evaluate(pd.read_csv(path))
    plot_detail()
