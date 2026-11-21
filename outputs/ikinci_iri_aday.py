"""Örtüşmeyen iki iri adayın ikincisini ölçer ve v3 ile eşleştirilmiş sınav yapar."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from v3_knn_esit_karisim import labels, published, sets, training, work
import features
import gsd


OUT = Path(__file__).resolve().parent
FEATURE = "ikinci_iri_aday"
PPM = 2
SIDE = 90 * PPM
YY, XX = np.indices((SIDE, SIDE))


def ring_kernel(diameter_mm):
    radius = diameter_mm * PPM / 2
    width = int(np.ceil(np.sqrt(2) * radius))
    y, x = np.ogrid[-width:width + 1, -width:width + 1]
    square = x*x + y*y
    disk = square <= radius**2
    ring = (square > radius**2) & (square <= 2 * radius**2)
    return (disk / disk.sum() - ring / ring.sum()).astype("float32")


KERNELS = [(d, ring_kernel(d)) for d in (20, 32, 50)]


def candidates(rgb):
    side = round(90 * features.CANON_PPM)
    y, x = (np.array(rgb.shape[:2]) - side) // 2
    assert min(y, x) >= 0
    patch = cv2.resize(rgb[y:y+side, x:x+side], (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype("float32")
    gray = (gray - gray.mean()) / max(float(gray.std()), 1e-6)
    maps = []
    for diameter, kernel in KERNELS:
        response = np.abs(cv2.filter2D(gray, -1, kernel))
        margin = kernel.shape[0] // 2
        valid = (XX >= margin) & (XX < SIDE-margin) & (YY >= margin) & (YY < SIDE-margin)
        maps.append(np.where(valid, response, -np.inf))
    found = []
    for _ in range(2):
        scale, y, x = np.unravel_index(np.argmax(maps), (len(KERNELS), SIDE, SIDE))
        score = maps[scale][y, x]
        if not np.isfinite(score):
            break
        radius = KERNELS[scale][0] * PPM / 2
        found.append((x, y, radius, float(score)))
        other_tile = (XX // (SIDE//3) != x // (SIDE//3)) | (YY // (SIDE//3) != y // (SIDE//3))
        for (diameter, _), response in zip(KERNELS, maps):
            apart = (XX - x)**2 + (YY - y)**2 >= (radius + diameter * PPM / 2)**2
            response[~(other_tile & apart)] = -np.inf
    return patch, found


def measure_photo(row):
    rgb, _ = features.load_canonical(row.path, row.phone)
    records = []
    for aug in range(9):
        image = features.augment(rgb, np.random.default_rng(aug * 1009 + row.Index)) if aug else rgb
        _, found = candidates(image)
        records.append((row.file, aug, found[1][3] if len(found) == 2 else 0.0))
    return records


def evaluate(train, test, ids, exam):
    columns = list(dict.fromkeys(sets["R1"] + sets["R2"] + [FEATURE]))
    train_mean = train.groupby("sample_id")[columns].mean().loc[ids]
    test_mean = test.groupby("sample_id")[columns].mean().loc[ids]
    assert test.aug.eq(0).all()
    quantiles = pd.DataFrame(gsd.curve_to_logq(labels.loc[ids]), index=ids)
    rows = []
    baseline_terms = []
    for sample in ids:
        pool = ids.drop(sample)
        true = labels.loc[sample].to_numpy()
        row = dict(exam=exam, sample_id=sample, lab_over20=100-labels.loc[sample, "20"],
                   feature=test_mean.loc[sample, FEATURE])
        for model, extra in [("v3", []), ("ekli", [FEATURE])]:
            predictions = []
            for cols in sets.values():
                cols = cols + extra
                reg = published.Component(cols, "pls").fit(train_mean.loc[pool, cols].to_numpy(),
                                                           quantiles.loc[pool].to_numpy())
                predictions.append(reg.predict(test_mean.loc[[sample], cols].to_numpy())[0])
            curve = gsd.logq_to_curve(np.mean(predictions, axis=0))[0]
            assert np.isfinite(curve).all() and np.all(np.diff(curve) >= 0) and curve[-1] == 100
            row[f"{model}_emd"] = gsd.emd(true, curve)[0]
            if exam == "24" and model == "v3":
                baseline_terms.append(np.abs(true[:-1] - curve[:-1]) * gsd.DLOG)
            tail = 100 - curve[gsd.COLS.index("20")]
            row[f"{model}_over20"] = tail
            row[f"{model}_tail_abs"] = abs(tail - row["lab_over20"])
        rows.append(row)
    if baseline_terms:
        averages = pd.Series(np.mean(baseline_terms, axis=0), index=gsd.COLS[:-1])
        np.testing.assert_allclose(averages.sum(), np.mean([r["v3_emd"] for r in rows]))
        print("24 toprakta ortalama v3 EMD, elek bazında:", averages.round(4).to_dict())
    return pd.DataFrame(rows)


def plot_examples(images):
    chosen = images[images.sample_id.isin(["H374", "H615"])].groupby("sample_id").first()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3))
    for (sample, row), ax in zip(chosen.iterrows(), axes):
        rgb, _ = features.load_canonical(row.path, row.phone)
        patch, found = candidates(rgb)
        ax.imshow(patch)
        for x, y, radius, _ in found[:1]:
            ax.add_patch(Circle((x, y), radius, ec="white", fc="none", lw=2))
        for x, y, radius, _ in found[1:]:
            ax.add_patch(Circle((x, y), radius, ec="#ff922b", fc="none", lw=2))
        ax.set_title(sample)
        ax.set_axis_off()
    fig.suptitle("90 × 90 mm kesitte iki ayrı kontrast bölgesi")
    fig.text(.5, .045, "Beyaz: en güçlü aday     Turuncu: modele verilen ikinci aday", ha="center")
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(OUT / "ikinci_iri_aday_ornekleri.png", dpi=140)
    plt.close(fig)


def main():
    cv2.setNumThreads(1)
    imgs = gsd.list_images(str(work / "soil-data")).query("split == 'train'")
    assert len(imgs) == 127 and len(labels) == 24
    with ThreadPoolExecutor(max_workers=4) as workers:
        records = [record for group in workers.map(measure_photo, imgs.itertuples()) for record in group]
    measurements = pd.DataFrame(records, columns=["file", "aug", FEATURE])
    assert len(measurements) == len(training) == 127 * 9
    data = training.merge(measurements, on=["file","aug"], how="left", validate="one_to_one")
    assert np.isfinite(data[FEATURE]).all() and data[FEATURE].ge(0).all()
    originals = data[data.aug == 0]
    motorola, samsung = "Motorola_Edge", "Samsung_A52"
    common = labels.index.intersection(originals[originals.phone == motorola].sample_id).intersection(
        originals[originals.phone == samsung].sample_id)
    assert len(common) == 21
    blocks = [evaluate(data, originals, labels.index, "24")]
    for src, dst in [(motorola, samsung), (samsung, motorola)]:
        blocks.append(evaluate(data[data.phone == src], originals[originals.phone == dst],
                               common, src + " -> " + dst))
    result = pd.concat(blocks, ignore_index=True)
    np.testing.assert_allclose([b.v3_emd.mean() for b in blocks], [38.758, 41.987, 42.831],
                               rtol=0, atol=.0006)
    result.to_csv(OUT / "ikinci_iri_aday_sonuclari.csv", index=False, float_format="%.8f")
    plot_examples(imgs)
    for name, group in list(result.groupby("exam")) + [("H374 haric", blocks[0].query("sample_id != 'H374'"))]:
        print(name, group[["v3_emd","ekli_emd","v3_tail_abs","ekli_tail_abs"]].mean().round(5).to_dict(),
              "wins", (group.ekli_emd < group.v3_emd).sum(), (group.ekli_tail_abs < group.v3_tail_abs).sum(), flush=True)
    print(blocks[0].query("sample_id in ['H374', 'H615']").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
