"""Üç fiziksel dalga boyunda ortalama enerjinin ve bölgesel farkın katkısını sınar."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import fft

from v3_knn_esit_karisim import labels, published, sets, training, work
import features
import gsd


OUT = Path(__file__).resolve().parent
WAVELENGTHS = (1, 2, 4)
MEANS = [f"mean_{w}mm" for w in WAVELENGTHS]
SPREADS = [f"p90_minus_median_{w}mm" for w in WAVELENGTHS]
MODELS = {"v3": [], "ortalama": MEANS, "bolgesel": SPREADS, "birlikte": MEANS + SPREADS}
FINE = ("0.063", "0.2", "0.63")
SIDE = int(90 * features.CANON_PPM)
PAD = int(3 * .75 * max(WAVELENGTHS) * features.CANON_PPM)


def morlet_filters():
    freq = fft.fftfreq(SIDE + 2 * PAD)
    fx, fy = freq[None, :], freq[:, None]
    bank = []
    for mm in WAVELENGTHS:
        wavelength = mm * features.CANON_PPM
        factor = 2 * np.pi**2 * (.75 * wavelength)**2
        dc = np.exp(-factor / wavelength**2) * np.exp(-factor * (fx**2 + fy**2))
        orientations = []
        for theta in np.deg2rad([0, 45, 90, 135]):
            distance = (fx - np.cos(theta) / wavelength)**2 + (fy - np.sin(theta) / wavelength)**2
            orientations.append((np.exp(-factor * distance) - dc).astype("float32"))
        bank.append(orientations)
    return bank


BANK = morlet_filters()


def wavelet_features(rgb):
    y, x = (np.asarray(rgb.shape[:2]) - SIDE) // 2
    assert min(y, x) >= 0
    patch = rgb[y:y + SIDE, x:x + SIDE]
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype("float32")
    gray = (gray - gray.mean()) / max(float(gray.std()), 1e-6)
    spectrum = fft.fft2(np.pad(gray, PAD, mode="reflect"))
    means, spreads = [], []
    for orientations in BANK:
        energy = np.zeros((SIDE, SIDE), dtype="float32")
        for kernel in orientations:
            response = fft.ifft2(spectrum * kernel)[PAD:PAD + SIDE, PAD:PAD + SIDE]
            energy += response.real**2 + response.imag**2
        energy /= len(orientations)
        tiles = energy.reshape(4, SIDE // 4, 4, SIDE // 4).mean(axis=(1, 3), dtype="float64")
        means.append(tiles.mean())
        spreads.append(np.quantile(tiles, .90) - np.median(tiles))
    return means + spreads


def measure_photo(row):
    rgb, _ = features.load_canonical(row.path, row.phone)
    records = []
    for aug in range(9):
        image = features.augment(rgb, np.random.default_rng(aug * 1009 + row.Index)) if aug else rgb
        records.append([row.file, aug, *wavelet_features(image)])
    return records


def evaluate(refs, queries, ids, exam):
    all_cols = list(dict.fromkeys(sets["R1"] + sets["R2"] + MEANS + SPREADS))
    train = refs.groupby("sample_id")[all_cols].mean().loc[ids]
    test = queries.groupby("sample_id")[all_cols].mean().loc[ids]
    assert queries.aug.eq(0).all()
    quantiles = pd.DataFrame(gsd.curve_to_logq(labels.loc[ids]), index=ids)
    results = []
    for sample in ids:
        pool = ids.drop(sample)
        true = labels.loc[sample].to_numpy()
        for model, extra in MODELS.items():
            components = []
            for base in sets.values():
                cols = base + extra
                reg = published.Component(cols, "pls").fit(train.loc[pool, cols].to_numpy(),
                                                          quantiles.loc[pool].to_numpy())
                components.append(reg.predict(test.loc[[sample], cols].to_numpy())[0])
            curve = gsd.logq_to_curve(np.mean(components, axis=0))[0]
            assert np.isfinite(curve).all() and 0 <= curve[0] and curve[-1] == 100 and (np.diff(curve) >= 0).all()
            row = dict(exam=exam, sample_id=sample, model=model, emd=gsd.emd(true, curve)[0])
            for i, sieve in enumerate(gsd.COLS):
                row[f"pred_{sieve}"] = curve[i]
                if sieve in FINE:
                    row[f"signed_{sieve}"] = curve[i] - true[i]
            results.append(row)
    return pd.DataFrame(results)


def plot_paired_gain(result):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scores = result[result.exam.astype(str) == "24"].pivot(index="sample_id", columns="model", values="emd")
    gain = (scores.ortalama - scores.birlikte).sort_values()
    colors = ["#b05b20" if s in ("H037", "H038") else "#246b77" for s in gain.index]
    fig, ax = plt.subplots(figsize=(8, 7.2))
    y = np.arange(len(gain))
    ax.axvline(0, color="#444444", lw=1)
    ax.hlines(y, 0, gain, colors=colors, lw=2)
    ax.scatter(gain, y, c=colors, s=27, zorder=3)
    ax.set_yticks(y, gain.index)
    ax.set_xlabel("EMD kazancı  ·  Pozitif değer daha düşük hata")
    ax.set_title("Bölgesel farkı eklemek her toprağı nasıl etkiledi?", loc="left", pad=14)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], color="#b05b20", marker="o", ls="none",
                              label="Fikre esin veren H037 ve H038")], loc="upper center",
              bbox_to_anchor=(.5, -.085), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "dalgacik_eslestirilmis_kazanc.png", dpi=145)
    plt.close(fig)


def main():
    cv2.setNumThreads(1)
    imgs = gsd.list_images(str(work / "soil-data")).query("split == 'train'")
    assert len(imgs) == 127 and len(labels) == 24
    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = []
        for n, records in enumerate(workers.map(measure_photo, imgs.itertuples()), 1):
            rows.extend(records)
            if n % 32 == 0:
                print(f"{n}/{len(imgs)} original photos and their augmentations", flush=True)
    measured = pd.DataFrame(rows, columns=["file", "aug"] + MEANS + SPREADS)
    data = training.merge(measured, on=["file", "aug"], how="left", validate="one_to_one")
    assert len(data) == len(measured) == 127 * 9
    assert np.isfinite(data[MEANS + SPREADS]).all().all() and data[MEANS + SPREADS].ge(0).all().all()
    originals = data[data.aug == 0]
    moto, sam = "Motorola_Edge", "Samsung_A52"
    common = labels.index.intersection(originals[originals.phone == moto].sample_id).intersection(
        originals[originals.phone == sam].sample_id)
    assert len(common) == 21
    blocks = [evaluate(data, originals, labels.index, "24")]
    for src, dst in [(moto, sam), (sam, moto)]:
        blocks.append(evaluate(data[data.phone == src], originals[originals.phone == dst], common, src + " -> " + dst))
    result = pd.concat(blocks, ignore_index=True)
    np.testing.assert_allclose([b.query("model == 'v3'").emd.mean() for b in blocks],
                               [38.75755, 41.98687, 42.83059], rtol=0, atol=.00001)
    assert len(result) == (24 + 2 * 21) * 4 and not result.duplicated(["exam","sample_id","model"]).any()
    result.to_csv(OUT / "dalgacik_sonuclari.csv", index=False, float_format="%.9f")
    plot_paired_gain(result)
    views = [*blocks, blocks[0][~blocks[0].sample_id.isin(["H037", "H038"])]]
    for block in views:
        scores = block.pivot(index="sample_id", columns="model", values="emd")
        gain = scores.ortalama - scores.birlikte
        print(block.exam.iloc[0], "n", len(scores), scores.mean().round(5).to_dict(),
              "primary gain mean / median / wins", gain.mean(), gain.median(), (gain > 0).sum())


if __name__ == "__main__":
    main()
