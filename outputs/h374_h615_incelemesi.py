"""Aynı 19 referansla H374/H615 çekimlerini, kadrajı ve v3 ölçülerini karşılaştır."""
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iri_toprak_sinavi import d50
from v3_knn_esit_karisim import columns, labels, original, published, training, work
import features
import gsd


SAMPLES = ["H374", "H615"]
SIDE_MM = 90
OUT = Path(__file__).resolve().parent


def predict_curves(models, frame):
    quantiles = np.mean([m.predict(frame[m.cols].to_numpy()) for m in models], axis=0)
    return gsd.logq_to_curve(quantiles)


def contributions(models, means):
    # Modelin iç log10(D50) çıktısı, kırpılmış standart özelliklerde doğrusaldır.
    j = np.flatnonzero(gsd.P_GRID == 50).item()
    parts = []
    gap = 0.0
    for model in models:
        x = means.loc[SAMPLES, model.cols].to_numpy()
        z = np.clip(model.sc.transform(x), -published.CLIP_Z, published.CLIP_Z)
        weight = model.m.coef_[j]
        parts.append(pd.Series((z[1] - z[0]) * weight / len(models), index=model.cols))
        gap += np.diff(model.predict(x)[:, j]).item() / len(models)
    terms = pd.concat(parts).groupby(level=0).sum()
    np.testing.assert_allclose(terms.sum(), gap, atol=1e-10)
    return terms


def plot_photos(photos, patches):
    rows = [
        ("H374", "Motorola_Edge_60_fusion", "Motorola Edge 60 Fusion"),
        ("H615", "Motorola_Edge", "Motorola Edge"),
        ("H615", "Samsung_A52", "Samsung A52"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.6))
    fig.subplots_adjust(left=.23, right=.97, top=.87, bottom=.085, wspace=.05, hspace=.08)
    fig.suptitle("İki toprağın aynı ölçekte sekiz çekimi", y=.976, fontsize=19)
    for col, ax in enumerate(axes[0]):
        ax.set_title(f"Çekim {col+1}", fontsize=11, pad=9)
    for (sample, phone, display), row in zip(rows, axes):
        subset = photos[(photos.sample_id == sample) & (photos.phone == phone)]
        for ax in row:
            ax.set_axis_off()
        for ax, filename in zip(row, subset.index):
            ax.imshow(patches[filename], interpolation="nearest")
        pos = row[0].get_position()
        fig.text(.215, pos.y0 + pos.height/2, f"{sample}\n{display}", ha="right", va="center", fontsize=10)
    ax = axes[1, 2]
    ax.set_box_aspect(1)
    ax.plot([.25, .25 + 20/SIDE_MM], [.50, .50], color="#273034", lw=3, transform=ax.transAxes)
    ax.text(.25, .39, "20 mm", fontsize=10, transform=ax.transAxes)
    fig.text(.5, .04, "Her kare fotoğrafın merkezinden 90 × 90 mm gösterir.", ha="center", fontsize=11)
    fig.savefig(OUT / "h374_h615_ayni_olcek.png", dpi=140)
    plt.close(fig)


def main():
    pool = labels.index.drop(labels.index[np.argsort(-d50(labels))[:5]])
    assert len(pool) == 19 and not set(SAMPLES).intersection(pool)
    refs = training[training.sample_id.isin(pool)].groupby("sample_id")[columns].mean().loc[pool]
    models = [m.fit(refs[m.cols].to_numpy(), gsd.curve_to_logq(labels.loc[pool].to_numpy()))
              for _, m in published.build_components(columns)]
    photos = original[original.sample_id.isin(SAMPLES)].set_index("file").sort_index()
    sources = gsd.list_images(str(work / "soil-data")).set_index("file")
    cropped = photos.copy()
    reconstructed = photos.copy()
    patches = {}
    side = int(SIDE_MM * features.CANON_PPM)
    for filename, row in photos.iterrows():
        path = sources.loc[filename, "path"]
        rgb, ppm = features.load_canonical(path, row.phone)
        y, x = (np.asarray(rgb.shape[:2]) - side) // 2
        assert min(x, y) >= 0
        square = rgb[y:y + side, x:x + side]
        patches[filename] = square
        with patch.object(features, "load_canonical", return_value=(rgb, ppm)):
            baseline = features.image_features(path, row.phone)
        reconstructed.loc[filename, columns] = [baseline[c] for c in columns]
        # Aynı v3 çıkarıcısını kullan, sadece ona verilen fiziksel alanı değiştir.
        with patch.object(features, "load_canonical", return_value=(square, ppm)):
            measurements = features.image_features(path, row.phone)
        cropped.loc[filename, columns] = [measurements[c] for c in columns]
    assert photos.groupby("sample_id").size().to_dict() == {"H374": 3, "H615": 5}
    rebuilt = d50(predict_curves(models, reconstructed))
    cached = d50(predict_curves(models, photos))
    print("Re-extracted vs cached photo D50, max absolute difference (mm):", abs(rebuilt - cached).max())
    np.testing.assert_allclose(rebuilt, cached, rtol=.01, atol=.01)

    shots = photos[["sample_id", "phone"]].copy()
    for mode, frame in [("original", photos), ("90mm", cropped)]:
        shots[mode + "_d50_mm"] = d50(predict_curves(models, frame))
        means = frame.groupby("sample_id")[columns].mean().loc[SAMPLES]
        curves = predict_curves(models, means)
        print(mode, "sample D50:", d50(curves), "EMD:", gsd.emd(labels.loc[SAMPLES], curves))
        if mode == "original":
            archived = pd.read_csv(OUT / "iri_toprak_sonuclari.csv").query("model == 'v3'").set_index("sample_id")
            np.testing.assert_allclose(d50(curves), archived.loc[SAMPLES, "d50_mm"], atol=1e-7, rtol=0)
            print("Predicted passing percentages:", pd.DataFrame(curves, index=SAMPLES, columns=gsd.COLS).round(2).to_string())
            terms = contributions(models, means)
            print("H615 minus H374, raw log10(D50) difference:", terms.sum())
            print("By feature family:", terms.groupby(terms.index.str.split("_").str[0]).sum().round(4).to_dict())
            print("Largest signed terms:", terms.reindex(terms.abs().nlargest(8).index).round(4).to_dict())
    print(shots.round(3).to_string())
    shots.to_csv(OUT / "h374_h615_cekimleri.csv", float_format="%.8f")
    plot_photos(photos, patches)


if __name__ == "__main__":
    main()
