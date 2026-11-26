"""İri taş alanının kütleye dönüşümünü iki boyut aralığında denetle."""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops_table
from threadpoolctl import threadpool_limits

from montreal_20mm_sinama import (
    MASKS, NATIVE_PPM, PHOTOS, PHYSICAL_CAMERA, PPM, calibration, crop_box, features,
    gsd, keys, landscape, load_model, out, segment, update20, v3,
)
from orta_elek_kismi_duzeltme import BANDS, band_curve
from v3_knn_esit_karisim import labels


def inputs():
    def load(name):
        return pd.read_csv(out / name, dtype={"sample_id": str}).set_index("sample_id")

    old = load("montreal_20mm_sonuclar.csv")
    used = old.index.union(load("montreal_egitim_secimi.csv").index).union(load("montreal_pilot_referans.csv").index)
    meta = pd.read_csv(PHOTOS.parent / "psd_mass_path_80_to_0_08mm_no_ballast.csv", dtype={"sample_id": str})
    dry = meta.loc[meta.dry_or_moist.eq("dry")].sort_values("image_path")
    first = dry.drop_duplicates("sample_id").set_index("sample_id").sort_index()
    pool = first.loc[~first.index.isin(used) & first["%_20mm"].le(85)]
    rng = np.random.default_rng(42)
    ids = []
    for kind in ("bulk", "splitspoon"):
        group = pool.loc[pool.image_path.str.contains(f"_{kind}_")]
        ids.extend(rng.choice(group.index, 2, replace=False))
    assert ids == ["0027", "0132", "0196", "0201"]
    natural = pool.loc[pool.image_path.str.contains(r"_(?:bulk|splitspoon)_")].drop(ids)
    stress = natural["%_20mm"].idxmin()
    assert stress == "0309"
    chosen = first.loc[[*ids, stress, *old.index]].copy()
    chosen["file"] = chosen.image_path.map(lambda s: Path(s).name)
    assert chosen.index.is_unique and chosen.loc[old.index, "file"].equals(old.file)
    chosen["new"] = chosen.index.isin(ids)
    chosen["stress"] = chosen.index == stress
    return chosen, old


def measure(chosen):
    model = None
    rows = []
    for soil, row in chosen.iterrows():
        photo = landscape(PHOTOS / row.file)
        assert photo.size == (8192, 5464)
        rect = crop_box(photo)
        stem = Path(row.file).stem
        path = MASKS / (stem + "_" + "_".join(map(str, rect)) + f"_{PPM}.npz")
        if not path.exists():
            if model is None:
                model = load_model(out.parent / ".venv" / "IG2_full_set_cp_SAM")
            segment(model, photo, rect, stem, PPM)
        with np.load(path) as data:
            mask = data["mask"]
        assert mask.shape == tuple(round((rect[j+2]-rect[j])*PPM/NATIVE_PPM) for j in (1, 0))
        props = regionprops_table(mask, properties=("axis_minor_length", "area"))
        size = props["axis_minor_length"]/PPM
        areas = {"a_"+s: 100*props["area"][size > float(s)].sum()/mask.size for s in BANDS}
        rows.append(dict(sample_id=soil, **areas))
        print(soil, round(areas["a_20"], 3), flush=True)
    return pd.DataFrame(rows).set_index("sample_id")


def middle_mass(curves, soil_areas, ref_areas):
    values = []
    for soil, curve in curves.iterrows():
        adjusted = band_curve(curve.to_numpy(), ref_areas, soil_areas.loc[soil])
        assert np.all(np.diff(adjusted) >= -1e-8)
        values.append(adjusted[gsd.COLS.index("20")]-adjusted[gsd.COLS.index("6.3")])
    return pd.Series(values, index=curves.index)


def plot_new(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fresh = result[result.new]
    y = np.arange(len(fresh))
    fig, (coarse, mid) = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    fig.subplots_adjust(left=.19, right=.98, top=.82, bottom=.30, wspace=.12)
    lab, area, model = "#167556", "#b26318", "#254d9b"
    coarse.scatter(fresh.mass20_lab, y-.14, color=lab, marker="D", label="Laboratuvar", zorder=3)
    coarse.scatter(fresh.mass20_from_area, y, color=area, marker="^", label="Alanın verdiği ilk tahmin", zorder=3)
    coarse.scatter(fresh.mass20_model, y+.14, color=model, label="Son tahmin", zorder=3)
    mid.hlines(y, fresh.mass6_3_20_lab_low, fresh.mass6_3_20_lab_high, color=lab, lw=5)
    mid.scatter(fresh.mass6_3_20_model, y, facecolors="white", edgecolors=model, linewidths=2, zorder=3)
    coarse.set_yticks(y, [f"{s}  (alan %{a:.1f})".replace(".", ",") for s, a in fresh.a_20.items()])
    coarse.invert_yaxis()
    coarse.set_title("20 mm üstü kütle", loc="left")
    mid.set_title("6,3–20 mm arasındaki kütle", loc="left")
    for ax in (coarse, mid):
        ax.set_xlabel("Numune kütlesindeki pay (%)")
        ax.set_xlim(left=-1)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.legend(*coarse.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(.015,.078),
               frameon=False, ncol=3)
    fig.text(.025, .04, "Sağda yeşil çizgi, 5 ve 10 mm ölçümlerinin izin verdiği aralığı, mavi halka modelin tahminini gösterir.")
    fig.savefig(out / "iri_alan_iki_aralik.png", dpi=165, facecolor="white")
    plt.close(fig)


def show_masks(result):
    sheet = Image.new("RGB", (820, 905), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("arial.ttf", 16)
    for i, (soil, row) in enumerate(result[result.new | result.stress].iterrows()):
        photo = landscape(PHOTOS / row.file)
        rect = crop_box(photo)
        key = Path(row.file).stem+"_"+"_".join(map(str, rect))+f"_{PPM}.npz"
        with np.load(MASKS / key) as data:
            mask = data["mask"]
        props = regionprops_table(mask, properties=("label", "axis_minor_length"))
        rgb = np.asarray(photo.crop(rect).resize((mask.shape[1], mask.shape[0]), Image.Resampling.LANCZOS)).copy()
        for label, diam in zip(props["label"], props["axis_minor_length"]/PPM):
            if diam <= 6.3:
                continue
            contours, _ = cv2.findContours((mask == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (255, 197, 0) if diam > 20 else (37, 219, 236), 4)
        tile = Image.fromarray(rgb)
        tile.thumbnail((400, 263), Image.Resampling.LANCZOS)
        x, y = 5 + (i%2)*410, (i//2)*290
        sheet.paste(tile, (x, y))
        kind = " (uç durum)" if row.stress else ""
        draw.text((x, y+263), f"{soil}{kind}    20 mm üstü görünür alan %{row.a_20:.1f}".replace(".", ","),
                  font=font, fill="black")
    draw.text((5, 876), "Sarı çizgiler 20 mm üstü, turkuaz çizgiler 6,3–20 mm kısa eksenli adayları gösteriyor.", font=font, fill="black")
    sheet.save(out / "iri_alan_iki_aralik_maskeler.jpg", quality=92)


def main():
    chosen, old = inputs()
    areas = measure(chosen)
    cv2.setNumThreads(1)
    with threadpool_limits(1), patch.dict(gsd.PHONES, {PHYSICAL_CAMERA: (8192, NATIVE_PPM)}):
        factor = calibration()
        rows = []
        for soil, name in chosen.file.items():
            values = features.image_features(PHOTOS / name, PHYSICAL_CAMERA)
            rows.append(dict(sample_id=soil, **{k: values[k] for k in keys}))
        raw = v3(pd.DataFrame(rows)).loc[chosen.index]
    all_areas = pd.read_csv(out / "maske_elek_alanlari.csv").groupby(["split", "sample_id"])[
        ["a_"+s for s in BANDS]].mean()
    ref = all_areas.loc["train"]
    archive = pd.read_csv(out / "submission_v3_63mm_kontrollu.csv", index_col="sample_id")
    baseline = pd.read_csv(out / "submission_v3_tam_yuzey_20mm.csv", index_col="sample_id")
    np.testing.assert_allclose(middle_mass(baseline, all_areas.loc["test"], ref),
                               (archive["20"]-archive["6.3"]).loc[baseline.index], atol=1.1e-6, rtol=0)
    h374_train = ref.drop("H374").a_20
    cutoff = h374_train.max()
    reduced_factor = (h374_train @ (100-labels.loc[h374_train.index, "20"])) / (h374_train @ h374_train)
    base = raw.copy()
    base["20"] = update20(raw, areas.a_20, factor)
    result = chosen[["file", "new", "stress"]].join(areas)
    result["above_H374_fold_range"] = areas.a_20.gt(cutoff)
    result["above_full_range"] = areas.a_20.gt(ref.a_20.max())
    result["mass20_lab"] = 100-chosen["%_20mm"]
    result["sample_mass_g"] = chosen["sample_mass (g)"]
    result["coarse_mass_g_from_lab_percent"] = result.sample_mass_g*result.mass20_lab/100
    result["mass20_from_area"] = factor*areas.a_20
    result["mass20_from_H374_fold_factor"] = reduced_factor*areas.a_20
    result["mass20_model"] = 100-base["20"]
    result["mass6_3_20_before_band"] = base["20"]-base["6.3"]
    result["mass6_3_20_model"] = middle_mass(base, areas, ref)
    # P5 <= P6.3 <= P10. Both endpoints use actual measured sieve passages.
    assert (chosen["%_5mm"] <= chosen["%_10mm"]).all() and (chosen["%_10mm"] <= chosen["%_20mm"]).all()
    result["mass6_3_20_lab_low"] = chosen["%_20mm"]-chosen["%_10mm"]
    result["mass6_3_20_lab_high"] = chosen["%_20mm"]-chosen["%_5mm"]
    mid = result.mass6_3_20_model
    result["mid_error_to_interval"] = mid-mid.clip(lower=result.mass6_3_20_lab_low,
                                                   upper=result.mass6_3_20_lab_high)
    np.testing.assert_allclose(result.loc[old.index, "a_20"], old.area20_pct, atol=1e-6, rtol=0)
    np.testing.assert_allclose(result.loc[old.index, "mass20_model"], old.corrected_retained, atol=1e-6, rtol=0)
    np.testing.assert_allclose(result.loc[old.index, "mass20_lab"], old.actual_retained, atol=1e-6, rtol=0)
    assert result.notna().all().all() and (mid + result.mass20_model <= 100+1e-8).all()
    result.to_csv(out / "iri_alan_iki_aralik_sonuclar.csv", float_format="%.6f")
    plot_new(result)
    show_masks(result)
    print("Calibration", factor, reduced_factor, "area thresholds", cutoff, ref.a_20.max())
    print(result.loc[result.new | result.stress | result.above_H374_fold_range, [
        "new", "a_20", "mass20_lab", "mass20_from_area", "mass20_model", "mass6_3_20_model",
        "mass6_3_20_lab_low", "mass6_3_20_lab_high", "mid_error_to_interval"]].round(2).to_string())


if __name__ == "__main__":
    main()
