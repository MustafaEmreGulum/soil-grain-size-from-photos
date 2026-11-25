"""Dondurulmuş 20 mm alan düzeltmesini seçilmiş 12 Montréal toprağında ölç."""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops_table
from threadpoolctl import threadpool_limits

from dinov2_hiyerarsik import landscape, soil_box
from imagegrains_pilot import PPM, load_model
from montreal_dis_sinama import PHYSICAL_CAMERA, features, gsd
from v3_knn_esit_karisim import cache, keys, labels, out, published, training, work

NATIVE_PPM = 1000 / 39.4
PHOTOS = work / "soil-data" / "montreal_20mm"
MASKS = PHOTOS / "masks"


def crop_box(photo):
    # The Montréal trays have pale rounded corners. Keep a fixed clearance on
    # all photos in addition to the original soil-box detector.
    w, h = photo.size
    x, y, r, b = soil_box(photo)
    return max(x, round(.05*w)), max(y, round(.05*h)), min(r, round(.95*w)), min(b, round(.95*h))


def v3(query):
    targets = pd.DataFrame(gsd.curve_to_logq(labels.to_numpy()), index=labels.index)
    idx, _, curves = published.fit_predict(training, query, targets, keys)
    return pd.DataFrame(curves, index=idx, columns=gsd.COLS).round(4)


def update20(raw, area, factor):
    return (100 - area*factor).clip(lower=raw["6.3"], upper=raw["63"]).round(4).where(area.gt(0), raw["20"])


def bound_error(raw, retained):
    """Smallest error possible by moving only the 20 mm point inside the original bounds."""
    return (retained-(100-raw["6.3"])).clip(lower=0) + ((100-raw["63"])-retained).clip(lower=0)


def calibration():
    areas = pd.read_csv(out / "v3_tam_yuzey_20mm_fotograflar.csv").groupby("sample_id").alan_pct.mean()
    assert len(areas) == 24 and set(areas.index) == set(labels.index)
    factor = max(0., (areas @ (100-labels.loc[areas.index, "20"])) / (areas @ areas))
    base = pd.read_csv(work / "v3_puan_kontrol.csv", index_col="sample_id")
    kaggle = v3(cache.query("split == 'test' and aug == 0")).loc[base.index]
    np.testing.assert_allclose(kaggle, base, atol=.00011, rtol=0)
    test = pd.read_csv(out / "v3_20mm_test_karsilastirma.csv", index_col="sample_id").loc[base.index]
    ref = pd.read_csv(out / "submission_v3_63mm_kontrollu.csv", index_col="sample_id").loc[base.index]
    np.testing.assert_allclose(update20(base, test.ortalama_alan_pct, factor), ref["20"], atol=.00001, rtol=0)
    return factor


def segment(model, photo, rect, name, ppm):
    roi = photo.crop(rect)
    size = tuple(round(n * ppm / NATIVE_PPM) for n in roi.size)
    path = MASKS / (name + "_" + "_".join(map(str, rect)) + f"_{ppm}.npz")
    if not path.exists():
        rgb = np.asarray(roi.resize(size, Image.Resampling.LANCZOS))
        mask, *_ = model.eval(rgb, batch_size=1, bsize=256, flow_threshold=.4, cellprob_threshold=0.)
        assert mask.min() >= 0 and mask.max() <= 65535 and mask.shape == size[::-1]
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, mask=mask.astype(np.uint16))
        tmp.replace(path)
    with np.load(path) as data:
        mask = data["mask"]
    assert mask.shape == size[::-1]
    props = regionprops_table(mask, properties=("axis_minor_length", "area"))
    diam = props["axis_minor_length"] / ppm
    large = diam > 20
    return diam.max(initial=0.), int(large.sum()), 100 * props["area"][large].sum() / mask.size


def render(result):
    sheet = Image.new("RGB", (1080, 4*285), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("arial.ttf", 15)
    for i, (soil, row) in enumerate(result.iterrows()):
        photo = landscape(PHOTOS / row.file)
        rect = tuple(int(row[k]) for k in ("left", "top", "right", "bottom"))
        name = Path(row.file).stem + "_" + "_".join(map(str, rect)) + f"_{PPM}.npz"
        with np.load(MASKS / name) as data:
            mask = data["mask"]
        props = regionprops_table(mask, properties=("label", "axis_minor_length"))
        ids = props["label"][props["axis_minor_length"] > 20*PPM]
        assert len(ids) == row.n20
        x, y = (i%3)*360, (i//3)*285
        sheet.paste(photo.resize((360, 240), Image.Resampling.BOX), (x, y))
        left, top, right, bottom = rect
        sx, sy = 360/photo.width, 240/photo.height
        draw.rectangle((x+left*sx, y+top*sy, x+right*sx, y+bottom*sy), outline="cyan", width=1)
        for label in ids:
            contours, _ = cv2.findContours((mask == label).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = [(x+(left + u*(right-left)/mask.shape[1])*sx,
                           y+(top + v*(bottom-top)/mask.shape[0])*sy) for u, v in contour[:, 0]]
                if len(points) > 2:
                    draw.line(points+[points[0]], fill="yellow", width=2)
        draw.text((x+3, y+243), f"{soil}   |   20 mm üstü aday: {int(row.n20)}", font=font, fill="black")
        draw.text((x+3, y+262), f"Laboratuvar %{row.actual_retained:.1f}    Tahmin %{row.corrected_retained:.1f}",
                  font=font, fill="black")
    sheet.save(out / "montreal_20mm_kadrajlar.jpg", quality=93)


def main():
    picked = pd.read_csv(out / "montreal_20mm_pilot_secimi.csv", dtype={"sample_id": str}).set_index("sample_id")
    assert picked.index.is_unique and len(picked) == 12
    for prior in ("montreal_pilot_referans.csv", "montreal_egitim_secimi.csv"):
        seen = pd.read_csv(out / prior, dtype={"sample_id": str}).sample_id
        assert picked.index.intersection(seen).empty
    metadata = pd.read_csv(PHOTOS.parent / "psd_mass_path_80_to_0_08mm_no_ballast.csv",
                           dtype={"sample_id": str}).set_index("image_path")
    chosen = metadata.loc[picked.image_path]
    assert chosen.sample_id.to_list() == picked.index.to_list() and chosen.dry_or_moist.eq("dry").all()
    np.testing.assert_allclose(chosen["%_20mm"], picked["%_20mm"], atol=1e-10, rtol=0)
    MASKS.mkdir(exist_ok=True, parents=True)
    paths = picked.image_path.map(lambda name: PHOTOS / Path(name).name)
    cv2.setNumThreads(1)
    with threadpool_limits(1), patch.dict(gsd.PHONES, {PHYSICAL_CAMERA: (8192, NATIVE_PPM)}):
        factor = calibration()
        rows = []
        for soil, path in paths.items():
            assert path.name.startswith(soil + "_")
            values = features.image_features(path, PHYSICAL_CAMERA)
            rows.append(dict(sample_id=soil, **{k: values[k] for k in keys}))
        raw = v3(pd.DataFrame(rows)).loc[picked.index]
    model = load_model(out.parent / ".venv" / "IG2_full_set_cp_SAM")
    print("Device", model.device, "factor", factor, flush=True)
    records = []
    for i, (soil, path) in enumerate(paths.items()):
        photo = landscape(path)
        assert photo.size == (8192, 5464)
        rect = crop_box(photo)
        peak, count, area = segment(model, photo, rect, path.stem, PPM)
        raw63 = raw.at[soil, "63"]
        peak63 = segment(model, photo, rect, path.stem, PPM/4)[0] if raw63 < 100 else np.nan
        records.append(dict(sample_id=soil, file=path.name, left=rect[0], top=rect[1], right=rect[2],
                            bottom=rect[3], n20=count, area20_pct=area, max_mm=peak, coarse_max_mm=peak63,
                            pass63=100 if raw63 == 100 or peak63 <= 63 else raw63))
        print(i+1, soil, count, round(area, 3), flush=True)
    result = pd.DataFrame(records).set_index("sample_id")
    result["area_factor"] = factor
    result["raw_pass6_3"] = raw["6.3"]
    result["raw_pass63"] = raw["63"]
    result["v3_retained"] = 100-raw["20"]
    result["area_estimate_before_bounds"] = (factor*result.area20_pct).where(result.n20.gt(0))
    result["corrected_retained"] = 100-update20(raw, result.area20_pct, factor)
    result["actual_retained"] = picked.iri_pay_pct
    result["min_error_with_original_bounds"] = bound_error(raw, result.actual_retained)
    np.testing.assert_allclose(result.actual_retained, 100-picked["%_20mm"], atol=.0001, rtol=0)
    assert (100-result.corrected_retained <= result.pass63).all()
    result["group"] = np.select([result.actual_retained == 0, result.actual_retained < 10], ["zero", "low"], default="high")
    assert result.groupby("group").size().eq(4).all()
    for name in ("v3", "corrected"):
        result[name+"_error"] = result[name+"_retained"] - result.actual_retained
    result.to_csv(out / "montreal_20mm_sonuclar.csv", float_format="%.6f")
    render(result)
    print(result.groupby("group")[["v3_error", "corrected_error"]].agg(["mean", lambda x: x.abs().mean()]))


if __name__ == "__main__":
    main()
