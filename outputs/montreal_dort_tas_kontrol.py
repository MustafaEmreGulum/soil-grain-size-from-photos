"""Dört farklı toprakta birer taşın görünen sınırı ve 20 mm tahminindeki etkisi."""
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops_table
from threadpoolctl import threadpool_limits

from montreal_20mm_sinama import (MASKS, PHOTOS, NATIVE_PPM, PPM, PHYSICAL_CAMERA, calibration,
                                  crop_box, features, gsd, keys, landscape, out, update20, v3)

# Coordinates are in the original 4.5 px/mm measurement grid. Each polygon is
# local to its listed zoom. Traced on the unmarked image before opening the masks.
STONES = {
    "0132": ((222,36,575,496),
             [(32,191),(34,168),(40,145),(44,122),(55,106),(75,89),(95,78),(109,62),
              (123,48),(141,39),(155,34),(173,36),(190,45),(201,57),(208,74),(216,91),
              (230,104),(253,111),(272,120),(290,135),(306,153),(316,173),(319,191),
              (314,215),(310,235),(312,257),(309,283),(307,305),(301,327),(289,349),
              (279,368),(273,389),(259,408),(246,423),(232,431),(216,425),(201,417),
              (183,402),(164,388),(148,370),(138,351),(127,336),(112,319),(102,300),
              (94,283),(78,264),(62,244),(48,226),(38,211)]),
    "0196": ((1061,123,1283,313),
             [(53,115),(55,96),(55,76),(56,57),(60,42),(64,33),(75,29),(87,30),
              (98,36),(113,34),(128,35),(142,39),(152,48),(159,59),(168,67),(171,79),
              (176,89),(174,103),(171,112),(169,125),(160,134),(146,138),(131,138),
              (118,135),(106,138),(93,134),(79,135),(68,130),(59,126)]),
    "0201": ((460,650,703,872),
             [(39,158),(43,143),(49,129),(55,115),(60,99),(66,85),(72,72),(83,60),
              (94,54),(106,55),(119,58),(131,58),(146,64),(160,73),(178,81),(189,93),
              (197,106),(201,121),(201,138),(203,154),(199,170),(190,183),(173,188),
              (156,194),(140,201),(125,205),(110,208),(93,204),(77,200),(61,197),
              (51,187),(44,175)]),
    "0309": ((144,133,481,423),
             [(45,95),(51,76),(60,61),(73,50),(92,44),(112,40),(133,36),(151,34),
              (170,37),(185,45),(198,58),(213,77),(229,94),(249,111),(269,128),
              (284,147),(295,167),(299,186),(296,205),(285,226),(270,241),(256,253),
              (235,258),(215,260),(192,259),(173,254),(154,245),(138,234),(124,219),
              (113,203),(99,189),(84,175),(70,158),(60,143),(53,124),(47,110)])
}


def measure(sample, path):
    photo = landscape(path)
    rect = crop_box(photo)
    key = path.stem+"_"+"_".join(map(str,rect))+f"_{PPM}.npz"
    with np.load(MASKS/key) as data:
        mask = data["mask"]
    assert rect == (410,273,7782,5191) and mask.shape == (872,1307)
    zoom, outline = STONES[sample]
    manual = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(manual, [np.array(outline, np.int32) + np.array(zoom[:2], np.int32)], 1)
    diam = regionprops_table(manual, properties=("axis_minor_length",))["axis_minor_length"].item()/PPM
    assert diam > 20
    manual = manual.astype(bool)
    props = regionprops_table(mask, properties=("axis_minor_length","label"))
    large_ids = props["label"][props["axis_minor_length"] > 20*PPM]
    coarse = np.isin(mask, large_ids)
    coverage = 100*(coarse & manual).sum()/manual.sum()
    # Coverage can also be high when two neighbouring stones were merged.
    overlaps = np.bincount(mask[manual].astype(int), minlength=int(mask.max())+1)
    sizes = np.bincount(mask.ravel().astype(int))
    iou = overlaps[large_ids]/(manual.sum()+sizes[large_ids]-overlaps[large_ids])
    corrected = coarse | manual
    metrics = dict(short_mm=diam, coverage=coverage, best_single_iou_pct=100*iou.max(initial=0), area_before=100*coarse.mean(),
                   area_after=100*corrected.mean(), stone_area_pct=100*manual.mean())
    tile = photo.crop(rect).resize((mask.shape[1],mask.shape[0]),Image.Resampling.LANCZOS).crop(zoom)
    x,y,r,b = zoom
    auto_tile, manual_tile = tile.copy(), tile.copy()
    contours, _ = cv2.findContours(coarse[y:b,x:r].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(map(int,p)) for p in contour[:,0]]
        if len(points)>2:
            ImageDraw.Draw(auto_tile).line(points+[points[0]], fill="#ffd400",width=3)
    ImageDraw.Draw(manual_tile).line(outline+[outline[0]],fill="#10dce5",width=3)
    return metrics,(auto_tile,manual_tile)


def main():
    old = pd.read_csv(out/"iri_alan_iki_aralik_sonuclar.csv",dtype={"sample_id":str}).set_index("sample_id")
    source = old.loc[list(STONES)]
    assert set(source.index) == set(old.loc[(old.new | old.stress) & (old.index != "0027")].index)
    measurements, drawings, inputs = {}, {}, []
    cv2.setNumThreads(1)
    with threadpool_limits(1), patch.dict(gsd.PHONES,{PHYSICAL_CAMERA:(8192,NATIVE_PPM)}):
        factor = calibration()
        for sample, row in source.iterrows():
            path = PHOTOS/row.file
            measurements[sample], drawings[sample] = measure(sample,path)
            values = features.image_features(path,PHYSICAL_CAMERA)
            inputs.append(dict(sample_id=sample,**{k:values[k] for k in keys}))
        raw = v3(pd.DataFrame(inputs)).loc[source.index]
    result = pd.DataFrame.from_dict(measurements,orient="index").rename_axis("sample_id")
    result["cap6_3"] = 100-raw["6.3"]
    for mode in ("before","after"):
        area = result["area_"+mode]
        result["area_mass_"+mode] = factor*area
        result["final_"+mode] = 100-update20(raw,area,factor)
    np.testing.assert_allclose(result[["area_before","area_mass_before","final_before"]],
                               source[["a_20","mass20_from_area","mass20_model"]],rtol=0,atol=1e-6)
    result["lab20"] = source.mass20_lab
    result["area_error_gain"] = (result.area_mass_before-result.lab20).abs()-(result.area_mass_after-result.lab20).abs()
    result["final_error_gain"] = (result.final_before-result.lab20).abs()-(result.final_after-result.lab20).abs()
    result.to_csv(out/"montreal_dort_tas_kontrol.csv",float_format="%.6f")
    print(result.round(3).to_string())
    clipped = old[(old.a_20>0) & (old.mass20_from_area-old.mass20_model>.0001)]
    gain = (clipped.mass20_from_area-clipped.mass20_lab).abs()-(clipped.mass20_model-clipped.mass20_lab).abs()
    print("Automatic downward clipping, gain in absolute error:", gain.round(3).to_dict())
    sheet = Image.new("RGB",(920,650),"white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("arial.ttf",18)
    for i,(sample,tiles) in enumerate(drawings.items()):
        x,y = (i%2)*460,(i//2)*303
        draw.text((x+8,y+4),f"{sample}    Modelin saydığı kısım %{result.loc[sample,'coverage']:.1f}".replace(".",","),
                  font=font,fill="#182428")
        for j,tile in enumerate(tiles):
            tile = tile.resize((round(tile.width*min(216/tile.width,255/tile.height)),
                                round(tile.height*min(216/tile.width,255/tile.height))),Image.Resampling.LANCZOS)
            sheet.paste(tile,(x+6+j*225,y+35))
    draw.text((8,612),"Her çiftte solda sarı otomatik iri maske, sağda turkuaz elle çizilen taş sınırı var.",font=font,fill="#182428")
    sheet.save(out/"montreal_dort_tas_kontrol.jpg",quality=94)


if __name__=="__main__":
    main()
