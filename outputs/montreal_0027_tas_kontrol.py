"""0027'de sağdaki tek taşın görünen sınırını aynı ölçüm kadrajında karşılaştır."""
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops_table
from threadpoolctl import threadpool_limits

from montreal_20mm_sinama import (MASKS, PHOTOS, NATIVE_PPM, PPM, PHYSICAL_CAMERA,
                                  calibration, crop_box, features, gsd, keys, landscape, out, update20, v3)

# Elle işaretlenen dış sınır, maskesiz fotoğraftaki 490 x 640 yakın görünümde.
ORIGIN = (880, 170)
OUTLINE = [(216,55),(235,51),(270,48),(301,53),(329,62),(351,77),(373,86),
           (389,100),(403,120),(423,135),(436,151),(442,176),(449,194),(454,220),
           (450,241),(444,252),(431,263),(415,278),(408,290),(397,318),(385,348),
           (374,373),(357,397),(346,411),(341,429),(335,457),(328,475),(321,491),
           (318,508),(310,516),(298,528),(288,548),(273,562),(257,569),(231,572),
           (203,572),(178,570),(158,565),(147,553),(137,534),(123,509),(109,485),
           (97,467),(93,455),(92,435),(93,420),(88,405),(91,387),(95,370),
           (97,346),(98,325),(103,297),(104,271),(103,249),(109,230),(116,212),
           (122,194),(126,181),(127,164),(132,150),(146,128),(165,106),(186,83)]


def main():
    sample = "0027"
    name = "0027_bulk_dry_tray024_01.jpg"
    photo = landscape(PHOTOS/name)
    rect = crop_box(photo)
    assert rect == (410, 273, 7782, 5191)
    with np.load(MASKS / (name[:-4]+"_410_273_7782_5191_4.5.npz")) as data:
        mask = data["mask"]
    h, w = mask.shape
    assert (w,h) == (1307,872)
    points = np.array(OUTLINE, np.int32) + ORIGIN
    manual = np.zeros((h,w), np.uint8)
    cv2.fillPoly(manual, [points.astype(np.int32)], 1)
    manual = manual.astype(bool)
    props = regionprops_table(mask, properties=("label","axis_minor_length"))
    coarse = np.isin(mask, props["label"][props["axis_minor_length"] > 20*PPM])
    assert regionprops_table(manual.astype(np.uint8), properties=("axis_minor_length",))["axis_minor_length"][0] > 20*PPM
    # Only add missing pixels of this stone. Existing coarse pixels are counted once.
    corrected = coarse | manual
    areas = np.array([coarse.mean(),corrected.mean()])*100
    old = pd.read_csv(out/"iri_alan_iki_aralik_sonuclar.csv", dtype={"sample_id":str}).set_index("sample_id").loc[sample]
    cv2.setNumThreads(1)
    with threadpool_limits(1), patch.dict(gsd.PHONES, {PHYSICAL_CAMERA: (8192,NATIVE_PPM)}):
        factor = calibration()
        values = features.image_features(PHOTOS/name, PHYSICAL_CAMERA)
        raw = v3(pd.DataFrame([dict(sample_id=sample, **{k:values[k] for k in keys})]))
    pre = factor*areas
    final = np.array([100-update20(raw,pd.Series(a,index=raw.index),factor).iloc[0] for a in areas])
    np.testing.assert_allclose([areas[0],pre[0],final[0]], [old.a_20,old.mass20_from_area,old.mass20_model],atol=1e-6,rtol=0)
    src = pd.read_csv(PHOTOS.parent/"psd_mass_path_80_to_0_08mm_no_ballast.csv",dtype={"sample_id":str})
    lab = (100-src.loc[src.sample_id.eq(sample),"%_20mm"]).unique()
    assert len(lab)==1 and np.isclose(lab[0],old.mass20_lab)
    coverage = 100*(coarse & manual).sum()/manual.sum()
    result = pd.DataFrame(dict(sinir=["otomatik","tek_tas_elle"], iri_alan_pct=areas, alandan_kutle_pct=pre,
                               son_kutle_pct=final, lab_kutle_pct=lab[0], v3_6_3_ustu_pct=100-raw["6.3"].iloc[0],
                               elle_cizilen_tasin_kadraj_payi_pct=100*manual.mean(),
                               otomatik_iri_maskenin_tasi_kapsama_pct=coverage))
    print(result.round(3).to_string(index=False))
    result.to_csv(out/"montreal_0027_tas_kontrol.csv",index=False,float_format="%.6f")

    sx,sy = (rect[2]-rect[0])/w,(rect[3]-rect[1])/h
    x,y = ORIGIN
    zoom = photo.transform((490,640),Image.Transform.AFFINE,
                           (sx,0,rect[0]+x*sx,0,sy,rect[1]+y*sy),resample=Image.Resampling.BICUBIC)
    panel = Image.new("RGB",(1000,746),"white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype("arial.ttf",18)
    for i,title in enumerate(["Modelin iri taş olarak saydığı alan","Fotoğraftan elle çizilen sınır"]):
        tile = zoom.copy()
        pen = ImageDraw.Draw(tile)
        if i==0:
            auto = np.zeros((640,490),np.uint8)
            auto[:,:w-x] = coarse[y:y+640,x:]
            contours,_ = cv2.findContours(auto,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                pts = [tuple(map(int,pt)) for pt in contour[:,0]]
                if len(pts)>2:
                    pen.line(pts+[pts[0]],fill="#ffd400",width=4)
        else:
            pen.line(OUTLINE+[OUTLINE[0]], fill="#11dbde",width=4)
        for y0 in range(0,640,16):
            pen.line((w-x,y0,w-x,y0+8),fill="white",width=2)
        panel.paste(tile,(i*500+5,39))
        draw.text((i*500+7,8),title,font=font,fill="#182428")
    draw.text((8,691),"Kesikli çizgi ölçüm kadrajının sağ kenarı. Hesapta taşın bu çizginin solunda kalan kısmı kullanıldı.",font=font,fill="#182428")
    panel.save(out/"montreal_0027_tas_kontrol.jpg",quality=94)


if __name__=="__main__":
    main()
