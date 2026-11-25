"""İki toprakta aynı fiziksel alanı dağıtılan JPEG ve gerçek v3 girdisinde göster."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from v3_knn_esit_karisim import out, work
import features
import gsd

SOILS = ("H037", "H668")
PHONES = {"Motorola_Edge": "Motorola Edge", "Samsung_A52": "Samsung A52"}
SIDE_MM = 12
TILE = 216


def zoom(image, box):
    # One display scale for both stages. Nearest neighbour preserves source pixels.
    return image.transform((TILE, TILE), Image.Transform.EXTENT, box, Image.Resampling.NEAREST)


def main():
    root = work / "soil-data"
    data = gsd.list_images(root)
    photos = data[data.sample_id.isin(SOILS) & data.split.eq("train")].sort_values("file")
    meta = pd.read_csv(root/"ppm_updated.csv").set_index("phone")
    assert len(photos)==13 and photos.file.is_unique and set(photos.phone)==set(PHONES)
    saved_ppm = pd.read_csv(out/"ince_doku_cozunurluk_olculer.csv").query("kol=='ozgun'").set_index("file").ppm
    records, panels = [], {}
    for row in photos.itertuples():
        with Image.open(row.path) as f:
            native = ImageOps.exif_transpose(f).convert("RGB")
        rgb, ppm = features.load_canonical(row.path,row.phone)
        canon = Image.fromarray(rgb)
        ref = meta.loc[PHONES[row.phone]]
        np.testing.assert_allclose(ppm, ref.ppm*max(native.size)/max(ref.width,ref.height),rtol=0,atol=1e-9)
        np.testing.assert_allclose(ppm, saved_ppm.loc[row.file],rtol=0,atol=1e-8)
        crop = np.array([int(v*features.CROP) for v in native.size])
        origin = (np.array(native.size)-crop)//2
        rescale = np.array(canon.size)/crop
        pxmm = rescale*ppm
        assert np.max(np.abs(pxmm-features.CANON_PPM)) < .01
        info = dict(file=row.file,sample_id=row.sample_id,phone=row.phone,jpeg_w=native.width,jpeg_h=native.height,
                    ref_width=ref.width,ref_height=ref.height,ref_ppm=ref.ppm,jpeg_ppm=ppm,
                    model_w=canon.width,model_h=canon.height,model_ppm_x=pxmm[0],model_ppm_y=pxmm[1])
        for mm in (.063,.2,.63,2):
            info.update({f"jpeg_px_{mm}":mm*ppm,f"model_x_px_{mm}":mm*pxmm[0],f"model_y_px_{mm}":mm*pxmm[1]})
        records.append(info)
        if row.file.endswith("_01.jpg"):
            lo, hi = (crop-SIDE_MM*ppm)/2, (crop+SIDE_MM*ppm)/2
            native_tile = zoom(native,tuple(np.r_[origin+lo,origin+hi]))
            model_tile = zoom(canon,tuple(np.r_[lo*rescale,hi*rescale]))
            panels[row.phone,row.sample_id] = native_tile,model_tile
    df = pd.DataFrame(records)
    df.to_csv(out/"h037_h668_piksel_olcegi.csv",index=False,float_format="%.8f")
    assert len(panels)==4
    canvas = Image.new("RGB",(958,598),"white")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype("arialbd.ttf",19)
    font = ImageFont.truetype("arial.ttf",17)
    for i,(phone,label) in enumerate(PHONES.items()):
        for j,s in enumerate(SOILS):
            for stage,im in enumerate(panels[phone,s]):
                x,y = 16+(2*j+stage)*230+j*10,52+i*266
                canvas.paste(im,(x,y))
                if i==0:
                    draw.text((x,14),s+(" · JPEG" if stage==0 else " · v3 girdisi"),font=title,fill="#182428")
                if j==0 and stage==0:
                    draw.text((x,y+TILE+4),label+"  /  01",font=font,fill="#182428")
    draw.text((16,555),"Her kare merkezdeki 12 × 12 mm alanı gösterir. Ekranda 1 mm",fill="#182428",font=font)
    draw.line((514,566,532,566),fill="#182428",width=3)
    canvas.save(out/"h037_h668_piksel_olcegi.png")
    print(df.groupby("phone")[["jpeg_ppm","model_ppm_x","model_ppm_y","jpeg_px_0.063","jpeg_px_0.2"]].agg(["min","max"]).round(4))


if __name__=="__main__":
    main()
