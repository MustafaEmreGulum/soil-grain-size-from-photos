"""H037 ve H038'in 14 tam çekimini ölçüm kadrajı ve taş sınırlarıyla karşılaştır."""
from argparse import ArgumentParser
from io import BytesIO
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.measure import regionprops_table

from iri_63_gorsel import PHONES, PPM, list_images, out, work
from v3_knn_esit_karisim import labels


def panels(row, box, mask):
    with Image.open(row.path) as src:
        photo = ImageOps.exif_transpose(src).convert("RGB")
    if photo.height > photo.width:
        photo = photo.transpose(Image.Transpose.ROTATE_90)
        mask = np.rot90(mask)
    assert photo.size==(box.width,box.height)
    ref, scale = PHONES[row.phone]
    ppm = scale * photo.width/ref
    rect = tuple(int(box[k]) for k in ("left","top","right","bottom"))
    roi = photo.crop(rect)
    assert tuple(round(n*PPM/ppm) for n in roi.size)==mask.shape[::-1]
    grains = pd.DataFrame(regionprops_table(mask,properties=("label","axis_minor_length","axis_major_length")))
    short = grains.axis_minor_length / PPM
    long = grains.axis_major_length / PPM
    largest = grains.loc[short.idxmax(),"label"]
    overlay = np.array(roi.resize(mask.shape[::-1],Image.Resampling.LANCZOS))
    for label in grains.loc[(short>=4)|(long>=10)|(grains.label==largest),"label"]:
        contours = cv2.findContours(np.uint8(mask==label),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(overlay,contours,-1,(255,217,0) if label==largest else (0,235,217),2)
    marked = photo.copy()
    marked.paste(Image.fromarray(overlay).resize(roi.size,Image.Resampling.LANCZOS),rect[:2])
    for panel in (photo,marked):
        draw = ImageDraw.Draw(panel)
        if panel is marked:
            draw.rectangle(rect,outline=(255,139,23),width=4)
        a=(24,panel.height-27)
        b=(a[0]+round(20*ppm),a[1])
        draw.line([a,b],fill="black",width=12)
        draw.line([a,b],fill="white",width=5)
        panel.thumbnail((560,420),Image.Resampling.LANCZOS)
    return photo,marked,dict(max_kisa_mm=short.max(),max_uzun_mm=long.max(),
                             adet_kisa_20=int((short>20).sum()),adet_uzun_20=int((long>20).sum()))


def main(archive):
    photos = list_images(work/"soil-data").query("sample_id in ['H037','H038']").set_index("file")
    boxes = pd.read_csv(out/"dinov2_hiyerarsik_kadrajlar.csv",index_col="file")
    coarse = pd.read_csv(out/"ikinci_olcek_egitim_fotograflar.csv",index_col="file")
    original = pd.read_csv(out/"maske_elek_alanlari.csv",index_col="file")
    font = ImageFont.truetype("arial.ttf",20)
    records = []
    with ZipFile(archive) as z:
        for sample,group in photos.sort_index().groupby("sample_id"):
            tiles=[]
            for name,row in group.iterrows():
                with np.load(BytesIO(z.read(name+".npz"))) as data:
                    plain,marked,stats = panels(row,boxes.loc[name],data["mask"])
                assert stats["adet_kisa_20"]==original.loc[name,"n_20"]
                low = coarse.loc[name,"max_mm"]
                records.append(dict(sample_id=sample,file=name,**stats,ikinci_max_kisa_mm=low,
                                    lab_20_ustu_pct=100-labels.loc[sample,"20"]))
                tile=Image.new("RGB",(1124,plain.height+32),"white")
                d=ImageDraw.Draw(tile)
                d.text((4,2),name.rsplit(".",1)[0],fill="black",font=font)
                d.text((565,2),f"En büyük kısa eksen {stats['max_kisa_mm']:.1f} mm",fill="black",font=font)
                tile.paste(plain,(0,32))
                tile.paste(marked,(564,32))
                tiles.append(tile)
            sheet=Image.new("RGB",(1124,sum(t.height for t in tiles)+34),"white")
            ImageDraw.Draw(sheet).text((4,4),"Beyaz çizgi 20 mm   |   Turuncu kadraj   |   Sarı en büyük kısa eksenli taş   |   Turkuaz diğer adaylar",fill="black",font=font)
            y=34
            for tile in tiles:
                sheet.paste(tile,(0,y))
                y+=tile.height
            sheet.save(out/f"20mm_kontrol_{sample}.jpg",quality=92)
    df=pd.DataFrame(records)
    assert df.groupby("sample_id").size().to_dict()=={"H037":6,"H038":8}
    df.to_csv(out/"h037_h038_20mm_olculer.csv",index=False,float_format="%.6f")
    print(df.round(2).to_string(index=False))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    main(parser.parse_args().archive)
