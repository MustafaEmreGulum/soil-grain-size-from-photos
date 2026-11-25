"""Görünen dış sınıra uydurulan elipsi, tek kenardan yapay kesilen maskelerde sına."""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from skimage.measure import regionprops

from iri_63_gorsel import PHONES, PPM, munster_outline, out


def short_axis(binary, ppm):
    return regionprops(binary.astype(np.uint8))[0].axis_minor_length / ppm


def cut_cases(binary, ppm):
    truth = short_axis(binary, ppm)
    for k in range(4):
        oriented = np.rot90(binary, k)
        for fraction in (.10, .25, .40):
            start = round(oriented.shape[1]*fraction)
            fragment = oriented[:,start:].copy().astype(np.uint8)
            contours = cv2.findContours(fragment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            points = max(contours, key=cv2.contourArea)[:,0,:]
            # x=0 ve komşu piksel yapay kesim yüzeyini temsil ediyor.
            arc = points[points[:,0] > 1]
            visible = short_axis(fragment,ppm)
            ellipse = cv2.fitEllipseDirect(arc.astype(np.float32)) if len(arc)>=5 else None
            fit = min(ellipse[1])/ppm if ellipse is not None else np.nan
            successful = bool(np.isfinite(fit) and fit > 0)
            yield dict(yon=k,kesilen_genislik=fraction,referans_mm=truth,
                       gorunen_mm=visible,elips_mm=fit if successful else visible,
                       kenar_orani=fragment[:,0].sum()/fragment.sum(axis=0).max(),
                       elips_basarili=successful), fragment, ellipse


def main(archive):
    photos = pd.read_csv(out/"maske_elek_alanlari.csv").query("split=='train'")
    assert len(photos)==127 and photos.file.is_unique
    records = []
    with ZipFile(archive) as z:
        for row in photos.itertuples():
            with np.load(BytesIO(z.read(row.file+".npz"))) as data:
                mask = data["mask"]
            for label, box in enumerate(find_objects(mask),1):
                if box is None:
                    continue
                y,x = box
                if y.start==0 or x.start==0 or y.stop==mask.shape[0] or x.stop==mask.shape[1]:
                    continue
                # Kısa eksen, en dar görüntü eksenindeki açıklığın iki katını aşamaz.
                if 2*min(y.stop-y.start,x.stop-x.start)<20*PPM:
                    continue
                stone = mask[box]==label
                if short_axis(stone,PPM)<20:
                    continue
                for result,_,_ in cut_cases(stone,PPM):
                    records.append(dict(kaynak="egitim",sample_id=row.sample_id,file=row.file,etiket=label,**result))
    # 63 mm üstünde elle izlenmiş tek örneği ayrı tut.
    outline = munster_outline()
    manual = np.zeros((615,820),np.uint8)
    cv2.fillPoly(manual,[outline],1)
    y,x = np.nonzero(manual)
    manual = manual[y.min():y.max()+1,x.min():x.max()+1]
    ref,scale = PHONES["iPhone14"]
    ppm = 820*scale/ref
    fig,axes = plt.subplots(1,3,figsize=(8,3.3),layout="constrained")
    for result,fragment,ellipse in cut_cases(manual,ppm):
        records.append(dict(kaynak="elle",sample_id="Münster",file="Münster 5",etiket=1,**result))
        if result["yon"]==0:
            ax=axes[(.10,.25,.40).index(result["kesilen_genislik"])]
            yy,xx = np.nonzero(fragment)
            ax.scatter(xx,-yy,s=.06,color="#8b806c")
            (cx,cy),(a,b),angle=ellipse
            ax.add_patch(Ellipse((cx,-cy),a,b,angle=-angle,fill=False,color="#147f80",lw=1.6))
            ax.axvline(0,color="#ba6b25",ls=":")
            ax.set(aspect="equal",title=f"%{round(100*result['kesilen_genislik'])} kesim · {result['elips_mm']:.1f} mm")
            ax.axis("off")
    fig.suptitle(f"Münster · Tam kısa eksen {short_axis(manual,ppm):.1f} mm · Yeşil çizgi tahmin edilen elips")
    fig.savefig(out/"yapay_kesik_tas.png",dpi=140)
    plt.close(fig)
    df = pd.DataFrame(records)
    keys=["kaynak","sample_id","file","etiket"]
    assert df.groupby(keys).size().eq(12).all()
    for mode in ["gorunen","elips"]:
        df[mode+"_hata_mm"] = (df[mode+"_mm"]-df.referans_mm).abs()
    df.to_csv(out/"yapay_kesik_tas.csv",index=False,float_format="%.6f")
    train=df.query("kaynak=='egitim'")
    errors=["gorunen_hata_mm","elips_hata_mm"]
    soil=train.groupby(["sample_id","kesilen_genislik"])[errors].mean()
    print(soil.groupby("kesilen_genislik").mean())
    print("training soils, photos, instances, maximum, fallback:",train.sample_id.nunique(),train.file.nunique(),
          len(train[keys].drop_duplicates()),train.referans_mm.max(),(~df.elips_basarili).sum())
    print("threshold false positives (visible, estimated), training total:",
          int((train.gorunen_mm>63).sum()),int((train.elips_mm>63).sum()),len(train))
    manual_rows=df.query("kaynak=='elle'")
    print("manual 63mm hits:",(manual_rows[["gorunen_mm","elips_mm"]]>63).sum().to_dict(),len(manual_rows))
    print("soils improved:",(soil.groupby(level="sample_id").mean().eval("elips_hata_mm < gorunen_hata_mm")).sum())


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    main(parser.parse_args().archive)
