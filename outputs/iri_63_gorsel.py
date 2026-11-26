"""Dört numunede tam çekim, maske sınırları ve fiziksel ölçek denetimi."""
import argparse
from io import BytesIO
from zipfile import ZipFile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.measure import regionprops_table

from v3_knn_esit_karisim import out, work
from gsd import PHONES, list_images

PPM = 4.5
SAMPLES = {"H038":"H038", "H374":"H374", "HPC_Muenster_BS6_9_0-10m":"Münster",
           "HPC_Testfeld Lidl WHV":"Testfeld"}
STEM = "iri_63_gorsel"


def shot(path, row, mask):
    with Image.open(path) as src:
        photo = ImageOps.exif_transpose(src).convert("RGB")
    ref, scale = PHONES[row.phone]
    ppm = scale * max(photo.size) / ref
    if photo.width < photo.height:
        photo = photo.transpose(Image.Transpose.ROTATE_90)
        mask = np.rot90(mask)
    rect = tuple(int(row[k]) for k in ["left","top","right","bottom"])
    roi = photo.crop(rect)
    assert tuple(round(n*PPM/ppm) for n in roi.size) == mask.shape[::-1]
    props = pd.DataFrame(regionprops_table(mask, properties=("label","axis_minor_length","axis_major_length","bbox")))
    props["kisa"] = props.axis_minor_length / PPM
    props["uzun"] = props.axis_major_length / PPM
    assert int((props.kisa > 63).sum()) == row.n_63 and int((props.kisa > 20).sum()) == row.n_20
    biggest = props.loc[props.kisa.idxmax()]
    edge = ((props["bbox-0"] == 0) | (props["bbox-1"] == 0) |
            (props["bbox-2"] == mask.shape[0]) | (props["bbox-3"] == mask.shape[1]))
    overlay = np.array(roi.resize(mask.shape[::-1], Image.Resampling.LANCZOS))
    for label in props.loc[(props.kisa >= 8) | (props.label == biggest.label), "label"]:
        region = np.uint8(mask == label)
        contours = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        color = (255,210,45) if label == biggest.label else (45,225,218)
        cv2.drawContours(overlay, contours, -1, color, 3 if label == biggest.label else 2)
    largest_edge = props.loc[edge, "kisa"].max() if edge.any() else 0.
    metrics = dict(max_kisa_mm=biggest.kisa, ayni_tane_uzun_mm=biggest.uzun,
                   max_uzun_mm=props.uzun.max(), en_buyuk_kenarda=bool(edge.loc[biggest.name]),
                   kenarda_max_kisa_mm=largest_edge, uzun_63_adet=int((props.uzun>63).sum()))
    full = np.asarray(photo.resize(tuple(round(n*PPM/ppm) for n in photo.size), Image.Resampling.LANCZOS))
    physical_rect = np.asarray(rect)/ppm
    return full, overlay, physical_rect, metrics


def draw(ax, image, x0, y0, width, height):
    ax.imshow(image, extent=(x0,x0+image.shape[1]/PPM,y0+image.shape[0]/PPM,y0))
    ax.set(xlim=(0,width),ylim=(height,0),aspect="equal")
    ax.set_axis_off()
    ax.plot([4,67],[height-4]*2,linewidth=4,color="#24303b",solid_capstyle="butt")
    ax.plot([4,67],[height-4]*2,linewidth=2,color="#ffcf48",solid_capstyle="butt")


def munster_outline():
    # Beşinci çekimin EXIF yönü düzeltilmiş 820 × 615 önizlemesinde elle çizilen taş.
    return np.array([(177,363),(176,346),(181,321),(184,298),(199,278),(225,263),
                        (253,246),(274,221),(302,196),(323,178),(340,174),(356,183),
                        (371,202),(386,231),(394,263),(401,306),(404,343),(406,368),
                        (398,397),(381,426),(363,450),(343,461),(323,465),(301,461),
                        (275,451),(252,437),(234,419),(219,395),(205,384),(186,375)])


def ruler_check(full, rect, mask, path):
    outline = munster_outline()
    manual = np.zeros((615,820),np.uint8)
    cv2.fillPoly(manual,[outline],1)
    axes = regionprops_table(manual,properties=("axis_minor_length","axis_major_length"))
    with Image.open(path) as src:
        oriented = ImageOps.exif_transpose(src)
        portrait = oriented.width < oriented.height
        assert sorted(oriented.size) == [3024,4032]
    if portrait:
        mask = np.rot90(mask)
    px_per_mm = PHONES["iPhone14"][1] * 820/4032
    local = np.rint((outline / px_per_mm - rect[:2]) * PPM).astype(np.int32)
    reference = np.zeros(mask.shape,np.uint8)
    cv2.fillPoly(reference,[local],1)
    overlap = np.bincount(mask[reference>0],minlength=int(mask.max())+1)
    overlap[0] = 0
    best = int(np.argmax(overlap))
    assert overlap[best] > 0
    matching = np.uint8(mask == best)
    auto_mm = regionprops_table(matching,properties=("axis_minor_length",))["axis_minor_length"][0]/PPM
    manual_mm = axes["axis_minor_length"][0]/px_per_mm
    ruler_mm = axes["axis_minor_length"][0]/((541-188)/120)
    print(f"Munster 5: hand minor {manual_mm:.1f} mm, ruler {ruler_mm:.1f} mm, best overlap auto {auto_mm:.1f} mm, coverage {overlap[best]/reference.sum():.3f}, all masks {overlap.sum()/reference.sum():.3f}")
    fig,ax = plt.subplots(figsize=(7.3,5.8))
    ax.imshow(full,extent=(0,full.shape[1]/PPM,full.shape[0]/PPM,0))
    line = np.vstack([outline,outline[:1]])/px_per_mm
    ax.plot(*line.T,color="#ffcd39",lw=2,label=f"Elle çizilen sınır · {manual_mm:.1f} mm".replace(".",","))
    contours = cv2.findContours(matching,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours,key=cv2.contourArea)[:,0,:]/PPM + rect[:2]
    ax.plot(*np.vstack([contour,contour[:1]]).T,color="#00e9da",lw=2,label="En çok örtüşen maske · komşu taş")
    ax.plot([8,71],[12,12],color="white",lw=3)
    ax.text(8,10,"63 mm",color="white",weight="bold")
    ax.set_title("Münster, 5. çekim · Aynı taşın kısa ekseni")
    ax.set_axis_off()
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.02),ncol=2)
    fig.tight_layout()
    fig.savefig(out/f"{STEM}_munster_cetvel.jpg",dpi=155)
    plt.close(fig)


def main(archive):
    photos = list_images(work / "soil-data").set_index("file")
    measures = pd.read_csv(out / "maske_elek_alanlari.csv").set_index("file")
    boxes = pd.concat([pd.read_csv(out / f).set_index("file")[["left","top","right","bottom"]]
                       for f in ["dinov2_hiyerarsik_kadrajlar.csv","test20_fotograflar.csv"]])
    selected = measures[measures.sample_id.isin(SAMPLES)].join(boxes,validate="one_to_one")
    assert len(selected) == 20 and selected.notna().all().all()
    shots, records = [], []
    with ZipFile(archive) as z:
        for name,row in selected.sort_index().iterrows():
            with np.load(BytesIO(z.read(name+".npz"))) as data:
                path = photos.loc[name,"path"]
                mask = data["mask"]
                full, overlay, rect, metrics = shot(path,row,mask)
                if SAMPLES[row.sample_id] == "Münster" and "(5)" in name:
                    ruler_check(full,rect,mask,path)
            shots.append((row.sample_id,name,full,overlay,rect,metrics))
            records.append(dict(sample_id=row.sample_id,file=name,**metrics))
    width = max(s[2].shape[1] for s in shots)/PPM
    height = max(s[2].shape[0] for s in shots)/PPM + 7
    for sample,title in SAMPLES.items():
        group = [s for s in shots if s[0]==sample]
        fig,axes = plt.subplots(len(group),2,figsize=(10,2.95*len(group)+.8))
        fig.suptitle(title+"   ·   63 mm kontrolü",fontsize=16,y=.995)
        fig.text(.5,1-.42/fig.get_figheight(),
                 "Solda tam fotoğraf, sağda maskelenen kadraj  •  Sarı sınır: en büyük kısa eksen  •  Çizgi: 63 mm",
                 ha="center",fontsize=9)
        for axrow,(_,name,full,overlay,rect,metrics) in zip(axes,group):
            draw(axrow[0],full,0,0,width,height)
            draw(axrow[1],overlay,rect[0],rect[1],width,height)
            axrow[0].set_title(name,fontsize=8,loc="left")
            axrow[1].set_title(f"En büyük kısa eksen  {metrics['max_kisa_mm']:.1f} mm".replace(".",","),fontsize=9,loc="left")
        fig.subplots_adjust(top=1-.85/fig.get_figheight(), bottom=.01,left=.02,right=.99,wspace=.03,hspace=.16)
        fig.savefig(out/f"{STEM}_{title.replace('ü','u').lower()}.jpg",dpi=140)
        plt.close(fig)
    results = pd.DataFrame(records)
    results.to_csv(out/f"{STEM}.csv", index=False, float_format="%.5f")
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    main(parser.parse_args().archive)
