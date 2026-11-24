"""Münster 5. çekimde ikinci çözünürlük, hedef sınır ve ayrı taş kontrolleri."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries

from imagegrains_bes_bolge import WEIGHTS
from imagegrains_pilot import load_model
from iri_63_gorsel import PHONES, PPM, list_images, munster_outline, out, work

COARSE = PPM / 4
STEM = "munster_ikinci_olcek"
# Özgün 820 × 615 önizlemede, tahminden önce seçilen on ayrı taşın iç noktası.
CENTERS = np.array([(305,330),(148,346),(228,442),(490,248),(565,350),
                    (665,264),(735,370),(647,450),(139,464),(671,113)])


def main(archive):
    images = list_images(work / "soil-data").set_index("file")
    name, = [n for n in images.index if images.loc[n,"sample_id"] == "HPC_Muenster_BS6_9_0-10m" and "(5)" in n]
    box = pd.read_csv(out/"test20_fotograflar.csv", index_col="file").loc[name]
    with Image.open(images.loc[name,"path"]) as src:
        photo = ImageOps.exif_transpose(src).convert("RGB")
    assert photo.size == (4032,3024)
    ref_side, scale = PHONES[box.phone]
    ppm = scale * photo.width / ref_side
    roi = photo.crop(tuple(box[["left","top","right","bottom"]]))
    def size(resolution):
        return tuple(round(n*resolution/ppm) for n in roi.size)
    with ZipFile(archive) as z, np.load(BytesIO(z.read(name+".npz"))) as data:
        old = data["mask"]
    assert old.shape[::-1] == size(PPM)
    rgb = np.asarray(roi.resize(size(COARSE), Image.Resampling.LANCZOS))
    model = load_model(WEIGHTS)
    print(f"ImageGrains {model.device}, {rgb.shape}", flush=True)
    new, *_ = model.eval(rgb,batch_size=1,bsize=256,flow_threshold=.4,cellprob_threshold=0.)
    assert new.shape == rgb.shape[:2] and 0 < new.max() <= 65535
    np.savez_compressed(out/f"{STEM}_maske.npz",photo_name=name,ppm=COARSE,mask=new.astype(np.uint16))

    # Her iki maskeyi arşivdeki yoğun ızgarada aynı dış çizgiye göre değerlendir.
    factor = PPM / ppm
    def local(points):
        return np.rint((points * photo.width / 820 - np.array([box.left,box.top]))*factor).astype(np.int32)
    outline = local(munster_outline())
    target = np.zeros(old.shape,np.uint8)
    cv2.fillPoly(target,[outline],1)
    coords = local(CENTERS)
    assert target[coords[0,1],coords[0,0]] and not target[coords[1:,1],coords[1:,0]].any()
    overlay = np.asarray(roi.resize(size(PPM),Image.Resampling.LANCZOS))
    methods = [(PPM,old),(COARSE,cv2.resize(new.astype(np.uint16),size(PPM),interpolation=cv2.INTER_NEAREST))]
    fig,axes = plt.subplots(1,2,figsize=(11.2,4.8),layout="constrained")
    rows = []
    for ax,(resolution,mask) in zip(axes,methods):
        overlap = np.bincount(mask[target>0],minlength=int(mask.max())+1)
        areas = np.bincount(mask.ravel(),minlength=len(overlap))
        overlap[0] = 0
        iou = overlap / (areas + target.sum() - overlap)
        best = int(iou.argmax())
        assert best > 0
        candidate = (mask == best).astype(np.uint8)
        mm = regionprops_table(candidate,properties=("axis_minor_length",))["axis_minor_length"][0]/PPM
        ids = mask[coords[:,1],coords[:,0]]
        detected = ids[ids>0]
        merged = sum(n*(n-1)//2 for n in np.unique(detected, return_counts=True)[1])
        rows.append(dict(ppm=resolution, iou=iou[best], kapsama=overlap[best]/target.sum(),
                         saflik=overlap[best]/areas[best], kisa_mm=mm,
                         merkez_bulunan=len(detected), birlesen_merkez_cifti=merged,
                         merkeze_atanan_maskeler=" ".join(map(str,ids))))
        view = overlay.copy()
        view[find_boundaries(mask, mode="thick")] = (45,185,205)
        cv2.polylines(view,[outline],True,(255,202,44),3)
        contours = cv2.findContours(candidate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(view,contours,-1,(232,72,148),3)
        ax.imshow(view)
        ax.scatter(coords[:,0],coords[:,1],s=15,c="white",edgecolors="black",linewidths=.7)
        ax.set_title(f"{resolution:g} piksel/mm    ·    Örtüşme %{100*iou[best]:.1f}".replace(".",","))
        ax.set_axis_off()
    fig.suptitle("Münster   ·   Sarı: elle çizilen taş     Pembe: en iyi örtüşen maske     Beyaz: ayrı taşlar",fontsize=12)
    fig.savefig(out/f"{STEM}.jpg", dpi=145)
    plt.close(fig)
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv(out/f"{STEM}.csv",index=False,float_format="%.7f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    main(parser.parse_args().archive)
