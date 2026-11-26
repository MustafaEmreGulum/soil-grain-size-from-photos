"""63 mm eleği için sınıra değen uzun maskeleri bul ve özgün çekimde yeniden ölç."""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.measure import regionprops

from munster_ikinci_olcek import COARSE, PHONES, PPM, WEIGHTS, list_images, load_model, out, work


def read_photo(row):
    with Image.open(row.path) as src:
        native = ImageOps.exif_transpose(src).convert("RGB")
    portrait = native.height > native.width
    view = native.transpose(Image.Transpose.ROTATE_90) if portrait else native
    ref, scale = PHONES[row.phone]
    return view, portrait, scale * view.width / ref


def edge_ids(mask):
    ids = np.unique(np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])))
    return ids[ids > 0]


def fit_rect(box, ppm, resolution, dims):
    # Eksenleri ayrı ayrı ölçeklemek, yeniden boyutlandırma yuvarlamasını da içerir.
    sx, sy = dims[0] / box.width, dims[1] / box.height
    assert max(abs(sx * ppm - resolution), abs(sy * ppm - resolution)) < .02
    return tuple(round(v*s) for v,s in zip((box.left, box.top, box.right, box.bottom),
                                           (sx, sy, sx, sy), strict=True))


def full_mask(view, portrait, ppm, model):
    native = view.transpose(Image.Transpose.ROTATE_270) if portrait else view
    dims = tuple(round(n * COARSE / ppm) for n in native.size)
    rgb = np.asarray(native.resize(dims, Image.Resampling.LANCZOS))
    raw, *_ = model.eval(rgb, batch_size=1, bsize=256, flow_threshold=.4, cellprob_threshold=0.)
    assert raw.shape == rgb.shape[:2]
    return np.rot90(raw) if portrait else raw


def match_region(target, mask):
    overlaps = np.bincount(mask[target], minlength=int(mask.max())+1)
    areas = np.bincount(mask.ravel(), minlength=len(overlaps))
    overlaps[0] = 0
    scores = overlaps / (target.sum() + areas - overlaps)
    label = int(np.argmax(scores))
    return label, float(scores[label])


def verify_against_other_view(photo, target, other):
    # Karşılıklı ayırt edici SIFT eşleşmeleri, yalnız görünür taşın içinden.
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    def describe(im, region=None):
        gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
        return sift.detectAndCompute(gray, region)
    keys, desc = describe(photo, target.astype(np.uint8)*255)
    others, dest = describe(other)
    if desc is None or dest is None or min(len(desc), len(dest)) < 2:
        return 0, 0., None
    def filtered(a,b):
        return {p.queryIdx:p.trainIdx for p,q in matcher.knnMatch(a,b,k=2) if p.distance < .75*q.distance}
    reverse = filtered(dest,desc)
    pairs = [(i,j) for i,j in filtered(desc,dest).items() if reverse.get(j)==i]
    if len(pairs)<3:
        return 0, 0., None
    a = np.float32([keys[i].pt for i,j in pairs])
    b = np.float32([others[j].pt for i,j in pairs])
    transform, inliers = cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3)
    pts = a[inliers.ravel().astype(bool)] if inliers is not None else a[:0]
    coverage = cv2.contourArea(cv2.convexHull(pts)) / target.sum() if len(pts)>=3 else 0.
    return len(pts), coverage, transform


def alternate_views(archive, photos, boxes, records, model=None):
    """Fotoğraf sınırından kurtulamayan taşları aynı numunenin diğer çekimlerinde ara."""
    rows = []
    def preview(name):
        view, portrait, _ = read_photo(photos.loc[name])
        dims = view.size
        view.thumbnail((1200,1200),Image.Resampling.LANCZOS)
        l,t,r,b = np.rint(boxes.loc[name].to_numpy() * np.tile(np.divide(view.size,dims),2)).astype(int)
        with np.load(BytesIO(archive.read(name+".npz"))) as d:
            mask = d["mask"]
        if portrait:
            mask = np.rot90(mask)
        canvas = np.zeros(view.size[::-1],np.uint16)
        canvas[t:b,l:r] = cv2.resize(mask,(r-l,b-t),interpolation=cv2.INTER_NEAREST)
        crop_area = np.zeros(canvas.shape,np.uint8)
        crop_area[t:b,l:r] = 1
        return view,canvas,mask,crop_area
    full_masks = {}
    for rec in records:
        if rec["durum"] == "ic_maske":
            continue
        name, label = rec["file"],rec["etiket"]
        im, canvas, _, source_area = preview(name)
        stone = canvas==label
        for other in photos.index[photos.sample_id==rec["sample_id"]]:
            if other == name:
                continue
            dest, labels, native_mask, _ = preview(other)
            n, spread, transform = verify_against_other_view(im,stone,dest)
            size, iou, agreement, extra = (np.nan,)*4
            status = "eslesme_belirsiz"
            if n>=8 and spread>=.08 and transform is not None:
                projected = cv2.warpAffine(stone.astype(np.uint8),transform,dest.size,flags=cv2.INTER_NEAREST)>0
                if projected.any():
                    match,iou = match_region(projected,labels)
                    if match and iou>=.5:
                        if match in edge_ids(native_mask):
                            status = "diger_kadrajda_da_kenar"
                        else:
                            # İçte kalan bir maske de taşın yalnızca aynı parçasını çizebilir.
                            # Yeni kadrajda eski sınırın ötesini görmeli ve iki ölçek de uyuşmalı.
                            interior = labels==match
                            warped = cv2.warpAffine(source_area,transform,dest.size,flags=cv2.INTER_NEAREST)
                            extra = np.mean(warped[interior]==0)
                            if other not in full_masks:
                                if model is None:
                                    model = load_model(WEIGHTS)
                                view, portrait, ppm = read_photo(photos.loc[other])
                                full_masks[other] = full_mask(view,portrait,ppm,model)
                            coarse = full_masks[other]
                            resized = cv2.resize(coarse.astype(np.uint16),dest.size,interpolation=cv2.INTER_NEAREST)
                            coarse_id, agreement = match_region(interior,resized)
                            if agreement < .75 or extra < .05 or coarse_id in edge_ids(coarse):
                                status = "tam_sinir_dogrulanamadi"
                            else:
                                status = "iki_olcekte_ic_maske"
                                size = regionprops(np.uint8(native_mask==match))[0].axis_minor_length/PPM
            rows.append(dict(file=name,etiket=label,diger=other,uyumlu_nokta=n,kapsama=spread,
                             iou=iou,iki_olcek_iou=agreement,yeni_alan_payi=extra,
                             durum=status,olculen_kisa_mm=size))
    return rows


def main(archive):
    photos = list_images(work/"soil-data").set_index("file")
    cols = ["left","top","right","bottom"]
    boxes = pd.concat([pd.read_csv(out/f,index_col="file")[cols] for f in
                       ("dinov2_hiyerarsik_kadrajlar.csv","test20_fotograflar.csv")])
    assert photos.index.is_unique and boxes.index.is_unique and set(photos.index)==set(boxes.index)
    records, scans = [], []
    model = None
    with ZipFile(archive) as z:
        for name,row in photos.sort_index().iterrows():
            with np.load(BytesIO(z.read(name+".npz"))) as d:
                raw = d["mask"]
            ids = edge_ids(raw)
            candidates = [p for p in regionprops(np.where(np.isin(raw,ids),raw,0))
                          if p.axis_major_length >= 63*PPM]
            scans.append(dict(file=name,split=row.split,sample_id=row.sample_id,aday=len(candidates)))
            if not candidates:
                continue
            view, portrait, ppm = read_photo(row)
            box = boxes.loc[name].copy()
            box["width"],box["height"] = view.size
            mask = np.rot90(raw) if portrait else raw
            assert mask.shape[::-1] == tuple(round(n*PPM/ppm) for n in
                                              (box.right-box.left,box.bottom-box.top))
            if model is None:
                model = load_model(WEIGHTS)
            full = full_mask(view,portrait,ppm,model)
            dims = full.shape[::-1]
            l,t,r,b = fit_rect(box,ppm,COARSE,dims)
            for p in candidates:
                target = np.zeros(full.shape,bool)
                target[t:b,l:r] = cv2.resize(np.uint8(mask==p.label),(r-l,b-t),interpolation=cv2.INTER_NEAREST)>0
                label, iou = match_region(target,full)
                found = full==label if label else np.zeros_like(target)
                reliable = iou>=.5
                # Tek piksellik boşluğu da belirsiz kenar olarak kabul et.
                border = bool(np.any(found[:2]) or np.any(found[-2:]) or np.any(found[:,:2]) or np.any(found[:,-2:]))
                size = regionprops(found.astype(np.uint8))[0].axis_minor_length/COARSE if reliable else np.nan
                status = "eslesme_belirsiz"
                if reliable:
                    if border:
                        status = "foto_kenarinda"
                    elif 1 - found[t:b,l:r].sum()/found.sum() >= .05:
                        status = "ic_maske"
                    else:
                        status = "ek_sinir_dogrulanamadi"
                records.append(dict(file=name,sample_id=row.sample_id,split=row.split,etiket=p.label,
                                    eski_kisa_mm=p.axis_minor_length/PPM,eski_uzun_mm=p.axis_major_length/PPM,
                                    tam_gorunen_kisa_mm=size,iou=iou,durum=status))
                print(name,status,round(size,1),round(iou,2),flush=True)
                if row.sample_id=="H374":
                    fig, axes = plt.subplots(1,3,figsize=(12,3.3),layout="constrained")
                    hd = view.resize((1200,round(1200*view.height/view.width)),Image.Resampling.LANCZOS)
                    stone = cv2.resize(np.uint8(target),hd.size,interpolation=cv2.INTER_NEAREST)>0
                    for ax,(other_name,other_row) in zip(axes,photos[photos.sample_id==row.sample_id].sort_index().iterrows(),strict=True):
                        other,_,_ = read_photo(other_row)
                        other.thumbnail((1200,1200),Image.Resampling.LANCZOS)
                        ax.imshow(other)
                        ax.set_axis_off()
                        if other_name==name:
                            h,w = full.shape
                            ax.contour((np.arange(w)+.5)*other.width/w,(np.arange(h)+.5)*other.height/h,
                                       found.astype(int),levels=[.5],colors=["#00e3c7"],linewidths=1.4)
                            ax.set_title("1. çekim · Tam fotoğraftaki maske")
                        else:
                            n,_,_ = verify_against_other_view(hd,stone,other)
                            ax.set_title(f"{Path(other_name).stem[-2:]}. çekim · {n} uyumlu nokta")
                    fig.savefig(out/"iri_kenar_h374.jpg",dpi=155)
                    plt.close(fig)
        matches = alternate_views(z,photos,boxes,records,model)
    for filename,rows in (("tarama",scans),("adaylar",records),("eslesme",matches)):
        pd.DataFrame(rows).to_csv(out/f"iri_kenar_{filename}.csv",index=False,float_format="%.6f")


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    main(parser.parse_args().archive)
