"""Tam yüzeydeki otomatik taş sınırını önceden çizilmiş iki H615 sınırıyla karşılaştır."""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.measure import regionprops

from h374_h615_tam_kadraj import OUTLINES, ig, prev

STEM = ig.out / "h615_otomatik_tas"
CACHE = ig.work.parent / ".venv" / "tam_kadraj_ig45"


def surface(row, box, resolution=ig.PPM):
    with Image.open(row.path) as src:
        photo = ImageOps.exif_transpose(src).convert("RGB")
    w, h = photo.size
    portrait = h > w
    assert tuple(sorted((w,h), reverse=True)) == (box.width, box.height)
    ref, base = ig.gsd.PHONES[row.phone]
    ppm = base * max(w,h) / ref
    rect = (w-box.bottom,box.left,w-box.top,box.right) if portrait else (
        box.left,box.top,box.right,box.bottom)
    roi = photo.crop(rect)
    dims = tuple(round(n * resolution / ppm) for n in roi.size)
    rgb = np.asarray(roi.resize(dims, Image.Resampling.LANCZOS))
    return rgb, portrait, ppm


def predict(model, name, rgb):
    CACHE.mkdir(parents=True, exist_ok=True)
    saved = CACHE / (name+".npz")
    if saved.exists():
        with np.load(saved) as data:
            mask = data["mask"]
    else:
        raw, *_ = model.eval(rgb,batch_size=1,bsize=256,flow_threshold=.4,cellprob_threshold=0.)
        assert raw.min() >= 0 and raw.max() <= 65535
        mask = raw.astype(np.uint16)
        tmp = saved.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,mask=mask)
        tmp.replace(saved)
    assert mask.shape == rgb.shape[:2] and mask.dtype == np.uint16
    return mask


def main():
    model = ig.load_model(prev.near.WEIGHTS)
    rows = prev.photos()
    boxes = pd.read_csv(ig.out / "dinov2_hiyerarsik_kadrajlar.csv",index_col="file")
    fig, axes = plt.subplots(1,2,figsize=(9,3.8),layout="constrained")
    results = []
    for ax,(file,outline) in zip(axes,OUTLINES.items(),strict=True):
        box = boxes.loc[file]
        rgb, portrait, ppm = surface(rows.loc[file],box)
        print("Full surface:",file,rgb.shape,flush=True)
        mask = predict(model,file,rgb)
        if portrait:
            rgb, mask = np.rot90(rgb), np.rot90(mask)
        h,w = mask.shape
        scale = np.array([w/(box.right-box.left),h/(box.bottom-box.top)])*ppm
        points = np.rint(np.array(outline)*scale-.5).astype(np.int32)
        ref = np.zeros(mask.shape,np.uint8)
        cv2.fillPoly(ref,[points],1)
        n = ref.sum()
        overlap = np.bincount(mask[ref.astype(bool)],minlength=int(mask.max())+1)
        overlap[0] = 0
        areas = np.bincount(mask.ravel(),minlength=len(overlap))
        ious = overlap/(areas+n-overlap)
        best = int(np.argmax(ious))
        matched = (mask == best) if best else np.zeros_like(ref,dtype=bool)
        short = regionprops(matched.astype(np.uint8))[0].axis_minor_length / ig.PPM if best else 0.
        other = overlap.copy()
        other[best] = 0
        result = dict(file=file,iou=ious[best],ref_kapsami=overlap[best]/n,
                      maske_safligi=overlap[best]/areas[best] if best else 0.,
                      ikinci_parca_ref_payi=other.max()/n,etiketsiz_ref_payi=1-overlap.sum()/n,
                      otomatik_kisa_eksen_mm=short,gecer=bool(ious[best]>=.5 and short>20))
        results.append(result)
        print(result,flush=True)
        extent = (0,w/scale[0],h/scale[1],0)
        ax.imshow(rgb,extent=extent)
        xy=np.vstack((outline,outline[0]))
        ax.plot(*xy.T,color="#ffdb3b",linewidth=1.5,label="Elle çizilen")
        for selected, color, label in [(matched,"#00e4eb","En iyi otomatik sınır"),
                                       (np.isin(mask,np.flatnonzero(other)), "#ec74e8","Diğer örtüşen parçalar")]:
            if selected.any():
                # Piksel merkezleri imshow ile aynı fiziksel koordinat sisteminde.
                ax.contour((np.arange(w)+.5)/scale[0],(np.arange(h)+.5)/scale[1],
                           selected,levels=[.5],colors=[color],linewidths=1)
                ax.plot([],[],color=color,label=label)
        lo,hi=xy.min(axis=0)-8,xy.max(axis=0)+8
        ax.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]),xlabel="mm",
               title=f"{rows.loc[file].phone.replace('_',' ')} · IoU {ious[best]:.2f}")
    handles = {}
    for ax in axes:
        for line in ax.lines:
            handles[line.get_label()] = line
    fig.legend(handles.values(),handles.keys(),loc="outside lower center",ncols=len(handles))
    fig.savefig(STEM.with_suffix(".png"),dpi=145)
    plt.close(fig)
    pd.DataFrame(results).to_csv(STEM.with_suffix(".csv"),index=False,float_format="%.8f")


if __name__ == "__main__":
    main()
