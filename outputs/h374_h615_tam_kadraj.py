"""Sekiz çekimde eski beş karenin fiziksel konumunu tam toprak kadrajına taşı."""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.measure import regionprops

import iri_tas_kalan_doku as prev

ig = prev.ig
STEM = ig.out / "h374_h615_tam_kadraj"
# Tam kadrajın sol üstünden mm olarak, iki çekimde ayrı ayrı okunan görünür sınır.
OUTLINES = {
    "Motorola_Edge_H615_02.jpg": [(80,130),(81,125),(83,122),(88,117),(91,112),
        (95,108),(100,104),(106,101),(110,100.5),(115,103),(121,105),(129,108.5),
        (126,112),(123,118),(120,122),(115,125),(111,129),(103,131),(95,133),
        (88,134.5),(83,134)],
    "Samsung_A52_H615_02.jpg": [(85,138),(85,134),(88,130),(93,125),(96,121),
        (99,117),(104,115),(108,111),(113,108.5),(117,109),(122,112),(129,114),
        (133,117),(130,121),(127,127),(124,130),(119,133),(115,137),(108,138.5),
        (102,140),(96,142),(89,142.5),(86,141)],
}


def panel(ax, row, box, masks, old):
    with Image.open(row.path) as src:
        native = ImageOps.exif_transpose(src).convert("RGB")
    w, h = native.size
    portrait = h > w
    rgb = native.transpose(Image.Transpose.ROTATE_90) if portrait else native
    assert rgb.size == (box.width, box.height)
    roi = rgb.crop((box.left, box.top, box.right, box.bottom))
    ref, base = ig.gsd.PHONES[row.phone]
    ppm = base * max(w, h) / ref
    width, height = np.array(roi.size) / ppm
    roi.thumbnail((1800, 1200), Image.Resampling.LANCZOS)
    ax.imshow(roi, extent=(0, width, height, 0))
    side = round(ig.SIDE_MM * ppm)
    pixels = np.flatnonzero(ig.CORE.any(axis=0))
    lo, hi = pixels[0] / ig.SIDE, (pixels[-1] + 1) / ig.SIDE
    area = positive_fields = 0
    covered = np.zeros((box.bottom-box.top, box.right-box.left), bool)
    context = np.zeros_like(covered)
    for mask, (_, (dx, dy)) in zip(masks, prev.fields(row), strict=True):
        x = (w-side)//2 + round(dx*ppm)
        y = (h-side)//2 + round(dy*ppm)
        expected = native.crop((x,y,x+side,y+side)).resize((ig.SIDE,ig.SIDE),Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(np.asarray(expected), ig.crop(row,(dx,dy)))
        large = np.isin(mask, [p.label for p in regionprops(mask) if p.axis_minor_length > 20*ig.PPM])
        hits = np.count_nonzero(large & ig.CORE)
        positive_fields += int(hits > 0)
        area += hits
        if portrait:
            x, y = y, w-x-side
            large = np.rot90(large)
        x -= box.left
        y -= box.top
        context[max(0,y):min(y+side,context.shape[0]),max(0,x):min(x+side,context.shape[1])] = True
        inner = round(x+lo*side), round(y+lo*side), round(x+hi*side), round(y+hi*side)
        left, top, right, bottom = inner
        assert 0 <= left < right <= covered.shape[1] and 0 <= top < bottom <= covered.shape[0]
        assert not covered[top:bottom,left:right].any()
        covered[top:bottom,left:right] = True
        ax.add_patch(Rectangle((x/ppm,y/ppm),side/ppm,side/ppm,fill=False,ec="white",ls=":",lw=.75))
        ax.add_patch(Rectangle((left/ppm,top/ppm),(right-left)/ppm,(bottom-top)/ppm,
                               fill=False,ec="#00e1ef",lw=1.25))
        if large.any() and not large.all():
            coords = (np.arange(ig.SIDE)+.5)*side/ig.SIDE
            ax.contour((x+coords)/ppm,(y+coords)/ppm,large.astype(int),levels=[.5],
                       colors=["#ff388a"],linewidths=.95)
    assert positive_fields == old.pozitif_kare
    np.testing.assert_allclose(100*area/(5*ig.CORE.sum()), old.alan20_pct, atol=1e-6)
    ax.plot([6,26],[height-5]*2,color="white",lw=2.5,
            path_effects=[patheffects.withStroke(linewidth=4.5, foreground="#222")])
    ax.text(16,height-7,"20 mm",ha="center",color="white",fontsize=8,clip_on=True,
            path_effects=[patheffects.withStroke(linewidth=2,foreground="#222")])
    ax.set(xlim=(0,width),ylim=(height,0),aspect="equal")
    ax.tick_params(axis="both",labelsize=7,length=2,pad=1)
    camera = row.phone.replace("Motorola_Edge_60_fusion","Edge 60 Fusion").replace("_"," ")
    ax.set_title(f"{row.sample_id} · {camera} · Çekim {Path(row.file).stem[-2:]}",loc="left",fontsize=10)
    candidate = {}
    if row.file in OUTLINES:
        points = np.array(OUTLINES[row.file],np.float32)
        outline = np.vstack((points,points[0]))
        ax.plot(*outline.T,color="#ffe245",lw=1.6)
        patch = np.zeros(covered.shape,np.uint8)
        cv2.fillPoly(patch,[np.rint(points*ppm).astype(np.int32)],1)
        # En dar paralel çizgi aralığı, dışbükey kılıfın kenar normallerinde bulunur.
        hull = cv2.convexHull(points)[:,0,:]
        edges = np.roll(hull,-1,axis=0)-hull
        normals = np.column_stack((-edges[:,1],edges[:,0]))/np.linalg.norm(edges,axis=1)[:,None]
        min_width = np.ptp(hull@normals.T,axis=0).min()
        candidate = dict(aday_min_genislik_mm=min_width,
                         aday_olcum_karelerinde_pct=100*patch[covered].sum()/patch.sum(),
                         aday_baglam_kesitlerinde_pct=100*patch[context].sum()/patch.sum())
    return dict(file=row.file,sample_id=row.sample_id,ppm=ppm,genislik_mm=width,yukseklik_mm=height,
                bes_kare_kapsami_pct=100*covered.mean(),eski_pozitif_kare=positive_fields,**candidate)


def main():
    rows = prev.photos().query("sample_id in ['H374', 'H615']")
    boxes = pd.read_csv(ig.out / "dinov2_hiyerarsik_kadrajlar.csv",index_col="file")
    old = pd.read_csv(ig.out / "h374_h615_20mm_alan.csv",index_col="file")
    assert list(rows.index) == list(old.index) and len(rows) == 8
    fig, axes = plt.subplots(4,2,figsize=(14,17),layout="constrained")
    fig.suptitle("H374 ve H615 · Ölçüm kareleri toprak yüzeyinin neresinde?\n"
                 "Turkuaz 26 mm ölçüm karesi · Noktalı 52 mm kesit · Pembe eski aday · Sarı elle izlenen aday · Eksenler mm",
                 fontsize=13)
    records = []
    detail, detail_axes = plt.subplots(1,3,figsize=(11,3.8),layout="constrained")
    detail.suptitle("Ölçüm karesinin kenarında görülen taşlar · eksenler mm")
    windows = {rows.index[0]:(0,85,35,120),
               "Motorola_Edge_H615_02.jpg":(70,143,87,150),
               "Samsung_A52_H615_02.jpg":(75,148,95,158)}
    with np.load(prev.ARCHIVE) as archive:
        for ax, (name,row) in zip(axes.flat,rows.iterrows(),strict=True):
            records.append(panel(ax,row,boxes.loc[name],archive[name],old.loc[name]))
        for ax,(name,(left,right,top,bottom)) in zip(detail_axes,windows.items(),strict=True):
            panel(ax,rows.loc[name],boxes.loc[name],archive[name],old.loc[name])
            ax.set(xlim=(left,right),ylim=(bottom,top))
            ax.plot([left+3,left+23],[bottom-3]*2,color="white",lw=2,
                    path_effects=[patheffects.withStroke(linewidth=4,foreground="#333")])
            ax.text(left+13,bottom-4,"20 mm",ha="center",color="white",fontsize=8,
                    path_effects=[patheffects.withStroke(linewidth=2,foreground="#333")])
    detail.savefig(STEM.with_name(STEM.name+"_detay.png"),dpi=150)
    plt.close(detail)
    frame = pd.DataFrame(records)
    frame.to_csv(STEM.with_suffix(".csv"),index=False,float_format="%.5f")
    fig.savefig(STEM.with_suffix(".png"),dpi=165)
    plt.close(fig)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
