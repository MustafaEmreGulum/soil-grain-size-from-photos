"""1,125 piksel/mm'de tüm eğitim maskeleri ve 63 mm için saklı kütle sınaması."""
from time import monotonic

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from h615_otomatik_tas import surface
from munster_ikinci_olcek import COARSE, WEIGHTS, list_images, load_model, out, work
from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels
from gsd import emd

STEM = "ikinci_olcek_egitim"
CACHE = work / "ig1125_egitim_maskeleri"


def measure():
    photos = list_images(work / "soil-data").query("split == 'train'").set_index("file")
    assert len(photos) == 127 and photos.index.is_unique
    boxes = pd.read_csv(out / "dinov2_hiyerarsik_kadrajlar.csv", index_col="file")
    CACHE.mkdir(exist_ok=True)
    model = None
    rows = []
    start = monotonic()
    ordered = photos.assign(priority=~photos.sample_id.isin(["H038","H374"])).sort_values(["priority","sample_id"])
    for n,(name,row) in enumerate(ordered.iterrows(),1):
        rgb, _, _ = surface(row, boxes.loc[name], resolution=COARSE)
        saved = CACHE / (name+".npz")
        if not saved.exists():
            if model is None:
                model = load_model(WEIGHTS)
            raw, *_ = model.eval(rgb,batch_size=1,bsize=256,flow_threshold=.4,cellprob_threshold=0.)
            assert raw.shape == rgb.shape[:2] and raw.min() >= 0 and raw.max() <= 65535
            tmp = saved.with_suffix(".tmp.npz")
            np.savez_compressed(tmp,mask=raw.astype(np.uint16),photo_name=name,ppm=COARSE)
            tmp.replace(saved)
        with np.load(saved) as data:
            mask = data["mask"]
            assert data["ppm"].item()==COARSE and data["photo_name"].item()==name and mask.shape==rgb.shape[:2]
        grains = regionprops_table(mask, properties=("axis_minor_length","area"))
        diam = grains["axis_minor_length"]/COARSE
        selected = diam > 63.
        rows.append(dict(file=name, sample_id=row.sample_id, phone=row.phone, n63=int(selected.sum()),
                         a63=100*grains["area"][selected].sum()/mask.size, max_mm=diam.max(initial=0.)))
        if n <= 11 or n % 5 == 0 or n == len(photos):
            print(f"{n}/{len(photos)} {name}: {rows[-1]['max_mm']:.1f} mm, n63={selected.sum()}, {monotonic()-start:.0f}s",flush=True)
    photos = pd.DataFrame(rows)
    photos.to_csv(out/f"{STEM}_fotograflar.csv",index=False,float_format="%.9f")
    return photos


def evaluate(photos):
    archive = pd.read_csv(out/"ince_paya_2mm_sonuclar.csv").set_index(["exam","sample_id"])
    rows = []
    for exam,hold,ids,src,dst in folds():
        assert set(hold).isdisjoint(ids)
        source = photos if src is None else photos[photos.phone==src]
        target = photos if dst is None else photos[photos.phone==dst]
        x = source.groupby("sample_id").a63.mean().loc[ids].to_numpy()
        area = target.groupby("sample_id").a63.mean().loc[hold]
        y = 100-labels.loc[ids,"63"].to_numpy()
        factor = x.dot(y)/x.dot(x) if x.dot(x)>0 else np.nan
        for sample in hold:
            ref = archive.loc[(exam,sample)]
            base = ref[COLS].to_numpy(float)
            truth = labels.loc[sample,COLS]
            np.testing.assert_allclose(emd(truth, base)[0],ref.emd,rtol=0,atol=1e-7)
            curves = {"mevcut":base,"maske":base.copy(),"tam_gecis":base.copy()}
            if area[sample]>0 and np.isfinite(factor):
                curves["maske"][-2] = np.clip(100-factor*area[sample],base[-3],100)
            curves["tam_gecis"][-2] = 100
            for method,curve in curves.items():
                rows.append(dict(exam=exam,sample_id=sample,model=method,gercek_iri=100-truth["63"],
                                 alan=area[sample],carpan=factor,tahmin_iri=100-curve[-2],emd=emd(truth,curve)[0]))
    result = pd.DataFrame(rows)
    assert len(result)==285 and not result.duplicated(["exam","sample_id","model"]).any()
    result.to_csv(out/f"{STEM}_sonuclar.csv",index=False,float_format="%.9f")
    print(result.groupby(["exam","model"]).emd.mean().unstack().round(4))


def report(photos):
    soils = photos.groupby("sample_id").agg(cekim=("file","size"), adayli_cekim=("n63",lambda n:(n>0).sum()),
                                             en_buyuk_kisa_mm=("max_mm","max"), alan_pct=("a63","mean"))
    soils["lab_63_ustu_pct"] = 100 - labels.loc[soils.index,"63"]
    assert len(soils) == 24 and soils.cekim.sum() == 127
    soils.to_csv(out/f"{STEM}_topraklar.csv",float_format="%.6f")
    # Bütün görüntüler bittikten sonra her pozitif toprağın ve en büyük negatifin örnek çekimi.
    positive = soils.lab_63_ustu_pct > 0
    selected = [*soils.index[positive], soils.loc[~positive,"en_buyuk_kisa_mm"].idxmax()]
    images = list_images(work/"soil-data").set_index("file")
    boxes = pd.read_csv(out/"dinov2_hiyerarsik_kadrajlar.csv",index_col="file")
    fig,axes = plt.subplots(1,len(selected),figsize=(12,3.6),layout="constrained")
    for sample,ax in zip(selected,axes,strict=True):
        group = photos.loc[photos.sample_id==sample].set_index("file")
        name = group.max_mm.idxmax()
        rgb,portrait,_ = surface(images.loc[name],boxes.loc[name])
        with np.load(CACHE/(name+".npz")) as data:
            mask = data["mask"]
        props = regionprops_table(mask,properties=("label","axis_minor_length"))
        if len(props["label"]):
            label = props["label"][np.argmax(props["axis_minor_length"])]
            candidate = cv2.resize(np.uint8(mask==label),rgb.shape[1::-1],interpolation=cv2.INTER_NEAREST)
            contours = cv2.findContours(candidate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            rgb = np.array(rgb,copy=True)
            cv2.drawContours(rgb,contours,-1,(255,198,20),3)
        if portrait:
            rgb=np.rot90(rgb)
        ax.imshow(rgb)
        ax.plot([14,14+63*4.5],[rgb.shape[0]-15]*2,color="black",lw=4)
        ax.plot([14,14+63*4.5],[rgb.shape[0]-15]*2,color="white",lw=2)
        ax.set_axis_off()
        ax.set_title(f"{sample} · {group.loc[name,'max_mm']:.1f} mm".replace(".",","))
    fig.suptitle("Her numunede en büyük görünen kısa eksen   ·   Beyaz çizgi 63 mm")
    fig.savefig(out/f"{STEM}_ornekler.jpg",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    photos = measure()
    evaluate(photos)
    report(photos)
