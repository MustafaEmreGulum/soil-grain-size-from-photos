"""Aynı Montréal bölgelerindeki yüksek frekanslı bilginin iki ince eleğe etkisi."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from matplotlib.figure import Figure
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
DATA = OUT.parent/"work"/"soil-data"
sys.path.insert(0,str(OUT.parent/"work"/"vitalclick"/"src"))
import granulometry as gran

NATIVE = 1000/39.4
LOW = 4.57
SIDE = round(24*NATIVE)
SMALL = round(SIDE*LOW/NATIVE)
RADII = np.array([.04,.08,.16,.32,.64,1.28])
COLS = [f"{kind}_{r}" for kind in ("bright","dark") for r in RADII]
TARGETS = ["%_0.08mm","%_0.16mm"]
STEM = "montreal_ince_cozunurluk"


def texture(tile):
    norm = gran.prepare(np.asarray(tile,dtype=np.float32),NATIVE)
    opened,closed,mean = gran.spectrum(norm,NATIVE,radii_mm=RADII)
    return np.r_[mean-opened,closed-mean]


def measure(files):
    rows = []
    for i,path in enumerate(files,1):
        with Image.open(path) as source:
            gray = ImageOps.exif_transpose(source).convert("L")
            assert gray.size == (8192,5464)
        values = {"jpeg":[],"kucuk":[]}
        for fy in (.28,.5,.72):
            for fx in (.28,.5,.72):
                x,y = round(fx*gray.width-SIDE/2),round(fy*gray.height-SIDE/2)
                tile = gray.crop((x,y,x+SIDE,y+SIDE))
                small = tile.resize((SMALL,SMALL),Image.Resampling.LANCZOS)
                # Both arms now use identical filter shapes and rounding. The
                # upsampling only returns the reduced image to the common grid.
                restored = small.resize(tile.size,Image.Resampling.BICUBIC)
                values["jpeg"].append(texture(tile))
                values["kucuk"].append(texture(restored))
        for arm,items in values.items():
            rows.append(dict(sample_id=path.name.split("_")[0],file=path.name,arm=arm,
                             **dict(zip(COLS,np.mean(items,axis=0)))))
        if i%7==0:
            print(f"{i}/{len(files)}",flush=True)
    return pd.DataFrame(rows)


def evaluate(features,truth):
    rows = []
    for arm,block in features.groupby("arm"):
        soil = block.groupby("sample_id")[COLS].mean().loc[truth.index]
        for held in soil.index:
            train = soil.drop(held)
            y = truth.loc[train.index]
            scale = StandardScaler().fit(train)
            model = Ridge(alpha=len(train)).fit(scale.transform(train),y)
            p = np.clip(model.predict(scale.transform(soil.loc[[held]]))[0],0,100)
            if p[0]>p[1]:
                p[:]=p.mean()
            row = dict(sample_id=held,arm=arm,n_ref=len(train),n_photos=int((block.sample_id==held).sum()))
            for j,s in enumerate(("008","016")):
                row[f"truth_{s}"]=truth.loc[held].iloc[j]
                row[f"pred_{s}"]=p[j]
                row[f"mae_{s}"]=abs(p[j]-row[f"truth_{s}"])
                row[f"median_control_mae_{s}"]=abs(y.median().iloc[j]-row[f"truth_{s}"])
            rows.append(row)
    return pd.DataFrame(rows)


def show_resolution(files):
    fig = Figure(figsize=(7, 6.8))
    axes = fig.subplots(2, 2)
    for row, sample in enumerate(("0196", "0201")):
        path = next(p for p in files if p.name.startswith(sample + "_"))
        with Image.open(path) as source:
            gray = ImageOps.exif_transpose(source).convert("L")
        x, y = round((gray.width-SIDE)/2), round((gray.height-SIDE)/2)
        tile = gray.crop((x, y, x+SIDE, y+SIDE))
        small = tile.resize((SMALL, SMALL), Image.Resampling.LANCZOS)
        for ax, view in zip(axes[row], (tile, small)):
            ax.imshow(view, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                      extent=(0, 24, 24, 0))
            ax.set_xlim(6, 18)
            ax.set_ylim(18, 6)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel("Toprak " + sample)
    axes[0, 0].set_title("Özgün JPEG · 25,38 piksel/mm")
    axes[0, 1].set_title("Küçültülmüş · 4,58 piksel/mm")
    fig.suptitle("Aynı çekimin merkezindeki 12 × 12 mm alan")
    fig.tight_layout()
    fig.savefig(OUT/f"{STEM}_detay.png", dpi=150)


def main():
    files = sorted((DATA/"montreal_20mm").glob("*_dry_tray*.jpg"))
    meta = pd.read_csv(DATA/"psd_mass_path_80_to_0_08mm_no_ballast.csv",dtype={"sample_id":str})
    meta["file"] = meta.image_path.map(lambda p:Path(p).name)
    chosen = meta[meta.file.isin([p.name for p in files])].copy()
    assert len(files)==len(chosen)==28 and chosen.file.is_unique and chosen.sample_id.nunique()==17
    assert chosen.dry_or_moist.eq("dry").all() and chosen.file.str[:4].equals(chosen.sample_id)
    assert chosen.groupby("sample_id")[TARGETS].nunique().eq(1).all().all()
    truth = chosen.groupby("sample_id")[TARGETS].first().sort_index()
    assert truth.ge(0).all().all() and truth.le(100).all().all() and truth.iloc[:,0].le(truth.iloc[:,1]).all()
    cv2.setNumThreads(1)
    features = measure(files)
    assert np.isfinite(features[COLS]).all().all() and not features.duplicated(["file","arm"]).any()
    features.to_csv(OUT/f"{STEM}_olculer.csv",index=False,float_format="%.10f")
    scores = evaluate(features,truth)
    assert len(scores)==34 and scores.n_ref.eq(16).all()
    scores.to_csv(OUT/f"{STEM}_sonuclar.csv",index=False,float_format="%.9f")
    show_resolution(files)
    print(scores.groupby("arm")[
        ["mae_008","mae_016","median_control_mae_008","median_control_mae_016"]].mean().round(4))


if __name__=="__main__":
    with threadpool_limits(1):
        main()
