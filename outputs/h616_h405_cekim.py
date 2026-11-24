"""Aynı saklı toprak için tek çekim, bir çekim eksik ve bütün çekim tahminleri."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from maske_kisa_cap_deneyi import adjusted_curve, soil_measures
from v3_knn_esit_karisim import COLS, labels, original, out, predict, training, work
from features import CROP

SAMPLES = ["H616", "H405"]
STEM = "h616_h405_cekim"


def evaluate(samples=SAMPLES):
    areas = pd.read_csv(out / "maske_elek_alanlari.csv").query("split == 'train'").set_index("file")
    archive = pd.read_csv(out / "ince_paya_2mm_sonuclar.csv").query("exam == 'toprak'").set_index("sample_id")
    rows = []
    for sample in samples:
        train = training[training.sample_id != sample]
        photos = original[original.sample_id == sample].sort_values("file")
        assert train.sample_id.nunique() == 23 and photos.file.is_unique
        ref = soil_measures(areas[areas.sample_id != sample], "a_")
        groups = [("birlikte", "", photos)]
        for name in photos.file:
            groups.append(("tek", name, photos[photos.file == name]))
            if len(photos) > 1:
                groups.append(("haric", name, photos[photos.file != name]))
        for phone, group in photos.groupby("phone"):
            groups.append(("telefon", phone, group))
        for mode, name, group in groups:
            raw = predict(train, group)["v3"].loc[sample, COLS].to_numpy()
            obs = soil_measures(areas.loc[group.file], "a_").loc[sample]
            curve = adjusted_curve(raw, ref, obs)
            if mode == "birlikte":
                np.testing.assert_allclose(curve, archive.loc[sample, COLS].to_numpy(float), rtol=0, atol=1e-7)
            rows.append(dict(sample_id=sample, grup=mode, secim=name, n=len(group),
                             gercek_0063=labels.loc[sample, "0.063"], **dict(zip(COLS, curve))))
    return pd.DataFrame(rows)


def pct(value):
    return f"%{value:.2f}".replace(".", ",")


def plot(rows):
    boxes = pd.read_csv(out / "dinov2_hiyerarsik_kadrajlar.csv").set_index("file")
    fig, axes = plt.subplots(4, 3, figsize=(11.4, 9.6))
    fig.subplots_adjust(left=.025, right=.975, bottom=.035, top=.86, wspace=.07, hspace=.66)
    fig.suptitle("0,063 mm eleğinden geçen kütle", y=.98, fontsize=21)
    fig.text(.5, .935, "Turkuaz çerçeve: v3 doku ölçümü     •     Sarı çerçeve: taş maskelerinin alanı",
             ha="center", fontsize=12)
    for i, sample in enumerate(SAMPLES):
        subset = rows[(rows.sample_id == sample) & (rows.grup == "tek")]
        fig.text(.025, .889 - i*.44, f"{sample}    Laboratuvar {pct(subset.gercek_0063.iloc[0])}",
                 fontsize=14, weight="bold")
        for j, phone in enumerate(["Motorola_Edge", "Samsung_A52"]):
            axrow = axes[2*i+j]
            for ax in axrow:
                ax.set_axis_off()
            for ax, (_, row) in zip(axrow, subset[subset.secim.str.startswith(phone)].iterrows()):
                name = row.secim
                path, = (work / "soil-data").rglob(name)
                with Image.open(path) as src:
                    im = ImageOps.exif_transpose(src).convert("RGB")
                w, h = im.size
                cw, ch = int(w*CROP), int(h*CROP)
                x, y = (w-cw)//2, (h-ch)//2
                if w < h:
                    im = im.transpose(Image.Transpose.ROTATE_90)
                    x, y, cw, ch = y, w-x-cw, ch, cw
                bw, bh = im.size
                box = boxes.loc[name]
                assert (bw,bh) == (box.width, box.height)
                im.thumbnail((680, 450))
                ax.imshow(im, extent=(0,bw,bh,0))
                for rect, color in [((box.left,box.top,box.right-box.left,box.bottom-box.top),"#ffce53"),
                                    ((x,y,cw,ch),"#00e1df")]:
                    ax.add_patch(Rectangle(rect[:2], *rect[2:], edgecolor=color, fill=False, lw=1.8))
                ax.set_title(f"{phone.split('_')[0]} {name[-6:-4]}   ·   Tahmin {pct(row['0.063'])}", fontsize=10)
    fig.savefig(out / f"{STEM}.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    results = evaluate()
    results.to_csv(out / f"{STEM}.csv", index=False, float_format="%.8f")
    plot(results)
    print(results[["sample_id","grup","secim","0.063","2","gercek_0063"]].round(3).to_string(index=False))
