"""Üç test toprağında maske hizası, çekimler arası fark ve alan-kütle dönüşümü."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries

from coklu_elek_deneyi import OUT, area_updates, bounds, labels
from gsd import COLS, PHONES

SAMPLES = {"HPC_Testfeld Lidl WHV": ("Testfeld", "testfeld"),
           "HPC_Audorfring": ("Audorfring", "audorfring"),
           "HPC_Muenster_BS6_9_0-10m": ("Münster", "muenster")}
SIEVES = ("2", "6.3")
COLORS = [(0, 255, 215), (255, 186, 45)]


def panel(path, row, mask):
    with Image.open(path) as src:
        photo = ImageOps.exif_transpose(src).convert("RGB")
    w, h = photo.size
    box = (row.left, row.top, row.right, row.bottom)
    if h > w:
        box = (w-row.bottom, row.left, w-row.top, row.right)
    roi = photo.crop(box)
    ref, ppm = PHONES[row.phone]
    dims = tuple(round(n * 4.5 / (ppm * max(w, h) / ref)) for n in roi.size)
    assert dims == mask.shape[::-1]
    rgb = np.asarray(roi.resize(dims, Image.Resampling.LANCZOS))
    if h > w:
        rgb, mask = np.rot90(rgb), np.rot90(mask)
    props = regionprops_table(mask, properties=("label", "axis_minor_length", "area"))
    images = [rgb]
    for sieve, color in zip(SIEVES, COLORS):
        selected = props["axis_minor_length"] > float(sieve) * 4.5
        keep = np.zeros(int(mask.max()) + 1, dtype=bool)
        keep[props["label"][selected]] = True
        labeled = np.where(keep[mask], mask, 0)
        assert selected.sum() == row["n_" + sieve]
        np.testing.assert_allclose(100 * np.count_nonzero(labeled) / mask.size, row["a_" + sieve], atol=1e-8)
        overlay = rgb.copy()
        overlay[find_boundaries(labeled, mode="thick")] = color
        images.append(overlay)
    return images


def main(archive):
    areas = pd.read_csv(OUT / "maske_elek_alanlari.csv")
    ref = areas.query("split == 'train'").groupby("sample_id").mean(numeric_only=True)
    frame = areas[areas.sample_id.isin(SAMPLES)].merge(pd.read_csv(OUT / "test20_fotograflar.csv"),
        on=["file", "phone"], validate="one_to_one").sort_values("file")
    paths = {p.name:p for p in (OUT.parent / "work/soil-data/Test_All_Photos").rglob("*.JPG")}
    old = pd.read_csv(OUT / "submission_v3_tam_yuzey_20mm.csv", index_col="sample_id")
    new = pd.read_csv(OUT / "submission_v3_tam_yuzey_2_6_3_20mm.csv", index_col="sample_id")
    results = []
    assert len(frame) == 14
    overview, preview = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    overview.suptitle("6,3 mm eşiğini geçen adayların sınırları", fontsize=14)
    with ZipFile(archive) as z:
        for sample, (name, slug) in SAMPLES.items():
            shots = frame[frame.sample_id == sample].set_index("file")
            base = old.loc[sample, COLS].values
            fig, axes = plt.subplots(len(shots), 3, figsize=(12, 2.65*len(shots)+.6), layout="constrained")
            fig.suptitle(name + "   |   Aynı kadrajda iki elek ölçümü", fontsize=17)
            for i, (file, shot) in enumerate(shots.iterrows()):
                with np.load(BytesIO(z.read(file + ".npz"))) as data:
                    images = panel(paths[file], shot, data["mask"])
                if i == len(shots) - 1:
                    tile = preview[list(SAMPLES).index(sample)]
                    tile.imshow(images[2])
                    tile.set_title(f"{name} · Çekim {i+1}", fontsize=11)
                    tile.set_axis_off()
                for ax, img, title in zip(axes[i], images, ["Fotoğraf", "Kısa eksen > 2 mm", "Kısa eksen > 6,3 mm"]):
                    ax.imshow(img)
                    ax.set_axis_off()
                    if title == "Fotoğraf":
                        caption = f"Çekim {i+1}"
                    else:
                        s = "2" if title.endswith(" 2 mm") else "6.3"
                        caption = f"Alan %{shot['a_'+s]:.1f}  ·  {int(shot['n_'+s])} aday".replace(".", ",")
                    ax.set_title((title + "\n" if i == 0 else "") + caption, fontsize=10)
            fig.savefig(OUT / f"uc_toprak_elek_{slug}.jpg", dpi=130, bbox_inches="tight")
            plt.close(fig)
            for omitted in [None, *shots.index]:
                active = shots if omitted is None else shots.drop(omitted)
                mean = active.mean(numeric_only=True)
                raw = area_updates(ref, mean, labels.loc[ref.index])
                pred = bounds(base, {s:raw[s] for s in SIEVES})
                if omitted is None:
                    np.testing.assert_allclose(pred, new.loc[sample, COLS], atol=.000051, rtol=0)
                for s in SIEVES:
                    x = ref["a_"+s]
                    mass = 100 - labels.loc[x.index,s]
                    nearest = (x - mean["a_"+s]).abs().idxmin()
                    results.append(dict(toprak=name, haric=omitted or "yok", elek_mm=s,
                        alan_pct=mean["a_"+s], alan_min=active["a_"+s].min(), alan_max=active["a_"+s].max(),
                        egitim_ust=x.max(), carpan=x.dot(mass)/x.dot(x), once=old.loc[sample,s],
                        sinir_oncesi=raw[s], sonra=pred[COLS.index(s)], alt_sinir=old.loc[sample,"0.63"],
                        ust_sinir=old.loc[sample,"20"], en_yakin=nearest, yakin_alan=x.loc[nearest],
                        yakin_lab_gecis=labels.loc[nearest,s]))
    overview.savefig(OUT / "uc_toprak_elek_ornek.jpg", dpi=130, bbox_inches="tight")
    plt.close(overview)
    pd.DataFrame(results).to_csv(OUT / "uc_toprak_elek_kontrol.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="soil_tane_maskeleri.zip")
    main(parser.parse_args().archive)
