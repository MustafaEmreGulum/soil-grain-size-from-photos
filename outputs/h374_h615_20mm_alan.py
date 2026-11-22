"""İki toprakta arşivlenmiş tane sınırlarından 20 mm üstü görünür alanı ölç."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

import iri_tas_kalan_doku as prev

ig = prev.ig
stem = ig.out / "h374_h615_20mm_alan"


def main():
    rows = prev.photos().query("sample_id in ['H374', 'H615']")
    assert rows.groupby("sample_id").size().to_dict() == {"H374": 3, "H615": 5}
    records = []
    fig, axes = plt.subplots(len(rows), 5, figsize=(8.8, 13))
    with np.load(prev.ARCHIVE) as archive:
        for row_axes, (name, row) in zip(axes, rows.iterrows()):
            counts = np.zeros(3, dtype=int)
            field_pcts = []
            for ax, mask, (_, offset) in zip(row_axes, archive[name], prev.fields(row)):
                props = regionprops(mask)
                edge_ids = np.unique(np.r_[mask[0], mask[-1], mask[:, 0], mask[:, -1]])
                ids = [p.label for p in props if p.axis_minor_length / ig.PPM > 20]
                large = np.isin(mask, ids)
                inside = large & ig.CORE
                counts += [inside.sum(), (inside & ~np.isin(mask, edge_ids)).sum(),
                           np.count_nonzero(ig.evaluated(mask)[ig.CORE])]
                field_pcts.append(100 * inside.sum() / ig.CORE.sum())
                rgb = ig.crop(row, offset).copy()
                rgb[find_boundaries(large, mode="inner")] = (255, 65, 120)
                ax.imshow(rgb, extent=(0, 52, 52, 0))
                ax.add_patch(Rectangle((13, 13), 26, 26, fill=False, edgecolor="#00eee1", lw=1))
                ax.set(xticks=[], yticks=[])
            short = row.phone.replace("Motorola_Edge_60_fusion", "Edge 60").replace("Motorola_Edge","Edge").replace("Samsung_A52","A52")
            row_axes[0].set_ylabel(f"{row.sample_id} · {short}\nÇekim {name[-6:-4]}", fontsize=9)
            pct = 100 * counts / (5 * ig.CORE.sum())
            records.append(dict(file=name, sample_id=row.sample_id, phone=row.phone,
                                alan20_pct=pct[0], alan20_sinira_degmeyen_pct=pct[1],
                                arsiv63_pct=pct[2], pozitif_kare=np.count_nonzero(field_pcts)))
    frame = pd.DataFrame(records).set_index("file")
    old = pd.read_csv(ig.out / "iri_tas_kalan_doku_fotograflar.csv", index_col="file")
    np.testing.assert_allclose(frame.arsiv63_pct, old.loc[frame.index, "iri_alan_pct"], atol=1e-8)
    frame.to_csv(stem.with_suffix(".csv"), float_format="%.6f")
    print(frame.to_string())
    print(frame.groupby("sample_id")[["alan20_pct","alan20_sinira_degmeyen_pct"]].agg(["mean","median","min","max"]))
    for ax, title in zip(axes[0], ["Merkez", "Yakın 1", "Yakın 2", "Uzak 1", "Uzak 2"]):
        ax.set_title(title, fontsize=10)
    fig.suptitle("H374 ve H615 · 20 mm üzeri taş adayları", fontsize=15)
    fig.text(.52,.955,"Pembe çizgi aday sınırı · Turkuaz karenin içi ölçüme giriyor (26 × 26 mm)",ha="center",fontsize=9)
    fig.subplots_adjust(left=.13,right=.99,top=.918,bottom=.012,hspace=.10,wspace=.035)
    fig.savefig(stem.with_suffix(".png"),dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    main()
