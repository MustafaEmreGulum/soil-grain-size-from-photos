"""Trace the controlled 63 mm submission's held-out cumulative errors from coarse to fine."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "work/vitalclick/src"))
import gsd


def main():
    base = pd.read_csv(OUT / "ince_paya_2mm_sonuclar.csv").query("exam == 'toprak'").set_index("sample_id")
    masks = pd.read_csv(OUT / "ikinci_olcek_egitim_fotograflar.csv")
    truth = pd.read_csv(OUT.parent / "work/soil-data/Training_labels_updated.csv", index_col="sample_id")
    assert len(base) == len(truth) == 24 and base.index.is_unique and truth.index.is_unique
    assert set(base.index) == set(truth.index) == set(masks.sample_id)
    assert len(masks) == 127 and masks.file.is_unique and masks.n63.eq(0).all()
    pred = base[gsd.COLS].copy()
    pred["63"] = 100.
    archive = pd.read_csv(OUT / "tekrar_cekimi_genel_emd.csv").query(
        "exam == 'toprak' and model == 'mevcut'").set_index("sample_id").loc[pred.index]
    truth = truth.loc[pred.index, gsd.COLS]
    np.testing.assert_allclose(pred, archive[gsd.COLS], atol=1e-7, rtol=0)
    error = pred - truth
    contribution = error.abs().mul(np.r_[gsd.DLOG, 0.], axis=1)
    np.testing.assert_allclose(contribution.sum(axis=1), archive.emd, atol=1e-7, rtol=0)
    np.testing.assert_allclose(contribution.sum(axis=1), gsd.emd(truth, pred))
    rows = []
    for soil in pred.index:
        previous = 0.
        for i in reversed(range(len(gsd.COLS))):
            col = gsd.COLS[i]
            signed = error.loc[soil, col]
            rows.append(dict(sample_id=soil, elek_mm=col, gercek=truth.loc[soil, col],
                             tahmin=pred.loc[soil, col], hata=signed, onceki_hata_farki=signed-previous,
                             aralik_ust_mm=gsd.COLS[i+1] if i+1 < len(gsd.COLS) else None,
                             emd_katkisi=contribution.loc[soil, col], toplam_emd=archive.at[soil, "emd"]))
            previous = signed
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "kaggle_elek_hata_izi.csv", index=False, float_format="%.9f")
    focus = ["H037", "H668"]
    print(long[long.sample_id.isin(focus)].round(2).to_string(index=False))
    print("Mean contribution per lower sieve", contribution.mean().round(3).to_dict())
    print("Mean EMD", contribution.sum(axis=1).mean())
    for threshold in (5, 10):
        first = long[long.hata.abs().ge(threshold)].groupby("sample_id", sort=False).first()
        print("First at", threshold, first.elek_mm.value_counts().to_dict(), first.loc[focus,"elek_mm"].to_dict())
    for soil in focus + ["mean"]:
        c = contribution.mean() if soil == "mean" else contribution.loc[soil]
        print(soil, "share .02 to .63", c[["0.02","0.063","0.2"]].sum()/c.sum(),
              "share .063 to .63", c[["0.063","0.2"]].sum()/c.sum())
    fig = Figure(figsize=(10, 6.4), layout="constrained")
    axes = fig.subplots(2, 2, height_ratios=[1.7, 1])
    pos = np.arange(len(gsd.COLS))
    for j, soil in enumerate(focus):
        ax = axes[0,j]
        for curve, color, label in ((truth, "#156960", "Laboratuvar"), (pred, "#bd5726", "Saklı tahmin")):
            ax.plot(pos, curve.loc[soil].to_numpy()[::-1], "o-", color=color, label=label, markersize=4)
        ax.set(title=soil, ylabel="Elekten geçen kütle (%)", ylim=(-3,103), xticks=[])
        ax.legend(fontsize=8)
        bar = axes[1,j]
        bar.bar(pos, contribution.loc[soil].to_numpy()[::-1], color="#346680")
        bar.set(ylabel="Aralığın EMD katkısı", xlabel="Alt sınırdaki elek (mm)",
                xticks=pos, xticklabels=[c.replace(".",",") for c in gsd.COLS[::-1]],
                ylim=(0,contribution.loc[focus].to_numpy().max()*1.15))
        bar.tick_params(axis="x", rotation=55)
    fig.suptitle("Kaggle eğitim topraklarında büyük elekten küçüğe hata")
    fig.savefig(OUT / "kaggle_elek_hata_izi.png", dpi=145)


if __name__ == "__main__":
    main()
