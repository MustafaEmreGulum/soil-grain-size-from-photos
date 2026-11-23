"""Saklı topraklarda iki orta elek modelinin EMD'sini elek bazında ayır."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "work/vitalclick/src"))
from gsd import COLS, DLOG


def main():
    lab = pd.read_csv(OUT.parent / "work/soil-data/Training_labels_updated.csv", index_col="sample_id")[COLS]
    base = pd.read_csv(OUT / "coklu_elek_sonuclar.csv").query("model == 'v3+20'")
    candidates = pd.read_csv(OUT / "orta_elek_kismi_sonuclar.csv")
    keys = ["exam", "sample_id"]
    base = base.set_index(keys, verify_integrity=True)
    weights = np.r_[DLOG, 0.]
    records = []
    for model in ("aralik", "yari"):
        rows = candidates[candidates.model == model].set_index(keys, verify_integrity=True)
        assert len(rows) == 71 and rows.index.difference(base.index).empty
        pred = base.loc[rows.index, COLS].copy()
        np.testing.assert_allclose(pred[["0.63","20"]], rows[["0.63","20"]], atol=1e-7, rtol=0)
        pred[["2","6.3"]] = rows[["2","6.3"]]
        actual = lab.loc[pred.index.get_level_values("sample_id")]
        err = pred - actual.to_numpy()
        contribution = err.abs() * weights
        np.testing.assert_allclose(contribution.sum(axis=1), rows.emd, rtol=0, atol=1e-7)
        for s in COLS:
            records.append(pd.DataFrame(dict(model=model, elek_mm=float(s), lab_pct=actual[s].to_numpy(),
                tahmin_pct=pred[s], hata_pp=err[s], emd_katkisi=contribution[s])).reset_index())
    detail = pd.concat(records, ignore_index=True)
    detail.to_csv(OUT / "iki_model_elek_toprak.csv", index=False, float_format="%.9f")
    main_exam = detail[detail.exam == "24"]
    assert main_exam.sample_id.nunique() == len(lab) == 24
    excluded = main_exam[~main_exam.sample_id.isin(["H037","H038"])].assign(exam="22_H037_H038_haric")
    groups = pd.concat([detail, excluded]).groupby(["exam","model","elek_mm"])
    summary = groups.agg(n=("sample_id","size"), emd=("emd_katkisi","mean"),
        ort_mutlak_pp=("hata_pp",lambda e:e.abs().mean()), ortanca_mutlak_pp=("hata_pp",lambda e:e.abs().median()),
        yanlilik_pp=("hata_pp","mean"), fazla=("hata_pp",lambda e:(e>1e-7).sum()),
        eksik=("hata_pp",lambda e:(e< -1e-7).sum()), en_az_10_pp=("hata_pp",lambda e:(e.abs()>=10).sum())).reset_index()
    summary["pay_pct"] = 100 * summary.emd / summary.groupby(["exam","model"]).emd.transform("sum")
    summary.to_csv(OUT / "iki_model_elek_ozet.csv", index=False, float_format="%.8f")

    errors = main_exam[(main_exam.model=="aralik") & (main_exam.elek_mm<200)].pivot(
        index="sample_id", columns="elek_mm", values="hata_pp")
    errors = errors.loc[errors.abs().mul(DLOG).sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8,7))
    limit = np.abs(errors.to_numpy()).max()
    im = ax.imshow(errors, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(errors.columns)), [f"{s:g}".replace(".",",") for s in errors.columns])
    ax.set_yticks(range(len(errors)), errors.index, fontsize=8)
    ax.set_xlabel("Elek açıklığı (mm)")
    ax.set_title("Üç boyut aralığı modeli · 24 saklı toprak")
    fig.colorbar(im, ax=ax, label="Tahmin − laboratuvar (yüzde puan)")
    fig.tight_layout()
    fig.savefig(OUT / "iki_model_elek_haritasi.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
