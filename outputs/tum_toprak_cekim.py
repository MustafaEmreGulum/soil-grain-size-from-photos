"""24 saklı toprakta 0,063 mm hatasının çekime bağlılığını ölç."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from h616_h405_cekim import evaluate, labels, original, out

STEM = "tum_toprak_cekim"


def summarize(rows):
    records = []
    for sample, group in rows.groupby("sample_id"):
        singles = group[group.grup == "tek"]
        omissions = group[group.grup == "haric"]
        n = len(singles)
        assert singles.secim.is_unique and n == len(original[original.sample_id == sample])
        assert set(omissions.secim) == (set(singles.secim) if n > 1 else set())
        truth = labels.loc[sample, "0.063"]
        pred = group.loc[group.grup == "birlikte", "0.063"].item()
        errors = singles["0.063"] - truth
        above, below = (errors > 1e-7).sum(), (errors < -1e-7).sum()
        if above == n:
            direction = "fazla"
        elif below == n:
            direction = "eksik"
        elif above and below:
            direction = "iki_yon"
        else:
            direction = "esitlik"
        records.append(dict(sample_id=sample, n=n, gercek=truth, birlikte=pred,
                            hata=pred-truth, yon=direction, fazla=above, eksik=below,
                            tek_min=singles["0.063"].min(), tek_max=singles["0.063"].max(),
                            haric_min=omissions["0.063"].min(), haric_max=omissions["0.063"].max(),
                            en_buyuk_degis=(omissions["0.063"]-pred).abs().max()))
    return pd.DataFrame(records).set_index("sample_id")


def plot(rows, summary):
    fig, ax = plt.subplots(figsize=(9, 8))
    for y, (sample, s) in enumerate(summary.sort_values("hata").iterrows()):
        points = rows.loc[(rows.sample_id == sample) & (rows.grup == "tek"), "0.063"] - s.gercek
        ax.plot([points.min(), points.max()], [y, y], color="#b7c6d0", linewidth=2)
        ax.scatter(points, np.full(len(points), y), color="#718d9e", s=17, zorder=2,
                   label="Tek çekim" if y == 0 else None)
        ax.scatter(s.hata, y, color="#a3480a", marker="D", s=31, zorder=3,
                   label="Tüm çekimler birlikte" if y == 0 else None)
    ax.axvline(0, color="#25282e", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(summary)), summary.sort_values("hata").index)
    ax.set_xlabel("Tahmin − laboratuvar (yüzde puan)     •     Solda eksik, sağda fazla")
    ax.set_title("0,063 mm eleğinde çekimlere göre tahmin hatası", loc="left", pad=14)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"{STEM}.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    rows = evaluate(labels.index)
    assert len(rows.query("grup == 'tek'")) == 127
    summary = summarize(rows)
    assert len(summary) == 24
    rows.to_csv(out / f"{STEM}_tahminler.csv", index=False, float_format="%.8f")
    summary.to_csv(out / f"{STEM}_ozet.csv", float_format="%.8f")
    plot(rows, summary)
    print(summary.round(2).to_string())
