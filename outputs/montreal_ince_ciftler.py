"""Find contrasting fine-sieve labels under one shared, leave-two-soils-out fit per pair."""
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from matplotlib.figure import Figure
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import montreal_ince_genis as exp
from montreal_ince_sabit import passing
from montreal_ince_genis import pilot, source

STEM = "montreal_ince_ciftler"


def mosaic(path):
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
    canvas = Image.new("L", (3 * 122 + 4, 3 * 122 + 4), 255)
    for row, fy in enumerate((.28, .5, .72)):
        for col, fx in enumerate((.28, .5, .72)):
            x = round(fx * gray.width - pilot.SIDE / 2)
            y = round(fy * gray.height - pilot.SIDE / 2)
            tile = gray.crop((x, y, x + pilot.SIDE, y + pilot.SIDE))
            canvas.paste(tile.resize((122, 122), Image.Resampling.LANCZOS), (124 * col, 124 * row))
    return canvas


def main():
    split = source.read_csv(exp.output("secim")).set_index("sample_id")
    added = split[split.role.eq("add_train")].index
    old = source.read_csv(pilot.OUT / (pilot.STEM + "_olculer.csv")).assign(folder=pilot.DATA/"montreal_20mm")
    new = source.read_csv(exp.output("olculer"))
    photos = pd.concat([old, new[new.sample_id.isin(added)].assign(folder=exp.PHOTO)]).query("arm == 'jpeg'")
    x = photos.groupby("sample_id")[pilot.COLS].mean().sort_index()
    y = exp.labels_for(x.index)
    assert len(x) == 37 and set(x.index).isdisjoint(split[split.role.eq("test")].index)
    pairs, details = [], {}
    for a, b in combinations(x.index, 2):
        if y.at[a, pilot.TARGETS[0]] > y.at[b, pilot.TARGETS[0]]:
            a, b = b, a
        train = x.drop([a, b])
        scale = StandardScaler().fit(train)
        model = Ridge(alpha=len(train)).fit(scale.transform(train), y.loc[train.index])
        z = scale.transform(x.loc[[a, b]])
        raw = model.predict(z)
        pred = passing(raw)[:, 0]
        truth = y.loc[[a, b], pilot.TARGETS[0]].to_numpy()
        contributions = (z[1] - z[0]) * model.coef_[0]
        np.testing.assert_allclose(contributions.sum(), raw[1, 0] - raw[0, 0], atol=1e-10)
        pairs.append(dict(a=a, b=b, truth_a=truth[0], truth_b=truth[1], pred_a=pred[0], pred_b=pred[1],
                          truth_gap=truth[1]-truth[0], pred_gap=abs(pred[1]-pred[0]),
                          texture_rms=np.sqrt(np.mean((z[1]-z[0])**2)),
                          positive=contributions.clip(min=0).sum(), negative=contributions.clip(max=0).sum(),
                          projected=not np.allclose(pred, raw[:, 0])))
        details[a, b] = pd.DataFrame(dict(a=a,b=b,feature=pilot.COLS,raw_a=x.loc[a].values,
                                          raw_b=x.loc[b].values, z_a=z[0],z_b=z[1],contribution=contributions))
    pairs = pd.DataFrame(pairs).sort_values(["truth_gap","pred_gap","a","b"], ascending=[False,True,True,True])
    close = pairs[pairs.pred_gap <= 2]
    used, chosen = set(), []
    for r in close.itertuples():
        if r.a in used or r.b in used:
            continue
        chosen.append(r)
        used.update([r.a, r.b])
        if len(chosen) == 3:
            break
    assert len(pairs) == 666 and len(chosen) == 3
    pairs["selected"] = pairs.index.isin([r.Index for r in chosen])
    pairs.to_csv(pilot.OUT / (STEM + "_eslesmeler.csv"), index=False, float_format="%.8f")
    pd.concat([details[r.a, r.b] for r in chosen]).to_csv(
        pilot.OUT / (STEM + "_olculer.csv"), index=False, float_format="%.8f")
    fig = Figure(figsize=(11, 9.8), layout="constrained")
    axes = fig.subplots(3, 3, gridspec_kw={"width_ratios": [1,1,1.45]})
    for axs, r in zip(axes, chosen):
        for ax, soil, truth, pred in zip(axs[:2], (r.a, r.b), (r.truth_a, r.truth_b), (r.pred_a, r.pred_b)):
            photo = photos[photos.sample_id.eq(soil)].sort_values("file").iloc[0]
            ax.imshow(mosaic(photo.folder / photo.file), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{soil}   ·   Gerçek %{truth:.1f}   ·   Tahmin %{pred:.1f}".replace(".",","), fontsize=10)
            ax.set_axis_off()
        ax = axs[2]
        feature = details[r.a, r.b]
        for name, color in ((r.a, "#06625c"), (r.b, "#b35322")):
            z = feature["z_a" if name == r.a else "z_b"]
            ax.plot(range(6), z.iloc[:6], "o-", color=color, label=name)
            ax.plot(range(6,12), z.iloc[6:], "o-", color=color)
        ax.axhline(0, color="0.65", linewidth=.7)
        ax.set_xticks(range(12), ["0,04","0,08","0,16","0,32","0,64","1,28"]*2, rotation=60, fontsize=8)
        ax.set_xlabel("Parlak yapı                 Koyu yapı\nFiltre yarıçapı (mm)", fontsize=9)
        ax.set_ylabel("Standartlaştırılmış ölçü", fontsize=9)
        ax.legend(fontsize=8, ncols=2)
    fig.suptitle("0,08 mm eleği  ·  Her fotoğrafta aynı dokuz bölge, her kare 24 × 24 mm")
    fig.savefig(pilot.OUT / (STEM + ".png"), dpi=135)
    print(f"{len(close)} close pairs; {(close.truth_gap >= 30).sum()} have truth gaps >= 30")
    print(pairs[pairs.selected].to_string(index=False))
    for r in chosen:
        d = details[r.a, r.b]
        print(r.a, r.b, "photos", [photos.sample_id.eq(s).sum() for s in (r.a,r.b)],
              "min/max contributions", d.loc[[d.contribution.idxmin(), d.contribution.idxmax()],
                                               ["feature","contribution"]].to_string(index=False))


if __name__ == "__main__":
    main()
