"""Beş eşit ölçüm karesinden uzun eksendeki ikisini 30 mm'den 100 mm'ye taşı."""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import spearmanr

import imagegrains_bes_bolge as five
import imagegrains_pilot as ig


STEM = "imagegrains_uzak_alan"
DISTANCE = 100
CACHE = five.work.parent / ".venv" / STEM
ARCHIVE = five.out / f"{STEM}_maskeleri.npz"


def layout(row):
    with Image.open(row.path) as image:
        w, h = ImageOps.exif_transpose(image).size
    if w > h:
        return ("sol", "sag"), [(-DISTANCE, 0), (DISTANCE, 0)]
    return ("ust", "alt"), [(0, -DISTANCE), (0, DISTANCE)]


def check_masks(stack):
    assert stack.shape == (2, ig.SIDE, ig.SIDE) and stack.dtype == np.uint16


def infer(photos):
    if ARCHIVE.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    model = ig.load_model(five.WEIGHTS)
    stacks = {}
    for i, (file, row) in enumerate(photos.iterrows(), start=1):
        saved = CACHE / f"{file}.npz"
        if saved.exists():
            with np.load(saved) as z:
                stack = z["masks"]
        else:
            _, offsets = layout(row)
            stack = np.stack([ig.segment(model, ig.crop(row, offset)) for offset in offsets])
            check_masks(stack)
            temp = saved.with_suffix(".tmp.npz")
            np.savez_compressed(temp, masks=stack)
            temp.replace(saved)
        check_masks(stack)
        stacks[file] = stack
        print(f"{i}/{len(photos)}", flush=True)
    temp = ARCHIVE.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **stacks)
    temp.replace(ARCHIVE)


def rank(x, lab):
    # All soil means are separated by at least one pixel or tied.
    return spearmanr(x.round(9), lab).statistic


def plot_layout(photos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    styles = [("Ortak üç alan", "#16dbfa", "-"), ("30 mm'deki iki alan", "#ffb524", "--"),
              ("100 mm'deki iki alan", "#e345c3", "-")]
    for ax, soil in zip(axes, ["H037", "H038"]):
        row = photos[(photos.phone == "Motorola_Edge") & (photos.sample_id == soil)].iloc[0]
        directions, far = layout(row)
        with Image.open(row.path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        ref_side, ref_ppm = five.gsd.PHONES[row.phone]
        ppm = ref_ppm * max(rgb.size) / ref_side
        portrait = rgb.height > rgb.width
        if portrait:
            rgb = rgb.transpose(Image.Transpose.ROTATE_90)
        w, h = np.array(rgb.size) / ppm
        ax.imshow(rgb, extent=(-w/2, w/2, h/2, -h/2))
        shared = [off for name, off in five.OFFSETS.items() if name not in directions]
        for offsets, (_, color, line) in zip([shared, [five.OFFSETS[d] for d in directions], far], styles):
            for x, y in offsets:
                if portrait:
                    x, y = y, -x
                ax.add_patch(Rectangle((x-13, y-13), 26, 26, fill=False, lw=2, edgecolor=color, linestyle=line))
        ax.set_title(f"{soil} · Motorola · ilk çekim", loc="left")
        ax.set_axis_off()
    handles = [Patch(facecolor="none", edgecolor=color, linestyle=line, lw=2, label=label) for label, color, line in styles]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(five.out / f"{STEM}_yerlesim.png", dpi=140)
    plt.close(fig)


def audit_panels(photos):
    rng = np.random.default_rng(20261123)
    soils = sorted(five.audit_selection(photos).sample_id.unique())
    records = []
    for soil in soils:
        index = int(rng.integers(2))
        for phone in five.PHONES:
            row = photos[(photos.sample_id == soil) & (photos.phone == phone)].iloc[0].copy()
            names, offsets = layout(row)
            row["direction"], row["offset"], row["mask_index"] = names[index], offsets[index], index
            records.append(row)
    panels = pd.DataFrame(records).iloc[rng.permutation(len(records))].copy()
    panels.index = pd.Index([f"U{i:02d}" for i in range(1, len(panels) + 1)], name="panel")
    return panels


def audit(photos, prepare):
    panels = audit_panels(photos)
    if prepare:
        panels[["file", "sample_id", "phone", "direction", "mask_index"]].to_csv(five.out / f"{STEM}_panel_secimi.csv")
        ig.blind_panels(panels, CACHE, panels.offset)
        return
    manifest = pd.read_csv(five.out / f"{STEM}_panel_secimi.csv", index_col="panel")
    pd.testing.assert_frame_equal(panels[manifest.columns], manifest)
    annotations = json.loads((five.out / f"{STEM}_referans.json").read_text(encoding="utf8"))
    with np.load(ARCHIVE) as z:
        scores = ig.audit_masks(panels, annotations, lambda panel, r: z[r.file][r.mask_index],
                                five.out / f"{STEM}_maske_kontrolu.png")
    scores.join(panels[["file", "direction"]], on="panel").to_csv(
        five.out / f"{STEM}_maske_kontrolu.csv", index=False, float_format="%.9f")
    print(ig.audit_by_phone(scores).to_string())


def score(photos):
    near = pd.read_csv(five.out / f"{five.STEM}_fotograf_olculeri.csv").set_index("file").loc[photos.index]
    pd.testing.assert_frame_equal(near[["sample_id", "phone"]], photos[["sample_id", "phone"]])
    rows = []
    with np.load(ARCHIVE) as z:
        assert set(z.files) == set(photos.index)
        for file, row in photos.iterrows():
            directions, _ = layout(row)
            stack = z[file]
            check_masks(stack)
            distant = [100 * (ig.evaluated(mask)[ig.CORE] > 0).mean() for mask in stack]
            old = near.loc[file]
            # Recover exact integer pixel counts from the nine-decimal reference CSV.
            old_counts = (old[list(five.OFFSETS)].astype(float) * ig.CORE.sum() / 100).round().astype(int)
            counts = old_counts.copy()
            counts.loc[list(directions)] = np.rint(np.array(distant) * ig.CORE.sum() / 100).astype(int)
            near_pct = 100 * old_counts.sum() / (5 * ig.CORE.sum())
            assert abs(near_pct - old.bes_ortalama) < 1e-8
            rows.append(dict(file=file, sample_id=row.sample_id, phone=row.phone, yon_eksi=directions[0],
                             yon_arti=directions[1], merkez=old.merkez, yakin_eksi=old[directions[0]],
                             yakin_arti=old[directions[1]], uzak_eksi=distant[0], uzak_arti=distant[1],
                             bes_yakin=near_pct, bes_uzak=100 * counts.sum() / (5 * ig.CORE.sum())))
    measures = pd.DataFrame(rows)
    grouped = measures.groupby(["sample_id", "phone"])[["bes_yakin", "bes_uzak"]]
    soils = grouped.mean()
    repeats = (grouped.max() - grouped.min()).where(grouped.size().eq(2), axis=0)
    soils = soils.join(repeats.add_prefix("tekrar_"))
    soils["lab_ust_6_3_pct"] = (100 - five.labels["6.3"]).reindex(soils.index, level="sample_id")
    print("Telefon bazında sıralama, çekim farkı ve tek toprak çıkarılınca sıra kazancı", flush=True)
    for phone in five.PHONES:
        s = soils.xs(phone, level="phone").copy()
        a, b = [rank(s[c], s.lab_ust_6_3_pct) for c in ("bes_yakin", "bes_uzak")]
        changes = []
        for sid in s.index:
            t = s.drop(sid)
            gain = rank(t.bes_uzak, t.lab_ust_6_3_pct) - rank(t.bes_yakin, t.lab_ust_6_3_pct)
            soils.loc[(sid, phone), "cikarilinca_kalan_sira_kazanci"] = gain
            changes.append(gain)
        print(phone, "rank", a, b, "repeat", s[["tekrar_bes_yakin", "tekrar_bes_uzak"]].mean().to_dict(),
              "rank gain omit one", min(changes), max(changes))
    for m in ["bes_yakin", "bes_uzak"]:
        pairs = soils[m].unstack("phone")
        print(m, "mean phone gap", (pairs[five.PHONES[0]] - pairs[five.PHONES[1]]).abs().mean())
    measures.to_csv(five.out / f"{STEM}_fotograflar.csv", index=False, float_format="%.9f")
    soils.to_csv(five.out / f"{STEM}_topraklar.csv", float_format="%.9f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["measure", "blind", "audit", "layout"], default="measure", nargs="?")
    step = parser.parse_args().step
    photos = five.selection()
    if step == "measure":
        infer(photos)
        score(photos)
    elif step == "layout":
        plot_layout(photos)
    else:
        audit(photos, prepare=step == "blind")
