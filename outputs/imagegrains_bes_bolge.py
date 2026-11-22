"""Tek ve beş örtüşmeyen fiziksel alan ile iri taş alan payının tekrarlanabilirliği."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import imagegrains_pilot as ig
from v3_knn_esit_karisim import labels, out, work
from v3_sabit_kadraj import PHONES
import gsd


OFFSETS = {"merkez": (0, 0), "sol": (-30, 0), "sag": (30, 0), "ust": (0, -30), "alt": (0, 30)}
STEM = "imagegrains_bes_bolge"
CACHE = work.parent / ".venv" / STEM
WEIGHTS = work.parent / ".venv/IG2_full_set_cp_SAM"
ARCHIVE = out / f"{STEM}_maskeleri.npz"
REFERENCE = out / f"{STEM}_referans.json"


def selection():
    old = set(pd.read_csv(ig.PILOT).sample_id)
    photos = gsd.list_images(str(work / "soil-data")).query("split == 'train'")
    pairs = set.intersection(*(set(photos.loc[photos.phone == p, "sample_id"]) for p in PHONES)) - old
    available = photos[photos.sample_id.isin(pairs) & photos.phone.isin(PHONES)]
    selected = available.sort_values(["sample_id", "phone", "file"]).groupby(["sample_id", "phone"]).head(2).copy()
    assert len(pairs) == 15 and len(selected) == 58 and selected.file.is_unique
    assert selected.groupby("sample_id").phone.nunique().eq(2).all()
    return selected.set_index("file", drop=False)


def audit_selection(photos):
    rng = np.random.default_rng(20261122)
    coarse = (100 - labels.loc[sorted(photos.sample_id.unique()), "6.3"]).sort_values(kind="stable")
    strata = np.array_split(coarse.index.to_numpy(), 3)
    ids = [s for group, n in zip(strata, (2, 2, 4)) for s in rng.choice(group, n, replace=False)]
    rows = []
    for soil in ids:
        region = rng.choice(list(OFFSETS)[1:])
        for phone in PHONES:
            row = photos[(photos.sample_id == soil) & (photos.phone == phone)].iloc[0].copy()
            row["region"] = region
            rows.append(row)
    panels = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].copy()
    panels.index = pd.Index([f"B{i:02d}" for i in range(1, len(rows)+1)], name="panel")
    return panels


def prepare(photos):
    panels = audit_selection(photos)
    panels[["sample_id", "phone", "file", "region"]].to_csv(out / f"{STEM}_panel_secimi.csv")
    photos[["sample_id", "phone", "file"]].to_csv(out / f"{STEM}_secimi.csv", index=False)
    ig.blind_panels(panels, CACHE, panels.region.map(OFFSETS))


def infer(photos):
    CACHE.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        with np.load(ARCHIVE) as archive:
            assert set(archive.files) == set(photos.file)
            for key in photos.file:
                validate_stack(archive[key])
        return
    model = ig.load_model(WEIGHTS)
    for i, (file, row) in enumerate(photos.iterrows(), start=1):
        saved = CACHE / f"{file}.npz"
        if not saved.exists():
            masks = np.stack([ig.segment(model, ig.crop(row, pos)) for pos in OFFSETS.values()])
            validate_stack(masks)
            tmp = saved.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, masks=masks)
            tmp.replace(saved)
        print(f"{i}/{len(photos)} photos", flush=True)
    all_masks = {}
    for file in photos.file:
        with np.load(CACHE / f"{file}.npz") as item:
            all_masks[file] = item["masks"]
            validate_stack(all_masks[file])
    tmp = ARCHIVE.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **all_masks)
    tmp.replace(ARCHIVE)


def validate_stack(masks):
    assert masks.shape == (len(OFFSETS), ig.SIDE, ig.SIDE) and masks.dtype == np.uint16


def photo_scores(photos, archive):
    rows = []
    for file, row in photos.iterrows():
        masks = archive[file]
        validate_stack(masks)
        areas = {region: 100 * np.mean(ig.evaluated(mask)[ig.CORE] > 0)
                 for region, mask in zip(OFFSETS, masks)}
        rows.append(dict(file=file, sample_id=row.sample_id, phone=row.phone, **areas,
                         bes_ortalama=np.mean(list(areas.values()))))
    return pd.DataFrame(rows)


def soil_scores(areas):
    soils = pd.DataFrame(index=sorted(areas.sample_id.unique()))
    soils.index.name = "sample_id"
    soils["lab_ust_6_3_pct"] = 100 - labels.loc[soils.index, "6.3"]
    measures = ["merkez", "bes_ortalama"]
    for phone in PHONES:
        subset = areas[areas.phone == phone].groupby("sample_id")[measures]
        soils[[f"{phone}_{m}" for m in measures]] = subset.mean()
        # Each soil contributes one absolute difference from its first two photos.
        repeats = (subset.max() - subset.min()).where(subset.size().eq(2), axis=0)
        soils[[f"{phone}_tekrar_{m}" for m in measures]] = repeats
    for m in measures:
        soils[f"telefon_farki_{m}"] = (soils[f"{PHONES[0]}_{m}"] - soils[f"{PHONES[1]}_{m}"]).abs()
    gain = soils.telefon_farki_merkez - soils.telefon_farki_bes_ortalama
    soils["kazanc_pp"] = gain
    soils["cikarilinca_kalan_kazanc_pp"] = (gain.sum() - gain) / (len(gain) - 1)
    return soils


def score(photos):
    panels = audit_selection(photos)
    manifest = pd.read_csv(out / f"{STEM}_panel_secimi.csv", index_col="panel")
    pd.testing.assert_frame_equal(panels[manifest.columns], manifest)
    panels["offset"] = panels.region.map(OFFSETS)
    annotations = json.loads(REFERENCE.read_text(encoding="utf-8"))
    with np.load(ARCHIVE) as archive:
        assert set(archive.files) == set(photos.file)
        areas = photo_scores(photos, archive)
        def panel_mask(panel, row):
            return archive[row.file][list(OFFSETS).index(row.region)]
        audit = ig.audit_masks(panels, annotations, panel_mask, out / f"{STEM}_karsilastirma.png")
    audit = audit.join(panels[["region", "file"]], on="panel")
    mask_stats = ig.audit_by_phone(audit)
    soils = soil_scores(areas)
    assert len(soils) == 15 and len(audit) == 16 and audit.groupby("phone").size().eq(8).all()
    rows = []
    for phone in PHONES:
        for m in ["merkez", "bes_ortalama"]:
            repeats = soils[f"{phone}_tekrar_{m}"].dropna()
            # Equal pixel counts must retain tied ranks across averaging paths.
            ranks = spearmanr(soils.lab_ust_6_3_pct, soils[f"{phone}_{m}"].round(10))
            rows.append(dict(phone=phone, method=m, rho=ranks.statistic,
                             repeat_n=len(repeats), repeat_gap_pp=repeats.mean()))
    comparisons = pd.DataFrame(rows)
    rho = comparisons.pivot(index="phone", columns="method", values="rho")
    gap_pass = (soils.kazanc_pp.mean() > 0) and (soils.cikarilinca_kalan_kazanc_pp > 0).all()
    rank_pass = ((rho.bes_ortalama > 0) & (rho.bes_ortalama >= rho.merkez)).all()
    for table, name in [(areas, "fotograf_olculeri"), (soils.reset_index(), "toprak_sonuclari"),
                        (audit, "maske_kontrolu"), (comparisons, "telefon_sonuclari")]:
        table.to_csv(out / f"{STEM}_{name}.csv", index=False, float_format="%.9f")
    print(soils[["telefon_farki_merkez", "telefon_farki_bes_ortalama", "kazanc_pp",
                 "cikarilinca_kalan_kazanc_pp"]].describe().to_string())
    print(comparisons.to_string(index=False))
    print(mask_stats.to_string())
    print("Lower phone gap with delete-one stability:", gap_pass, "lab ranks:", rank_pass,
          "masks:", mask_stats.passed.all(), "advance:", gap_pass and rank_pass and mask_stats.passed.all())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "infer", "score"])
    args = parser.parse_args()
    photos = selection()
    if args.step == "prepare":
        prepare(photos)
    elif args.step == "infer":
        infer(photos)
    else:
        score(photos)
