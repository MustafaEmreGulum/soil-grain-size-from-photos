"""Dondurulmuş 17 topraklık ince doku modelini yeni Montréal numunelerinde sına."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from matplotlib.figure import Figure
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import montreal_ince_cozunurluk as pilot

STEM = "montreal_ince_bagimsiz"
SEED = 20261125
COUNT = 20
META = pilot.DATA / "psd_mass_path_80_to_0_08mm_no_ballast.csv"
PHOTO = pilot.DATA / STEM
BASE = "https://www.frdr-dfdr.ca/repo/files/1/published/publication_1311/submitted_data/images_soil/"


def read_csv(path, **kwargs):
    return pd.read_csv(path, dtype={"sample_id": str}, **kwargs)


def selection():
    metadata = read_csv(META, usecols=["sample_id", "dry_or_moist", "image_path"])
    previous = set()
    for name in ("montreal_pilot_referans", "montreal_egitim_secimi", pilot.STEM + "_olculer"):
        previous.update(read_csv(pilot.OUT / (name + ".csv"), usecols=["sample_id"]).sample_id)
    dry = metadata[metadata.dry_or_moist.eq("dry")].copy()
    dry["file"] = dry.image_path.map(lambda path: Path(path).name)
    dry = dry[dry.file.str.contains(r"_(?:bulk|split[^_]*)_", case=False)]
    candidates = (dry[~dry.sample_id.isin(previous)].sort_values("file")
                  .drop_duplicates("sample_id").set_index("sample_id").sort_index())
    ids = sorted(np.random.default_rng(SEED).choice(candidates.index, COUNT, replace=False))
    selected = candidates.loc[ids, ["file"]]
    assert selected.file.is_unique and not set(ids).intersection(previous)
    path = pilot.OUT / (STEM + "_secim.csv")
    if path.exists():
        pd.testing.assert_frame_equal(read_csv(path).set_index("sample_id"), selected)
    else:
        selected.to_csv(path)
    print(f"Excluded {len(previous)} old soils; selected {COUNT}/{len(candidates)}: {ids}", flush=True)
    return selected


def download(name, folder=PHOTO):
    target = folder / name
    if target.is_file():
        return
    part = target.with_suffix(".part")
    for attempt in range(3):
        try:
            with requests.get(BASE + quote(name), stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                with part.open("wb") as dest:
                    for chunk in response.iter_content(1024 * 1024):
                        dest.write(chunk)
            with Image.open(part) as image:
                assert image.size == (8192, 5464) and image.format == "JPEG"
                image.verify()
            part.replace(target)
            print("Downloaded", name, flush=True)
            return
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def plot_predictions(results):
    fig = Figure(figsize=(9, 4.6), layout="constrained")
    for ax, sieve, title in zip(fig.subplots(1, 2), ("008", "016"), ("0,08 mm", "0,16 mm")):
        for arm, label, marker, color in (("kucuk", "Küçültülmüş", "x", "#ae6b2f"),
                                          ("jpeg", "Özgün", "o", "#176a73")):
            rows = results[results.arm.eq(arm)]
            ax.scatter(rows["truth_" + sieve], rows["pred_" + sieve], marker=marker,
                       color=color, alpha=0.8, s=25, label=label)
        median = results.loc[results.arm.eq("ortanca"), "pred_" + sieve].iloc[0]
        ax.axhline(median, color="#6c507a", linestyle="--", label="Eğitim ortancası")
        ax.plot([0, 100], [0, 100], color="#555555", linewidth=1, label="Tam isabet")
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Laboratuvarda geçen (%)",
               ylabel="Tahmin edilen geçiş (%)", title=title, aspect="equal")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=4)
    fig.suptitle("20 yeni Montréal toprağında ince elek tahminleri")
    fig.savefig(pilot.OUT / (STEM + "_tahminler.png"), dpi=150)


def validate(selected):
    pilot.cv2.setNumThreads(1)
    train_rows = read_csv(pilot.OUT / (pilot.STEM + "_olculer.csv"))
    new_rows = pilot.measure([PHOTO / name for name in selected.file])
    assert len(new_rows) == 2 * COUNT and set(train_rows.sample_id).isdisjoint(selected.index)
    assert np.isfinite(new_rows[pilot.COLS]).all().all()
    pilot_scores = read_csv(pilot.OUT / (pilot.STEM + "_sonuclar.csv"))
    targets = pilot_scores.query("arm == 'jpeg'").set_index("sample_id")[["truth_008", "truth_016"]]
    assert len(targets) == 17 and targets.index.is_unique
    predictions = {}
    for arm, rows in train_rows.groupby("arm"):
        x = rows.groupby("sample_id")[pilot.COLS].mean().loc[targets.index]
        test = new_rows[new_rows.arm.eq(arm)].set_index("sample_id").loc[selected.index, pilot.COLS]
        scale = StandardScaler().fit(x)
        model = Ridge(alpha=len(x)).fit(scale.transform(x), targets)
        pred = np.clip(model.predict(scale.transform(test)), 0, 100)
        reverse = pred[:, 0] > pred[:, 1]
        pred[reverse] = pred[reverse].mean(axis=1, keepdims=True)
        predictions[arm] = pred
    predictions["ortanca"] = np.tile(targets.median().to_numpy(), (COUNT, 1))

    # The selection, feature transforms, fits and predictions are complete before scoring.
    metadata = read_csv(META, usecols=["sample_id", *pilot.TARGETS])
    labels = metadata[metadata.sample_id.isin(selected.index)].groupby("sample_id")[pilot.TARGETS]
    assert labels.nunique(dropna=False).eq(1).all().all()
    truth = labels.first().loc[selected.index].to_numpy()
    assert np.isfinite(truth).all() and (truth >= 0).all() and (truth <= 100).all() and (np.diff(truth) >= 0).all()
    frames = []
    for arm, pred in predictions.items():
        result = selected.assign(arm=arm)
        for j, sieve in enumerate(("008", "016")):
            result["truth_" + sieve] = truth[:, j]
            result["pred_" + sieve] = pred[:, j]
            result["mae_" + sieve] = abs(pred[:, j] - truth[:, j])
        frames.append(result)
    results = pd.concat(frames)
    new_rows.to_csv(pilot.OUT / (STEM + "_olculer.csv"), index=False, float_format="%.10f")
    results.to_csv(pilot.OUT / (STEM + "_sonuclar.csv"), float_format="%.9f")
    plot_predictions(results)
    print(results.groupby("arm")[["mae_008", "mae_016"]].mean().round(4))
    indices = np.random.default_rng(SEED).integers(COUNT, size=(10000, COUNT))
    for comparator in ("ortanca", "kucuk"):
        errors = ["mae_008", "mae_016"]
        gains = (results.loc[results.arm.eq(comparator), errors]
                 - results.loc[results.arm.eq("jpeg"), errors])
        for sieve, gain in gains.items():
            interval = np.quantile(gain.to_numpy()[indices].mean(axis=1), [0.025, 0.975])
            print(comparator, sieve, "gain", round(gain.mean(), 3), "wins", (gain > 0).sum(),
                  "median", round(gain.median(), 3), "bootstrap 95%", interval.round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["select", "download", "score"])
    args = parser.parse_args()
    group = selection()
    if args.step == "download":
        PHOTO.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(download, group.file))
    if args.step == "score":
        with threadpool_limits(1):
            validate(group)
