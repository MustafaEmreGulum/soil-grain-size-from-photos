"""Compare the 17-soil Ridge and an expanded 37-soil fit on a fresh holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import montreal_ince_bagimsiz as source
import montreal_ince_cozunurluk as pilot

STEM = "montreal_ince_genis"
SEED = 23817
SIEVES = ("008", "016")
TEST_SIZE = 24
PHOTO = pilot.DATA / STEM


def output(suffix):
    return pilot.OUT / f"{STEM}_{suffix}.csv"


def choose():
    meta = source.read_csv(source.META, usecols=["sample_id", "image_path", "dry_or_moist"])
    known = set()
    for stem in ("montreal_pilot_referans", "montreal_egitim_secimi",
                 pilot.STEM + "_olculer", source.STEM + "_secim"):
        known.update(source.read_csv(pilot.OUT / (stem + ".csv"), usecols=["sample_id"]).sample_id)
    meta["file"] = meta.image_path.map(lambda name: Path(name).name)
    dry = meta[meta.dry_or_moist.eq("dry") & meta.file.str.contains(r"_(?:bulk|split[^_]*)_", case=False)]
    pool = (dry[~dry.sample_id.isin(known)].sort_values("file").drop_duplicates("sample_id")
            .set_index("sample_id").sort_index())
    rng = np.random.default_rng(SEED)
    test_ids = sorted(rng.choice(pool.index, TEST_SIZE, replace=False))
    test = pool.loc[test_ids, ["file"]].assign(role="test")

    # Read target values for training candidates only, after fixing the test IDs.
    remaining = pool.drop(test_ids)
    target = pilot.TARGETS[1]
    meta_labels = source.read_csv(source.META, usecols=["sample_id", target])
    grouped = meta_labels[meta_labels.sample_id.isin(remaining.index)].groupby("sample_id")[target]
    assert grouped.nunique(dropna=False).eq(1).all()
    values = grouped.first().loc[remaining.index]
    bands = pd.cut(values, [-0.01, 25, 50, 75, 100], labels=False)
    assert bands.notna().all()
    added = []
    for band in range(4):
        members = bands[bands.eq(band)].index
        added.extend(rng.choice(members, 5, replace=False))
    extra = pool.loc[sorted(added), ["file"]].assign(role="add_train")
    selected = pd.concat([test, extra])
    assert selected.index.is_unique and not set(selected.index).intersection(known)
    if output("secim").exists():
        pd.testing.assert_frame_equal(source.read_csv(output("secim")).set_index("sample_id"), selected)
    else:
        selected.to_csv(output("secim"))
    print("Available", len(pool), "excluded", len(known), "test", len(test), "new train", len(extra), flush=True)
    return selected


def labels_for(ids):
    data = source.read_csv(source.META, usecols=["sample_id", *pilot.TARGETS])
    grouped = data[data.sample_id.isin(ids)].groupby("sample_id")[pilot.TARGETS]
    assert grouped.nunique(dropna=False).eq(1).all().all()
    targets = grouped.first().loc[ids]
    assert targets.ge(0).all().all() and targets.le(100).all().all()
    assert targets.iloc[:, 0].le(targets.iloc[:, 1]).all()
    return targets


def fit_models(features, train, test):
    y = labels_for(train)
    x = features.loc[train]
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=len(train)).fit(scaler.transform(x), y)
    predicted = np.clip(ridge.predict(scaler.transform(features.loc[test])), 0, 100)
    reverse = predicted[:, 0] > predicted[:, 1]
    predicted[reverse] = predicted[reverse].mean(axis=1, keepdims=True)
    return {"ridge": predicted, "ortanca": np.tile(y.median(), (len(test), 1))}


def evaluate(selected):
    pilot.cv2.setNumThreads(1)
    original = source.read_csv(pilot.OUT / (pilot.STEM + "_olculer.csv"))
    new = pilot.measure([PHOTO / name for name in selected.file])
    assert set(new.arm) == {"jpeg", "kucuk"} and len(new) == 2 * len(selected)
    for _, rows in new.groupby("arm"):
        pd.testing.assert_series_equal(rows.set_index("sample_id").file.sort_index(),
                                       selected.file.sort_index())
    new.to_csv(output("olculer"), index=False, float_format="%.10f")
    test_ids = selected[selected.role.eq("test")].index
    old_ids = sorted(original.sample_id.unique())
    expanded_ids = sorted(old_ids + list(selected[selected.role.eq("add_train")].index))
    assert len(expanded_ids) == len(set(expanded_ids)) == 37 and set(expanded_ids).isdisjoint(test_ids)
    frames = []
    for resolution, rows in pd.concat([original, new]).groupby("arm"):
        features = rows.groupby("sample_id")[pilot.COLS].mean()
        assert np.isfinite(features).all().all()
        for train in (old_ids, expanded_ids):
            for model, pred in fit_models(features, train, test_ids).items():
                frame = pd.DataFrame(pred, index=test_ids, columns=["pred_" + s for s in SIEVES])
                frames.append(frame.assign(arm=resolution, n_train=len(train), model=model))
    predictions = pd.concat(frames)
    actual = labels_for(test_ids)
    for target, suffix in zip(pilot.TARGETS, SIEVES):
        predictions["actual_" + suffix] = actual[target]
        predictions["error_" + suffix] = predictions["pred_" + suffix] - predictions["actual_" + suffix]
    predictions.to_csv(output("sonuclar"), float_format="%.9f")
    return predictions


def summarize(results):
    for arm, rows in results.groupby("arm"):
        ridge = rows[rows.model.eq("ridge")]
        old = ridge[ridge.n_train.eq(17)].sort_index()
        new = ridge[ridge.n_train.eq(37)].sort_index()
        median = rows[rows.model.eq("ortanca") & rows.n_train.eq(37)].sort_index()
        indices = np.random.default_rng(SEED).integers(len(new), size=(10000, len(new)))
        for sieve in SIEVES:
            err = "error_" + sieve
            for comparator, control in (("ridge17", old), ("median37", median)):
                gain = control[err].abs() - new[err].abs()
                interval = np.quantile(gain.to_numpy()[indices].mean(axis=1), [0.025, 0.975])
                print(arm, sieve, comparator, "gain", round(gain.mean(), 3), "wins", (gain > 0).sum(),
                      "CI", interval.round(3))
        for high, group in new.groupby(new.actual_016.ge(50)):
            old_group = old.loc[group.index]
            print(arm, "high" if high else "low", len(group), "bias old/new",
                  old_group.error_016.mean(), group.error_016.mean(),
                  "MAE old/new", old_group.error_016.abs().mean(), group.error_016.abs().mean())


def plot_native(results):
    fig = Figure(figsize=(9, 4.8), layout="constrained")
    ridge = results[results.arm.eq("jpeg") & results.model.eq("ridge")]
    for ax, suffix, title in zip(fig.subplots(1, 2), SIEVES, ("0,08 mm", "0,16 mm")):
        for count, color, marker in ((17, "#b46a32", "x"), (37, "#006775", "o")):
            subset = ridge[ridge.n_train.eq(count)]
            ax.scatter(subset["actual_" + suffix], subset["pred_" + suffix], s=27,
                       marker=marker, color=color, alpha=0.8, label=f"{count} eğitim toprağı")
        ax.plot([0, 100], [0, 100], lw=1, color="#555555", label="Tam isabet")
        ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal", title=title,
               xlabel="Laboratuvarda geçen (%)", ylabel="Fotoğraftan tahmin edilen (%)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle("Aynı 24 toprakta eğitim grubunu genişletmenin etkisi")
    fig.savefig(pilot.OUT / (STEM + "_tahminler.png"), dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["select", "download", "score"])
    args = parser.parse_args()
    group = choose()
    if args.step == "download":
        PHOTO.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda name: source.download(name, folder=PHOTO), group.file))
    if args.step == "score":
        with threadpool_limits(1):
            results = evaluate(group)
        errors = ["error_" + s for s in SIEVES]
        print(results.assign(**results[errors].abs()).groupby(["arm", "model", "n_train"])[errors].mean())
        summarize(results)
        plot_native(results)
