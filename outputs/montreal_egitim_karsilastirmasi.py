"""Saf v3'ü aynı Kaggle ayrımlarında 12 ilave Montréal toprağıyla karşılaştır."""
import hashlib
import io
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from threadpoolctl import threadpool_limits
from urllib3.util.retry import Retry

from montreal_dis_sinama import META_PATH, PHYSICAL_CAMERA, ROOT, eligible_images, image_targets, read_photo
from v3_knn_esit_karisim import cache, keys, labels, original, out, published, training, work
from v3_sabit_kadraj import PHONES, gain_summary, passes
import features
import gsd


N_EXTERNAL = 12
SEED = 2026
MANIFEST = out / "montreal_egitim_secimi.csv"
FEATURES = out / "montreal_egitim_olculeri.csv"
SCORES = out / "montreal_egitim_sonuclari.csv"


def save_atomic(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def selected_soils(session):
    pilot = pd.read_csv(out / "montreal_pilot_referans.csv", dtype={"sample_id": str}).sample_id
    assert pilot.nunique() == len(pilot) == 8
    if not MANIFEST.exists():
        response = session.get(ROOT + META_PATH, timeout=(15, 60))
        response.raise_for_status()
        metadata = pd.read_csv(io.BytesIO(response.content), dtype={"sample_id": str})
        eligible, sizes, columns = eligible_images(metadata)
        pool = eligible.drop(index=pilot)
        assert len(pool) == 61
        ids = sorted(np.random.default_rng(SEED).choice(pool.index.to_numpy(), N_EXTERNAL, replace=False))
        names, truth = image_targets(pool.loc[ids], sizes, columns)
        save_atomic(truth.assign(filename=names).reset_index(), MANIFEST)
    selected = pd.read_csv(MANIFEST, dtype={"sample_id": str}).set_index("sample_id")
    assert selected.index.is_unique and len(selected) == N_EXTERNAL and selected.index.is_monotonic_increasing
    assert set(selected.index).isdisjoint(pilot) and selected.filename.is_unique
    assert all(name.startswith(soil + "_") for soil, name in selected.filename.items())
    validate_curves(selected[gsd.COLS].to_numpy())
    return selected


def validate_curves(curves):
    assert np.isfinite(curves).all() and (curves >= 0).all() and (curves <= 100).all()
    assert (np.diff(curves, axis=1) >= -1e-9).all() and (curves[:, -1] == 100).all()


def validate_features(frame, selected):
    assert not frame.duplicated(["sample_id", "aug"]).any()
    assert np.isfinite(frame[keys]).all().all() and frame.sha256.str.fullmatch(r"[a-f0-9]{64}").all()
    for soil, group in frame.groupby("sample_id"):
        assert soil in selected.index and group.filename.eq(selected.at[soil, "filename"]).all()
        assert set(group.aug) == set(range(9)) and group.sha256.nunique() == 1


def prepare():
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1,
                                                                status_forcelist=(429, 500, 502, 503, 504))))
        selected = selected_soils(session)
        measured = pd.read_csv(FEATURES, dtype={"sample_id": str}) if FEATURES.exists() else pd.DataFrame()
        if not measured.empty:
            validate_features(measured, selected)
        done = set(measured.sample_id) if not measured.empty else set()
        print("Selected:", ", ".join(selected.index), "already measured:", len(done), flush=True)
        cv2.setNumThreads(1)
        with patch.dict(gsd.PHONES, {PHYSICAL_CAMERA: (8192, 1000 / 39.4)}):
            for position, (soil, filename) in enumerate(selected.filename.items()):
                if soil in done:
                    continue
                with read_photo(session, filename, max_bytes=60 * 1024**2) as photo:
                    digest = hashlib.sha256(photo.getbuffer()).hexdigest()
                    canonical = features.load_canonical(photo, PHYSICAL_CAMERA)
                # Reuse the decoded and physically scaled RGB image for v3's eight training variants.
                rows = []
                with patch.object(features, "load_canonical", return_value=canonical):
                    for aug in range(9):
                        seed = 1009 * aug + position if aug else None
                        vector = features.image_features(filename, PHYSICAL_CAMERA, seed)
                        rows.append(dict(sample_id=soil, filename=filename, sha256=digest, aug=aug,
                                         **{name: vector[name] for name in keys}))
                block = pd.DataFrame(rows)
                validate_features(block, selected)
                measured = pd.concat([measured, block], ignore_index=True)
                save_atomic(measured, FEATURES)
                print("Measured", soil, f"({len(measured) // 9}/{N_EXTERNAL})", flush=True)
    assert set(measured.sample_id) == set(selected.index) and measured.sha256.nunique() == N_EXTERNAL
    return selected, measured


def evaluate(selected, measured):
    # The published fit averages features within soil, so each external soil has one vote.
    source = measured[["sample_id", *keys]].copy()
    source["sample_id"] = "MTL_" + source.sample_id
    ext_y = selected[gsd.COLS].add_prefix("MTL_", axis="index")
    assert set(source.sample_id) == set(ext_y.index) and set(ext_y.index).isdisjoint(labels.index)
    all_y = pd.concat([labels, ext_y])
    targets = pd.DataFrame(gsd.curve_to_logq(all_y.to_numpy()), index=all_y.index)
    common = labels.index
    for phone in PHONES:
        common = common.intersection(original.loc[original.phone == phone, "sample_id"])
    assert len(labels) == 24 and len(common) == 21 and published.LINEAR_ONLY and published.RANGE_MARGIN is None
    rows = []
    for ids, src, dst in [(labels.index, None, None), (common, *PHONES), (common, *PHONES[::-1])]:
        exam = "24" if src is None else f"{src} -> {dst}"
        local = training if src is None else training.loc[training.phone == src]
        views = original if dst is None else original.loc[original.phone == dst]
        assert views.aug.eq(0).all()
        for held in ids:
            refs = ids.drop(held)
            local_train = local.loc[local.sample_id.isin(refs), ["sample_id", *keys]]
            query = views.loc[views.sample_id == held]
            assert set(local_train.sample_id) == set(refs) and len(query) > 0
            for variant, fit_data, pool in [
                ("v3", local_train, refs),
                ("v3_montreal", pd.concat([local_train, source], ignore_index=True), refs.append(ext_y.index)),
            ]:
                assert held not in pool and pool.is_unique and set(fit_data.sample_id) == set(pool)
                order, _, predictions = published.fit_predict(fit_data, query, targets.loc[pool], keys)
                assert list(order) == [held]
                mean = all_y.loc[pool].mean().to_numpy()
                mean_model = "ortalama" if variant == "v3" else "ortalama_montreal"
                for model, curve in [(variant, predictions[0]), (mean_model, mean)]:
                    validate_curves(curve[None, :])
                    rows.append(dict(exam=exam, sample_id=held, model=model, n_local=len(refs),
                                     n_external=len(pool)-len(refs), emd=gsd.emd(labels.loc[held], curve)[0],
                                     **dict(zip(gsd.COLS, curve))))
    scores = pd.DataFrame(rows)
    assert len(scores) == 264 and not scores.duplicated(["exam", "sample_id", "model"]).any()
    by = ["exam", "sample_id"]
    base = scores[scores.model == "v3"]
    candidate = scores[scores.model == "v3_montreal"]
    reference = pd.read_csv(out / "v3_kadraj_sonuclari.csv").query("model == 'v3'").set_index(by).sort_index()
    current = base.set_index(by).sort_index()
    pd.testing.assert_index_equal(reference.index, current.index)
    np.testing.assert_allclose(current[["emd", *gsd.COLS]], reference[["emd", *gsd.COLS]], atol=1e-7, rtol=0)
    save_atomic(scores, SCORES)
    print(scores.pivot_table(index="exam", columns="model", values="emd").round(4).to_string())
    report = gain_summary(base, candidate)
    print(report.round(4).to_string())
    if passes(report):
        template = pd.read_csv(work / "soil-data/sample_submission.csv", index_col="sample_id")
        test = cache.query("split == 'test' and aug == 0")
        assert len(template) == 10 and set(test.sample_id) == set(template.index)
        all_training = pd.concat([training[["sample_id", *keys]], source], ignore_index=True)
        order, _, predictions = published.fit_predict(all_training, test, targets, keys)
        validate_curves(predictions)
        pred = pd.DataFrame(predictions, index=order, columns=gsd.COLS).loc[template.index]
        assert pred.columns.equals(template.columns)
        save_atomic(pred.reset_index(), out / "submission_v3_montreal_12.csv")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        evaluate(*prepare())
