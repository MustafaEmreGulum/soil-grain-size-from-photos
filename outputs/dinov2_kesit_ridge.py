"""Tek ve beş fiziksel kesitin DINOv2 temsillerini aynı Ridge ile sınar."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import timm
import torch

from v3_knn_esit_karisim import labels, work
import gsd


OUT = Path(__file__).resolve().parent
OFFSETS_MM = [(0, 0), (-20, -20), (20, -20), (-20, 20), (20, 20)]
MODES = ("tek", "bes")


def crop_photo(row):
    ref_side, ppm = gsd.PHONES[row.phone]
    with Image.open(row.path) as source:
        im = ImageOps.exif_transpose(source)
        w, h = im.size
        eff_ppm = ppm * max(w, h) / ref_side
        side = round(100 * eff_ppm)
        images = []
        for dx, dy in OFFSETS_MM:
            x = (w - side) // 2 + round(dx * eff_ppm)
            y = (h - side) // 2 + round(dy * eff_ppm)
            assert 0 <= x <= w - side and 0 <= y <= h - side, row.file
            crop = im.crop((x, y, x + side, y + side)).convert("RGB")
            crop = crop.resize((400, 400), Image.Resampling.LANCZOS)
            images.append(np.asarray(crop.resize((224, 224), Image.Resampling.BICUBIC)))
    return images


def extract(photos):
    weights = work / "dinov2_small.safetensors"
    expected = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
    with weights.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == expected
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False,
                              dynamic_img_size=True, num_classes=0)
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.eval()
    mean, std = (torch.tensor(model.pretrained_cfg[k]).view(1, 3, 1, 1) for k in ("mean", "std"))
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(photos), 2):
            rows = photos.iloc[start:start+2]
            images = [crop for row in rows.itertuples() for crop in crop_photo(row)]
            batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
            embeddings.append(model((batch - mean) / std).numpy().reshape(len(rows), 5, 384))
            if (start + len(rows)) % 32 == 0 or start + len(rows) == len(photos):
                print("Photos encoded:", start + len(rows), "/", len(photos), flush=True)
    vectors = np.concatenate(embeddings).astype("float64")
    assert vectors.shape == (127, 5, 384) and np.isfinite(vectors).all()
    return {mode: pd.DataFrame(array, index=photos.index)
            for mode, array in zip(MODES, (vectors[:, 0], vectors.mean(axis=1)))}


def previous_knn_control(photos, centers):
    median = centers.groupby(photos.sample_id).median().loc[labels.index]
    errors = []
    for sample in labels.index:
        pool = labels.index.drop(sample)
        scaler = StandardScaler().fit(median.loc[pool])
        distances = np.linalg.norm(scaler.transform(median.loc[pool]) -
                                   scaler.transform(median.loc[[sample]]), axis=1)
        neighbors = pool[np.argsort(distances, kind="stable")[:3]]
        errors.append(gsd.emd(labels.loc[sample], labels.loc[neighbors].mean())[0])
    return np.mean(errors)


def evaluate(photos, vectors, ids, source, dest, reference):
    tr = photos if source is None else photos[photos.phone == source]
    te = photos if dest is None else photos[photos.phone == dest]
    exam = "24" if source is None else f"{source} -> {dest}"
    rows = []
    for mode, per_photo in vectors.items():
        train = per_photo.loc[tr.index].groupby(tr.sample_id).mean().loc[ids]
        test = per_photo.loc[te.index].groupby(te.sample_id).mean().loc[ids]
        for sample in ids:
            pool = ids.drop(sample)
            pipe = make_pipeline(StandardScaler(), PCA(n_components=8, whiten=True, svd_solver="full"),
                                 Ridge(alpha=10.0, solver="svd"))
            pipe.fit(train.loc[pool].to_numpy(), gsd.curve_to_logq(labels.loc[pool]))
            curve = gsd.logq_to_curve(pipe.predict(test.loc[[sample]].to_numpy()))[0]
            assert np.isfinite(curve).all() and curve[0] >= 0 and curve[-1] == 100 and (np.diff(curve) >= 0).all()
            error = gsd.emd(labels.loc[sample], curve)[0]
            rows.append({"exam": exam, "sample_id": sample, "model": mode, "emd": error,
                         "v3_emd": reference.loc[(exam, sample)], **dict(zip(gsd.COLS, curve))})
    return pd.DataFrame(rows)


def paired_summary(result):
    all_soils = result[result.exam == "24"]
    winner = all_soils.groupby("model").emd.mean().idxmin()
    rows = []
    for exam, block in result.groupby("exam", sort=False):
        scores = block.pivot(index="sample_id", columns="model", values="emd")
        scores["v3"] = block[block.model == winner].set_index("sample_id").v3_emd
        for base, candidate in [("tek", "bes"), ("v3", winner)]:
            gain = scores[base] - scores[candidate]
            leave_one_out = (gain.sum() - gain) / (len(gain) - 1)
            rows.append(dict(exam=exam, base=base, candidate=candidate, n=len(gain),
                             base_emd=scores[base].mean(), candidate_emd=scores[candidate].mean(),
                             mean_gain=gain.mean(), median_gain=gain.median(), wins=(gain > 0).sum(),
                             ties=(gain == 0).sum(), loo_min=leave_one_out.min(), loo_max=leave_one_out.max(),
                             loo_positive=(leave_one_out > 0).sum()))
    return pd.DataFrame(rows)


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    photos = gsd.list_images(str(work / "soil-data")).query("split == 'train'").set_index("file").sort_index()
    assert len(photos) == 127 and photos.index.is_unique and set(photos.sample_id) == set(labels.index)
    vectors = extract(photos)
    ref = pd.read_csv(OUT / "ikinci_iri_aday_sonuclari.csv").set_index(["exam", "sample_id"]).v3_emd
    moto, sam = "Motorola_Edge", "Samsung_A52"
    common = labels.index.intersection(photos[photos.phone == moto].sample_id).intersection(
        photos[photos.phone == sam].sample_id)
    assert len(common) == 21
    with threadpool_limits(limits=1):
        control = previous_knn_control(photos, vectors["tek"])
        print("Original center-only KNN:", control, flush=True)
        np.testing.assert_allclose(control, 53.428, atol=.0006, rtol=0)
        result = pd.concat([evaluate(photos, vectors, ids, src, dst, ref)
                            for ids, src, dst in [(labels.index, None, None), (common, moto, sam),
                                                  (common, sam, moto)]], ignore_index=True)
    assert len(result) == 132 and not result.duplicated(["exam", "sample_id", "model"]).any()
    result.to_csv(OUT / "dinov2_kesit_ridge_sonuclari.csv", index=False, float_format="%.9f")
    print(result.pivot_table(index="exam", columns="model", values="emd").round(5))
    print(paired_summary(result).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
