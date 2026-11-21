"""Tek Weibull eğrisini açık v3 ile aynı görüntüler ve PLS modelleriyle sınar."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

work = Path(__file__).resolve().parent
sys.path.insert(0, str(work / "vitalclick" / "src"))
import gsd
import run as published


def weibull_curve(log_params):
    log_scale, log_shape = log_params
    log_power = np.exp(log_shape) * (np.log(gsd.SIZES) - log_scale)
    curve = -100 * np.expm1(-np.exp(np.clip(log_power, -80, 6)))
    curve[-1] = 100
    return curve


def fit_lab_curve(row):
    def loss(params):
        return gsd.emd(row, weibull_curve(params))[0]

    bounds = [(np.log(0.00002), np.log(20000)), (np.log(0.02), np.log(30))]
    return differential_evolution(loss, bounds, seed=42, tol=1e-8).x


def evaluate(refs, queries, ids, columns, labels, log_quantiles, log_params):
    train = refs.groupby("sample_id")[columns].mean()
    query = queries.groupby("sample_id")[columns].mean()
    rows = []
    for sample in ids:
        pool = ids.drop(sample)
        predictions = []
        for target in (log_quantiles, log_params):
            parts = []
            for _, model in published.build_components(columns):
                model.fit(train.loc[pool, model.cols].to_numpy(), target.loc[pool].to_numpy())
                parts.append(model.predict(query.loc[[sample], model.cols].to_numpy())[0])
            predictions.append(np.mean(parts, axis=0))
        truth = labels.loc[sample].to_numpy()
        v3 = gsd.logq_to_curve(predictions[0])[0]
        weibull = weibull_curve(predictions[1])
        rows.append([gsd.emd(truth, curve)[0] for curve in (v3, weibull)])
    return pd.DataFrame(rows, index=ids, columns=["v3", "Weibull"])


if __name__ == "__main__":
    published.LINEAR_ONLY = True
    labels = pd.read_csv(work / "soil-data/Training_labels_updated.csv", index_col="sample_id")[gsd.COLS]
    cache = pd.read_csv(work / "vitalclick/cache/features.csv")
    training = cache[cache.split == "train"]
    original = training[training.aug == 0]
    columns = [c for c in cache if c not in published.META]
    log_quantiles = pd.DataFrame(gsd.curve_to_logq(labels.to_numpy()), index=labels.index)
    # Each fit uses just that soil's lab measurements. Only pool targets enter the predictor.
    log_params = pd.DataFrame([fit_lab_curve(row) for row in labels.to_numpy()], index=labels.index)
    fitted = np.array([weibull_curve(row) for row in log_params.to_numpy()])
    fit_errors = pd.Series(gsd.emd(labels.to_numpy(), fitted), index=labels.index)
    print("Lab fit mean / max:", fit_errors.mean(), fit_errors.idxmax(), fit_errors.max())
    moto, samsung = "Motorola_Edge", "Samsung_A52"
    paired = labels.index.intersection(original[original.phone == moto].sample_id).intersection(
        original[original.phone == samsung].sample_id)
    for name, ids, refs, queries in [
        ("All soils", labels.index, training, original),
        ("Motorola -> Samsung", paired, training[training.phone == moto], original[original.phone == samsung]),
        ("Samsung -> Motorola", paired, training[training.phone == samsung], original[original.phone == moto]),
    ]:
        errors = evaluate(refs, queries, ids, columns, labels, log_quantiles, log_params)
        assert np.isfinite(errors).all().all()
        print(name, len(ids), errors.mean().round(4).to_dict(),
              "Weibull wins:", (errors.Weibull < errors.v3).sum())
