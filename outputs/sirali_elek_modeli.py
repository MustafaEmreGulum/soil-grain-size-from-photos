"""Konum ve yayılımı görüntüden öğrenen birikimli elek modeli, iç içe numune doğrulaması."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from v3_knn_esit_karisim import cache, keys, labels, original, out, sets, training, work
from v3_sabit_kadraj import PHONES, gain_summary, passes
import gsd


N_PC = 4
N_CUT = len(gsd.DLOG)
WEIGHTS = gsd.DLOG / gsd.DLOG.sum()
PENALTIES = (0.03, 0.003, 0.0003)  # Prefer stronger shrinkage in an exact tie.
WIDTH_LIMIT = 1.5
SMOOTH = 0.01
ANCHOR_PENALTY = 0.0001
MODES = ("konum", "konum_yayilim")


def coordinates(train, test, targets=None, projection="pca4"):
    scaler = StandardScaler().fit(train)
    x = np.clip(scaler.transform(train), -3.5, 3.5)
    xt = np.clip(scaler.transform(test), -3.5, 3.5)
    if projection == "pca4":
        pca = PCA(n_components=N_PC, whiten=True, svd_solver="full").fit(x)
        return pca.transform(x), pca.transform(xt)
    if projection not in ("grup_pca", "grup_pls") or targets is None:
        raise ValueError("Two-group projection requires training targets and a recognized method.")
    train_axes, test_axes = [], []
    for group in sets.values():
        cols = [keys.index(k) for k in group]
        if projection == "grup_pca":
            axis = PCA(n_components=1, svd_solver="full").fit(x[:, cols])
        else:
            axis = PLSRegression(n_components=1, scale=False).fit(x[:, cols], gsd.curve_to_logq(targets))
        train_axes.append(axis.transform(x[:, cols])[:, 0])
        test_axes.append(axis.transform(xt[:, cols])[:, 0])
    train_axes = np.column_stack(train_axes)
    scale_axes = StandardScaler().fit(train_axes)
    return scale_axes.transform(train_axes), scale_axes.transform(np.column_stack(test_axes))


def distribution(params, x, variable_width):
    cuts = params[:N_CUT].cumsum()
    end_location = N_CUT + x.shape[1]
    location = x @ params[N_CUT:end_location]
    width_coordinate = x @ params[end_location:] if variable_width else np.zeros(len(x))
    bounded = np.tanh(width_coordinate / WIDTH_LIMIT)
    inverse_width = np.exp(-WIDTH_LIMIT * bounded)
    logits = (cuts - location[:, None]) * inverse_width[:, None]
    return expit(logits), logits, inverse_width, bounded


def objective(params, x, y, variable_width, penalty, anchor):
    prob, logits, inv_width, bounded = distribution(params, x, variable_width)
    residual = prob - y
    radius = np.hypot(residual, SMOOTH)
    loss = ((radius - SMOOTH) @ WEIGHTS).mean()
    d_logits = (residual / radius) * prob * (1 - prob) * WEIGHTS / len(y)
    d_cut_by_soil = d_logits * inv_width[:, None]
    cut_deviation = params[:N_CUT].cumsum() - anchor
    slopes = params[N_CUT:]
    loss += 0.5 * (penalty * (slopes @ slopes) + ANCHOR_PENALTY * np.mean(cut_deviation**2))
    d_cut = d_cut_by_soil.sum(axis=0) + ANCHOR_PENALTY * cut_deviation / N_CUT
    gradients = [np.cumsum(d_cut[::-1])[::-1], -(x.T @ d_cut_by_soil.sum(axis=1))]
    if variable_width:
        gradients.append(-(x.T @ ((d_logits * logits).sum(axis=1) * (1 - bounded**2))))
    grad = np.concatenate(gradients)
    grad[N_CUT:] += penalty * slopes
    return loss, grad


def fit_parameters(x, y_pct, mode, penalty):
    variable_width = mode == "konum_yayilim"
    y = y_pct[:, :N_CUT] / 100
    anchor = logit(np.clip(y.mean(axis=0), 0.005, 0.995))
    gaps = np.r_[anchor[0], np.diff(anchor)]
    n_slopes = x.shape[1] * (2 if variable_width else 1)
    start = np.r_[gaps, np.zeros(n_slopes)]
    bounds = [(None, None)] + [(0, None)] * (N_CUT - 1) + [(None, None)] * n_slopes
    result = minimize(objective, start, args=(x, y, variable_width, penalty, anchor), jac=True,
                      method="L-BFGS-B", bounds=bounds,
                      options={"ftol": 1e-11, "gtol": 2e-7, "maxiter": 1500, "maxls": 50})
    if not result.success:
        raise RuntimeError(f"{mode}, penalty {penalty}: {result.message}")
    return result.x


def fit_predict(x, y_pct, xt, mode, penalty):
    params = fit_parameters(x, y_pct, mode, penalty)
    probability = distribution(params, xt, mode == "konum_yayilim")[0]
    curves = np.column_stack([100 * probability, np.full(len(xt), 100.0)])
    assert np.isfinite(curves).all() and (np.diff(curves, axis=1) >= 0).all()
    return curves


def select_penalties(train, original_views, targets, projection="pca4", modes=MODES):
    # Input arrays contain one row per training soil, in sorted sample_id order.
    losses = {(mode, p): 0.0 for mode in modes for p in PENALTIES}
    for refs, val in KFold(n_splits=4, shuffle=True, random_state=42).split(train):
        x, xt = coordinates(train[refs], original_views[val], targets[refs], projection)
        for (mode, p) in losses:
            curves = fit_predict(x, targets[refs], xt, mode, p)
            losses[mode, p] += gsd.emd(targets[val], curves).sum()
    return {mode: min(PENALTIES, key=lambda p: losses[mode, p]) for mode in modes}


def soil_features(rows):
    return rows.groupby("sample_id")[keys].mean()


def predict_fold(refs, source_originals, targets, test, projection, modes):
    penalties = select_penalties(refs, source_originals, targets, projection, modes)
    x, xt = coordinates(refs, test, targets, projection)
    return {mode: (p, fit_predict(x, targets, xt, mode, p)) for mode, p in penalties.items()}


def evaluate(projection="pca4", modes=MODES):
    common = labels.index
    for phone in PHONES:
        common = common.intersection(original.loc[original.phone == phone, "sample_id"])
    assert len(common) == 21 and len(labels) == 24
    rows = []
    for ids, src, dst in [(labels.index, None, None), (common, *PHONES), (common, *PHONES[::-1])]:
        ids = ids.sort_values()
        exam = "24" if src is None else f"{src} -> {dst}"
        tr = soil_features(training if src is None else training[training.phone == src]).loc[ids]
        orig = soil_features(original if src is None else original[original.phone == src]).loc[ids]
        te = soil_features(original if dst is None else original[original.phone == dst]).loc[ids]
        for soil in ids:
            pool = ids.drop(soil)
            targets = labels.loc[pool].to_numpy()
            fitted = predict_fold(tr.loc[pool].to_numpy(), orig.loc[pool].to_numpy(),
                                  targets, te.loc[[soil]].to_numpy(), projection, modes)
            for mode, (p, curves) in fitted.items():
                rows.append(dict(exam=exam, sample_id=soil, model=mode, projection=projection, penalty=p,
                                 emd=gsd.emd(labels.loc[soil], curves)[0], **dict(zip(gsd.COLS, curves[0]))))
        print(projection, exam, "completed", flush=True)
    return pd.DataFrame(rows)


def submission(projection="pca4"):
    template = pd.read_csv(work / "soil-data/sample_submission.csv", index_col="sample_id")
    ids = labels.index.sort_values()
    test = cache.query("split == 'test' and aug == 0")
    assert len(test) == 35 and set(test.sample_id) == set(template.index)
    tr, orig = (soil_features(frame).loc[ids].to_numpy() for frame in (training, original))
    y = labels.loc[ids].to_numpy()
    p = select_penalties(tr, orig, y, projection, ("konum_yayilim",))["konum_yayilim"]
    x, xt = coordinates(tr, soil_features(test).loc[template.index].to_numpy(), y, projection)
    curves = fit_predict(x, y, xt, "konum_yayilim", p)
    pred = pd.DataFrame(curves, index=template.index, columns=gsd.COLS)
    assert pred.shape == template.shape == (10, 11) and pred.columns.equals(template.columns)
    dest = out / f"submission_sirali_konum_yayilim_{projection}.csv"
    pred.to_csv(dest, float_format="%.8f")
    pd.testing.assert_frame_equal(pd.read_csv(dest, index_col="sample_id"), pred, atol=1e-8, rtol=0)
    print("Submission prepared", dest.name, "penalty", p)


def shape_diagnostic(scores):
    # Held-out labels enter the two-parameter fit ONLY after predictive scores are fixed.
    saved = scores.query("exam == '24' and model == 'konum_yayilim'").set_index("sample_id")
    train, views = soil_features(training), soil_features(original)
    rows = []
    for soil, prediction in saved.iterrows():
        pool = labels.index.drop(soil).sort_values()
        x, xt = coordinates(train.loc[pool], views.loc[[soil]])
        params = fit_parameters(x, labels.loc[pool].to_numpy(), "konum_yayilim", prediction.penalty)
        cuts = params[:N_CUT].cumsum()
        start = np.array([(xt @ params[N_CUT:N_CUT+N_PC]).item(),
                          WIDTH_LIMIT * np.tanh((xt @ params[N_CUT+N_PC:]).item() / WIDTH_LIMIT)])
        observed = labels.loc[soil].to_numpy()[:-1]

        def error(pair):
            estimated = 100 * expit((cuts - pair[0]) * np.exp(-pair[1]))
            return float(np.abs(observed - estimated) @ gsd.DLOG)

        image_error = error(start)
        np.testing.assert_allclose(image_error, prediction.emd, atol=1e-7, rtol=0)
        bounds = [(min(cuts[0], start[0]) - 20, max(cuts[-1], start[0]) + 20),
                  (-WIDTH_LIMIT, WIDTH_LIMIT)]
        errors = [image_error]
        for initial in (start, [np.median(cuts), 0]):
            result = minimize(error, initial, method="Powell", bounds=bounds)
            if not result.success:
                raise RuntimeError(f"Shape fit {soil}: {result.message}")
            errors.append(error(result.x))
        rows.append(dict(sample_id=soil, prediction_emd=image_error, lab_fit_emd=min(errors)))
    result = pd.DataFrame(rows)
    result.to_csv(out / "sirali_elek_bicim_tanisi.csv", index=False, float_format="%.9f")
    print("Lab-informed diagnostic", result.lab_fit_emd.mean())


def main():
    scores = evaluate()
    assert len(scores) == 132 and not scores.duplicated(["exam", "model", "sample_id"]).any()
    scores.to_csv(out / "sirali_elek_sonuclari.csv", index=False, float_format="%.9f")
    base = pd.read_csv(out / "v3_kadraj_sonuclari.csv").query("model == 'v3'")
    for mode in MODES:
        block = scores[scores.model == mode]
        report = gain_summary(base, block)
        print(mode, block.groupby("exam").emd.mean().round(5).to_dict(), report.round(4), sep="\n")
        if mode == "konum_yayilim" and passes(report):
            submission()
    shape_diagnostic(scores)
    groups = pd.concat([evaluate(proj, ("konum_yayilim",)) for proj in ("grup_pca", "grup_pls")],
                       ignore_index=True)
    assert len(groups) == 132 and not groups.duplicated(["exam", "projection", "sample_id"]).any()
    groups.to_csv(out / "sirali_elek_iki_grup_sonuclari.csv", index=False, float_format="%.9f")
    control = groups[groups.projection == "grup_pca"]
    candidate = groups[groups.projection == "grup_pls"]
    print(groups.groupby(["exam", "projection"]).emd.mean())
    print("PLS vs PCA", gain_summary(control, candidate))
    report = gain_summary(base, candidate)
    print("PLS vs pure v3", report)
    if passes(report):
        submission("grup_pls")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
