"""Compare a refitted Ridge intercept with full retraining on the same 37 soils."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import montreal_ince_genis as experiment
from montreal_ince_genis import pilot, source


def passing(prediction):
    result = np.clip(prediction, 0, 100)
    inverted = result[:, 0] > result[:, 1]
    result[inverted] = result[inverted].mean(axis=1, keepdims=True)
    return result


def main():
    old = source.read_csv(pilot.OUT / (pilot.STEM + "_olculer.csv"))
    new = source.read_csv(experiment.output("olculer"))
    split = source.read_csv(experiment.output("secim")).set_index("sample_id")
    train = sorted(old.sample_id.unique())
    expanded = sorted(train + list(split.index[split.role.eq("add_train")]))
    test = split.index[split.role.eq("test")]
    assert len(train) == 17 and len(set(expanded)) == len(expanded) == 37
    assert len(test) == 24 and test.is_unique and set(test).isdisjoint(expanded)
    truth = experiment.labels_for(expanded)
    archive = source.read_csv(experiment.output("sonuclar")).query("model == 'ridge'")
    rows = []
    for arm, photos in pd.concat([old, new]).groupby("arm"):
        x = photos.groupby("sample_id")[pilot.COLS].mean()
        scale = StandardScaler().fit(x.loc[train])
        ridge = Ridge(alpha=len(train)).fit(scale.transform(x.loc[train]), truth.loc[train])
        # With fixed slopes, the mean raw residual gives the least-squares intercept.
        offset = (truth.to_numpy() - ridge.predict(scale.transform(x.loc[expanded]))).mean(axis=0)
        raw = ridge.predict(scale.transform(x.loc[test]))
        previous = archive[archive.arm.eq(arm)]
        reference = previous[previous.n_train.eq(17)].set_index("sample_id").loc[test]
        full = previous[previous.n_train.eq(37)].set_index("sample_id").loc[test]
        corrected = passing(raw + offset)
        np.testing.assert_allclose(passing(raw), reference[["pred_008", "pred_016"]], atol=2e-7, rtol=0)
        for j, sieve in enumerate(experiment.SIEVES):
            rows.append(pd.DataFrame(dict(arm=arm, sieve=sieve, offset=offset[j],
                                          old=reference["pred_" + sieve], corrected=corrected[:, j],
                                          full=full["pred_" + sieve], actual=reference["actual_" + sieve]),
                                     index=test))
    results = pd.concat(rows)
    results.to_csv(pilot.OUT / "montreal_ince_sabit_sonuclar.csv", float_format="%.9f")
    for (arm, sieve), group in results.groupby(["arm", "sieve"]):
        errors = group[["old", "corrected", "full"]].sub(group.actual, axis=0).abs()
        paired = errors.corrected - errors.full
        indices = np.random.default_rng(experiment.SEED).integers(len(group), size=(10000, len(group)))
        interval = np.quantile(paired.to_numpy()[indices].mean(axis=1), [.025, .975])
        print(arm, sieve, "offset", group.offset.iloc[0], "MAE", errors.mean().round(4).to_dict(),
              "full wins", paired.gt(0).sum(), "CI", interval.round(3))
        if sieve == "016":
            print(errors.groupby(group.actual.ge(50)).mean().round(3))


if __name__ == "__main__":
    main()
