"""Saf v3 ile beş kesitli Ridge tahminlerini aynı toprak ve elekte eşleştir."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
WORK = OUT.parent / "work"
sys.path.insert(0, str(WORK / "vitalclick" / "src"))
from gsd import COLS, DLOG, emd


SIEVES = ("0.063", "0.2", "0.63")
KEYS = ["exam", "sample_id"]
SEED_SOILS = ("H037", "H038")
TOL = 1e-8  # Predictions are stored to nine decimal places.


def absolute_error_slope(v3, dino, lab):
    # Right derivative at weight 0 of |v3 + weight * (dino - v3) - lab|.
    delta = dino - v3
    return np.where(v3 == lab, np.abs(delta), np.sign(v3 - lab) * delta)


def paired_errors():
    v3 = pd.read_csv(OUT / "dalgacik_sonuclari.csv").query("model == 'v3'").set_index(KEYS).sort_index()
    dino = pd.read_csv(OUT / "dinov2_kesit_ridge_sonuclari.csv").query("model == 'bes'").set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(v3.index, dino.index)
    assert v3.index.is_unique and sorted(v3.groupby(level="exam").size()) == [21, 21, 24]
    lab = pd.read_csv(WORK / "soil-data" / "Training_labels_updated.csv", index_col="sample_id")
    assert lab.index.is_unique
    actual = lab.loc[v3.index.get_level_values("sample_id"), COLS].set_axis(v3.index)
    for predictions, columns in [(v3, ["pred_" + c for c in COLS]), (dino, COLS)]:
        np.testing.assert_allclose(emd(actual, predictions[columns]), predictions.emd, atol=TOL, rtol=0)
    np.testing.assert_allclose(v3.emd, dino.v3_emd, atol=TOL, rtol=0)

    v3_curves = v3[["pred_" + c for c in COLS]].to_numpy()
    slopes = absolute_error_slope(v3_curves, dino[COLS].to_numpy(), actual.to_numpy())
    curve_slope = pd.Series(slopes[:, :-1] @ DLOG, index=v3.index)
    rows = []
    for sieve in SIEVES:
        part = pd.DataFrame({"sieve_mm": sieve, "lab_pct": actual[sieve],
                             "v3_pct": v3["pred_" + sieve], "dino_pct": dino[sieve]})
        v = part["v3_signed_pp"] = part.v3_pct - part.lab_pct
        d = part["dino_signed_pp"] = part.dino_pct - part.lab_pct
        np.testing.assert_allclose(v, v3["signed_" + sieve], atol=TOL, rtol=0)
        gain = part["dino_gain_pp"] = v.abs() - d.abs()
        part["mae_slope_at_v3"] = slopes[:, COLS.index(sieve)]
        exact = (v.abs() <= TOL) | (d.abs() <= TOL)
        part["error_direction"] = np.select([exact, (v * d) > 0], ["exact", "same"], default="opposite")
        part["closer"] = np.select([gain > TOL, gain < -TOL], ["dino", "v3"], default="tie")
        rows.append(part.reset_index())
    return pd.concat(rows, ignore_index=True).sort_values(KEYS + ["sieve_mm"]), curve_slope


def print_summary(rows, curve_slope):
    for exam, block in rows.groupby("exam", sort=False):
        groups = [(exam, block)]
        if exam == "24":
            groups.append(("22", block[~block.sample_id.isin(SEED_SOILS)]))
        for label, subset in groups:
            print("\n", label)
            for sieve, p in subset.groupby("sieve_mm"):
                print(sieve, "MAE v3/DINO", round(p.v3_signed_pp.abs().mean(), 3),
                      round(p.dino_signed_pp.abs().mean(), 3),
                      "mean/median gain", round(p.dino_gain_pp.mean(), 3), round(p.dino_gain_pp.median(), 3),
                      "DINO closer/tied", (p.closer == "dino").sum(), (p.closer == "tie").sum(),
                      "same/opposite/exact", *(int((p.error_direction == value).sum())
                                               for value in ("same", "opposite", "exact")),
                      "slope", round(p.mae_slope_at_v3.mean(), 3))
            soils = subset.groupby("sample_id").dino_gain_pp.mean()
            print("three-sieve soil mean: gain/median/wins/ties",
                  round(soils.mean(), 3), round(soils.median(), 3), (soils > TOL).sum(), (soils.abs() <= TOL).sum())
            print("initial slope: three-sieve MAE / full EMD",
                  round(subset.mae_slope_at_v3.mean(), 3),
                  round(curve_slope.xs(exam).loc[soils.index].mean(), 3))
            if label == "24":
                print(soils.sort_values(ascending=False).round(2).to_string())


if __name__ == "__main__":
    rows, curve_slope = paired_errors()
    rows.to_csv(OUT / "dinov2_ince_elek_sonuclari.csv", index=False, float_format="%.9f")
    print_summary(rows, curve_slope)
