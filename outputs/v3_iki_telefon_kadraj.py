"""Aynı eğitim modeliyle saklı toprağın Motorola ve Samsung eğrilerini karşılaştır."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from v3_knn_esit_karisim import keys, labels, original, out, published, training
from v3_sabit_kadraj import PHONES, TARGETS, fixed_features
import gsd


def predict_phone_pairs(data, name, shared):
    records = []
    data = data[data.sample_id.isin(shared) & data.phone.isin(PHONES)]
    for source in PHONES:
        for soil in shared:
            pool = shared.drop(soil)
            train = data[(data.phone == source) & data.sample_id.isin(pool)]
            test = data[(data.sample_id == soil) & (data.aug == 0)]
            assert set(train.sample_id) == set(pool) and set(test.phone) == set(PHONES)
            # fit_predict groups test shots by sample_id; the two keys here denote views of one soil.
            views = test.assign(sample_id=test.phone)
            order, _, curves = published.fit_predict(train, views, TARGETS.loc[pool], keys)
            by_phone = pd.DataFrame(curves, index=order, columns=gsd.COLS).loc[list(PHONES)]
            assert np.isfinite(by_phone).all().all() and np.all(np.diff(by_phone, axis=1) >= 0)
            assert (by_phone.iloc[:, 0] >= 0).all() and (by_phone.iloc[:, -1] == 100).all()
            distance = gsd.emd(by_phone.iloc[0], by_phone.iloc[1])[0]
            for target, curve in by_phone.iterrows():
                records.append(dict(sample_id=soil, model=name, train_phone=source, test_phone=target,
                                    lab_emd=gsd.emd(labels.loc[soil], curve)[0], phone_gap_emd=distance,
                                    **curve.to_dict()))
    return pd.DataFrame(records)


def paired_summary(table, metric):
    scores = table.pivot(index="sample_id", columns="model", values=metric)
    assert len(scores) == 21 and scores.notna().all().all()
    gain = scores.v3 - scores["100mm"]
    loo = (gain.sum() - gain) / (len(gain) - 1)
    return dict(v3=scores.v3.mean(), fixed=scores["100mm"].mean(), gain=gain.mean(),
                median=gain.median(), wins=int((gain > 0).sum()), loo_min=loo.min(), loo_max=loo.max())


def main():
    shared = labels.index
    for phone in PHONES:
        shared = shared[shared.isin(original.loc[original.phone == phone, "sample_id"])]
    assert len(shared) == 21
    pairs = pd.concat([predict_phone_pairs(data, name, shared)
                       for name, data in [("v3", training), ("100mm", fixed_features())]], ignore_index=True)
    assert len(pairs) == 168 and not pairs.duplicated(["sample_id", "train_phone", "test_phone", "model"]).any()
    cross = pairs[pairs.train_phone != pairs.test_phone].copy()
    cross["exam"] = cross.train_phone + " -> " + cross.test_phone
    archive = pd.read_csv(out / "v3_kadraj_sonuclari.csv").query("exam != '24' and model in ['v3','100mm']")
    index = ["exam", "sample_id", "model"]
    a, b = cross.set_index(index).sort_index(), archive.set_index(index).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_allclose(a[gsd.COLS], b[gsd.COLS], rtol=0, atol=1e-7)
    np.testing.assert_allclose(a.lab_emd, b.emd, rtol=0, atol=1e-7)
    pairs.to_csv(out / "v3_iki_telefon_kadraj_sonuclari.csv", index=False, float_format="%.9f")
    gains = []
    for source in PHONES:
        rows = pairs[pairs.train_phone == source]
        for target in PHONES:
            result = paired_summary(rows[rows.test_phone == target], "lab_emd")
            print(source, "->", target, {k: round(v, 4) for k, v in result.items()})
            if target != source:
                gains.append(result["gain"])
        result = paired_summary(rows[rows.test_phone == source], "phone_gap_emd")
        gains.append(result["gain"])
        print(source, "between views", {k: round(v, 4) for k, v in result.items()})
    print("Five-crop criterion satisfied:", all(g > 0 for g in gains))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
