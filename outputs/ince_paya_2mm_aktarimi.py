"""Düzeltilen 2 mm geçiş oranını v3'ün daha ince eleklerine orantılı aktar."""
import numpy as np
import pandas as pd

from saha_ve_egri_deneyi import folds, original, training, predict
from v3_knn_esit_karisim import COLS, labels, out, work
from gsd import emd


def transfer(base, raw):
    k = COLS.index("2")
    result = base.copy()
    if raw[k] > 0:
        result[:k] = raw[:k] / raw[k] * base[k]
    assert np.isfinite(result).all() and np.all(np.diff(result) >= -1e-8)
    np.testing.assert_array_equal(result[k:], base[k:])
    return result


def write_submission():
    base = pd.read_csv(out / "submission_v3_orta_elek_aralik.csv", index_col="sample_id")
    raw = pd.read_csv(work / "v3_puan_kontrol.csv", index_col="sample_id")
    template = pd.read_csv(work / "soil-data/sample_submission.csv", index_col="sample_id")
    assert base.index.is_unique and raw.index.is_unique and len(template) == 10
    assert set(template.index) == set(base.index) == set(raw.index)
    assert list(base.columns) == list(raw.columns) == list(template.columns) == list(COLS)
    adjusted = base.loc[template.index].copy()
    for sample in adjusted.index:
        adjusted.loc[sample] = transfer(base.loc[sample].to_numpy(), raw.loc[sample].to_numpy())
    path = out / "submission_v3_ince_pay_2mm.csv"
    adjusted.to_csv(path, float_format="%.6f")
    saved = pd.read_csv(path, index_col="sample_id")
    assert saved.index.equals(template.index) and saved.ge(0).all().all() and saved.le(100).all().all()
    assert (np.diff(saved.values, axis=1) >= 0).all() and saved["200"].eq(100).all()
    pd.testing.assert_frame_equal(saved.loc[:, "2":], base.loc[saved.index, "2":])


def main():
    ref = pd.read_csv(out/"saha_ve_egri_sonuclar.csv").query("model=='referans'").set_index(["exam","sample_id"])
    results = []
    for exam, hold, ids, src, dst in folds():
        train = training[training.sample_id.isin(ids)]
        query = original[original.sample_id.isin(hold)]
        if src:
            train, query = train[train.phone==src],query[query.phone==dst]
        raw = predict(train,query)["v3"]
        for s in hold:
            row = ref.loc[(exam,s)]
            curve = row[COLS].to_numpy(float)
            original_curve = raw.loc[s,COLS].values
            np.testing.assert_allclose(original_curve[:6],curve[:6],atol=1e-7,rtol=0)
            np.testing.assert_allclose(emd(labels.loc[s],curve)[0],row.emd,atol=1e-7,rtol=0)
            adjusted = transfer(curve, original_curve)
            score = emd(labels.loc[s],adjusted)[0]
            results.append(dict(exam=exam,sample_id=s,site=row.site,referans=row.emd,emd=score,
                                kazanc=row.emd-score, **dict(zip(COLS,adjusted))))
    results = pd.DataFrame(results)
    assert len(results)==95 and not results.duplicated(["exam","sample_id"]).any()
    results.to_csv(out/"ince_paya_2mm_sonuclar.csv", index=False,float_format="%.9f")
    print(results.groupby("exam").agg(emd=("emd","mean"),kazanc=("kazanc","mean"),
          iyilesen=("kazanc",lambda g:(g>1e-7).sum()), en_az=("kazanc",lambda g:((g.sum()-g)/(len(g)-1)).min())))
    write_submission()


if __name__=="__main__":
    main()
