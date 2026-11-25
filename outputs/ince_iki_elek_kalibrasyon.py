"""İki ince eleğin mevcut tahminini iç sınamada öğrenilmiş monoton eğriyle kalibre et."""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

from coklu_elek_deneyi import bounds
from orta_elek_kismi_duzeltme import BANDS
from saha_ve_egri_deneyi import SITE, folds
from tekrar_cekimi_genel_emd import corrected, factor_for
from v3_knn_esit_karisim import COLS, keys, labels, original, out, published, training
from gsd import DLOG, curve_to_logq, emd

TARGETS = ["0.063", "0.2"]
POS = [COLS.index(s) for s in TARGETS]
AREAS = pd.read_csv(out/"maske_elek_alanlari.csv").query("split=='train'")
AREA_COLS = ["a_"+s for s in BANDS]


@lru_cache(maxsize=4096)
def baseline(train_ids, query_ids, src=None, dst=None):
    assert set(train_ids).isdisjoint(query_ids)
    refs = training[training.sample_id.isin(train_ids)]
    obs = original[original.sample_id.isin(query_ids)]
    a = AREAS if src is None else AREAS[AREAS.phone==src]
    b = AREAS if dst is None else AREAS[AREAS.phone==dst]
    if src is not None:
        refs = refs[refs.phone==src]
    if dst is not None:
        obs = obs[obs.phone==dst]
    assert set(refs.sample_id)==set(train_ids) and set(obs.sample_id)==set(query_ids)
    target = pd.DataFrame(curve_to_logq(labels.loc[list(train_ids),COLS].values),index=train_ids)
    ids, _, values = published.fit_predict(refs, obs, target, keys)
    raw = pd.DataFrame(values,index=ids,columns=COLS)
    ref = a.groupby("sample_id")[AREA_COLS].mean().loc[list(train_ids)]
    query = b.groupby("sample_id")[AREA_COLS].mean().loc[list(query_ids)]
    factor = factor_for(ref)
    return pd.DataFrame([corrected(raw.loc[s], query.loc[s], ref, factor, 100.) for s in query_ids],
                        index=pd.Index(query_ids,name="sample_id"),columns=COLS)


def inner_predictions(ids, src, by_site):
    groups = {}
    for s in ids:
        groups.setdefault(SITE[s] if by_site else s, []).append(s)
    inner = []
    for group in groups.values():
        train_ids = tuple(sorted(set(ids)-set(group)))
        inner.append(baseline(train_ids,tuple(sorted(group)),src,src))
    frame = pd.concat(inner)
    assert frame.index.is_unique and set(frame.index)==set(ids)
    return frame.loc[list(ids),TARGETS]


def calibrate(inner, base):
    proposals = {}
    for sieve in TARGETS:
        mapping = IsotonicRegression(out_of_bounds="clip").fit(inner[sieve],labels.loc[inner.index,sieve])
        proposals[sieve] = (base[sieve].to_numpy()+mapping.predict(base[sieve]))/2
    return np.stack([bounds(row.to_numpy(),{s:proposals[s][i] for s in TARGETS})
                     for i,(_,row) in enumerate(base.iterrows())])


def main():
    archive = pd.read_csv(out/"ustten_20_63_sonuclar.csv").query("model=='mevcut'").set_index(["exam","sample_id"])
    rows = []
    fixed = [k for k in range(len(COLS)) if k not in POS]
    for exam,hold,ids,src,dst in folds():
        ids, hold = tuple(sorted(ids)), tuple(sorted(hold))
        base = baseline(ids,hold,src,dst)
        saved = archive.loc[[(exam,s) for s in hold]]
        np.testing.assert_allclose(base,saved[COLS],rtol=0,atol=1e-7)
        inner = inner_predictions(ids,src,by_site=exam=="saha")
        pred = calibrate(inner,base)
        np.testing.assert_array_equal(pred[:,fixed],base.to_numpy()[:,fixed])
        assert np.isfinite(pred).all() and (np.diff(pred,axis=1)>=-1e-8).all() and pred.min()>=0 and pred.max()<=100
        y = labels.loc[list(hold),COLS].to_numpy()
        old_score, new_score = emd(y,base), emd(y,pred)
        np.testing.assert_allclose(old_score,saved.emd,rtol=0,atol=1e-7)
        err_before, err_after = abs(y[:,POS]-base.to_numpy()[:,POS]), abs(y[:,POS]-pred[:,POS])
        np.testing.assert_allclose(new_score-old_score,(err_after-err_before)@DLOG[POS],atol=1e-8,rtol=0)
        for i,s in enumerate(hold):
            record = dict(exam=exam,sample_id=s,site=SITE[s],n_train=len(ids),
                          emd_before=old_score[i],emd_after=new_score[i],gain=old_score[i]-new_score[i])
            for j,col in enumerate(TARGETS):
                record.update({f"lab_{col}":y[i,POS[j]],f"before_{col}":base.loc[s,col],f"after_{col}":pred[i,POS[j]],
                               f"mae_before_{col}":err_before[i,j],f"mae_after_{col}":err_after[i,j]})
            rows.append(record)
    d = pd.DataFrame(rows)
    assert len(d)==95 and not d.duplicated(["exam","sample_id"]).any()
    d.to_csv(out/"ince_iki_elek_kalibrasyon_sonuclar.csv",index=False,float_format="%.9f")
    print(d.groupby("exam")[["emd_before","emd_after","mae_before_0.063","mae_after_0.063",
                             "mae_before_0.2","mae_after_0.2"]].mean().round(4).to_string())


if __name__=="__main__":
    with threadpool_limits(1):
        main()
