"""Tekrarlı doku ölçülerini denetle ve tek özellikli hata düzeltmesini saklı toprakta sına."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coklu_elek_deneyi import TEXTURE, bounds
from ince_iki_elek_kalibrasyon import POS, TARGETS, baseline, inner_predictions
from saha_ve_egri_deneyi import folds
from v3_knn_esit_karisim import COLS, labels, original, out, work
from gsd import DLOG, emd


def photographs():
    ids = ["file", "sample_id", "phone"]
    center = [c for c in original if c.startswith(("log_", "edge_", "open_", "close_"))]
    frame = original[ids + center].copy()
    gran = pd.read_csv(work/"vitalclick/cache/granulometry_train.csv")
    grid = pd.read_csv(out/"bes_ve_onalti_doku_fotograflar.csv")
    for data, columns in [(gran,TEXTURE), (grid,[c for c in grid if c.startswith("onalti_")])]:
        frame = frame.merge(data[ids+columns], on=ids, validate="one_to_one")
    assert len(frame)==127 and frame.sample_id.nunique()==24 and frame.file.is_unique
    columns = frame.columns.drop(ids).tolist()
    assert np.isfinite(frame[columns]).all().all()
    return frame, columns


def ranking(photos, cols, residual):
    grouped = photos.groupby("sample_id")[cols]
    means = grouped.mean().loc[residual.index]
    # Give each soil equal weight in both variance components. Singletons
    # supply a mean but cannot supply a within-soil variance estimate.
    between = means.var()
    within_soil = grouped.var().loc[means.index]
    within = within_soil.mean()
    noise_of_mean = (within_soil.div(grouped.size(),axis=0)).mean()
    signal = (between-noise_of_mean).clip(lower=0)
    repeat = signal.div(signal+within).fillna(0)
    stats = pd.DataFrame(dict(repeat=repeat,within_sd=np.sqrt(within),between_sd=np.sqrt(between)))
    ranks = means.rank()
    for s in TARGETS:
        stats["rho_lab_"+s] = ranks.corrwith(labels.loc[means.index,s].rank())
        stats["rho_err_"+s] = ranks.corrwith(residual[s].rank())
    stats["score"] = repeat * stats[["rho_err_"+s for s in TARGETS]].abs().fillna(0).mean(axis=1)
    return stats


def audit(frame, cols, exam, hold, train, src, dst):
    train, hold = tuple(sorted(train)), tuple(sorted(hold))
    assert set(hold).isdisjoint(train)
    ref = frame[frame.sample_id.isin(train)]
    query = frame[frame.sample_id.isin(hold)]
    if src:
        ref, query = ref[ref.phone==src], query[query.phone==dst]
    inner = inner_predictions(train,src,by_site=exam=="saha")
    residual = labels.loc[inner.index,TARGETS]-inner
    stats = ranking(ref,cols,residual)
    best = stats.score.idxmax()
    x = ref.groupby("sample_id")[best].mean().loc[inner.index]
    mean, sd = x.mean(), x.std(ddof=0)
    z = (x-mean)/sd
    # Ridge with alpha=n on one standardized feature; zero correction at mean.
    slopes = z.to_numpy() @ residual.to_numpy() / (z@z+len(z))
    xt = query.groupby("sample_id")[best].mean().loc[list(hold)]
    delta = np.outer((xt-mean)/sd, slopes)
    base = baseline(train,hold,src,dst)
    rows = []
    for i,sample in enumerate(hold):
        b = base.loc[sample]
        pred = bounds(b.to_numpy(),dict(zip(TARGETS,b[TARGETS].to_numpy()+delta[i])))
        fixed = [j for j in range(len(COLS)) if j not in POS]
        np.testing.assert_array_equal(pred[fixed],b.to_numpy()[fixed])
        assert np.isfinite(pred).all() and (np.diff(pred)>=-1e-8).all()
        actual = labels.loc[sample,COLS].to_numpy()
        old,new = emd(actual,[b.to_numpy(),pred])
        diff = abs(actual[POS]-b.to_numpy()[POS])-abs(actual[POS]-pred[POS])
        np.testing.assert_allclose(old-new,diff@DLOG[POS],rtol=0,atol=1e-7)
        row = dict(exam=exam,sample_id=sample,feature=best,repeat=stats.at[best,"repeat"],
                   x=xt.loc[sample],x_min=query.loc[query.sample_id==sample,best].min(),
                   x_max=query.loc[query.sample_id==sample,best].max(),emd_base=old,emd_new=new,gain=old-new)
        for j,t in zip(POS,TARGETS):
            row.update({f"true_{t}":actual[j],f"base_{t}":b[t],f"new_{t}":pred[j]})
        rows.append(row)
    return rows


def main():
    frame,cols = photographs()
    train = tuple(sorted(labels.index))
    inner = inner_predictions(train,None,False)
    full = ranking(frame,cols,labels.loc[inner.index,TARGETS]-inner)
    phones = frame.groupby(["phone","sample_id"])[cols].mean()
    a,b = (phones.loc[name] for name in ["Motorola_Edge","Samsung_A52"])
    shared = a.index.intersection(b.index)
    assert len(shared)==21
    full["phone_rank_rho"] = a.loc[shared].rank().corrwith(b.loc[shared].rank())
    full.index.name = "feature"
    full.sort_values("score",ascending=False).to_csv(out/"ince_doku_tutarliligi_olculer.csv")
    records = []
    for exam,hold,ids,src,dst in folds():
        records.extend(audit(frame,cols,exam,hold,ids,src,dst))
    result = pd.DataFrame(records)
    archive = pd.read_csv(out/"ustten_20_63_sonuclar.csv").query("model=='mevcut'").set_index(["exam","sample_id"])
    idx = result.set_index(["exam","sample_id"]).index
    assert idx.is_unique and len(idx)==len(archive)==95
    np.testing.assert_allclose(result.emd_base,archive.loc[idx,"emd"],rtol=0,atol=1e-7)
    pair = ["H037","H668"]
    records.extend(audit(frame,cols,"iki_ornek",pair,set(train)-set(pair),None,None))
    pd.DataFrame(records).to_csv(out/"ince_doku_tutarliligi_sonuclar.csv",index=False,float_format="%.9f")
    print(full.nlargest(4,"score").round(3).to_string())
    print(result.groupby("exam")[["emd_base","emd_new","gain"]].mean().round(3))
    print(pd.DataFrame(records[-2:]).round(3).to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(1):
        main()
