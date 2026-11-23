"""Tam yüzeydeki iri taş alanıyla yalnız v3'ün 20 mm geçişini güncelle."""
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from h615_otomatik_tas import CACHE, STEM as AUDIT, ig, predict, prev, surface

STEM = ig.out / "v3_tam_yuzey_20mm"


def measure(rows):
    audit = pd.read_csv(AUDIT.with_suffix(".csv"),index_col="file")
    from h374_h615_tam_kadraj import OUTLINES
    assert set(audit.index) == set(OUTLINES) and audit.iou.ge(.5).all()
    assert audit.otomatik_kisa_eksen_mm.gt(20).all()
    boxes = pd.read_csv(ig.out / "dinov2_hiyerarsik_kadrajlar.csv",index_col="file")
    model = None
    records = []
    # Bir örtü maskesi ve H615 doğrulaması önceki tam yüzey çalışmalarıyla ortak.
    for i,(name,row) in enumerate(rows.iterrows(),start=1):
        rgb,_,_ = surface(row,boxes.loc[name])
        if name == prev.H037:
            with np.load(ig.out / "h037_tam_yuzey_maske.npz") as data:
                assert data["photo_name"].item() == name
                mask = data["mask"]
        else:
            if model is None and not (CACHE/(name+".npz")).exists():
                model = ig.load_model(prev.near.WEIGHTS)
            mask = predict(model,name,rgb)
        assert mask.shape == rgb.shape[:2]
        grains = [p for p in regionprops(mask) if p.axis_minor_length > 20*ig.PPM]
        area = sum(p.area for p in grains)*100/mask.size
        records.append(dict(file=name,sample_id=row.sample_id,phone=row.phone,
                            aday_sayisi=len(grains),alan_pct=area))
        print(f"{i}/{len(rows)} {name} · {len(grains)} · {area:.3f}%",flush=True)
    frame = pd.DataFrame(records)
    assert len(frame)==127 and frame.file.is_unique and frame.sample_id.nunique()==24
    frame.to_csv(STEM.with_name(STEM.name+"_fotograflar.csv"),index=False,float_format="%.9f")
    return frame


def evaluate(frame,curves,labels,cols,widths):
    records = []
    idx = cols.index("20")
    for exam,block in curves.groupby("exam"):
        src,dst = (frame,frame) if exam=="24" else (
            frame[frame.phone==exam.split(" -> ")[0]],frame[frame.phone==exam.split(" -> ")[1]])
        train_area = src.groupby("sample_id").alan_pct.mean().loc[block.sample_id]
        test_area = dst.groupby("sample_id").alan_pct.mean().loc[block.sample_id]
        for sample,base in block.set_index("sample_id").iterrows():
            train = train_area.drop(sample)
            y = 100-labels.loc[train.index,"20"].to_numpy()
            x = train.to_numpy()
            denominator = x @ x
            factor = max(0., x @ y / denominator) if denominator>0 else np.nan
            curve = base[cols].to_numpy(dtype=float)
            actual = labels.loc[sample,cols].to_numpy()
            np.testing.assert_allclose(np.abs(actual-curve)[:-1] @ widths,base.emd,atol=1e-7,rtol=0)
            corrected = curve.copy()
            a = test_area.loc[sample]
            if a>0 and np.isfinite(factor):
                corrected[idx] = np.clip(100-factor*a,curve[idx-1],curve[idx+1])
            assert (np.diff(corrected)>=-1e-8).all()
            records.append(dict(exam=exam,sample_id=sample,alan_pct=a,carpan=factor,
                ref_sayisi=len(train),gercek_iri=100-actual[idx],v3_iri=100-curve[idx],
                duzeltilmis_iri=100-corrected[idx],v3_mae20=abs(curve[idx]-actual[idx]),
                duzeltilmis_mae20=abs(corrected[idx]-actual[idx]),v3_emd=base.emd,
                duzeltilmis_emd=np.abs(actual-corrected)[:-1] @ widths))
    result = pd.DataFrame(records)
    assert len(result)==66 and not result.duplicated(["exam","sample_id"]).any()
    difference = result.v3_emd-result.duzeltilmis_emd
    np.testing.assert_allclose(difference,(result.v3_mae20-result.duzeltilmis_mae20)*widths[idx],atol=1e-7,rtol=0)
    return result


def summarize(result):
    groups = list(result.groupby("exam"))+[("22",result[(result.exam=="24") & ~result.sample_id.isin(["H374","H615"])])]
    summaries = []
    for name,g in groups:
        gain = g.v3_emd-g.duzeltilmis_emd
        loo = (gain.sum()-gain)/(len(g)-1)
        summaries.append(dict(sinama=name,n=len(g),**g[["v3_mae20","duzeltilmis_mae20","v3_emd","duzeltilmis_emd"]].mean(),
                              ortalama_kazanc=gain.mean(),ortanca_kazanc=gain.median(),iyilesen=sum(gain>1e-7),
                              ayni=sum(abs(gain)<=1e-7),kotulesen=sum(gain < -1e-7),
                              tek_puan_haric_min=loo.min(),tek_puan_haric_max=loo.max()))
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    curves = pd.read_csv(ig.out/"kutle_payi_kontrol_sonuclari.csv").query("model == 'v3'")
    result = evaluate(measure(prev.photos()),curves,ig.labels,ig.gsd.COLS,ig.gsd.DLOG)
    result.to_csv(STEM.with_name(STEM.name+"_sonuclar.csv"),index=False,float_format="%.9f")
    print(summarize(result).round(5).to_string(index=False))
