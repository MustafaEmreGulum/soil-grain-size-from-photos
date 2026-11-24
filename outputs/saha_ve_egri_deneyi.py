"""Gerçek saha ayrımlarında v3 ve aynı girdili Ridge hedeflerini karşılaştır."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from coklu_elek_deneyi import AREA, OUT, area_updates, bounds
from orta_elek_kismi_duzeltme import band_curve
from v3_knn_esit_karisim import COLS, keys, labels, original, predict, training
from gsd import curve_to_logq, emd, logq_to_curve

# Soranzo ve ark., Tablo 1 (https://www.deepsoil.at/grai1_data.pdf).
GROUPS = {"Zistersdorf":"F827", "Aichdorf":"G190", "Wien":"H030", "St. Corona":"H037 H038",
          "Wimpassing":"H126", "Ybbs":"H181 H183", "Steinakirchen":"H366 H367 H368 H371 H372 H374",
          "Hagsdorf":"H405", "Gajuri":"H493", "Omding":"H516", "Gottsdorf":"H549", "Pöchlarn":"H615",
          "Simbach":"H616 H617", "Krems":"H637", "Heiligenkreuz":"H666 H668"}
SITE = {s:site for site, samples in GROUPS.items() for s in samples.split()}


def folds():
    all_ids = labels.index
    for s in all_ids:
        yield "toprak", [s], all_ids.drop(s), None, None
    for site, members in GROUPS.items():
        hold = members.split()
        yield "saha", hold, all_ids.drop(hold), None, None
    phones = ["Motorola_Edge","Samsung_A52"]
    paired = set.intersection(*(set(original[original.phone==p].sample_id) for p in phones))
    assert len(paired)==21
    for src,dst in [phones, phones[::-1]]:
        for s in sorted(paired):
            yield f"{src} -> {dst}", [s], sorted(paired-{s}), src, dst
    coarse = ["H368","H374","H615","H367","H366"]
    yield "iri_5", coarse, all_ids.drop(coarse), None, None


def ridge_curves(refs, query, ids, hold):
    x = refs.groupby("sample_id")[keys].mean().loc[ids]
    q = query.groupby("sample_id")[keys].mean().loc[hold]
    scaler = StandardScaler().fit(x)
    x, q = scaler.transform(x), scaler.transform(q)
    truth = labels.loc[ids,COLS].to_numpy()
    z = logit(np.clip(truth/100, .005, .995))
    pca = PCA(n_components=3, svd_solver="full").fit(z)
    result = {}
    for name, target in [("ridge_cap",curve_to_logq(truth)), ("ridge_logit",z),
                         ("ridge_pca3",pca.transform(z))]:
        prediction = Ridge(alpha=30).fit(x, target).predict(q)
        if name == "ridge_cap":
            result[name] = logq_to_curve(prediction)
        else:
            logits = pca.inverse_transform(prediction) if name=="ridge_pca3" else prediction
            curve = np.array([isotonic_regression(row) for row in 100*expit(logits)])
            curve[:,-1] = 100.
            result[name] = curve
    return result


def main():
    photos = pd.read_csv(OUT/"maske_elek_alanlari.csv").query("split=='train'")
    assert set(SITE)==set(labels.index) and len(GROUPS)==15 and len(labels)==24
    assert len(original)==127 and len(training)==1143 and original.file.is_unique
    assert set(original.file)==set(photos.file) and not training.duplicated(["file","aug"]).any()
    archive = pd.read_csv(OUT/"orta_elek_kismi_sonuclar.csv").query("model=='aralik'").set_index(["exam","sample_id"])
    areas = ["a_"+s for s in AREA]
    data = []
    for exam,hold,ids,src,dst in folds():
        refs = training[training.sample_id.isin(ids)]
        query = original[original.sample_id.isin(hold)]
        source, dest = photos, photos
        if src:
            refs, query = refs[refs.phone==src], query[query.phone==dst]
            source, dest = photos[photos.phone==src], photos[photos.phone==dst]
        assert set(refs.sample_id)==set(ids) and set(query.sample_id)==set(hold) and set(ids).isdisjoint(hold)
        if exam=="saha":
            assert {SITE[s] for s in ids}.isdisjoint(SITE[s] for s in hold)
        tr = source.groupby("sample_id")[areas].mean().loc[ids]
        te = dest.groupby("sample_id")[areas].mean().loc[hold]
        preds = ridge_curves(refs, query, ids, hold)
        preds["referans"] = predict(refs,query)["v3"].loc[hold,COLS].to_numpy()
        for i,s in enumerate(hold):
            obs = te.loc[s]
            updates = area_updates(tr,obs,labels.loc[ids])
            models = {}
            for name, curves in preds.items():
                raw = curves[i]
                base = bounds(raw, {k:v for k,v in updates.items() if k=="20"})
                models[name] = (raw, band_curve(base, tr, obs))
            # Swathi V6 pick 3 uyarlaması: eşikler kaynak sürümden, medyan yalnız eğitimden.
            y = labels.loc[ids]
            donors = y[(y["2"] >= 90) & (y["0.063"] <= 10)]
            cur = models["referans"][1].copy()
            fine = COLS.index("0.063")
            routed = cur[COLS.index("2")] >= 90
            if routed and len(donors) and cur[fine] > donors["0.063"].median():
                cur[:fine+1] *= donors["0.063"].median() / cur[fine]
            models["temiz_kum"] = (models["referans"][0], cur)
            for name, (raw, pred) in models.items():
                assert np.isfinite(pred).all() and (np.diff(pred)>=-1e-8).all() and pred.min()>=0 and pred[-1]==100
                score = emd(labels.loc[s,COLS],pred)[0]
                if name=="referans" and exam!="saha":
                    prev = archive.loc[("24" if exam=="toprak" else exam,s)]
                    np.testing.assert_allclose(score,prev.emd,atol=1e-7,rtol=0)
                data.append(dict(exam=exam, site=SITE[s], sample_id=s, n_ref=len(ids), model=name,
                                 kum_adayi=routed, temiz_ref=len(donors),
                                 emd=score, ham_emd=emd(labels.loc[s,COLS],raw)[0], **dict(zip(COLS,pred))))
    rows = pd.DataFrame(data)
    assert len(rows)==95*5 and not rows.duplicated(["exam","model","sample_id"]).any()
    rows.to_csv(OUT/"saha_ve_egri_sonuclar.csv",index=False,float_format="%.9f")
    print(rows.pivot_table(index="model",columns="exam",values="emd").round(3).to_string())
    print("Sahalara esit agirlik",rows.query("exam=='saha'").groupby(["model","site"]).emd.mean().groupby("model").mean().round(3))


if __name__=="__main__":
    main()
