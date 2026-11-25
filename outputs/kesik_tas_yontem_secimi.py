"""Görünen kesim yüzüne göre çap yöntemini, her seferinde bir toprağı ayırarak seç."""
import numpy as np
import pandas as pd

from yapay_kesik_tas import out


# İlk ve son seçenek sırasıyla her zaman görünen ölçüm ve her zaman elipstir.
# Eşit eğitim hatasında ilk seçenek tercih edilir.
THRESHOLDS = [1.01, .8, .6, .4, 0.]


def prediction(df, threshold):
    return np.where(df.kenar_orani >= threshold, df.elips_mm, df.gorunen_mm)


def fit(train):
    def loss(threshold):
        errors = abs(prediction(train, threshold) - train.referans_mm)
        return errors.groupby(train.sample_id).mean().mean()
    return min(THRESHOLDS, key=loss)


def main():
    data = pd.read_csv(out/"yapay_kesik_tas.csv")
    assert data.kenar_orani.between(0,1).all()
    train = data[data.kaynak=="egitim"]
    rows = []
    for (source, sample), held in data.groupby(["kaynak","sample_id"]):
        refs = train[train.sample_id!=sample]
        threshold = fit(refs)
        values = prediction(held,threshold)
        rows.append(held.assign(esik=threshold,secim_mm=values,secim_hata_mm=abs(values-held.referans_mm)))
    result = pd.concat(rows,ignore_index=True)
    result.to_csv(out/"kesik_tas_yontem_secimi.csv",index=False,float_format="%.6f")
    local = result[result.kaynak=="egitim"]
    metrics = ["gorunen_hata_mm","elips_hata_mm","secim_hata_mm"]
    soils = local.groupby("sample_id")[metrics].mean()
    print(soils.mean())
    print("improved vs ellipse:",(soils.secim_hata_mm<soils.elips_hata_mm).sum(),"/",len(soils))
    print(local.groupby(["sample_id","esik"]).size())
    check=result[result.kaynak=="elle"]
    print("manual check:",check.esik.unique(),(check.secim_mm>63).sum(),len(check))
    print(local.groupby(["sample_id","kesilen_genislik"])[metrics].mean().groupby("kesilen_genislik").mean())


if __name__=="__main__":
    main()
