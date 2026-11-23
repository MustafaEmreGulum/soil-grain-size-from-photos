"""Görüntüdeki en yakın toprağın 0,063 mm laboratuvar geçişini karşılaştır."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from coklu_elek_deneyi import OUT, TEXTURE
from v3_knn_esit_karisim import keys, labels, original


def main():
    gran = pd.read_csv(OUT.parent / "work/vitalclick/cache/granulometry_train.csv")
    assert original.file.is_unique and set(gran.file) == set(original.file) and len(gran) == 127
    ids = labels.index.sort_values()
    lab = labels.loc[ids, "0.063"].to_numpy()
    all_gaps = np.abs(lab[:, None] - lab)
    random_mean = all_gaps.sum() / (len(ids)*(len(ids)-1))
    rng = np.random.default_rng(42)
    shuffled = np.stack([rng.permutation(lab) for _ in range(10000)])
    details, summaries = [], []
    for name, photos, columns in [("v3_30", original, keys), ("doku_6", gran, TEXTURE)]:
        centers = photos.groupby("sample_id")[columns].median().loc[ids]
        near = []
        for i, sample in enumerate(ids):
            ref = centers.drop(sample)
            scaler = StandardScaler().fit(ref)
            delta = scaler.transform(ref) - scaler.transform(centers.loc[[sample]])
            dist = np.linalg.norm(delta, axis=1)
            j = dist.argmin()
            neighbor = ref.index[j]
            near.append(ids.get_loc(neighbor))
            details.append(dict(ozellik=name, toprak=sample, komsu=neighbor, uzaklik=dist[j],
                                gecis=lab[i], komsu_gecis=lab[near[-1]], fark_pp=all_gaps[i,near[-1]],
                                rastgele_fark_pp=all_gaps[i].sum()/(len(ids)-1)))
        gaps = np.abs(lab-lab[near])
        null = np.abs(shuffled-shuffled[:,near]).mean(axis=1)
        summaries.append(dict(ozellik=name, toprak=len(ids), benzersiz_cift=len({tuple(sorted((i,j))) for i,j in enumerate(near)}),
                              ort_fark_pp=gaps.mean(), ortanca_fark_pp=np.median(gaps),
                              rastgele_ort_pp=random_mean, en_az_10_pp=(gaps>=10).sum(),
                              en_az_20_pp=(gaps>=20).sum(), permutasyon_p=(1+(null<=gaps.mean()).sum())/(1+len(null))))
    details = pd.DataFrame(details)
    assert len(details) == 48 and (details.toprak != details.komsu).all()
    details.to_csv(OUT/"ince_pay_komsu_ciftleri.csv", index=False, float_format="%.7f")
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/"ince_pay_komsu_ozet.csv", index=False, float_format="%.7f")
    print(summary.round(4).to_string(index=False))
    print(details[details.toprak.isin(["H616","H405"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
