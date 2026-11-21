import contextlib
import io
import json
from pathlib import Path

out = Path("outputs")
book = json.loads((out / "toprak_goreli_doku.ipynb").read_text(encoding="utf8"))
cells = book["cells"][:4]
cells[0]["source"] = ["# Yayımlanmış v3 ve aynı ölçülerle üç komşu\n", "24 toprakta karşılaştırma. Kaynak: [vitalclick, sabit sürüm](https://github.com/vitalclick/Predicting-Soil-Grain-Size/tree/0808d05def19e03a6478410827a644bfd411ba81). Yerelde work/vitalclick altında src ve cache klasörleri gerekir."]
def add(kind, source):
    cells.append(dict(cell_type=kind, metadata={}, source=source.strip().splitlines(True)))

add("markdown", "## v3 doğrulaması\nYazarın iki doğrusal PLS modelini birleştiren v3 kodunu ve paylaştığı hazır ölçüleri kullanıyoruz. Her turda saklanan toprağın tüm çekimleri ve türetilmiş görüntüleri eğitim dışında kalır. Eğitimde türetilmiş çekimler de vardır, sınavda yalnızca özgün çekimler vardır.")
add("code", """
import sys
from scipy.stats import spearmanr
repo = next(p for p in [Path("work/vitalclick"), Path("../work/vitalclick")] if p.is_dir())
sys.path.insert(0, str((repo / "src").resolve()))
import run as published
import gsd
cache = pd.read_csv(repo / "cache/features.csv")
cols = [c for c in cache if c not in published.META]
original = cache[(cache.split == "train") & (cache.aug == 0)].copy()
assert sorted(original.file) == sorted(p.name for p in photos)
assert all(r.sample_id == Path(r.file).stem.rsplit("_", 2)[1] for r in original.itertuples())
assert list(labels.columns) == gsd.COLS
published.LINEAR_ONLY = True
v3 = published.loso(cache, labels, pd.DataFrame(gsd.curve_to_logq(labels.values), index=labels.index), cols)
print(v3.mean().round(3))
""")
add("markdown", "## Ölçüler değiştiğinde komşular değişiyor mu?\nYazarın v3’te kullandığı iki özellik listesinin birleşimini alıyoruz. İki model için de özgün çekimleri numune başına ortancayla birleştiriyoruz. Saklanan toprağı çıkardıktan sonra her özelliği standartlaştırıp en yakın üç numunenin eğrisini ortalıyoruz. Laboratuvar komşularını yalnızca seçimleri değerlendirmek için kullanıyoruz.")
add("code", """
sets = published.feature_sets(cols)
keys = list(dict.fromkeys(sets["R1"] + sets["R2"]))
external = original.groupby("sample_id")[keys].median().loc[labels.index]
assert len(external) == 24 and np.isfinite(external).all().all()
print("Özellik sayıları:", {k: len(v) for k,v in sets.items()}, "Birleşim:", len(keys))
records = []
for sample in labels.index:
    others = labels.index.drop(sample)
    truth = labels.loc[sample].to_numpy()
    lab_distance = gsd.emd(labels.loc[others].values, truth)
    lab_order = others[np.argsort(lab_distance, kind="stable")]
    for name, frame in [("Sade", features), ("Açık ölçüler", external)]:
        train = frame.loc[others]
        scaler = StandardScaler().fit(train)
        dist = np.linalg.norm(scaler.transform(train) - scaler.transform(frame.loc[[sample]]), axis=1)
        idx = np.argsort(dist, kind="stable")[:3]
        chosen = others[idx]
        curve = labels.loc[chosen].mean().to_numpy()
        assert sample not in chosen and len(set(chosen)) == 3
        records.append([sample, name, gsd.emd(truth, curve)[0], len(set(chosen)&set(lab_order[:3])),
                        lab_distance[idx].mean(), spearmanr(dist, lab_distance).statistic, ", ".join(chosen)])
detail = pd.DataFrame(records, columns=["Numune","Model","EMD","İsabet","Komşu EMD","Sıra uyumu","Komşular"])
assert len(detail) == 48
print(detail.groupby("Model").agg({"EMD":"mean","İsabet":"sum","Komşu EMD":"mean","Sıra uyumu":"mean"}).round(3))
per_soil = detail.pivot(index="Numune", columns="Model", values="EMD")
per_soil["v3"] = v3["ENSEMBLE"]
assert round(per_soil.Sade.mean(),3) == 44.744
print(per_soil.round(2).to_string())
for name in ["Açık ölçüler","v3"]:
    delta = per_soil[name] - per_soil.Sade
    print(name, "iyileşen / eşit / kötüleşen:", (delta < -1e-8).sum(), (delta.abs() <= 1e-8).sum(), (delta > 1e-8).sum())
print(detail[detail.Numune.isin(["H038","H405"])].round(2).to_string(index=False))
""")
env = {}
for i, cell in enumerate((c for c in cells if c["cell_type"]=="code"),1):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec("".join(cell["source"]), env)
    msg = stream.getvalue()
    print(msg, end="", flush=True)
    cell.update(execution_count=i, outputs=[dict(output_type="stream",name="stdout",text=msg.splitlines(True))] if msg else [])
book["cells"] = cells
(out / "v3_ve_komsu_secimi.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf8")
