import contextlib
import io
import json
from pathlib import Path

out = Path("outputs")
book = json.loads((out / "toprak_bolge_deneyi.ipynb").read_text(encoding="utf8"))
cells = book["cells"][:3]
cells[0]["source"] = ["# Telefon değişince 16 bölgeli ve sade modellerin hatası\n",
                      "Aynı topraklar, aynı fotoğraflar ve aynı üç komşu kuralı ile iki yöntemi karşılaştırıyoruz. Ölçüm birimi toprak numunesi."]
cells[1]["source"] = "".join(cells[1]["source"]).replace('else Path("work/soil-data")', 'else next(p for p in [Path("work/soil-data"), Path("../work/soil-data")] if p.exists())').splitlines(True)

def add(kind, source):
    cells.append(dict(cell_type=kind, metadata={}, source=source.strip().splitlines(True)))

add("markdown", """
## Bir telefonda öğrenip iki telefonda ölçme

Her telefon ve toprak için çekimlerin ortancasını alıyoruz. Her turda iki telefonda da bulunan 21 topraktan birini saklıyoruz. Ölçekleme ve üç komşu seçimi için kalan 20 toprağın yalnızca eğitim telefonuyla çekilmiş fotoğraflarını kullanıyoruz. Önce altı doku ölçüsünü, sonra bu altı ölçüyle birlikte 16 bölgedeki değişkenliği kullanıyoruz. Her model için iki tahminin laboratuvara olan hatalarını ve birbirlerine olan uzaklıklarını aynı EMD ölçeğinde hesaplıyoruz.
""")
add("code", """
phones = ["Motorola_Edge", "Samsung_A52"]
frame["telefon"] = [p.stem.rsplit("_", 2)[0] for p in photos]
assert frame["sample_id"].tolist() == [p.stem.rsplit("_", 2)[1] for p in photos]
per_phone = frame.groupby(["telefon", "sample_id"])[columns].median()
motorola, samsung = [per_phone.loc[p].index for p in phones]
paired = labels.index[labels.index.isin(motorola) & labels.index.isin(samsung)]
assert len(paired) == 21 and paired.is_unique
widths = np.diff(np.log10(labels.columns.astype(float)))

def emd(a, b):
    return np.abs(a - b)[:-1] @ widths

records = []
for name, keys in [("Sade", scales), ("16 bölge", columns)]:
    for source, other in [phones, phones[::-1]]:
        for sample in paired:
            train = per_phone.loc[source, keys].loc[paired.drop(sample)]
            assert len(train) == 20 and sample not in train.index
            scaler = StandardScaler().fit(train)
            reference = scaler.transform(train)
            curves = []
            for viewed in [source, other]:
                query = per_phone.loc[viewed, keys].loc[[sample]]
                distance = np.linalg.norm(reference - scaler.transform(query), axis=1)
                nearest = train.index[np.argsort(distance, kind="stable")[:3]]
                curves.append(labels.loc[nearest].mean().to_numpy())
            truth = labels.loc[sample].to_numpy()
            same_error, cross_error = [emd(curve, truth) for curve in curves]
            records.append([name, source, sample, same_error, cross_error, cross_error - same_error, emd(*curves)])
results = pd.DataFrame(records, columns=["Model", "Eğitim telefonu", "Numune", "Aynı", "Diğer", "Hata farkı", "Tahminler arası EMD"])
assert len(results) == 84 and not results.duplicated(["Model", "Eğitim telefonu", "Numune"]).any()
metrics = ["Aynı", "Diğer", "Hata farkı", "Tahminler arası EMD"]
summary = results.groupby(["Model","Eğitim telefonu"])[metrics].mean()
assert np.allclose(summary.loc["Sade", ["Aynı", "Diğer"]], [[45.98,49.10],[45.01,45.26]], atol=0.005, rtol=0)
print(summary.round(3).to_string())
print("İki eğitim yönünün eşit ağırlıklı ortalaması")
print(results.groupby("Model")[metrics].mean().round(3).to_string())
""")
add("code", """
print("Numune başına EMD")
print(results.pivot(index=["Numune", "Eğitim telefonu"], columns="Model", values=["Aynı", "Diğer"]).round(2).to_string())
""")
env = {}
for i, cell in enumerate((c for c in cells if c["cell_type"] == "code"), 1):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec("".join(cell["source"]), env)
    msg = stream.getvalue()
    print(msg, end="", flush=True)
    cell.update(execution_count=i, outputs=[dict(output_type="stream", name="stdout", text=msg.splitlines(True))] if msg else [])
book["cells"] = cells
(out / "toprak_bolge_kamera_karsilastirmasi.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf8")
