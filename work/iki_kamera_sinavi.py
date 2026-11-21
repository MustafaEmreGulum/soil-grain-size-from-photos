import contextlib
import io
import json
from pathlib import Path

out = Path("outputs")
book = json.loads((out / "toprak_doku_dersi.ipynb").read_text(encoding="utf-8"))
cells = book["cells"][:4]
cells[0]["source"] = ["# Yeni bir telefonla çekilen toprak fotoğrafları\n",
                      "Üç komşu modelini iki telefonla çekilmiş aynı numuneler üzerinde karşılaştırıyoruz."]

def add(kind, source):
    cells.append({"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(True)})

add("markdown", """
## İki telefonla da çekilmiş numuneler

Motorola Edge ve Samsung A52 fotoğraflarını ayrı tutuyoruz. Her toprak ve telefon için altı doku ölçüsünün ortancasını alıyoruz. Her iki yönde aynı eğitim ve değerlendirme numunelerini kullanmak için yalnızca ortak numuneleri seçiyoruz.
""")
add("code", '''
phones = ["Motorola_Edge", "Samsung_A52"]
image_rows = pd.DataFrame(rows, columns=["numune", *features.columns])
image_rows["telefon"] = [path.stem.rsplit("_", 2)[0] for path in photos]
soil_phone = image_rows.groupby(["telefon", "numune"])[features.columns.tolist()].median()
by_phone = {p: soil_phone.loc[p] for p in phones}
paired = labels.index[labels.index.isin(by_phone[phones[0]].index) & labels.index.isin(by_phone[phones[1]].index)]
assert len(paired) == 21 and paired.is_unique
print(f"{len(paired)} ortak numune; her turda {len(paired)-1} eğitim numunesi.")
''')
add("markdown", """
## Her numuneyi iki telefonla da sınama

Her turda aynı numunenin tüm fotoğraflarını eğitimden çıkarıyoruz. Seçilen eğitim telefonunda kalan 20 numuneyle ölçeklemeyi ve üç komşu modelini kuruyoruz. Aynı modelle saklanan numunenin iki telefonla çekilmiş görüntülerini ayrı ayrı değerlendiriyoruz. Hata, 11 boyuttaki tahmin eğrisinin laboratuvar eğrisinden logaritmik ağırlıklı uzaklığıdır (EMD). Düşük değer daha iyi.
""")
add("code", '''
widths = np.diff(np.log10(labels.columns.astype(float).to_numpy()))
records = []
for sample in paired:
    others = paired.drop(sample)
    for training_phone in phones:
        train = by_phone[training_phone].loc[others]
        assert sample not in train.index and len(train) == 20
        scaler = StandardScaler().fit(train)
        reference = scaler.transform(train)
        for viewing_phone in phones:
            held_out = by_phone[viewing_phone].loc[[sample]]
            distance = np.linalg.norm(reference - scaler.transform(held_out), axis=1)
            nearest = train.index[np.argsort(distance, kind="stable")[:3]]
            curve = labels.loc[nearest].mean()
            assert len(set(nearest)) == 3 and sample not in nearest
            error = np.abs((curve - labels.loc[sample]).to_numpy()[:-1]) @ widths
            records.append([sample, training_phone, viewing_phone, error, curve["2"]])

results = pd.DataFrame(records, columns=["Numune", "Eğitim", "Çekim", "EMD", "2 mm altı (%)"])
assert len(results) == 84 and not results.duplicated(["Numune", "Eğitim", "Çekim"]).any()
scores = results.pivot(index=["Numune", "Eğitim"], columns="Çekim", values="EMD")
for phone, other in [phones, phones[::-1]]:
    selected = scores.xs(phone, level="Eğitim")
    gap = selected[other] - selected[phone]
    print(phone, "eğitimi")
    print(f"Aynı telefon {selected[phone].mean():.2f} | Diğer telefon {selected[other].mean():.2f}")
    print(f"Ortalama fark {gap.mean():+.2f} | Ortanca fark {gap.median():+.2f}")
    print("Diğer telefonda hata artan / azalan / aynı:", (gap > 1e-9).sum(), (gap < -1e-9).sum(), np.isclose(gap, 0).sum())

print("H038 laboratuvar ölçümü (2 mm altı, %):", labels.loc["H038", "2"])
print(results[results.Numune == "H038"].round(2).to_string(index=False))
print(scores.round(2).to_string())
''')
namespace = {}
count = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    count += 1
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec("".join(cell["source"]), namespace)
    cell.update(execution_count=count, outputs=[])
    if stream.getvalue():
        cell["outputs"].append({"output_type":"stream", "name":"stdout", "text":stream.getvalue().splitlines(True)})
        print(stream.getvalue(), end="")
book["cells"] = cells
(out / "toprak_iki_kamera_sinavi.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
