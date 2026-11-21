import contextlib
import io
import json
from pathlib import Path

out = Path("outputs")
book = json.loads((out / "toprak_doku_dersi.ipynb").read_text(encoding="utf8"))
camera = json.loads((out / "toprak_bolge_kamera_karsilastirmasi.ipynb").read_text(encoding="utf8"))
cells = book["cells"][:6]
cells[0]["source"] = ["# Altı ölçekte göreli doku ve üç komşu\n", "Doku ölçülerinin toplam büyüklüğünü kaldırmanın komşu seçimine etkisini iki yerel sınavda ölçüyoruz."]
cells[1]["source"] = camera["cells"][1]["source"]

def add(kind, source):
    cells.append(dict(cell_type=kind, metadata={}, source=source.strip().splitlines(True)))

add("markdown", """
## Her çekimde ölçüleri kendi toplamına bölme

Önceki deneydeki görüntü kırpma ve gri tonları düzeltme adımlarını kullanıyoruz. Bu kez, örneğin 1 mm aralığındaki açık-koyu farkını, aynı çekimdeki altı farkın toplamına bölüyoruz. Altı ölçü birden aynı katsayıyla büyürse oranlar değişmez. Tümü sıfır olan bir fotoğrafın göreli özellikleri de sıfırdır. Her toprağın fotoğraflarını bundan sonra ortanca ile birleştiriyoruz. Başka ayar seçmiyoruz.
""")
add("code", """
scales = features.columns.tolist()
frame = pd.DataFrame(rows, columns=["sample_id", *scales])
relative = ["pay_" + s for s in scales]
total = frame[scales].sum(axis=1)
shares = frame[scales].div(total.where(total > 0, 1), axis=0)
assert np.allclose(shares.sum(axis=1), (total > 0).astype(float))
frame[relative] = shares.to_numpy()
columns = scales + relative
soil = frame.groupby("sample_id")[relative].median().loc[labels.index]
err = []
for sample in labels.index:
    train = soil.drop(index=sample)
    scaler = StandardScaler().fit(train)
    distance = np.linalg.norm(scaler.transform(train) - scaler.transform(soil.loc[[sample]]), axis=1)
    selected = train.index[np.argsort(distance, kind="stable")[:3]]
    assert len(set(selected)) == 3 and sample not in selected
    curve = labels.loc[selected].mean().to_numpy()
    err.append(emd(labels.loc[sample].to_numpy(), curve))
comparison = scores[["Üç komşu"]].rename(columns={"Üç komşu":"Sade"}).copy()
comparison["Göreli"] = err
comparison["Kazanç"] = comparison["Sade"] - comparison["Göreli"]
assert len(comparison) == 24 and np.isfinite(comparison).all().all()
assert round(comparison["Sade"].mean(), 3) == 44.744
print(comparison.round(2).to_string())
print(comparison.mean().round(3))
print("İyileşen / aynı / kötüleşen:", (comparison.Kazanç > 1e-8).sum(), (comparison.Kazanç.abs() <= 1e-8).sum(), (comparison.Kazanç < -1e-8).sum())
""")
add("markdown", "## Telefon değişiminde aynı kontrol\n\nHer iki telefonla çekilen aynı 21 toprakta, eğitimde kullandığımız 20 toprağın çekimlerini tek telefondan alıyoruz. Saklanan toprağın iki telefondaki çekimlerini ayrı ayrı tahmin ediyoruz.")
camera_code = next("".join(c["source"]) for c in camera["cells"] if c["cell_type"] == "code" and "per_phone = " in "".join(c["source"]))
add("code", camera_code.replace('("16 bölge", columns)', '("Göreli", relative)'))
add("markdown", "## On test toprağı için gönderim dosyası\n\nEğitimdeki 24 toprağı referans alıyoruz. Her test çekiminde altı doku ölçüsünü kendi toplamına bölüyor, ardından aynı toprağın çekimleri arasında ortanca alıyoruz.")
add("code", r'''
import re
template = pd.read_csv(data / "sample_submission.csv").set_index("sample_id")
assert template.index.is_unique and template.columns.equals(labels.columns)
camera_map = {c.phone.replace(" ", "").lower(): c for c in cameras.itertuples()}
test_rows = []
for path in sorted((data / "Test_All_Photos").rglob("*")):
    if path.suffix.lower() != ".jpg":
        continue
    phone, suffix = path.stem.split("_HPC_", 1)
    sample = "HPC_" + re.sub(r" \(\d+\)$", "", suffix)
    sample = sample.replace("Münster_BS6_9,0-10m", "Muenster_BS6_9_0-10m")
    assert sample in template.index
    values = np.array(texture(physical_patch(path, camera_map[phone.replace("_", "").lower()])))
    test_rows.append([sample, *(values / values.sum() if values.sum() > 0 else values)])
test_features = pd.DataFrame(test_rows, columns=["sample_id", *relative]).groupby("sample_id").median()
assert len(test_rows) == 35 and len(test_features) == 10 and set(test_features.index) == set(template.index)
scaler = StandardScaler().fit(soil)
reference = scaler.transform(soil)
for sample, query in zip(template.index, scaler.transform(test_features.loc[template.index])):
    distances = np.linalg.norm(reference - query, axis=1)
    nearest = soil.index[np.argsort(distances, kind="stable")[:3]]
    template.loc[sample] = labels.loc[nearest].mean()
destination = next(p for p in [Path("/kaggle/working"), Path("outputs"), Path(".")] if p.is_dir()) / "submission_3nn_goreli_doku.csv"
template.to_csv(destination, float_format="%.8f")
saved = pd.read_csv(destination).set_index("sample_id")
pd.testing.assert_frame_equal(saved, template, atol=1e-8, rtol=0)
assert saved.shape == (10, 11) and np.isfinite(saved).all().all()
assert np.all(np.diff(saved, axis=1) >= 0) and saved.ge(0).all().all() and saved.le(100).all().all() and saved["200"].eq(100).all()
print(destination, "· 10 numune, 11 geçerli yüzde")
''')
env = {}
for i, cell in enumerate((c for c in cells if c["cell_type"] == "code"), 1):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec("".join(cell["source"]), env)
    msg = stream.getvalue()
    print(msg, end="", flush=True)
    cell.update(execution_count=i, outputs=[dict(output_type="stream", name="stdout", text=msg.splitlines(True))] if msg else [])
book["cells"] = cells
(out / "toprak_goreli_doku.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf8")
