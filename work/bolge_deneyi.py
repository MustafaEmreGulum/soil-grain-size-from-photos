import base64
import contextlib
import io
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
root = Path("outputs")
previous = json.loads((root / "toprak_doku_dersi.ipynb").read_text(encoding="utf-8"))
old_code = ["".join(cell["source"]) for cell in previous["cells"] if cell["cell_type"] == "code"]
cells = []
namespace = {}


def markdown(source):
    cells.append(dict(cell_type="markdown", metadata={}, source=source.strip().splitlines(True)))


def run(source, figure=False):
    source = source.strip()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(source.replace("plt.show()", "fig.canvas.draw()"), namespace)
    outputs = []
    if stream.getvalue():
        print(stream.getvalue(), end="")
        outputs.append(dict(output_type="stream", name="stdout", text=stream.getvalue().splitlines(True)))
    if figure:
        path = root / "h038_bolgeler.png"
        namespace["fig"].savefig(path, dpi=120, bbox_inches="tight")
        outputs.append(dict(output_type="display_data", data={"image/png":base64.b64encode(path.read_bytes()).decode()}, metadata={}))
    count = 1 + sum(cell["cell_type"] == "code" for cell in cells)
    cells.append(dict(cell_type="code", execution_count=count, metadata={}, outputs=outputs, source=source.splitlines(True)))


markdown("""# Doku fotoğrafın içinde ne kadar değişiyor?

Önceki üç komşu modeline tek bir bilgi ekliyoruz. Her fiziksel ölçekte 16 bölgenin doku değerlerinin standart sapmasını hesaplıyoruz. Değer yüksekse bölgelerin dokuları birbirinden daha farklı. Her numuneyi sırayla dışarıda bırakarak iki yöntemi aynı fotoğraflarda karşılaştırıyoruz.""")
run(old_code[0])
crop_code = old_code[1].split("def texture(patch):")[0]
run(crop_code + '''
def normalized_gray(patch):
    gray = np.asarray(patch.convert("L"), dtype=float)
    low, high = np.percentile(gray, [5, 95])
    return np.clip((gray - low) / max(high - low, 1.0), 0, 1)

def texture_at_six_scales(gray):
    values = []
    for offset in [1, 2, 4, 8, 16, 32]:
        x = np.abs(gray[:, offset:] - gray[:, :-offset]).mean()
        y = np.abs(gray[offset:, :] - gray[:-offset, :]).mean()
        values.append((x + y) / 2)
    return values

cameras_by_name = {c.phone.replace(" ", "_").lower(): c for c in cameras.itertuples()}
feature_rows = []
for path in photos:
    phone, sample, _ = path.stem.rsplit("_", 2)
    patch = physical_patch(path, cameras_by_name[phone.lower()])
    gray = normalized_gray(patch)
    regions = [gray[y:y+100, x:x+100] for y in range(0, 400, 100) for x in range(0, 400, 100)]
    region_textures = np.array([texture_at_six_scales(region) for region in regions])
    spread = region_textures.std(axis=0)
    feature_rows.append([sample, *texture_at_six_scales(gray), *spread])
    if path.stem == "Motorola_Edge_H038_03":
        example = patch.copy()

scales = ["0.25", "0.5", "1", "2", "4", "8"]
columns = scales + ["sd_" + s for s in scales]
frame = pd.DataFrame(feature_rows, columns=["sample_id", *columns])
features = frame.groupby("sample_id").median().loc[labels.index]
assert features.shape == (24, 12) and np.isfinite(features).all().all()
''')
markdown("""## İki yöntemin yerel sınavı

Üç komşuyu bulmadan önce doku ölçülerini yalnızca geride kalan 23 numune üzerinden standartlaştırıyoruz. Her numuneye bir oy veriyoruz. Bir numunenin bütün çekimleri aynı tarafta kalıyor. EMD küçüldükçe laboratuvar eğrisine daha çok yaklaşıyoruz.""")
run('''
sizes = labels.columns.astype(float).to_numpy()
widths = np.diff(np.log10(sizes))
predictions, errors = {}, {}
for name, keys in [("Önceki", scales), ("Bölgeli", columns)]:
    curves = []
    for sample in labels.index:
        train = features[keys].drop(index=sample)
        scaler = StandardScaler().fit(train)
        delta = scaler.transform(train) - scaler.transform(features.loc[[sample], keys])
        nearest = train.index[np.argsort(np.linalg.norm(delta, axis=1), kind="stable")[:3]]
        assert sample not in nearest and len(set(nearest)) == 3
        curves.append(labels.loc[nearest].mean())
    predicted = pd.DataFrame(curves, index=labels.index)
    assert np.all(np.diff(predicted, axis=1) >= -1e-9) and predicted["200"].eq(100).all()
    predictions[name] = predicted
    errors[name] = np.abs((predicted - labels).to_numpy()[:, :-1]) @ widths

scores = pd.DataFrame(errors, index=labels.index)
change = scores["Bölgeli"] - scores["Önceki"]
assert round(scores["Önceki"].mean(), 2) == 44.74
print(scores.round(2).to_string())
print("Ortalama EMD", scores.mean().round(3).to_dict())
print("H038 hariç ortalama EMD", scores.drop("H038").mean().round(3).to_dict())
same = np.isclose(change, 0)
print("İyileşen / kötüleşen / aynı:", sum((change < 0) & ~same), sum((change > 0) & ~same), sum(same))
values = {name: pred.loc["H038", "2"] for name, pred in predictions.items()}
print("H038 • 2 mm altı (%)", {**values, "Laboratuvar": labels.loc["H038", "2"]})
''')
markdown("""## H038’de model hangi alanları gördü?

Solda sekiz çekimden birinde kullanılan yaklaşık 10 × 10 cm kesit var. Sağdaki tahmin, numunenin bütün çekimlerinden hesaplandı.""")
run('''
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
left.imshow(example)
for edge in [99.5, 199.5, 299.5]:
    left.axhline(edge, color="white", linewidth=1)
    left.axvline(edge, color="white", linewidth=1)
left.set(title="H038 · 16 bölge")
left.set_axis_off()
right.semilogx(sizes, labels.loc["H038"], "o-", label="Laboratuvar", color="#146b60")
right.semilogx(sizes, predictions["Önceki"].loc["H038"], "--", label="Önceki model", color="#75467f")
right.semilogx(sizes, predictions["Bölgeli"].loc["H038"], "s-", label="Bölgeli model", color="#bf5117", markersize=4)
right.set(xlabel="Tane çapı (mm)", ylabel="Daha ince tanelerin kütle payı (%)", ylim=(-3, 104))
right.grid(alpha=0.2)
right.legend()
plt.show()
''', figure=True)
book = dict(nbformat=4, nbformat_minor=4, metadata=previous["metadata"], cells=cells)
(root / "toprak_bolge_deneyi.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
