import base64
import contextlib
import io
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
out = Path("outputs")
book = json.loads((out / "toprak_doku_dersi.ipynb").read_text(encoding="utf-8"))
cells = book["cells"][:6]
cells[0]["source"] = ["# İri tane adaylarının kapladığı alan\n",
                      "Altı doku ölçüsüne, görüntüde ayırt edilen iri tane adaylarının yüzey alanı yüzdesini ekliyoruz."]

def add(kind, text):
    cells.append(dict(cell_type=kind, metadata={}, source=text.strip().splitlines(True)))

add("markdown", """
## Görüntüden yüzey alanını yaklaşık ölçmek

Yaklaşık 100 mm genişliğindeki 400 piksellik kesitte 2 mm, 8 piksele denk gelir. Zeminden daha açık görünen bağlı bölgeleri bulup alanı 8 piksel çaplı bir dairenin alanına ulaşanları sayıyoruz. Eşik her görüntünün parlaklık dağılımından Otsu yöntemiyle belirleniyor. Seçilen piksellerin kesitteki payı tek yeni ölçümüz. Birleşen taneler tek bölge olabilir, düşük kontrastlı veya koyu taneler gözden kaçabilir. Laboratuvarın kütle yüzdesi ayrı bir ölçümdür.
""")
add("code", '''
import cv2

def coarse_mask(patch):
    gray = np.asarray(patch.convert("L"), dtype=np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1)
    background = cv2.GaussianBlur(gray, (0, 0), 16)
    relief = np.clip(smooth - background, 0, 255).astype(np.uint8)
    _, bright = cv2.threshold(relief, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, components, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    min_area = np.pi * (8 / 2) ** 2
    selected = stats[:, cv2.CC_STAT_AREA] >= min_area
    selected[0] = False
    return selected[components]

area_rows = []
for path in photos:
    phone, sample, _ = path.stem.rsplit("_", 2)
    patch = physical_patch(path, camera_by_name[phone.lower()])
    mask = coarse_mask(patch)
    area_rows.append((sample, 100 * mask.mean()))
    if path.stem == "Motorola_Edge_H038_03":
        example, example_mask = patch, mask

area = pd.DataFrame(area_rows, columns=["numune", "aday alan (%)"]).groupby("numune").median()
extended = features.join(area, how="left")
assert extended.shape == (24, 7) and np.isfinite(extended).all().all()
assert extended["aday alan (%)"].between(0, 100).all()
''')
add("markdown", """
## Aynı 24 numunede iki model

Her turda saklanan toprağın bütün çekimleri eğitim dışında kalır. Ölçeklemeyi yalnızca diğer 23 numuneden öğreniyoruz. Her iki modelde de en yakın üç numunenin laboratuvar eğrilerinin ortalamasını alıyoruz.
""")
add("code", '''
curves = {}
for sample in labels.index:
    train = extended.drop(index=sample)
    scaler = StandardScaler().fit(train)
    delta = scaler.transform(train) - scaler.transform(extended.loc[[sample]])
    nearest = train.index[np.argsort(np.linalg.norm(delta, axis=1), kind="stable")[:3]]
    assert sample not in nearest and len(set(nearest)) == 3
    curves[sample] = labels.loc[nearest].mean().to_numpy()
new_curves = pd.DataFrame.from_dict(curves, orient="index", columns=labels.columns).loc[labels.index]
assert np.all(np.diff(new_curves, axis=1) >= -1e-10) and new_curves["200"].eq(100).all()
comparison = scores[["Üç komşu"]].copy()
comparison["Alan ekli"] = np.abs((new_curves - labels).to_numpy()[:, :-1]) @ widths
assert round(comparison["Üç komşu"].mean(), 2) == 44.74
changes = comparison["Alan ekli"] - comparison["Üç komşu"]
same = np.isclose(changes, 0)
print(comparison.round(2).to_string())
print("Ortalama EMD:", comparison.mean().round(3).to_dict())
print("İyileşen / kötüleşen / aynı:", sum((changes < 0) & ~same), sum((changes > 0) & ~same), sum(same))
print("H038, 2 mm altı (ölçüm / önceki tahmin / yeni tahmin):",
      labels.loc["H038", "2"], predictions["H038"][list(labels.columns).index("2")], new_curves.loc["H038", "2"])
''')
add("code", '''
fig, (left, right) = plt.subplots(1, 2, figsize=(7.7, 3.9), layout="constrained")
left.imshow(example)
left.set_title("H038 · Motorola 03")
right.imshow(example)
overlay = np.zeros((*example_mask.shape, 4))
overlay[example_mask] = [0.94, 0.39, 0.04, 0.68]
right.imshow(overlay)
pct = f"{100 * example_mask.mean():.1f}".replace(".", ",")
right.set_title(f"Turuncu aday alan · %{pct}")
for ax in (left, right):
    ax.set_axis_off()
plt.show()
''')
env = {}
idx = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    idx += 1
    src = "".join(cell["source"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src.replace("plt.show()", "fig.canvas.draw()"), env)
    cell.update(outputs=[], execution_count=idx)
    if buf.getvalue():
        print(buf.getvalue(), end="")
        cell["outputs"].append(dict(output_type="stream", name="stdout", text=buf.getvalue().splitlines(True)))
    if "plt.show()" in src:
        png = out / "h038_iri_tane_alani.png"
        env["fig"].savefig(png, dpi=130, bbox_inches="tight")
        cell["outputs"].append(dict(output_type="display_data", metadata={},
                                    data={"image/png": base64.b64encode(png.read_bytes()).decode()}))
book["cells"] = cells
(out / "toprak_iri_tane_alani.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
