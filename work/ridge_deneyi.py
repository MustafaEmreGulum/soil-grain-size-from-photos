import base64
import contextlib
import io
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
out = Path("outputs")
previous = json.loads((out / "toprak_doku_dersi.ipynb").read_text(encoding="utf-8"))
cells = previous["cells"][:6]
cells[0]["source"] = ["# Altı doku ölçüsüyle Ridge regresyonu\n",
                      "Aynı numuneler üzerinde ortalama eğri, üç komşu ve Ridge tahminlerini karşılaştırıyoruz."]

def add(kind, source):
    cells.append({"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(True)})

add("markdown", """
## 3. Ridge ile tahmin

Ridge her tane çapı için altı doku ölçüsünden bir yüzde tahmini öğrenir. Büyük katsayıları cezalandırma gücünü belirleyen alfa değerini, her turda yalnızca eğitim için ayrılmış 23 numune içinde seçiyoruz. Dört adayımız 0,1, 1, 10 ve 100. Her adayın başarısını içeride bir numuneyi sırayla ayırıp geçerli tahmin eğrisinin EMD hatasını ölçerek karşılaştırıyoruz.

Çıktıyı 0 ile 100 arasında tutuyoruz. Azalan yüzdeler çıkarsa izotonik regresyonla eğriyi artmayan parçaları düzeltecek biçimde düzenliyoruz. Son değer her zaman 100.
""".replace("artmayan parçaları düzeltecek biçimde", "azalan kısımları düzeltecek biçimde"))
add("code", '''
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

def ridge_curve(x, y, query, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, y[:, :-1])
    raw = model.predict(query)[0]
    valid = isotonic_regression(raw, increasing=True, y_min=0, y_max=100)
    return np.append(valid, 100.0)

x = features.to_numpy()
y = labels.to_numpy()
ids = np.arange(len(x))
alphas = [0.1, 1, 10, 100]
curves = []
chosen = []
for holdout in ids:
    train_ids = ids[ids != holdout]
    inner_scores = []
    for alpha in alphas:
        inner_errors = []
        for check in train_ids:
            fit_ids = train_ids[train_ids != check]
            curve = ridge_curve(x[fit_ids], y[fit_ids], x[[check]], alpha)
            inner_errors.append(emd(y[check], curve))
        inner_scores.append(np.mean(inner_errors))
    alpha = alphas[np.argmin(inner_scores)]
    chosen.append(alpha)
    curves.append(ridge_curve(x[train_ids], y[train_ids], x[[holdout]], alpha))

ridge = pd.DataFrame(curves, index=labels.index, columns=labels.columns)
assert ridge.shape == (24, 11) and np.isfinite(ridge).all().all()
assert np.all(np.diff(ridge, axis=1) >= -1e-10) and ridge["200"].eq(100).all()
assert ridge.ge(0).all().all() and ridge.le(100).all().all()
scores["Ridge"] = np.abs((ridge - labels).to_numpy()[:, :-1]) @ widths
assert round(scores["Üç komşu"].mean(), 2) == 44.74
print(scores.mean().round(2).to_string())
print("Ridge daha düşük hata:", (scores["Ridge"] < scores["Üç komşu"]).sum(), "/", len(scores))
print("H038 · 2 mm altı (%)", labels.loc["H038", "2"], predictions["H038"][list(labels.columns).index("2")], ridge.loc["H038", "2"])
print(scores.round(2).to_string())
print("Seçilen alfa ve tur sayısı:", pd.Series(chosen).value_counts().sort_index().to_dict())
''')
add("markdown", """
## H038 ve F827’de tahmin eğrileri

Yeşil eğri laboratuvar ölçümünü gösterir. Her tahmin, ilgili numunenin tüm fotoğrafları ve ölçümleri eğitim dışında tutularak üretilmiştir.
""")
add("code", '''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, layout="constrained")
for ax, sample in zip(axes, ["H038", "F827"]):
    ax.semilogx(sizes, labels.loc[sample], "o-", color="#146b60", label="Laboratuvar")
    ax.semilogx(sizes, predictions[sample], "--", color="#705678", label="Üç komşu")
    ax.semilogx(sizes, ridge.loc[sample], "s-", markersize=4, color="#bb551f", label="Ridge")
    ax.set(title=sample, xlabel="Tane çapı (mm)", ylim=(-3, 105))
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Daha ince tanelerin kütle payı (%)")
axes[1].legend()
plt.show()
''')
env = {}
count = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    count += 1
    src = "".join(cell["source"])
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(src.replace("plt.show()", "fig.canvas.draw()"), env)
    cell.update(execution_count=count, outputs=[])
    if capture.getvalue():
        print(capture.getvalue(), end="")
        cell["outputs"].append(dict(output_type="stream", name="stdout", text=capture.getvalue().splitlines(True)))
    if "plt.show()" in src:
        png = io.BytesIO()
        env["fig"].savefig(png, format="png", dpi=120)
        cell["outputs"].append(dict(output_type="display_data", metadata={},
                                    data={"image/png":base64.b64encode(png.getvalue()).decode()}))
previous["cells"] = cells
(out / "toprak_ridge_deneyi.ipynb").write_text(json.dumps(previous, ensure_ascii=False, indent=1), encoding="utf-8")
