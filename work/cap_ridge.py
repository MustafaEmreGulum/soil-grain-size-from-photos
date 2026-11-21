import contextlib
import io
import json
from pathlib import Path

out = Path("outputs")
book = json.loads((out / "toprak_ridge_deneyi.ipynb").read_text(encoding="utf-8"))
cells = book["cells"][:8]
cells[0]["source"] = ["# Fotoğraf dokusundan kütle yüzdelerine karşılık gelen çapları tahmin etme\n",
                      "Aynı eğitim verisi ve aynı Ridge ayarlarıyla iki hedef biçimini karşılaştırıyoruz."]
cells[1]["source"] = "".join(cells[1]["source"]).replace('else Path("work/soil-data")', 'else next(p for p in [Path("work/soil-data"), Path("../work/soil-data")] if p.exists())').splitlines(True)

def add(kind, source):
    cells.append(dict(cell_type=kind, metadata={}, source=source.strip().splitlines(True)))

add("markdown", """
## Bir toprağın kütlesini yüz eşit payla gösterme

Yüzdelik noktalarını deneyi çalıştırmadan önce %0,5, %1,5, …, %99,5 olarak seçtik. Laboratuvar yüzdeleri arasında logaritmik çap ekseninde doğrusal ara değer alıyoruz. En küçük elekten geçen payın çapını 0,002 mm sınırında tutuyoruz. Modelin çıktısı her pay için bir logaritmik çap. Tahmini tersine çevirmek için her elekten kaç payın geçtiğini sayıyoruz. Böylece yüzdeler birer puanlık adımlarla değişebilir. Aşağıdaki dönüşüm kontrolü, hiç model kullanmadan bu yaklaşık gösterimin tek başına yarattığı hatayı ölçer.
""")
add("code", """
log_sizes = np.log10(sizes)
levels = np.arange(100) + 0.5
assert np.all(np.diff(y, axis=1) >= 0) and np.all(y[:, -1] == 100)

def to_log_diameters(curve):
    upper = np.searchsorted(curve, levels, side="left")
    lower = np.maximum(upper - 1, 0)
    output = log_sizes[upper].copy()
    between = upper > 0
    hi, lo = upper[between], lower[between]
    fraction = (levels[between] - curve[lo]) / (curve[hi] - curve[lo])
    output[between] = log_sizes[lo] + fraction * (log_sizes[hi] - log_sizes[lo])
    return output

def to_percentages(raw):
    ordered = isotonic_regression(raw, increasing=True, y_min=log_sizes[0], y_max=log_sizes[-1])
    return np.searchsorted(ordered, log_sizes, side="right") * (100 / len(levels))

diameters = np.stack([to_log_diameters(row) for row in y])
restored = np.stack([to_percentages(row) for row in diameters])
assert np.abs(restored - y).max() <= 0.500001
print("Model kullanmadan dönüşümün ortalama EMD’si:", round(np.mean([emd(a, b) for a, b in zip(y, restored)]), 3))
""")
add("markdown", """
## Yüzde tahmini yapan eski Ridge ile çap tahmini yapan yeni Ridge

Üstte çalıştırılan yüzde Ridge ile aynı dört alfa adayını kullanıyoruz. Dış turda saklanan toprağın fotoğraflarını ve hedeflerini model eğitiminden ve alfa seçiminden çıkarıyoruz. İç turlarda çap tahminini tekrar yüzdelere çevirip yarışmanın yerel EMD hesabını uyguluyoruz. İki yöntemin de hedefi sonunda aynı 11 noktadaki hatayı azaltmak.
""")
add("code", """
def diameter_curve(train_idx, sample_idx, alpha):
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    estimator.fit(x[train_idx], diameters[train_idx])
    return to_percentages(estimator.predict(x[[sample_idx]])[0])

diameter_curves, diameter_alphas = [], []
for holdout in ids:
    pool = ids[ids != holdout]
    losses = []
    for alpha in alphas:
        errors = [emd(y[check], diameter_curve(pool[pool != check], check, alpha)) for check in pool]
        losses.append(np.mean(errors))
    best_alpha = alphas[np.argmin(losses)]
    diameter_alphas.append(best_alpha)
    diameter_curves.append(diameter_curve(pool, holdout, best_alpha))

diameter_curves = pd.DataFrame(diameter_curves, index=labels.index, columns=labels.columns)
assert diameter_curves.shape == (24, 11) and np.isfinite(diameter_curves).all().all()
assert np.all(np.diff(diameter_curves, axis=1) >= 0) and diameter_curves["200"].eq(100).all()
assert diameter_curves.ge(0).all().all() and diameter_curves.le(100).all().all()
assert round(scores["Ridge"].mean(), 2) == 48.01
comparison = scores[["Ridge", "Üç komşu"]].copy()
comparison["Çap Ridge"] = np.abs((diameter_curves - labels).to_numpy()[:, :-1]) @ widths
comparison["Kazanç"] = comparison["Ridge"] - comparison["Çap Ridge"]
print(comparison.round(2).to_string())
print("Ortalamalar:", comparison.mean().round(3).to_dict())
print("Önceki Ridge'e göre iyileşen / eşit / kötüleşen:", (comparison.Kazanç > 1e-8).sum(), (comparison.Kazanç.abs() <= 1e-8).sum(), (comparison.Kazanç < -1e-8).sum())
print("Seçilen alfa ve tur sayısı:", pd.Series(diameter_alphas).value_counts().sort_index().to_dict())
print("H038'de 2 mm altı kütle yüzdesi (ölçüm / eski Ridge / çap Ridge):", labels.at["H038", "2"], round(ridge.at["H038", "2"], 2), diameter_curves.at["H038", "2"])
""")

env = {}
for i, cell in enumerate((c for c in cells if c["cell_type"] == "code"), start=1):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec("".join(cell["source"]), env)
    msg = buf.getvalue()
    print(msg, end="", flush=True)
    cell.update(execution_count=i, outputs=[dict(output_type="stream", name="stdout", text=msg.splitlines(True))] if msg else [])
book["cells"] = cells
(out / "toprak_cap_ridge.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
