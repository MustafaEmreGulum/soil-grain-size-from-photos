import contextlib
import io
import json
from pathlib import Path

out = Path("outputs")
book = json.loads((out / "toprak_doku_dersi.ipynb").read_text(encoding="utf-8"))
cells = book["cells"][:6]
cells[0]["source"] = ["# Üç komşu yönteminde ulaşılabilir en düşük hata\n",
                      "Fotoğraflı modelin üçlüsü ile laboratuvar eğrisi bilindiğinde seçilebilecek en iyi üçlüyü karşılaştırıyoruz."]

def add(kind, source):
    cells.append(dict(cell_type=kind, metadata={}, source=source.strip().splitlines(True)))

add("markdown", """
## Her saklı numune için 1.771 üçlü

Kalan 23 numuneden üç farklı numune seçip eğrilerinin eşit ağırlıklı ortalamasını alıyoruz. Aynı üçlü bütün çaplarda kullanılıyor. Laboratuvar sonucuna bakıp EMD'yi en aza indiren üçlüyü seçmek, yalnızca bu yönteme ve bu referans numunelerine özgü matematiksel bir alt sınır verir. Gerçek test numuneleri için bu seçim yapılamaz.
""")
add("code", '''
from itertools import combinations
from math import comb

records = []
for sample in labels.index:
    pool = labels.drop(index=sample)
    groups = np.array(list(combinations(range(len(pool)), 3)))
    assert len(groups) == comb(23, 3) == 1771
    candidates = pool.to_numpy()[groups].mean(axis=1)
    observed = labels.loc[sample].to_numpy()
    errors = np.abs(candidates[:, :-1] - observed[:-1]) @ widths
    best = int(errors.argmin())
    chosen = neighbors[sample]
    chosen_indices = sorted(pool.index.get_loc(n) for n in chosen)
    match = np.flatnonzero((groups == chosen_indices).all(axis=1))
    assert len(match) == 1 and np.isclose(errors[match[0]], scores.loc[sample, "Üç komşu"])
    floor = errors[best]
    model = scores.loc[sample, "Üç komşu"]
    ids = pool.index[groups[best]].tolist()
    assert sample not in ids and floor <= model + 1e-10
    assert np.isclose(floor, emd(observed, labels.loc[ids].mean().to_numpy()))
    records.append([sample, model, floor, model - floor, ", ".join(chosen), ", ".join(ids)])

table = pd.DataFrame(records, columns=["Numune", "Fotoğraf EMD", "Alt sınır", "Fark", "Fotoğrafın seçimi", "En iyi üçlü"]).set_index("Numune")
assert len(table) == 24 and round(table["Fotoğraf EMD"].mean(), 2) == 44.74
print(table.sort_values("Fark", ascending=False).round(2).to_string())
print("Ortalama:", table.iloc[:, :3].mean().round(3).to_dict())
print("Ortanca:", table.iloc[:, :3].median().round(3).to_dict())
''')
env = {}
n = 0
for cell in cells:
    if cell["cell_type"] == "code":
        n += 1
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec("".join(cell["source"]), env)
        msg = buf.getvalue()
        print(msg, end="")
        cell.update(execution_count=n, outputs=[dict(output_type="stream", name="stdout", text=msg.splitlines(True))] if msg else [])
book["cells"] = cells
(out / "toprak_en_iyi_uclu.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
