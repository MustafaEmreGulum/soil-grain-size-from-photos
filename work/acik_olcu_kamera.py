import contextlib
import io
import json
from pathlib import Path

out = Path("outputs")
path = out / "v3_ve_komsu_secimi.ipynb"
book = json.loads(path.read_text(encoding="utf8"))
book["cells"] = book["cells"][:8]
camera = json.loads((out / "toprak_bolge_kamera_karsilastirmasi.ipynb").read_text(encoding="utf8"))
protocol = next("".join(c["source"]) for c in camera["cells"] if "per_phone = " in "".join(c["source"]))
intro = """
## 21 ortak toprakta telefon değişimi

Özgün fotoğrafların dosya adlarını, toprak ve telefon kimliklerini eşleştiriyoruz. Aynı toprak iki telefonla da çekilmişse sınava giriyor. Her turda bir toprağı saklayıp kalan 20 toprağın tek telefondaki ölçüleriyle üç komşuyu buluyoruz. İkinci telefonun da aynı toprağa ait ölçüleriyle ayrıca tahmin yapıyoruz. İki yönün ortalamasında her toprağın ağırlığı eşit.
"""
setup = """
scales = features.columns.tolist()
frame = pd.DataFrame(rows, columns=["sample_id", *scales])
frame["file"] = [p.name for p in photos]
matched = original.set_index("file").loc[frame.file]
assert np.array_equal(matched.sample_id, frame.sample_id)
assert matched.phone.tolist() == [p.stem.rsplit("_", 2)[0] for p in photos]
open_columns = keys.copy()
assert len(open_columns) == 30
frame[open_columns] = matched[open_columns].to_numpy()
columns = scales + open_columns
"""
results = """
paired_errors = results.groupby(["Numune","Model"])["Diğer"].mean().unstack()
gain = paired_errors["Sade"] - paired_errors["30 ölçü"]
assert len(gain) == 21 and np.isfinite(gain).all()
print("İki yön ortalamasında iyileşen/eşit/kötüleşen:", (gain>1e-8).sum(), (gain.abs()<=1e-8).sum(), (gain< -1e-8).sum())
print(paired_errors.assign(Kazanç=gain).round(2).sort_values("Kazanç").to_string())
"""
book["cells"].extend([
    dict(cell_type="markdown", metadata={}, source=intro.strip().splitlines(True)),
    dict(cell_type="code", metadata={}, source=(setup + protocol.replace('("16 bölge", columns)', '("30 ölçü", open_columns)') + "\n" + results).strip().splitlines(True))
])
env = {}
for i, cell in enumerate((c for c in book["cells"] if c["cell_type"] == "code"), 1):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec("".join(cell["source"]), env)
    msg = stream.getvalue()
    cell.update(execution_count=i, outputs=[dict(output_type="stream", name="stdout", text=msg.splitlines(True))] if msg else [])
    if cell is book["cells"][-1]:
        print(msg, flush=True)
path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf8")
