import base64, contextlib, io, json, re, os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
from matplotlib import pyplot as plt
plt.show = lambda: None
out = Path("outputs")
code = Path("work/doku_dersi.py").read_text(encoding="utf8").replace("üç tahmin eğrisini", "üç eğriyi")
parts = re.split(r"^# %% (md|code)\n", code, flags=re.M)
cells, env, idx = [], {}, 0
for typ, source in zip(parts[1::2], parts[2::2]):
    cell = {"cell_type":"markdown" if typ == "md" else "code", "source":source.strip().splitlines(True), "metadata":{}}
    if typ == "code":
        idx += 1
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(source, env)
        outputs = []
        if buffer.getvalue():
            outputs.append({"output_type":"stream", "name":"stdout", "text":buffer.getvalue().splitlines(True)})
            print(buffer.getvalue())
        if "plt.show()" in source:
            file = out / "f827_uc_komsu.png"
            env["fig"].savefig(file, dpi=115)
            outputs.append({"output_type":"display_data","data":{"image/png":base64.b64encode(file.read_bytes()).decode()},"metadata":{}})
        cell.update(outputs=outputs,execution_count=idx)
    cells.append(cell)
meta = {"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"},"language_info":{"name":"python"}}
(out/"toprak_doku_dersi.ipynb").write_text(json.dumps(dict(cells=cells,metadata=meta,nbformat=4,nbformat_minor=4),ensure_ascii=False,indent=1),encoding="utf8")
print(env["neighbors"]["F827"])
j = list(env["sizes"]).index(.063)
print([obj["F827"][j] for obj in [env["averages"],env["predictions"]]],env["labels"].loc["F827"].iloc[j])
print(env["scores"].to_string())
