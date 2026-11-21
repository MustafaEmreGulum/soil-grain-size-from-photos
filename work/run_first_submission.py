import base64
import contextlib
import io
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out = Path("outputs")
book_path = out / "toprak_ilk_gonderim.ipynb"
book = json.loads(book_path.read_text(encoding="utf-8"))
state = {}
count = 0
for cell in book["cells"]:
    if cell["cell_type"] != "code":
        continue
    count += 1
    source = "".join(cell["source"])
    local_source = source.replace('Path("/kaggle/input")', 'Path("work/soil-data")')
    local_source = local_source.replace('Path("/kaggle/working")', 'Path("outputs")')
    local_source = local_source.replace('plt.show()', 'fig.canvas.draw()')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(local_source, f"<notebook-cell-{count}>", "exec"), state)
    output = []
    if buf.getvalue():
        output.append({"output_type":"stream","name":"stdout","text":buf.getvalue().splitlines(True)})
        print(buf.getvalue(), end="")
    if "plt.show()" in source:
        assert all(len(ax.images) == 1 for ax in state["axes"][:, 0])
        path = out / "iki_toprak_numunesi.png"
        state["fig"].savefig(path, dpi=150, bbox_inches="tight")
        png = base64.b64encode(path.read_bytes()).decode("ascii")
        output.append({"output_type":"display_data","data":{"image/png":png},"metadata":{}})
    cell.update(execution_count=count, outputs=output)

template = state["template"]
saved = pd.read_csv(out / "submission.csv")
truth = state["curves"]
cols = state["target_cols"]
assert saved.shape == template.shape and list(saved) == list(template)
assert len(saved) == saved.sample_id.nunique() and list(saved.sample_id) == list(template.sample_id)
assert np.isfinite(saved[cols]).all().all() and saved[cols].ge(0).all().all() and saved[cols].le(100).all().all()
assert np.all(np.diff(saved[cols], axis=1) >= 0) and saved[cols[-1]].eq(100).all()
avg_others = (truth.sum(axis=0) - truth) / (len(truth) - 1)
crosscheck = (abs(truth - avg_others)[:, :-1] @ np.diff(np.log10(state["diameters"]))).mean()
assert np.isclose(crosscheck, np.mean(state["errors"]))
book_path.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
for idx in state["order"][[0,-1]]:
    print(state["samples"].iloc[idx], truth[idx, state["two_mm"]])
print("Validated", saved.shape, "local EMD", crosscheck)
