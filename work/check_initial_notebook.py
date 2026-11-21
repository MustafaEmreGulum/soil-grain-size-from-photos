import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import numpy as np
from PIL import Image

os.environ["MPLBACKEND"] = "Agg"
book = json.loads(Path("outputs/toprak_ilk_gonderim.ipynb").read_text(encoding="utf-8"))
headers = "0.002 0.0063 0.02 0.063 0.2 0.63 2 6.3 20 63 200".split()
with TemporaryDirectory(dir="work") as tmp:
    root = Path(tmp).resolve()
    data = root / "input" / "competitions" / "soil-grain-size-from-photos"
    data.mkdir(parents=True)
    curves = np.array([[0]*10 + [100], [100]*11])
    labels = pd.DataFrame(curves, columns=headers)
    labels.insert(0, "sample_id", ["F001", "F002"])
    labels.to_csv(data / "Training_labels_updated.csv", index=False, sep=";")
    template = labels.copy()
    template["sample_id"] = ["002", "001"]
    template.rename(columns={"2":"2.0"}).to_csv(data / "sample_submission.csv", index=False)
    for sample in labels.sample_id:
        Image.new("RGB", (100, 100)).save(data / f"camera_{sample}_0.JPG")
    state = {}
    for entry in book["cells"]:
        if entry["cell_type"] == "code":
            code = "".join(entry["source"])
            code = code.replace('Path("/kaggle/input")', repr(str(root / "input")).join(["Path(", ")"]))
            code = code.replace('Path("/kaggle/working")', repr(str(root / "working")).join(["Path(", ")"]))
            exec(code, state)
    assert all(abs(value-500) < 1e-9 for value in state["errors"])
    assert state["saved"].iloc[:, 0].tolist() == ["002", "001"]
    assert (state["saved"].iloc[:, 1:-1] == 50).all().all()
    assert all(len(ax.images) == 1 for ax in state["axes"][:, 0])
    print("Notebook calculations, chart and CSV verified.")
