import contextlib
import io
import json
from pathlib import Path

out = Path("outputs")
book = json.loads((out / "toprak_doku_dersi.ipynb").read_text(encoding="utf8"))
cells = book["cells"][:6]
cells[0]["source"] = ["# DINOv2 üç komşu seçimini iyileştiriyor mu?\n", "Her turda bir toprağın bütün çekimlerini saklıyoruz. Altı doku ölçüsü ile hazır DINOv2'nin görüntüden çıkardığı 384 sayıyı karşılaştırıyoruz. Her iki yöntem de seçtiği üç toprağın laboratuvar eğrilerinin ortalamasını kullanıyor."]
cells[1]["source"] = "".join(cells[1]["source"]).replace('else Path("work/soil-data")', 'else next(p for p in [Path("work/soil-data"), Path("../work/soil-data")] if p.exists())').splitlines(True)

def add(kind, source):
    cells.append(dict(cell_type=kind, metadata={}, source=source.strip().splitlines(True)))

add("markdown", """
## Her fotoğraftan 384 ölçü

DINOv2-S/14 önceden eğitilmiş bir görüntü modeli. Ağırlıklarını sabit tutuyoruz. Modelin tüm görüntüyü özetleyen vektörünü (CLS) alıyoruz. Aynı 100 mm genişliğindeki renkli kesiti 224 × 224 piksele indiriyoruz. Doku yöntemi aynı kesiti 400 × 400 pikselde gri tonlarda işler. Fark bu iki görüntü işleme yolunun toplam etkisini ölçer.

Yerelde çalışma klasöründeki ağırlık dosyası kullanılır. Kaggle'da aynı dosyayı `dinov2_small.safetensors` adıyla girdi olarak ekleyebilir veya internet açıkken modelin indirilmesini sağlayabilirsiniz. [Model ağırlıkları](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
""")
add("code", """
import hashlib
import timm
import torch
from safetensors.torch import load_file

torch.set_num_threads(min(4, torch.get_num_threads()))
name = "vit_small_patch14_dinov2.lvd142m"
paths = [Path("work/dinov2_small.safetensors"), Path("../work/dinov2_small.safetensors")]
if Path("/kaggle/input").exists():
    paths.extend(Path("/kaggle/input").rglob("dinov2_small.safetensors"))
weights = next((p for p in paths if p.is_file()), None)
if weights is None:
    from huggingface_hub import hf_hub_download
    weights = Path(hf_hub_download("timm/" + name, "model.safetensors"))
with weights.open("rb") as f:
    assert hashlib.file_digest(f, "sha256").hexdigest() == "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
model = timm.create_model(name, pretrained=False, dynamic_img_size=True, num_classes=0)
model.load_state_dict(load_file(str(weights)), strict=True)
model.eval()
mean = torch.tensor(model.pretrained_cfg["mean"]).view(3, 1, 1)
std = torch.tensor(model.pretrained_cfg["std"]).view(3, 1, 1)
ids, vectors = [], []
with torch.inference_mode():
    for start in range(0, len(photos), 8):
        batch = []
        for path in photos[start:start + 8]:
            phone, sample, _ = path.stem.rsplit("_", 2)
            patch = physical_patch(path, camera_by_name[phone.lower()])
            rgb = np.array(patch.resize((224, 224), Image.Resampling.BICUBIC))
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
            batch.append((tensor - mean) / std)
            ids.append(sample)
        vectors.extend(model(torch.stack(batch)).numpy())
embeddings = pd.DataFrame(vectors, index=ids).groupby(level=0).median().loc[labels.index]
assert embeddings.shape == (24, 384) and np.isfinite(embeddings).all().all()
print(len(ids), "fotoğraf,", len(embeddings), "toprak,", embeddings.shape[1], "özellik")
""")
add("markdown", """
## Aynı 24 toprakta iki yöntemi karşılaştır

Her turda uzaklık ölçeğini yalnızca diğer 23 topraktan öğreniyoruz. Saklanan toprağın laboratuvar eğrisine tahmini yaptıktan sonra bakıyoruz. Pozitif “Kazanç” DINOv2'nin hatayı kaç puan azalttığını gösterir. H038 için H371 ve H617'nin görüntü sırasını da kaydediyoruz.
""")
add("code", """
records = []
for sample in labels.index:
    train = embeddings.drop(index=sample)
    scaler = StandardScaler().fit(train)
    distance = np.linalg.norm(scaler.transform(train) - scaler.transform(embeddings.loc[[sample]]), axis=1)
    order = train.index[np.argsort(distance, kind="stable")].tolist()
    chosen = order[:3]
    assert sample not in chosen and len(set(chosen)) == 3
    predicted = labels.loc[chosen].mean().to_numpy()
    assert np.all(np.diff(predicted) >= -1e-10) and np.isclose(predicted[-1], 100)
    observed = labels.loc[sample].to_numpy()
    err = emd(observed, predicted)
    old = scores.loc[sample, "Üç komşu"]
    lab = sorted(train.index, key=lambda other: emd(observed, labels.loc[other].to_numpy()))[:3]
    records.append([sample, old, err, old - err, len(set(chosen) & set(lab)), ", ".join(chosen)])
    if sample == "H038":
        print("H038 için görüntü sırası:", {s: order.index(s) + 1 for s in ["H371", "H617"]})
report = pd.DataFrame(records, columns=["Numune", "Doku", "DINOv2", "Kazanç", "Ortak", "Yeni komşular"]).set_index("Numune")
assert len(report) == 24 and abs(report.Doku.mean() - 44.744) < 0.001
print(report.round(2).to_string())
print("Ortalama EMD:", report[["Doku", "DINOv2"]].mean().round(3).to_dict())
print("Daha iyi / eşit / daha kötü:", int((report.Kazanç > 1e-8).sum()), int((report.Kazanç.abs() <= 1e-8).sum()), int((report.Kazanç < -1e-8).sum()))
print("Laboratuvarın ilk üçüyle ortak seçim:", report.Ortak.sum(), "/ 72")
""")
env = {}
for i, cell in enumerate((c for c in cells if c["cell_type"] == "code"), start=1):
    print("Hücre", i, flush=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec("".join(cell["source"]), env)
    msg = buf.getvalue()
    print(msg, flush=True)
    cell.update(execution_count=i, outputs=[dict(output_type="stream", name="stdout", text=msg.splitlines(True))] if msg else [])
book["cells"] = cells
(out / "toprak_dinov2_uc_komsu.ipynb").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf8")
