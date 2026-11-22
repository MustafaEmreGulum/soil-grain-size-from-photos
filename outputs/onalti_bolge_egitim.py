"""Toprak etiketiyle 16 bölgenin ortak eğitimi: eşit ortalama ve öğrenilmiş dikkat."""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

import bes_ve_onalti_doku as base


ig = base.ig
STEM = "onalti_bolge_egitim"
CACHE = ig.work.parent / ".venv" / STEM
INPUT = CACHE / "bolgeler.npz"
LARGER_INPUT = CACHE / "bolgeler_64x32.npz"
LARGER_STEM = STEM + "_64x32"
EPOCHS = 48
SEED = 20261122
MODES = ("esit", "dikkat")


def prepare(input_file=INPUT, width=32, height=16):
    photos = base.prev.photos()
    meta = pd.read_csv(ig.out / "dinov2_hiyerarsik_kadrajlar.csv", index_col="file")
    old = pd.read_csv(base.PHOTO, index_col="file")
    images, textures = [], []
    for file, row in photos.iterrows():
        photo, _, _, gray = base.calibrated_surface(row, meta.loc[file])
        tiles = np.empty((16,height,width), dtype=np.uint8)
        stats = np.empty((16,len(base.LAGS),2), dtype=np.int64)
        for i, (y,x,y1,x1) in enumerate(base.tile_grid(gray.shape)):
            tile = gray[y:y1, x:x1]
            pos = i
            if photo.height > photo.width:
                tile = np.rot90(tile)
                pos = (3-i%4)*4 + i//4
            tiles[pos] = cv2.resize(tile, (width,height), interpolation=cv2.INTER_AREA)
            stats[pos] = base.pair_totals(gray, [(y,x,y1,x1)], shared_edges=True)
        np.testing.assert_allclose(base.contrasts(stats.sum(0)), old.loc[file,base.MODES["onalti"]].to_numpy(float),
                                   rtol=0, atol=1e-14)
        images.append(tiles)
        textures.append(stats[:,:,0] / stats[:,:,1] / 255.)
    CACHE.mkdir(exist_ok=True)
    temp = input_file.with_suffix(".tmp.npz")
    np.savez_compressed(temp, files=photos.index.to_numpy(str), images=np.stack(images), textures=np.stack(textures))
    temp.replace(input_file)
    print("Verified region inputs", np.shape(images), np.shape(textures), flush=True)


class SoilNet(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.conv = nn.Conv2d(1, 8, kernel_size=4, stride=4)
        self.region = nn.Sequential(nn.Linear(22, 16), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 1, bias=False))
        self.head = nn.Linear(24, len(ig.gsd.COLS))
        nn.init.zeros_(self.gate[-1].weight)
        if not attention:
            self.gate.requires_grad_(False)

    def forward(self, pixels, texture, photo_mean, global_features):
        # Nonoverlapping 4x4 blocks were unpacked once. This is the convolution
        # above in matrix form and backpropagates to its kernel and bias.
        act = torch.nn.functional.linear(pixels, self.conv.weight.flatten(1), self.conv.bias).tanh()
        descriptors = torch.cat([act.mean(2), (act.var(2, unbiased=False)+1e-6).sqrt(), texture], dim=-1)
        regions = self.region(descriptors)
        if self.attention:
            weights = self.gate(regions).squeeze(-1).softmax(dim=1)
        else:
            weights = torch.full_like(regions[:,:,0], 1 / regions.size(1))
        photos = (weights.unsqueeze(-1) * regions).sum(1)
        soil = torch.cat([photo_mean @ photos, global_features], dim=1)
        curve = self.head(soil).softmax(1).cumsum(1) * 100
        return curve, weights


def inputs(input_file=INPUT):
    meta = base.prev.photos()
    with np.load(input_file) as z:
        np.testing.assert_array_equal(z["files"], meta.index)
        raw = torch.from_numpy(z["images"]).float() / 64 - 2
        # [photo, region, block within region, 16 grayscale values in block]
        height, width = raw.shape[-2:]
        assert height % 4 == 0 and width % 4 == 0
        tiles = (raw.reshape(len(meta),16,height//4,4,width//4,4).permute(0,1,2,4,3,5)
                 .reshape(len(meta),16,(height//4)*(width//4),16))
        tex = z["textures"]
    with np.load(ig.out / "dinov2_hiyerarsik_temsiller.npz") as z:
        glob = pd.DataFrame(z["genel_yerel"][:,:384].astype(float),index=z["files"]).loc[meta.index].to_numpy()
    assert torch.isfinite(tiles).all() and np.isfinite(tex).all() and tex.shape==(len(meta),16,6)
    return meta, tiles, tex, glob


def make_groups(meta, ids, phone):
    take = meta.sample_id.isin(ids)
    if phone is not None:
        take &= meta.phone == phone
    idx = np.flatnonzero(take)
    group = np.equal.outer(ids, meta.sample_id.to_numpy()[idx]).astype(float)
    assert (group.sum(0) == 1).all() and (group.sum(1) > 0).all()
    return idx, group / group.sum(1, keepdims=True)


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def fold_inputs(data, ids, held, source, dest):
    meta, tiles, textures, glob = data
    train, a = make_groups(meta, ids, source)
    test, b = make_groups(meta, held, dest)
    if set(ids) != set(held):
        assert set(ids).isdisjoint(held) and not np.intersect1d(train,test).size
    gs = StandardScaler().fit(a @ glob[train])
    pca = PCA(n_components=8, whiten=True, svd_solver="full").fit(gs.transform(a @ glob[train]))
    ts = StandardScaler().fit(textures[train].reshape(-1,6))
    def pack(idx, group):
        local = (textures[idx] - ts.mean_) / ts.scale_
        general = pca.transform(gs.transform(group @ glob[idx]))
        return tiles[idx], tensor(local), tensor(group), tensor(general)
    prep = {key: torch.from_numpy(arr.copy()) for key, arr in {
        "texture_mean": ts.mean_, "texture_scale": ts.scale_, "global_mean": gs.mean_,
        "global_scale": gs.scale_, "pca_mean": pca.mean_, "pca_components": pca.components_,
        "pca_std": np.sqrt(pca.explained_variance_)}.items()}
    return pack(train, a), pack(test, b), test, prep


def fit(pack, query, truth, mode):
    torch.manual_seed(SEED)
    model = SoilNet(mode == "dikkat")
    with torch.no_grad():
        avg = torch.cat([truth[:,:1],truth[:,1:]-truth[:,:-1]],1).mean(0)
        model.head.bias.copy_((avg / 100).clamp_min(1.e-5).log())
        model.head.weight.mul_(.1)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.006, weight_decay=.01)
    logwidth = tensor(ig.gsd.DLOG)
    initial = model.conv.weight.detach().clone()
    losses = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        curve, _ = model(*pack)
        loss = ((curve[:,:-1] - truth[:,:-1]).abs() * logwidth).sum(1).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert not torch.allclose(initial,model.conv.weight) and np.isfinite(losses).all()
    with torch.no_grad():
        model.eval()
        curve, weights = model(*query)
        assert torch.allclose(weights.sum(1),torch.ones(len(weights)),atol=1e-6)
        curve = ig.gsd.finalize(curve.numpy())
    return model, curve, weights.numpy(), losses


def predict_saved(path, data, soils):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SoilNet(checkpoint["mode"] == "dikkat")
    model.load_state_dict(checkpoint["state"])
    model.eval()
    prep = checkpoint["prep"]
    meta, tiles, textures, glob = data
    assert tiles.shape[2] == checkpoint.get("blocks_per_region",32)
    idx, group = make_groups(meta, soils, None)
    local = (textures[idx]-prep["texture_mean"].numpy()) / prep["texture_scale"].numpy()
    general = (group @ glob[idx]-prep["global_mean"].numpy()) / prep["global_scale"].numpy()
    general = (general-prep["pca_mean"].numpy()) @ prep["pca_components"].numpy().T / prep["pca_std"].numpy()
    with torch.no_grad():
        curve, weights = model(tiles[idx], tensor(local), tensor(group), tensor(general))
    return ig.gsd.finalize(curve.numpy()), weights.numpy()


def run(larger=False):
    torch.set_num_threads(2)
    input_file = LARGER_INPUT if larger else INPUT
    stem = LARGER_STEM if larger else STEM
    modes = ("esit",) if larger else MODES
    if not input_file.exists():
        prepare(input_file, 64 if larger else 32, 32 if larger else 16)
    data = inputs(input_file)
    if larger:
        with np.load(INPUT) as previous:
            np.testing.assert_array_equal(data[2],previous["textures"])
            np.testing.assert_array_equal(data[0].index,previous["files"])
        assert data[1].shape[2]==128
    meta = data[0]
    ids = sorted(ig.labels.index)
    source, dest = ig.PHONES
    paired = sorted(set(meta[meta.phone==source].sample_id) & set(meta[meta.phone==dest].sample_id))
    assert len(ids)==24 and len(paired)==21
    exams = [("24", ids, None, None), (f"{source} -> {dest}",paired,source,dest),
             (f"{dest} -> {source}",paired,dest,source)]
    records, weight_rows, history = [], [], []
    for exam, samples, train_phone, test_phone in exams:
        for idx, soil in enumerate(samples,1):
            others = [s for s in samples if s != soil]
            pack, query, chosen, _ = fold_inputs(data, others, [soil], train_phone, test_phone)
            truth = tensor(ig.labels.loc[others].to_numpy())
            for mode in modes:
                _, predicted, weights, losses = fit(pack,query,truth,mode)
                score = ig.gsd.emd(ig.labels.loc[soil],predicted)[0]
                records.append(dict(exam=exam,sample_id=soil,model=mode,emd=score,**dict(zip(ig.gsd.COLS,predicted[0]))))
                history.append(dict(exam=exam,sample_id=soil,model=mode,
                                    **{f"epoch_{j}":v for j,v in enumerate(losses,1)}))
                if mode=="dikkat" and exam=="24":
                    for file, w in zip(meta.index[chosen],weights):
                        weight_rows.append(dict(file=file,sample_id=soil,**{f"bolge_{j}":v for j,v in enumerate(w,1)}))
            if idx % 4 == 0 or idx == len(samples):
                print(exam, idx, "/", len(samples), flush=True)
    frame = pd.DataFrame(records)
    assert len(frame)==66*len(modes) and not frame.duplicated(["exam","sample_id","model"]).any()
    frame.to_csv(ig.out / f"{stem}_sonuclar.csv", index=False, float_format="%.9f")
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(ig.out / f"{stem}_agirliklar.csv",index=False,float_format="%.9f")
    # Deployment fits use all soils. Their in-sample losses are kept apart from held-out scores.
    pack, query, _, prep = fold_inputs(data,ids,ids,None,None)
    for mode in modes:
        model, pred, w, losses = fit(pack,query,tensor(ig.labels.loc[ids].to_numpy()),mode)
        path = ig.out / f"{stem}_{mode}.pt"
        temp = path.with_suffix(".tmp.pt")
        torch.save(dict(mode=mode, state=model.state_dict(), prep=prep, soils=ids, epochs=EPOCHS, seed=SEED,
                        blocks_per_region=data[1].shape[2]),temp)
        temp.replace(path)
        restored, restored_w = predict_saved(path, data, ids)
        np.testing.assert_allclose(pred, restored, atol=2e-5, rtol=0)
        np.testing.assert_allclose(w, restored_w, atol=1e-6, rtol=0)
        history.append(dict(exam="tam_egitim",sample_id="hepsi",model=mode,
                            **{f"epoch_{j}":v for j,v in enumerate(losses,1)}))
    pd.DataFrame(history).to_csv(ig.out / f"{stem}_egitim_kaybi.csv",index=False,float_format="%.8f")
    print(frame.groupby(["exam","model"]).emd.mean().unstack())


def plot_weights(weights):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    photos = base.prev.photos()
    roi = pd.read_csv(ig.out / "dinov2_hiyerarsik_kadrajlar.csv",index_col="file")
    files = ["Motorola_Edge_H037_01.jpg","Samsung_A52_H037_01.jpg"]
    fig, axes = plt.subplots(2,1,figsize=(8,8.5),layout="constrained")
    for ax, file in zip(axes, files):
        image, _, _, gray = base.calibrated_surface(photos.loc[file],roi.loc[file])
        rotated = image.height > image.width
        w = weights.loc[file,[f"bolge_{i}" for i in range(1,17)]].to_numpy(float)
        ax.imshow(np.rot90(gray) if rotated else gray, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        for i, (y,x,y1,x1) in enumerate(base.tile_grid(gray.shape)):
            pos = i
            if rotated:
                pos = (3-i%4)*4 + i//4
                y,x,y1,x1 = gray.shape[1]-x1,y,gray.shape[1]-x,y1
            ax.add_patch(Rectangle((x-.5,y-.5),x1-x,y1-y, ec="#fff4d9",fc="none",lw=.8))
            if pos==int(np.argmax(w)):
                ax.add_patch(Rectangle((x+2,y+2),x1-x-5,y1-y-5, ec="#ffd34c",fc="none",lw=2.2))
            label = ("%"+f"{100*w[pos]:.1f}").replace(".",",")
            ax.text((x+x1)/2,(y+y1)/2,label,color="white",ha="center",va="center",
                    bbox=dict(facecolor="#26323a",edgecolor="none",alpha=.82,pad=2.1),fontsize=9)
        ax.set_title(file.split("_H037")[0].replace("_"," "),loc="left",fontweight="bold",fontsize=12)
        ax.set_axis_off()
    fig.suptitle("H037 · eğitimde ayrılan toprağın bölge ağırlıkları",fontsize=15,fontweight="bold")
    fig.legend(handles=[Line2D([],[],color="#d8aa22",lw=2,label="En yüksek ağırlık · eşit ağırlıkta her bölge %6,25")],
               loc="outside lower center",frameon=False,fontsize=9)
    fig.savefig(ig.out / f"{STEM}_H037.png",dpi=140)
    plt.close(fig)


def report():
    results = pd.read_csv(ig.out / f"{STEM}_sonuclar.csv")
    old = pd.read_csv(base.RESULT).query("model == 'onalti'").replace({"model":{"onalti":"ridge"}})
    v3 = pd.read_csv(ig.out / "v3_kadraj_sonuclari.csv").query("model == 'v3'")
    all_rows = pd.concat([results,old,v3],ignore_index=True)
    wide = all_rows.pivot(index=["exam","sample_id"],columns="model",values="emd")
    assert wide.shape == (66,4) and wide.notna().all().all()
    curves = all_rows[ig.gsd.COLS].to_numpy()
    actual = ig.labels.loc[all_rows.sample_id,ig.gsd.COLS].to_numpy()
    assert (np.diff(curves,axis=1)>=0).all() and (curves[:,-1]==100).all()
    np.testing.assert_allclose(abs(curves-actual)[:,:-1] @ np.diff(np.log10(ig.gsd.SIZES)), all_rows.emd,
                               rtol=0,atol=1e-8)
    print(wide.groupby(level="exam").mean().round(5))
    for exam, rows in wide.groupby(level="exam"):
        for control in ("esit","ridge","v3"):
            gain = rows[control]-rows.dikkat
            omitted = (gain.sum()-gain)/(len(gain)-1)
            print(exam,control,"vs dikkat: wins/median/omit range",sum(gain>0),gain.median(),omitted.min(),omitted.max())
    main = all_rows.query("exam == '24'").copy()
    main["fine_error"] = abs(main["0.2"].to_numpy()-ig.labels.loc[main.sample_id,"0.2"].to_numpy())
    print("H037",main.query("sample_id=='H037'")[["model","emd","0.2"]].to_dict("records"))
    print("0.2 mm MAE",main.groupby("model").fine_error.mean().to_dict())
    w = pd.read_csv(ig.out / f"{STEM}_agirliklar.csv",index_col="file")
    col = [f"bolge_{i}" for i in range(1,17)]
    assert len(w)==127 and w.index.is_unique and (w[col]>0).all().all()
    np.testing.assert_allclose(w[col].sum(1),1,rtol=0,atol=2e-7)
    print("H037 weights min/max",w.query("sample_id=='H037'")[col].agg(["min","max"],axis=1).round(5))
    print("All photos largest weight median",w[col].max(axis=1).median())
    print("All-data training losses",pd.read_csv(ig.out / f"{STEM}_egitim_kaybi.csv").query("exam=='tam_egitim'")[["model","epoch_1","epoch_48"]])
    plot_weights(w)


def report_resolution():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {"32x16": STEM, "64x32": LARGER_STEM}
    frames = []
    for name, stem in paths.items():
        frame = pd.read_csv(ig.out / f"{stem}_sonuclar.csv").query("model=='esit'").copy()
        frame["size"] = name
        frames.append(frame)
    result = pd.concat(frames,ignore_index=True)
    curves = result[ig.gsd.COLS].to_numpy()
    target = ig.labels.loc[result.sample_id,ig.gsd.COLS].to_numpy()
    assert (np.diff(curves,axis=1)>=0).all() and (curves[:,-1]==100).all()
    np.testing.assert_allclose(abs(curves-target)[:,:-1] @ np.diff(np.log10(ig.gsd.SIZES)),
                               result.emd,rtol=0,atol=1e-8)
    wide = result.pivot(index=["exam","sample_id"],columns="size",values="emd")
    assert wide.shape==(66,2) and wide.notna().all().all()
    print(wide.groupby(level="exam").mean().round(5))
    for exam, sub in wide.groupby(level="exam"):
        g = sub["32x16"]-sub["64x32"]
        omitted = (g.sum()-g)/(len(g)-1)
        print(exam, "wins, mean gain, median gain, omission range", (g>0).sum(),g.mean(),g.median(),omitted.min(),omitted.max())
    main = result.query("exam=='24'").copy()
    main["fine_error"] = abs(main["0.2"].to_numpy()-ig.labels.loc[main.sample_id,"0.2"].to_numpy())
    print("0.2 mm MAE",main.groupby("size").fine_error.mean().to_dict())
    print("H037",main.query("sample_id=='H037'")[["size","emd","0.2"]].to_dict("records"))
    file = "Motorola_Edge_H037_01.jpg"
    row = base.prev.photos().loc[file]
    meta = pd.read_csv(ig.out / "dinov2_hiyerarsik_kadrajlar.csv",index_col="file").loc[file]
    _, _, _, gray = base.calibrated_surface(row,meta)
    # Row 3, column 4 in the landscape display contains stone 2 in the earlier full-surface audit.
    pos = 11
    native_i = (pos%4)*4 + 3-pos//4
    y,x,y1,x1 = base.tile_grid(gray.shape)[native_i]
    original = np.rot90(gray[y:y1,x:x1])
    images = [original]
    for path in (INPUT,LARGER_INPUT):
        with np.load(path) as z:
            tile = z["images"][np.flatnonzero(z["files"]==file).item(),pos]
            np.testing.assert_array_equal(tile,cv2.resize(original,(tile.shape[1],tile.shape[0]),interpolation=cv2.INTER_AREA))
            images.append(tile)
    fig, axes = plt.subplots(1,3,figsize=(10,2.6),layout="constrained")
    for ax, im, title in zip(axes,images,("Kalibre edilmiş bölge","Ağın girdisi · 32 × 16","Ağın girdisi · 64 × 32")):
        ax.imshow(im,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
        ax.set_title(title,fontsize=11,loc="left")
        ax.set_axis_off()
    fig.suptitle("H037 · Motorola · aynı bölgenin görüntü koluna girişi",fontsize=14,fontweight="bold")
    fig.savefig(ig.out / f"{STEM}_cozunurluk_H037.png",dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare","run","report","run64","report64"])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.step=="prepare":
            prepare()
        elif args.step in ("run","run64"):
            run(larger=args.step=="run64")
        elif args.step=="report64":
            report_resolution()
        else:
            report()
