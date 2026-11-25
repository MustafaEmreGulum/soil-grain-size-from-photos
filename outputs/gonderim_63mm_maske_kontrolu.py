"""63 mm üstü aday görüldüğünde referansı koruyan tam geçiş kontrolünün gönderimi."""
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops_table

from h615_otomatik_tas import surface
from munster_ikinci_olcek import COARSE, WEIGHTS, list_images, load_model, out, work
from v3_knn_esit_karisim import COLS, labels
from gsd import emd


def main():
    train = pd.read_csv(out/"ikinci_olcek_egitim_fotograflar.csv")
    assert len(train)==127 and train.file.is_unique and train.n63.eq(0).all()
    local = pd.read_csv(out/"ince_paya_2mm_sonuclar.csv").query("exam=='toprak'").set_index("sample_id")
    assert set(local.index)==set(train.sample_id) and len(local)==24
    curves = local[COLS].copy()
    curves["63"] = 100.
    scores = emd(labels.loc[curves.index,COLS],curves)
    saved = pd.read_csv(out/"ikinci_olcek_egitim_sonuclar.csv").query("exam=='toprak' and model=='tam_gecis'")
    np.testing.assert_allclose(scores,saved.set_index("sample_id").loc[curves.index,"emd"],rtol=0,atol=1e-7)
    print("Yerel EMD",local.emd.mean(),scores.mean(),flush=True)

    base = pd.read_csv(out/"submission_v3_ince_pay_2mm.csv",index_col="sample_id")
    template = pd.read_csv(work/"soil-data/sample_submission.csv",index_col="sample_id")
    assert base.index.equals(template.index) and list(base.columns)==list(template.columns)==COLS
    photos = list_images(work/"soil-data").query("split=='test'").set_index("file")
    boxes = pd.read_csv(out/"test20_fotograflar.csv",index_col="file")
    assert set(photos.index)==set(boxes.index) and set(photos.sample_id)==set(base.index)
    with np.load(out/"munster_ikinci_olcek_maske.npz") as old:
        known_name = old["photo_name"].item()
        assert old["ppm"].item()==COARSE
        known_mask = old["mask"]
    model = None
    candidate = base.copy()
    audit = []
    for sample in base.index[base["63"]<100]:
        subset = photos[photos.sample_id==sample]
        # Tek bir pozitif çekim koruma kararı için yeterli.
        for name,row in sorted(subset.iterrows(),key=lambda item:item[0]!=known_name):
            with Image.open(row.path) as photo:
                width,height = sorted(photo.size,reverse=True)
            box = boxes.loc[name].copy()
            box["width"],box["height"] = width,height
            rgb,_,_ = surface(row,box,resolution=COARSE)
            if name==known_name:
                mask = known_mask
            else:
                if model is None:
                    model = load_model(WEIGHTS)
                mask,*_ = model.eval(rgb,batch_size=1,bsize=256,flow_threshold=.4,cellprob_threshold=0.)
            assert mask.shape==rgb.shape[:2]
            axes = regionprops_table(mask,properties=("axis_minor_length",))["axis_minor_length"]/COARSE
            peak = float(axes.max(initial=0.))
            audit.append(dict(sample_id=sample,file=name,max_kisa_mm=peak))
            print(name,round(peak,2),flush=True)
            if peak>63:
                break
        else:
            assert len(subset)>0
            candidate.loc[sample,"63"] = 100.
    assert candidate.drop(columns="63").equals(base.drop(columns="63"))
    assert np.isfinite(candidate).all().all() and candidate.ge(0).all().all() and candidate.le(100).all().all()
    assert (np.diff(candidate.to_numpy(),axis=1)>=0).all()
    pd.DataFrame(audit).to_csv(out/"gonderim_63mm_kontrol_cekimleri.csv",index=False,float_format="%.6f")
    dest = out/"submission_v3_63mm_kontrollu.csv"
    candidate.to_csv(dest,float_format="%.6f")
    pd.testing.assert_frame_equal(pd.read_csv(dest,index_col="sample_id"),candidate)
    print(base["63"].compare(candidate["63"]))


if __name__=="__main__":
    main()
