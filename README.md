# Toprak fotoğraflarından tane boyutu tahmini

Bir toprak fotoğrafından kütlenin yüzde kaçının 11 farklı elekten geçeceğini tahmin ediyoruz. Çalışma 24 eğitim toprağına dayanıyor. D50, toplam kütlenin yarısının geçtiği elek çapıdır. EMD, yarışmanın bütün eğri üzerindeki hata ölçüsüdür ve düşük değer daha iyidir. [Kaggle yarışması](https://www.kaggle.com/competitions/soil-grain-size-from-photos)

## Deney defteri

- **25 Kasım.** En iyi doğrulanmış Kaggle public EMD **33,85350**. Bu sürüm 2 mm’deki düzeltmeyi ince eleklere de aktarıyor. Yerel ortalama 24 toprakta **35,0112**. Sonraki 63 mm deneyinde 162 çekimi taradık. Kenarda kesilen dört adayın tam kısa eksenini doğrulayamadık. [En iyi gönderim](outputs/submission_v3_ince_pay_2mm.csv) · [Deney sonuçları ve sıradaki adım](outputs/GUNCEL_DURUM.md)

- **23 Kasım.** O tarihteki en iyi doğrulanmış Kaggle public EMD **35,63420**. Bu sürüm, v3 ve 20 mm düzeltmesini kullanarak orta boy tanelerin kütlesini üç boyut aralığına paylaştırıyor. Ardından iki ince elek deneyi ve 24 toprakta komşu karşılaştırması tamamlandı. v3’ün 30 görüntü ölçüsünde en yakın toprağın 0,063 mm laboratuvar geçiş farkı ortalama 15,93 yüzde puan, rastgele eşleşmede 31,70. [Gönderim dosyası](outputs/submission_v3_orta_elek_aralik.csv) · [Komşu karşılaştırması](outputs/GUNCEL_DURUM.md#incekomsu)

- **22 Kasım araştırması.** Beş açık Kaggle notebook’unu saf v3 ile karşılaştırdık. İri taşların alanını doğrudan ölçme fikrini H374 ve H615’in sekiz çekiminde inceledik. 20 mm üzerindeki görünür alan ortalaması sırasıyla %12,52 ve %0,00, laboratuvardaki iri kütle payı %23,51 ve %8,86. [Notebook karşılaştırması ve deney defteri](outputs/GUNCEL_DURUM.md) · [40 ölçüm karesi](outputs/h374_h615_20mm_alan.png) · [Çekim sonuçları](outputs/h374_h615_20mm_alan.csv)

- Saf v3’te tüm görüntü özelliklerini 100 × 100 mm kadrajda hesaplamak iki telefon yönünde EMD’yi 1,032 ve 0,898 düşürdü. 24 toprak ortalamasında EMD 38,758’den 40,243’e çıktı. Özgün kadrajda fotoğraf özelliklerinin ortancasını kullanmak 38,794 EMD verdi. Ana model, özgün kadrajlı saf v3. [Kadraj ve ortanca sonuçları](outputs/v3_kadraj_sonuclari.csv) · [Tek çekimi çıkarmanın etkisi](outputs/v3_cekim_duyarliligi.csv)
- Doku ölçüleri, KNN, Ridge, DINOv2, Weibull ve v3 deneylerinin sırası, notebook'lar ve Kaggle gönderimlerinin puanları. [Güncel durum](outputs/GUNCEL_DURUM.md)
- H374 ve H615'in sekiz fotoğrafı aynı fiziksel ölçekte incelendi. Önceki iri toprak sınavında laboratuvar D50'leri 5,91 ve 5,34 mm iken v3 tahminleri 1,37 ve 6,20 mm oldu. [Görüntü karşılaştırması](outputs/h374_h615_ayni_olcek.png) · [Çekim bazında sonuçlar](outputs/h374_h615_cekimleri.csv)
- İkinci bir iri tane adayını ölçmek, 24 toprakta ortalama EMD'yi 38,758'den 38,709'a indirdi. H615'in puanı dışarıda bırakıldığında kazanç tersine döndü. Saf v3 referans model olarak kaldı. [Deney kodu](outputs/ikinci_iri_aday.py) · [24 toprak ve iki telefon sınavının sonuçları](outputs/ikinci_iri_aday_sonuclari.csv)
- Dalgacık enerjisinin bölgelere göre farkı, ortalama enerjiyi kullanan modelin 24 topraktaki EMD’sini 0,708 artırdı. Kütle paylarını öğrenmek saf v3’e göre 17/24 toprakta hatayı azalttı, ancak H493’ün puanı çıkarıldığında sıralama tersine döndü. [Dalgacık tahminleri](outputs/dalgacik_sonuclari.csv) · [Kütle payı tahminleri](outputs/kutle_payi_kontrol_sonuclari.csv) · [Literatür ve deney raporu](outputs/Toprak_Tane_Boyutu_Literatur_ve_Deney_Sonuclari.docx)
- DINOv2 + Ridge için beş kesit, tek kesite göre EMD’yi 50,617’den 48,849’a indirdi. Saf v3 aynı numunelerde 38,758 EMD verdi. 0,063 / 0,2 / 0,63 mm eleklerinin ortalamasında beş kesitli model yalnızca 7/24 toprakta laboratuvar ölçümüne saf v3’ten daha çok yaklaştı. [Tek ve beş kesit karşılaştırması](outputs/dinov2_kesit_ridge_sonuclari.csv) · [198 elek karşılaştırması](outputs/dinov2_ince_elek_sonuclari.csv)

## Beş iri toprağı birlikte saklama deneyi

1. Laboratuvar D50'si en büyük beş toprağı birlikte sakladık. Kalan 19 toprağın en büyük D50'si 4,585 mm idi. Her saklı numunenin bütün fotoğrafları ve türevleri eğitim dışında kaldı.
2. Kontrollü karşılaştırmada aynı toprağı sakladık, diğer dört iri toprağı eğitime aldık ve onların yerine daha ince dört toprağı çıkardık. Her turda eğitimde 19 toprak kaldı. Rastgele seçilen 100 farklı dörtlüyü her hedef ve iki model için ortak kullandık (tohum 42). Önce her hedef için kontrol hatalarının ortalamasını aldık.
3. Vitalclick'in saf v3 sürümü, fiziksel görüntü ölçülerinden yüzdeliklere karşılık gelen çapları tahmin eder. İnce/iri KNN yöntemimiz, üç referans toprağın laboratuvar eğrilerini ortalar. İki yöntemin ayarları deneyden önce sabitlendi. [V3 kaynağı](https://github.com/vitalclick/Predicting-Soil-Grain-Size/tree/0808d05def19e03a6478410827a644bfd411ba81)

## Sonuçlar

| Model | Beş iri toprak birlikte saklı, ortalama EMD | İri referanslı kontrol, ortalama EMD | Fark | Saklı grupta ortanca tahmin/gerçek D50 |
|---|---:|---:|---:|---:|
| Saf v3 | 29,821 | 26,532 | +3,289 | 0,492 |
| İnce/iri KNN | 21,161 | 15,747 | +5,414 | 0,665 |

İri referansları çıkarmak v3'ün hatasını beş toprağın dördünde, KNN'nin hatasını beşinin tamamında artırdı. Beş satırdan hangisini dışarıda bırakırsak bırakalım, ortalama EMD farkı iki model için de pozitif kaldı. Dört toprakla hesaplanan ortalama farklar v3 için +1,960 ile +5,766, KNN için +5,015 ile +5,806 arasındaydı.

| Saklı toprak | Laboratuvar D50 (mm) | v3 D50 (mm) | KNN D50 (mm) | v3 EMD | KNN EMD |
|---|---:|---:|---:|---:|---:|
| H368 | 6,13 | 4,58 | 3,84 | 17,49 | 20,47 |
| H374 | 5,91 | 1,37 | 2,52 | 56,71 | 36,98 |
| H615 | 5,34 | 6,20 | 3,84 | 19,41 | 16,35 |
| H367 | 4,79 | 2,36 | 3,19 | 28,07 | 17,29 |
| H366 | 4,79 | 2,21 | 3,19 | 27,42 | 14,72 |

Saf v3 eğitimdeki en büyük D50'yi yalnızca H615'te aştı. H374'te gerçek çap 5,91 mm iken tahmini 1,37 mm'de kaldı. Diğer dört iri numune eğitime girdiğinde de ortalama tahmini ancak 1,59 mm'ye çıktı. Bu iki numunenin yakın çekimlerinde hangi fiziksel ölçülerin iri taneleri yakaladığını karşılaştırmak bir sonraki deney için öncelikli. Sınav için uçtan seçilmiş beş toprak var. Kontrol tekrarları görüntü sayısını veya bağımsız numune sayısını artırmıyor. Sonuçlar iPhone test kümesi için bir puan tahmini vermiyor. [Numune ve model bazında tam sonuçlar](outputs/iri_toprak_sonuclari.csv)

## Yeniden çalıştırma

Python, Git ve Kaggle eğitim etiketleri gerekiyor. Aşağıdaki komutlar kaynak v3 kodunu ve ona ait hazır görüntü ölçülerini de indirir.

```shell
git clone --recurse-submodules https://github.com/MustafaEmreGulum/soil-grain-size-from-photos.git
cd soil-grain-size-from-photos
python -m pip install -r work/vitalclick/requirements.txt
```

Yarışmadan indirdiğiniz `Training_labels_updated.csv` dosyasını `work/soil-data/` klasörüne koyup aşağıdaki komutu depo kökünden çalıştırın. Deney, alt modüldeki hazır görüntü ölçülerini kullanır ve sonuçları `outputs/iri_toprak_sonuclari.csv` dosyasına yazar.

```shell
python outputs/iri_toprak_sinavi.py
```

H374/H615 karşılaştırması ve ikinci iri aday deneyi için yarışmanın eğitim fotoğraflarını arşivdeki klasör yapısıyla `work/soil-data/` altına çıkarın. Bu iki betik görselleri üretmek için Matplotlib kullanır.

```shell
python -m pip install matplotlib
python outputs/h374_h615_incelemesi.py
python outputs/ikinci_iri_aday.py
```

Sabit kadraj deneyinin 1.143 satırlık görüntü ölçüleri depoda yer alır. Aşağıdaki ilk komut eğitim etiketleriyle tahminleri ve çekim duyarlılığını yeniden hesaplar. İkinci komut 100 × 100 mm ölçülerini de özgün eğitim fotoğraflarından üretir.

```shell
python outputs/v3_sabit_kadraj.py
python outputs/v3_sabit_kadraj.py --rebuild-features
```

Kütle payı deneyi için `python outputs/kutle_payi_kontrolu.py`, dalgacık karşılaştırması için `python outputs/dalgacik_karsilastirmasi.py` çalıştırılır. DINOv2 deneyini görüntülerden tekrarlamak için PyTorch, timm ve safetensors gerekir. `model.safetensors` dosyasını `work/dinov2_small.safetensors` olarak kaydedip `python outputs/dinov2_kesit_ridge.py` çalıştırın. Saklı tahminlerle ince elek denetimi `python outputs/dinov2_ince_elek_kontrolu.py` ile çalışır. [Kullanılan DINOv2 ağırlıkları](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)

## Görsellerin kaynağı

Notebook ve rapor görsellerindeki toprak fotoğrafları, Lukas Leibold ve Geotechnical Resilience Project'in Kaggle'da yayımladığı *Predicting Soil Grain Size Distributions from Images* veri kümesinden alınmıştır. Analizlerde fotoğraflar kırpıldı, milimetre ölçeğine getirildi ve bazı örneklerde ölçüm bölgeleri işaretlendi. Veri sayfasında belirtilen lisans CC BY 4.0'dır. [Veri kaynağı](https://www.kaggle.com/competitions/soil-grain-size-from-photos/data) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
