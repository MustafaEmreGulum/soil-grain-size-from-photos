# Toprak fotoğraflarından tane boyutu tahmini

Bu deney, fotoğraftan bir toprağın 11 elek çapındaki kümülatif kütle yüzdelerini tahmin eden iki yöntemi karşılaştırıyor. Sorumuz şu: Model, eğitimde benzer irilikte hiç numune görmediğinde ne kadar hata yapıyor? D50, toplam kütlenin yarısının geçtiği elek çapıdır. EMD, yarışmanın bütün eğri üzerindeki hata ölçüsüdür ve düşük değer daha iyidir. [Kaggle yarışması](https://www.kaggle.com/competitions/soil-grain-size-from-photos)

## Deney

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
