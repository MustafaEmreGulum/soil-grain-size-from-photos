# Güncel durum — Fotoğraftan toprak tane boyutu tahmini

## Açık Kaggle notebooklarından v3’e aktarım planı · 23 Kasım

**En iyi Kaggle public sonucumuz 35,63420.** Saf v3 ve 20 mm düzeltmesinin üzerine üç boyut aralığının görünür alanlarını ayrı ölçen model, 24 toprakta 35,486 yerel EMD verdi. Yarı yarıya düzeltme public sınamada 35,73986 aldı. [Orta elek karşılaştırması](#ortaelek)

| Sorun ve kaynak fikir | Bizdeki bulgu | Durum ve sıradaki karar |
|---|---|---|
| İri taşlar doku ortalamasında kayboluyor. SAM notebook’u ve Sadam, fotoğrafta ölçülen iri taneleri ilgili eleklerde kullanıyor. | H615’te eski beş ölçüm karesi adayı kaçırdı. Tam yüzeyde tek otomatik maskenin elle çizilen sınırla IoU’su Motorola’da 0,915 ve Samsung’da 0,938. [Otomatik sınır kontrolü](h615_otomatik_tas.png) | **127 çekim tamamlandı.** Yerel kazanç 24 toprakta 1,361 EMD, H374 ve H615 hariç 0,718 EMD. 10 test toprağında karşılaştırma tamamlandı. Üç tahmin değişti. [Gönderim adayı](submission_v3_tam_yuzey_20mm.csv) [GPU sonuçları](#gpu20) |
| Eğitim aralığını aşan dokuda koşullu çap tahmini (Sadam’dan esinlenen yerel uyarlama). | Her sınamada yalnız H615 eşiği geçti. En iri beş toprak birlikte saklandığında 20 mm referansının EMD’si 27,408 iken koşullu çap 50,381 verdi. | Bu çap kuralını bıraktık. [Koşullu çap ve elek sonuçları](#cokluelek) |
| Tam yüzey düzeltmesini diğer eleklere taşıma. | 2 ve 6,3 mm tam düzeltme yerelde 35,711, public sınamada 37,68440 verdi. | Üç boyut aralığı 35,63420, yarı düzeltme 35,73986 public EMD verdi. [Kısmi düzeltme sonuçları](#ortaelek) |
| İnce eleklerde fazla veya eksik tahmin. Swathi, tahmini 2 mm geçişine göre kum için ayrı eğri seçiyor. | 0,063, 0,2 ve 0,63 mm birlikte saf v3’ün yerel EMD’sinin %45,49’unu oluşturuyor. Tüm topraklara Weibull uyguladığımızda EMD 39,210 oldu. Swathi’nin bir adayındaki %2,78 ince pay tek eğitim toprağına dayanıyor. | **Açık.** Yalnız görüntüden kum ağırlıklı tahmin edilen topraklarda ayrı eğri kullanmayı sına. Üç ince eleği, H037/H038 gibi karışık toprakları ve tüm eğriyi ayrı raporla. [Mevcut Weibull karşılaştırması](../work/weibull_karsilastirma.py) |
| Elek eğrisinin şeklini az veriyle öğrenmek. Sadam, geçiş yüzdelerini logit dönüşümünden sonra üç temel bileşenle özetleyip Ridge ile tahmin ediyor. | Biz sıralı elek ve kütle payı hedeflerini denedik. Bu iki deney Sadam’ın üç bileşenli eğri temsilini sınamıyor. [Sıralı elek sonuçları](sirali_elek_sonuclari.csv) | **Yedekte.** Aynı görüntü özellikleri ve aynı Ridge ile çap yüzdelikleri ve üç eğri bileşenini karşılaştır. Bileşenleri yalnızca eğitim eğrilerinden çıkar. |
| Tamamlayıcı tahminleri birleştirmek veya basit regresyonla kontrol etmek. prvsiyan iki yayımlanmış tahmini birleştiriyor. Soil grain V4’te Elastic Net seçiliyor. | Kendi %50 v3 + %50 KNN gönderimimiz 52,88353 public EMD verdi. O karışım farklı modeller kullanıyordu. | **Düşük öncelik.** Yeni bir bileşen farklı topraklarda v3’ün hatasını telafi ederse birleştirmeyi değerlendir. Elastic Net aynı veride basit bir karşılaştırma adayı olarak kayıtlı. |

**Deney öncesi plan.** İri taş ölçümü için ilk karşılaştırma aynı saklı topraklarda saf v3 ile **saf v3 + 20 mm düzeltmesi** olacak. Yüzey alanını kütleye bağlayan ilişkiyi yalnız eğitim topraklarından öğren. 20 mm hata payını ve 11 elekli eğrinin EMD’sini birlikte raporla. H374 ve H615 geliştirmede kullanıldığı için kalan 22 toprağın sonucunu ayrıca ver. Telefon geçişlerini iki yönde ve tek bir toprağın ortalamaya etkisini kontrol et. Her denemenin sonunda ilgili satıra sonuç dosyasını, sayısal farkı ve devam kararını işle.

### Önceki deneylerden özet

**22 Kasım · 16 bölgeli eğitim.** Yerel sınamada sabit DINOv2 genel görünümü ile birlikte eğitilen küçük CNN (48 epoch) eşit bölge ağırlıklarıyla 42,393 EMD, dikkat ağıyla 43,027 EMD verdi. Bölge görüntüsünü 32 × 16’dan 64 × 32 piksele çıkarmak eşit ağırlıklı ağda 42,379 EMD verdi. Ana model saf v3 (38,758 yerel EMD). [Eğitimin kodu](onalti_bolge_egitim.py) · [Ağın gördüğü gerçek bölge görüntüleri](onalti_bolge_egitim_cozunurluk_H037.png)

**21 Kasım 2026 · En iyi Kaggle public sonucumuz 41,67360.** Özgün kadrajlı v3’te fotoğraf özelliklerinin ortancasını kullanmak, önceki en iyi gönderimimiz olan %50–%50 v3/KNN karışımına (52,88353) göre public hatayı 11,20993 EMD düşürdü.

| Gönderim | Dosya | 24 toprakta yerel EMD | Kaggle public EMD |
|---|---|---:|---:|
| 1 · Altı doku ölçüsü + üç komşu | [submission_3nn_doku.csv](submission_3nn_doku.csv) | 44,744 | 67,96723 |
| 2 · Altı doku ölçüsü ve 16 bölgedeki değişkenlik + üç komşu | [submission_3nn_16_bolge.csv](submission_3nn_16_bolge.csv) | 44,184 | 71,50011 |
| 3 · Her ölçekteki doku farkının toplam içindeki payı + üç komşu | [submission_3nn_goreli_doku.csv](submission_3nn_goreli_doku.csv) | 43,567 | 80,56630 |
| 4 · Açık v3 + ince/iri KNN, %50–%50 | [submission_v3_knn_50_50.csv](submission_v3_knn_50_50.csv) | 38,002 | 52,88353 |
| 5 · v3, sabit 100 × 100 mm kadraj | [submission_v3_100mm.csv](submission_v3_100mm.csv) | 40,243 | 44,03121 |
| 6 · v3, özgün kadraj ve ortanca | [submission_v3_ortanca.csv](submission_v3_ortanca.csv) | 38,794 | 41,67360 |
| 7 · Saf v3 ve tam yüzeyde 20 mm düzeltmesi | [submission_v3_tam_yuzey_20mm.csv](submission_v3_tam_yuzey_20mm.csv) | 37,396 | 37,15842 |
| 8 · Saf v3 ve tam yüzeyde 2, 6,3 ve 20 mm düzeltmesi | [submission_v3_tam_yuzey_2_6_3_20mm.csv](submission_v3_tam_yuzey_2_6_3_20mm.csv) | 35,711 | 37,68440 |
| 9 · Üç boyut aralığı | [submission_v3_orta_elek_aralik.csv](submission_v3_orta_elek_aralik.csv) | 35,486 | **35,63420** |
| 10 · Yarı yarıya düzeltme | [submission_v3_orta_elek_yari.csv](submission_v3_orta_elek_yari.csv) | 36,167 | 35,73986 |

**Bugün hangi model kullanılıyor?** Saf v3, fotoğrafın merkezinde genişliğin ve yüksekliğin %60’ını kullanır (toplam görüntü alanının yaklaşık %36’sı). Fiziksel ölçeği eşitler ve farklı boyutlardaki doku ile kenarları 30 sayıyla özetler. İki özellik grubunda birer tek bileşenli PLS regresyonu, toprak başına tane çapı yüzdeliklerini tahmin eder. İki tahmin log-çap uzayında ortalanıp yarışmanın 11 çapındaki birikimli kütle yüzdelerine dönüştürülür. Saf v3’te çekim özellikleri ortalama ile, 6. gönderimdeki v3 sürümünde ise toprak içindeki çekimler ortanca ile birleştirilir. [Model kodu](../work/vitalclick/src/run.py) · [Kadraj ve ölçümler](../work/vitalclick/src/features.py) · [Bir fotoğraf, modelin kullandığı alan ve gerçek tahmin](v3_fotograf_ve_tahmin_H037.png)

**Eğitim ve epoch.** 127 özgün fotoğraf 24 farklı laboratuvar eğrisine bağlı. Standart saf v3’te bir özgün ve sekiz yapay sürüm, 1.143 görüntü-özellik satırı oluşturuyor. Yerel sınavda bunlar eğitim tarafında toprak başına birleştiriliyor. 23 topraktan öğreniliyor ve saklanan 24. toprağın yalnız özgün fotoğrafları kullanılıyor. KNN örnek araması yapar. Ridge ve PLS, bu çalışmada epoch çizelgesi kullanmadan regresyon katsayıları öğrenir. DINOv2 ile ImageGrains hazır ağırlıklarıyla çalıştırıldı. Yarışma fotoğraflarında iki sinir ağına da sıfır epoch ek eğitim verildi. 23 Kasım deneyi bu ölçümü saf v3’ün 20 mm tahminine ekleyerek yeni yerel doğrulama puanı üretti. [GPU sonuçları](#gpu20)

21 Kasım’da paylaşılan Kaggle ekranında altı dosya da **Complete** durumunda. Saf v3 için yazarın bildirdiği public EMD 40,90, bizim yeniden ürettiğimiz yerel EMD 38,758. Saf v3’ün kendisi ekrandaki altı gönderim arasında yer almıyor. Yerelde dosyaların 10 test kimliğini, 11 yüzde sütununu, yüzdelerin sınırlarını ve artan çaplarda azalmadığını kontrol ettik. İlk hazırlanan [submission.csv](submission.csv) ise her numuneye eğitim eğrilerinin ortalamasını verir. [Yazarın gönderim kaydı](https://github.com/vitalclick/Predicting-Soil-Grain-Size#submission-log)

## Lider tablosundaki 0,00000 nasıl okunmalı?

21 Kasım’da yarışma sayfasını incelerken lider tablosunda **0,00000** görüldüğü bildirildi. Güncel takım satırını bu raporu hazırlarken bağımsız olarak doğrulayamadık. Yarışma açıklamasına göre EMD bir hata ölçüsüdür. Gerçek değer tam sıfırsa puanlanan numunelerin tahmin eğrileri, değerlendirilen noktalarda laboratuvar eğrileriyle örtüşür. Tabloda beş ondalık gösterildiğinden çok küçük pozitif bir hata da sıfır gibi görünebilir. Puan tek başına takımın yöntemini açıklamaz. [Yarışmanın puanlama açıklaması](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview/frequently-asked-questions) · [Lider tablosu](https://www.kaggle.com/competitions/soil-grain-size-from-photos/leaderboard)

Bizim **44,18** ve **44,74** değerlerimiz 24 eğitim numunesini sırayla saklayarak hesaplandı. Kaggle’ın puanladığı test alt kümesinde ilk üç komşu modelinin hatası **67,96723** oldu. Dördüncü gönderimle public karşılaştırma değerimiz **52,88353** oldu.

## Veri ve sınav düzeni

- **Girdi ve hedef.** 24 fiziksel eğitim numunesine ait 127 yüzey fotoğrafı, 10 test numunesine ait 35 fotoğraf var. Her numune için 0,002–200 mm arasında 11 çapta, daha ince tanelerin toplam kütle yüzdesi tahmin ediliyor. Eğitim etiketleri eleme ve sedimentasyon ölçümlerinden geliyor. [Kaggle veri açıklaması](https://www.kaggle.com/competitions/soil-grain-size-from-photos/data)
- **Fotoğrafı sayıya çevirme.** Kamera tablosundaki piksel/mm değerini indirilen görüntünün gerçek çözünürlüğüne uyarladık. Merkezden yaklaşık 100 × 100 mm alanı alıp 400 × 400 piksele getirdik. Gri görüntüde 0,25, 0,5, 1, 2, 4 ve 8 mm aralıklı noktalar arasındaki açık-koyu farklarını ölçtük. Aynı toprağın çekimlerini her özellik için ortancayla birleştirdik.
- **Yerel sınav.** Her turda bir toprağın bütün fotoğraflarını ayırdık. Ölçeklemeyi kalan 23 topraktan öğrendik. Üç komşu modeli doku bakımından en yakın üç toprağın laboratuvar eğrilerini ortaladı. Puan, tahmin ile ölçüm arasındaki farkı tane çaplarının logaritmik aralıklarına göre ağırlıklandıran EMD’nin numune ortalaması. Düşük puan daha iyi.
- **Gönderim biçimi.** Her test kimliği için bir satır ve 11 yüzde gerekiyor. Yüzdeler 0–100 aralığında olmalı, çap büyüdükçe azalmamalı ve 200 mm’de 100 olmalı. Hazır ortalama dosyasında 10 satırı, kimlik sırasını, başlıkları ve eğri koşullarını kontrol ettik. [İlk notebook](toprak_ilk_gonderim.ipynb)

## Deney defteri

Tüm satırlar aynı 24 numunelik yerel sınava ait. H038 sütunu yalnızca o numunenin **tüm eğri** hatasını gösterir.

| Deney | Ortalama EMD | H038 EMD | Bulgumuz ve kararımız |
|---|---:|---:|---|
| Eğitim eğrilerinin ortalaması | 89,89 | 57,24 | İlk çalışan dosyayı üretti. Fotoğraftan kazanımı ölçmek için başlangıç noktası. |
| Altı doku ölçüsü + üç komşu | 44,74 | 93,28 | Başlangıcı 24 toprağın 18’inde geçti. Sade ve yeniden üretilebilir ana referans. |
| Üç komşu + 4 × 4 bölgedeki doku değişkenliği | **44,18** | 138,95 | Ana modele göre 9 toprakta iyileşti, 11’inde kötüleşti, 4’ünde aynı kaldı. H374’te hata 128,87’den 21,44’e indi. Ortalamadaki 0,56 puanlık avantaj ve H038’deki kayıp birlikte kaydedildi. |
| Aynı altı ölçü + Ridge regresyonu | 48,01 | 102,04 | Ana modeli 12 numunede geçti. F827’de hata 63,98’den 53,22’ye inerken H366’da 8,11’den 75,81’e çıktı. Düzenlileştirme ayarı her turda yalnızca eğitim tarafında seçildi. |
| Üç komşu ve Ridge eğrilerinin %50–%50 ortalaması | 45,28 | 97,66 | Sabit eşit ağırlığı denedik. Ana modele göre ortalama hatayı azaltmadı. |
| Üç komşu + parlak iri tane adaylarının alanı | 45,865 | 93,28 | 4 toprakta iyileşti, 3’ünde kötüleşti, 17’sinde aynı kaldı. H038’in 2 mm tahmini de aynı kaldı. Bu ölçüyü ana modele eklemedik. |
| Hazır DINOv2-S/14 + üç komşu | 53,428 | 148,81 | Altı doku ölçüsüne göre 8 toprakta iyileşti, 13’ünde kötüleşti, 3’ünde eşit kaldı. Bu sürüm için Kaggle gönderimi yapılmadı. |
| Aynı altı ölçü + yüzdeliklerin çapını tahmin eden Ridge | 47,01 | 102,32 | Önceki Ridge’e göre 10 numunede iyileşti, 14’ünde kötüleşti. Ortalama kazanç 0,99 EMD. [Çalıştırılmış deney](toprak_cap_ridge.ipynb) |

Çalıştırılmış deneyler ve numune sonuçları: [üç komşu](toprak_doku_dersi.ipynb) · [16 bölge](toprak_bolge_deneyi.ipynb) · [Ridge](toprak_ridge_deneyi.ipynb) · [iri tane alanı](toprak_iri_tane_alani.ipynb)

## Hataları anlamak için neye baktık?

**F827 ve H368 — fotoğraftaki farkın laboratuvar karşılığı.** Kütlece 2 mm’den ince pay H368’de %22,76, F827’de %100. F827’de daha zor bir eşik olan 0,063 mm’de laboratuvar sonucu %89,56, ortalama eğri %27,65 ve üç komşu tahmini %45,65 çıktı. Seçilen komşular G190, H617 ve H493 oldu. F827 saklandığında kalan en yüksek üç ölçümün ortalaması bile yaklaşık %73. Bu örnekte komşu ortalamasının ulaşabileceği değerin matematiksel bir sınırı var. [İki numunenin fotoğrafı ve eğrisi](iki_toprak_numunesi.png) · [F827 ve üç komşusu](f827_uc_komsu.png)

**H038 — iri tanelerin kütle payı düşük tahmin edildi.** Laboratuvara göre kütlenin %47,23’ü 2 mm’den ince. Üç komşu %85,15, 16 bölgeli model %87,06, Ridge %81,97 tahmin etti. Ana modelin seçtiği komşular H037, H617 ve H030 oldu. Aynı 23 referansla H038’in sekiz çekimini tek tek sınadık. Üç Motorola çekimi %97,2–98,4, beş Samsung çekimi %75,7–85,3 verdi. Sekizi de ölçümün üzerindeydi. [Sekiz çekimin sonuçları](h038_sekiz_cekim.png) · [16 bölgeli karşılaştırma](h038_bolgeler.png)

**İri taneleri doğrudan görme denemesi.** Arka plana göre açık renkli bağlı bölgelerden, alanı yaklaşık 2 mm çaplı bir daireye ulaşanları seçtik. H038’in Motorola 03 çekiminde kesitin %19,4’ü işaretlendi. Bu, o fotoğraftaki seçili piksel alanıdır. Koyu taneler gözden kaçabilir, bir taşın yalnızca aydınlık kısmı seçilebilir. Bu ölçüyü gerçek iri tane kütlesi olarak kullanmak hatalı olur. [İşaretlenen bölgeler](h038_iri_tane_alani.png)

## Telefon değişince ne oldu?

Eğitim görüntüleri Motorola ve Samsung, test görüntüleri iPhone 14 ve 16 ile çekilmiş. İki ayrı kontrol yaptık.

1. **Aynı model, aynı toprak, farklı telefon.** Her iki telefonla çekilmiş 21 toprakta, diğer 23 topraktan kurulan aynı modele ayrı ayrı Motorola Edge ve Samsung A52 görüntülerini verdik. 2 mm tahminleri arasındaki mutlak farkın ortancası **8,79 yüzde puan**, en büyüğü **44,22 yüzde puan** çıktı.
2. **Tek telefonda eğitim, iki telefonda sınav.** İki telefonla da çekilmiş aynı 21 toprağı seçtik. Her turda 20 toprakla tek telefonda eğitim yaptık ve saklanan toprağı iki telefonda sınadık. Aşağıdaki sonuçlar bu 21 numunelik ayrı deneyden geliyor. [Çalıştırılmış kamera deneyi](toprak_iki_kamera_sinavi.ipynb)

| Eğitim telefonu | Aynı telefondan görüntü, EMD | Diğer telefondan görüntü, EMD |
|---|---:|---:|
| Motorola Edge | 45,98 | 49,10 |
| Samsung A52 | 45,01 | 45,26 |

H668’de Motorola ile eğitilen modelin hatası Motorola çekiminde 10,03, Samsung çekiminde 101,44 oldu. Bu, bazı numunelerin çekim değişimine duyarlı olduğunu gösteriyor. Aynı çekimlerde telefonla birlikte ışık ve kadraj da değişebiliyor. iPhone çekimlerinin laboratuvar etiketleri elimizde olmadığı için iPhone hatası açık bir soru.

### 16 bölge eklenince telefon değişimine duyarlılık

Aynı 21 toprağın her birini sırayla sakladık. İki model de her turda aynı 20 topraktan ve aynı eğitim telefonundan komşu seçti. Tabloda iki modelin diğer telefonun çekimlerindeki ortalama EMD’si var. [Çalıştırılmış karşılaştırma ve her toprağın sonuçları](toprak_bolge_kamera_karsilastirmasi.ipynb)

| Eğitim → sınav telefonu | Sade | 16 bölge |
|---|---:|---:|
| Motorola Edge → Samsung A52 | 49,10 | 44,48 |
| Samsung A52 → Motorola Edge | 45,26 | 56,62 |
| İki yönün ortalaması | 47,18 | 50,55 |

Aynı telefonda sınandığında iki yönün ortalama hatası sade modelde 45,50, bölgeli modelde 51,57 oldu. Aynı toprağın iki çekim grubuna verilen tahminleri doğrudan birbirleriyle karşılaştırdığımızda ortalama eğri farkı sade modelde 23,73, bölgeli modelde 25,99 EMD çıktı. Bölgeli model burada çekim değişiminden daha çok etkilendi. Bununla birlikte diğer telefona geçişin laboratuvar hatasına etkisi eğitim yönüne bağlı. 24 topraklık önceki sınavda bölgeli modelin en büyük kazanımı gösterdiği H374, ortak 21 toprak arasında yok. Bu sınavda iPhone çekimleri de yok. Kaggle’daki puan farkının nedeni açık kalıyor.

## Üç komşuyla ulaşılabilecek hata sınırı

Her eğitim numunesini dışarıda bırakıp kalan 23 numuneden kurulabilen 1.771 farklı üçlüyü karşılaştırdık. Her üçlüde aynı üç eğrinin eşit ağırlıklı ortalamasını, 11 çapın tamamında kullandık. Fotoğrafa göre seçilen üçlünün ortalama EMD’si 44,74. Saklanan numunenin laboratuvar sonucuna bakarak en iyi üçlüyü seçince ortalama 10,36 çıktı. Aradaki 34,38 puan, doğru cevap bilindiğinde yapılabilecek seçimin avantajı. Testte beklenen kazanım olarak kullanılamaz.

H038’de fotoğrafın seçimi H037, H617 ve H030 ile 93,28 hata verdi. H037, H374 ve H637 üçlüsü aynı eğriyi 5,26 hatayla temsil etti. H616’da ise en iyi üçlü bile 43,47’de kaldı. Komşu bulma ve üçlü ortalamanın temsil gücü farklı numunelerde farklı ölçülerde sınırlayıcı. Bir sonraki yöntem tartışmasında fotoğraf benzerliğinin laboratuvar eğrilerinin benzerliğiyle ne kadar örtüştüğünü incelemeye öncelik veriyoruz. [24 numunenin üçlüleri ve hata değerleri](toprak_en_iyi_uclu.ipynb)

## Fotoğraf ve laboratuvar hangi komşuları seçiyor?

Her numune için diğer 23 toprağı hem altı doku ölçüsündeki uzaklığa hem de laboratuvar eğrileri arasındaki EMD’ye göre sıraladık. Toplam 72 komşu seçiminin 22’si (%30,6) laboratuvardaki ilk üçe girdi. 24 numunenin 8’inde ortak komşu yoktu, 10’unda bir, 6’sında iki ortak komşu vardı. Tüm 23 adayın sıra uyumunu ölçen Spearman katsayısının numune ortalaması 0,57 iken H038’de −0,56 çıktı.

H038 için fotoğrafın seçtiği H037, H617, H030 laboratuvara göre sırasıyla 5., 14. ve 15. sıradaydı. Laboratuvarda tek tek en yakın üç eğri H637, H371 ve H366’ya aitti. Özellikle H371 laboratuvarda 2., fotoğrafta 23. sıradaydı. Önceki deneyde üç eğrinin **ortalamasının** başarısını ölçmüştük; burada her komşunun **kendi eğrisini** H038 ile karşılaştırdık. Bu teşhis için test numuneleri kullanılmadı.

### H038, H371 ve H617’nin altı ölçekte karşılaştırması

Üç toprağın 21 çekimini kamera ve numune bazında ortancayla özetledik. İki telefonda da H038 dışındaki aynı 23 eğitim numunesinden öğrenilmiş standart sapmaları kullandık. Altı ölçünün her birinde H617, H038’e H371’den daha yakın çıktı. Kareli uzaklık farkına en büyük katkı Motorola Edge’de 1 mm, Samsung A52’de 2 mm aralığından geldi. 4 mm iki telefonda da en az etkili aralıktı. Herhangi bir tek ölçeği kaldırmak bu iki adayın sırasını çevirmiyor. Laboratuvar eğrisinin H038’e uzaklığı ise H371 için 37,12, H617 için 116,23 EMD. Bu karşılaştırma, üç numune için açık-koyu doku benzerliğinin kütle dağılımı benzerliğinden ayrıldığını gösteriyor. [Aynı fiziksel ölçekte altı kesit ve katkı grafiği](h038_alti_olcek.png)

## DINOv2 ile komşu seçimi ve açık yarışma sonuçları

127 çekimin her birinden aynı 100 × 100 mm fiziksel kesiti aldık. DINOv2’ye 224 × 224 renkli piksel verdik ve hazır modelin ürettiği 384 özelliği çekimler arasında ortancayla birleştirdik. Numune ayırma, eğitim tarafında standartlaştırma, üç komşu ve eğri ortalaması önceki deneyle aynı. Yerel hata 44,744’ten 53,428’e yükseldi. H374’te 128,87’den 19,78’e indi, H038’de 93,28’den 148,81’e çıktı, H405’te 133,53 kaldı. H038 için laboratuvarda 2. olan H371 görüntü sıralamasında 23.’lükten 15.’liğe çıktı. Laboratuvarda 14. olan H617 görüntüde yine 2. sıradaydı. Laboratuvarın ilk üçüne isabet toplamda 22/72’den 23/72’ye yükseldiği halde tahmin eğrilerinin ortalama hatası arttı. [Çalıştırılmış DINOv2 deneyi ve 24 numune sonucu](toprak_dinov2_uc_komsu.ipynb)

Nomannic'in açık DINOv2 + Sqrt-Mass Ridge notebook'unda V1 için Kaggle puanı 61,83175 görünüyor. Bizim gönderdiğimiz üç komşu modelinin public puanı 67,96723. İki gönderimde görüntü işleme ve tahmin yöntemi birlikte değiştiği için farkı tek başına DINOv2’ye bağlayamayız. [Nomannic’in Kaggle notebook’u](https://www.kaggle.com/code/nomannic19/cv-psgsdi-dinov2-features-sqrt-mass-ridge)

vitalclick’in açık deney kaydında v3, 24 numunelik yerel sınavda 38,8 ve Kaggle public'te 40,90 olarak bildiriliyor. Yazar, milimetre ölçekli görüntü özelliklerinden sabit kütle yüzdelerine karşılık gelen tane çaplarını tahmin ediyor. Daha sonraki v31 için public 22,31 bildiriliyor. Bu sürümün ayarları public puana bakılarak seçilmiş ve aynı sürüm için yerel puan verilmemiş. Bu fikirden hareketle altı doku ölçüsünü koruyarak çap tahminli Ridge deneyini yaptık. Sonuçlar bir sonraki bölümde. [Yöntem ve sürüm bazında sonuçlar](https://github.com/vitalclick/Predicting-Soil-Grain-Size#submission-log)

## Yüzde yerine çap tahmin edince ne oldu?

127 fotoğrafın aynı altı doku ölçüsünü kullandık. Her toprak için kütleyi 100 eşit payla temsil ettik. %0,5 ile %99,5 arasındaki yüzdeliklere karşılık gelen çapları laboratuvar eğrisinden logaritmik çap ekseninde ara değer alarak çıkardık. Ridge bu çapların logaritmalarını tahmin etti. Her elekte kaç payın geçtiğini sayarak yarışmanın istediği 11 yüzdeye döndük. Ölçülmüş eğrileri yalnızca bu dönüşümden geçirmek ortalama 0,975 EMD hata verdi. Bu sayı yüzdeye dönüşteki bir puanlık adımların etkisini gösterir.

Her dış turda bir numunenin tüm çekimlerini ayırdık. Her iki Ridge modelinde aynı dört alfa arasından, kalan 23 numunede iç doğrulama ile seçim yaptık. Eski Ridge 48,006, çap tahminli Ridge 47,014 EMD aldı. 10 numunede hata azaldı, 14 numunede arttı. H030’da 36,38 → 13,97 ve H617’de 43,41 → 26,73 iyileşmeleri ortalamayı aşağı çekti. H038’de 2 mm’den ince kütlenin ölçümü %47,23, eski Ridge tahmini %81,97 ve yeni tahmin %94 oldu. Tüm eğride H038 hatası 102,04 → 102,32. Yerel gönderim adayı 44,18 ile 16 bölgeli sürüm. Çap Ridge için public puan ölçülmedi. [24 numunenin sonuçları ve dönüşüm kontrolü](toprak_cap_ridge.ipynb)

## Altı doku ölçüsünün oranları

Aynı fotoğrafın altı doku ölçüsünü kendi toplamlarına böldük ve ardından aynı toprağın çekimlerini ortancayla birleştirdik. Kamera ölçeği, merkez kesit, eğitim tarafında standartlaştırma ve üç komşu seçimi ilk gönderimle aynı. 24 numunelik yerel sınavda sade modelin 44,744 EMD’si 43,567’ye indi. H374, H405 ve H516’da büyük kazanımlar oldu. H038’de hata 93,28’den 112,90’a çıktı. Toplamda 8 toprakta hata azaldı, 11’inde arttı, 5’inde eşit kaldı.

21 ortak toprağın çapraz telefon sınavında Motorola’dan Samsung’a hata 49,10’dan 47,03’e, Samsung’dan Motorola’ya 45,26’dan 40,30’a indi. İki yönün ortalama hatası 47,180’den 43,662’ye düştü. Aynı toprağa iki telefondan verilen tahminlerin birbirine uzaklığı ise 23,730’dan 23,541’e indi. Kaggle public sonucu 80,56630 oldu. [Çalıştırılmış deney ve gönderim kodu](toprak_goreli_doku.ipynb)

İlk ve üçüncü gönderimlerin tahmin eğrilerini aynı 10 test kimliğinde karşılaştırdık. Eğriler arasındaki en büyük üç EMD farkı Kleinkummerfeld 9-4 (75,42), Airbus BS12-5bis8 (74,69) ve Kleinkummerfeld 18-3 (66,62) numunelerinde. Üçünde de ilk modelin 2 mm altı tahmini %75,72 iken üçüncü model sırasıyla %97,52, %96,31 ve %96,23 verdi. Bu farklar modellerin ne kadar ayrıştığını ölçüyor. Kaggle numune bazında hata göstermediği için her birinin public puana katkısı bilinmiyor.

### En çok ayrışan üç test toprağında komşular

İki gönderimdeki komşu seçimlerini fotoğraflardan yeniden hesapladık. Seçilen üç eğitim numunesinin laboratuvar eğrilerini ortalayınca iki CSV’deki 10 toprağın 11 çap tahmini de sekiz ondalık basamakta kayıtlı değerlerle uyumlu çıktı.

En çok ayrışan üç test toprağında sade model aynı üç komşuyu seçti. H666, H374 ve H030’un laboratuvarda 2 mm’den ince kütle payları sırasıyla %99,79, %29,42 ve %97,95. Bu üçlünün ortalaması %75,72. Göreli model üçünde de H030’u korudu, H666 ve H374’ün yerine başka numuneler seçti.

| Test toprağı | Göreli modelin komşuları | 2 mm altı tahmini, sade → göreli | 0,063 mm altı tahmini, sade → göreli |
|---|---|---:|---:|
| Kleinkummerfeld 9-4 | H030, H549, H126 | %75,72 → %97,52 | %31,12 → %58,48 |
| Airbus BS12-5bis8 | G190, F827, H030 | %75,72 → %96,31 | %31,12 → %59,06 |
| Kleinkummerfeld 18-3 | H030, G190, H549 | %75,72 → %96,23 | %31,12 → %52,47 |

2 mm eşiğindeki artışın ana açıklaması, H374’ün üç seçimden de çıkması. Yeni seçilen numunelerde bu eşikten geçen kütle %90,99–100,00. Üç test toprağının gerçek eğrileri gizli olduğu için hangi modelin bu satırlarda ölçüme daha yakın olduğunu belirleyemiyoruz. Çalışmaya bu karşılaştırmayla ara verildi.

## Açık v3 yöntemini doğrulama ve görüntü ölçülerini üç komşuda sınama

vitalclick’in 0808d05 sürümündeki hazır görüntü ölçüleri, kendi laboratuvar etiketlerimiz ve özgün `--linear-only --cv-only` komutuyla 24 numunelik yerel EMD 38,758 çıktı. Kod her turda saklanan numuneyi tüm çekimleriyle eğitimden çıkarıyor. Yazarın bildirdiği public 40,90 o çalışmanın gönderimine ait. [Kaynak](https://github.com/vitalclick/Predicting-Soil-Grain-Size/tree/0808d05def19e03a6478410827a644bfd411ba81)

Kontrollü karşılaştırmada yalnızca görüntüden çıkarılan özellikleri değiştirdik. Açık yöntemin iki listesinin birleşimi olan 30 ölçüyü kullandık. Bizim önceki KNN gibi sadece 127 özgün çekimi aldık, numune başına ortancayı ve eğitim tarafında standartlaştırmayı koruduk. Tahmin yine en yakın üç toprağın eğri ortalaması. 24 toprakta hata 44,744’ten 39,743’e, seçilen tekil komşuların ortalama laboratuvar uzaklığı 53,043’ten 47,144’e düştü. Laboratuvar sıralamasının ilk üçüne isabet 22/72’den 20/72’ye indi, bütün adaylar üzerinden ortalama Spearman sıra uyumu 0,574’ten 0,660’a çıktı. 9 toprak iyileşti, 10’u kötüleşti, 5’i aynı kaldı. H374 ve H405’te büyük kazanım var, H038’de aynı komşular yeniden seçildi. Sonuç yerel. [Çalıştırılmış deney ve 24 numunenin karşılaştırması](v3_ve_komsu_secimi.ipynb)

### 30 ölçülü KNN için telefon kontrolü

Motorola Edge ve Samsung A52 ile çekilmiş aynı 21 toprakta her turda bir toprağı dışarıda tuttuk. İki model aynı 20 referans toprağı, aynı eğitim telefonunu ve aynı üç komşu kuralını kullandı. Aşağıda sınav telefonu eğitim telefonundan farklı. İki yönün ortalamasında 30 ölçülü KNN 7 toprakta iyileşti, 13’ünde kötüleşti, 1’inde eşit kaldı.

| Eğitim → sınav | Sade KNN | 30 ölçülü KNN |
|---|---:|---:|
| Motorola Edge → Samsung A52 | 49,10 | 58,95 |
| Samsung A52 → Motorola Edge | 45,26 | 40,24 |
| İki yönün ortalaması | 47,18 | 49,59 |

İki yönün ortalamasında aynı telefonla sınandığında hata 30 ölçülü KNN’de 40,159, sade KNN’de 45,495 oldu. Farklı telefonla sınandığında sıralama tersine döndü. Aynı toprak için iki çekim grubunun tahmin eğrileri arasındaki ortalama EMD sırasıyla 33,495 ve 23,730. Deney Motorola ve Samsung ile sınırlı. 30 ölçülü KNN’yi bu aşamada Kaggle’a göndermeyi bekletiyoruz. [Çalıştırılmış telefon karşılaştırması](v3_ve_komsu_secimi.ipynb)

### Komşu mesafesini eğitim eğrilerinden öğrenme

Aynı 30 görüntü ölçüsü, özgün fotoğrafların ortancası ve üç komşunun eşit ağırlıklı ortalaması korundu. Standartlaştırılmış 30 boyutta Öklid uzaklığı yerine her turda yalnızca eğitim topraklarının değişken 10 laboratuvar yüzdesiyle tek PLS ekseni öğrendik. Komşuları o eksendeki uzaklığa göre sıraladık. Sonuçlara bakarak eksen sayısı seçilmedi. [Deneyi çalıştıran kod](ogrenilen_komsu_mesafesi.py)

| Yerel sınav | Öklid EMD | Öğrenilen mesafe EMD |
|---|---:|---:|
| 24 toprak | 39,743 | 39,722 |
| Motorola → Samsung | 58,949 | 48,343 |
| Samsung → Motorola | 40,239 | 38,422 |
| İki telefon yönünün ortalaması | 49,594 | 43,383 |

24 toprakta seçilen komşuların ortalama laboratuvar uzaklığı 47,144’ten 46,153’e indi ve laboratuvarın ilk üçüne isabet 20/72’den 23/72’ye çıktı. Tahmin hatası 10 toprakta azaldı, 13’ünde arttı ve 1’inde eşit kaldı. İki telefon yönü numune bazında ortalandığında 21 toprağın 13’ünde hata düştü. 24 topraklık sınavda öğrenilen eksen H038 ile H405’i karşılıklı komşu seçti. Laboratuvarda 0,063 mm altı payları %18,28 ve %75,66, 2 mm altı payları %47,23 ve %98,45. Her ikisinin de hatası arttı.

### Komşuları ince ve iri tane paylarına göre seçmek

Her dış turda aynı 30 görüntü ölçüsünden iki ayrı tek bileşenli PLS modeli öğrendik. Hedefler 0,063 mm altı ve 2 mm üstü laboratuvar kütle yüzdeleri. Saklanan toprağın etiketleri iki eğitime de girmedi. Hem referanslar hem de sorgu için görüntüden tahmin edilen payları 0–100 aralığına kırptık, iki yüzde puanı eksenine eşit ağırlık verdik. Son tahmin yine üç komşunun 11 çapta ölçülmüş eğrilerinin ortalaması.

| EMD | Tek eksen | İnce ve iri payı |
|---|---:|---:|
| 24 toprak | 39,722 | 40,144 |
| Motorola → Samsung | 48,343 | 48,464 |
| Samsung → Motorola | 38,422 | 37,162 |
| İki telefon yönünün ortalaması | 43,383 | 42,813 |

24 toprakta 21 komşu üçlüsü aynı kaldı. Bir toprakta hata azaldı ve ikisinde arttı. Laboratuvar ilk üçüne isabet 23/72’den 24/72’ye çıktı. İki telefon yönünün numune ortalamasında 2 iyileşme, 17 eşitlik, 2 kötüleşme var. H038 ve H405 yine birbirini seçti. Sırayla dışarıda tutulduklarında H038 için ince/iri pay tahminleri %45,55/%15,34, ölçümleri %18,28/%52,77 oldu. H405 için aynı tahminler %41,20/%18,31, ölçümler %75,66/%1,55. Bunlar komşu aramasında kullanılan ara tahminler. Bu iki hedef ve bu modelle amaçlanan ayrım sağlanmadı. [Çalıştırılan deney](ogrenilen_komsu_mesafesi.py)

### İnce ve iri pay tahminleri fotoğrafsız ortalamayı geçiyor mu?

Komşuları seçmeden önce üretilen iki pay tahminini ayrı ayrı değerlendirdik. Referans her turda yalnızca eğitimdeki 23 toprağın ilgili pay ortalaması. Ölçü, kütle yüzdesindeki mutlak hatanın 24 numune üzerinden ortalaması (yüzde puan).

| Hedef | Eğitim ortalaması | Fotoğraftan tahmin | Fotoğrafın daha yakın olduğu toprak |
|---|---:|---:|---:|
| 0,063 mm’den ince pay | 25,71 | 14,87 | 19/24 |
| 2 mm’den iri pay | 31,21 | 9,21 | 22/24 |

21 ortak topraktaki telefon deneyinde referansı her turda kalan 20 topraktan hesapladık. Motorola → Samsung yönünde fotoğrafın ortalama mutlak hatası ince/iri için 16,32/14,47 puan, Samsung → Motorola yönünde 17,53/7,74 puan oldu. Eğitim ortalamasının hataları iki yönde de 26,85/31,43. İnce payda her yönde 16/21, iri payda 19/21 toprakta fotoğraf daha yakındı. H038 ve H405 örnekleri tek başına bu genel başarıyı yansıtmıyor. İki payın tahmini ile üç komşudan elde edilen tüm eğrinin başarısı ayrı değerlendirilmeli.

### Doğrudan pay tahmini üç komşunun ortalamasına geçince ne değişiyor?

Aynı saklı numunede, ince ve iri eksenler için fotoğraftan tahmin edilen payları bu iki eksende seçilen üç komşunun laboratuvar eğrilerinin ortalamasıyla karşılaştırdık. 24 toprağın her birinde her iki tahmin aynı gerçek paya göre ölçüldü. İnce payda mutlak hata ortalaması doğrudan 14,869, komşularda 14,917 yüzde puan. Komşular 16 toprakta daha yakın, doğrudan tahmin 8 toprakta daha yakın. İri payda değerler sırasıyla 9,209 ve 10,422, iki yöntem de 12’şer toprakta daha yakın. H038’de gerçek iri pay %52,77, doğrudan %15,34 ve komşular %2,82. H405’te gerçek ince pay %75,66, doğrudan %41,20 ve komşular %18,59.

21 ortak toprağın çapraz telefon sınavında iri pay için komşu ortalaması iki yönde de daha düşük hata verdi. Motorola → Samsung yönünde doğrudan 14,474, komşu 13,304 yüzde puan. Ters yönde doğrudan 7,736, komşu 6,840 yüzde puan. Komşu ortalamasına geçişin etkisi hedefe, toprağa ve sınava göre değişiyor.

24 toprakta son komşu eğrisinin 40,144 EMD’sini her çapta mutlak yüzde hatasını bir sonraki çapa kadar logaritmik aralıkla çarparak ayırdık. En büyük katkılar 0,063 mm’de 7,484 EMD (%18,64), 0,2 mm’de 6,221 (%15,50) ve 0,63 mm’de 5,748 (%14,32). Birlikte toplam hatanın %48,46’sı. 2 mm’nin katkısı 5,193 (%12,94). Numune bazındaki katkılar toplamı ilgili numunenin EMD’si ile doğrulandı. [24 satırlık karşılaştırma ve elek bazında hesap](ogrenilen_komsu_mesafesi.py)

### 0,2 ve 0,63 mm elekleri, bütün eğri ve v3 kamera kontrolü

30 özgün görüntü ölçüsü ve 3 komşulu ince/iri modelini önceki deneyden aldık. Her dış turda tek toprak ve ona ait tüm çekimler dışarıda. 24 topraklık sınavda 23 referans, 21 ortak topraklık telefon sınavında yalnızca kaynak telefonla çekilmiş 20 referans var. Aynı eğitim tarafında her elek için tek bileşenli ayrı bir PLS regresyonuyla elekten geçen kütle yüzdesini doğrudan tahmin ettik. [Tüm ölçümleri, üç sınavın her eleğini ve numune karşılaştırmasını üreten kod](elek_dogrudan_karsilastirma.py)

| Ortalama mutlak hata, yüzde puan | Komşu 0,2 mm | Doğrudan 0,2 mm | Komşu 0,63 mm | Doğrudan 0,63 mm |
|---|---:|---:|---:|---:|
| 24 toprak | 12,484 | 13,940 | 11,457 | 12,470 |
| Motorola → Samsung | 18,290 | 19,014 | 14,919 | 18,767 |
| Samsung → Motorola | 14,600 | 14,904 | 9,172 | 11,493 |

Tabloda doğrudan tahminler 0–100’e kırpıldı. Elekler arası düzeltmeden önce 24 toprakta doğrudan tahmin 0,2 mm’de 7, 0,63 mm’de 11 numunede daha yakındı. Fotoğrafsız, eğitim ortalamasına dayalı karşılaştırmanın hataları sırasıyla 35,491 ve 38,005 yüzde puan.

Tüm eğride 10 değişken eşiğin tahminlerine izotonik regresyon uyguladık ve 200 mm noktasını 100 olarak sabitledik. Bu düzeltmeye ana sınavda 4/24, Motorola → Samsung yönünde 3/21, ters yönde 12/21 eğri ihtiyaç duydu. Sonraki deneyde düzeltilmiş doğrudan eğri ile üç komşu eğrisinin ortasını aldık. Ağırlık baştan ½ olarak belirlendi. Bu iki bileşen, önceki altı ölçülü Ridge karışımından farklı.

| Tüm eğride EMD | 24 toprak | Motorola → Samsung | Samsung → Motorola | Çapraz ortalama |
|---|---:|---:|---:|---:|
| İnce/iri tahminine göre üç komşu | 40,144 | 48,464 | 37,162 | 42,813 |
| Doğrudan, monotonluk düzeltmesinden önce (tanısal) | 40,047 | 50,800 | 40,304 | 45,552 |
| Doğrudan, geçerli eğri | 40,047 | 50,800 | 40,202 | 45,501 |
| Komşu ve geçerli doğrudan eğrinin eşit ortalaması | 39,122 | 48,419 | 37,782 | 43,100 |
| Yayımlanmış v3, özgün ve yapay eğitim çekimleri | 38,758 | 41,987 | 42,831 | 42,409 |
| Aynı v3, yalnızca özgün eğitim çekimleri | 38,440 | 47,716 | 38,656 | 43,186 |

Doğrudan eğri, komşu modeline göre 24 toprağın 10’unda hatayı azalttı, 14’ünde artırdı. H405’te 120,14 → 87,71 ve H038’de 126,29 → 109,32 iyileşmelerine karşılık H374’te 33,55 → 56,22 oldu. Eşit karışım 24 toprakta yalnız komşuyu 13, yalnız doğrudan modeli 16 toprakta geçti. Telefon yönlerini numune başına ortalayınca yalnız komşuyu 10/21, yalnız doğrudan modeli 17/21 toprakta geçti. Küçük yerel ortalama farklarıyla yöntem seçmek için yeterli bağımsız veri yok.

v3 için kaynak kodun aynı iki doğrusal bileşenini ve ayarlarını koruduk. Eğitim toprakları dışında kalan türetilmiş görüntüler kullanılmadı. Saklanan toprakta yalnız özgün fotoğraflar kullanıldı. Eğitimde yapay çekimleri ekleyince Motorola → Samsung yönünde 16/21 toprak iyileşti ve ortalama hata 5,729 puan düştü. Samsung → Motorola yönünde 4/21 toprak iyileşti, ortalama hata 4,175 puan arttı. Ana sınavda 9/24 toprak iyileşti. Yapay çekimler yön farkını küçülttü. Yarar eğitim ve sınav telefonuna bağlı. Bu çalışmada test numuneleri kullanılmadı.

### Aynı toprakta telefonla değişen ölçüler

21 ortak toprağın özgün çekimlerini telefon ve numune bazında ortancayla birleştirdik. Her ölçü için Samsung ve Motorola arasındaki mutlak farkı, iki telefonun kendi içindeki topraklar arası varyanslarının ortalamasının kareköküne böldük. Aşağıdaki sıralama, çıkan farkların 21 topraktaki ortancasına göre. Fotoğraftaki ışık, kadraj ve yüzey bölgesi de değişebildiğinden farkı yalnız telefon donanımına bağlayamıyoruz.

| En çok değişen ölçü | Standartlaştırılmış mutlak fark | Samsung çekiminde daha yüksek |
|---|---:|---:|
| 22 mm ölçekli doku yanıtı | 3,546 | 21/21 |
| 16 mm ölçekli doku yanıtı | 1,835 | 19/21 |
| 9 mm yarıçaplı filtreyle giderilen parlaklık | 1,830 | 19/21 |
| Doku yanıtlarının ölçekler arasındaki yayılımı | 0,899 | 21/21 |
| Doku yanıtının ağırlıklı ortalama ölçeği | 0,849 | 20/21 |

Doku yanıtları filtre ölçeğini ifade eder. Örneğin “22 mm” belirli tane çapının kütle payı değildir. Her dış turda saklanan toprak çıkarıldı. Diğer eşleşmiş topraklarda en çok değişen beş özelliği sildik. İlk üç ölçü 21 turun hepsinde, ortalama ölçek 19, yayılım 17, 0,5 mm doku yanıtı 6 turda silindi. İkinci deneyde 30 ölçüyü koruduk ve her ölçü için diğer 20 toprakta gözlenen telefonlar arası işaretli farkın ortancasını sınav çekiminden çıkardık. Ölçekleme ve komşu referansları eğitim telefonundan geldi. Ölçü seçimi ve kayma düzeltmesi için diğer 20 toprağın her iki telefondaki çekimleri kullanıldı. Saklanan toprağın laboratuvar eğrisi yalnız puanlamada kullanıldı.

| Model ve EMD sınavı | 30 ölçü | En çok kayan 5 ölçü çıkarıldı | 30 ölçü, ortanca fark düzeltildi |
|---|---:|---:|---:|
| Öklid, 24 toprak | 39,743 | 40,390 | — |
| Öklid, Motorola → Samsung | 58,949 | 45,182 | 47,332 |
| Öklid, Samsung → Motorola | 40,239 | 43,295 | 41,042 |
| Öklid, iki yön ortalaması | 49,594 | 44,238 | 44,187 |
| İnce/iri payına göre, 24 toprak | 40,144 | 40,266 | — |
| İnce/iri, Motorola → Samsung | 48,464 | 48,945 | 44,988 |
| İnce/iri, Samsung → Motorola | 37,162 | 39,478 | 39,002 |
| İnce/iri, iki yön ortalaması | 42,813 | 44,211 | 41,995 |

İki yönü toprak başına ortalayınca beş ölçüyü silmek Öklid modelinde 15 toprağı iyileştirdi, 6’sını kötüleştirdi. İnce/iri modelinde 4 iyileşme, 12 eşitlik ve 5 kötüleşme var. Ortanca fark düzeltmesi Öklid modelinde 12 iyileşme ve 9 kötüleşme, ince/iri modelinde 8 iyileşme, 2 eşitlik ve 11 kötüleşme verdi. Bu ölçüleri genel olarak dışlamak için sonuçlar tutarlı değil. Düzeltmede eşleşmiş hedef telefon fotoğrafları gerekiyor. Eğitim topraklarının iPhone eşleri elimizde olmadığı için bu değerler iPhone’a doğrudan uygulanamaz. [30 ölçünün tam sıralaması ve yeniden çalıştırılabilir deney](telefon_olcu_kaymasi.py)

<details>
<summary>30 ölçünün tamamı ve değişimin yönü</summary>

| Ölçü | Ortanca mutlak fark | Ortanca yönlü fark (Samsung eksi Motorola) | Samsung yüksek (21 içinden) |
|---|---:|---:|---:|
| log_22.0 | 3,546 | 3,546 | 21 |
| log_16.0 | 1,835 | 1,835 | 19 |
| open_9.0 | 1,830 | 1,830 | 19 |
| scale_spread | 0,899 | 0,899 | 21 |
| scale_centroid | 0,849 | 0,849 | 20 |
| log_0.5 | 0,823 | 0,305 | 14 |
| log_11.0 | 0,728 | 0,728 | 17 |
| slope_coarse | 0,525 | 0,525 | 21 |
| open_0.5 | 0,487 | -0,487 | 1 |
| open_1.5 | 0,477 | -0,477 | 2 |
| open_1.0 | 0,469 | -0,469 | 2 |
| open_0.75 | 0,462 | -0,462 | 2 |
| open_2.0 | 0,449 | -0,449 | 2 |
| log_8.0 | 0,432 | 0,432 | 17 |
| log_0.7 | 0,416 | 0,416 | 18 |
| edge_2.0 | 0,408 | 0,408 | 18 |
| log_1.0 | 0,398 | 0,398 | 19 |
| log_1.4 | 0,375 | 0,375 | 18 |
| log_4.0 | 0,358 | 0,358 | 18 |
| slope_fine | 0,357 | 0,357 | 19 |
| open_6.0 | 0,353 | 0,203 | 13 |
| open_3.0 | 0,327 | -0,327 | 4 |
| log_5.6 | 0,325 | 0,314 | 18 |
| edge_0.5 | 0,322 | -0,313 | 2 |
| log_2.0 | 0,307 | 0,307 | 19 |
| log_2.8 | 0,276 | 0,276 | 18 |
| open_4.5 | 0,268 | -0,087 | 9 |
| edge_1.0 | 0,202 | -0,189 | 4 |
| edge_4.0 | 0,051 | 0,033 | 17 |
| scale_argmax | 0,000 | 0,000 | 9 |

</details>

### Üç ölçüyü eşit fiziksel kadrajda yeniden hesaplama

İki telefonda ortak olan 21 toprağın 117 özgün fotoğrafını yeniden işledik. Standart, kenar başına %60 merkez kesitin fiziksel kısa kenarı 4 piksel/mm’ye dönüştürüldükten sonra Motorola’da 94 mm, Samsung’da 158,25 mm çıktı. Aynı görüntülerde standart kesitle yeniden hesaplanan üç ölçü, arşivlenmiş fotoğraf ölçüleriyle en çok 0,0000032 mutlak fark gösterdi. Deney kesitini doğrudan özgün fotoğrafın merkezinden 100 × 100 mm aldık ve aynı 4 piksel/mm çözünürlüğe getirdik. Yeni kesitte aydınlatma ve kontrast normalizasyonunu yeniden uyguladık.

| Ölçü | Özgün standart fark | 100 mm kesitte standart fark | Özgün ham mutlak fark → 100 mm |
|---|---:|---:|---:|
| 22 mm doku yanıtı | 3,546 | 2,331 | 0,53880 → 0,42819 |
| 16 mm doku yanıtı | 1,835 | 2,009 | 0,32291 → 0,39231 |
| 9 mm parlaklık filtresi | 1,830 | 0,978 | 0,16305 → 0,09366 |

Her sütunda kendi kesitinin topraklar arası yayılımıyla standartlaştırma yapıldı. 100 mm kesitin farklarını özgün kesitin paydasına böldüğümüzde de aynı yönde değişim gözlendi (sırasıyla 2,818, 2,230 ve 1,051). 22 mm ölçüsü yeni kesitte 19/21 toprakta Samsung’da daha yüksek kaldı. [Fotoğraflardan yeniden hesaplama ve komşu kontrolü](esit_kadraj_kontrolu.py)

Yalnız bu üç ölçüyü eşit fiziksel kesitten yeniden hesaplayıp kalan 27 ölçüyü sabit tuttuk. Özgün kesitle tekrar üretilen tahminler arşivdeki sonuçları yeniden verdi. Kesişimdeki 21 toprak için çapraz telefon EMD’si aşağıda.

| Model ve kesit | Motorola → Samsung | Samsung → Motorola | Ortalama |
|---|---:|---:|---:|
| Öklid, özgün | 58,949 | 40,239 | 49,594 |
| Öklid, üç ölçü 100 mm | 52,290 | 42,880 | 47,585 |
| İnce/iri, özgün | 48,464 | 37,162 | 42,813 |
| İnce/iri, üç ölçü 100 mm | 47,055 | 38,645 | 42,850 |

İki yönün toprak ortalamasında Öklid için 11 iyileşme, 1 eşitlik ve 9 kötüleşme, ince/iri için 4 iyileşme, 13 eşitlik ve 4 kötüleşme oldu. Eşit kadraj bazı ölçü farklarını ve Motorola → Samsung hatasını azalttı. Ters yöndeki hata arttı. İnce/iri modelinde bu üç ölçünün kesitini değiştirmeye geçmiyoruz.

### Eğitim toprakları test görüntülerini ne kadar kapsıyor?

30 görüntü ölçüsüyle her toprağın en yakın üç eğitim toprağına ortalama standartlaştırılmış Öklid uzaklığını hesapladık. Eğitim için numunenin kendisi dışarıda bırakıldı ve ölçekleme kalan 23 topraktan öğrenildi. Testte 24 eğitim toprağının tamamı referans alındı. İkinci sütun görüntü uzaklığıdır. Üçüncü sütun, numunenin 30 ölçüsünden kaçının eğitimdeki en küçük ve en büyük değerler arasından çıktığını gösterir.

| Numune | En yakın üç toprağa ortalama uzaklık | Eğitim aralığı dışındaki ölçü |
|---|---:|---:|
| Airbus BS10-4bis7 | 2,587 | 0 |
| Kleinkummerfeld 9-4 | 2,731 | 2 |
| Kleinkummerfeld 2-3 | 2,863 | 0 |
| Airbus BS6-3 | 3,154 | 2 |
| Kleinkummerfeld 18-3 | 3,458 | 4 |
| Testfeld Lidl WHV | 3,656 | 7 |
| Audorfring | 4,091 | 1 |
| Airbus BS12-5bis8 | 4,495 | 3 |
| Kleinkummerfeld 2-2 | 5,284 | 4 |
| Münster BS6 9,0–10 m | 12,755 | 20 |

Eğitimde üç komşuya uzaklığın ortancası 2,406, en büyüğü 4,573. Testte ortanca 3,557. İki test numunesi eğitimde gözlenen en yüksek ortalama uzaklığı aştı. Münster için fark belirgin. Bu uzaklık toprağın laboratuvar dağılımını ya da test hatasını ölçmüyor. Çekim koşulları ve toprağın kendisi olası iki açıklama. [Kapsam ölçümü ve yakın referans kontrolü](test_komsu_kapsami.py)

**Çekimler tek tek incelendiğinde.** 127 eğitim fotoğrafının her birinde kendi toprağını bütünüyle dışarıda bıraktık. Her çekimin en yakın üç referans toprağa ortalama uzaklığında üst sınır 7,314 çıktı. Münster’in beş fotoğrafı da bu değeri aştı (12,101–15,346). Kleinkummerfeld 2-2’nin üç çekimi 5,210–5,882 aralığında kaldı. Münster dışındaki 30 test fotoğrafının hiçbiri 7,314’ü aşmadı. Bu yüzden en belirgin kapsam sorunu Münster’e ait.

**Yakın eğitim örneğini çıkarmanın etkisi.** 24 eğitim toprağını sırayla sakladık. Her turda kalan 23 referanstan iki tanesini çıkarıp modelleri tekrar eğittik. Rastgele çıkarma karşılaştırması için tüm 253 çifti eşit ağırlıkla ortaladık. Böylece en yakın çift ile herhangi iki örnek için eğitim büyüklüğü aynı (21). En yakın çift, 30 özgün görüntü ölçüsündeki standartlaştırılmış uzaklığa göre seçildi. v3 özgün ve türetilmiş eğitim görüntülerini kullandı. Saklanan üç toprağın hiçbir çekimi eğitime girmedi.

| Koşul, 24 saklı toprakta ortalama EMD | İnce/iri 3 komşu | Açık v3 |
|---|---:|---:|
| 23 referansın hepsi | 40,144 | 38,758 |
| Herhangi iki referans çıkarıldı, tüm çiftlerin ortalaması | 40,342 | 38,925 |
| Görüntüde en yakın iki referans çıkarıldı | 41,387 | 40,788 |

Rastgele çifte göre yakın çifti çıkarmanın ek ortalama hatası sırasıyla 1,045 ve 1,863 EMD. Bu fark üç komşuda 12/24, v3’te 16/24 toprakta pozitif. Üç komşuda ortanca fark −0,359, v3’te +2,401. Örneğin H038’de +15,21 ve +7,76, H405’te −11,89 ve −6,10 çıktı. Görüntüde yakın olan referansları kaybetmek ortalamada zararlı ancak bütün topraklarda aynı yönde sonuç vermiyor. Eğitimdeki bu deney iPhone test hatasını ölçmüyor.

### V3 ile ince/iri KNN birbirini tamamlıyor mu?

İki modelin aynı saklı toprak için ürettiği 11 yüzdeyi çap başına eşit ağırlıkla ortaladık. Her turda saklanan toprağın bütün çekimleri ve türevleri eğitim dışında tutuldu. Adaylık ölçütü deneyden önce belirlendi: 24 numunenin ortalama EMD’si ve iki çapraz telefon yönünün eşit ağırlıklı ortalama EMD’si v3’ten düşük olmalı. Başka karışım oranı denenmedi.

| EMD sınavı | v3 | İnce/iri KNN | %50–%50 | v3’e göre daha iyi / eşit / daha kötü toprak |
|---|---:|---:|---:|---:|
| 24 toprak | 38,758 | 40,144 | 38,002 | 16 / 0 / 8 |
| Motorola → Samsung | 41,987 | 48,464 | 43,728 | 12 / 0 / 9 |
| Samsung → Motorola | 42,831 | 37,162 | 38,474 | 12 / 0 / 9 |
| İki yönün ortalaması | 42,409 | 42,813 | 41,101 | 13 / 0 / 8 |

24 toprakta v3’e göre en büyük kazanç H493’te 17,858, en büyük kayıp H668’de 24,749 EMD oldu. Ortanca kazanç 2,067 EMD. Telefon sınavında toprak başına iki yön ortalandıktan sonra ortanca kazanç 0,690. Motorola → Samsung yönündeki ortalama hata arttı. Bunlar daha önce yöntem geliştirmede kullandığımız topraklardaki yerel sonuçlar. İki adaylık koşulu sağlandığı için tüm eğitim topraklarıyla tahminler üretildi. 35 özgün test çekimi, 10 kimlik ve 11 çap kullanıldı. Dosya şablon sırası, geçerli yüzdeler ve monotonluk kontrollerinden geçti. [Deney ve aday üretim kodu](v3_knn_esit_karisim.py)

### V3 ve diğer katılımcılardan ne öğrendik? · 21 Kasım

V3 yazarının komşu bileşeni içeren v1 modeli 35,9 yerel ve 55,19 public EMD aldı. Aynı tariften komşu bileşenleri çıkarınca v3, 38,8 yerel ve 40,90 public aldı. Bizim ince/iri KNN algoritmamız yazarın komşu yönteminden farklı. Yine de bizde de sıralama tersine döndü. Saf v3'ten eşit karışıma geçince yerel EMD 38,758'den 38,002'ye indi, public EMD 40,90'dan 52,88353'e çıktı. Yazar sonraki sürümlerde public sonuçlara göre bazı test tahminlerinin tane çaplarını kaydırdı. Örneğin v31 için 22,31 public bildirdi, buna karşılık gelen bir yerel sınav sonucu yayımlamadı. [Yazarın kodu ve puan kaydı](https://github.com/vitalclick/Predicting-Soil-Grain-Size#submission-log)

V3, aynı fiziksel ölçekteki doku ve kenarları ölçüyor ve eğitim fotoğraflarına sekiz kamera benzetimi uyguluyor. İki özellik grubuyla birer tek bileşenli PLS eğitip her kütle yüzdeliği için logaritmik çap tahminlerini ortalıyor. Önceki çap Ridge deneyimizde de hedefi yüzdelik çaplara çevirmiştik. İnce/iri KNN ise görüntü ölçülerinden öğrendiklerini üç referans seçmek için kullanıyor ve son eğriyi bu referansların ortalamasından kuruyor. Yalnız ölçüleri değiştirip üç komşu kuralını sabit tuttuğumuz önceki deneyde yerel hata 44,744'ten 39,743'e inmişti.

**Münster örneği.** D50, kütlenin yarısının altında kaldığı tane çapı. Yarışmanın çap ekseninde logaritmik ara değer alarak eğitim etiketlerinde en büyük D50'yi 6,126 mm (H368), Münster için saf v3'te 12,498 mm ve yüklediğimiz eşit karışımda 7,393 mm bulduk. Üç eğitim eğrisini ortalayınca D50, üç referansın en büyük D50'sini geçemez. Eşit karışımın bu toprakta v3 tahminini hangi yöne çektiğini ölçebiliyoruz. Gerçek Münster eğrisi açıklanmadığı için bu toprağın toplam public hataya katkısı bilinmiyor. [Saf v3 tahminleri](../work/v3_puan_kontrol.csv) · [Eşit karışım](submission_v3_knn_50_50.csv)

**Weibull deneyi.** John Draper, 20 Ağustos tarihli yarışma notunda o sırada public dördüncülüğünü, Weibull eğrisi uydurmayı, kameralar arasında tutarlı görüntü ölçülerini ve deneme gönderimleriyle puanlamayı incelemeyi birlikte anlatıyor. AmbrosM'nin ConvNeXt + Weibull notebook'unda V2 için 82,29941 public görünüyor. Bu yüzden tek başına yöntem adından üstünlük çıkaramıyoruz. Lider tablosunda görülen 0,00000 için kullandığı yöntemi açıklayan bir kaynak bulamadık. [Draper'ın yarışma notu](https://john-draper.github.io/index.html) · [AmbrosM'nin Kaggle notebook'u](https://www.kaggle.com/code/ambrosm/soil-with-convnext-and-weibull)

Kendi denememizde Weibull'un iki parametresini (ölçek ve şekil) her toprağın 11 laboratuvar yüzdesine ayrı ayrı uydurduk. Yarışma EMD'sini küçülttük ve 200 mm'de yüzdeyi 100'e sabitledik. Ölçüm önceden bilindiğinde ortalama hata 7,649 çıktı. Ardından parametrelerin logaritmalarını yalnızca fotoğraflardan tahmin ettik. Saf v3 ile aynı görüntü ölçüleri, veri artırma, örnek ortalamaları, iki PLS modeli ve saklı numuneler kullanıldı. Saklı toprağın laboratuvar ölçümü yalnızca puanlamaya girdi. Bu, Draper'ın modelinin yeniden üretimi değil, Weibull fikrinin bizim koşullarımızdaki karşılaştırmasıdır. [Yeniden çalıştırılabilir karşılaştırma](../work/weibull_karsilastirma.py)

| Saklı topraklarda ortalama EMD | Saf v3 | İki parametreli Weibull |
|---|---:|---:|
| 24 toprak | 38,758 | 39,210 |
| Motorola → Samsung, 21 toprak | 41,987 | 42,889 |
| Samsung → Motorola, 21 toprak | 42,831 | 43,961 |

Weibull, 24 toprağın 11'inde daha düşük hata verdi. Üç sınavın ortalamasında da saf v3 önde kaldı. Yeni Kaggle gönderimi yapılmadı. Sonraki karşılaştırmada özellikle iri toprakları birlikte saklayıp modelin D50'yi sistematik olarak küçük tahmin edip etmediğine ve bütün eğrinin EMD'sine bakacağız.

### 21 Kasım · İri topraklar için önceden belirlenen deney planı

**Soru.** Eğitimde benzer irilikte toprak kalmayınca, saf v3 tipik tane çapını ne yönde tahmin ediyor ve bütün eğrinin hatası nasıl değişiyor? Saf v3 referans, kendi ince/iri üç komşu modelimiz karşılaştırma olacak. Etiketlerden sınav grubu ve eğitim sınırı belirlendi. Bu plan kaydedilirken modeller bu sınav için çalıştırılmadı.

1. **Beş toprağı birlikte sakla.** D50'ye göre en iri beş numuneyi ayır. H368 (6,126 mm), H374 (5,911), H615 (5,339), H367 (4,792) ve H366 (4,791). Kalan 19 toprağın en büyük D50'si 4,585 mm. Bu ayrım yazarın açık kodundaki beş iri toprağı saklama düzenini izliyor. Her modelin özellikleri ve ayarları sabit kalacak. Saklanan toprağın tüm fotoğrafları ve türevleri eğitim dışında olacak. [Yazarın deney kodu](../work/vitalclick/src/experiments.py)
2. **Eğitim sayısını sabit tutan bir kontrol kur.** Aynı beş toprağın her birini sırayla tahmin et. O toprağın kendisi her zaman dışarıda kalsın. Bu kez diğer dört iri toprak eğitime girsin, onların yerine daha ince 19 topraktan dördü rastgele çıkarılsın. Böylece her iki koşulda da 19 toprakla eğitim yapılır ve aynı hedefler karşılaştırılır. Kontrolü sabit tohumla (42) 100 farklı dörtlü üzerinde tekrarla, aynı dörtlüleri iki modele de uygula. Önce her saklı toprak için kontrol hatalarını ortala. Bu tekrarlar beş ayrı sınav toprağının yerini tutmaz.
3. **Her toprağı kendi kontrolüyle karşılaştır.** Birincil ölçü bütün eğride yarışma EMD'si. Her numune için “iri örnekler yokken EMD eksi iri örnekler varken ortalama EMD” farkını ver. Pozitif fark, iri örneklerin çıkarılmasının o hedefte zarar verdiğini gösterir. İkinci ölçü D50 tahmini / laboratuvar D50'si. 1'in altı küçük tahmin demektir. Beş satırı, ortalama EMD farkını ve ortanca D50 oranını göster. Her seferinde bir satırı kaldırınca ortalama EMD farkının yönünün değişip değişmediğini de yaz.
4. **Bulguya göre öncelik seç.** V3'te çoğu iri toprakta çap küçük kalır ve kontrol koşuluna göre ortalama EMD artarsa, sonraki deneyi iri tanelerin görüntüden milimetre cinsinden ölçülmesine ayır. Çaplar yakın olduğu halde eğri hatası yüksekse, hatayı yarışmanın çap aralıklarına ayırarak hangi kütle payının kaçtığını bul. Sonuç tek toprağa dayanıyorsa önce o toprağın çekimlerini ve laboratuvar eğrisini incele. Bu beş örnek bilerek uçtan seçildi. Sonuç, 10 iPhone toprağının ortalama Kaggle puanı için bir tahmin olarak kullanılmayacak.

### 21 Kasım · Beş iri toprak sınavının sonucu

Önceden belirlenen 19 eğitim toprağı ve beş hedefle çalıştık. Kontrol için tohum 42 ile 100 farklı ince toprak dörtlüsü seçildi. Her iki modele aynı bölümler verildi. Aşağıda fark, iri örnekler çıkarıldığında EMD'deki artış. [Deney kodu](iri_toprak_sinavi.py) · [Beş numunenin tam sonuçları](iri_toprak_sonuclari.csv)

| Model | Beşi birlikte saklı EMD | İri referanslı kontrol EMD | Ortalama fark | Saklı grupta ortanca D50 oranı |
|---|---:|---:|---:|---:|
| Saf v3 | 29,821 | 26,532 | +3,289 | 0,492 |
| İnce/iri KNN | 21,161 | 15,747 | +5,414 | 0,665 |

V3 dört, KNN beş toprakta D50'yi küçük tahmin etti. Tek tek bir toprağı değerlendirmeden çıkarınca ortalama EMD farkı v3 için +1,960 ile +5,766, KNN için +5,015 ile +5,806 arasında kaldı. KNN'nin saklı grupta daha düşük EMD alması, yanlış çap tahmininin bütün eğrideki hatayı tek başına açıklamadığını gösteriyor. V3, H374'te gerçek 5,91 mm'ye karşı 1,37 mm (kontrolde 1,59 mm), H615'te gerçek 5,34 mm'ye karşı 6,20 mm (kontrolde 9,04 mm) verdi. Bir sonraki incelemede bu iki toprağın iri tanelerini aynı fiziksel ölçekte ve farklı çekimlerde karşılaştıracağız. [Deney raporu](../README.md)

### 21 Kasım · H374 ve H615'in sekiz çekimi

Önceki iri toprak sınavındaki aynı 19 referansla saf v3'ü yeniden kurduk. Aşağıdaki numune tahminlerinde önce görüntü ölçüleri çekimler arasında ortalandı. Tek çekim aralıkları her fotoğrafı tek başına modele verdiğimizde elde edildi. [Deney kodu](h374_h615_incelemesi.py) · [Çekim bazında sayılar](h374_h615_cekimleri.csv) · [Aynı ölçekte fotoğraflar](h374_h615_ayni_olcek.png)

| Ölçüm | H374 | H615 |
|---|---:|---:|
| Laboratuvar D50 (mm) | 5,91 | 5,34 |
| Saf v3 D50, özgün kadraj (mm) | 1,37 | 6,20 |
| Tek çekim tahmin aralığı (mm) | 1,04–2,25 (3 çekim) | 3,97–10,66 (5 çekim) |
| Görüntünün yalnız ortadaki 90 × 90 mm'sini verince D50 (mm) | 0,97 | 5,17 |
| 0,063 mm'den ince kütle, laboratuvar | %9,30 | %4,84 |
| 20 mm'den iri kütle, laboratuvar | %23,51 | %8,86 |
| 20 mm'den iri kütle, v3 özgün kadraj | %2,95 | %23,65 |

**Fotoğraftan okuduğumuz.** Aynı fiziksel ölçekte H374'ün iri taşlarının arasında ve üzerinde ince dokulu malzeme görülüyor. H615'in çekimlerinde pek çok tane kenarı ve taneler arasındaki koyu boşluklar daha belirgin. Laboratuvar H374'te hem ince malzemenin hem de 20 mm üzerindeki iri malzemenin kütle payını daha yüksek ölçmüş. V3, 20 mm üzerindeki kütle payında bu iki toprağın sırasını ters çevirdi.

**Modelden ölçtüğümüz.** H615 ile H374 arasındaki model içi log10(D50) farkı +0,6433. Sabit iki PLS bileşeninin katsayılarını, kırpılmış ve standartlaştırılmış özellik farklarıyla çarpıp ortaladık. Toplamı doğrudan modeldeki farkla eşleşti. Ölçeğe göre açık-koyu doku farkları +0,2176, kenar yoğunlukları +0,2090, morfolojik açma ölçüleri +0,1577, ölçek özetleri +0,0389 ve eğimler +0,0202 katkı verdi. İlk iki grubun net payı yaklaşık %66. Birbirleriyle ilişkili ölçülerin bu modeldeki katkıları, fiziksel neden veya başka topraklar için önem sırası olarak yorumlanamaz.

**Kadraj ve kamera.** Her çekimi 4 piksel/mm'ye getirip merkezden 90 × 90 mm aldık. Yalnız tahmin görüntülerinin alanını değiştirip aynı modeli uyguladık. Sıralama korundu ve H374 daha da küçük tahmin edildi. H374'ün üç fotoğrafı yalnız Motorola Edge 60 Fusion ile çekilmiş. 24 toprak içinde bu telefondan başka örnek yok. H615'te iki Motorola Edge, üç Samsung A52 çekimi var. Çekim farkı ile toprağın kendi yüzeyi bu iki numunede birbirinden bağımsız olarak ölçülemiyor.

**Bir sonraki sınanabilir fikir.** Görüntü genelindeki ortalama dokuya, birden fazla büyük taşın göründüğü bölgeleri ayrıca tarif eden tek bir ölçü ekle. Önceki 4 × 4 doku değişkenliği ve en büyük tek parlak bölge denemelerini tekrarlamamak için fiziksel çap aralığını ve birleştirme kuralını önceden belirle. Başarıyı 24 toprakta numune dışarıda bırakılarak bütün eğrinin EMD'si ve 20 mm üzeri kütle hatasıyla değerlendir. H374 bu fikrin geliştirilmesinde kullanıldığı için onun tek başına iyileşmesi yeterli kanıt olmaz. Bu inceleme ve görsel yerelde kaydedildi.

### 21 Kasım · Birden çok iri tane adayı için deney kararı

Sonuçlara bakmadan tek bir ek ölçü ve karşılaştırma düzeni belirledik. Her çekimin merkezindeki 90 × 90 mm alanı 2 piksel/mm'ye indirip dokuz eşit bölgeye ayıracağız. 20, 32 ve 50 mm çaplı dairelerin içindeki ortalama gri tonu, çevresindeki eşit alanlı halkanın ortalamasıyla karşılaştıracağız. Mutlak farkı kesitin gri ton standart sapmasına böleceğiz. Çevre halkası dahil kesitin içine sığan en kuvvetli adayı bulduktan sonra, başka bir bölgede merkezi bulunan ve ilk adayın diskiyle örtüşmeyen en kuvvetli ikinci adayın değerini alacağız. Bu sayı fiziksel çapı bilinen bir kontrast ölçüsü. Gerçek taş adedi veya kütle yüzdesi ölçümü sayılmayacak.

Ölçüyü v3'ün iki özellik listesine de birer kez ekleyeceğiz. Her iki sürümde aynı özgün çekimler, aynı sekiz eğitim türevi, aynı iki PLS ve aynı 24 numune bölümü kullanılacak. Saklanan toprağın hiçbir çekimi eğitime girmeyecek. Karar için bütün eğride ortalama EMD ve 20 mm üzeri kütle payında ortalama mutlak hata (yüzde puan) birlikte düşmeli ve bu iki düşüş H374 dışındaki 23 toprakta da korunmalı. Kazançların kaç numuneye yayıldığını ve aynı toprağın farklı telefonlarla çekildiği 21 numunenin çapraz telefon sonuçlarını ayrıca raporlayacağız.

**Ölçüm ve sonuç.** 127 özgün eğitim fotoğrafındaki ve sekizer kamera türevindeki ölçüleri çıkardık. 1.143 dosya/türev eşleşmesini v3 önbelleğiyle doğruladık. Sabit görüntüde ölçü sıfırdı. Sentetik görüntüde biri açık, biri koyu, çapları 20 mm olan iki dairenin merkezleri bir piksel içinde bulundu. Gerçek görüntülerde büyük bir taş, gölge veya birden çok küçük taş benzer bir açık-koyu farkı üretebiliyor. H615'in ilk Motorola çekiminde ikinci işaret birkaç taneyi ve aralarındaki koyu boşluğu kapsıyor. [Örnek işaretlemeler](ikinci_iri_aday_ornekleri.png) · [Deney kodu](ikinci_iri_aday.py) · [Numune bazında sonuçlar](ikinci_iri_aday_sonuclari.csv)

| Sınav | Saf v3 EMD | Ek ölçülü EMD | Saf v3, 20 mm üzeri mutlak hata (yüzde puan) | Ek ölçülü mutlak hata (yüzde puan) |
|---|---:|---:|---:|---:|
| 24 toprak | 38,75755 | 38,70930 | 6,76191 | 6,74876 |
| H374 dışındaki 23 sonuç | 38,16423 | 38,12727 | 6,16840 | 6,15578 |
| Motorola → Samsung, 21 toprak | 41,98687 | 41,87208 | 4,21519 | 4,22006 |
| Samsung → Motorola, 21 toprak | 42,83059 | 42,26061 | 4,23076 | 4,11829 |

Önceden belirlenen koşul sağlandı. 24 numunede ortalama EMD 0,04825, iri payın ortalama mutlak hatası 0,01315 yüzde puan düştü. EMD 14 toprakta azaldı ve 10'unda arttı. İri payın hatası 7 toprakta azaldı, 13'ünde aynı kaldı, 4'ünde arttı. Her seferinde bir numunenin **puanını** dışarıda bıraktığımızda ortalama EMD kazancı −0,0948 ile +0,1375 arasında değişti. H615 çıkarılınca iki ortalama kazanç da kayboldu. H374 çıkarılarak hesaplanan satır dahil bu kontrollerde saklı numune tahminleri aynıdır.

H374'te gerçek iri pay %23,5125. Yeni karşılaştırmadaki saf v3 tahmini %3,0998, ek ölçülü tahmin %3,1251. H615'te gerçekte %8,8597 olan pay için tahmin %30,7918'den %29,1456'ya indi. Bu turda her toprak için 23 referans kullandık. Önceki iki fotoğraf grubu incelemesinde beş iri toprağı birlikte saklayıp 19 referans kullanmıştık. Sonuçları doğrudan bu iki farklı eğitim düzeni arasında kıyaslamıyoruz.

**Karar.** Saf v3 ana referans olarak kalıyor. Bu küçük ve numune seçimine duyarlı kazanç için Kaggle adayı hazırlanmadı. Sonraki önceliği belirlemek için saf v3'ün 24 saklı tahminindeki her elek mutlak hatasını ilgili logaritmik çap aralığıyla çarptık. En büyük üç katkı 0,063 mm'de 6,830 EMD, 0,2 mm'de 5,839 ve 0,63 mm'de 4,962. Bunlar toplam EMD'nin %45,49'unu oluşturuyor. 20 mm'nin katkısı 3,370 EMD (%8,69). H374'ün puanı dışarıda tutulduğunda ilk üç elek sıralaması aynı kaldı. İnce/iri KNN'de de aynı üç eleğin payı %48,46 idi. Bir sonraki karşılaştırma için bu üç elekte iki modelin aynı topraklarda aynı yönde yanılıp yanılmadığını incelemek öncelikli. Bu sonuçlar yereldir.

İnce eleklerdeki hataların yönünü de aynı 24 saklı numunede saf v3 ve ince/iri KNN için karşılaştırdık. Modeller 0,063 mm'de 22/24, 0,2 mm'de 20/24 ve 0,63 mm'de 21/24 toprakta aynı yönde yanıldı. H037 ve H038'de iki model de üç elekten geçen kütleyi fazla tahmin etti. H038'de 0,63 mm'de v3 +55,00, KNN +60,64 yüzde puan; H037'de sırasıyla +46,98 ve +46,22 yüzde puan saptı. Bu iki numune, üç elekte her iki modelin paylaştığı hata büyüklüğüne göre öne çıktı (aynı yöndeyse iki mutlak hatanın küçüğü, aksi halde sıfır; ardından üç eleğin ortalaması). Elde edilen 72 değer aynı 24 topraktan geliyor. H037 ve H038'i bir sonraki tanıda laboratuvar eğrisi benzer ve tahmin hatası daha düşük referanslarla karşılaştıracağız.

### 21 Kasım · Literatür önerilerinin değerlendirmesi

Bu aşamada üç öneriyi değerlendirdik. Yeni model çalıştırılmadı. 24 eğitim kimliğinin tamamı Zenodo kaydındaki 26 numuneyle eşleşiyor. Kayıttaki diğer iki kimlik H031 ve H693. Örtüşme kimlik düzeyinde doğrulandı. Tüm kayıtla eğitilmiş bir toprak modelinin aynı 24 numunedeki puanı bağımsız doğrulama sağlayamaz. [Zenodo veri kaydı](https://zenodo.org/records/14725633)

1. **Öncelik bölgeler arası dalgacık enerjisi.** 1, 2 ve 4 mm desen ölçeklerinin her birinde, aynı fiziksel alanlı bölgelerin enerjilerinden ortalama ve 90. yüzdelik eksi ortanca hesaplanacak. İkinci özet, enerjisi yüksek bölgelerin tipik bir bölgeden ne kadar ayrıldığını ölçer. Bölgelerin yerleri değiştirildiğinde değeri aynı kalır. Merkezde, kenarda veya yan yana birikme ayrı bir araştırma sorusudur. Buscombe'un yüzey dokusu çalışmasından hareketle bu özetlerin laboratuvardaki ince kütle payı hakkında dolaylı bilgi taşıyıp taşımadığını sınayacağız. Önceki 4 × 4 deneyimiz altı doku ölçüsünün bölgeler arası standart sapmasını KNN'ye eklemişti. [Buscombe](https://doi.org/10.1111/sed.12049)
2. **Çok bölgeli DINOv2 ve Ridge.** Montréal tezinin 316 numunedeki çözünürlük karşılaştırması bölgesel ayrıntıyı korumayı destekliyor. Eski DINOv2 denememizde tek kesit üç komşu seçmek için kullanılmıştı. DINOv2'ye geçilirse tek kesit + Ridge ile çok bölge + Ridge karşılaştırılmalı. Her toprağın bölgeleri önce kendi içinde birleştirilmeli ve her toprak bir eğitim örneği olarak kalmalı. Öğrenilmiş bölge ağırlıklarını sonraya bırakıyoruz. [Tez](https://espace.etsmtl.ca/id/eprint/4065/)
3. **Elek aralığı kütle payları.** V3 zaten geçerli, monoton bir eğri üretiyor. Yeni soru, aynı görüntü ölçülerinden aralık paylarını öğrenmenin kümülatif eğrideki EMD'yi azaltıp azaltmadığı. Görüntü girdilerini ve doğrulamayı sabit tutan ayrı bir deney olarak değerlendirilecek. [Anh ve arkadaşları](https://doi.org/10.1186/s40703-026-00263-x)

**Dört sürümlü dalgacık planı.** Saf v3, v3 + ortalama, v3 + bölgesel fark ve v3 + ortalama + bölgesel fark aynı fiziksel kesitlerde, aynı PLS ayarlarıyla ve aynı saklı topraklarda karşılaştırılacak. Sırasıyla 0, 3, 3 ve 6 yeni özellik kullanılacak.

| Karşılaştırma | Ölçülen katkı |
|---|---|
| Saf v3 ile v3 + ortalama | Ortalama dalgacık enerjisinin mevcut v3 girdilerine katkısı |
| v3 + ortalama ile v3 + bölgesel fark | Eşit sayıda özellik kullanıldığında iki özetin tahmin başarısı |
| **v3 + ortalama ile v3 + ortalama + bölgesel fark** | **Ortalama zaten modeldeyken bölgesel farkı eklemenin tahmin başarısına katkısı** |

**Karar ölçüleri.** Ana karşılaştırmada her saklı toprağın kazancı, ortalamalı modelin EMD'sinden ortalama ve bölgesel farkı birlikte kullanan modelin EMD'si çıkarılarak hesaplanacak. Pozitif değer iyileşme demektir. Birincil sonuç 24 toprağın tüm eğri üzerindeki ortalama EMD farkıdır. Ortanca kazanç, iyileşen numune sayısı ve üç ince elekteki işaretli hatalar ayrıca kaydedilecek. H037 ve H038 bu fikre esin verdiği için aynı saklı tahminlerin diğer 22 topraktaki ortalama kazancı da raporlanacak. İki telefon yönü (Motorola → Samsung ve Samsung → Motorola) ayrı gösterilecek. Kazancın kalan 22 toprakta ve her iki telefon yönünde de korunması bölgesel farkı kullanma gerekçesini güçlendirecek. Dört sürümün puanları deney çalıştırıldığında eklenecek.

**Çalıştırma düzeni.** Sonuçlar görülmeden merkezdeki 90 × 90 mm kesit, 4 piksel/mm ve 4 × 4 eşit bölge seçildi. Gri kesit kendi ortalaması ve standart sapmasıyla normalize edilecek. 1, 2 ve 4 mm dalga boylarında dört yönün (0°, 45°, 90°, 135°) karmaşık Morlet yanıtlarının karesel büyüklükleri ortalanacak. Gauss penceresinin standart sapması dalga boyunun 0,75 katı. Süzme tüm kesitte yapıldıktan sonra her 22,5 × 22,5 mm bölgede ortalama enerji hesaplanacak. Her dalga boyunda 16 bölge değerinin ortalaması ve 90. yüzdeliği eksi ortancası alınacak. Ek özellikler v3'ün iki PLS bileşenine de girecek. Eski görüntü özellikleri ve eğitim türevleri korunacak.

### 21 Kasım · Dört sürümlü dalgacık deneyinin sonuçları

127 özgün fotoğraf ve sekizer eğitim türevi için altı ölçü çıkarıldı. 1.143 fotoğraf/türev anahtarı v3 önbelleğiyle eşleşti. Saklı numunenin yalnızca özgün çekimleriyle tahmin yapıldı. Düz sentetik görüntüde altı ölçü de sıfırdı. 1, 2 ve 4 mm periyotlu desenlerin her biri en yüksek enerjiyi kendi ölçeğinde verdi. Desen yalnızca görüntünün yarısında kaldığında bölgesel fark her üç ölçekte de büyüdü. 264 tahmin eğrisinden EMD ve ince eleklerin işaretli hataları yeniden hesaplandı. Üç sınavdaki 66 saf v3 puanı, önceki deneyin aynı numuneleriyle birebir kontrol edildi (mutlak tolerans 0,0000001 EMD). [Çalıştırma kodu](dalgacik_karsilastirmasi.py) · [Tahmin eğrileri ve numune sonuçları](dalgacik_sonuclari.csv) · [24 eşleştirilmiş EMD farkı](dalgacik_eslestirilmis_kazanc.png)

| Saklı numunelerde ortalama EMD | Saf v3 | v3 + ortalama | v3 + bölgesel fark | v3 + ikisi |
|---|---:|---:|---:|---:|
| 24 toprak | **38,758** | 38,981 | 39,649 | 39,689 |
| H037 ve H038 dışındaki 22 puan | **32,685** | 33,350 | 33,820 | 34,260 |
| Motorola → Samsung, 21 toprak | **41,987** | 42,944 | 43,335 | 44,130 |
| Samsung → Motorola, 21 toprak | 42,831 | **42,120** | 43,376 | 42,824 |

| Ana karşılaştırma (ortalama enerjili EMD eksi iki özeti de kullanan EMD) | Ortalama kazanç | Ortanca kazanç | İyileşen / eşit / kötüleşen |
|---|---:|---:|---:|
| 24 toprak | −0,708 | −0,066 | 11 / 0 / 13 |
| Seçili ikili dışındaki 22 puan | −0,909 | −0,258 | 9 / 0 / 13 |
| Motorola → Samsung | −1,185 | −0,700 | 8 / 0 / 13 |
| Samsung → Motorola | −0,704 | −0,740 | 7 / 0 / 14 |

H037'de EMD 86,592'den 84,782'ye, H038'de 115,242'den 114,036'ya düştü. En büyük kayıplar F827'de 7,047, H668'de 6,897 ve H126'da 6,807 EMD idi. Her seferinde bir numunenin puanını çıkardığımızda ortalama kazanç −0,874 ile −0,432 arasında kaldı. 22 puanlık satırda mevcut saklı tahminlerin ikisini puanlamadan çıkardık.

| 24 toprakta ince elek hatası (yüzde puan) | Ortalama enerji, işaretli | İki özet, işaretli | Ortalama enerji, mutlak | İki özet, mutlak |
|---|---:|---:|---:|---:|
| 0,063 mm | −1,270 | −2,124 | 13,376 | 13,854 |
| 0,2 mm | +0,566 | −0,333 | 11,972 | 12,458 |
| 0,63 mm | +4,030 | +3,632 | 10,066 | 10,315 |

İşaretli hata tahmin eksi laboratuvar ölçümüdür. H037'de 0,063 / 0,2 / 0,63 mm'deki üç işaretli hata sırasıyla +15,57 / +40,01 / +46,09'dan +14,44 / +38,87 / +45,74'e indi. H038'de +17,97 / +41,60 / +53,65'ten +17,35 / +40,77 / +53,28'e indi. Buna karşılık 24 topraktaki ortalama mutlak hata üç elekte de arttı.

**Karar.** Seçilen ölçek ve bölge özeti, ortalama enerjiyi kullanan modele genellenebilir bir kazanç göstermedi. Saf v3 referans olarak kalıyor. Görüntü girdilerini sabit tutup ardışık elekler arasındaki kütle paylarını tahmin etmek, görüntü tarafında ek değişiklik gerektirmeyen bir sonraki bağımsız deney seçeneği.

### 21 Kasım · Kütle payı modeli ve saf v3

Paylaşılan sayısal özeti aynı kamera özelliklerinden bağımsız olarak yeniden hesapladık. Her turda aynı toprağın tüm çekimleri dışarıda kaldı. Pay modeli, 11 elek yüzdesinin ardışık farklarını (ilk elek için doğrudan o elekten geçen payı) öğrendi. İki PLS tahmini ortalandıktan sonra pay vektörü, negatif olmayan ve toplamı %100 olan vektörler kümesine Öklid uzaklığıyla izdüşürüldü. Kümülatif toplamdan eğri elde edildi. Saf v3 için önceki 66 numune/senaryo EMD'si 0,0000001 toleransla doğrulandı. [Yerel doğrulama kodu](kutle_payi_kontrolu.py) · [İki modelin saklı tahminleri](kutle_payi_kontrol_sonuclari.csv) · [Güncellenen Word raporu](Toprak_Tane_Boyutu_Literatur_ve_Deney_Sonuclari.docx)

| Yerel sınama | Saf v3 EMD | Kütle payı EMD | Fark (pay − v3) |
|---|---:|---:|---:|
| 24 toprak | 38,758 | 37,218 | −1,540 |
| H037 ve H038 hariç, 22 puan | 32,685 | 30,970 | −1,715 |
| H493 hariç, 23 puan | 37,998 | 38,133 | +0,135 |
| Motorola → Samsung, 21 toprak | 41,987 | 46,152 | +4,166 |
| Samsung → Motorola, 21 toprak | 42,831 | 39,322 | −3,509 |

24 toprağın 17'sinde kütle payı modeli daha düşük EMD verdi. Saf v3'e göre ortanca kazanç 0,645 EMD. H493'te 56,238'den 16,173'e inen hata ortalama kazancın yönünü belirledi. Hariç satırları aynı saklı tahminleri yeniden puanlar. İşaretli tablo farkları tam hassasiyetli ortalamalardan hesaplandı.

0,063, 0,2 ve 0,63 mm'de kütle payı modelinin ortalama mutlak hatası sırasıyla 1,123, 2,199 ve 0,658 yüzde puan arttı. En büyük iyileşme 20 mm'de 2,369 yüzde puan. İri elekteki kazanç, öncelikli ince eleklerdeki ve Motorola → Samsung aktarımındaki kaybı gidermediği için ana model saf v3 olarak korundu. Bunlar yerel sonuçlardır.

### Tek kesitli ve beş kesitli DINOv2 + Ridge için deney düzeni

Sonuçları görmeden kesitleri ve Ridge ayarını sabitledik. Her kesit 100 × 100 mm. İkinci sürüm, ilk sürümün merkez vektörüne sağ/sol ve üst/alt yönlerinde 20'şer mm kaydırılmış dört çapraz kesitin vektörlerini ekleyip beşini eşit ağırlıkla ortalayacak. Kesitler örtüşüyor. 140 × 140 mm'lik toplam alan, 127 eğitim fotoğrafının en kısa fiziksel kenarına (156,63 mm) sığıyor. Her iki sürüm aynı 127 özgün fotoğrafı, önceki deneyin DINOv2-S/14 ağırlıklarını ve 224 × 224 renkli girdiyi kullanacak. Her numunede fotoğraf vektörlerinin aritmetik ortalaması alınacak.

Her dış turda yalnızca eğitim topraklarında standartlaştırma, sekiz bileşenli PCA (bileşen varyansları bir olacak şekilde) ve sabit α = 10 ile Ridge uygulanacak. Hedef, laboratuvar eğrisinden v3'ün dönüşümüyle elde edilen logaritmik tane çapı yüzdelikleri. Numune bazında saklama ve iki çapraz telefon sınavı önceki v3 sınavlarıyla aynı olacak. İlk karar 24 topraktaki ortalama EMD ile tek ve beş kesit arasında verilecek. Kazananın saf v3 karşısındaki numune bazlı EMD farkları, her seferinde bir puanın dışarıda kalması ve iki telefon yönü ayrıca incelenecek.

### 21 Kasım · Tek ve beş kesitli DINOv2 + Ridge sonuçları

127 fotoğraftan 635 kesitin özelliklerini çıkardık. İki Ridge sürümünde merkez kesitini ve her turda sınava ayrılan toprakları birebir eşledik. Önceki üç komşulu model aynı merkez kesitleriyle 53,427662 EMD verdi ve kayıtlı 53,428 sonucunu yeniden üretti. Aşağıdaki karşılaştırmada karar ölçütü, her toprağa eşit ağırlık veren ortalama EMD’dir. Düşük değer daha iyi.

| Yerel sınama (eğitim → sınama) | Saf v3 | Tek kesit + Ridge | Beş kesit + Ridge | Beş kesitin tek kesite göre kazancı | Beş kesitin tek kesiti geçtiği toprak |
|---|---:|---:|---:|---:|---:|
| 24 toprak, tüm telefonlar | 38,758 | 50,617 | 48,849 | +1,769 | 16/24 |
| Motorola Edge → Samsung A52 | 41,987 | 53,780 | 55,375 | −1,595 | 10/21 |
| Samsung A52 → Motorola Edge | 42,831 | 58,310 | 54,668 | +3,642 | 16/21 |

Tek kesitten beş kesite geçince toprak başına ortanca kazanç 1,900 EMD oldu. Sırasıyla bir toprağın puanını ortalamadan çıkarıp aynı tahminleri değerlendirdiğimizde beş kesitin ortalama kazancı 24 karşılaştırmanın tamamında korundu (1,334–2,531 EMD). İki telefon arasındaki aktarımda ise kazancın yönü, eğitim için kullanılan telefona bağlıydı.

Beş kesitli model saf v3’ü yalnızca 6/24 toprakta geçti. 24 toprakta ortalama hatası saf v3’ten 10,091 EMD daha yüksekti. Her seferinde bir puan çıkarıldığında saf v3’ün üstünlüğü 8,234–12,103 EMD arasında kaldı. Özellikle H493 hariç tutulduğunda 23 toprakta saf v3 37,998, beş kesit 48,810, tek kesit 50,230 EMD verdi.

**Karar.** Merkeze dört örtüşen kesit eklemek bu sabit Ridge ayarında 24 topraktaki ortalama tahmini iyileştirdi. Saf v3 hem 24 toprakta hem iki telefon yönünde daha düşük EMD verdiği için ana model olarak kalıyor. [Deney kodu](dinov2_kesit_ridge.py) · [132 tahmin eğrisi, numune hataları ve eşleştirilmiş v3 puanları](dinov2_kesit_ridge_sonuclari.csv)

### 21 Kasım · Saf v3 ve beş kesitli DINOv2 ince eleklerde hangi hataları paylaşıyor?

Her toprak ve sınama için iki modelin saklı tahminini aynı laboratuvar sonucuyla eşleştirdik. Her modelin 66 tam eğrisinden EMD’yi yeniden hesaplayarak dosyalarda yazan puanları doğruladık. Aşağıdaki hatalar elekten geçen kütle yüzdesindeki **ortalama mutlak sapmayı** gösterir. Birim yüzde puandır. “Aynı yön” iki modelin de ölçümün üzerinde veya ikisinin de altında kalmasıdır.

| Elek | Saf v3 hatası | DINOv2 hatası | DINOv2 daha yakın | İkisi aynı yönde yanıldı |
|---|---:|---:|---:|---:|
| 0,063 mm | 13,61 | 16,63 | 8/24 | 20/24 |
| 0,2 mm | 11,72 | 19,72 | 8/24 | 20/24 |
| 0,63 mm | 9,89 | 15,89 | 6/24 | 20/24 |

Her toprakta üç eleğin mutlak hatalarına eşit ağırlık verdik. H037’de DINOv2 üç elek ortalamasında 19,03 yüzde puan, H038’de 4,33 puan kazandırdı. İkisinde de üç elekteki fazla tahmini küçülttü. H037’de 0,2 mm’deki fazla tahmin +42,77’den +18,15’e indi. F827’de aynı elekteki eksik tahmin −12,81’den −34,49’a büyüdü. Üç elek ortalamasında DINOv2’nin v3’ü geçtiği yedi toprak H037, H616, H038, H372, H371, H637 ve H030. H030’daki fark yalnızca 0,06 yüzde puan. Aynı toprağın üç ölçümü tek sonuç olarak sayıldı.

| Sınama | DINOv2’nin daha yakın olduğu toprak | DINOv2’nin saf v3’e göre ek ortalama mutlak hatası (yüzde puan) |
|---|---:|---:|
| 24 toprak | 7/24 | +5,67 |
| H037 ve H038 puanları hariç | 5/22 | +7,25 |
| Motorola Edge → Samsung A52 | 6/21 | +4,34 |
| Samsung A52 → Motorola Edge | 6/21 | +6,74 |

**İki tahmini karıştırınca ne olabilir?** Sabit bir DINOv2 ağırlığı için karışım (1−w) × v3 + w × DINOv2 olarak tanımlandı. Saf v3’ten başlayınca hatanın değişme yönünü ayrıca hesapladık. Bir elekte v3’ün işaretli hatası sıfırdan farklıysa başlangıç eğimi, o hatanın işaretiyle (DINOv2 tahmini − v3 tahmini) çarpılarak bulunur. V3 tam isabet etmişse iki tahminin mutlak farkı kullanılır. Her mutlak hata ağırlığa göre dışbükey olduğu için ortalama başlangıç eğimi pozitifse hiçbir sabit pozitif ağırlık o ortalamayı azaltamaz. Negatif eğim, yeterince küçük ağırlıkla yerel azalmanın mümkün olduğunu gösterir. Tablodaki eğimler model puanı değildir.

| Sınama | 0,063 mm başlangıç eğimi | 0,2 mm | 0,63 mm | Üç eleğin eşit ağırlıklı ortalaması | Tüm eğride EMD başlangıç eğimi |
|---|---:|---:|---:|---:|---:|
| 24 toprak | −1,191 | +5,435 | +3,548 | +2,597 | −2,073 |
| H037 ve H038 puanları hariç | −0,287 | +7,284 | +4,688 | +3,895 | −0,132 |
| Motorola Edge → Samsung A52 | −2,865 | +3,776 | +3,958 | +1,623 | −4,266 |
| Samsung A52 → Motorola Edge | −7,261 | +0,048 | +2,803 | −1,470 | −16,443 |

**Karar.** 24 toprakta üç ince eleğin ortak hatasını azaltmak için iki modele sabit ağırlık vermeyi eledik. 0,063 mm tek başına ele alındığında küçük bir DINOv2 payı aynı örneklerde fayda sağlıyor. Tüm eğride de küçük bir payın hata azaltma yönü var, ancak fikre esin veren iki toprak çıkarıldığında bu eğim sıfıra çok yaklaşıyor. Bu inceleme ana model seçimini değiştirmiyor. [Toprak, telefon ve elek bazında 198 karşılaştırma](dinov2_ince_elek_sonuclari.csv) · [Hesaplama ve doğrulama kodu](dinov2_ince_elek_kontrolu.py)

### 21 Kasım · Kaggle gönderim ve final seçim önerisi (arka plan)

Kaggle’ın yarışma servisi 21 Kasım’da son gönderim zamanını 30 Kasım 2026 11.00 UTC (Türkiye saatiyle 14.00), günlük üst sınırı 5 olarak verdi. Kaggle gönderim ekranında final için en fazla iki dosya seçilebiliyor. Public tablo test verisinin yaklaşık %30’unu kapsıyor. [Yarışma](https://www.kaggle.com/competitions/soil-grain-size-from-photos) · [Public tablonun kapsamı](https://www.kaggle.com/competitions/soil-grain-size-from-photos/leaderboard)

1. **21 Kasım için önerilen referans gönderimi.** Kendi çalıştırmamızla üretilen saf v3 dosyasındaki 10 numune kimliğini, 11 elek yüzdesini ve eğri koşullarını tekrar doğruladık. Dosya, satır sonları eşitlendiğinde yazarın v3 dosyasıyla birebir aynı. Önceki karar deney puanını tekrar ölçmenin bilgi değerine ilişkindi. Bu gönderimin amacı saf v3’ü kendi hesabımızdaki iki final seçeneğinden biri olarak kullanılabilir hale getirmek. 40,90 yazarın bildirdiği public puan. Bizim şu ana kadarki en iyi doğrulanmış gönderimimiz 52,88353. [Saf v3 dosyası](../work/v3_puan_kontrol.csv) · [Yazarın puan kaydı](https://github.com/vitalclick/Predicting-Soil-Grain-Size#submission-log)
2. **22 Kasım için önerilen yeni deneysel gönderim.** Kütle payı modelinin daha önce sınadığımız sürümünü 24 toprakla eğitip 10 test toprağına uygulayalım. Aynı görüntü özelliklerini, PLS ayarlarını ve eğri düzeltmesini koruyalım. Şablon ve geçerli eğri kontrollerinden sonra public sonucu aynı hesaptaki saf v3 ile karşılaştıralım. Gerekçe 17/24 toprakta iyileşme ve 37,218 yerel EMD. H493’ün puanı çıkarıldığında sıralama tersine döndü, Motorola → Samsung yönünde hata arttı. Bu adayın public puanı henüz bilinmiyor.
3. **29 Kasım için final seçimi.** Saf v3’ü ilk aday olarak tutalım. İkinci yer için kütle payı modelini ve daha önce yüklenen %50–%50 v3/KNN karışımını kendi public puanları, iki telefon sınavı ve numune bazında kazançlarıyla birlikte değerlendirelim. Yeni bir deney yapılırsa yanıtlayacağı soruyu public sonucu görmeden kaydedelim.

### 21 Kasım · Aktif model geliştirme planı

**İlk soru.** Saf v3 her fotoğrafta aynı büyüklükteki toprak yüzeyine bakarsa, yeni toprağın laboratuvar eğrisini daha iyi tahmin eder mi? Bugünkü v3 fotoğrafın her kenarının merkezdeki %60’ını kullanıyor ve görüntüyü 4 piksel/mm’ye getiriyor. Ortak 21 toprağın çekimlerinde bu kadrajın kısa kenarı Motorola Edge’de 94 mm, Samsung A52’de 158,25 mm. Önceki 100 mm deneyi üç özelliği değiştirdi ve iki komşu yöntemini sınadı. Bazı özelliklerin telefonlar arasındaki farkı küçüldü, iki telefon yönünde hata aynı yönde değişmedi. Saf v3’ün kullandığı 30 özelliği birlikte aynı fiziksel alanda hesaplama sorusu açık. [Önceki kadraj kontrolü](esit_kadraj_kontrolu.py)

1. **Aynı eğitim düzeninde iki kadrajı karşılaştır.** Kontrol kolu saf v3’ün kayıtlı 38,758 EMD’sini yeniden üretir. Deney kolu görüntünün merkezinden doğrudan 100 × 100 mm alır ve 400 × 400 piksele dönüştürür. Her iki kolda aynı fotoğraflar, özellik tanımları, eğitim görüntüsü türetme ayarları ve tohumları, iki PLS bileşeni ve tane çapı yüzdelikleri kullanılır. Bu deney kadraj kuralının toplam etkisini ölçer. Karenin biçimi ve görülen yüzey alanı birlikte değişir. Daha dar bir kadraj bazı iri taneleri dışarıda bırakabilir.
2. **Kararı sonuçtan önce tanımla.** Her turda bir toprağın tüm fotoğraflarını ve türevlerini ayır. Birincil ölçü 24 toprağın tüm eğri üzerindeki ortalama EMD’si. Saf v3’ün Motorola Edge → Samsung A52 referansı 41,987, ters yön referansı 42,831 (21 ortak toprak). Yeni kadraj için ortalama ve ortanca eşleşmiş kazancı, kaç toprağın iyileştiğini ve her seferinde bir toprağın puanı çıkarılınca ortalama kazancın aralığını raporla. 0,063 / 0,2 / 0,63 mm hatalarını ve iri uçtaki 20 mm hatasını ayrıca göster. Ana modele geçiş için öneri: 24 toprakta kazanç tek bir puana bağlı kalmasın ve iki telefon yönünde de ortalama EMD artmasın. Tek yönde yarar görülürse saf v3 referans kalır. Android sınavları iPhone’daki başarıyı doğrudan ölçmez.
3. **Sonraki deneyin gereğini çekimlerden belirle.** İlk karşılaştırmada seçilen kadrajı sabitle. Her saklı numunenin özgün çekimlerinden birini sırayla çıkardığında v3 tahmininin ne kadar değiştiğini ölç. Tek bir çekime duyarlılık birkaç toprakta tekrarlanırsa v3’ün fotoğraf özelliklerini numune düzeyinde ortalama veya ortancayla birleştirmesini karşılaştır. Önce her özgün fotoğrafın eğitim türevlerini kendi içinde ortala, sonra fotoğrafları birleştir. Bulgular kararlıysa yeni özelliği veya modeli denemeden önce etiketli, bağımsız topraklarla sınama imkanını araştır. Her adımda sonraki adayın hangi hatayı azaltması gerektiğini kaydet.

Çekim duyarlılığı için sonucu görmeden bir çalışma eşiği belirledik. Bir özgün çekimi çıkarmak, saklı numunenin tam tahminiyle eksik çekimli tahmini arasında en az 10 EMD fark yaratıyorsa bunu kayda değer bir değişim sayacağız. Bu durum en az üç ayrı toprakta görülürse ortalama ile ortancayı sınayacağız. 10 EMD, saf v3’ün yerel hatasının yaklaşık dörtte biri. Bu eşik, kontrolün çalıştırılmasına karar vermek için kullandığımız bir tercihtir.

### 21 Kasım · Sabit kadraj ve çekim duyarlılığı sonuçları

Saf v3 ile arşivdeki 66 eğrinin 11’er tahminini ve EMD’lerini yeniden ürettik. 100 × 100 mm adayında 127 özgün fotoğrafın ve her biri için aynı sekiz eğitim türevinin 30 özelliğini hesapladık (1.143 satır). Aynı fotoğraflar, fiziksel çözünürlük, özellikler, toprak ayrımları, hedef dönüşümü ve iki PLS modeli kullanıldı. Her saklı toprak yalnızca özgün fotoğraflarıyla sınandı.

| Yerel sınama | Saf v3 | 100 mm kadraj | Özgün kadraj, ortanca | 100 mm ile iyileşen toprak |
|---|---:|---:|---:|---:|
| 24 toprak, tüm telefonlar | **38,758** | 40,243 | 38,794 | 12/24 |
| Motorola Edge → Samsung A52 | 41,987 | **40,955** | 42,136 | 11/21 |
| Samsung A52 → Motorola Edge | 42,831 | **41,932** | 43,474 | 11/21 |

Sabit kadrajın saf v3’e göre ortalama kazancı 24 toprakta −1,486, ortanca kazancı −0,642 EMD oldu. Her seferinde bir toprağın puanını çıkardığımızda ortalama kazanç −1,998 ile −1,086 arasında kaldı. Buna karşılık iki telefon yönünde ortalama kazanç +1,032 ve +0,898 EMD. Bu kazançlar da tek bir puan çıkarıldığında yönünü korudu. F827’de sabit kadraj 10,289 EMD kazandırdı, H374’te 10,681 ve H038’de 9,883 EMD kaybettirdi.

| 24 toprakta ortalama mutlak hata (yüzde puan) | Saf v3 | 100 mm kadraj | Özgün kadraj, ortanca |
|---|---:|---:|---:|
| 0,063 mm | 13,613 | 14,237 | 13,400 |
| 0,2 mm | 11,718 | 12,011 | 11,552 |
| 0,63 mm | 9,891 | 10,308 | 10,133 |
| 20 mm | 6,762 | 6,810 | 6,701 |

**Tek çekime duyarlılık.** Seçilen saf v3 kadrajında, diğer 23 toprakla eğitilmiş aynı modele saklı toprağın sırasıyla bir fotoğrafı eksik çekimlerini verdik. H637, H666, G190, H367 ve H126’da en az bir çekim tahmini 10 EMD’den fazla oynattı. En büyük değişim H637’de 15,502 EMD oldu. O çekim çıkarılınca laboratuvara göre hata 32,675’ten 33,276’ya yükseldi. H367’de ise en etkili çekim çıkarılınca hata 20,445’ten 10,219’a indi. Bir çekimin tahmini ne kadar değiştirdiği ve doğruluğu nasıl etkilediği ayrı ölçüldü.

Beş toprakta eşik aşıldığı için özgün kadrajda ortanca ile birleştirmeyi de sınadık. 13/24 toprakta hata düştü ve ortanca kazanç +1,018 EMD oldu. Buna karşılık ortalama hata 0,036 EMD arttı. İki telefon yönünde de ortalama hata arttı. Bu küçük fark, ortancaya geçmek için gerekçe sağlamıyor.

**Karar.** Saf v3’ün özgün kadrajı ve fotoğraf özelliklerinin aritmetik ortalaması ana model olarak kaldı. 100 mm kadraj, iki Android telefonu arasında aktarımda gözlenen iyileşmeyle kayda geçti. Etiketli iPhone örneğiyle doğrulama değerli bir sonraki sınama olur. [Kod](v3_sabit_kadraj.py) · [198 saklı tahmin eğrisi](v3_kadraj_sonuclari.csv) · [127 çekim çıkarma sonucu](v3_cekim_duyarliligi.csv) · [Sabit kadraj görüntü ölçüleri](v3_100mm_fotograf_olculeri.csv)

### 21 Kasım · İki yeni public puandan sonra geliştirme planı

**Referanslar.** Kendi en iyi public sonucumuz özgün kadraj + ortanca ile 41,67360. Sabit 100 mm kadraj 44,03121 aldı. Ana model saf v3’ün yerel referansı 38,758, yazarın bildirdiği public referans 40,90.

**Ön kontrol.** 24 topraklık sınavın mevcut tahminlerini yalnızca iki telefonla da çekilmiş 21 toprakta puanladık. Saf v3 39,207, 100 mm kadraj 40,214 EMD verdi. Eğitim hâlâ her turda diğer 23 topraktan tüm telefonları kullanıyor. Böylece 24 topraklık sınavla iki çapraz telefon sınavı arasındaki sıralama değişimini yalnızca üç numunenin kapsam dışında kalması açıklamıyor. [Saklı tahminler](v3_kadraj_sonuclari.csv)

1. **İlk iş, telefonu ve eğitim grubunu kontrol etmek.** Aynı 21 toprağı sırayla ayır. Diğer 20 toprağın Motorola çekimleriyle her kadraj için bir model eğit ve aynı saklı toprağın hem Motorola hem Samsung çekiminde sına. Ardından Samsung çekimleriyle eğitip iki sınamayı tekrarla. Mevcut 30 görüntü ölçüsü bu dört koşula yeter. Her koşulda iki kadrajın laboratuvara göre EMD farkını ve kaç toprakta iyileşme olduğunu kaydet. Aynı eğitilmiş modelin iki telefondan verdiği tahminlerin birbirinden ne kadar uzaklaştığını da ölç. Işık ve görülen yüzey bölgesi çekimler arasında değişebildiğinden sonuç bütün çekim koşullarını kapsar.
2. **Bulgular desteklerse tek yeni model dene.** Sabit kadrajda her iki eğitim yönünde hem çapraz telefon hatası hem telefonlar arası tahmin farkı azalırsa, fiziksel ölçeği koruyup yüzeyden daha fazla bölge örneklemeyi sına. Tek 100 mm merkez kesitine karşı merkez ve dört çapraz çevre kesitini karşılaştır (yatay ve düşey yönde 20’şer mm kaydırma). Her kesitte aynı 30 v3 özelliğini hesapla, beş özellik vektörünü eşit ağırlıkla ortala. Tek kesit doğrudan kontrol, saf v3 ana referans. Bu kol için sonuç henüz yok.
3. **Karar.** Adayı saf v3 ile aynı 24 toprakta ve iki çapraz telefon yönünde karşılaştır. 24 topraktaki ortalama kazanç her seferinde bir puan çıkarıldığında korunuyor ve iki telefon yönünde ortalama EMD artmıyorsa yeni bir Kaggle gönderimi için gerekçe oluşur. İlk kontrolde beklenen ilişki çıkmazsa bağımsız, laboratuvar etiketli iPhone topraklarıyla sınama olanağını araştır.

### 21 Kasım · Aynı toprakta iki telefon ve iki kadraj

GitHub’daki ana dal ile yerel son commit `d4d23e2` üzerinde eşleşti. v3 kodu, deponun işaret ettiği `0808d05` sürümündeydi. Son iki gönderimin CSV’leri, bunları üreten betik ve ekrandan doğrulanan 41,67360 / 44,03121 public puanları yerel çalışma alanındaydı.

**Deney.** Her turda iki telefonla da çekilmiş 21 topraktan biri saklandı. Kalan **aynı 20 toprağın** yalnızca kaynak telefondaki örnekleriyle model eğitildi. Aynı eğitilmiş modele saklı toprağın Motorola ve Samsung özgün çekimleri ayrı ayrı verildi. Her iki kadrajda aynı eğitim türevleri ve PLS kullanıldı. Böylece her karşılaştırmada hem eğitim toprağı sayısı hem kimlikleri eşleşti. Önceki çapraz sınavların 84 tahmin eğrisinin ve EMD değerinin tamamı yeniden üretildi.

| Eğitim → sınama | Saf v3 EMD | 100 mm EMD | 100 mm ile iyileşen toprak |
|---|---:|---:|---:|
| Motorola → Motorola | 37,696 | 38,735 | 9/21 |
| Motorola → Samsung | 41,987 | 40,955 | 11/21 |
| Samsung → Motorola | 42,831 | 41,932 | 11/21 |
| Samsung → Samsung | 39,561 | 40,055 | 8/21 |

| Eğitim telefonu | İki telefon tahmininin uzaklığı, saf v3 | Aynı uzaklık, 100 mm | Uzaklığı azalan toprak |
|---|---:|---:|---:|
| Motorola | 29,259 | 24,964 | 17/21 |
| Samsung | 23,728 | 19,305 | 15/21 |

İkinci tablodaki EMD aynı toprağın iki çekim grubundan çıkan eğrilerin birbirine uzaklığıdır. İlk tablo ise her tahmini laboratuvarla karşılaştırır. Fiziksel kadraj sabitlenince iki telefonun tahminleri iki eğitim yönünde de yaklaştı. Çapraz hata da her iki yönde azaldığından plandaki beş kesit deneyi için koşul sağlandı. Eğitim ve sınama telefonu aynı olduğunda özgün kadrajın ortalama hatası daha düşüktü. Bu ayrım telefon donanımının, ışığın ve görüntülenen yüzey bölgesinin etkilerini birbirinden ayırmıyor. [Eşleştirilmiş kontrol kodu](v3_iki_telefon_kadraj.py) · [168 tahmin eğrisi ve puanı](v3_iki_telefon_kadraj_sonuclari.csv)

### 21 Kasım · Tek ve beş sabit kesitte v3 özellikleri

127 eğitim fotoğrafının her birinde 100 × 100 mm merkez kesitini koruduk. Yatayda ve düşeyde 20’şer mm kaydırılmış dört çapraz kesit ekledik. Her birinde aynı 30 fiziksel görüntü özelliğini hesapladık. Beş özellik vektörünün eşit ağırlıklı ortalamasını her fotoğraf ve türevi için ayrı aldık. Kesitler örtüşür. Her fotoğrafın sekiz eğitim türevi, aynı tohumlar ve PLS ayarlarıyla işlendi. 1.143 satırın her birindeki merkez katkısı önceki tek kesit önbelleğinden geldi. Bir özgün Motorola ve bir türetilmiş Samsung görüntüsünde ayrı kesitlerden yeniden kurulan ortalama, kaydedilmiş 30 özelliğin tamamıyla eşleşti. Saklı örneklerin tamamında yalnız özgün fotoğraflar kullanıldı.

| Yerel EMD | Saf v3 | Tek kesit (100 mm) | Beş kesit | Beş kesitin tek kesiti geçtiği toprak |
|---|---:|---:|---:|---:|
| 24 toprak, tüm telefonlar | **38,758** | 40,243 | 40,419 | 12/24 |
| Motorola → Samsung | 41,987 | 40,955 | **40,860** | 12/21 |
| Samsung → Motorola | 42,831 | **41,932** | 42,683 | 7/21 |

24 toprakta beş kesitin tek kesite göre ortalama kaybı 0,176 EMD, ortanca kaybı 0,136 EMD. Sırayla bir puanı çıkarınca beş kesitin ortalama kazancı −0,366 ile +0,026 arasında değişti. Saf v3’e karşı karşılaştırma daha belirgin: Beş kesit yalnızca 9/24 toprakta kazandı, ortalama hatası 1,662 EMD daha yüksek kaldı. Her seferinde bir puan çıkarılınca saf v3 lehine fark 1,273–2,204 EMD aralığında kaldı. H037 ve H038 birlikte dışarıda bırakıldığında da saf v3 lehine fark 0,963 EMD idi.

| 24 toprakta elek bazında ortalama mutlak hata (yüzde puan) | Saf v3 | Tek kesit | Beş kesit |
|---|---:|---:|---:|
| 0,063 mm | 13,613 | 14,237 | 14,320 |
| 0,2 mm | 11,718 | 12,011 | 11,929 |
| 0,63 mm | 9,891 | 10,308 | 10,162 |
| 20 mm | 6,762 | 6,810 | 6,890 |

**Karar.** Sabit fiziksel kadraj iki Android telefonun tahminlerini birbirine yaklaştırıyor. Bu kadraja çevreden dört kesit eklemek ana sınavda kazanç sağlamadı. Saf v3 ana model olarak kaldı. Beş kesitli sürüm Kaggle adaylık koşulunu karşılamadı. Bu düzenle kesit sayısı denemesini kapattık. Bağımsız, laboratuvar eğrisi bilinen iPhone çekimleri edinilirse aynı modelleri yeni topraklarda sınamak, telefon aktarımı için doğrudan kanıt sağlayabilir. [Deney kodu](v3_bes_kesit.py) · [66 saklı tahmin ve EMD](v3_bes_kesit_sonuclari.csv) · [Beş kesitli görüntü özellikleri](v3_bes_kesit_fotograf_olculeri.csv)

### 21 Kasım · Sıralı elek modeli için deney planı

**Hipotez.** Aynı 30 v3 görüntü özelliğinden toprağın incelik–irilik düzeyini ve dağılımın yayılımını ayrı öğrenmek, saklı toprakların bütün eğrisindeki EMD’yi azaltabilir. Liu ve arkadaşları birikimli olasılık modelleriyle koşullu dağılımın doğrudan tahminini, Cox ise sıralı veride konum ve ölçek etkilerini tanımlar. Bu çalışmalardan aldığımız fikir modelin biçimidir. Elekten geçen kütle yüzdelerine ve yarışmanın hata ölçüsüne uyarlamayı burada sınayacağız. [Liu ve arkadaşları](https://pubmed.ncbi.nlm.nih.gov/28872693/) · [Cox](https://pubmed.ncbi.nlm.nih.gov/7667560/)

- **Ana aday ve kontrol.** On sıralı elek için ortak artan eşikler öğren. Her toprağın görüntü özellikleri eşiklere göre konumu belirler. Ana adayda görüntü özellikleri yayılımı da belirler. Kontrolde yayılım sabittir. Son elek her zaman %100’dür. İki sürüm aynı kaybı, fotoğrafları ve eğitim türevlerini kullanır. Ana referans saf v3’tür.
- **Öğrenme.** Toprak başına özellikleri v3 gibi ortala. Eğitim topraklarında standartlaştır, ±3,5’te sınırla ve dört standartlaştırılmış temel bileşene indir. Yarışmadaki log-çap ağırlıklarıyla mutlak hatanın bir yüzde puanlık yumuşatılmış karşılığını kullan. Her toprak eşit ağırlıklı. Görüntü katsayılarının küçültme cezasını 0,0003 / 0,003 / 0,03 arasından, yalnız dış eğitim topraklarında dört parçalı iç doğrulamayla seç (sabit tohum 42). İç sınamada özgün çekimleri kullan. Telefon sınavlarında iç ayarı kaynak telefonla yap.
- **Karar.** 24 toprakta ve iki yönde 21 ortak toprakta tam eğri EMD’sini ölç. Kazancı, ortancayı, kazanan toprak sayısını ve her seferinde bir puan çıkarıldığında ortalama kazancın aralığını ver. Üç ince elek, H037/H038 ve H493 ayrı tanı amaçlıdır. Ana adayın 24 topraktaki kazancı her tek-puan çıkarımında korunur ve iki çapraz telefon yönünde EMD artmazsa tüm 24 toprakla test tahmini üret. Karşılaştırma yereldir.
- **Eğri biçiminin sınırı.** Bayat ve arkadaşları 160 toprağın ölçülmüş dağılımlarına 36 eğri biçimi uydurduğunda yöntemlerin sıralamasının tane aralığına göre değiştiğini buldu. Biz de fotoğraftan tahmin performansı ile eğrinin ölçülmüş dağılıma uyabilme kapasitesini ayıracağız. Gerçek tahminler kaydedildikten sonra her saklı toprağın laboratuvar eğrisine bakarak yalnız o toprağın konumunu ve yayılımını uyduracağız. Eğitimde öğrenilmiş eşikler sabit kalacak. Bu uyum hatası sadece tanı amaçlıdır ve aday seçimine girmez. [Bayat ve arkadaşları](https://doi.org/10.1111/ejss.12423)

### Sıralı elek modeli için ikinci, keşif amaçlı karşılaştırma

İlk çalıştırmada dört PCA bileşeninden konum ve yayılım tahmini 47,561 EMD verdi. Aynı 24 eğrinin eşiklerini koruyup iki parametreyi saklı laboratuvar eğrisine göre ayarladığımızdaki uyum hatası 7,008 idi. Bu fark, görüntülerden iki parametrenin tahminini incelemeyi gerekçelendiriyor. İkinci aşamayı bu bulgudan sonra, yeni puanları görmeden tanımladık.

V3’ün aynı iki özellik grubunun her birinden tek görüntü bileşeni çıkaracağız. Kontrolde PCA, adayda laboratuvar eğrisinin log-çap yüzdelikleriyle eğitilmiş tek bileşenli PLS kullanılacak. İki giriş de kendi eğitim tarafında standartlaştırılacak. Her iki modele de **aynı iki girdili konum–yayılım yapısı**, iç doğrulama ve ceza ızgarası uygulanacak. PLS iç ve dış doğrulamada her seferinde yalnız eğitim numunelerinin etiketlerini görecek. Bu kez de saf v3’e karşı aynı adaylık ölçütü geçerli. İkinci aşama aynı 24 toprakta yapılan keşif amaçlı bir deney. [V3’ün referans uygulaması](../work/vitalclick/src/run.py)

### Sıralı elek modeli ve özellik seçimi sonuçları

Saf v3 için eski 66 tahmin eğrisinin ve EMD puanının tamamı tekrar üretildi. Aşağıdaki yeni sürümlerin her biri için saklı toprağın hiçbir fotoğrafı, türevi veya laboratuvar eğrisi model kurulumuna ve ceza seçimine girmedi. İç sınamada özgün çekimlerle gerçek EMD hesaplandı. Sıralı modelin bütün yeni 264 tahmininde 11 elek yüzdesi, monotonluk, son değerin %100 olması ve EMD bağımsız olarak denetlendi.

| Yerel model | 24 toprak | Motorola → Samsung | Samsung → Motorola |
|---|---:|---:|---:|
| Saf v3 | **38,758** | **41,987** | **42,831** |
| Dört PCA bileşeni, yalnız konum | 45,529 | 55,798 | 66,412 |
| Dört PCA bileşeni, konum ve yayılım | 47,561 | 63,141 | 71,279 |
| İki özellik grubu, her grupta PCA, konum ve yayılım | 41,424 | 43,764 | 48,598 |
| İki özellik grubu, her grupta PLS, konum ve yayılım | 41,784 | 43,876 | 49,883 |

Birinci aşamada yayılımı görüntüden öğrenmek genel hatayı artırdı. Ana aday saf v3’ü 8/24 toprakta geçti. Laboratuvar eğrisini iki parametreye uydurarak yapılan ayrı tanıda ortalama 7,008, ortanca 5,900 EMD elde edildi. Bu sayı, gerçek dağılımı gördüğümüzde öğrenilmiş eşiklerle hangi uyumun mümkün olduğunu gösterir. Fotoğraftan elde edilebilecek puanı veya hatanın tek nedenini vermez.

İkinci aşamada iki grubun PLS özeti, eşlenmiş PCA kontrolünü yalnızca 8/24 toprakta geçti. Saf v3’e göre 9/24 toprakta daha yakındı. Saf v3 eksi aday olarak ortalama kazanç −3,026, ortanca kazanç −1,771 EMD. Her seferinde bir puan çıkarıldığında ortalama kazanç −4,040 ile −0,043 arasında kaldı. H493’te saf v3 56,24, PLS adayı 127,89 EMD aldı. H493’ün puanı çıkarıldığında ortalamalar 37,998 ve 38,040’a yaklaştı. H037 ve H038 birlikte çıkarıldığında saf v3 32,685, aday 36,177 EMD verdi. İki telefon sınavında da saf v3 öndeydi.

| 24 toprakta ortalama mutlak hata (yüzde puan) | 0,063 mm | 0,2 mm | 0,63 mm | 20 mm |
|---|---:|---:|---:|---:|
| Saf v3 | 13,613 | 11,718 | 9,891 | 6,762 |
| Dört PCA bileşeni, konum ve yayılım | 16,106 | 12,714 | 11,804 | 5,140 |
| İki PLS girdisi, konum ve yayılım | 15,084 | 13,077 | 11,436 | 4,934 |

**Yerel karar.** 20 mm’deki daha düşük hata genel eğriye ve üç ince eleğe taşınmadı. Saf v3 ana model olarak korundu. İki yeni aday da Kaggle gönderimi için önceden tanımlanan koşulu karşılamadı. [Kod](sirali_elek_modeli.py) · [Birinci aşamanın tahminleri](sirali_elek_sonuclari.csv) · [İki grup karşılaştırması](sirali_elek_iki_grup_sonuclari.csv) · [Eğri biçimi tanısı](sirali_elek_bicim_tanisi.csv)

### Montréal dış sınaması için plan

İki yerel aşamadan sonra modelleri dondurduk. FRDR'nin Photogranulometry v1.0 veri kümesinde 321 numune var. Resmî dosya açıklaması, hidrometre eğrilerini de içeren CSV’de 77 numune bildiriyor. Dış puanları görmeden belirlenen pilot yalnız bu CSV’yi kullanır. Kuru çekimi bulunan doğal bulk veya split-spoon topraklarından, geçerli artan eğriye ve 80 mm’de %100 geçişe sahip olanları sırala. Numune kimliklerinden sabit 42 tohumu ile sekizini tekrarsız seç. Her birinin dosya adı sırasındaki ilk kuru fotoğrafını kullan. Elek yüzdelerini yarışma çaplarına logaritmik çapta aradeğerle, 200 mm’yi %100 al. Canon fotoğraflarında yayındaki nominal 39,4 µm/piksel ile özgün v3 kadrajı uygulanacak. Numune kalınlığı gerçek ölçeği etkileyebilir.

Yalnız Kaggle’ın 24 toprağıyla eğitilen saf v3, ikinci aşamadaki iki PLS girdili konum–yayılım modeli ve fotoğrafsız eğitim eğrisi ortalaması aynı sekiz numunede sınanacak. Dış etiketler model eğitimine ve ayar seçimine girmeyecek. Bu ayrı cihazda, kuru laboratuvar görüntüleriyle, toprak başına tek çekim kullanılan bir aktarım sınavı. [Araştırmacıların veri ve çekim tanımı](https://arxiv.org/pdf/2506.17469) · [FRDR veri kaydı](https://doi.org/10.20383/103.01316)

### Montréal pilotunun sonucu ve tek çekim kontrolü

77 numuneden 69’u seçim ölçütlerini karşıladı. Seçilen sekiz toprak 0048, 0143, 0181, 0243, 0255, 0274, 0285 ve 0288 oldu. Her birinden bir Canon fotoğrafı işlendi. Aşağıdaki EMD’ler tüm eğri içindir. Her toprak eşit ağırlıklıdır.

| Model | Kaggle, 24 saklı toprak, tüm çekimler | Kaggle, 24 saklı toprak, tek çekim | Montréal, 8 yeni toprak, tek çekim |
|---|---:|---:|---:|
| Saf v3 | **38,758** | **43,013** | 86,274 |
| İki PLS girdisiyle konum ve yayılım | 41,784 | Ölçülmedi | 90,562 |
| Fotoğrafsız eğitim ortalaması | 89,889 | 89,889 | **71,291** |

Montréal’de iki model de yalnız 0288’de eğitim ortalamasını geçti. Sırayla bir toprağı çıkardığımız sekiz ortalamanın tamamında fotoğrafsız yöntem saf v3’ten daha iyi kaldı (7,899–23,877 EMD fark). Saf v3 sekiz toprağın altısında sıralı modeli geçti. 0,2 mm eleğinden geçen kütle payını saf v3 sekiz toprağın altısında eksik tahmin etti. Sekiz toprak üzerindeki işaretli ortalama fark −24,16 yüzde puan, ortalama mutlak hata 31,72 yüzde puandı. Fotoğrafsız ortalamanın aynı elekteki mutlak hatası 21,31 yüzde puandı. 20 mm’de ise saf v3 ortalama eğriden daha yakındı (4,18 ve 6,22 yüzde puan).

Pilotun sonucunu gördükten sonra tek fotoğraf kullanımının etkisini Kaggle eğitiminde ölçtük. 24 turun her birinde bir toprağın bütün çekimlerini ayırıp kalan 23 toprakla saf v3’ü eğittik. Saklı toprağın özgün fotoğraflarını önce ayrı ayrı, sonra birlikte değerlendirdik. Tek çekim sütununda 127 tekil fotoğrafın EMD’lerini önce kendi toprağında, sonra 24 toprakta ortaladık. Tüm çekimlerde 24 eski tahmin eğrisi tekrar elde edildi. Her turdaki fotoğrafsız ortalama yalnız 23 eğitim etiketini kullandı. Montréal için aynı kontrol Kaggle’ın 24 eğitim etiketini kullandı.

Kendi verimizde tek çekim saf v3’e ortalama 4,255 EMD kaybettirdi, ancak bu durumda dahi 22/24 toprakta fotoğrafsız ortalamayı geçti. 0,2 mm’de tek çekimle işaretli ortalama fark yalnızca +1,02 yüzde puandı. Montréal ve Kaggle’daki sıralama tersliğini açıklamak için yalnız çekim sayısına bakmak yeterli görünmüyor. Sekiz örnekten kamera, toprak bileşimi, yüzey hazırlığı ve ölçüm protokolünün etkilerini tek tek ayıramıyoruz. Bu pilot Kaggle’daki iPhone puanını ölçmüyor. [Pilot kodu](montreal_dis_sinama.py) · [Dış tahminler](montreal_pilot_sonuclari.csv) · [Eşlenmiş referans eğrileri](montreal_pilot_referans.csv) · [Fotoğraf özellikleri ve dosya kimlikleri](montreal_pilot_olculeri.csv) · [127 tekil çekimin tahmini](montreal_tek_cekim_yerel_kontrol.csv)

**Sıradaki geliştirme planı.** Tek değişken olarak eğitimdeki toprak çeşitliliğini sınayacağız. Pilotun sekiz kimliğini dışarıda tutup kalan uygun Montréal topraklarından sonuçlarına bakmadan sabit bir grup seçeceğiz. Aynı v3 özellikleri ve ayarlarıyla, her Kaggle turunda yalnız o turun eğitim topraklarını kullanan sürümü, bunlara dış toprakları da ekleyen sürümle karşılaştıracağız. Aynı sınav topraklarında fotoğrafsız referansı, tüm eğrinin EMD’sini, üç ince eleğin hatalarını ve iki telefon yönünü raporlayacağız. Dış numunelerle birlikte eğitimin faydasını Kaggle’ın 24 toprağındaki eşleştirilmiş hatalar belirleyecek. Sekiz pilotun sonuçlarını gördüğümüz için yeni bir dış doğrulama iddiası ayrıca hiç açılmamış topraklar gerektirir.

### Montréal topraklarını eğitime ekleme deneyi · sonuçlardan önce

Sekiz pilot kimliğini ayırdıktan sonra uygun 61 topraktan 12’sini, sıralı kimlikler üzerinde 2026 tohumuyla tekrarsız seçeceğiz. 12, Kaggle’daki 24 toprağın yarısıdır. Dış verinin katkısını tek grup büyüklüğüyle sınayacağız. Her numuneden dosya adı sırasındaki ilk kuru fotoğraf, yayımlanmış fiziksel kalibrasyon ve v3’teki özgün + sekiz eğitim türevi kullanılacak. Her toprak modele eşit ağırlıkla girecek. Dış eğriler önceki pilotla aynı log-çap aradeğerlemesiyle 11 eleğe dönüştürülecek. Fotoğrafsız kontrolü hem Kaggle eğitim eğrilerinin hem birleşik eğitim eğrilerinin ortalamasıyla hesaplayacağız.

Ana sınavda her seferinde 23 Kaggle toprağı ile aynı 23 + 12 Montréal toprağı karşılaştırılacak. Telefon sınavlarında iki tarafta da aynı 21 ortak toprak kullanılacak; saklı toprağı çıkardıktan sonra eğitimde yalnız kaynak telefonun özgün çekimleri ve türevleri, sorguda hedef telefonun özgün çekimleri yer alacak. Montréal kolunda aynı 12 Canon toprağı her dış eğitime eklenecek. Birincil ölçüt 24 eşleştirilmiş toprağın tam eğri EMD’si. Model adaylığı için tek tek her puan çıkarıldığında ana kazancın korunması ve iki telefon yönünde de ortalama hatanın artmaması şeklindeki önceki kural geçerli. 0,063, 0,2 ve 0,63 mm için mutlak hataları, kazanan toprak sayısını, H037/H038 ve H493 duyarlılığını ayrıca kaydedeceğiz.

### Montréal topraklarını eğitime ekleme deneyinin sonucu

**Karar.** Bu 12 dış toprağa eşit ağırlık vermek 24 saklı Kaggle toprağında EMD’yi 38,758’den 53,899’a yükseltti. Saf v3 ana model olarak kaldı. Bu deneyden Kaggle gönderimi çıkmadı.

| Saklı Kaggle sınavı · ortalama EMD | Saf v3 | V3 + Montréal | Dış veriyle iyileşen toprak | Fotoğrafsız Kaggle ortalaması | Fotoğrafsız birleşik ortalama |
|---|---:|---:|---:|---:|---:|
| 24 toprak, tüm çekimler | **38,758** | 53,899 | 7/24 | 89,889 | 92,268 |
| Motorola → Samsung | **41,987** | 57,410 | 7/21 | 90,838 | 90,426 |
| Samsung → Motorola | **42,831** | 56,795 | 7/21 | 90,838 | 90,426 |

Ana sınavda ortanca toprakta EMD 8,549 arttı. Sırayla bir toprak puanı çıkarıldığında ortalama kayıp 13,564–16,064 EMD aralığında kaldı. H493 çıkarılınca kayıp 15,918, H037 ve H038 birlikte çıkarılınca 15,933 EMD oldu. En büyük kayıplar H368 (51,42), H367 (50,21) ve H366’da (45,61) görüldü. H405’te 6,07 EMD kazanıldı. Her iki telefon yönünde fotoğrafsız ortalama 0,412 EMD iyileşirken, görüntülü modelin hatası arttı.

| İnce eleklerde ortalama mutlak hata (yüzde puan) | 0,063 mm | 0,2 mm | 0,63 mm |
|---|---:|---:|---:|
| 24 toprak · saf v3 | 13,613 | 11,718 | 9,891 |
| 24 toprak · v3 + Montréal | 15,312 | 15,717 | 17,878 |
| Motorola → Samsung · saf v3 | 16,480 | 18,057 | 12,572 |
| Motorola → Samsung · v3 + Montréal | 16,751 | 19,727 | 20,711 |
| Samsung → Motorola · saf v3 | 17,018 | 12,726 | 9,930 |
| Samsung → Motorola · v3 + Montréal | 17,729 | 16,483 | 17,705 |

**Seçilen toprakların bileşimi.** 0,63 mm’den geçen ortalama kütle payı Montréal grubunda %86,02, Kaggle’daki 24 toprakta %55,86 idi. Aynı elekte saklı Kaggle tahminlerinin ortalama işaretli hatası +4,64’ten +12,91 yüzde puana çıktı. 0,2 mm’de de seçilen grupların laboratuvar ortalamaları %73,17 ve %45,75 idi. Bu gözlem, dış grubun ince malzeme ağırlıklı olmasının etkili olabileceğini düşündürüyor. Fotoğraflar da farklı bir cihaz ve protokolle çekildiğinden bu karşılaştırma iki etkiyi birbirinden ayırmıyor.

Sonuçları gördükten sonra kaybı 0,63 mm eleğine göre ayırdık. Montréal grubunda kütlesinin yarıdan azı bu elekten geçen toprak yoktu (0/12). Kaggle’da bu tanıma uyan 13 toprağın tamamında hata arttı. Bu grupta ortalama EMD 39,984’ten 67,339’a çıktı. Diğer 11 toprakta ortalamalar 37,308 ve 38,016 idi ve 7’sinde dış veri hatayı azalttı. Bu alt gruplar açıklayıcıdır. Bir sonraki dış veri deneyi için, laboratuvar ölçümlerine göre bu iri grubu da temsil eden eğitim numuneleri aranabilir.

**Yeniden üretim.** Seçilen 12 numunenin kimlikleri 0025, 0045, 0066, 0112, 0119, 0125, 0137, 0165, 0179, 0240, 0247 ve 0249’dur. Kaynak tablodan dosya eşleşmeleri ile 11 elek değeri ayrı bir hesapla doğrulandı. 12 fotoğrafın özgün ve sekiz türeviyle toplam 108 özellik satırı üretildi. Bir Kaggle çekiminde dokuz sürümün 30 özelliği de arşivle eşleşti. 66 saf v3 eğrisi önceki deneyi yeniden üretti. Dört yöntem için toplam 264 eğri ve bunların EMD’leri numune bazında bağımsız olarak yeniden hesaplandı. `python outputs/montreal_egitim_karsilastirmasi.py` çalışma klasöründe kayıtlı seçim ve özelliklerle çevrimdışı tekrar çalışır. [Deney kodu](montreal_egitim_karsilastirmasi.py) · [Numuneler ve laboratuvar eğrileri](montreal_egitim_secimi.csv) · [Görüntü özellikleri](montreal_egitim_olculeri.csv) · [264 tahmin ve EMD](montreal_egitim_sonuclari.csv) · [FRDR veri kaydı](https://doi.org/10.20383/103.01316)

### Montréal’de kalan iri toprakların sayımı ve karar

FRDR’nin ince tane ölçümlerini içeren resmî tablosunda aynı kuru doğal toprak ve geçerli eğri koşulları uygulandı. Kaynakta 0,63 mm doğrudan ölçülmüş. Bu elekten kütlesinin %50’sinden azı geçen her numuneyi bir kez saydık. Uygun 69 toprağın yalnız 3’ü iri tanımını karşıladı. İkisi önceki 8 topraklık pilotta, hiçbiri 12 topraklık eğitim ekinde değildi. Kalan 49 numunede tek iri toprak 0177 idi (%37,48 geçen). Kaggle eğitiminde aynı tanıma 24 toprağın 13’ü uyuyor. Dolayısıyla bu tablonun kullanılmamış kısmından yarışmadaki ince-iri oranına yakın, 12 farklı topraklık bir ek grup kurulamıyor. Sayım yalnız laboratuvar tablosuna dayanıyor. [FRDR kaynak tablosu](https://www.frdr-dfdr.ca/repo/files/1/published/publication_1311/submitted_data/metadata/psd_mass_path_80_to_0_0018mm.csv)

**Karar.** Bu kaynaktan fotoğraf indirme ve toprak ekleme denemelerini burada kapatıyoruz. Yarışma için eldeki ilk pratik adım, paylaşılan son altı gönderide görünmeyen saf v3’ü kendi hesap kaydımıza almak. Dosya tekrar kontrol edildi; 10 numune ve 11 elek şablonla eşleşiyor, eğriler geçerli ve normalleştirilmiş Git içeriği yazarın v3 kaydıyla aynı. Bu adım puanı kendi hesabımızda ölçer. Sonraki veri toplama çalışması, yarışmadaki saha ve telefon çekimlerine benzeyen ve laboratuvar eğrisi bilinen iri örnekler sağlayabilirse anlamlı olur. [Doğrulanan saf v3 dosyası](../work/v3_puan_kontrol.csv)

### Mevcut fotoğraflarla yeni yöntem arayışı · kaynaklar ve öneri

| Birincil çalışma | Ölçtüğü hedef ve yöntemi | Bizim deneye etkisi |
|---|---|---|
| [Soranzo, Guardiani ve Wu, 2025](https://doi.org/10.1016/j.sandf.2025.101575) | Kontrollü ışıkta yakın plan toprak fotoğraflarından, ImageNet ile önceden eğitilmiş MobileNet kullanarak Weibull eğrisinin iki parametresini tahmin ediyor. | Az sayıda parametreyle tam dağılım üretmek mümkün. İlişkili açık veri kümesindeki 26 kimliğin 24’ü Kaggle eğitimimizle örtüştüğünden, bu kaynağın tamamıyla eğitilmiş bir modeli aynı 24 numunede sınamak bağımsız sonuç vermez. |
| [Plante St-Cyr, 2026, yüksek lisans tezi](https://espace.etsmtl.ca/id/eprint/4065/) | 316 toprağa ait 12.714 görüntüde DINOv2 ile kesit özellikleri çıkarılıyor. Dikkat mekanizması, kesitleri numune düzeyindeki laboratuvar eğrisine göre ağırlıklandırıyor. | Biz beş kesiti eşit ağırlıkla ortaladık. Yerel 48,849 EMD veren o Ridge deneyi, öğrenilmiş bölge ağırlıklarını sınamadı. Yazarın deposunda eğitim kodu var; eğitilmiş dağılım modelinin ağırlıkları paylaşılmıyor. [Kod ve kapsam](https://github.com/thomas-plante-st-cyr/abmil-optical-psd) |
| [Lang ve arkadaşları, GRAINet, 2021](https://hess.copernicus.org/articles/25/2567/2021/) | Altı nehirdeki 25 çakıl barında 1.491 dijital çizgi örneği kullanarak görüntüden tane boyutu dağılımı öğreniyor. Yeni barlara aktarımı ayrı sınıyor. | Yüzeyde çizgi boyunca sayılan çakıllarla elde edilen dağılım, bizim elek ve sedimentasyonla ölçülen toplam kütle eğrimizden farklı bir hedef. |
| [Mair ve arkadaşları, ImageGrains 2.0, 2026](https://esurf.copernicus.org/articles/14/527/2026/) | Jeolojik görüntülerde tek tek görünen tanelerin sınırlarını ve eksen uzunluklarını çıkarıyor. Çalışma, maske ve boyut doğruluğunu elle çizilmiş sınırlarla karşılaştırıyor. | Saf v3’e yeni bir görüntü ölçüsü sağlama olanağı var. Önceki halka/kontrast ölçümüz gerçek tane maskesi üretmiyordu. Yazarlar eğitilmiş ağırlıkları da yayımladı. [Model ağırlıkları](https://doi.org/10.5281/zenodo.15309323) |

**Neden önce ImageGrains pilotu?** Mevcut özgün fotoğrafların kayıtlı etkin ölçeği doğrulandı. 127 eğitim çekiminin 124’ünde 4,55–4,60 piksel/mm var. Bu grupta 0,2 mm yaklaşık 0,91–0,92 piksel, 6,3 mm yaklaşık 29 piksel. Diğer üç çekimde 6,3 mm yaklaşık 79 piksel. Saklı saf v3 eğrilerinde 0,063, 0,2 ve 0,63 mm birlikte ortalama EMD’nin %45,49’unu oluşturuyor. Bu nedenle ince malzemeyi görüntüden tek tek saymak yerine, yeni bilginin görülebilen iri tanelerden gelip gelmediğini sınamak daha anlamlı.

**Önerilen tek ilk test.** Her iki telefonla çekilmiş topraklardan, eski v3 hatalarını kullanmadan ve ince/iri örnekleri kapsayacak şekilde altı toprak seç. Her birinden telefon başına bir özgün çekim kullan. Aynı fiziksel görüş alanındaki taş sınırlarını elle işaretlenmiş küçük alanlarla karşılaştır; özellikle gölgeyi taş sayma ve komşu taşları birleştirme hatalarına bak. Model ölçüsünü baştan sabitle: kısa ekseni en az 6,3 mm olan taş maskelerinin kapladığı alanın incelenen alana oranı. Bu yüzey alanı payı laboratuvar kütle yüzdesinden farklı bir niceliktir. Maskeler her iki telefonda da güvenilirse, saf v3 ile tek ek ölçü kullanan v3’ü aynı 24 saklı toprak ve iki telefon yönünde karşılaştır. Ana ölçüt tam eğri EMD’si; 6,3 mm üzerindeki kütle hatası yardımcı ölçüt. Bu turda araştırma ve piksel ölçeği kontrolü yapıldı; ImageGrains için yerel puan henüz yok.

### ImageGrains pilotu · maskeleri görmeden sabitlenen düzen

Her iki telefonla çekilmiş 21 toprak arasından 0,63 mm eleğinde %50 geçişin iki yanında üçer toprak, sabit 20261121 tohumuyla seçilir. Her toprak ve telefon için dosya adı sırasındaki ilk çekim alınır. 12 çekimde aynı 52 × 52 mm merkez alanı 4,5 piksel/mm ile işlenir; merkezdeki 26 × 26 mm alan değerlendirilir. Çevresi, merkeze uzanan taşların sınırlarını görmek içindir. Fotoğraflar iki cihazda aynı bakış noktasından çekilmiş kabul edilmez.

Pilotun referans çizimlerini asistan, özgün fotoğraflarda model sonucunu ve kimlikleri görmeden yapar. Bu yaklaşık görsel referanslar uzman anotasyonu değildir. Kısa ekseni en az 6,3 mm olan ve değerlendirme alanına en az 1 mm² giren belirgin taşlar sayılır. Karar verilemeyen bölgeler ayrıca işaretlenir. Nesne eşleşmesi için kesişim/birleşim en az 0,5 aranır. İlerleme eşiği her telefon için en az üç değerlendirilebilir referans taşı, en az %70 kesinlik ve %70 duyarlılık, ortalama mutlak alan payı farkında en fazla 15 yüzde puandır. Yeterli referans taşı bulunmazsa sınır kalitesi o telefon için belirsiz kalır. İki telefonda eşikler karşılanırsa tek ölçü v3’e eklenip saklı topraklarda sınanır.

Kısa eksen, maskenin ikinci momentlerinden elde edilen eşdeğer elipsin kısa çapıdır. Kesişim/birleşim oranı, merkez alanına giren adayların 52 mm bağlamdaki bütün görünen sınırları üzerinden hesaplanır. Alan payı merkezdeki piksellerden hesaplanır. İlerlenirse her özgün eğitim çekiminden bir fiziksel alan payı ölçülür. Özellik, mevcut modelin sentetik eğitim kopyalarında aynı fotoğrafa ait sabit ölçüm olarak taşınır. V3’ün iki PLS girdisine de yalnız bu özellik eklenecektir. 24 topraklık ortalama EMD kazancının sırayla her bir puan çıkarıldığında korunması ve iki telefon yönünde ortalama EMD’nin artmaması şeklindeki mevcut adaylık kuralı uygulanır. 6,3 mm üzeri kütle payı hatası ayrıca verilir.

### ImageGrains 2.0 · altı toprak, iki telefon, 12 çekim

**Karar.** Samsung A52 önceden kayda alınan üç kalite ölçütünü geçti. Motorola Edge’de kesinlik ve duyarlılık %60 oldu, eşik her ikisi için %70’ti. Bu nedenle tek merkez karesinden elde edilen alan payıyla 24 topraklık v3 karşılaştırmasına geçilmedi. Saf v3’ün yerel karşılaştırma puanı 38,758 EMD. Yeni Kaggle tahmini veya model EMD’si üretilmedi.

| Telefon (6 çekim) | Referans iri taş | Modelin iri saydığı | Eşleşen | Kesinlik | Duyarlılık | Alan payı ortalama mutlak farkı |
|---|---:|---:|---:|---:|---:|---:|
| Motorola Edge | 5 | 5 | 3 | %60 | %60 | 0,397 yüzde puan |
| Samsung A52 | 8 | 8 | 8 | %100 | %100 | 0,328 yüzde puan |

**Sınır çizimi ile boyut kararı.** Eşik tanısı sırasında, modelin bütün maskeleri ile 13 iri referans taşını tekrar eşleştirdik. Her referans taşı için ayrı bir model maskesi bulundu. Tam görünür alandaki maske kesişim/birleşim oranının en küçüğü 0,740, ortancası 0,866, en büyüğü 0,965 idi. Yalnız modele de 6,3 mm kısa eksen koşulu konduğunda ortak 11 taş kaldı. P05’te iki taş için referans/model çapları 6,65/5,92 mm ve 6,58/5,06 mm idi. P10’da bir taş referansta 6,22 mm, modelde 6,52 mm ile eşik tarafını değiştirdi. P05’te ayrıca yalnız modelin saydığı, değerlendirme karesine 1,63 mm² giren bir maske vardı. Bu tanı ilk geçiş kuralını veya referans poligonlarını değiştirmedi.

**Yüzeyin temsil gücü.** 12 çekimin sekizinde referans alana giren iri taş yoktu. Alan payı farkları bu nedenle küçük görünebilir. H368’de Samsung karesinin referans/model alan payları %64,82/%62,98, Motorola karesininkiler %0,83/%0,24 idi. Ölçülen iki yüzey bölgesi arasındaki fark, çizim yöntemleri arasındaki farktan çok daha büyük. Ayrı çekimlerde farklı yüzeyler görülebildiğinden bu örnek cihaz yanlılığını tek başına ölçmüyor. Bu yöntem tekrar ele alınacaksa önce aynı çekimden birkaç örtüşmeyen fiziksel alan kullanmanın, toprak başına alan payını ne kadar kararlı yaptığı ölçülebilir. Tek kareyi kullanarak laboratuvar kütle payını iyileştirdiğimiz gösterilmedi.

**Yeniden üretim ve sınırlamalar.** Seçim H181, H368, H516, H549, H666 ve H668 topraklarını verdi. Referans poligonlarını asistan, dosya/telefon/toprak bilgisini ve model tahminlerini görmeden yaklaşık olarak çizdi. Ayrı bir uzman doğrulaması ve bağımsız ikinci çizici yok. Modelin Zenodo kaydındaki `IG2_full_set_cp_SAM` (1.218.643.463 bayt, MD5 `5a5b74bc70a63b6605afd50465240969`) Cellpose 4.0.6 ile CPU’da kullanıldı. `blind`, `infer --weights <dosya>` ve `score` komutları sırasıyla anonim görselleri ve ayrı kimlik listesini üretir, modeli çalıştırır ve sonuçları özetler. Kaydedilmiş maskelerle `score` adımı model yüklenmeden tekrarlanabilir. İkinci bir hesapta 12 panelin kısa eksenleri, eşleşme sayıları ve alan yüzdeleri doğrulandı. [Kod](imagegrains_pilot.py) · [Seçim](imagegrains_pilot_secimi.csv) · [Referans çizimleri](imagegrains_pilot_referans.json) · [Ham model maskeleri](imagegrains_pilot_maskeleri.npz) · [Karşılaştırma görseli](imagegrains_pilot_karsilastirma.png) · [12 çekimin sonuçları](imagegrains_pilot_sonuclari.csv) · [ImageGrains 2.0 makalesi](https://esurf.copernicus.org/articles/14/527/2026/) · [Ağırlıklar](https://doi.org/10.5281/zenodo.15309323)

### ImageGrains sonrası strateji · bir merkez karesi mi, beş ayrı alan mı?

**Kararın dayanağı.** Bir merkez karesinde altı pilot toprağın iki telefonu arasındaki mutlak model alan payı farkının ortalaması 11,28, ortancası 0,44 yüzde puandır. Altı mutlak farkın toplamının %92,7’si H368’den gelir. Bu nedenle yeni yöntemin başarısını bu örneğe bağlamayacağız. Önceki beş kesit deneyinde örtüşen bölgelerin doku özellikleri ortalanmıştı. Buradaki soru, ayrı yüzey alanlarında görülen iri taşları saymanın numune düzeyindeki ölçümü kararlı kılıp kılmadığıdır.

**Önerilen ölçüm sınavı.** İki telefonla da çekilmiş 21 Kaggle toprağından bu altısını ayırınca ImageGrains pilotunda hiç kullanılmamış 15 toprak kalıyor. 15’inin tamamında en az iki Motorola, 13’ünde en az iki Samsung çekimi var. Her toprak ve cihazda dosya adına göre ilk iki özgün çekimi, tek çekim varsa onu kullanmak 58 fotoğraf eder. 15 toprağın 13’ünde laboratuvar 6,3 mm üstünde sıfırdan büyük kütle payı ölçmüş; 15 topraktaki aralık %0–45,60.

1. Aynı fotoğrafta merkezdeki 26 × 26 mm alanı kontrol olarak tut. Adaya merkezden sağa, sola, yukarı ve aşağı 30 mm kaydırılmış aynı boyutta dört alan ekle. Beş ölçüm karesi birbirinden ayrıdır; her biri için sınırları görebilmek amacıyla 52 × 52 mm bağlam korunur. Tüm eğitim fotoğraflarında gerekli 112 mm görüş alanı dosya boyutları ve mevcut kalibrasyona göre sığıyor. Önceki model, eşikler ve fiziksel çözünürlük sabit kalır.
2. Her iki yöntemi aynı çekimlerde karşılaştır. Aynı telefonda tekrarlanan çekimlerin mutlak farkını ve iki telefonun ortalama ölçümleri arasındaki mutlak farkı ayrı raporla; her toprağa eşit ağırlık ver. Birincil ölçüt 15 toprakta iki telefon arasındaki ortalama farkın azalması, tek tek topraklar çıkarıldığında da bu azalmanın korunmasıdır. İki telefonda da ölçünün laboratuvardaki 6,3 mm üstü kütle payıyla pozitif bir sıralama ilişkisi göstermesi ve bu ilişkinin tek kareye göre zayıflamaması aranır. Sabit bir ölçü farkları yapay biçimde küçültür; azalan farkın tek başına yeterli sayılmamasının nedeni bu. Alan yüzdesi ile kütle yüzdesi doğrudan eşitlenmez.
3. Beş bölge, 6,3 mm yakınındaki çap ölçüm hatasını kendiliğinden gidermez. Yeni topraklardan önceden seçilmiş alanlarda model sonucunu görmeden referans çizip önceki telefon bazlı maske kalite eşiklerini tekrar kontrol et. Ölçüm ve maske kontrolleri desteklerse tek yeni özellik ile aynı v3’ü karşılaştır. Model adaylığını 24 saklı toprağın tüm eğri EMD’si ve iki telefon yönündeki mevcut ölçütler belirler. Kontroller desteklemezse bu dal kapanır ve saf v3 referans kalır. Bu turda kayıtlı pilot ve dosya envanteri incelendi; yeni bölgelere model uygulanmadı.

### Beş alan doğrulaması · sonuçlar görülmeden kaydedilen uygulama

Yukarıdaki stratejideki 15 toprakta 58 çekim ve 290 fiziksel alan kullanılır. Telefon başına çekim ortalamaları alındıktan sonra telefonlar arası mutlak fark her toprağa eşit ağırlıkla hesaplanır. Birincil kazanç, tek merkezin telefonlar arası farkından beş alanın farkını çıkarmaktır. On beş toprağın her biri sırayla çıkarıldığında kalan ortalama kazanç pozitif olmalıdır. Laboratuvarda 6,3 mm üstü kütle payıyla ilişki, her telefon için 15 toprakta ayrı Spearman sıra katsayısıdır. Beş alanın katsayısı iki telefonda da pozitif ve merkezin katsayısına en az eşit olmalıdır. Tekrar çekim kontrolünde iki çekimi olan topraklar kullanılır; Motorola’da 15, Samsung’da 13 toprak. Aynı toprak/telefon içindeki ilk iki dosya adı kullanılır.

Maske denetimi için 15 toprak, 6,3 mm üstü laboratuvar kütle payına göre üç eşit gruba ayrılır. 20261122 tohumuyla alt, orta, üst gruptan 2, 2, 4 toprak seçilir. İri taş görebilme olasılığını artırmak için üst grup daha fazla temsil edilir. Her toprakta iki telefonun da ilk çekimi kullanılır. İki çekimden, merkeze göre aynı yönde birer çevre alanı alınır. Bu 16 panel anonim ve karışık sırada çizilir. Model sonuçlarına bakılmadan kaydedilen çizimler önceki pilotta kullanılan nesne ve alan payı eşikleriyle değerlendirilir. Üst grubun daha fazla temsil edilmesi alan denetiminin yorumunu sınırlar.

Asistanın model maskelerini görmeden çizdiği 47 aday sınırdan 30’u kısa eksen ve merkez kareye giren alan koşullarını karşıladı. Motorola’da 18, Samsung’da 12 referans taş var. Çevre panellerindeki bu yaklaşık çizimler bağımsız uzman denetiminden geçmedi. [16 panelin referans çizimleri](imagegrains_bes_bolge_referans.json) için sonuç öncesi SHA-256 değeri `c106ec45737d5d15702419af1157ef57295c343b461ef33c4af3fa6e725325d6`.

### 21 Kasım · ImageGrains ile beş alanın sonucu

**Karar.** 15 toprakta 58 özgün fotoğrafın 290 alanı ölçüldü. Beş alanın ortalaması telefonlar arasındaki farkı azalttı. Laboratuvar sıralamasıyla uyum ve maske doğruluğu için önceden belirlenen iki koşul karşılanmadı. Bu özellik saf v3’e eklenmedi ve ImageGrains ile iri taş alan payı deneyi burada kapandı. Saf v3’ün 24 topraktaki yerel EMD’si 38,758. Bu tur yeni bir yarışma tahmini veya EMD puanı üretmedi.

| Ölçüm | Toprak sayısı | Tek merkez | Beş alan |
|---|---:|---:|---:|
| İki telefonun ölçümleri arasındaki ortalama mutlak fark (yüzde puan) | 15 | 4,929 | 1,887 |
| Motorola’da iki çekim arasındaki ortalama mutlak fark (yüzde puan) | 15 | 4,442 | 5,491 |
| Samsung’da iki çekim arasındaki ortalama mutlak fark (yüzde puan) | 13 | 3,657 | 2,238 |
| Motorola’nın laboratuvar sıralamasıyla uyumu (Spearman) | 15 | 0,780 | 0,742 |
| Samsung’un laboratuvar sıralamasıyla uyumu (Spearman) | 15 | 0,822 | 0,834 |

Ortalama telefon farkı 3,042 yüzde puan azaldı. Altı toprakta fark küçüldü, ikisinde büyüdü, yedisinde sıfır kaldı. Toprak başına kazancın ortancası sıfır. En büyük iyileşme H367’de görüldü (36,515 → 4,488). Bu toprağı çıkardığımızda kalan 14 toprakta azalma 0,972 yüzde puan oldu. Tek tek diğer toprakları da çıkararak elde edilen 15 ortalama kazancın aralığı 0,972–3,584 yüzde puandı.

**Yüzey ölçümünün sınırı.** Yedi toprakta beş alanın ortalaması her iki telefonda da sıfırdı. Bunların beşinde laboratuvar 6,3 mm üstünde kütle ölçtü. En belirgin örnek H037’ydi. Laboratuvarda kütlenin %20,95’i bu eleğin üzerinde kalırken model incelenen alanların hiçbirinde eşikleri geçen taş saptamadı. Ölçülen yüzey, taşların görünürlüğü ve modelin taşları kaçırması bu gözlemde ayrı ayrı ayırt edilemiyor.

| Modelden habersiz çizilmiş sekizer çevre alanı | Referans taş | Modelin işaretlediği taş | Eşleşen | Kesinlik | Duyarlılık | Ortalama mutlak alan hatası |
|---|---:|---:|---:|---:|---:|---:|
| Motorola | 18 | 18 | 17 | %94,4 | %94,4 | 1,029 yüzde puan |
| Samsung | 12 | 15 | 10 | %66,7 | %83,3 | 3,238 yüzde puan |

İki telefonda alan hatası 15 yüzde puan sınırının altındaydı. Samsung’un kesinliği baştan belirlenen %70 eşiğinin altında kaldı. İkincil kontrolde modelin boyut eşiğini uygulamadan önceki sınırları Motorola’da 18 referans taşın 18’iyle, Samsung’da 12 taşın 10’uyla eşleşti (kesişim/birleşim ≥ 0,5). Sınırlı sayıdaki, yaklaşık ve tek çiziciye ait referanslar nedeniyle bu oranlar iki telefonun genel doğruluğunu belirlemek için yeterli değil.

**Yeniden üretim.** `outputs/imagegrains_bes_bolge.py` için `prepare` seçimleri ve kör panelleri oluşturur, `infer` ImageGrains’i çalıştırır, `score` arşivlenmiş maskelerden raporu üretir. İkinci hesapta kısa eksenler piksel koordinatlarının kovaryansından, eşleşmeler bağımsız bir ikili eşleştirme algoritmasıyla hesaplandı. 290 alan yüzdesi, 16 paneldeki taş sayıları ve eşleşmeler, çekim farkları, sıra katsayıları ve toprak çıkarma sonuçları doğrulandı. [Kod](imagegrains_bes_bolge.py) · [58 çekim ve beş alanın ölçüleri](imagegrains_bes_bolge_fotograf_olculeri.csv) · [15 toprağın karşılaştırması](imagegrains_bes_bolge_toprak_sonuclari.csv) · [Telefon sonuçları](imagegrains_bes_bolge_telefon_sonuclari.csv) · [Panel sonuçları](imagegrains_bes_bolge_maske_kontrolu.csv) · [Fotoğraflar ve çizimler](imagegrains_bes_bolge_karsilastirma.png) · [Ham maskeler](imagegrains_bes_bolge_maskeleri.npz)

### Motorola’da iri taşların konumu · uzak alanları ölçmeden önce

Kullanıcının gözlemini sınamak için önce kayıtlı 30 Motorola çekiminde merkeze ve dört yakın alana baktık. Merkezde 13, dört alanın herhangi birinde 12 çekimde model iri taş saptadı. Yalnız çevrede saptanan tek çekim H038’in ilk çekimiydi. 15 toprak eşit ağırlık aldığında dört yakın alanın ortalama payı merkeze göre üç toprakta yüksek, beşinde düşük, yedisinde eşitti. Bunlar bir telefonla çekilmiş yüzeylerdeki model ölçümleridir.

**Yeni karşılaştırma.** İki yöntemde de aynı 58 özgün çekim ve aynı beş adet 26 × 26 mm ölçüm alanı kullanılacak. Merkez ve kısa eksendeki iki alan aynı kalacak. Yatay veya dikey çekime göre uzun eksendeki iki alanın uzaklığı 30 mm’den 100 mm’ye çıkacak. Merkezler görüntü içeriğine bakılarak seçilmeyecek. İki alanın konumu dışında model, fiziksel çözünürlük, kısa eksen eşiği ve beş alanın eşit ağırlığı sabit. Bu düzen uzun eksende uzaklaşmayı sınar. Fotoğrafın köşeleri ve yüzey altındaki malzeme kapsam dışındadır.

Birincil soru, 15 toprakta Motorola’nın laboratuvar sıralamasıyla uyumunun önceki beş alana göre artıp artmadığı. Aynı değişimin Samsung’daki yönünü, her toprak sırayla çıkarıldığında Motorola’daki değişimi, telefonlar arası ve tekrar çekim farklarını da raporlayacağız. Motorola’da ve Samsung’da sonuçları birlikte destekleyen bir kazanç çıkarsa uzak alanlardaki taş sınırları ayrıca kontrol edilmeli. Önceki 16 panelin maske kontrolü uzak alanlar için yeni doğrulama sağlamaz. Aynı topraklar tekrar kullanıldığı için bu karşılaştırma keşif amaçlıdır.

Uzak alanlar için, önceki maskeleri denetlenen sekiz toprağın her telefondaki ilk çekiminden birer yeni alan seçilecek. Uzun eksenin hangi tarafının kullanılacağı her toprak için 20261123 tohumuyla seçilecek. Aynı işaret iki telefona uygulanacak. Model işaretleri ve panel kimlikleri gizlenerek 16 alan çizilecek, önceki %70 kesinlik, %70 duyarlılık ve en çok 15 yüzde puan alan hatası ölçütleriyle okunacak. Laboratuvarın elekte tarttığı pay ile görünen yüzeydeki alan payı farklı niceliklerdir.

Anonim panellerde modelin sonuçları açılmadan asistan 55 aday sınır çizdi. Fiziksel eşikleri uygulayınca Motorola’da 19, Samsung’da 18 referans taş kaldı. Referans dosyasının sonuç öncesi SHA-256 değeri `a4cd132ee7705d7840fcd16f315c2d4ccbd6e483748dfe2d32248a98f6edd697`. 116 yeni karenin işaretlenmemiş fotoğrafları küçük önizlemede incelendi. Ölçüm karelerinde belirgin tepsi çerçevesi görünmedi.

### 21 Kasım · Aynı beş kareyle uzak alanları ölçmenin sonucu

**Motorola’da uzun eksende daha uzağa bakmak işe yaradı.** Önceki beş alan yöntemiyle karşılaştırıldığında, laboratuvarın 6,3 mm üstü kütle payına göre 15 toprağı sıralama uyumu 0,7423’ten 0,8584’e çıktı. 15 toprağın her biri sırayla çıkarıldığında sıra katsayısındaki artış +0,0665 ile +0,1400 arasında kaldı. Samsung’da ise değişim −0,0116 oldu. Bu küçük fark, çıkarılan toprağa göre yön değiştirdi (−0,0195 ile +0,0049). Bu çalışmada yarışmanın tam eğri EMD’si ölçülmedi. Saf v3 yerel referans olarak kaldı.

| Aynı 58 fotoğrafta karşılaştırma | Yakın beş alan | İki alan 100 mm’ye taşınınca |
|---|---:|---:|
| Motorola’da laboratuvar sıralamasıyla uyum (15 toprak) | 0,7423 | 0,8584 |
| Samsung’da laboratuvar sıralamasıyla uyum (15 toprak) | 0,8336 | 0,8220 |
| Motorola’da tekrar çekimlerin ortalama mutlak farkı (15 toprak, yüzde puan) | 5,491 | 5,042 |
| Samsung’da tekrar çekimlerin ortalama mutlak farkı (13 toprak, yüzde puan) | 2,238 | 3,044 |
| İki telefonun ortalama ölçümleri arasındaki mutlak fark (15 toprakta ortalama, yüzde puan) | 1,887 | 1,848 |

**H037 ve H038 farklı örnekler.** H037’nin ilk Motorola çekiminde önceki beş karenin tamamı sıfırdı. Merkezden 100 mm uzaktaki iki karenin birinde model bir iri taş buldu. O karenin %6,00’sını kaplayan bu taş, beş karenin ortalamasını o çekimde %1,20’ye ve iki çekimlik toprak ortalamasını %0,60’a getirdi. Laboratuvar kütlesinin %20,95’i 6,3 mm üzerindeydi. Samsung bu toprağın iki düzende de ölçülen yüzeyinde iri taş saptamadı. H038’in ilk Motorola çekiminde yakın çevrede saptanan taşla ilgili önceki gözlem korundu. H038’in iki çekimlik alan ortalaması her iki düzende de %1,16 oldu. 58 çekimde önceki beş alan sıfırken yeni düzende taş saptanan tek çekim H037’nin ilk Motorola çekimiydi.

**16 uzak alanda maske denetimi.** Motorola’da 19 referans taşın 19’u eşleşti, model 20 taş işaretledi (kesinlik %95,0, duyarlılık %100, alan hatası 0,514 yüzde puan). Samsung’da 18 referans taşın 15’i eşleşti, model 17 taş işaretledi (kesinlik %88,2, duyarlılık %83,3, alan hatası 1,762 yüzde puan). İki cihaz da bu panellerde önceki pilotun maske eşiklerini geçti. Asistanın özgün fotoğraflarda yaptığı yaklaşık çizimler bağımsız uzman çizimiyle doğrulanmadı. Denetim önceki sekiz toprağın yeni alanlarıyla sınırlı.

**Yorum.** Ölçüm konumu kimi Motorola çekimlerinde sonucu değiştiriyor. Bu sınama, iri taşların genel olarak Motorola fotoğraflarının kenarlarında toplandığını göstermiyor. Kayıtlı 15 toprakta iki telefon aynı yönde kazanım vermediği için saf v3’e yeni özellik eklenmedi. Bu sonuçla telefonlara ayrı kural seçmek, kararı iki kez aynı veriye dayandırır. Bundan sonra en bilgilendirici kontrol, düzeni sabit tutarak bu 15 toprağın dışında kalan altı eşli pilot toprağında yakın ve uzak beş alanı karşılaştırmak. O altı toprağın merkez kareleri önceki pilotta incelenmişti. Uzak alanlar henüz karşılaştırılmadı.

**Doğrulama.** Kaydedilen 290 yakın ve 116 uzak maske üzerinden kısa eksenleri piksel koordinatlarının kovaryansından yeniden hesapladık. İkinci hesap 58 çekimin iki ölçümünü, 16 panelin eşleşmelerini, iki telefonda 15’er toprak ortalamasını ve 30 ayrı toprak çıkarma sonucunu doğruladı. [Deney kodu](imagegrains_uzak_alan.py) · [Alanların yerleşimi](imagegrains_uzak_alan_yerlesim.png) · [58 çekim](imagegrains_uzak_alan_fotograflar.csv) · [Toprak sonuçları](imagegrains_uzak_alan_topraklar.csv) · [16 panel](imagegrains_uzak_alan_panel_secimi.csv) · [Kör çizimler](imagegrains_uzak_alan_referans.json) · [Maske sonuçları](imagegrains_uzak_alan_maske_kontrolu.csv) · [Maske karşılaştırması](imagegrains_uzak_alan_maske_kontrolu.png) · [Uzak alan maskeleri](imagegrains_uzak_alan_maskeleri.npz)

### Bütün fotoğrafı kullanmak · aynı PLS ile doğrudan EMD karşılaştırması

127 özgün fotoğrafın tamamında bir tarafta v3’ün her kenarda %60’lık merkez kadrajını, diğer tarafta görüntünün tamamını kullandık. Tam karede tepsi, ışık ve diğer arka plan pikselleri de kalır. İki tarafta da 4 piksel/mm fiziksel çözünürlük, aynı 30 görüntü özelliği, aynı iki PLS bileşen modeli, toprak başına ortalama ve aynı saklı topraklar kullanıldı. Hızlı ve eşlenmiş bir karşılaştırma için her iki tarafta yalnız özgün çekimleri kullandık. Standart saf v3’ün sekiz türetilmiş çekimi de kullanan 38,758 EMD’si ayrı bir eğitim düzenine aittir.

| Sınavın ortalama EMD’si | Merkez kadraj | Tam fotoğraf | Tam fotoğrafın daha iyi olduğu toprak |
|---|---:|---:|---:|
| 24 toprak, bütün eğitim telefonları | 38,440 | 40,147 | 11/24 |
| Motorola Edge ile eğit, Samsung A52 ile sına | 47,717 | 80,612 | 2/21 |
| Samsung A52 ile eğit, Motorola Edge ile sına | 38,656 | 50,437 | 5/21 |

Tam kare ortalama hatayı 1,706 EMD artırdı. 24 topraktan biri sırayla çıkarıldığında hata artışı 0,612 ile 2,937 EMD arasında kaldı. Alansal kapsamla arka plan birlikte değişti. Bu karşılaştırma, tepsi duvarının kaybın ne kadarını açıkladığını veya yalnızca görünür toprak yüzeyini kullanmanın sonucunu ölçmez. Bir sonraki doğrudan soru, arka planı çıkarıp tüm toprak yüzeyinde aynı fiziksel ölçüleri kullanmanın 24 topraktaki hatayı azaltıp azaltmadığıdır. Önce her kamera için toprak sınırı gösterilmeli. Model ayarı ve toprak ayrımı sabit kalmalı. Bu yerel deneyden Kaggle gönderimi üretilmedi. [Deney kodu](v3_tam_fotograf.py) · [132 tahmin eğrisi ve EMD değerleri](v3_tam_fotograf_sonuclari.csv) · [H037’nin fotoğrafı ve standart saf v3 tahmini](v3_fotograf_ve_tahmin_H037.png)

### H037 · merkez dışındaki doku ve hiyerarşik temsil fikri

İlk Samsung A52 çekimini aynı yükseklikte, görüntü genişliğinin %14,5, %50 ve %85,5 konumlarında karşılaştırdık. Üç kare de yaklaşık 25 × 25 mm ve 114 × 114 özgün piksel. Merkezde tane sınırları daha belirgin görünüyor. Soldaki dış karede zemin daha düz görünüyor, sağdaki dış karede iri taneler de var. Bu tek çekim ve üç alan, tüm dış bölgenin daha ince olduğunu veya bulanıklığın ne kadar etkili olduğunu belirlemeye yetmiyor. Bu dosyanın kalibrasyonu 4,553 piksel/mm. 0,2 mm yaklaşık 0,91 piksel ediyor. [Eşit ölçekte üç yakın plan](H037_merkez_dis_ayni_olcek.png)

H037’nin laboratuvar sonucu 0,2 mm altında %34,6629 kütle. Aynı 24 toprak ayrımını ve yalnız özgün çekimleri kullanan PLS karşılaştırmasında merkez kadrajı %68,2925, tam kare %70,8673 verdi. İki sürüm de ince payı fazla tahmin etti. Tam kareye geçince hata 2,575 yüzde puan büyüdü. Bu tahminler numunenin altı çekimini birleştiriyor. Tam karede tepsi de görüldüğü için dış toprak bölgesinin tek başına etkisi ayrıştırılmadı. Önceki grafikteki %77,4310, eğitimde yapay çekimleri de kullanan standart saf v3 tahminiydi. [Eşlenmiş kadraj sonuçları](v3_tam_fotograf_sonuclari.csv)

Tartışılan hiyerarşik seçenek, tepsi dışlandıktan sonra tüm yüzeyin genel görünümüyle sabit mm ölçeğindeki yerel doku bölgelerini birlikte kullanmak. Aynı toprak maskesi, öğrenicisi ve eğitim ayrımlarıyla genel görünüm ile genel görünüm + yerel özet karşılaştırılabilir. 24 bağımsız toprakta ilk deneme için görsel kodlayıcı sabit tutulabilir. Karar bütün eğrinin numune bazında EMD’siyle, 0,2 mm ise yardımcı ölçüyle verilir. Autoencoder ve VQ-VAE bu turda kavramsal olarak değerlendirildi. Yeni model eğitimi veya Kaggle puanı yok. [Toprak fotoğraflarında DINOv2 ve çoklu bölge öğrenmesi](https://espace.etsmtl.ca/id/eprint/4065/) · [Maskeli autoencoder](https://arxiv.org/abs/2111.06377) · [VQ-VAE](https://arxiv.org/abs/1711.00937)

### Hiyerarşik DINOv2 + Ridge · karşılaştırma düzeni

127 özgün eğitim fotoğrafında görünen tepsi duvarları ve lamba için renk geçişlerinden toprağın içinde kalan muhafazakâr bir dikdörtgen seçilir. Islaklık, bulanıklık, renk veya etiket üzerinden parça seçimi yapılmaz. Her iki sürüm aynı dikdörtgenin genel DINOv2-S/14 temsilini kullanır. İkinci sürümde yaklaşık 50 × 50 mm kareler aynı alanı kenarlara kadar örter. Komşu kareler gerektiğinde kısmen örtüşür. Her karenin ağırlığı, kendi merkezine diğer merkezlerden daha yakın olan toprak alanıdır. Yerel temsillerin ağırlıklı ortalaması ve standart sapması genel temsile eklenir. Global, yerel ve numune düzeylerinin ardından hedef aynı laboratuvar eğrisidir.

Önceki deneyin sabit ağırlıkları, sekiz bileşenli PCA ve Ridge katsayısı 10 kullanılır. Fotoğraf özellikleri toprak içinde eşit ağırlıkla birleştirilir. Ölçekleyici, PCA ve Ridge her seferinde yalnız eğitim topraklarında öğrenilir. Ana sınama 24 toprağın tam eğri EMD ortalamasıdır. Her toprak için eşlenmiş hata, tek tek toprakların etkisi, iki telefon geçişi ve H037’de 0,2 mm ince payı da raporlanır. Hiyerarşik sürüm genel görünümü geçerse, yarışma adayı için aynı topraklarda saf v3 ile karşılaştırılır. Codebook veya ek sinir ağı eğitimi bu deneye dahil edilmedi.

### Hiyerarşik DINOv2 + Ridge · 24 toprakta sonuç

**Karar.** Genel görünüme 50 mm bölgelerin doku ortalamasını ve standart sapmasını eklemek ortalama yerel EMD’yi 50,326’dan 50,599’a getirdi. İki değer arasındaki 0,273 EMD farkı tek bir toprağın çıkarılmasıyla yön değiştirebiliyor. Saf v3 referansı 38,758 ile her iki sürümü de geçti. Bu hiyerarşik sürümü Kaggle’a gönderme önerisi çıkmadı.

| Yerel EMD sınaması | Saf v3 | Yalnız genel görünüm | Genel + yerel doku | Yerelin geneli geçtiği toprak |
|---|---:|---:|---:|---:|
| 24 toprak, tüm eğitim telefonları | 38,758 | 50,326 | 50,599 | 13/24 |
| Motorola Edge → Samsung A52 | 41,987 | 56,129 | 57,291 | 9/21 |
| Samsung A52 → Motorola Edge | 42,831 | 54,386 | 52,956 | 10/21 |

Yerelin genele karşı toprak başına ortanca kazancı +0,443 EMD, birer toprak çıkarıldığında ortalama kazanç aralığı −0,909 ile +0,278 EMD oldu. Yerel sürüm saf v3’ten 6/24 toprakta daha düşük hata verdi. Bütün 24 toprakta saf v3’ün ortalama avantajı 11,842 EMD, tek toprak çıkarıldığındaki avantaj aralığı 9,631–13,573 EMD oldu.

**H037.** Laboratuvarda 0,2 mm’den ince kütle %34,66. Sırasıyla saf v3, yalnız genel ve genel + yerel modelleri %77,43, %56,01 ve %56,82 tahmin etti. H037’nin tüm eğrisinde EMD değerleri 91,448, 62,482 ve 63,483 oldu. Genel görünümle alınan iyileşme yerel özet eklendiğinde büyümedi. 24 toprakta 0,2 mm’de ortalama mutlak hata sırasıyla 11,718, 20,261 ve 20,016 yüzde puan oldu.

**Görülen alan ve kapsam.** Tepsi içi yaklaşık dikdörtgenler 127 özgün fotoğrafta görsel olarak kontrol edildi. Aydınlatma donanımı ve tepsi kenarının etkisini azaltmak için dar bir toprak şeridi de dışarıda kalabilir. Kadrajlar fotoğraf alanının %62,3–93,7’sini korudu. Fotoğraf başına 18–35, toplam 3.440 fiziksel kare kullanıldı. Her eksende tüm piksellerin en az bir kareye girdiği ve 127 fotoğraf özetinde iki modelin genel görünüm vektörlerinin eşleştiği kontrol edildi. Her sürümde 24 + 21 + 21 saklı tahmin üretildi. Kaydedilen temsillerden PCA ve Ridge’i ayrı bir matris hesabıyla tekrar kurarak 132 tahmin eğrisini, EMD’leri ve saf v3 karşılaştırmalarını doğruladık. Codebook, öğrenilmiş dikkat ağırlıkları veya bölgelerin açık konum bilgisi bu karşılaştırmada sınanmadı. Yeni Kaggle puanı yok.

[Kullanılan kadrajların görseli](dinov2_hiyerarsik_alanlar.png) · [H037 ve 24 toprağın grafiği](dinov2_hiyerarsik_karsilastirma.png) · [132 tahmin ve EMD](dinov2_hiyerarsik_sonuclar.csv) · [127 fotoğrafın kadrajları](dinov2_hiyerarsik_kadrajlar.csv) · [Fotoğraf temsilleri](dinov2_hiyerarsik_temsiller.npz) · [Deney kodu](dinov2_hiyerarsik.py)

### Kullanıcının hiyerarşi fikri · tane boyutuna göre iki aşama

Kullanıcı küçük görüntü parçalarının aynı numuneye ait bir grup olarak öğrenilmesini ve görünür iri taneler tespit edildikten sonra kalan ince dokunun ayrıca değerlendirilmesini kastediyor. Önceki 50 mm deneyinde kareler hazır DINOv2’den sekizerli gruplarla geçirilmiş, bölge özellikleri sabit ağırlıklarla özetlenmiş, Ridge numune eğrisini öğrenmişti. Laboratuvar hatasından hangi karenin ne kadar ağırlık alacağını öğrenen bir mekanizma kullanılmadı. ImageGrains deneyleri ise görünür iri tanelerin maskelerini ve alanını sınadı. Bu maskelerin dışında kalan dokuyu ayrı işleyip iki bilgiyi birlikte kullanarak 11 elekli eğriyi tahmin eden düzen henüz sınanmadı. Maskelenmemiş yüzeyde orta boy taneler ve kaçırılan iri taşlar da bulunabilir.

Bu fikre yönelik ilk karşılaştırma, aynı kareler ve aynı numune ayrımıyla tüm toprak dokusunu kullanan sürüm ile görünür iri taşların ölçülerini ve kalan yüzeyin dokusunu birlikte kullanan sürüm olabilir. Ölçüm birimi laboratuvar eğrisi olan topraktır. H037 için 0,2 mm altındaki %34,66 kütle bütün numuneye aittir. Tek tek karelerin kütle dağılımı bilinmiyor. Bu görüşmede yeni model puanı üretilmedi.

### Görünür iri taş ve kalan doku · sonuçlardan önceki karşılaştırma düzeni

24 topraktaki 127 özgün eğitim fotoğrafı kullanılır. Üç sürümün ortak bilgisi, önceki deneyde kaydedilmiş tepsi içi genel DINOv2 vektörüdür. İri taş sınırları için ImageGrains 2.0’ın aynı ağırlıkları, 4,5 piksel/mm ölçeği ve 6,3 mm kısa eksen eşiği korunur. Kısmen görülen taşları ölçerken 52 mm bağlamın ortasındaki 26 × 26 mm kareye daha önceki gibi en az 1 mm² giriş koşulu uygulanır. Her çekimde merkez, kısa eksende ±30 mm ve uzun eksende ±100 mm olmak üzere beş örtüşmeyen kare kullanılır. Önceki DINOv2 deneyinin 3.440 yerel karesinin tamamına taş maskesi uygulanmıyor. Bu ilk karşılaştırma geniş yüzey bilgisinin yanında 635 sabit yerel ölçüm alanını sınar. Örnek olarak önceden bilinen ilk Motorola H037 uzak alanı görselleştirilir. Fotoğraf veya alan seçimi yeni tahmin hatalarına bağlı değildir.

**Üç eşlenmiş sürüm.** Kontrol bütün yerel dokunun fiziksel uzaklıklardaki parlaklık farklarını kullanır. İkinci sürüm aynı dokuya görünen iri taşların kapladığı alanı, merkez yoğunluğunu ve alanla ağırlıklandırılmış kısa eksenini ekler. Üçüncü sürüm aynı taş ölçülerini koruyup dokuyu yalnız taş maskelerinin dışındaki piksel çiftlerinde hesaplar. Gri görselleştirme pikselleri hesaplamaya girmez. İkinci ve üçüncü sürümlerin karşılaştırması, doğrudan taşları doku hesabından çıkarmanın katkısını ölçer. İlk ve üçüncü sürümün karşılaştırması kullanıcının önerdiği bütün değişikliği ölçer.

Genel DINOv2 vektöründeki sekiz beyazlatılmış PCA bileşeni ve standardize edilmiş yerel ölçüler Ridge’e verilir (alpha 10). Fotoğraflar aynı toprak içinde eşit ağırlıkla birleştirilir. Ölçekleme, PCA, gerekirse eksik doku değerinin ortanca ile tamamlanması ve regresyon her turda yalnız eğitim topraklarında yapılır. Ana ölçüt 24 toprağın tam elek eğrisi EMD’sidir. H037’nin 0,2 mm ince payı, 6,3 mm üstü kütle payı hatası, tek toprak çıkarma sonuçları ve iki telefon yönü ayrıca incelenir. Saf v3 aynı sınavdaki sabit yarışma referansıdır.

**Kapsam ve H037 görüntüsü.** 635 ölçüm karesinin tamamı genel modelin kullandığı toprak dikdörtgenine sığıyor. Beş karenin toplam alanı 33,8 cm². Bu, çekim başına görülen toprak dikdörtgeninin %4,91–8,53’ü (ortanca %7,23). H037’nin iki telefondan altı çekiminde taranan 30 karenin birinde kısa ekseni en az 6,3 mm olan bir taş işaretlendi. İlk Motorola çekiminin sağdaki uzak karesinde işaret alanın %6,00’sını kaplıyor. Dedektörün işareti ve laboratuvarın numune üzerinde ölçtüğü kütle payı ayrı ölçümlerdir. [H037’nin 30 karesi](iri_tas_kalan_doku_H037_tum_alanlar.png) · [Özgün alan, maske ve kalan doku](iri_tas_kalan_doku_H037.png)

### 22 Kasım · İri taş ölçülerini eklemek ve taşları dokudan çıkarmak

127 özgün çekimde 635 alan tamamlandı. En az bir iri taşın işaretlendiği 251 kare ve 60 fotoğraf var. 24 toprağın her biri için o toprağa ait tüm çekimler eğitimden birlikte çıkarıldı. Telefon sınamalarında ortak 21 topraktan her defasında biri saklandı ve diğer 20 toprakla eğitim yapıldı. Aşağıdaki değerler yerel ortalama EMD, düşük değer daha iyi.

| Model | 24 toprak | Motorola → Samsung | Samsung → Motorola |
|---|---:|---:|---:|
| Genel görüntü + bütün yerel doku | 44,670 | 51,532 | 40,902 |
| Aynı model + taş ölçüleri | 44,402 | 49,231 | 41,088 |
| Genel görüntü + kalan doku + aynı taş ölçüleri | 45,663 | 48,787 | 41,024 |
| Saf v3 referansı | **38,758** | **41,987** | 42,831 |

**Taşları çıkarmanın tek başına etkisi.** Taş ölçüleri sabitken maskeyi doku hesabına uygulamak hatayı 44,402’den 45,663’e çıkardı. 10/24 toprak iyileşti, toprak başına ortanca hata artışı 0,924 EMD oldu. Hangi tek toprağın puanını çıkarırsak çıkaralım ortalama hata artışı 0,771–1,472 arasında kaldı. Motorola → Samsung yönünde maskeleme 0,444 EMD kazandırdı. Ters yönde ortalama kazanç 0,064 EMD oldu ve tek toprak çıkarılınca işareti değişebildi.

**İki değişikliğin toplamı.** Bütün dokudan kalan doku + taş ölçülerine geçince ortalama hata 0,993 EMD arttı. 13/24 toprakta iyileşme ve 0,016 EMD ortanca kazanç var. Birer toprak çıkarıldığında ortalama hata artışı 0,164–1,539 EMD. H037 ve H038 birlikte çıkarılınca da hata 0,525 EMD arttı. Yalnız taş ölçülerini eklemek ortalamada 0,269 EMD kazandırdı (15/24 toprak iyileşti). Bu küçük kazancın işareti tek toprak çıkarılınca değişebiliyor.

**H037 ve elekler.** 0,2 mm’den geçen gerçek kütle %34,66. Bütün doku %58,10, taş ölçülerini ekleyen model %61,56, kalan doku ve taş ölçüleri %62,00, saf v3 %77,43 tahmin etti. H037’nin bütün eğri EMD’leri aynı sırada 63,411, 68,438, 69,388 ve 91,448. H037’de maske öncesi bütün doku en iyi tahmini verdi. Numunede 6,3 mm üstü laboratuvar kütlesi %20,95. Kalan doku modeli bu payı %1,56 tahmin etti. Bu örnekte işaretlenen yüzey alanı, iri kütle payını açıklamakta yetersiz kaldı. Bunun alan seçimi, taş tespiti ve yüzey-kütle ilişkisi arasındaki dağılımı bu deneyle belirlenemiyor.

24 toprakta 0,2 mm kütle payının ortalama mutlak hataları aynı sırada 17,228, 17,660, 17,636 ve 11,718 yüzde puan. 6,3 mm üstü kütle payında hatalar 8,777, 8,177, 9,026 ve 8,310 yüzde puan. Saf v3 ana model olarak korunuyor. Bu deney için Kaggle gönderimi hazırlanmadı. H037’de alan seçimini sınamak için sonraki küçük kontrol, aynı çekimin tepsi içindeki tüm yüzeyinde görülen iri taşları beş karenin kapsadığı taşlarla karşılaştırmak olabilir.

**Doğrulama ve dosyalar.** Önceki 58 fotoğrafın yeniden hesaplanan iri alan yüzdeleri eski kayıtlarla 0,00000001 yüzde puan toleransında eşleşti. Üç yeni sürümün 198 saklı tahminini ölçekleme, tekil değer ayrışımı ve Ridge normal denklemleriyle ikinci kez hesapladık. En büyük eğri farkı 0,000000002 yüzde puanın altında kaldı. Saf v3 dahil 264 satırın EMD’si kayıtlı eğriler ve laboratuvar etiketlerinden doğrulandı. [Sonuçların grafiği](iri_tas_kalan_doku_karsilastirma.png) · [198 tahmin eğrisi ve EMD](iri_tas_kalan_doku_sonuclar.csv) · [127 çekimin ölçüleri](iri_tas_kalan_doku_fotograflar.csv) · [635 alanın maskeleri](iri_tas_kalan_doku_maskeler.npz) · [Deney kodu](iri_tas_kalan_doku.py)

### 22 Kasım · H037’nin ilk Motorola çekiminde tam yüzey ve eski beş kare

Önceki görselde kullanılan `Motorola_Edge_H037_01.jpg` üzerinde çalıştık. Önceden belirlenmiş tepsi içi toprak dikdörtgeni yaklaşık 454 cm². Aynı ImageGrains ağırlıkları, 4,5 piksel/mm ve 6,3 mm kısa eksen eşiğiyle tüm dikdörtgeni taradık. Modelin örtüşen görüntü parçalarından hesapladığı tek maske üzerinden her taşı bir kez saydık. Eski beş karenin kayıtlı piksel konumlarını bu maskeye taşıdık. Karelerin toplamı 33,8 cm², bu çekimde taranan alanın %7,45’i. Önceki karenin bir taşı kapsaması için en az 1 mm² kesişim koşulunu koruduk.

**Sonuç.** Model tam yüzeyde eşiği geçen üç taş buldu. Sağdaki eski karenin içinde kalan taşın kısa ekseni 6,47 mm. Diğer iki taşın kısa eksenleri 6,34 ve 6,44 mm. İkisi de beş karenin tamamen dışında ve taranan kadrajın sınırından içeride. Eski maskenin işaretlediği taş tam yüzeyde yeniden bulundu. İki sınırın kesişim/birleşim oranı 0,967. Üç taşın toplam izdüşüm alanı yaklaşık 110,02 mm². Beş kare bu alanın %37,03’ünü içeriyor. Bu oran fotoğraftaki işaretli taş alanına ait, laboratuvarın ölçtüğü kütle payı ayrı bir ölçüm.

Kaydedilen maskede kısa eksenleri piksel koordinatlarının kovaryansından bağımsız olarak yeniden hesaplayıp üç taşın kimliğini, boyutlarını ve karelerle kesişimlerini doğruladık. Üçü de 6,3 mm eşiğine yakın ve iki başka maske 6,0–6,3 mm aralığında. Sonuç tek çekimdeki alan seçimini sınar. Bu aşamada 16 bölgeli eğitim ya da yeni bir EMD ölçümü yapılmadı. Bir sonraki seçenek, aynı toprak alanını boşluksuz kaplayan 16 bölgenin yerleşimini ve sınırdaki taşların nasıl birleştirileceğini önce bu çekimde kontrol etmek. [Özgün yüzey ve işaretli taşlar](h037_tam_yuzey.png) · [Taş ölçüleri](h037_tam_yuzey_taslar.csv) · [Tam çözünürlüklü model maskesi](h037_tam_yuzey_maske.npz) · [Deney kodu](h037_tam_yuzey.py)

### H037 · aynı çekimde 4 × 4 bölge ve sınır kontrolü

İlk Motorola çekiminin önceki tam yüzey taramasındaki 4,5 piksel/mm çözünürlüğü, dikdörtgen kadrajı, tam yüzey renk normalizasyonu, ImageGrains ağırlıkları ve 6,3 mm kısa eksen eşiği korunur. Yüzeyin tamamı 16 dikdörtgene ayrılır. Her piksel yalnız bir sayım bölgesine aittir. Her bölge, görüntünün izin verdiği yönlerde komşu bölgeden 59 piksel (yaklaşık 13 mm) ek bağlamla taranır. Taşın büyüklüğü bu geniş görüntüde ölçülür. Merkez noktası sayım bölgesinde kalan maskeler alınır. Farklı bölgeler aynı taşı sahiplenirse kesişim/birleşim oranı en az 0,50 olan iki maskeden çevresinde daha çok görüntü bulunan korunur. Tam yüzey maskesi sonuçları sayım tamamlandıktan sonra konuma göre bire bir eşleştirmek için kullanılır.

Olağan ızgaranın hiçbir iç sınırı önceki üç iri taşın içinden geçmiyor. Sınır davranışını ayrıca ölçmek için, önceki beş karede de görülen sağdaki taş önceden seçildi. Yakınındaki bölme çizgisi bu taşın merkezine kaydırılır ve yalnız iki komşu bölge yeniden taranır. Bu kontrol asıl 16 bölgenin sonucundan ayrı raporlanır.

**16 bölgenin sonucu.** 918.669 pikselin her biri tam bir sayım bölgesine girdi. 16 taramada 6,3 mm eşiğini geçen dört görünüm vardı. Merkez konumuna göre birleştirilince üç ayrı taş kaldı. Üçü de tam yüzey modelinin işaretlediği üç taşla bire bir eşleşti. Eşiği aşan yeni veya kayıp taş yoktu. Boyutlar fotoğraftaki maskelerin kısa eksenleridir.

| Görseldeki taş | Tam yüzey (mm) | 16 bölge (mm) | Maskelerin kesişim/birleşim oranı |
|---|---:|---:|---:|
| 1 | 6,338 | 6,307 | 0,977 |
| 2 | 6,472 | 6,324 | 0,970 |
| 3 | 6,440 | 6,476 | 0,972 |

**Sınır kontrolü.** Çizgi 2 numaralı taşın içinden geçirildiğinde komşu iki bölge taşı 6,375 ve 6,329 mm ölçtü. İki maske de tam yüzey maskesine göre aynı taşla eşleşti (oranlar 0,977 ve 0,970). Merkez kuralıyla birincisi sayıma girdi ve toplam bir taş oldu. Doğal ızgarada bir bölge çizgisini kesen, her boyuttan 100 tam yüzey maskesinin 89’u 16 bölgeli taramayla 0,50 eşiğinde bire bir eşleşti. Kalan 11’i bu ölçüte göre eşleşmedi. Bu karşılaştırmanın referansı aynı modelin tam yüzey çıktısıdır. Bu deney laboratuvar tahmininin doğruluğunu ölçmedi. Saf v3’ün ana model kararı korunuyor.

**Doğrulama.** Arşivlenen maskelerden kısa eksenleri piksel koordinatlarının kovaryansı, merkezleri ve alan örtüşmelerini doğrudan piksel hesabıyla yeniden bulduk. Üç iri eşleşme, sınır kontrolündeki tek sayım ve her pikselin bir bölgeye ait olduğu doğrulandı. [16 bölge ve taşlar](h037_onalti_bolge.png) · [Sınırdaki taşın iki görünümü](h037_onalti_bolge_sinir.png) · [Taşların eşleşme tablosu](h037_onalti_bolge_eslesmeler.csv) · [Sınır kontrolünün ölçüleri](h037_onalti_bolge_sinir.csv) · [Kod](h037_onalti_bolge.py)

### 24 toprak · beş sabit alan ve toprak yüzeyini kaplayan 16 bölge

Ana soru, aynı altı doku ölçüsünü daha geniş alandan almanın tahmin hatasını değiştirip değiştirmediği. Önce önceki beş alanın 44,670 EMD sonucunu özgün ölçüm düzeniyle yeniden hesaplarız. Eşlenmiş karşılaştırma için her çekimin önceden kaydedilmiş toprak dikdörtgenini bir kez 4,5 piksel/mm ölçeğine getiririz. Hem beş alanı hem 16 bölgeyi bu ortak piksel ızgarasından okuruz. Böylece tek tek küçük kesitleri yeniden boyutlandırmanın getirdiği farkı ölçüm alanının etkisinden ayırabiliriz.

Beş karede iki piksel arasındaki yatay veya dikey ölçüm aynı karenin içinde kalır. 16 bölge tüm toprak dikdörtgenini kaplar. Her geçerli piksel çifti ilk pikselinin bölgesinde bir kez sayılır. Komşu bölgelerin sınırındaki doku da ölçülür. Altı fiziksel uzaklığın her birinde tüm geçerli çiftlerin ortalama mutlak parlaklık farkını alırız. Fotoğraf başına altı doku özelliği ve ortak genel DINOv2 vektörü korunur. Taş maskesi veya ayrı bir bölge etiketi gerekmez. Değişkenler numune içinde önceki gibi ortalanır. Her ayrımda eğitim topraklarında sekiz PCA bileşeni ve aynı Ridge (alpha 10) kullanılır.

Ana karar 24 toprağın tüm eğri ortalama EMD’siyle verilir. Toprak başına eşlenmiş kazanç, her seferinde bir toprak çıkarılınca ortalama kazanç, iki telefon yönü, H037 ve 0,2 mm hatası ayrıca incelenir. Saf v3 aynı ayrımların yarışma referansıdır.

**Sonuç.** 127 özgün fotoğrafın tamamında beş alan ve tam yüzeyi kaplayan 16 bölge ölçüldü. Her fotoğrafta altı fiziksel ölçekteki doku özeti, her toprakta 11 elekli tek tahmin eğrisi üretildi. Her toprak tüm çekimleriyle birlikte sınama için ayrıldı. 16 bölgedeki doku ölçüleri geçerli piksel çifti sayısıyla ağırlıklı olarak birleştirildi. Model ayrı bölgelerin önemini veya konumunu öğrenmedi.

| Yerel sınama | Beş alan (ortak ızgara) | 16 bölge | Saf v3 | 16 bölgenin beş alana göre iyileştirdiği toprak |
|---|---:|---:|---:|---:|
| 24 toprak | 44,760 | 43,918 | 38,758 | 9/24 |
| Motorola → Samsung | 51,953 | 48,385 | 41,987 | 16/21 |
| Samsung → Motorola | 41,383 | 48,301 | 42,831 | 4/21 |

**Ortalama ve toprak bazında fark.** 16 bölge 24 toprakta ortalama 0,842 EMD kazandırdı. Ortanca toprakta hata 1,841 EMD arttı. H374’te hata 82,972’den 60,643’e indi. H374’ün puanı çıkarılınca kalan 23 toprakta beş alanın ortalaması 43,098 ve 16 bölgeninki 43,191 oldu. Her defasında bir toprağın puanı çıkarıldığında ortalama kazanç −0,093 ile +1,175 EMD arasında değişti. H037 ve H038 birlikte çıkarılınca ortalama kazanç +1,471 EMD. Motorola’dan Samsung’a geçişte 16 bölgenin kazancı +3,567 EMD, ters yönde −6,918 EMD oldu. Bu iki yönün işaretleri herhangi bir tek toprağın puanı çıkarıldığında da korundu.

**H037 ve ince elek.** 0,2 mm altında laboratuvar %34,66, ortak ızgaradaki beş alan %57,99, 16 bölge %62,69 tahmin etti. H037’nin tüm eğri EMD’si 63,288’den 70,119’a yükseldi. 24 toprakta 0,2 mm ortalama mutlak hatası beş alanda 17,187, 16 bölgede 16,942, saf v3’te 11,718 yüzde puan. Tam yüzeyin altı doku ortalamasına indirgenmesi bu çekimlerde yarışma referansını geçmedi. Ana model saf v3. Yeni Kaggle gönderimi yok.

**Önceki değerle bağlantı ve doğrulama.** Geçmiş beş alan deneyinin altı özelliği 127 çekimde yeniden hesaplandı. Önceki 66 sınama eğrisi yeniden üretildi (24 toprak 44,670 EMD). Aynı beş alanı doğrudan yeni ortak piksel ızgarasından okuyunca 44,760 EMD çıkıyor. İki yöntemin alan etkisi için tablo bu ikinci beş alan sonucunu kullanıyor. Beş kare, ortak ızgarada fotoğraf başına yüzeyin %4,93–8,54’ünü kapladı (ortanca %7,23). 16 bölümün her ölçek için ölçtüğü parlaklık farkı toplamları ve çift sayıları, 127 fotoğrafın tamamında kesintisiz tam görüntü hesabıyla bire bir eşleşti. Üç düzenin toplam 198 tahmin eğrisi PCA ve Ridge’in ayrı matris hesabıyla yeniden üretildi (en büyük fark 0,000000001 yüzde puanın altında). Saf v3 dahil kayıtlı EMD’ler laboratuvar değerleri ve eğrilerden tekrar hesaplandı. [Toprak bazında fark ve H037 eğrisi](bes_ve_onalti_doku.png) · [198 tahmin ve puan](bes_ve_onalti_doku_sonuclar.csv) · [127 fotoğrafın doku ölçüleri](bes_ve_onalti_doku_fotograflar.csv) · [Deney kodu](bes_ve_onalti_doku.py)

### 16 bölgeyi birlikte eğitme · deney düzeni

Önceki DINOv2 + Ridge deneyinde 16 bölgenin doku bilgisi fotoğraf başına altı ortalamaya indirgenmişti. Yeni karşılaştırmada 16 bölge ayrı kalır. Her bölgenin gri görüntüsünün 32 × 16 piksel özeti ve önceki altı fiziksel uzaklıkta tam çözünürlüklü doku ölçüleri aynı küçük ağa verilir. Bir evrişim katmanı ve bölge temsili laboratuvar hatasından öğrenir. İki sürümün birinde 16 temsile eşit ağırlık verilir, diğerinde küçük bir dikkat ağı her fotoğrafta toplamı bir olan 16 ağırlık öğrenir. Aynı topraktaki fotoğraf temsilleri eşit ortalanır. Her iki modelde önceki sabit genel DINOv2 vektörünün sekiz eğitim-PCA bileşeni de kullanılır. Ağın son katmanı 11 kütle aralığını tahmin eder, birikimli toplam elek eğrisini verir.

Her ayrımda aynı başlangıç tohumu, AdamW, 0,006 öğrenme hızı ve önceden sabit 48 epoch kullanılır. Bir epoch bütün eğitim toprakları üzerinden bir güncellemedir. Kayıp, elekler arası logaritmik aralıklarla ağırlıklı mutlak kümülatif yüzde hatasıdır. Ayarlar saklanan topraklara göre seçilmez. 24 topraklı sınama ile 21 topraklı iki telefon yönünde iki yeni model eşlenir. Ayrıca önceki 16 bölge + Ridge ve saf v3 referansları okunur. İlk iki yeni sürüm arasındaki deney değişkeni bölge ağırlıklandırmasıdır. Doğrulamadan sonra 24 toprağın tamamında yapılan son eğitim ayrı kaydedilir. Saklı H037 çekimine ait ağırlıklar görselleştirilir.

**16 bölgeyi birlikte eğitmenin sonucu.** 127 fotoğrafın her birinde 16 bölge aynı ağdan birlikte geçti. Toplam 2.032 bölgenin bağlı olduğu 24 bağımsız laboratuvar eğrisi var. Bir epoch bütün eğitim topraklarının tek toplu güncellemesi. Hem 24 toprak hem iki telefon yönü için saklanan toprağın bütün çekimleri eğitimden çıkarıldı. İki yerel görüntü ağı, bölge temsili ve son tahmin katmanı 48 epoch eğitildi. Genel görüntü kolundaki DINOv2 sabit kaldı.

| EMD (düşük iyi) | Eşit ağırlıklı ağ | Dikkat ağı | Önceki 16 bölge + Ridge | Saf v3 |
|---|---:|---:|---:|---:|
| 24 toprak | 42,393 | 43,027 | 43,918 | 38,758 |
| Motorola → Samsung | 41,937 | 41,563 | 48,385 | 41,987 |
| Samsung → Motorola | 45,752 | 46,107 | 48,301 | 42,831 |

Eşit ağırlıklı ağ, önceki Ridge’e göre 14/24 toprakta hatayı düşürdü. Ortalama kazanç 1,525 EMD. Tek bir toprağın puanı çıkarıldığında ortalama kazanç 0,239–2,895 aralığında kaldı. İki sürüm arasında dikkat ağı 11/24 toprakta önde. Dikkatin eşit ağırlığa karşı ortalama kazancı −0,634 EMD, tek toprak çıkarıldığında −0,996 ile +0,078. Saf v3 24 topraktaki en düşük ortalama hatayı korudu.

H037’de laboratuvar 0,2 mm altında %34,66 ölçtü. Eşit ağırlıklı ağ %83,54, dikkat ağı %83,92, önceki 16 bölge + Ridge %62,69 tahmin etti. İki ağın H037 tam eğri EMD’si sırasıyla 92,502 ve 92,206. 24 toprakta ince payın ortalama mutlak hatası 17,654 ve 17,868 yüzde puan. Saklı H037’nin ilk Motorola çekiminde dikkat ağı bölgelere %3,2–8,9 ağırlık verdi. Ağırlıklar ağın bölgeleri birleştirme katsayılarıdır. Her karenin laboratuvar kütle payı bilinmiyor.

Arşivlenmiş 132 sınama tahmini için EMD doğrulandı. H037 dışarıda tutularak iki model de tekrar eğitildi ve kayıtlı eğriler, ayrıca bölge ağırlıkları yeniden üretildi. 24 toprağın tamamında eğitilen iki kaydın tekrar açılmasıyla alınan tahminler yeniden eğitim çıktılarıyla eşleşti. [Eğitim kodu](onalti_bolge_egitim.py) · [132 saklı tahmin](onalti_bolge_egitim_sonuclar.csv) · [48 epoch eğitim kayıpları](onalti_bolge_egitim_egitim_kaybi.csv) · [H037 ağırlık haritası](onalti_bolge_egitim_H037.png) · [Eşit ağırlıklı model](onalti_bolge_egitim_esit.pt) · [Dikkat modeli](onalti_bolge_egitim_dikkat.pt)

### 16 bölge · eşit ağırlıklı ağda görüntü çözünürlüğü kontrolü

H037’nin ilk Motorola fotoğrafında 32 × 16 piksellik bölge özetinin her pikseli yaklaşık 2,34 × 2,37 mm’lik alanı temsil ediyor. Altı fiziksel doku ölçüsü ilk deneyde de kalibre edilen daha yüksek çözünürlüklü görüntüden geliyordu. Bu ek kontrolde yalnız CNN’e giren görüntüler her iki eksende iki kat büyütülür (64 × 32). Önceki 32 × 16 özetini büyütmek yerine özgün kalibre edilmiş yüzey yeniden örneklenir. Eşit ağırlıklar, evrişim parametre sayısı, altı doku özelliği, sabit DINOv2, hedef, başlangıç tohumu, 48 epoch ve 24 + 21 + 21 sınama düzeni korunur. 4 × 4 filtre yeni girdide yarı fiziksel genişliği görür. Önceki sonuç görüldükten sonra seçilen bu ek karşılaştırma tek tohumla keşif amaçlıdır. Karar ana EMD, toprak bazında eşlenmiş farklar ve telefon yönleriyle verilir.

**Çözünürlük sonucu.** 64 × 32 ile 24 toprakta 42,379 EMD, 32 × 16 ile 42,393 EMD. Ortalama kazanç yalnız 0,014 EMD. Dokuz toprak iyileşti, 15 toprakta hata arttı. Ortanca kazanç −0,247 ve herhangi bir tek toprak çıkarıldığında ortalama kazanç −0,186 ile +0,107 arasında. Motorola → Samsung yönünde hata 41,937’den 42,512’ye, Samsung → Motorola yönünde 45,752’den 46,679’a yükseldi. Artışın işareti iki yönde de herhangi bir tek toprak çıkarıldığında korundu. Bu ölçekte daha büyük bölge görüntüsüne geçmeyi destekleyen tutarlı bir kazanç görülmedi.

H037’de 0,2 mm ince pay tahmini %83,54’ten %83,02’ye, bütün eğri EMD’si 92,502’den 91,897’ye indi. Gerçek ince pay %34,66. 24 toprakta ince payın ortalama mutlak hatası iki çözünürlükte 17,654 ve 17,609 yüzde puan. 32 × 16 eşit ağırlıklı ağ ile saf v3’ün EMD farkını elek bazında ayırdık. 0,063 mm ve 0,2 mm sırasıyla +2,151 ve +2,958 EMD katkı verdi. Buna karşılık 20 mm’de CNN’in avantajı 1,413 EMD. İki modelin de 0,2 mm’de 12/24 toprakta fazla tahmini var. Tek bir sabit sapma H037’deki fazla tahmini genele açıklamıyor.

**Karar ve doğrulama.** 16 bölge araştırma kolunda 32 × 16, eşit ağırlıklı küçük CNN referansımız. Saf v3 yarışma ana modeli. 16 bölge yaklaşımı sürdürülürse, sonraki sorumuz bölgelere saf v3’ün ince ölçekli doku ölçülerini eklemenin 0,063 mm ve 0,2 mm eleklerinde hatayı indirip bütün eğriyi iyileştirip iyileştirmediği. 64 × 32 düzeninde 66 saklı tahmin eğrisi, grup ayrımları ve EMD’ler denetlendi. Tam eğitim kaydı tekrar açılarak çıktı eşitliği, H037’yi ayırarak tekrar eğitildiğinde de kayıtlı eğri doğrulandı. Görüntü bloklarını işleyen matris işlemi standart PyTorch evrişimiyle sayısal olarak eşleşti. Makaledeki dikkat fikri toprağın sürekli elek eğrisini tahmin etmeye uyarlandı. [İlham alınan dikkat modeli](https://proceedings.mlr.press/v80/ilse18a.html) · [Gerçek CNN girdileri](onalti_bolge_egitim_cozunurluk_H037.png) · [64 × 32 sonuçları](onalti_bolge_egitim_64x32_sonuclar.csv) · [Tam veriyle eğitilen 64 × 32 model](onalti_bolge_egitim_64x32_esit.pt)

<a id="kaggle-surumleri"></a>

### 22 Kasım · İyi public puanlı açık Kaggle notebookları

Kaggle’ın Code sekmesinde görünen public EMD puanlarını ilgili notebook sürümleriyle eşleştirdik. Karşılaştırma için saf v3 yazarının bildirdiği public EMD 40,90, kendi en iyi gönderimimizdeki public EMD 41,67360. Lider tablosundaki 0,00000 puanlı gönderimlerin yöntemini açık kodla eşleştiremedik. [Yarışmanın kodları](https://www.kaggle.com/competitions/soil-grain-size-from-photos/code) · [Lider tablosu](https://www.kaggle.com/competitions/soil-grain-size-from-photos/leaderboard)

| Public EMD | Yayımlanan yöntemin saf v3’ten farkı |
|---:|---|
| 19,42515 | Swathi, tahmini 2 mm geçişine göre kuma veya çakıla uygun bir eğri seçiyor. Bir adayda 0,063 mm altı için tek bir eğitim toprağından %2,78 alıyor. Sürüm birkaç dosya üretiyor, hangisinin bu puanı aldığı belirsiz. [Swathi V6](https://www.kaggle.com/code/swathiha/photos-to-sieve-curves-the-fines-question?scriptVersionId=370964352) |
| 22,21172 | prvsiyan, Sadam ve Swathi’nin yayımladığı iki tahmini birleştiriyor. [prvsiyan V21](https://www.kaggle.com/code/prvsiyan/trying-some-soil-stuff?scriptVersionId=368162415) |
| 22,28153 | Sadam, elek eğrisini üç bileşenle özetleyip Ridge ile öğreniyor. Eğitim aralığı dışındaki doku ölçekleri için ayrı bir çap tahmini kullanıyor. Yazar bu adımda public EMD’nin 51’den 25,6’ya indiğini bildiriyor. [Sadam V4](https://www.kaggle.com/code/sadamtorres/soil-psd-physics-not-deep-learning?scriptVersionId=366985271) |
| 36,18733 | SAM ile ölçülen iri taşları kullanarak 20 ve 63 mm geçişlerini düzeltiyor. Yazar on test toprağından yalnızca Münster tahmininin değiştiğini bildiriyor. [SAM ile iri taş ölçümü V3](https://www.kaggle.com/code/takahitomizunobyts/soil-texture-model-sam-cobble-measurement?scriptVersionId=355111280) |
| 40,22430 | Aynı yerel sınamada karşılaştırılan regresyonlar arasında Elastic Net seçiliyor. [Soil grain V4, notebook sürümü 2](https://www.kaggle.com/code/krishnayadav456wrsty/soil-grain-v4?scriptVersionId=347775378) |

### 22 Kasım · H374 ve H615'te 20 mm üzerindeki görünür taş alanı

Sekiz çekimin önceden kaydedilmiş ImageGrains sınırlarını kullandık. Her çekimde beş adet 52 mm'lik kesitte kısa ekseni 20 mm'yi aşan adayları seçtik. Yüzey yüzdesini her kesitin ortasındaki 26 × 26 mm içinde ölçtük, sonra çekimlere eşit ağırlık verdik. Aynı maskelerle eski 6,3 mm alan sonuçlarını sekiz çekimde doğruladık.

| Toprak | 20 mm üzeri gerçek kütle | Saf v3, 23 referans | Ölçülen görünür alan | Pozitif kare |
|---|---:|---:|---:|---:|
| H374 | %23,51 | %3,10 | %12,52 | 6/15 |
| H615 | %8,86 | %30,79 | %0,00 | 0/25 |

İki toprağın sırası laboratuvarla uyuştu. H374'te kesit sınırına değen adaylar dışarıda kalınca ortalama %3,37'ye indi. H615'te iki telefonda da ölçülen sıfır, laboratuvardaki pozitif iri kütle payını karşılamadı. H374 yalnız Edge 60 Fusion ile çekildi. Bu iki örnek yöntemin geliştirilmesinde kullanıldı. Kütleyi tahmin eden bir düzeltme eğitilmedi, yeni EMD ölçülmedi. Sonraki aday, daha geniş toprak kadrajında aynı fiziksel ölçüyü kontrol etmek ve 20 mm eleği için saf v3'e katkısını saklı topraklarda sınamak. [Görsel](h374_h615_20mm_alan.png) · [Çekim sonuçları](h374_h615_20mm_alan.csv) · [Hesap](h374_h615_20mm_alan.py)

### 22 Kasım · Sekiz tam kadrajda H615’in sıfır alan bulgusunu inceleme

Sekiz fotoğrafın önceden kaydedilen toprak kadrajına eski beş ölçüm karesini, onları çevreleyen kesitleri ve arşivlenmiş 20 mm üstü aday sınırlarını taşıdık. Özgün görüntü ile 40 kesitin piksel eşleşmesini ve önceki sekiz çekimin alan yüzdelerini doğruladık. Beş ölçüm karesi kullanılabilir yüzeyin H374’te %5,93’ünü, H615’te %5,44–7,38’ini kapsıyordu.

H615’in ikinci Motorola ve ikinci Samsung çekiminde benzer biçimli belirgin bir taşı iki görünümde ayrı ayrı elle izledik. Görüntü düzleminde sınırı içine alan en dar iki paralel çizgi arasındaki mesafe 26,5 ve 25,5 mm. İşaretli alanın eski ölçüm karelerine düşen payı sırasıyla %0,06 ve %0,00. Daha geniş 52 mm kesitlerin tümünde bile aday alanının yalnızca %17,3 ve %5,9’u vardı. Bunlar aynı taşın iki görünümü olabilir. H374’ün ilk çekiminde soldaki büyük taşın yalnızca bir parçası eski kesite giriyor. Laboratuvar elek çapı ile yüzeyde ölçülen en arasındaki ilişki ve bütün fotoğraftaki toplam iri pay henüz ölçülmedi.

Sonraki sınama için bütün toprak kadrajında taş sınırlarını bulup H615’te elle işaretlenen bu adayın tek parça yakalanıp yakalanmadığını kontrol edeceğiz. [Sekiz fotoğraf](h374_h615_tam_kadraj.png) · [Yakın görünüm](h374_h615_tam_kadraj_detay.png) · [Çekim ölçümleri](h374_h615_tam_kadraj.csv) · [Hesap ve elle izlenen sınırlar](h374_h615_tam_kadraj.py)

### H615 otomatik sınır kontrolü ve koşullu 20 mm deneyi

İki telefonun ikinci çekiminde önceden çizilmiş aday sınırını kullanıyoruz. ImageGrains’in kayıtlı ağırlıkları, önceki tam yüzey deneyindeki 4,5 piksel/mm ve aynı maske ayarlarıyla kullanılabilir toprak yüzeyinin tamamı işlenir. Önceki maske kontrolündeki gibi tek bir otomatik taş ile elle çizilen alanın kesişim/birleşim oranına (IoU) bakılır. Her iki telefonda da IoU en az 0,50 olmalı ve eşleşen taşın otomatik kısa ekseni 20 mm’yi geçmeli. Ayrıca en iyi parçanın elle işaretlenen alanı ne kadar kapladığı, ikinci parçanın payı ve etiketsiz kalan alan raporlanır.

İki çekim geçerse tüm eğitim çekimlerinde 20 mm üzeri adayların görünür alanı toprağa göre ortalanır. Denenecek basit düzeltme, görünen alanı kütle payına bağlayan negatif olmayan tek bir çarpanı diğer eğitim topraklarında en küçük karelerle öğrenir. Görünür alan sıfırsa v3 tahmini kullanılır. Düzeltilen 20 mm geçişi v3’ün komşu 6,3 ve 63 mm geçişlerinin arasında tutulur. Telefonlar arası sınamada çarpan kaynak telefonun çekimlerinden öğrenilir. Karşılaştırma için önceden saklanmış saf v3 eğrileri kullanılır.

**Yerel sonuç.** Motorola’da en iyi otomatik maske elle çizilen alanın %94,04’ünü kapsadı. IoU 0,915 ve otomatik kısa eksen 24,73 mm. Samsung’da karşılıkları %98,23, 0,938 ve 24,96 mm. İkinci otomatik parça elle çizilen alanın sırasıyla yalnız %1,16 ve %0,48’ini içerdi. Kontrol aynı adayın iki görünümü ile sınırlı. [Sınırlar](h615_otomatik_tas.png) · [İki çekimin sayıları](h615_otomatik_tas.csv) · [Yerel kontrol kodu](h615_otomatik_tas.py)

**Kaggle’a devir.** Kullanıcının seçimiyle 127 fotoğraf için GPU notebook’u yerelde hazırlandı. Yarışma verisi eklendiğinde hücreler sırasıyla 66 arşivlenmiş saf v3 puanını yeniden hesaplar, Zenodo ağırlığını doğrular, H615’in iki çekimini GPU’da tekrar sınar ve 24 toprak ile iki telefon yönünde 20 mm düzeltmesini değerlendirir. H374 ve H615’in puanları çıkarılarak kalan 22 toprak ayrıca raporlanır. Okunacak çıktı `soil20/ozet.csv`. Çalıştırma ve sonuç kontrolü 23 Kasım’da tamamlandı. [Çalıştırılabilir notebook](v3_tam_yuzey_20mm_kaggle.ipynb)

**Beklenebilecek kazancın üst sınırı.** Arşivlenmiş saf v3 eğrisinin diğer on eleğini sabit tutup saklı toprağın gerçek 20 mm değerini komşu iki elek arasına kırptığımızda 24 toprakta EMD 36,146 oluyor. Dolayısıyla bu tek eleği değiştirerek kazanılabilecek en yüksek ortalama 2,612 EMD. Bu hesap laboratuvar cevabını kullanır ve yeni model puanı olarak yorumlanamaz.

<a id="gpu20"></a>

### 23 Kasım · Tam yüzeyde 20 mm düzeltmesinin Kaggle GPU sonucu

Hazır ImageGrains modeli, 127 çekimin 48’inde kısa ekseni 20 mm’yi aşan en az bir aday buldu. Bunlar 24 toprağın 10’una aitti. H615’te elle çizilen sınırla örtüşme Motorola’da 0,915, Samsung’da 0,938 oldu. GPU’dan indirilen görüntü ölçümlerini yerel değerlendirme koduna verdiğimizde 66 saklı sonucun ve dört özet satırının tümü 0,0000001 tolerans içinde eşleşti. [Fotoğraf ölçümleri](v3_tam_yuzey_20mm_fotograflar.csv) · [GPU sınır kontrolü](h615_gpu_kontrol.csv)

| Yerel sınama | Saf v3 EMD | Düzeltilmiş EMD | Ortalama kazanç | İyileşen / aynı / kötüleşen toprak |
|---|---:|---:|---:|---:|
| 24 toprak | 38,758 | 37,396 | 1,361 | 7 / 14 / 3 |
| H374 ve H615 puanları hariç | 38,802 | 38,084 | 0,718 | 5 / 14 / 3 |
| Motorola → Samsung | 41,987 | 41,558 | 0,429 | 5 / 14 / 2 |
| Samsung → Motorola | 42,831 | 42,383 | 0,448 | 4 / 14 / 3 |

24 toprakta 20 mm geçişinin ortalama mutlak hatası 6,762’den 4,030 yüzde puana indi. H615’te gerçek iri kütle payı %8,86 iken tahmin %30,79’dan %7,29’a, H374’te gerçek %23,51 iken %3,10’dan %16,61’e geldi. Diğer 22 toprağın puanlarında da ortalama kazanç korundu. 24 puandan herhangi biri çıkarıldığında ortalama kazanç 0,980–1,566 EMD aralığında kaldı. Samsung → Motorola yönünde H615’in puanı çıkarılırsa ortalama kazanç −0,003 EMD’ye düşüyor. [Toprak bazında sonuçlar](v3_tam_yuzey_20mm_sonuclar.csv) · [Dört sınamanın özeti](v3_tam_yuzey_20mm_ozet.csv)

**Karar.** Saf v3 referansımız. 20 mm düzeltmesi ayrı bir gönderim adayı. Sıradaki karşılaştırma, çarpanı 24 eğitim toprağında sabitleyip 10 test toprağı için iki sürümün tahminlerini yan yana incelemek. Bu deneyin puanı eğitim topraklarını sırayla ayırarak hesaplandı. Yeni bir public puan elde etmek için ayrıca gönderim gerekir.

### 23 Kasım · 10 test toprağı için gönderim adayı

127 eğitim çekimindeki alan yüzdelerini toprak bazında ortalayıp 24 laboratuvar ölçüsünden tek bir çarpan öğrendik (1,975284203). Kaggle GPU üzerinde aynı ImageGrains ağırlıklarıyla 35 iPhone fotoğrafını işledik. Ölçeği telefon modeline göre ayarladık. Tüm kadraj koordinatları önceki yöntemin yerel koduyla birebir eşleşti. 14 çekimde kısa ekseni 20 mm'yi aşan aday bulundu. Saf v3 kaynak dosyasının Git içerik kimliğini, şablon satır sırasını, 11 elek sütununu, yüzde sınırlarını ve eğrilerin azalmadığını doğruladık.

| Toprak | Aday bulunan çekim | Ortalama görünür iri alan | Saf v3, 20 mm geçiş | Düzeltilmiş 20 mm geçiş | Fark (yüzde puan) |
|---|---:|---:|---:|---:|---:|
| Münster | 5/5 | %44,93 | %59,9571 | %35,4911 | −24,4660 |
| Testfeld Lidl WHV | 4/4 | %20,12 | %80,5039 | %60,2591 | −20,2448 |
| Audorfring | 5/5 | %2,34 | %99,7560 | %97,8411 | −1,9149 |

Diğer yedi toprakta tüm eğri saf v3 ile aynı. Eğitim toprakları içinde en yüksek ortalama görünür iri alan %15,22 idi. Münster ve Testfeld bu aralığı aşıyor. Münster ve Audorfring'de komşu 6,3 mm eleğinin geçişi alt sınırı belirledi. Münster’in beşinci çekiminde cetvel kadrajda ve kenarı turkuaz görünüyor. Bu çekim çıkarıldığında diğer dört çekimde ortalama alan %46,92 ve sınırlanmış 20 mm tahmini yine %35,4911. Gönderim adayında 35 çekimin tamamı kullanıldı. 23 Kasım’da paylaşılan Kaggle ekranında bu dosya **Complete** ve public EMD **37,15842**. Önceki en iyi gönderimimiz ortancalı v3 (41,67360) ile aradaki fark 4,51518 EMD. Public puan toplu verildiği için üç toprağın ayrı ayrı hata değişimini bilmiyoruz.

[Gönderim dosyası](submission_v3_tam_yuzey_20mm.csv) · [10 toprak karşılaştırması](v3_20mm_test_karsilastirma.csv) · [35 çekimin taş sınırları](test20_kontrol.jpg) · [Çekim ölçümleri](test20_fotograflar.csv) · [Katsayı ve dosya üretimi](v3_20mm_gonderim.py)

<a id="cokluelek"></a>

## 23 Kasım · Tam yüzeyde başka eleklerin düzeltilmesi

Kaggle oturumundan alınan 162 maskede (127 eğitim, 35 test çekimi) her tanenin kısa eksenini 4,5 piksel/mm üzerinden ölçtük. Önceki 20 mm ölçümündeki aday sayısını ve alanı her fotoğrafta yeniden ürettik. Alan yüzdesini fotoğraflar arasında toprak başına ortaladık. Her eleğin alan–kütle katsayısı yalnız o turdaki eğitim topraklarından geldi. Görünür alan sıfırsa o elekte önceki tahmin kullanıldı. Birlikte değişen komşu eleklerin geçiş yüzdelerini sıralı tuttuk ve önceki 20 mm değerini koruduk. Arşivdeki 66 referans EMD’si de yeniden üretildi.

| Sürüm · düşük EMD iyi | 24 toprak | Motorola → Samsung | Samsung → Motorola | En iri 5 birlikte saklı |
|---|---:|---:|---:|---:|
| Saf v3 | 38,758 | 41,987 | 42,831 | 29,821 |
| v3 + 20 mm | 37,396 | 41,558 | 42,383 | 27,408 |
| Buna 2 mm ekle | 36,617 | 41,898 | 41,597 | 22,603 |
| Buna 6,3 mm ekle | 36,355 | 41,069 | 42,418 | 23,959 |
| Buna 2 ve 6,3 mm ekle | **35,711** | **41,408** | **41,619** | **18,995** |
| Buna 0,63 mm maske ölçümü ekle | 37,840 | 41,580 | 42,993 | 24,172 |
| Buna ince doku tahmini ekle | 43,380 | 44,149 | 45,574 | 41,986 |
| Koşullu çap ve 20 mm | 38,710 | 46,365 | 47,171 | 50,381 |

Birleşik deneyde 63 mm eşiği de vardı. 162 fotoğrafta bu eşiği geçen maske bulunmadığı için dosya yalnız 2 ve 6,3 mm sütunlarını değiştiriyor. 24 toprakta 15 kazanç, 9 kayıp var. Ortanca kazanç 0,424 EMD. En çok iyileşen toprak H374 (+22,888), en çok kötüleşen H405 (−4,690). Tek bir puanı dışarıda bıraktığımızda en düşük ortalama kazanç 0,763. Geliştirmede baktığımız H374 ve H615 birlikte çıkarılınca kazanç 0,627. 2 ve 6,3 mm geçiş oranlarındaki ortalama mutlak hata sırasıyla 8,755 → 7,192 ve 8,310 → 6,503 yüzde puan. Beş iri toprakta D50 mutlak hatası 2,393 → 1,349 mm.

İnce doku deneyi, mevcut doğal çözünürlüklü gri ton granülometrisinden altı özellik alıp 0,063 / 0,2 / 0,63 mm geçişlerini Ridge ile öğrendi. Tahminleri referansla yarı yarıya birleştirdik. Bu sürümde 18 toprağın hatası arttı. Maskedeki 0,63 mm eşiği yaklaşık 2,8 piksel, 0,2 mm ise 0,9 piksel. Daha küçük elekler için doğrudan maske alanı kullanmadık.

Koşullu çap deneyinde gri ton granülometrisinin %50 noktasından bir görüntü çapı hesapladık. Toprağın ortanca görüntü çapı eğitimdeki en yüksek değeri geçince, eğitimdeki laboratuvar/görüntü çap oranlarının ortancasıyla kalibre edip v3 eğrisini log-çap ekseninde kaydırdık ve 20 mm düzeltmesini uyguladık. Bu bizim uyarlamamız. Beşli sınavda 19 referans kullanıldı. Yalnız H615 kuralı tetikledi ve grupta D50 mutlak hatası 2,393’ten 3,235 mm’ye çıktı.

**Kaggle sonucu.** 2 + 6,3 + 20 mm gönderimi tamamlandı. Public EMD 37,68440. Yalnız 20 mm düzeltmesiyle aldığımız 37,15842 bu aşamadaki en iyi public puanımızdı. İki ek elekle hata yerelde 1,685 azalırken public sınamada 0,52598 arttı.

On test toprağının onunda da tahmin değişti. 6,3 mm eleğinden geçtiği tahmin edilen kütle payı Testfeld’de %52,33’ten %17,13’e, Audorfring’de %97,84’ten %74,40’a, Münster’de %35,49’dan %12,64’e indi. Münster ve Testfeld her iki yeni elekte de eğitimdeki en yüksek ortalama görünür alanı aşıyor. Audorfring ise her ikisinde de eğitim aralığında. Public puan tek başına hangi numunenin veya eleğin hatayı artırdığını göstermiyor. Sonraki yerel kontrol, bu üç toprağın 2 ve 6,3 mm maskelerini çekim bazında incelemek.

[Gönderim adayı](submission_v3_tam_yuzey_2_6_3_20mm.csv) · [Testte değişen tahminler](coklu_elek_test_kontrol.csv) · [Numune tahminleri](coklu_elek_sonuclar.csv) · [Elek hataları ve özet](coklu_elek_ozet.csv) · [Deney kodu](coklu_elek_deneyi.py) · [Maske ölçüm kodu](maske_elek_alanlari.py) · [Çekim başına alanlar](maske_elek_alanlari.csv)

### 23 Kasım · Üç test toprağında taş sınırları ve kütleye dönüşüm

Testfeld’in 4, Audorfring’in 5 ve Münster’in 5 iPhone çekiminde özgün kadraja maske sınırlarını bindirdik. 14 çekimin iki eleğinde de alan yüzdeleri ve aday sayıları eski ölçümlerle eşleşti. Bütün çekimler dahilken yayımlanan iki tahmin sütununu üç toprakta yeniden ürettik. Taş sınırları görsel olarak incelendi. Test numunelerinde doğrulanmış elle çizilmiş sınırlar veya laboratuvar etiketleri yok.

| Toprak | 6,3 mm üzeri görünür alan, çekim aralığı | 6,3 mm geçişi, eski → ham hesap → gönderilen | Bir çekim çıkarılınca 6,3 mm geçişi |
|---|---:|---:|---:|
| Testfeld | %61,75–72,01 | %52,33 → %13,71 → %17,13 | %17,13–18,14 |
| Audorfring | %15,58–21,75 | %97,84 → %74,40 → %74,40 | %73,19–75,24 |
| Münster | %61,77–84,40 | %35,49 → %1,17 → %12,64 | %12,64–13,10 |

Testfeld’in dört çekiminde ve Münster’in beşinde iri taşlar seçiliyor. İki toprağın tüm çekimlerinde 6,3 mm üzeri alan eğitim topraklarının en yüksek ortalamasını (%43,15) aşıyor. Münster’in beşinci fotoğrafında cetvel kenarı da kontur aldı. Bu çekimi çıkarınca 6,3 mm son tahmini yine %12,64. Münster’de kimi farklı çekimlerde aynı taşlar ayırt edilebiliyor. Tek tek fotoğraf çıkarma kontrolü, bir çekimin ortalamaya etkisini ölçüyor.

6,3 mm eleğinde öğrendiğimiz katsayı yaklaşık 1,33. Geçişi 100 − katsayı × alan yüzdesiyle hesaplıyoruz. Münster ve Testfeld’de 6,3 mm’nin ham geçişi 2 mm’nin altına indi. Elek sırası ve sabit 0,63 mm alt sınırı uygulandığında iki yeni elek de 0,63 mm ile aynı değere geldi. Dolayısıyla gönderilen eğri her iki örnekte 0,63–6,3 mm aralığına sıfır kütle ayırdı. Audorfring’de ise 6,3 mm tahminindeki 23,44 puanlık düşüş doğrudan alan-kütle dönüşümünden geliyor. 2 mm değeri ayrıca 0,63 mm’ye eşitlendi.

**Sonraki yerel deney önerisi.** 2 ve 6,3 mm düzeltmesini v3 eğrisine kısmen uygulayan sabit bir sürümü, aynı 24 toprak, iki telefon yönü ve beş iri toprak sınamasında tam düzeltme ve 20 mm referansıyla karşılaştır. İncelenecek ölçüler tüm eğrinin EMD’si ve düzeltilen eğride 0,63–6,3 mm aralığına sıfır kütle atanma sıklığı. Bu incelemede yeni gönderim veya puan üretilmedi.

[Testfeld çekimleri](uc_toprak_elek_testfeld.jpg) · [Audorfring çekimleri](uc_toprak_elek_audorfring.jpg) · [Münster çekimleri](uc_toprak_elek_muenster.jpg) · [Çekim çıkarma hesapları](uc_toprak_elek_kontrol.csv) · [İnceleme kodu](uc_toprak_elek_kontrol.py)

<a id="ortaelek"></a>

### Orta boy taneler · Yarı düzeltme ve üç boyut aralığı

Yarı düzeltme, 2 ve 6,3 mm'de 20 mm referansı ile geçerli tam düzeltme eğrisinin tam ortasıdır. İkinci yöntem 0,63–2 / 2–6,3 / 6,3–20 mm tane alanlarını birikimli maske ölçümlerinin farkından hesaplar. Her aralığın görünür alanını laboratuvar kütlesine bağlayan üç ayrı katsayıyı yalnız eğitim topraklarından öğrenir. Üç tahmini kendi toplamlarına bölerek oranlara çevirir ve 0,63–20 mm arasındaki mevcut kütleyi bu oranlarla paylaştırır. Görüntüden kullanılabilir pay çıkmazsa referans eğrisi kullanılır. İki sürümde de yalnızca 2 ve 6,3 mm sütunları değişir. Karışım oranı ve aralıklar sınama sonuçlarıyla seçilmedi. Önceki iki modelin 71'er tahmini ve EMD'si yeniden doğrulandı.

| Yöntem | 24 toprak | Motorola → Samsung | Samsung → Motorola | En iri 5 birlikte saklı |
|---|---:|---:|---:|---:|
| v3 + 20 mm | 37,396 | 41,558 | 42,383 | 27,408 |
| Tam düzeltme | 35,711 | 41,408 | 41,619 | 18,995 |
| Yarı düzeltme | 36,167 | 41,140 | 41,945 | 22,231 |
| Üç boyut aralığı | 35,486 | 41,311 | 42,173 | 20,758 |

Aralık yöntemi 20 mm referansına göre 17/24 toprakta iyileşti. Tek bir puan çıkarılınca kalan ortalama kazanç en az 1,117 EMD. H374 ve H615 birlikte çıkarılınca kazanç 0,960. Tam düzeltmeye göre ayrıca bakıldığında 11 toprakta kazanç ve 13 kayıp var. Ortalama üstünlük 0,225 EMD. 2–6,3 mm kütle payının 24 topraktaki ortalama mutlak hatası sırasıyla 5,887 / 5,303 / 4,721 / 4,505 yüzde puan oldu.

24 toprakta dört sürümün de 0,63–6,3 mm payı sıfırdan büyüktü. On yarışma toprağında ise tam düzeltme iki kez sıfır verdi, yarı düzeltme ve aralık yöntemi sıfır vermedi. Münster ve Testfeld için bu aralıkta yarı düzeltme sırasıyla %11,43 ve %17,60, aralık yöntemi %3,85 ve %9,93 kütle tahmin ediyor. Bunlar etiketsiz test tahminleri. Yarı düzeltme iki telefon yönünde de aralık yöntemini geçti.

**Kaggle sonucu.** İki gönderim de tamamlandı. Üç boyut aralığı 35,63420, yarı düzeltme 35,73986 public EMD aldı. Önceki en iyi gönderime (37,15842) göre hata sırasıyla 1,52422 ve 1,41856 azaldı. Aralarındaki fark 0,10566. Üç boyut aralığı yeni karşılaştırma modelimiz.

[Aralık gönderimi](submission_v3_orta_elek_aralik.csv) · [Yarı düzeltme gönderimi](submission_v3_orta_elek_yari.csv) · [Sınama özeti](orta_elek_kismi_ozet.csv) · [Numune sonuçları](orta_elek_kismi_sonuclar.csv) · [Test tahminleri](orta_elek_kismi_test.csv) · [Deney kodu](orta_elek_kismi_duzeltme.py)

### En iyi iki gönderimin ortak hatası · 24 saklı toprak

Üç boyut aralığı ve yarı düzeltme için her toprakta elek geçiş hatasının mutlak değerini bir sonraki eleğe kadar olan log10 çap genişliğiyle çarptık. Katkıların toplamı her iki modelde dört yerel sınamadaki 71'er arşiv EMD’sini yeniden verdi. Modellerin yalnızca 2 ve 6,3 mm tahminleri farklı. İnce eleklerdeki hatalar birebir ortak.

| Elek | 24 toprakta ortalama mutlak hata (yüzde puan) | Ortalama EMD katkısı | En az 10 yüzde puan hata görülen toprak |
|---|---:|---:|---:|
| 0,063 mm | 13,61 | 6,83 | 11/24 |
| 0,2 mm | 11,72 | 5,84 | 10/24 |
| 0,63 mm | 9,89 | 4,96 | 5/24 |

Bu üç elek 17,63 EMD ile üç boyut aralığı modelinin yerel hatasının %49,68’ini, yarı düzeltmenin %48,75’ini oluşturuyor. 0,063 mm her tek-toprak çıkarımında birinci. H037/H038 birlikte çıkarılınca da ilk sırada (6,56 EMD). 0,063 mm’de 14 fazla ve 10 eksik tahmin var. Ortalama işaretli hata +0,31 yüzde puan. H616’da tahmin %48,67 ve laboratuvar %2,78; H405’te tahmin %35,04 ve laboratuvar %75,66. Motorola → Samsung sınavında en büyük katkı 0,2 mm, ters yönde 0,063 mm, beş iri toprak sınavında ise 6,3 mm.

**Sonraki yerel öneri.** Üç boyut aralığı sürümünde yalnız 0,063 mm geçişini incele. 0,02 ve 0,2 mm arasındaki kütlenin ne kadarının 0,063 mm altında kaldığını yalnız eğitim numunelerinden tahmin et. İki uçta mevcut geçişleri koru. Sabit bir oran ile aynı oranı görüntü özelliklerinden öğrenen sürümü karşılaştır. 24 saklı toprakta tüm eğri EMD’sini, 0,063 mm mutlak hatasını ve iki telefon yönünü raporla. Bu analizde yeni model eğitilmedi veya gönderim yapılmadı.

[Elek bazında özet](iki_model_elek_ozet.csv) · [Toprak, elek ve model sonuçları](iki_model_elek_toprak.csv) · [Hata haritası](iki_model_elek_haritasi.png) · [Hesaplama kodu](iki_model_elek_hatasi.py)

### 0,063 mm · Komşu elekler sabitken ortak oran ve görüntü oranı

Üç boyut aralığı sürümünün her saklı toprak için arşivlenen eğrisinde yalnız 0,063 mm tahminini değiştirdik. Eğitim hedefi, 0,02–0,2 mm arasındaki kütlenin 0,063 mm altındaki payıydı. Ortak oran, payların ilgili aralıktaki laboratuvar kütlesiyle ağırlıklı ortalaması. Görüntü sürümünde aynı hedef ve ağırlıklarla altı gri ton doku özelliği kullanan Ridge eğitildi. Toprak başına özellikler fotoğrafların ortancası. Ölçekleme, ortak oran ve Ridge yalnız eğitim tarafında hesaplandı. Ridge cezası etkin eğitim numunesi sayısına eşitlendi ve sonuç 0–1 arasına sınırlandı. Her iki sürüm de saklı toprağın alt ve üst uçları için mevcut modelin tahminlerini kullandı.

| Sınama | Referans EMD | Ortak oran EMD | Doku + Ridge EMD |
|---|---:|---:|---:|
| 24 toprak | 35,486 | 35,652 | 35,726 |
| Motorola → Samsung | 41,311 | 41,180 | 41,065 |
| Samsung → Motorola | 42,173 | 42,077 | 42,193 |
| En iri beş birlikte saklı | 20,758 | 21,131 | 21,202 |

24 toprakta 0,063 mm ortalama mutlak hata sırasıyla 13,613 / 13,943 / 14,090 yüzde puan. İki deney de referansa göre 10 toprakta kazanç ve 14 toprakta kayıp verdi. H037/H038 hariç kalan 22 toprakta EMD 29,157 / 29,145 / 29,219. Her senaryoda referans toplam EMD'si ve tek elek değişiminin EMD'ye katkısı doğrulandı.

**Komşu eleklerin sınırı.** 24 toprağın 7’sinde gerçek 0,063 mm geçişi, tahmin edilen 0,02 ve 0,2 mm geçişlerinin dışında. H616’da alt sınır %26,14, gerçek %2,78. H405’te üst sınır %63,23, gerçek %75,66. Diğer uçta H372 yalnızca 0,03 yüzde puan dışarıda. Uçlar sabitken en iyi mümkün 0,063 mm mutlak hatasının 24 toprak ortalaması 2,324 yüzde puan. Bu, etiketlerle hesaplanan kuramsal alt sınırdır. Uçlardaki hata tek başına mevcut 13,613 puanın tamamını açıklamıyor.

**Karar.** Yeni gönderim hazırlanmadı. Public referans üç boyut aralığı, 35,63420. Sonraki deney önerisi, 0,02 / 0,063 / 0,2 mm noktalarını birlikte ve sıralı biçimde güncellemeyi aynı saklı topraklarda sınamak. Önceki tüm eğri kütle payı deneyinden farklı olarak yalnız bu üç nokta hareket edecek.

[Eşleştirilmiş numune sonuçları](ince_0063_oran_sonuclar.csv) · [Sınama özeti](ince_0063_oran_ozet.csv) · [Deney kodu](ince_0063_oran.py)

### 0,02 / 0,063 / 0,2 mm birlikte · aynı saklı topraklarda sonuç

Önceki oran deneyinin dış sınırlarını 0,0063 ve 0,63 mm’ye genişlettik. Aradaki her elek için bu aralıktaki kütlenin ne kadarının o elekten geçtiğini öğrendik. Birinci sürüm, eğitim toprağının aralıktaki kütlesiyle ağırlıklandırılmış üç ortak oran kullandı. İkinci sürüm aynı ağırlıklar, aynı altı gri ton doku özelliği ve aynı Ridge ceza kuralıyla üç oranı birlikte öğrendi. Tahminleri komşu eleklerin sınırları içinde ve küçükten büyüğe sıraladık. Her turda saklanan toprağın tüm çekimleri eğitim dışında kaldı. 71 saklı numune/senaryo için diğer sekiz elek ve toplam EMD değişiminin üç elek hatasından geldiği doğrulandı.

| Yerel EMD | Üç boyut aralığı referansı | Ortak oranlar | Doku + Ridge |
|---|---:|---:|---:|
| 24 toprak | 35,486 | 36,961 | 36,772 |
| Motorola → Samsung | 41,311 | 40,500 | 40,359 |
| Samsung → Motorola | 42,173 | 41,862 | 41,873 |
| En iri beş birlikte saklı | 20,758 | 26,357 | 24,582 |

| 24 toprakta ortalama mutlak hata, yüzde puan | Referans | Ortak oranlar | Doku + Ridge |
|---|---:|---:|---:|
| 0,02 mm | 9,443 | 9,236 | 9,359 |
| 0,063 mm | 13,613 | 14,425 | 14,584 |
| 0,2 mm | 11,718 | 14,067 | 13,403 |

Ortak oranlar 9/24, Ridge 7/24 toprakta toplam hatayı azalttı. Deneye gerekçe olan H405 ile H616 birlikte çıkarılınca kalan 22 toprakta EMD sırasıyla referans 32,547, ortak oranlar 34,456, Ridge 34,109. H405’te 0,2 mm üst sınırı %63,23’ten ortak oranlarla %69,93’e ve Ridge ile %68,89’a yükseldi. Gerçek 0,063 mm geçişi %75,66 ile iki yeni sınırın da üstünde. H616’da 0,02 mm alt sınırı %26,14’ten sırasıyla %25,82’ye ve %26,98’e gitti. Gerçek 0,063 mm geçişi %2,78. Bu örnekte sabit dış sınırın kendisi (0,0063 mm tahmini %11,00) gerçek 0,063 mm değerinden büyük. Tek eleği değiştiren önceki deneyde ortak oran 35,652 ve doku 35,726 EMD vermişti.

**Karar.** 24 toprakta iki yeni sürüm de referansın gerisinde kaldı. İki telefon yönündeki ortalama iyileşmeye rağmen iri beşli sınavda da hata büyüdü. Yeni Kaggle adayı hazırlanmadı. Public referans 35,63420. [Deney kodu](ince_uc_elek_ortak.py) · [Sınama özeti](ince_uc_elek_ozet.csv) · [Toprak ve elek bazında tahminler](ince_uc_elek_sonuclar.csv)

<a id="incekomsu"></a>

### Duraklama noktası · Görüntüde en yakın toprak ve 0,063 mm laboratuvar farkı

24 toprağın 127 özgün çekimini kullandık. Her toprağın fotoğraf ölçülerinin ortancasını aldık. Bir toprağı sorgularken kalan 23 toprakla standartlaştırma yapıp Öklid uzaklığı en küçük komşuyu seçtik. Laboratuvar ölçüleri yalnız seçimin ardından karşılaştırmaya girdi. Önceki komşu deneyleri tüm elek eğrisini inceliyordu. Bu karşılaştırma özellikle 0,063 mm’den geçen kütlenin yüzde puan farkını ölçüyor.

| Özellikler | En yakın komşuyla ortalama mutlak fark | Ortanca fark | Farkı en az 20 puan olan sorgular |
|---|---:|---:|---:|
| v3’ün 30 ölçüsü | 15,93 | 7,09 | 6/24 |
| Önceki ince elek Ridge deneyindeki 6 doku ölçüsü | 16,52 | 13,47 | 7/24 |

Her toprak için diğer 23 olası komşunun tümü üzerinden rastgele seçimin beklenen farkı 31,70 yüzde puan. Aynı çift iki yönde seçilebildiğinden 24 sorgu, v3’te 21 ve doku ölçülerinde 17 ayrı çift oluşturdu. Komşu seçimleri sabitken 10.000 ortak rastgele etiket permütasyonunda gözlenen ortalama farka veya daha küçüğüne ait tek yönlü, düzeltilmemiş p değerleri 0,0009 ve 0,0025 oldu. Bu sonuçlar eğitim kümesi içindeki ilişkiyi ölçer.

v3’te H616’nın en yakın komşusu H126. Geçişler %2,78 ve %68,36 (65,58 puan fark). H405’in en yakın komşusu H617. Geçişler %75,66 ve %34,70 (40,95 puan fark). Altı doku ölçüsü iki toprağı da H037 ile eşleştirdi. Görüntü benzerliği ortalamada ince pay hakkında bilgi taşıyor, ancak bu iki zor toprakta yakınlık yeterli ayrım sağlamıyor.

**Kaldığımız yer.** Son deney bir tanı çalışmasıdır. En iyi Kaggle gönderimi üç boyut aralığı, public EMD 35,63420. Yarı düzeltme ikinci sırada (35,73986). 0,063 mm’yi tek başına ve komşu iki elekle birlikte değiştiren sürümler yerel ana sınamada referansın gerisinde kaldı. Bu karşılaştırmayla çalışmaya ara verildi.

[24 sorgunun iki özellik grubundaki eşleşmeleri](ince_pay_komsu_ciftleri.csv) · [Özet](ince_pay_komsu_ozet.csv) · [Analiz kodu](ince_pay_komsu_ciftleri.py)

## Yeniden başlarken hatırlanacak kararlar

1. **Ayırma birimi toprak numunesi.** Bir toprağın farklı fotoğraflarını eğitim ve sınava dağıtmak, yeni bir toprağı tahmin etme başarısını olduğundan yüksek gösterebilir. Ölçekleme ve model ayarları her turda eğitim tarafında kalmalı.
2. **Her iyileşme iddiasında iki kaydı birlikte oku.** 24 numunenin ortalama EMD’sine ve numune bazında kazanç/kayıplara bak. H038’i ayrıca izle. Tek bir çapta yaklaşmak da bütün eğrinin hatasını azaltmayabiliyor.
3. **Aynı deneyi yeni bir fikir gibi tekrarlama.** 4 × 4 doku değişkenliği, altı özellikli Ridge, sabit %50–%50 karışım ve tek parlak-alan özelliği denendi. Yeniden ele alınacaksa hangi varsayımın değiştiğini ve değişikliğin nasıl sınanacağını önce yaz.
4. **Ölçü birimlerini açık tut.** Yüzeydeki piksel alanı, laboratuvardaki kütle yüzdesi ve EMD puanı üç farklı nicelik. Pikseli milimetreye çevirirken kameranın kayıtlı çözünürlüğünü indirilen fotoğrafın gerçek boyutuna uyarla.
5. **Puanın kaynağını yaz.** Kayıtta “24 numunelik yerel”, “21 numunelik kamera sınavı” veya “Kaggle public” ifadesini kullan. Aynı 24 numuneyi birkaç kez incelediğimiz için küçük farkların yeni topraklarda korunup korunmadığı belirsiz.
6. **Güncel karar.** Kendi doğrulanmış gönderimlerimiz içinde en düşük public EMD, üç boyut aralığıyla 35,63420. Her iki yeni sürüm saf v3 tabanını kullanıyor. Kendi verimizle ürettiğimiz saf v3 dosyası yazarın sürümüyle eşleşiyor (Git blob fa5b1134728960728c87a3114d5f49392578b194). Yazar 40,90 public EMD bildirdi. [Yazarın gönderim kaydı](https://github.com/vitalclick/Predicting-Soil-Grain-Size#submission-log)
