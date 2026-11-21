# Güncel durum — Fotoğraftan toprak tane boyutu tahmini

**21 Kasım 2026 · En iyi Kaggle public sonucumuz 52,88353.** Açık v3 ve ince/iri KNN eğrilerinin sabit %50–%50 ortalamasıyla önceki en düşük hatamız 67,96723’ten 52,88353’e indi. İyileşme 15,08370 EMD (%22,19). Dördüncü gönderim ekranda Complete olarak doğrulandı.

| Gönderim | Dosya | 24 toprakta yerel EMD | Kaggle public EMD |
|---|---|---:|---:|
| 1 · Altı doku ölçüsü + üç komşu | [submission_3nn_doku.csv](submission_3nn_doku.csv) | 44,744 | 67,96723 |
| 2 · Altı doku ölçüsü ve 16 bölgedeki değişkenlik + üç komşu | [submission_3nn_16_bolge.csv](submission_3nn_16_bolge.csv) | 44,184 | 71,50011 |
| 3 · Her ölçekteki doku farkının toplam içindeki payı + üç komşu | [submission_3nn_goreli_doku.csv](submission_3nn_goreli_doku.csv) | 43,567 | 80,56630 |
| 4 · Açık v3 + ince/iri KNN, %50–%50 | [submission_v3_knn_50_50.csv](submission_v3_knn_50_50.csv) | 38,002 | **52,88353** |

Paylaşılan Kaggle ekranında dört dosya da **Complete** durumunda. Yerelde dosyaların 10 test kimliğini, 11 yüzde sütununu, yüzdelerin sınırlarını ve artan çaplarda azalmadığını kontrol ettik. İlk hazırlanan [submission.csv](submission.csv) ise her numuneye eğitim eğrilerinin ortalamasını verir.

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

## Yeniden başlarken hatırlanacak kararlar

1. **Ayırma birimi toprak numunesi.** Bir toprağın farklı fotoğraflarını eğitim ve sınava dağıtmak, yeni bir toprağı tahmin etme başarısını olduğundan yüksek gösterebilir. Ölçekleme ve model ayarları her turda eğitim tarafında kalmalı.
2. **Her iyileşme iddiasında iki kaydı birlikte oku.** 24 numunenin ortalama EMD’sine ve numune bazında kazanç/kayıplara bak. H038’i ayrıca izle. Tek bir çapta yaklaşmak da bütün eğrinin hatasını azaltmayabiliyor.
3. **Aynı deneyi yeni bir fikir gibi tekrarlama.** 4 × 4 doku değişkenliği, altı özellikli Ridge, sabit %50–%50 karışım ve tek parlak-alan özelliği denendi. Yeniden ele alınacaksa hangi varsayımın değiştiğini ve değişikliğin nasıl sınanacağını önce yaz.
4. **Ölçü birimlerini açık tut.** Yüzeydeki piksel alanı, laboratuvardaki kütle yüzdesi ve EMD puanı üç farklı nicelik. Pikseli milimetreye çevirirken kameranın kayıtlı çözünürlüğünü indirilen fotoğrafın gerçek boyutuna uyarla.
5. **Puanın kaynağını yaz.** Kayıtta “24 numunelik yerel”, “21 numunelik kamera sınavı” veya “Kaggle public” ifadesini kullan. Aynı 24 numuneyi birkaç kez incelediğimiz için küçük farkların yeni topraklarda korunup korunmadığı belirsiz.
6. **Güncel karar.** Kendi dört gönderimimiz içinde en düşük public EMD, sabit karışımla 52,88353. Yazarın saf v3 satırında 38,8 yerel ve 40,90 public EMD kayıtlı. Kendi verimizle yeniden ürettiğimiz saf v3 CSV’si, satır sonları eşitlendiğinde depodaki dosyayla birebir eşleşti (Git blob fa5b1134728960728c87a3114d5f49392578b194). Saf v3’ü yeniden göndermek aynı tahminleri tekrarlayacağı için sonraki model karşılaştırmalarında yazarın bildirdiği 40,90’ı referans alıyoruz. [Yazarın gönderim kaydı](https://github.com/vitalclick/Predicting-Soil-Grain-Size#submission-log)
