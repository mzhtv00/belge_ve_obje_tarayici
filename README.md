# Mobil Tarama ve CAD Aktarımı

Telefonu masaüstü tarama paneline QR kodla bağlayan, fotoğrafları WebSocket
üzerinden yerel makineye alan ve STL dosyalarını masaüstü CAD araçlarına
aktarmayı kolaylaştıran FastAPI uygulaması.

## Kurulum ve çalıştırma

Python 3.11 veya üzeri gerekir.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

`py -3` Windows Python Launcher'daki mevcut Python 3 sürümlerinden birini
kullanır (proje Python 3.11 veya üzerini gerektirir). `Activate.ps1` çalıştırma
ilkesi nedeniyle engellenebildiğinden kurulum ve başlatma doğrudan sanal
ortamın `python.exe` dosyasıyla yapılır. Belirli bir sürüm kullanmak için
`py -3.12` gibi makinede kurulu olan sürümü seçin.

Panel varsayılan tarayıcıda açılır. Telefonun bağlanabilmesi için sunucu aynı
güvenilir Wi-Fi ağında `8000` portunu dinler. Güvenlik duvarında bu porta yalnız
özel ağlarda izin verin. QR kod, oturum boyunca geçerli rastgele bir erişim
anahtarı içerir; bağlantıyı başkalarıyla paylaşmayın.

Telefonda QR kodunu okutun. Canlı `getUserMedia` kamera önizlemesi HTTPS veya
`localhost` güvenli bağlamı ister. Yerel ağdaki düz HTTP bağlantısında sayfa,
telefon tarayıcısının yerel kamera/fotoğraf seçicisini kullanır. Canlı önizleme
için güvenilir yerel bir HTTPS ters vekil sunucu yapılandırılabilir.

## İşlevler ve sınırlar

- Panel, mobil oturum ve QR kodu oluşturur; tarama modunu seçer, çekimleri arar
  ve WebSocket üzerinden canlı durum güncellemelerini alır.
- Mobil sayfa JPEG, PNG ve WebP verisini 64 KiB WebSocket parçalarıyla yollar.
  Sunucu 12 MiB ve 40 megapiksel sınırlarını uygular, içeriği doğrular ve
  küçük önizlemeyi arka plan işçisinde üretir.
- Çekimler 2D/3D moduyla etiketlenir. Bu depoda OCR, belge tanıma,
  fotogrametri veya STL üretim servisi yoktur; uygulama bunların yapıldığını
  iddia etmez. Harici üretim hattı `app/pipeline.py` üzerinden bağlanabilir.
- STL dosyaları panele yüklenebilir, Three.js ile önizlenebilir veya
  Cura/PrusaSlicer'da açılabilir. Üç boyutlu önizleme için tarayıcının
  `cdn.jsdelivr.net` adresine erişebilmesi gerekir.
- Tinkercad'in yerel dosya içe aktarma API'si bulunmadığından aktarım,
  Tinkercad'i açar ve STL dosyasını indirir; kullanıcı tasarım düzenleyicisinde
  dosyayı elle içe aktarır.
- Slicer `PATH` içinde bulunamazsa `SLICER_EXECUTABLE` değişkenini Cura veya
  PrusaSlicer çalıştırılabilir dosyasının tam yolu olarak ayarlayın. Süreç,
  shell kullanılmadan argüman listesiyle başlatılır.

Görüntüler, küçük önizlemeler ve STL dosyaları varsayılan olarak `data/`
altında tutulur; `SCANNER_DATA_DIR` ile değiştirilebilir. Oturum anahtarları
bellektedir ve uygulama yeniden başlayınca geçersiz olur.

## Yapılandırma

| Değişken | Varsayılan | Açıklama |
| --- | --- | --- |
| `SCANNER_HOST` | `0.0.0.0` | Sunucunun dinleme adresi |
| `SCANNER_PORT` | `8000` | HTTP/WebSocket portu |
| `SCANNER_DATA_DIR` | `data` | Yerel dosya dizini |
| `SCANNER_WORKERS` | `2` | Küçük önizleme işçisi sayısı |
| `SLICER_EXECUTABLE` | Otomatik bulma | Cura/PrusaSlicer tam yolu |

Testleri kurup çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```
