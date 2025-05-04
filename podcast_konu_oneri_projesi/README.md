
# 🎙️ Podcast Konu Öneri Sistemi (Gerçekçi Verilerle)

Bu proje, uzun transkriptlere sahip 1000 podcast bölümünden oluşan gerçekçi bir veri seti ile çalışır. NLP tekniklerini kullanarak her bölümü işler, analiz eder ve sonrasında yeni bir özet verildiğinde benzer konulu podcast bölümleri önerir.

---

## 📁 Proje Klasör Yapısı

```
podcast_konu_oneri_projesi/
├── ciktilar/                  → TF-IDF, grafik ve model çıktıları
├── code/                      → Modüller: lemmatization, stemming, vektörler
├── islem_baslatma/           → Her adımı çalıştırmak için script dosyaları
├── veri/raw/                 → Gerçekçi podcast transkriptleri (.json)
├── requirements.txt
└── README.md
```

---

## 📦 Kullanılan Veri

- **Kaynak:** Simüle edilmiş 1000 uzun transkript
- **Format:** `veri/raw/podcast_transcripts.json`
- **Boyut:** ≈ 3.3 MB
- **İçerik:** Her biri 500+ kelimelik 1000 bölüm

---

## 🛠️ Kurulum

1. Gereksinimleri yükle:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## ▶️ Çalıştırma Sırası

### 1. Lemmatize + Stem işlemleri:
```bash
python islem_baslatma/lema_stem_islem.py
```

### 2. Zipf grafikleri:
```bash
python islem_baslatma/zipf_islem.py
```

### 3. TF-IDF + Word2Vec işlemleri:
```bash
python islem_baslatma/vektor_islem.py
```



## 👤 Geliştirici

Bu proje, metin tabanlı yapay zeka teknikleriyle podcast benzerlik analizine örnek teşkil eder.
