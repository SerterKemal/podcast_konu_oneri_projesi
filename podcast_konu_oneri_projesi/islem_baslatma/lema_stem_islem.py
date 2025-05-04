
import sys
import os
sys.path.append(os.path.abspath("code"))

from lema_stem_code.lema import lemmatize_texts
from lema_stem_code.stem import stem_texts
from kaynak import load_texts

import pandas as pd

def save_to_csv(data, path):
    df = pd.DataFrame({'text': data})
    df.to_csv(path, index=False)
    print(f"Kayıt edildi: {path}")

def main():
    print("Veriler yükleniyor...")
    raw_texts = load_texts()

    print("Lemmatization başlatılıyor...")
    lema_texts = lemmatize_texts(raw_texts)
    save_to_csv(lema_texts, "ciktilar/tfidf_lemmatized.csv")

    print("Stemming başlatılıyor...")
    stem_texts_ = stem_texts(raw_texts)
    save_to_csv(stem_texts_, "ciktilar/tfidf_stemmed.csv")

    print("İşlem tamamlandı.")

if __name__ == "__main__":
    main()
