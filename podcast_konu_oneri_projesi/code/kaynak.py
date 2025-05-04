
import os
import json

def download_podcast_data():
    path = "veri/raw/podcast_transcripts.json"
    if not os.path.exists(path):
        raise FileNotFoundError("Veri dosyası bulunamadı. Lütfen podcast_transcripts.json dosyasını 'veri/raw/' klasörüne yerleştirin.")
    return path

def load_texts():
    path = download_podcast_data()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for item in data:
        transcript = item.get("transcript", "")
        if transcript and len(transcript) > 100:
            texts.append(transcript.strip())

    print(f"{len(texts)} bölüm yüklendi.")
    return texts

if __name__ == "__main__":
    print("Toplam kayıt sayısı:", len(load_texts()))
