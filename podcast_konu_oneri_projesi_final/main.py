import os
import joblib
import gensim
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✔️ TF-IDF vektörlerinden kelime bazlı metin üretimi
def load_csv(version):
    df = pd.read_csv(f"data/{version}.csv")
    def row_to_text(row):
        sorted_words = row.sort_values(ascending=False).head(20).index.tolist()
        return " ".join(sorted_words)
    df["text"] = df.apply(row_to_text, axis=1)
    return df

def train_tfidf(version):
    df = load_csv(version)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    joblib.dump((vectorizer, tfidf_matrix), f"models/tfidf_{version}.pkl")

def tfidf_similarity(input_text, version):
    vectorizer, tfidf_matrix = joblib.load(f"models/tfidf_{version}.pkl")
    df = load_csv(version)
    input_vec = vectorizer.transform([input_text])
    scores = cosine_similarity(input_vec, tfidf_matrix)[0]
    top5_idx = np.argsort(scores)[-5:][::-1]
    return df.iloc[top5_idx]["text"].tolist(), scores[top5_idx].tolist()

def avg_vector(model, text):
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else None

def word2vec_similarity(input_text, model_path, csv_path):
    model = gensim.models.Word2Vec.load(model_path)
    df = load_csv(csv_path.replace("data/", "").replace(".csv", ""))
    input_vec = avg_vector(model, input_text)
    if input_vec is None:
        return [], []
    df["vector"] = df["text"].apply(lambda x: avg_vector(model, x))
    df = df[df["vector"].notnull()]
    if df.empty:
        return [], []
    mat = np.vstack(df["vector"].values)
    scores = cosine_similarity([input_vec], mat)[0]
    top5_idx = np.argsort(scores)[-5:][::-1]
    top_texts = df.iloc[top5_idx]["text"].tolist()
    top_scores = scores[top5_idx].tolist()
    return top_texts, top_scores

# 🔢 Anlamsal puanı 1–5 arası hesapla (cosine skoru üzerinden)
def score_to_rating(score):
    if score >= 0.9:
        return 5
    elif score >= 0.75:
        return 4
    elif score >= 0.5:
        return 3
    elif score >= 0.3:
        return 2
    elif score > 0:
        return 1
    else:
        return 0

def generate_excel_report_flat(results):
    os.makedirs("outputs", exist_ok=True)
    records = []
    for model, data in results.items():
        for text, score in zip(data["texts"], data["scores"]):
            records.append({
                "Model": model,
                "Benzer Metin": text,
                "Benzerlik Skoru": round(score, 4),
                "Anlamsal Puan (1-5)": score_to_rating(score)
            })
    df = pd.DataFrame(records)
    df.to_excel("outputs/benzerlik_raporu.xlsx", index=False)
    print("✅ Rapor oluşturuldu: outputs/benzerlik_raporu.xlsx")

def main():
    os.makedirs("outputs", exist_ok=True)

    for version in ["lemmatized", "stemmed"]:
        train_tfidf(version)

    df = load_csv("lemmatized")
    input_text = df["text"].iloc[0]
    results = {}

    # TF-IDF benzerlikleri
    for version in ["lemmatized", "stemmed"]:
        texts, scores = tfidf_similarity(input_text, version)
        results[f"tfidf_{version}"] = {"texts": texts, "scores": scores}

    # Word2Vec benzerlikleri
    for m in os.listdir("models"):
        if m.endswith(".model"):
            path = os.path.join("models", m)
            vtype = "lemmatized" if "lema" in m else "stemmed"
            texts, scores = word2vec_similarity(input_text, path, f"data/{vtype}.csv")
            short_name = m.replace(".model", "")
            if texts and scores:
                results[short_name] = {"texts": texts, "scores": scores}

    generate_excel_report_flat(results)

if __name__ == "__main__":
    main()
