
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(input_path, output_path):
    df = pd.read_csv(input_path)
    texts = df['text'].fillna('')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.to_csv(output_path, index=False)
    print(f"TF-IDF kaydedildi: {output_path}")

def main():
    compute_tfidf("ciktilar/tfidf_lemmatized.csv", "ciktilar/tfidf_lemmatized.csv")
    compute_tfidf("ciktilar/tfidf_stemmed.csv", "ciktilar/tfidf_stemmed.csv")

if __name__ == "__main__":
    main()
