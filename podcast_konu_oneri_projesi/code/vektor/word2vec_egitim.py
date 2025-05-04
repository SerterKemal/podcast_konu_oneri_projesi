
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import os

parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

def convert_df_to_texts(df):
    # TF-IDF matrisinden metin benzeri veri üret
    return [' '.join([col for col, val in row.items() if val > 0]) for _, row in df.iterrows()]

def train_models(input_path, output_dir, prefix):
    df = pd.read_csv(input_path)
    texts = convert_df_to_texts(df)
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]

    for params in parameters:
        sg = 1 if params['model_type'] == 'skipgram' else 0
        model = Word2Vec(sentences=tokenized_texts, vector_size=params['vector_size'],
                         window=params['window'], sg=sg, min_count=1, workers=2)
        model_name = f"{prefix}_{params['model_type']}_win{params['window']}_dim{params['vector_size']}.model"
        model.save(os.path.join(output_dir, model_name))
        print(f"Model kaydedildi: {model_name}")

def main():
    train_models("ciktilar/tfidf_lemmatized.csv", "ciktilar/vektor/modeller/lema", "lema")
    train_models("ciktilar/tfidf_stemmed.csv", "ciktilar/vektor/modeller/stem", "stem")

if __name__ == "__main__":
    main()
