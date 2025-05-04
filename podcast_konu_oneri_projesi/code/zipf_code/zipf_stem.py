
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def plot_zipf(texts, output_path):
    words = ' '.join(texts).split()
    freq_dist = Counter(words)
    sorted_freqs = sorted(freq_dist.values(), reverse=True)
    ranks = list(range(1, len(sorted_freqs) + 1))

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, sorted_freqs)
    plt.xlabel("Log Rank")
    plt.ylabel("Log Frequency")
    plt.title("Zipf Grafiği - Lemmatized")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    df = pd.read_csv("ciktilar/tfidf_stemmed.csv")
    texts = df['text'].dropna().tolist()
    plot_zipf(texts, "ciktilar/zipf_dosyalari/grafikler/zipf_stem_loglog.png")
    print("Zipf grafiği (stemmed) oluşturuldu.")

if __name__ == "__main__":
    main()
