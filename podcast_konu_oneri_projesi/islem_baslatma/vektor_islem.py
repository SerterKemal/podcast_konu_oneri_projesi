
import subprocess

print("TF-IDF hesaplaması başlatılıyor...")
subprocess.run(["python", "code/vektor/tfidf_vektor.py"])

print("Word2Vec modelleri eğitiliyor...")
subprocess.run(["python", "code/vektor/word2vec_egitim.py"])
