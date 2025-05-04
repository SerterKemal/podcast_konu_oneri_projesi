
import subprocess

print("Lemmatized Zipf grafiği hesaplanıyor...")
subprocess.run(["python", "code/zipf_code/zipf_lema.py"])

print("Stemmed Zipf grafiği hesaplanıyor...")
subprocess.run(["python", "code/zipf_code/zipf_stem.py"])
