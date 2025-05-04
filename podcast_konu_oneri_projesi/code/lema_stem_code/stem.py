
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

def stem_texts(texts):
    return [' '.join([stemmer.stem(word) for word in word_tokenize(text)]) for text in texts]
