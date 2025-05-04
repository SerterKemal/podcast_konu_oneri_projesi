
import spacy
nlp = spacy.load("en_core_web_sm")

def lemmatize_texts(texts):
    return [' '.join([token.lemma_ for token in nlp(text)]) for text in texts]
