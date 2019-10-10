from gensim.corpora.wikicorpus import tokenize
import spacy

nlp = spacy.load("en_core_web_sm") # en or  en_core_web_sm
nlp.max_length = 93621305


def tokenize_spacy(text, min_token_length=2):
    doc = nlp(text)
    sentences = []
    for sentence in doc.sents:
        tokens = [token.string.strip() for token in sentence if token.string.strip()]
        if len(tokens) > min_token_length:
            sentences.append(' '.join(tokens))
    return sentences

def tokenize_gensim(text):
    tokenize(text)

