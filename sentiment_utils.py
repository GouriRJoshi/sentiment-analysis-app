import string
from sklearn.feature_extraction import text

stopwords = text.ENGLISH_STOP_WORDS

def preprocess(text_input):
    text_input = text_input.lower()
    text_input = ''.join([c for c in text_input if c not in string.punctuation])
    words = text_input.split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)
