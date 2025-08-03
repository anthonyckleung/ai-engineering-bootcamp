import spacy
import pandas as pd
import re
import nltk
import numpy as np
import string
from nltk.stem import PorterStemmer

import spacy
from scipy.sparse import hstack

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',str(text))

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',str(text))

def remove_punctuation(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def final_preprocess(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text

def standardize_text(text):
    text = remove_URL(text)
    text = remove_emoji(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = final_preprocess(text)
    return text


def job_description_classifier(model, vectorizer, input_text):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    doc = nlp(input_text)
    X_lemma, X_pos = ' '.join([token.lemma_ for token in doc]), ' '.join([token.pos_ for token in doc])

    X_lemma_transform = vectorizer.transform([X_lemma])
    X_pos_transform = vectorizer.transform([X_pos])

    X_in = hstack([X_lemma_transform, X_pos_transform])

    proba = float(model.predict_proba(X_in)[0][1])
    is_fraud = bool(proba > 0.5)

    return {
        "is_fraud": is_fraud,
        "probability": round(proba, 3)
    }