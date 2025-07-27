import spacy
from scipy.sparse import hstack
from api.ml.preprocessor import standardize_text


def job_description_classifier(model, vectorizer, input_text):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    doc = nlp(input_text)
    X_lemma, X_pos = ' '.join([token.lemma_ for token in doc]), ' '.join([token.pos_ for token in doc])

    X_lemma_transform = vectorizer.transform([X_lemma])
    X_pos_transform = vectorizer.transform([X_pos])

    X_in = hstack([X_lemma_transform, X_pos_transform])

    proba = model.predict_proba(X_in)[0][1]
    is_fraud = proba > 0.5

    return {
        "is_fraud": is_fraud,
        "probability": round(proba, 3)
    }

