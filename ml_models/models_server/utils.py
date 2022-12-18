import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import pickle


# Utilities for Passive Aggressive Algorithm
def preprocesFactDescription(sentence):
    stop_words = set(stopwords.words('english')) 
    lemma_words = []
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(sentence) 
    for word in word_tokens: 
        if word not in stop_words: 
            new_word = re.sub('[^a-zA-Z]', '',word)
            new_word = new_word.lower()
            new_word = wordnet_lemmatizer.lemmatize(new_word)
            lemma_words.append(new_word)
    return " ".join(lemma_words)

def fact_validator(news, model, vectorizer):
    lnews = preprocesFactDescription(news)

    df = pd.DataFrame([lnews])

    x = df.iloc[:,0]
    x = vectorizer.transform(x)

    x_pred = model.predict(x)

    return x_pred

def load_model_vectorizer(model_path, vectorizer_path):
    with open(model_path, "rb") as file:
        pac_model = pickle.load(file)

    with open(vectorizer_path, "rb") as file:
        pac_vectorizer = pickle.load(file)

    return (pac_model, pac_vectorizer)