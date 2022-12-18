import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

def fact_validator(fact, model, vectorizer):
    lfact = preprocesFactDescription(fact)

    df = pd.DataFrame([lfact])

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



# Utilities for LSTM Algorithm
def load_model_tokenizer_lstm(model_path, tokenizer_path):

    lstm_model = load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        lstm_tokenizer = pickle.load(file)

    return (lstm_model, lstm_tokenizer)


def fact_validator_lstm(fact, model, tokenizer, maxlen):

    fact_Ser = [fact]
    tokenized_fact = tokenizer.texts_to_sequences(fact_Ser)
    padded_tokenized_fact = pad_sequences(tokenized_fact, maxlen=maxlen)
    pred = model.predict(padded_tokenized_fact)

    return pred[0][0]

# Utilities for SVM Algorithm
def load_model_vectorizer_svm(model_path, vectorizer_path):
    svm_model = load_model(model_path)
    with open(vectorizer_path, "rb") as file:
        svm_vectorizer = pickle.load(file)
    return (svm_model, svm_vectorizer)

def fact_validator_svm(fact, model, vectorizer):
    lfact = preprocesFactDescription(fact)

    df = pd.DataFrame([lfact])

    x = df.iloc[:,0]
    x = vectorizer.transform(x)
    x_pred = model.predict(x)
    return x_pred=='1'

# Utilities for Naive Bayes Algorithm
def load_model_vectorizer_nb(model_path, vectorizer_path):
    nb_model = load_model(model_path)
    with open(vectorizer_path, "rb") as file:
        nb_vectorizer = pickle.load(file)
    return (nb_model, nb_vectorizer)

def fact_validator_nb(fact, model, vectorizer):
    lfact = preprocesFactDescription(fact)

    df = pd.DataFrame([lfact])

    x = df.iloc[:,0]
    x = vectorizer.transform(x)
    x_pred = model.predict(x)
    return x_pred