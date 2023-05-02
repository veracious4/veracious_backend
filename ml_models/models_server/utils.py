import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import load_model
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import spacy
import numpy as np

from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import requests
import re
import os
from nltk import tokenize

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
    with open(model_path, "rb") as file:
        svm_model = pickle.load(file)

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
    with open(model_path, "rb") as file:
        nb_model = pickle.load(file)

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

# Utilities for BERT
def load_model_bert(model_path):
    bert_model = tf.saved_model.load(model_path)
    return(bert_model)

def fact_validator_bert(fact, model):
    fact_Ser = [fact]
    fact_pred_tensor = tf.sigmoid(model(tf.constant(fact_Ser)))
    pred = fact_pred_tensor[0][0].numpy()
    return pred

class SemanticMatcher: 

    ''''
        This is used to compare the input query with the statements extracted from
        scrapped webpage and return most similar statements.
    '''

    TOP_MATCHES = 5
    embedder = None
    facts = None
    nlp = None

    def __init__(self, top_matches = 5):

        self.TOP_MATCHES = top_matches
        self.embedder = SentenceTransformer("bert-base-nli-mean-tokens")
        self.nlp = spacy.load("en_core_web_md")


    def get_facts(self):
        return self.facts

    def load_data(self, data_path):
        self.facts = pd.read_csv("C:/Users/hp/Desktop/projects/Veracious/veracious_backend/generic_fact_validator/semantic_matcher/facts_train.csv")


    def encode_embeddings(self, corupus):

        corpus_embeddings = self.embedder.encode(corupus)
        return corpus_embeddings

    def generate_candidate(self, d, corpus, corpus_embeddings, query):
        index = faiss.IndexFlatL2(d)
        print(index.is_trained)
        index.add(np.stack(corpus_embeddings, axis=0))
        print(index.ntotal)
        query_embeddings = self.embedder.encode(query)

        k = 2 # Best possible matches 
        D, I = index.search(np.stack(query_embeddings, axis=0), k) 
        print(I)    

        for query, query_embedding in zip(query, query_embeddings):
            distances, indices = index.search(np.asarray(query_embedding).reshape(1,768),k)
            print("\n======================\n")
            print("Query Fact:", query)
            print("\nMost similar facts in corpus:")
            print(corpus[indices[0,1]], "(Distance: %.4f)" % distances[0,1])

        return (corpus[indices[0,1]], distances[0,1])

    def generate_simarity_score(self, ref_sentence, sim_sentence):
        ref_sentence_vec = self.nlp(ref_sentence)
        sim_sentence_vec = self.nlp(sim_sentence)
        sim_score = round(ref_sentence_vec.similarity(sim_sentence_vec)*100, 2)
        return sim_score


class Scraper:
    '''
    It is used to search for a particular text on the web and scrape the top x pages from the search results.
    By default, it scrapes the top 3 pages.
    '''
    num_pages = 1
    DRIVER_PATH = ""
    service = None

    
    def __init__(self, num_pages=1):
        '''
        Constructor for the Scraper class
        @param query: The text to be searched on the web
        @param num_pages: The maximum number of pages to be scraped from the search results. Default is 3.
        '''
        self.num_pages = num_pages
        self.DRIVER_PATH = os.environ.get('SELENIUM_WEB_DRIVER_CHROME_PATH')
        if self.DRIVER_PATH == None:
            raise Exception("Please set the environment variable SELENIUM_WEB_DRIVER_CHROME_PATH")

        self.service = Service(executable_path=self.DRIVER_PATH)
    

    def __parse_web_search_results(self, content, article_list=[]) -> None:
        '''
        This function is used to parse the google search results, and stores the article url and text in form of a list.
        @param content: The content of the page to be parsed
        @param article_list: The list to store the article url and text
        '''
        soup = BeautifulSoup(content, 'html.parser')
        for script in soup(["script", "style", "header", "footer"]):
            script.decompose()
        body = soup.find('body')
        articles = body.find_all('div', class_='kvH3mc')
        for article in articles:
            if(article.find('h3') == None or article.find('a') == None):
                continue
            obj = {
                'url': article.find('a')['href'],
                'title': article.find('h3').text,
            }
            article_list.append(obj)

    
    def __search_text_on_web(self, query, num_pages=num_pages) -> list:
        '''
        This function is used to search for the given text on the web and store the search results in a list.
        @param query: The text to be searched on the web
        @param num_pages: The maximum number of pages to be scraped from the search results. Default is passed from the constructor.
        '''
        browser = webdriver.Chrome(service=self.service)
        browser.implicitly_wait(0.5)
        try:
            browser.get("https://www.google.com")
        
            #identify search box
            search = browser.find_element(By.NAME, "q")
            #enter search text
            search.send_keys(query+" ?")
            time.sleep(0.2)
            
            #perform Google search with Keys.ENTER
            search.send_keys(Keys.ENTER)

            # extract the search results and store it in a list
            article_list = []

            # for next page id is 'pnnext'
            hasContent = True
            pageNo = 1
            
            # fetch the content of top 3 pages
            while(pageNo<=num_pages and hasContent):
                content = browser.page_source
                self.__parse_web_search_results(content, article_list)
                try:
                    next_page = browser.find_element(By.ID, "pnnext")
                    next_page.click()
                    time.sleep(0.5)
                    pageNo += 1
                except:
                    hasContent = False
            
            browser.quit()
            return article_list
        except:
            browser.quit()
            raise Exception("Error while searching for the text on the web. Please check the internet connection and try again.")
    
    
    def __scrape_content_from_url(self, url) -> str:
        '''
        This function is used to scrape the content from the given url. 
        It removes all the unnecessary tags and returns the text.
        @param url: The url from which the content is to be scraped. 
        '''
        try:
            # get the content from the url
            content = requests.get(url).content
            # parse the content using beautiful soup
            soup = BeautifulSoup(content, 'html.parser')
            body = soup.find('body')

            if body == None:
                return ""
            
            # Remove the header and footer
            if body.header:
                body.header.decompose()
            if body.footer:
                body.footer.decompose()
            
            # Remove the media tags
            media_tags = ['video', 'audio', 'picture', 'source', 'img']
            for tag in media_tags:
                for media in body.find_all(tag):
                    media.decompose()
            
            # Remove all inputs
            input_tags = ['input', 'form', 'select', 'textarea']
            for tag in input_tags:
                for input in body.find_all(tag):
                    input.decompose()

            # Remove the scripts
            for script in body.find_all('script'):
                script.decompose()
            
            # Remove all the buttons
            for button in body.find_all('button'):
                button.decompose()

            # Remove navigation
            for nav in body.find_all('nav'):
                nav.decompose()
            
            # Remove all icons
            for i in body.find_all('i'):
                i.decompose()
            
            # Remove all links
            for link in body.find_all('a'):
                link.decompose()
            
            # Remove all code tags
            code_tags = ['code', 'kbd', 'samp', 'var']
            for tag in code_tags:
                for code in body.find_all(tag):
                    code.decompose() 
            
            # Remove unnecessary tags
            unnecessary_tags = ['s', 'del']
            for tag in unnecessary_tags:
                for t in body.find_all(tag):
                    t.decompose()
                
            text = body.text
            
            # remove the newlines
            text = text.replace('\n', ' ')
            # remove the extra spaces
            text = re.sub(' +', ' ', text)
            return text
        except:
            return ""
    

    def scrape(self, query, num_pages=num_pages) -> list:
        '''
        This function is used to scrape the web for the given query and returns the web articles in the form of a list.
        @param query: The text to be searched on the web
        @param num_pages: The maximum number of pages to be scraped from the search results. Default is passed from the constructor.
        '''
        # search for the given query on the web
        article_list = self.__search_text_on_web(query, num_pages)
        # scrape the content from the url
        for article in article_list:
            article['text'] = self.__scrape_content_from_url(article['url'])
        return article_list
    
def run_generic_model(fact):

    sc = Scraper(1)
    scrape_res = sc.scrape(fact)
    eff_score = 0
    scores_cnt = 0


    for i in range(0, len(scrape_res)):

        if scrape_res[i] != None and scrape_res[i]["text"] != None:
            corpus = scrape_res[i]["text"]
            if len(corpus) >= 50:
                corpus_facts = tokenize.sent_tokenize(corpus)
                sm = SemanticMatcher()

                corpus_embeddings = sm.encode_embeddings(corpus_facts)
                query = [fact]

                (sim_sentence, sim_sentence_dis) = sm.generate_candidate(768, corpus_facts, corpus_embeddings, query)

                sim_score = sm.generate_simarity_score(query[0], sim_sentence)

                eff_score += sim_score
                scores_cnt += 1

                print("Current Web-Page similarity score: \n")
                print(sim_score)
                temp = eff_score/scores_cnt
                print("Current effective Web-Page similarity score: \n")
                print("{:.2f}".format(temp))

    print("Effective Trust Score: \n")
    eff_score /= scores_cnt
    print(eff_score)

    normalized_score = eff_score/100

    return normalized_score

def validate_fact(fact:str):
    pred_gen = run_generic_model(fact)
    '''
    # PAC Model
    (model, vectorizer) = load_model_vectorizer("../passive_aggressive_model.pkl", "../passive_aggressive_vectorizer.pkl")
    
    pred_pac = fact_validator(fact, model, vectorizer)

    # LSTM Model
    (model, tokenizer) = load_model_tokenizer_lstm("../lstm_classifier.h5", "../lstm_tokenizer.pkl")

    pred_lstm = fact_validator_lstm(fact, model, tokenizer, 300)

    # SVM Model
    #(svm_model, svm_vectorizer) = load_model_vectorizer_svm("../saved_models/svm_classifier.pkl", "../saved_models/vectorizer_SVM_NB.pickle")
    #pred_svm = fact_validator_svm(fact, svm_model, svm_vectorizer)

    # Naive Bayes Model
    (nb_model, nb_vectorizer) = load_model_vectorizer_svm("../saved_models/bernoulli_nb_classifier.pkl", "../saved_models/vectorizer_SVM_NB.pickle")
    pred_nb = fact_validator_nb(fact, nb_model, nb_vectorizer)

    # # Bert Model
    # bert_model = load_model_bert("../saved_models/bert/bert_classifier")
    # pred_bert = fact_validator_bert(fact, bert_model)  

    # Ensembling Result
    pred_lstm = float(pred_lstm)
    pred_pac = float(pred_pac[0])
    #pred_svm = float(pred_svm)
    pred_nb = float(pred_nb)
    # pred_bert = float(pred_bert)

    if(pred_pac == 1.0):
        pred_pac = 0.75
    else :
        pred_pac = 0.25
    
    if(pred_svm==1.0):
        pred_svm = 0.75
    else:
        pred_svm = 0.25

    ensembled_result = (pred_lstm + pred_pac + pred_nb)/3
    # ensembled_result = (pred_lstm + pred_pac + pred_nb + pred_bert)/4
    '''
    return pred_gen




def get_verdict_from_trust_score(trust_score:float):
    if trust_score <0.4:
        return "REFUTES"
    elif trust_score >0.6:
        return "SUPPORTS"
    else:
        return "NOT ENOUGH INFO"