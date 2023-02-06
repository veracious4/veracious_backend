from sentence_transformers import SentenceTransformer
import faiss
import spacy
import gensim
import numpy as np
import pandas as pd

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

def main(sm):
    sm.load_data("facts_train.csv")
    facts = sm.get_facts()
    facts_1 = facts[['qid1', 'question1']]
    facts_1.columns = ['id', 'question']
    facts_2 = facts[['qid2', 'question2']]
    facts_2.columns = ['id', 'question']
    facts_list = pd.concat([facts_1,facts_2]).sort_values('id')
    corpus = facts_list["question"].tolist()
    corpus_embeddings = sm.encode_embeddings(corpus)

    query = ["What is purpose of life?", "What is the step by step guide to invest in share market in india?"]
    (sim_sentence, sim_sentence_dis) = sm.generate_candidate(768, corpus, corpus_embeddings, query)

    sim_score = sm.generate_simarity_score(query[1], sim_sentence)
    print(sim_score)
    return (sim_score, query[1], sim_sentence)

if __name__=="__main__":
    
    sm = SemanticMatcher()
    main(sm)
