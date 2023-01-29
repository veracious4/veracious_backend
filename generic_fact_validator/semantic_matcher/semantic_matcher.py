from sentence_transformers import SentenceTransformer
import scipy.spatial


class SemanticMatcher: 

    ''''
        This is used to compare the input query with the statements extracted from
        scrapped webpage and return most similar statements.
    '''

    TOP_MATCHES = 5
    embedder = None

    def __init__(self, top_matches = 5):

        self.TOP_MATCHES = top_matches
        self.embedder = SentenceTransformer("bert-base-nli-mean-tokens")

    def encode_embeddings(self, corupus):

        corpus_embeddings = self.embedder.encode(corupus)
        return corpus_embeddings

    
    
