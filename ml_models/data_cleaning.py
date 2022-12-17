import numpy as np
import  pandas as pd

class CleanRawData:
    fact_categories = ['false', 'mixture', 'true', 'unproven']

    def __clean_PUBHEALTH_data(self, corpus):
        '''Removes unnecessary columns and rows from the PUBHEALTH dataset'''

        corpus.drop(['fact_checkers', 'claim_id', 'sources', 'date_published', 'explanation'], axis=1, inplace=True)
        corpus.dropna(inplace=True)
        corpus = corpus.drop_duplicates()
        corpus['label'] = corpus['label'].str.lower().str.strip()
        corpus = corpus[corpus['label'].isin(self.fact_categories)]

        return corpus
    
    def __filter_PUBHEALTH_medical_data(self, cleanedCorpus):
        '''Filters the PUBHEALTH dataset to only include medical claims, and removes unnecessary columns'''
        medical_data = cleanedCorpus[-cleanedCorpus['subjects'].str.lower().str.contains(
                                        'foreign policy|political|social|economy|religion|religious|culture|sport|sports|entertainment|other')]
        medical_data.drop(['subjects'], axis=1, inplace=True)
        return medical_data
    
    
    def clean_PUBHEALTH_dataset(self):
        '''Cleans out the raw PUBHEALTH dataset and saves them into csv files'''
        np.random.seed(500)
        train_corpus = pd.read_csv(r"./data/PUBHEALTH/train.tsv", sep='\t')
        test_corpus = pd.read_csv(r"./data/PUBHEALTH/test.tsv", sep='\t')
        dev_corpus = pd.read_csv(r"./data/PUBHEALTH/dev.tsv", sep='\t')

        train_corpus = pd.concat([train_corpus, dev_corpus], ignore_index=True)
        train_corpus = self.__clean_PUBHEALTH_data(train_corpus)
        test_corpus = self.__clean_PUBHEALTH_data(test_corpus)
        medical_data_train = self.__filter_PUBHEALTH_medical_data(train_corpus)
        medical_data_test = self.__filter_PUBHEALTH_medical_data(test_corpus)
        medical_data_train.to_csv(r"./data/PUBHEALTH/medical_data_cleaned_train.tsv", sep='\t', index=False)
        medical_data_test.to_csv(r"./data/PUBHEALTH/medical_data_cleaned_test.tsv", sep='\t', index=False)



data_cleaner = CleanRawData()
# data_cleaner.clean_PUBHEALTH_dataset() 