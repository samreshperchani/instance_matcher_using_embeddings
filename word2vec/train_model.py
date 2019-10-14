import glob
from gensim.models import Word2Vec
import logging
import datetime
import os
import re
import gensim
import sys
from pathlib import Path
import shutil
import subprocess
import pandas as pd

# set configuration file path
config_path = os.path.dirname(os.getcwd()) + '/config' 

# add config file path to system
sys.path.append(config_path)

import config

# base directory of the solution
BASE_DIR = config.BASE_DIR

# path to data directory
DATA_DIR = config.DATA_DIR

# path where extracted text will be stored
EXTRACTED_TEXT_DIR = config.EXTRACTED_TEXT_DIR


print(gensim.models.word2vec.FAST_VERSION)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# count of epochs
WORD2VEC_EPOCHS = config.WORD2VEC_EPOCHS

# length of embedding vector
EMBEDDING_VECTOR_LENGTH = config.EMBEDDING_VECTOR_LENGTH

# path where untared xml dumps are present
UNTARED_DUMPS_DIR = config.UNTARED_DUMPS_DIR

# path where processed dumps are present
PROCESSED_DUMPS_DIR = config.PROCESSED_DUMPS_DIR


# class to traing word2vec
class WORD2VEC:

    # extract text from all xml dumps
    def extract_text(self):
        '''
        if not os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' +EXTRACTED_TEXT_DIR):
            os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' +EXTRACTED_TEXT_DIR)
        # get list of all folders in untared dumps directory
        file_names = os.listdir(BASE_DIR + '/' + DATA_DIR + '/' + UNTARED_DUMPS_DIR + '/')
        
        for file in file_names:
            text_extraction.extract_text_from_xml(file)
            #text_extraction.extract_text_for_all_wikis()
        '''
        #ta = wikimain_parallel.TEXT_EXTRACTION()
        #ta.extract_text()
        cmd = ['python', 'wikimain_parallel.py']
        subprocess.Popen(cmd).wait()

    def train_model(self):
        # build itertor for vocabulary
        class TextLine(object):
            def __init__(self, dirname):
                self.dirname = dirname
                self.iter_count = 0
                # chunksize determines how many lines needs to be read at a time from file
                self.chunksize = 10000
        
            def __iter__(self):
                self.iter_count = self.iter_count + 1
                self.proc_count = 1
                # loop through directory and read text files one by one
                for file in glob.glob(self.dirname + '/' + DATA_DIR + '/' +  EXTRACTED_TEXT_DIR + '/*.txt'):
                    print('Processing ', file , ' iteration ', self.iter_count, 'Processed: ', self.proc_count)
                    with open(file,'r', encoding='utf-8') as text_contents:
                        file_lines = text_contents.readlines(self.chunksize)
                        while file_lines:
                            lines = []
                            # read line by line and replace new line characters
                            for line in file_lines:
                                lines.append(line.replace('\n','').replace('\r\n','').replace('\r','').strip())
                            # splot line into tokens
                            yield " ".join(lines).split(' ')
                            # read another chunk
                            file_lines = text_contents.readlines(self.chunksize)
                        self.proc_count = self.proc_count + 1
                    # close file
                    text_contents.close()
                    


        corpus_data = TextLine(BASE_DIR)

        # set model parameters
        model = Word2Vec(size=EMBEDDING_VECTOR_LENGTH, window=10, min_count=1, workers=2)

        # build model vocabulary
        model.build_vocab(corpus_data)

        print('Training the Model: ........')

        # train model
        model.train(corpus_data, total_examples=model.corpus_count, epochs=WORD2VEC_EPOCHS)

        # remove directory if already present
        if os.path.exists('model'):
            shutil.rmtree('model')

        # create model directory to save model    
        os.mkdir('model')

        # save model
        model.save('model/word2vec.model')
    

    # the function is to remove extra characters from labels
    def pre_process_labels(self, label):
        pre_processed_label = label
        if label.find("\"^^") != -1:
            pre_processed_label = label[1: label.index("\"^^")].lower()
        elif len(label) > 0:
            pre_processed_label = label[1:len(label)-6].lower()
        
        return pre_processed_label
    
    # extract vectors from two wikis, wikiname should be folder names
    def extract_vectors(self, wiki_1, wiki_2):
        if os.path.exists('model/word2vec.model'):
            print('loading model')
            model = Word2Vec.load('model/word2vec.model', mmap='r')
            print('loading model done ..')
            
            words = model.wv.vocab

            df_vectors_wiki_1 = pd.DataFrame(columns = ['entity_id', 'wiki_name', 'label', 'vector'])
            df_vectors_wiki_2 = pd.DataFrame(columns = ['entity_id', 'wiki_name', 'label',  'vector'])
            
            wiki_1_label_file = [file for file in glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + wiki_1 + '/*-labels.ttl') if "category-labels" not in file]
            wiki_2_label_file = [file for file in glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + wiki_2 + '/*-labels.ttl') if "category-labels" not in file]
            
            if len(wiki_1_label_file) > 0:
                my_cols = ["details"]
                df = pd.read_table(wiki_1_label_file[0], names=my_cols)
                df = df.iloc[1:len(df)-1]
                
                if len(df) > 0:
                    new_cols = df["details"].str.split(" ", n = 2, expand=True)
                    df['entity_id'] = new_cols[0].str.lower()
                    df['predicate'] = new_cols[1].str.lower()
                    df['label'] = new_cols[2].apply(self.pre_process_labels)

                    for index, row in df.iterrows():
                        if row['entity_id'] in words:
                            entity_id = row['entity_id']
                            predicate = row['predicate']
                            label =  row['label'] if predicate == '<http://www.w3.org/2000/01/rdf-schema#label>' else entity_id[entity_id.rfind('/')+1:len(entity_id)-1]
                            df_vectors_wiki_1 = df_vectors_wiki_1.append({'entity_id': entity_id, 'wiki_name':wiki_1, 'label': label, 'vector':model.wv[entity_id]}, ignore_index=True)
            
            if len(wiki_2_label_file) > 0:
                my_cols = ["details"]
                df = pd.read_table(wiki_2_label_file[0], names=my_cols)
                df = df.iloc[1:len(df)-1]
                
                if len(df) > 0:
                    new_cols = df["details"].str.split(" ", n = 2, expand=True)
                    df['entity_id'] = new_cols[0].str.lower()
                    df['predicate'] = new_cols[1].str.lower()
                    df['label'] = new_cols[2].apply(self.pre_process_labels)

                    for index, row in df.iterrows():
                        if row['entity_id'] in words:
                            entity_id = row['entity_id']
                            predicate = row['predicate']
                            label =  row['label'] if predicate == '<http://www.w3.org/2000/01/rdf-schema#label>' else entity_id[entity_id.rfind('/')+1:len(entity_id)-1]
                            df_vectors_wiki_2 = df_vectors_wiki_2.append({'entity_id': entity_id, 'wiki_name':wiki_2, 'label': label, 'vector':model.wv[entity_id]}, ignore_index=True)
            

            df_vectors_wiki_1.drop_duplicates(subset=['entity_id'], inplace = True)
            df_vectors_wiki_1= df_vectors_wiki_1.reset_index(drop=True)
            
            df_vectors_wiki_2.drop_duplicates(subset=['entity_id'], inplace = True)
            df_vectors_wiki_2= df_vectors_wiki_2.reset_index(drop=True)

            print(df_vectors_wiki_1.head())
            print(df_vectors_wiki_2.head())
            return df_vectors_wiki_1, df_vectors_wiki_2
        else:
            print('model file not present, please re-run the model')
            return None