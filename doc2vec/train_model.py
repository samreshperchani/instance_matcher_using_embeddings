from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import nltk
from nltk.tokenize import word_tokenize
import datetime
import sys 
import glob
import shutil
import pandas as pd
import logging
from gensim.models import Doc2Vec


# set configuration file path
config_path = os.path.dirname(os.getcwd()) + '/config' 

# add config file path to system
sys.path.append(config_path)

import config

# base directory of the solution
BASE_DIR = config.BASE_DIR

# path to data directory
DATA_DIR = config.DATA_DIR

# path to processed extracted dumps
PROCESSED_DUMPS_DIR = config.PROCESSED_DUMPS_DIR

# count of epochs
DOC2VEC_EPOCHS = config.DOC2VEC_EPOCHS

# length of embedding vector
EMBEDDING_VECTOR_LENGTH = config.EMBEDDING_VECTOR_LENGTH

# class to pre-process long abstracts and train model
class DOC2VEC:
    
    def pre_process_long_abstracts(self):

        # path where pre processed long abstracts will be stored
        PR_LONG_ABSTRACTS_DIR_PATH = BASE_DIR + '/' + DATA_DIR + '/pre_processed_long_abstracts' 
        
        # if the directory for pre processed long abstracts is present then remove it
        if os.path.exists(PR_LONG_ABSTRACTS_DIR_PATH):
            shutil.rmtree(PR_LONG_ABSTRACTS_DIR_PATH)
        
        # create pre processed long abstracts directory
        os.mkdir(PR_LONG_ABSTRACTS_DIR_PATH)
        
        

        # loop through extracted processed dumps and pre-process long abstracts in CSV format
        # output format has 2 columns one column to store subject other is tokenized string
        for dir in os.listdir(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/'):
            for f_name in os.listdir(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + dir):
                # check if file name ends with long-abstracts, change it to short abstracts if you want to use short abstracts
                if f_name.endswith('long-abstracts.ttl'):

                    with open(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + dir + '/' + f_name,'r', encoding='utf-8') as abs_file:
                        with open(PR_LONG_ABSTRACTS_DIR_PATH + '/' + dir + '_' + 'long_abstracts_tokenized.csv','w+', encoding='utf-8') as abs_tkn_file:
                            print('Processing: ' , dir)
                            listOfLines = abs_file.readlines()
                            # write header
                            abs_tkn_file.write("subject,tokens\n") 
                            for line in listOfLines:
                                line = line.lower()
                                if not line.startswith('#'):
                                    s_p_o = line.split(' ', 2)
                                    subject = str(s_p_o[0]).strip() #.replace("<","").replace(">","")
                                    obj = s_p_o[2].replace(".","")
                                    obj = obj[1:obj.rfind("\"")].lower().replace("\n","").replace("\"","").replace(";","")
                                    tokens = obj.split(" ")
                                    abs_tkn_file.write("\""+subject + "\",\"" + ",".join(tokens) + "\"\n") 
                        abs_tkn_file.close()
                    abs_file.close()

    
    def train_model(self):
        # enable longgine
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        
        class LabeledDocs(object):
            def __init__(self, doc_list):
                self.docs_path = doc_list
                self.iter_cnt = 1
                self.proc_cnt = 0
                
            def __iter__(self):

                print('Starting Iteration: ', self.iter_cnt)
                self.proc_cnt = 0
                
                # loop through pre processed long abstaracts and return data
                for f_name in os.listdir(self.docs_path + '/' + 'pre_processed_long_abstracts'):
                    self.proc_cnt = self.proc_cnt + 1
                    print(str(datetime.datetime.now()),': Iteration: ', self.iter_cnt,' Count: ', self.proc_cnt ,', Processing Wiki: ', f_name)
                    # just for cross check, veruify if file has valid name
                    if f_name.endswith('abstracts_tokenized.csv'):
                        df = pd.read_csv(self.docs_path + '/' + 'pre_processed_long_abstracts/' + f_name,  sep=',')
                        df = df.fillna('')
                        
                        for index, row in df.iterrows():
                            # get list of tokens
                            tokens = str(row['tokens'])
                            
                            # return data
                            yield TaggedDocument(words=tokens.split(","), tags=[row['subject'].lower()])
                
                self.iter_cnt = self.iter_cnt + 1

        
        # call labeled docs class with data folder path which will return data in iterative way
        corpus_data = LabeledDocs(BASE_DIR + '/' + DATA_DIR)
        
        # initialize model parameters
        model = Doc2Vec(vector_size=EMBEDDING_VECTOR_LENGTH, window=10, dm=0, dbow_words=0, min_count = 1, workers=2)
        
        print('Process started: ', str(datetime.datetime.now()))
        
        print('building vocabulary...')
        
        # build vocab for model
        model.build_vocab(corpus_data)
        
        print('training model...')
        
        # train model
        model.train(corpus_data, total_examples=model.corpus_count, epochs=DOC2VEC_EPOCHS)
        
        print('Process End: ', str(datetime.datetime.now()))
        
        print('saving model...')

        # remove directory if already present
        if os.path.exists('model'):
            shutil.rmtree('model')

        # create model directory to save model    
        os.mkdir('model')

        # save model
        model.save('model/doc2vec.model')

    # function to remove extra characters from labels
    def pre_process_labels(self, label):
        pre_processed_label = label
        if label.find("\"^^") != -1:
            pre_processed_label = label[1: label.index("\"^^")].lower()
        elif len(label) > 0:
            pre_processed_label = label[1:len(label)-6].lower()
        
        return pre_processed_label

    # extract vectors from two wikis, wikiname should be folder names
    def extract_vectors(self, wiki_1, wiki_2):
        if os.path.exists('model/doc2vec.model'):
            print('loading model')
            model = Doc2Vec.load('model/doc2vec.model', mmap='r')
            print('loading model done ..')
            
            tags = model.docvecs.doctags

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
                        if row['entity_id'] in tags:
                            entity_id = row['entity_id']
                            predicate = row['predicate']
                            label =  row['label'] if predicate == '<http://www.w3.org/2000/01/rdf-schema#label>' else entity_id[entity_id.rfind('/')+1:len(entity_id)-1]
                            df_vectors_wiki_1 = df_vectors_wiki_1.append({'entity_id': entity_id, 'wiki_name':wiki_1, 'label': label, 'vector':model.docvecs[entity_id]}, ignore_index=True)
            
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
                        if row['entity_id'] in tags:
                            entity_id = row['entity_id']
                            predicate = row['predicate']
                            label =  row['label'] if predicate == '<http://www.w3.org/2000/01/rdf-schema#label>' else entity_id[entity_id.rfind('/')+1:len(entity_id)-1]
                            df_vectors_wiki_2 = df_vectors_wiki_2.append({'entity_id': entity_id, 'wiki_name':wiki_2, 'label': label, 'vector':model.docvecs[entity_id]}, ignore_index=True)
            

            df_vectors_wiki_1.drop_duplicates(subset=['entity_id'], inplace = True)
            df_vectors_wiki_1= df_vectors_wiki_1.reset_index(drop=True)
            
            df_vectors_wiki_2.drop_duplicates(subset=['entity_id'], inplace = True)
            df_vectors_wiki_2= df_vectors_wiki_2.reset_index(drop=True)

            return df_vectors_wiki_1, df_vectors_wiki_2
        else:
            print('model file not present, please re-run the model')
            return None

