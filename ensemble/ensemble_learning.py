from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import glob
import logging
import datetime
import re
import sys
from pathlib import Path
import shutil
import multiprocessing as mp
import random
import time
import psycopg2
import sql
from sqlalchemy import create_engine
import pyodbc
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from numpy.random import seed

logger = logging.getLogger(__name__)

path = Path(os.path.abspath(__file__))

# set configuration file path
#config_path = os.path.dirname(os.getcwd()) + '/config' 
config_path = str(path.parent.parent) + '/config' 

# add config file path to system
sys.path.append(config_path)

import config

# base directory of the solution
BASE_DIR = config.BASE_DIR

# path to data directory
DATA_DIR = config.DATA_DIR

# ensemble dataset folder
ENSEMBLE_DATASET_PATH = config.ENSEMBLE_DATASET_PATH

# ensemble approach
ENSEMBLE_APPROACH = config.ENSEMBLE_APPROACH

# embedding vector length
EMBEDDING_VECTOR_LENGTH = config.EMBEDDING_VECTOR_LENGTH

#Reduced_dimensions
REDUCED_DIMENSIONS = 100


class ENSEMBLE_LEARNING:

    # function to transform dataset using concatenate ensemble approach
    def transform_dataset_using_concat(self, X_train, y_train, X_test):
        print('*********** using concatenation ensemble learning ****************')
        
        def get_concatenated_vectors(df):
            df_vec_en_1 = df[['doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1']]
            
            vector_col_name = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_1[vector_col_name]= pd.DataFrame(df.doc2vec_vector_entity_1.tolist(), index=df.index)
            df_vec_en_1.drop(columns=['doc2vec_vector_entity_1'], inplace=True)
        
            vector_col_name = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_1[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_entity_1.tolist(), index=df.index)
            df_vec_en_1.drop(columns=['rdf2vec_vector_entity_1'], inplace=True)
            
            vector_col_name = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_1[vector_col_name]= pd.DataFrame(df.word2vec_vector_entity_1.tolist(), index=df.index)
            df_vec_en_1.drop(columns=['word2vec_vector_entity_1'], inplace=True)

            print(df_vec_en_1.head())

            
            df_vec_en_2 = df[['doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2']]
            

            vector_col_name = ['doc2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_2[vector_col_name]= pd.DataFrame(df.doc2vec_vector_entity_2.tolist(), index=df.index)
            df_vec_en_2.drop(columns=['doc2vec_vector_entity_2'], inplace=True)
            
            vector_col_name = ['rdf2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_2[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_entity_2.tolist(), index=df.index)
            df_vec_en_2.drop(columns=['rdf2vec_vector_entity_2'], inplace=True)
            
            vector_col_name = ['word2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_2[vector_col_name]= pd.DataFrame(df.word2vec_vector_entity_2.tolist(), index=df.index)
            df_vec_en_2.drop(columns=['word2vec_vector_entity_2'], inplace=True)

            
            df = df[['entity_id_wiki_1','entity_id_wiki_2']]
            df = pd.merge(df, df_vec_en_1, left_index=True, right_index=True )
            df = pd.merge(df, df_vec_en_2, left_index=True, right_index=True )

            
            return df
        
        #### tranformation on the training set ################
        print('***********applying transofrmation on training set************')
        X_train = get_concatenated_vectors(X_train)
        
        #### tranformation on the test set ################
        print('***********applying transofrmation on test set************')
        X_test = get_concatenated_vectors(X_test)

        return X_train, y_train, X_test
    
    # function to transform dataset using average ensemble approach
    def transform_dataset_using_average(self, X_train, y_train, X_test):
        
        print('*********** using average ensemble learning ****************')
        # functions to calculate calculate average vectors
        def avg_vector_ent1(x):
            vectors = [x['doc2vec_vector_entity_1'], x['word2vec_vector_entity_1'], x['rdf2vec_vector_entity_1']]
            return  np.mean(vectors, axis = 0) 
        
        def avg_vector_ent2(x):
            vectors = [x['doc2vec_vector_entity_2'], x['word2vec_vector_entity_2'], x['rdf2vec_vector_entity_2']]
            return  np.mean(vectors, axis = 0)

        def calculate_average_vectors(df):
            df['avg_vector_ent1'] = df.apply(avg_vector_ent1, axis=1)
            vector_col_name = ['ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df[vector_col_name]= pd.DataFrame(df.avg_vector_ent1.tolist(), index=df.index)
            df.drop(columns=['avg_vector_ent1'], inplace=True)
            df.drop(columns=['doc2vec_vector_entity_1'], inplace=True)
            df.drop(columns=['word2vec_vector_entity_1'], inplace=True)
            df.drop(columns=['rdf2vec_vector_entity_1'], inplace=True)
            
            df['avg_vector_ent2'] = df.apply(avg_vector_ent2, axis=1)
            vector_col_name = ['ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df[vector_col_name]= pd.DataFrame(df.avg_vector_ent2.tolist(), index=df.index)
            df.drop(columns=['avg_vector_ent2'], inplace=True)
            df.drop(columns=['doc2vec_vector_entity_2'], inplace=True)
            df.drop(columns=['word2vec_vector_entity_2'], inplace=True)
            df.drop(columns=['rdf2vec_vector_entity_2'], inplace=True)

            return df

        ############ transformation on the training set ###################
        X_train = calculate_average_vectors(X_train)
        ############ transformation on the test set ###################    
        X_test = calculate_average_vectors(X_test)

        return X_train, y_train, X_test

    
    # PCA transformation using network 1
    def transform_dataset_using_pca_network1(self, X_train, y_train, X_test):
        print('*********** using PCA(Network 1) ensemble learning ****************')

        vector_col_name_doc2vec = ['doc2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        vector_col_name_word2vec = ['word2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        vector_col_name_rdf2vec = ['rdf2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]


        def get_entity_vectors(df):
            df_vec_en_1 = df[['entity_id_wiki_1','doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1']]
            df_vec_en_1.rename(columns = {'entity_id_wiki_1': 'entity_id','doc2vec_vector_entity_1' : 'doc2vec_vector', 
            'word2vec_vector_entity_1' : 'word2vec_vector', 'rdf2vec_vector_entity_1' : 'rdf2vec_vector'}, inplace=True)
            
            df_vec_en_2 = df[['entity_id_wiki_2','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2']]
            df_vec_en_2.rename(columns = {'entity_id_wiki_2': 'entity_id', 'doc2vec_vector_entity_2' : 'doc2vec_vector', 
            'word2vec_vector_entity_2' : 'word2vec_vector', 'rdf2vec_vector_entity_2' : 'rdf2vec_vector'}, inplace=True)
            
            df_concat = pd.concat([df_vec_en_1, df_vec_en_2])
            df_concat = df_concat.reset_index(drop=True)
            df_concat[vector_col_name_doc2vec]= pd.DataFrame(df_concat.doc2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['doc2vec_vector'], inplace=True)
            
            df_concat[vector_col_name_word2vec]= pd.DataFrame(df_concat.word2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['word2vec_vector'], inplace=True)
            
            df_concat[vector_col_name_rdf2vec]= pd.DataFrame(df_concat.rdf2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['rdf2vec_vector'], inplace=True)
            
            df_concat = df_concat.drop_duplicates(subset=['entity_id'])
            df_concat = df_concat.reset_index(drop=True)
            
            return df_concat
        
        
        df_train = get_entity_vectors(X_train)
        df_test = get_entity_vectors(X_test)
        
        df_vectors_train = df_train.drop(columns = ['entity_id'])
        df_vectors_test = df_test.drop(columns = ['entity_id'])
        
        pca = PCA(n_components=300, whiten = False, random_state = 2019)
        pca.fit(df_vectors_train)
        
        X_pca_train = pca.transform(df_vectors_train)
        df_pca_train = pd.DataFrame.from_records(X_pca_train)
        
        X_pca_test = pca.transform(df_vectors_test)
        df_pca_test = pd.DataFrame.from_records(X_pca_test)
        
        
        df_train = pd.merge(df_train, df_pca_train, how='inner', left_index=True, right_index=True)
        df_train.drop(columns = vector_col_name_doc2vec + vector_col_name_word2vec + vector_col_name_rdf2vec, inplace=True)
        
        df_test = pd.merge(df_test, df_pca_test, how='inner', left_index=True, right_index=True)
        df_test.drop(columns = vector_col_name_doc2vec + vector_col_name_word2vec + vector_col_name_rdf2vec, inplace=True)

        X_train.drop(columns=['doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2'], inplace=True)
        df_model_train = pd.merge(X_train, df_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        df_model_train.drop(columns=['entity_id'], inplace=True)

        df_model_train = pd.merge(df_model_train, df_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        df_model_train.drop(columns=['entity_id'], inplace=True)
        df_model_train = df_model_train.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])
        
        
        X_test.drop(columns=['doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2'], inplace=True)
        df_model_test = pd.merge(X_test, df_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        df_model_test.drop(columns=['entity_id'], inplace=True)
        
        df_model_test = pd.merge(df_model_test, df_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        df_model_test.drop(columns=['entity_id'], inplace=True)
        df_model_test = df_model_test.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])


        df_train = pd.concat([df_model_train, y_train], axis=1)
        df_test = df_model_test


        df = pd.concat([X_train, y_train], axis=1)
        df = df[['entity_id_wiki_1', 'entity_id_wiki_2','label']]

        df_model_train = pd.merge(df_model_train, df, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_train = df_model_train['label']
        y_train=y_train.astype('int')
        df_model_train = df_model_train.drop(['label'], axis=1)

        return df_model_train, y_train, df_test


    # SVD transformation using network 1
    def transform_dataset_using_svd_network1(self, X_train, y_train, X_test):
        vector_col_name_doc2vec = ['doc2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        vector_col_name_word2vec = ['word2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        vector_col_name_rdf2vec = ['rdf2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        
        def get_entity_vectors(df):
            df_vec_en_1 = df[['entity_id_wiki_1','doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1']]
            df_vec_en_1.rename(columns = {'entity_id_wiki_1': 'entity_id','doc2vec_vector_entity_1' : 'doc2vec_vector', 
            'word2vec_vector_entity_1' : 'word2vec_vector', 'rdf2vec_vector_entity_1' : 'rdf2vec_vector'}, inplace=True)
            
            df_vec_en_2 = df[['entity_id_wiki_2','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2']]
            df_vec_en_2.rename(columns = {'entity_id_wiki_2': 'entity_id', 'doc2vec_vector_entity_2' : 'doc2vec_vector', 
            'word2vec_vector_entity_2' : 'word2vec_vector', 'rdf2vec_vector_entity_2' : 'rdf2vec_vector'}, inplace=True)
            
            df_concat = pd.concat([df_vec_en_1, df_vec_en_2])
            df_concat = df_concat.reset_index(drop=True)
            
            df_concat[vector_col_name_doc2vec]= pd.DataFrame(df_concat.doc2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['doc2vec_vector'], inplace=True)
            
            df_concat[vector_col_name_word2vec]= pd.DataFrame(df_concat.word2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['word2vec_vector'], inplace=True)
            
            df_concat[vector_col_name_rdf2vec]= pd.DataFrame(df_concat.rdf2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['rdf2vec_vector'], inplace=True)
            
            df_concat = df_concat.drop_duplicates(subset=['entity_id'])
            df_concat = df_concat.reset_index(drop=True)
            
            return df_concat
            
            
        df_train = get_entity_vectors(X_train)
        df_test = get_entity_vectors(X_test)
        
        df_vectors_train = df_train.drop(columns = ['entity_id'])
        df_vectors_test = df_test.drop(columns = ['entity_id'])
        
        svd = TruncatedSVD(n_components=300, random_state = 2019)
        svd.fit(df_vectors_train)
        
        X_svd_train = svd.transform(df_vectors_train)
        df_svd_train = pd.DataFrame.from_records(X_svd_train)
        
        X_svd_test = svd.transform(df_vectors_test)
        df_svd_test = pd.DataFrame.from_records(X_svd_test)
        
        
        
        df_train = pd.merge(df_train, df_svd_train, how='inner', left_index=True, right_index=True)
        df_train.drop(columns = vector_col_name_doc2vec + vector_col_name_word2vec + vector_col_name_rdf2vec, inplace=True)
        
        df_test = pd.merge(df_test, df_svd_test, how='inner', left_index=True, right_index=True)
        df_test.drop(columns = vector_col_name_doc2vec + vector_col_name_word2vec + vector_col_name_rdf2vec, inplace=True)
        
        X_train.drop(columns=['doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2'], inplace=True)
        df_model_train = pd.merge(X_train, df_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        df_model_train.drop(columns=['entity_id'], inplace=True)
        
        df_model_train = pd.merge(df_model_train, df_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        df_model_train.drop(columns=['entity_id'], inplace=True)
        df_model_train = df_model_train.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])
        
        
        X_test.drop(columns=['doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2'], inplace=True)
        df_model_test = pd.merge(X_test, df_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        df_model_test.drop(columns=['entity_id'], inplace=True)
        
        df_model_test = pd.merge(df_model_test, df_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        df_model_test.drop(columns=['entity_id'], inplace=True)
        df_model_test = df_model_test.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])
        
        df_train = pd.concat([df_model_train, y_train], axis=1)
        df_test = df_model_test


        df = pd.concat([X_train, y_train], axis=1)
        df = df[['entity_id_wiki_1', 'entity_id_wiki_2','label']]

        df_model_train = pd.merge(df_model_train, df, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_train = df_model_train['label']
        y_train=y_train.astype('int')
        df_model_train = df_model_train.drop(['label'], axis=1)


        return df_model_train, y_train, df_model_test


    # transformation using autoencoder network 1
    def transform_dataset_using_autoencoder_network1(self, X_train, y_train, X_test):
        
        # reduce dimensions using autoencoder
        def reduce_dimension(dataset):
            encoding_dim = 300
            dataset['label'] = 0
            X = dataset
            
            X = dataset.drop(columns=['label'])
            Y = dataset['label']
            sX = minmax_scale(X, axis = 0)
            ncol = sX.shape[1]
            
            X_train, X_test, Y_train, Y_test = train_test_split(sX, Y, train_size = 0.6, random_state = seed(2017))
            
            input_dim = Input(shape = (ncol, ))
            
            # Encoder Layers
            encoded1 = Dense(900, activation = 'relu')(input_dim)
            encoded2 = Dense(750, activation = 'relu')(encoded1)
            encoded3 = Dense(600, activation = 'relu')(encoded2)
            encoded4 = Dense(450, activation = 'relu')(encoded3)
            encoded5 = Dense(encoding_dim, activation = 'relu')(encoded4)
            
            # Decoder Layers
            decoded1 = Dense(450, activation = 'relu')(encoded5)
            decoded2 = Dense(600, activation = 'relu')(decoded1)
            decoded3 = Dense(750, activation = 'relu')(decoded2)
            decoded4 = Dense(900, activation = 'relu')(decoded3)
            
            # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
            autoencoder = Model(input = input_dim, output = decoded4)
            
            # CONFIGURE AND TRAIN THE AUTOENCODER
            autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
            autoencoder.fit(X_train, X_train, nb_epoch = 15, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
            
            # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
            encoder = Model(input = input_dim, output = encoded5)
            encoded_input = Input(shape = (encoding_dim, ))
            encoded_out = encoder.predict(X)
            
            return encoder, pd.DataFrame(encoded_out)
            
        vector_col_name_doc2vec = ['doc2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        vector_col_name_word2vec = ['word2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        vector_col_name_rdf2vec = ['rdf2vec_ent_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        
        def get_entity_vectors(df):
            df_vec_en_1 = df[['entity_id_wiki_1','doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1']]
            df_vec_en_1.rename(columns = {'entity_id_wiki_1': 'entity_id','doc2vec_vector_entity_1' : 'doc2vec_vector', 
            'word2vec_vector_entity_1' : 'word2vec_vector', 'rdf2vec_vector_entity_1' : 'rdf2vec_vector'}, inplace=True)
            
            df_vec_en_2 = df[['entity_id_wiki_2','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2']]
            df_vec_en_2.rename(columns = {'entity_id_wiki_2': 'entity_id', 'doc2vec_vector_entity_2' : 'doc2vec_vector', 
            'word2vec_vector_entity_2' : 'word2vec_vector', 'rdf2vec_vector_entity_2' : 'rdf2vec_vector'}, inplace=True)
            
            df_concat = pd.concat([df_vec_en_1, df_vec_en_2])
            df_concat = df_concat.reset_index(drop=True)
            
            df_concat[vector_col_name_doc2vec]= pd.DataFrame(df_concat.doc2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['doc2vec_vector'], inplace=True)
            
            df_concat[vector_col_name_word2vec]= pd.DataFrame(df_concat.word2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['word2vec_vector'], inplace=True)
            
            df_concat[vector_col_name_rdf2vec]= pd.DataFrame(df_concat.rdf2vec_vector.tolist(), index=df_concat.index)
            df_concat.drop(columns=['rdf2vec_vector'], inplace=True)
            
            df_concat = df_concat.drop_duplicates(subset=['entity_id'])
            df_concat = df_concat.reset_index(drop=True)
            
            return df_concat

        df_train = get_entity_vectors(X_train)
        df_test = get_entity_vectors(X_test)

        df_vectors_train = df_train.drop(columns = ['entity_id'])
        df_vectors_test = df_test.drop(columns = ['entity_id'])


        encoder, df_ae_train = reduce_dimension(df_vectors_train)
        df_ae_test = pd.DataFrame(encoder.predict(df_vectors_test))
        
        
        
        df_train = pd.merge(df_train, df_ae_train, how='inner', left_index=True, right_index=True)
        df_train.drop(columns = vector_col_name_doc2vec + vector_col_name_word2vec + vector_col_name_rdf2vec, inplace=True)

        df_test = pd.merge(df_test, df_ae_test, how='inner', left_index=True, right_index=True)
        df_test.drop(columns = vector_col_name_doc2vec + vector_col_name_word2vec + vector_col_name_rdf2vec, inplace=True)

        X_train.drop(columns=['doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2'], inplace=True)
        df_model_train = pd.merge(X_train, df_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        df_model_train.drop(columns=['entity_id'], inplace=True)

        df_model_train = pd.merge(df_model_train, df_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        df_model_train.drop(columns=['entity_id'], inplace=True)
        df_model_train = df_model_train.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])


        X_test.drop(columns=['doc2vec_vector_entity_1','word2vec_vector_entity_1','rdf2vec_vector_entity_1','doc2vec_vector_entity_2','word2vec_vector_entity_2','rdf2vec_vector_entity_2'], inplace=True)
        df_model_test = pd.merge(X_test, df_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        df_model_test.drop(columns=['entity_id'], inplace=True)


        df_model_test = pd.merge(df_model_test, df_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        df_model_test.drop(columns=['entity_id'], inplace=True)
        df_model_test = df_model_test.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])


        df_model = pd.concat([df_train, df_test])
        
        df = pd.concat([X_train, y_train], axis=1)
        df = df[['entity_id_wiki_1', 'entity_id_wiki_2','label']]

        df_model_train = pd.merge(df_model_train, df, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_train = df_model_train['label']
        y_train=y_train.astype('int')
        df_model_train = df_model_train.drop(['label'], axis=1)


        return df_model_train, y_train, df_model_test


    # PCA transformation using network 2
    def transform_dataset_using_pca_network2(self, X_train, y_train, X_test):

        df_train = pd.concat([X_train, y_train], axis= 1)
        df_test= X_test

        vector_col_name_doc2vec_ent1 = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_doc2vec_ent1]= pd.DataFrame(X_train.doc2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['doc2vec_vector_entity_1'], inplace=True)

        vector_col_name_doc2vec_ent2 = ['doc2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_doc2vec_ent2]= pd.DataFrame(X_train.doc2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['doc2vec_vector_entity_2'], inplace=True)

        vector_col_name_word2vec_ent1 = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_word2vec_ent1]= pd.DataFrame(X_train.word2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['word2vec_vector_entity_1'], inplace=True)

        vector_col_name_word2vec_ent2 = ['word2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_word2vec_ent2]= pd.DataFrame(X_train.word2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['word2vec_vector_entity_2'], inplace=True)

        vector_col_name_rdf2vec_ent1 = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_rdf2vec_ent1]= pd.DataFrame(X_train.rdf2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['rdf2vec_vector_entity_1'], inplace=True)

        vector_col_name_rdf2vec_ent2 = ['rdf2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_rdf2vec_ent2]= pd.DataFrame(X_train.rdf2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['rdf2vec_vector_entity_2'], inplace=True)


        vector_col_name_doc2vec_ent1 = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_doc2vec_ent1]= pd.DataFrame(X_test.doc2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['doc2vec_vector_entity_1'], inplace=True)

        vector_col_name_doc2vec_ent2 = ['doc2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_doc2vec_ent2]= pd.DataFrame(X_test.doc2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['doc2vec_vector_entity_2'], inplace=True)

        vector_col_name_word2vec_ent1 = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_word2vec_ent1]= pd.DataFrame(X_test.word2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['word2vec_vector_entity_1'], inplace=True)

        vector_col_name_word2vec_ent2 = ['word2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_word2vec_ent2]= pd.DataFrame(X_test.word2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['word2vec_vector_entity_2'], inplace=True)

        vector_col_name_rdf2vec_ent1 = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_rdf2vec_ent1]= pd.DataFrame(X_test.rdf2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['rdf2vec_vector_entity_1'], inplace=True)

        vector_col_name_rdf2vec_ent2 = ['rdf2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_rdf2vec_ent2]= pd.DataFrame(X_test.rdf2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['rdf2vec_vector_entity_2'], inplace=True)


        
        print('*********** using PCA(Network 2) ensemble learning ****************')
        ########################### PCA on DOC2Vec #############################################
        vector_col_name_doc2vec = ['doc2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_doc2vec.append('entity_id')
        df_doc2vec_ent1_train = X_train[vector_col_name_doc2vec_ent1 + ['entity_id_wiki_1']]
        df_doc2vec_ent1_train.columns = vector_col_name_doc2vec

        df_doc2vec_ent2_train = X_train[vector_col_name_doc2vec_ent2 + ['entity_id_wiki_2']]
        df_doc2vec_ent2_train.columns = vector_col_name_doc2vec

        df_doc2vec_entities_train = pd.concat([df_doc2vec_ent1_train, df_doc2vec_ent2_train])
        df_doc2vec_entities_train = df_doc2vec_entities_train.reset_index(drop=True)
        df_doc2vec_entities_train_vectors = df_doc2vec_entities_train.drop(columns=['entity_id'])


        pca_doc2vec = PCA(n_components=REDUCED_DIMENSIONS, whiten = False, random_state = 2019)
        pca_doc2vec.fit(df_doc2vec_entities_train_vectors)

        X_pca_doc2vec_train = pca_doc2vec.transform(df_doc2vec_entities_train_vectors)
        df_pca_doc2vec_train = pd.DataFrame.from_records(X_pca_doc2vec_train)
        df_pca_doc2vec_train.columns = ['doc2vec_pca_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_doc2vec_entities_train = df_doc2vec_entities_train[['entity_id']]
        df_pca_doc2vec_train = pd.merge(df_pca_doc2vec_train, df_doc2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_doc2vec_ent1_test = X_test[vector_col_name_doc2vec_ent1 + ['entity_id_wiki_1']]
        df_doc2vec_ent1_test.columns = vector_col_name_doc2vec
        df_doc2vec_ent2_test = X_test[vector_col_name_doc2vec_ent2+ ['entity_id_wiki_2']]
        df_doc2vec_ent2_test.columns = vector_col_name_doc2vec
        df_doc2vec_entities_test = pd.concat([df_doc2vec_ent1_test, df_doc2vec_ent2_test])
        df_doc2vec_entities_test = df_doc2vec_entities_test.reset_index(drop=True)
        df_doc2vec_entities_test_vectors = df_doc2vec_entities_test.drop(columns=['entity_id'])

        X_pca_doc2vec_test = pca_doc2vec.transform(df_doc2vec_entities_test_vectors)
        df_pca_doc2vec_test = pd.DataFrame.from_records(X_pca_doc2vec_test)
        df_pca_doc2vec_test.columns = ['doc2vec_pca_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_doc2vec_entities_test = df_doc2vec_entities_test[['entity_id']]

        print('After PCA Test: ', len(df_pca_doc2vec_test))

        df_pca_doc2vec_test = pd.merge(df_pca_doc2vec_test, df_doc2vec_entities_test, how='inner', right_index=True, left_index=True)


        ########################### PCA on Word2Vec #############################################
        vector_col_name_word2vec = ['word2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_word2vec.append('entity_id')
        df_word2vec_ent1_train = X_train[vector_col_name_word2vec_ent1 + ['entity_id_wiki_1']]
        df_word2vec_ent1_train.columns = vector_col_name_word2vec

        df_word2vec_ent2_train = X_train[vector_col_name_word2vec_ent2 + ['entity_id_wiki_2']]
        df_word2vec_ent2_train.columns = vector_col_name_word2vec

        df_word2vec_entities_train = pd.concat([df_word2vec_ent1_train, df_word2vec_ent2_train])
        df_word2vec_entities_train = df_word2vec_entities_train.reset_index(drop=True)
        df_word2vec_entities_train_vectors = df_word2vec_entities_train.drop(columns=['entity_id'])

        print(df_word2vec_entities_train.head())

        pca_word2vec = PCA(n_components=REDUCED_DIMENSIONS, whiten = False, random_state = 2019)
        pca_word2vec.fit(df_word2vec_entities_train_vectors)

        X_pca_word2vec_train = pca_word2vec.transform(df_word2vec_entities_train_vectors)
        df_pca_word2vec_train = pd.DataFrame.from_records(X_pca_word2vec_train)
        df_pca_word2vec_train.columns = ['word2vec_pca_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_word2vec_entities_train = df_word2vec_entities_train[['entity_id']]
        df_pca_word2vec_train = pd.merge(df_pca_word2vec_train, df_word2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_word2vec_ent1_test = X_test[vector_col_name_word2vec_ent1 + ['entity_id_wiki_1']]
        df_word2vec_ent1_test.columns = vector_col_name_word2vec
        df_word2vec_ent2_test = X_test[vector_col_name_word2vec_ent2+ ['entity_id_wiki_2']]
        df_word2vec_ent2_test.columns = vector_col_name_word2vec
        df_word2vec_entities_test = pd.concat([df_word2vec_ent1_test, df_word2vec_ent2_test])
        df_word2vec_entities_test = df_word2vec_entities_test.reset_index(drop=True)
        df_word2vec_entities_test_vectors = df_word2vec_entities_test.drop(columns=['entity_id'])

        X_pca_word2vec_test = pca_word2vec.transform(df_word2vec_entities_test_vectors)
        df_pca_word2vec_test = pd.DataFrame.from_records(X_pca_word2vec_test)
        df_pca_word2vec_test.columns = ['word2vec_pca_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_word2vec_entities_test = df_word2vec_entities_test[['entity_id']]

        print('After PCA Test: ', len(df_pca_word2vec_test))

        df_pca_word2vec_test = pd.merge(df_pca_word2vec_test, df_word2vec_entities_test, how='inner', right_index=True, left_index=True)

        ########################### PCA on RDF2Vec #############################################
        vector_col_name_rdf2vec = ['rdf2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_rdf2vec.append('entity_id')
        df_rdf2vec_ent1_train = X_train[vector_col_name_rdf2vec_ent1 + ['entity_id_wiki_1']]
        df_rdf2vec_ent1_train.columns = vector_col_name_rdf2vec

        df_rdf2vec_ent2_train = X_train[vector_col_name_rdf2vec_ent2 + ['entity_id_wiki_2']]
        df_rdf2vec_ent2_train.columns = vector_col_name_rdf2vec

        df_rdf2vec_entities_train = pd.concat([df_rdf2vec_ent1_train, df_rdf2vec_ent2_train])
        df_rdf2vec_entities_train = df_rdf2vec_entities_train.reset_index(drop=True)
        df_rdf2vec_entities_train_vectors = df_rdf2vec_entities_train.drop(columns=['entity_id'])

        print(df_rdf2vec_entities_train.head())

        pca_rdf2vec = PCA(n_components=REDUCED_DIMENSIONS, whiten = False, random_state = 2019)
        pca_rdf2vec.fit(df_rdf2vec_entities_train_vectors)

        X_pca_rdf2vec_train = pca_rdf2vec.transform(df_rdf2vec_entities_train_vectors)
        df_pca_rdf2vec_train = pd.DataFrame.from_records(X_pca_rdf2vec_train)
        df_pca_rdf2vec_train.columns = ['rdf2vec_pca_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]

        df_rdf2vec_entities_train = df_rdf2vec_entities_train[['entity_id']]
        df_pca_rdf2vec_train = pd.merge(df_pca_rdf2vec_train, df_rdf2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_rdf2vec_ent1_test = X_test[vector_col_name_rdf2vec_ent1 + ['entity_id_wiki_1']]
        df_rdf2vec_ent1_test.columns = vector_col_name_rdf2vec
        df_rdf2vec_ent2_test = X_test[vector_col_name_rdf2vec_ent2+ ['entity_id_wiki_2']]
        df_rdf2vec_ent2_test.columns = vector_col_name_rdf2vec
        df_rdf2vec_entities_test = pd.concat([df_rdf2vec_ent1_test, df_rdf2vec_ent2_test])
        df_rdf2vec_entities_test = df_rdf2vec_entities_test.reset_index(drop=True)
        df_rdf2vec_entities_test_vectors = df_rdf2vec_entities_test.drop(columns=['entity_id'])

        X_pca_rdf2vec_test = pca_rdf2vec.transform(df_rdf2vec_entities_test_vectors)
        df_pca_rdf2vec_test = pd.DataFrame.from_records(X_pca_rdf2vec_test)
        df_pca_rdf2vec_test.columns = ['rdf2vec_pca_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_rdf2vec_entities_test = df_rdf2vec_entities_test[['entity_id']]

        print('After PCA Test: ', len(df_pca_rdf2vec_test))

        df_pca_rdf2vec_test = pd.merge(df_pca_rdf2vec_test, df_rdf2vec_entities_test, how='inner', right_index=True, left_index=True)


        X_train = X_train[['entity_id_wiki_1', 'entity_id_wiki_2']]
        X_train = pd.merge(X_train, df_pca_doc2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_pca_doc2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_train = pd.merge(X_train, df_pca_word2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_pca_word2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_train = pd.merge(X_train, df_pca_rdf2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_pca_rdf2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])


        print(len(X_train))

        X_test = X_test[['entity_id_wiki_1', 'entity_id_wiki_2']]
        X_test = pd.merge(X_test, df_pca_doc2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_pca_doc2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_test = pd.merge(X_test, df_pca_word2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_pca_word2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_test = pd.merge(X_test, df_pca_rdf2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_pca_rdf2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])



        df_train = df_train[['entity_id_wiki_1', 'entity_id_wiki_2','label']]
        df_test = df_test[['entity_id_wiki_1', 'entity_id_wiki_2']]

        X_train = pd.merge(X_train, df_train, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_train = X_train['label']
        y_train = y_train.astype('int')
        X_train = X_train.drop(['label'], axis=1)


        X_test = pd.merge(X_test, df_test, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])

        return X_train, y_train, X_test



    # SVD transformation using network 2
    def transform_dataset_using_svd_network2(self, X_train, y_train, X_test):
        print('*********** using SVD(Network 2) ensemble learning ****************')
        
        df_train = pd.concat([X_train, y_train], axis= 1)
        df_test= X_test

        vector_col_name_doc2vec_ent1 = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_doc2vec_ent1]= pd.DataFrame(X_train.doc2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['doc2vec_vector_entity_1'], inplace=True)

        vector_col_name_doc2vec_ent2 = ['doc2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_doc2vec_ent2]= pd.DataFrame(X_train.doc2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['doc2vec_vector_entity_2'], inplace=True)

        vector_col_name_word2vec_ent1 = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_word2vec_ent1]= pd.DataFrame(X_train.word2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['word2vec_vector_entity_1'], inplace=True)

        vector_col_name_word2vec_ent2 = ['word2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_word2vec_ent2]= pd.DataFrame(X_train.word2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['word2vec_vector_entity_2'], inplace=True)

        vector_col_name_rdf2vec_ent1 = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_rdf2vec_ent1]= pd.DataFrame(X_train.rdf2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['rdf2vec_vector_entity_1'], inplace=True)

        vector_col_name_rdf2vec_ent2 = ['rdf2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_rdf2vec_ent2]= pd.DataFrame(X_train.rdf2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['rdf2vec_vector_entity_2'], inplace=True)


        vector_col_name_doc2vec_ent1 = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_doc2vec_ent1]= pd.DataFrame(X_test.doc2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['doc2vec_vector_entity_1'], inplace=True)

        vector_col_name_doc2vec_ent2 = ['doc2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_doc2vec_ent2]= pd.DataFrame(X_test.doc2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['doc2vec_vector_entity_2'], inplace=True)

        vector_col_name_word2vec_ent1 = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_word2vec_ent1]= pd.DataFrame(X_test.word2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['word2vec_vector_entity_1'], inplace=True)

        vector_col_name_word2vec_ent2 = ['word2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_word2vec_ent2]= pd.DataFrame(X_test.word2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['word2vec_vector_entity_2'], inplace=True)

        vector_col_name_rdf2vec_ent1 = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_rdf2vec_ent1]= pd.DataFrame(X_test.rdf2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['rdf2vec_vector_entity_1'], inplace=True)

        vector_col_name_rdf2vec_ent2 = ['rdf2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_rdf2vec_ent2]= pd.DataFrame(X_test.rdf2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['rdf2vec_vector_entity_2'], inplace=True)

        ########################### SVD on DOC2Vec #############################################
        vector_col_name_doc2vec = ['doc2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_doc2vec.append('entity_id')
        df_doc2vec_ent1_train = X_train[vector_col_name_doc2vec_ent1 + ['entity_id_wiki_1']]
        df_doc2vec_ent1_train.columns = vector_col_name_doc2vec

        df_doc2vec_ent2_train = X_train[vector_col_name_doc2vec_ent2 + ['entity_id_wiki_2']]
        df_doc2vec_ent2_train.columns = vector_col_name_doc2vec

        df_doc2vec_entities_train = pd.concat([df_doc2vec_ent1_train, df_doc2vec_ent2_train])
        df_doc2vec_entities_train = df_doc2vec_entities_train.reset_index(drop=True)
        df_doc2vec_entities_train_vectors = df_doc2vec_entities_train.drop(columns=['entity_id'])


        svd_doc2vec = TruncatedSVD(n_components=REDUCED_DIMENSIONS, random_state = 2019)
        svd_doc2vec.fit(df_doc2vec_entities_train_vectors)

        X_svd_doc2vec_train = svd_doc2vec.transform(df_doc2vec_entities_train_vectors)
        df_svd_doc2vec_train = pd.DataFrame.from_records(X_svd_doc2vec_train)
        df_svd_doc2vec_train.columns = ['doc2vec_svd_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_doc2vec_entities_train = df_doc2vec_entities_train[['entity_id']]
        df_svd_doc2vec_train = pd.merge(df_svd_doc2vec_train, df_doc2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_doc2vec_ent1_test = X_test[vector_col_name_doc2vec_ent1 + ['entity_id_wiki_1']]
        df_doc2vec_ent1_test.columns = vector_col_name_doc2vec
        df_doc2vec_ent2_test = X_test[vector_col_name_doc2vec_ent2+ ['entity_id_wiki_2']]
        df_doc2vec_ent2_test.columns = vector_col_name_doc2vec
        df_doc2vec_entities_test = pd.concat([df_doc2vec_ent1_test, df_doc2vec_ent2_test])
        df_doc2vec_entities_test = df_doc2vec_entities_test.reset_index(drop=True)
        df_doc2vec_entities_test_vectors = df_doc2vec_entities_test.drop(columns=['entity_id'])

        X_svd_doc2vec_test = svd_doc2vec.transform(df_doc2vec_entities_test_vectors)
        df_svd_doc2vec_test = pd.DataFrame.from_records(X_svd_doc2vec_test)
        df_svd_doc2vec_test.columns = ['doc2vec_svd_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_doc2vec_entities_test = df_doc2vec_entities_test[['entity_id']]

        print('After svd Test: ', len(df_svd_doc2vec_test))

        df_svd_doc2vec_test = pd.merge(df_svd_doc2vec_test, df_doc2vec_entities_test, how='inner', right_index=True, left_index=True)


        ########################### SVD on Word2Vec #############################################
        vector_col_name_word2vec = ['word2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_word2vec.append('entity_id')
        df_word2vec_ent1_train = X_train[vector_col_name_word2vec_ent1 + ['entity_id_wiki_1']]
        df_word2vec_ent1_train.columns = vector_col_name_word2vec

        df_word2vec_ent2_train = X_train[vector_col_name_word2vec_ent2 + ['entity_id_wiki_2']]
        df_word2vec_ent2_train.columns = vector_col_name_word2vec

        df_word2vec_entities_train = pd.concat([df_word2vec_ent1_train, df_word2vec_ent2_train])
        df_word2vec_entities_train = df_word2vec_entities_train.reset_index(drop=True)
        df_word2vec_entities_train_vectors = df_word2vec_entities_train.drop(columns=['entity_id'])

        print(df_word2vec_entities_train.head())

        svd_word2vec = TruncatedSVD(n_components=REDUCED_DIMENSIONS, random_state = 2019)
        svd_word2vec.fit(df_word2vec_entities_train_vectors)

        X_svd_word2vec_train = svd_word2vec.transform(df_word2vec_entities_train_vectors)
        df_svd_word2vec_train = pd.DataFrame.from_records(X_svd_word2vec_train)
        df_svd_word2vec_train.columns = ['word2vec_svd_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_word2vec_entities_train = df_word2vec_entities_train[['entity_id']]
        df_svd_word2vec_train = pd.merge(df_svd_word2vec_train, df_word2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_word2vec_ent1_test = X_test[vector_col_name_word2vec_ent1 + ['entity_id_wiki_1']]
        df_word2vec_ent1_test.columns = vector_col_name_word2vec
        df_word2vec_ent2_test = X_test[vector_col_name_word2vec_ent2+ ['entity_id_wiki_2']]
        df_word2vec_ent2_test.columns = vector_col_name_word2vec
        df_word2vec_entities_test = pd.concat([df_word2vec_ent1_test, df_word2vec_ent2_test])
        df_word2vec_entities_test = df_word2vec_entities_test.reset_index(drop=True)
        df_word2vec_entities_test_vectors = df_word2vec_entities_test.drop(columns=['entity_id'])

        X_svd_word2vec_test = svd_word2vec.transform(df_word2vec_entities_test_vectors)
        df_svd_word2vec_test = pd.DataFrame.from_records(X_svd_word2vec_test)
        df_svd_word2vec_test.columns = ['word2vec_svd_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_word2vec_entities_test = df_word2vec_entities_test[['entity_id']]

        print('After svd Test: ', len(df_svd_word2vec_test))

        df_svd_word2vec_test = pd.merge(df_svd_word2vec_test, df_word2vec_entities_test, how='inner', right_index=True, left_index=True)

        ########################### SVD on RDF2Vec #############################################
        vector_col_name_rdf2vec = ['rdf2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_rdf2vec.append('entity_id')
        df_rdf2vec_ent1_train = X_train[vector_col_name_rdf2vec_ent1 + ['entity_id_wiki_1']]
        df_rdf2vec_ent1_train.columns = vector_col_name_rdf2vec

        df_rdf2vec_ent2_train = X_train[vector_col_name_rdf2vec_ent2 + ['entity_id_wiki_2']]
        df_rdf2vec_ent2_train.columns = vector_col_name_rdf2vec

        df_rdf2vec_entities_train = pd.concat([df_rdf2vec_ent1_train, df_rdf2vec_ent2_train])
        df_rdf2vec_entities_train = df_rdf2vec_entities_train.reset_index(drop=True)
        df_rdf2vec_entities_train_vectors = df_rdf2vec_entities_train.drop(columns=['entity_id'])

        print(df_rdf2vec_entities_train.head())

        svd_rdf2vec = TruncatedSVD(n_components=REDUCED_DIMENSIONS, random_state = 2019)
        svd_rdf2vec.fit(df_rdf2vec_entities_train_vectors)

        X_svd_rdf2vec_train = svd_rdf2vec.transform(df_rdf2vec_entities_train_vectors)
        df_svd_rdf2vec_train = pd.DataFrame.from_records(X_svd_rdf2vec_train)
        df_svd_rdf2vec_train.columns = ['rdf2vec_svd_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]

        df_rdf2vec_entities_train = df_rdf2vec_entities_train[['entity_id']]
        df_svd_rdf2vec_train = pd.merge(df_svd_rdf2vec_train, df_rdf2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_rdf2vec_ent1_test = X_test[vector_col_name_rdf2vec_ent1 + ['entity_id_wiki_1']]
        df_rdf2vec_ent1_test.columns = vector_col_name_rdf2vec
        df_rdf2vec_ent2_test = X_test[vector_col_name_rdf2vec_ent2+ ['entity_id_wiki_2']]
        df_rdf2vec_ent2_test.columns = vector_col_name_rdf2vec
        df_rdf2vec_entities_test = pd.concat([df_rdf2vec_ent1_test, df_rdf2vec_ent2_test])
        df_rdf2vec_entities_test = df_rdf2vec_entities_test.reset_index(drop=True)
        df_rdf2vec_entities_test_vectors = df_rdf2vec_entities_test.drop(columns=['entity_id'])

        X_svd_rdf2vec_test = svd_rdf2vec.transform(df_rdf2vec_entities_test_vectors)
        df_svd_rdf2vec_test = pd.DataFrame.from_records(X_svd_rdf2vec_test)
        df_svd_rdf2vec_test.columns = ['rdf2vec_svd_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_rdf2vec_entities_test = df_rdf2vec_entities_test[['entity_id']]

        print('After svd Test: ', len(df_svd_rdf2vec_test))

        df_svd_rdf2vec_test = pd.merge(df_svd_rdf2vec_test, df_rdf2vec_entities_test, how='inner', right_index=True, left_index=True)


        X_train = X_train[['entity_id_wiki_1', 'entity_id_wiki_2']]
        X_train = pd.merge(X_train, df_svd_doc2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_svd_doc2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_train = pd.merge(X_train, df_svd_word2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_svd_word2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_train = pd.merge(X_train, df_svd_rdf2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_svd_rdf2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])


        print(len(X_train))

        X_test = X_test[['entity_id_wiki_1', 'entity_id_wiki_2']]
        X_test = pd.merge(X_test, df_svd_doc2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_svd_doc2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_test = pd.merge(X_test, df_svd_word2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_svd_word2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_test = pd.merge(X_test, df_svd_rdf2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_svd_rdf2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])



        df_train = df_train[['entity_id_wiki_1', 'entity_id_wiki_2','label']]
        df_test = df_test[['entity_id_wiki_1', 'entity_id_wiki_2']]

        X_train = pd.merge(X_train, df_train, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_train = X_train['label']
        y_train = y_train.astype('int')
        X_train = X_train.drop(['label'], axis=1)


        X_test = pd.merge(X_test, df_test, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])

        return X_train, y_train, X_test
    
    # Autoencoder using Network 2
    def transform_dataset_using_autoencoder_network2(self, X_train, y_train, X_test):
        print('*********** using Autoencoder(Network 2) ensemble learning ****************')
        df_train = pd.concat([X_train, y_train], axis= 1)
        df_test= X_test
        
        def reduce_dimension(dataset):
            encoding_dim = 100
            dataset['label'] = 0
            X = dataset
            
            X = dataset.drop(columns=['label'])
            Y = dataset['label']
            sX = minmax_scale(X, axis = 0)
            ncol = sX.shape[1]
            
            X_train, X_test, Y_train, Y_test = train_test_split(sX, Y, train_size = 0.6, random_state = seed(2017))
            
            input_dim = Input(shape = (ncol, ))
            
            # Encoder Layers
            encoded1 = Dense(300, activation = 'relu')(input_dim)
            encoded2 = Dense(250, activation = 'relu')(encoded1)
            encoded3 = Dense(200, activation = 'relu')(encoded2)
            encoded4 = Dense(150, activation = 'relu')(encoded3)
            encoded5 = Dense(encoding_dim, activation = 'relu')(encoded4)
            
            # Decoder Layers
            decoded1 = Dense(150, activation = 'relu')(encoded5)
            decoded2 = Dense(200, activation = 'relu')(decoded1)
            decoded3 = Dense(250, activation = 'relu')(decoded2)
            decoded4 = Dense(300, activation = 'relu')(decoded3)
            
            # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
            autoencoder = Model(input = input_dim, output = decoded4)
            
            # CONFIGURE AND TRAIN THE AUTOENCODER
            autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
            autoencoder.fit(X_train, X_train, nb_epoch = 15, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
            
            # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
            encoder = Model(input = input_dim, output = encoded5)
            encoded_input = Input(shape = (encoding_dim, ))
            encoded_out = encoder.predict(X)
            
            return encoder, pd.DataFrame(encoded_out)

        vector_col_name_doc2vec_ent1 = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_doc2vec_ent1]= pd.DataFrame(X_train.doc2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['doc2vec_vector_entity_1'], inplace=True)

        vector_col_name_doc2vec_ent2 = ['doc2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_doc2vec_ent2]= pd.DataFrame(X_train.doc2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['doc2vec_vector_entity_2'], inplace=True)

        vector_col_name_word2vec_ent1 = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_word2vec_ent1]= pd.DataFrame(X_train.word2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['word2vec_vector_entity_1'], inplace=True)

        vector_col_name_word2vec_ent2 = ['word2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_word2vec_ent2]= pd.DataFrame(X_train.word2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['word2vec_vector_entity_2'], inplace=True)

        vector_col_name_rdf2vec_ent1 = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_rdf2vec_ent1]= pd.DataFrame(X_train.rdf2vec_vector_entity_1.tolist(), index=X_train.index)
        X_train.drop(columns=['rdf2vec_vector_entity_1'], inplace=True)

        vector_col_name_rdf2vec_ent2 = ['rdf2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_train[vector_col_name_rdf2vec_ent2]= pd.DataFrame(X_train.rdf2vec_vector_entity_2.tolist(), index=X_train.index)
        X_train.drop(columns=['rdf2vec_vector_entity_2'], inplace=True)


        vector_col_name_doc2vec_ent1 = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_doc2vec_ent1]= pd.DataFrame(X_test.doc2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['doc2vec_vector_entity_1'], inplace=True)

        vector_col_name_doc2vec_ent2 = ['doc2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_doc2vec_ent2]= pd.DataFrame(X_test.doc2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['doc2vec_vector_entity_2'], inplace=True)

        vector_col_name_word2vec_ent1 = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_word2vec_ent1]= pd.DataFrame(X_test.word2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['word2vec_vector_entity_1'], inplace=True)

        vector_col_name_word2vec_ent2 = ['word2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_word2vec_ent2]= pd.DataFrame(X_test.word2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['word2vec_vector_entity_2'], inplace=True)

        vector_col_name_rdf2vec_ent1 = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_rdf2vec_ent1]= pd.DataFrame(X_test.rdf2vec_vector_entity_1.tolist(), index=X_test.index)
        X_test.drop(columns=['rdf2vec_vector_entity_1'], inplace=True)

        vector_col_name_rdf2vec_ent2 = ['rdf2vec_ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        X_test[vector_col_name_rdf2vec_ent2]= pd.DataFrame(X_test.rdf2vec_vector_entity_2.tolist(), index=X_test.index)
        X_test.drop(columns=['rdf2vec_vector_entity_2'], inplace=True)

        ########################### autoencoder on DOC2Vec #############################################
        vector_col_name_doc2vec = ['doc2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_doc2vec.append('entity_id')
        df_doc2vec_ent1_train = X_train[vector_col_name_doc2vec_ent1 + ['entity_id_wiki_1']]
        df_doc2vec_ent1_train.columns = vector_col_name_doc2vec

        df_doc2vec_ent2_train = X_train[vector_col_name_doc2vec_ent2 + ['entity_id_wiki_2']]
        df_doc2vec_ent2_train.columns = vector_col_name_doc2vec

        df_doc2vec_entities_train = pd.concat([df_doc2vec_ent1_train, df_doc2vec_ent2_train])
        df_doc2vec_entities_train = df_doc2vec_entities_train.reset_index(drop=True)
        df_doc2vec_entities_train_vectors = df_doc2vec_entities_train.drop(columns=['entity_id'])

        print(df_doc2vec_entities_train.head())


        doc2vec_encoder, df_autoencoder_doc2vec_train = reduce_dimension(df_doc2vec_entities_train_vectors)
        df_autoencoder_doc2vec_train.columns = ['doc2vec_autoencoder_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_doc2vec_entities_train = df_doc2vec_entities_train[['entity_id']]
        df_autoencoder_doc2vec_train = pd.merge(df_autoencoder_doc2vec_train, df_doc2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_doc2vec_ent1_test = X_test[vector_col_name_doc2vec_ent1 + ['entity_id_wiki_1']]
        df_doc2vec_ent1_test.columns = vector_col_name_doc2vec
        df_doc2vec_ent2_test = X_test[vector_col_name_doc2vec_ent2+ ['entity_id_wiki_2']]
        df_doc2vec_ent2_test.columns = vector_col_name_doc2vec
        df_doc2vec_entities_test = pd.concat([df_doc2vec_ent1_test, df_doc2vec_ent2_test])
        df_doc2vec_entities_test = df_doc2vec_entities_test.reset_index(drop=True)
        df_doc2vec_entities_test_vectors = df_doc2vec_entities_test.drop(columns=['entity_id'])

        X_autoencoder_doc2vec_test = doc2vec_encoder.predict(df_doc2vec_entities_test_vectors)
        df_autoencoder_doc2vec_test = pd.DataFrame.from_records(X_autoencoder_doc2vec_test)
        df_autoencoder_doc2vec_test.columns = ['doc2vec_autoencoder_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_doc2vec_entities_test = df_doc2vec_entities_test[['entity_id']]

        print('After autoencoder Test: ', len(df_autoencoder_doc2vec_test))

        df_autoencoder_doc2vec_test = pd.merge(df_autoencoder_doc2vec_test, df_doc2vec_entities_test, how='inner', right_index=True, left_index=True)


        ########################### autoencoder on Word2Vec #############################################
        vector_col_name_word2vec = ['word2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_word2vec.append('entity_id')
        df_word2vec_ent1_train = X_train[vector_col_name_word2vec_ent1 + ['entity_id_wiki_1']]
        df_word2vec_ent1_train.columns = vector_col_name_word2vec

        df_word2vec_ent2_train = X_train[vector_col_name_word2vec_ent2 + ['entity_id_wiki_2']]
        df_word2vec_ent2_train.columns = vector_col_name_word2vec

        df_word2vec_entities_train = pd.concat([df_word2vec_ent1_train, df_word2vec_ent2_train])
        df_word2vec_entities_train = df_word2vec_entities_train.reset_index(drop=True)
        df_word2vec_entities_train_vectors = df_word2vec_entities_train.drop(columns=['entity_id'])

        word2vec_encoder, df_autoencoder_word2vec_train = reduce_dimension(df_doc2vec_entities_train_vectors)
        df_autoencoder_word2vec_train.columns = ['word2vec_autoencoder_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_word2vec_entities_train = df_word2vec_entities_train[['entity_id']]
        df_autoencoder_word2vec_train = pd.merge(df_autoencoder_word2vec_train, df_word2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_word2vec_ent1_test = X_test[vector_col_name_word2vec_ent1 + ['entity_id_wiki_1']]
        df_word2vec_ent1_test.columns = vector_col_name_word2vec
        df_word2vec_ent2_test = X_test[vector_col_name_word2vec_ent2+ ['entity_id_wiki_2']]
        df_word2vec_ent2_test.columns = vector_col_name_word2vec
        df_word2vec_entities_test = pd.concat([df_word2vec_ent1_test, df_word2vec_ent2_test])
        df_word2vec_entities_test = df_word2vec_entities_test.reset_index(drop=True)
        df_word2vec_entities_test_vectors = df_word2vec_entities_test.drop(columns=['entity_id'])

        X_autoencoder_word2vec_test = word2vec_encoder.predict(df_word2vec_entities_test_vectors)
        df_autoencoder_word2vec_test = pd.DataFrame.from_records(X_autoencoder_word2vec_test)
        df_autoencoder_word2vec_test.columns = ['word2vec_autoencoder_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_word2vec_entities_test = df_word2vec_entities_test[['entity_id']]

        print('After autoencoder Test: ', len(df_autoencoder_word2vec_test))

        df_autoencoder_word2vec_test = pd.merge(df_autoencoder_word2vec_test, df_word2vec_entities_test, how='inner', right_index=True, left_index=True)

        ########################### autoencoder on RDF2Vec #############################################
        vector_col_name_rdf2vec = ['rdf2vec_ent' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH)]
        vector_col_name_rdf2vec.append('entity_id')
        df_rdf2vec_ent1_train = X_train[vector_col_name_rdf2vec_ent1 + ['entity_id_wiki_1']]
        df_rdf2vec_ent1_train.columns = vector_col_name_rdf2vec

        df_rdf2vec_ent2_train = X_train[vector_col_name_rdf2vec_ent2 + ['entity_id_wiki_2']]
        df_rdf2vec_ent2_train.columns = vector_col_name_rdf2vec

        df_rdf2vec_entities_train = pd.concat([df_rdf2vec_ent1_train, df_rdf2vec_ent2_train])
        df_rdf2vec_entities_train = df_rdf2vec_entities_train.reset_index(drop=True)
        df_rdf2vec_entities_train_vectors = df_rdf2vec_entities_train.drop(columns=['entity_id'])

        print(df_rdf2vec_entities_train.head())

        rdf2vec_encoder, df_autoencoder_rdf2vec_train = reduce_dimension(df_doc2vec_entities_train_vectors)
        df_autoencoder_rdf2vec_train.columns = ['rdf2vec_autoencoder_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]

        df_rdf2vec_entities_train = df_rdf2vec_entities_train[['entity_id']]
        df_autoencoder_rdf2vec_train = pd.merge(df_autoencoder_rdf2vec_train, df_rdf2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_rdf2vec_ent1_test = X_test[vector_col_name_rdf2vec_ent1 + ['entity_id_wiki_1']]
        df_rdf2vec_ent1_test.columns = vector_col_name_rdf2vec
        df_rdf2vec_ent2_test = X_test[vector_col_name_rdf2vec_ent2+ ['entity_id_wiki_2']]
        df_rdf2vec_ent2_test.columns = vector_col_name_rdf2vec
        df_rdf2vec_entities_test = pd.concat([df_rdf2vec_ent1_test, df_rdf2vec_ent2_test])
        df_rdf2vec_entities_test = df_rdf2vec_entities_test.reset_index(drop=True)
        df_rdf2vec_entities_test_vectors = df_rdf2vec_entities_test.drop(columns=['entity_id'])

        X_autoencoder_rdf2vec_test = rdf2vec_encoder.predict(df_rdf2vec_entities_test_vectors)
        df_autoencoder_rdf2vec_test = pd.DataFrame.from_records(X_autoencoder_rdf2vec_test)
        df_autoencoder_rdf2vec_test.columns = ['rdf2vec_autoencoder_ent' + str(i) for i in range(0,REDUCED_DIMENSIONS)]
        df_rdf2vec_entities_test = df_rdf2vec_entities_test[['entity_id']]

        print('After autoencoder Test: ', len(df_autoencoder_rdf2vec_test))

        df_autoencoder_rdf2vec_test = pd.merge(df_autoencoder_rdf2vec_test, df_rdf2vec_entities_test, how='inner', right_index=True, left_index=True)


        X_train = X_train[['entity_id_wiki_1', 'entity_id_wiki_2']]
        X_train = pd.merge(X_train, df_autoencoder_doc2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_autoencoder_doc2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_train = pd.merge(X_train, df_autoencoder_word2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_autoencoder_word2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_train = pd.merge(X_train, df_autoencoder_rdf2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_autoencoder_rdf2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])


        print(len(X_train))

        X_test = X_test[['entity_id_wiki_1', 'entity_id_wiki_2']]
        X_test = pd.merge(X_test, df_autoencoder_doc2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_autoencoder_doc2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_test = pd.merge(X_test, df_autoencoder_word2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_autoencoder_word2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_test = pd.merge(X_test, df_autoencoder_rdf2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_autoencoder_rdf2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])



        df_train = df_train[['entity_id_wiki_1', 'entity_id_wiki_2','label']]
        df_test = df_test[['entity_id_wiki_1', 'entity_id_wiki_2']]

        X_train = pd.merge(X_train, df_train, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_train = X_train['label']
        y_train = y_train.astype('int')
        X_train = X_train.drop(['label'], axis=1)


        X_test = pd.merge(X_test, df_test, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])


        return X_train, y_train, X_test

    def apply_transformation(self, approach, X_train, y_train, X_test):
        if approach == 'CN':
            return self.transform_dataset_using_concat(X_train, y_train, X_test)
        elif approach == 'AVG':
            return self.transform_dataset_using_average(X_train, y_train, X_test)
        elif approach == 'PCA1':
            return self.transform_dataset_using_pca_network1(X_train, y_train, X_test)
        elif approach == 'SVD1':
            return self.transform_dataset_using_svd_network1(X_train, y_train, X_test)
        elif approach == 'AE1':
            return self.transform_dataset_using_autoencoder_network1(X_train, y_train, X_test)
        elif approach == 'PCA2':
            return self.transform_dataset_using_pca_network2(X_train, y_train, X_test)
        elif approach == 'SVD2':
            return self.transform_dataset_using_svd_network2(X_train, y_train, X_test)
        elif approach == 'AE2':
            return self.transform_dataset_using_autoencoder_network2(X_train, y_train, X_test)
        else:
            return self.transform_dataset_using_concat(X_train, y_train, X_test)

    def generate_ensemble_dataset(self):
        df_training_set = pd.read_pickle(BASE_DIR + '/' + DATA_DIR + '/' + ENSEMBLE_DATASET_PATH + '/'  + 'training_set' + '/'  + 'training_set.pkl')
        df_training_set = df_training_set.dropna()
        print(df_training_set.head())
        
        y_train = df_training_set['label'].astype('int')
        X_train = df_training_set.drop(['label'], axis=1)
        
        
        df_test_set = pd.read_pickle(BASE_DIR + '/' + DATA_DIR + '/' + ENSEMBLE_DATASET_PATH + '/'  + 'test_set' + '/'  + 'test_set.pkl')
        df_test_set = df_test_set.dropna()
        X_test = df_test_set #df_test_set.drop(['label'], axis=1)
        
        print(len(X_train),  ' ', len(y_train), ' ', len(X_test))
        
        X_train, y_train, X_test = self.apply_transformation(ENSEMBLE_APPROACH, X_train, y_train, X_test)

        return X_train, y_train, X_test
    
    def get_dataset(self):
        X_train, y_train, X_test = self.generate_ensemble_dataset()
        return X_train, y_train, X_test