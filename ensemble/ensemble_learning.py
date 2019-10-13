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

logger = logging.getLogger(__name__)

# set configuration file path
config_path = os.path.dirname(os.getcwd()) + '/config' 

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


class ENSEMBLE_LEARNING:
    def transform_dataset_using_concat(self, X_train, y_train, X_test):

        def get_concatenated_vectors(df):
            df_vec_en_1 = df[['doc2vec_vector_ent1','word2vec_vector_ent1','rdf2vec_vector_ent1']]
            vector_col_name = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_1[vector_col_name]= pd.DataFrame(df.doc2vec_vector_ent1.tolist(), index=df.index)
            df_vec_en_1.drop(columns=['doc2vec_vector_ent1'], inplace=True)
        
            vector_col_name = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_1[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_ent1.tolist(), index=df.index)
            df_vec_en_1.drop(columns=['rdf2vec_vector_ent1'], inplace=True)
            
            vector_col_name = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df_vec_en_1[vector_col_name]= pd.DataFrame(df.word2vec_vector_ent1.tolist(), index=df.index)
            df_vec_en_1.drop(columns=['word2vec_vector_ent1'], inplace=True)
            
            df_vec_en_2 = X_train[['doc2vec_vector_ent2','word2vec_vector_ent2','rdf2vec_vector_ent2']]
            vector_col_name = ['doc2vec_ent2_' + str(i) for i in range(0,vector_length) ]
            df_vec_en_2[vector_col_name]= pd.DataFrame(df.doc2vec_vector_ent2.tolist(), index=df.index)
            df_vec_en_2.drop(columns=['doc2vec_vector_ent2'], inplace=True)
            
            vector_col_name = ['rdf2vec_ent2_' + str(i) for i in range(0,vector_length) ]
            df_vec_en_2[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_ent2.tolist(), index=df.index)
            df_vec_en_2.drop(columns=['rdf2vec_vector_ent2'], inplace=True)
            
            vector_col_name = ['word2vec_ent2_' + str(i) for i in range(0,vector_length) ]
            df_vec_en_2[vector_col_name]= pd.DataFrame(df.word2vec_vector_ent2.tolist(), index=df.index)
            df_vec_en_2.drop(columns=['word2vec_vector_ent2'], inplace=True)

            
            df = df[['entity_id_wiki_1','entity_id_wiki_2']]
            df = pd.merge(df, df_vec_en_1, left_index=True, right_index=True )
            df = pd.merge(df, df_vec_en_2, left_index=True, right_index=True )

            return df
        
        #### tranformation on the training set ################
        X_train = get_concatenated_vectors(X_train)
        
        #### tranformation on the test set ################
        X_test = get_concatenated_vectors(X_test)

        return X_train, y_train, X_test


    def transform_dataset_using_average(self, X_train, y_train, X_test):
        
        # functions to calculate calculate average vectors
        def avg_vector_ent1(x):
            vectors = [x['doc2vec_vector_ent1'], x['word2vec_vector_ent1'], x['rdf2vec_vector_ent1']]
            return  np.mean(vectors, axis = 0) 
        
        def avg_vector_ent2(x):
            vectors = [x['doc2vec_vector_ent2'], x['word2vec_vector_ent2'], x['rdf2vec_vector_ent2']]
            return  np.mean(vectors, axis = 0)

        def calculate_average_vectors(df):
            df['avg_vector_ent1'] = df.apply(avg_vector_ent1, axis=1)
            vector_col_name = ['ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df[vector_col_name]= pd.DataFrame(df.avg_vector_ent1.tolist(), index=df.index)
            df.drop(columns=['avg_vector_ent1'], inplace=True)
            df.drop(columns=['doc2vec_vector_ent1'], inplace=True)
            df.drop(columns=['word2vec_vector_ent1'], inplace=True)
            df.drop(columns=['rdf2vec_vector_ent1'], inplace=True)
            
            df['avg_vector_ent2'] = df.apply(avg_vector_ent2, axis=1)
            vector_col_name = ['ent2_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
            df[vector_col_name]= pd.DataFrame(df.avg_vector_ent2.tolist(), index=df.index)
            df.drop(columns=['avg_vector_ent2'], inplace=True)
            df.drop(columns=['doc2vec_vector_ent2'], inplace=True)
            df.drop(columns=['word2vec_vector_ent2'], inplace=True)
            df.drop(columns=['rdf2vec_vector_ent2'], inplace=True)

            return df

        ############ transformation on the training set ###################
        X_train = calculate_average_vectors(X_train)
        ############ transformation on the test set ###################    
        X_test = calculate_average_vectors(X_test)

        return X_train, y_train, X_test



    def apply_transformation(self, approach, X_train, y_train, X_test):
        if approach == 'CN':
            return transform_dataset_using_concat(X_train, y_train, X_test)
        elif approach == 'AVG':
            return transform_dataset_using_concat(X_train, y_train, X_test)
        elif approach == 'PCA1':
            return transform_dataset_using_concat(X_train, y_train, X_test)
        elif approach == 'SVD1':
            return transform_dataset_using_concat(X_train, y_train, X_test)
        elif approach == 'AE1':
            return transform_dataset_using_concat(X_train, y_train, X_test)
        elif approach == 'PCA2':
            return ransform_dataset_using_concat(X_train, y_train, X_test)
        elif approach == 'SVD2':
            return transform_dataset_using_concat(X_train, y_train, X_test)
        elif approach == 'AE2':
            return transform_dataset_using_concat(X_train, y_train, X_test)
        else:
            return transform_dataset_using_concat(X_train, y_train, X_test)

    def generate_ensemble_dataset(self):
        df_training_set = pd.read_pickle(BASE_DIR + '/' + DATA_DIR + '/' + ENSEMBLE_DATASET_PATH + '/'  + 'training_set' + '/'  + 'training_set.pkl')
        X_train = df_training_set.drop(['label'], axis=1)
        y_train = df_training_set.astype('int')
        
        df_test_set = pd.read_pickle(BASE_DIR + '/' + DATA_DIR + '/' + ENSEMBLE_DATASET_PATH + '/'  + 'test_set' + '/'  + 'test_set.pkl')
        X_test = df_test_set.drop(['label'], axis=1)

        X_train, y_train, X_test = self.apply_transformation(ENSEMBLE_APPROACH, X_train, y_train, X_test)

        return X_train, y_train, X_test
    
    def get_dataset(self):
        X_train, y_train, X_test = self.generate_ensemble_dataset()

        return X_train, y_train, X_test





 '''
        ##### transformation on training set ####################
        df_vec_en_1 = X_train[['doc2vec_vector_ent1','word2vec_vector_ent1','rdf2vec_vector_ent1']]
        vector_col_name = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        df_vec_en_1[vector_col_name]= pd.DataFrame(df.doc2vec_vector_ent1.tolist(), index=df.index)
        df_vec_en_1.drop(columns=['doc2vec_vector_ent1'], inplace=True)
        
        vector_col_name = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        df_vec_en_1[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_ent1.tolist(), index=df.index)
        df_vec_en_1.drop(columns=['rdf2vec_vector_ent1'], inplace=True)
        
        vector_col_name = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        df_vec_en_1[vector_col_name]= pd.DataFrame(df.word2vec_vector_ent1.tolist(), index=df.index)
        df_vec_en_1.drop(columns=['word2vec_vector_ent1'], inplace=True)
        
        df_vec_en_2 = X_train[['doc2vec_vector_ent2','word2vec_vector_ent2','rdf2vec_vector_ent2']]
        vector_col_name = ['doc2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df_vec_en_2[vector_col_name]= pd.DataFrame(df.doc2vec_vector_ent2.tolist(), index=df.index)
        df_vec_en_2.drop(columns=['doc2vec_vector_ent2'], inplace=True)
        
        vector_col_name = ['rdf2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df_vec_en_2[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_ent2.tolist(), index=df.index)
        df_vec_en_2.drop(columns=['rdf2vec_vector_ent2'], inplace=True)
        
        vector_col_name = ['word2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df_vec_en_2[vector_col_name]= pd.DataFrame(df.word2vec_vector_ent2.tolist(), index=df.index)
        df_vec_en_2.drop(columns=['word2vec_vector_ent2'], inplace=True)

        
        X_train = X_train[['entity_id_wiki_1','entity_id_wiki_2']]
        X_train = pd.merge(X_train, df_vec_en_1, left_index=True, right_index=True )
        X_train = pd.merge(X_train, df_vec_en_2, left_index=True, right_index=True )

        #################### transformation on test set ####################################
        df_vec_en_1 = X_test[['doc2vec_vector_ent1','word2vec_vector_ent1','rdf2vec_vector_ent1']]
        vector_col_name = ['doc2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        df_vec_en_1[vector_col_name]= pd.DataFrame(df.doc2vec_vector_ent1.tolist(), index=df.index)
        df_vec_en_1.drop(columns=['doc2vec_vector_ent1'], inplace=True)
        
        vector_col_name = ['rdf2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        df_vec_en_1[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_ent1.tolist(), index=df.index)
        df_vec_en_1.drop(columns=['rdf2vec_vector_ent1'], inplace=True)
        
        vector_col_name = ['word2vec_ent1_' + str(i) for i in range(0,EMBEDDING_VECTOR_LENGTH) ]
        df_vec_en_1[vector_col_name]= pd.DataFrame(df.word2vec_vector_ent1.tolist(), index=df.index)
        df_vec_en_1.drop(columns=['word2vec_vector_ent1'], inplace=True)
        
        df_vec_en_2 = X_test[['doc2vec_vector_ent2','word2vec_vector_ent2','rdf2vec_vector_ent2']]
        vector_col_name = ['doc2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df_vec_en_2[vector_col_name]= pd.DataFrame(df.doc2vec_vector_ent2.tolist(), index=df.index)
        df_vec_en_2.drop(columns=['doc2vec_vector_ent2'], inplace=True)
        
        vector_col_name = ['rdf2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df_vec_en_2[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_ent2.tolist(), index=df.index)
        df_vec_en_2.drop(columns=['rdf2vec_vector_ent2'], inplace=True)
        
        vector_col_name = ['word2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df_vec_en_2[vector_col_name]= pd.DataFrame(df.word2vec_vector_ent2.tolist(), index=df.index)
        df_vec_en_2.drop(columns=['word2vec_vector_ent2'], inplace=True)

        
        X_test = X_test[['entity_id_wiki_1','entity_id_wiki_2']]
        X_test = pd.merge(X_test, df_vec_en_1, left_index=True, right_index=True )
        X_test = pd.merge(X_test, df_vec_en_2, left_index=True, right_index=True )
''''

