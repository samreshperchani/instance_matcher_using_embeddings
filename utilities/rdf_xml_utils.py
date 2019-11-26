import os
import glob
import tarfile
import multiprocessing as mp
import shutil
import sys
from pathlib import Path
import time
import rdflib
import time
import pandas as pd
from annoy import AnnoyIndex

path = Path(os.path.abspath(__file__))

# set configuration file path
#config_path = os.path.dirname(os.getcwd()) + '/config' 
config_path = str(path.parent.parent) + '/config'  
doc2vec_path = str(path.parent.parent) + '/doc2vec'  
word2vec_path = str(path.parent.parent) + '/word2vec'  
rdf2vec_path = str(path.parent.parent) + '/rdf2vec'  

# add config file path to system
sys.path.append(config_path)
sys.path.append(doc2vec_path)
sys.path.append(word2vec_path)
sys.path.append(rdf2vec_path)

import config
from rdf2vec.rdf2vec import RDF2Vec
from doc2vec.doc2vec import DOC2Vec
from word2vec.word2vec import WORD2Vec

# base directory of the solution
BASE_DIR = config.BASE_DIR

# path to data directory
DATA_DIR = config.DATA_DIR

# path where processed dumps have to be placed
PROCESSED_DUMPS_DIR = config.PROCESSED_DUMPS_DIR

# path where xml dumps are present
XML_DUMPS_DIR = config.XML_DUMPS_DIR

# embedding vector length
EMBEDDING_VECTOR_LENGTH = config.EMBEDDING_VECTOR_LENGTH

class RDF_XML_UTILS:
    # the function is to remove extra characters from labels
    def pre_process_labels(self, label):
        pre_processed_label = label
        if label.find("\"^^") != -1:
            pre_processed_label = label[1: label.index("\"^^")].lower()
        elif len(label) > 0:
            pre_processed_label = label[1:len(label)-6].lower()
        return pre_processed_label


    def revise_labels(self, x):
        revised_label = x['label']

        if x['predicate'] != '<http://www.w3.org/2000/01/rdf-schema#label>':
            revised_label = revised_label[revised_label.rfind('/')+1:len(revised_label)]
        
        return revised_label
    
    # this method is to generate negative training examples using vectors of entities and positive examples
    def generate_negative_training_examples(self, df_positives_cases, df_wiki_1_vectors, df_wiki_2_vectors):

        # initialize data frame
        df_negative_examples = pd.DataFrame(columns=['entity_id_wiki_1', 'entity_id_wiki_2', 'vector_entity_1', 'vector_entity_2','label'])

        dict_wiki_1 = df_wiki_1_vectors['entity_id'].to_dict()
        index_map_wiki_1 = dict((v,k) for k,v in dict_wiki_1.items())

        dict_wiki_2 = df_wiki_2_vectors['entity_id'].to_dict()
        index_map_wiki_2 = dict((v,k) for k,v in dict_wiki_2.items())

        t_wiki_1 = None  # Length of item vector that will be indexed
        t_wiki_2 = None


        if len(df_wiki_1_vectors) > 0:
            print('building index for entities in wiki 1')
            t_wiki_1 = AnnoyIndex(EMBEDDING_VECTOR_LENGTH, 'angular')
            for index, row in df_wiki_1_vectors.iterrows():
                v = row['vector']
                t_wiki_1.add_item(index, v)
            t_wiki_1.build(20) # 10 trees
            print('building index for entities in wiki 1 done')
        
        if len(df_wiki_2_vectors) > 0:
            print('building index for entities in wiki 2')
            t_wiki_2 = AnnoyIndex(EMBEDDING_VECTOR_LENGTH, 'angular')    
            for index, row in df_wiki_2_vectors.iterrows():
                v = row['vector']
                t_wiki_2.add_item(index, v)
            t_wiki_2.build(20) # 10 trees
            print('building index for entities in wiki 2 done')
            
        print('total positive cases:', len(df_positives_cases))
        df_positives_cases_slice = df_positives_cases.head(1000)
        for index, row in df_positives_cases_slice.iterrows():
            #print(df_wiki_1_vectors.head())
            #print(row['entity_id_wiki_1'])

            
            entity_1 = row['entity_id_wiki_1']
            entity_2 = row['entity_id_wiki_2']

            if entity_1 in index_map_wiki_1:
                filtered_entity_1 = df_wiki_1_vectors.loc[index_map_wiki_1[entity_1],:]
                

                #print(filtered_entity_1)
                if len(filtered_entity_1) > 0:
                    vector = filtered_entity_1['vector']

                    nn_ent1 = t_wiki_2.get_nns_by_vector(vector, 5, include_distances=False)
                    for i in range(0,len(nn_ent1)):
                        df_negative_examples.loc[len(df_negative_examples)] = [entity_1] + [dict_wiki_2[i]] + [vector] + [t_wiki_2.get_item_vector(i)] +[0]
                        #df_negative_examples = df_negative_examples.append({'entity_id_wiki_1': entity_1, 'entity_id_wiki_2': dict_wiki_2[i], 'vector_entity_1': vector ,'vector_entity_2': t_wiki_2.get_item_vector(i), 'label':0}, ignore_index=True)
            
            
            if entity_2 in index_map_wiki_2:
                filtered_entity_2 = df_wiki_2_vectors.loc[index_map_wiki_2[entity_2],:]
                if len(filtered_entity_2) > 0:
                    vector = filtered_entity_2['vector']
                    nn_ent2 = t_wiki_1.get_nns_by_vector(vector, 5, include_distances=False)

                    for i in range(0,len(nn_ent2)):
                        df_negative_examples = df_negative_examples.append({'entity_id_wiki_1': dict_wiki_1[i], 'entity_id_wiki_2': entity_2, 'vector_entity_1': t_wiki_1.get_item_vector(i) ,'vector_entity_2': vector, 'label':0}, ignore_index=True)
        return df_negative_examples



    # function to generate training and test set
    def generate_train_test(self, wiki_1, wiki_2):
        wiki_1_label_file = [file for file in glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + wiki_1 + '/*-instance-labels.ttl') if "category-labels" not in file]
        wiki_2_label_file = [file for file in glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + wiki_2 + '/*-instance-labels.ttl') if "category-labels" not in file]
        

        df_wiki_1 = pd.DataFrame(columns = ['entity_id','predicate','label','wiki_name'])
        df_wiki_2 = pd.DataFrame(columns = ['entity_id','predicate','label','wiki_name'])

        if len(wiki_1_label_file) > 0:
            my_cols = ["details"]
            df_wiki_1 = pd.read_table(wiki_1_label_file[0], names=my_cols)
            df_wiki_1 = df_wiki_1.iloc[1:len(df_wiki_1)-1]
            
            if len(df_wiki_1) > 0:
                new_cols = df_wiki_1["details"].str.split(" ", n = 2, expand=True)
                df_wiki_1['entity_id'] = new_cols[0].str.lower()
                df_wiki_1['predicate'] = new_cols[1].str.lower()
                df_wiki_1['label'] = new_cols[2].apply(self.pre_process_labels)
                #df_wiki_1['label'] = df_wiki_1.apply(revise_labels, axis=1)
                df_wiki_1['wiki_name'] = wiki_1
                df_wiki_1 = df_wiki_1[df_wiki_1['predicate']=='<http://www.w3.org/2000/01/rdf-schema#label>']

        else:
            print('No labels present for wiki: ',  wiki_1)

        print(wiki_2_label_file)
        if len(wiki_2_label_file) > 0:
            my_cols = ["details"]
            df_wiki_2 = pd.read_table(wiki_2_label_file[0], names=my_cols)
            df_wiki_2 = df_wiki_2.iloc[1:len(df_wiki_2)-1]
            
            if len(df_wiki_2) > 0:
                new_cols = df_wiki_2["details"].str.split(" ", n = 2, expand=True)
                df_wiki_2['entity_id'] = new_cols[0].str.lower()
                df_wiki_2['predicate'] = new_cols[1].str.lower()
                df_wiki_2['label'] = new_cols[2].apply(self.pre_process_labels)
                #df_wiki_2['label'] = df_wiki_2.apply(revise_labels, axis=1)
                df_wiki_2['wiki_name'] = wiki_2
                df_wiki_2 = df_wiki_2[df_wiki_2['predicate']=='<http://www.w3.org/2000/01/rdf-schema#label>']

        else:
            print('No labels present for wiki: ',  wiki_2)


        df_wiki_1.drop_duplicates(subset=['entity_id'], inplace = True)
        df_wiki_1= df_wiki_1.reset_index(drop=True)

        df_wiki_2.drop_duplicates(subset=['entity_id'], inplace = True)
        df_wiki_2 = df_wiki_2.reset_index(drop=True)

        print('Instances in Wiki 1:', len(df_wiki_1))
        print('Instances in Wiki 2:', len(df_wiki_2))


        print('*******************Priniting Wiki 2**************')
        print(df_wiki_2.head())


        #initialize rdf2vec_model
        rdf2vec_model = RDF2Vec()


        print('Extracting RDF2Vec Vectors')
        # get rdf2vec vectors
        rdf2vec_wiki_1, rdf2vec_wiki_2 = rdf2vec_model.extract_vectors(df_wiki_1,df_wiki_2)



        #initialize word2vec_model
        word2vec_model = WORD2Vec()
        print('Extracting Word2Vec Vectors')
        word2vec_wiki_1, word2vec_wiki_2 = word2vec_model.extract_vectors(df_wiki_1,df_wiki_2)



        #initialize doc2vec_model
        doc2vec_model = DOC2Vec()
        print('Extracting DOC2Vec Vectors')
        doc2vec_wiki_1, doc2vec_wiki_2 = doc2vec_model.extract_vectors(df_wiki_1,df_wiki_2)


        # write vectors to file
        '''
        rdf2vec_wiki_1.to_pickle('rdf2vec_wiki1.pkl')
        rdf2vec_wiki_2.to_pickle('rdf2vec_wiki2.pkl')

        word2vec_wiki_1.to_pickle('word2vec_wiki1.pkl')
        word2vec_wiki_2.to_pickle('word2vec_wiki2.pkl')

        doc2vec_wiki_1.to_pickle('doc2vec_wiki1.pkl')
        doc2vec_wiki_2.to_pickle('doc2vec_wiki2.pkl')
        
        rdf2vec_wiki_1  = pd.read_pickle('rdf2vec_wiki1.pkl')
        rdf2vec_wiki_2 = pd.read_pickle('rdf2vec_wiki2.pkl')

        word2vec_wiki_1  = pd.read_pickle('word2vec_wiki1.pkl')
        word2vec_wiki_2 = pd.read_pickle('word2vec_wiki2.pkl')

        doc2vec_wiki_1  = pd.read_pickle('doc2vec_wiki1.pkl')
        doc2vec_wiki_2  = pd.read_pickle('doc2vec_wiki2.pkl')
        '''

        df_wiki_1 = df_wiki_1[['entity_id','wiki_name','label']]
        df_wiki_2 = df_wiki_2[['entity_id','wiki_name','label']]

        df_wiki_1.rename(columns ={'entity_id':'entity_id_wiki_1', 'wiki_name':'wiki_1'}, inplace=True)
        df_wiki_2.rename(columns ={'entity_id':'entity_id_wiki_2', 'wiki_name':'wiki_2'}, inplace=True)


        df_common_labels = pd.merge(df_wiki_1, df_wiki_2, how='inner', on='label')

        print('*********Common labels**********')
        print(df_common_labels.head())

        rdf2vec_negative_examples = self.generate_negative_training_examples(df_common_labels, rdf2vec_wiki_1, rdf2vec_wiki_2)
        doc2vec_negative_examples = self.generate_negative_training_examples(df_common_labels, doc2vec_wiki_1, doc2vec_wiki_2)
        word2vec_negative_examples = self.generate_negative_training_examples(df_common_labels, word2vec_wiki_1, word2vec_wiki_2)


        print('***********Printing Negative Examples*****************')
        print(rdf2vec_negative_examples.head())
        print(doc2vec_negative_examples.head())
        print(word2vec_negative_examples.head())


        ######################### Training Set Positives Examples ################################################

        ################## RDF2Vec vectors for positives examples ##############################
        rdf2vec_wiki_1 = rdf2vec_wiki_1[['entity_id','vector']]
        rdf2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'rdf2vec_vector_entity_1'}, inplace=True)

        rdf2vec_wiki_2 = rdf2vec_wiki_2[['entity_id','vector']]
        rdf2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'rdf2vec_vector_entity_2'}, inplace=True)

        df_positive_examples = pd.merge(df_common_labels, rdf2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_positive_examples = pd.merge(df_positive_examples, rdf2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 



        ################## DOC2Vec vectors for positives examples ##############################
        doc2vec_wiki_1 = doc2vec_wiki_1[['entity_id','vector']]
        doc2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'doc2vec_vector_entity_1'}, inplace=True)

        doc2vec_wiki_2 = doc2vec_wiki_2[['entity_id','vector']]
        doc2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'doc2vec_vector_entity_2'}, inplace=True)

        df_positive_examples = pd.merge(df_positive_examples, doc2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_positive_examples = pd.merge(df_positive_examples, doc2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 


        ################## word2Vec vectors for positives examples ##############################
        word2vec_wiki_1 = word2vec_wiki_1[['entity_id','vector']]
        word2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'word2vec_vector_entity_1'}, inplace=True)

        word2vec_wiki_2 = word2vec_wiki_2[['entity_id','vector']]
        word2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'word2vec_vector_entity_2'}, inplace=True)

        df_positive_examples = pd.merge(df_positive_examples, word2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_positive_examples = pd.merge(df_positive_examples, word2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 

        df_positive_examples['label'] = 1


        df_positive_examples = df_positive_examples[['entity_id_wiki_1', 'entity_id_wiki_2', 'rdf2vec_vector_entity_1', 'rdf2vec_vector_entity_2','doc2vec_vector_entity_1', 'doc2vec_vector_entity_2','word2vec_vector_entity_1','word2vec_vector_entity_2', 'label']]

        df_positive_examples['doc2vec_vector_entity_1'].fillna(df_positive_examples['rdf2vec_vector_entity_1'], inplace=True)
        df_positive_examples['doc2vec_vector_entity_2'].fillna(df_positive_examples['rdf2vec_vector_entity_2'], inplace=True)
        df_positive_examples['rdf2vec_vector_entity_1'].fillna(df_positive_examples['doc2vec_vector_entity_1'], inplace=True)
        df_positive_examples['rdf2vec_vector_entity_2'].fillna(df_positive_examples['doc2vec_vector_entity_2'], inplace=True)
        df_positive_examples['word2vec_vector_entity_1'].fillna(df_positive_examples['doc2vec_vector_entity_1'], inplace=True)
        df_positive_examples['word2vec_vector_entity_2'].fillna(df_positive_examples['doc2vec_vector_entity_2'], inplace=True)



        print('******************* Prininting postive examples (training set) *******************')
        print(df_positive_examples.head())




        ######################### Training Set Negative Examples ################################################
        rdf2vec_negative_entities = rdf2vec_negative_examples[['entity_id_wiki_1','entity_id_wiki_2']]
        doc2vec_negative_entities = doc2vec_negative_examples[['entity_id_wiki_1','entity_id_wiki_2']] 
        word2vec_negative_entities = word2vec_negative_examples[['entity_id_wiki_1','entity_id_wiki_2']] 

        df_negative_entities = pd.concat([rdf2vec_negative_entities, doc2vec_negative_entities, word2vec_negative_entities], ignore_index=True)


        df_negative_entities = pd.merge(df_negative_entities, rdf2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_negative_entities = pd.merge(df_negative_entities, rdf2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 

        df_negative_entities = pd.merge(df_negative_entities, doc2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_negative_entities = pd.merge(df_negative_entities, doc2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 

        df_negative_entities = pd.merge(df_negative_entities, word2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_negative_entities = pd.merge(df_negative_entities, word2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 
        
        # set label for negative examples
        df_negative_entities['label'] = 0

        df_negative_entities = df_negative_entities[['entity_id_wiki_1', 'entity_id_wiki_2', 'rdf2vec_vector_entity_1', 'rdf2vec_vector_entity_2','doc2vec_vector_entity_1', 'doc2vec_vector_entity_2','word2vec_vector_entity_1','word2vec_vector_entity_2','label']]

        df_negative_entities['doc2vec_vector_entity_1'].fillna(df_negative_entities['rdf2vec_vector_entity_1'], inplace=True)
        df_negative_entities['doc2vec_vector_entity_2'].fillna(df_negative_entities['rdf2vec_vector_entity_2'], inplace=True)
        df_negative_entities['rdf2vec_vector_entity_1'].fillna(df_negative_entities['doc2vec_vector_entity_1'], inplace=True)
        df_negative_entities['rdf2vec_vector_entity_2'].fillna(df_negative_entities['doc2vec_vector_entity_2'], inplace=True)
        df_negative_entities['word2vec_vector_entity_1'].fillna(df_negative_entities['doc2vec_vector_entity_1'], inplace=True)
        df_negative_entities['word2vec_vector_entity_2'].fillna(df_negative_entities['doc2vec_vector_entity_2'], inplace=True)




        df_training_set = pd.concat([df_positive_examples, df_negative_entities], ignore_index=True)
        df_training_set = df_training_set.drop_duplicates(['entity_id_wiki_1', 'entity_id_wiki_2'])
        df_training_set = df_training_set.reset_index(drop=True)

        print(df_training_set.head())
        #df_training_set.to_pickle('training_set.pkl')


        df_training_set = pd.read_pickle('training_set.pkl')

        df_positive_training_set = df_training_set[df_training_set['label']==1]

        print(df_training_set.head())
        ############################## Generation of Test set ###############################

        entities_training_wiki_1 = df_positive_training_set['entity_id_wiki_1'].tolist()
        entities_training_wiki_2 = df_positive_training_set['entity_id_wiki_2'].tolist()


        print('***********************test set********************')
        print(df_positive_training_set.head())

        print('***********************removing positive examples entities********************')
        df_wiki_test_set_1 = df_wiki_1[~(df_wiki_1['entity_id_wiki_1'].isin(entities_training_wiki_1))]
        df_wiki_test_set_2 = df_wiki_2[~(df_wiki_2['entity_id_wiki_2'].isin(entities_training_wiki_2))]

        df_wiki_test_set_1 = df_wiki_test_set_1[['entity_id_wiki_1']]
        df_wiki_test_set_2 = df_wiki_test_set_2[['entity_id_wiki_2']]

        # setting to 100 because of cost of cartersion product
        df_wiki_test_set_1 = df_wiki_test_set_1.head(100)
        df_wiki_test_set_2 = df_wiki_test_set_2.head(100)

        df_wiki_test_set_1.index = [1 for i in range(0,len(df_wiki_test_set_1))]
        df_wiki_test_set_2.index = [1 for i in range(0,len(df_wiki_test_set_2))]

        df_wiki_test_set = pd.merge(df_wiki_test_set_1, df_wiki_test_set_2, left_index=True, right_index=True)

        print(df_wiki_test_set.head())


        rdf2vec_wiki_1 = rdf2vec_wiki_1[['entity_id_wiki_1','rdf2vec_vector_entity_1']]
        #rdf2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'rdf2vec_vector_entity_1'}, inplace=True)

        rdf2vec_wiki_2 = rdf2vec_wiki_2[['entity_id_wiki_2','rdf2vec_vector_entity_2']]
        #rdf2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'rdf2vec_vector_entity_2'}, inplace=True)


        doc2vec_wiki_1 = doc2vec_wiki_1[['entity_id_wiki_1','doc2vec_vector_entity_1']]
        #doc2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'doc2vec_vector_entity_1'}, inplace=True)

        doc2vec_wiki_2 = doc2vec_wiki_2[['entity_id_wiki_2','doc2vec_vector_entity_2']]
        #doc2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'doc2vec_vector_entity_2'}, inplace=True)


        word2vec_wiki_1 = word2vec_wiki_1[['entity_id_wiki_1','word2vec_vector_entity_1']]
        #word2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'word2vec_vector_entity_1'}, inplace=True)

        word2vec_wiki_2 = word2vec_wiki_2[['entity_id_wiki_2','word2vec_vector_entity_2']]
        #word2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'word2vec_vector_entity_2'}, inplace=True)





        df_wiki_test_set = pd.merge(df_wiki_test_set, rdf2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_wiki_test_set = pd.merge(df_wiki_test_set, rdf2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 

        df_wiki_test_set = pd.merge(df_wiki_test_set, doc2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_wiki_test_set = pd.merge(df_wiki_test_set, doc2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 

        df_wiki_test_set = pd.merge(df_wiki_test_set, word2vec_wiki_1, how='left', left_on=['entity_id_wiki_1'], right_on=['entity_id_wiki_1'])
        df_wiki_test_set = pd.merge(df_wiki_test_set, word2vec_wiki_2, how='left', left_on=['entity_id_wiki_2'], right_on=['entity_id_wiki_2']) 


        df_wiki_test_set['doc2vec_vector_entity_1'].fillna(df_wiki_test_set['rdf2vec_vector_entity_1'], inplace=True)
        df_wiki_test_set['doc2vec_vector_entity_2'].fillna(df_wiki_test_set['rdf2vec_vector_entity_2'], inplace=True)
        df_wiki_test_set['rdf2vec_vector_entity_1'].fillna(df_wiki_test_set['doc2vec_vector_entity_1'], inplace=True)
        df_wiki_test_set['rdf2vec_vector_entity_2'].fillna(df_wiki_test_set['doc2vec_vector_entity_2'], inplace=True)
        df_wiki_test_set['word2vec_vector_entity_1'].fillna(df_wiki_test_set['doc2vec_vector_entity_1'], inplace=True)
        df_wiki_test_set['word2vec_vector_entity_2'].fillna(df_wiki_test_set['doc2vec_vector_entity_2'], inplace=True)

        print(df_wiki_test_set_1.head())
        print(df_wiki_test_set_2.head())
        print(df_wiki_test_set.head())
        print(df_wiki_test_set.columns)

        df_wiki_test_set.to_pickle('test_set.pkl')    

        return df_training_set, df_wiki_test_set  



    # function to convert RDF in XML format to ttl
    def convert_xml_rdf_ttl(self):
        if not os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR):
            os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' + 'processed_dumps')
        # loop through all xmls and convert to ttl one by one
        for dump_file in os.listdir(BASE_DIR + '/' + DATA_DIR + '/' + XML_DUMPS_DIR):
            print('converting', dump_file, 'rdf-xml to ttl')
            dump_folder = dump_file.replace('.xml','')

            if os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' +  dump_folder):
                shutil.rmtree(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' +  dump_folder)
                time.sleep(2)
            
            os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' +  dump_folder)

            catgeory_labels_file = open(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' +  dump_folder + '/' + 'wiki-category-labels.ttl', 'w+', encoding='utf-8')
            instance_labels_file = open(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' +  dump_folder + '/' + 'wiki-instance-labels.ttl', 'w+', encoding='utf-8')
            property_labels_file = open(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' +  dump_folder + '/' + 'wiki-property-labels.ttl', 'w+', encoding='utf-8')
            class_labels_file = open(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' +  dump_folder + '/' + 'wiki-class-labels.ttl', 'w+', encoding='utf-8')
            abstract_file = open(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' +  dump_folder + '/' + 'wiki-long-abstracts.ttl', 'w+', encoding='utf-8')
            
            g=rdflib.Graph()
            g.load(BASE_DIR + '/' + DATA_DIR + '/' + XML_DUMPS_DIR + '/' + dump_file)

            # loop through triples and check if it contain information about label, if it does then write to different file based on resource type(i.e. class, property, instance, category)
            for s,p,o in g:
                if p.lower().strip() == 'http://www.w3.org/2000/01/rdf-schema#label':
                    if s.lower().find('/resource/category:') != -1:
                        catgeory_labels_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
                        g.remove( (s, p, o) )
                    elif s.lower().find('/resource/') != -1:
                        instance_labels_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
                        g.remove( (s, p, o) )
                    elif s.lower().find('/property/') != -1:
                        property_labels_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
                        g.remove( (s, p, o) )
                    elif s.lower().find('/class/') != -1:
                        class_labels_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
                        g.remove( (s, p, o) )
                elif p.lower().strip() == 'http://dbkwik.webdatacommons.org/ontology/abstract':
                    abstract_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.replace('\n','').strip() +  '\" .' + '\n')
                    g.remove( (s, p, o) )
            g.serialize(destination= BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + dump_folder + '/' + dump_file.replace('.xml','.ttl'), format='nt')


            catgeory_labels_file.close()
            instance_labels_file.close()
            property_labels_file.close()
            class_labels_file.close()
            abstract_file.close()