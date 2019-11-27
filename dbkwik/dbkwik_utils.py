import glob
from pathlib import Path
import os
import sys
import pandas as pd
from lxml import etree

path = Path(os.path.abspath(__file__))

# set configuration file path
#config_path = os.path.dirname(os.getcwd()) + '/config' 
config_path = str(path.parent.parent) + '/config' 
config_path = str(path.parent.parent) + '/rdf2vec' 

# add config file path to system
sys.path.append(config_path)

import config
from rdf2vec.rdf2vec import RDF2Vec
from doc2vec.doc2vec import DOC2Vec
from word2vec.word2vec import WORD2Vec

# base directory of the solution
BASE_DIR = config.BASE_DIR

# directory where data is saved
DATA_DIR = config.DATA_DIR

# directory where processed dumps are saved
PROCESSED_DUMPS_DIR = config.PROCESSED_DUMPS_DIR

# directory where gold standard files are present
GS_DIR = config.GS_DIR

class DBKWIK_UTILS:

    # the function is to remove extra characters from labels
    def pre_process_labels(self, label):
        pre_processed_label = label
        if label.find("\"^^") != -1:
            pre_processed_label = label[1: label.index("\"^^")].lower()
        elif len(label) > 0:
            pre_processed_label = label[1:len(label)-6].lower()
        return pre_processed_label

    def get_test_set(self, wiki_1, wiki_2):
        print(BASE_DIR)

        wiki_1_label_file = [file for file in glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + wiki_1 + '/*-labels.ttl') if "category-labels" not in file]
        wiki_2_label_file = [file for file in glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + PROCESSED_DUMPS_DIR + '/' + wiki_2 + '/*-labels.ttl') if "category-labels" not in file]
        

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

        # use only important columns
        rdf2vec_wiki_1 = rdf2vec_wiki_1[['entity_id','vector']]
        doc2vec_wiki_1 = doc2vec_wiki_1[['entity_id','vector']]
        word2vec_wiki_1 = word2vec_wiki_1[['entity_id','vector']]

        rdf2vec_wiki_2 = rdf2vec_wiki_2[['entity_id','vector']]
        doc2vec_wiki_2 = doc2vec_wiki_2[['entity_id','vector']]
        word2vec_wiki_2 = word2vec_wiki_2[['entity_id','vector']]

        df_wiki_1 = df_wiki_1[['entity_id']]
        df_wiki_2 = df_wiki_2[['entity_id']]

        # rename columns
        rdf2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'rdf2vec_vector_entity_1'}, inplace=True)
        doc2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'doc2vec_vector_entity_1'}, inplace=True)
        word2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'word2vec_vector_entity_1'}, inplace=True)

        rdf2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'rdf2vec_vector_entity_2'}, inplace=True)
        doc2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'doc2vec_vector_entity_2'}, inplace=True)
        word2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'word2vec_vector_entity_2'}, inplace=True)

        df_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1'}, inplace=True)
        df_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2'}, inplace=True)


        #create dataframe for entities in wiki 1
        df_wiki_1 = pd.merge(df_wiki_1, rdf2vec_wiki_1, how='left', on=['entity_id_wiki_1'])
        df_wiki_1 = pd.merge(df_wiki_1, doc2vec_wiki_1, how='left', on=['entity_id_wiki_1'])
        df_wiki_1 = pd.merge(df_wiki_1, word2vec_wiki_1, how='left', on=['entity_id_wiki_1'])


        #create dataframe for entities in wiki 2
        df_wiki_2 = pd.merge(df_wiki_2, rdf2vec_wiki_2, how='left', on=['entity_id_wiki_2'])
        df_wiki_2 = pd.merge(df_wiki_2, doc2vec_wiki_2, how='left', on=['entity_id_wiki_2'])
        df_wiki_2 = pd.merge(df_wiki_2, word2vec_wiki_2, how='left', on=['entity_id_wiki_2'])

        # fallback strategy
        df_wiki_1['rdf2vec_vector_entity_1'] = df_wiki_1['rdf2vec_vector_entity_1'].fillna(df_wiki_1['doc2vec_vector_entity_1'])
        df_wiki_1['doc2vec_vector_entity_1'] = df_wiki_1['doc2vec_vector_entity_1'].fillna(df_wiki_1['rdf2vec_vector_entity_1'])
        df_wiki_1['word2vec_vector_entity_1'] = df_wiki_1['word2vec_vector_entity_1'].fillna(df_wiki_1['doc2vec_vector_entity_1'])
        
        df_wiki_2['rdf2vec_vector_entity_2'] = df_wiki_2['rdf2vec_vector_entity_2'].fillna(df_wiki_2['doc2vec_vector_entity_2'])
        df_wiki_2['doc2vec_vector_entity_2'] = df_wiki_2['doc2vec_vector_entity_2'].fillna(df_wiki_2['rdf2vec_vector_entity_2'])
        df_wiki_2['word2vec_vector_entity_2'] = df_wiki_2['word2vec_vector_entity_2'].fillna(df_wiki_2['doc2vec_vector_entity_2'])

        df_wiki_1.index = [1 for i in range(0,len(df_wiki_1))]
        df_wiki_2.index = [1 for i in range(0,len(df_wiki_2))]

        df_wiki_test_set = pd.merge(df_wiki_1, df_wiki_2, left_index=True, right_index=True)

        df_wiki_test_set= df_wiki_test_set.reset_index(drop=True)
        
        print(df_wiki_test_set.head())

        return df_wiki_test_set

    
    def get_gs_entities(self):
        gs_standard_directory = BASE_DIR + '/' + DATA_DIR + '/' + GS_DIR + '/'
        gs_entities = []
        for file in os.listdir(gs_standard_directory):
            
            count_entities = 0
            if file.endswith('.xml'):
                print(file)
                
                # parse xml tree from gold standard file
                tree = etree.parse(gs_standard_directory + '/' + file)
                
                # initialize dictionary with namespaces
                ns = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'n':'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'}
                
                # get nodes with mapping
                gs_mappings = tree.xpath("n:Alignment/n:map", namespaces=ns)
                
                # initialize dataframe 
                df_gs = pd.DataFrame(columns=['entity_id_wiki_1', 'entity_id_wiki_2','label'])
                
                # loop through mapping and insert mapping in dataframe
                for mapping in gs_mappings:
                    entity1_node = mapping.xpath('n:Cell/n:entity1/@rdf:resource', namespaces=ns)
                    
                    entity2_node = mapping.xpath('n:Cell/n:entity2/@rdf:resource', namespaces=ns)

                    label = mapping.xpath('n:Cell/n:relation/text()', namespaces=ns)

                    if label[0] == '=':
                        label = 1
                    else:
                        label = 0

                    #print(label)
                    
                    if '/resource/' in entity1_node[0].lower() or '/resource/' in entity2_node[0].lower():
                        count_entities = count_entities + 1
                        
                        # Pass the row elements as key value pairs to append() function 
                        wiki1 = entity1_node[0].replace('http://dbkwik.webdatacommons.org/','')
                        wiki1 = wiki1[0:wiki1.find('/resource/')]
                        #df_gs = df_gs.append({'entity_id_wiki_1' : entity1_node[0].lower() , 'wiki_name' : wiki1} , ignore_index=True)
                        
                        wiki2 = entity2_node[0].replace('http://dbkwik.webdatacommons.org/','')
                        wiki2 = wiki2[0:wiki2.find('/resource/')]
                        #df_gs = df_gs.append({'entity_id' : entity2_node[0].lower() , 'wiki_name' : wiki2} , ignore_index=True)
                        df_gs = df_gs.append({'entity_id_wiki_1' : '<' + str(entity1_node[0].lower()) + '>' , 'entity_id_wiki_2' : '<' + str(entity2_node[0].lower()) + '>', 'label': str(label)} , ignore_index=True)
                        
                #print(df_gs.head())
                print('Total Entities: ', count_entities)
                gs_entities.append(df_gs)


        df_gs_entities = pd.concat(gs_entities)

        print(len(df_gs_entities))

        df_gs_entities.drop_duplicates(inplace=True)

        print(df_gs_entities.head())

        return df_gs_entities
    

    def get_training_set(self):

        print('*********Extracting Gold Standard Entities ************')
        df_gs_entities = self.get_gs_entities()

        df_wiki_1 = pd.DataFrame(columns = ['entity_id','predicate','label','wiki_name'])
        df_wiki_2 = pd.DataFrame(columns = ['entity_id','predicate','label','wiki_name'])

        df_wiki_1['entity_id'] = df_gs_entities['entity_id_wiki_1']
        df_wiki_2['entity_id'] = df_gs_entities['entity_id_wiki_2']
        
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


        # use only important columns
        rdf2vec_wiki_1 = rdf2vec_wiki_1[['entity_id','vector']]
        doc2vec_wiki_1 = doc2vec_wiki_1[['entity_id','vector']]
        word2vec_wiki_1 = word2vec_wiki_1[['entity_id','vector']]

        rdf2vec_wiki_2 = rdf2vec_wiki_2[['entity_id','vector']]
        doc2vec_wiki_2 = doc2vec_wiki_2[['entity_id','vector']]
        word2vec_wiki_2 = word2vec_wiki_2[['entity_id','vector']]

        df_wiki_1 = df_wiki_1[['entity_id']]
        df_wiki_2 = df_wiki_2[['entity_id']]

        # rename columns
        rdf2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'rdf2vec_vector_entity_1'}, inplace=True)
        doc2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'doc2vec_vector_entity_1'}, inplace=True)
        word2vec_wiki_1.rename(columns={'entity_id':'entity_id_wiki_1', 'vector':'word2vec_vector_entity_1'}, inplace=True)

        rdf2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'rdf2vec_vector_entity_2'}, inplace=True)
        doc2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'doc2vec_vector_entity_2'}, inplace=True)
        word2vec_wiki_2.rename(columns={'entity_id':'entity_id_wiki_2', 'vector':'word2vec_vector_entity_2'}, inplace=True)
        
       

        #create dataframe for entities in wiki 1
        df_gs_entities = pd.merge(df_gs_entities, rdf2vec_wiki_1, how='left', on=['entity_id_wiki_1'])
        df_gs_entities = pd.merge(df_gs_entities, doc2vec_wiki_1, how='left', on=['entity_id_wiki_1'])
        df_gs_entities = pd.merge(df_gs_entities, word2vec_wiki_1, how='left', on=['entity_id_wiki_1'])


        #create dataframe for entities in wiki 2
        df_gs_entities = pd.merge(df_gs_entities, rdf2vec_wiki_2, how='left', on=['entity_id_wiki_2'])
        df_gs_entities = pd.merge(df_gs_entities, doc2vec_wiki_2, how='left', on=['entity_id_wiki_2'])
        df_gs_entities = pd.merge(df_gs_entities, word2vec_wiki_2, how='left', on=['entity_id_wiki_2'])

        df_gs_entities.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'], inplace=True)


        # fallback strategy
        df_gs_entities['rdf2vec_vector_entity_1'] = df_gs_entities['rdf2vec_vector_entity_1'].fillna(df_gs_entities['doc2vec_vector_entity_1'])
        df_gs_entities['doc2vec_vector_entity_1'] = df_gs_entities['doc2vec_vector_entity_1'].fillna(df_gs_entities['rdf2vec_vector_entity_1'])
        df_gs_entities['word2vec_vector_entity_1'] = df_gs_entities['word2vec_vector_entity_1'].fillna(df_gs_entities['doc2vec_vector_entity_1'])
        
        df_gs_entities['rdf2vec_vector_entity_2'] = df_gs_entities['rdf2vec_vector_entity_2'].fillna(df_gs_entities['doc2vec_vector_entity_2'])
        df_gs_entities['doc2vec_vector_entity_2'] = df_gs_entities['doc2vec_vector_entity_2'].fillna(df_gs_entities['rdf2vec_vector_entity_2'])
        df_gs_entities['word2vec_vector_entity_2'] = df_gs_entities['word2vec_vector_entity_2'].fillna(df_gs_entities['doc2vec_vector_entity_2'])

        print(df_gs_entities.head())

        return df_gs_entities


