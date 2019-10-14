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
import multiprocessing as mp
import pandas as pd
import random
import time
import psycopg2
import sql
from sqlalchemy import create_engine
import pyodbc
from gensim.models import Word2Vec

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

# path where extracted text will be stored
EXTRACTED_TEXT_DIR = config.EXTRACTED_TEXT_DIR

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# count of epochs
RDF2VEC_EPOCHS = config.RDF2VEC_EPOCHS

# length of embedding vector
EMBEDDING_VECTOR_LENGTH = config.EMBEDDING_VECTOR_LENGTH

# path where untared xml dumps are present
PROCESSED_DUMPS_DIR = config.PROCESSED_DUMPS_DIR

# path to store knowledge graphs
KG_DIR = config.KG_DIR

# path where labels mapping is present
LB_MAP_DIR = config.LB_MAP_DIR

# revised kg directory
REVISED_KG_DIR = config.REVISED_KG_DIR

# merged knowledge graph folder path
MERGED_KG_RW = config.MERGED_KG_RW

# final name for the final graph that will be used for random walks
MERGED_KG_RW_FILE_NAME = config.MERGED_KG_RW_FILE_NAME

# folder where random walks will be stored
WALKS_DIR = config.WALKS_DIR

# instance labels db dsn
INS_LABELS_DB = config.INS_LABELS_DB

# category labels db dsn
CAT_LABELS_DB = config.CAT_LABELS_DB

# property labels db dsn
PROP_LABELS_DB = config.PROP_LABELS_DB

# class labels db dsn
CLASS_LABELS_DB = config.CLASS_LABELS_DB

# instance labels count db
INS_DUPL_LABELS_DB = config.INS_DUPL_LABELS_DB

# categories labels count db
CAT_DUPL_LABELS_DB = config.CAT_DUPL_LABELS_DB

# properties labels count db
PROP_DUPL_LABELS_DB = config.PROP_DUPL_LABELS_DB

# classes labels count db
CLASS_DUPL_LABELS_DB = config.CLASS_DUPL_LABELS_DB

# class to traing word2vec
class RDF2Vec:
    
    def merge_all_ttl_files(self, wiki):
        print('Processing Wiki: ', wiki)
        kg_files = []
        CHUNKSIZE = 10000

        # get file ttl names of extracted dumps
        for file in os.listdir(wiki):
            if file.endswith(".ttl") and file != 'kg.ttl' and not (file.endswith('labels_map.ttl')) and ('nif' not in file) and not (file.endswith('abstracts.ttl')):
                #print(file)
                kg_files.append(file)

        # create knowledge graph
        with open(BASE_DIR + '/' + DATA_DIR +  '/' + KG_DIR + '/' + os.path.basename(wiki) + '_kg.ttl', 'w+', encoding='utf-8') as out_file:
            for fname in kg_files:
                with open(wiki + '/' + fname, 'r', encoding='utf-8') as ttl_file:
                    file_lines = ttl_file.readlines(CHUNKSIZE)
                    while file_lines:
                        for line in file_lines:
                            if not line.startswith('#'):
                                out_file.write(line.lower())
                
                        file_lines = ttl_file.readlines(CHUNKSIZE)
                    ttl_file.close()

    # merge all ttl files to generate knowledge graphs
    def generate_knowledge_graphs(self):
        wikis = glob.glob(BASE_DIR + '/' + DATA_DIR + '/'  + PROCESSED_DUMPS_DIR + '/*')
        if len(wikis) > 0:
            if not os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + KG_DIR):
                os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' + KG_DIR)
                
            processes = max(1, mp.cpu_count()-1)
                
            with mp.Pool(processes) as pool:
                result = pool.map_async(self.merge_all_ttl_files, wikis)
                pool.close()
                pool.join()
                
                
    def pre_process_object(self, o):
        if o is None:
            return ""
        elif o.endswith(" ."):
            return o[:len(o)-2]

    def update_duplicate_labels(self, wiki):

        print('Processing Wiki: ', wiki)
        use_mappings= False
        df_mappings = pd.DataFrame()
        with open(BASE_DIR + '/' + DATA_DIR + '/'+ REVISED_KG_DIR + '/' + os.path.basename(wiki), 'w+', encoding='utf-8') as out_file:
            print(os.path.basename(wiki).replace('_kg.ttl','_labels_map.ttl'))
            labels_file = os.path.basename(wiki).replace('_kg.ttl','_labels_map.ttl')
            if os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + LB_MAP_DIR + '/'  + labels_file):
                print('Using Labels File: ', labels_file)
                df_mappings = pd.read_csv(BASE_DIR + '/' +  DATA_DIR + '/' + LB_MAP_DIR + '/' + labels_file, sep=" ", encoding='utf-8', header=None, comment='#')
                df_mappings.columns = ['id', 'revised_id']
                df_mappings = df_mappings.drop_duplicates(subset='id', keep='first')
                use_mappings = True
            

            if use_mappings:
                for chunk in pd.read_csv(wiki,  header=None, delimiter='\n',skip_blank_lines=True, names=['details'], quotechar="\"", chunksize=100000 , iterator=True):
                    df_wiki = chunk
                    new_cols = df_wiki["details"].str.split(r'\s', n=2, expand = True)
                    df_wiki.fillna("")
                    df_wiki["s"] = new_cols[0]
                    df_wiki["p"] = new_cols[1]
                    df_wiki["o"] = new_cols[2]    
                    df_wiki["o"] = df_wiki["o"].apply(lambda x:self.pre_process_object(x))

                    df_wiki = df_wiki.drop("details", axis = 1)
                    merged = pd.merge(df_wiki, df_mappings, how='left', left_on='s', right_on='id')
                    merged['revised_id'] = merged['revised_id'].fillna(merged['s'])
                    merged = pd.merge(merged, df_mappings, how='left', left_on='o', right_on='id')
                    merged['revised_id_y'] = merged['revised_id_y'].fillna(merged['o'])
                    merged = merged.drop(['s','o','id_x', 'id_y'] , axis=1)
                    merged.rename(columns={'revised_id_x':'s',
                                'revised_id_y':'o'},inplace=True)
                    merged['triple'] = merged["s"] + " " + merged["p"] + " " + merged["o"] + " ."

                    merged = merged.drop(["s","p","o"], axis = 1)
                    
                    with open(BASE_DIR + '/' + DATA_DIR + '/' + REVISED_KG_DIR + '/' + os.path.basename(wiki), mode='a+', encoding='utf-8') as out_file:
                        out_file.write(merged['triple'].str.cat(sep='\n'))                      
                    out_file.close()
            else:
                print('No mapping found: ', wiki)
                copyfile(wiki, BASE_DIR + '/' + DATA_DIR + '/' + REVISED_KG_DIR + '/' + os.path.basename(wiki))
                
                
                
    # generate revised knowlege graphs with updated labels
    def generate_revised_knowledge_graphs(self):
        
        wikis = glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + KG_DIR + '/*.ttl')
        if len(wikis) > 0:
            if not os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + REVISED_KG_DIR):
                os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' + REVISED_KG_DIR)
            
            for wiki in wikis:
                self.update_duplicate_labels(wiki)


    # this method is used to generate merged knowledge graph after revision of labels
    def generate_merged_kg_graphs(self):
        
        if os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + MERGED_KG_RW):
            shutil.rmtree(BASE_DIR + '/' + DATA_DIR + '/' + MERGED_KG_RW)
        
        os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' + MERGED_KG_RW)
        
        FILES_LIST = glob.glob(BASE_DIR + '/' + DATA_DIR + '/' + REVISED_KG_DIR + '/*.ttl' )
        count = 0
        CHUNKSIZE = 10000
        for file in FILES_LIST:
            count = count + 1
            print('Processing:', os.path.basename(file), ' Count:', count)
            
            with open(file,'r', encoding='utf-8') as kg:
                with open(BASE_DIR + '/' + DATA_DIR + '/' + MERGED_KG_RW + '/' + MERGED_KG_RW_FILE_NAME, 'a+', encoding='utf-8') as mergedkg:
                    file_lines = kg.readlines(CHUNKSIZE)
                    
                    while file_lines:
                        mergedkg.write(''.join(file_lines))
                        file_lines = kg.readlines(CHUNKSIZE)
                mergedkg.close()
            kg.close()

    # method to write generated walks in a file 
    def write_walks_to_file(self, contents, wiki_name):
        with open(BASE_DIR + '/' + DATA_DIR + '/' + WALKS_DIR  + '/' + wiki_name + '_walk.txt', 'a+', encoding='utf-8') as out_file:
            for path in contents:
                out_file.write(path)
        out_file.close()

    # method to generate hashmap representation of knowledge graphs
    def generate_subject_predicate_object_dict(self, kg):
        sub_pred_obj_dict ={}

        chunksize = 1000000
        with open(kg, 'r', encoding='utf-8') as rdf_graph:
            #print('Reading Next Chunk ....')
            lines = rdf_graph.readlines(chunksize)
            while lines:

                for line in lines:
                    if line.endswith(" ."):
                        line = line[:len(line)-2]
                    #print(line)
                    parts = line.split(" ",2)
                    #if len(parts)<3:
                    #    print(line)
                    s = str(parts[0]).strip()
                    p = str(parts[1]).strip()
                    o = str(parts[2]).strip()
                    if o.endswith(' .'):
                        o = o[:len(o)-2]
                    if not o.lower().startswith('<http://dbkwik.webdatacommons.org/'):
                        continue

                    if s in sub_pred_obj_dict:
                        if p in sub_pred_obj_dict[s]:
                            sub_pred_obj_dict[s][p].append(o)
                        else:
                            sub_pred_obj_dict[s][p] = [o]
                            
                    else:
                        sub_pred_obj_dict[s] = {p:[o]}
                #print('Reading Next Chunk ....')
                lines = rdf_graph.readlines(chunksize)
        print('Building Subject Predicate Object Done ....')
        print('Going to sleep for 5 Secs')
        time.sleep(5)
        print('Back from sleep')
        return sub_pred_obj_dict

    # method to get predicate and object for a random walk
    def __get_next_pred_obj(self, sub_pred_obj_dict, current_resource):
        pred_obj_map = sub_pred_obj_dict.get(current_resource)
        if pred_obj_map is None:
            return None, None
        chosen_predicate = random.choice(list(pred_obj_map.keys()))
        objects = pred_obj_map[chosen_predicate]
        chosen_object = next(iter(random.sample(objects, 1)))
        return chosen_predicate, chosen_object


    def generate_random_walks(self, kg, wiki_name, number_of_walks_per_resource=10,
                            maximum_length_of_a_walk=50, only_unique_walks=True):
        all_lines= []
        sub_pred_obj_dict = self.generate_subject_predicate_object_dict(kg)
        subject_count = len(sub_pred_obj_dict.keys())
        node_sep= '~EON~'
        for i, resource in enumerate(sub_pred_obj_dict.keys()):
            walks_per_resource = []

            for k in range(number_of_walks_per_resource):
                current = resource
                one_walk = [(current + node_sep).strip()]
                for l in range(maximum_length_of_a_walk):
                    (chosen_predicate, chosen_object) = self.__get_next_pred_obj(sub_pred_obj_dict, current)
                    if chosen_predicate is None or chosen_object is None:
                        break
                    one_walk.append((chosen_predicate + node_sep).strip())
                    one_walk.append((chosen_object + node_sep).strip())
                    current = chosen_object
                walks_per_resource.append(tuple(one_walk))
            
            for walk in set(walks_per_resource): 
                walk_str = ''.join(walk)
                all_lines.append(walk_str.replace(' .',''))
            all_lines.append('\n')
            if len(all_lines) >= 10000000: 
                self.write_walks_to_file(all_lines, wiki_name)
                all_lines.clear()
            if i % 1000 == 0:
                logger.info("%d / %d", i, subject_count)
        
        # for reamining lines
        self.write_walks_to_file(all_lines, wiki_name)
        all_lines.clear()




    # this method generate random walks on final knowledge graph
    def generate_walks(self):
        
        if os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + WALKS_DIR):
            shutil.rmtree(BASE_DIR + '/' + DATA_DIR + '/' + WALKS_DIR, ignore_errors=False)
            time.sleep(5)
        
        os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' + WALKS_DIR)


        wikis_list = glob.glob(BASE_DIR + '/' + DATA_DIR  + '/' + MERGED_KG_RW + '/*.ttl')
        total_wikis = len(wikis_list)
        print('Total Wikis Found: ', total_wikis)
        count = 0
        for wiki in wikis_list:
            count += 1
            print('Processing ', wiki, ' ' , count, ' out of ', total_wikis)
            self.generate_random_walks(kg=wiki, wiki_name=os.path.basename(wiki))

    # train RDF2Vec model
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
                
                for file in glob.glob(self.dirname  + '/' + WALKS_DIR + '/*.txt'):
                    print('Processing ', file , ' iteration ', self.iter_count, 'Processed: ', self.proc_count)
                    with open(file,'r', encoding='utf-8') as text_contents:
                        file_lines = text_contents.readlines(self.chunksize)
                        while file_lines:
                            lines = []
                            
                            for line in file_lines:
                                lines.append(line.replace('\n','').replace('\r\n','').replace('\r','').strip())
                                yield " ".join(lines).split('~EON~') #text_contents.read().replace('\r\n','').replace('\n','').split()
                            file_lines = text_contents.readlines(self.chunksize)
                        self.proc_count = self.proc_count + 1
                    text_contents.close()

        # set the text iterator
        corpus_data = TextLine(BASE_DIR + '/' + DATA_DIR)

        # set model parameters
        model = Word2Vec(size=EMBEDDING_VECTOR_LENGTH, window=10, min_count=1, workers=2, sg=1)

        # build model vocabulary
        model.build_vocab(corpus_data)

        print('Training the Model: ........')

        # train model
        model.train(corpus_data, total_examples=model.corpus_count, epochs=RDF2VEC_EPOCHS)

        # remove directory if already present
        if os.path.exists('model'):
            shutil.rmtree('model')
            time.sleep(5)

        # create model directory to save model    
        os.mkdir('model')

        # save model
        model.save('model/rdf2vec.model')

    #################################### Database functions ###################################
    def pre_process_labels(self, label):
        pre_processed_label = label
        if label.find("\"^^") != -1:
            pre_processed_label = label[1: label.index("\"^^")].lower()
        elif len(label) > 0:
            pre_processed_label = label[1:len(label)-6].lower()
        
        return pre_processed_label


    def insert_labels_to_db(self, dsn_name, id_column_name, wiki_name, file_name, target_table, schema):
        connection_string = 'mssql+pyodbc://@%s' % dsn_name
        engine = create_engine(connection_string)
        my_cols = ["details"]
        df = pd.read_table(file_name, names=my_cols)
        
        df = df.iloc[1:len(df)-1]
        if len(df) > 0:
            new_cols = df["details"].str.split(" ", n = 2, expand=True)
            df[id_column_name] = new_cols[0].str.lower()
            df['predicate'] = new_cols[1].str.lower()
            df['label'] = new_cols[2].apply(self.pre_process_labels) #(lambda x: x[1: x.index("\"^^")].lower())#[1:len(x)-6].lower())
            df['wiki_name'] = wiki_name #new_cols[0].apply(lambda x: x.split('/')[3].lower())  #'' #wiki_name.lower()
            df['revised_id'] = ''
            df = df[df['predicate'] == '<http://www.w3.org/2000/01/rdf-schema#label>']
            df = df.reset_index(drop=True)
            df = df.drop(['details','predicate'], axis=1)
            df['length_label'] = df['label'].apply(lambda x:len(x))
            df['length_el'] = df[id_column_name].apply(lambda x:len(x))

            print('Total Length:', len(df))

            df = df[df['length_label'] < 3900]
            df = df.reset_index(drop=True)

            df = df.drop(['length_label','length_el'], axis=1)
            
            df.drop_duplicates(subset=[id_column_name], inplace=True)
            df = df.reset_index(drop=True)
            df.to_sql(target_table, engine, schema=schema, if_exists="append", index=False)

    def truncate_tables_from_db(self, dsn_name, table_name, schema):
        conn = pyodbc.connect('DSN=' + dsn_name +';')
        sql_query_string = "TRUNCATE TABLE " + schema + '.' + table_name
        print(sql_query_string)
        cursor = conn.cursor()
        cursor.execute(sql_query_string)
        conn.commit()

    def truncate_tables(self):
        self.truncate_tables_from_db(INS_LABELS_DB, 'entity_labels', 'dbo')
        self.truncate_tables_from_db(CAT_LABELS_DB, 'category_labels', 'dbo')
        self.truncate_tables_from_db(PROP_LABELS_DB, 'prop_labels', 'dbo')
        self.truncate_tables_from_db(CLASS_LABELS_DB, 'class_labels', 'dbo')
        self.truncate_tables_from_db(INS_DUPL_LABELS_DB, 'entity_dup_labels', 'dbo')
        self.truncate_tables_from_db(CAT_DUPL_LABELS_DB, 'category_dup_labels', 'dbo')
        self.truncate_tables_from_db(PROP_DUPL_LABELS_DB, 'prop_dup_labels', 'dbo')
        self.truncate_tables_from_db(CLASS_DUPL_LABELS_DB, 'class_dup_labels', 'dbo')

    def process_labels(self):
        wiki_names = glob.glob(BASE_DIR + '/'  +  DATA_DIR +  '/' + PROCESSED_DUMPS_DIR + '/*')
        self.truncate_tables()
        for file in wiki_names:
            wiki_name = os.path.basename(file)

            data_files = glob.glob(file + '/*.ttl')

            label_files = list(filter(lambda x: x.endswith('labels.ttl') or x.endswith('-infobox-property-definitions.ttl') or x.endswith('-template-type-definitions.ttl'), data_files))

            for lb_file in label_files:
                print(lb_file)
                if lb_file.endswith('-category-labels.ttl'): 
                    self.insert_labels_to_db(CAT_LABELS_DB, 'category_id', wiki_name, lb_file, 'category_labels', 'dbo')
                elif lb_file.endswith('-labels.ttl'): 
                    self.insert_labels_to_db(INS_LABELS_DB, 'entity_id', wiki_name, lb_file, 'entity_labels', 'dbo')
                elif lb_file.endswith('-infobox-property-definitions.ttl'):
                    self.insert_labels_to_db(PROP_LABELS_DB, 'prop_id', wiki_name, lb_file, 'prop_labels', 'dbo')
                elif lb_file.endswith('-template-type-definitions.ttl'):
                    self.insert_labels_to_db(CLASS_LABELS_DB, 'class_id', wiki_name, lb_file, 'class_labels', 'dbo')


    
    def pop_duplicate_label_count_tables(self, dsn_name, source_table, target_table, schema):
        conn = pyodbc.connect('DSN=' + dsn_name +';')
        
        sql_query_string = "INSERT INTO " + schema + "." + target_table + " select label label, count(*) count from  " + schema + "." + source_table + " group by label having count(*) > 1"
        print(sql_query_string)
        cursor = conn.cursor()
        cursor.execute(sql_query_string)
        
        conn.commit()

    def run_revise_uris_db(self, dsn_name, source_table, target_table, schema, prefix, type):
        conn = pyodbc.connect('DSN=' + dsn_name +';')
        
        sql_query_string = "UPDATE e_l SET revised_id =CONCAT('<http://dbkwik.webdatacommons.org/" + type +"/" + prefix + "', REPLACE(e_l.label, ' ','_'),'>') FROM " + schema + "." + source_table + " e_l " + "inner join " + schema + '.' + target_table + " ins on ins.label = e_l.label and ins.count > 1 "
        print(sql_query_string)
        cursor = conn.cursor()
        cursor.execute(sql_query_string)

        conn.commit()

    
    def revise_uris(self):

        #### populdate duplicate labels tables #######
        self.pop_duplicate_label_count_tables(INS_DUPL_LABELS_DB, 'entity_labels', 'entity_dup_labels', 'dbo')
        self.pop_duplicate_label_count_tables(CAT_DUPL_LABELS_DB, 'category_labels', 'category_dup_labels', 'dbo')
        self.pop_duplicate_label_count_tables(PROP_DUPL_LABELS_DB, 'prop_labels', 'prop_dup_labels', 'dbo')
        self.pop_duplicate_label_count_tables(CLASS_DUPL_LABELS_DB, 'class_labels', 'class_dup_labels', 'dbo')
        
        # populate labels count for instances
        self.run_revise_uris_db(INS_DUPL_LABELS_DB, 'entity_labels', 'entity_dup_labels', 'dbo', '', 'resource')

        # populate labels count for categories
        self.run_revise_uris_db(CAT_DUPL_LABELS_DB, 'category_labels', 'category_dup_labels', 'dbo','category:', 'resource')

        # populate labels count for properties
        self.run_revise_uris_db(PROP_DUPL_LABELS_DB, 'prop_labels', 'prop_dup_labels', 'dbo','', 'property')

        # populate labels count for classes
        self.run_revise_uris_db(CLASS_DUPL_LABELS_DB, 'class_labels', 'class_dup_labels', 'dbo','', 'class')

    
    
    def get_revise_ids_from_db(self, dsn_name, source_table, id_column_name, schema):
        conn = pyodbc.connect('DSN=' + dsn_name +';')
        query = "select wiki_name," +  id_column_name + " , revised_id from " + schema +"." + source_table  + " where revised_id <> '' order by wiki_name" 
        
        for df in pd.read_sql(query, conn, chunksize=10000): 
            while(len(df)>0):
                wiki_name = df.iloc[0]['wiki_name']
                print('Processing Wiki: ', wiki_name)
                df_wiki = df[df['wiki_name']==wiki_name]
                df_wiki = df_wiki.drop('wiki_name', axis = 1)
                df_wiki.to_csv(BASE_DIR + '/' + DATA_DIR + '/' + LB_MAP_DIR +'/' + wiki_name + '_' + 'labels_map.ttl' , sep=' ', mode='a+', header= None, index= False, encoding = 'utf-8')
                df.drop(df[df['wiki_name']==wiki_name].index, inplace=True)
                print('Remaining Record: ', len(df))
    
    def generate_labels_mapping_file(self):
        if os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' + LB_MAP_DIR):
            shutil.rmtree(BASE_DIR + '/' + DATA_DIR + '/' + LB_MAP_DIR, ignore_errors=False)
            time.sleep(5)
        os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' + LB_MAP_DIR)
        
        #### populdate duplicate labels tables #######
        self.get_revise_ids_from_db(INS_DUPL_LABELS_DB, 'entity_labels', 'entity_id', 'dbo')
        self.get_revise_ids_from_db(CAT_DUPL_LABELS_DB, 'category_labels', 'category_id', 'dbo')
        self.get_revise_ids_from_db(PROP_DUPL_LABELS_DB, 'prop_labels', 'prop_id', 'dbo')
        self.get_revise_ids_from_db(CLASS_DUPL_LABELS_DB, 'class_labels', 'class_id', 'dbo')

    
    # extract vectors from two wikis, wikiname should be folder names
    def extract_vectors(self, wiki_1, wiki_2):
        if os.path.exists(str(path.parent) + '/model/rdf2vec.model'):
            print('loading model')
            model = Word2Vec.load(str(path.parent) + '/model/rdf2vec.model', mmap='r')
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

            return df_vectors_wiki_1, df_vectors_wiki_2
        else:
            print('model file not present, please re-run the model')
            return None
