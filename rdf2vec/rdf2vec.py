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

# path where extracted text will be stored
EXTRACTED_TEXT_DIR = config.EXTRACTED_TEXT_DIR


print(gensim.models.word2vec.FAST_VERSION)

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