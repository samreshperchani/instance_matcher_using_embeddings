import multiprocessing as mp
import glob
import os
import shutil
import pandas as pd
import gc


BASE_DIR = '/work/sperchan/bkp_latest_dumps/'
DUMPS_DIR = 'extracted_dumps/*'
KG_PATH_SAMPLE ='knowledge_graphs'
KG_PATH ='knowledge_graphs_final_en_all'
LB_DIR ='labels_mapping_en_all'
chunksize= 10000000
gc.collect()

def update_duplicate_label(wiki):
    try:
        print('Processing Wiki: ', wiki)
        use_mappings = False
        replace_dictionary = {}
        with open(KG_PATH + '/' + os.path.basename(wiki), 'w+', encoding='utf-8') as out_file:
            print(os.path.basename(wiki).replace('_kg.ttl','_labels_map.ttl'))
            labels_file = os.path.basename(wiki).replace('_kg.ttl','_labels_map.ttl')
            if os.path.exists(LB_DIR + '/'  + labels_file):
                print('Using Labels File: ', labels_file)
                df = pd.DataFrame.from_csv(LB_DIR + '/' + labels_file, sep=' ', header=None, encoding='utf-8')
                df.columns = ['value']
                #replace_dictionary = df.to_dict('index')
                #print(type(df))
                use_mappings = True
            
            with open(wiki, 'r', encoding='utf-8') as ttl_file:
                file_contents = ttl_file.read(chunksize)
                while file_contents:
                    if use_mappings == True:
                        for index, row in df.iterrows():
                            #print(index)
                            file_contents = file_contents.lower().replace(index, row['value'])
                    out_file.write(file_contents)
                    file_contents = ttl_file.read(chunksize)
                ttl_file.close()
            #os.remove(wiki)
            out_file.close()
    except Exception as e:
        print(wiki , ': Exception: ', str(e))


if __name__ == '__main__':

    wikis = glob.glob(BASE_DIR + '/' + KG_PATH_SAMPLE + '/*.ttl')
    print(len(wikis))
    #update_duplicate_label(wikis[0])
    processes = max(1, mp.cpu_count()-12)
    
    with mp.Pool(processes) as pool:
        pool.map_async(update_duplicate_label, wikis)
        pool.close()
        pool.join()