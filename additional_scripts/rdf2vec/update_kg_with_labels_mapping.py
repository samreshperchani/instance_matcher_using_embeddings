import pandas as pd

KG_PATH = '/work/sperchan/bkp_latest_dumps/merged_kg.ttl'
LABELS_PATH = 'merged_labels.ttl'
OUTPUT_FILE = 'final_graph.ttl'

chunksize = 1000
print('reading labels')
df = pd.read_csv(LABELS_PATH, sep=' ', header=None, encoding='utf-8')
df.columns = ['key','value']
print('reading labels done')
with open(OUTPUT_FILE,'w', encoding='utf-8') as final_kg:
    with open(KG_PATH,'r', encoding='utf-8') as kg:
        file_lines = kg.readlines(chunksize)
        while file_lines:
            
            file_lines = kg.readlines(chunksize)
            kg_lines = ''.join(file_lines)
            
            for index, row in df.iterrows():
                kg_lines = kg_lines.replace(row['key'],row['value'])


            #with open(LABELS_PATH,'r', encoding='utf-8') as lb:
                #labels = lb.readlines(chunksize)
                #df_chunck = pd.read_csv(lb, sep=' ', header=None, encoding='utf-8', chunksize=chunksize)
                #df = df_chunck.next()
                #lb_lines = lb.readlines(chunksize)
                #print(lb_lines)
                #for df in pd.read_csv(lb, sep=' ', header=None, encoding='utf-8', chunksize=chunksize):
                    #print(df.head())
            #    while lb_lines:
            #        for line in lb_lines:
            #            key_value = line.split(' ')
            #            kg_lines = kg_lines.lower().replace(key_value[0], key_value[1])
            #        lb_lines = lb.readlines(chunksize)
                        #for index, row in df.iterrows():
                        
                    #df = df_chunck.next()
            
            final_kg.write(kg_lines)
            file_lines = kg.readlines(chunksize)



