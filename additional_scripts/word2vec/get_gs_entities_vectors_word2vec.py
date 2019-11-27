import pandas as pd

'''
df_gs = pd.read_csv('gs_entities.csv', encoding='utf-8')

df_gs['entity_id'] = df_gs['entity_id'].apply(lambda x: str(x).lower().strip())
df_gs['revised_id'] = df_gs['entity_id'].apply(lambda x: x.replace('.wikia.com/resource/','/resource/'))
#print(df_gs.head())

df_gs.to_pickle('word2vec_gs_entities.pkl')

print(len(df_gs))

'''


import gensim
from gensim.models import Word2Vec
import pandas as pd

BASE_DIR = '/work/sperchan/Word2Vec/'
MODEL_FILE = 'model/word2vec_en_all_v2.model'
GS_ENTITIES = 'word2vec_gs_entities.pkl'


print('Loading Model .....')
model = Word2Vec.load(BASE_DIR + '/' + MODEL_FILE, mmap='r')
print('Loading Model Done .....')

vocab = model.wv

print('total words found: ', len(model.wv.vocab))

vocab = {str(k).lower().strip():v for k,v in vocab.items()}


df_gs = pd.read_pickle(BASE_DIR + '/' + GS_ENTITIES)

df_gs['vector'] = ''
for index, row in df_gs.iterrows():
    if df_gs['revised_id'] in vocab:
        df_gs[index,'vector'].loc = vocab[df_gs['revised_id']]



df_gs = df_gs[['entity_id','wiki_name','vector']]

print('Total Records: ', len(df_gs))
print('Total Records(NULL VECTOR): ', len(df_gs[df_gs['vector']=='']))

df_gs.to_pickle(BASE_DIR + '/word2vec_gs_ent_vectors.pkl')