import pandas as pd
from gensim.models import Word2Vec

print('loading model')
model = Word2Vec.load('model/word2vec_en_all_v2.model', mmap='r')
print('loading model done ..')

words = model.wv.vocab
df_vocab = pd.DataFrame(columns=['original_vocab'])
df_vocab['original_vocab'] = list(words.keys())
df_vocab['entity_id'] = df_vocab['original_vocab'].apply(lambda x: str(x).lower().strip())

#df_gs_entties = pd.read_csv('gs_entities.csv', encoding='utf-8')
df_gs_entties = pd.read_pickle('final_ent_incl_old_no_lyrics_27082019.pkl')
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x: x.lower().strip())
df_gs_entties['original_gs_id'] = df_gs_entties['entity_id']
df_gs_entties['vector'] = ''

df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/stexpanded.wikia.com/','/stexpanded/'))
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/lyrics.wikia.com/','/lyrics/'))
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/swg.wikia.com/','/swg/'))
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/swtor.wikia.com/','/swtor/'))
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/starwars.wikia.com/','/starwars/'))
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/marvelcinematicuniverse.wikia.com/','/marvelcinematicuniverse/'))
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/marvel.wikia.com/','/marvel/'))
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/memory-alpha.wikia.com/','/memory_alpha/'))
df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x:x.replace('/memory-beta.wikia.com/','/memory_beta/'))

df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x: x.lower().strip())

df_commons = pd.merge(df_gs_entties, df_vocab, how='inner', on=['entity_id'])

print('Total Commons Wikis: ', len(df_commons))

print('extracting vectors')

for index, row in df_commons.iterrows():
    #if row['entity_id'] in words:
    df_commons.loc[index]['vector'] = model.wv[row['original_vocab']]

df_commons['entity_id'] = df_commons['original_gs_id']
df_commons = df_commons[['entity_id','vector','wiki_name']]

print('Total Records: ', len(df_commons))
print('Total Records(NULL VECTOR): ', len(df_commons[df_commons['vector']=='']))

df_commons.to_pickle('all_word2vec_vector_27082019.pkl')




