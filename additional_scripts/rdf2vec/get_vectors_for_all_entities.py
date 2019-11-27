import pandas as pd
from gensim.models import Word2Vec

print('loading model')
model = Word2Vec.load('model_new/RDF2Vec_en_all.model', mmap='r')
print('loading model done ..')

words = model.wv.vocab
df_vocab = pd.DataFrame(columns=['original_vocab'])
df_vocab['original_vocab'] = list(words.keys())
df_vocab['entity_id'] = df_vocab['original_vocab'].apply(lambda x: str(x).lower().strip())

#df_gs_entties = pd.read_csv('gs_entities.csv', encoding='utf-8')
df_gs_entties = pd.read_pickle('final_ent_incl_old_no_lyrics_27082019.pkl')
df_gs_entties['original_gs_id'] = df_gs_entties['entity_id'].apply(lambda x: x.lower().strip())
df_gs_entties['vector'] = ''


df_gs_entties['entity_id'] = df_gs_entties['entity_id'].apply(lambda x: '<' + x.lower().strip() + '>')

df_commons = pd.merge(df_gs_entties, df_vocab, how='inner', left_on=['revised_id'], right_on=['entity_id'])

print('Total Commons Wikis: ', len(df_commons))

print('extracting vectors')

for index, row in df_commons.iterrows():
    #if row['entity_id'] in words:
    df_commons.loc[index]['vector'] = model.wv[row['original_vocab']]


df_commons = df_commons[['original_gs_id','wiki_name','label','vector']]

df_commons.rename(columns = {'original_gs_id': 'entity_id'}, inplace=True)

df_commons.to_pickle('all_entities_rdf2vec_vector_28082019.pkl')




