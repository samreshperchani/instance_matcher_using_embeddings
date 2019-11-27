import pandas as pd
from sklearn.metrics import jaccard_similarity_score
import nltk

df = pd.read_pickle('dataset_ensembles/ensemble_training_ds.pkl')




df_ent_1 = df[['entity_id_wiki_1', 'doc2vec_vector_ent1', 'word2vec_vector_ent1', 'rdf2vec_vector_ent1']]
df_ent_1 = df_ent_1[df_ent_1['entity_id_wiki_1'].str.find('/lyrics.wikia.com/') == -1]
df_ent_1['label'] = df_ent_1['entity_id_wiki_1'].apply(lambda x: x[x.rfind('/')+1:] )
df_ent_1['wiki_name'] = df_ent_1['entity_id_wiki_1'].apply(lambda x: x[32+1: x.find('/', 42)] )
df_ent1_sample = df_ent_1.sample(n=70)
df_ent1_sample.index = [1 for i in range(0, len(df_ent1_sample))]
#df_ent_1.rename(columns={'entity_id_wiki_1': 'entity_id', 'doc2vec_vector_ent1':'doc2vec_vector', 'word2vec_vector_ent1' : 'word2vec_vector', 'rdf2vec_vector_ent1': 'rdf2vec_vector'}, inplace=True)
print(df_ent_1.head())


df_ent_2 = df[['entity_id_wiki_2', 'doc2vec_vector_ent2', 'word2vec_vector_ent2', 'rdf2vec_vector_ent2']]
df_ent_2 = df_ent_2[df_ent_2['entity_id_wiki_2'].str.find('/lyrics.wikia.com/') == -1]
df_ent_2['label'] = df_ent_2['entity_id_wiki_2'].apply(lambda x: x[x.rfind('/')+1:] )
df_ent_2['wiki_name'] = df_ent_2['entity_id_wiki_2'].apply(lambda x: x[32+1: x.find('/', 42)] )
df_ent2_sample = df_ent_2.sample(n=70)
df_ent2_sample.index = [1 for i in range(0, len(df_ent2_sample))]



#df_ent_2.rename(columns={'entity_id_wiki_2': 'entity_id', 'doc2vec_vector_ent2':'doc2vec_vector', 'word2vec_vector_ent2' : 'word2vec_vector', 'rdf2vec_vector_ent2': 'rdf2vec_vector'}, inplace=True)
print(df_ent_2.head())


#df_vectors = pd.concat([df_ent_1, df_ent_2])

#df_vectors = df_vectors.drop_duplicates(subset=['entity_id'])

#print(len(df_vectors))


def string_sim(x):
    return 1 - nltk.jaccard_distance(set(x['label_x']), set(x['label_y']))


df_merged = pd.merge(df_ent1_sample, df_ent2_sample, how ='inner', left_index=True, right_index=True)
df_merged = df_merged[df_merged['label_x'] != df_merged['label_y'] ]
df_merged = df_merged[df_merged['wiki_name_x'] != df_merged['wiki_name_y'] ]
df_merged = df_merged[df_merged['entity_id_wiki_1'] != df_merged['entity_id_wiki_2'] ]
df_merged['similarity'] = df_merged.apply(string_sim, axis=1)
df_merged = df_merged[df_merged['similarity'] <= 0.4]


print(len(df_merged))
print(df_merged.head())

df_merged['label'] = 0
df_merged = df_merged[['entity_id_wiki_1', 'entity_id_wiki_2', 'label','doc2vec_vector_ent1','doc2vec_vector_ent2', 'rdf2vec_vector_ent1' , 'rdf2vec_vector_ent2', 'word2vec_vector_ent1', 'word2vec_vector_ent2']]
df_merged = df_merged.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])
df_merged = df_merged.reset_index(drop=True)
print(df_merged.head())

print(df_merged.loc[0])

print(len(df_merged))

print(df_merged.columns)
print(df.columns)


df_final = pd.concat([df,df_merged])
df_final =  df_final.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])
df_final = df_final.reset_index(drop=True)
print(len(df_final))

df_final.to_pickle('more_negative_example_ds.pkl')





