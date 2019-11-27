import pandas as pd
from sklearn.decomposition import TruncatedSVD

df = pd.read_pickle('data_revised_negative_example/more_negative_example_ds.pkl')
df = df.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])
df = df.reset_index(drop=True)

vector_length = 300
reduced_dimensions = 300


df_vec_en_1 = df[['doc2vec_vector_ent1','word2vec_vector_ent1','rdf2vec_vector_ent1']]

vector_col_name = ['doc2vec_ent1_' + str(i) for i in range(0,vector_length) ]
df_vec_en_1[vector_col_name]= pd.DataFrame(df.doc2vec_vector_ent1.tolist(), index=df.index)
df_vec_en_1.drop(columns=['doc2vec_vector_ent1'], inplace=True)


vector_col_name = ['rdf2vec_ent1_' + str(i) for i in range(0,vector_length) ]
df_vec_en_1[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_ent1.tolist(), index=df.index)
df_vec_en_1.drop(columns=['rdf2vec_vector_ent1'], inplace=True)


vector_col_name = ['word2vec_ent1_' + str(i) for i in range(0,vector_length) ]
df_vec_en_1[vector_col_name]= pd.DataFrame(df.word2vec_vector_ent1.tolist(), index=df.index)
df_vec_en_1.drop(columns=['word2vec_vector_ent1'], inplace=True)




df_vec_en_2 = df[['doc2vec_vector_ent2','word2vec_vector_ent2','rdf2vec_vector_ent2']]

vector_col_name = ['doc2vec_ent2_' + str(i) for i in range(0,vector_length) ]
df_vec_en_2[vector_col_name]= pd.DataFrame(df.doc2vec_vector_ent2.tolist(), index=df.index)
df_vec_en_2.drop(columns=['doc2vec_vector_ent2'], inplace=True)


vector_col_name = ['rdf2vec_ent2_' + str(i) for i in range(0,vector_length) ]
df_vec_en_2[vector_col_name]= pd.DataFrame(df.rdf2vec_vector_ent2.tolist(), index=df.index)
df_vec_en_2.drop(columns=['rdf2vec_vector_ent2'], inplace=True)

vector_col_name = ['word2vec_ent2_' + str(i) for i in range(0,vector_length) ]
df_vec_en_2[vector_col_name]= pd.DataFrame(df.word2vec_vector_ent2.tolist(), index=df.index)
df_vec_en_2.drop(columns=['word2vec_vector_ent2'], inplace=True)







df = df[['entity_id_wiki_1','entity_id_wiki_2','label']]

df = pd.merge(df, df_vec_en_1, left_index=True, right_index=True )

df = pd.merge(df, df_vec_en_2, left_index=True, right_index=True )

print(len(df))
print(df.head())

df.to_pickle('more_negative_example_ds_concat.pkl')