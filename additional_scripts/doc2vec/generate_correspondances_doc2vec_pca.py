import pandas as pd
import numpy as np
import faiss
from sklearn import preprocessing
from sklearn.decomposition import PCA

df_gs_index = pd.read_csv('gs_index.csv', encoding='utf-8')

df = pd.read_pickle('docvec_final_no_lytics.pkl')
print('Reading data done ....')

vector_length = 300
reduced_dimensions = 100


vector_col_name = ['model' + str(i) for i in range(0,vector_length) ]

def create_vector(x):
    return np.array(x[vector_col_name[0:reduced_dimensions]])
print('Running PCA')
df_pca = pd.DataFrame()
    
df[vector_col_name]= pd.DataFrame(df.vector.values.tolist(), index=df.index)
    
df_pca = df[vector_col_name]
    
pca = PCA(n_components=reduced_dimensions, whiten = False, random_state = 2019)
X_pca = pca.fit_transform(df_pca)
    
df_pca = pd.DataFrame.from_records(X_pca)
df_pca = df_pca.reset_index(drop=True)
df_pca.columns = vector_col_name[0:reduced_dimensions]

print('Running PCA Done....')

print('Creating Vectors....')

df_pca['vector_pca'] = df_pca.apply(create_vector, axis=1)

print('Creating Vectors Done....')

df_pca.drop(columns=vector_col_name[0:reduced_dimensions], inplace=True)

df = pd.merge(df, df_pca, how='inner', left_index=True, right_index=True)

df.drop(columns=vector_col_name, inplace=True)


for index, row in df_gs_index.iterrows():
    wiki_1 = row['wiki_1'].strip()
    wiki_2 = row['wiki_2'].strip()

    print('Processing: ', wiki_1, ' ', wiki_2 )

    wiki_1_folder = row['wiki_1_folder'].strip()
    wiki_2_folder = row['wiki_2_folder'].strip()

    df_wiki_1 = df[df['wiki_name'] == wiki_1]
    df_wiki_1 = df_wiki_1.reset_index(drop=True)

    df_wiki_1['norm_vector'] = df_wiki_1.vector_pca.apply(lambda x: preprocessing.normalize(np.array([x]), norm='l2')[0])


    df_wiki_2 = df[df['wiki_name'] == wiki_2]
    df_wiki_2 = df_wiki_2.reset_index(drop=True)
    
    df_wiki_2['norm_vector'] = df_wiki_2.vector_pca.apply(lambda x: preprocessing.normalize(np.array([x]), norm='l2')[0])

    print(df_wiki_2.head())
    
    dimension = reduced_dimensions
    n = len(df_wiki_2)

    db_vectors = np.stack(df_wiki_2['norm_vector'].to_list(), axis=0).astype('float32')

    nlist = 5

    index = faiss.IndexFlatL2(dimension)

    index.train(db_vectors)

    index.add(db_vectors)

    query_vectors = np.stack(df_wiki_1['norm_vector'].to_list(), axis=0).astype('float32')
    nq = len(df_wiki_1)
    
    #k = 10

    #distances, indices = index.search(query_vectors, k)

    lim, dis, ids = index.range_search(query_vectors, 0.316)
    #df_final_corr = pd.DataFrame(columns = ['wiki_1', 'wiki_2', 'sim_score'])
    
    final_corr = []
    #with open(wiki_1 + '_' + wiki_2 +'word2vec_corr.csv', 'w+', encoding='utf-8') as corrs_file:
        #corrs_file.write('wiki_1,wiki_2,dist'+'\n')
    if True:
        for i in range(nq):
            l0, l1 = lim[i], lim[i + 1]
            if l1 > l0:
                wiki_1_id = df_wiki_1.loc[i, 'entity_id']
                ind_wiki_2 = ids[l0:l1]
                dist_wiki_2 = dis[l0:l1]

                for j in range(len(ind_wiki_2)):
                    #corrs_file.write(str(wiki_1_id) + ',' + str(df_wiki_2.loc[ind_wiki_2[j],"entity_id"]) + ',' + str(dist_wiki_2[j]) + '\n')
                    #df_final_corr = df_final_corr.append({'wiki_1': wiki_1_id, 'wiki_2': df_wiki_2.loc[ind_wiki_2[j],"entity_id"], 'dist': dist_wiki_2[j]}, ignore_index=True)
                    final_corr.append((wiki_1_id,df_wiki_2.loc[ind_wiki_2[j],"entity_id"],dist_wiki_2[j]))
    
    df_final_corr = pd.DataFrame(final_corr, columns=['wiki_1', 'wiki_2','dist'])
    df_final_corr = df_final_corr.drop_duplicates(subset=['wiki_1','wiki_2'])
    df_final_corr.to_pickle(wiki_1 + '_' + wiki_2 +'doc2vec_corr_pca.pkl')

