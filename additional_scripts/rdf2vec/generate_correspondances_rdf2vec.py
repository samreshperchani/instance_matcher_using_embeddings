import pandas as pd
import numpy as np
import faiss
from sklearn import preprocessing

df_gs_index = pd.read_csv('gs_index.csv', encoding='utf-8')

df = pd.read_pickle('all_entities_rdf2vec_vector_28082019.pkl')
print('Reading data done ....')

for index, row in df_gs_index.iterrows():
    wiki_1 = row['wiki_1'].strip()
    wiki_2 = row['wiki_2'].strip()

    print('Processing: ', wiki_1, ' ', wiki_2 )

    wiki_1_folder = row['wiki_1_folder'].strip()
    wiki_2_folder = row['wiki_2_folder'].strip()

    df_wiki_1 = df[df['wiki_name'] == wiki_1_folder]
    df_wiki_1 = df_wiki_1.reset_index(drop=True)
    df_wiki_1['norm_vector'] = df_wiki_1.vector.apply(lambda x: preprocessing.normalize(np.array([x]), norm='l2')[0])
    
    df_wiki_2 = df[df['wiki_name'] == wiki_2_folder]
    df_wiki_2 = df_wiki_2.reset_index(drop=True)
    df_wiki_2['norm_vector'] = df_wiki_2.vector.apply(lambda x: preprocessing.normalize(np.array([x]), norm='l2')[0])
    
    dimension = 300
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

    lim, dis, ids = index.range_search(query_vectors, 0.6)
    #df_final_corr = pd.DataFrame(columns = ['wiki_1', 'wiki_2', 'sim_score'])
    
    with open(wiki_1 + '_' + wiki_2 +'word2vec_corr.csv', 'w+', encoding='utf-8') as corrs_file:
        corrs_file.write('wiki_1,wiki_2,dist'+'\n')
        for i in range(nq):
            l0, l1 = lim[i], lim[i + 1]
            if l1 > l0:
                wiki_1_id = df_wiki_1.loc[i, 'entity_id']
                ind_wiki_2 = ids[l0:l1]
                dist_wiki_2 = dis[l0:l1]

                for j in range(len(ind_wiki_2)):
                    corrs_file.write(str(wiki_1_id) + ',' + str(df_wiki_2.loc[ind_wiki_2[j],"entity_id"]) + ',' + str(dist_wiki_2[j]) + '\n')
                    #df_final_corr = df_final_corr.append({'wiki_1': wiki_1_id, 'wiki_2': df_wiki_2.loc[ind_wiki_2[j],"entity_id"], 'dist': dist_wiki_2[j]}, ignore_index=True)

    
    
    #df_final_corr.to_csv(wiki_1 + '_' + wiki_2 +'rdf2vec_corr.csv', index=False)

