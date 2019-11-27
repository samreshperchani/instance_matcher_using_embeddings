import pandas as pd
import numpy as np
import faiss

df_gs_index = pd.read_csv('gs_index.csv', encoding='utf-8')

df = pd.read_pickle('all_word2vec_vector_27082019.pkl')
print('Reading data done ....')

for index, row in df_gs_index.iterrows():
    wiki_1 = row['wiki_1'].strip()
    wiki_2 = row['wiki_2'].strip()

    print('Processing: ', wiki_1, ' ', wiki_2 )

    wiki_1_folder = row['wiki_1_folder'].strip()
    wiki_2_folder = row['wiki_2_folder'].strip()

    df_wiki_1 = df[df['wiki_name'] == wiki_1_folder]
    df_wiki_1 = df_wiki_1.reset_index(drop=True)
    
    df_wiki_2 = df[df['wiki_name'] == wiki_2_folder]
    df_wiki_2 = df_wiki_2.reset_index(drop=True)
    
    dimension = 300
    n = len(df_wiki_2)

    db_vectors = np.stack(df_wiki_2['vector'].to_list(), axis=0).astype('float32')

    nlist = 5

    index = faiss.IndexFlatL2(dimension)

    index.train(db_vectors)

    index.add(db_vectors)

    query_vectors = np.stack(df_wiki_1['vector'].to_list(), axis=0).astype('float32')
    
    k = 10

    distances, indices = index.search(query_vectors, k)

    df_distances = pd.DataFrame(distances)

    df_indices = pd.DataFrame(indices)

    df_dist_ind = pd.merge(df_distances, df_indices, how="inner", left_index=True, right_index=True)

    df_dist_ind = df_dist_ind[df_dist_ind['0_x'] <= 3]

    df_final_corr = pd.DataFrame(columns = ['wiki_1', 'wiki_2', 'dist'])


    for index, row in df_dist_ind.iterrows():

        if row['0_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['0_y'],"entity_id"], 'dist': row['0_x']}, ignore_index=True)
        
        if row['1_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['1_y'],"entity_id"], 'dist': row['1_x']}, ignore_index=True)

        if row['2_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['2_y'],"entity_id"], 'dist': row['2_x']}, ignore_index=True)
        
        if row['3_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['3_y'],"entity_id"], 'dist': row['3_x']}, ignore_index=True)
        
        if row['4_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['4_y'],"entity_id"], 'dist': row['4_x']}, ignore_index=True)
        
        if row['5_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['5_y'],"entity_id"], 'dist': row['5_x']}, ignore_index=True)
        
        if row['6_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['6_y'],"entity_id"], 'dist': row['6_x']}, ignore_index=True)
        
        if row['7_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['7_y'],"entity_id"], 'dist': row['7_x']}, ignore_index=True)
        
        if row['8_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['8_y'],"entity_id"], 'dist': row['8_x']}, ignore_index=True)

        if row['9_x'] <= 3:
            df_final_corr = df_final_corr.append({'wiki_1': df_wiki_1.loc[index,"entity_id"], 'wiki_2': df_wiki_2.loc[row['9_y'],"entity_id"], 'dist': row['9_x']}, ignore_index=True)
        
    df_final_corr.to_csv(wiki_1 + '_' + wiki_2 +'word2vec_corr.csv', index=False)

