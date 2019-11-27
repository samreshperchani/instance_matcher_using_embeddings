import os
import pandas as pd
from matching import Player
from matching.games import StableMarriage
from matching.games import HospitalResident


MODEL_OUTPUT = 'model_output/'
MODEL = 'doc2vec_pca'
COR_FOLDER = 'correspondances/'

model_output_files = os.listdir(MODEL_OUTPUT + '/' + MODEL)


for file in model_output_files:

    print('Processing: ', file)
    
    #df = pd.read_csv(MODEL_OUTPUT + '/' + MODEL + '/' + file, encoding='utf-8')
    df = pd.read_pickle(MODEL_OUTPUT + '/' + MODEL + '/' + file)
    
    #wiki_1_ent = df['wiki_1'].tolist()
    #wiki_2_ent = df['wiki_2'].tolist()

    #entities = wiki_1_ent + wiki_2_ent
  

    print('*********Preparing Ids**********')

    #df['entity_wiki1_id'] = df['wiki_1'].apply(lambda x: entities.index(x))
    #df['entity_wiki2_id'] = df['wiki_2'].apply(lambda x: entities.index(x))
    df['entity_wiki1_id'] = df.index
    df['entity_wiki2_id'] = df['entity_wiki1_id'].apply(lambda x: x + len(df))

    print('*********Preparing Ids Done**********')

    entity_1_dict = pd.Series(df.wiki_1.values, index=df.entity_wiki1_id).to_dict()
    entity_2_dict = pd.Series(df.wiki_2.values, index=df.entity_wiki2_id).to_dict()
    entities = {**entity_1_dict, **entity_2_dict}

    print(len(entities))

    print('*********Preparing Dictionary for Men**********')
    df_wiki_1 = df
    dict_of_men = {}
    '''
    while len(df_wiki_1) > 0:
        entity_wiki1 = df_wiki_1.iloc[0]['wiki_1']
        entity_wiki1_id = df_wiki_1.iloc[0]['entity_wiki1_id']

        df_subset = df[df['wiki_1'] == entity_wiki1]
        
        df_subset = df_subset.sort_values(by=['dist'], ascending=True)
        
        dict_of_men[entity_wiki1_id] = df_subset['entity_wiki2_id'].tolist()
        df_wiki_1 = df_wiki_1[df_wiki_1['wiki_1'] != entity_wiki1]
    '''

    df_sorted = df_wiki_1.sort_values(by=['entity_wiki1_id','dist'], ascending=True)
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted = df_sorted[['entity_wiki1_id','entity_wiki2_id']]
    #results = df_sorted.groupby(['entity_wiki1_id'])['entity_wiki2_id'].apply(list)
    #df_results = pd.DataFrame(results)
    #dict_of_men = df_results.to_dict()['entity_wiki2_id']
    dict_of_men = df_sorted.groupby(['entity_wiki1_id'])['entity_wiki2_id'].apply(list).to_dict()

    #print(dict_of_men)

    print('*********Preparing Dictionary for Men Done**********')
    print('*********Preparing Dictionary for Women**********')
    df_wiki_2 = df
    dict_of_women = {}
    '''
    while len(df_wiki_2) > 0:
        entity_wiki2 = df_wiki_2.iloc[0]['wiki_2']
        entity_wiki2_id = df_wiki_2.iloc[0]['entity_wiki2_id']

        df_subset = df[df['wiki_2'] == entity_wiki2]
        
        df_subset = df_subset.sort_values(by=['dist'], ascending=True)
        
        dict_of_women[entity_wiki2_id] = df_subset['entity_wiki1_id'].tolist()
        df_wiki_2 = df_wiki_2[df_wiki_2['wiki_2'] != entity_wiki2]
    '''
    df_sorted = df_wiki_2.sort_values(by=['entity_wiki2_id','dist'], ascending=True)
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted = df_sorted[['entity_wiki2_id','entity_wiki1_id']]
    #results = df_sorted.groupby(['entity_wiki2_id'])['entity_wiki1_id'].apply(list)
    #df_results = pd.DataFrame(results)
    #dict_of_women = df_results.to_dict()['entity_wiki1_id']
    dict_of_women = df_sorted.groupby(['entity_wiki2_id'])['entity_wiki1_id'].apply(list).to_dict()


    print('*********Preparing Dictionary for Women Done**********')
    print('Data Preparation Done,  running algo . . . . ')
    capacities = {women: 1 for women in dict_of_women}
    #capacities = {women: 1 for women in dict_of_women}

    game = HospitalResident.create_from_dictionaries(dict_of_men, dict_of_women, capacities)

    result = game.solve()

    def get_name(x): return entities[x.name] #entities[x.name]


    #print(entities[437])
    #print(entities[292])

    with open(COR_FOLDER + '/' + MODEL + '/' + file + '.csv', 'w+', encoding = 'utf-8') as stable_marriage_op:
        stable_marriage_op.write("entity_id_wiki_2,entity_id_wiki_1\n")
        for key in result:
            #print(type(str(key.name)))
            #print(type(result[key]))
            #print(result[key][0].name)
            stable_marriage_op.write('\"' + entities[key.name] + '\",\"'+ ''.join(list(map(get_name, result[key]))) + '\"')
            stable_marriage_op.write("\n")
        
        stable_marriage_op.close()
