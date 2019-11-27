import pandas as pd

matching_pairs =[]

df = pd.read_csv('correspondances_marvel.wikia.com_marvelcinematicuniverse.wikia.com.csv', encoding='utf-8')

df.columns = ['entity_wiki1','entity_wiki2','sim_score']

df.sort_values(by='sim_score', ascending=False, inplace=True)


print(df.head())


while len(df)!=0:
    record = df.iloc[0]

    matching_pairs.append((record.entity_wiki1.lower(),record.entity_wiki2.lower(),record.sim_score))

    df = df[df['entity_wiki1']!=record.entity_wiki1]
    df = df[df['entity_wiki2']!=record.entity_wiki2]


    print(len(df))
    df.sort_values(by='sim_score', ascending=False, inplace=True)

    #break


df_matching_pairs = pd.DataFrame(matching_pairs, columns=['entity_id_wiki_1','entity_id_wiki_2','sim_score'])

df_matching_pairs.to_csv('stable_marriage.csv', encoding='utf-8', index= False)