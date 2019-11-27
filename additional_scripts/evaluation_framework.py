from lxml import etree
import pandas as pd


CORRESPONDENCES_FILE = 'E:/University/Thesis/final-code/correspondances/rdf2vec/starwars_swgrdf2vec_corr.pkl.csv'
GS_FILE = 'starwars~swg~evaluation.xml'

def parse_gs_file(file_name):

    # parse xml tree from gold standard file
    tree = etree.parse(file_name)

    # initialize dictionary with namespaces
    ns = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'n':'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'}
    

    # get nodes with mapping
    gs_mappings = tree.xpath("n:Alignment/n:map", namespaces=ns)
    
    # get ontology and wiki names
    ontology_1 = tree.xpath("n:Alignment/n:onto1/n:Ontology/n:location/text()", namespaces=ns)
    wiki_1 = tree.xpath("n:Alignment/n:onto1/n:Ontology/@rdf:about", namespaces=ns)
    ontology_2 = tree.xpath("n:Alignment/n:onto2/n:Ontology/n:location/text()", namespaces=ns)
    wiki_2 = tree.xpath("n:Alignment/n:onto2/n:Ontology/@rdf:about", namespaces=ns)
    
    # initialize dataframe 
    df_gs = pd.DataFrame(columns=['entity_id_wiki_1', 'entity_id_wiki_2','wiki_1','wiki_2','ontology_loc_1', 'ontology_loc_2'])

    # loop through mapping and insert mapping in dataframe
    for mapping in gs_mappings:

        entity1_node = mapping.xpath('n:Cell/n:entity1/@rdf:resource', namespaces=ns)
        entity2_node = mapping.xpath('n:Cell/n:entity2/@rdf:resource', namespaces=ns)
        
        if '/resource/' in entity1_node[0].lower() or '/resource/' in entity2_node[0].lower():
            # Pass the row elements as key value pairs to append() function 
            df_gs = df_gs.append({'entity_id_wiki_1' : entity1_node[0].lower()  , 'entity_id_wiki_2' : entity2_node[0].lower() } , ignore_index=True)


    print(df_gs.head())
    #df_gs['wiki_1'] = wiki_1[0].lower()
    #df_gs['wiki_2'] = wiki_2[0].lower()
    #df_gs['ontology_loc_1'] = ontology_1[0].lower()
    #df_gs['ontology_loc_2'] = ontology_2[0].lower()

    df_gs.to_csv('gs.csv', encoding='utf-8')
    

    return df_gs
    


def calculate_performance_parameters(true_mappings, model_mappings):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    count = 0

    print('True Mappings: ', len(true_mappings))
    
    true_map_not_null = true_mappings[(true_mappings['entity_id_wiki_1']!='null') & (true_mappings['entity_id_wiki_2']!='null')]

    print('True Mappings(Not Null): ', len(true_map_not_null))
    print('Model Mappings: ', len(model_mappings))
    
    model_map_not_null = model_mappings[(model_mappings['entity_id_wiki_1']!='null') & (model_mappings['entity_id_wiki_2']!='null')]
    total_retrieved = len(model_map_not_null)
    
    print('Model Mappings(Not Null): ', len(model_map_not_null))
    
    # join to get true positives
    matches = pd.merge(true_map_not_null, model_map_not_null, how='inner', on=['entity_id_wiki_1', 'entity_id_wiki_2'] )
    print('Matches: ', len(matches))
    
    tp = len(matches)
    fp = total_retrieved - tp
    fn = max(0,len(true_map_not_null) - len(matches))
    tn = len(model_mappings) - (tp + fp + fn)
    #print(model_map_not_null.head())


    
    '''
    for index, row in true_mappings.iterrows():
        count = count + 1
        if count == 500:
            break
        if row['entity_id_wiki_1'] != 'null' and  row['entity_id_wiki_2'] != 'null':
            match = model_mappings[(model_mappings['entity_id_wiki_1'] == row['entity_id_wiki_1']) & (model_mappings['entity_id_wiki_2'] == row['entity_id_wiki_2'])]

            if len(match) > 0:
                #print(match)
                match.to_csv('match.csv')
                tp = tp + 1
            else:
                fn = fn + 1

        
        elif row['entity_1'] == 'null':
            match = model_mappings[(model_mappings['entity_id_wiki_2'] == row['entity_id_wiki_2'])]
            
            if len(match) > 0:
                fp = fp + 1
            else:
                tn = tn + 1
        
        elif row['entity_2'] == 'null':
            match = model_mappings[(model_mappings['entity_id_wiki_1'] == row['entity_id_wiki_1'])]
            
            if len(match) > 0:
                fp = fp + 1
            else:
                tn = tn + 1
    '''
    print('True Positives: ', tp)
    print('False Positives: ', fp)
    print('True Negatives: ', tn)
    print('False Negatives: ', fn)
    return tp, fp, fn, tn


def calculate_performance_measures(true_positives, false_positives, false_negatives, true_negatives):

    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)

    precision = (true_positives) / (true_positives + false_positives)

    recall = (true_positives) / (true_positives + false_negatives)

    f1_measure = (2 * (precision*recall)) / (precision+recall)

    return accuracy, precision, recall, f1_measure



df_gs = parse_gs_file(GS_FILE)

#df_gs = pd.read_pickle('E:/University/Thesis/final-code/ml_classification/test_set.pkl')

df_cor = pd.read_csv(CORRESPONDENCES_FILE, encoding='utf-8')
df_cor = df_cor.fillna('null')
df_cor['entity_id_wiki_1'] = df_cor['entity_id_wiki_1'].apply(lambda x: x.lower())
df_cor['entity_id_wiki_2'] = df_cor['entity_id_wiki_2'].apply(lambda x: x.lower())
df_cor.drop_duplicates(inplace=True) 

tp, fp, fn, tn  = calculate_performance_parameters(df_gs, df_cor)

accuracy, precision, recall, f1_measure  = calculate_performance_measures(tp, fp, fn, tn)

print('True Positives: ', tp)
print('False Positives: ', fp)
print('False Negatives: ', fn)
print('True Negatives: ', tn)


print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Measure: ', f1_measure)

#print(df_gs.head())
#print(df_cor.head())
