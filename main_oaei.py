from utilities.rdf_xml_utils import RDF_XML_UTILS
from doc2vec.doc2vec import DOC2Vec
from rdf2vec.rdf2vec import RDF2Vec
from word2vec.word2vec import WORD2Vec
import os
import config
from ensemble.ensemble_learning import ENSEMBLE_LEARNING
from classification.classification import CLASSIFICATION
from oaei.oaei import OAEI
import pandas as pd

print(config.BASE_DIR)

# extract path to base directory
BASE_DIR = config.BASE_DIR

# extract path to data directory
DATA_DIR = config.DATA_DIR

# model to use for classification
CLASSIFICATION_MODEL= config.CLASSIFICATION_MODEL

rdf_xml_utils = RDF_XML_UTILS()
ensemble_ds = ENSEMBLE_LEARNING()
classifier = CLASSIFICATION()
oaei_file_formater = OAEI()

print('***********processing data dumps************')
rdf_xml_utils.convert_xml_rdf_ttl()

print('***** Executing RDF2Vec model ************')
os.system('python rdf2vec/main.py')

print('***** Executing DOC2Vec model ************')
os.system('python doc2vec/main.py')

print('***** Executing Word2Vec model ************')
#os.system('python word2vec/main.py')

print('***** Generate training and test environment ************')
df_train_set, df_test_set = rdf_xml_utils.generate_train_test(wiki_1 = 'darkscape', wiki_2 = 'oldschoolrunscape')

print('****** Save training and test set *************')
#df_train_set.to_pickle(BASE_DIR + '/' + DATA_DIR + '/' + 'training_set_oaei.pkl')
#df_test_set.to_pickle(BASE_DIR + '/' + DATA_DIR + '/' + 'test_set_oaei.pkl')



#df_train_set = pd.read_pickle(BASE_DIR + '/' + DATA_DIR + '/' + 'training_set_oaei.pkl')
#df_test_set  = pd.read_pickle(BASE_DIR + '/' + DATA_DIR + '/' + 'test_set_oaei.pkl')

print('getting esemble dataset')
# get ensemble dataset
X_train, y_train, X_test = ensemble_ds.get_dataset(df_train_set, df_test_set)

print(X_train.head())
print(y_train.head())
print(X_test.head())

print('running classifier')
# get ensemble dataset
y_pred = classifier.run_classification(X_train, y_train, X_test, CLASSIFICATION_MODEL)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['label']

print('***************X train************')
print(X_train.head())

df_training_set = pd.concat([X_train, y_train], axis=1)

print(df_training_set.head())
df_test_set = pd.concat([X_test, y_pred], axis=1)

df_final_set = pd.concat([df_training_set, df_test_set], axis=0, ignore_index=True)

print('***************Final set************')
df_final_set = df_final_set[df_final_set['label']==1.0]

print('***************Generating OAEI File************')
oaei_file = oaei_file_formater.generate_file_oaei_format(df_final_set)

print('***************Writing output to file************')
oaei_file.write("oaei_output.xml", encoding='utf-8',  xml_declaration=True, method = 'xml')


#print(df_train_set.head())
#print(df_test_set.head())



# convert rdf xmls to ttl
#rdf_xml_utils.convert_xml_rdf_ttl()

# create object of a class doc2vec
#doc2vec = DOC2Vec()

# call function to pre-process long abstracts
#doc2vec.pre_process_long_abstracts()

# call function to train doc2vec model
#doc2vec.train_model()


'''
# create object of a class RDF2Vec
rdf2vec_model = RDF2Vec()

#insert labels to db
rdf2vec_model.process_labels()

# revise uri based on duplicate labels
rdf2vec_model.revise_uris()

# generate labels mapping files
rdf2vec_model.generate_labels_mapping_file()

# extract text
rdf2vec_model.generate_knowledge_graphs()

# revised knowledge graphs with labels mapping
rdf2vec_model.generate_revised_knowledge_graphs()

# generate final knowledge graph for random walks
rdf2vec_model.generate_merged_kg_graphs()

# generate walks on a graph
rdf2vec_model.generate_walks()

# train RDF2Vec model
rdf2vec_model.train_model()

'''

